"""Experiment orchestration (proposal Section 3.4).

For each scenario: generate the replicate instances (common random numbers), write them in the case
format with per-algorithm seeds, run every algorithm in Java (gcs.opt.RunExperiment), and keep the
per-scenario CSVs. Finished scenarios are skipped on restart (checkpointing); a failed scenario is re-run
once with the same seeds, and every attempt is logged in run_log.csv.

The run configuration goes to config_<hash>.json; its hash is logged with every attempt and stored with
every selected schedule (sNN_schedules.csv.gz). Scenarios, instances and algorithms run in a seeded
random order (--order-seed); the results do not depend on it.

    python -m gcs.experiment --out ../results/main                       # full matrix, all algorithms
    python -m gcs.experiment --out ../results/pilot --scenarios 0,40,80 --reps 3
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import time

from .export_case import write_case
from .scenarios import algorithm_seed, instance_seed, java_seed, load_config, scenario_matrix
from .workload import generate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JAVA_OUT = os.path.join(ROOT, "java", "out")


def all_algorithms(cfg) -> list[str]:
    return list(cfg["algorithms"]) + list(cfg.get("ablations", [])) + list(cfg.get("supplementary", []))


def parse_ids(spec: str, n: int) -> list[int]:
    if spec == "all":
        return list(range(n))
    ids = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return ids


def compile_java():
    """Compile gcs.core and gcs.opt into java/out, unless the classes are newer than every source:
    another experiment may be running JVMs from java/out, and rewriting classes under it is unsafe."""
    srcs = []
    for sub in ("core", "opt"):
        d = os.path.join(ROOT, "java", "src", "main", "java", "gcs", sub)
        srcs += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".java")]
    stamp = os.path.join(JAVA_OUT, "gcs", "opt", "RunExperiment.class")
    if os.path.exists(stamp) and os.path.getmtime(stamp) >= max(os.path.getmtime(f) for f in srcs):
        return
    os.makedirs(JAVA_OUT, exist_ok=True)
    subprocess.run(["javac", "-d", JAVA_OUT] + srcs, check=True)


def source_hash() -> str:
    """SHA-256 of the Java (core, opt) and Python (gcs) sources, line endings normalised."""
    h = hashlib.sha256()
    dirs = [os.path.join(ROOT, "java", "src", "main", "java", "gcs", s) for s in ("core", "opt")]
    dirs.append(os.path.join(ROOT, "python", "gcs"))
    for d in dirs:
        for f in sorted(os.listdir(d)):
            if f.endswith((".java", ".py")):
                h.update(os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, "/").encode())
                with open(os.path.join(d, f), "rb") as fh:
                    h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def write_manifest(out_dir, manifest) -> str:
    """Write the configuration as sorted JSON to config_<hash>.json and return the hash."""
    text = json.dumps(manifest, sort_keys=True, indent=1, default=str)
    digest = hashlib.sha256(text.encode()).hexdigest()
    path = os.path.join(out_dir, f"config_{digest[:12]}.json")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(text + "\n")
    return digest


class OutputLock:
    """Exclusive lock on an output directory, held for the whole run and released by the OS if the
    process dies. Two experiments writing the same directory would race on the same files."""

    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, ".lock")

    def __enter__(self):
        self.f = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close()
            raise SystemExit(f"{os.path.dirname(self.path)} is in use by another experiment process; "
                             "wait for it to finish (finished scenarios are kept and skipped on rerun).")
        return self

    def __exit__(self, *exc):
        if os.name == "nt":
            import msvcrt
            self.f.seek(0)
            msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        self.f.close()


def run_scenario(cfg, sc, reps, algos, out_dir, threads, budget, params, power_profile, seed_stream, dump_fronts,
                 config_hash="", order_seed=None):
    tmp = tempfile.mkdtemp(prefix=f"gcs_s{sc['scenario_id']}_")
    try:
        names = all_algorithms(cfg)
        for rep in reps:
            iseed = instance_seed(cfg, sc["scenario_id"], rep)
            inst = generate(cfg, sc, iseed, power_profile)
            seeds = {a: java_seed(algorithm_seed(cfg, sc["scenario_id"], rep, names.index(a))) for a in algos}
            meta = {"scenario_id": sc["scenario_id"], "rep": rep, "tasks": sc["tasks"], "vms": sc["vms"],
                    "heterogeneity": sc["heterogeneity"], "arrival": sc["arrival"],
                    "power_profile": power_profile or cfg["fixed_parameters"]["power_profile"], "seed_stream": seed_stream}
            write_case(os.path.join(tmp, f"s{sc['scenario_id']:02d}_r{rep:02d}.txt"), inst, meta=meta, seeds=seeds)
        sid = f"s{sc['scenario_id']:02d}"
        part_runs, part_trace = os.path.join(out_dir, f"{sid}.runs.part"), os.path.join(out_dir, f"{sid}.trace.part")
        part_sched = os.path.join(out_dir, f"{sid}.schedules.part")
        cmd = ["java", "-Xmx4g", "-cp", JAVA_OUT, "gcs.opt.RunExperiment", "--in", tmp, "--out", part_runs,
               "--trace", part_trace, "--algos", ",".join(algos), "--budget", str(budget), "--threads", str(threads),
               "--schedules", part_sched, "--config-hash", config_hash]
        if order_seed is not None:
            cmd += ["--order-seed", str(order_seed + sc["scenario_id"])]
        for k, v in params.items():
            cmd += ["--param", f"{k}={v}"]
        if dump_fronts:
            cmd += ["--fronts", os.path.join(out_dir, "fronts")]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        os.replace(part_sched, os.path.join(out_dir, f"{sid}_schedules.csv.gz"))
        os.replace(part_trace, os.path.join(out_dir, f"{sid}_trace.csv"))
        os.replace(part_runs, os.path.join(out_dir, f"{sid}_runs.csv"))   # last: marks the scenario as finished
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--algos", default=None, help="comma list; default = all algorithms, ablations, supplementary")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--param", action="append", default=[], help="override a search parameter, key=value")
    ap.add_argument("--power-profile", default=None)
    ap.add_argument("--tuning-seeds", action="store_true", help="use the separate tuning seed stream")
    ap.add_argument("--dump-fronts", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--order-seed", type=int, default=None,
                    help="seed of the random execution order (default: the master seed of the stream)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config) if a.config else load_config()
    if a.tuning_seeds:
        cfg["experiment"]["master_seed"] = cfg["experiment"]["tuning_seed"]
    search = dict(cfg["search"])
    budget = a.budget or search.pop("budget")
    search.pop("budget", None)
    tol = cfg["decision_rules"]
    params = {**search, "tol_makespan": tol["max_makespan_degradation"], "tol_cost": tol["max_cost_increase"],
              "tol_sla": tol["max_sla_increase_pp"] / 100.0}
    for kv in a.param:   # after the config tolerances, so that --param tol_*=... overrides them
        k, v = kv.split("=", 1)
        params[k] = v
    algos = a.algos.split(",") if a.algos else all_algorithms(cfg)
    reps = list(range(a.reps or cfg["experiment"]["replications"]))
    matrix = scenario_matrix(cfg)
    if a.order_seed is None:
        a.order_seed = cfg["experiment"]["master_seed"]
    os.makedirs(a.out, exist_ok=True)
    with OutputLock(a.out):
        compile_java()
        manifest = {"config": cfg, "algorithms": algos, "budget": budget, "params": params, "reps": len(reps),
                    "power_profile": a.power_profile or cfg["fixed_parameters"]["power_profile"],
                    "seed_stream": "tuning" if a.tuning_seeds else "test", "source_sha256": source_hash()}
        config_hash = write_manifest(a.out, manifest)
        run_all(a, cfg, matrix, reps, algos, budget, params, config_hash)


LOG_HEADER = ["scenario_id", "attempt", "status", "seconds", "budget", "algorithms", "reps", "master_seed",
              "config_hash", "order_seed", "order_position"]


def open_log(path):
    """Open run_log.csv for appending. An older log without the provenance columns gets them, empty."""
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if rows and rows[0] != LOG_HEADER:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(LOG_HEADER)
                w.writerows(r + [""] * (len(LOG_HEADER) - len(r)) for r in rows[1:])
        return open(path, "a", newline=""), False
    return open(path, "a", newline=""), True


def run_all(a, cfg, matrix, reps, algos, budget, params, config_hash=""):
    log_path = os.path.join(a.out, "run_log.csv")
    logf, new_log = open_log(log_path)
    with logf:
        log = csv.writer(logf)
        if new_log:
            log.writerow(LOG_HEADER)
        order = parse_ids(a.scenarios, len(matrix))
        random.Random(a.order_seed).shuffle(order)
        for pos, sid in enumerate(order):
            if os.path.exists(os.path.join(a.out, f"s{sid:02d}_runs.csv")):
                print(f"scenario {sid}: done, skipping")
                continue
            for attempt in (1, 2):
                t0 = time.time()
                try:
                    run_scenario(cfg, matrix[sid], reps, algos, a.out, a.threads, budget, params, a.power_profile,
                                 "tuning" if a.tuning_seeds else "test", a.dump_fronts, config_hash, a.order_seed)
                    status = "ok"
                except Exception as e:   # recorded, then retried once with the same seeds
                    status = f"failed: {e!r}"
                dt = time.time() - t0
                log.writerow([sid, attempt, status, round(dt, 1), budget, "|".join(algos), len(reps),
                              cfg["experiment"]["master_seed"], config_hash, a.order_seed, pos])
                logf.flush()
                print(f"scenario {sid} {matrix[sid]}: {status} in {dt:.1f}s", flush=True)
                if status == "ok":
                    break


if __name__ == "__main__":
    main()
