"""Replay baseline schedules in CloudSim Plus and compare with the evaluator (proposal Section 3.4).

Needs the CloudSim Plus jars in java/lib (see `make cloudsim-lib`). Covers one replicate of every
independent-task scenario with 500 or 2000 tasks; CloudSim Plus has no native precedence support,
so workflow scenarios are validated only through the Python/Java parity cases.

    python -m gcs.validate_cloudsim --out ../results/validation/cloudsim.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import subprocess
import tempfile

from .export_case import write_case
from .scenarios import instance_seed, load_config, scenario_matrix
from .workload import generate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COLS = ["schedule", "tasks", "unfinished", "max_finish_diff_s", "makespan_cloudsim", "makespan_model",
        "energy_cloudsim_j", "energy_model_j", "rel_diff_makespan", "rel_diff_energy", "agree"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "validation", "cloudsim.csv"))
    a = ap.parse_args(argv)
    cfg = load_config()
    out_sim = os.path.join(ROOT, "java", "out-sim")
    src = [os.path.join(ROOT, "java", "src", "main", "java", "gcs", d, f)
           for d in ("core", "sim") for f in os.listdir(os.path.join(ROOT, "java", "src", "main", "java", "gcs", d))]
    cp = os.pathsep.join([out_sim, os.path.join(ROOT, "java", "lib", "*")])
    subprocess.run(["javac", "-d", out_sim, "-cp", os.path.join(ROOT, "java", "lib", "*")] + src, check=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for sc in scenario_matrix(cfg):
            if sc["arrival"] == "workflow" or sc["tasks"] > 2000:
                continue
            path = os.path.join(tmp, f"s{sc['scenario_id']}.txt")
            write_case(path, generate(cfg, sc, instance_seed(cfg, sc["scenario_id"], 0)))
            res = subprocess.run(["java", "-cp", cp, "gcs.sim.CloudSimValidation", path], capture_output=True, text=True)
            for line in res.stdout.splitlines():
                if line.startswith("CSV,"):
                    rows.append({**{k: sc[k] for k in ("scenario_id", "tasks", "vms", "heterogeneity", "arrival")},
                                 **dict(zip(COLS, line.split(",")[1:]))})
            print(sc, "agree" if res.returncode == 0 else "DISAGREE", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} comparisons written to {a.out}")


if __name__ == "__main__":
    main()
