"""Run the deterministic baselines on one scenario and write a CSV of objectives.

    python -m gcs.run_baselines --scenario 0 --reps 3 --out ../results/demo_baselines.csv
"""
from __future__ import annotations
import argparse
import csv
import time

from .baselines import BASELINES
from .model import OBJECTIVE_NAMES, evaluate
from .scenarios import instance_seed, load_config, scenario_matrix
from .workload import generate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="../results/demo_baselines.csv")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config) if a.config else load_config()
    sc = scenario_matrix(cfg)[a.scenario]
    rows = []
    for rep in range(a.reps):
        inst = generate(cfg, sc, instance_seed(cfg, sc["scenario_id"], rep))
        for name, fn in BASELINES.items():
            t0 = time.perf_counter()
            assign = fn(inst)
            runtime = time.perf_counter() - t0
            obj = evaluate(inst, assign)
            rows.append({**{k: sc[k] for k in ("scenario_id", "tasks", "vms", "heterogeneity", "arrival")},
                         "rep": rep, "algorithm": name, **obj, "optimiser_runtime_s": runtime})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"scenario {sc} -> {len(rows)} rows written to {a.out}")
    for r in rows[: len(BASELINES)]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in OBJECTIVE_NAMES + ("algorithm",)})


if __name__ == "__main__":
    main()
