"""Parameter tuning on development scenarios with the separate tuning seed stream (proposal Section 3.2).

Each candidate configuration is run on the same development instances; the score is the median, over
instances, of the selected schedule's energy divided by the reference baseline's energy (1.0 when the
selected schedule is outside the tolerance envelope). The best configuration is then frozen in
config/experiment.yaml by hand, before any test-seed run.

    python -m gcs.tune --algo hpeas --out ../results/tuning
    python -m gcs.tune --algo ga --out ../results/tuning
"""
from __future__ import annotations
import argparse
import glob
import itertools
import os

import pandas as pd

from . import experiment
from .scenarios import load_config, scenario_matrix

DEV_SCENARIOS = [  # (tasks, vms, heterogeneity, arrival): one per arrival type and size, mixed heterogeneity
    (500, 20, "medium", "bursty"), (2000, 50, "high", "steady"), (2000, 50, "low", "workflow"),
    (5000, 100, "medium", "bursty"), (500, 50, "high", "workflow"), (5000, 20, "low", "steady")]

GRIDS = {
    "hpeas": {"pm": [1, 5, 20], "pc": [0.6, 0.9], "stagnation": [1, 3, 6], "ls_steps": [10, 20, 50]},
    "ga": {"ga_pm": [1, 5, 20], "ga_pc": [0.6, 0.9]},
}


def dev_ids(cfg) -> list[int]:
    m = scenario_matrix(cfg)
    return [next(s["scenario_id"] for s in m if (s["tasks"], s["vms"], s["heterogeneity"], s["arrival"]) == d)
            for d in DEV_SCENARIOS]


def energy_ratio(df: pd.DataFrame, algo: str) -> pd.Series:
    """Selected energy / reference energy per instance (reference = faster of FCFS and Min-Min)."""
    key = ["scenario_id", "rep"]
    base = df[df.algorithm.isin(["fcfs", "minmin"])].sort_values("makespan").groupby(key).first()
    sel = df[(df.algorithm == algo) & (df.selection.isin(["tcme", "single"]))].set_index(key)
    r = sel.energy / base.energy.reindex(sel.index)
    return r.where(sel.feasible == 1, 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="hpeas", choices=sorted(GRIDS))
    ap.add_argument("--out", default="../results/tuning")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)
    cfg = load_config()
    ids = ",".join(map(str, dev_ids(cfg)))
    grid = GRIDS[a.algo]
    rows = []
    for combo in itertools.product(*grid.values()):
        params = dict(zip(grid, combo))
        tag = a.algo + "_" + "_".join(f"{k}{v}" for k, v in params.items())
        out = os.path.join(a.out, tag)
        argv2 = ["--out", out, "--scenarios", ids, "--reps", str(a.reps), "--algos", f"fcfs,minmin,{a.algo}",
                 "--tuning-seeds"] + [x for k, v in params.items() for x in ("--param", f"{k}={v}")]
        experiment.main(argv2)
        df = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(out, "*_runs.csv")))
        r = energy_ratio(df, a.algo)
        rows.append({**params, "median_ratio": r.median(), "mean_ratio": r.mean(), "n": len(r)})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows).sort_values(["median_ratio", "mean_ratio"])
    res.to_csv(os.path.join(a.out, f"tuning_{a.algo}.csv"), index=False)
    print(res.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
