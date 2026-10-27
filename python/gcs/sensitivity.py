"""Sensitivity, scalability and construct-validity runs (proposal Sections 3.2, 3.5 and 3.6).

One-at-a-time variations of HPEAS settings around the frozen configuration, on a subset of nine
scenarios (every size x arrival pattern, 50 VMs, medium heterogeneity) with 10 test replicates; the
tolerance thresholds are varied the same way; and the main comparison is repeated under an
energy-proportional power curve on all 81 scenarios with 10 replicates.

    python -m gcs.sensitivity --out ../results/sensitivity            # run
    python -m gcs.sensitivity --out ../results/sensitivity --analyse  # tables and figures
"""
from __future__ import annotations
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import experiment
from .analysis import HIGHLIGHT, INK, MAIN, MUTED, deployed, load, summary_table
from .scenarios import load_config, scenario_matrix

SUBSET = [(t, 50, "medium", a) for t in (500, 2000, 5000) for a in ("steady", "bursty", "workflow")]
VARIANTS = {   # name -> extra experiment arguments
    "default": [],
    "pop20": ["--param", "pop=20"], "pop80": ["--param", "pop=80"],
    "archive50": ["--param", "archive=50"], "archive200": ["--param", "archive=200"],
    "stagnation3": ["--param", "stagnation=3"], "stagnation6": ["--param", "stagnation=6"],
    "budget2500": ["--budget", "2500"], "budget5000": ["--budget", "5000"], "budget20000": ["--budget", "20000"],
    "tol_tight": ["--param", "tol_makespan=0.025", "--param", "tol_cost=0.025", "--param", "tol_sla=0.005"],
    "tol_loose": ["--param", "tol_makespan=0.10", "--param", "tol_cost=0.10", "--param", "tol_sla=0.02"],
}
GROUPS = [("Population", ["pop20", "default", "pop80"], ["20", "40*", "80"]),
          ("Archive", ["archive50", "default", "archive200"], ["50", "100*", "200"]),
          ("LS trigger", ["default", "stagnation3", "stagnation6"], ["1*", "3", "6"]),
          ("Budget", ["budget2500", "budget5000", "default", "budget20000"], ["2.5k", "5k", "10k*", "20k"]),
          ("Tolerance", ["tol_tight", "default", "tol_loose"], ["2.5%", "5%*", "10%"])]


def subset_ids(cfg):
    m = scenario_matrix(cfg)
    return [s["scenario_id"] for s in m if (s["tasks"], s["vms"], s["heterogeneity"], s["arrival"]) in SUBSET]


def run(out):
    cfg = load_config()
    ids = ",".join(map(str, subset_ids(cfg)))
    for name, extra in VARIANTS.items():
        experiment.main(["--out", os.path.join(out, name), "--scenarios", ids, "--reps", "10",
                         "--algos", "fcfs,minmin,hpeas"] + extra)
    experiment.main(["--out", os.path.join(out, "power_proportional"), "--reps", "10", "--algos", ",".join(MAIN),
                     "--power-profile", "proportional"])


def analyse(out, figures):
    rows = []
    for name, extra in VARIANTS.items():
        d = deployed(load(os.path.join(out, name)))
        if name.startswith("tol_"):   # judge acceptance against the variant's own envelope
            mk = 2.5 if name == "tol_tight" else 10.0
            sla = 0.5 if name == "tol_tight" else 2.0
            d["accepted"] = (d.makespan_chg <= mk + 1e-9) & (d.cost_chg <= mk + 1e-9) & (d.sla_chg_pp <= sla + 1e-9)
            d["saving_accepted"] = d.accepted & (d.energy_chg < 0)
        h = d[d.algorithm == "hpeas"]
        rows.append({"variant": name, "median_energy_chg": h.energy_chg.median(),
                     "q1": h.energy_chg.quantile(0.25), "q3": h.energy_chg.quantile(0.75),
                     "median_makespan_chg": h.makespan_chg.median(), "accepted_saving": 100 * h.saving_accepted.mean(),
                     "median_hv": h.hv.median(), "median_runtime_s": h.runtime_s.median(), "n": len(h)})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "table_sensitivity.csv"), index=False)
    print(t.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, len(GROUPS), figsize=(7.2, 2.6), sharey=True,
                             gridspec_kw={"width_ratios": [len(g[1]) for g in GROUPS]})
    for ax, (title, names, labels) in zip(axes, GROUPS):
        s = t.set_index("variant").loc[names]
        x = range(len(names))
        ax.vlines(x, s.q1, s.q3, color=MUTED, linewidth=1.2)
        ax.plot(x, s.median_energy_chg, marker="o", color=HIGHLIGHT, linewidth=1.5, markersize=5)
        ax.set_xticks(list(x), labels, fontsize=7.5)
        ax.set_xlim(-0.5, len(names) - 0.5)
        ax.set_title(title, fontsize=8.5, loc="left", color=INK)
        ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("HPEAS energy change (%)\nmedian and IQR")
    fig.savefig(os.path.join(figures, "fig_sensitivity.png"))
    plt.close(fig)

    p = deployed(load(os.path.join(out, "power_proportional")))
    tp = summary_table(p, MAIN)
    tp.to_csv(os.path.join(out, "table_power_proportional.csv"), index=False)
    print("\n== energy-proportional power curve\n" + tp.to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/sensitivity")
    ap.add_argument("--figures", default="../report/figures")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args(argv)
    if a.analyse:
        os.makedirs(a.figures, exist_ok=True)
        analyse(a.out, a.figures)
    else:
        run(a.out)


if __name__ == "__main__":
    main()
