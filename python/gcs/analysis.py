"""Statistical analysis and figures for the report (proposal Section 3.5).

Every algorithm's deployed schedule is compared, instance by instance, with the reference baseline
(the faster of FCFS and Min-Min on that instance). Changes are relative (energy, makespan, cost) or in
percentage points (SLA). "Accepted" means inside the tolerance envelope: makespan +5%, cost +5%,
SLA +1 pp. Statistics: Friedman across algorithms, Holm-corrected Wilcoxon signed-rank against HPEAS,
Cliff's delta, bootstrap 95% CIs of medians.

    python -m gcs.analysis --main ../results/main --out ../results/analysis --figures ../report/figures
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .scenarios import load_config
from .stats import bootstrap_ci, cliffs_delta, friedman, holm_adjust
from scipy import stats as sps

MAIN = ["fcfs", "minmin", "ga", "mogwo", "hpeas"]
ABLATION = ["hpeas", "hpeas_noseed", "hpeas_nols", "hpeas_static", "nsga2"]
FAIRNESS = ["hpeas", "ga_seeded", "mogwo_seeded", "energy_greedy"]
LABEL = {"fcfs": "FCFS", "minmin": "Min-Min", "ga": "GA", "mogwo": "MOGWO", "hpeas": "HPEAS",
         "hpeas_noseed": "HPEAS − seeding", "hpeas_nols": "HPEAS − local search",
         "hpeas_static": "HPEAS static LS", "nsga2": "Global search only", "ga_seeded": "GA + seeding",
         "mogwo_seeded": "MOGWO + seeding", "energy_greedy": "Energy-greedy"}
INSTANCE = ["scenario_id", "rep"]

# reference palette (dataviz skill): HPEAS highlighted in slot-1 blue, everything else neutral ink
HIGHLIGHT, NEUTRAL, INK, MUTED, GRID = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.dpi": 150,
                     "savefig.bbox": "tight", "legend.frameon": False})


def colour(a):
    return HIGHLIGHT if a == "hpeas" else NEUTRAL


def load(dirpath: str, kind: str = "runs") -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(dirpath, f"s*_{kind}.csv")))
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def deployed(df: pd.DataFrame, selection: str = "tcme") -> pd.DataFrame:
    """One row per (instance, algorithm): the schedule each method would deploy, with changes vs the reference."""
    d = df[df.selection.isin([selection, "single"])].copy()
    base = (df[df.algorithm.isin(["fcfs", "minmin"])].sort_values(["makespan", "algorithm"])
            .groupby(INSTANCE).first()[["energy", "makespan", "cost", "sla", "imbalance"]].add_prefix("ref_"))
    d = d.join(base, on=INSTANCE)
    d["energy_chg"] = 100 * (d.energy / d.ref_energy - 1)
    d["makespan_chg"] = 100 * (d.makespan / d.ref_makespan - 1)
    d["cost_chg"] = 100 * (d.cost / d.ref_cost - 1)
    d["sla_chg_pp"] = 100 * (d.sla - d.ref_sla)
    d["accepted"] = (d.makespan_chg <= 5 + 1e-9) & (d.cost_chg <= 5 + 1e-9) & (d.sla_chg_pp <= 1 + 1e-9)
    d["saving_accepted"] = d.accepted & (d.energy_chg < 0)
    return d


def med_iqr(s: pd.Series, fmt="{:.2f}") -> str:
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{fmt.format(q2)} [{fmt.format(q1)}, {fmt.format(q3)}]"


def summary_table(d: pd.DataFrame, algos) -> pd.DataFrame:
    rows = []
    for a in algos:
        x = d[d.algorithm == a]
        if x.empty:
            continue
        rows.append({"Algorithm": LABEL[a], "Energy change %": med_iqr(x.energy_chg),
                     "Makespan change %": med_iqr(x.makespan_chg), "Cost change %": med_iqr(x.cost_chg),
                     "SLA change (pp)": med_iqr(x.sla_chg_pp), "Imbalance (CV)": med_iqr(x.imbalance, "{:.3f}"),
                     "Within tolerance %": f"{100 * x.accepted.mean():.1f}",
                     "Accepted saving %": f"{100 * x.saving_accepted.mean():.1f}",
                     "HV": med_iqr(x.hv, "{:.3f}"), "IGD": med_iqr(x.igd, "{:.3f}"),
                     "Runtime s": med_iqr(x.runtime_s, "{:.2f}"), "n": len(x)})
    return pd.DataFrame(rows)


def paired(d: pd.DataFrame, metric: str, algos) -> pd.DataFrame:
    return d[d.algorithm.isin(algos)].pivot_table(index=INSTANCE, columns="algorithm", values=metric).dropna()[algos]


def tests_vs(d: pd.DataFrame, metric: str, ref: str, others) -> pd.DataFrame:
    """Wilcoxon signed-rank (Holm-corrected) and Cliff's delta of `ref` against each other algorithm."""
    w = paired(d, metric, [ref] + list(others))
    raw, rows = [], []
    for o in others:
        diff = w[ref] - w[o]
        p = 1.0 if np.allclose(diff, 0) else float(sps.wilcoxon(w[ref], w[o]).pvalue)
        raw.append(p)
        rows.append({"comparison": f"{LABEL[ref]} vs {LABEL[o]}", "metric": metric,
                     "median difference": float(np.median(diff)),
                     "HPEAS better in %": 100 * float(((diff > 0) if metric == "hv" else (diff < 0)).mean()),
                     "cliffs_delta": cliffs_delta(w[ref].values, w[o].values), "p_raw": p})
    for r, p in zip(rows, holm_adjust(raw)):
        r["p_holm"] = float(p)
    return pd.DataFrame(rows)


def magnitude(delta: float) -> str:
    a = abs(delta)   # Romano et al. (2006) thresholds
    return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"


# ---------------------------------------------------------------- figures

def fig_box(d, metric, algos, ylabel, path, clip=None, ref_line=0.0, title=None):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    data = [d.loc[d.algorithm == a, metric].clip(*(clip or (None, None))).values for a in algos]
    bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True, medianprops={"color": INK, "linewidth": 1.4},
                    whiskerprops={"color": MUTED}, capprops={"color": MUTED}, boxprops={"linewidth": 0.8})
    for patch, a in zip(bp["boxes"], algos):
        patch.set_facecolor(colour(a)); patch.set_alpha(0.55 if a != "hpeas" else 0.85); patch.set_edgecolor(MUTED)
    ax.set_xticks(range(1, len(algos) + 1), [LABEL[a] for a in algos], rotation=0 if len(algos) <= 6 else 20)
    if ref_line is not None:
        ax.axhline(ref_line, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.grid(axis="x", visible=False)
    fig.savefig(path); plt.close(fig)


def fig_tradeoff(d, algos, path):
    """Median energy vs makespan change per scenario: where each method sits relative to the envelope."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    g = d[d.algorithm.isin(algos)].groupby(["algorithm", "scenario_id"])[["energy_chg", "makespan_chg"]].median().reset_index()
    markers = {"fcfs": "s", "minmin": "D", "ga": "^", "mogwo": "v", "hpeas": "o"}
    ax.axvspan(-100, 5, color=HIGHLIGHT, alpha=0.06, linewidth=0)
    for a in algos:
        x = g[g.algorithm == a]
        ax.scatter(x.makespan_chg.clip(-30, 200), x.energy_chg.clip(-40, 200), s=18 if a == "hpeas" else 14,
                   marker=markers.get(a, "o"), facecolor=colour(a) if a == "hpeas" else "none",
                   edgecolor=colour(a), linewidth=0.9, label=LABEL[a], zorder=3 if a == "hpeas" else 2)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--"); ax.axvline(5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xscale("symlog", linthresh=10); ax.set_yscale("symlog", linthresh=10)
    ax.set_xlim(left=-12)
    ax.set_xlabel("Makespan change vs reference (%), symlog"); ax.set_ylabel("Energy change vs reference (%), symlog")
    ax.text(4, ax.get_ylim()[1] * 0.6, "makespan tolerance +5%", color=MUTED, fontsize=7, ha="right")
    ax.legend(loc="upper left", ncol=5, fontsize=7.5, handletextpad=0.3, columnspacing=1.0, bbox_to_anchor=(0, 1.13))
    fig.savefig(path); plt.close(fig)


def fig_factors(d, path):
    """HPEAS accepted energy change by each scenario factor (small multiples, one axis scale)."""
    h = d[d.algorithm == "hpeas"]
    factors = [("tasks", "Tasks"), ("vms", "VMs"), ("heterogeneity", "Heterogeneity"), ("arrival", "Arrival")]
    order = {"heterogeneity": ["low", "medium", "high"], "arrival": ["steady", "bursty", "workflow"]}
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.6), sharey=True)
    for ax, (f, name) in zip(axes, factors):
        levels = order.get(f, sorted(h[f].unique()))
        data = [h.loc[h[f] == l, "energy_chg"].values for l in levels]
        bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                        medianprops={"color": INK, "linewidth": 1.3}, whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        for p in bp["boxes"]:
            p.set_facecolor(HIGHLIGHT); p.set_alpha(0.75); p.set_edgecolor(MUTED)
        ax.set_xticks(range(1, len(levels) + 1), [str(l) for l in levels], fontsize=8)
        ax.set_title(name, fontsize=9, loc="left", color=INK); ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("HPEAS energy change (%)")
    fig.savefig(path); plt.close(fig)


def fig_convergence(trace, deployed_df, algos, path):
    """Median best in-envelope energy (relative to the reference) against the share of the evaluation budget."""
    ref = deployed_df[deployed_df.algorithm == "hpeas"].set_index(INSTANCE).ref_energy
    t = trace[trace.algorithm.isin(algos)].join(ref, on=INSTANCE)
    cols = [c for c in trace.columns if c.startswith("t") and c[1:].isdigit()]
    x = np.arange(1, len(cols) + 1) / len(cols) * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    styles = {"hpeas": "-", "hpeas_nols": "--", "hpeas_noseed": ":", "nsga2": "-.", "ga_seeded": (0, (5, 2, 1, 2)),
              "mogwo_seeded": (0, (1, 1))}
    for a in algos:
        r = t[t.algorithm == a]
        rel = r[cols].div(r.ref_energy, axis=0).replace(np.inf, np.nan)
        y = 100 * (rel.median(skipna=True) - 1)
        ax.plot(x, y.values, linestyle=styles.get(a, "-"), color=HIGHLIGHT if a == "hpeas" else MUTED,
                linewidth=2 if a == "hpeas" else 1.4, label=LABEL[a])
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Share of evaluation budget used (%)"); ax.set_ylabel("Best in-envelope energy\nvs reference (%)")
    ax.legend(fontsize=7.5, ncol=2)
    fig.savefig(path); plt.close(fig)


def fig_runtime(d, algos, path):
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    g = d[d.algorithm.isin(algos)].groupby(["algorithm", "tasks"]).runtime_s.median().unstack(0)
    # end labels, spread apart vertically (in decades) where the lines finish close together
    ends = sorted(algos, key=lambda a: g[a].iloc[-1])
    ys = [np.log10(g[a].iloc[-1]) for a in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.18)
    label_y = {a: 10 ** y for a, y in zip(ends, ys)}
    for a in algos:
        ax.plot(g.index, g[a], marker="o", markersize=4, color=HIGHLIGHT if a == "hpeas" else MUTED,
                linewidth=2 if a == "hpeas" else 1.2, linestyle="-" if a in ("hpeas", "fcfs") else "--")
        ax.annotate(LABEL[a], (g.index[-1] * 1.08, label_y[a]), fontsize=7.5, color=INK, va="center")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(g.index, [str(i) for i in g.index])
    ax.set_xlabel("Tasks"); ax.set_ylabel("Median wall-clock time (s)")
    ax.set_xlim(g.index[0] * 0.8, g.index[-1] * 2.2)
    fig.savefig(path); plt.close(fig)


def fig_front(front_dir, stem, algos, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    markers = {"mogwo": "v", "ga": "^", "fcfs": "s", "minmin": "D", "hpeas": "o", "energy_greedy": "P"}
    for a in algos:
        f = os.path.join(front_dir, f"{stem}_{a}.csv")
        if not os.path.exists(f):
            continue
        x = pd.read_csv(f)
        ax.scatter(x.makespan, x.energy / 1e6, s=26 if a in ("fcfs", "minmin", "energy_greedy") else 12,
                   marker=markers.get(a, "o"), facecolor=HIGHLIGHT if a == "hpeas" else "none",
                   edgecolor=HIGHLIGHT if a == "hpeas" else INK if a in ("fcfs", "minmin", "energy_greedy") else NEUTRAL,
                   linewidth=0.9, label=LABEL[a], alpha=0.9)
    ax.set_xlabel("Makespan (s)"); ax.set_ylabel("Energy (MJ)")
    ax.legend(fontsize=7.5, ncol=3)
    fig.savefig(path); plt.close(fig)

# ---------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="../results/main")
    ap.add_argument("--out", default="../results/analysis")
    ap.add_argument("--figures", default="../report/figures")
    ap.add_argument("--front", default=None, help="dir with dumped fronts for the example-front figure")
    ap.add_argument("--front-stem", default=None)
    a = ap.parse_args(argv)
    cfg = load_config()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.figures, exist_ok=True)
    runs, trace = load(a.main, "runs"), load(a.main, "trace")
    d = deployed(runs)
    d.to_csv(os.path.join(a.out, "deployed.csv"), index=False)
    res = {"instances": int(d[INSTANCE].drop_duplicates().shape[0])}

    tables = {"main": summary_table(d, MAIN), "ablation": summary_table(d, ABLATION),
              "fairness": summary_table(d, FAIRNESS), "knee": summary_table(deployed(runs, "knee"), ["mogwo", "hpeas"] + ABLATION[1:])}
    for name, t in tables.items():
        t.to_csv(os.path.join(a.out, f"table_{name}.csv"), index=False)

    # omnibus and pairwise tests on energy change and HV, main comparison
    w = paired(d, "energy_chg", MAIN)
    res["friedman_energy"] = dict(zip(["chi2", "p"], friedman({k: w[k].values for k in MAIN})))
    w = paired(d, "hv", MAIN)
    res["friedman_hv"] = dict(zip(["chi2", "p"], friedman({k: w[k].values for k in MAIN})))
    pw = pd.concat([tests_vs(d, m, "hpeas", [x for x in MAIN if x != "hpeas"]) for m in ("energy_chg", "makespan_chg", "hv")]
                   + [tests_vs(d, m, "hpeas", ABLATION[1:] + FAIRNESS[1:]) for m in ("energy_chg", "hv")])
    pw["effect"] = pw.cliffs_delta.map(magnitude)
    pw.to_csv(os.path.join(a.out, "pairwise_tests.csv"), index=False)

    # HPEAS headline: median energy change with bootstrap CI, acceptance, per-scenario significance
    h = d[d.algorithm == "hpeas"]
    res["hpeas_energy_median"] = float(h.energy_chg.median())
    res["hpeas_energy_ci"] = bootstrap_ci(h.energy_chg.values, resamples=cfg["decision_rules"]["bootstrap_resamples"])
    res["hpeas_within_tolerance"] = float(h.accepted.mean())
    res["hpeas_accepted_saving"] = float(h.saving_accepted.mean())
    per = []
    for sid, g in d[d.algorithm.isin(["hpeas"])].groupby("scenario_id"):
        diff = g.energy - g.ref_energy
        p = 1.0 if np.allclose(diff, 0) else float(sps.wilcoxon(g.energy, g.ref_energy, alternative="less").pvalue)
        b = bootstrap_ci(g.energy_chg.values, resamples=2000, seed=sid)
        per.append({**g.iloc[0][["scenario_id", "tasks", "vms", "heterogeneity", "arrival"]].to_dict(),
                    "median_energy_chg": g.energy_chg.median(), "ci_lo": b[0], "ci_hi": b[1],
                    "median_makespan_chg": g.makespan_chg.median(), "median_sla_chg_pp": g.sla_chg_pp.median(),
                    "within_tolerance": g.accepted.mean(), "p_raw": p})
    per = pd.DataFrame(per)
    per["p_holm"] = holm_adjust(per.p_raw.values)
    per["significant"] = per.p_holm < cfg["decision_rules"]["alpha"]
    per.to_csv(os.path.join(a.out, "hpeas_per_scenario.csv"), index=False)
    res["scenarios_significant_saving"] = int(per.significant.sum())
    res["scenarios_total"] = len(per)
    for f in ("tasks", "vms", "heterogeneity", "arrival"):
        res[f"hpeas_energy_by_{f}"] = h.groupby(f).energy_chg.median().round(3).to_dict()
        res[f"hpeas_accepted_by_{f}"] = h.groupby(f).saving_accepted.mean().round(3).to_dict()

    # carbon: secondary scenario with fixed intensities (estimated, not measured emissions)
    ci = cfg["decision_rules"]["carbon_intensity_g_per_kwh"]
    kwh_ref, kwh_h = h.ref_energy / 3.6e6, h.energy / 3.6e6
    res["carbon_g_per_run"] = {k: {"reference": float(np.median(kwh_ref * v)), "hpeas": float(np.median(kwh_h * v)),
                                   "saving": float(np.median((kwh_ref - kwh_h) * v))} for k, v in ci.items()}
    res["energy_kwh_median"] = {"reference": float(kwh_ref.median()), "hpeas": float(kwh_h.median())}

    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)

    # figures
    fig_box(d, "energy_chg", MAIN + ["energy_greedy"], "Energy change vs reference (%)",
            os.path.join(a.figures, "fig_energy_box.png"), clip=(-60, 400))
    fig_box(d, "hv", MAIN, "Hypervolume (normalised)", os.path.join(a.figures, "fig_hv_box.png"), ref_line=None)
    fig_tradeoff(d, MAIN, os.path.join(a.figures, "fig_tradeoff.png"))
    fig_factors(d, os.path.join(a.figures, "fig_factors.png"))
    fig_box(d, "energy_chg", ABLATION + ["ga_seeded", "mogwo_seeded"], "Energy change vs reference (%)",
            os.path.join(a.figures, "fig_ablation_box.png"), clip=(-60, 400))
    fig_convergence(trace, d, ["hpeas", "hpeas_nols", "hpeas_noseed", "nsga2", "ga_seeded", "mogwo_seeded"],
                    os.path.join(a.figures, "fig_convergence.png"))
    fig_runtime(d, MAIN, os.path.join(a.figures, "fig_runtime.png"))
    if a.front and a.front_stem:
        fig_front(a.front, a.front_stem, ["mogwo", "hpeas", "ga", "fcfs", "minmin", "energy_greedy"],
                  os.path.join(a.figures, "fig_front.png"))
    for name, t in tables.items():
        print(f"\n== {name}\n" + t.to_string(index=False))
    print("\n" + pw.round(4).to_string(index=False))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
