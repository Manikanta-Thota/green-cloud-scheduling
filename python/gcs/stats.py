"""Statistical analysis (proposal Section 3.5): Friedman, Holm-corrected Wilcoxon, Cliff's delta,
bootstrap CIs and the practical-tolerance decision rule."""
from __future__ import annotations
import numpy as np
from scipy import stats


def friedman(samples: dict[str, np.ndarray]):
    """samples: algorithm -> metric values, aligned by scenario/replicate. Returns (statistic, p)."""
    res = stats.friedmanchisquare(*samples.values())
    return float(res.statistic), float(res.pvalue)


def holm_adjust(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def pairwise_wilcoxon_holm(samples: dict[str, np.ndarray]) -> list[dict]:
    names = list(samples)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    raw = []
    for a, b in pairs:
        diff = np.asarray(samples[a]) - np.asarray(samples[b])
        raw.append(1.0 if np.all(diff == 0) else float(stats.wilcoxon(samples[a], samples[b]).pvalue))
    adj = holm_adjust(raw)
    return [{"a": a, "b": b, "p_raw": r, "p_holm": float(h)} for (a, b), r, h in zip(pairs, raw, adj)]


def cliffs_delta(x, y) -> float:
    """P(x > y) - P(x < y); +1 means x is always larger."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def bootstrap_ci(values, stat=np.median, resamples=10000, alpha=0.05, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(resamples, len(v)))
    boots = stat(v[idx], axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def practical_acceptance(candidate: dict, baseline: dict, rules: dict) -> dict:
    """Section 3.5 tolerances applied to per-algorithm medians (dicts with makespan, cost, sla as a fraction)."""
    mk = candidate["makespan"] / baseline["makespan"] - 1.0
    co = candidate["cost"] / baseline["cost"] - 1.0
    sla_pp = (candidate["sla"] - baseline["sla"]) * 100.0
    ok = (mk <= rules["max_makespan_degradation"] and co <= rules["max_cost_increase"]
          and sla_pp <= rules["max_sla_increase_pp"])
    return {"makespan_change": mk, "cost_change": co, "sla_change_pp": sla_pp, "within_tolerance": bool(ok)}
