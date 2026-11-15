"""Seeded workload generator (proposal Section 3.3).

Both families come from published datasets (protocol D15, D16); nothing is generated except the
simulated data centre, the deadlines and the random choices:
* independent tasks: windows of consecutive jobs from two cleaned Parallel Workloads Archive logs,
  HPC2N-2002-2.2-cln and NASA-iPSC-1993-3.1-cln (Feitelson et al., 2014), listed with SHA-256 in
  data/joblogs_manifest.csv. A job's length is its recorded runtime; its arrival is its recorded submit
  time, rescaled in time to the configured load. "Steady" and "bursty" are the windows whose
  inter-arrival times are least and most irregular (lowest and highest third of the coefficient of
  variation over all windows of that size in that log);
* scientific workflows taken from published execution traces: real Pegasus runs of Montage,
  Epigenomics, Seismology and 1000Genome on the Chameleon testbed from the WfInstances repository of
  the WfCommons project (Coleman et al., 2022; LGPL v3.0). The 73 trace files, their commit and
  SHA-256 are listed in data/wfinstances_manifest.csv and fetched by python/fetch_traces.py
  (protocol D15). Tasks and dependencies are used unchanged; recorded runtimes become task lengths.

Infrastructure: VMs with heterogeneous MIPS packed onto hosts; host power figures come from the
configured power profile; VM price grows super-linearly with speed so that cost depends on placement.
"""
from __future__ import annotations
import csv
import json
import math
import os
from functools import lru_cache
import numpy as np
from .model import Instance

# ---------------------------------------------------------------- published workflow traces

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACE_DIR = os.path.join(ROOT, "data", "wfinstances")
TRACE_MANIFEST = os.path.join(ROOT, "data", "wfinstances_manifest.csv")
LOG_DIR = os.path.join(ROOT, "data", "joblogs")
LOG_MANIFEST = os.path.join(ROOT, "data", "joblogs_manifest.csv")


@lru_cache(maxsize=None)
def trace_catalogue() -> dict:
    """{application: [(file, n_tasks), ...]} from the committed manifest, sorted for reproducibility."""
    cat = {}
    with open(TRACE_MANIFEST, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cat.setdefault(r["application"], []).append((r["file"], int(r["tasks"])))
    return {app: sorted(v) for app, v in sorted(cat.items())}


@lru_cache(maxsize=None)
def load_trace(app: str, name: str, min_runtime_s: float) -> tuple:
    """(runtimes in seconds, predecessor lists) of one WfFormat trace, tasks in topological order.

    Runtimes recorded as 0 (below the measurement resolution) are raised to `min_runtime_s`."""
    path = os.path.join(TRACE_DIR, app, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run python/fetch_traces.py first")
    with open(path, encoding="utf-8") as f:
        wf = json.load(f)["workflow"]
    tasks = wf["specification"]["tasks"]
    runtime = {t["id"]: float(t["runtimeInSeconds"]) for t in wf["execution"]["tasks"]}
    key = {t["id"]: t["id"] for t in tasks}
    key.update({t["name"]: t["id"] for t in tasks})
    idx = {t["id"]: i for i, t in enumerate(tasks)}
    parents = [[idx[key[q]] for q in t.get("parents", [])] for t in tasks]
    # Kahn's algorithm, ties broken by file order, so predecessors always come first
    n = len(tasks)
    indeg = [len(ps) for ps in parents]
    children = [[] for _ in range(n)]
    for i, ps in enumerate(parents):
        for q in ps:
            children[q].append(i)
    ready, order = [i for i in range(n) if indeg[i] == 0], []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for c in children[i]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    if len(order) != n:
        raise ValueError(f"{name}: dependency cycle")
    pos = {old: new for new, old in enumerate(order)}
    w = tuple(max(runtime[tasks[i]["id"]], min_runtime_s) for i in order)
    preds = tuple(tuple(sorted(pos[q] for q in parents[i])) for i in order)
    return w, preds


# ---------------------------------------------------------------- published job logs

@lru_cache(maxsize=None)
def log_catalogue() -> tuple:
    """((log name, file), ...) from the committed manifest, sorted for reproducibility."""
    with open(LOG_MANIFEST, encoding="utf-8") as f:
        return tuple(sorted((r["log"], r["file"]) for r in csv.DictReader(f)))


@lru_cache(maxsize=None)
def job_log(log: str) -> tuple:
    """(submit times, runtimes) in seconds of the jobs of one Standard Workload Format log whose recorded
    runtime is positive, in submit order."""
    import gzip
    name = dict(log_catalogue())[log]
    path = os.path.join(LOG_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: download it as described in python/fetch_traces.py")
    with gzip.open(path, "rt", encoding="latin-1") as f:
        rows = [line.split() for line in f if line.strip() and not line.startswith(";")]
    submit = np.array([float(r[1]) for r in rows])
    run = np.array([float(r[3]) for r in rows])
    keep = run > 0
    order = np.argsort(submit[keep], kind="stable")
    return submit[keep][order], run[keep][order]


@lru_cache(maxsize=None)
def window_starts(log: str, n: int, arrival: str, q_lo: float, q_hi: float) -> np.ndarray:
    """Start indices of the n-job windows of `log` whose inter-arrival coefficient of variation lies in the
    lowest (steady) or highest (bursty) part of its distribution over all n-job windows of that log."""
    submit, _ = job_log(log)
    gaps = np.diff(submit)
    c1 = np.concatenate([[0.0], np.cumsum(gaps)])
    c2 = np.concatenate([[0.0], np.cumsum(gaps * gaps)])
    starts = np.arange(len(submit) - n + 1)
    m = n - 1                                               # gaps inside a window
    s1, s2 = c1[starts + m] - c1[starts], c2[starts + m] - c2[starts]
    mean = s1 / m
    var = np.maximum(s2 / m - mean * mean, 0.0)
    cv = np.sqrt(var) / np.where(mean > 0, mean, np.inf)
    lo, hi = np.quantile(cv, [q_lo, q_hi])
    return starts[cv <= lo] if arrival == "steady" else starts[cv >= hi]


def _depth_and_path(weights, preds):
    """DAG depth of each task and the heaviest path weight ending at (and including) it."""
    n = len(weights)
    depth = np.zeros(n, dtype=int)
    path = np.zeros(n)
    for t in range(n):          # tasks are in topological order
        if preds[t]:
            depth[t] = 1 + max(depth[p] for p in preds[t])
            path[t] = weights[t] + max(path[p] for p in preds[t])
        else:
            path[t] = weights[t]
    return depth, path

# ---------------------------------------------------------------- instance generation

def _infrastructure(cfg, scenario, rng, power_profile=None):
    p = cfg["fixed_parameters"]
    m = scenario["vms"]
    lo, hi = p["heterogeneity_mips_range"][scenario["heterogeneity"]]
    mips = rng.uniform(lo, hi, size=m)
    n_hosts = math.ceil(m / p["vms_per_host"])
    host_of = np.arange(m) // p["vms_per_host"]
    price = p["vm_price_per_second_at_1000_mips"] * (mips / 1000.0) ** p["vm_price_exponent"]
    prof = cfg["power_profiles"][power_profile or p["power_profile"]]
    types = np.arange(n_hosts) % len(prof["p_idle_w"])        # host models alternate
    p_idle = np.array(prof["p_idle_w"], dtype=float)[types]
    p_max = np.array(prof["p_max_w"], dtype=float)[types]
    return mips, price, host_of, p_idle, p_max


def generate(cfg: dict, scenario: dict, seed: np.random.SeedSequence, power_profile: str | None = None) -> Instance:
    p = cfg["fixed_parameters"]
    rng = np.random.default_rng(seed)
    mips, price, host_of, p_idle, p_max = _infrastructure(cfg, scenario, rng, power_profile)
    if scenario["arrival"] == "workflow":
        return _workflow_instance(cfg, scenario, rng, mips, price, host_of, p_idle, p_max)
    n = scenario["tasks"]
    if scenario["arrival"] not in ("steady", "bursty"):
        raise ValueError(scenario["arrival"])
    q_lo, q_hi = p["joblog_window_quantiles"]
    logs = [name for name, _ in log_catalogue()]
    log = logs[rng.integers(len(logs))]
    starts = window_starts(log, n, scenario["arrival"], q_lo, q_hi)
    s0 = int(starts[rng.integers(len(starts))])
    submit, run = job_log(log)
    w = run[s0:s0 + n] * p["trace_reference_mips"]                         # MI at the reference speed
    length = w * (p["task_length_mean_mi"] / w.mean())                      # one common factor per instance

    # recorded submit times, rescaled in time so that the offered load is datacentre_load * capacity
    t = submit[s0:s0 + n] - submit[s0]
    horizon = length.sum() / (p["datacentre_load"] * mips.sum())
    arrival = t * (horizon / t[-1]) if t[-1] > 0 else np.zeros(n)

    k = rng.uniform(*p["deadline_slack_range"], size=n)
    deadline = arrival + k * length / mips.mean()
    return Instance(length=length, arrival=arrival, deadline=deadline, mips=mips, price=price,
                    host_of=host_of, p_idle=p_idle, p_max=p_max)


def _workflow_instance(cfg, scenario, rng, mips, price, host_of, p_idle, p_max) -> Instance:
    """A Poisson stream of published workflow traces totalling exactly `tasks` tasks.

    Each draw picks an application at random among those with a trace that still fits, then one of its
    fitting traces; a remainder smaller than every trace is filled with jobs drawn from a job log."""
    p = cfg["fixed_parameters"]
    n = scenario["tasks"]
    ref_mips, min_rt = p["trace_reference_mips"], p["workflow_trace_min_runtime_s"]
    logs = [name for name, _ in log_catalogue()]
    cat = trace_catalogue()
    weights, preds, rank, path, wf_of = [], [], [], [], []
    n_wf = 0
    while len(weights) < n:
        room = n - len(weights)
        fits = {app: [f for f, k in files if k <= room] for app, files in cat.items()}
        apps = [a for a in fits if fits[a]]
        if apps:
            app = apps[rng.integers(len(apps))]
            name = fits[app][rng.integers(len(fits[app]))]
            rt, ps = load_trace(app, name, min_rt)
            w = np.asarray(rt) * ref_mips                       # million instructions at the reference speed
            ps = [list(q) for q in ps]
        else:                                                   # filler: jobs drawn from a job log
            _, run = job_log(logs[rng.integers(len(logs))])
            w = run[rng.integers(len(run), size=room)] * ref_mips
            ps = [[] for _ in range(room)]
        d, pth = _depth_and_path(w, ps)
        base = len(weights)
        singletons = all(not q for q in ps)
        for t in range(len(w)):
            weights.append(w[t]); preds.append([base + q for q in ps[t]])
            rank.append(int(d[t])); path.append(pth[t])
            wf_of.append(n_wf + (t if singletons else 0))
        n_wf += len(w) if singletons else 1
    weights, path, wf_of = np.array(weights), np.array(path), np.array(wf_of)
    scale = p["task_length_mean_mi"] / weights.mean()          # one common factor: relative runtimes kept
    length, path = weights * scale, path * scale

    # workflows arrive as a Poisson process; offered load = datacentre_load * total capacity
    horizon = length.sum() / (p["datacentre_load"] * mips.sum())
    wf_arrival = np.sort(rng.uniform(0, horizon, size=n_wf))
    arrival = wf_arrival[wf_of]
    k = rng.uniform(*p["deadline_slack_range"], size=n_wf)[wf_of]
    deadline = arrival + k * path / mips.mean()
    return Instance(length=length, arrival=arrival, deadline=deadline, mips=mips, price=price,
                    host_of=host_of, p_idle=p_idle, p_max=p_max, rank=np.array(rank), preds=preds)
