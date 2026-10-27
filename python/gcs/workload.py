"""Seeded workload generator (proposal Section 3.3).

Two families:
* independent tasks with steady (Poisson) or bursty arrivals;
* scientific workflows: a stream of DAGs whose shapes follow Montage, CyberShake, Epigenomics and
  Inspiral (LIGO) as characterised by Bharathi et al. (2008) and Juve et al. (2013). The relative task
  weights are illustrative approximations of those characterisations, not measured traces.

Infrastructure: VMs with heterogeneous MIPS packed onto hosts; host power figures come from the
configured power profile; VM price grows super-linearly with speed so that cost depends on placement.
"""
from __future__ import annotations
import math
import numpy as np
from .model import Instance

# ---------------------------------------------------------------- workflow shapes
# Each builder returns (weights, preds) for one workflow of roughly `size` tasks.

def _montage(size, rng):
    k = max(3, (size - 3) // 4)
    w, preds = [], []
    def add(weight, ps):
        w.append(weight); preds.append(list(ps)); return len(w) - 1
    proj = [add(1.7, []) for _ in range(k)]                                  # mProjectPP
    diffs = [add(0.7, [proj[i], proj[j]]) for i in range(k) for j in (i + 1, i + 2) if j < k]   # mDiffFit
    concat = add(11.0, diffs)                                                  # mConcatFit
    bg = add(14.0, [concat])                                                   # mBgModel
    back = [add(1.7, [bg, proj[i]]) for i in range(k)]                         # mBackground
    tbl = add(2.8, back)                                                       # mImgtbl
    madd = add(14.0, [tbl])                                                    # mAdd
    shrink = add(4.0, [madd])                                                  # mShrink
    add(0.7, [shrink])                                                         # mJPEG
    return w, preds


def _cybershake(size, rng):
    k = max(2, (size - 4) // 2)
    w, preds = [110.0, 110.0], [[], []]                                        # ExtractSGT x2
    seis = []
    for i in range(k):
        w.append(40.0); preds.append([i % 2]); seis.append(len(w) - 1)         # SeismogramSynthesis
    peak = []
    for s in seis:
        w.append(1.0); preds.append([s]); peak.append(len(w) - 1)              # PeakValCalc
    w.append(25.0); preds.append(list(seis))                                   # ZipSeis
    w.append(20.0); preds.append(list(peak))                                   # ZipPSA
    return w, preds


def _epigenomics(size, rng):
    k = max(2, (size - 4) // 4)
    w, preds = [35.0], [[]]                                                    # fastQSplit
    maps = []
    for _ in range(k):
        prev = 0
        for weight in (2.5, 0.5, 1.4, 60.0):        # filterContams, sol2sanger, fastq2bfq, map
            w.append(weight); preds.append([prev]); prev = len(w) - 1
        maps.append(prev)
    w.append(11.0); preds.append(maps)                                         # mapMerge
    w.append(43.0); preds.append([len(w) - 2])                                 # maqIndex
    w.append(55.0); preds.append([len(w) - 2])                                 # pileup
    return w, preds


def _inspiral(size, rng):
    k = max(4, int(size / 4.5))
    g = max(1, k // 4)
    groups = np.array_split(np.arange(k), g)
    w, preds = [], []
    def add(weight, ps):
        w.append(weight); preds.append(list(ps)); return len(w) - 1
    bank = [add(18.0, []) for _ in range(k)]                                   # TmpltBank
    insp = [add(460.0, [bank[i]]) for i in range(k)]                           # Inspiral
    thinca = [add(5.4, [insp[i] for i in grp]) for grp in groups]              # Thinca
    gid = {int(i): j for j, grp in enumerate(groups) for i in grp}
    trig = [add(5.1, [thinca[gid[i]]]) for i in range(k)]                      # TrigBank
    insp2 = [add(460.0, [trig[i]]) for i in range(k)]                          # Inspiral (second pass)
    for grp in groups:
        add(5.4, [insp2[i] for i in grp])                                      # Thinca (second pass)
    return w, preds


WORKFLOW_SHAPES = {"montage": _montage, "cybershake": _cybershake, "epigenomics": _epigenomics,
                   "inspiral": _inspiral}


def _depth_and_path(weights, preds):
    """DAG depth of each task and the heaviest path weight ending at (and including) it."""
    n = len(weights)
    depth = np.zeros(n, dtype=int)
    path = np.zeros(n)
    for t in range(n):          # builders emit tasks in topological order
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

    mu = math.log(p["task_length_mean_mi"]) - 0.5 * p["task_length_sigma"] ** 2   # lognormal with the stated mean
    length = rng.lognormal(mu, p["task_length_sigma"], size=n)

    # offered load = datacentre_load * total capacity -> mean arrival rate
    rate = p["datacentre_load"] * mips.sum() / p["task_length_mean_mi"]            # tasks per second
    if scenario["arrival"] == "steady":
        arrival = np.cumsum(rng.exponential(1.0 / rate, size=n))
    elif scenario["arrival"] == "bursty":
        arrival = _mmpp_arrivals(n, rate, p["bursty"], rng)
    else:
        raise ValueError(scenario["arrival"])

    k = rng.uniform(*p["deadline_slack_range"], size=n)
    deadline = arrival + k * length / mips.mean()
    return Instance(length=length, arrival=arrival, deadline=deadline, mips=mips, price=price,
                    host_of=host_of, p_idle=p_idle, p_max=p_max)


def _mmpp_arrivals(n, rate, b, rng) -> np.ndarray:
    """Two-phase Markov-modulated Poisson process with mean rate `rate`: burst phases at
    peak_factor * rate alternate with quiet phases at (2 - peak_factor) * rate; phase lengths are
    exponential with a mean of `phase_mean_arrivals` mean inter-arrival times, so burst size does not
    depend on the number of tasks."""
    hi, lo = b["peak_factor"] * rate, (2.0 - b["peak_factor"]) * rate
    mean_phase = b["phase_mean_arrivals"] / rate
    out, t, burst = [], 0.0, bool(rng.integers(2))
    while len(out) < n:
        end = t + rng.exponential(mean_phase)
        lam = hi if burst else lo
        while True:
            t += rng.exponential(1.0 / lam)
            if t >= end:
                break
            out.append(t)
        t, burst = end, not burst
    return np.array(out[:n])


def _workflow_instance(cfg, scenario, rng, mips, price, host_of, p_idle, p_max) -> Instance:
    """A Poisson stream of workflows totalling exactly `tasks` tasks (a remainder smaller than the
    minimum workflow size is filled with independent tasks)."""
    p = cfg["fixed_parameters"]
    n = scenario["tasks"]
    lo_size, hi_size = p["workflow_size_range"]
    names = sorted(WORKFLOW_SHAPES)
    weights, preds, rank, path, wf_of = [], [], [], [], []
    n_wf = 0
    while len(weights) < n:
        room = n - len(weights)
        if room < lo_size:
            w, ps = [1.0] * room, [[] for _ in range(room)]
        else:
            name = names[rng.integers(len(names))]
            w, ps = WORKFLOW_SHAPES[name](int(rng.integers(lo_size, hi_size + 1)), rng)
            if len(w) > room:   # too big for the remaining room: rebuild at the room's size, else fill
                w, ps = WORKFLOW_SHAPES[name](room, rng)
                if len(w) > room:
                    w, ps = [1.0] * room, [[] for _ in range(room)]
        w = np.asarray(w) * rng.lognormal(0.0, 0.3, size=len(w))
        d, pth = _depth_and_path(w, ps)
        base = len(weights)
        singletons = all(not q for q in ps)
        for t in range(len(w)):
            weights.append(w[t]); preds.append([base + q for q in ps[t]])
            rank.append(int(d[t])); path.append(pth[t])
            wf_of.append(n_wf + (t if singletons else 0))
        n_wf += len(w) if singletons else 1
    weights, path, wf_of = np.array(weights), np.array(path), np.array(wf_of)
    scale = p["task_length_mean_mi"] / weights.mean()
    length, path = weights * scale, path * scale

    # workflows arrive as a Poisson process; offered load = datacentre_load * total capacity
    horizon = length.sum() / (p["datacentre_load"] * mips.sum())
    wf_arrival = np.sort(rng.uniform(0, horizon, size=n_wf))
    arrival = wf_arrival[wf_of]
    k = rng.uniform(*p["deadline_slack_range"], size=n_wf)[wf_of]
    deadline = arrival + k * path / mips.mean()
    return Instance(length=length, arrival=arrival, deadline=deadline, mips=mips, price=price,
                    host_of=host_of, p_idle=p_idle, p_max=p_max, rank=np.array(rank), preds=preds)
