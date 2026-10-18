"""Seeded synthetic workload generator (proposal Section 3.3, first workload family)."""
from __future__ import annotations
import math
import numpy as np
from .model import Instance


def generate(cfg: dict, scenario: dict, seed: np.random.SeedSequence) -> Instance:
    if scenario["arrival"] == "workflow":
        raise NotImplementedError("workflow (DAG) workloads are not implemented yet (milestone M2)")
    p = cfg["fixed_parameters"]
    rng = np.random.default_rng(seed)
    n, m = scenario["tasks"], scenario["vms"]

    lo, hi = p["heterogeneity_mips_range"][scenario["heterogeneity"]]
    mips = rng.uniform(lo, hi, size=m)
    n_hosts = math.ceil(m / p["vms_per_host"])
    host_of = np.arange(m) // p["vms_per_host"]
    price = mips * p["vm_price_per_mips_second"]

    mu = math.log(p["task_length_mean_mi"]) - 0.5 * p["task_length_sigma"] ** 2   # lognormal with the stated mean
    length = rng.lognormal(mu, p["task_length_sigma"], size=n)

    # offered load = datacentre_load * total capacity -> arrival horizon
    rate = p["datacentre_load"] * mips.sum() / p["task_length_mean_mi"]            # tasks per second
    horizon = n / rate
    if scenario["arrival"] == "steady":
        arrival = np.cumsum(rng.exponential(1.0 / rate, size=n))
    elif scenario["arrival"] == "bursty":
        n_burst = int(0.7 * n)
        centres = rng.uniform(0.1, 0.9, size=5) * horizon
        burst = rng.normal(rng.choice(centres, size=n_burst), 0.02 * horizon)
        rest = rng.uniform(0, horizon, size=n - n_burst)
        arrival = np.sort(np.clip(np.concatenate([burst, rest]), 0, None))
    else:
        raise ValueError(scenario["arrival"])

    k = rng.uniform(*p["deadline_slack_range"], size=n)
    deadline = arrival + k * length / mips.mean()
    return Instance(length=length, arrival=arrival, deadline=deadline, mips=mips, price=price,
                    host_of=host_of, p_idle=np.full(n_hosts, p["host_p_idle_w"]),
                    p_max=np.full(n_hosts, p["host_p_max_w"]))
