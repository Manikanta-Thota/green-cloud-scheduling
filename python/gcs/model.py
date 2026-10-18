"""Problem model and objective evaluation (proposal Section 3.2).

Conventions (documented in docs/protocol.md):
* A VM runs one task at a time, non-preemptively. Tasks queued on a VM run in
  order of (arrival, task index).
* start = max(arrival, time the VM becomes free); finish = start + length / mips.
* Hosts are powered on for the whole interval [0, makespan]. A host's
  utilisation at time t is the MIPS of its busy VMs divided by the total MIPS
  of its VMs, and power is linear: P = p_idle + (p_max - p_idle) * u.
* Cost = sum over VMs of (price per second x busy seconds).
* Imbalance = coefficient of variation (population std / mean) of VM
  utilisation, where VM utilisation = busy seconds / makespan.
* SLA violation = fraction of tasks that finish after their deadline.
All objectives are minimised.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

OBJECTIVE_NAMES = ("energy", "makespan", "cost", "imbalance", "sla")


@dataclass
class Instance:
    length: np.ndarray      # task length, million instructions (MI)
    arrival: np.ndarray     # arrival time, seconds
    deadline: np.ndarray    # absolute deadline, seconds
    mips: np.ndarray        # VM speed, MI per second
    price: np.ndarray       # VM price, currency per second of busy time
    host_of: np.ndarray     # host index of each VM
    p_idle: np.ndarray      # per host, watts
    p_max: np.ndarray       # per host, watts

    @property
    def n_tasks(self) -> int:
        return len(self.length)

    @property
    def n_vms(self) -> int:
        return len(self.mips)

    def exec_matrix(self) -> np.ndarray:
        """Execution time of every task on every VM, shape (n_tasks, n_vms)."""
        return self.length[:, None] / self.mips[None, :]


def simulate(inst: Instance, assign) -> dict:
    """Return per-task start/finish and per-VM busy time for an assignment."""
    assign = np.asarray(assign, dtype=int)
    if assign.shape != (inst.n_tasks,) or assign.min() < 0 or assign.max() >= inst.n_vms:
        raise ValueError("invalid assignment")
    order = np.lexsort((np.arange(inst.n_tasks), inst.arrival))
    free = np.zeros(inst.n_vms)
    busy = np.zeros(inst.n_vms)
    start = np.zeros(inst.n_tasks)
    finish = np.zeros(inst.n_tasks)
    for t in order:
        v = assign[t]
        s = max(inst.arrival[t], free[v])
        d = inst.length[t] / inst.mips[v]
        start[t], finish[t] = s, s + d
        free[v] = s + d
        busy[v] += d
    return {"start": start, "finish": finish, "busy": busy, "assign": assign}


def _energy(inst: Instance, sim: dict, makespan: float) -> float:
    total = 0.0
    n_hosts = len(inst.p_idle)
    host_cap = np.bincount(inst.host_of, weights=inst.mips, minlength=n_hosts)
    events = [[] for _ in range(n_hosts)]
    for t in range(inst.n_tasks):
        v = sim["assign"][t]
        h = inst.host_of[v]
        events[h].append((sim["start"][t], +inst.mips[v]))
        events[h].append((sim["finish"][t], -inst.mips[v]))
    for h in range(n_hosts):
        ev = sorted(events[h])
        busy_mips, prev = 0.0, 0.0
        span = inst.p_max[h] - inst.p_idle[h]
        for time, delta in ev:
            if time > prev:
                u = busy_mips / host_cap[h] if host_cap[h] > 0 else 0.0
                total += (inst.p_idle[h] + span * u) * (time - prev)
                prev = time
            busy_mips += delta
        if makespan > prev:   # trailing idle time up to makespan
            total += inst.p_idle[h] * (makespan - prev)
    return total


def evaluate(inst: Instance, assign) -> dict:
    """Objective vector for an assignment: energy (J), makespan (s), cost, imbalance, SLA fraction."""
    sim = simulate(inst, assign)
    makespan = float(sim["finish"].max())
    util = sim["busy"] / makespan if makespan > 0 else np.zeros(inst.n_vms)
    mean_u = util.mean()
    imbalance = float(util.std() / mean_u) if mean_u > 0 else 0.0
    return {
        "energy": float(_energy(inst, sim, makespan)),
        "makespan": makespan,
        "cost": float((inst.price * sim["busy"]).sum()),
        "imbalance": imbalance,
        "sla": float((sim["finish"] > inst.deadline + 1e-9).mean()),
    }


def load_case(path: str):
    """Parse testdata/tiny_case.txt -> (Instance, assignment, expectations dict)."""
    vms, hosts, tasks, expect = [], {}, [], {}
    assign = None
    with open(path) as fh:
        lines = fh.readlines()
    for raw in lines:
        line = raw.split("#")[0].strip()
        if not line:
            continue
        tok = line.split()
        key, vals = tok[0], tok[1:]
        if key == "vm":
            vms.append((float(vals[0]), float(vals[1]), int(vals[2])))
        elif key == "host":
            hosts[int(vals[0])] = (float(vals[1]), float(vals[2]))
        elif key == "task":
            tasks.append(tuple(float(v) for v in vals))
        elif key == "assign":
            assign = [int(v) for v in vals]
        elif key == "expect":
            expect[vals[0]] = float(vals[1])
        elif key in ("expect_fcfs", "expect_minmin"):
            expect[key] = [int(v) for v in vals]
    ids = sorted(hosts)
    inst = Instance(
        length=np.array([t[0] for t in tasks]), arrival=np.array([t[1] for t in tasks]),
        deadline=np.array([t[2] for t in tasks]), mips=np.array([v[0] for v in vms]),
        price=np.array([v[1] for v in vms]), host_of=np.array([ids.index(v[2]) for v in vms]),
        p_idle=np.array([hosts[i][0] for i in ids]), p_max=np.array([hosts[i][1] for i in ids]))
    return inst, assign, expect
