"""Deterministic baseline schedulers (proposal comparison methods 1 and 2).

Tie-breaking is fixed (lowest task index, then lowest VM index) so results are reproducible and match
the Java implementations exactly. Workflow tasks become ready when all their predecessors are done.
"""
from __future__ import annotations
import numpy as np
from .model import Instance


def fcfs(inst: Instance) -> np.ndarray:
    """Tasks in execution order; each goes to the VM that can start it earliest."""
    free = np.zeros(inst.n_vms)
    finish = np.zeros(inst.n_tasks)
    assign = np.zeros(inst.n_tasks, dtype=int)
    ex = inst.exec_matrix()
    for t in inst.order():
        ready = max([inst.arrival[t]] + [finish[p] for p in inst.preds[t]])
        start = np.maximum(ready, free)
        v = int(np.argmin(start))
        assign[t] = v
        finish[t] = free[v] = start[v] + ex[t, v]
    return assign


def min_min(inst: Instance) -> np.ndarray:
    """Classic Min-Min with ready times: among tasks whose predecessors are scheduled, repeatedly
    schedule the one whose best completion time is smallest."""
    ex = inst.exec_matrix()
    free = np.zeros(inst.n_vms)
    ct = np.zeros(inst.n_tasks)
    assign = np.full(inst.n_tasks, -1, dtype=int)
    left = np.array([len(p) for p in inst.preds])
    succ = [[] for _ in range(inst.n_tasks)]
    for t, ps in enumerate(inst.preds):
        for p in ps:
            succ[p].append(t)
    ready = inst.arrival.astype(float).copy()
    remaining = [t for t in range(inst.n_tasks) if left[t] == 0]
    while remaining:
        rem = np.array(sorted(remaining))
        c = np.maximum(ready[rem, None], free[None, :]) + ex[rem]
        best_v = c.argmin(axis=1)
        best_ct = c[np.arange(len(rem)), best_v]
        i = int(np.argmin(best_ct))          # first minimum -> lowest task index
        t, v = int(rem[i]), int(best_v[i])
        assign[t] = v
        ct[t] = free[v] = best_ct[i]
        remaining.remove(t)
        for s in succ[t]:
            left[s] -= 1
            if left[s] == 0:
                ready[s] = max([inst.arrival[s]] + [ct[p] for p in inst.preds[s]])
                remaining.append(s)
    return assign
