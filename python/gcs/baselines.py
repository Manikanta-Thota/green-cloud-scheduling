"""Deterministic baseline schedulers (proposal comparison methods 1 and 2).

Tie-breaking is fixed (lowest task index, then lowest VM index) so results are reproducible.
"""
from __future__ import annotations
import numpy as np
from .model import Instance


def fcfs(inst: Instance) -> np.ndarray:
    """Tasks in arrival order; each goes to the VM that can start it earliest."""
    free = np.zeros(inst.n_vms)
    assign = np.zeros(inst.n_tasks, dtype=int)
    order = np.lexsort((np.arange(inst.n_tasks), inst.arrival))
    ex = inst.exec_matrix()
    for t in order:
        start = np.maximum(inst.arrival[t], free)
        v = int(np.argmin(start))
        assign[t] = v
        free[v] = start[v] + ex[t, v]
    return assign


def min_min(inst: Instance) -> np.ndarray:
    """Classic Min-Min with ready times: repeatedly schedule the task whose best completion time is smallest."""
    ex = inst.exec_matrix()
    free = np.zeros(inst.n_vms)
    assign = np.full(inst.n_tasks, -1, dtype=int)
    remaining = np.arange(inst.n_tasks)
    while len(remaining):
        ct = np.maximum(inst.arrival[remaining, None], free[None, :]) + ex[remaining]
        best_v = ct.argmin(axis=1)
        best_ct = ct[np.arange(len(remaining)), best_v]
        i = int(np.argmin(best_ct))          # first minimum -> lowest task index
        t, v = remaining[i], int(best_v[i])
        assign[t] = v
        free[v] = best_ct[i]
        remaining = np.delete(remaining, i)
    return assign


BASELINES = {"fcfs": fcfs, "minmin": min_min}
