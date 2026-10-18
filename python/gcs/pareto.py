"""Pareto utilities: dominance, non-dominated sorting, crowding distance, hypervolume, IGD.
All objectives are minimised. Inputs are arrays of shape (n_points, n_objectives)."""
from __future__ import annotations
import numpy as np


def dominates(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(F) -> list[list[int]]:
    F = np.asarray(F, dtype=float)
    n = len(F)
    dominated_by = [[] for _ in range(n)]
    count = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(F[i], F[j]):
                dominated_by[i].append(j); count[j] += 1
            elif dominates(F[j], F[i]):
                dominated_by[j].append(i); count[i] += 1
    fronts, current = [], [i for i in range(n) if count[i] == 0]
    while current:
        fronts.append(current)
        nxt = []
        for i in current:
            for j in dominated_by[i]:
                count[j] -= 1
                if count[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def crowding_distance(F) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    n, d = F.shape
    dist = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(d):
        order = np.argsort(F[:, k], kind="stable")
        span = F[order[-1], k] - F[order[0], k]
        dist[order[0]] = dist[order[-1]] = np.inf
        if span == 0:
            continue
        dist[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / span
    return dist


def _nd(P):
    return P[non_dominated_sort(P)[0]] if len(P) else P


def hypervolume(F, ref) -> float:
    """Exact hypervolume by slicing (HSO); fine for the front sizes used here."""
    P = np.asarray(F, dtype=float)
    ref = np.asarray(ref, dtype=float)
    P = P[np.all(P < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = _nd(P)
    d = len(ref)
    if d == 1:
        return float(ref[0] - P[:, 0].min())
    P = P[np.argsort(P[:, -1], kind="stable")]
    vol = 0.0
    for i in range(len(P)):
        upper = P[i + 1, -1] if i + 1 < len(P) else ref[-1]
        if upper > P[i, -1]:
            vol += hypervolume(P[: i + 1, :-1], ref[:-1]) * (upper - P[i, -1])
    return float(vol)


def igd(F, reference_front) -> float:
    """Inverted generational distance: mean distance from each reference point to the nearest obtained point."""
    F, R = np.asarray(F, dtype=float), np.asarray(reference_front, dtype=float)
    d = np.linalg.norm(R[:, None, :] - F[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def normalise(F, ideal, nadir) -> np.ndarray:
    F, ideal, nadir = (np.asarray(a, dtype=float) for a in (F, ideal, nadir))
    return (F - ideal) / np.where(nadir - ideal == 0, 1.0, nadir - ideal)
