"""Scenario matrix and deterministic seeding (common random numbers)."""
from __future__ import annotations
import itertools
import os
import yaml
import numpy as np

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "config", "experiment.yaml")


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def scenario_matrix(cfg: dict) -> list[dict]:
    f = cfg["scenario_factors"]
    keys = ["tasks", "vms", "heterogeneity", "arrival"]
    rows = [dict(zip(keys, combo)) for combo in itertools.product(*(f[k] for k in keys))]
    for i, r in enumerate(rows):
        r["scenario_id"] = i
    return rows


def instance_seed(cfg: dict, scenario_id: int, rep: int) -> np.random.SeedSequence:
    """Seed for the *instance*: identical for every algorithm (common random numbers)."""
    return np.random.SeedSequence([cfg["experiment"]["master_seed"], 1, scenario_id, rep])


def algorithm_seed(cfg: dict, scenario_id: int, rep: int, algo_index: int) -> np.random.SeedSequence:
    """Seed for a stochastic algorithm's own random choices."""
    return np.random.SeedSequence([cfg["experiment"]["master_seed"], 2, scenario_id, rep, algo_index])
