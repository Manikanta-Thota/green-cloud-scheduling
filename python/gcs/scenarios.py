"""Scenario matrix and deterministic seeding (common random numbers)."""
from __future__ import annotations
import ast
import itertools
import os
import numpy as np

try:
    import yaml
except ImportError:          # PyYAML has no wheel for some new Python versions; see _mini_yaml
    yaml = None

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "config", "experiment.yaml")


def _scalar(s: str):
    s = s.strip()
    if s in ("true", "false"):
        return s == "true"
    if s.startswith("[") and s.endswith("]"):
        return [_scalar(x) for x in s[1:-1].split(",") if x.strip()]
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def _mini_yaml(text: str) -> dict:
    """Parser for the YAML subset used in config/: nested mappings, inline lists, '- scalar' lists, comments."""
    root: dict = {}
    stack = [(-1, root)]
    lines = [(len(l) - len(l.lstrip()), l.split("#")[0].strip()) for l in text.splitlines()]
    lines = [(ind, l) for ind, l in lines if l]
    for i, (ind, line) in enumerate(lines):
        while stack[-1][0] >= ind:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            parent.append(_scalar(line[2:]))
            continue
        key, _, val = line.partition(":")
        if val.strip():
            parent[key.strip()] = _scalar(val)
        else:
            nxt = lines[i + 1] if i + 1 < len(lines) else (0, "")
            child = [] if nxt[1].startswith("- ") and nxt[0] > ind else {}
            parent[key.strip()] = child
            stack.append((ind, child))
    return root


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        text = f.read()
    return yaml.safe_load(text) if yaml else _mini_yaml(text)


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


def java_seed(seq: np.random.SeedSequence) -> int:
    """A signed 64-bit integer derived from a SeedSequence, for java.util.SplittableRandom."""
    return int(seq.generate_state(1, dtype=np.uint64)[0].astype(np.int64))
