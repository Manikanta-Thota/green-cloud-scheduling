# HPEAS: energy-aware multi-objective task scheduling for green clouds

Practicum project: a Hybrid Pareto Energy-Aware Scheduler (HPEAS) compared against FCFS, Min-Min,
a single-objective genetic algorithm and a multi-objective grey wolf optimiser (MOGWO) under one
reproducible protocol (`docs/protocol.md`).

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows; use bin/activate elsewhere
make setup            # numpy, scipy, pandas, matplotlib, PyYAML (optional)
make test             # Python tests, Java/Python parity cases, optimiser self-checks (JDK 17+)
make validate-cloudsim   # downloads CloudSim Plus 8.5.7 jars, replays schedules, compares (optional)
make tune             # parameter tuning on the separate tuning seeds (already frozen in config)
make experiment       # full matrix: 81 scenarios x 30 replicates x 12 algorithms (~1 h on 8 cores)
make sensitivity      # one-at-a-time sensitivity, tolerance and power-curve runs
make analysis         # tables, statistics and figures
```

Without `make`, run the module commands shown in the `Makefile` from the `python/` directory.

## Layout

| Path | Purpose |
|---|---|
| `config/experiment.yaml` | Frozen experiment definition: factors, seeds, power profiles, search settings, tolerances. |
| `docs/protocol.md` | The proposal turned into exact rules; decisions log. |
| `python/gcs/model.py` | Reference oracle: objective evaluation with event-by-event energy integration. |
| `python/gcs/workload.py`, `scenarios.py` | Seeded workloads (steady, bursty MMPP, workflow DAGs); 81-scenario matrix; common random numbers. |
| `python/gcs/baselines.py` | FCFS and Min-Min (precedence-aware), used for parity tests. |
| `python/gcs/experiment.py` | Orchestration with checkpointing, retry and run log; calls the Java runner. |
| `python/gcs/tune.py`, `sensitivity.py` | Tuning grid; sensitivity, tolerance and power-curve experiments. |
| `python/gcs/analysis.py`, `stats.py`, `pareto.py` | Statistics (Friedman, Holm-Wilcoxon, Cliff's delta, bootstrap) and figures. |
| `python/gcs/validate_cloudsim.py` | CloudSim Plus replay of baseline schedules. |
| `java/src/main/java/gcs/core/` | Instance, fast evaluator, FCFS, Min-Min, energy-greedy, case-file reader, `SelfCheck`. |
| `java/src/main/java/gcs/opt/` | GA, MOGWO, HPEAS (+ ablations), archive, indicators, `RunExperiment`, `OptSelfCheck`. |
| `java/src/main/java/gcs/sim/` | CloudSim Plus validation layer (needs `java/lib`, see `make cloudsim-lib`). |
| `testdata/` | Hand-checkable and parity cases shared by both languages. |
| `results/` | Raw per-run CSVs, traces, run logs, tuning, sensitivity and analysis outputs. |

## Reproducibility

Every instance and algorithm seed derives from `master_seed` in the config (`docs/protocol.md` §2).
Results are written per scenario (`results/main/sNN_runs.csv`, `sNN_trace.csv`, and every selected
schedule in `sNN_schedules.csv.gz`); rerunning skips finished scenarios, and `run_log.csv` records every
attempt with the hash of the run configuration saved in `config_<hash>.json`. Scenarios, instances and
algorithms run in a seeded random order (`--order-seed`); results do not depend on it. PyYAML is optional: on Python versions
without a wheel, a small built-in reader handles the config.
