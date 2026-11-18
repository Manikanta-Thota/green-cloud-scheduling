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
make analysis         # tables, statistics, H0/H1 decision and figures
```

Without `make`, run the module commands shown in the `Makefile` from the `python/` directory.

### Windows (PowerShell, no `make`)

From the repository root:
```powershell
python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force java\out | Out-Null
javac -d java\out (Get-ChildItem java\src\main\java\gcs\core\*.java, java\src\main\java\gcs\opt\*.java).FullName
java -cp java\out gcs.core.SelfCheck testdata\parity_case.txt
java -cp java\out gcs.opt.OptSelfCheck testdata\parity_workflow_case.txt
```
From `python\`:
```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests          # 31 tests, last line OK
..\.venv\Scripts\python.exe fetch_traces.py                        # once, before any experiment
..\.venv\Scripts\python.exe demo.py                                # FCFS, Min-Min, HPEAS on one scenario (~40 s)
..\.venv\Scripts\python.exe -m gcs.experiment --out ..\results\main
..\.venv\Scripts\python.exe -m gcs.analysis --main ..\results\main --out ..\results\analysis --figures ..\results\figures --front ..\results\fronts\fronts --front-stem s76_r00
..\.venv\Scripts\python.exe -m gcs.validate_cloudsim                # needs the CloudSim Plus jars in java\lib (Makefile: cloudsim-lib)
```
One CloudSim Plus case, from `python\` (after one `gcs.validate_cloudsim` run has compiled `java\out-sim`):
```powershell
java -cp "..\java\out-sim;..\java\lib\*" gcs.sim.CloudSimValidation ..\testdata\parity_case.txt
```

## Layout

| Path | Purpose |
|---|---|
| `config/experiment.yaml` | Frozen experiment definition: factors, seeds, power profiles, search settings, tolerances. |
| `docs/protocol.md` | The proposal turned into exact rules; decisions log. |
| `python/gcs/model.py` | Reference oracle: objective evaluation with event-by-event energy integration. |
| `python/gcs/workload.py`, `scenarios.py` | Seeded workloads (steady and bursty: windows of published Parallel Workloads Archive job logs; workflows: published WfInstances traces); 81-scenario matrix; common random numbers. |
| `python/fetch_traces.py`, `data/wfinstances_manifest.csv` | Downloads the 73 WfInstances workflow traces (pinned commit, LGPL v3.0, not redistributed) and verifies them against the committed SHA-256 manifest; also verifies the two job logs listed in `data/joblogs_manifest.csv`, which must be downloaded by hand in a browser (the archive blocks scripted downloads). Run once before any experiment. |
| `python/gcs/baselines.py` | FCFS and Min-Min (precedence-aware), used for parity tests. |
| `python/gcs/experiment.py` | Orchestration with checkpointing, retry and run log; calls the Java runner. |
| `python/gcs/tune.py`, `sensitivity.py` | Tuning grid; sensitivity, tolerance and power-curve experiments. |
| `python/gcs/analysis.py`, `stats.py`, `pareto.py` | Statistics (Friedman, Holm-Wilcoxon, Cliff's delta, bootstrap), the H0/H1 decision of the protocol (`hypothesis_tests.csv`) and figures. |
| `python/gcs/validate_cloudsim.py` | CloudSim Plus replay of baseline schedules. |
| `java/src/main/java/gcs/core/` | Instance, fast evaluator, FCFS, Min-Min, energy-greedy, case-file reader, `SelfCheck`. |
| `java/src/main/java/gcs/opt/` | GA, MOGWO, HPEAS (+ ablations), archive, indicators, `RunExperiment`, `OptSelfCheck`. |
| `java/src/main/java/gcs/sim/` | CloudSim Plus validation layer (needs `java/lib`, see `make cloudsim-lib`). |
| `testdata/` | Hand-checkable and parity cases shared by both languages. |
| `results/` | Raw per-run CSVs, traces, run logs, tuning, sensitivity and analysis outputs; figures in `results/figures/`. |

## Reproducibility

Every instance and algorithm seed derives from `master_seed` in the config (`docs/protocol.md` §2).
Results are written per scenario (`results/main/sNN_runs.csv`, `sNN_trace.csv`, and every selected
schedule in `sNN_schedules.csv.gz`); rerunning skips finished scenarios, and `run_log.csv` records every
attempt with the hash of the run configuration saved in `config_<hash>.json`. Scenarios, instances and
algorithms run in a seeded random order (`--order-seed`); results do not depend on it. PyYAML is optional: on Python versions
without a wheel, a small built-in reader handles the config.
