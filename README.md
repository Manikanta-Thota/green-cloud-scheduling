# HPEAS: energy-aware multi-objective task scheduling for green clouds

Practicum project: a Hybrid Pareto Energy-Aware Scheduler (HPEAS) compared against FCFS, Min-Min,
a single-metaheuristic scheduler and a recent multi-objective baseline, under one reproducible
protocol. See `docs/protocol.md` for the exact rules and the list of open decisions.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
make setup          # installs numpy, scipy, pandas, matplotlib, PyYAML
make test           # Python unit tests + Java parity checks (needs a JDK 17+ for the Java part)
make demo           # runs FCFS and Min-Min on a scenario, writes results/demo_baselines.csv
```

Without `make`: `cd python && python3 -m unittest discover -s tests -v`.

## Layout

| Path | Purpose |
|---|---|
| `config/experiment.yaml` | Frozen experiment definition: factors, seeds, tolerances, placeholder parameters. |
| `docs/protocol.md` | The proposal turned into exact rules, plus open decisions. |
| `python/gcs/model.py` | Objective evaluation (energy, makespan, cost, imbalance, SLA): the reference oracle. |
| `python/gcs/workload.py`, `scenarios.py` | Seeded synthetic workloads; 81-scenario matrix; common-random-number seeds. |
| `python/gcs/baselines.py` | FCFS and Min-Min. |
| `python/gcs/pareto.py`, `stats.py` | Non-dominated sorting, crowding, hypervolume, IGD; Friedman, Holm-Wilcoxon, Cliff's delta, bootstrap CIs, tolerance rule. |
| `python/gcs/run_baselines.py`, `export_case.py` | Run baselines to CSV; export a case for Java parity. |
| `java/src/main/java/gcs/core/` | Dependency-free Java mirror of the model and baselines, plus `SelfCheck`. |
| `java/pom.xml` | Maven build with the CloudSim Plus dependency (not yet built, see below). |
| `testdata/` | Hand-checkable and parity cases shared by both languages. |

## Status against the proposal milestones

| Milestone | Status |
|---|---|
| M1 Protocol, objectives, metrics, ethics | Draft in `docs/protocol.md`; 10 open decisions to confirm. Ethics form not started. |
| M2 Working simulator and validated baselines | Partly done: evaluator, workload generator, FCFS, Min-Min, tests, Java parity. **Still to do:** CloudSim Plus integration and validation against this oracle; workflow workloads. |
| M3 Hybrid scheduler and unit tests | Not started. |
| M4-M6 | Not started. |

## Not verified in the setup session

The setup environment had no network access, so CloudSim Plus was never downloaded. `java/pom.xml` is
unbuilt and its CloudSim Plus version number must be confirmed. Everything under `python/` and the
dependency-free Java core was compiled and tested.

## Suggested next steps

1. Settle open decisions D1-D4 and D6 (they change what is implemented).
2. On a networked machine run `cd java && mvn -q compile` and fix the CloudSim Plus version if needed.
3. Write the CloudSim Plus layer that builds hosts, VMs and cloudlets from an `Instance` and reproduces
   the oracle's objectives on `testdata/*.txt`.
4. Add the single-metaheuristic and multi-objective baselines, then HPEAS (M3).
