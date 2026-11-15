# Changelog

Versions follow [Semantic Versioning](https://semver.org/). Each release is a git tag (`vX.Y.Z`).

## [2.0.0] - 2026-11-14
### Changed
- Workloads are published data, not generated (protocol D15, D16): workflow scenarios use 73
  WfInstances traces (Montage, Epigenomics, Seismology, 1000Genome; fetched at a pinned commit by
  `python/fetch_traces.py`, listed in `data/wfinstances_manifest.csv`); steady and bursty scenarios use
  windows of the HPC2N-2002 and NASA-iPSC-1993 logs from the Parallel Workloads Archive
  (`data/joblogs_manifest.csv`, with SHA-256 and access date). CyberShake and Inspiral have no published
  traces and are replaced. Only the data centre, deadlines and random choices remain generated.
- Configuration: `workflow_size_range`, `task_length_sigma` and `bursty` removed;
  `trace_reference_mips`, `workflow_trace_min_runtime_s` and `joblog_window_quantiles` added.
- GA and HPEAS re-tuned on the tuning seeds and frozen before the test runs (HPEAS pm 1, ls_steps 20;
  GA ga_pm 1, ga_pc 0.6).
- All experiments re-run: main (81 scenarios x 30 replicates, config hash `fe8cc50d3ffb`, no failed
  attempt), example fronts, CloudSim Plus validation (108/108 agree), sensitivity and power-curve runs,
  analysis. The headline median energy saving of HPEAS changed from 3.1% to 8.1%; it is now
  largest for workflows instead of smallest. v1.2.1 outputs are kept in `results/*_v2`.
- Windows run commands added to `README.md`. `python/demo.py` added: runs FCFS, Min-Min and HPEAS on one scenario and prints the comparison.

## [1.2.1] - 2026-11-10
### Changed
- Documentation aligned with the supervisor's comments on the methodology (protocol D14): the
  workload description names the public workflow sources (WfInstances,
  LGPL v3.0, Montage and Epigenomics only; Pegasus WorkflowGenerator, no licence), states that neither
  is used, and describes how the generated workflows are modified and scaled; Coleman et al. (2022)
  added to the references.
- The run-time pilot is treated as a project-plan activity (milestone M4): the methodology
  now calls the tuning-seed workload check a calibration run (section 3.2 "Workload calibration and
  parameter tuning").
- No code under `python/gcs` or `java/src`, configuration or result changed, so the source hash stored
  with `results/main` still matches.

## [1.2.0] - 2026-10-28
### Added
- Per-run provenance promised in proposal §3.4 (protocol D13): every selected schedule (assignment
  vector, SHA-256, algorithm seed) in `sNN_schedules.csv.gz`; the full run configuration and a hash of
  the Java and Python sources in `config_<hash>.json`; the configuration hash with every schedule and
  every `run_log.csv` attempt.
- Seeded random execution order of scenarios, instances and algorithms (`--order-seed`), logged per
  attempt; output rows keep their canonical order.
### Changed
- Main experiment re-run with provenance and randomised order; it is now `results/main`, and the earlier run is kept as `results/main_v1`.
- `run_log.csv` gains `config_hash`, `order_seed` and `order_position`; older logs are upgraded in place.
- A scenario's runs file is now written last, so it alone marks the scenario as finished.

## [1.0.0] - 2026-10-28
### Added
- Protocol v1 decisions log (D1-D12) in `docs/protocol.md`; README with the full reproduction steps.

## [0.3.0] - 2026-10-28
### Added
- Experiment outputs under `results/`: pilot and calibration runs, tuning grids, CloudSim Plus validation,
  main experiment (81 scenarios x 30 replicates, 12 methods), example Pareto fronts, sensitivity,
  tolerance, budget and energy-proportional power-curve runs, statistical analysis tables and run logs.
- Figures in `results/figures/`.
### Fixed
- Tolerance-sensitivity variants re-run with the 0.2.0 `--param tol_*` fix; the earlier runs, which used
  the default envelope, are kept in `results/sensitivity_invalid_tol/` (protocol D12).

## [0.2.0] - 2026-10-28
### Added
- Java optimisers: HPEAS (heuristic seeding, NSGA-II, adaptive local search, Pareto archive), GA, MOGWO,
  ablation and fairness variants, energy-greedy heuristic, quality indicators and a multi-threaded
  experiment runner (`gcs.opt.RunExperiment`), with self-checks.
- CloudSim Plus 8.5.7 replay of schedules (`gcs.sim.CloudSimValidation`).
- Python orchestration: bursty (MMPP) and workflow workloads, scenario matrix, experiment runner with
  checkpointing and an output-directory lock, tuning, sensitivity runs, statistical analysis and figures.
- Precedence-aware evaluator and baselines, parity test cases and unit tests.
### Changed
- Super-linear VM price, idle-host switch-off energy model and calibrated workloads (protocol D3-D9).
### Fixed
- `--param tol_*` overrides were overwritten by the configured tolerances (protocol D12).
### Removed
- Unused code (dead helpers, unused counters and constructors) with results verified identical to the
  previous build.

## [0.1.0] - 2026-10-18
### Added
- Project setup: model oracle, baselines, workload generator, Pareto and statistics tools, Java core,
  protocol draft.
