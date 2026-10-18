# Experiment protocol (working draft for milestone M1)

Source: *Energy-Aware Multi-Objective Task Scheduling for Green Cloud Computing Using Hybrid
Optimisation* (practicum proposal). This file turns Sections 3.1-3.6 of the proposal into the exact
rules the code implements. Where the proposal is ambiguous the chosen reading is listed under
**Open decisions** at the end; please confirm or overrule each one before the protocol is frozen.

## 1. Problem model (proposal 3.2)

A schedule assigns every task to one VM. Objective vector, all minimised:

    F(x) = [E(x), M(x), C(x), I(x), S(x)]
           energy, makespan, execution cost, utilisation imbalance, SLA violation

| Symbol | Definition used by the code |
|---|---|
| Execution | A VM runs one task at a time, non-preemptively, in (arrival, index) order. `start = max(arrival, VM free time)`, `finish = start + length / mips`. |
| M (makespan) | Latest task finish time (s). |
| E (energy) | Hosts are on during `[0, M]`. Host power `P = p_idle + (p_max - p_idle) * u`, where `u` = MIPS of the host's busy VMs / total MIPS of its VMs. Integrated over time (J). |
| C (cost) | Sum over VMs of `price per second x busy seconds`. |
| I (imbalance) | Coefficient of variation (population std / mean) of VM utilisation, utilisation = busy seconds / M. |
| S (SLA) | Fraction of tasks that finish after their deadline. |

Feasibility-first dominance (proposal 3.2) applies when constraints are added to the search; the
evaluation functions above are unconstrained and are the ground truth for all algorithms.

## 2. Scenario matrix and seeding (proposal 3.1)

`config/experiment.yaml` holds the factors: tasks {500, 2000, 5000} x VMs {20, 50, 100} x
heterogeneity {low, medium, high} x arrival {steady, bursty, workflow} = **81 scenarios**, 30
replicates, 5 algorithms: **12,150 runs** for the full factorial.

* Instance seed = `SeedSequence([master_seed, 1, scenario_id, replicate])`, identical for every
  algorithm (common random numbers).
* Algorithm seed = `SeedSequence([master_seed, 2, scenario_id, replicate, algorithm_index])`.
* Failed or interrupted runs are re-run with the same seed and logged, never silently dropped.

## 3. Decision rules (proposal 3.5)

An energy improvement counts only if, against the best baseline's medians: makespan degrades by
at most 5%, cost rises by at most 5%, and SLA violations rise by at most 1 percentage point.
Statistics: Friedman test, Holm-corrected pairwise Wilcoxon signed-rank, Cliff's delta, bootstrap
95% CIs (10,000 resamples). Pareto quality: hypervolume and IGD on normalised objectives.
Implemented in `python/gcs/stats.py` and `python/gcs/pareto.py`.

## 4. Validation strategy (proposal 3.4)

1. `testdata/tiny_case.txt`: three tasks, two VMs, every number checkable by hand.
2. `testdata/parity_case.txt`: a 500-task bursty instance exported from Python; the Java
   `SelfCheck` must reproduce all objectives and both baseline assignments.
3. When the CloudSim Plus layer exists (M2) it must reproduce the same objectives on the same
   cases, or the differences must be explained and documented.

## 5. Open decisions

| # | Decision needed | Current default |
|---|---|---|
| D1 | Which algorithm is the "single global metaheuristic" baseline (e.g. GA or PSO)? | Not chosen; slot reserved in config. |
| D2 | Which algorithm is the "recent multi-objective baseline" (NSGA-II as a reference, or a 2023-25 method from the review)? | Not chosen. |
| D3 | Power-model parameters. | Placeholders: 100 W idle, 250 W peak per host. Replace with a cited source. |
| D4 | Proposal 3.2 says "no VM performs more than one task". | Read as one task at a time per VM, with queueing. |
| D5 | Billing granularity. | Per-second of busy time. |
| D6 | Section 3.5 tolerance wording is garbled ("U+1 percentage point", "statistically insignificant or unfeasible cost increase over 5%"). | Read as: SLA +1 pp, cost +5%. |
| D7 | Workflow workloads (Montage, CyberShake, Epigenomics, Inspiral) need DAG traces or a generator. | Not implemented; the 27 `workflow` scenarios cannot run yet. |
| D8 | Figure 1 (Gantt chart) labels a task "Hybrid GSCOA implementation", but the proposed scheduler is HPEAS and GSCOA is Kumar and Karri's algorithm. | Rename the task in the figure. |
| D9 | Workload calibration. With the placeholder load (0.7) and deadlines, FCFS violates ~0% of SLAs in steady scenarios but ~34% in the medium-heterogeneity bursty one. | Decide the target stress range during the week 6 pilot. |
| D10 | Full factorial (12,150 runs) versus a balanced sample. | Full factorial until the pilot gives timings. |
