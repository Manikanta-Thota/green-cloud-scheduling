# Experiment protocol (v1, frozen before the test runs)

Source: *Energy-Aware Multi-Objective Task Scheduling for Green Cloud Computing Using Hybrid
Optimisation* (practicum proposal). This file turns proposal Sections 3.1-3.6 into the exact rules the
code implements. Section 6 records how every open decision of the v0 draft was settled.

## 1. Problem model (proposal 3.2)

A schedule assigns every task to one VM. Objective vector, all minimised:

    F(x) = [E(x), M(x), C(x), I(x), S(x)]
           energy, makespan, execution cost, utilisation imbalance, SLA violation

| Symbol | Definition used by the code |
|---|---|
| Execution | A VM runs one task at a time, non-preemptively. Every VM runs its tasks in the global order (arrival, DAG rank, index). `start = max(arrival, latest predecessor finish, VM free time)`, `finish = start + length / mips`. |
| M (makespan) | Latest task finish time (s). |
| E (energy) | A host is on from the first start to the last finish of the tasks placed on it; hosts that get no task stay off. While on, `P = p_idle + (p_max - p_idle) * u`, with `u` = MIPS of the host's busy VMs / total MIPS of its VMs. Integrated over time (J). Equivalent closed form: `p_idle * on-time + (p_max - p_idle) / capacity * MI executed`. |
| C (cost) | Sum over VMs of `price per second x busy seconds`, with `price = 1.4e-5 * (mips / 1000)^1.3`. |
| I (imbalance) | Coefficient of variation (population std / mean) of VM utilisation, utilisation = busy seconds / M. |
| S (SLA) | Fraction of tasks finishing after their deadline. Deadline = arrival + k x (heaviest path to and including the task, in MI) / mean VM MIPS, k ~ U(3, 10). |

**Tolerance envelope (constraint).** The reference schedule of an instance is the faster (smaller
makespan) of FCFS and Min-Min. A schedule is *feasible* / *accepted* when, relative to the reference,
makespan rises by at most 5%, cost by at most 5% and SLA violation by at most 1 percentage point.
Constraint violation `CV = max(0, M/M_ref - 1.05) + max(0, C/C_ref - 1.05) + max(0, S - S_ref - 0.01)`.
Search uses feasibility-first dominance (Deb et al., 2002): feasible beats infeasible; between two
infeasible schedules the smaller CV wins; between feasible ones, Pareto dominance.

**Deployed schedule.** A multi-objective method returns an archive; the schedule it deploys is chosen
by a rule fixed in advance: *tolerance-constrained minimum energy* (lowest energy among feasible
archive members, else smallest CV). A *knee* rule (closest to the ideal point after min-max
normalisation) is reported as a sensitivity check.

## 2. Workloads, infrastructure and seeding (proposal 3.1, 3.3)

* Factors: tasks {500, 2000, 5000} x VMs {20, 50, 100} x heterogeneity {low, medium, high} x arrival
  {steady, bursty, workflow} = **81 scenarios**, 30 replicates each = **2,430 instances**.
* VM MIPS ~ U(range of the heterogeneity level); 4 VMs per host; host models alternate between the
  HP ProLiant ML110 G4 (86 / 117 W) and G5 (93.7 / 135 W) SPECpower profiles used by CloudSim.
* Independent tasks (D16): published job logs, not generated. Windows of consecutive jobs from the
  cleaned Parallel Workloads Archive logs HPC2N-2002-2.2-cln (202,871 jobs) and NASA-iPSC-1993-3.1-cln
  (18,066 jobs with positive runtime of 18,239), listed with SHA-256 in `data/joblogs_manifest.csv` and
  downloaded by hand (the archive blocks scripted downloads). Each instance picks a log at random, then a
  window of n consecutive jobs. Length (MI) = recorded runtime x 1,000 MIPS, rescaled by one factor to a
  mean of 50,000 MI; parallelism is not modelled (one job = one task). Arrivals = recorded submit times
  of the window, shifted to 0 and rescaled in time so the offered load is 0.7 of capacity. *Steady* /
  *bursty*: windows whose inter-arrival coefficient of variation is in the lowest / highest third of all
  n-job windows of that log (`joblog_window_quantiles`). Jobs with runtime <= 0 are dropped.
* *Workflow* (D15): published execution traces, not generated. 73 WfFormat 1.6 traces of real Pegasus runs
  on the Chameleon testbed from WfInstances (Coleman et al., 2022; LGPL v3.0), pinned to commit 89e1f4c
  (2026-11-09) and listed with task count and SHA-256 in `data/wfinstances_manifest.csv`: Montage (14 of 17
  instances, those with at most 5,000 tasks), Epigenomics (26), Seismology (11) and 1000Genome (22).
  `python/fetch_traces.py` downloads them (not redistributed). Tasks and dependencies are used unchanged;
  length (MI) = recorded runtime x 1,000 MIPS, with the 127 runtimes recorded as 0 s raised to 0.003 s;
  each draw picks an application at random among those with a trace that still fits, then one of its
  fitting traces; traces arrive as a Poisson stream until the scenario's task total; a remainder below 41
  tasks is filled with jobs drawn at random from a job log; one common factor rescales all lengths to the same mean task
  length and offered load. Deadlines follow the same rule as independent tasks.
* Instance seed = `SeedSequence([master_seed, 1, scenario, replicate])`, identical for every algorithm
  (common random numbers). Algorithm seed = `SeedSequence([master_seed, 2, scenario, replicate,
  algorithm index])`. Tuning uses a different master seed (777) and never touches test seeds.
* Failed or interrupted runs are re-run with the same seeds and logged in `run_log.csv`.

## 3. Algorithms (proposal 3.1, 3.2)

| Name | Role | Description |
|---|---|---|
| `fcfs` | baseline | Tasks in execution order; earliest-start VM. |
| `minmin` | baseline | Min-Min with ready times and precedence. |
| `ga` | single global metaheuristic (D1) | Generational GA, equal-weight scalarisation normalised by the reference, CV penalty 10, random initial population. |
| `mogwo` | recent multi-objective baseline (D2) | Multi-objective grey wolf optimiser (Mirjalili et al., 2016) as applied to cloud scheduling by Khan et al. (2024); continuous positions decoded by truncation; grid archive. |
| `hpeas` | proposed | Heuristic seeding + NSGA-II global search + adaptive local search + external archive. |
| `hpeas_noseed`, `hpeas_nols`, `hpeas_static`, `nsga2` | ablation | HPEAS without seeding; without local search; local search on a fixed schedule with uniform operator choice; global search component alone. |
| `ga_seeded`, `mogwo_seeded` | fairness check | Competitors given HPEAS's seeded initial population. |
| `energy_greedy` | supplementary | HPEAS's energy-preferred seed heuristic on its own. |

All metaheuristics stop after the same number of objective evaluations (10,000, including the two
reference evaluations) and share population size 40 and archive size 100.

## 4. Decision rules and statistics (proposal 3.5)

* Per algorithm and instance: change in energy, makespan and cost relative to the reference (%), SLA
  change (pp), imbalance, whether the deployed schedule is inside the envelope, hypervolume and IGD
  (objectives min-max normalised by the instance's reference front = non-dominated union of all
  algorithms' solutions; HV reference point 1.1 in every objective), Schott spacing and wall-clock time.
* An energy improvement counts only when the deployed schedule is inside the envelope.
* Friedman test across the five main methods (blocks = instances); Holm-corrected pairwise Wilcoxon
  signed-rank tests of HPEAS against each other method; Cliff's delta with Romano et al. (2006)
  magnitude labels; bootstrap 95% CIs of medians (10,000 resamples); per-scenario one-sided Wilcoxon
  tests of HPEAS energy against the reference, Holm-corrected over the 81 scenarios. alpha = 0.05.
* Carbon (secondary scenario): energy x fixed grid intensity of 50, 300 and 700 gCO2/kWh. These are
  estimates from simulated energy, not measured emissions.

## 5. Validation (proposal 3.4)

1. `testdata/tiny_case.txt`: three tasks, two VMs, every objective computed by hand (in the file).
2. `testdata/parity_case.txt` (bursty) and `parity_workflow_case.txt` (workflow DAGs): Python's event-by-
   event oracle and Java's closed-form evaluator must agree to 1e-6 relative, and both baseline
   assignments must match exactly (`make test`).
3. `gcs.opt.OptSelfCheck`: dominance, archive invariants, hypervolume/IGD against Python, exact budget
   accounting and seed determinism for every algorithm.
4. `gcs.validate_cloudsim`: FCFS, Min-Min and energy-greedy schedules of every independent-task scenario
   with up to 2,000 tasks are replayed in CloudSim Plus 8.5.7; makespan and energy must agree within 0.1%.

## 6. Decisions log (v0 open decisions -> v1)

| # | Decision | Resolution |
|---|---|---|
| D1 | Single global metaheuristic | Genetic algorithm with equal-weight scalarisation. |
| D2 | Recent multi-objective baseline | MOGWO (Khan et al., 2024); it is reviewed in the proposal's Section 2.2. |
| D3 | Power model parameters | SPECpower profiles of the HP ML110 G4/G5 (as in CloudSim); alternative energy-proportional curve (idle 30% of peak) as a construct-validity check. Hosts now switch off when idle before their first or after their last task, so consolidation matters. |
| D4 | "No VM performs more than one task" | One task at a time per VM, FIFO queue. |
| D5 | Billing | Per second of busy time; price super-linear in speed (exponent 1.3), otherwise cost would be identical for every schedule. |
| D6 | Garbled tolerance wording | Makespan +5%, cost +5%, SLA +1 pp, relative to the faster deterministic baseline. |
| D7 | Workflow workloads | v1.x: synthetic DAG generator following published structures (Bharathi et al., 2008; Juve et al., 2013). Superseded in v2.0 by published traces (D15). |
| D8 | Gantt task name "Hybrid GSCOA implementation" | Report refers to "HPEAS implementation". |
| D9 | Workload stress | v1.x: bursts made independent of run length (MMPP). Calibration run on tuning seeds: median FCFS SLA violation 0% steady, 3-26% bursty, 23-35% workflow. Superseded in v2.0 by published job logs and traces (D15, D16). |
| D10 | Full factorial vs sample | Full factorial: the run-time pilot (a project-plan activity, milestone M4, not part of the method) showed 12 algorithms x 2,430 instances fit in under an hour on 8 cores. |
| D11 | Tuning | Grid on six development scenarios x 3 tuning replicates, scored by median in-envelope energy ratio; see `results/tuning/`. GA tuned separately (`ga_pm`, `ga_pc`). |
| D12 | Tolerance sensitivity defect | `--param tol_*` was overwritten by the configured tolerances, so the first tol_tight / tol_loose runs used the default envelope (results identical to `default` apart from run time). Fixed in `experiment.py`; both variants re-run; default runs verified identical to the golden baseline after the fix; the invalid runs are kept in `results/sensitivity_invalid_tol`. |
| D13 | Provenance and run order (proposal 3.4 gaps) | Every selected schedule (assignment vector, its SHA-256, algorithm seed) is stored in `sNN_schedules.csv.gz`; the run configuration (config file, algorithms, budget, parameters, seed stream, SHA-256 of the Java and Python sources) is written to `config_<hash>.json` and its hash is stored with every schedule and every `run_log.csv` attempt. Scenarios, instances and algorithms execute in a seeded random order (`--order-seed`, default the master seed; position logged); outputs keep canonical order. Regression on 4 scenarios x 2 replicates x 12 methods: run rows identical except `runtime_s`, traces byte-identical, stored schedules re-evaluate to the reported objectives with the Python oracle. Main experiment re-run with these features into `results/main`; the earlier run is kept as `results/main_v1` for comparison. |
| D14 | Supervisor feedback on the methodology (Research_Report_MK, Nov 2026) | Workflow data sources, licences, versions, access dates and the modification and scaling of the generated workflows are stated in the report (D7). The run-time pilot belongs to the project plan (proposal 3.8, milestone M4), not the method: the report's methodology calls the tuning-seed workload check a calibration run (D9). Documentation and report text only; no code, configuration or result changed. |
| D15 | Published workflow traces (v2.0; supervisor feedback, Nov 2026) | Workflow scenarios use real WfInstances traces instead of generated DAGs (section 2). CyberShake and Inspiral have no published traces and are replaced by Seismology and 1000Genome. Config: `workflow_size_range` replaced by `trace_reference_mips: 1000` (renamed in D16) and `workflow_trace_min_runtime_s: 0.003`. Regression: all 324 steady/bursty instances checked (54 scenarios x 3 replicates, each compared twice) are byte-identical to v1.2.1; `parity_workflow_case.txt` regenerated from scenario 5 and passes SelfCheck and OptSelfCheck. Calibration on tuning seeds (`results/calib2_traces`, 27 workflow scenarios x 2 replicates): median FCFS SLA violation 51% (4-85% by size and VMs), Min-Min 22-88%, energy-greedy 0-82%; heavier than the generated workflows (FCFS 23-35%) but not collapsed, since the deadline-aware heuristic meets most deadlines except with 20 VMs and 2,000-5,000 tasks, and the tolerance envelope is relative to the instance's reference. Deadline rule therefore unchanged. Parameters re-tuned on the tuning seeds and all experiments re-run; v1.2.1 outputs kept in `results/*_v2`. |
| D16 | Published job logs for independent tasks (v2.0; student's instruction: no self-created datasets) | Steady and bursty scenarios use windows of the HPC2N and NASA iPSC logs from the Parallel Workloads Archive (Feitelson et al., 2014) instead of lognormal lengths with Poisson / MMPP arrivals (section 2). This departs from proposal 3.3, which planned synthetic independent tasks; the steady/bursty factor is kept by classifying real windows by arrival irregularity, but it is no longer set by the experimenter, and even steady windows are burstier than Poisson (median coefficient of variation about 2-3 against 5-8 for bursty). Config: `task_length_sigma` and `bursty` removed; `joblog_window_quantiles: [1/3, 2/3]` added. Only the data centre (VM speeds, prices, power), deadlines and random choices remain generated. The 30 replicates are random windows and may overlap, most in the shorter NASA log. `parity_case.txt` and `parity_workflow_case.txt` regenerated (scenarios 4 and 5); 31 Python tests and all Java checks pass. Calibration on tuning seeds (`results/calib3_published`, 81 scenarios x 2 replicates): median SLA violation steady FCFS 10% / Min-Min 26% / energy-greedy 0%, bursty 20% / 34% / 0%, workflow 45% / 51% / 8%; deadline rule unchanged. GA and HPEAS re-tuned on the tuning seeds (`results/tuning`) and frozen before any test run: HPEAS pm 1, pc 0.9, stagnation 1, ls_steps 20 (median in-envelope energy ratio 0.901 on the development instances, against 0.967 for the v1.x setting); GA ga_pm 1, ga_pc 0.6 (all GA medians tie at 1.0, so the best mean decides, as in D11). v1.x tuning kept in `results/tuning_v2`. All experiments re-run. |
