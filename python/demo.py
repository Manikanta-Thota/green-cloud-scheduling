"""Live demo: run FCFS, Min-Min and HPEAS on one scenario and print the comparison as a table.

The comparison is the same one the analysis uses (gcs.analysis.deployed): each method's deployed
schedule against the reference, the faster of FCFS and Min-Min, with the tolerance envelope
makespan +5%, cost +5%, SLA +1 pp.

    python demo.py                 # scenario 4: 500 tasks, 20 VMs, bursty job-log window
    python demo.py --scenario 8    # scenario 8: 500 tasks, 20 VMs, published workflows
"""
import argparse
import os
import subprocess
import sys
import tempfile

import pandas as pd

from gcs.analysis import LABEL, deployed

p = argparse.ArgumentParser()
p.add_argument("--scenario", type=int, default=4)
p.add_argument("--rep", type=int, default=1, help="number of replicates")
p.add_argument("--out", default=None, help="output folder (default: a temporary folder)")
args = p.parse_args()

out = args.out or tempfile.mkdtemp(prefix="hpeas_demo_")
subprocess.run([sys.executable, "-m", "gcs.experiment", "--out", out, "--scenarios", str(args.scenario),
                "--reps", str(args.rep), "--algos", "fcfs,minmin,hpeas"], check=True)

runs = pd.read_csv(os.path.join(out, f"s{args.scenario:02d}_runs.csv"))
d = deployed(runs)
first = d.iloc[0]
ref = runs[runs.algorithm.isin(["fcfs", "minmin"])].sort_values(["makespan", "algorithm"]).iloc[0].algorithm

print()
print(f"Scenario {args.scenario}: {first.tasks} tasks, {first.vms} VMs, {first.heterogeneity} heterogeneity, "
      f"{first.arrival} arrivals; {d.rep.nunique()} replicate(s)")
print(f"Reference = {LABEL[ref]} (the faster of FCFS and Min-Min); tolerance: makespan +5%, cost +5%, SLA +1 pp")
print()
t = (d.groupby("algorithm")
     .agg(energy_kwh=("energy", lambda e: e.median() / 3.6e6), makespan_s=("makespan", "median"),
          energy_chg=("energy_chg", "median"), makespan_chg=("makespan_chg", "median"),
          cost_chg=("cost_chg", "median"), sla_pp=("sla_chg_pp", "median"),
          within=("accepted", "mean"), runtime_s=("runtime_s", "median"))
     .reindex(["fcfs", "minmin", "hpeas"]))
t.index = [LABEL[a] for a in t.index]
t["within"] = (100 * t.within).map("{:.0f}%".format)
t.columns = ["Energy kWh", "Makespan s", "Energy %", "Makespan %", "Cost %", "SLA pp", "In envelope", "Runtime s"]
print(t.round(3).to_string())
print()
h = t.loc["HPEAS"]
print(f"HPEAS uses {-h['Energy %']:.1f}% less energy than {LABEL[ref]} "
      f"({'inside' if h['In envelope'] == '100%' else 'not always inside'} the tolerance envelope).")
print(f"Raw files: {out}")
