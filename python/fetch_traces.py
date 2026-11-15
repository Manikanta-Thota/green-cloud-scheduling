"""Fetch and verify the published datasets behind every workload (protocol D15, D16).

Workflow scenarios: WfInstances workflow traces. The traces are real Pegasus executions on the Chameleon testbed, published by the WfCommons project
(Coleman et al., 2022) under the GNU LGPL v3.0. They are not redistributed with this repository: this
script fetches them from a pinned commit and records name, task count and SHA-256 of every file in
data/wfinstances_manifest.csv, which is committed so that anyone can verify they use the same data.

Independent-task scenarios: two cleaned job logs from the Parallel Workloads Archive (Feitelson et al.,
2014), HPC2N-2002-2.2-cln and NASA-iPSC-1993-3.1-cln, listed with SHA-256 in data/joblogs_manifest.csv.
The archive's web server blocks scripted downloads, so these two files are downloaded by hand in a
browser from the URLs in the manifest into data/joblogs/; this script then verifies them.

    python fetch_traces.py            # download missing traces, (re)write the trace manifest, check the logs
    python fetch_traces.py --verify   # check every local file against the committed manifests
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import urllib.request

REPO = "wfcommons/WfInstances"
COMMIT = "89e1f4cce20a96f5aed369781f99e0b335ebcf41"   # main on 2026-11-09, accessed 2026-11-10
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "wfinstances")
MANIFEST = os.path.join(ROOT, "data", "wfinstances_manifest.csv")
LOG_DIR = os.path.join(ROOT, "data", "joblogs")
LOG_MANIFEST = os.path.join(ROOT, "data", "joblogs_manifest.csv")
MAX_TASKS = 5000   # every trace must fit the largest scenario

MONTAGE = ([f"montage-chameleon-2mass-{d}-001.json" for d in
            ("005d", "01d", "015d", "02d", "025d", "03d", "04d", "05d", "10d")] +
           [f"montage-chameleon-dss-{d}-001.json" for d in ("05d", "075d", "10d", "125d", "15d")])
EPIGENOMICS = [f"epigenomics-chameleon-{d}-{n}seq-{b}-001.json"
               for d, top in (("hep", 7), ("ilmn", 6)) for n in range(1, top + 1) for b in ("50k", "100k")]
SEISMOLOGY = [f"seismology-chameleon-{p}p-001.json" for p in range(100, 1101, 100)]
GENOME = [f"1000genome-chameleon-{c}ch-{s}-001.json" for c in range(2, 23, 2) for s in ("100k", "250k")]
FILES = {"montage": MONTAGE, "epigenomics": EPIGENOMICS, "seismology": SEISMOLOGY, "1000genome": GENOME}


def local_path(app: str, name: str) -> str:
    return os.path.join(DATA, app, name)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def n_tasks(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return len(json.load(f)["workflow"]["specification"]["tasks"])


def download(app: str, name: str) -> None:
    url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/pegasus/{app}/{name}"
    os.makedirs(os.path.join(DATA, app), exist_ok=True)
    tmp = local_path(app, name) + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    os.replace(tmp, local_path(app, name))


def fetch() -> None:
    rows = []
    for app, names in FILES.items():
        for name in names:
            path = local_path(app, name)
            if not os.path.exists(path):
                print("downloading", app, name)
                download(app, name)
            n = n_tasks(path)
            assert n <= MAX_TASKS, (name, n)
            rows.append({"application": app, "file": name, "tasks": n, "sha256": sha256(path),
                         "source": f"https://github.com/{REPO}/blob/{COMMIT}/pegasus/{app}/{name}"})
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} traces, manifest written to {MANIFEST}")
    verify_logs()


def verify_logs() -> None:
    """The job logs cannot be fetched by script: check them, or say exactly what to download."""
    with open(LOG_MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        path = os.path.join(LOG_DIR, r["file"])
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}: download {r['source']} in a browser and save it there unchanged")
        if sha256(path) != r["sha256"]:
            raise SystemExit(f"{path} differs from the manifest (SHA-256); download it again from {r['source']}")
    print(f"all {len(rows)} job logs match the manifest")


def verify() -> None:
    with open(MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bad = [r["file"] for r in rows
           if not os.path.exists(local_path(r["application"], r["file"]))
           or sha256(local_path(r["application"], r["file"])) != r["sha256"]]
    if bad:
        raise SystemExit(f"{len(bad)} trace file(s) missing or different: {bad[:5]} - run fetch_traces.py")
    print(f"all {len(rows)} traces match the manifest")
    verify_logs()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true")
    (verify if ap.parse_args().verify else fetch)()
