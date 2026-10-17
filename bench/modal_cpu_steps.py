"""scfast CPU PCA / kNN / SNN / Leiden at 100k-1.3M cells on Modal, before and after a change.

    uvx modal run bench/modal_cpu_steps.py::inputs                 # HVG matrices -> volume (once)
    uvx modal run bench/modal_cpu_steps.py::steps                  # both builds, JSON -> bench/results

Inputs (volume `scfast-data`, folder cpu_steps/): the prepared 10x brain subsets (prep/brain,
see modal_baselines.py) and a synthetic 1M-cell set with planted types (bench/synth.py), each
reduced to its 2,000 HVGs by scfast.preprocess and stored as hvg2000.npz plus cells.tsv.
The image builds two scfast wheels from source snapshots taken first, so editing the tree
during an upload cannot break the build:

    rm -rf /tmp/scfast_before /tmp/scfast_after && mkdir -p /tmp/scfast_before /tmp/scfast_after
    git archive f51a3ba rust | tar -x -C /tmp/scfast_before
    rsync -a --exclude target rust /tmp/scfast_after/
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
CPUS, MEM_GB = 16, 48
BEFORE_SRC, AFTER_SRC = Path("/tmp/scfast_before"), Path("/tmp/scfast_after")
RUST_IGNORE = ["target", "**/target", "scfast-r"]

app = modal.App("scfast-cpu-steps")
vol = modal.Volume.from_name("scfast-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal")
    .pip_install("maturin", "numpy", "scipy", "pandas", "h5py", "scikit-learn", "anndata")
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(BEFORE_SRC / "rust", "/src/before/rust", copy=True, ignore=RUST_IGNORE)
    .add_local_dir(AFTER_SRC / "rust", "/src/after/rust", copy=True, ignore=RUST_IGNORE)
    .run_commands(*[
        f"cd /src/{b}/rust/scfast-py && maturin build --release -o /wheels/{b} "
        f"&& pip install --no-deps --target /opt/{b} /wheels/{b}/*.whl" for b in ("before", "after")
    ])
    .add_local_dir(ROOT / "bench", "/work/bench", ignore=["results", "hpc", "__pycache__"])
)

DATA = {
    "brain100k": "prep/brain/n100000/matrix.h5",
    "brain500k": "prep/brain/n500000/matrix.h5",
    "brain1290k": "prep/brain/n1290549/matrix.h5",
    "synth1m": None,
}


def sh(cmd: list[str], env: dict | None = None) -> str:
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd="/work", env={**os.environ, **(env or {})}, capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-4000:], flush=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[:3]} exited with {r.returncode}")
    return r.stdout


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=96 * 1024, timeout=2 * 3600)
def make_input(name: str):
    import shutil
    import sys

    import numpy as np
    import scipy.sparse as sp

    sys.path[:0] = ["/opt/after", "/work/bench"]
    import scfast
    from h5io import read_counts

    out = Path(f"/data/cpu_steps/{name}")
    if (out / "hvg2000.npz").exists():
        return f"{name}: exists"
    out.mkdir(parents=True, exist_ok=True)
    if DATA[name] is None:
        tmp = Path("/tmp/synth")
        sh(["python", "bench/synth.py", "--cells", "1000000", "--genes", "5000", "--types", "20",
            "--out", str(tmp / "synth_1m.h5")], env={"PYTHONPATH": "/work/bench"})
        h5 = tmp / "synth_1m.h5"
        shutil.copy(f"{h5}.labels.tsv", out / "cells.tsv")
    else:
        h5 = Path("/data") / DATA[name]
        shutil.copy(h5.parent / "cells.tsv", out / "cells.tsv")
    x = read_counts(h5)[0]
    x.sum_duplicates()
    hv = scfast.preprocess(x, n_hvg=2000)["X_hvg"]
    sp.save_npz(out / "hvg2000.npz", sp.csr_matrix(hv, dtype=np.float32), compressed=False)
    vol.commit()
    return f"{name}: {x.shape[0]} cells, {x.nnz} counts, hvg nnz {hv.nnz}"


@app.function(image=image, volumes={"/data": vol}, cpu=CPUS, memory=MEM_GB * 1024, timeout=3600)
def run_steps(build: str, name: str, methods: str, reps: int) -> dict:
    data = Path(f"/data/cpu_steps/{name}")
    work = Path(f"/tmp/{name}")  # cpu_steps caches pcs30.npy next to the data; keep the volume clean
    work.mkdir(parents=True, exist_ok=True)
    for f in ("hvg2000.npz", "cells.tsv"):
        (work / f).symlink_to(data / f)
    labels = Path(f"/data/cpu_steps/labels/{build}/{name}")
    out = sh(["python", "bench/cpu_steps.py", "--data", str(work), "--methods", methods, "--reps", str(reps),
              "--save-labels", str(labels)],
             env={"PYTHONPATH": f"/opt/{build}", "RAYON_NUM_THREADS": str(CPUS)})
    vol.commit()
    res = json.loads(out.strip().splitlines()[-1])
    return {"build": build, "name": name, "cpus": CPUS, **res}


@app.local_entrypoint()
def inputs(names: str = ",".join(DATA)):
    for msg in make_input.map(names.split(","), order_outputs=False):
        print(msg)


@app.local_entrypoint()
def steps(names: str = "brain100k,brain500k,brain1290k,synth1m", reps: int = 3, tag: str = "modal"):
    jobs = []
    for n in names.split(","):
        jobs.append(("before", n, "exact", reps))
        jobs.append(("after", n, "exact,kdtree", reps))
    dst = ROOT / "bench" / "results" / "cpu_steps"
    dst.mkdir(parents=True, exist_ok=True)
    for r in run_steps.starmap(jobs, order_outputs=False):
        (dst / f"{tag}_{r['build']}_{r['name']}.json").write_text(json.dumps(r, indent=1))
        print(json.dumps(r))
