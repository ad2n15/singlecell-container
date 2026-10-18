"""Times scfast.preprocess and scfast.pca on Modal cpu=16 for several scfast builds side by side.

    SCFAST_PROBE_BUILDS=base=/tmp/base/rust,new=rust uvx modal run bench/modal_pca_probe.py::main --sizes 100000,1290549
    SCFAST_PROBE_BUILDS=base=/tmp/base/rust,new=rust uvx modal run bench/modal_pca_probe.py::bench_main

`main` times preprocess and pca alone with phase timers; `bench_main` runs bench/run_py.py.

Each build is a `rust/` tree (`git archive <rev> rust | tar -x -C /tmp/base` for a commit; the
default `new` is this checkout's `rust/`; point at a snapshot when editing the checkout during
the run, as Modal rejects files modified mid-upload). Machines differ by up to ~40% between
runs, so builds are compared within one container: every size runs in a fresh process per
build, with SCFAST_TIMING=1 phase timers, and each build's embeddings are compared with the
first build's.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
# Read where the image is defined (locally); inside the container only the names matter.
BUILDS = dict(b.split("=", 1) for b in os.environ.get("SCFAST_PROBE_BUILDS", f"new={ROOT / 'rust'}").split(","))
CPUS = 16

app = modal.App("scfast-pca-probe")
vol = modal.Volume.from_name("scfast-data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "curl")
    .pip_install("numpy", "scipy", "h5py", "maturin>=1.9.4,<2")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable")
    .env({"PYTHONUNBUFFERED": "1", "PATH": "/root/.cargo/bin:$PATH", "RUSTFLAGS": "-C target-cpu=x86-64-v3"})
)
for name, path in BUILDS.items():
    image = image.add_local_dir(path, f"/src/{name}/rust", copy=True, ignore=["**/target"]).run_commands(
        f"cd /src/{name} && maturin build --release -m rust/scfast-py/Cargo.toml -o /tmp/w{name} "
        f"&& pip install --target /opt/{name} /tmp/w{name}/*.whl")
image = image.add_local_dir(ROOT / "bench", "/work/bench", ignore=["results", "hpc", "__pycache__"])

STEP = r'''
import sys, time, numpy as np, scfast
path, out = sys.argv[1], sys.argv[2]
_kernel = scfast._scfast.preprocess_csr
def _timed(*a):
    t = time.perf_counter(); r = _kernel(*a); print(f"  preprocess_csr {time.perf_counter() - t:.3f}", flush=True); return r
scfast._scfast.preprocess_csr = _timed
t = time.perf_counter(); X, _, _ = scfast.read_10x_h5(path)
print(f"read {time.perf_counter() - t:.2f} s, {X.shape}, nnz {X.nnz}, {X.indices.dtype}", flush=True)
for r in range(2):
    t = time.perf_counter(); pp = scfast.preprocess(X, n_hvg=2000); print(f"normalize {time.perf_counter() - t:.3f}", flush=True)
    if r == 0:
        del pp
h = pp["X_hvg"]; del pp, X
print(f"hvg nnz {h.nnz} ({h.nnz / h.shape[0]:.0f} per cell)", flush=True)
for r in range(2):
    t = time.perf_counter(); e, l, s = scfast.pca(h, n_comps=30, seed=0, device="cpu"); print(f"pca {time.perf_counter() - t:.3f}", flush=True)
np.save(out, e)
'''

COMPARE = r'''
import sys, numpy as np
a, b = (np.load(p).astype(np.float64) for p in sys.argv[1:3])
for k in (10, 30):
    qa, _ = np.linalg.qr(a[:, :k]); qb, _ = np.linalg.qr(b[:, :k])
    print(f"top-{k} subspace similarity {np.linalg.norm(qa.T @ qb) ** 2 / k:.10f}")
c = [abs(np.corrcoef(a[:, i], b[:, i])[0, 1]) for i in range(a.shape[1])]
print(f"min column correlation {min(c):.8f}")
'''


@app.function(image=image, volumes={"/data": vol}, cpu=CPUS, memory=96 * 1024, timeout=3600)
def probe(sizes: str, builds: list[str]):
    env = {**os.environ, "RAYON_NUM_THREADS": str(CPUS), "SCFAST_TIMING": "1"}
    Path("/tmp/step.py").write_text(STEP)
    Path("/tmp/compare.py").write_text(COMPARE)
    for size in sizes.split(","):
        path = f"/data/prep/brain/n{size}/matrix.h5"
        for build in builds:
            print(f"=== n{size} {build}", flush=True)
            subprocess.run([sys.executable, "/tmp/step.py", path, f"/tmp/emb_{build}_{size}.npy"],
                           env={**env, "PYTHONPATH": f"/opt/{build}"}, check=True)
        for build in builds[1:]:
            print(f"--- {build} vs {builds[0]}", flush=True)
            subprocess.run([sys.executable, "/tmp/compare.py", f"/tmp/emb_{builds[0]}_{size}.npy",
                            f"/tmp/emb_{build}_{size}.npy"], check=True)


@app.function(image=image, volumes={"/data": vol}, cpu=CPUS, memory=96 * 1024, timeout=3600)
def bench(sizes: str, builds: list[str], reps: int):
    """bench/run_py.py --engine scfast-cpu per build and size (outputs in /tmp, not on the volume)."""
    import json
    import statistics

    env = {**os.environ, "PYTHONPATH": ""}
    env.update({v: str(CPUS) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS")})
    for size in sizes.split(","):
        for build in builds:
            out = Path(f"/tmp/bench_{build}_{size}")
            subprocess.run([sys.executable, "bench/run_py.py", "--engine", "scfast-cpu", "--data", f"/data/prep/brain/n{size}",
                            "--out", str(out), "--reps", str(reps), "--threads", str(CPUS)], cwd="/work",
                           env={**env, "PYTHONPATH": f"/opt/{build}"}, check=True, stdout=subprocess.DEVNULL)
            rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
            steps = {}
            for r in rows:
                if not r.get("warmup") and r.get("status") == "ok" and r.get("wall_s") is not None:
                    steps.setdefault(r["step"], []).append(r["wall_s"])
            print(f"bench n{size} {build}: " + ", ".join(f"{k} {statistics.median(v):.3f}" for k, v in steps.items()),
                  flush=True)


@app.local_entrypoint()
def bench_main(sizes: str = "100000,500000,1290549", reps: int = 3):
    bench.remote(sizes, list(BUILDS), reps)


@app.local_entrypoint()
def main(sizes: str = "100000,500000,1290549"):
    probe.remote(sizes, list(BUILDS))
