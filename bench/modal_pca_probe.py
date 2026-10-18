"""Times scfast.preprocess and scfast.pca on Modal cpu=16 for two scfast builds side by side.

    SCFAST_PROBE_BASE=/tmp/base/rust uvx modal run bench/modal_pca_probe.py --sizes 100000,500000,1290549

SCFAST_PROBE_BASE is a `rust/` tree of the reference commit (`git archive <rev> rust | tar -x -C
/tmp/base`); SCFAST_PROBE_NEW (default: this checkout's `rust/`) is the candidate. Point it at a
snapshot when editing the checkout during the run (Modal rejects files modified mid-upload). Both wheels are installed side by side, every
size runs in a fresh process per build on the same container, with SCFAST_TIMING=1 phase
timers, and the candidate's embeddings are compared with the reference's.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
BASE = Path(os.environ.get("SCFAST_PROBE_BASE", "/tmp/base/rust"))
NEW = Path(os.environ.get("SCFAST_PROBE_NEW", ROOT / "rust"))
CPUS = 16

app = modal.App("scfast-pca-probe")
vol = modal.Volume.from_name("scfast-data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "curl")
    .pip_install("numpy", "scipy", "maturin>=1.9.4,<2")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable")
    .env({"PYTHONUNBUFFERED": "1", "PATH": "/root/.cargo/bin:$PATH", "RUSTFLAGS": "-C target-cpu=x86-64-v3"})
    .add_local_dir(BASE, "/src/base/rust", copy=True, ignore=["**/target"])
    .run_commands("cd /src/base && maturin build --release -m rust/scfast-py/Cargo.toml -o /tmp/wb "
                  "&& pip install --target /opt/base /tmp/wb/*.whl")
    .add_local_dir(NEW, "/src/new/rust", copy=True, ignore=["**/target"])
    .run_commands("cd /src/new && maturin build --release -m rust/scfast-py/Cargo.toml -o /tmp/wn "
                  "&& pip install --target /opt/new /tmp/wn/*.whl")
    .pip_install("h5py")
)

STEP = r'''
import sys, time, numpy as np, scfast
path, out = sys.argv[1], sys.argv[2]
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
def probe(sizes: str, builds: str):
    env = {**os.environ, "RAYON_NUM_THREADS": str(CPUS), "SCFAST_TIMING": "1"}
    Path("/tmp/step.py").write_text(STEP)
    Path("/tmp/compare.py").write_text(COMPARE)
    for size in sizes.split(","):
        path = f"/data/prep/brain/n{size}/matrix.h5"
        for build in builds.split(","):
            print(f"=== n{size} {build}", flush=True)
            subprocess.run([sys.executable, "/tmp/step.py", path, f"/tmp/emb_{build}_{size}.npy"],
                           env={**env, "PYTHONPATH": f"/opt/{build}"}, check=True)
        if builds == "base,new":
            subprocess.run([sys.executable, "/tmp/compare.py", f"/tmp/emb_base_{size}.npy", f"/tmp/emb_new_{size}.npy"],
                           check=True)


@app.local_entrypoint()
def main(sizes: str = "100000,500000,1290549", builds: str = "base,new"):
    probe.remote(sizes, builds)
