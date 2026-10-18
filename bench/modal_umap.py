"""UMAP-only timing on Modal (cpu=16): `scfast.umap` on cached kNN graphs, A/B across git revisions.

    uvx modal run bench/modal_umap.py::setup                      # brain kNN graphs + PBMC 68k -> volume
    uvx modal run bench/modal_umap.py::main --revs 9fa435b,WORKTREE --sizes 100000,500000,1290549

Each revision's `rust/` tree is built into a wheel once (cached on the volume by commit SHA;
WORKTREE is always rebuilt) and timed in its own process, revisions interleaved per dataset.
PBMC 68k reports the silhouette of Seurat's Leiden clusters (seeds 0-2, 15k cells); the largest
brain size writes a PNG per revision coloured by scfast Leiden cluster to bench/results/umap/.
"""

from __future__ import annotations

import io
import os
import subprocess
import tarfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App("scfast-umap")
vol = modal.Volume.from_name("scfast-data")
BASE = "/data/umap_bench"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "curl")
    .pip_install("numpy", "scipy", "scikit-learn", "matplotlib", "maturin>=1.9.4,<2")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable")
    .env({"PYTHONUNBUFFERED": "1", "PATH": "/root/.cargo/bin:$PATH", "RUSTFLAGS": "-C target-cpu=x86-64-v3"})
)

WORKER = r"""
import sys, time, json, numpy as np, scfast
ds, reps, out, n_sil = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
d = "/data/umap_bench/" + ds
i, dist, p = np.load(d + "/idx.npy"), np.load(d + "/dist.npy"), np.load(d + "/init.npy")
ts = []
for r in range(reps):
    t = time.perf_counter(); u = scfast.umap(i, dist, p, seed=0); ts.append(time.perf_counter() - t)
res = {"times": ts}
if ds == "pbmc68k":
    from sklearn.metrics import silhouette_score
    lab = np.load(d + "/labels.npy")
    s = np.random.default_rng(0).choice(len(lab), 15000, replace=False)
    res["sil"] = [float(silhouette_score(u[s], lab[s]))]
    for seed in range(1, n_sil):
        v = scfast.umap(i, dist, p, seed=seed); res["sil"].append(float(silhouette_score(v[s], lab[s])))
np.save(out, u)
r = np.linalg.norm(u - np.median(u, 0), axis=1)
res["strays"] = int((r > 1.5 * np.percentile(r, 99.9)).sum())
print(json.dumps(res))
"""


@app.function(image=image, volumes={"/data": vol}, cpu=16, memory=32768, timeout=1800)
def prep(pbmc: dict[str, bytes]):
    import numpy as np

    os.makedirs(f"{BASE}/pbmc68k", exist_ok=True)
    for name, data in pbmc.items():
        Path(f"{BASE}/pbmc68k/{name}").write_bytes(data)
    vol.commit()


def build(rev: str, tar: bytes) -> str:
    """Returns a directory holding the installed scfast of `rev`."""
    whl_dir = f"{BASE}/wheels/{rev}"
    if rev == "WORKTREE" or not os.path.isdir(whl_dir):
        src = f"/tmp/src/{rev}"
        os.makedirs(src, exist_ok=True)
        tarfile.open(fileobj=io.BytesIO(tar)).extractall(src)
        subprocess.run(f"rm -rf {whl_dir} && maturin build --release -q -m rust/scfast-py/Cargo.toml -o {whl_dir}",
                       shell=True, cwd=src, check=True)
        if rev == "WORKTREE":
            subprocess.run("cargo test --release -q --manifest-path rust/Cargo.toml -p scfast-core umap 2>&1 | grep -E 'panicked|test result' -A4",
                           shell=True, cwd=src)
        vol.commit()
    site = f"/tmp/site/{rev}"
    subprocess.run(f"pip install -q --no-deps --target {site} {whl_dir}/*.whl", shell=True, check=True)
    return site


@app.function(image=image, volumes={"/data": vol}, cpu=16, memory=16384, timeout=3600)
def run(srcs: dict[str, bytes], datasets: list[str], reps: int, png: str | None, threads: list[int],
        sil_seeds: int = 3) -> dict:
    import json

    sites = {rev: build(rev, tar) for rev, tar in srcs.items()}
    results, pngs = {}, {}
    cpu = subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)|L3'", shell=True, capture_output=True, text=True)
    results["cpu"] = cpu.stdout
    for ds in datasets:
        for rev, site in sites.items():
            for th in threads:
                out = f"/tmp/{rev}_{ds}.npy"
                env = {**os.environ, "PYTHONPATH": site, "RAYON_NUM_THREADS": str(th)}
                r = subprocess.run(["python", "-c", WORKER, ds, str(reps), out, str(sil_seeds)], env=env, capture_output=True, text=True)
                key = f"{rev} {ds}" + (f" t{th}" if th != 16 else "")
                results[key] = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
                if r.stderr.strip():
                    results[key]["stderr"] = r.stderr[-1500:]
                print(key, results[key], flush=True)
                if ds == png and r.returncode == 0:
                    pngs[rev] = render(out, f"{BASE}/{ds}/labels.npy", f"{rev} {ds}")
    return {"results": results, "pngs": pngs}


def render(emb_path: str, lab_path: str, title: str) -> bytes:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u, lab = np.load(emb_path), np.load(lab_path)
    order = np.random.default_rng(0).permutation(len(u))
    f, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(u[order, 0], u[order, 1], c=lab[order] % 20, s=0.05, cmap="tab20", linewidths=0, rasterized=True)
    ax.set_title(title)
    buf = io.BytesIO()
    f.savefig(buf, format="png", dpi=110)
    return buf.getvalue()


@app.function(image=image, volumes={"/data": vol}, cpu=16, memory=65536, timeout=1800)
def prep_brain(tar: bytes, sizes: list[int]):
    """kNN (k=20 on 10 PCs) from the scfast-cpu PCs saved by modal_baselines.py, plus its Leiden labels."""
    import numpy as np

    site = build("prep", tar)
    code = r"""
import sys, numpy as np, scfast
n, src, dst = int(sys.argv[1]), sys.argv[2], sys.argv[3]
pcs = np.fromfile(src + "/pcs.f32", dtype=np.float32).reshape(n, 30)
idx, dist = scfast.knn(np.ascontiguousarray(pcs[:, :10]), k=20, method="auto")
lab = np.unique(np.array(open(src + "/labels.txt").read().split()), return_inverse=True)[1]
np.save(dst + "/idx.npy", idx); np.save(dst + "/dist.npy", dist)
np.save(dst + "/init.npy", np.ascontiguousarray(pcs[:, :2])); np.save(dst + "/labels.npy", lab)
"""
    for n in sizes:
        dst = f"{BASE}/brain{n}"
        os.makedirs(dst, exist_ok=True)
        src = f"/data/results/brain/scfast-cpu_n{n}/outputs/scfast-cpu/n{n}"
        subprocess.run(["python", "-c", code, str(n), src, dst], env={**os.environ, "PYTHONPATH": site}, check=True)
        print("prepared", dst, flush=True)
    vol.commit()


def rust_tar(rev: str) -> bytes:
    if rev == "WORKTREE":
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as t:
            t.add(ROOT / "rust", "rust", filter=lambda m: None if "/target" in m.name else m)
        return buf.getvalue()
    return subprocess.run(["git", "archive", rev, "rust"], cwd=ROOT, capture_output=True, check=True).stdout


def sha(rev: str) -> str:
    if rev == "WORKTREE":
        return rev
    return subprocess.run(["git", "rev-parse", "--short", rev], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


@app.local_entrypoint()
def setup(sizes: str = "100000,500000,1290549", pbmc_dir: str = "/tmp/sc/mu",
          labels: str = "bench/results/pbmc68k_cpu16/outputs/seurat/n68260/labels.txt"):
    import numpy as np

    lab = np.unique(np.array((ROOT / labels).read_text().split()), return_inverse=True)[1]
    buf = io.BytesIO()
    np.save(buf, lab)
    pbmc = {"labels.npy": buf.getvalue()}
    for src, dst in [("p68_idx.npy", "idx.npy"), ("p68_dist.npy", "dist.npy")]:
        pbmc[dst] = Path(pbmc_dir, src).read_bytes()
    pca = np.load(Path(pbmc_dir, "p68_pca.npy"))
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(pca[:, :2]))
    pbmc["init.npy"] = buf.getvalue()
    prep.remote(pbmc)
    prep_brain.remote(rust_tar("HEAD"), [int(s) for s in sizes.split(",")])


@app.local_entrypoint()
def main(revs: str = "HEAD,WORKTREE", sizes: str = "1290549", pbmc: bool = True, reps: int = 2, threads: str = "16",
         sil_seeds: int = 3):
    srcs = {sha(r): rust_tar(r) for r in revs.split(",")}
    datasets = (["pbmc68k"] if pbmc else []) + [f"brain{s}" for s in sizes.split(",") if s]
    out = run.remote(srcs, datasets, reps, datasets[-1] if sizes else None, [int(t) for t in threads.split(",")], sil_seeds)
    for k, v in out["results"].items():
        print(k, v)
    dest = ROOT / "bench/results/umap"
    dest.mkdir(parents=True, exist_ok=True)
    for rev, data in out["pngs"].items():
        (dest / f"{rev}_{datasets[-1]}.png").write_bytes(data)
        print("wrote", dest / f"{rev}_{datasets[-1]}.png")
