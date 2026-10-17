"""Benchmark one Python engine on one prepared dataset (docs/RUST_ENGINE.md section 8).

Engines:
  scanpy      Scanpy on CPU
  rapids      rapids-singlecell on GPU (CuPy / RAPIDS)
  scfast-cpu  scfast Rust engine, CPU backend   (records 'unavailable' until built)
  scfast-gpu  scfast Rust engine, CUDA backend  (records 'unavailable' until built)

Steps (same names and parameters as bench/run_seurat.R): read, normalize, hvg,
scale, pca, neighbors, cluster, umap; GPU engines add to_gpu / to_cpu transfers.
Repeat 0 is a warm-up and is flagged; repeats 1..N are measured.

    python bench/run_py.py --engine scanpy --data bench/results/run1/data/n2638 \
        --out bench/results/run1 --reps 3 --threads 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

T_PROCESS = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (stdlib-only import; numpy not loaded yet)

ENGINES = ("scanpy", "rapids", "scfast-cpu", "scfast-gpu")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--engine", choices=ENGINES, required=True)
    p.add_argument("--data", required=True, help="prepared folder with matrix.h5 (bench/prep.py)")
    p.add_argument("--out", required=True, help="run folder; results.jsonl is appended here")
    p.add_argument("--reps", type=int, default=3, help="measured repeats (plus one warm-up)")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--k", type=int, default=20, help="neighbours including the cell itself (Seurat k.param)")
    p.add_argument("--resolution", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-umap", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------- engines

class ScanpyEngine:
    device = "cpu"

    def startup(self):
        import importlib.metadata as md

        import scanpy as sc

        self.sc, self.md = sc, md
        return {"scanpy": md.version("scanpy"), "anndata": md.version("anndata")}

    def read(self, path):
        a = self.sc.read_10x_h5(path)
        a.var_names_make_unique()
        return a

    def normalize(self, a, cfg):
        a.layers["counts"] = a.X.copy()
        self.sc.pp.normalize_total(a, target_sum=1e4)
        self.sc.pp.log1p(a)

    def hvg(self, a, cfg):
        self.sc.pp.highly_variable_genes(a, n_top_genes=cfg.n_hvg, flavor="seurat_v3", layer="counts")

    def scale(self, a, cfg):
        b = a[:, a.var.highly_variable].copy()
        del b.layers["counts"]
        self.sc.pp.scale(b, max_value=10)
        return b

    def pca(self, a, cfg):
        self.sc.pp.pca(a, n_comps=cfg.n_pcs, random_state=cfg.seed)

    def neighbors(self, a, cfg):
        self.sc.pp.neighbors(a, n_neighbors=cfg.k, n_pcs=cfg.n_pcs, random_state=cfg.seed)

    def cluster(self, a, cfg):
        self.sc.tl.leiden(a, resolution=cfg.resolution, random_state=cfg.seed, flavor="igraph",
                          n_iterations=2, directed=False, key_added="cluster")

    def umap(self, a, cfg):
        self.sc.tl.umap(a, random_state=cfg.seed)


class RapidsEngine(ScanpyEngine):
    device = "gpu"

    def startup(self):
        versions = super().startup()
        import cupy as cp
        import rapids_singlecell as rsc

        cp.zeros(1)  # create the CUDA context now, so it counts as start-up
        cp.cuda.Device().synchronize()
        self.cp, self.rsc = cp, rsc
        self.pool = cp.get_default_memory_pool()
        versions.update({"rapids_singlecell": self.md.version("rapids_singlecell"), "cupy": cp.__version__})
        return versions

    def sync(self):
        self.cp.cuda.Device().synchronize()

    def gpu_mem(self):
        # bytes held by CuPy's pool after the step: an upper bound on live allocations
        return int(self.pool.total_bytes())

    def to_gpu(self, a):
        self.rsc.get.anndata_to_GPU(a)

    def to_cpu(self, a):
        self.rsc.get.anndata_to_CPU(a, convert_all=True)

    def normalize(self, a, cfg):
        a.layers["counts"] = a.X.copy()
        self.rsc.pp.normalize_total(a, target_sum=1e4)
        self.rsc.pp.log1p(a)

    def hvg(self, a, cfg):
        self.rsc.pp.highly_variable_genes(a, n_top_genes=cfg.n_hvg, flavor="seurat_v3", layer="counts")

    def scale(self, a, cfg):
        b = a[:, a.var.highly_variable].copy()
        del b.layers["counts"]
        self.rsc.pp.scale(b, max_value=10)
        return b

    def pca(self, a, cfg):
        self.rsc.pp.pca(a, n_comps=cfg.n_pcs, random_state=cfg.seed)

    def neighbors(self, a, cfg):
        self.rsc.pp.neighbors(a, n_neighbors=cfg.k, n_pcs=cfg.n_pcs, random_state=cfg.seed)

    def cluster(self, a, cfg):
        self.rsc.tl.leiden(a, resolution=cfg.resolution, random_state=cfg.seed, n_iterations=2,
                           key_added="cluster")

    def umap(self, a, cfg):
        self.rsc.tl.umap(a, random_state=cfg.seed)


class Unavailable(RuntimeError):
    pass


def make_engine(name: str):
    if name == "scanpy":
        return ScanpyEngine()
    if name == "rapids":
        return RapidsEngine()
    # scfast engines: the Python module arrives in Phase 1 (rust/scfast-py).
    try:
        import scfast  # noqa: F401
    except ImportError as e:
        raise Unavailable(f"scfast is not built yet ({e})") from e
    raise Unavailable("scfast is importable but its bench adapter is not implemented yet (Phase 5)")


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    cfg = parse(argv)
    common.set_threads(cfg.threads)  # before numpy / scanpy are imported
    data = Path(cfg.data)
    h5 = data / "matrix.h5"
    prep = json.loads((data.parent / "prep.json").read_text()) if (data.parent / "prep.json").exists() else {}
    n_cells = sum(1 for _ in (data / "cells.tsv").open()) - 1
    out = Path(cfg.out)
    base = {"engine": cfg.engine, "cells": n_cells, "genes": prep.get("genes_qc"),
            "dataset": data.name, "threads": cfg.threads, "commit": common.repo_commit()}

    try:
        engine = make_engine(cfg.engine)
    except Unavailable as e:
        common.Recorder(out / "results.jsonl", {**base, "device": "n/a"}).write(
            step="all", kind="startup", status="unavailable", error=str(e), wall_s=None)
        print(f"{cfg.engine}: unavailable: {e}")
        return 0

    rec = common.Recorder(out / "results.jsonl", {**base, "device": engine.device})
    rec.warmup = True
    t0 = time.perf_counter()
    try:
        versions = engine.startup()
    except Exception as e:
        rec.write(step="startup", kind="startup", status="unavailable", error=f"{type(e).__name__}: {e}",
                  wall_s=None)
        print(f"{cfg.engine}: start-up failed: {e}")
        return 0
    rss, _ = common.peak_rss_bytes()
    rec.write(step="startup", kind="startup", status="ok", error=None,
              wall_s=round(time.perf_counter() - t0, 6),
              process_to_ready_s=round(time.perf_counter() - T_PROCESS, 6), peak_rss_gb=round(rss / 1e9, 4))
    if engine.device == "gpu":
        rec.sync, rec.gpu_mem = engine.sync, engine.gpu_mem

    env = common.environment(versions)
    env_path = out / "env" / f"{cfg.engine}_{data.name}.json"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(json.dumps({**env, "config": vars(cfg)}, indent=2))
    rec.base["node"] = env["node"]

    import numpy as np

    for rep in range(cfg.reps + 1):
        rec.rep, rec.warmup = rep, rep == 0
        try:
            with rec.step("read", kind="io"):
                a = engine.read(str(h5))
            if engine.device == "gpu":
                with rec.step("to_gpu", kind="transfer"):
                    engine.to_gpu(a)
            with rec.step("normalize"):
                engine.normalize(a, cfg)
            with rec.step("hvg"):
                engine.hvg(a, cfg)
            with rec.step("scale"):
                a = engine.scale(a, cfg)
            with rec.step("pca"):
                engine.pca(a, cfg)
            with rec.step("neighbors"):
                engine.neighbors(a, cfg)
            with rec.step("cluster"):
                engine.cluster(a, cfg)
            if not cfg.skip_umap:
                with rec.step("umap"):
                    engine.umap(a, cfg)
            if engine.device == "gpu":
                with rec.step("to_cpu", kind="transfer"):
                    engine.to_cpu(a)
        except Exception as e:  # the failed step is already recorded; stop this engine
            print(f"{cfg.engine} rep {rep}: {type(e).__name__}: {e}")
            return 1
        if rep == 1 or (cfg.reps == 0 and rep == 0):  # parity outputs from the first measured repeat
            knn = common.knn_from_distances(a.obsp["distances"], cfg.k - 1)
            common.write_outputs(out / "outputs" / cfg.engine / data.name, a.obs_names,
                                 np.asarray(a.obsm["X_pca"])[:, : cfg.n_pcs], knn,
                                 a.obs["cluster"].astype(str).tolist())
        del a
    print(f"{cfg.engine} {data.name}: done ({cfg.reps} measured repeats)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
