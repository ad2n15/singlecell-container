"""Real datasets and CPU baselines (Seurat, Scanpy) on Modal, for comparison with scfast.

    uvx modal run bench/modal_baselines.py::download                  # raw data -> volume scfast-data
    uvx modal run bench/modal_baselines.py::prepare --dataset pbmc68k  # QC + nested subsamples
    uvx modal run bench/modal_baselines.py::baselines --dataset brain --sizes 100000 --engines seurat,scanpy
    uvx modal run bench/modal_baselines.py::pull --dataset brain --name brain_cpu16

Volume `scfast-data` layout:
  raw/            downloaded 10x files and the PBMC 68k labels
  prep/<dataset>/ prep.py output: prep.json, n<cells>/matrix.h5 + cells.tsv
  results/<dataset>/<engine>_n<cells>/  one runner's results.jsonl, outputs/, env/

`pull` merges every runner folder of a dataset into bench/results/<name>/ (results.jsonl,
outputs/, env/, and data/ without the matrices), ready for parity.py and report.py.
Every run uses CPUS physical cores and MEM_GB of memory, with threads = CPUS.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
CPUS, MEM_GB = 16, 64
RUN_TIMEOUT = 7200

DATASETS = {
    "brain": {
        "input": "raw/1M_neurons_filtered_gene_bc_matrices_h5.h5",
        "sizes": "100000,500000,all",
        "qc": ["--mt-prefix", "mt-", "--min-genes", "200", "--max-genes", "6000", "--max-mt", "20"],
        "mem_gb": 160,
    },
    "pbmc68k": {
        "input": "raw/pbmc68k/filtered_matrices_mex/hg19",
        "labels": "raw/pbmc68k_labels.tsv",
        "sizes": "all",
        "qc": ["--min-genes", "200", "--max-genes", "2500", "--max-mt", "5"],
        "mem_gb": 16,
    },
}
DOWNLOADS = {
    "raw/1M_neurons_filtered_gene_bc_matrices_h5.h5":
        "https://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/1M_neurons_filtered_gene_bc_matrices_h5.h5",
    "raw/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz":
        "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/"
        "fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz",
    "raw/68k_pbmc_barcodes_annotation.tsv":
        "https://raw.githubusercontent.com/10XGenomics/single-cell-3prime-paper/master/pbmc68k_analysis/"
        "68k_pbmc_barcodes_annotation.tsv",
}

app = modal.App("scfast-baselines")
vol = modal.Volume.from_name("scfast-data", create_if_missing=True)
BENCH = dict(ignore=["results", "hpc", "__pycache__"])

data_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl")
    .pip_install("numpy", "scipy", "pandas", "h5py", "anndata")
    .add_local_dir(ROOT / "bench", "/work/bench", **BENCH)
)

# r2u: CRAN as Ubuntu binaries, so Seurat installs in seconds. OpenBLAS so R gets a threaded BLAS.
bench_image = (
    modal.Image.from_registry("rocker/r2u:24.04", add_python="3.12")
    .apt_install("r-cran-seurat", "r-cran-hdf5r", "r-cran-leidenbase", "r-cran-uwot", "r-cran-jsonlite",
                 "libopenblas0-pthread")
    .run_commands("Rscript -e 'stopifnot(packageVersion(\"Seurat\") >= \"5.1.0\"); sessionInfo()'")
    .pip_install("scanpy", "anndata", "scikit-misc", "leidenalg", "igraph", "h5py", "scikit-learn", "matplotlib")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "bench", "/work/bench", **BENCH)
)


def sh(cmd: list[str], env: dict | None = None, check: bool = True) -> int:
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd="/work", env={**os.environ, **(env or {})}).returncode
    if check and rc:
        raise RuntimeError(f"{cmd[0]} exited with {rc}")
    return rc


@app.function(image=data_image, volumes={"/data": vol}, cpu=2, memory=8192, timeout=3600)
def download():
    import h5py
    import numpy as np
    import pandas as pd

    for rel, url in DOWNLOADS.items():
        dst = Path("/data") / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print("download", url, flush=True)
        sh(["curl", "-fsSL", "--retry", "3", "-o", f"{dst}.part", url])
        os.replace(f"{dst}.part", dst)
    if not Path("/data/raw/pbmc68k").exists():
        sh(["tar", "-xzf", "/data/raw/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz",
            "-C", "/data/raw", "--one-top-level=pbmc68k"])
    lab = pd.read_csv("/data/raw/68k_pbmc_barcodes_annotation.tsv", sep="\t")
    lab[["barcodes", "celltype"]].rename(columns={"barcodes": "barcode", "celltype": "cell_type"}).to_csv(
        "/data/raw/pbmc68k_labels.tsv", sep="\t", index=False)
    with h5py.File("/data/" + DATASETS["brain"]["input"]) as f:
        g = f[list(f.keys())[0]]
        genes_per_cell = np.diff(g["indptr"][()])
        print(f"brain: groups {list(f.keys())}, keys {list(g.keys())}, shape {g['shape'][()]}, "
              f"nnz {g['data'].shape[0]}, dtypes data {g['data'].dtype} indices {g['indices'].dtype}; "
              f"genes per cell p1/p50/p99/p99.9: {np.quantile(genes_per_cell, [.01, .5, .99, .999]).tolist()}")
    sh(["ls", "-laR", "/data/raw"])
    vol.commit()


@app.function(image=data_image, volumes={"/data": vol}, cpu=4, memory=16 * 1024, timeout=3 * 3600)
def prep(dataset: str):
    d = DATASETS[dataset]
    cmd = ["python", "bench/prep.py", "--input", f"/data/{d['input']}", "--out", f"/data/prep/{dataset}",
           "--sizes", d["sizes"], *d["qc"]]
    if "labels" in d:
        cmd += ["--labels", f"/data/{d['labels']}"]
    sh(cmd)
    vol.commit()


@app.function(image=bench_image, volumes={"/data": vol}, cpu=CPUS, memory=MEM_GB * 1024,
              timeout=RUN_TIMEOUT + 600)
def run(engine: str, dataset: str, size: str, reps: int):
    out = Path(f"/data/results/{dataset}/{engine}_{size}")
    out.mkdir(parents=True, exist_ok=True)
    threads = {v: str(CPUS) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                      "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS")}
    runner = ["Rscript", "bench/run_seurat.R"] if engine == "seurat" else \
        ["python", "bench/run_py.py", "--engine", engine]
    rc = sh(["timeout", str(RUN_TIMEOUT), *runner, "--data", f"/data/prep/{dataset}/{size}", "--out", str(out),
             "--reps", str(reps), "--threads", str(CPUS)], env=threads, check=False)
    if rc == 124:
        with (out / "results.jsonl").open("a") as f:
            f.write(f'{{"schema":"1.0","engine":"{engine}","dataset":"{size}","cells":{size[1:]},"rep":-1,'
                    f'"warmup":false,"step":"all","kind":"compute","status":"error",'
                    f'"error":"timeout after {RUN_TIMEOUT}s","wall_s":null}}\n')
    vol.commit()
    return engine, size, rc


@app.local_entrypoint()
def prepare(dataset: str):
    prep.with_options(memory=DATASETS[dataset]["mem_gb"] * 1024).remote(dataset)


@app.local_entrypoint()
def baselines(dataset: str, sizes: str, engines: str = "seurat,scanpy", reps: int = 3):
    """sizes: comma-separated prepared folder names without the 'n', e.g. 100000 or 68579."""
    jobs = [(e, dataset, f"n{s}", reps) for s in sizes.split(",") for e in engines.split(",")]
    for engine, size, rc in run.starmap(jobs, order_outputs=False):
        print(f"{engine} {size}: exit {rc}")


@app.local_entrypoint()
def pull(dataset: str, name: str):
    dst = ROOT / "bench" / "results" / name
    files = [e.path for e in vol.listdir(f"results/{dataset}", recursive=True) if e.type.name == "FILE"]
    files += [e.path for e in vol.listdir(f"prep/{dataset}", recursive=True)
              if e.type.name == "FILE" and not e.path.endswith(".h5")]
    jsonl = []
    for path in files:
        body = b"".join(vol.read_file(path))
        parts = Path(path).parts
        if parts[0] == "prep":
            rel = Path("data", *parts[2:])
        elif parts[-1] == "results.jsonl":
            jsonl.append(body)
            continue
        else:
            rel = Path(*parts[3:])  # results/<dataset>/<engine>_<size>/...
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        (dst / rel).write_bytes(body)
    (dst / "results.jsonl").write_bytes(b"".join(jsonl))
    print(f"pulled {len(files)} files into {dst}")
