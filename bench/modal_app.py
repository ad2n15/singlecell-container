"""Run the GPU benchmark on Modal: build scfast with CUDA, generate data, run engines, pull results.

    uvx modal run bench/modal_app.py::build                              # wheel + Rust test binaries (CPU box)
    uvx modal run bench/modal_app.py::rust_tests                         # cargo test --features cuda,cugraph on a GPU
    uvx modal run bench/modal_app.py::gen --tag synth_500k --sizes 100000,500000
    uvx modal run bench/modal_app.py::bench --tag synth_500k --name l4 --sizes 100000 \
        --engines scanpy,rapids,scfast-cpu,scfast-gpu
    python bench/parity.py --run bench/results/l4 --reference scanpy && python bench/report.py --run bench/results/l4

GPU type, CPU count and memory come from SCFAST_GPU (default L4), SCFAST_CPUS (8) and
SCFAST_MEM_GB (48), read when the file is loaded. Threads for every engine equal SCFAST_CPUS.
Data and results live in the Modal volume `scfast-bench`; the cargo cache in `scfast-cargo`.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
GPU = os.environ.get("SCFAST_GPU", "L4")
CPUS = int(os.environ.get("SCFAST_CPUS", "8"))
MEM_GB = int(os.environ.get("SCFAST_MEM_GB", "48"))
RAPIDS = "26.2.*"

app = modal.App("scfast-bench")
bench_vol = modal.Volume.from_name("scfast-bench", create_if_missing=True)
cargo_vol = modal.Volume.from_name("scfast-cargo", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "curl", "git", "pkg-config")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable")
    .env({"PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
          "CARGO_HOME": "/cache/cargo", "CARGO_TARGET_DIR": "/cache/target", "PYTHONUNBUFFERED": "1"})
    .pip_install(
        "scanpy", "anndata", "scikit-misc", "leidenalg", "python-igraph", "h5py", "scikit-learn", "matplotlib",
        "pandas", "scipy", "maturin>=1.9.4,<2", "pytest", "cupy-cuda12x[ctk]",
        f"cudf-cu12=={RAPIDS}", f"cuml-cu12=={RAPIDS}", f"cugraph-cu12=={RAPIDS}", f"cuvs-cu12=={RAPIDS}",
        f"libcugraph-cu12=={RAPIDS}",
        extra_index_url="https://pypi.nvidia.com",
    )
    .env({"CC": "gcc", "CXX": "g++", "SKBUILD_CMAKE_DEFINE_CMAKE_CUDA_ARCHITECTURES": "86;89;90"})
    .run_commands(
        "pip install rapids-singlecell --no-deps && pip install docrep natsort scikit-image statsmodels tqdm "
        "'scverse-misc[settings]'",
    )
    .add_local_dir(ROOT, "/work", ignore=["**/.git", "**/target", "**/__pycache__", "bench/results", "data",
                                          "plots", "plots_scanpy", "R", "tests", "docker", "containers"])
)


def sh(cmd: list[str], env: dict | None = None, cwd: str = "/work", check: bool = True) -> int:
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}).returncode
    if check and rc:
        raise RuntimeError(f"{cmd[0]} exited with {rc}")
    return rc


def gpu_libs() -> dict:
    """LD_LIBRARY_PATH covering the pip-installed CUDA / RAPIDS libraries, for dlopen from Rust."""
    site = sysconfig.get_paths()["purelib"]
    dirs = sorted(glob.glob(f"{site}/nvidia/*/lib") + glob.glob(f"{site}/lib*/lib64") +
                  glob.glob(f"{site}/rapids_logger/lib64") + glob.glob(f"{site}/lib*/lib"))
    libs = glob.glob(f"{site}/libcugraph/lib64/libcugraph_c.so")
    env = {"LD_LIBRARY_PATH": ":".join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")])}
    if libs:
        env["SCFAST_CUGRAPH_LIB"] = libs[0]
    return env


def install_wheel() -> None:
    bench_vol.reload()
    wheel = sorted(glob.glob("/vol/wheels/scfast-*.whl"))[-1]
    sh([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "-q", wheel])


@app.function(image=image, cpu=16, memory=32768, timeout=3000, volumes={"/cache": cargo_vol, "/vol": bench_vol})
def build_remote(tests: bool = True) -> str:
    feats = ["--features", "cuda,cugraph"]
    shutil.rmtree("/tmp/wheels", ignore_errors=True)
    sh(["maturin", "build", "--release", "-m", "rust/scfast-py/Cargo.toml", *feats, "-o", "/tmp/wheels"])
    Path("/vol/wheels").mkdir(parents=True, exist_ok=True)
    for old in glob.glob("/vol/wheels/*.whl"):
        os.remove(old)
    wheel = glob.glob("/tmp/wheels/*.whl")[0]
    shutil.copy(wheel, "/vol/wheels/")
    if tests:
        sh(["cargo", "test", "--no-run", "--release", "-p", "scfast-core", *feats], cwd="/work/rust")
    cargo_vol.commit()
    bench_vol.commit()
    return os.path.basename(wheel)


@app.function(image=image, cpu=4, memory=65536, timeout=3600, volumes={"/vol": bench_vol})
def gen_remote(tag: str, sizes: str, genes: int = 5000, types: int = 10, force: bool = False) -> str:
    root = Path("/vol/data") / tag
    if (root / "data" / "prep.json").exists() and not force:
        return f"{tag}: already prepared"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    wanted = [int(s) for s in sizes.split(",")]
    raw = root / "raw.h5"
    sh([sys.executable, "bench/synth.py", "--cells", str(int(max(wanted) * 1.03)), "--genes", str(genes),
        "--types", str(types), "--out", str(raw)])
    sh([sys.executable, "bench/prep.py", "--input", str(raw), "--out", str(root / "data"), "--sizes", sizes,
        "--labels", f"{raw}.labels.tsv", "--max-mt", "100", "--max-genes", "1e9"])
    raw.unlink()
    bench_vol.commit()
    return f"{tag}: sizes {sizes} written"


@app.function(image=image, gpu=GPU, cpu=CPUS, memory=MEM_GB * 1024, timeout=3000,
              volumes={"/cache": cargo_vol, "/vol": bench_vol})
def rust_tests_remote(args: str = "") -> str:
    cargo_vol.reload()
    sh(["nvidia-smi"], check=False)
    cmd = ["cargo", "test", "--release", "-p", "scfast-core", "--features", "cuda,cugraph", *args.split(),
           "--", "--include-ignored", "--nocapture", "--test-threads=1"]
    env = {**gpu_libs(), "SCFAST_CUGRAPH_TEST": "1"}
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd="/work/rust", env={**os.environ, **env}, capture_output=True, text=True)
    out = p.stdout + p.stderr
    print(out[-30000:])
    return f"exit {p.returncode}\n{out}"


@app.function(image=image, gpu=GPU, cpu=CPUS, memory=MEM_GB * 1024, timeout=7000, volumes={"/vol": bench_vol})
def bench_remote(tag: str, name: str, sizes: str, engines: str, reps: int = 1, extra: str = "") -> str:
    install_wheel()
    bench_vol.reload()
    run = Path("/tmp/run") / name
    shutil.rmtree(run, ignore_errors=True)
    res = Path("/vol/results") / name
    run.mkdir(parents=True)
    if res.exists():
        shutil.copytree(res, run, dirs_exist_ok=True)
    engine_list = engines.split(",")
    keep = run / "results.jsonl"
    if keep.exists():
        lines = [ln for ln in keep.read_text().splitlines() if json.loads(ln)["engine"] not in engine_list]
        keep.write_text("".join(ln + "\n" for ln in lines))
    for e in engine_list:
        shutil.rmtree(run / "outputs" / e, ignore_errors=True)
    data = Path("/vol/data") / tag / "data"
    stage = Path("/tmp/stage")
    stage.mkdir(exist_ok=True)
    (run / "data").mkdir(exist_ok=True)
    for f in (stage, run / "data"):
        shutil.copy(data / "prep.json", f)
    env = {**gpu_libs(), "NUMBA_CACHE_DIR": "/tmp/numba", "MPLCONFIGDIR": "/tmp/mpl"}
    for size in sizes.split(","):
        ds = f"n{size}"
        shutil.copytree(data / ds, stage / ds, dirs_exist_ok=True)
        (run / "data" / ds).mkdir(exist_ok=True)
        shutil.copy(data / ds / "cells.tsv", run / "data" / ds)
        for e in engine_list:
            sh([sys.executable, "bench/run_py.py", "--engine", e, "--data", str(stage / ds), "--out", str(run),
                "--reps", str(reps), "--threads", str(CPUS), *extra.split()], env=env, check=False)
    shutil.rmtree(res, ignore_errors=True)
    shutil.copytree(run, res, ignore=shutil.ignore_patterns("matrix.h5"))
    bench_vol.commit()
    return f"results/{name}"


def pull(name: str) -> Path:
    out = ROOT / "bench" / "results" / name
    shutil.rmtree(out, ignore_errors=True)
    for entry in bench_vol.iterdir(f"results/{name}", recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        dest = out / Path(entry.path).relative_to(f"results/{name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for chunk in bench_vol.read_file(entry.path):
                f.write(chunk)
    return out


@app.local_entrypoint()
def build(tests: bool = True):
    print(build_remote.remote(tests))


@app.local_entrypoint()
def rust_tests(args: str = ""):
    print(rust_tests_remote.remote(args)[:200])


@app.local_entrypoint()
def gen(tag: str, sizes: str, genes: int = 5000, types: int = 10, force: bool = False):
    print(gen_remote.remote(tag, sizes, genes, types, force))


@app.local_entrypoint()
def bench(tag: str, name: str, sizes: str, engines: str = "scanpy,rapids,scfast-cpu,scfast-gpu", reps: int = 1,
          extra: str = ""):
    print(bench_remote.remote(tag, name, sizes, engines, reps, extra))
    print("pulled to", pull(name))
