# Installing and running scpipe

Three targets: the **Iridis X L4 login node**, any **Linux** machine, and **Windows**.
All use the same `config.yaml`; only the install and the `site` differ.

Which engine runs the CPU steps is set by `engine:` in `config.yaml`
([scfast-integration.md](scfast-integration.md)):

| `engine` | S1 doublets/pre-clusters, S2 PCA/neighbours/Leiden | exchange `.h5ad` files |
|---|---|---|
| `auto` (default) | Rust `scfast` if `scfastR` is installed and the plan is CPU, else Seurat | none on the scfast path |
| `scfast` | Rust `scfast`, error if not installed or if `device: gpu` | none |
| `seurat` | Seurat (CPU) or rapids-singlecell via exchange (GPU) | only on the GPU path |

On a GPU plan `engine_gpu: auto | scfast | rapids` picks between the Rust CUDA backend
(`gpu_scfast`, no exchange) and rapids-singlecell via the exchange; `auto` uses the Rust backend
only when `scfastR` was built with `--features cuda` and a device is visible. Rust is needed only
to build `scfast`; without it `engine: auto` falls back to Seurat.

**The CUDA and cuGraph code has never run on a GPU.** It was written and compiled in a workspace
without one; the kernels were checked by CPU emulation and the cuGraph C-API bindings are
hand-written against the 24.x headers. Section 1b lists the acceptance tests to run on the L4
first. Until they pass, leave `engine_gpu: auto` (which stays on rapids until `scfastR` reports a
CUDA build) or set `engine_gpu: rapids`.

## 1. Iridis X, L4 login node

Not yet run on the real node; the steps below are what the code expects. Check the points marked
**confirm** with the Iridis team (hpc@soton.ac.uk) first, especially whether multi-hour GPU work on
the shared login node is allowed. If it is not, set `execution: split` (see section 5 of the user guide).

```bash
ssh <user>@iridisx.soton.ac.uk            # confirm the login host name
nvidia-smi                                # the L4 should be listed; note free memory
module avail 2>&1 | grep -i -E "^R/|python|apptainer|rust|cuda"   # confirm module names
```

**One-time setup** (put everything on scratch, not home):

```bash
cd /scratch/$USER && git clone https://github.com/mukeshkbj/singlecell.git && cd singlecell
module load R python                      # confirm names/versions: R >= 4.3, Python >= 3.12
python3 -m venv /scratch/$USER/venv-scpipe && source /scratch/$USER/venv-scpipe/bin/activate
pip install -e "python[harmony,llm,report,test,scfast]"
pip install -e "python[gpu]"              # rapids-singlecell, CUDA 12; large download

# R packages (user library on scratch)
mkdir -p /scratch/$USER/Rlib && export R_LIBS_USER=/scratch/$USER/Rlib
Rscript -e 'install.packages(c("Seurat","SeuratObject","hdf5r","Matrix","jsonlite","digest","yaml","optparse","leidenbase","uwot","harmony","qs2","presto"))'
Rscript -e 'BiocManager::install(c("DESeq2","limma"))'

# Rust engine (optional but recommended): needs cargo >= 1.80
curl --proto '=https' -sSf https://sh.rustup.rs | sh -s -- -y && source $HOME/.cargo/env
R CMD INSTALL rust/scfast-r                                       # scfastR
maturin develop --release -m rust/scfast-py/Cargo.toml            # scfast for Python
```

If the node has Apptainer, the container route replaces all of the above:
`apptainer build /scratch/$USER/scpipe.sif containers/scpipe.def`, then set `container.image` in the
site file (`conf/sites/iridisx.yaml` already passes `--nv`).

**Configure and check:**

```bash
cp -r config/examples /scratch/$USER/my_study && cd /scratch/$USER/my_study
# edit config.yaml: site: iridisx, run_dir: /scratch/$USER/scpipe/my_study, engine: auto,
# device: auto, plus samples.csv and qc_policy.yaml
export ANTHROPIC_API_KEY=...              # optional; without it the gates use rules only
scpipe doctor --config config.yaml        # packages, scfastR/scfast, GPU, LLM key, budget, Open Targets
```

`doctor` prints the engine `auto` would choose. For datasets under about 100k cells it will pick
the Rust CPU engine; above that, with the L4 free, S2 uses rapids-singlecell through the exchange.
(The exact cut-over is set by the device plan; `S2/summary.json` records which path ran.)

**Run, detached so a dropped SSH session does not kill it:**

```bash
scpipe run --config config.yaml --detach
scpipe status --run-dir /scratch/$USER/scpipe/my_study
scpipe approve --run-dir /scratch/$USER/scpipe/my_study --stage G1 --by $USER   # when a gate waits (exit code 3)
scpipe resume  --run-dir /scratch/$USER/scpipe/my_study
```

Results: `S5/report.html` and `S5/explorer.html` in the run directory. Copy them to your laptop with
`scp`. Be considerate on the shared node: keep `resources.threads` at 8 or fewer, `nice: 10`, and
`gpu_mem_cap_gb: 20` as in the site file.

### 1b. Enabling the Rust GPU backend on the L4 (acceptance tests first)

```bash
# needs the CUDA 12 toolkit (nvrtc, cublas) and, for GPU Leiden, a RAPIDS env with libcugraph_c
module load cuda/12.4                                   # confirm the name; or activate the RAPIDS conda env
cd /scratch/$USER/singlecell/rust

# 1. Rust parity tests on the device (PCA subspace > 0.99, kNN overlap >= 90%, Leiden ARI >= 0.9)
cargo test -p scfast-core --release --features cuda --test cuda -- --ignored --nocapture
export SCFAST_CUGRAPH_LIB=$CONDA_PREFIX/lib/libcugraph_c.so   # if not on the loader path
export SCFAST_CUGRAPH_API=legacy                              # or 24.08 / 24.10 to match `conda list cugraph`
SCFAST_CUGRAPH_TEST=1 cargo test --features cugraph -- --ignored cugraph --nocapture

# 2. Rebuild the R package and the wheel with the features on
SCFASTR_CARGO_FEATURES=cuda,cugraph R CMD INSTALL rust/scfast-r
Rscript -e 'str(scfastR::scfast_info())'               # cuda_built TRUE, cuda_available TRUE, device_name "NVIDIA L4"
cd rust/scfast-py && maturin build --release --features cuda,cugraph && pip install target/wheels/scfast-*.whl

# 3. Pipeline-level check, then the scale runs (bench/hpc/README.md)
scpipe doctor --config config.yaml                       # "scfastR cuda built: ok", "CUDA device visible: ok"
sbatch --export=ALL,SIZES=50000,ENGINES=seurat-cpu,scfast-cpu,scfast-cuda,rapids bench/hpc/slurm_scale.sh
```

If a parity test fails, report the numbers; `engine_gpu: rapids` keeps the pipeline on the proven
path meanwhile. The most likely failure is the cuGraph graph-constructor layout: try the other
`SCFAST_CUGRAPH_API` values before anything else.

## 2. Linux (workstation, server, cloud VM)

Same as above without the `module` lines. On Ubuntu 22.04/24.04:

```bash
sudo apt-get install -y build-essential gfortran git tmux cmake pkg-config \
  libhdf5-dev libcurl4-openssl-dev libssl-dev libxml2-dev libglpk-dev libgsl-dev \
  python3-venv python3-dev r-base r-base-dev
git clone https://github.com/mukeshkbj/singlecell.git && cd singlecell
python3 -m venv .venv && source .venv/bin/activate
pip install -e "python[harmony,llm,report,test,scfast]"
# R packages, Rust, scfastR, scfast: as in section 1
scpipe init-site --out conf/sites/mybox.yaml     # detects scheduler, GPU, container runtime
```

Set `site: workstation` (or your generated file) and run exactly as in section 1.
With Docker and an NVIDIA GPU: `docker build -t scpipe -f containers/Dockerfile .` then
`docker run --rm --gpus all -v $PWD:/work scpipe scpipe doctor`. Without a GPU, drop `--gpus all`
and use `device: cpu`.

## 3. Windows

Run it **inside WSL2**, which gives you a real Ubuntu, so the Linux steps apply unchanged.
Native Windows is not supported: the runner relies on `nice`, `tmux`/`nohup` and POSIX process
handling, and rapids-singlecell is Linux-only. Native installs have not been tested.

```powershell
# PowerShell as Administrator, once; then reboot
wsl --install -d Ubuntu-24.04
```

Open the Ubuntu terminal and follow **section 2**. Notes:

- Keep the project and run directory **inside the Linux filesystem** (`~/singlecell`), not under
  `/mnt/c/...`; I/O there is much slower.
- GPU: install the current NVIDIA Windows driver (it exposes the GPU to WSL2); do not install a Linux
  driver in WSL. Check with `nvidia-smi`. Without a GPU use `device: cpu`.
- Memory: WSL2 takes half the RAM by default. Raise it in `%UserProfile%\.wslconfig`
  (`[wsl2]` / `memory=32GB`) and run `wsl --shutdown`.
- Alternative: Docker Desktop with the WSL2 backend, then the `docker run` line in section 2.
- View results by opening `S5\report.html` from Explorer at `\\wsl$\Ubuntu\home\<you>\...`.

## 4. Check that it worked

```bash
PYTHONPATH=python python3 -m pytest -q tests/python     # unit tests, offline
bash tests/integration/run_full.sh                      # whole pipeline S0-S5 on the synthetic fixture
```

On the scfast path `S2/summary.json` shows `"path": "cpu_scfast"` and no `.h5ad` files appear under
the run directory.
