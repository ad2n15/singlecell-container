# Getting started with scpipe

scpipe takes single-cell RNA-seq count files and runs the whole analysis:

- QC, doublet removal, normalisation, batch correction and clustering;
- cell-type annotation;
- differential expression, with drug and disease evidence from Open Targets;
- a self-contained HTML report.

At four points ("review gates") LLM agents check the decisions; you approve them, or let the
agents decide.

This page covers the everyday path: install, check, run, find the results. The details are in the
[User guide](USER_GUIDE.md) and the [Pipeline reference](PIPELINE.md).

**Contents**
1. [What you need](#1-what-you-need)
2. [Install](#2-install)
3. [Quick test (5 minutes)](#3-quick-test-5-minutes)
4. [Run your own data](#4-run-your-own-data)
5. [Guided setup (`-i`)](#5-guided-setup--i)
6. [Review gates and LLM agents](#6-review-gates-and-llm-agents)
7. [Differential expression](#7-differential-expression)
8. [Big datasets: compute nodes and GPUs](#8-big-datasets-compute-nodes-and-gpus)
9. [Results](#9-results)
10. [Everyday commands](#10-everyday-commands)
11. [Troubleshooting](#11-troubleshooting)
12. [FAQ](#12-faq)

---

## 1. What you need

| | Requirement |
|---|---|
| **Computer** | Linux x86_64: an HPC login or compute node, a workstation, or a cloud VM. macOS / Windows through Docker Desktop (section 2C) or WSL2 ([INSTALL.md](INSTALL.md)). |
| **Container runtime** | Apptainer ≥ 1.3 (HPC) or Docker (workstation, cloud, macOS / Windows via Docker Desktop). Everything else (R 4.6, Seurat 5, Bioconductor, Python 3.12, the Rust engine, CellTypist, RAPIDS) is inside the scpipe image. |
| **Disk** | The image is 6.6 GB as a `.sif` file, or 12 GB / ~133k files unpacked as a sandbox. Each run needs a few hundred MB to a few GB, depending on the number of cells. |
| **Memory** | PBMC 3k (2,700 cells) peaks at about 1.4 GB. Memory grows with the number of cells; for 100k+ cells start with 64 GB, and check the peak memory shown per stage. |
| **CPU / GPU** | 8 cores is a good default. A GPU (e.g. NVIDIA L4) is optional; the CPU path is complete. |
| **Internet** | Needed for the LLM review agents and the Open Targets evidence. Not needed for the stages in between (most of the computing). On Iridis, compute nodes have **no** internet: see [section 8](#8-big-datasets-compute-nodes-and-gpus). |
| **LLM key (optional)** | One of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `NVIDIA_API_KEY`. Without a key the gates use rules only and stop for you each time. |
| **Data** | 10x `.h5` or `matrix.mtx` folders, `.h5ad`, Seurat `.rds` / `.qs2`, `.loom`, or csv/tsv count tables. **Human** data. Raw 10x matrices are reduced to cells automatically. |

---

## 2. Install

Pick one option.

### Option A: Iridis X, shared copy (ready now)

The image is already unpacked for the DDnB group at `/iridisfs/ddnb/images/scpipe_sandbox`.
There is nothing to install:

```bash
module load apptainer/1.4.0
apptainer exec --cleanenv /iridisfs/ddnb/images/scpipe_sandbox scpipe doctor
```

To save typing, define a shell function, for example in `~/.bashrc`:

```bash
scpipe() { apptainer exec --cleanenv --env-file "${SCPIPE_ENV:-/dev/null}" -B /iridisfs,/scratch /iridisfs/ddnb/images/scpipe_sandbox scpipe "$@"; }
```

After `module load apptainer/1.4.0`, `scpipe run ...` then works like a local command.
`SCPIPE_ENV=.env` passes your API keys (section 6).

### Option B: build the image yourself (any site with Apptainer)

This takes about 40–60 minutes and needs internet, so on an HPC run it on the login node.
Keep Apptainer's temporary folder on a local disk:

```bash
git clone https://github.com/mukeshkbj/singlecell.git && cd singlecell
APPTAINER_TMPDIR=/tmp/$USER-apptainer apptainer build --fakeroot scpipe.sif containers/scpipe-build.def
```

- **Sites without `fusermount` (Iridis):** a `.sif` only runs with `--unsquash`, which unpacks it
  into RAM on every start. Unpack it once into a sandbox folder instead:
  ```bash
  apptainer build --sandbox /tmp/$USER-sbx scpipe.sif
  rsync -a /tmp/$USER-sbx/ /path/to/shared/scpipe_sandbox/ && rm -rf /tmp/$USER-sbx
  ```
  The `rsync -a` step matters on GPFS: unpacking straight onto it loses the execute permissions.
- **After a code change:** update the sandbox in place in about 20 s, instead of rebuilding.
  Without `--code-only` it also rebuilds the Rust engine, in about 5 minutes:
  ```bash
  containers/update_sandbox.sh /path/to/scpipe_sandbox --code-only
  ```

### Option C: Docker (workstation, cloud VM, macOS / Windows)

You need Docker. For a GPU you also need the NVIDIA Container Toolkit (Linux).

> These steps follow `containers/Dockerfile`. The Docker build hasn't yet been run end to end in
> CI; the same software is checked through the Apptainer build. Report problems, please.

**Build the image** once, in about 40–60 minutes. It needs internet and about 15 GB of free disk:

```bash
git clone https://github.com/mukeshkbj/singlecell.git && cd singlecell
docker build -t scpipe -f containers/Dockerfile .
```

After the first release you can pull it instead:
`docker pull ghcr.io/mukeshkbj/scpipe:latest && docker tag ghcr.io/mukeshkbj/scpipe:latest scpipe`.

**Make a short command.** The image starts `scpipe` in `/work`, so mount your project folder
there:

```bash
alias scpipe-docker='docker run --rm -it --user "$(id -u):$(id -g)" -e HOME=/home/scpipe -v "$HOME/.scpipe:/home/scpipe/.scpipe" -v "$PWD:/work" scpipe'
mkdir -p ~/.scpipe
scpipe-docker doctor
```

| Part | Why |
|---|---|
| `-it` | Keeps the terminal, for the progress display, `-i` and the DE questions. |
| `--user` | Results belong to you, not the image's user. |
| `~/.scpipe` mount | The LLM budget ledger and the time-left history persist between runs. |
| `-v "$PWD:/work"` | Data paths must be inside the current folder, given as relative paths. For data elsewhere, add `-v /data:/data:ro` and use `/data/...` paths. |

**Run.** From the folder that holds the data:

```bash
cd ~/scpipe-test
scpipe-docker run filtered_gene_bc_matrices/hg19 --name pbmc3k --out pbmc3k_full
```

Add options after `scpipe-docker` and before `run` by editing the alias, or run `docker run ...`
in full:

| Option | Add |
|---|---|
| API keys | `--env-file .env` |
| GPU | `--gpus all` |
| More cores / memory | `--cpus 16 --memory 64g` |
| Keep running after you close the terminal | Instead of `--detach`, use `-d` in place of `-it`, then `docker logs -f <id>` |

**macOS / Windows:** use Docker Desktop. The image is linux/amd64: on Apple silicon add
`--platform linux/amd64` (emulated, so slower, and no GPU). On Windows, run the commands inside
WSL2 ([INSTALL.md](INSTALL.md)).

### Option D: published image for Apptainer (from the first release)

Once a `v*` release is published:

```bash
apptainer pull scpipe.sif docker://ghcr.io/mukeshkbj/scpipe:latest
```

Until then, use A, B or C.

### Check the install

```bash
scpipe doctor
```

With Docker this is `scpipe-docker doctor`. It lists the Python and R packages, the GPU, the LLM provider (with a test call), the budget and
Open Targets. Every **required** line must say `ok`.

---

## 3. Quick test (5 minutes)

This runs the classic 10x PBMC 3k dataset. Use the copy in the repository, or download it:

```bash
mkdir -p ~/scpipe-test && cd ~/scpipe-test
curl -sL https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz | tar -xz
```

**Without an LLM.** This stops after clustering and markers, and needs no key or internet:

```bash
scpipe run filtered_gene_bc_matrices/hg19 --name pbmc3k --llm none --no-gates --until evidence --out pbmc3k_test
```

**Full pipeline with the agents.** Run this on a node with internet, after setting a key
(section 6):

```bash
scpipe run filtered_gene_bc_matrices/hg19 --name pbmc3k --out pbmc3k_full
```

In a terminal you see one line per finished stage. The current stage shows with its latest step,
a progress bar, the elapsed time and an estimate of the time left:

```
✓ Ingest (S0) 9.6s · peak 604.01 MB
✓ QC metrics (S1a) 23.1s · peak 1.01 GB
⠼ Normalise, PCA, batch correction, clustering  S2 · 12.3s
  ↳ harmony
  ━━━━━━━━━━━━━━━━━━━╸━━━━━━━━━━━━━━━━━━━━  5/12 stages · 50s elapsed · about 1 min left
```

Then open `pbmc3k_full/S5/report.html` in a browser. It should show about 2,500 cells after QC,
around 9 clusters, and T, B, NK, monocyte, dendritic and platelet populations, each with marker
genes and Open Targets evidence.

To check the image itself, run the built-in smoke test. It takes about 10 minutes, with a mock
LLM and offline:

```bash
srun --partition=amd --cpus-per-task=8 --mem=16G --time=01:00:00 apptainer exec --cleanenv /iridisfs/ddnb/images/scpipe_sandbox bash /opt/scpipe/containers/smoke_test.sh
```

---

## 4. Run your own data

**Give the files directly.** The format is detected, and each file or folder is one sample, named
after it:

```bash
scpipe run data/donorA.h5 data/donorB.h5 data/donorC.h5ad --meta condition=control --out results/my_study
```

**With sample information, use a sample sheet.** Any extra column (donor, condition, sex, age,
batch…) becomes cell metadata, which you can use for batch correction and comparisons:

```csv
sample_id,path,condition,donor,sex
ctrl1,data/ctrl1.h5,control,D1,F
ctrl2,data/ctrl2.h5,control,D2,M
stim1,data/stim1.h5,stimulated,D1,F
stim2,data/stim2.h5,stimulated,D2,M
```

```bash
scpipe run samples.csv --integrate donor --out results/my_study      # Harmony batch correction on donor
```

**Full control: a config file.** Copy [`config/examples/`](../config/examples) next to your data,
edit `config.yaml` (stages, gates, DE), `samples.csv` and `qc_policy.yaml` (QC limits), then:

```bash
scpipe doctor --config config.yaml
scpipe run --config config.yaml
```

Every stage is checkpointed. Running the same command again skips finished, unchanged stages.
Changing a setting reruns only the stages that use it, and the ones after them.

---

## 5. Guided setup (`-i`)

Add `-i` and scpipe asks a few questions first. Enter keeps the suggested value:

```bash
scpipe run data/*.h5 -i --out results/my_study
```

1. **Mode:** *Automatic* uses the defaults, with adaptive QC and the agents deciding. *Guided*
   asks the questions below.
2. **QC hard limits:** minimum genes (200) and UMIs (400) per cell, maximum mitochondrial % (25).
   `off` disables a limit.
3. **Adaptive QC:** the number of MADs (3), or `off`.
4. **Doublets:** Scrublet, scDblFinder or none.
5. **Clustering:** PCs (`auto`), Leiden resolution (0.5), variable genes (2000).
6. **DE:** the method, the test, and either markers per cell type, two groups, or **choose after
   annotation** (section 7).
7. **Batch correction** with Harmony (only with more than one sample).
8. **LLM**, from the keys it finds.
9. **Review gates:** agents decide, or stop at every gate.

It shows a summary, then asks before starting. The answers are saved, and it prints a command
that repeats the run without questions. `-i` needs a terminal, and is used **without** `--config`.

---

## 6. Review gates and LLM agents

| Gate | Name on the command line | What is reviewed | Default |
|---|---|---|---|
| G1 | `qc-review` | QC decisions per sample | agents decide |
| G2 | `context-review` | tissue and disease context | stops for you |
| G3 | `annotation-review` | cell-type labels of the clusters | agents decide |
| G4 | `interpretation-review` | interpretation of the DE genes | stops for you |

**API keys.** Put them in a `.env` file next to your data. Keep it private: it is never written
into the run.

```bash
NVIDIA_API_KEY=nvapi-...
# or ANTHROPIC_API_KEY=... / OPENAI_API_KEY=...
```

Pass the file with `apptainer exec --env-file .env ...` (or `SCPIPE_ENV=.env` with the function
in section 2). With several keys, scpipe uses Claude first, then OpenAI, then NVIDIA. Choose one
with `--llm nvidia`, and a model with `--llm-model <name>`.

**Letting the agents decide at every gate.** Use `-i` and choose "Agents decide", or use a config
with:

```yaml
gates: {qc-review: auto, context-review: auto, annotation-review: auto, interpretation-review: auto}
```

Even in `auto`, a gate stops when the reviewer agent disagrees, or when the LLM call fails.

**When a gate stops.** Open its report, then approve and continue:

```bash
scpipe approve --run-dir results/my_study --stage context-review --by "Your Name"
scpipe resume  --run-dir results/my_study
```

The report is at `results/my_study/gates/G2/report.html`. To change the proposal, edit a copy of
`gates/G2/proposal.json` and pass `--from-file edited.json`.

**Cost.** Every call is priced and logged. The default budget is £5 per run and £30 in total
(`llm.budget`). A call that would cross it is not made, and that gate falls back to rules.

---

## 7. Differential expression

The default is Seurat **FindMarkers** with the **MAST** test. The two common setups:

| You have | You get, with no settings |
|---|---|
| one sample, or no comparison set | the **markers of each cell type** against the other cells |
| comparisons in `dge.contrasts` | each comparison, within each cell type |

**Choose the groups after annotation (recommended).** Cell types and clusters only have names
after clustering and the annotation review. With `-i`, choose *"choose after clustering and
annotation"*, or set `dge: {interactive: true}`. The run then pauses before the DE stage and asks:

```
Clustering and the cell-type annotation review are done. Choose the differential expression comparison(s).
  cell types: T cell (1361 cells), monocyte (581 cells), B cell (282 cells), NK cell (111 cells), ...
  clusters: 0 (663 cells), 1 (440 cells), 2 (417 cells), ...
  [1] markers of each cell type against the other cells
  [2] two cell types or clusters
  [3] two groups of samples (condition, donor, sex)
```

You pick groups by number or name. Group 2 can be `rest` (all other cells), and you can add more
comparisons. The choice is saved, so `resume` won't ask again.

**Comparisons in the config.** Any metadata column works for either side:

```yaml
dge:
  test: MAST                    # or wilcox (fast), LR, t, bimod, negbinom, poisson, DESeq2
  contrasts:
    - {name: stim_vs_ctrl, column: condition, numerator: stimulated, denominator: control}     # per cell type
    - {name: F_vs_M,       column: sex,       numerator: F,  denominator: M, covariates: [donor]}
    - {name: old_vs_rest,  column: age_group, numerator: [60-70, 70-80], denominator: rest, within: none}
    - {name: CD8_vs_CD4,   column: cell_type_fine, numerator: CD8 T cell, denominator: CD4 T cell, within: none}
```

- `within: none` compares all cells together. The default compares cell type by cell type.
- Covariates are used by MAST, LR, negbinom and poisson.

**Pseudobulk.** Use `method: pseudobulk` for DESeq2 on samples. You need two or more samples per
side, and the column must have one value per sample.

**Speed.** MAST takes about 1–2 minutes per cell type on PBMC 3k, and cell types run in
parallel. For hundreds of thousands of cells, use `test: wilcox` or set `max_cells_per_side: 2000`.

Each DE result lists the top 50 up and 50 down genes. Each gene comes with its Open Targets
record: associated diseases, drugs in the clinic (phase, indication, mechanism) and tractability.

---

## 8. Big datasets: compute nodes and GPUs

On Iridis the compute nodes have no internet, but the agents and Open Targets need it. Split the
run into three steps that share the run folder; each step continues where the last one stopped:

```bash
# 1. login node: setup questions, ingest, QC and the QC review (light; needs the terminal and internet)
scpipe run data/*.h5 -i --until qc-review --out /scratch/$USER/my_study

# 2. compute node: normalisation, batch correction, clustering, UMAP, markers, CellTypist (heavy, offline)
sbatch --export=ALL,RUN=/scratch/$USER/my_study conf/jobs/slurm_compute_offline.sbatch

# 3. login node, after the job: context and annotation reviews, DE + Open Targets, interpretation, report
scpipe resume --run-dir /scratch/$USER/my_study
```

- **Resources:** edit the `#SBATCH` lines of
  [`conf/jobs/slurm_compute_offline.sbatch`](../conf/jobs/slurm_compute_offline.sbatch). The
  defaults are 8 cores, 64 GB and 4 hours on `amd`; the job's threads follow `--cpus-per-task`.
- **Following the job:** `squeue -u $USER`, `scpipe status --run-dir ...`, or
  `<run>/dashboard.html`.
- **GPU:** in the job script, use `--partition=l4 --gres=gpu:1` and add `--nv` after
  `apptainer exec`. In a direct run, add `--nv` too.
- **Long runs on the login node:** `scpipe run ... --detach` runs inside tmux and survives a
  dropped SSH connection.

---

## 9. Results

Everything is in the run folder (`--out`):

| Where | What |
|---|---|
| `S5/report.html` | **Start here**: one self-contained page with QC, clusters, markers, annotation, DE, Open Targets, gate decisions, LLM cost and software versions |
| `S5/explorer.html` | Interactive UMAP: colour by cell type, cluster, sample or gene |
| `dashboard.html` | Run progress: stages, timings, memory, gates (updated after every stage) |
| `S4/dge/<comparison>__<group>.tsv.gz` | Full DE tables: `gene, baseMean, log2fc, pvalue, padj, direction, pct_1, pct_2` |
| `S4/opentargets_summary.json` | Open Targets records of the top DE genes |
| `S3/markers.tsv.gz`, `S3/annotation_consensus.tsv` | Cluster markers and cell-type calls |
| `S2/object.rds` | The final Seurat object (`readRDS()` in R) |
| `gates/G1..G4/` | Each gate's evidence, proposal, review, decision and `report.html` |
| `logs/<stage>.log` | Full log of each stage |
| `config/` | The exact settings used (`config.resolved.yaml`), for reproducibility |

---

## 10. Everyday commands

| Task | Command |
|---|---|
| Check the install and keys | `scpipe doctor` |
| Run | `scpipe run <files or samples.csv> --out DIR` |
| Run with questions | `scpipe run <files> -i --out DIR` |
| Stop after a stage | `--until clustering` (stage names below, or IDs like `S2`) |
| Continue / rerun after a change | `scpipe resume --run-dir DIR` |
| Progress and timings | `scpipe status --run-dir DIR` |
| Approve a gate | `scpipe approve --run-dir DIR --stage context-review --by NAME` |
| Run in the background (tmux) | add `--detach` |
| One line per stage (for logs) | add `--progress plain` |

**Stage names:** `ingest`, `qc-metrics`, `doublets`, `qc-filter`, `qc-review`, `clustering`,
`evidence`, `context-review`, `annotation-review`, `dge`, `interpretation-review`, `report`.

---

## 11. Troubleshooting

| You see | Cause and fix |
|---|---|
| `fusermount: No such file or directory` when starting a `.sif` | The site has no FUSE. Use the sandbox folder (section 2), or add `--unsquash`. `--unsquash` unpacks into RAM, so jobs need ~96 GB. |
| `bash: executable file not found` in a sandbox | It was unpacked straight onto GPFS and lost its execute permissions. Unpack on `/tmp` and copy with `rsync -a` (section 2B). |
| A gate stops with `fallback: true` | The LLM call failed. See `logs/G<n>.log`: e.g. `HTTP 410` means the model was retired (`--llm-model <current model>`), `401` means a wrong key, and no internet happens on compute nodes. |
| A gate stops with `reviewer: escalate` | The reviewer agent disagreed, or found a problem. Read `gates/G<n>/report.html`, then approve or edit. |
| `the worker made no valid proposal: ... must be null or match EFO_/MONDO_` | The agent gave an invalid ID; the policy check refused it. Approve the gate by hand, or rerun it. |
| `-i needs a terminal` | `-i` was used in a batch job or a pipe. Run it on the login node, then submit the heavy part (section 8). |
| The run waits before DE (`waits for the comparisons`) | You chose "after annotation" and the run has no terminal. `scpipe resume` in a terminal asks. |
| A stage `FAILED` | The last 20 lines of the log are printed. The full log is `logs/<stage>.log`. Fix the cause, then `scpipe resume`. |
| `disk quota exceeded` (home or scratch) | On Iridis the limit is the **number of files**. Keep runs, images and caches on a project folder or `/iridisfs`, not in home. |
| DE is slow | `dge.test: wilcox`, or `dge.max_cells_per_side: 2000` (section 7). |

---

## 12. FAQ

**What do the LLM agents see?** Aggregated evidence only: QC summaries, cluster statistics,
marker lists and DE summaries, never the expression matrix. Free-text sample metadata is passed
as data, never as instructions. Every call is logged in `llm/agent_log.jsonl`.

**Can I run without an LLM?** Yes, with `--llm none`. The rules make every decision, and each
gate stops for your approval. Use `--no-gates` to skip the gates entirely.

**Is it reproducible?** The exact settings, package versions, random seed and every decision are
recorded in the run folder. Rerunning with the same config skips everything unchanged.

**Mouse data?** Not supported: the annotation references and Open Targets are human.

**Where is the full reference?** [User guide](USER_GUIDE.md) (configuration, sites, schedulers,
containers), [Pipeline reference](PIPELINE.md) (every stage and setting), [Install](INSTALL.md)
(native installs, building, GPUs), and [Architecture](ARCHITECTURE.md).
