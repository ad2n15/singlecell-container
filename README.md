# singlecell — `scpipe`, an agentic Seurat pipeline

A production single-cell RNA-seq pipeline with **Seurat (R)** as the backbone, **GPU steps
through a validated Seurat ⇄ AnnData exchange format**, and **LLM agents that advise at four
review gates** while rules, a separate reviewer agent and humans keep the final say. It runs on
one internet-connected node (first target: the University of Southampton Iridis X L4 login node)
and moves to any HPC, workstation or cloud VM with one site file.

> Rules decide, agents advise, reviewers check, humans approve, everything is logged.

```
S0 ingest → S1 QC metrics · doublets · rules → G1 QC gate → S2 core (CPU or GPU)
→ S3 evidence + consensus annotation → G2 context gate → G3 annotation gate
→ S4 pseudobulk DGE + Open Targets → G4 interpretation gate → S5 report + explorer
```

## What it does

| | |
|---|---|
| **Any dataset** | 10x `.h5` / mtx, `.h5ad`, loom, csv/tsv, Seurat `.rds`/`.qs2`; format auto-detected; samples merged with a gene union; human-organism check |
| **QC by rules, advised by an agent** | Hard limits, per-sample adaptive bounds, doublets, protected populations (platelets, erythrocytes…) and review triggers from `qc_policy.yaml`. The QC agent can only tighten thresholds within the policy; every cell carries its reason codes |
| **Rust engine** | `scfast` (Rust) does doublets, PCA, neighbours, SNN and Leiden in process on CPU, so no Seurat ⇄ AnnData exchange is needed and Scanpy is not used; `engine: auto|scfast|seurat` |
| **CPU ⇄ GPU** | Seurat on CPU; rapids-singlecell on an NVIDIA GPU (24 GB L4 is enough for ~1M cells) through an exchange format whose counts checksum is identical in R and Python, so results are only ever merged into the object they came from |
| **Annotation** | Consensus of lineage signatures, CellTypist and markers against 72 human cell types with Cell Ontology IDs; the annotation agent touches only low-confidence or conflicting clusters |
| **Tissue and disease context** | Inferred from the evidence by an agent (organism fixed to human), with UBERON/EFO IDs resolved through Open Targets; speculative, so a human reviews it by default |
| **Pseudobulk DGE + Open Targets** | Conditions, replicates and contrasts from config; DESeq2 (limma-voom fallback); up/down genes annotated with Open Targets drugs, diseases and tractability; an interpretation agent writes labelled hypotheses citing evidence IDs |
| **Review gates** | Each gate writes `bundle / proposal / review / decision` JSON and a report; modes `auto`, `review`, `escalate`; `scpipe approve` for humans; full audit log and replay |
| **Any LLM, £30 cap** | Claude Haiku 4.5 by default; OpenAI-compatible servers (vLLM, Ollama) supported; a ledger refuses any call that would cross the per-run or total budget and falls back to rules |
| **Any HPC** | `direct`, SLURM/PBS/LSF/SGE or `split` execution; `scpipe init-site` and `doctor`; Apptainer and Docker recipes; resource caps and `nice` for shared login nodes; checkpoints and `resume` |

## Quick start

**R** 4.3+ with Seurat 5, and **Python** 3.12+:

```r
install.packages(c("Seurat", "SeuratObject", "hdf5r", "Matrix", "jsonlite", "digest", "yaml",
                   "leidenbase", "uwot", "DESeq2", "limma"))
install.packages(c("harmony", "qs2", "presto"))   # optional
```

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "python[harmony,llm,report,test]"
pip install -e "python[gpu]"                      # NVIDIA GPU, CUDA 12
# optional Rust engine (needs cargo >= 1.80): R CMD INSTALL rust/scfast-r ; see docs/INSTALL.md
```

Copy [`config/examples/`](config/examples) next to your data, edit `config.yaml`,
`samples.csv` and `qc_policy.yaml`, then:

```bash
export ANTHROPIC_API_KEY=...                       # or any provider; omit for rules-only gates
scpipe doctor --config config.yaml                # packages, GPU, LLM key + tool-use test, budget, Open Targets
scpipe run    --config config.yaml --detach       # runs inside tmux; survives a dropped SSH session
scpipe status --run-dir /scratch/$USER/scpipe/my_study
scpipe approve --run-dir ... --stage G2 --by me   # when a gate stops for you
scpipe resume  --run-dir ...                      # continue after a gate, disconnect or kill
```

Open `S5/report.html` (one self-contained page) and `S5/explorer.html` (interactive UMAP) in the
run directory. The full walkthrough is in [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

## Architecture

The as-built architecture, with diagrams of the stages, the CPU/GPU exchange and the gate
protocol, is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The documents behind it:

| Document | What it is |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | How the pieces fit: stages, runner, exchange, gates, LLM, portability, verification |
| [`docs/DESIGN.md`](docs/DESIGN.md) | The design spec (goals, login-node safeguards, memory plan, risks, open questions) |
| [`docs/exchange-spec.md`](docs/exchange-spec.md) | The Seurat ⇄ AnnData contract and checksum |
| [`docs/stages-spec.md`](docs/stages-spec.md) | S0–S2, the runner, the QC rules engine, config schema |
| [`docs/agents-spec.md`](docs/agents-spec.md) | Gates, agents, LLM adapter and budget, S3–S5, portability |
| [`docs/INSTALL.md`](docs/INSTALL.md) | Install and run on the Iridis X L4 node, Linux and Windows (WSL2) |
| [`docs/scfast-integration.md`](docs/scfast-integration.md) | How the Rust engine is wired into S1/S2 and `gpu_core` |
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | Install, configure, run, review gates, sites and schedulers |
| [`tests/README.md`](tests/README.md) | Fixtures, unit and integration tests, benchmarks |
| [`bench/README.md`](bench/README.md) | Benchmark harness: times Seurat, Scanpy, rapids and scfast step by step and checks parity with Seurat |
| [`docs/RUST_ENGINE.md`](docs/RUST_ENGINE.md) | `scfast`, an experimental Rust engine (PCA, kNN, SNN, Leiden) with Python and R bindings |

## Repository layout

```
R/exchange/       exchange format (R)          python/scpipe/   runner, config, qc, exchange, gpu_core,
R/stages/         S0 S1a S2 S3 S4 S5-export                     stages/, llm/, agents/, gates/, annotation/,
config/examples/  config, samples, qc_policy                    opentargets/, report/, site.py
conf/sites/       iridisx, workstation, docker  tests/           R + Python unit tests, fixtures, integration, benchmark
conf/templates/   slurm pbs lsf sge             containers/      Apptainer .def and Dockerfile
docs/             architecture, design, specs, user guide
```

## Status

All seven build steps are implemented and tested offline: ≈980 R expectations, ≈300 Python
tests, the exchange round trip, the S0–S2 CPU-vs-GPU comparison and the full S0 → S5 run with a
mock LLM all pass (see [`tests/README.md`](tests/README.md)). There is no hosted CI; tests run
locally or on the HPC. The Rust engine path (`engine: scfast`) passes the full S0 → S5 run and
≈1,080 R expectations; the GPU path is unchanged.

Not yet exercised outside the HPC: a real GPU, the live Claude API and the live Open Targets API
(unreachable from the development workspace). Before the first real run on Iridis: confirm with
the HPC team that multi-hour GPU work on the L4 login node is allowed (otherwise use
`execution: split`), put `run_dir` on scratch, and run `scpipe doctor`. Development benchmarks
are limited to 2k and 5k cells (`tests/benchmark/`).

## Tutorial scripts

The repository began as side-by-side **Seurat** and **Scanpy** workflows on the 10x PBMC 3k
dataset; both apply the same QC and end with the same 9 cell types.

| File | What it does |
|---|---|
| `seurat_pbmc3k.R` | Seurat v5: QC, normalisation, PCA, UMAP, Louvain, markers (presto), annotation, plots → `plots/` |
| `scanpy_pbmc3k.py` | Scanpy: the same workflow with Leiden → `plots_scanpy/` |

| Seurat | Scanpy |
| --- | --- |
| ![Seurat UMAP](plots/04_dimplot_celltypes.png) | ![Scanpy UMAP](plots_scanpy/umap_03_celltypes.png) |

```bash
Rscript seurat_pbmc3k.R      # downloads PBMC 3k on first run
python scanpy_pbmc3k.py      # reuses the same download
```

Setup for the scripts on macOS: `brew install --cask r`, then in R
`install.packages(c("Seurat","harmony","remotes","BiocManager","patchwork","dplyr"), type="binary")`,
`BiocManager::install("glmGamPoi")`, `remotes::install_github("immunogenomics/presto")`; for Python
`uv venv .venv-scanpy && source .venv-scanpy/bin/activate && uv pip install "scanpy[leiden]" harmonypy`.
