# singlecell

Side-by-side single-cell RNA-seq workflows in **Seurat (R)** and **Scanpy (Python)**, run on the 10x Genomics PBMC 3k dataset.

Both scripts apply the same QC thresholds and end with the same 9 annotated cell types, so they're a handy reference for moving between the two ecosystems.

## Contents

| File | What it does |
| --- | --- |
| `seurat_pbmc3k.R` | Seurat v5: QC, normalisation, PCA, UMAP, Louvain clustering, markers (presto), annotation, plots |
| `scanpy_pbmc3k.py` | Scanpy: the same workflow with Leiden clustering |
| `plots/` | Seurat figures |
| `plots_scanpy/` | Scanpy figures |

## Results

2,638 cells after QC, 9 clusters:

Naive CD4 T, Memory CD4 T, CD14 Mono, B, CD8 T, FCGR3A Mono, NK, DC, Platelet

| Seurat | Scanpy |
| --- | --- |
| ![Seurat UMAP](plots/04_dimplot_celltypes.png) | ![Scanpy UMAP](plots_scanpy/umap_03_celltypes.png) |

Cluster sizes differ slightly between the two (Louvain vs Leiden and different neighbour graphs), as expected.

## Setup (macOS, Apple Silicon)

**R**

```bash
brew install --cask r
```

```r
install.packages(c("Seurat", "harmony", "remotes", "BiocManager", "patchwork", "dplyr"), type = "binary")
BiocManager::install("glmGamPoi")
remotes::install_github("immunogenomics/presto")
```

**Python**

```bash
uv venv .venv-scanpy && source .venv-scanpy/bin/activate
uv pip install "scanpy[leiden]" harmonypy ipykernel
```

## Run

```bash
Rscript seurat_pbmc3k.R      # downloads PBMC 3k on first run
python scanpy_pbmc3k.py      # reuses the same download
```

Data: [10x Genomics PBMC 3k](https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz)
