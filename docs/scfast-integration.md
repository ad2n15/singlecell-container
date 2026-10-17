# scfast integration spec

Goal: use the Rust engine (`scfast`, see [RUST_ENGINE.md](RUST_ENGINE.md)) in the pipeline wherever it
replaces Scanpy, so no Seurat ⇄ AnnData exchange is needed on that path.

## Config

Top-level `engine: auto | scfast | seurat` (default `auto`), validated in `python/scpipe/config.py`.

| value | meaning |
|---|---|
| `scfast` | in-process Rust engine; error with install hint if `scfastR` is missing |
| `seurat` | previous behaviour (Seurat on CPU, exchange + rapids/scanpy on GPU) |
| `auto` | `scfast` if `scfastR` is installed and the device plan says CPU, else `seurat` |

`device` keeps its meaning. `auto` never selects scfast when the plan is GPU (scfast has no CUDA backend yet):
GPU plans keep rapids via exchange. `engine: scfast` with `device: gpu` is a config error.
`engine` is added to the fingerprint keys of S1a, S1b and S2.

## S2 (R, `R/stages/s2_core.R`)

New path name `cpu_scfast`, in process, no exchange files:
normalise -> HVG -> scale (Seurat) -> `RunPCAFast` -> (Harmony via R `harmony` if `integration: harmony`;
if absent, fall back to the `exchange_cpu` path with a warning) -> `FindNeighborsFast` -> `FindClustersFast`
-> `uwot::umap` on the PCs. Same `S2/object.rds` layout, same reductions/graph names as the Seurat path.
`S2/summary.json` gets `path: "cpu_scfast"` and `engine: "scfast"`, plus `scfast_version` from `scfast_info()`.

## S1b doublets and pre-clustering

Python `scrublet`/scanpy and `S1/metrics_counts.h5ad` are replaced when the engine is scfast:
S1a (R) computes doublet scores and `pre_cluster` itself with a scrublet-style detector
(`R/stages/doublets_scfast.R`: per sample, simulate doublets by summing random cell pairs, log-normalise,
HVG, `RunPCAFast`, kNN via `scfast_knn`, score = fraction of simulated neighbours with a threshold from
the bimodal score distribution; pre-cluster = PCA -> kNN -> SNN -> Leiden res 0.3, seed fixed).
It writes `doublet_score`, `doublet_call`, `pre_cluster` straight into `S1/qc_metrics.tsv.gz`,
S1b becomes a pass-through (`method` recorded, nothing to do) and no `.h5ad` is written. S1c is unchanged.
`doublets.method: scrublet` still works with `engine: seurat`.

## gpu_core

`python -m scpipe.gpu_core --engine {rapids,scanpy,rust}` (default unchanged). `rust` uses `scfast`
(Python) for PCA, kNN, SNN, Leiden; UMAP still uses `umap-learn` / rapids. For the Python/exchange route only.

## Install and checks

- R: `R CMD INSTALL rust/scfast-r` (needs cargo >= 1.80). Python: `maturin develop --release -m rust/scfast-py/Cargo.toml`.
- `scpipe doctor` reports scfastR / scfast presence and version, and the engine that `auto` would choose.
- `containers/` recipes install the Rust toolchain and both packages.

## Tests

- Unit: config validation, engine resolution, summary fields.
- Integration (skip cleanly when `scfastR` is absent): `run_stages.sh` variant with `engine: scfast`
  asserts no `*.h5ad` under the run dir, clusters vs planted types ARI >= the Seurat path - 0.05,
  doublet recall on planted doublets, and the full `run_full.sh` passes with `engine: scfast`.
