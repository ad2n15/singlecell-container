# Performance hillclimb (branch `perf/hillclimb`)

Goal: make the core pipeline (read → normalize → HVG → PCA → neighbours → Leiden → UMAP) as fast as
possible, keep the parity gates (ARI ≥ 0.9 vs Seurat and vs planted types; bench/README.md), and
produce a benchmark for a 3-minute demo talk.

Constraints from the user (2026-10-17):
- Modal budget: soft $30 total (raised 23:15). Allocated: GPU agent $8, real-data agent $8, scale agent $4, headline ~$10. L4 for sweeps, one H100 run for the headline.
- Parity gates stay. Commit after each change. Local branch only; no push.
- Modal CLI: `uvx modal` (profile mukeshkbj). No Docker locally. `goalctl` is not installed.

## Spend log (Modal, estimated)

| When | What | GPU-h / CPU-h | Est. $ |
|---|---|---|---|
| 10-17 22:40-23:50 UTC | GPU agent: image build, wheel builds, CUDA tests, L4 benches and profiling (app `scfast-bench`, Modal billing report) | ~1.3 L4-h + builds | 2.06 |

## Done
- Baseline, Mac M-series 12 threads, 100k synthetic cells, compute s (excl. read): Seurat 87.3, Scanpy 50.3, scfast-cpu 33.9 (UMAP 29.5).
- Rust UMAP (`scfast.umap`, f51a3ba): UMAP 29.5 -> 2.6 s; then latency fixes (6ee2e62, 7716230): 1.3 s at 12 threads, 5.9 s single thread. Quality equal to umap-learn (PBMC 3k silhouette 0.458 vs 0.460).

## GPU path (branch t3/837279cd, GPU agent round 2)
- Image: prebuilt `rapids-singlecell-cu12` 0.18.1 + RAPIDS 26.10 wheels; builds on Modal.
- CUDA bugs fixed: cuGraph dtype enum (INT32 = 2, not 0); Leiden `max_level` = n_iterations (2) gave 580 fragments, now 100;
  cargo did not rebuild mounted sources (mtime), now touched; kNN `topk` O(n^2) with L1 thrash, replaced by a fused kernel;
  serial symmetry check before cuGraph (2 s at 500k), now parallel. All CUDA tests pass on L4 (incl. new device-input PCA test).
- Bench: rapids `n_iterations=2` was the same max_level bug, removed; both GPU engines use cudaMallocAsync via RMM.
- scfast-gpu now keeps data on the device: PCA takes the CuPy HVG matrix (Gram path, colstats on device), SNN is built
  with cuSPARSE from CuPy indices, Leiden gets cuGraph device views. kNN: fused kernel, Morton order, tile pruning.
- L4, 8 CPU cores, compute s (excl. read), 100k / 500k: rapids 2.14 / 13.68, scfast-gpu 1.82 / 8.74.
  500k scfast-gpu steps: to_gpu 0.69, pca 0.64, neighbors 2.15, cluster (cuGraph) 3.16, umap (cuML) 1.27.
- Parity vs rapids: 100k all pass (ARI 1.0). 500k: PC 0.9997, kNN overlap 0.94, ARI vs truth 0.85 (12 clusters, truth 10).
  Evidence that this is Seurat's objective, not a GPU bug: RB quality (directed, res 0.5) on the 500k SNN:
  truth 2,696,230 < CPU Leiden (11 clusters, ARI 0.936) 2,697,092 < cuGraph (12 clusters) 2,698,180. Decision needed:
  accept, or scale resolution with n. Diagonal at half weight in cuGraph under-splits at res 1 and 2 (100k: 13/20 vs CPU 14/22).
- Rust UMAP on the Modal box: 11.7 s at 100k (8 threads) vs cuML 0.25 s: GPU engine keeps cuML. Mac: 1.3 s. Unexplained.
- Next: final L4 run (all 4 engines, `l4_final`), H100 run at 500k and 1M (`synth_1m`).

## In progress
Subagents (T3 delegate_task, own worktrees, branches from perf/hillclimb):
- GPU path on Modal (Sonnet 5.5): bench/modal_app.py, CUDA tests on L4, scfast-gpu engine.
- CPU front end (Sonnet 5.5): read/normalize/HVG/scale.
- Real data + baselines on Modal (Opus 5.5 medium): 1.3M brain, PBMC 68k, Seurat/Scanpy on Modal CPU, REFERENCES.md.
- Scale path (Opus 5.5 medium): kNN (exact is O(n^2)), SNN, Leiden, PCA at 500k-1M.

## Remaining ideas (ranked)

## Evidence
