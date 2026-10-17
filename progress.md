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

## Done
- Baseline, Mac M-series 12 threads, 100k synthetic cells, compute s (excl. read): Seurat 87.3, Scanpy 50.3, scfast-cpu 33.9 (UMAP 29.5).
- Rust UMAP (`scfast.umap`, f51a3ba): UMAP 29.5 -> 2.6 s; then latency fixes (6ee2e62, 7716230): 1.3 s at 12 threads, 5.9 s single thread. Quality equal to umap-learn (PBMC 3k silhouette 0.458 vs 0.460).

- Parity on real data (PBMC 3k vs Seurat 5.6) was FAILING before tonight: PC sim 0.73, kNN 0.36, ARI 0.68. Causes and fixes:
  1. Seurat clips scaled values at 10; the implicit PCA did not. Exact sparse clip correction (e47b108).
  2. Python pca defaults (10/4) too inaccurate; block Krylov SVD replaces randomised SVD, default 10/6 (a10115e): more accurate AND faster (1.12 s vs 1.41 s at 100k).
  3. bench used scanpy seurat_v3 HVGs; now scfast.preprocess (Seurat vst) (ae357a8): ARI 0.85 -> 0.94.
  Now: PC sim 0.999, kNN 0.906, ARI 0.947, all gates pass.
  4. Remaining: scfast loess uses surface="direct"; Seurat uses "interpolate" -> 2/2000 HVGs differ -> kNN 0.906 instead of ~0.977. Delegated (loess agent).

## In progress
Subagents (T3 delegate_task, own worktrees, branches from perf/hillclimb):
- GPU path on Modal (Sonnet 5.5): bench/modal_app.py, CUDA tests on L4, scfast-gpu engine.
- CPU front end (Sonnet 5.5): read/normalize/HVG/scale.
- Real data + baselines on Modal (Opus 5.5 medium): 1.3M brain, PBMC 68k, Seurat/Scanpy on Modal CPU, REFERENCES.md.
- Scale path (Opus 5.5 medium): kNN (exact is O(n^2)), SNN, Leiden at 500k-1M. (PCA now done by me: block Krylov.)
- Loess interpolate port (Opus 5.5 medium): Seurat-exact HVGs.
- Round-1 Sonnet agents (GPU, front end) were interrupted at 22:51; relaunched as Opus 5.5 medium.

## Surprises (evidence)
- 3 Python tests fail on main too (not touched by this branch): test_runner::test_status_output, test_runner::test_failure_then_resume (log-tail line assertions), test_s3::test_consensus_recovers_planted_types (platelet cluster low_confidence). 498 others pass.
- Tried and rejected for UMAP: polynomial fast_pow (scalar and 8-lane batched): no gain; the SGD is bound by random gathers.

## Remaining ideas (ranked)

## Evidence
