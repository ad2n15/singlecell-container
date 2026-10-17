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
| 10-17 23:11 | baselines agent: download raw data (2 cores, 8 GB, ~3 min) | 0.1 core-h | 0.01 |
| 10-17 23:16 | prep PBMC 68k (4 cores, 16 GB, ~1 min) | 0.1 core-h | 0.01 |
| 10-17 23:16 | prep brain 100k/500k/all (4 cores, 160 GB, ~12 min) | 0.8 core-h | 0.30 |
| 10-17 23:17 | Seurat + Scanpy PBMC 68k, 3 reps (16 cores, 64 GB, ~14 min total) | 3.7 core-h | 0.30 |
| 10-17 23:28 | Seurat + Scanpy brain 100k, 3 reps (16 cores, 64 GB, ~23 min total) | 6.1 core-h | 0.49 |
| 10-17 23:28 | Seurat + Scanpy brain 500k, 1 rep + warm-up (16 cores, 128 GB; Seurat 38 min, Scanpy 20 min) | 15 core-h | ~1.7 |
| 10-17 23:45 | Scanpy brain 1.29M, 1 rep + warm-up (16 cores, 192 GB, 43 min) | 11 core-h | ~1.7 |
| 10-18 00:30 | baselines agent total: `modal billing report` says $3.36 for app scfast-baselines up to 00:00 BST; estimate incl. last 30 min | | ~4.5 |

## Done
- Baseline, Mac M-series 12 threads, 100k synthetic cells, compute s (excl. read): Seurat 87.3, Scanpy 50.3, scfast-cpu 33.9 (UMAP 29.5).
- Rust UMAP (`scfast.umap`, f51a3ba): UMAP 29.5 -> 2.6 s; then latency fixes (6ee2e62, 7716230): 1.3 s at 12 threads, 5.9 s single thread. Quality equal to umap-learn (PBMC 3k silhouette 0.458 vs 0.460).
- CPU front end (branch t3/2e65cab6): `scfast.read_10x_h5` inflates plain-gzip h5 chunks in parallel in Rust (libdeflate) straight into CSR; `scfast.preprocess` builds X_hvg in Rust; the scfast-cpu engine never imports scanpy/AnnData. 100k synthetic, 12 threads, medians of 5: read 3.05 -> 0.17, normalize+hvg+scale 2.40 -> 0.22 (fused in normalize), process-to-ready 1.75 -> 0.05 s. Parity unchanged: PBMC 3k vs Seurat PC 0.9985, ARI 0.944 (kNN overlap 0.886, as before); synthetic 20k/100k pass_all, ARI vs truth 1.0. Fat LTO: within noise, not kept; panic = "abort" unsafe (run_kernel catches panics).

- Parity on real data (PBMC 3k vs Seurat 5.6) was FAILING before tonight: PC sim 0.73, kNN 0.36, ARI 0.68. Causes and fixes:
  1. Seurat clips scaled values at 10; the implicit PCA did not. Exact sparse clip correction (e47b108).
  2. Python pca defaults (10/4) too inaccurate; block Krylov SVD replaces randomised SVD, default 10/6 (a10115e): more accurate AND faster (1.12 s vs 1.41 s at 100k).
  3. bench used scanpy seurat_v3 HVGs; now scfast.preprocess (Seurat vst) (ae357a8): ARI 0.85 -> 0.94.
  Now: PC sim 0.999, kNN 0.906, ARI 0.947, all gates pass.
  4. scfast loess used surface="direct"; Seurat uses "interpolate" (2/2000 HVGs differed). Ported by loess agent, merged: HVGs 2000/2000 = Seurat, fit within 6e-12 of R.
  Final PBMC 3k vs Seurat: PC sim 0.99995, kNN 0.976, ARI 0.936 (scanpy: 0.9997 / 0.968 / 0.818).

- CPU front end merged (agent): parallel Rust h5 read (libdeflate), X_hvg in Rust, engine without scanpy/AnnData. 100k: read+normalize+hvg+scale 5.45 -> 0.39 s; process ready 0.05 s (was 1.75 s).
- Current scfast-cpu at 100k synthetic, Mac 12 threads under load avg ~20 (s): read 0.15, normalize 0.19, pca 1.13, neighbors 0.68, cluster 0.87, umap 1.28 -> ~4.3 s incl. read (Seurat 93 s, Scanpy 54 s incl. read, both measured on a quieter machine).
- PCA experiments rejected: Krylov 10/4 (PBMC kNN 0.87, fails); f32 product kernels (~10% single-thread, loses f64 determinism).

- scpipe S2 fast path: scfastR::RunUMAPFast replaces single-thread uwot (R agent, merged): S2 UMAP 100k 50.7 s -> 2.7 s, 20k 9.15 -> 0.43 s; silhouette 0.797 vs uwot 0.793. R test failures (3 scfastR parity, 5 tests/R) also fail on the base commit.
- Agent finding: kNN method "auto" switches to HNSW at >= 50k cells; at 100k x 20 dims HNSW is 10.3 s vs exact 0.97 s and less pure. Forwarded to the scale work (S2 FindNeighborsFast uses auto).

- Real data and CPU baselines (`bench/modal_baselines.py`, branch `t3/scfast-real-data-baselines`).
  Modal volume `scfast-data`: `raw/` (10x files, `pbmc68k_labels.tsv` from Zheng et al.),
  `prep/brain/{n100000,n500000,n1290549}`, `prep/pbmc68k/n68260` (with `cells.tsv` labels),
  `results/<dataset>/<engine>_n<cells>/`. Local copies: `bench/results/{pbmc68k_cpu16,brain_cpu16}`.
  QC brain: `--mt-prefix mt- --min-genes 200 --max-genes 6000 --max-mt 20` (1,306,127 → 1,290,549
  cells, 22,788 genes). QC PBMC: defaults (200 < genes < 2500, MT- < 5 %) → 68,260 of 68,579 cells.
  Spec: Modal `cpu=16` (32 logical CPUs), 64 GB (128/192 GB at 500k / all), threads = 16.
  To compare scfast: run it on the same `prep/<dataset>/n<cells>` folder, put its results.jsonl
  lines and `outputs/` into the same local run folder, then `parity.py` + `report.py`.
- Published 1.3M-brain references: `bench/results/REFERENCES.md`.
- CPU baselines (compute = normalize → UMAP, median of measured reps; read separate):

  | Dataset | Cells | Engine | Reps | Compute s | Read s | Slowest step |
  |---|---:|---|---:|---:|---:|---|
  | PBMC 68k | 68,260 | Seurat 5.6.0 | 3 | 121.4 | 4.6 | UMAP 69.5 s |
  | PBMC 68k | 68,260 | Scanpy 1.12.4 | 3 | 57.2 | 2.0 | UMAP 34.5 s |
  | brain | 100,000 | Seurat | 3 | 194.6 | 21.0 | UMAP 106.5 s |
  | brain | 100,000 | Scanpy | 3 | 75.8 | 8.8 | UMAP 49.5 s |
  | brain | 500,000 | Seurat | 1 | 966.8 | 88.4 | UMAP 579 s |
  | brain | 500,000 | Scanpy | 1 | 488.7 | 52.0 | UMAP 325 s |
  | brain | 1,290,549 | Scanpy | 1 | 1161.6 | 101.0 | UMAP 796 s |

  Seurat at 1.29M not run (published ≈ 3 h on 16 cores; REFERENCES.md).

## In progress
Subagents (T3 delegate_task, own worktrees, branches from perf/hillclimb):
- GPU path on Modal (Sonnet 5.5): bench/modal_app.py, CUDA tests on L4, scfast-gpu engine.
- Real data + baselines on Modal (Opus 5.5 medium): 1.3M brain, PBMC 68k, Seurat/Scanpy on Modal CPU, REFERENCES.md.
- Scale path (Opus 5.5 medium): kNN (exact is O(n^2)), SNN, Leiden at 500k-1M. (PCA now done by me: block Krylov.)
- Loess interpolate port (Opus 5.5 medium): Seurat-exact HVGs.
- Round-1 Sonnet agents (GPU, front end) were interrupted at 22:51; relaunched as Opus 5.5 medium.

## Surprises (evidence)
- 3 Python tests fail on main too (not touched by this branch): test_runner::test_status_output, test_runner::test_failure_then_resume (log-tail line assertions), test_s3::test_consensus_recovers_planted_types (platelet cluster low_confidence). 498 others pass.
- Tried and rejected for UMAP: polynomial fast_pow (scalar and 8-lane batched): no gain; the SGD is bound by random gathers.

## Remaining ideas (ranked)

## Evidence
