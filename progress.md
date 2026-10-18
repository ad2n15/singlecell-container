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
| to 00:30 | Real-data agent: downloads, prep, Seurat/Scanpy baselines (cpu=16) | CPU | ~4.5 |
| to 01:30 | Scale agent: cpu_steps on Modal | CPU | ~2.5 |
| to 01:00 | GPU agent: builds, CUDA tests, L4 runs | L4 + builds | ~2.1+ (more since) |
| 00:30-01:40 | Orchestrator: scfast-cpu PBMC 68k / brain 100k, UMAP probes | CPU | ~0.6 |
| to 02:45 | GPU agent round 2 total (L4 + H100 synthetic runs), Modal billing report | GPU | 6.15 |
| 01:50-02:00 | Orchestrator: final scfast-cpu brain 500k/1.29M (160 GB) | CPU | ~0.4 |
| from 02:00 | Caps: UMAP-large $3, PCA-large $3, GPU round 3 $7 | | <= 13 |
| 10-17 23:11 | baselines agent: download raw data (2 cores, 8 GB, ~3 min) | 0.1 core-h | 0.01 |
| 10-17 23:16 | prep PBMC 68k (4 cores, 16 GB, ~1 min) | 0.1 core-h | 0.01 |
| 10-17 23:16 | prep brain 100k/500k/all (4 cores, 160 GB, ~12 min) | 0.8 core-h | 0.30 |
| 10-17 23:17 | Seurat + Scanpy PBMC 68k, 3 reps (16 cores, 64 GB, ~14 min total) | 3.7 core-h | 0.30 |
| 10-17 23:28 | Seurat + Scanpy brain 100k, 3 reps (16 cores, 64 GB, ~23 min total) | 6.1 core-h | 0.49 |
| 10-17 23:28 | Seurat + Scanpy brain 500k, 1 rep + warm-up (16 cores, 128 GB; Seurat 38 min, Scanpy 20 min) | 15 core-h | ~1.7 |
| 10-17 23:45 | Scanpy brain 1.29M, 1 rep + warm-up (16 cores, 192 GB, 43 min) | 11 core-h | ~1.7 |
| 10-18 00:30 | baselines agent total: `modal billing report` says $3.36 for app scfast-baselines up to 00:00 BST; estimate incl. last 30 min | | ~4.5 |
| 10-17 22:40-23:50 UTC | GPU agent: image build, wheel builds, CUDA tests, L4 benches and profiling (app `scfast-bench`, Modal billing report) | ~1.3 L4-h + builds | 2.06 |
| 10-18 00:00-02:10 UTC | GPU agent: L4 profiling, final L4 run (4 engines), H100 runs 500k / 1M (app `scfast-bench`, billing report) | ~2 L4-h, ~0.4 H100-h | ~4.1 |
| 10-18 03:14-03:26 BST | 60k hashing GEM-X on L4: scfast-gpu, rapids, scfast-cpu, scanpy, 3 reps (app `scfast-bench`, 11 min) | ~0.2 L4-h | ~0.25 |

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

- Real-data baselines (agent, merged): Modal cpu=16, same image for every engine. Data on volume scfast-data.
- scfast-cpu on Modal cpu=16 (compute s, normalize..umap; speed-up vs Seurat / Scanpy):
  - PBMC 68k: Seurat 121.4, Scanpy 57.2, scfast 5.76 (21x / 10x). PC 1.000, kNN 0.994, ARI 0.840.
  - brain 100k: Seurat 197.6, Scanpy 78.6, scfast 8.55 (23x / 9x). PC 0.99997, kNN 0.996, ARI 0.598 (Scanpy 0.585).
- ARI gate is below Leiden's noise floor on real data (evidence): PBMC 68k, Seurat vs Seurat with Leiden seeds 0..3: ARI 1.0, 0.709, 0.700; scfast seed vs seed 0.67-0.79 (0.73-0.81 converged). Not changed; reported.
- UMAP on Modal x86 16 cores is ~4x slower than on the Mac (PBMC 68k: 3.7 s vs 0.9 s). Probing thread scaling.

CPU kNN / SNN / Leiden / PCA (branch t3/110617a5; bench/cpu_steps.py, bench/modal_cpu_steps.py).
Per-step medians of 3 (s). before = f51a3ba (GEMM exact kNN, single-clustering Leiden);
after = HEAD with `method="auto"` kNN (k-d tree, eps 0.7, above 50k cells).

| data | machine | PCA | kNN | SNN | Leiden | sum |
|---|---|---|---|---|---|---|
| synth 100k | Mac 12-core, before | 0.42 | 0.51 | 0.07 | 0.78 | 1.78 |
| synth 100k | Mac 12-core, after | 0.24 | 0.10 | 0.07 | 0.31 | 0.72 |
| synth 500k | Mac 12-core, before | 2.07 | 11.26 | 0.55 | 5.05 | 18.93 |
| synth 500k | Mac 12-core, after | 1.07 | 0.83 | 0.45 | 1.56 | 3.91 |
| brain 1.29M | Mac 12-core, after | 1.91 | 4.00 | 1.37 | 5.07 | 12.35 |
| brain 100k | Modal 16 CPU, before | 0.16 | 15.51 | 0.08 | 1.01 | 16.76 |
| brain 100k | Modal 16 CPU, after | 0.16 | 0.14 | 0.09 | 1.02 | 1.41 |
| brain 500k | Modal 16 CPU, before | 0.88 | 179.46 | 0.59 | 12.16 | 193.09 |
| brain 500k | Modal 16 CPU, after | 0.59 | 0.95 | 0.41 | 2.99 | 4.94 |
| synth 1M | Modal 16 CPU, before | 1.87 | 28.71 | 0.96 | 15.09 | 46.63 |
| synth 1M | Modal 16 CPU, after | 2.09 | 2.60 | 1.30 | 7.36 | 13.35 |
| brain 1.29M | Modal 16 CPU, after | 1.61 | 3.98 | 1.68 | 10.11 | 17.38 |

Raw JSON: bench/results/cpu_steps/ in the t3-110617a5 worktree (gitignored, local only):
modal16_{before,after,final_after}_*.json, mac12_after_*.json.
Brain 1.29M "before" was stopped (GEMM exact kNN ~20 min per repeat by n^2 from 500k).
Modal "after" rows are HEAD eee161a; Mac rows HEAD 148d418+ (same code paths).

Approximation adopted: kNN eps 0.7 k-d tree. Overlap with exact neighbours: 0.957 (synth
100k), 0.955 (synth 500k), 0.951 (synth 1M), 0.963 / 0.962 / 0.946 (brain 100k / 500k /
1.29M). Leiden ARI against labels from the exact graph: 0.998 / 0.996 / 0.999 (synth 100k /
500k / 1M); vs planted types unchanged (0.953 / 0.964 / 0.949). On brain, Leiden itself is
unstable at resolution 0.5: seed-to-seed ARI on the exact graph is 0.55-0.58, and the
approximate graph's ARI (0.56-0.59) sits inside that noise, so ARI cannot judge kNN there.
Leiden changes vs single-clustering: ARI vs leidenalg 0.999 (synth 100k).

What changed (all deterministic for any thread count; bit-identical where stated):
- kNN: exact k-d tree for <= 16 dims (bit-identical to brute force), eps-approximate mode,
  points/boxes in f32 widened per use. HNSW was 16x slower than exact at 100k (10 dims).
- Leiden: own copy of single-clustering's loop; parallel batched local move (validated
  moves), per-community parallel refinement, parallel aggregation (identical graph), BFS
  node order.
- PCA: register-tiled Z V / Z^T U (bit-identical): 1.8x on Apple silicon, neutral on Modal x86.
- SNN: rows in BFS order of the kNN graph (bit-identical): 1.70 -> 1.36 s at 1.29M.

Tried and dropped: f32 inner power iterations in PCA (5% after tiling); fixed leaf cap for
the k-d tree (recall fell with n); Leiden apply shortcuts via speculative weights (fallback
54% of moves) and via corrections (net 7%, not bit-identical).

- Scale path merged (agent): k-d tree kNN (exact <= 16 dims; eps=0.7 approximate in 'auto' above 50k, overlap >= 0.946 with exact), parallel Leiden (local moving, refinement, aggregation; thread-count independent; ARI 0.999 vs leidenalg at 100k), tiled PCA kernels, SNN in BFS order. PCA+kNN+SNN+Leiden: brain 500k Modal 193 -> 4.9 s; brain 1.29M Modal 17.4 s.
- UMAP x86 fix (34f4535): Z-order relabelling + head-only updates; Modal 16 threads PBMC 68k 3.82 -> 1.07 s.
| 10-18 02:41-03:29 BST | GPU real-data agent (app `scfast-gpu-real`): 7 wheel builds (16-core CPU), CUDA / Python tests + PBMC 3k on L4 (x4), 3 rounds of H100 runs on 4 real datasets, 2 H100 PCA profiles, L4 brain 100k / 500k (Modal billing report at 03:29: $3.69; the last ~10 min may still be missing) | ~1.4 H100-h, ~0.5 L4-h | ~3.9 |

## GPU path on real data, exact clip on the device (branch t3/1fc1e0a8) - done
Merged `t3/gpu-bench-modal-hillclimb` into the perf/hillclimb base. CPU PCA = block Krylov + `Clipped`. CUDA PCA:
- Clip, exact: a scaled value (x - m) / sd exceeds the clip c only at a stored entry, and there the clipped value
  equals the scaled value of x' = m + c sd. So the `cap` kernel replaces the stored values on the device by
  min(x, m + c sd) (constant genes: no cap), and the operator keeps the unclipped means: Z_c = (X' - 1 m^T) diag(1/sd)
  exactly. Z_c V, Z_c^T U and the scores U = Z_c V need nothing else. The Gram correction needs the column sums s of X'
  (one more `colstats` pass): G_c = diag(1/sd) (X'^T X' - m s^T - s m^T + n m m^T) diag(1/sd). Same for host-input
  and device-input (CuPy) paths; this is equivalent to G - Z^T C - (Z^T C)^T + C^T C without forming C.
- Gram eigendecomposition: on the device (`gram_finish` kernel + cuSOLVER `syevd`) for <= 4096 genes. The host one (faer)
  took 2.3 s of 2.4 s for the PCA at brain 100k on the H100 box; the device one 0.06 s, PCA 0.17 s warm (CPU 1.0 s).
  The exact eigenpairs also beat the Krylov Rayleigh-Ritz on parity (PBMC 3k kNN overlap 0.998 vs 0.968).
  Above 4096 genes: gene-side block Krylov on the host Gram matrix (same `power_iters` meaning as the CPU).
- Device preprocess: `scfast.preprocess` takes a cupyx CSR or, above 2^31 - 1 non-zeros, a `(data, indices, indptr,
  shape)` tuple of CuPy arrays. CuPy kernels (row_stats, clipped_sums, hvg_count, hvg_fill) do the passes; the per-gene
  vst step is the CPU's Rust code (`preprocess::vst_from_sums`, split out of `preprocess()`). The counts are integers,
  so library sizes and per-gene sums are exact: brain 100k gives the CPU's HVG list and a bitwise-identical HVG CSR in
  0.08 s (CPU 0.40 s); PBMC 3k the same 2,000 HVGs with two near-ties (2e-14) swapped in order.
- scfast-gpu engine (bench/run_py.py): scfast.read_10x_h5 on the CPU, `to_gpu` uploads the raw counts (as rapids),
  then preprocess / PCA / kNN / SNN / Leiden on the device, cuML UMAP. `pca()` on a CuPy matrix now passes its clip
  (it was always 10).
- Tests (L4): `cargo test --release --features cuda,cugraph` all pass, incl. new `gpu_pca_clip_matches_cpu_clip`
  (18,581 clipped entries; clip vs no clip on the CPU 0.972; GPU clip vs CPU clip top-10 0.9999999984, loadings
  0.9999999993, both input paths). Python `tests/python/test_scfast_py.py`: 106 pass, 2 fail by design on a GPU box
  (`test_pca_parameter_errors` / `test_knn_parameter_errors` with `device="cuda"` expect the no-GPU RuntimeError).
- PBMC 3k parity, scfast-gpu on L4 vs Seurat: PC similarity 1.0, kNN overlap 0.998, ARI 0.89 (CPU: 0.99995 / 0.976).

Real data, H100 80 GB box (cpu=16, i.e. 32 logical; 64-200 GB RAM), volume `scfast-data`, 1 warm-up + 2 measured reps,
medians in s. Compute = normalize..umap. Results: bench/results/gpu_h100_real (merged; parity vs Seurat in parity.jsonl,
vs Scanpy in parity_vs_scanpy.jsonl; outputs and per-run folders bench/results/h100_* are git-ignored, also on volume
`scfast-bench` under results/h100_*).

| engine | cells | read | to_gpu | normalize | hvg | scale | pca | neighbors | cluster | umap | to_cpu | compute | peak GPU GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rapids | PBMC 68,260 | 2.00 | 0.04 | 0.00 | 0.03 | 0.02 | 0.06 | 0.09 | 0.26 | 0.16 | 0.18 | 0.62 | 3.6 |
| scfast-gpu | PBMC 68,260 | 0.09 | 0.06 | 0.01 | 0.00 | 0.02 | 0.05 | 0.09 | 0.20 | 0.15 | 0.01 | 0.52 | 2.3 |
| rapids | brain 100,000 | 10.27 | 0.32 | 0.01 | 0.04 | 0.04 | 0.06 | 0.14 | 0.34 | 0.57 | 0.27 | 1.20 | 7.6 |
| scfast-gpu | brain 100,000 | 0.28 | 0.29 | 0.02 | 0.00 | 0.03 | 0.08 | 0.14 | 0.37 | 0.57 | 0.05 | 1.20 | 3.0 |
| rapids | brain 500,000 | 57.74 | 1.44 | 0.03 | 0.11 | 0.23 | 0.22 | 1.32 | 1.31 | 1.06 | 1.37 | 4.28 | 33.6 |
| scfast-gpu | brain 500,000 | 1.18 | 1.07 | 0.06 | 0.00 | 0.15 | 0.28 | 0.68 | 0.85 | 1.04 | 0.19 | 3.06 | 10.2 |
| rapids | brain 1,290,549 | 150 (warm-up) | ok | out of memory | | | | | | | | fails | 85 |
| scfast-gpu | brain 1,290,549 | 14.31 | 3.97 | 0.15 | 0.00 | 0.49 | 0.70 | 2.83 | 2.23 | 0.40 | 0.62 | 6.80 | 43.3 |

rapids at 1.29M: `normalize` fails allocating 20.9 GB (the `counts` layer copy of 2.6B non-zeros) with 85 GB in use.
File to result (read + transfers + compute), scfast-gpu vs rapids: 68k 0.68 vs 2.84 s; 100k 1.82 vs 12.06; 500k 5.50 vs
64.83; 1.29M 25.7 vs fails. Compute only: 0.52 vs 0.62, 1.20 vs 1.20, 3.06 vs 4.28, 6.80 vs fails.
Before the device preprocess (CPU Rust front end, HVG CSR uploaded in `pca`): compute 0.69 / 1.58 / 5.83 / 19.07 s
(normalize 0.10 / 0.40 / 2.34 / 11.62 s); before the cuSOLVER eigendecomposition PCA was 0.80 / 1.00 / 1.58 / 2.14 s.
Remaining scfast-gpu floor: cuGraph Leiden (2.2 s at 1.29M), kNN (2.8 s), the raw upload (4.0 s, pageable memory).

L4 24 GB box (cpu=8, 64 / 96 GB RAM), same data and reps, bench/results/gpu_l4_real:

| engine | cells | read | to_gpu | normalize | hvg | scale | pca | neighbors | cluster | umap | to_cpu | compute | peak GPU GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rapids | brain 100,000 | 8.22 | 0.18 | 0.05 | 0.08 | 0.09 | 0.26 | 0.30 | 0.52 | 0.91 | 2.23 | 2.21 | 6.8 |
| scfast-gpu | brain 100,000 | 0.50 | 0.16 | 0.07 | 0.00 | 0.03 | 0.29 | 0.23 | 1.02 | 0.91 | 0.21 | 2.56 | 2.2 |
| rapids | brain 500,000 | | | out of memory | | | | | | | | fails | 23.7 |
| scfast-gpu | brain 500,000 | 3.34 | 1.53 | 0.34 | 0.00 | 0.28 | 0.76 | 1.85 | 4.14 | 3.60 | 0.33 | 10.97 | 9.4 |

rapids on the L4 at 500k: `normalize` fails allocating 4.1 GB with 23.7 GB in use. scfast-gpu's L4 Leiden (cuGraph on the SNN
graph) is 2x rapids' at 100k (1.02 vs 0.52 s; on the H100 0.37 vs 0.34). L4 parity vs Seurat: scfast-gpu 100k 0.99997 /
0.996 / ARI 0.60 (17 clusters; 21 on the H100: cuGraph Leiden is not deterministic across GPUs), 500k 0.99967 / 0.994 /
0.54; rapids 100k 0.99997 / 0.995 / 0.61.

Parity vs Seurat (top-10 PC similarity / kNN overlap / ARI): scfast-gpu 68k 0.99998 / 0.994 / 0.73, 100k 0.99997 / 0.996
/ 0.55, 500k 0.99967 / 0.994 / 0.48; rapids 68k 0.99998 / 0.974 / 0.53, 100k 0.99997 / 0.995 / 0.59, 500k 0.99967 /
0.992 / 0.50. ARI vs Seurat is low for every engine on real data, scanpy included (0.56-0.60). 1.29M has no Seurat run;
vs Scanpy: scfast-gpu 0.99868 / 0.999 / 0.59.

Reproduce (per-session wheel / target / app so concurrent sessions do not collide):
    export SCFAST_APP=scfast-gpu-real SCFAST_WHEELS=/vol/wheels-gpureal SCFAST_TARGET=/cache/target-gpureal
    uvx modal run bench/modal_app.py::build
    SCFAST_GPU=L4 uvx modal run bench/modal_app.py::rust_tests
    SCFAST_GPU=L4 uvx modal run bench/modal_app.py::bench --tag pbmc3k --name gpu_pbmc3k_l4 --sizes 2638 --engines scfast-gpu --reps 1
    SCFAST_GPU=H100 SCFAST_CPUS=16 SCFAST_MEM_GB=128 uvx modal run bench/modal_app.py::bench --tag real:brain --name h100_brain500k --sizes 500000 --engines scfast-gpu,rapids --reps 2
    (PBMC 3k: data/pbmc3k/data and the Seurat outputs in results/gpu_pbmc3k_l4/outputs/seurat on volume scfast-bench)

## GPU path (branch t3/837279cd, GPU agent round 2) - done
Compute seconds (excl. read), 1 measured rep after a warm-up, synthetic 10-type data (bench/synth.py + prep.py).
L4 box: 8 CPU cores, 48 GB. H100 box: 8 CPU cores, 160 GB. Results in bench/results/{l4_final,h100_500k,h100_1m} (git-ignored; also on volume `scfast-bench`).

| engine | GPU | cells | to_gpu | norm | hvg | scale | pca | neighbors | cluster | umap | to_cpu | compute |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scanpy | (L4 box CPU) | 100k | - | 0.99 | 2.34 | 3.61 | 33.20 | 9.46 | 5.97 | 87.15 | - | 142.72 |
| scfast-cpu | (L4 box CPU) | 100k | - | 0.96 | 2.29 | 2.34 | 0.46 | 1.43 | 1.99 | 11.19 | - | 20.66 |
| rapids | L4 | 100k | 0.22 | 0.03 | 0.05 | 0.07 | 0.27 | 0.30 | 0.54 | 0.23 | 0.36 | 2.07 |
| scfast-gpu | L4 | 100k | 0.22 | 0.03 | 0.05 | 0.04 | 0.20 | 0.27 | 0.62 | 0.24 | 0.07 | 1.74 |
| scanpy | (L4 box CPU) | 500k | - | 4.65 | 11.09 | 16.90 | 232.83 | 62.23 | 37.99 | 485.02 | - | 850.71 |
| scfast-cpu | (L4 box CPU) | 500k | - | 4.71 | 11.12 | 11.16 | 1.99 | 27.19 | 13.15 | 46.42 | - | 115.74 |
| rapids | L4 | 500k | 1.10 | 0.14 | 0.17 | 0.36 | 1.17 | 3.96 | 3.41 | 1.29 | 1.77 | 13.36 |
| scfast-gpu | L4 | 500k | 0.63 | 0.14 | 0.17 | 0.21 | 0.65 | 1.57 | 3.19 | 1.31 | 0.32 | 8.20 |
| rapids | H100 | 500k | 0.69 | 0.02 | 0.08 | 0.13 | 0.23 | 1.35 | 1.19 | 0.40 | 1.40 | 5.49 |
| scfast-gpu | H100 | 500k | 0.56 | 0.02 | 0.07 | 0.08 | 0.31 | 0.65 | 0.72 | 0.38 | 0.26 | 3.05 |
| rapids | H100 | 1M | 2.51 | 0.04 | 0.13 | 0.27 | 0.43 | 4.23 | 1.60 | 0.32 | 2.68 | 12.21 |
| scfast-gpu | H100 | 1M | 1.51 | 0.04 | 0.13 | 1.11 | 0.42 | 1.93 | 1.27 | 0.30 | 0.29 | 7.00 |

Read (scanpy read_10x_h5, all engines): ~8 s at 100k, ~39 s at 500k, ~55-68 s at 1M. It dwarfs GPU compute.

Parity (bench/parity.py): vs scanpy at 100k every engine ARI 1.0, PC similarity >= 0.9999. scfast kNN overlap vs scanpy 0.94
(both scfast engines, from the shared unclipped PCA), rapids 0.999. ARI vs truth at 500k: rapids 1.0, scfast-cpu 0.93 (11
clusters), scfast-gpu 0.85 (12); H100 1M scfast-gpu 0.93 (11). The 500k split is Seurat's objective: directed RB quality at
res 0.5 on the 500k SNN is truth 2,696,230 < CPU Leiden (11) 2,697,092 < cuGraph (12) 2,698,180. Open decision: accept,
or scale resolution with n. (Diagonal at half weight in cuGraph under-splits at res 1 and 2: 100k 13/20 vs CPU 14/22.)

What changed (commits 99f1ee0..3b74cba):
- Image: prebuilt `rapids-singlecell-cu12` 0.18.1 + RAPIDS 26.10 wheels (no CMake). Modal: touch mounted sources so cargo
  rebuilds; `script` entrypoint runs a profiling script on the GPU box.
- CUDA bugs: cuGraph dtype enum (INT32 = 2, not 0); Leiden max_level = n_iterations (2) gave 580 fragments, now 100 (rapids
  engine had the same bug via rsc); kNN `topk` O(n^2) with L1 thrash; serial symmetry check before cuGraph (2 s at 500k).
- kNN: fused kernel (|r|^2 - 2 q.r, 4 queries per thread, NVRTC per dims/cap), Morton order, tile pruning by bounding box.
- PCA: Gram path (f64-atomic X^T X on device, subspace iteration on host); takes a CuPy CSR (device copy, colstats kernel).
- SNN with cuSPARSE from CuPy indices; Leiden on cuGraph device views; UMAP fuzzy graph with CuPy (= cuML to 4e-6).
- Both GPU engines: cudaMallocAsync via RMM (RMM's fixed pool ran out at 500k on the L4).
- Rust UMAP vs cuML: cuML (0.24 s at 100k on L4) kept; Rust UMAP is 11.7 s on the 8-core Modal box (1.3 s on the Mac; unexplained).

Reproduce:
    uvx modal run bench/modal_app.py::build --no-tests
    uvx modal run bench/modal_app.py::rust_tests                      # all CUDA / cuGraph tests on an L4
    uvx modal run bench/modal_app.py::gen --tag synth --sizes 100000,500000
    uvx modal run bench/modal_app.py::gen --tag synth_1m --sizes 1000000
    uvx modal run bench/modal_app.py::bench --tag synth --name l4_final --sizes 100000,500000 --engines scfast-gpu,rapids,scfast-cpu,scanpy
    SCFAST_GPU=H100 SCFAST_MEM_GB=160 uvx modal run bench/modal_app.py::bench --tag synth --name h100_500k --sizes 500000 --engines scfast-gpu,rapids
    SCFAST_GPU=H100 SCFAST_MEM_GB=160 uvx modal run bench/modal_app.py::bench --tag synth_1m --name h100_1m --sizes 1000000 --engines scfast-gpu,rapids
    uv run --no-project --with numpy --with pandas --with scikit-learn --with scipy python bench/parity.py --run bench/results/l4_final --reference scanpy

Remaining ideas: cuGraph Leiden is the floor of `cluster` (3.0 s at 500k on L4, renumbering does not matter); kNN host parts
(Morton order, re-rank) at 1M; read path (CPU agent).

- GPU round 2 (agent, branch t3/gpu-bench-modal-hillclimb, NOT merged yet): synthetic, compute s: L4 100k rapids 2.07 / scfast-gpu 1.74; L4 500k 13.36 / 8.20; H100 500k 5.49 / 3.05; H100 1M 12.21 / 7.00. CUDA bugs fixed: cuGraph dtype enum, Leiden max_level=2 (580 fragments), O(n^2) kNN top-k, stale cargo builds, serial symmetry check. Its device Gram-matrix PCA does not apply Seurat's clip -> merge aborted; GPU round 3 merges it with an exact device clip and runs the real-data H100 headline.

## PCA and normalize at 500k-1.3M on x86 (branch t3/66475d21, from perf/hillclimb 9fa435b)

Modal cpu=16 (AMD EPYC), real brain data, `bench/run_py.py --engine scfast-cpu`, median of 3
after a warm-up, base (9fa435b) and new in the same container (`bench/modal_pca_probe.py::bench_main`;
machines differ by up to 40% between runs, so compare within a row):

| cells | normalize base -> new | pca base -> new |
|---:|---|---|
| 100k | 0.48 -> 0.26 | 1.11 -> 0.60 |
| 500k | 2.13 -> 1.11 | 4.32 -> 1.86 |
| 1.29M | 7.91 -> 3.10 | 10.33 -> 3.74 |

PBMC 3k vs Seurat after every PCA change: PC similarity 0.99995, kNN overlap 0.9757, ARI 0.947
(identical to before). Brain embeddings vs base: top-30 subspace similarity 0.999999998.

What changed (each a commit; bitwise-identical output unless stated):
- Clip correction (`Clipped`) was serial: 3x the sparse product itself. Now parallel, then
  folded into the `Z v` row tasks.
- Block Krylov: each new block is orthonormalised against the earlier ones as it is made
  (BCGS2 with CholeskyQR, TSQR fallback; block Lanczos with full reorthogonalisation), so
  the `Z^T K` of the final SVD is the products the iteration already made (14 sparse
  products instead of 20) and the cells x 280 Householder QR is gone. The first projection
  coefficient comes from the stored `Z^T K`. Passes fused (basis read twice per block).
  Blocks computed in place in K; K lazily zeroed (faer's `Mat::zeros` fills serially).
  Not bitwise equal to base; matches it to 1e-9.
- preprocess: parallel column-index check in `CsrView::new` (was serial over 2.6e9
  indices); `check_canonical` folded into pass 1; no scipy dtype scan of int64 indices
  when wrapping outputs; log1p table for small integer counts; per-gene pass-1 sums in one
  cache line; duplicate check only for rows that are not strictly increasing.

Tried and dropped (evidence in commit messages / this list):
- Rayleigh-Ritz through the Gram matrix `K^T K` (cheapest): drops directions below
  sqrt(eps); PBMC 3k kNN overlap 0.9757 -> 0.9719, Ritz values lower.
- Skipping the per-block orthonormalisation of `y`: same loss (it is the Gram route).
- Tile-major product kernels (L2-sized gene tables for Zen3): +5% on x86, inside the
  +-10% run-to-run noise; reverted.
- `select_columns` in one pass with per-chunk buffers: slower on the Mac (0.22 -> 0.32 s)
  and on x86 at 1.29M (0.75 -> 0.97 s).
- `select_columns` with a per-row rank bitset instead of the per-row sort: 0.63 -> 0.61 s
  at 1.29M on x86, not kept (the sort is not where the time goes).
- Gram-matrix exact PCA: not tried; sum of nnz_row^2 is ~4e10 scattered updates at 1.29M,
  more than the whole Krylov run.
- MAP_POPULATE / huge pages for K: page faults for 2.9 GB cost 0.06 s on 16 threads on
  Modal, not worth it.

Where the time goes now (1.29M, x86, `modal_pca_probe.py::main`, second call): pca 2.6 s =
Z^T u x7 0.84 + Z v x6 0.55 + Gram-Schmidt 0.72 + the rest 0.5; normalize 2.3-2.7 s =
pass 1 1.4 + HVG select 0.75 + pass 2 0.1. The harness pca step (3.7 s) is slower than
a second call in the probe (2.6-2.9 s): the first call at 1.29M is ~0.5 s slower, all in
the first Z v products; page faults explain only ~0.06 s of it. Not resolved.
Next candidates: select_columns on x86 (count pass is 0.02 s on the Mac, so the 0.75 s is
not memory bandwidth; needs a profiler), f32 sparse products (parity risk).

Spend: Modal billing for scfast-pca-probe + scfast-pf-probe apps: $0.69 (budget $3).

## UMAP at scale (agent, branch t3/bf1bab2f, 2026-10-18 01:45-06:00 BST)

Harness: `bench/modal_umap.py` (cached kNN inputs on volume `scfast-data:umap_bench/`; A/B of git
revisions in one container; PBMC 68k silhouette of Seurat's clusters over N seeds; stray count;
PNG of the 1.29M layout to bench/results/umap/, which is git-ignored). Modal `cpu=16` containers
vary: AMD Milan (family 25 model 1), Genoa (25/17), Intel Xeon (6/85); speeds differ up to 2.5x,
so only compare revisions within one run. All runs use 16 threads.

Result, base e290045 -> da3c506, same container each (s, median of warm reps):
| container | PBMC 68k | brain 100k | 500k | 1.29M |
|---|---|---|---|---|
| Milan (e290045 vs 8384569, before the last 3 wins) | 1.28 -> 0.70 | 1.75 -> 0.96 | 8.2 -> 4.2 | 20.2 -> 11.5 |
| Intel Xeon (final da3c506) | 1.57 -> 0.62 | 2.5 -> 0.86 | 15.0 -> 4.6 | 50.1 -> 12.2 |
| AMD Genoa (final da3c506) | 1.00 -> 0.47 | 1.49 -> 0.66 | 7.2 -> 2.7 | 19.5 -> 7.1 (cold 24.5 -> 7.6) |
Later wins on Genoa/Milan, each vs its parent: 32 lanes 8.5 -> 7.75 s, lane count in a register +
rng streams 7.6 -> 7.15 s, branch-free due packing 7.95 -> 6.6 s (1.29M). Cold first call at 1.29M
(fresh process): Milan 27.3 -> 11.6 s (fuzzy set 6 s -> 0.4-0.6 s). Mac M-series 12 threads (loaded
machine): PBMC 0.56 -> 0.46-0.55 s, 1.29M 12.6-13.0 -> 8.1-10.1 s (scalar path; no AVX2 on arm).
Quality: PBMC 68k silhouette mean over 10 seeds, Modal x86: 0.4029 base, 0.4037 final; Mac:
0.4037 base, 0.4095 final (range 0.39-0.43 for every version). 1.29M PNGs (base, final): same
islands; strays (cells beyond 1.5x the 99.9th-percentile radius) 0-4 in every version, the same
outlier cells (nearest kNN neighbour far away), e.g. Mac seeds 0-2 base 3/0/3, final 2/2/3.

Where the time went (Milan, 500k, 1 thread, after the AVX2 kernel): 62 s; negatives without memory
reads 63 s (memory is not the bound); no force maths 35 s. The 8-lane kernel was a ~170-cycle
dependency chain per call -> 32 lanes (four independent groups). Thread scaling 1 -> 16 threads 13.4x.

Kept (in order): 49352c4 prefetch negatives 16 ahead; b1e4684 stateless epoch schedule (edge due
when floor(epoch/eps) steps up, negative_sample_rate negatives per sample; 8 read-only bytes per
edge per epoch); 219e759 fuzzy union via parallel transpose (same graph bit for bit); 45d2838 AVX2+FMA
8-lane force kernel with polynomial pow (runtime-detected; scalar powf elsewhere); 98d9d3d blob test by
layout neighbours (the centre criterion failed 7/30 seeds on the base too); 8384569 parallel edge
setup; 4245389 32 lanes; 810200f lane count in a register, rng stream per prefetch slot; c05c84e
branch-free due packing; da3c506 cleanup.

Rejected (same-container A/B): autovectorised per-row batch (+50 %); scalar polynomial pow (+30 %);
prefetch distance 32; rcp+Newton instead of divisions; negatives in blocks of 8 from 8 streams;
rows sorted by rate; one-floor due test; 64 lanes (all within noise).

Remaining ideas: NEON kernel for arm (Mac); the scalar lane-filling loop (rng, ring, two loads and
three stores per sample) is now about half the time; per-epoch barrier cost at PBMC size.
Spend: about $1.9 of the $3 budget (`modal billing report`, app scfast-umap).
||||||| Stash base

## 60k hashing GEM-X PBMC (10x 9.0.0 raw matrix) - 2026-10-18

Input: `60k_Human_TotalSeqB_hashing_3p_gem-x_Multiplex_count_raw_feature_bc_matrix.h5` (in ~/Downloads):
2,397,363 barcodes, 38,606 genes + 8 hashtags (dropped). No filtered matrix at that URL.

- `pipeline_steps.py` default `--min-genes 200` keeps 29,763 empty droplets (a cluster with median 211 genes,
  243 UMIs, ambient LYZ). The count of barcodes falls from 83,937 (>200 genes) to 56,746 (>250), then flattens
  (54,150 at >500). Results: `bench/results/hash60k_raw/` (default) and `hash60k_raw_min500/`.
- Modal L4 bench, 54,150 cells (prep `--min-genes 500 --max-genes 1e9 --max-mt 20`, volume
  `scfast-data:prep/hash60k`), medians of 3: scfast-gpu 2.03 s total (compute 1.12), rapids 14.4 (read 12.9),
  scfast-cpu 5.10, scanpy 79.0. PC sim 0.99999 and kNN 0.96-0.9998 vs scanpy for all; ARI vs scanpy
  0.74-0.77 for all, including rapids (scanpy on GPU), so the gap is Leiden on this data, not scfast.
  `bench/results/gpu_l4_hash60k/`.
- scpipe S0-S5 locally (CPU, Seurat path, live Claude Haiku 4.5 gates): run dir `~/scpipe/hash60k/run`, input
  barcodes with >= 250 genes (56,934). 14m23s total; G1 escalated (4 clusters lose 75-99%, mostly scrublet
  doublets 11,893 = 21%), approved by the user; G2-G4 auto-approved; 22 Claude calls, GBP 0.46. 42,732 cells,
  21 clusters (T, NK, B, CD14/CD16 mono, cDC, pDC, HSPC). Key: `~/.config/scpipe/anthropic_api_key` (0600).
- scpipe on Modal GPU: `bench/modal_scpipe.py` builds `containers/Dockerfile` (first build ever). Fixed two
  Dockerfile bugs: `nice-utils` is not an Ubuntu package; the R 4.4.2 pin conflicts with CRAN's r-recommended
  (needs R >= 4.5/4.6), so R is unpinned (same fix in `scpipe.def`).

## In progress
Subagents (T3 delegate_task, own worktrees, branches from perf/hillclimb):
- GPU path on Modal (Sonnet 5.5): bench/modal_app.py, CUDA tests on L4, scfast-gpu engine.
- Real data + baselines on Modal (Opus 5.5 medium): 1.3M brain, PBMC 68k, Seurat/Scanpy on Modal CPU, REFERENCES.md.
- Scale path (Opus 5.5 medium): kNN (exact is O(n^2)), SNN, Leiden at 500k-1M. (PCA now done by me: block Krylov.)
- Loess interpolate port (Opus 5.5 medium): Seurat-exact HVGs.
- Round-1 Sonnet agents (GPU, front end) were interrupted at 22:51; relaunched as Opus 5.5 medium.

## Surprises (evidence)
- `scpipe doctor` tool-use test uses `max_tokens=16`; Claude Haiku 4.5 needs more for a tool call (16: stop_reason max_tokens, 0 calls; 256: tool_use, 1 call), so doctor warns although the key and tool use work.
- 3 Python tests fail on main too (not touched by this branch): test_runner::test_status_output, test_runner::test_failure_then_resume (log-tail line assertions), test_s3::test_consensus_recovers_planted_types (platelet cluster low_confidence). 498 others pass.
- Tried and rejected for UMAP: polynomial fast_pow (scalar and 8-lane batched): no gain; the SGD is bound by random gathers.

Nothing running. Modal spend for this work: about $2.5 (estimate: inputs ~12 min of up to
4 x 8-core/96 GB, step runs ~27 + ~10 min of up to 8 x 16-core/48 GB, one 16-core tile test).

## Remaining ideas (ranked)

- Leiden level-0 local move is the largest step at 1M+ (Modal x86 7-10 s): its serial
  apply re-scans each moved node's edges (1.5 s of 5 s on the Mac at 1.29M brain).
- kNN: share traversal across a leaf's queries, or an f32 leaf scan for the eps mode
  (search is now 36% leaf scan, 25% loop, 19% box bounds, 14% heap).
- Pre-existing failures, not from this work: tests/python test_public_api (umap in __all__),
  umap::tests::two_blobs_separate is flaky under parallel test load.

## Evidence

## Final state (orchestrator, 2026-10-18 morning)

Headline numbers and sources: `bench/results/FINAL.md`; talk page: `bench/results/demo.html`;
live demo: `python bench/demo_run.py --input bench/results/demo_data/<set>/matrix.h5`.

Merged into perf/hillclimb overnight (each with evidence in its commits):
- CPU: Rust UMAP (Z-order, head-only, AVX2 + NEON force kernels, prefetch), block Krylov PCA with
  exact Seurat clip (incremental orthonormalisation), interpolated-loess vst (HVGs = Seurat's),
  fused Rust preprocess and parallel h5 read (int64 above 2^31 nnz), k-d tree kNN, parallel Leiden,
  SNN in BFS order.
- GPU: CUDA PCA via the clipped Gram matrix (cuSOLVER), device preprocess, fused kNN kernel, SNN +
  cuGraph Leiden on device, raw int32-index upload; five CUDA bugs fixed (see GPU sections).
- R/scpipe: RunUMAPFast replaces single-thread uwot on the S2 fast path.

Surprises (evidence):
- Parity on real data was failing before tonight (PBMC 3k PC sim 0.73): missing clip, too-few
  PCA iterations, scanpy HVGs, direct loess. Fixed; now 0.99995 / 0.976 / 0.947.
- The ARI >= 0.9 gate is below Leiden's own noise on PBMC 68k / brain: Seurat vs Seurat across
  seeds 0.70-0.71 (PBMC 68k). Not changed; reported.
- Modal cpu=16 lands on AMD Milan, AMD Genoa or Intel at random; the same build varies up to 2.5x.
- `modal volume get` returned a corrupted 2.5 GB file once (same size, different md5; h5py
  "filter returned failure"); a second download was intact. Check large downloads.
- rapids-singlecell runs out of GPU memory at 1.29M cells on an 80 GB H100 (and at 500k on a 24 GB L4).
- 3 Python tests fail on main too (runner log tails, s3 platelet confidence).
