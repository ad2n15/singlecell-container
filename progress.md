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
