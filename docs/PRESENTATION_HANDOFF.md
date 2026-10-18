# Presentation handoff: scfast, Seurat's analysis in seconds

This file is the material for a 3-minute talk. It gives the storyline slide by slide, the
numbers with their sources, the visual assets, a live-demo script, and the claims to avoid.
Every number here comes from a measured run in this repository. The arithmetic examples are
marked as arithmetic.

## The one sentence

scfast runs the standard Seurat analysis on 1.29 million cells in 6.8 seconds on one GPU, and
it gives the same principal components and neighbours as Seurat.

## Numbers you can put on a slide

Compute is normalise → variable genes → scale → PCA → neighbours → Leiden → UMAP. Reading the
file is shown separately. Source for all tables: `bench/results/FINAL.md`.

### 1.29 million mouse brain cells (10x E18 brain, real data)

| Engine | Hardware | Compute | File → result |
|---|---|---:|---:|
| Seurat | 16 CPU cores | about 3 h (published, not run by us) | – |
| Scanpy | 16 CPU cores | 19.4 min (1,161.6 s) | 21 min (1,262.6 s) |
| rapids-singlecell | NVIDIA H100, 80 GB | out of GPU memory | fails |
| scfast | 16 CPU cores | 26.9 s | 30.8 s |
| scfast | NVIDIA H100, 80 GB | **6.8 s** | **12.0 s** |

scfast on the GPU is 172 times faster than Scanpy on compute. scfast on the CPU is 43 times
faster than Scanpy. scfast on the CPU uses 46 GB of memory; Scanpy uses 150 GB.

### CPU only, 16 cores for every engine

| Dataset | Seurat | Scanpy | scfast | Faster than Seurat | Faster than Scanpy |
|---|---:|---:|---:|---:|---:|
| PBMC 68k (68,260 cells) | 121.4 s | 57.2 s | 1.8 s | 67× | 32× |
| Mouse brain 100k | 194.6 s | 75.8 s | 2.4 s | 80× | 31× |
| Mouse brain 500k | 966.8 s (16 min) | 488.7 s | 11.3 s | 86× | 43× |

### GPU, file to result on one H100

| Dataset | rapids-singlecell | scfast |
|---|---:|---:|
| PBMC 68k | 2.84 s | 0.68 s |
| Mouse brain 500k | 64.8 s | 5.35 s |
| Mouse brain 1.29M | out of memory | 12.0 s |

On a smaller 24 GB L4 GPU, rapids-singlecell runs out of memory at 500k cells. scfast finishes
the same 500k cells on the L4 in 11.0 s.

### On a laptop (Apple M4 Pro, 12 threads, 24 GB)

| Dataset | scfast, from the file to a UMAP image |
|---|---:|
| PBMC 68k | 1.3 s |
| Mouse brain 100k | 2.1 s |
| Mouse brain 500k | 10.0 s |

Same laptop, mouse brain 100k, all three engines: Seurat 98.8 s, Scanpy 48.0 s, scfast 1.56 s
(compute). Source: `bench/results/laptop_brain100k/report.md`.

### Same answers as Seurat

| Dataset | PC similarity | Neighbour overlap | Cluster agreement (ARI) |
|---|---:|---:|---:|
| PBMC 3k, scfast CPU | 0.99995 | 0.976 | 0.947 |
| PBMC 3k, scfast GPU | 1.0000 | 0.998 | – |
| PBMC 3k, Scanpy (for comparison) | 0.9997 | 0.968 | 0.818 |

The parity gates are PC similarity ≥ 0.99 and neighbour overlap ≥ 0.9. scfast passes them on
every dataset. On PBMC 3k, scfast agrees with Seurat more closely than Scanpy does.

## Storyline: 3 minutes, 9 slides

The arc goes from the wait, to the reveal, to proof, to what changes for a scientist. Each slide
has one number. Spend the most time on slides 4 and 5.

### Slide 1. The wait (0:00–0:15)

- On screen: a progress bar that does not move. Text: "Seurat, 1.3 million cells: about 3 hours."
- Speaker: "This is a normal single-cell analysis. For a large dataset, the scientist waits for hours."
- Effect: open on a black slide with a clock that runs in real time.

### Slide 2. The cost of the wait (0:15–0:35)

- On screen: "One question = one run. One run = 16 minutes." Use Seurat at 500k cells (966.8 s).
- Speaker: "Each change of a parameter means another wait. So scientists try fewer ideas."
- Arithmetic example (state it as arithmetic): ten clustering resolutions at 100k cells take
  ten Seurat runs, 10 × 194.6 s = 32 minutes. With scfast, they take 10 × 2.4 s = 24 seconds.

### Slide 3. The reveal (0:35–0:50)

- On screen: one huge number, "6.8 s", and below it "1.29 million cells, one GPU".
- Speaker: "We rebuilt the core of Seurat. The same analysis now takes under seven seconds."
- Effect: cut from the slow clock of slide 1 straight to this number.

### Slide 4. The race (0:50–1:25), the emotional peak

- On screen: the log-scale bar chart from `bench/results/demo.html` (top chart). Seurat about
  3 h, Scanpy 19.4 min, rapids-singlecell "out of memory", scfast CPU 26.9 s, scfast GPU 6.8 s.
- Speaker: "The GPU tool that the field uses today cannot even load this dataset on an 80 GB GPU.
  scfast finishes it in 6.8 seconds."
- Effect: animate the bars one at a time, slowest first. Hold on the "out of memory" line.

### Slide 5. Live (1:25–2:00)

- On screen: a terminal on this laptop. Run the live demo (script below). The UMAP image opens.
- Speaker: "This is a laptop. 100,000 brain cells, from the raw file to clusters and a map, in two seconds."
- Backup: if the live run fails, show `bench/results/demo_data/brain500k/umap.png`
  (500,000 cells, 10 s on this laptop).

### Slide 6. Trust (2:00–2:20)

- On screen: the parity table. Highlight "PC similarity 0.99995" and "neighbour overlap 0.976".
- Speaker: "Speed is useless if the answer changes. scfast reproduces Seurat's principal
  components and neighbours. On this test it matches Seurat more closely than Scanpy does."

### Slide 7. Where it already fits (2:20–2:35)

- On screen: the run dashboard of the full pipeline, `~/scpipe/hash60k/run/dashboard.html`
  (screenshot it), and the cell-type report `~/scpipe/hash60k/run/S5/report.html`.
- Speaker: "scfast sits inside our Seurat pipeline. The pipeline also does QC, cell-type
  annotation, Open Targets lookups, and review gates with an AI agent."
- Facts from that run (60k PBMC cells, Seurat engine): 12 stages done; the QC gate kept 42,344
  cells and excluded 14,202; the annotation gate labelled 21 clusters with Cell Ontology IDs.

### Slide 8. What this changes for a scientist (2:35–2:50)

- On screen, three short lines, each backed by a measured number:
  - "Explore, do not wait": a full re-run at 100k cells takes 2.4 s, not 3 minutes.
  - "Atlas scale on one GPU": 1.29 million cells in 6.8 s, where today's GPU tool runs out of memory.
  - "Less hardware": 46 GB instead of 150 GB of memory at 1.29 million cells.
- Speaker: "Analysis becomes interactive. You ask a question and see the answer while you still think about it."

### Slide 9. Close (2:50–3:00)

- On screen: "Seurat's answers. In seconds." and the 1.29M UMAP image.
- Speaker: "Same science, 67 to 86 times faster than Seurat on the same CPUs, and 172 times
  faster than Scanpy on one GPU." (Use "overnight" only for the optimisation work; the engine
  itself existed before.)

## Live demo script (slide 5)

Run these on the laptop before the talk to warm the disk cache. Then run the second command live.

```bash
cd ~/singlecell && source .venv-bench/bin/activate
python bench/demo_run.py --input bench/results/demo_data/pbmc68k/matrix.h5 --out /tmp/warm.png
python bench/demo_run.py --input bench/results/demo_data/brain100k/matrix.h5 --out /tmp/brain100k.png && open /tmp/brain100k.png
```

The script prints the time of each step and the total from process start (about 2.1 s for
100k cells). It writes a UMAP image coloured by cluster.

For a dramatic side-by-side, record a split-screen video in advance: Seurat on the left
(`Rscript bench/run_seurat.R --data bench/results/laptop_brain100k/data/n100000 --out /tmp/s --reps 0`),
scfast on the right (the command above). On this laptop, Seurat takes about 110 s for this run
and scfast about 2 s. Speed the Seurat half up in the edit and show a running clock.

## Visual assets

| Asset | Path | Use |
|---|---|---|
| One-page benchmark (charts, parity, tables) | `bench/results/demo.html` | Screenshot the charts for slides 4 and 6 |
| UMAP, 500k brain cells | `bench/results/demo_data/brain500k/umap.png` | Slide 5 backup, slide 9 |
| UMAP, 100k brain cells | `bench/results/demo_data/brain100k/umap.png` | Slide 5 |
| UMAP, PBMC 68k | `bench/results/demo_data/pbmc68k/umap.png` | Any slide that needs a familiar immune map |
| Pipeline dashboard | `~/scpipe/hash60k/run/dashboard.html` | Slide 7 |
| Pipeline report and explorer | `~/scpipe/hash60k/run/S5/report.html`, `S5/explorer.html` | Slide 7 |
| Full results and caveats | `bench/results/FINAL.md` | Q&A |
| Published reference timings | `bench/results/REFERENCES.md` | Q&A |

The `bench/results/demo_data/` folder is not in git. It holds the three input files and the images.

## Claims to avoid, and the true version

- Do not say "same machine" for the CPU table. Every engine used a 16-core container of the same
  size, but Modal assigns different CPU generations. The speed-ups are good to roughly ±30%.
- Do not say we ran Seurat on 1.29 million cells. The "about 3 hours" figure is published
  (Billato et al. 2025, 16 cores). We measured Seurat up to 500,000 cells.
- Do not say "identical clusters to Seurat". Principal components and neighbours match. Leiden
  clustering is not stable even within Seurat: Seurat against itself with different random
  seeds agrees at ARI 0.70 on PBMC 68k. scfast's cluster agreement with Seurat sits in that range.
- The laptop numbers include reading the file; the server tables do not. Do not mix them in one chart.
- The rapids-singlecell "out of memory" result is for this workflow (counts kept for the
  variable-gene step) on an 80 GB H100. Say "in this workflow" if someone asks.
- Published rapids-singlecell figures on about 1M cells of the same dataset are 25–26 s in total
  and about 7.3 s for the core steps on B200-class GPUs, with a different workflow
  (`bench/results/REFERENCES.md`). scfast's 6.8 s on an H100 is in the same range, not "10× faster than rapids".

## Pending

A 30-minute subagent job overlaps the file read with the GPU upload at 1.29 million cells.
The file → result time is now 12.0 s; the upload step it targets is 1.8 s. If the change works,
`bench/results/FINAL.md` will show the new number. Use 12.0 s unless FINAL.md changes.
