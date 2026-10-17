# scpipe user guide

How to run the pipeline on your own machine or on an HPC, read its report, and move it to a
new site. The design is in [`DESIGN.md`](DESIGN.md); the stage-by-stage contracts are in
[`stages-spec.md`](stages-spec.md) (S0–S2) and [`agents-spec.md`](agents-spec.md) (gates,
S3–S5, portability). The `scpipe` command is the only thing you run.

**The principle behind every decision the pipeline makes:** rules decide, agents advise,
reviewers check, humans approve, everything is logged. No agent changes data directly; it makes
a proposal that code validates against policy, and you can always see (and overrule) the result.

## 1. Quick start

```bash
# 1. install (see README.md), then check the node you are on
scpipe doctor --config config.yaml

# 2. run; --detach keeps it alive inside tmux when your SSH session drops
scpipe run --config config.yaml --detach

# 3. follow it
scpipe status --run-dir /scratch/$USER/scpipe/my_study

# 4. if a gate stops the run for your approval (exit code 3), it prints what to do:
scpipe approve --run-dir /scratch/$USER/scpipe/my_study --stage G2 [--from-file edited.json] [--by you]
scpipe resume  --run-dir /scratch/$USER/scpipe/my_study

# 5. open the result
<run-dir>/S5/report.html      # everything, one self-contained page
<run-dir>/S5/explorer.html    # interactive UMAP
```

Stages run in this order: `S0 S1a S1b S1c G1 S2 S3 G2 G3 S4 G4 S5`. Every stage is
checkpointed in `state.json` with a fingerprint of its inputs and settings, so `scpipe run`
again with the same config skips what is done, and a config change reruns only the stages that
use the changed settings and those after them.

| Stage | What it does |
|---|---|
| S0 | Reads every sample (10x h5, 10x mtx folder, h5ad, Seurat rds/qs2, loom, csv/tsv), harmonises genes, merges |
| S1a, S1b, S1c | Per-cell QC metrics, doublets, pre-clustering, then rules-based QC decisions (`keep` / `exclude` / `review`) |
| G1 | QC agent reviews the rules' decisions (gate) |
| S2 | Filter, normalise, variable genes, PCA, optional Harmony, neighbours, Leiden, UMAP; CPU (Seurat) or GPU (rapids-singlecell) |
| S3 | Markers, signature scores, CellTypist (if installed), consensus annotation in code |
| G2, G3 | Context agent (tissue, disease candidates) and annotation agent (gates) |
| S4 | Pseudobulk DGE per contrast and cell type, Open Targets evidence for the DE genes |
| G4 | Interpretation agent writes labelled hypotheses, the reviewer checks every cited ID (gate) |
| S5 | `report.html` and `explorer.html` |

## 2. Configuration

Three files, usually next to your data (relative paths resolve against the folder holding
`config.yaml`):

| File | What goes in it |
|---|---|
| `config.yaml` | Project, run directory, `site`, `execution`, resource caps, stage settings, `llm`, `gates`, `dge`, `opentargets`, `report` |
| `samples.csv` | `sample_id`, `path`, optional `format`; any other column (donor, condition, batch, ...) becomes cell metadata and can be used by `dge` |
| `qc_policy.yaml` | Hard limits, adaptive MAD bounds, doublet handling, protected populations, review triggers |

Start from [`config/examples/`](../config/examples). Settings are validated before anything
runs and every problem is reported at once. The sections added by build steps 3–7:

```yaml
site: iridisx                    # a name in conf/sites/<name>.yaml, or a path; supplies defaults for
                                 # execution, resources, container, scheduler, scratch (config.yaml wins per key)
execution: direct                # direct | slurm | pbs | lsf | sge | split   (section 5)

gates:                           # who gets the last word at each gate (section 4)
  G1: auto                       # auto: agent proposal approved by the reviewer -> run continues
  G2: review                     # review: run stops for a human even when the reviewer approves
  G3: auto                       # escalate: always a human
  G4: review

llm:                             # omit the section for rules-only gates (every gate then needs a human)
  default:
    provider: anthropic          # anthropic | openai_compatible (vLLM, Ollama, ...) | mock (tests)
    model: claude-haiku-4-5
    api_key_env: ANTHROPIC_API_KEY   # the NAME of the variable holding the key; the key itself is never written anywhere
    # base_url: http://localhost:11434/v1     # openai_compatible only
  roles: {}                      # per-role overrides: qc_worker, context_worker, annotation_worker, interpretation_worker, reviewer
  budget:
    per_run_limit: 5             # GBP; a call that would cross a limit is not made
    total_limit: 30              # GBP across all runs (ledger in ~/.scpipe/ledger.json)
    usd_to_gbp: 0.78
  replay: false                  # true: reuse logged outputs for identical prompts (no calls, no cost)

dge:
  sample_col: sample_id
  condition_col: condition
  group_by: cell_type_broad      # or cluster
  contrasts:
    - {name: disease_vs_control, numerator: disease, denominator: control, covariates: [batch]}
  min_cells_per_sample: 20
  min_replicates: 2
  method: deseq2                 # limma_voom is the fallback when DESeq2 is missing
  padj: 0.05
  lfc: 0.5

opentargets:
  enabled: true
  offline: false                 # true: read the cache only, never the network
  max_genes_per_direction: 50
  cache_dir: refs/opentargets    # shared across runs; entries never expire

report:
  explorer_max_cells: 100000     # explorer.html subsamples above this
  expr_genes: 200                # variable genes exported for colour-by-gene
  expr_max_cells: 20000          # cells with expression values in the explorer
  top_markers: 10                # markers per cluster shown in report.html
```

## 3. The run directory

```
<run-dir>/
  config/            copies of your files + config.resolved.yaml (what the stages actually read) + hashes.json
  state.json         per stage: status, start/end, wall time, peak memory, exit code, input fingerprint
  logs/<STAGE>.log
  S0/  object.rds  ingest_report.json
  S1/  qc_metrics.tsv.gz  qc_decisions.tsv.gz  qc_summary.json
  S2/  object.rds  summary.json
  S3/  markers.tsv.gz  cluster_summary.json  signatures.tsv.gz  celltypist.tsv.gz  evidence_bundle.json
       annotation_consensus.tsv  annotation.tsv (after G3)
  S4/  pseudobulk/  dge/<contrast>__<group>.tsv.gz  dge_summary.json  opentargets_summary.json
  S5/  report.html  explorer.html  umap.tsv.gz  expr_subset.tsv.gz  export.json
  gates/G1..G4/      bundle.json  proposal.json  review.json  decision.json  report.html
  llm/               ledger.json (cost)  agent_log.jsonl (every call, replayable)
  submit/            <scheduler>.sh and last_job.json when a job was submitted (section 5)
```

Put the run directory on scratch; results you keep belong on project storage.

## 4. Gates and approvals

At each gate the pipeline builds `bundle.json` (aggregated tables only, never per-cell data or
free-text metadata), the rules engine makes a baseline decision, the worker agent proposes,
code validates the proposal against policy, and a separate reviewer agent (which never sees the
worker's rationale) returns `approve`, `request_changes` or `escalate`. What happens next
depends on `gates.<G>`:

| Mode | Outcome |
|---|---|
| `auto` | reviewer approves -> `decision.json` written, run continues; otherwise the run stops for you |
| `review` | the run always stops for you with the proposal and review laid out |
| `escalate` | same, and the reviewer is told to be strict |

When a run stops at a gate it exits with code 3, writes `gates/<G>/report.html`, and prints:

```
scpipe approve --run-dir D --stage G2 [--from-file edited.json] [--by NAME]
scpipe resume  --run-dir D
```

Open the gate's `report.html`, optionally copy `decision.json`, edit it and pass it with
`--from-file`; the edited decision is validated against the same policy as the agent's. The
decision records `decided_by: human:<name>`. Without an `llm` section, or when the budget guard
refuses a call, the rules baseline becomes the decision (`fallback: true`) and the gate goes to
`review` mode, so nothing is ever applied silently.

Every LLM call is logged to `llm/agent_log.jsonl` with hashes of the prompt and input, the
tokens used and the cost in GBP; `llm/ledger.json` holds the totals. `scpipe status` shows the
spend so far, and `report.html` has a per-gate, per-role cost table.

## 5. Execution modes and sites

`scpipe` drives the stages itself. Where they run is a configuration switch, not a different
pipeline:

| `execution` | Where the stages run | Use when |
|---|---|---|
| `direct` | on the current node (inside `tmux` with `--detach`), with the resource caps and `nice` from `resources` | the node with internet is a login node or workstation (Iridis X default) |
| `slurm`, `pbs`, `lsf`, `sge` | one scheduler job on a compute node that has internet | a site's compute nodes have internet |
| `split` | heavy stages (S0–S4) as an offline scheduler job; gates, Open Targets and the report on the node you submit from | a site does not allow compute on its internet node |

`scpipe run --execution slurm` (or `execution: slurm` in the config or site file) prepares the
run directory, writes `<run-dir>/submit/slurm.sh` from `conf/templates/slurm.sh.j2` with the
`scheduler` settings (partition, account, time, cpus, mem_gb, gpus, extra directives), and
calls `sbatch`. The job runs `scpipe resume --run-dir D --execution direct`, so the job and your
login shell share one `state.json`: `scpipe status` works from either, and `scpipe resume` after
the job (or after a gate) continues from wherever it stopped. The API key is never written into
the script; the scheduler propagates your environment (`--export=ALL`, `-V`). Set
`container.runtime` and `container.image` to run the job inside the Apptainer/Docker image.

`split` submits the stages up to S4 with `--no-gates` (gate stages are recorded as skipped and
S2 uses the rules' QC baseline), then prints the `scpipe resume` command to run on the
submitting node once the job has finished: it runs the skipped gates in place, Open Targets and
S5. With `execution: split` and a gate mode of `review`, expect the run to stop at each gate
for your approval as usual.

### Site files

A site is one small YAML file, `conf/sites/<name>.yaml`, that supplies defaults for
`execution`, `resources`, `container`, `scheduler` and `scratch`. Three are shipped:

| Site | For |
|---|---|
| `iridisx` | the Iridis X L4 login node: `direct`, 8 threads, 64 GB, 20 GB GPU cap, `nice 10`, Apptainer with `--nv`; `scheduler.partition: gpu` for `slurm`/`split` |
| `workstation` | your own machine or VM: no scheduler, no container, higher caps, `nice 0` |
| `docker` | inside the scpipe Docker image with `--gpus all` |

Point `config.yaml` at one with `site: iridisx` (or a path). Anything set in `config.yaml`
overrides the site file key by key. Moving to a new HPC means writing a new site file, not
touching pipeline code.

### `scpipe init-site`

On a new machine, let the pipeline draft the site file:

```bash
scpipe init-site                                    # print the draft
scpipe init-site --write conf/sites/mycluster.yaml --name mycluster
scpipe init-site --json                             # the raw detection
scpipe init-site --no-internet-check                # skip the HTTPS probes (offline node)
```

It detects the scheduler (`sbatch` / `qsub` / `bsub`, telling PBS and SGE apart), the GPU and
its memory (`nvidia-smi`), the container runtime (`apptainer` / `singularity` / `docker`), CPU
count and RAM, scratch candidates, and whether HTTPS reaches the LLM endpoint and Open Targets.
It then drafts conservative values: half the node when a scheduler is present (a shared node),
`--nv` when Apptainer and a GPU are both there, `split` when there is a scheduler but the LLM
endpoint is not reachable from this node. Review the draft, fill in `scheduler.partition` and
`account`, and point `config.yaml` at it.

### `scpipe doctor`

```bash
scpipe doctor --config config.yaml
```

Preflight checks, grouped: R and Python packages (required and optional), GPU, write access to
the run directory and scratch, the site file (valid, known keys), the execution mode (submit
command on PATH), the container runtime and image, the LLM API key (present or not; the key is
never printed), the LLM endpoint, a 1-token tool-use test (skipped with a note when no key is
set), the budget remaining across runs, and Open Targets (reachable, or `offline`). Required
checks failing give exit code 1; the LLM and Open Targets checks are informational, since the
pipeline degrades to rules-only gates and cached evidence without them. `SCPIPE_DOCTOR_OFFLINE=1`
skips the network probes (the tests set it).

### Containers

`containers/scpipe.def` (Apptainer) and `containers/Dockerfile` build the same image (CUDA 12
base, R 4.4 with Seurat 5, BPCells, presto, DESeq2, limma, harmony, leidenbase; Python 3.12 with
scpipe, rapids-singlecell, CellTypist, jinja2, matplotlib):

```bash
apptainer build scpipe.sif containers/scpipe.def
docker build -t scpipe:latest -f containers/Dockerfile .
```

Set `container.runtime: apptainer` and `container.image: /path/to/scpipe.sif` (with
`options: ["--nv"]` for the GPU) in the site file; scheduler jobs then run inside it.

## 6. The report and the explorer

`S5/report.html` is a single self-contained file (inline CSS and JavaScript, plots as inline
SVG or base64 PNG, no external resources), so it can be emailed or kept with the results. Its
sections, in order:

| Section | Contents |
|---|---|
| Run summary | project, run id, samples, execution mode and site, the per-sample ingest table (format, cells, genes, UMIs, warnings), and the stage table from `state.json` (status, start, wall time, peak memory, exit code) |
| Resources | caps in force, host, container, the device and memory plan S2 chose and why, peak memory over all stages |
| Quality control | per-sample table (kept / review / excluded, thresholds actually applied), decisions and reason-code charts, protected populations, triggered review rules, per-sample histograms of genes, UMIs and % mito with the thresholds drawn |
| Clusters and annotation | UMAP coloured by cluster, by broad and fine cell type, by sample and by condition; cells per cluster; per-cluster medians, labels with CL ids, confidence and status, top markers |
| Markers | top N markers per cluster (log2FC, % in, % out, adjusted p) |
| Differential expression | per contrast and group: method used, samples per side, tested / up / down, top genes, why a group was skipped; a bar chart per contrast |
| Open Targets evidence | per contrast, group and direction: each DE gene with Ensembl id, known drugs (phase, disease), top associated diseases, tractability; offline badge when the cache alone was used |
| Gate decisions | one card per gate: mode, who decided, reviewer verdict and findings, policy check, the agent's rationale (advisory), changes versus the proposal, the decision JSON, link to the gate's own report; "waiting for a human" when a decision is pending |
| LLM cost | provider, number of calls (and replays), total GBP against the per-run and total limits, a per-gate / per-role token and cost table from `llm/ledger.json` |
| Software | scpipe, Python and R package versions recorded by the stages |

A section whose inputs are missing (a stage that has not run, or was skipped) says so instead of
failing, so the report can be built at any point: `python -m scpipe.stages.s5_report --run-dir D
--skip-export` rebuilds it from what is there without rerunning the R export.

`S5/explorer.html` draws the UMAP on a canvas from data embedded in the page: colour by cluster,
cell type, sample, condition or any other sample-sheet column, by QC metrics, or by the
expression of any of the exported variable genes (type the name). Hover shows the cell, the
legend toggles groups, wheel zooms, drag pans, and a button saves the view as PNG. Above
`report.explorer_max_cells` (100k) cells a seeded subsample is embedded, keeping the cells that
have expression values first.

Each gate's `gates/<G>/report.html` is built from the same templates and ends with the
`scpipe approve` command for that gate.

## 7. Tests

```bash
PYTHONPATH=python python -m pytest -q tests/python     # unit tests, all offline
tests/integration/run_stages.sh                        # S0-S2 end to end, CPU and GPU path
tests/integration/run_full.sh                          # S0-S5 with a mock LLM and offline Open Targets
```

`run_full.sh` prints a PASS / FAIL / SKIP table; a stage that is not registered in the runner yet
is SKIP, so the script passes on a partial tree. Details in [`tests/README.md`](../tests/README.md).
