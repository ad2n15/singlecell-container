"""Run the full scpipe pipeline (S0-S5, live LLM gates) on a Modal GPU, from containers/Dockerfile.

    uvx modal volume put scfast-data cells.h5 scpipe/<name>/input.h5
    uvx modal run bench/modal_scpipe.py::doctor --name hash60k         # builds the image on first use
    uvx modal run bench/modal_scpipe.py::run --name hash60k            # stops at a gate that needs a human
    uvx modal run bench/modal_scpipe.py::approve --name hash60k --stage G1
    uvx modal run bench/modal_scpipe.py::resume --name hash60k
    uvx modal run bench/modal_scpipe.py::pull --name hash60k           # -> ~/scpipe/<name>/modal_run

The Anthropic key is read locally from $ANTHROPIC_API_KEY or ~/.config/scpipe/anthropic_api_key and
passed to the container for that call only. The run directory lives on the scfast-data volume under
scpipe/<name>/run, so a gate can be approved and the run resumed in later calls.
GPU type from SCFAST_GPU (default L4).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
GPU = os.environ.get("SCFAST_GPU", "L4")
KEY_FILE = Path("~/.config/scpipe/anthropic_api_key").expanduser()

app = modal.App("scpipe-gpu")
data_vol = modal.Volume.from_name("scfast-data", create_if_missing=True)
image = modal.Image.from_dockerfile(ROOT / "containers" / "Dockerfile", context_dir=ROOT)

CONFIG = """\
project: {name}
run_dir: /tmp/run
seed: 42
samples: samples.csv
qc_policy: qc_policy.yaml
object_format: rds
site: workstation
execution: direct
resources: {{threads: 8, max_memory_gb: 60, gpu_mem_cap_gb: 20, nice: 0}}
doublets: {{method: scrublet}}
qc: {{review_action: keep}}
normalization: lognorm
n_hvg: 2000
device: gpu
engine: auto
engine_gpu: auto
integration: {{method: none, batch_key: sample_id}}
pcs: auto
n_neighbors: 20
clustering: {{resolution: 0.5}}
gates: {{G1: auto, G2: auto, G3: auto, G4: auto}}
llm:
  default: {{provider: anthropic, model: claude-haiku-4-5, api_key_env: ANTHROPIC_API_KEY}}
  roles: {{}}
  budget: {{per_run_limit: 5, total_limit: 30, usd_to_gbp: 0.78, ledger_path: /vol/scpipe/ledger.json}}
  replay: false
dge: {{sample_col: sample_id, condition_col: condition, group_by: cell_type_broad, contrasts: [],
      min_cells_per_sample: 20, min_replicates: 2, method: deseq2, padj: 0.05, lfc: 0.5}}
opentargets: {{enabled: true, offline: false, max_genes_per_direction: 50, cache_dir: /vol/scpipe/opentargets}}
report: {{explorer_max_cells: 100000, expr_genes: 200, expr_max_cells: 20000, top_markers: 10}}
"""


def key_secret() -> modal.Secret:
    key = os.environ.get("ANTHROPIC_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else "")
    return modal.Secret.from_dict({"ANTHROPIC_API_KEY": key} if key else {})


@app.function(image=image, gpu=GPU, cpu=8, memory=65536, timeout=4 * 3600, volumes={"/vol": data_vol},
              secrets=[key_secret()])
def scpipe_remote(name: str, args: list[str]) -> str:
    """Restore the run dir from the volume, run `scpipe <args>`, save the run dir back."""
    home = Path("/vol/scpipe") / name
    saved, run = home / "run", Path("/tmp/run")
    shutil.rmtree(run, ignore_errors=True)
    if saved.exists():
        shutil.copytree(saved, run)
    cfg = Path("/tmp/config")
    cfg.mkdir(exist_ok=True)
    (cfg / "config.yaml").write_text(CONFIG.format(name=name))
    shutil.copy("/opt/scpipe/config/examples/qc_policy.yaml", cfg)
    (cfg / "samples.csv").write_text(f"sample_id,path,format,donor,condition,batch\n{name},{home}/input.h5,,pooled,pbmc,B1\n")
    if args[0] in ("run", "doctor"):
        args = [*args, "--config", str(cfg / "config.yaml")]
    else:
        args = [*args, "--run-dir", str(run)]
    rc = 0
    for cmd in ([["scpipe", "doctor", "--config", str(cfg / "config.yaml")]] if args[0] == "run" else []) + [["scpipe", *args]]:
        print("$", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd="/work").returncode
    if run.exists():
        shutil.rmtree(saved, ignore_errors=True)
        shutil.copytree(run, saved)
    data_vol.commit()
    return f"scpipe {args[0]} exited {rc}"


@app.local_entrypoint()
def doctor(name: str):
    print(scpipe_remote.remote(name, ["doctor"]))


@app.local_entrypoint()
def run(name: str):
    print(scpipe_remote.remote(name, ["run"]))


@app.local_entrypoint()
def approve(name: str, stage: str, by: str = "human"):
    print(scpipe_remote.remote(name, ["approve", "--stage", stage, "--by", by]))


@app.local_entrypoint()
def resume(name: str):
    print(scpipe_remote.remote(name, ["resume"]))


@app.local_entrypoint()
def pull(name: str):
    out = Path(f"~/scpipe/{name}/modal_run").expanduser()
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run(["uvx", "modal", "volume", "get", "scfast-data", f"scpipe/{name}/run", str(out)], check=True)
    print("pulled to", out)
