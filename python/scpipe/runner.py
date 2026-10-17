"""scpipe built-in runner (docs/stages-spec.md section 1).

Commands::

    scpipe run --config config.yaml [--run-dir DIR] [--until STAGE] [--detach]
    scpipe resume --run-dir DIR [--until STAGE] [--detach]
    scpipe status --run-dir DIR [--json]
    scpipe doctor [--config config.yaml] [--run-dir DIR] [--json]

(``python -m scpipe ...`` is the same.)

Stages run in order S0 (R) -> S1a (R) -> S1b (Python) -> S1c (Python) ->
S2 (R), each as its own process under ``nice -n <resources.nice>``, with its
stdout and stderr going to ``<run-dir>/logs/<STAGE>.log``. The runner writes
a header and a footer line (prefixed ``# scpipe:``) around the stage's own
output, so stages print to stdout/stderr and never open their log file.

Checkpoints (``<run-dir>/state.json``)
--------------------------------------
Per stage: ``status`` (pending / running / done / failed), ``started_utc``,
``ended_utc``, ``wall_s``, ``peak_rss_bytes``, ``exit_code``,
``fingerprint`` and ``fingerprint_parts`` (what went into it), ``outputs``
(checksum of each output when the stage finished), ``log`` and ``command``.

The **input fingerprint** of a stage is SHA-256 over the canonical JSON of:

* the values of the config keys the stage uses (:data:`STAGE_CONFIG_KEYS`;
  dotted keys into ``config.resolved.yaml``, ``policy.<key>`` into the parsed
  QC policy), and
* the checksum of each input file. Inputs written by an earlier stage are
  identified by the checksum recorded when that stage finished (so a file
  that a later stage completes in place, like ``S1/qc_metrics.tsv.gz``, or
  deletes, like ``S1/metrics_counts.h5ad``, does not invalidate anything);
  other inputs (config copies, the sample files) are hashed when needed.

File checksums (:func:`file_checksum`): files up to 64 MiB are hashed in full
(``sha256:<hex>``); larger files hash their size, mtime (ns) and first and
last MiB (``sampled:<hex>``), so multi-GB inputs cost milliseconds; touching
such a file therefore counts as a change. Directories (10x ``mtx`` folders)
hash the sorted list of relative paths and file checksums.

Peak RSS is the kernel's ``ru_maxrss`` for the stage process and its
descendants (the largest single process, not a sum), measured through a
small launcher so the runner's own memory is not counted (see
``_LAUNCHER``).

A ``done`` stage is skipped only if its fingerprint is unchanged and its
outputs still exist; otherwise it and every later stage are rerun. If a stage
that must rerun needs a file that a skipped earlier stage wrote and that is
gone (S1c deletes ``S1/metrics_counts.h5ad``), the run starts from that
earlier stage instead.

Environment given to every stage
--------------------------------
``SCPIPE_RUN_DIR``, ``SCPIPE_REPO_ROOT``, ``SCPIPE_PYTHON`` (this
interpreter), ``SCPIPE_CONFIG`` (``config.resolved.yaml``), ``SCPIPE_STAGE``,
``SCPIPE_RUN_ID``, ``PYTHONPATH`` (``<repo>/python`` first), the caps
``SCPIPE_THREADS``, ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``,
``MKL_NUM_THREADS``, ``VECLIB_MAXIMUM_THREADS``, ``NUMBA_NUM_THREADS`` (all
``resources.threads``), ``SCPIPE_MAX_MEMORY_GB``, ``R_MAX_VSIZE``
(``<max_memory_gb>Gb``), ``SCPIPE_GPU_MEM_CAP_GB``, and ``NUMBA_CACHE_DIR``
(``<run-dir>/.cache/numba`` unless already set). Anything else in the
runner's environment (e.g. ``SCPIPE_ALLOW_CPU_AS_GPU``) is inherited.
``SCPIPE_RSCRIPT`` overrides the ``Rscript`` executable.

Test hook: ``SCPIPE_STAGES=module:attribute`` replaces the stage registry
(a list of :class:`Stage`) in a detached or subprocess run.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import config as cfgmod
from .config import ConfigError, canonical_json, get_key

__all__ = [
    "Stage",
    "STAGES",
    "STAGE_CONFIG_KEYS",
    "default_stages",
    "file_checksum",
    "stage_env",
    "run_pipeline",
    "status_text",
    "doctor",
    "main",
]

STATE_FILE = "state.json"
STATE_SCHEMA = 1
GATE_WAITING = 3  # exit code of a gate stage (and of scpipe run/resume) while a human decision is pending
LOG_TAIL = 20
FULL_HASH_LIMIT = 64 * 1024**2
SAMPLE_BYTES = 1024**2

# --------------------------------------------------------------------------
# Stage registry
# --------------------------------------------------------------------------

STAGE_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "S0": ("object_format", "samples_resolved", "sample_columns"),
    "S1a": ("object_format", "doublets.method", "engine", "policy.protected_populations"),
    "S1b": ("doublets.method", "engine", "seed", "device", "resources.gpu_mem_cap_gb"),
    "S1c": ("policy",),
    # G1 = the QC gate (scpipe.gates.g1_qc): rules baseline, optional LLM worker + reviewer, human approval.
    "G1": ("policy", "gates.G1", "llm"),
    "S2": (
        "object_format",
        "normalization",
        "n_hvg",
        "engine",
        "device",
        "integration",
        "pcs",
        "n_neighbors",
        "clustering",
        "seed",
        "qc.review_action",
        "resources.gpu_mem_cap_gb",
    ),
    # S3 = R evidence (markers, signatures) + CellTypist + consensus, one stage (scpipe.stages.s3).
    "S3": ("object_format", "annotation", "signatures"),
    # G2 context (tissue/disease), G3 annotation, G4 DGE interpretation: scpipe.gates.g{2,3,4}_*.
    "G2": ("gates.G2", "llm", "annotation.celltypist_model", "opentargets"),
    "G3": ("gates.G3", "llm", "annotation"),
    "G4": ("gates.G4", "llm", "dge", "opentargets"),
    "S4": ("object_format", "seed", "dge", "opentargets.enabled", "opentargets.max_genes_per_direction",
           "opentargets.offline"),
    "S5": ("project", "seed", "report"),
}

OBJ = "S0/object.rds|S0/object.qs2"
OBJ2 = "S2/object.rds|S2/object.qs2"
# Written by G3 / S3 when those stages exist; S4 uses them when present (optional inputs).
S4_OPTIONAL = ("gates/G3/decision.json", "S3/annotation.tsv")
# S5 (report + explorer) reads whatever earlier stages produced; only the S2 object is required.
S5_OPTIONAL = ("S0/ingest_report.json", "S1/qc_summary.json", "S1/qc_metrics.tsv.gz", "S2/summary.json",
               "S3/cluster_summary.json", "S3/markers.tsv.gz", "S3/annotation.tsv", "S3/annotation_consensus.tsv",
               "S4/dge_summary.json", "S4/opentargets_summary.json",
               "gates/G1/decision.json", "gates/G2/decision.json", "gates/G3/decision.json", "gates/G4/decision.json",
               "llm/ledger.json")


@dataclass
class Stage:
    """One pipeline stage.

    ``command(ctx)`` returns the argv to run (without ``nice``). ``inputs``
    and ``outputs`` are paths relative to the run directory; ``a|b`` means
    either file satisfies it. ``external_inputs(cfg)`` returns absolute paths
    hashed live (e.g. the sample files). ``transient`` outputs may be deleted
    by later stages and are not required for skipping. ``optional_inputs``
    are fingerprinted when present (``"absent"`` otherwise), never required.
    """

    id: str
    lang: str
    command: Callable[["RunContext"], list[str]]
    config_keys: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    transient: tuple[str, ...] = ()
    external_inputs: Callable[[Mapping[str, Any]], list[str]] | None = None
    optional_inputs: tuple[str, ...] = ()


@dataclass
class RunContext:
    run_dir: Path
    cfg: dict[str, Any]
    repo_root: Path


def _rscript() -> str:
    return os.environ.get("SCPIPE_RSCRIPT") or "Rscript"


def _r_stage(stage_id: str) -> Callable[[RunContext], list[str]]:
    def cmd(ctx: RunContext) -> list[str]:
        script = ctx.repo_root / "R" / "stages" / "run_stage.R"
        return [_rscript(), str(script), "--stage", stage_id, "--run-dir", str(ctx.run_dir)]

    return cmd


def _py_stage(module: str) -> Callable[[RunContext], list[str]]:
    def cmd(ctx: RunContext) -> list[str]:
        return [sys.executable, "-m", module, "--run-dir", str(ctx.run_dir)]

    return cmd


def _sample_paths(cfg: Mapping[str, Any]) -> list[str]:
    return [s["path"] for s in cfg.get("samples_resolved") or []]


def default_stages() -> list[Stage]:
    return [
        Stage("S0", "R", _r_stage("S0"), STAGE_CONFIG_KEYS["S0"],
              inputs=("config/samples.csv",), outputs=(OBJ, "S0/ingest_report.json"),
              external_inputs=_sample_paths),
        Stage("S1a", "R", _r_stage("S1a"), STAGE_CONFIG_KEYS["S1a"],
              inputs=(OBJ,), outputs=("S1/metrics_counts.h5ad", "S1/qc_metrics.tsv.gz"),
              transient=("S1/metrics_counts.h5ad",)),
        Stage("S1b", "python", _py_stage("scpipe.stages.s1b_doublets"), STAGE_CONFIG_KEYS["S1b"],
              inputs=("S1/metrics_counts.h5ad", "S1/qc_metrics.tsv.gz"), outputs=("S1/qc_metrics.tsv.gz",)),
        Stage("S1c", "python", _py_stage("scpipe.stages.s1c_qc"), STAGE_CONFIG_KEYS["S1c"],
              inputs=("S1/qc_metrics.tsv.gz", "config/qc_policy.yaml"),
              outputs=("S1/qc_decisions.tsv.gz", "S1/qc_summary.json")),
        # G1 reads S1c's decisions (the baseline) and rewrites S1/qc_decisions.tsv.gz through the rules
        # engine when a decision differs from it, so it is listed as a G1 output (S2 then depends on G1).
        # Exit code 3 = waiting for `scpipe approve`; `scpipe resume` reruns the gate, which applies it.
        Stage("G1", "python", _py_stage("scpipe.gates.g1_qc"), STAGE_CONFIG_KEYS["G1"],
              inputs=("S1/qc_metrics.tsv.gz", "S1/qc_decisions.tsv.gz", "config/qc_policy.yaml"),
              outputs=("gates/G1/decision.json", "gates/G1/bundle.json", "gates/G1/proposal.json",
                       "gates/G1/review.json", "gates/G1/report.html", "S1/qc_decisions.tsv.gz")),
        Stage("S2", "R", _r_stage("S2"), STAGE_CONFIG_KEYS["S2"],
              inputs=(OBJ, "S1/qc_decisions.tsv.gz"), outputs=(OBJ2, "S2/summary.json")),
        # S3 = the R evidence step (markers, cluster summary, signatures), then CellTypist (skipped with a
        # note when not installed), then the consensus annotation; one stage, one log (scpipe.stages.s3).
        Stage("S3", "python", _py_stage("scpipe.stages.s3"), STAGE_CONFIG_KEYS["S3"],
              inputs=(OBJ2, "config/samples.csv"),
              outputs=("S3/markers.tsv.gz", "S3/cluster_summary.json", "S3/signatures.tsv.gz",
                       "S3/evidence_bundle.json", "S3/annotation_consensus.tsv", "S3/celltypist_note.json")),
        Stage("G2", "python", _py_stage("scpipe.gates.g2_context"), STAGE_CONFIG_KEYS["G2"],
              inputs=("S3/evidence_bundle.json",), optional_inputs=("S3/annotation_consensus.tsv",),
              outputs=("gates/G2/decision.json",)),
        Stage("G3", "python", _py_stage("scpipe.gates.g3_annotation"), STAGE_CONFIG_KEYS["G3"],
              inputs=("S3/evidence_bundle.json", "S3/annotation_consensus.tsv"),
              outputs=("gates/G3/decision.json",)),
        # S4 = the R pseudobulk DGE step, then the Open Targets Python step (one stage, one log).
        Stage("S4", "python", _py_stage("scpipe.stages.s4"), STAGE_CONFIG_KEYS["S4"],
              inputs=(OBJ2,), outputs=("S4/dge_summary.json", "S4/opentargets_summary.json"),
              optional_inputs=S4_OPTIONAL),
        Stage("G4", "python", _py_stage("scpipe.gates.g4_interpretation"), STAGE_CONFIG_KEYS["G4"],
              inputs=("S4/dge_summary.json",), optional_inputs=("S4/opentargets_summary.json",),
              outputs=("gates/G4/decision.json",)),
        # S5 must stay last: the R export (S5/umap.tsv.gz, S5/expr_subset.tsv.gz) then report.html/explorer.html.
        Stage("S5", "python", _py_stage("scpipe.stages.s5_report"), STAGE_CONFIG_KEYS["S5"],
              inputs=(OBJ2,), outputs=("S5/report.html", "S5/explorer.html", "S5/umap.tsv.gz"),
              optional_inputs=S5_OPTIONAL),
    ]


STAGES = default_stages()


def _registry(stages: Sequence[Stage] | None) -> list[Stage]:
    if stages is not None:
        return list(stages)
    hook = os.environ.get("SCPIPE_STAGES")
    if hook:
        mod, _, attr = hook.partition(":")
        obj = getattr(importlib.import_module(mod), attr or "STAGES")
        return list(obj() if callable(obj) else obj)
    return default_stages()


def _stage_index(stages: Sequence[Stage], stage_id: str | None) -> int:
    if stage_id is None:
        return len(stages) - 1
    for i, s in enumerate(stages):
        if s.id.lower() == stage_id.lower():
            return i
    raise ConfigError(f"unknown stage {stage_id!r} (stages: {[s.id for s in stages]})")


# --------------------------------------------------------------------------
# Checksums and fingerprints
# --------------------------------------------------------------------------


def file_checksum(path: str | Path) -> str:
    """Checksum of a file or directory (see module docstring)."""
    p = Path(path)
    if p.is_dir():
        h = hashlib.sha256()
        for f in sorted(x for x in p.rglob("*") if x.is_file()):
            h.update(f.relative_to(p).as_posix().encode() + b"\0" + file_checksum(f).encode() + b"\n")
        return "dir:" + h.hexdigest()
    st = p.stat()
    if st.st_size <= FULL_HASH_LIMIT:
        return "sha256:" + cfgmod.sha256_file(p)
    h = hashlib.sha256()
    h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    with open(p, "rb") as f:
        h.update(f.read(SAMPLE_BYTES))
        f.seek(max(0, st.st_size - SAMPLE_BYTES))
        h.update(f.read(SAMPLE_BYTES))
    return "sampled:" + h.hexdigest()


def _alternatives(spec: str) -> list[str]:
    return spec.split("|")


def _existing(run_dir: Path, spec: str) -> Path | None:
    for alt in _alternatives(spec):
        if (run_dir / alt).exists():
            return run_dir / alt
    return None


def _policy_dict(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    from . import qc

    p = cfg.get("qc_policy")
    if not p or not Path(p).exists():
        return None
    return qc.load_policy(p).to_dict()


def _config_value(cfg: Mapping[str, Any], key: str, policy: dict[str, Any] | None) -> Any:
    if key == "policy":
        return policy
    if key.startswith("policy."):
        return get_key(policy or {}, key[len("policy."):])
    return get_key(cfg, key)


def _producer(stages: Sequence[Stage], idx: int, spec: str) -> int | None:
    for j in range(idx - 1, -1, -1):
        if spec in stages[j].outputs:
            return j
    return None


def _fingerprint(stages: Sequence[Stage], idx: int, ctx: RunContext, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    st = stages[idx]
    policy = _policy_dict(ctx.cfg) if any(k == "policy" or k.startswith("policy.") for k in st.config_keys) else None
    parts: dict[str, Any] = {"config": {k: _config_value(ctx.cfg, k, policy) for k in st.config_keys}, "inputs": {}}
    for spec in st.inputs:
        j = _producer(stages, idx, spec)
        rec = None
        if j is not None:
            rec = (state["stages"].get(stages[j].id) or {}).get("outputs", {}).get(spec)
        if rec is None:
            p = _existing(ctx.run_dir, spec)
            rec = file_checksum(p) if p is not None else "missing"
        parts["inputs"][spec] = rec
    if st.external_inputs is not None:
        for p in st.external_inputs(ctx.cfg):
            parts["inputs"][p] = file_checksum(p) if Path(p).exists() else "missing"
    for spec in st.optional_inputs:
        p = _existing(ctx.run_dir, spec)
        parts["inputs"][spec] = file_checksum(p) if p is not None else "absent"
    fp = hashlib.sha256(canonical_json(parts).encode()).hexdigest()
    return fp, parts


def _diff(old: Mapping[str, Any] | None, new: Mapping[str, Any]) -> list[str]:
    if not old:
        return ["no previous fingerprint"]
    out = []
    for k, v in new.get("config", {}).items():
        if old.get("config", {}).get(k, object()) != v:
            out.append(f"config {k}")
    for k in old.get("config", {}):
        if k not in new.get("config", {}):
            out.append(f"config {k}")
    for k, v in new.get("inputs", {}).items():
        if old.get("inputs", {}).get(k) != v:
            out.append(f"input {k}")
    return out or ["fingerprint changed"]


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state(run_dir: Path, stages: Sequence[Stage] | None = None) -> dict[str, Any]:
    p = Path(run_dir) / STATE_FILE
    if p.exists():
        state = json.loads(p.read_text())
    else:
        state = {"schema": STATE_SCHEMA, "stages": {}}
    for s in stages or []:
        state["stages"].setdefault(s.id, {"status": "pending", "lang": s.lang})
    return state


def save_state(run_dir: Path, state: dict[str, Any]) -> None:
    state["updated_utc"] = _utc()
    p = Path(run_dir) / STATE_FILE
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, p)


class _Lock:
    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / ".scpipe.lock"
        self.fh = None

    def __enter__(self):
        self.fh = open(self.path, "a+")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.fh.seek(0)
            who = self.fh.read().strip()
            self.fh.close()
            raise RunnerError(f"another scpipe runner is active in {self.path.parent} ({who or 'unknown pid'})")
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(f"pid {os.getpid()} on {socket.gethostname()} since {_utc()}\n")
        self.fh.flush()
        return self

    def __exit__(self, *exc):
        try:
            self.fh.seek(0)
            self.fh.truncate()
            fcntl.flock(self.fh, fcntl.LOCK_UN)
        finally:
            self.fh.close()


class RunnerError(RuntimeError):
    pass


class _Terminated(BaseException):
    def __init__(self, signum: int):
        self.signum = signum


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def stage_env(cfg: Mapping[str, Any], run_dir: Path, stage_id: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a stage process (see module docstring)."""
    env = dict(os.environ if base is None else base)
    repo = Path(cfg.get("repo_root") or cfgmod.REPO_ROOT)
    res = cfg.get("resources") or {}
    threads = str(int(res.get("threads", 1)))
    mem = res.get("max_memory_gb")
    gpu = res.get("gpu_mem_cap_gb")
    py_path = str(repo / "python")
    old = env.get("PYTHONPATH", "")
    parts = [py_path] + [p for p in old.split(os.pathsep) if p and p != py_path]
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(parts),
            "SCPIPE_PYTHON": sys.executable,
            "SCPIPE_RUN_DIR": str(run_dir),
            "SCPIPE_REPO_ROOT": str(repo),
            "SCPIPE_CONFIG": str(Path(run_dir) / "config" / "config.resolved.yaml"),
            "SCPIPE_STAGE": stage_id,
            "SCPIPE_RUN_ID": str(cfg.get("run_id", "none")),
            "SCPIPE_THREADS": threads,
            "OMP_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "VECLIB_MAXIMUM_THREADS": threads,
            "NUMBA_NUM_THREADS": threads,
        }
    )
    if mem is not None:
        env["SCPIPE_MAX_MEMORY_GB"] = f"{float(mem):g}"
        env["R_MAX_VSIZE"] = f"{float(mem):g}Gb"
    if gpu is not None:
        env["SCPIPE_GPU_MEM_CAP_GB"] = f"{float(gpu):g}"
    env.setdefault("NUMBA_CACHE_DIR", str(Path(run_dir) / ".cache" / "numba"))
    env.setdefault("MPLCONFIGDIR", str(Path(run_dir) / ".cache" / "matplotlib"))
    return env


def _nice_prefix(cfg: Mapping[str, Any]) -> list[str]:
    n = int((cfg.get("resources") or {}).get("nice", 0) or 0)
    if n > 0 and shutil.which("nice"):
        return ["nice", "-n", str(n)]
    return []


def _fmt_bytes(b: int | None) -> str:
    if b is None:
        return "-"
    for unit, div in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if b >= div:
            return f"{b / div:.2f} {unit}"
    return f"{b} B"


def _fmt_s(s: float | None) -> str:
    if s is None:
        return "-"
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def tail(path: Path, n: int = LOG_TAIL) -> list[str]:
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:]


# Stages are started through this launcher (``python -S -c``, ~10 MB). Linux
# carries the parent's high-water RSS into a child across fork + exec, so
# ru_maxrss measured from the runner (which has numpy/pandas loaded) would
# never be below the runner's own size. The launcher spawns the stage from a
# fresh small process, waits for it and writes the stage's rusage as JSON.
# ru_maxrss covers the stage process and every descendant it waited for (the
# largest of them, not their sum).
_LAUNCHER = r"""
import json, os, signal, sys
out, argv = sys.argv[1], sys.argv[3:]
try:
    pid = os.posix_spawnp(argv[0], argv, os.environ)
except OSError as e:
    print("# scpipe: cannot start %s: %s" % (argv[0], e), file=sys.stderr)
    sys.exit(127)
for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(s, lambda n, f: os.kill(pid, n))
while True:
    try:
        _, st, ru = os.wait4(pid, 0)
        break
    except InterruptedError:
        continue
with open(out, "w") as f:
    json.dump({"ru_maxrss": ru.ru_maxrss, "utime": ru.ru_utime, "stime": ru.ru_stime}, f)
code = os.waitstatus_to_exitcode(st)
sys.exit(code if code >= 0 else 128 - code)
"""


def _run_process(argv: list[str], env: dict[str, str], cwd: Path, log_path: Path) -> tuple[int, int, float]:
    """Run argv with output to log_path; return (exit_code, peak_rss_bytes, wall_s).

    ``argv`` may start with a ``nice -n N`` prefix; the launcher goes after it.
    """
    t0 = time.monotonic()
    nice: list[str] = []
    rest = list(argv)
    if rest[:1] == ["nice"] and len(rest) > 3 and rest[1] == "-n":
        nice, rest = rest[:3], rest[3:]
    ru_file = log_path.with_name(f".{log_path.stem}.rusage.json")
    ru_file.unlink(missing_ok=True)
    full = nice + [sys.executable, "-S", "-c", _LAUNCHER, str(ru_file), "--"] + rest
    with open(log_path, "a", buffering=1) as log:
        log.write(f"# scpipe: started {_utc()} on {socket.gethostname()}\n# scpipe: command {shlex.join(argv)}\n")
        log.flush()
        try:
            proc = subprocess.Popen(full, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    env=env, cwd=str(cwd), start_new_session=True)
        except OSError as e:
            log.write(f"# scpipe: could not start: {e}\n")
            return 127, 0, time.monotonic() - t0
        try:
            _, status, ru = os.wait4(proc.pid, 0)
        except (KeyboardInterrupt, _Terminated):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                os.wait4(proc.pid, 0)
            except ChildProcessError:
                pass
            proc.returncode = -1
            log.write(f"# scpipe: interrupted at {_utc()}\n")
            raise
        code = os.waitstatus_to_exitcode(status)
        proc.returncode = code
        kb = None
        try:
            kb = int(json.loads(ru_file.read_text())["ru_maxrss"])
        except (OSError, ValueError, KeyError):
            kb = ru.ru_maxrss  # launcher died: fall back (includes the runner's own size)
        finally:
            ru_file.unlink(missing_ok=True)
        peak = int(kb if sys.platform == "darwin" else kb * 1024)
        wall = time.monotonic() - t0
        log.write(f"# scpipe: finished {_utc()} exit code {code}, wall {wall:.1f}s, peak RSS {_fmt_bytes(peak)}\n")
    return code, peak, wall


def _outputs_present(run_dir: Path, st: Stage) -> list[str]:
    """Missing non-transient outputs."""
    return [spec for spec in st.outputs if spec not in st.transient and _existing(run_dir, spec) is None]


def is_gate(st: Stage) -> bool:
    """Gate stages are G1..G4 (docs/agents-spec.md section 2)."""
    return len(st.id) >= 2 and st.id[0].upper() == "G" and st.id[1:].isdigit()


def _plan(stages: Sequence[Stage], until: int, ctx: RunContext, state: dict[str, Any],
          no_gates: bool = False) -> tuple[int, dict[str, str]]:
    """Return (index of the first stage to run, {stage id: reason}); len(stages) if nothing to run.

    A gate recorded as ``skipped`` (an earlier ``--no-gates`` run) counts as done
    here; :func:`run_pipeline` runs such gates in place without invalidating
    the stages after them (their fingerprints change if the gate writes a
    file they consume).
    """
    reasons: dict[str, str] = {}
    start = until + 1
    for i in range(until + 1):
        st = stages[i]
        rec = state["stages"].get(st.id) or {}
        if is_gate(st) and (no_gates or rec.get("status") == "skipped"):
            continue
        if rec.get("status") != "done":
            reasons[st.id] = {"failed": "previous attempt failed", "running": "previous attempt did not finish",
                              "waiting": "applying the human decision (or still waiting for one)"}.get(
                rec.get("status"), "not run yet")
            start = i
            break
        fp, parts = _fingerprint(stages, i, ctx, state)
        if fp != rec.get("fingerprint"):
            reasons[st.id] = "changed: " + ", ".join(_diff(rec.get("fingerprint_parts"), parts))
            start = i
            break
        missing = _outputs_present(ctx.run_dir, st)
        if missing:
            reasons[st.id] = "output missing: " + ", ".join(missing)
            start = i
            break
    # A stage that reruns may need a file that a skipped stage wrote and that is gone.
    changed = True
    while changed and start <= until:
        changed = False
        for k in range(start, until + 1):
            for spec in stages[k].inputs:
                j = _producer(stages, k, spec)
                if j is not None and j < start and _existing(ctx.run_dir, spec) is None:
                    reasons[stages[j].id] = f"{spec} is needed by {stages[k].id} but was removed"
                    start = j
                    changed = True
                    break
            if changed:
                break
    return start, reasons


def run_pipeline(
    run_dir: str | Path,
    cfg: dict[str, Any],
    *,
    until: str | None = None,
    stages: Sequence[Stage] | None = None,
    out=None,
    no_gates: bool = False,
) -> int:
    """Run the stages for an already prepared run directory. Returns 0 on success, 1 on failure.

    ``no_gates`` (``execution: split`` jobs) records gate stages as ``skipped``
    instead of running them; a later run without it runs them in place.
    """
    out = out or sys.stdout
    stages = _registry(stages)
    run_dir = Path(run_dir)
    last = _stage_index(stages, until)
    ctx = RunContext(run_dir, cfg, Path(cfg.get("repo_root") or cfgmod.REPO_ROOT))
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    def say(msg: str) -> None:
        print(msg, file=out, flush=True)

    prev_handlers = {}

    def _raise(signum, frame):
        raise _Terminated(signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            if signal.getsignal(sig) == signal.SIG_IGN:
                continue  # e.g. SIGHUP under nohup: keep ignoring it
            prev_handlers[sig] = signal.signal(sig, _raise)
        except (ValueError, OSError):  # not in the main thread
            pass
    try:
        with _Lock(run_dir):
            state = load_state(run_dir, stages)
            state.update(schema=STATE_SCHEMA, run_id=cfg.get("run_id"), config_hash=cfg.get("config_hash"),
                         run_dir=str(run_dir), stage_order=[s.id for s in stages])
            state["runner"] = {"pid": os.getpid(), "host": socket.gethostname(), "started_utc": _utc(),
                               "until": stages[last].id}
            start, reasons = _plan(stages, last, ctx, state, no_gates=no_gates)
            # Gates skipped by an earlier --no-gates run and not rerun anyway: run them in place.
            pending_gates = [] if no_gates else [
                i for i in range(min(start, last + 1)) if is_gate(stages[i])
                and (state["stages"].get(stages[i].id) or {}).get("status") == "skipped"]
            for i in pending_gates:
                reasons[stages[i].id] = "skipped by an earlier --no-gates run"
            say(f"scpipe: run {cfg.get('run_id')} in {run_dir}")
            for i in range(start):
                if i in pending_gates:
                    continue
                if no_gates and is_gate(stages[i]):
                    say(f"  [{stages[i].id:>4}] skip (gate; --no-gates)")
                else:
                    say(f"  [{stages[i].id:>4}] skip (done, inputs unchanged)")
            if start > last and not pending_gates:
                save_state(run_dir, state)
                say(f"scpipe: nothing to do up to {stages[last].id}; all stages are done and unchanged")
                return 0
            for i in range(start, len(stages)):
                rec = state["stages"][stages[i].id]
                if rec.get("status") == "done":
                    rec["status"] = "pending"
                    rec["note"] = f"invalidated: {stages[start].id} is rerun"
            save_state(run_dir, state)

            for i in pending_gates + list(range(start, last + 1)):
                st = stages[i]
                if no_gates and is_gate(st):
                    state["stages"][st.id] = {"status": "skipped", "lang": st.lang, "note": "skipped: --no-gates",
                                              "skipped_utc": _utc()}
                    save_state(run_dir, state)
                    say(f"  [{st.id:>4}] skip (gate; --no-gates)")
                    continue
                reason = reasons.get(st.id) or f"upstream {stages[start].id} was rerun"
                fp, parts = _fingerprint(stages, i, ctx, state)
                missing_in = [s for s in st.inputs if _existing(run_dir, s) is None]
                rec = {"status": "running", "lang": st.lang, "started_utc": _utc(), "ended_utc": None,
                       "wall_s": None, "peak_rss_bytes": None, "exit_code": None, "fingerprint": fp,
                       "fingerprint_parts": parts, "outputs": {}, "log": f"logs/{st.id}.log", "reason": reason}
                argv = _nice_prefix(cfg) + st.command(ctx)
                rec["command"] = argv
                state["stages"][st.id] = rec
                save_state(run_dir, state)
                log_path = run_dir / "logs" / f"{st.id}.log"
                log_path.write_text("")
                say(f"  [{st.id:>4}] run  ({reason})")
                if missing_in:
                    code, peak, wall = 1, 0, 0.0
                    log_path.write_text(f"# scpipe: missing input(s): {', '.join(missing_in)}\n")
                else:
                    try:
                        code, peak, wall = _run_process(argv, stage_env(cfg, run_dir, st.id), run_dir, log_path)
                    except (KeyboardInterrupt, _Terminated) as e:
                        rec.update(status="failed", ended_utc=_utc(), exit_code=-getattr(e, "signum", signal.SIGINT),
                                   note="interrupted")
                        save_state(run_dir, state)
                        say(f"  [{st.id:>4}] interrupted; resume with: scpipe resume --run-dir {run_dir}")
                        return 130
                rec.update(ended_utc=_utc(), wall_s=round(wall, 3), peak_rss_bytes=peak, exit_code=code)
                if code == 0:
                    absent = [s for s in st.outputs if _existing(run_dir, s) is None]
                    if absent:
                        code = rec["exit_code"] = 1
                        with open(log_path, "a") as fh:
                            fh.write(f"# scpipe: stage exited 0 but did not write: {', '.join(absent)}\n")
                if code == GATE_WAITING and is_gate(st):
                    # The gate wrote its report and waits for a human (docs/agents-spec.md section 2).
                    rec["status"] = "waiting"
                    rec["note"] = "waiting for human approval"
                    save_state(run_dir, state)
                    say(f"  [{st.id:>4}] WAITING for approval after {_fmt_s(wall)}; "
                        f"report: {run_dir / 'gates' / st.id / 'report.html'}")
                    say(f"scpipe: review the gate, then: scpipe approve --run-dir {run_dir} --stage {st.id} "
                        f"[--from-file edited.json] [--by NAME]\n        and continue with: scpipe resume --run-dir {run_dir}")
                    return GATE_WAITING
                if code != 0:
                    rec["status"] = "failed"
                    save_state(run_dir, state)
                    say(f"  [{st.id:>4}] FAILED (exit code {code}) after {_fmt_s(wall)}. "
                        f"Last {LOG_TAIL} lines of {log_path}:")
                    for line in tail(log_path):
                        say(f"         | {line}")
                    say(f"scpipe: fix the problem, then: scpipe resume --run-dir {run_dir}")
                    return 1
                rec["outputs"] = {
                    spec: file_checksum(p) for spec in st.outputs if (p := _existing(run_dir, spec)) is not None
                }
                rec["status"] = "done"
                rec.pop("note", None)
                save_state(run_dir, state)
                say(f"  [{st.id:>4}] done in {_fmt_s(wall)}, peak RSS {_fmt_bytes(peak)}")
            say(f"scpipe: finished up to {stages[last].id}")
            return 0
    finally:
        for sig, h in prev_handlers.items():
            signal.signal(sig, h)


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------


def status_text(run_dir: str | Path, stages: Sequence[Stage] | None = None) -> str:
    run_dir = Path(run_dir)
    stages = _registry(stages)
    if not (run_dir / STATE_FILE).exists():
        return f"no {STATE_FILE} in {run_dir}: the run has not started"
    state = load_state(run_dir, stages)
    order = state.get("stage_order") or [s.id for s in stages]
    for sid in state["stages"]:
        if sid not in order:
            order.append(sid)
    lines = [f"run_id:  {state.get('run_id')}", f"run_dir: {run_dir}", f"updated: {state.get('updated_utc')}", ""]
    header = ("STAGE", "LANG", "STATUS", "STARTED (UTC)", "WALL", "PEAK RSS", "EXIT")
    rows = []
    for sid in order:
        r = state["stages"].get(sid, {})
        status = r.get("status", "pending")
        if status == "running" and not _runner_alive(run_dir):
            status = "running?"  # runner gone: killed or crashed
        rows.append((sid, r.get("lang", ""), status, (r.get("started_utc") or "-").replace("T", " ").rstrip("Z"),
                     _fmt_s(r.get("wall_s")), _fmt_bytes(r.get("peak_rss_bytes")),
                     "-" if r.get("exit_code") is None else str(r.get("exit_code"))))
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(header)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines.append(fmt.format(*header))
    lines.append(fmt.format(*("-" * w for w in widths)))
    lines += [fmt.format(*row) for row in rows]
    total = sum(r.get("wall_s") or 0 for r in state["stages"].values() if r.get("status") == "done")
    peak = max([r.get("peak_rss_bytes") or 0 for r in state["stages"].values()] or [0])
    lines += ["", f"total wall (done stages): {_fmt_s(total)}; max peak RSS: {_fmt_bytes(peak or None)}"]
    for sid in order:
        r = state["stages"].get(sid, {})
        if r.get("status") in ("failed", "running"):
            lines.append(f"\nlast lines of logs/{sid}.log ({r['status']}):")
            lines += [f"  | {x}" for x in tail(run_dir / "logs" / f"{sid}.log", 10)]
        elif r.get("status") == "waiting":
            lines.append(f"\n{sid} is waiting for a human decision: see {run_dir / 'gates' / sid / 'report.html'}, then\n"
                         f"  scpipe approve --run-dir {run_dir} --stage {sid} [--from-file edited.json] [--by NAME]\n"
                         f"  scpipe resume --run-dir {run_dir}")
    if _runner_alive(run_dir):
        lines.append("\na runner is active in this run directory")
    return "\n".join(lines)


def _runner_alive(run_dir: Path) -> bool:
    p = Path(run_dir) / ".scpipe.lock"
    if not p.exists():
        return False
    with open(p, "a+") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fh, fcntl.LOCK_UN)
    return False


# --------------------------------------------------------------------------
# Doctor
# --------------------------------------------------------------------------

PY_REQUIRED = ("anndata", "scanpy", "h5py", "numpy", "scipy", "pandas", "yaml", "sklearn", "igraph", "leidenalg", "skimage")
PY_OPTIONAL = ("harmonypy", "rapids_singlecell", "cupy", "psutil", "jinja2", "matplotlib", "requests", "celltypist")
R_REQUIRED = ("Seurat", "SeuratObject", "Matrix", "hdf5r", "jsonlite", "digest", "yaml")
R_OPTIONAL = ("harmony", "leidenbase", "uwot", "qs2", "scDblFinder", "glmGamPoi", "BPCells", "presto", "DESeq2", "limma")
_DIST = {"yaml": "PyYAML", "sklearn": "scikit-learn", "skimage": "scikit-image", "igraph": "igraph",
         "rapids_singlecell": "rapids-singlecell", "cupy": "cupy"}


def _py_pkg(name: str) -> str | None:
    import importlib.metadata as md
    import importlib.util

    if importlib.util.find_spec(name) is None:
        return None
    for dist in (_DIST.get(name, name), name, name.replace("_", "-")):
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            continue
    if name == "cupy":
        for dist in ("cupy-cuda12x", "cupy-cuda11x"):
            try:
                return md.version(dist)
            except md.PackageNotFoundError:
                pass
    return "installed"


def _r_packages(pkgs: Sequence[str]) -> tuple[dict[str, str | None], str | None, str | None]:
    """({pkg: version or None}, R version, error)."""
    rs = _rscript()
    if shutil.which(rs) is None and not Path(rs).exists():
        return {p: None for p in pkgs}, None, f"{rs} not found (set SCPIPE_RSCRIPT)"
    expr = (
        "cat('R', as.character(getRversion()), '\\n');"
        f"for (p in c({', '.join(repr(p) for p in pkgs)})) "
        "cat(p, if (nzchar(system.file(package = p))) as.character(utils::packageVersion(p)) else 'NA', '\\n')"
    ).replace("'", '"')
    try:
        r = subprocess.run([rs, "--vanilla", "-e", expr], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {p: None for p in pkgs}, None, f"{rs} failed: {e}"
    if r.returncode != 0:
        return {p: None for p in pkgs}, None, f"{rs} exited {r.returncode}: {r.stderr.strip()[-300:]}"
    found: dict[str, str | None] = {p: None for p in pkgs}
    rver = None
    for line in r.stdout.splitlines():
        bits = line.split()
        if len(bits) == 2:
            if bits[0] == "R":
                rver = bits[1]
            elif bits[0] in found:
                found[bits[0]] = None if bits[1] == "NA" else bits[1]
    return found, rver, None


def _writable(path: Path) -> tuple[bool, str]:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    if not os.access(p, os.W_OK | os.X_OK):
        return False, f"{p} is not writable"
    if p.is_dir():
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(dir=p, prefix=".scpipe-doctor-"):
                pass
        except OSError as e:
            return False, f"cannot create files in {p}: {e}"
    shown = "exists" if p == path else f"will be created under {p}"
    return True, f"{path} ({shown})"


def doctor(config_path: str | None = None, run_dir: str | None = None) -> dict[str, Any]:
    """Preflight checks. Returns a report dict with ``ok`` (required checks all pass)."""
    from . import resources

    report: dict[str, Any] = {"checks": []}

    def add(group: str, name: str, ok: bool, detail: str, required: bool = True) -> None:
        report["checks"].append({"group": group, "name": name, "ok": bool(ok), "required": required, "detail": detail})

    add("python", "python", sys.version_info >= (3, 10), f"{sys.version.split()[0]} ({sys.executable})")
    for p in PY_REQUIRED:
        v = _py_pkg(p)
        add("python", p, v is not None, v or "missing")
    for p in PY_OPTIONAL:
        v = _py_pkg(p)
        add("python", p, v is not None, v or "missing (optional)", required=False)

    rpk, rver, err = _r_packages(R_REQUIRED + R_OPTIONAL)
    add("R", "Rscript", err is None, err or f"R {rver} ({shutil.which(_rscript()) or _rscript()})")
    if err is None:
        for p in R_REQUIRED:
            add("R", p, rpk[p] is not None, rpk[p] or "missing")
        for p in R_OPTIONAL:
            add("R", p, rpk[p] is not None, rpk[p] or "missing (optional)", required=False)
    stage_script = cfgmod.REPO_ROOT / "R" / "stages" / "run_stage.R"
    add("R", "R/stages/run_stage.R", stage_script.exists(), str(stage_script))

    cfg = None
    if config_path:
        try:
            cfg = cfgmod.load_config(config_path, run_dir)
            add("config", "config", True, f"{config_path} is valid")
            for w in cfg.get("_warnings", []):
                add("config", "warning", True, w, required=False)
        except ConfigError as e:
            for p in e.problems:
                add("config", "config", False, p)
    cap = float(get_key(cfg or {}, "resources.gpu_mem_cap_gb", None) or os.environ.get("SCPIPE_GPU_MEM_CAP_GB") or 20)
    gi = resources.gpu_info(cap)
    detail = gi["reason"]
    if gi.get("total_gb") is not None:
        detail += f"; total {gi['total_gb']} GB, free {gi['free_gb']} GB"
    want_gpu = (cfg or {}).get("device") == "gpu"
    add("gpu", "gpu", gi["available"], detail, required=want_gpu)
    report["gpu"] = gi

    for tool in ("nice", "tmux", "nohup"):
        w = shutil.which(tool)
        add("system", tool, w is not None, w or "missing", required=False)
    if not (shutil.which("tmux") or shutil.which("nohup")):
        add("system", "detach", False, "neither tmux nor nohup found: --detach unavailable", required=False)

    target = run_dir or (cfg or {}).get("run_dir")
    if target:
        ok, detail = _writable(Path(target).expanduser().resolve())
        add("storage", "run_dir", ok, detail)
    # Portability and LLM checks (docs/agents-spec.md section 8): site file,
    # execution mode, container, API key + tool-use test, budget, Open Targets.
    try:
        from . import site as sitemod

        report["checks"].extend(sitemod.doctor_checks(cfg))
    except Exception as e:  # never let an extra check break doctor
        add("site", "portability checks", True, f"skipped: {e}", required=False)
    report["ok"] = all(c["ok"] for c in report["checks"] if c["required"])
    return report


def _print_doctor(report: dict[str, Any], out) -> None:
    group = None
    for c in report["checks"]:
        if c["group"] != group:
            group = c["group"]
            print(f"\n{group}:", file=out)
        mark = "ok" if c["ok"] else ("FAIL" if c["required"] else "--")
        print(f"  [{mark:>4}] {c['name']:<22} {c['detail']}", file=out)
    print("\ndoctor: " + ("all required checks passed" if report["ok"] else "some required checks FAILED"), file=out)


# --------------------------------------------------------------------------
# Detach
# --------------------------------------------------------------------------


def detach_command(argv: list[str], run_dir: Path, run_id: str, method: str | None = None,
                   env_file: Path | None = None) -> tuple[str, list[str], str]:
    """Return (method, command to launch, text telling the user how to follow it).

    With tmux the detached runner does not inherit this shell's environment
    (a tmux server may already be running with its own), so the environment is
    handed over in ``env_file`` (mode 0600, read and deleted by the detached
    runner via the hidden ``--env-file`` option) rather than on the command
    line, where other users of a shared node could read it with ``ps``.
    """
    method = method or os.environ.get("SCPIPE_DETACH_WITH") or ("tmux" if shutil.which("tmux") else "nohup")
    log = run_dir / "logs" / "runner.log"
    inner = [sys.executable, "-m", "scpipe", *argv]
    if method == "tmux":
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in run_id)[:60]
        session = f"scpipe-{safe}"
        if env_file is not None:
            inner += ["--env-file", str(env_file)]
        shell = (f"cd {shlex.quote(str(Path.cwd()))} && " + shlex.join(inner)
                 + f" 2>&1 | tee -a {shlex.quote(str(log))}")
        cmd = ["tmux", "new-session", "-d", "-s", session, shell]
        how = (f"scpipe: running detached in tmux session {session!r}\n"
               f"  reattach:  tmux attach -t {session}   (detach again with Ctrl-b d)\n"
               f"  log:       tail -f {log}\n"
               f"  status:    scpipe status --run-dir {run_dir}")
        return method, cmd, how
    if method == "nohup":
        cmd = ["nohup", *inner]
        how = (f"scpipe: running detached with nohup (tmux not found)\n"
               f"  log:       tail -f {log}\n"
               f"  status:    scpipe status --run-dir {run_dir}")
        return method, cmd, how
    raise RunnerError(f"unknown detach method {method!r} (tmux or nohup)")


def _write_env_file(run_dir: Path) -> Path:
    p = run_dir / f".scpipe-env-{os.getpid()}.json"
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(dict(os.environ), fh)
    return p


def _load_env_file(path: str) -> None:
    p = Path(path)
    try:
        env = json.loads(p.read_text())
    finally:
        p.unlink(missing_ok=True)
    os.environ.clear()
    os.environ.update(env)


def _launch_detached(argv: list[str], run_dir: Path, run_id: str, out) -> int:
    if _runner_alive(run_dir):
        raise RunnerError(f"another scpipe runner is active in {run_dir}")
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    method = os.environ.get("SCPIPE_DETACH_WITH") or ("tmux" if shutil.which("tmux") else "nohup")
    env_file = _write_env_file(run_dir) if method == "tmux" else None
    method, cmd, how = detach_command(argv, run_dir, run_id, method, env_file)
    log = run_dir / "logs" / "runner.log"
    if method == "tmux":
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            if env_file is not None:
                env_file.unlink(missing_ok=True)
            raise RunnerError(f"tmux failed: {r.stderr.strip()}")
    else:
        with open(log, "a") as fh:
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
        how += f"\n  pid:       {p.pid}"
    print(how, file=out)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scpipe", description="scpipe pipeline runner (S0-S2)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="validate the config and run the stages")
    r.add_argument("--config", required=True)
    r.add_argument("--run-dir", default=None, help="overrides run_dir in the config")
    r.add_argument("--until", default=None, metavar="STAGE", help="stop after this stage")
    r.add_argument("--detach", action="store_true", help="run inside tmux (or nohup) and return")
    r.add_argument("--env-file", default=None, help=argparse.SUPPRESS)
    s = sub.add_parser("resume", help="continue from the first incomplete stage")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--until", default=None, metavar="STAGE")
    s.add_argument("--detach", action="store_true")
    s.add_argument("--env-file", default=None, help=argparse.SUPPRESS)
    for sp in (r, s):
        sp.add_argument("--execution", default=None, choices=cfgmod.EXECUTION_MODES,
                        help="override execution from the config/site file (direct runs here; slurm/pbs/lsf/sge "
                             "submit one job; split submits the heavy stages and runs the gates here)")
        sp.add_argument("--no-gates", action="store_true", help="record gate stages as skipped instead of running them")
    i = sub.add_parser("init-site", help="detect scheduler, GPU, container runtime and internet; draft a site file")
    i.add_argument("--write", default=None, metavar="PATH")
    i.add_argument("--name", default=None)
    i.add_argument("--no-internet-check", action="store_true")
    i.add_argument("--config", default=None)
    i.add_argument("--json", action="store_true")
    t = sub.add_parser("status", help="stage states, timings and peak memory")
    t.add_argument("--run-dir", required=True)
    t.add_argument("--json", action="store_true", help="print state.json")
    d = sub.add_parser("doctor", help="preflight checks")
    d.add_argument("--config", default=None)
    d.add_argument("--run-dir", default=None)
    d.add_argument("--json", action="store_true")
    ap = sub.add_parser("approve", help="record a human decision at a gate (G1-G4), then `scpipe resume`")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", required=True, metavar="GATE", help="G1, G2, G3 or G4")
    ap.add_argument("--from-file", default=None, metavar="JSON",
                    help="an edited decision (the proposal.json content, or {\"decision\": {...}}); default: the proposal as is")
    ap.add_argument("--by", default=None, metavar="NAME", help="who approves (default: $SCPIPE_APPROVER or the login name)")
    return p


def _submit_mode(mode: str, resolved: dict[str, Any], rd: Path, stages: Sequence[Stage], until: str | None, out) -> int:
    """``execution: slurm|pbs|lsf|sge`` submits one job; ``split`` submits the heavy stages (docs/agents-spec.md 8)."""
    from . import site as sitemod

    if _runner_alive(rd):
        raise RunnerError(f"another scpipe runner is active in {rd}")
    if mode == "split":
        sitemod.submit_split(resolved, rd, [s.id for s in stages], until=until, out=out)
    else:
        sitemod.submit(resolved, rd, mode, until=until, out=out)
    return 0


def main(argv: list[str] | None = None, *, stages: Sequence[Stage] | None = None, out=None) -> int:
    out = out or sys.stdout
    a = _parser().parse_args(argv)
    if getattr(a, "env_file", None):
        _load_env_file(a.env_file)
    try:
        if a.cmd == "run":
            cfg = cfgmod.load_config(a.config, a.run_dir)
            for w in cfg.get("_warnings", []):
                print(f"scpipe: warning: {w}", file=out)
            _stage_index(_registry(stages), a.until)
            rd = Path(cfg["run_dir"])
            if _runner_alive(rd):
                raise RunnerError(f"another scpipe runner is active in {rd}")
            resolved = cfgmod.prepare_run_dir(cfg)
            mode = a.execution or resolved.get("execution") or "direct"
            if mode != "direct":
                return _submit_mode(mode, resolved, rd, _registry(stages), a.until, out)
            if a.detach:
                args = ["run", "--config", str(cfg["config_file"]), "--run-dir", str(rd), "--execution", "direct"]
                if a.until:
                    args += ["--until", a.until]
                if a.no_gates:
                    args += ["--no-gates"]
                return _launch_detached(args, rd, resolved["run_id"], out)
            return run_pipeline(rd, resolved, until=a.until, stages=stages, out=out, no_gates=a.no_gates)
        if a.cmd == "resume":
            rd = Path(a.run_dir).expanduser().resolve()
            resolved = cfgmod.load_resolved(rd)
            _stage_index(_registry(stages), a.until)
            mode = a.execution or "direct"  # resume runs here unless asked to submit again
            if mode != "direct":
                return _submit_mode(mode, resolved, rd, _registry(stages), a.until, out)
            if a.detach:
                args = ["resume", "--run-dir", str(rd), "--execution", "direct"] + (["--until", a.until] if a.until else [])
                if a.no_gates:
                    args += ["--no-gates"]
                return _launch_detached(args, rd, resolved["run_id"], out)
            return run_pipeline(rd, resolved, until=a.until, stages=stages, out=out, no_gates=a.no_gates)
        if a.cmd == "init-site":
            from . import site as sitemod

            args = []
            if a.write:
                args += ["--write", a.write]
            if a.name:
                args += ["--name", a.name]
            if a.no_internet_check:
                args += ["--no-internet-check"]
            if a.config:
                args += ["--config", a.config]
            if a.json:
                args += ["--json"]
            return sitemod.init_site_main(args, out=out)
        if a.cmd == "status":
            rd = Path(a.run_dir).expanduser().resolve()
            if a.json:
                p = rd / STATE_FILE
                print(p.read_text() if p.exists() else "{}", file=out)
                return 0 if p.exists() else 1
            print(status_text(rd, stages), file=out)
            return 0 if (rd / STATE_FILE).exists() else 1
        if a.cmd == "doctor":
            rep = doctor(a.config, a.run_dir)
            if a.json:
                print(json.dumps(rep, indent=2, default=str), file=out)
            else:
                _print_doctor(rep, out)
            return 0 if rep["ok"] else 1
        if a.cmd == "approve":
            from . import gates as gatesmod

            rd = Path(a.run_dir).expanduser().resolve()
            resolved = cfgmod.load_resolved(rd)
            try:
                dec = gatesmod.approve(a.stage, rd, resolved, from_file=a.from_file, by=a.by)
            except gatesmod.GateError as e:
                raise RunnerError(str(e)) from e
            ch = dec.get("changes_vs_proposal") or []
            print(f"scpipe: {dec['gate']} approved by {dec['decided_by']} at {dec['decided_utc']}"
                  + (f" with {len(ch)} change(s) vs the proposal:" if ch else " (proposal unchanged)"), file=out)
            for c in ch:
                print(f"  - {c}", file=out)
            print(f"scpipe: continue with: scpipe resume --run-dir {rd}", file=out)
            return 0
    except ConfigError as e:
        print(f"scpipe: {e}", file=out)
        return 2
    except RunnerError as e:
        print(f"scpipe: error: {e}", file=out)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
