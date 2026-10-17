"""Run configuration: load, fill defaults, validate, and write the run's config folder.

Implements docs/stages-spec.md section 4 (``config.yaml``) and the
``<run-dir>/config/`` part of section 2.

* :func:`load_config` reads ``config.yaml``, ``samples.csv`` and
  ``qc_policy.yaml``, fills in defaults, resolves paths and validates
  everything, raising one :class:`ConfigError` that lists **every** problem.
* :func:`prepare_run_dir` writes ``config.yaml``, ``samples.csv`` and
  ``qc_policy.yaml`` (copies), ``config.resolved.yaml`` and ``hashes.json``.
* :func:`load_resolved` reads ``config.resolved.yaml`` back (what stages use).

Path rules:

* ``run_dir``, ``samples`` and ``qc_policy`` in ``config.yaml``: environment
  variables (``$USER``, ``${VAR}``) and ``~`` are expanded; relative paths are
  resolved against the directory containing ``config.yaml``. A ``--run-dir``
  override is resolved against the current directory.
* ``path`` values in ``samples.csv``: relative paths are also resolved against
  the directory containing ``config.yaml`` (spec section 4: "relative paths
  are resolved against the directory containing config.yaml"), whatever
  directory ``samples.csv`` itself is in; env vars and ``~`` are expanded.
  The absolute results are written to ``samples_resolved`` and that
  directory to ``config_dir`` in ``config.resolved.yaml`` (S0 in R resolves
  the copied sheet's paths against ``config_dir``).
* ``config.resolved.yaml`` points ``samples`` and ``qc_policy`` at the copies
  in ``<run-dir>/config/``; ``samples_source`` / ``qc_policy_source`` hold
  the user's originals. When ``qc_policy`` is not set, the built-in default
  policy (spec section 3.4) is written as the copy.

``config_hash`` = SHA-256 of the canonical JSON (sorted keys) of the resolved
analysis settings (every section, but not ``run_id``, ``run_dir``,
``repo_root`` or the ``*_source`` paths) plus the SHA-256 of the
``samples.csv`` and ``qc_policy.yaml`` contents.
"""

from __future__ import annotations

import copy
import csv
import getpass
import hashlib
import io
import json
import os
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import qc as qcmod

__all__ = [
    "ConfigError",
    "DEFAULTS",
    "SAMPLE_FORMATS",
    "ALLOWED",
    "REPO_ROOT",
    "load_config",
    "prepare_run_dir",
    "load_resolved",
    "get_key",
    "sha256_file",
    "canonical_json",
    "detect_format",
]

REPO_ROOT = Path(os.environ.get("SCPIPE_REPO_ROOT") or Path(__file__).resolve().parents[2])

DEFAULTS: dict[str, Any] = {
    "project": "scpipe",
    "run_dir": None,
    "seed": 42,
    "samples": None,
    "qc_policy": None,
    "object_format": "rds",
    "resources": {"threads": 8, "max_memory_gb": 64, "gpu_mem_cap_gb": 20, "nice": 10},
    "doublets": {"method": "scrublet"},
    "qc": {"review_action": "keep"},
    "normalization": "lognorm",
    "n_hvg": 2000,
    "device": "auto",
    "integration": {"method": "none", "batch_key": "sample_id"},
    "pcs": "auto",
    "n_neighbors": 20,
    "clustering": {"resolution": 0.5},
}

ALLOWED = {
    "object_format": ("rds", "qs2"),
    "doublets.method": ("scrublet", "scdblfinder", "none"),
    "qc.review_action": ("keep", "exclude"),
    "normalization": ("lognorm", "sct"),
    "device": ("auto", "cpu", "gpu"),
    "integration.method": ("none", "harmony"),
}

SAMPLE_FORMATS = ("10x_h5", "10x_mtx", "h5ad", "seurat", "loom", "table")

# Keys the runner adds to config.resolved.yaml (never read from the user's file).
_DERIVED_KEYS = (
    "run_id",
    "config_hash",
    "repo_root",
    "config_file",
    "config_dir",
    "samples_source",
    "qc_policy_source",
    "samples_resolved",
    "sample_columns",
    "policy_hash",
    "scpipe_version",
)
_HASH_EXCLUDE = ("run_id", "config_hash", "run_dir", "repo_root", "config_file", "config_dir", "samples_source",
                 "qc_policy_source", "samples", "qc_policy", "scpipe_version")


class ConfigError(ValueError):
    """The configuration is invalid. ``problems`` lists every problem found."""

    def __init__(self, problems: list[str] | str):
        if isinstance(problems, str):
            problems = [problems]
        self.problems = list(problems)
        super().__init__("invalid configuration:\n  - " + "\n  - ".join(self.problems))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_key(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """``get_key(cfg, "clustering.resolution")``; missing keys give ``default``."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


_ENV_RE = re.compile(r"\$(\w+)|\$\{([^}]+)\}")


def _expand(value: str, problems: list[str], what: str) -> str:
    env = dict(os.environ)
    if "USER" not in env:
        try:
            env["USER"] = getpass.getuser()
        except Exception:
            pass

    def sub(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        if name not in env:
            problems.append(f"{what}: environment variable ${name} is not set (in {value!r})")
            return m.group(0)
        return env[name]

    return os.path.expanduser(_ENV_RE.sub(sub, value))


def _resolve_path(value: str, base: Path, problems: list[str], what: str) -> Path:
    p = Path(_expand(value, problems, what))
    if not p.is_absolute():
        p = base / p
    return Path(os.path.normpath(p))


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def detect_format(path: Path) -> str | None:
    """Best-effort format detection (spec 3.1). ``None`` if unrecognised.

    S0 (R) is authoritative; this only lets the runner validate the sheet early.
    """
    name = path.name.lower()
    if path.is_dir():
        if any((path / f).exists() for f in ("matrix.mtx", "matrix.mtx.gz")):
            return "10x_mtx"
        return None
    if name.endswith(".h5ad"):
        return "h5ad"
    if name.endswith(".h5"):
        try:
            import h5py

            with h5py.File(path, "r") as f:
                return "10x_h5" if "matrix" in f else None
        except Exception:
            return None
    if name.endswith((".rds", ".qs2")):
        return "seurat"
    if name.endswith(".loom"):
        return "loom"
    if re.search(r"\.(csv|tsv|txt)(\.gz)?$", name):
        return "table"
    return None


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------


def _merge_defaults(raw: Mapping[str, Any], problems: list[str], warnings: list[str]) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in raw.items():
        if k in _DERIVED_KEYS:
            warnings.append(f"key {k!r} is set by the runner; the value in config.yaml is ignored")
            continue
        if k not in DEFAULTS:
            # Later build steps add sections (llm, gates, dge, ...); keep them, but say so.
            warnings.append(f"unknown top-level key {k!r} (kept as is, not used by S0-S2)")
            cfg[k] = v
            continue
        if isinstance(DEFAULTS[k], dict):
            if v is None:
                continue
            if not isinstance(v, Mapping):
                problems.append(f"{k} must be a mapping, got {type(v).__name__}")
                continue
            for kk, vv in v.items():
                if kk not in DEFAULTS[k]:
                    warnings.append(f"unknown key {k}.{kk} (kept as is, not used by S0-S2)")
                cfg[k][kk] = vv
        else:
            cfg[k] = v
    return cfg


def _check_values(cfg: dict[str, Any], problems: list[str]) -> None:
    def need(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    need(isinstance(cfg["project"], str) and bool(cfg["project"].strip()), f"project must be a non-empty string, got {cfg['project']!r}")
    need(_is_int(cfg["seed"]) and cfg["seed"] >= 0, f"seed must be a non-negative integer, got {cfg['seed']!r}")
    for key, allowed in ALLOWED.items():
        v = get_key(cfg, key)
        need(isinstance(v, str) and v in allowed, f"{key} must be one of {list(allowed)}, got {v!r}")
    r = cfg["resources"]
    need(_is_int(r["threads"]) and r["threads"] >= 1, f"resources.threads must be an integer >= 1, got {r['threads']!r}")
    for k in ("max_memory_gb", "gpu_mem_cap_gb"):
        need(_is_num(r[k]) and r[k] > 0, f"resources.{k} must be a positive number, got {r[k]!r}")
    need(_is_int(r["nice"]) and 0 <= r["nice"] <= 19, f"resources.nice must be an integer from 0 to 19, got {r['nice']!r}")
    need(_is_int(cfg["n_hvg"]) and cfg["n_hvg"] >= 10, f"n_hvg must be an integer >= 10, got {cfg['n_hvg']!r}")
    pcs = cfg["pcs"]
    need(pcs == "auto" or (_is_int(pcs) and 2 <= pcs <= 200), f"pcs must be 'auto' or an integer from 2 to 200, got {pcs!r}")
    need(_is_int(cfg["n_neighbors"]) and cfg["n_neighbors"] >= 2, f"n_neighbors must be an integer >= 2, got {cfg['n_neighbors']!r}")
    res = cfg["clustering"]["resolution"]
    need(_is_num(res) and res > 0, f"clustering.resolution must be a positive number, got {res!r}")
    bk = cfg["integration"]["batch_key"]
    need(isinstance(bk, str) and bool(bk.strip()), f"integration.batch_key must be a non-empty string, got {bk!r}")
    if cfg["normalization"] == "sct" and cfg["device"] == "gpu":
        problems.append("normalization: sct runs on CPU only; use device: cpu or auto with sct")


def _read_samples(path: Path, problems: list[str], warnings: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        problems.append(f"samples: cannot read {path}: {e}")
        return [], []
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        problems.append(f"samples: {path} is empty")
        return [], []
    header = [h.strip() for h in rows[0]]
    for col in ("sample_id", "path"):
        if col not in header:
            problems.append(f"samples: {path.name} lacks required column {col!r} (has {header})")
    dup_cols = sorted({h for h in header if header.count(h) > 1})
    if dup_cols:
        problems.append(f"samples: duplicated column name(s) {dup_cols}")
    if any(not h for h in header):
        problems.append("samples: empty column name in header")
    records = []
    for i, r in enumerate(rows[1:], start=2):
        if len(r) != len(header):
            problems.append(f"samples: line {i} has {len(r)} fields, header has {len(header)}")
            continue
        records.append({h: v.strip() for h, v in zip(header, r)})
    if not records:
        problems.append(f"samples: {path.name} has no sample rows")
    return header, records


def _check_samples(header, records, base: Path, problems: list[str], warnings: list[str]) -> list[dict[str, str]]:
    if "sample_id" not in header or "path" not in header:
        return []
    out = []
    seen: dict[str, int] = {}
    for i, rec in enumerate(records, start=2):
        sid = rec["sample_id"]
        if not sid:
            problems.append(f"samples: line {i}: sample_id is empty")
            continue
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", sid):
            problems.append(
                f"samples: line {i}: sample_id {sid!r} may only contain letters, digits, '.', '_' and '-' "
                "(it prefixes cell barcodes)"
            )
        if sid in seen:
            problems.append(f"samples: sample_id {sid!r} appears more than once (lines {seen[sid]} and {i})")
            continue
        seen[sid] = i
        raw_path = rec["path"]
        if not raw_path:
            problems.append(f"samples: line {i} ({sid}): path is empty")
            continue
        p = _resolve_path(raw_path, base, problems, f"samples line {i} path")
        fmt = rec.get("format", "") or ""
        if fmt and fmt not in SAMPLE_FORMATS:
            problems.append(f"samples: line {i} ({sid}): format {fmt!r} is not one of {list(SAMPLE_FORMATS)}")
        detected = None
        if not p.exists():
            problems.append(f"samples: line {i} ({sid}): path does not exist: {p}")
        else:
            detected = detect_format(p)
            if not fmt and detected is None:
                problems.append(
                    f"samples: line {i} ({sid}): cannot detect the format of {p}; set the format column "
                    f"to one of {list(SAMPLE_FORMATS)}"
                )
            elif fmt and detected and fmt != detected:
                warnings.append(f"samples: line {i} ({sid}): format {fmt!r} given, but the path looks like {detected!r}")
        out.append({"sample_id": sid, "path": str(p), "format": fmt, "format_detected": detected or ""})
    return out


def load_config(config_path: str | Path, run_dir: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a run configuration; returns the resolved config dict.

    Raises :class:`ConfigError` listing every problem. Non-fatal remarks are
    returned under ``cfg["_warnings"]`` (not written to the resolved file).
    """
    problems: list[str] = []
    warnings: list[str] = []
    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config_path = Path(os.path.normpath(config_path))
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"cannot read config file {config_path}: {e}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"config file {config_path} is not valid YAML: {e}") from e
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"config file {config_path} must hold a mapping, got {type(raw).__name__}")
    base = config_path.parent
    cfg = _merge_defaults(raw, problems, warnings)
    _check_values(cfg, problems)

    # run_dir
    if run_dir is not None:
        rd = _resolve_path(str(run_dir), Path.cwd(), problems, "--run-dir")
    elif isinstance(cfg["run_dir"], str) and cfg["run_dir"].strip():
        rd = _resolve_path(cfg["run_dir"], base, problems, "run_dir")
    else:
        rd = None
        problems.append("run_dir is not set (set it in config.yaml or pass --run-dir)")
    cfg["run_dir"] = str(rd) if rd is not None else None

    # samples
    samples_path = None
    if not isinstance(cfg["samples"], str) or not cfg["samples"].strip():
        problems.append("samples is not set (path to samples.csv)")
    else:
        samples_path = _resolve_path(cfg["samples"], base, problems, "samples")
        if not samples_path.is_file():
            problems.append(f"samples: file not found: {samples_path}")
            samples_path = None
    header: list[str] = []
    resolved_samples: list[dict[str, str]] = []
    if samples_path is not None:
        header, records = _read_samples(samples_path, problems, warnings)
        resolved_samples = _check_samples(header, records, base, problems, warnings)
    bk = cfg["integration"]["batch_key"]
    if cfg["integration"]["method"] == "harmony" and header and isinstance(bk, str) and bk not in header:
        problems.append(
            f"integration.batch_key {bk!r} is not a samples.csv column (columns: {header})"
        )
    if cfg["integration"]["method"] == "harmony" and resolved_samples and len(resolved_samples) < 2:
        warnings.append("integration.method is harmony but there is only one sample")

    # qc_policy
    policy_path = None
    policy = None
    if cfg["qc_policy"] is not None:
        if not isinstance(cfg["qc_policy"], str) or not cfg["qc_policy"].strip():
            problems.append(f"qc_policy must be a path or null, got {cfg['qc_policy']!r}")
        else:
            policy_path = _resolve_path(cfg["qc_policy"], base, problems, "qc_policy")
            if not policy_path.is_file():
                problems.append(f"qc_policy: file not found: {policy_path}")
                policy_path = None
    try:
        policy = qcmod.load_policy(policy_path) if policy_path else qcmod.policy_from_dict(None)
    except qcmod.QCError as e:
        problems.extend(f"qc_policy: {p}" for p in e.problems)

    if problems:
        raise ConfigError(problems)

    from . import __version__

    cfg["config_file"] = str(config_path)
    cfg["config_dir"] = str(base)
    cfg["samples_source"] = str(samples_path)
    cfg["qc_policy_source"] = str(policy_path) if policy_path else "builtin-default"
    cfg["samples"] = str(Path(rd) / "config" / "samples.csv")
    cfg["qc_policy"] = str(Path(rd) / "config" / "qc_policy.yaml")
    cfg["samples_resolved"] = resolved_samples
    cfg["sample_columns"] = header
    cfg["policy_hash"] = policy.hash
    cfg["repo_root"] = str(REPO_ROOT)
    cfg["scpipe_version"] = __version__
    cfg["_warnings"] = warnings
    cfg["_policy"] = policy
    return cfg


def _policy_yaml(policy: qcmod.Policy) -> str:
    return "# Built-in default QC policy (docs/stages-spec.md section 3.4)\n" + yaml.safe_dump(
        policy.to_dict(), sort_keys=False
    )


def _copy(src: Path, dst: Path) -> None:
    if dst.exists() and src.resolve() == dst.resolve():
        return
    shutil.copyfile(src, dst)


def _new_run_id(project: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", project).strip("-") or "run"
    return f"{slug}-{stamp}-{secrets.token_hex(3)}"


def config_hash(cfg: Mapping[str, Any], samples_sha: str, policy_sha: str) -> str:
    body = {k: v for k, v in cfg.items() if not k.startswith("_") and k not in _HASH_EXCLUDE}
    body["samples_sha256"] = samples_sha
    body["qc_policy_sha256"] = policy_sha
    return sha256_bytes(canonical_json(body).encode("utf-8"))


def prepare_run_dir(cfg: dict[str, Any]) -> dict[str, Any]:
    """Create ``<run-dir>/config/`` and write the copies, resolved config and hashes.

    ``run_id`` is kept from an existing ``config.resolved.yaml`` in the same
    run directory (so it is stable across reruns), otherwise newly generated.
    Returns the resolved config as written (without ``_``-prefixed keys).
    """
    rd = Path(cfg["run_dir"])
    cdir = rd / "config"
    try:
        cdir.mkdir(parents=True, exist_ok=True)
        (rd / "logs").mkdir(exist_ok=True)
    except OSError as e:
        raise ConfigError(f"cannot create run directory {rd}: {e}") from e

    run_id = None
    prev = cdir / "config.resolved.yaml"
    if prev.exists():
        try:
            run_id = (yaml.safe_load(prev.read_text()) or {}).get("run_id")
        except Exception:
            run_id = None
    run_id = run_id or _new_run_id(cfg["project"])

    _copy(Path(cfg["config_file"]), cdir / "config.yaml")
    _copy(Path(cfg["samples_source"]), cdir / "samples.csv")
    if cfg["qc_policy_source"] == "builtin-default":
        (cdir / "qc_policy.yaml").write_text(_policy_yaml(cfg["_policy"]), encoding="utf-8")
    else:
        _copy(Path(cfg["qc_policy_source"]), cdir / "qc_policy.yaml")

    samples_sha = sha256_file(cdir / "samples.csv")
    policy_sha = sha256_file(cdir / "qc_policy.yaml")
    out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    out["run_id"] = run_id
    out["config_hash"] = config_hash(out, samples_sha, policy_sha)

    header = (
        "# Written by scpipe: defaults filled in, paths absolute, validated.\n"
        "# Stages read this file, never the user's config.yaml.\n"
    )
    tmp = cdir / "config.resolved.yaml.tmp"
    tmp.write_text(header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(tmp, cdir / "config.resolved.yaml")

    hashes = {
        name: sha256_file(cdir / name)
        for name in ("config.yaml", "samples.csv", "qc_policy.yaml", "config.resolved.yaml")
    }
    hashes["config_hash"] = out["config_hash"]
    hashes["policy_hash"] = out["policy_hash"]
    (cdir / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return out


def load_resolved(run_dir: str | Path) -> dict[str, Any]:
    """Read ``<run-dir>/config/config.resolved.yaml``."""
    p = Path(run_dir) / "config" / "config.resolved.yaml"
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"{p} not found or unreadable ({e}); run 'scpipe run --config ...' first") from e
    if not isinstance(cfg, dict) or "run_id" not in cfg:
        raise ConfigError(f"{p} is not a resolved scpipe config")
    return cfg
