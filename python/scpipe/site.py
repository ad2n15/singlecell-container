"""Portability: site detection, site files, scheduler submission, doctor extras.

docs/agents-spec.md section 8 and docs/DESIGN.md section 7.

* :func:`detect_site` looks at the current node: scheduler (``sbatch`` /
  ``qsub`` / ``bsub``), GPU (``nvidia-smi``), container runtime (``apptainer``
  / ``singularity`` / ``docker``), CPU count and RAM, and internet access
  (HTTPS to the LLM endpoint and Open Targets). Nothing detected is fine: the
  draft then holds workstation defaults.
* :func:`draft_site` turns the detection into the site YAML of DESIGN 7.2;
  ``scpipe init-site [--write conf/sites/<name>.yaml]`` prints or writes it.
* :func:`render_batch_script` fills ``conf/templates/<scheduler>.sh.j2`` and
  :func:`submit` writes ``<run-dir>/submit/<scheduler>.sh`` and calls
  ``sbatch`` / ``qsub`` / ``bsub``. The job runs ``scpipe resume --run-dir D
  --execution direct``, so the batch job and the login node share one
  ``state.json`` and resume from wherever the other stopped.
* ``execution: split`` (:func:`submit_split`) submits the heavy stages up to
  the last non-gate stage before S5 as one job with ``--no-gates``; gates,
  Open Targets and the report then run on the submitting node with
  ``scpipe resume`` once the job has finished.
* :func:`doctor_checks` adds the portability and LLM checks that
  ``scpipe doctor`` prints.

Tests use a fake ``sbatch`` on ``PATH`` (``SCPIPE_SCHEDULER_PATH`` is not
needed: :func:`submit` looks the submit command up with ``shutil.which``).
Network checks honour ``SCPIPE_DOCTOR_OFFLINE=1`` (skip) and use short
timeouts.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import config as cfgmod

__all__ = [
    "SCHEDULERS",
    "detect_site",
    "draft_site",
    "init_site_main",
    "render_batch_script",
    "submit",
    "submit_split",
    "doctor_checks",
    "load_site_file",
    "validate_site_file",
]

# scheduler -> (submit command, job-id regex on stdout, how the script is passed)
SCHEDULERS: dict[str, dict[str, Any]] = {
    "slurm": {"submit": "sbatch", "probe": ("sbatch", "squeue"), "jobid": r"Submitted batch job (\d+)", "stdin": False},
    "pbs": {"submit": "qsub", "probe": ("qsub", "qstat"), "jobid": r"^(\S+)", "stdin": False},
    "lsf": {"submit": "bsub", "probe": ("bsub", "bjobs"), "jobid": r"Job <(\d+)>", "stdin": True},
    "sge": {"submit": "qsub", "probe": ("qsub", "qstat"), "jobid": r"job (\d+)", "stdin": False},
}

LLM_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com",
    "openai_compatible": None,  # base_url from config
}
OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
DEFAULT_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai_compatible": "OPENAI_API_KEY"}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def detect_scheduler() -> str | None:
    """``slurm`` / ``pbs`` / ``lsf`` / ``sge`` / None, from the commands on PATH."""
    if shutil.which("sbatch"):
        return "slurm"
    if shutil.which("bsub"):
        return "lsf"
    if shutil.which("qsub"):
        # PBS and SGE both have qsub; SGE ships qconf, PBS ships pbsnodes.
        if shutil.which("qconf") and not shutil.which("pbsnodes"):
            return "sge"
        return "pbs"
    return None


def detect_gpu() -> dict[str, Any]:
    """``{"present": bool, "name": str|None, "total_gb": float|None, "driver": str|None}`` via nvidia-smi."""
    out: dict[str, Any] = {"present": False, "name": None, "total_gb": None, "driver": None, "count": 0}
    smi = shutil.which("nvidia-smi")
    if not smi:
        return out
    try:
        r = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return out
    rows = [ln.split(",") for ln in r.stdout.splitlines() if ln.strip()]
    if r.returncode != 0 or not rows:
        return out
    name, mem, drv = (x.strip() for x in rows[0][:3])
    out.update(present=True, name=name, driver=drv, count=len(rows))
    try:
        out["total_gb"] = round(float(mem) / 1024, 1)
    except ValueError:
        pass
    return out


def detect_container_runtime() -> str:
    for rt in ("apptainer", "singularity", "docker"):
        if shutil.which(rt):
            return rt
    return "none"


def detect_memory_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * size / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def https_reachable(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """HEAD/GET ``url``; any HTTP response (even 4xx) means the host is reachable."""
    if os.environ.get("SCPIPE_DOCTOR_OFFLINE") == "1":
        return False, "skipped (SCPIPE_DOCTOR_OFFLINE=1)"
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "scpipe-doctor"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return True, f"HTTP {e.code} (host reachable)"
    except Exception as e:  # URLError, socket.timeout, ssl errors
        return False, f"unreachable: {str(e)[:120]}"


def detect_site(check_internet: bool = True, cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What this node offers. Pure detection; never raises."""
    llm_cfg = (_llm_section(cfg or {}) or {}).get("default") or {}
    provider = llm_cfg.get("provider") or "anthropic"
    llm_url = llm_cfg.get("base_url") or LLM_ENDPOINTS.get(provider) or LLM_ENDPOINTS["anthropic"]
    d: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "cpus": os.cpu_count() or 1,
        "memory_gb": detect_memory_gb(),
        "scheduler": detect_scheduler(),
        "gpu": detect_gpu(),
        "container_runtime": detect_container_runtime(),
        "internet": {"llm": None, "opentargets": None, "checked": bool(check_internet)},
        "scratch_candidates": [p for p in (os.environ.get("SCRATCH"), "/scratch", "/tmp") if p and Path(p).is_dir()],
        "detected_utc": _utc(),
    }
    if check_internet:
        ok, note = https_reachable(llm_url)
        d["internet"]["llm"] = {"ok": ok, "url": llm_url, "detail": note}
        ok, note = https_reachable(OPENTARGETS_URL)
        d["internet"]["opentargets"] = {"ok": ok, "url": OPENTARGETS_URL, "detail": note}
    return d


# --------------------------------------------------------------------------
# Site YAML
# --------------------------------------------------------------------------


def draft_site(det: Mapping[str, Any], name: str | None = None) -> dict[str, Any]:
    """Site YAML (DESIGN 7.2) from a detection. Workstation defaults when nothing is detected."""
    cpus = int(det.get("cpus") or 1)
    mem = det.get("memory_gb")
    gpu = det.get("gpu") or {}
    sched = det.get("scheduler")
    internet = det.get("internet") or {}
    llm_ok = (internet.get("llm") or {}).get("ok")
    name = name or ("hpc" if sched else "workstation")
    # Leave half the node for others when it looks shared (a scheduler is present).
    shared = sched is not None
    threads = max(1, min(cpus, 8) if shared else cpus)
    mem_cap = 64 if mem is None else (min(64, max(4, int(mem * 0.5))) if shared else max(4, int(mem * 0.8)))
    gpu_total = gpu.get("total_gb")
    gpu_cap = 20 if gpu_total is None else max(1, int(gpu_total * 0.85))
    scratch = (det.get("scratch_candidates") or ["~/scpipe"])[0]
    if scratch == "/scratch":
        scratch = "/scratch/$USER/scpipe"
    elif scratch == "/tmp":
        scratch = "~/scpipe"
    execution = "direct"
    if sched and internet.get("checked") and llm_ok is False:
        execution = "split"  # compute nodes may have internet; the login node does not reach the LLM
    site = {
        "site": name,
        "execution": execution,
        "resources": {"threads": threads, "max_memory_gb": mem_cap, "gpu_mem_cap_gb": gpu_cap, "nice": 10 if shared else 0},
        "scratch": scratch,
        "container": {"runtime": det.get("container_runtime") or "none", "image": None,
                      "options": ["--nv"] if det.get("container_runtime") in ("apptainer", "singularity") and gpu.get("present") else []},
        "scheduler": {"partition": None, "account": None, "queue": None, "time": "24:00:00",
                      "cpus": threads if sched else None, "mem_gb": mem_cap if sched else None,
                      "gpus": 1 if gpu.get("present") else 0, "extra": []},
    }
    return site


def site_yaml_text(site: Mapping[str, Any], det: Mapping[str, Any] | None = None) -> str:
    lines = ["# scpipe site file drafted by `scpipe init-site` (docs/DESIGN.md 7.2)."]
    if det:
        gpu = det.get("gpu") or {}
        lines += [
            f"# host {det.get('hostname')} ({det.get('os')}), {det.get('cpus')} CPUs, "
            f"{det.get('memory_gb') or '?'} GB RAM, scheduler {det.get('scheduler') or 'none'}, "
            f"container {det.get('container_runtime')}",
            f"# GPU: {gpu.get('name') or 'none'}" + (f" ({gpu.get('total_gb')} GB, driver {gpu.get('driver')})" if gpu.get("present") else ""),
        ]
        inet = det.get("internet") or {}
        if inet.get("checked"):
            for k in ("llm", "opentargets"):
                v = inet.get(k) or {}
                lines.append(f"# internet {k}: {'ok' if v.get('ok') else 'NO'} ({v.get('detail')})")
        else:
            lines.append("# internet: not checked")
    lines += ["# Review the values, set scheduler.partition/account if you submit jobs, then",
              "# point config.yaml at this file with `site: <name>` or `site: path/to/file.yaml`.", ""]
    return "\n".join(lines) + yaml.safe_dump(dict(site), sort_keys=False, allow_unicode=True)


def load_site_file(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise cfgmod.ConfigError(f"site file {path} must hold a mapping")
    return data


def validate_site_file(path: str | Path) -> list[str]:
    """Problems with a site file (empty list = valid)."""
    problems: list[str] = []
    try:
        data = load_site_file(path)
    except (OSError, yaml.YAMLError, cfgmod.ConfigError) as e:
        return [str(e)]
    for k in data:
        if k not in cfgmod.SITE_SECTIONS and k != "site":
            problems.append(f"unknown site key {k!r} (allowed: site, {', '.join(cfgmod.SITE_SECTIONS)})")
    ex = data.get("execution", "direct")
    if ex not in cfgmod.EXECUTION_MODES:
        problems.append(f"execution must be one of {list(cfgmod.EXECUTION_MODES)}, got {ex!r}")
    for sec in ("resources", "container", "scheduler"):
        v = data.get(sec)
        if v is not None and not isinstance(v, dict):
            problems.append(f"{sec} must be a mapping")
        elif isinstance(v, dict):
            for kk in v:
                if kk not in cfgmod.DEFAULTS[sec]:
                    problems.append(f"unknown key {sec}.{kk}")
    return problems


def init_site_main(args: list[str] | None = None, out=None) -> int:
    import argparse

    out = out or sys.stdout
    p = argparse.ArgumentParser(prog="scpipe init-site", description="detect this node and draft a site file")
    p.add_argument("--write", default=None, metavar="PATH", help="write the YAML here (default: print)")
    p.add_argument("--name", default=None, help="site name (default: hpc or workstation)")
    p.add_argument("--no-internet-check", action="store_true", help="skip the HTTPS checks")
    p.add_argument("--config", default=None, help="config.yaml (for the LLM provider endpoint)")
    p.add_argument("--json", action="store_true", help="print the detection as JSON instead")
    a = p.parse_args(args)
    cfg = None
    if a.config:
        try:
            cfg = yaml.safe_load(Path(a.config).read_text()) or {}
        except Exception:
            cfg = None
    det = detect_site(check_internet=not a.no_internet_check, cfg=cfg)
    if a.json:
        print(json.dumps(det, indent=2), file=out)
        return 0
    site = draft_site(det, a.name)
    text = site_yaml_text(site, det)
    if a.write:
        dest = Path(a.write).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        print(f"scpipe: site file written to {dest}\n  use it with:  site: {dest}  in config.yaml", file=out)
    else:
        print(text, file=out)
    return 0


# --------------------------------------------------------------------------
# Scheduler submission
# --------------------------------------------------------------------------


def _template_env():
    import jinja2

    tdir = cfgmod.REPO_ROOT / "conf" / "templates"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(tdir)), keep_trailing_newline=True,
                             undefined=jinja2.StrictUndefined, autoescape=False)
    env.filters["sh"] = shlex.quote
    return env


def _hhmm(t: str) -> str:
    """``D-HH:MM:SS`` / ``HH:MM:SS`` -> ``HH:MM`` for LSF."""
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    bits = t.split(":")
    h = int(bits[0]) + 24 * days
    m = int(bits[1]) if len(bits) > 1 else 0
    return f"{h}:{m:02d}"


def container_prefix(cfg: Mapping[str, Any], run_dir: Path) -> str:
    """Command prefix that runs the stage inside the configured container image ('' when none)."""
    c = cfg.get("container") or {}
    rt, image = c.get("runtime", "none"), c.get("image")
    if rt == "none" or not image:
        return ""
    opts = " ".join(shlex.quote(o) for o in c.get("options") or [])
    repo = shlex.quote(str(cfg.get("repo_root") or cfgmod.REPO_ROOT))
    rd = shlex.quote(str(run_dir))
    if rt in ("apptainer", "singularity"):
        return f"{rt} exec {opts} --bind {repo},{rd} {shlex.quote(str(image))}"
    if rt == "docker":
        return f"docker run --rm {opts} -v {repo}:{repo} -v {rd}:{rd} -w {repo} {shlex.quote(str(image))}"
    return ""


def render_batch_script(cfg: Mapping[str, Any], run_dir: Path, scheduler: str, *, until: str | None = None,
                        no_gates: bool = False) -> str:
    if scheduler not in SCHEDULERS:
        raise cfgmod.ConfigError(f"unknown scheduler {scheduler!r} (one of {list(SCHEDULERS)})")
    s = dict(cfgmod.DEFAULTS["scheduler"])
    s.update(cfg.get("scheduler") or {})
    res = cfg.get("resources") or {}
    run_dir = Path(run_dir)
    job_name = "scpipe-" + re.sub(r"[^A-Za-z0-9_.-]+", "-", str(cfg.get("project") or "run"))[:40]
    log_path = run_dir / "logs" / f"{scheduler}-%j.log" if scheduler == "slurm" else run_dir / "logs" / f"{scheduler}.log"
    # Pass-through environment, by name only: secrets stay in the scheduler's environment propagation.
    env: dict[str, str] = {"SCPIPE_REPO_ROOT": str(cfg.get("repo_root") or cfgmod.REPO_ROOT)}
    for k in ("SCPIPE_RSCRIPT", "SCPIPE_ALLOW_CPU_AS_GPU", "SCPIPE_DOCTOR_OFFLINE"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    ctx = {
        "job_name": job_name,
        "log_path": str(log_path),
        "time": s.get("time") or "24:00:00",
        "time_hhmm": _hhmm(s.get("time") or "24:00:00"),
        "partition": s.get("partition"),
        "queue": s.get("queue"),
        "account": s.get("account"),
        "cpus": s.get("cpus") or res.get("threads"),
        "mem_gb": s.get("mem_gb") or res.get("max_memory_gb"),
        "gpus": s.get("gpus") or 0,
        "extra": list(s.get("extra") or []),
        "written_utc": _utc(),
        "run_id": cfg.get("run_id"),
        "repo_root": str(cfg.get("repo_root") or cfgmod.REPO_ROOT),
        "pythonpath": str(Path(cfg.get("repo_root") or cfgmod.REPO_ROOT) / "python"),
        "env": env,
        "python": sys.executable,
        "run_dir": str(run_dir),
        "until": until,
        "no_gates": no_gates,
        "container_cmd": container_prefix(cfg, run_dir),
    }
    return _template_env().get_template(f"{scheduler}.sh.j2").render(**ctx)


def submit(cfg: Mapping[str, Any], run_dir: Path, scheduler: str, *, until: str | None = None,
           no_gates: bool = False, out=None) -> dict[str, Any]:
    """Write ``<run-dir>/submit/<scheduler>.sh`` and submit it. Returns ``{job_id, script, command, stdout}``."""
    out = out or sys.stdout
    run_dir = Path(run_dir)
    spec = SCHEDULERS[scheduler]
    cmd = shutil.which(spec["submit"])
    if cmd is None:
        raise cfgmod.ConfigError(f"execution: {scheduler} but {spec['submit']!r} is not on PATH "
                                 f"(run `scpipe init-site` on a node that has it, or use execution: direct)")
    sdir = run_dir / "submit"
    sdir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    script = sdir / f"{scheduler}.sh"
    script.write_text(render_batch_script(cfg, run_dir, scheduler, until=until, no_gates=no_gates), encoding="utf-8")
    script.chmod(0o750)
    if spec["stdin"]:
        argv = [cmd]
        r = subprocess.run(argv, stdin=open(script), capture_output=True, text=True)
    else:
        argv = [cmd, str(script)]
        r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        raise cfgmod.ConfigError(f"{spec['submit']} failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")
    m = re.search(spec["jobid"], r.stdout.strip(), re.M)
    job_id = m.group(1) if m else r.stdout.strip().split()[0] if r.stdout.strip() else None
    rec = {"scheduler": scheduler, "job_id": job_id, "script": str(script), "command": argv,
           "stdout": r.stdout.strip(), "submitted_utc": _utc(), "until": until, "no_gates": no_gates,
           "mode": "split" if no_gates else scheduler}
    (sdir / "last_job.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"scpipe: submitted {scheduler} job {job_id or '?'} ({script})\n"
          f"  follow:   scpipe status --run-dir {run_dir}\n"
          f"  job log:  {run_dir / 'logs'}/{scheduler}*.log", file=out)
    return rec


def heavy_until(stage_ids: list[str]) -> str | None:
    """Last non-gate stage before S5 (the stages a split job runs)."""
    last = None
    for sid in stage_ids:
        if sid.upper() == "S5":
            break
        if not sid.upper().startswith("G"):
            last = sid
    return last


def submit_split(cfg: Mapping[str, Any], run_dir: Path, stage_ids: list[str], *, until: str | None = None,
                 out=None) -> dict[str, Any]:
    """``execution: split``: one offline job for the heavy stages, gates and S5 on this node.

    The job runs ``scpipe resume --until <last heavy stage> --no-gates``. When it
    has finished, ``scpipe resume --run-dir D`` here runs the gate stages (LLM,
    Open Targets) and S5. TODO(coordinator): once G1-G4 are registered, decide
    whether a gate placed before a heavy stage (G1 before S2) should instead
    split the job in two; with ``--no-gates`` S2 uses the rules baseline
    (``qc.review_action``) as it does today.
    """
    out = out or sys.stdout
    sched = detect_scheduler()
    if sched is None:
        raise cfgmod.ConfigError("execution: split needs a scheduler (sbatch/qsub/bsub) on this node")
    stop = heavy_until(stage_ids)
    if until is not None and until.upper() != "S5":
        stop = until
    rec = submit(cfg, run_dir, sched, until=stop, no_gates=True, out=out)
    print(f"  then, on this node (gates, Open Targets, report):  scpipe resume --run-dir {run_dir}", file=out)
    return rec


# --------------------------------------------------------------------------
# Doctor extras
# --------------------------------------------------------------------------


def _llm_section(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    """``llm:`` normalised to the DESIGN 4.8 shape (``default``, ``roles``, ``budget``, ``replay``), or None when unset.

    The older flat shape (``llm.provider`` / ``llm.model`` / ``llm.api_key_env`` /
    ``llm.base_url``) is folded into ``default``. Budget defaults come from
    ``config.LLM_DEFAULTS`` when that exists. ``api_key_env`` is left unset
    unless the user set it, so the doctor can pick the provider's own default.
    """
    raw = cfg.get("llm")
    if not raw or not isinstance(raw, Mapping):
        return None
    defaults = getattr(cfgmod, "LLM_DEFAULTS", {}) or {}
    out: dict[str, Any] = {"default": dict(raw.get("default") or {}), "roles": dict(raw.get("roles") or {}),
                           "budget": dict(defaults.get("budget") or {}), "replay": raw.get("replay", False)}
    out["budget"].update(raw.get("budget") or {})
    for k in ("provider", "model", "api_key_env", "base_url"):
        if k in raw and k not in out["default"]:
            out["default"][k] = raw[k]
    out["default"].setdefault("provider", (defaults.get("default") or {}).get("provider") or "anthropic")
    out["default"].setdefault("model", (defaults.get("default") or {}).get("model") or "claude-haiku-4-5")
    for k in ("ledger_path", "script"):
        if k in raw:
            out[k] = raw[k]
    return out


def _llm_provider_module():
    try:
        from . import llm  # noqa: F401  (written by another developer; may not exist yet)
    except Exception:
        return None
    return llm


def doctor_checks(cfg: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Checks appended by ``scpipe doctor``: site, execution, container, LLM, Open Targets, budget."""
    checks: list[dict[str, Any]] = []

    def add(group: str, name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({"group": group, "name": name, "ok": bool(ok), "required": required, "detail": detail})

    cfg = cfg or {}
    # Site file
    src = cfg.get("site_source")
    if cfg.get("site"):
        probs = validate_site_file(src) if src else [f"site {cfg.get('site')!r} could not be resolved"]
        add("site", "site file", not probs, src if not probs else "; ".join(probs))
    else:
        add("site", "site file", True, "none (settings from config.yaml; `scpipe init-site` drafts one)", required=False)
    # Execution mode
    ex = cfg.get("execution") or "direct"
    if ex == "direct":
        add("site", "execution", True, "direct (runs on this node)")
    else:
        sched = detect_scheduler() if ex == "split" else ex
        spec = SCHEDULERS.get(sched or "")
        have = spec is not None and shutil.which(spec["submit"]) is not None
        add("site", "execution", have, f"{ex}: " + (f"{spec['submit']} found" if have else
                                                   (f"{spec['submit']} (for {sched}) is not on PATH" if spec else "no scheduler (sbatch/qsub/bsub) on PATH")))
    # Container
    c = cfg.get("container") or {}
    rt = c.get("runtime", "none")
    if rt == "none":
        add("site", "container", True, "none (host environment)", required=False)
    else:
        w = shutil.which(rt)
        img = c.get("image")
        img_ok = img is None or str(img).startswith(("docker://", "oras://", "library://")) or Path(str(img)).exists() or rt == "docker"
        add("site", "container", w is not None and img_ok,
            f"{rt}: {w or 'not on PATH'}; image {img or 'none'}{'' if img_ok else ' (not found)'}")
    # LLM (config shape of DESIGN 4.8: llm.default.{provider,model,api_key_env,base_url}, llm.roles, llm.budget)
    llm_cfg = _llm_section(cfg)
    if llm_cfg is None:
        add("llm", "provider", True, "llm: not configured (gates run rules-only with a human review)", required=False)
        llm_cfg = {}
        provider = None
    else:
        provider = (llm_cfg.get("default") or {}).get("provider") or "anthropic"
    if provider is None:
        pass
    elif provider == "mock":
        add("llm", "provider", True, "mock (tests; no network)", required=False)
    elif provider in ("bedrock", "vertex"):
        add("llm", "provider", False, f"{provider}: not implemented yet (use anthropic or openai_compatible)", required=False)
    else:
        dflt = llm_cfg.get("default") or {}
        key_env = dflt.get("api_key_env") or DEFAULT_KEY_ENV.get(provider, "ANTHROPIC_API_KEY")
        have_key = bool(os.environ.get(key_env))
        add("llm", "api key", have_key, f"${key_env} is {'set' if have_key else 'not set'} (provider {provider})", required=False)
        base = dflt.get("base_url") or LLM_ENDPOINTS.get(provider)
        if base:
            ok, note = https_reachable(base)
            add("llm", "endpoint", ok, f"{base}: {note}", required=False)
        if not have_key:
            add("llm", "tool-use test", True, "skipped (no API key)", required=False)
        else:
            mod = _llm_provider_module()
            if mod is None or not hasattr(mod, "get_provider"):
                add("llm", "tool-use test", True, "skipped (scpipe.llm not available yet)", required=False)
            else:
                try:
                    prov = mod.get_provider(llm_cfg, "doctor")
                    resp = prov.complete(system="Reply by calling the tool.", messages=[{"role": "user", "content": "ping"}],
                                         tools=[{"name": "ping", "description": "reply", "input_schema": {"type": "object", "properties": {}}}],
                                         max_tokens=16)
                    used = getattr(resp, "tool_calls", None)
                    add("llm", "tool-use test", bool(used) or bool(getattr(resp, "text", "")),
                        f"ok: stop_reason={getattr(resp, 'stop_reason', '?')}, tool_calls={len(used or [])}", required=False)
                except Exception as e:
                    add("llm", "tool-use test", False, f"failed: {str(e)[:200]}", required=False)
    if provider is not None:
        # Budget: the total ledger shared across runs (scpipe.llm.ledger: {"runs": {run_id: gbp}})
        budget = llm_cfg.get("budget") or {}
        ledger_path = os.path.expanduser(str(budget.get("ledger_path") or llm_cfg.get("ledger_path") or "~/.scpipe/ledger.json"))
        limit = budget.get("total_limit", 30.0)
        spent = None
        try:
            if Path(ledger_path).exists():
                data = json.loads(Path(ledger_path).read_text())
                runs = data.get("runs") or {}
                spent = float(sum(float(v) for v in runs.values())) if isinstance(runs, dict) else float(data.get("total_gbp") or 0)
            else:
                spent = 0.0
        except Exception as e:
            add("llm", "budget", True, f"ledger {ledger_path} unreadable: {e}", required=False)
        if spent is not None:
            try:
                lim = float(limit)
                add("llm", "budget", spent < lim, f"GBP {spent:.2f} of {lim:.2f} spent across runs ({ledger_path})", required=False)
            except (TypeError, ValueError):
                add("llm", "budget", True, f"GBP {spent:.2f} spent; total_limit {limit!r} is not a number", required=False)
    # Open Targets
    ot = cfg.get("opentargets") or {}
    if ot.get("enabled", True) is False:
        add("opentargets", "api", True, "disabled in config", required=False)
    elif ot.get("offline"):
        add("opentargets", "api", True, f"offline mode; cache {ot.get('cache_dir')}", required=False)
    else:
        ok, note = https_reachable(OPENTARGETS_URL)
        add("opentargets", "api", ok, f"{OPENTARGETS_URL}: {note}", required=False)
    return checks


if __name__ == "__main__":
    raise SystemExit(init_site_main())
