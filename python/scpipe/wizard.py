"""``scpipe run -i``: a few questions before the run.

The wizard starts from the settings the run would use anyway (the config
defaults and the built-in QC policy) and asks for:

1. the mode: automatic (keep the defaults, agents decide at the review gates)
   or guided (the questions below);
2. QC hard limits (min genes, min UMIs, max mito %; ``off`` disables one);
3. adaptive QC (number of MADs, or ``off``);
4. doublets (method, and whether called doublets are removed);
5. clustering (PCs, resolution, HVGs);
6. batch correction (only with more than one sample);
7. the LLM for the review agents (from the API keys that are set);
8. the review gates (agents decide, stop at every gate, or the defaults).

Enter keeps the value in brackets. A bad value is explained and asked again.
:func:`run_wizard` returns the answers as config overrides, a QC policy and an
LLM choice; ``runner.main`` writes them to ``<run-dir>/config/wizard.yaml`` and
``wizard_qc_policy.yaml`` and prints the command that repeats the run without
questions. Prompts go through ``ask(text, default) -> str`` so tests can
script them.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from . import qc as qcmod
from .llm.provider import AUTO_ORDER, PRESETS
from .stage_names import title

__all__ = ["WizardResult", "run_wizard", "terminal_ask"]

Ask = Callable[[str, str], str]
Say = Callable[[str], None]

GATE_IDS = ("G1", "G2", "G3", "G4")
PROVIDER_NAMES = {"anthropic": "Claude (ANTHROPIC_API_KEY)", "openai": "OpenAI (OPENAI_API_KEY)",
                  "nvidia": "NVIDIA NIM (NVIDIA_API_KEY)"}


@dataclass
class WizardResult:
    overrides: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    llm: str | None = None  # provider name, "none", or None (keep the run's choice)
    summary: list[tuple[str, str]] = field(default_factory=list)


def terminal_ask(text: str, default: str) -> str:
    """Prompt on the terminal (``rich`` when available); Enter returns ``default``."""
    try:
        from rich.prompt import Prompt

        return Prompt.ask(text, default=default, show_default=True) or default
    except ImportError:
        got = input(f"{text} [{default}]: ").strip()
        return got or default


class _Asker:
    def __init__(self, ask: Ask, say: Say):
        self.ask, self.say = ask, say

    def choice(self, text: str, options: Sequence[tuple[str, str]], default: str) -> str:
        """Numbered options; the answer may be the number or the key."""
        for n, (key, label) in enumerate(options, 1):
            self.say(f"  [{n}] {label}")
        keys = [k for k, _ in options]
        dflt = str(keys.index(default) + 1)
        while True:
            got = self.ask(text, dflt).strip().lower()
            if got.isdigit() and 1 <= int(got) <= len(keys):
                return keys[int(got) - 1]
            if got in keys:
                return got
            self.say(f"  please answer 1-{len(keys)}")

    def number(self, text: str, default: Any, *, integer: bool = False, allow_off: bool = False,
               allow_auto: bool = False, lo: float | None = None, hi: float | None = None,
               lo_open: bool = False) -> Any:
        """A number, or ``off`` (None) / ``auto`` when allowed."""
        shown = "off" if default is None else str(default)
        while True:
            got = self.ask(text, shown).strip().lower()
            if allow_off and got in ("off", "none", "no"):
                return None
            if allow_auto and got == "auto":
                return "auto"
            try:
                v = float(got)
            except ValueError:
                extra = "".join([", 'off'" if allow_off else "", ", 'auto'" if allow_auto else ""])
                self.say(f"  please enter a number{extra}")
                continue
            if integer and not v.is_integer():
                self.say("  please enter a whole number")
                continue
            if (lo is not None and (v <= lo if lo_open else v < lo)) or (hi is not None and v > hi):
                rng = f"{'above' if lo_open else 'at least'} {lo:g}" if lo is not None else ""
                rng += (" and " if rng and hi is not None else "") + (f"at most {hi:g}" if hi is not None else "")
                self.say(f"  must be {rng}")
                continue
            return int(v) if integer or v.is_integer() else v

    def yes(self, text: str, default: bool) -> bool:
        while True:
            got = self.ask(f"{text} (y/n)", "y" if default else "n").strip().lower()
            if got in ("y", "yes"):
                return True
            if got in ("n", "no"):
                return False
            self.say("  please answer y or n")


def _keys_set(env: Mapping[str, str]) -> list[str]:
    return [p for p in AUTO_ORDER if env.get(PRESETS[p]["api_key_env"])]


def run_wizard(cfg: Mapping[str, Any], *, ask: Ask = terminal_ask, say: Say = print,
               n_samples: int = 1, ask_llm: bool = True, ask_batch: bool = True,
               env: Mapping[str, str] | None = None) -> WizardResult | None:
    """Ask the questions; None when the user declines to start."""
    env = os.environ if env is None else env
    q = _Asker(ask, say)
    res = WizardResult()
    policy = copy.deepcopy(cfg["_policy"].to_dict()) if cfg.get("_policy") is not None else \
        copy.deepcopy(qcmod.DEFAULT_POLICY)
    gates_default = dict(cfg.get("gates") or {})

    say("\nscpipe setup. Press Enter to keep the suggested value; Ctrl-C cancels.\n")
    say("How would you like to run the pipeline?")
    mode = q.choice("Mode", [("auto", "Automatic: defaults and adaptive QC; the review agents decide at each gate"),
                             ("guided", "Guided: choose the QC cut-offs, doublets, clustering and review gates")],
                    "auto")

    hard, adaptive = policy["hard_limits"], policy["adaptive"]
    if mode == "guided":
        say("\nQuality control: cells outside these limits are removed ('off' disables a limit).")
        hard["min_genes"] = q.number("Minimum genes per cell", hard["min_genes"], allow_off=True, lo=0)
        hard["min_umis"] = q.number("Minimum UMIs per cell", hard["min_umis"], allow_off=True, lo=0)
        hard["max_mito_pct"] = q.number("Maximum mitochondrial %", hard["max_mito_pct"], allow_off=True, lo=0,
                                        hi=100)
        say("\nAdaptive QC also removes outliers per sample: median ± N × MAD of genes, UMIs and mito %.")
        n_mads = q.number("Number of MADs ('off' = hard limits only)",
                          adaptive["n_mads"] if adaptive.get("metrics") else None, allow_off=True, lo=0, lo_open=True)
        if n_mads is None:
            adaptive["metrics"] = []
        else:
            adaptive["n_mads"] = n_mads
            adaptive["metrics"] = adaptive.get("metrics") or list(qcmod.ADAPTIVE_METRICS)

        say("\nDoublets (two cells captured together):")
        method = q.choice("Doublet detection", [("scrublet", "Scrublet"), ("scdblfinder", "scDblFinder (R)"),
                                                ("none", "none")],
                          str((cfg.get("doublets") or {}).get("method", "scrublet")))
        res.overrides["doublets"] = {"method": method}
        if method != "none":
            policy["doublets"]["exclude_called"] = q.yes("Remove called doublets",
                                                         bool(policy["doublets"].get("exclude_called", True)))

        say("\nClustering:")
        res.overrides["pcs"] = q.number("Principal components ('auto' = elbow, 10-50)", cfg.get("pcs", "auto"),
                                        integer=True, allow_auto=True, lo=2, hi=200)
        res.overrides["clustering"] = {"resolution": q.number(
            "Leiden resolution (higher = more clusters)", (cfg.get("clustering") or {}).get("resolution", 0.5),
            lo=0, lo_open=True, hi=10)}
        res.overrides["n_hvg"] = q.number("Highly variable genes", cfg.get("n_hvg", 2000), integer=True,
                                          lo=100, hi=20000)

    if ask_batch and n_samples > 1:
        integ = cfg.get("integration") or {}
        default = integ.get("method") == "harmony"
        say(f"\n{n_samples} samples: batch correction aligns them before clustering.")
        if q.yes("Correct batch effects between samples with Harmony", default):
            res.overrides["integration"] = {"method": "harmony", "batch_key": integ.get("batch_key", "sample_id")}
        elif default:
            res.overrides["integration"] = {"method": "none"}

    have = _keys_set(env)
    if ask_llm:
        say("\nReview agents (LLM) at the QC, context, annotation and DGE gates:")
        if have:
            opts = [(p, PROVIDER_NAMES[p]) for p in have] + [("none", "none: rules only, every gate waits for you")]
            res.llm = q.choice("LLM", opts, have[0]) if mode == "guided" or len(have) > 1 else have[0]
            if mode == "auto" and len(have) == 1:
                say(f"  using {PROVIDER_NAMES[have[0]]}")
        else:
            say("  no API key found (ANTHROPIC_API_KEY, OPENAI_API_KEY or NVIDIA_API_KEY): rules only, "
                "and every gate waits for you")
            res.llm = "none"
    llm_on = res.llm not in ("none",) and (res.llm is not None or bool(cfg.get("llm")))

    if mode == "guided":
        say("\nReview gates: after QC, context, annotation and DGE the pipeline can stop for you.")
        g = q.choice("Gates", [("auto", "Agents decide; stop only when they flag a problem"),
                               ("review", "Stop at every gate for my decision"),
                               ("default", "Pipeline defaults (QC and annotation automatic, context and DGE stop)")],
                     "auto" if llm_on else "review")
    else:
        g = "auto"
    if g in ("auto", "review"):
        res.overrides["gates"] = {gid: g for gid in GATE_IDS}

    try:
        qcmod.policy_from_dict(policy)
    except qcmod.QCError as e:  # the questions keep values in range, so this is a safety net
        raise ValueError(f"QC settings are not valid: {e}") from e
    res.policy = policy

    def off(v: Any, unit: str = "") -> str:
        return "off" if v is None else f"{v}{unit}"

    gates_eff = res.overrides.get("gates") or gates_default
    res.summary = [
        ("Mode", "automatic" if mode == "auto" else "guided"),
        ("QC hard limits", f"min genes {off(hard['min_genes'])}, min UMIs {off(hard['min_umis'])}, "
                           f"max mito {off(hard['max_mito_pct'], '%')}"),
        ("Adaptive QC", f"{adaptive['n_mads']} MADs" if adaptive.get("metrics") else "off"),
        ("Doublets", (res.overrides.get("doublets") or cfg.get("doublets") or {}).get("method", "scrublet")
         + ("" if policy["doublets"].get("exclude_called", True) else ", kept")),
        ("PCs / resolution / HVGs", f"{res.overrides.get('pcs', cfg.get('pcs'))} / "
                                    f"{(res.overrides.get('clustering') or cfg.get('clustering') or {}).get('resolution')}"
                                    f" / {res.overrides.get('n_hvg', cfg.get('n_hvg'))}"),
    ]
    if n_samples > 1:
        im = (res.overrides.get("integration") or cfg.get("integration") or {}).get("method", "none")
        res.summary.append(("Batch correction", "Harmony" if im == "harmony" else "none"))
    res.summary.append(("Review agents", {None: "as configured", "none": "none (rules only)"}.get(
        res.llm, PROVIDER_NAMES.get(str(res.llm), str(res.llm)))))
    res.summary.append(("Gates", "; ".join(f"{title(gid)}: {gates_eff.get(gid, '-')}" for gid in GATE_IDS)))

    say("\nSummary:")
    w = max(len(k) for k, _ in res.summary)
    for k, v in res.summary:
        say(f"  {k:<{w}}  {v}")
    if not q.yes("\nStart the run", True):
        return None
    return res
