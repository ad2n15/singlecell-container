"""Thin compatibility shim for the G2/G3/G4 worker agents.

The agents in ``context.py``, ``annotation.py`` and ``interpretation.py`` import
exactly five names from this module:

    get_provider, Agent, Ledger, run_gate, write_gate_files

Each name is taken from the real module when it exists
(``scpipe.llm``, ``scpipe.agents.base``, ``scpipe.gates.protocol``,
docs/agents-spec.md sections 2 and 3) and otherwise from the minimal fallback
defined below so that the agents and their tests run before those modules
land. **The fallbacks are meant to be deleted** once the real modules exist;
nothing outside this file should depend on fallback-only names.

Interfaces assumed (spec sections 2 and 3, plus the conventions below that the
spec leaves to the developer):

Provider / Response (spec 3)
    ``provider.complete(*, system, messages, tools, max_tokens, json_schema=None)``
    returns an object with ``.text``, ``.tool_calls`` (list of
    ``{id, name, arguments}``), ``.usage`` (``{input_tokens, output_tokens,
    cached_tokens}``), ``.stop_reason``. ``provider.model`` names the model.
    ``get_provider(cfg_llm, role)`` builds one. The fallback ``mock`` provider
    takes its scripted responses from ``cfg_llm["script"]`` (a list of
    ``{"text": str, "tool_calls": [...]}`` dicts, one per call).

Messages
    Provider-neutral list of dicts: ``{"role": "user", "content": str}``,
    ``{"role": "assistant", "content": str, "tool_calls": [...]}`` and
    ``{"role": "tool", "tool_call_id": str, "name": str, "content": str}``.

Tools
    ``{"name", "description", "input_schema"}`` (JSON schema) sent to the
    provider; agents keep the callable alongside as ``"fn"`` and mark the
    proposal tool with ``"proposal": True``. The base loop removes those two
    keys before the provider sees the definitions.

Agent (spec 3, "agent base")
    ``Agent(provider, ledger=None, *, log_path=None, max_tokens=2048)``;
    subclasses set ``role`` (prompt file ``prompts/<role>.md``) and ``gate``,
    and implement ``build_tools(bundle) -> list[tool dict]`` and
    ``user_message(bundle) -> str``. ``run(bundle)`` returns a dict in the
    ``proposal.json`` shape of spec section 2. A proposal tool's function
    returns ``{"accepted": bool, "violations": [...], "proposal": {...},
    "rationale": str}``; the loop stops on the first accepted proposal, after
    the second rejected one (one retry, spec 2 step 3), after 12 tool calls,
    or when the model stops calling tools. Free text from data is wrapped by
    :func:`wrap_data`.

Gate modules (``scpipe.gates.g2_context`` etc.)
    expose ``GATE``, ``WORKER_ROLE``, ``build_bundle(run_dir, cfg)``,
    ``baseline(bundle, cfg)``, ``policy_check(proposal, bundle)`` and
    ``make_worker(provider, ledger=None, **kw)``. ``run_gate(gate, run_dir,
    cfg)`` (spec 2) drives them and writes the four gate files through
    ``write_gate_files(gate_dir, bundle, proposal, review, decision,
    report_html=None)``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = ["get_provider", "Agent", "Ledger", "run_gate", "write_gate_files", "wrap_data"]

MAX_TOOL_CALLS = 12
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def wrap_data(text: Any) -> str:
    """Wrap free text that comes from data so prompts can mark it as never-instructions."""
    s = "" if text is None else str(text)
    s = s.replace("</data>", "</ data>")
    return f"<data>{s}</data>"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(obj: Any) -> str:
    if not isinstance(obj, str):
        obj = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(obj.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Provider + Ledger (scpipe.llm)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only once the real module exists
    from scpipe.llm import Ledger, get_provider  # type: ignore
except ImportError:  # fallback

    @dataclass
    class _Response:
        text: str = ""
        tool_calls: list[dict] = field(default_factory=list)
        usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0})
        stop_reason: str = "end_turn"

    class _MockProvider:
        """Scripted provider: one scripted response per ``complete`` call, then silence."""

        model = "mock"

        def __init__(self, script: list[dict] | None = None, model: str = "mock"):
            self.script = [dict(s) for s in (script or [])]
            self.model = model
            self.calls: list[dict] = []
            self._i = 0

        def complete(self, *, system, messages, tools, max_tokens, json_schema=None):
            self.calls.append({"system": system, "messages": list(messages), "tools": tools})
            if self._i >= len(self.script):
                return _Response(text="", stop_reason="end_turn")
            step = self.script[self._i]
            self._i += 1
            calls = []
            for k, tc in enumerate(step.get("tool_calls", [])):
                calls.append({"id": tc.get("id", f"call_{self._i}_{k}"), "name": tc["name"],
                              "arguments": dict(tc.get("arguments", {}))})
            usage = {"input_tokens": len(json.dumps(messages, default=str)) // 4,
                     "output_tokens": 50 * max(1, len(calls)), "cached_tokens": 0}
            return _Response(text=step.get("text", ""), tool_calls=calls, usage=usage,
                             stop_reason="tool_use" if calls else "end_turn")

    def get_provider(cfg_llm: Mapping[str, Any] | None, role: str):  # type: ignore[no-redef]
        cfg_llm = dict(cfg_llm or {})
        default = dict(cfg_llm.get("default") or {})
        roles = dict(cfg_llm.get("roles") or {})
        merged = {**default, **cfg_llm, **dict(roles.get(role) or {})}
        provider = merged.get("provider", "mock")
        if provider != "mock":
            raise NotImplementedError(f"fallback get_provider supports only 'mock', not {provider!r}")
        script = merged.get("script")
        if isinstance(script, Mapping):
            script = script.get(role, [])
        return _MockProvider(script or [], model=merged.get("model", "mock"))

    class Ledger:  # type: ignore[no-redef]
        """Minimal budget ledger: USD prices per MTok, GBP totals, hard limits."""

        PRICES = {"mock": (0.0, 0.0), "claude-haiku-4-5": (1.0, 5.0)}

        def __init__(self, path: str | Path | None = None, *, per_run_limit: float | None = None,
                     total_limit: float | None = None, usd_to_gbp: float = 0.78):
            self.path = Path(path) if path else None
            self.per_run_limit = per_run_limit
            self.total_limit = total_limit
            self.usd_to_gbp = usd_to_gbp
            self.run_total_gbp = 0.0
            self.total_gbp = 0.0
            self.entries: list[dict] = []
            if self.path and self.path.exists():
                try:
                    self.total_gbp = float(json.loads(self.path.read_text()).get("total_gbp", 0.0))
                except (ValueError, OSError):
                    self.total_gbp = 0.0

        def _cost(self, model: str, in_tok: float, out_tok: float) -> float:
            pin, pout = self.PRICES.get(model, (1.0, 5.0))
            return (in_tok * pin + out_tok * pout) / 1e6 * self.usd_to_gbp

        def charge(self, role: str, model: str, usage: Mapping[str, Any]) -> float:
            cost = self._cost(model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            self.run_total_gbp += cost
            self.total_gbp += cost
            self.entries.append({"ts": _utc(), "role": role, "model": model, "usage": dict(usage), "cost_gbp": cost})
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps({"total_gbp": self.total_gbp, "entries": self.entries}, indent=1))
            return cost

        def would_exceed(self, est_input_tokens: int, est_output_tokens: int, model: str = "mock") -> bool:
            est = self._cost(model, est_input_tokens, est_output_tokens)
            if self.per_run_limit is not None and self.run_total_gbp + est > self.per_run_limit:
                return True
            if self.total_limit is not None and self.total_gbp + est > self.total_limit:
                return True
            return False


# ---------------------------------------------------------------------------
# Agent base (scpipe.agents.base)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from scpipe.agents.base import Agent  # type: ignore
except ImportError:

    class Agent:  # type: ignore[no-redef]
        """Fallback agent loop: system prompt + tools + ``run(bundle) -> proposal``."""

        role: str = ""
        gate: str = ""
        max_tool_calls: int = MAX_TOOL_CALLS

        def __init__(self, provider, ledger=None, *, log_path: str | Path | None = None, max_tokens: int = 2048):
            self.provider = provider
            self.ledger = ledger
            self.log_path = Path(log_path) if log_path else None
            self.max_tokens = max_tokens
            self.model = getattr(provider, "model", "unknown")

        # -- subclass hooks --------------------------------------------------
        def build_tools(self, bundle: Mapping[str, Any]) -> list[dict]:
            raise NotImplementedError

        def user_message(self, bundle: Mapping[str, Any]) -> str:
            return "Evidence bundle:\n" + json.dumps(bundle, indent=1, default=str)

        def system_prompt(self) -> str:
            return (PROMPT_DIR / f"{self.role}.md").read_text(encoding="utf-8")

        # -- helpers ---------------------------------------------------------
        @staticmethod
        def _validate_args(schema: Mapping[str, Any], args: Mapping[str, Any]) -> list[str]:
            problems = []
            props = schema.get("properties", {})
            for k in schema.get("required", []):
                if k not in args:
                    problems.append(f"missing argument {k!r}")
            for k in args:
                if props and k not in props:
                    problems.append(f"unknown argument {k!r}")
            return problems

        def _log(self, entry: dict) -> None:
            if not self.log_path:
                return
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")

        # -- loop ------------------------------------------------------------
        def run(self, bundle: Mapping[str, Any]) -> dict:
            tools = self.build_tools(bundle)
            by_name = {t["name"]: t for t in tools}
            defs = [{k: v for k, v in t.items() if k in ("name", "description", "input_schema")} for t in tools]
            system = self.system_prompt()
            user = self.user_message(bundle)
            messages: list[dict] = [{"role": "user", "content": user}]
            n_calls = 0
            rejections = 0
            accepted: dict | None = None
            last_rejected: dict | None = None
            text_out = ""
            prompt_sha, input_sha = _sha(system), _sha(user)
            while True:
                est_in = len(json.dumps(messages, default=str)) // 4 + len(system) // 4
                if self.ledger is not None and self.ledger.would_exceed(est_in, self.max_tokens):
                    break
                resp = self.provider.complete(system=system, messages=messages, tools=defs,
                                              max_tokens=self.max_tokens)
                if self.ledger is not None:
                    self.ledger.charge(self.role, self.model, resp.usage)
                self._log({"ts": _utc(), "gate": self.gate, "role": self.role, "model": self.model,
                           "prompt_sha256": prompt_sha, "input_sha256": input_sha,
                           "tool_calls": resp.tool_calls, "output": resp.text, "usage": resp.usage})
                text_out = resp.text or text_out
                if not resp.tool_calls:
                    break
                messages.append({"role": "assistant", "content": resp.text, "tool_calls": resp.tool_calls})
                stop = False
                for tc in resp.tool_calls:
                    n_calls += 1
                    tool = by_name.get(tc["name"])
                    args = tc.get("arguments") or {}
                    if tool is None:
                        result: Any = {"error": f"unknown tool {tc['name']!r}"}
                    else:
                        probs = self._validate_args(tool["input_schema"], args) if isinstance(args, Mapping) else ["arguments must be an object"]
                        if probs:
                            result = {"error": "; ".join(probs)}
                        else:
                            try:
                                result = tool["fn"](**args)
                            except (ValueError, TypeError, KeyError) as exc:
                                result = {"error": f"{type(exc).__name__}: {exc}"}
                    if tool is not None and tool.get("proposal") and isinstance(result, Mapping) and "accepted" in result:
                        if result["accepted"]:
                            accepted = dict(result)
                            stop = True
                        else:
                            rejections += 1
                            last_rejected = dict(result)
                            if rejections >= 2:
                                stop = True
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"],
                                     "content": json.dumps(result, default=str)})
                    if stop or n_calls >= self.max_tool_calls:
                        stop = True
                        break
                if stop:
                    break
            worker = {"model": self.model, "role": self.role}
            if accepted is not None:
                return {"gate": self.gate, "worker": worker, "proposal": accepted["proposal"],
                        "rationale": accepted.get("rationale", ""),
                        "policy_check": {"ok": True, "violations": []}, "fallback": False}
            if last_rejected is not None:
                return {"gate": self.gate, "worker": worker, "proposal": last_rejected.get("proposal", {}),
                        "rationale": last_rejected.get("rationale", ""),
                        "policy_check": {"ok": False, "violations": list(last_rejected.get("violations", []))},
                        "fallback": True}
            return {"gate": self.gate, "worker": worker, "proposal": {}, "rationale": text_out,
                    "policy_check": {"ok": False, "violations": ["no proposal tool call"]}, "fallback": True}


# ---------------------------------------------------------------------------
# Gate protocol (scpipe.gates.protocol)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from scpipe.gates.protocol import run_gate, write_gate_files  # type: ignore
except ImportError:
    GATE_DEFAULT_MODES = {"G1": "auto", "G2": "review", "G3": "auto", "G4": "review"}

    def write_gate_files(gate_dir: str | Path, bundle: Mapping[str, Any], proposal: Mapping[str, Any],
                         review: Mapping[str, Any], decision: Mapping[str, Any],
                         report_html: str | None = None) -> dict[str, Path]:  # type: ignore[no-redef]
        gate_dir = Path(gate_dir)
        gate_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for name, obj in (("bundle", bundle), ("proposal", proposal), ("review", review), ("decision", decision)):
            p = gate_dir / f"{name}.json"
            p.write_text(json.dumps(obj, indent=1, default=str, sort_keys=True))
            out[name] = p
        if report_html is None:
            report_html = _simple_report(decision.get("gate", gate_dir.name), proposal, review, decision)
        p = gate_dir / "report.html"
        p.write_text(report_html)
        out["report"] = p
        return out

    def _simple_report(gate: str, proposal: Mapping[str, Any], review: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
        def esc(s: Any) -> str:
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(gate)} gate</title></head><body>"
                f"<h1>Gate {esc(gate)}</h1><h2>Decision ({esc(decision.get('mode'))}, by {esc(decision.get('decided_by'))})</h2>"
                f"<pre>{esc(json.dumps(decision.get('decision'), indent=1))}</pre>"
                f"<h2>Worker rationale</h2><pre>{esc(proposal.get('rationale', ''))}</pre>"
                f"<h2>Reviewer</h2><pre>{esc(json.dumps(review, indent=1))}</pre></body></html>")

    def _gate_module(gate: str):
        import importlib
        names = {"G2": "g2_context", "G3": "g3_annotation", "G4": "g4_interpretation"}
        if gate not in names:
            raise ValueError(f"fallback run_gate knows only G2-G4, not {gate!r}")
        return importlib.import_module(f"scpipe.gates.{names[gate]}")

    def run_gate(gate: str, run_dir: str | Path, cfg: Mapping[str, Any]) -> dict:  # type: ignore[no-redef]
        """Spec section 2 flow with a rules-only reviewer (no reviewer LLM in the fallback)."""
        run_dir = Path(run_dir)
        mod = _gate_module(gate)
        bundle = mod.build_bundle(run_dir, cfg)
        base = mod.baseline(bundle, cfg)
        mode = str((cfg.get("gates") or {}).get(gate, GATE_DEFAULT_MODES.get(gate, "review")))
        llm_cfg = cfg.get("llm")
        worker_role = mod.WORKER_ROLE
        proposal: dict
        if not llm_cfg:
            proposal = {"gate": gate, "worker": {"model": None, "role": worker_role}, "proposal": base,
                        "rationale": "rules baseline (llm not configured)",
                        "policy_check": {"ok": True, "violations": []}, "fallback": True}
        else:
            budget = dict(llm_cfg.get("budget") or {})
            ledger = Ledger(llm_cfg.get("ledger_path") or (run_dir / "llm" / "ledger.json"),
                            per_run_limit=budget.get("per_run_limit"), total_limit=budget.get("total_limit"),
                            usd_to_gbp=float(budget.get("usd_to_gbp", 0.78)))
            provider = get_provider(llm_cfg, worker_role)
            worker = mod.make_worker(provider, ledger, log_path=run_dir / "llm" / "agent_log.jsonl")
            proposal = worker.run(bundle)
            if not proposal["policy_check"]["ok"]:
                proposal = dict(proposal, proposal=base, fallback=True)
        if proposal["fallback"]:
            mode = "review" if mode == "auto" else mode
        ok, viol = mod.policy_check(proposal["proposal"], bundle)
        review = {"verdict": "approve" if ok else "escalate",
                  "findings": [{"severity": "block", "text": v} for v in viol] or
                  ([{"severity": "info", "text": "rules baseline used (fallback)"}] if proposal["fallback"] else []),
                  "reviewer": {"model": "rules"}}
        decided_by = "rules" if proposal["fallback"] else "agent"
        decision = {"gate": gate, "mode": mode, "decided_by": decided_by, "decided_utc": _utc(),
                    "decision": proposal["proposal"], "changes_vs_proposal": [],
                    "status": "approved" if (mode == "auto" and review["verdict"] == "approve") else "pending"}
        files = write_gate_files(run_dir / "gates" / gate, bundle, proposal, review, decision)
        return {"gate": gate, "mode": mode, "continue": decision["status"] == "approved",
                "decision": decision, "proposal": proposal, "review": review, "files": files}
