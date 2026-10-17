"""Gate G2 (tissue / disease context): bundle builder, rules baseline, policy check.

Spec: docs/agents-spec.md sections 2, 4 and 5; DESIGN.md 4.2.

Inputs read from the run directory:

* ``S3/evidence_bundle.json`` (spec section 4). Accepted layout::

      {"organism": "human", "n_cells": int, "n_samples": int, "samples": [...],
       "sample_metadata_hint": [str, ...],          # free text from the sample sheet
       "celltypist_model": str,                      # optional
       "clusters": {"<id>": {"n_cells", "pct_cells", "per_sample", "median_genes",
                             "median_umis", "median_pct_mito", "top_markers": [..],
                             "markers": [{gene, avg_log2fc, pct_in, pct_out, p_adj, rank}],
                             "signatures": {"<set>": float},
                             "celltypist": {"<label>": fraction}}}}

  ``clusters`` may also be a list of such dicts with a ``cluster`` key.
* ``S3/annotation_consensus.tsv`` (spec section 5), optional for G2.

``bundle.json`` holds aggregated tables only; the sample metadata hint is the
one free-text field and it is carried as a list of strings that the agent's
user message wraps in ``<data>`` tags.

G2 proposal shape (spec 5)::

    {"tissue": {"label", "uberon_id", "confidence"},
     "disease_candidates": [{"label", "efo_id", "confidence", "evidence"}],
     "celltypist_model": str}
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping

GATE = "G2"
WORKER_ROLE = "context_worker"
MAX_DISEASE_CANDIDATES = 5
UBERON_RE = re.compile(r"^UBERON_\d+$")
DISEASE_ID_RE = re.compile(r"^EFO_\d+$|^MONDO_\d+$|^Orphanet_\d+$")
DEFAULT_CELLTYPIST_MODEL = "Immune_All_Low.pkl"

# Lineage signature sets that are immune (blood) vs. tissue-resident; used by the rules baseline.
_IMMUNE_SETS = {"T", "CD4", "CD8", "NK", "B", "plasma", "mono_CD14", "mono_CD16", "cDC", "pDC", "neutrophil", "mast"}
_TISSUE_SETS = {"epithelial", "endothelial", "fibroblast", "hepatocyte", "neuron", "glia"}


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def read_evidence_bundle(path: str | Path) -> dict[str, Any]:
    """Read ``evidence_bundle.json`` and normalise ``clusters`` to a dict keyed by cluster id (str)."""
    raw = _load_json(Path(path))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    clusters = raw.get("clusters", {})
    if isinstance(clusters, list):
        clusters = {str(c.get("cluster", i)): c for i, c in enumerate(clusters)}
    out = dict(raw)
    out["clusters"] = {str(k): dict(v) for k, v in clusters.items()}
    hint = raw.get("sample_metadata_hint") or []
    if isinstance(hint, (str, Mapping)):
        hint = [hint]
    out["sample_metadata_hint"] = [h if isinstance(h, str) else json.dumps(h, sort_keys=True) for h in hint]
    return out


def read_consensus(path: str | Path) -> list[dict[str, Any]]:
    """Read ``annotation_consensus.tsv`` (spec section 5) into a list of row dicts."""
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            r = dict(row)
            r["cluster"] = str(r.get("cluster", "")).strip()
            try:
                r["confidence"] = float(r.get("confidence") or 0.0)
            except ValueError:
                r["confidence"] = 0.0
            if r.get("cl_id") in ("", "NA", "None", None):
                r["cl_id"] = None
            r["status"] = (r.get("status") or "consensus").strip()
            try:
                r["votes"] = json.loads(r.get("votes_json") or "{}")
            except ValueError:
                r["votes"] = {}
            rows.append(r)
    return rows


def cluster_table(evidence: Mapping[str, Any], consensus: list[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Per-cluster aggregate rows used by G2 and G3 bundles (no per-cell data)."""
    cons = {str(r["cluster"]): r for r in (consensus or [])}
    rows = []
    for cid, c in evidence["clusters"].items():
        row = {
            "cluster": cid,
            "n_cells": c.get("n_cells"),
            "pct_cells": c.get("pct_cells"),
            "n_samples_present": sum(1 for v in (c.get("per_sample") or {}).values() if v),
            "median_genes": c.get("median_genes"),
            "median_umis": c.get("median_umis"),
            "median_pct_mito": c.get("median_pct_mito"),
            "top_markers": list(c.get("top_markers") or [])[:20],
            "signatures": dict(c.get("signatures") or {}),
            "celltypist": dict(c.get("celltypist") or {}),
        }
        if cid in cons:
            r = cons[cid]
            row["consensus"] = {k: r.get(k) for k in ("label_broad", "label_fine", "cl_id", "confidence", "status")}
            row["consensus"]["votes"] = r.get("votes", {})
        rows.append(row)
    rows.sort(key=lambda r: (len(r["cluster"]), r["cluster"]))
    return rows


def build_bundle(run_dir: str | Path, cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``gates/G2/bundle.json``: aggregated cluster evidence + sample metadata hint."""
    run_dir = Path(run_dir)
    cfg = cfg or {}
    evidence = read_evidence_bundle(run_dir / "S3" / "evidence_bundle.json")
    cons_path = run_dir / "S3" / "annotation_consensus.tsv"
    consensus = read_consensus(cons_path) if cons_path.exists() else []
    clusters = cluster_table(evidence, consensus)
    sig_table = {r["cluster"]: r["signatures"] for r in clusters}
    composition = {r["cluster"]: r["pct_cells"] for r in clusters}
    ann_cfg = dict(cfg.get("annotation") or {})
    return {
        "gate": GATE,
        "organism": "human",
        "n_cells": evidence.get("n_cells"),
        "n_samples": evidence.get("n_samples", len(evidence.get("samples") or [])),
        "samples": list(evidence.get("samples") or []),
        "clusters": clusters,
        "signature_table": sig_table,
        "signature_sets": list(evidence.get("signature_sets") or sorted({s for v in sig_table.values() for s in v})),
        "composition_pct": composition,
        "sample_metadata_hint": list(evidence.get("sample_metadata_hint") or []),
        "celltypist_model": ann_cfg.get("celltypist_model") or evidence.get("celltypist_model") or DEFAULT_CELLTYPIST_MODEL,
    }


def baseline(bundle: Mapping[str, Any], cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Rules-only decision: tissue from lineage signatures, no disease claims."""
    sig = bundle.get("signature_table") or {}
    comp = bundle.get("composition_pct") or {}
    immune = tissue = 0.0
    for cid, scores in sig.items():
        w = float(comp.get(cid) or 0.0) / 100.0
        best = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None
        if best in _IMMUNE_SETS or best == "platelet" or best == "erythrocyte":
            immune += w
        elif best in _TISSUE_SETS:
            tissue += w
    if immune >= 0.9 and tissue < 0.05:
        t = {"label": "peripheral blood", "uberon_id": "UBERON_0000178", "confidence": 0.6}
    elif immune >= 0.5:
        t = {"label": "immune-rich tissue (unresolved)", "uberon_id": None, "confidence": 0.3}
    else:
        t = {"label": "unknown", "uberon_id": None, "confidence": 0.0}
    return {"tissue": t, "disease_candidates": [],
            "celltypist_model": bundle.get("celltypist_model") or DEFAULT_CELLTYPIST_MODEL}


def _num01(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and 0.0 <= float(x) <= 1.0


def policy_check(proposal: Mapping[str, Any], bundle: Mapping[str, Any] | None = None) -> tuple[bool, list[str]]:
    """Validate a G2 proposal in code. Returns ``(ok, violations)``."""
    v: list[str] = []
    if not isinstance(proposal, Mapping):
        return False, ["proposal must be an object"]
    tissue = proposal.get("tissue")
    if not isinstance(tissue, Mapping):
        v.append("tissue must be an object {label, uberon_id, confidence}")
    else:
        label = tissue.get("label")
        if not isinstance(label, str) or not label.strip():
            v.append("tissue.label must be a non-empty string")
        if not _num01(tissue.get("confidence")):
            v.append("tissue.confidence must be a number in [0, 1]")
        uid = tissue.get("uberon_id")
        if uid is not None and not (isinstance(uid, str) and UBERON_RE.match(uid)):
            v.append(f"tissue.uberon_id must be null or match ^UBERON_\\d+$ (got {uid!r})")
    cands = proposal.get("disease_candidates", [])
    if not isinstance(cands, list):
        v.append("disease_candidates must be a list")
    else:
        if len(cands) > MAX_DISEASE_CANDIDATES:
            v.append(f"at most {MAX_DISEASE_CANDIDATES} disease candidates (got {len(cands)})")
        for i, c in enumerate(cands):
            if not isinstance(c, Mapping):
                v.append(f"disease_candidates[{i}] must be an object")
                continue
            if not isinstance(c.get("label"), str) or not c["label"].strip():
                v.append(f"disease_candidates[{i}].label must be a non-empty string")
            if not isinstance(c.get("evidence"), str) or not c["evidence"].strip():
                v.append(f"disease_candidates[{i}].evidence must be non-empty text")
            if not _num01(c.get("confidence")):
                v.append(f"disease_candidates[{i}].confidence must be a number in [0, 1]")
            eid = c.get("efo_id")
            if eid is not None and not (isinstance(eid, str) and DISEASE_ID_RE.match(eid)):
                v.append(f"disease_candidates[{i}].efo_id must be null or match EFO_/MONDO_/Orphanet_ + digits (got {eid!r})")
    model = proposal.get("celltypist_model")
    if not isinstance(model, str) or not model.strip():
        v.append("celltypist_model must be a non-empty string")
    for key in proposal:
        if key not in ("tissue", "disease_candidates", "celltypist_model"):
            v.append(f"unexpected field {key!r}")
    return (not v), v


def make_worker(provider, ledger=None, **kw):
    from scpipe.agents.context import ContextAgent

    return ContextAgent(provider, ledger, **kw)
