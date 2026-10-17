"""S5 report and explorer (docs/agents-spec.md section 7).

* :func:`scpipe.report.build.build_all` writes ``S5/report.html`` and
  ``S5/explorer.html`` for a run directory.
* :func:`scpipe.report.build.render_gate_report` renders one gate's
  ``report.html`` from its ``bundle/proposal/review/decision.json``.
* :mod:`scpipe.report.plots` holds the small inline-SVG / matplotlib plots.
"""

from .build import build_all, build_explorer, build_report, collect, render_gate_report, render_report

__all__ = ["build_all", "build_explorer", "build_report", "collect", "render_gate_report", "render_report"]
