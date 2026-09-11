#!/usr/bin/env python3
"""Render-time and PDF quality gates for the publication figures.

Two modes:
  python3 code/scripts/figure_qa.py            # full ``make figures`` build with the gates registered
  python3 code/scripts/figure_qa.py --audit output/figures   # audit already exported PDFs only

Gates (all from the local ``nature-figure`` skill scripts, path via ``NATURE_FIGURE_SCRIPTS``):
  * panel alignment measured on the final drawn layout (shared edges, widths, heights, gutters;
    1.5 pt tolerance) — registered as a ``style.PREFLIGHT_HOOKS`` entry, so a failure blocks export;
  * a CJK tofu guard around the whole render (missing-glyph reports become errors);
  * PDF glyph-size floor and PDF collision audit (text crossed by strokes, text over text).

matplotlib writes bar hatches as PDF tiling patterns; PyMuPDF reports each pattern cell as a
page-space drawing (72 or 144 pt squares on a 36 pt grid made only of lines, only of circles, or
one filled rect). Those cells are not ink at that location, so they are dropped before the
collision audit; every real stroke, fill and text box is audited unchanged.
Reports go to ``output/figure_qa/``. Figure modules may declare ``fig.qa_alignment`` (a dict of
``exclude_axes`` / ``exemptions`` / ``groups``) for colorbars or intentional unequal layouts.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILL = Path(os.environ.get("NATURE_FIGURE_SCRIPTS", Path.home()/".claude/skills/nature-figure/scripts"))
REPORTS = ROOT/"output/figure_qa"
sys.path[:0] = [str(ROOT/"code/src"), str(ROOT/"code"), str(SKILL)]


def _require_skill():
    if not (SKILL/"audit_panel_alignment.py").exists():
        raise SystemExit(f"nature-figure skill scripts not found at {SKILL}; set NATURE_FIGURE_SCRIPTS")


def alignment_hook(fig, folder, name):
    from audit_panel_alignment import require_matplotlib_panel_alignment
    options = getattr(fig, "qa_alignment", {})
    report = require_matplotlib_panel_alignment(
        fig, json_out=REPORTS/f"{name}.alignment.json", overlay_svg=REPORTS/f"{name}.alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=False, strict=True, **options)
    print(f"[align] {name}: {report.get('verdict', report.get('status'))}")


def build_with_gates():
    import matplotlib
    matplotlib.use("Agg")
    from cjk_fonts import tofu_guard
    import vis.style as style
    from vis.build import build
    style.PREFLIGHT_HOOKS.append(alignment_hook)
    with tofu_guard():
        return build()


def _is_pattern_tile(drawing, fitz):
    r = drawing["rect"]; items = drawing.get("items", ()); w = (drawing.get("width") or 0)/2
    inner = fitz.Rect(r.x0+w, r.y0+w, r.x1-w, r.y1-w)
    grid = all(abs(v/36-round(v/36)) < 0.05 for v in (inner.x0, inner.y0, inner.x1, inner.y1))
    size_ok = any(abs(inner.width-t) < 3 for t in (72, 144)) and any(abs(inner.height-t) < 3 for t in (72, 144))
    kinds = {it[0] for it in items}
    lines = kinds == {"l"} and len(items) >= 8
    dots = kinds == {"c"} and len(items) >= 40
    square = len(items) == 1 and items[0][0] == "re"
    return grid and size_ok and (lines or dots or square)


def audit_pdfs(folder):
    import fitz
    import audit_figure_collisions as afc
    original = fitz.Page.get_drawings
    fitz.Page.get_drawings = lambda self, *a, **k: [d for d in original(self, *a, **k) if not _is_pattern_tile(d, fitz)]
    blocking = 0
    for pdf in sorted(Path(folder).glob("*.pdf")):
        text = subprocess.run([sys.executable, str(SKILL/"audit_pdf_text.py"), str(pdf), "--min-pt", "5"], capture_output=True, text=True)
        result = afc.audit_pdf(pdf)
        (REPORTS/f"{pdf.stem}.collision.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        if result["findings"]:
            afc.write_overlay_pdf(pdf, REPORTS/f"{pdf.stem}.collision.pdf", result["findings"])
        code = afc.exit_code(result, False); summary = result["summary"]
        print(f"[pdf] {pdf.name}: glyph-floor exit={text.returncode} | collisions fail={summary['fail']} warn={summary['warn']}")
        for finding in result["findings"]:
            if finding["severity"] != "INFO":
                print(f"   {finding['severity']} {finding['kind']}: {finding.get('text')!r} vs {finding.get('other_text')!r}")
        blocking += int(code == 1 or text.returncode == 1)
    return blocking


if __name__ == "__main__":
    _require_skill(); REPORTS.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[1] == "--audit":
        folder = Path(sys.argv[2])
    else:
        folder = Path(build_with_gates())/"figures"
    bad = audit_pdfs(folder)
    print(f"figure QA: {bad} PDF(s) with blocking findings; reports in {REPORTS}")
    raise SystemExit(1 if bad else 0)
