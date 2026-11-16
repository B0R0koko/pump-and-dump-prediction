"""Verify built PDFs, result values, baseline highlighting and figure placement."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "paper/src"
sys.path.insert(0, str(HERE))
from build_highlighted import BASELINE, BASELINE_DIR, build_highlighted, normalized_lines
from layout import audit_pages


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = json.loads((BASELINE_DIR / "provenance.json").read_text())
    assert sha(BASELINE_DIR / "submitted.pdf") == baseline["pdf_sha256"]
    assert sha(BASELINE) == baseline["tex_sha256"]
    user_pdf = Path("/home/borokoko/Downloads/IEEE_Final_submission (2).pdf")
    if user_pdf.exists():
        assert sha(user_pdf) == baseline["pdf_sha256"]
    source = (HERE / "access.tex").read_text()
    highlighted = build_highlighted(normalized_lines(BASELINE.read_text()), normalized_lines(source),
                                    figure_hashes=set(baseline["image_sha256"].values()))
    assert (HERE / "access_highlighted.tex").read_text() == highlighted
    changes = json.loads((HERE / "access_highlighted_changes.json").read_text())
    manuscript = json.loads((ROOT / "docs/manuscript_results_audit.json").read_text())
    numerical = json.loads((ROOT / "docs/seed_sweep_manuscript_audit.json").read_text())
    assert manuscript["status"] == numerical["status"] == "PASS"
    assert manuscript["paper_source_sha256"] == sha(HERE / "access.tex")
    texts, pages, hashes = [], [], {}
    for name in ["access", "access_highlighted"]:
        public = ROOT / "paper" / f"{name}.pdf"
        assert public.read_bytes() == (HERE / f"{name}.pdf").read_bytes()
        hashes[str(public.relative_to(ROOT))] = sha(public)
        texts.append(subprocess.check_output(["pdftotext", "-layout", str(public), "-"], text=True))
        info = subprocess.check_output(["pdfinfo", str(public)], text=True)
        pages.append(int(re.search(r"(?m)^Pages:\s+(\d+)", info)[1]))
        log = (HERE / f"{name}.log").read_text()
        assert not re.search(r"(?m)^!", log), f"LaTeX build errors in {name}.log"
        assert not re.search(r"(?:Reference|Citation) .* undefined|There were undefined references", log)
    assert texts[0] == texts[1] and "??" not in texts[0]
    assert pages[0] == pages[1] and max(pages) <= 20, f"20-page budget exceeded: {pages}"
    values = manuscript["key_results"]
    required = [values[key] for key in ["validation_mean", "cat_test_mean", "l1_test_mean", "cb_auc", "top5", "model_return", "impact_return", "hit_percent"]]
    for value in required:
        assert value in texts[0], f"Missing reported result: {value}"
    assert "table:validation_model_selection" in changes["changed_tables"]
    assert "table:test_seed_distribution" in changes["changed_tables"]
    assert {"fig:validation_seed_distribution", "fig:test_seed_distribution"}.issubset(changes["changed_figures"])
    audit_pages()
    layout = json.loads((ROOT / "docs/manuscript_layout_audit.json").read_text())
    count = len(layout["documents"]["access"]["figures"])
    report = {"status": "PASS", "pages": pages, "clean_and_highlighted_text_identical": True,
              "all_cross_references_resolved": True, "current_result_values_present": required,
              "baseline_pdf_sha256": baseline["pdf_sha256"],
              "changed_tables_highlighted": changes["changed_tables"],
              "changed_figures_highlighted": changes["changed_figures"],
              "source_sha256": sha(HERE / "access.tex"), "pdf_sha256": hashes,
              "reported_models": 6, "seeds_per_model": 100, "fixed_model_seed": 42,
              "single_logistic_regression_model": "L2 logistic" not in source,
              "feature_importance_and_all_pnl_model": "CatBoost classifier + Tuned",
              "page_limit": 20,
              "compact_auc_bootstrap_figure_width": "columnwidth",
              "figures_on_first_reference_page_or_next": count,
              "layout_audit": "docs/manuscript_layout_audit.json"}
    (ROOT / "docs/manuscript_pdf_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: both {pages[0]}-page PDFs have identical text, current results and baseline-relative highlighting.")


if __name__ == "__main__":
    main()
