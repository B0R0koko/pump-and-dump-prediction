"""Keep manuscript figures near their discussion and audit compiled page gaps."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from build_highlighted import group_end

HERE = Path(__file__).resolve().parent
POLICY = r"""% BEGIN FLOAT LAYOUT
\usepackage{stfloats}
\usepackage{needspace}
\usepackage[nospread]{flushend}
\makeatletter
\def\@IEEEBIOskipN{10pt}
\makeatother
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{5}
\setcounter{dbltopnumber}{3}
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\floatpagefraction}{0.7}
\renewcommand{\dblfloatpagefraction}{0.7}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\dblfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{10pt plus 2pt minus 2pt}
% END FLOAT LAYOUT
"""
BARRIER = r"\FloatBarrier % local figure placement" + "\n"
KEEP_RESULTS = r"\Needspace{6\baselineskip} % keep portfolio results together" + "\n"
BIO_START = "\\noindent\\begin{minipage}{\\columnwidth} % compact biography\n"
BIO_END = "\\end{minipage}\\par % compact biography\n"
FLOAT = re.compile(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", re.S)


def strip_layout(source):
    source = re.sub(r"% BEGIN FLOAT LAYOUT\n.*?% END FLOAT LAYOUT\n", "", source, flags=re.S)
    source = source.replace(BARRIER, "")
    source = source.replace(KEEP_RESULTS, "")
    source = source.replace(BIO_START, "").replace(BIO_END, "")
    return re.sub(r"\\label\{firstref:fig:[^}]+\}", "", source)


def apply_layout(source):
    source = strip_layout(source)
    source = source.replace(r"\usepackage{placeins}" + "\n", r"\usepackage{placeins}" + "\n" + POLICY)
    # Wide figures can use both ends of a page instead of queuing for page tops.
    source = re.sub(r"(\\begin\{(?:figure|table)\*?\})\[[^]]*\]", r"\1[!tbp]", source)
    # Let prose continue around deferred wide plots. Barriers before the
    # portfolio and limitations sections otherwise leave partly empty columns.
    for title in ("Model Comparison", "Conclusion"):
        heading = rf"\section{{{title}}}"
        source = source.replace(heading, BARRIER + heading)
    # Keep the impact examples ahead of their portfolio-performance discussion.
    heading = r"\subsubsection{Portfolio performance with price impact}"
    source = source.replace(heading, BARRIER + heading)
    # Keep the first portfolio-result reference with enough of its paragraph to
    # avoid placing it two pages ahead of the plot behind the uncertainty floats.
    reference = r"Table~\ref{table:performance_by_k} reports portfolio statistics"
    source = source.replace(reference, KEEP_RESULTS + reference, 1)
    # Natural-height boxes prevent the class's elastic biography spacing from
    # stretching. Keep each portrait with its full biography, then balance the
    # final page instead of forcing the last author into a separate column.
    start = source.index(r"\begin{IEEEbiography}")
    biographies = source[start:].replace("\\newpage\n", "")
    biographies = re.sub(r"\\begin\{IEEEbiography\}.*?\\end\{IEEEbiography\}\n",
                         lambda match: BIO_START + match[0] + BIO_END,
                         biographies, flags=re.S)
    source = source[:start] + biographies
    figures = [re.search(r"\\label\{([^}]+)\}", m[0])[1]
               for m in FLOAT.finditer(source) if m[1].startswith("figure")]
    for label in figures:
        reference = rf"\ref{{{label}}}"
        assert reference in source, f"Missing in-text figure reference: {label}"
        source = source.replace(reference, reference + rf"\label{{firstref:{label}}}", 1)
    return source


def audit_pages():
    report = {"status": "PASS", "rule": "Every figure is on its first-reference page or the following page", "documents": {}}
    failures = []
    for name in ("access", "access_highlighted"):
        source = (HERE / f"{name}.tex").read_text()
        auxiliary = (HERE / f"{name}.aux").read_text()
        labels = {}
        for match in re.finditer(r"\\newlabel\{((?:firstref:)?fig:[^}]+)\}\{", auxiliary):
            start = match.end()
            end = group_end(auxiliary, start)
            number = auxiliary[start + 1:end - 1]
            page_end = group_end(auxiliary, end)
            labels[match[1]] = (number, int(auxiliary[end + 1:page_end - 1]))
        figures = [label for label in labels if label.startswith("fig:")]
        assert len(figures) == len([m for m in FLOAT.finditer(source) if m[1].startswith("figure")])
        rows = []
        for label in figures:
            number, page = labels[label]
            reference_page = labels[f"firstref:{label}"][1]
            gap = page - reference_page
            rows.append({"label": label, "number": number, "first_reference_page": reference_page,
                         "figure_page": page, "page_gap": gap})
            if gap not in (0, 1):
                failures.append(f"{name}: {label}, reference p.{reference_page}, figure p.{page}")
        report["documents"][name] = {
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "pdf_sha256": hashlib.sha256((HERE / f"{name}.pdf").read_bytes()).hexdigest(),
            "figures": rows,
        }
    report["status"] = "FAIL" if failures else "PASS"
    (HERE.parents[1] / "docs/manuscript_layout_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise AssertionError("\n".join(failures))
    print(f"PASS: {len(figures)} figures in each PDF appear on their first-reference page or the next page.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-pages", action="store_true", required=True)
    parser.parse_args()
    audit_pages()
