"""Highlight the current paper against the verified IEEE submission.

Blue marks revised text, including headings, references, and table cells.
Changed tables have blue captions (including tables with deleted rows).
Changed figures have blue borders and captions without recoloring their data.
Pure text deletions mark the remaining paragraph or the following text in blue.
The JSON audit records removed source text without restoring it to the paper.
"""

from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BASELINE_DIR = HERE.parent / "baselines/ieee_final_submission"
BASELINE = BASELINE_DIR / "access.tex"


def group_end(text, start, opening="{", closing="}"):
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "\\":
            continue
        if index and text[index - 1] == "\\":
            continue
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError(f"Unbalanced TeX group: {text[start:]}")


def tex_tokens(text):
    """Keep math and complete macro arguments intact when comparing words."""
    result = []
    index = 0
    while index < len(text):
        start = index
        char = text[index]
        if char == "$":
            index += 1
            while index < len(text) and (text[index] != "$" or text[index - 1] == "\\"):
                index += 1
            index += 1
        elif char == "\\":
            command = re.match(r"\\(?:[a-zA-Z]+\*?|.)", text[index:])
            index += len(command.group())
            while index < len(text) and text[index] in "[{":
                opening = text[index]
                index = group_end(text, index, opening, "]" if opening == "[" else "}")
        elif char == "{":
            index = group_end(text, index)
        else:
            token = re.match(r"\s+|[^\s\\${}]+", text[index:])
            if token is None:
                raise ValueError(f"Unexpected TeX token: {text[index:]}")
            index += len(token.group())
        result.append(text[start:index])
    return result


def blue(text):
    content = text.strip()
    if not content:
        return text
    start = len(text) - len(text.lstrip())
    return text[:start] + r"\textcolor{revisionblue}{" + content + "}" + text[start + len(content):]


def highlight_words(before, after):
    old_tokens, new_tokens = tex_tokens(before), tex_tokens(after)
    output = []
    matcher = SequenceMatcher(lambda token: token.isspace(), old_tokens, new_tokens, autojunk=False)
    for tag, _, _, start, end in matcher.get_opcodes():
        fragment = "".join(new_tokens[start:end])
        output.append(blue(fragment) if tag in {"insert", "replace"} else fragment)
    highlighted = "".join(output)
    # A deletion-only edit otherwise leaves no visible revision mark.
    if before.strip() != after.strip() and highlighted == after:
        return blue(after)
    return highlighted


def normalized_lines(text):
    """Join multiline captions so they are compared as complete TeX groups."""
    lines = text.splitlines()
    output = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.lstrip().startswith(r"\caption{"):
            caption = re.sub(r"(?<!\\)%.*$", "", line).rstrip()
            start = caption.index("{")
            while True:
                try:
                    group_end(caption, start)
                    break
                except ValueError:
                    index += 1
                    if index >= len(lines):
                        raise ValueError("Unclosed caption") from None
                    addition = re.sub(r"(?<!\\)%.*$", "", lines[index]).strip()
                    caption += " " + addition
            line = caption
        output.append(line)
        index += 1
    return output


def floats(lines):
    """Return complete table/figure blocks, including their labels."""
    blocks = []
    start = None
    kind = None
    for index, line in enumerate(lines):
        begin = re.match(r"\s*\\begin\{(table\*?|figure\*?)\}", line)
        if begin:
            start, kind = index, begin.group(1)
        if start is not None and line.strip() == rf"\end{{{kind}}}":
            body = lines[start:index + 1]
            labels = re.findall(r"\\label\{([^}]+)\}", "\n".join(body))
            blocks.append({"start": start, "end": index, "kind": kind,
                           "label": labels[-1] if labels else "", "lines": body})
            start = kind = None
    return blocks


def table_body(block):
    return [line.strip() for line in block["lines"]
            if "&" in line or re.search(r"\\(?:toprule|midrule|bottomrule)", line)]


def image_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_row(line):
    return " & " in line and line.rstrip().endswith(r"\\")


def matching_line(line, candidates):
    candidates = [candidate for candidate in candidates if bool(candidate.strip()) and is_row(candidate) == is_row(line)]
    if is_row(line):
        key = line.split(" & ", 1)[0].strip()
        candidates = [candidate for candidate in candidates if candidate.split(" & ", 1)[0].strip() == key]
    if not candidates:
        return ""
    return max(candidates, key=lambda candidate: SequenceMatcher(None, candidate, line, autojunk=False).ratio())


def highlight_line(before, after, in_equation):
    stripped = after.strip()
    if not stripped or stripped.startswith("%"):
        return after
    caption = re.match(r"(\s*\\(?:caption|paragraph|section|subsection|subsubsection)\*?\{)(.*)(\})$", after)
    if caption:
        old_caption = re.match(r"\s*\\(?:caption|paragraph|section|subsection|subsubsection)\*?\{(.*)\}$", before)
        previous = old_caption.group(1) if old_caption else ""
        return caption.group(1) + highlight_words(previous, caption.group(2)) + caption.group(3)
    prefix = re.match(r"(\s*\\(?:item(?:\[[^]]*\])?|bibitem(?:\[[^]]*\])?\{[^}]+\})\s*)(.*)$", after)
    if prefix:
        old_prefix = re.match(r"\s*\\(?:item(?:\[[^]]*\])?|bibitem(?:\[[^]]*\])?\{[^}]+\})\s*(.*)$", before)
        previous = old_prefix.group(1) if old_prefix else ""
        return prefix.group(1) + highlight_words(previous, prefix.group(2))
    if is_row(after):
        old_cells = re.split(r"(?<!\\)&", before.strip()[:-2]) if is_row(before) else []
        cells = re.split(r"(?<!\\)&", stripped[:-2])
        highlighted = [highlight_words(old_cells[i] if i < len(old_cells) else "", cell.strip()) for i, cell in enumerate(cells)]
        return after[:len(after) - len(after.lstrip())] + " & ".join(highlighted) + r" \\"
    if in_equation and not stripped.startswith((r"\begin", r"\end", r"\label")):
        return r"\color{revisionblue} " + after
    # Multiline table headers retain their alignment delimiters outside color groups.
    if stripped.startswith(r"\makecell"):
        ending = re.search(r"\s*(&|\\\\)\s*$", after)
        if ending:
            old_content = re.sub(r"\s*(&|\\\\)\s*$", "", before)
            return highlight_words(old_content, after[:ending.start()]) + after[ending.start():]
    # Grouping a standalone font declaration in \textcolor would cancel its
    # effect on subsequent text and change table dimensions and pagination.
    structural = r"\s*\\(?:begin|end|label|centering|setlength|renewcommand|toprule|midrule|bottomrule|hline|resizebox|includegraphics|FloatBarrier|newpage|clearpage|vspace|hspace|EOD|tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge|bfseries|mdseries|itshape|upshape|rmfamily|sffamily|ttfamily)\b"
    if re.match(structural, after) or stripped.startswith(("}", "&")):
        return after
    if stripped in {r"\\", r"\\ \\"}:
        return after
    return highlight_words(before, after)


def build_highlighted(before, after, *, audit=None, figure_hashes=None):
    previous = {}
    forced = set()
    changes = []
    opcodes = SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
    for tag, a, b, c, d in opcodes:
        if tag != "equal":
            changes.append({"kind": tag, "baseline_lines": [a + 1, b],
                            "current_lines": [c + 1, d],
                            "removed": before[a:b], "replacement": after[c:d]})
        if tag in {"insert", "replace"}:
            for index in range(c, d):
                previous[index] = matching_line(after[index], before[a:b])
        elif tag == "delete":
            # Mark the next surviving visible text; table captions are handled below.
            if any(line.strip() and not line.lstrip().startswith("%") for line in before[a:b]):
                for index in range(c, len(after)):
                    stripped = after[index].strip()
                    if stripped and (not stripped.startswith("\\") or re.match(
                            r"\\(?:section|subsection|subsubsection|caption|textbf|item)\b", stripped)):
                        forced.add(index)
                        break

    old_tables = {block["label"]: block for block in floats(before) if block["kind"].startswith("table")}
    changed_tables = []
    changed_figures = []
    figure_lines = set()
    for block in floats(after):
        if block["kind"].startswith("table"):
            label = block["label"]
            old = old_tables.get(label)
            if old is None or table_body(old) != table_body(block):
                changed_tables.append(label)
                for index in range(block["start"], block["end"] + 1):
                    if after[index].lstrip().startswith(r"\caption{"):
                        forced.add(index)
                    if old is None and (" & " in after[index] or after[index].lstrip().startswith(r"\makecell")):
                        previous[index] = ""
        elif figure_hashes is not None:
            changed = False
            for index in range(block["start"], block["end"] + 1):
                match = re.match(r"\s*\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", after[index])
                if match and image_digest(HERE / match.group(1)) not in figure_hashes:
                    figure_lines.add(index)
                    changed = True
            if changed:
                changed_figures.append(block["label"])
                for index in range(block["start"], block["end"] + 1):
                    if after[index].lstrip().startswith(r"\caption{"):
                        forced.add(index)
    output = []
    in_document = in_equation = False
    marked_lines = []
    for index, line in enumerate(after):
        stripped = line.strip()
        if stripped == r"\begin{document}":
            output.extend([
                r"\usepackage{color}",
                r"\definecolor{revisionblue}{RGB}{0,70,170}",
                r"\definecolor{REVISIONBLUE}{RGB}{0,70,170}",
                r"\newcommand{\revisionfigure}[1]{{\setlength{\fboxsep}{0pt}\setlength{\fboxrule}{0.8pt}\smash{\rlap{\hspace{-\fboxrule}\fcolorbox{revisionblue}{white}{\phantom{#1}}}}#1}}",
                "% Baseline: paper/baselines/ieee_final_submission/submitted.pdf (user-supplied IEEE_Final_submission (2).pdf).",
                "% Blue text marks revisions; blue captions also flag changed tables and figures.",
                "% Blue borders flag changed figure images. Removed text is recorded in access_highlighted_changes.json.",
            ])
            in_document = True
        if stripped.startswith(r"\begin{equation}"):
            in_equation = True
        if index in figure_lines:
            indent = line[:len(line) - len(line.lstrip())]
            line = indent + r"\revisionfigure{" + line.strip() + "}"
        elif in_document and index in forced:
            line = highlight_line("", line, in_equation)
        elif index in previous and in_document:
            line = highlight_line(previous[index], line, in_equation)
        if line != after[index]:
            marked_lines.append(index + 1)
        if stripped.startswith(r"\end{equation}"):
            in_equation = False
        output.append(line)
    if audit is not None:
        audit.update({"baseline_pdf": str(BASELINE_DIR / "submitted.pdf"),
                      "baseline_pdf_sha256": image_digest(BASELINE_DIR / "submitted.pdf"),
                      "line_numbering": "TeX source after joining multiline captions",
                      "marked_lines": marked_lines,
                      "changed_tables": changed_tables, "changed_figures": changed_figures,
                      "changes": changes})
    return "\n".join(output) + "\n"


if __name__ == "__main__":
    provenance = json.loads((BASELINE_DIR / "provenance.json").read_text())
    assert image_digest(BASELINE_DIR / "submitted.pdf") == provenance["pdf_sha256"]
    assert image_digest(BASELINE) == provenance["tex_sha256"]
    before = normalized_lines(BASELINE.read_text())
    after = normalized_lines((HERE / "access.tex").read_text())
    figure_hashes = set(provenance["image_sha256"].values())
    audit = {}
    (HERE / "access_highlighted.tex").write_text(build_highlighted(before, after, audit=audit, figure_hashes=figure_hashes))
    (HERE / "access_highlighted_changes.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Highlighted {len(audit['marked_lines'])} source lines, "
          f"{len(audit['changed_tables'])} changed tables, and {len(audit['changed_figures'])} changed figures.")
