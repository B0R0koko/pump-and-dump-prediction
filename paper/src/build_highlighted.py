"""Highlight revised words in blue against the pre-review manuscript."""

from difflib import SequenceMatcher
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent.parent / "research_snapshots/submitted_results/access.tex"


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
    return "".join(output)


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
    caption = re.match(r"(\s*\\(?:caption|paragraph)\{)(.*)(\})$", after)
    if caption:
        old_caption = re.match(r"\s*\\(?:caption|paragraph)\{(.*)\}$", before)
        previous = old_caption.group(1) if old_caption else ""
        return caption.group(1) + highlight_words(previous, caption.group(2)) + caption.group(3)
    if is_row(after):
        old_cells = re.split(r"(?<!\\)&", before.strip()[:-2]) if is_row(before) else []
        cells = re.split(r"(?<!\\)&", stripped[:-2])
        highlighted = [highlight_words(old_cells[i] if i < len(old_cells) else "", cell.strip()) for i, cell in enumerate(cells)]
        return after[:len(after) - len(after.lstrip())] + " & ".join(highlighted) + r" \\"
    if in_equation and not stripped.startswith((r"\begin", r"\end", r"\label")):
        return r"\color{revisionblue} " + after
    if stripped.startswith(("\\", "}", "&")):
        return after
    return highlight_words(before, after)


def build_highlighted(before, after):
    previous = {}
    for tag, a, b, c, d in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if tag in {"insert", "replace"}:
            for index in range(c, d):
                previous[index] = matching_line(after[index], before[a:b])
    output = []
    in_document = in_equation = False
    for index, line in enumerate(after):
        stripped = line.strip()
        if stripped == r"\begin{document}":
            output.extend([
                r"\usepackage{color}",
                r"\definecolor{revisionblue}{RGB}{0,70,170}",
                "% Blue marks revised words/numbers relative to before_reviewer2_20261102.",
                "% Superseded text is omitted; updated figures retain their original colors.",
            ])
            in_document = True
        if stripped.startswith(r"\begin{equation}"):
            in_equation = True
        if index in previous and in_document:
            line = highlight_line(previous[index], line, in_equation)
        if stripped.startswith(r"\end{equation}"):
            in_equation = False
        output.append(line)
    return "\n".join(output) + "\n"


if __name__ == "__main__":
    before = BASELINE.read_text().splitlines()
    after = (HERE / "access.tex").read_text().splitlines()
    (HERE / "access_highlighted.tex").write_text(build_highlighted(before, after))
