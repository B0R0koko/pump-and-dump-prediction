"""Revision highlighting preserves TeX structure and marks deleted table content."""

import pytest

from paper.src.build_highlighted import build_highlighted, normalized_lines


def test_multiline_captions_preserve_escaped_percent_and_following_text():
    text = "\\caption{First % comment\nsecond 95\\%\nline}\nFollowing paragraph"
    assert normalized_lines(text) == [r"\caption{First second 95\% line}", "Following paragraph"]
    with pytest.raises(ValueError, match="Unclosed caption"):
        normalized_lines("\\caption{Never closed\nlast line")


def test_removed_table_rows_mark_caption_without_recoloring_surviving_cells():
    before = [r"\begin{document}", r"\begin{table}", r"\caption{Scores}",
              r"\label{scores}", r"A & 0.5 \\", r"B & 0.4 \\", r"\end{table}", r"\end{document}"]
    after = [line for line in before if not line.startswith("B &")]
    audit = {}
    highlighted = build_highlighted(before, after, audit=audit)
    assert r"\caption{\textcolor{revisionblue}{Scores}}" in highlighted
    assert r"A & 0.5 \\" in highlighted
    assert "B &" not in highlighted
    assert audit["changed_tables"] == ["scores"]


def test_comment_only_deletions_do_not_mark_surviving_text():
    before = [r"\begin{document}", "% old comment", "Unchanged text", r"\end{document}"]
    after = [line for line in before if not line.startswith("%")]
    audit = {}
    highlighted = build_highlighted(before, after, audit=audit)
    assert "Unchanged text\n" in highlighted
    assert not audit["marked_lines"]
