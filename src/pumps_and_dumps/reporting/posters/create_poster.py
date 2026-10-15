#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["reportlab>=4", "qrcode>=8"]
# ///
"""Build the original deterministic Top-K results poster in Russian.

Run: uv run --no-project --script src/pumps_and_dumps/reporting/posters/create_poster.py
Reads poster/results.json and writes poster/poster_saimf_ru_A1_metrics.pdf.
Other existing poster editions and archived outputs are preserved.
"""

from pathlib import Path
import runpy


def main() -> None:
    source = Path(__file__).resolve()
    root = next(parent for parent in source.parents if (parent / "pyproject.toml").is_file())
    directory = root / "poster"
    renderer = runpy.run_path(str(source.parent / "archive" / "create_poster_metrics.py"))
    print(
        renderer["build_poster"](
            directory,
            output_path=directory / "poster_saimf_ru_A1_metrics.pdf",
        )
    )


if __name__ == "__main__":
    main()
