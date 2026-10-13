#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["reportlab>=4", "matplotlib>=3.9", "svglib>=1.5", "qrcode>=8"]
# ///
"""Build the current English scientific poster from saved research outputs.

Run: uv run --no-project --script src/pumps_and_dumps/reporting/posters/create_poster.py
Writes poster/poster_saimf_en_A1_scientific.pdf and a result snapshot.
The original Russian edition and older archived editions are preserved.
"""

from pathlib import Path
import runpy


def main() -> None:
    runpy.run_path(
        str(Path(__file__).with_name("create_scientific_poster.py")),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
