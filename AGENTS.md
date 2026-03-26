# Repository Guidelines

## Project Structure & Module Organization
`backtest/` contains model pipelines, portfolio simulation, and robustness analysis. `core/` holds shared domain types and path/time utilities. `features/` and `preprocessing/` cover feature generation and raw-data transformation. `market_data/` contains exchange parsers. `notebooks/` stores research notebooks plus generated figures and analysis outputs. `paper/` contains the LaTeX manuscript and paper images. Tests live under `test/` and mostly mirror the runtime package layout.

## Build, Test, and Development Commands
Use Python 3.13 with Poetry.

- `poetry install`: install runtime and dev dependencies.
- `poetry run pytest -q`: run the full test suite.
- `poetry run pytest -q test/analysis/portfolio/test_price_impact.py`: run one focused test module.
- `just format-all`: format `core`, `features`, `market_data`, `preprocessing`, `backtest`, and `test` with Black.
- `just pylint`: run lint checks across the main packages.
- `just mypy`: run static typing checks.
- `poetry run jupyter lab`: open notebooks for local analysis work.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints, and small docstrings where behavior is not obvious. Black is the formatter, with a 120-character line limit. Prefer `snake_case` for new functions, variables, and test files. This repository also has legacy CamelCase module names such as `BasePortfolio.py` and `TOPKPortfolio.py`; follow the surrounding style in the directory you are editing rather than renaming files opportunistically.

## Testing Guidelines
Tests use `pytest` with `pythonpath = ["."]` configured in `pyproject.toml`. Add tests under the matching `test/...` area and name them `test_*.py`. Prefer deterministic synthetic fixtures and `tmp_path` for filesystem-backed cases. When changing portfolio execution, impact modeling, preprocessing, or notebook-backed analysis, add or update regression-style tests that pin the expected behavior.

## Commit & Pull Request Guidelines
Recent history uses short, direct commit messages such as `added portfolio strategy optimization`. Keep messages concise and descriptive; avoid `WIP` in shared history. Pull requests should explain the problem, the behavioral change, and the validation commands you ran. If notebook or paper outputs change, include the affected paths and before/after figures where relevant.

## Data & Configuration Notes
`core/paths.py` expects local datasets under `/var/lib/pumps/data`. Do not hardcode alternative machine-specific paths in feature, backtest, or notebook code. Commit generated artifacts only when they are intentional deliverables, not incidental local outputs.
