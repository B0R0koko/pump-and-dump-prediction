# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

Python 3.13 with Poetry.

- `poetry install` — install all dependencies
- `poetry run pytest -q` — run full test suite
- `poetry run pytest -q test/analysis/portfolio/test_price_impact.py` — run a single test module
- `just format-all` — format with Black (120-char line limit)
- `just pylint` — lint checks
- `just mypy` — static type checking
- `poetry run jupyter lab` — launch notebooks

## Architecture

This project detects cryptocurrency pump-and-dump schemes using ML on Binance market data. The pipeline flows:

**Raw data → Preprocessing → Features → Model training → Portfolio simulation → Analysis**

### Core modules

- **`core/`** — Shared domain types (`PumpEvent`, `CurrencyPair`, `FeatureType`, `NamedTimeDelta`, `Exchange`), column name constants, data path definitions, and time utilities. Used by all other modules.
- **`market_data/`** — Scrapy-based Binance parsers that download trade/kline data.
- **`preprocessing/`** — Transforms raw Binance trades into HIVE-partitioned parquet format.
- **`features/`** — `PumpsFeatureWriter` computes features (asset returns, flow imbalance, slippage, etc.) at multiple time offsets (5min to 14 days) per pump event.
- **`backtest/`** — The central system, containing three subsystems:
  - **`pipelines/`** — ML model implementations (`BasePipeline` → CatboostClassifier, CatboostClassifierSMOTE, CatboostClassifierTOPKAUC, CatboostRanker, LogisticRegression, RandomForest). Handles data splitting, preprocessing (cross-section standardization), training, and Optuna hyperparameter tuning.
  - **`portfolio/`** — Execution simulation with top-k portfolio construction, price impact modeling, VWAP estimation, and PnL calculation. `TOPKPortfolio` is the main orchestrator.
  - **`utils/`** — Dataset management (`build_dataset`, `sample`), evaluation metrics (`calculate_topk`, `calculate_topk_percent_auc`), and robustness testing.
- **`notebooks/`** — Research notebooks and analysis outputs.
- **`paper/`** — LaTeX manuscript (IEEE Access format).

### Key concepts

- **Cross-section**: For each pump event, all assets in the same time window form a cross-section. Models rank assets within each cross-section to predict the manipulation target.
- **Time splits**: Train < 2020-09-01, Validation 2020-09-01–2021-05-01, Test > 2021-05-01 (defined in `BasePipeline`).
- **Data paths**: `core/paths.py` expects datasets under `/var/lib/pumps/data`. Do not hardcode alternative paths.

## Code Style

- 4-space indent, type hints throughout, Black formatter (120-char lines)
- `snake_case` for new code; legacy CamelCase module names (e.g., `BasePortfolio.py`, `TOPKPortfolio.py`) — follow the surrounding style rather than renaming
- Interfaces use both ABCs (`BasePipeline`, `BaseModel`) and `typing.Protocol` (`QuoteToUSDTProvider`, `ExecutionImpactModel`)
- Tests go under `test/` mirroring runtime layout, named `test_*.py`. Use deterministic fixtures and `tmp_path`. Add regression tests when changing portfolio/impact/preprocessing behavior.
