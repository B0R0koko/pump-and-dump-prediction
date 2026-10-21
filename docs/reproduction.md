# Reproduction guide

[README](../README.md) · [Full results](results.md)

Run commands from the repository root. Research notebooks need Git and Python 3.13+. The data installer needs Python 3.10+ and Docker; its published containers target Linux/amd64.

## Environment

```bash
./scripts/install.sh --setup-environment --no-update
source .venv/bin/activate
```

Setup installs runtime and development dependencies from `pyproject.toml`, verifies editable imports and dependency consistency, and registers **Pumps and dumps (.venv)** for Jupyter. It reuses an existing valid `.venv`. Logs go to `logs/setup-environment-<timestamp>-<pid>.log`.

`--no-update` keeps the current checkout. Without it, setup fast-forwards a clean tracking branch, skips pulling when local edits exist, and stops on detached or diverged branches. Use `--python /path/to/python3.13` to choose an interpreter or `--dry-run` to preview setup.

Poetry is an alternative:

```bash
poetry install
poetry run jupyter lab
```

When using Poetry, prefix the Python, Jupyter and pytest commands below with `poetry run`. Imports use the installed `pumps_and_dumps` package; no `PYTHONPATH` edits are needed.

## Data pipeline

The interactive installer selects a data directory and stages:

```bash
./scripts/install.sh
```

For selected stages or a noninteractive run:

```bash
./scripts/install.sh --all --data-dir /mnt/market-data --no-tui
./scripts/install.sh --stages parquet features --data-dir /mnt/market-data
./scripts/install.sh --stages trades candles --data-dir /mnt/market-data --dry-run
./scripts/install.sh --help
```

Stages run in order: trade archives, BTC/USDT candles, Hive Parquet, features. Each selected stage runs once; prerequisites must already exist when earlier stages are omitted. Conversion and reading of the same Hive partitions must run sequentially. `--dry-run` starts no jobs and writes no data or configuration. `Ctrl-C` cancels a run; stage logs are saved under the selected data directory's `logs/install-<timestamp>-<pid>/`.

The default download range is **2018-01-01 inclusive to 2025-01-01 exclusive**, using the 488 historical BTC pairs in `resources/btc_currencies.json`. Download the candidate universe and enough history for the 30-day feature normalizers, plus outcome windows. A short sample download cannot reproduce the research dataset.

`--start` / `--end` override archive dates; `--hive-workers` and `--feature-workers` override worker counts. Dates are UTC, with an inclusive start and exclusive end. Archive reuse verifies file sizes and complete ZIP CRCs, repairing missing or damaged files. Features are recomputed by default; use `--skip-existing-features` only when inputs and feature definitions are unchanged.

The chosen data directory is stored in gitignored `.pumps-and-dumps.json`. `PUMPS_AND_DUMPS_DATA_DIR` overrides it for a process. Changing the path does not move files; restart notebook kernels afterward.

```text
<data directory>/
├── raw/binance/spot/trades/          # Daily trade ZIPs
├── raw/binance/spot/klines/          # BTC/USDT candles
├── transformed/binance/spot/trades/  # date=.../symbol=... Hive partitions
├── features/pumps/                  # One feature file per event
├── features/logs/                   # Input timestamp audits
└── studies_split_60_10_20_10_v1.db    # Current tuning studies
```

### Configuration and direct Docker use

Stages accept HOCON or JSON through `--config`. Examples are in [configs/](../configs/): `trades.conf`, `klines.conf`, `hive.conf`, `features.conf` and `features-inline.conf`. CLI options override config values, which override defaults. Relative paths in configs resolve beside the config file; relative CLI paths resolve from the working directory.

The installer accepts `--trades-config`, `--candles-config`, `--parquet-config` and `--features-config`. It mounts each config's directory read-only at `/config`; use `/data/...` for container data paths. This example uses the bundled one-day trade and conversion jobs:

```bash
./scripts/install.sh --stages trades parquet --data-dir /mnt/market-data \
  --trades-config configs/trades.conf --parquet-config configs/hive.conf --no-tui
```

For direct container use, share a writable `/data` mount across stages:

| Image | Default command |
| --- | --- |
| `b0r0koko/pds-dataloader:latest` | `pumps-download-trades` |
| `b0r0koko/pds-preprocessor:latest` | `pumps-write-hive` |
| `b0r0koko/pds-feature-writer:latest` | `pumps-compute-features` |

For example, after preparing the required Hive data:

```bash
docker run --rm --init --network none \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=/mnt/market-data,target=/data" \
  --mount "type=bind,source=$(pwd)/configs,target=/config,readonly" \
  b0r0koko/pds-feature-writer:latest --config /config/features.conf
```

Use `--entrypoint pumps-download-klines` on the downloader for candles, and `--help` on any command for all options. The same commands work in the local research environment. The installer pulls images before running stages; direct container commands use Docker's normal image policy.

Feature configs accept either an inline `pumps` list or `pumps-file`. Each event supplies `symbol`, `exchange` and a UTC `time` at whole-second precision. Feature `--start` / `--end` filter announcement times while retaining earlier events for historical features. `--no-skip-existing` overrides configured feature reuse and produces fresh input audits.

To build local images, run the corresponding Dockerfile from the repository root and substitute your tag in direct Docker commands:

```bash
docker build -f docker/dataloader/Dockerfile -t pumps-and-dumps-dataloader:local .
docker build -f docker/preprocessor/Dockerfile -t pumps-and-dumps-preprocessor:local .
docker build -f docker/feature-writer/Dockerfile -t pumps-and-dumps-feature-writer:local .
```

Each image installs its own `requirements.txt` and the project without notebook/training dependencies. Container builds do not use `poetry.lock`; retain the verification environment when replaying recorded results.

## Event labels and features

[resources/pumps.json](../resources/pumps.json) contains 517 usable events. [pump_event_evidence.json](../resources/pump_event_evidence.json) records reviewed additions and the recovered YOYO event; [pump_event_exclusions.json](../resources/pump_event_exclusions.json) preserves duplicate and out-of-scope records. Events are not filtered by post-announcement profitability. The original labels have not all been comprehensively reaudited; historical review artifacts remain under `data/research/`.

For an announcement at T, candidate eligibility requires an actual trade in `[T−1 day−15min, T−15min)`. The [feature schema](../src/pumps_and_dumps/features/schema.py) defines 81 regressors: eight families at ten horizons, plus prior-pump count. All predictor inputs precede T−15min. `target_return@...` columns are outcomes and are excluded from model inputs.

Current feature files use schema version 4. The loader accepts version-2/3 files containing the same required predictors and discards additional columns; unversioned or incomplete files are rejected. Changes to inputs or definitions require feature regeneration, a fresh notebook kernel and retuning.

## Feature audits

Every feature-writer run saves JSON-lines logs in `<data directory>/features/logs/feature_audit_<run>.log`. Each computed pump/asset has an `asset_inputs` record containing the decision cutoff, latest predictor input (`feature_max_time`), earliest target input (`target_min_time`), and two checks:

- `features_before_cutoff`: all predictor inputs precede T−15min.
- `non_overlapping`: `feature_max_time < target_min_time`; equality counts as overlap.

`OK` requires complete provenance and both checks. `LEAKAGE` indicates an input at or after the cutoff; `OVERLAP` indicates crossing predictor/target boundaries, which can also arise from a shared historical starting price. `UNKNOWN` and `NO_DATA` indicate missing provenance or data. Cached features are marked `reused` with `audit_provenance="not_recomputed"`; they do not receive a fresh audit.

To choose a log path and inspect checks requiring review:

```bash
python -m pumps_and_dumps.features.writer --cpu-count 4 --audit-log /tmp/feature-audit.log
jq 'select(.record_type == "asset_inputs" and .status != "OK")' /tmp/feature-audit.log
```

Automated perturbation tests complement timestamp logs: see [tests/leakage/](../tests/leakage/) and [the full feature-writer causality test](../tests/features/test_feature_writer_causality.py).

## Training and evaluation

The [research notebook](../notebooks/research_notebook.ipynb) refits seven candidates from saved Optuna parameters. It selects the largest comparison-validation Top@k%AUC before evaluating test results. The chronological partitions contain 310 training, 52 tuning/early-stopping, 103 comparison and 52 test events; complete cross-sections and simultaneous events remain together.

For a fresh data directory without studies, run:

```bash
python -m pumps_and_dumps.backtest.research.retune \
  --output-dir data/research_runs/split_60_10_20_10_v1 \
  --trials 100
```

This uses the selected data directory's `studies_split_60_10_20_10_v1.db` and refuses to overwrite an existing database. For another search, pass `--storage sqlite:////absolute/new/studies.db`, use a separate output directory, and set the notebook's `STUDY_STORAGE` to that same URL. Each search records parameters, split membership, logs and fitted artifacts. Retune after changing the data, predictors or split policy.

Then launch Jupyter and run the notebook from a fresh kernel:

```bash
jupyter lab notebooks/research_notebook.ipynb
```

Outputs go under `notebooks/analysis_outputs/split_60_10_20_10_v1/deterministic_selection/`: metric tables, bootstrap predictions/draws/comparisons, robustness runs, plots and fit logs. Matplotlib images are displayed once and copied unchanged to `paper/src/images/`. The source notebook has no saved outputs; the verified executed copy is `research_notebook.executed.ipynb` in the output directory.

Bootstrap inference uses the same event draws across fixed models, with 10,000 replicates and seed 42. Holm correction covers 21 validation AUC comparisons and all 84 test pair/metric hypotheses. Precision, recall and F1 coincide under Top@1 but remain in that family. [Results and inference limits](results.md#statistical-significance) describe the findings.

Portfolios use the validation-selected model and equal weights. Missing fills retain their cash allocation. Entry is the first trade in `[T−15min, T)`; other assets exit at or after T, and the manipulated asset at or after T+1min. Returns use fixed event notional, USDT conversion and 25 bp round-trip costs. Impact fits use only data strictly before the transaction being priced; the separate size sweep uses `I_vwap(Q) = (2/3) × β × sqrt(Q)`.

## Build the paper

With `just` and a LaTeX installation available:

```bash
just paper
python paper/src/build_highlighted.py
just paper-highlighted
```

Builds write to `paper/src/`. To refresh the shareable copies after checking the build:

```bash
cp paper/src/access.pdf paper/access.pdf
cp paper/src/access_highlighted.pdf paper/access_highlighted.pdf
```

Sources are in `paper/src/`, figures in `paper/src/images/`, and the IEEE Access class/fonts in `paper/styles/`. The manuscript's inline tables are traceable to `paper/src/results_20261016/`. The highlighted edition compares against `paper/archive/before_results_update_20261016/`. The archived bootstrap replay and environment are bundled in `paper/reproducibility_20261016.zip`; see [the results guide](results.md#robustness-reproducibility-and-leakage-checks).
