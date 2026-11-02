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

## Historical notebook (73ba716)

`notebooks/research_notebook.ipynb` uses the **473-event cohort analyzed in the paper stored at `0167dddfa6b55819e276812836ac7537c5d3a00b`**. Its PDF is the same Git blob as at `73ba716`, where the original 498-event manifest and 25 recorded unavailable events are both committed. The active cohort is their exact set difference, preserving event identities and order. It lives in [research_snapshots/paper_0167ddd/pumps.json](../research_snapshots/paper_0167ddd/pumps.json), with source hashes and the paper blob ID in its [provenance](../research_snapshots/paper_0167ddd/provenance.json). Historical model imports load the isolated `research_snapshots/73ba716/runtime.zip`; its dataset loader reads this pinned manifest for every model pipeline. Original date boundaries give **290 train / 103 validation / 80 test**. Saved parameters still load from `<data directory>/studies.db`. The broader `resources/pumps.json` retains 517 events for the modern pipeline and installer; it does not determine the primary notebook cohort.

The compatibility snapshot respects the installer’s configured directory and writes dropped-event diagnostics under `notebooks/analysis_outputs/research_73ba716/`. Portfolio reads use the shared partition-aware trade reader, with the same timestamp boundaries and execution rules. Original model, metric and split behaviour is preserved. The snapshot includes only model feature-offset constants; feature construction uses the current data pipeline. Plotting and analysis helpers live in `notebooks/analysis_helpers.ipynb`, which both research notebooks load. `python research_snapshots/build_snapshot.py` regenerates the runtime archive from the original Git commit, reconstructs the paper cohort from the historical manifest and exclusions and verifies the paper PDF identity, and records separate model-code and active-manifest provenance. The research notebook also rejects missing requested events before training.

Launch `jupyter lab notebooks/research_notebook.ipynb` from the repository root and use the existing project kernel. The standalone statsmodels diagnostic summary is saved near the beginning and excluded from comparisons. Earlier 517-event run artifacts remain under `notebooks/analysis_outputs/pumps_0167ddd/`. To execute just the seven-model comparison and its paired test bootstrap (plus the standalone diagnostic fit), run `MPLBACKEND=Agg .venv/bin/python scripts/run_test_auc_bootstrap.py`. This executes the tagged primary-notebook cells, with eight CatBoost threads and file logging disabled; statistical parameters are unchanged. It writes an executed notebook under `notebooks/analysis_outputs/paper_cohort_473_l2/test_auc_bootstrap/`.

The bootstrap scores each fitted model once, then resamples 80 whole test events with replacement 10,000 times (seed 42), using identical draws for every model. It integrates Top@k% from 0 to 20% in 0.5% increments and normalises by 0.2. Repeated events retain their multiplicity. Outputs include `auc_distributions.csv`, `auc_summary.csv`, `auc_pairwise.csv`, `event_aucs.csv`, `resampled_events.npz` (indices into the saved event-ID array), `test_predictions.parquet`, `validation_auc.csv`, `metadata.json`, and PNG/SVG plots. Pointwise percentile intervals describe fixed-model sampling uncertainty; two-sided centred bootstrap tests use an add-one correction and Holm adjustment across all 21 pairs. Events are assumed independent. The test has been inspected previously, so this comparison remains exploratory. Metadata records the validation winner and the primary model; these are the same fitted candidate. The `--full` option additionally runs all additional notebook analyses and exports tables, figures, feature-file hashes, partition manifests and package versions under `analysis_outputs/paper_cohort_473_l2/paper_run/`. It uses the existing regenerated features without recomputing them. The L2 logistic comparator shares the L1 preprocessing and historical tuned class weights. Its C is selected from 25 logarithmically spaced values from 1e-6 to 1e2 using validation Top@k%AUC only, with smaller C breaking ties. The search uses liblinear, tol=1e-8 and max_iter=2000; a convergence warning fails the search. For training-subset robustness, the selected L2 C and class weights stay fixed while the logistic preprocessing and model are refitted on each subset; the historical L1 study is not reloaded. The validation path, selected parameters, and coefficients are exported as `logistic_l2_*` under `paper_run/tables/`. The original 12-model run remains in `analysis_outputs/paper_0167ddd_473/`; the current seven-model comparison uses `paper_cohort_473_l2/`. Restoring event membership does not claim that feature values or results are bit-for-bit identical to the historical run; the hashes identify the inputs used now.

## Refactored training and evaluation

The retained [refactored notebook](../notebooks/research_refactored_19063b5.ipynb) refits seven candidates from saved Optuna parameters. It selects the largest comparison-validation Top@k%AUC before evaluating test results. The chronological partitions contain 310 training, 52 tuning/early-stopping, 103 comparison and 52 test events; complete cross-sections and simultaneous events remain together.

For a fresh data directory without studies, run:

```bash
python -m pumps_and_dumps.backtest.pipelines.tuning \
  --output-dir data/research_runs/split_60_10_20_10_v1 \
  --trials 100
```

This uses the selected data directory's `studies_split_60_10_20_10_v1.db` and refuses to overwrite an existing database. For another search, pass `--storage sqlite:////absolute/new/studies.db`, use a separate output directory, and set the notebook's `STUDY_STORAGE` to that same URL. Each search records parameters, split membership, logs and fitted artifacts. Retune after changing the data, predictors or split policy.

Then launch Jupyter and run the notebook from a fresh kernel:

```bash
jupyter lab notebooks/research_refactored_19063b5.ipynb
```

Outputs go under `notebooks/analysis_outputs/split_60_10_20_10_v1/deterministic_selection/`: metric tables, bootstrap predictions/draws/comparisons, robustness runs, plots and fit logs. Matplotlib images are displayed once and copied unchanged to `paper/src/images/`. The source notebook has no saved outputs; the verified executed copy is `research_notebook.executed.ipynb` in the output directory.

Bootstrap inference uses the same event draws across fixed models, with 10,000 replicates and seed 42. Holm correction covers 21 validation AUC comparisons and all 84 test pair/metric hypotheses. Precision, recall and F1 coincide under Top@1 but remain in that family. [Results and inference limits](results_refactored_19063b5.md#statistical-significance) describe the findings.

The alternative refactored notebook's portfolios use its validation-selected model. The primary historical notebook selects its portfolio model automatically using validation Top@k%AUC. The current winner is CatBoost + Top@k%AUC early stopping. It first fits a standalone untuned statsmodels Logit diagnostic and prints `summary2()` with pump-event clustered standard errors before any plots or other models. This diagnostic fit is excluded from all validation/test comparisons and the 21-pair bootstrap; the validation-selected CatBoost model is the primary predictor. Additional centering/scaling uses training rows only. The model implementation lives in `src/pumps_and_dumps/backtest/pipelines/logistic_regression/statsmodels.py`; summary formatting and exports stay in the notebook. Both portfolio workflows use equal weights. Missing fills retain their cash allocation. Entry is the first trade in `[T−15min, T)`; other assets exit at or after T, and the manipulated asset at or after T+1min. Returns use fixed event notional, USDT conversion and 25 bp round-trip costs. Impact fits use only data strictly before the transaction being priced; the separate size sweep uses `I_vwap(Q) = (2/3) × β × sqrt(Q)`.

## Build the paper

With `just` and a LaTeX installation available:

```bash
just paper
just paper-highlighted
```

Builds write to `paper/src/`. To refresh the shareable copies after checking the build:

```bash
cp paper/src/access.pdf paper/access.pdf
cp paper/src/access_highlighted.pdf paper/access_highlighted.pdf
```

Sources are in `paper/src/`, figures in `paper/src/images/`, and the IEEE Access class/fonts in `paper/styles/`. The manuscript retains the original numerical results for tuned models and adds the current 473-event notebook's paired bootstrap, L2 comparator, validation-selected/random portfolios, and reviewer clarifications. One reporting note identifies the original estimates and the fits used for the bootstrap. The duplicated historical portfolio panels have been removed; the current portfolio, impact and temporal sensitivity analyses use the validation-selected CatBoost fit. `research_snapshots/submitted_results/` pins the original PDF, source, reported tables and provenance; `submitted_*` figures retain the original graphics. Matching fitted models and event predictions for the submitted 80-event experiment were not found, so current bootstrap intervals must not be paired with its historical scores. The older October prediction archive uses 52 test events and cannot substitute for that experiment. Reproduction commands execute the additional notebook experiment; its saved predictions, resamples and feature hashes identify the results they reproduce. Historical sources remain under `paper/archive/`. The highlighted edition marks changes relative to the pre-review 73ba716 manuscript; `paper/src/build_highlighted.py` regenerates it from the tracked `research_snapshots/submitted_results/access.tex` baseline. The required paper figures, LaTeX class and fonts are versioned with the source.

The local [reproduction bundle](../paper/reproducibility_paper_cohort_random.zip) contains frozen feature inputs, saved studies, source code, and run artifacts, with checksums and `REPRODUCE.txt` instructions. It supports replay of model fitting and bootstrap inference without rebuilding features. Raw/Hive market data and original Telegram messages are not bundled; rerunning trade execution and price-impact calculations requires the market data described above. Random asset selections and portfolio metrics can also be replayed from the bundled execution outcomes without raw-market reads. The archive is ignored by Git and must accompany the manuscript separately.

### Random portfolio comparison

The full notebook run also draws 10,000 random portfolios per size (k = 1, 2, 5, 10, 20, 30; seed 42). Each event samples distinct eligible assets with equal probability and assigns equal weights, retaining unfilled allocations as cash. Events stay fixed in chronological order. The random reference shares the main portfolio chart's execution rules, 25 bp round-trip cost and USDT conversion, without price impact. The k=5 mean path and pointwise 95% selection range appear beside validation-selected CatBoost portfolios and BTC buy-and-hold. Return, volatility and Sharpe are computed separately for each draw before averaging; the mean path is not one traded portfolio.

`paper_run/random_portfolio/` stores the event/asset execution outcomes, exact selection indices, per-draw event returns, metrics, path quantiles and replay metadata. `tables/portfolio_statistics.csv` includes random means at every k; `portfolio_vs_random.csv` gives matched-size differences and the fraction of random draws at least as large as the model; `random_portfolio_metrics.csv` includes 95% ranges and target hit rates. These ranges describe asset-selection randomness conditional on this test period and are separate from the paired event bootstrap. The price-impact size-sensitivity section remains a separate execution scenario. Expected small-sample exit-impact fallbacks are silent; failures to fit an impact model still log warnings. The primary notebook filters routine missing-data notices while retaining input validation and error reporting. The AUC bootstrap figure shows interval summaries without cumulative-distribution panels.


### Portfolios with slippage at 1,000 USDT per asset

`notebooks/portfolio_1000usdt_slippage.ipynb` reuses the saved validation-selected CatBoost predictions and executes all eligible test assets with the same price-impact engine at **1,000 USDT per asset** (k × 1,000 USDT intended portfolio capital). It keeps the 25 bp round-trip cost, divides each USDT PnL by its per-asset notional, and leaves unfilled allocations as cash. Both ranked and random portfolios use these shared outcomes. No predictive model is refitted and no feature file is regenerated. Running the execution cell requires the local market archive.

Outputs are in `notebooks/analysis_outputs/paper_cohort_473_l2/portfolio_1000usdt_slippage/`: the figure, per-asset execution/impact details, ranked event returns, model metrics, matched-size random comparisons, and `random_portfolio/` containing all 10,000 selections and their metrics. The k=5 model return is checked against the existing 1,000-USDT impact-sensitivity result. The random selections can be replayed from the saved execution outcomes without raw trades. BTC buy-and-hold is the same continuous market reference as in the main chart. Source snapshots and run manifests distinguish the original full run, the comparison rerun, and the supplemental execution.

The latest comparison rerun executes the main notebook through `paper-run-exports` for seven tuned models (21 paired comparisons). `paper_run/comparison_tuned.executed.ipynb` records that execution. Later portfolio, impact and robustness outputs are retained from the preceding full run after verifying identical retained-model AUCs, shared event draws, selected-model test predictions, feature files and partitions. The assembled main notebook displays the refreshed comparisons alongside those verified analyses; `run_manifest.json` records the execution scopes and prior snapshots. The paper presents the no-impact comparison first, then the price-impact methodology, then Table XIII and Figure 17 with 1,000 USDT per asset.
