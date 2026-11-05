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
  --mount "type=bind,source=$(pwd)/resources/pumps.json,target=/app/resources/pumps.json,readonly" \
  b0r0koko/pds-feature-writer:latest --config /config/features.conf
```

Use `--entrypoint pumps-download-klines` on the downloader for candles, and `--help` on any command for all options. The same commands work in the local research environment. The installer pulls images before running stages and mounts the checkout's `resources/pumps.json` read-only over `/app/resources/pumps.json`, so the feature stage uses the checked-out event list even with an older image. A custom inline or file-based event selection in a feature config still takes precedence. Direct container commands use Docker's normal image policy; include the manifest mount shown above to use the checkout's events.

Feature configs accept either an inline `pumps` list or `pumps-file`. Each event supplies `symbol`, `exchange` and a UTC `time` at whole-second precision. Feature `--start` / `--end` filter announcement times while retaining earlier events for historical features. `--no-skip-existing` overrides configured feature reuse and produces fresh input audits.

To build local images, run the corresponding Dockerfile from the repository root and substitute your tag in direct Docker commands:

```bash
docker build -f docker/dataloader/Dockerfile -t pumps-and-dumps-dataloader:local .
docker build -f docker/preprocessor/Dockerfile -t pumps-and-dumps-preprocessor:local .
docker build -f docker/feature-writer/Dockerfile -t pumps-and-dumps-feature-writer:local .
```

Each image installs its own `requirements.txt` and the project without notebook/training dependencies. Container builds do not use `poetry.lock`; retain the verification environment when replaying recorded results.

## Event labels and features

[resources/pumps.json](../resources/pumps.json) contains exactly **498 announcements**, with identical contents and order to commit `73ba716dfe39be3f2da9b0ccd7d6663ff544b570` (SHA-256 `3f0cab25093a2c304d15b071b7b40344a7ce58c03ef79dd7b1e9b3a29b319467`). The matching [dropped_pumps.json](../resources/dropped_pumps.json) records the 25 historical exclusions, leaving the 473 events used in the paper. The primary notebook pins those 473 events independently of current data availability. The modern dataset loader rewrites `resources/dropped_pumps.json` with its actual missing-event report when run; the immutable historical exclusions are also preserved in [the paper snapshot](../research_snapshots/paper_0167ddd/excluded_pumps.json).

The expanded 517-event manifest is [archived separately](../research_snapshots/expanded_517/pumps.json), with its hash and source commit in [provenance.json](../research_snapshots/expanded_517/provenance.json). [pump_event_evidence.json](../resources/pump_event_evidence.json) and [pump_event_exclusions.json](../resources/pump_event_exclusions.json) describe that archived expansion, not the default 498-event manifest. Events are not filtered by post-announcement profitability. The original labels have not all been comprehensively reaudited; historical review artifacts remain under `data/research/`.

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

`notebooks/research_notebook.ipynb` uses the **473-event cohort analyzed in the paper stored at `0167dddfa6b55819e276812836ac7537c5d3a00b`**. Its PDF is the same Git blob as at `73ba716`, where the original 498-event manifest and 25 recorded unavailable events are both committed. The active cohort is their exact set difference, preserving event identities and order. It lives in [research_snapshots/paper_0167ddd/pumps.json](../research_snapshots/paper_0167ddd/pumps.json), with source hashes and the paper blob ID in its [provenance](../research_snapshots/paper_0167ddd/provenance.json). Historical model imports load the isolated `research_snapshots/73ba716/runtime.zip`; its dataset loader reads this pinned manifest for every model pipeline. Original date boundaries give **290 train / 103 validation / 80 test**. Saved parameters still load from `<data directory>/studies.db`. The modern pipeline and installer default to the restored 498-event `resources/pumps.json`; the primary notebook continues to use the pinned 473-event paper cohort.

The compatibility snapshot respects the installer’s configured directory and writes dropped-event diagnostics under `notebooks/analysis_outputs/research_73ba716/`. Portfolio reads use the shared partition-aware trade reader, with the same timestamp boundaries and execution rules. Original model, metric and split behaviour is preserved. The snapshot includes only model feature-offset constants; feature construction uses the current data pipeline. Plotting and analysis helpers live in `notebooks/analysis_helpers.ipynb`, which both research notebooks load. `python research_snapshots/build_snapshot.py` regenerates the runtime archive from the original Git commit, reconstructs the paper cohort from the historical manifest and exclusions and verifies the paper PDF identity, and records separate model-code and active-manifest provenance. The research notebook also rejects missing requested events before training.

Launch `jupyter lab notebooks/research_notebook.ipynb` from the repository root and use the existing project kernel. The standalone statsmodels diagnostic summary is saved near the beginning and excluded from comparisons. Earlier 517-event run artifacts remain under `notebooks/analysis_outputs/pumps_0167ddd/`. To execute just the seven-model comparison and its paired test bootstrap (plus the standalone diagnostic fit), run `MPLBACKEND=Agg .venv/bin/python scripts/run_test_auc_bootstrap.py`. This executes the tagged primary-notebook cells, with seed 42, one CPU thread per model by default, and CatBoost file logging disabled. `--threads N` explicitly changes the thread budget recorded for that run. It writes an executed notebook under `notebooks/analysis_outputs/paper_cohort_473_l2/test_auc_bootstrap/`.

The bootstrap scores each fitted model once, then resamples 80 whole test events with replacement 10,000 times (seed 42), using identical draws for every model. It integrates Top@k% from 0 to 20% in 0.5% increments and normalises by 0.2. Repeated events retain their multiplicity. Outputs include `auc_distributions.csv`, `auc_summary.csv`, `auc_pairwise.csv`, `event_aucs.csv`, `resampled_events.npz` (indices into the saved event-ID array), `test_predictions.parquet`, `validation_auc.csv`, `metadata.json`, and PNG/SVG plots. Pointwise percentile intervals describe fixed-model sampling uncertainty; two-sided centred bootstrap tests use an add-one correction and Holm adjustment across all 21 pairs. Events are assumed independent. The test has been inspected previously, so this comparison remains exploratory. Metadata records the validation winner and the primary model; these are the same fitted candidate. The `--full` option additionally runs all notebook analyses and exports tables, figures, feature-file hashes, partition manifests and package versions under `analysis_outputs/paper_cohort_473_l2/paper_run/`. It uses the existing regenerated features without recomputing them. The L2 logistic model shares the L1 preprocessing and historical tuned class weights. Its C is selected from 25 logarithmically spaced values from 1e-6 to 1e2 using validation Top@k%AUC only, with smaller C breaking ties. The search uses liblinear, tol=1e-8 and max_iter=2000; a convergence warning fails the search. For training-subset robustness, the selected L2 C and class weights stay fixed while the logistic preprocessing and model are refitted on each subset; the historical L1 study is not reloaded. The validation path, selected parameters, and coefficients are exported as `logistic_l2_*` under `paper_run/tables/`. The original 12-model run remains in `analysis_outputs/paper_0167ddd_473/`; the current seven-model comparison uses `paper_cohort_473_l2/`. Restoring event membership does not claim that feature values or results are bit-for-bit identical to the historical run; the hashes identify the inputs used now.

### Repeating tuning and training

The primary notebook initializes Python and NumPy with seed **42** and uses the same explicit seed for Optuna TPE, L1/L2 logistic regression, random forests, all four CatBoost pipelines, SMOTE, and resampling. The policy is defined in [research_snapshots/reproducibility.py](../research_snapshots/reproducibility.py) and imported by the historical runtime. Optuna trials run sequentially (`n_jobs=1`). CPU models and native numerical libraries use one thread by default. The batch launcher sets `PYTHONHASHSEED=42` before starting the kernel; feature columns also have an explicit stable order. In an interactive session, restart the kernel and run from the beginning.

The notebook fits with saved study parameters. To reproduce a *hyperparameter search*, start a fresh study with identical data, search space, trial count, package versions, and thread settings; a seeded sampler does not replay a partially completed study automatically. Preserve `studies.db` before invoking the historical pipelines' `optimize_parameters()`: their existing `start_new=True` behavior replaces the named study. L2 uses the fixed validation grid with smaller C breaking ties, while the unpenalized statsmodels diagnostic starts Newton optimization from zero coefficients.

`tables/reproducibility.json`, bootstrap metadata, and the full run manifest record the seed, sampler, thread policy, and software versions. The full manifest also records fitted parameters and feature order alongside the input hashes. These settings make repeat runs checkable; they do not reconstruct an unavailable earlier database or feature snapshot. A regression test repeats miniature searches and fits for all seven candidates in separate processes with different Python hash seeds and compares the complete trial records and predictions.

## Refactored training and evaluation

The retained [refactored notebook](../notebooks/research_refactored_19063b5.ipynb) refits seven candidates from saved Optuna parameters. It selects the largest comparison-validation Top@k%AUC before evaluating test results. Its archived 517-event run used 310 training, 52 tuning/early-stopping, 103 comparison and 52 test events; complete cross-sections and simultaneous events remain together. A new run reads the restored default manifest, so available-event counts and partitions can differ from that archived run. Regenerate features and retune for the changed manifest rather than treating the archived studies as a reproduction of the 498-event input.

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

Sources are in `paper/src/`, figures in `paper/src/images/`, and the IEEE Access class/fonts in `paper/styles/`. All manuscript model results, resampling and portfolios come from the current full notebook execution. The previous submitted PDF is used only for highlighting. The exact baseline is `paper/baselines/ieee_final_submission/submitted.pdf` (SHA-256 `0c3b40e93a729f8840ed8f62e714e827981703acf29cee6082f59fb1451ad361`), matching `IEEE_Final_submission (2).pdf`. `paper/src/build_highlighted.py` uses its reconstructed source and extracted images to mark changed text, tables and figures. Historical submissions remain archived for provenance.

Run the analysis and refresh manuscript results in this order:

```bash
MPLBACKEND=Agg .venv/bin/python scripts/run_test_auc_bootstrap.py --full
MPLBACKEND=Agg .venv/bin/python scripts/run_test_auc_bootstrap.py --portfolio
.venv/bin/python scripts/audit_notebook_results.py
.venv/bin/python paper/src/update_results.py
.venv/bin/python paper/src/update_results.py --check
python3 paper/src/build_highlighted.py
just paper
just paper-highlighted
```

The replay audit independently reconstructs test point estimates and classification scores, every bootstrap draw and paired comparison, selected-asset portfolio returns and hit rates, and random-selection returns. The manuscript updater reads the exported tables and copies current figures. It stops if a changed winner requires revised interpretation. The numerical audit is `docs/notebook_rerun_audit.json`; table and figure provenance is `docs/manuscript_results_audit.json`. The full run saves all seven fitted models in `paper_run/models/`, as well as parameters, feature hashes, split membership, seed policy and software versions. Pickle artifacts require the recorded Python/packages and the historical runtime; load only trusted local artifacts. The rerun's model files have been reloaded and checked against every saved test prediction.

Executed notebooks are written beneath the output directory. The checked-in research notebooks contain the completed run outputs. `source_snapshot/` under the local run preserves the input notebooks and runtime used for that execution. A rerun fits the six saved study configurations and repeats the L2 validation grid; it does not perform six fresh Optuna searches. Archive the current output directory before a new run if its exact artifacts must be retained.

Older local reproduction ZIPs may describe earlier experiments. They are not the source of the current manuscript results. To share this execution, bundle the current output directory, source snapshot, feature files matching `feature_hashes.csv`, unchanged `studies.db`, and the recorded software environment. Raw/Hive market data are needed to repeat execution and impact fitting; saved execution outcomes and selections suffice for the independent portfolio replay audit.
