# Mitigating Class Imbalance in Pump-and-Dump Detection

Code and experiments for the IEEE Access submission *Mitigating Class Imbalance in Pump-and-Dump Detection*. Given an upcoming manipulation announcement, models rank eligible Binance BTC pairs using 81 predictors available **15 minutes before the announcement**.

[Paper](paper/access.pdf) · [Highlighted revisions](paper/access_highlighted.pdf) · [Research notebook](notebooks/research_notebook.ipynb) · [Full results](docs/results.md) · [Reproduction guide](docs/reproduction.md)

## Results

The study contains **517 events** from January 2018 through August 2022, split chronologically into **310 training / 52 tuning / 103 comparison validation / 52 test events**. Whole events and simultaneous announcements stay together. These historical periods were inspected previously, so the study is retrospective.

`Top@k%AUC` is the normalized area under the hit curve over the top 0–20% of candidates. All seven models use saved tuning studies with 100 trials per family.

| Model | Validation AUC | Test AUC | Test 95% CI |
| --- | --- | --- | --- |
| Logistic regression | 0.599 | 0.769 | [0.681, 0.846] |
| CatBoost classifier | 0.597 | 0.723 | [0.626, 0.812] |
| CatBoost + AUC early stopping | 0.551 | 0.725 | [0.631, 0.813] |
| CatBoost + Top-5 | 0.514 | 0.588 | [0.474, 0.702] |
| Random forest | 0.487 | 0.688 | [0.598, 0.772] |
| CatBoost ranker | 0.472 | 0.591 | [0.479, 0.700] |
| CatBoost + SMOTE | 0.461 | 0.612 | [0.507, 0.711] |

**Logistic regression is selected on validation** and frozen for the test portfolios. Its validation difference from CatBoost is inconclusive. Pointwise intervals use 10,000 paired whole-event bootstrap draws with seed 42. Two-sided tests apply Holm correction across 21 validation comparisons and 84 test comparisons. Only three test AUC differences are significant; all F1/precision/recall comparisons are inconclusive. See the [paired comparisons and limitations](docs/results.md#statistical-significance).

The fixed **k=5 logistic portfolio** returns **0.570 cumulatively**, with Sharpe **2.29**, using equal weights, no reinvestment and 25 bp round-trip costs. Adding fitted price impact reduces cumulative ROE from **0.486 at 100 USDT per trade** to **−0.261 at 10,000 USDT**. These are simulated outcomes, not evidence of live profitability. [Portfolio and robustness details](docs/results.md#portfolio-performance-validation-selected-logistic-regression).

## Run the research

### Set up Python

Requires Git and Python 3.13+. From the repository root:

```bash
./scripts/install.sh --setup-environment --no-update
source .venv/bin/activate
```

This installs the current checkout and registers the **Pumps and dumps (.venv)** Jupyter kernel. Poetry is also supported; see the [environment guide](docs/reproduction.md#environment).

### Prepare the data

If feature files are not already available, install Docker and run:

```bash
./scripts/install.sh
```

Choose a data directory and stages. The installer downloads Binance archives, converts them to Hive Parquet, and computes features for the checked-in events. Use `--help` for options or `--dry-run` to preview commands. [Docker, configuration and data requirements](docs/reproduction.md#data-pipeline).

### Run the notebook

The notebook needs feature files and `studies_split_60_10_20_10_v1.db` in the selected data directory. If the studies are missing, [run tuning first](docs/reproduction.md#training-and-evaluation).

```bash
jupyter lab notebooks/research_notebook.ipynb
```

Select **Pumps and dumps (.venv)**, restart the kernel and run all cells. The notebook refits the seven models using saved parameters, selects on validation, then evaluates test metrics and portfolios.

All plots use Matplotlib and export the exact displayed images to `paper/src/images/`. Tables, bootstrap draws and logs go under `notebooks/analysis_outputs/split_60_10_20_10_v1/deterministic_selection/`. The source notebook omits embedded outputs. [Paper build instructions](docs/reproduction.md#build-the-paper).

## Reproducibility and leakage checks

The **2026-10-21** rerun reproduced the paper's results; 124 targeted research, bootstrap and leakage tests passed. Replay uses fixed feature hashes, event splits, parameters, seeds and the recorded environment. The archived bootstrap replay is described in the [results guide](docs/results.md#robustness-reproducibility-and-leakage-checks).

Feature windows and candidate eligibility end strictly before T−15min. Tests perturb future trades, listings, labels and announcements and check that predictors and portfolio choices remain unchanged. Other checks enforce TRAIN-only fallback imputation, the regressor allowlist, and separation of fitting/tuning from comparison/test data. These checks cover the tested leakage paths; bootstrap inference assumes independent events and excludes retraining uncertainty.

Event labels and review evidence are in [resources/](resources/). [Feature audit details](docs/reproduction.md#feature-audits) explain how actual input timestamps are recorded.

## Development

The research package is in [src/pumps_and_dumps/](src/pumps_and_dumps/): `features/` builds predictors, `backtest/` handles models and evaluation, and `market_data/` plus `preprocessing/` prepare the data. The installer lives in [src/installer/](src/installer/); ready-to-edit jobs are in [configs/](configs/).

With the research environment activated:

```bash
pytest -q
pytest -q tests/leakage tests/features/test_feature_writer_causality.py
```

Formatting, linting, type checks and paper builds are also available in the [justfile](justfile).

## License

[MIT](LICENSE).
