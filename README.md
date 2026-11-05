# Mitigating Class Imbalance in Pump-and-Dump Detection

Empirical comparison of pump-target prediction models on Binance BTC pairs, with event-level ranking metrics and an execution-sensitivity backtest.

The active experiment restores the **473 pumps analyzed in the paper at `0167ddd`**: **290 train / 103 validation / 80 test**. The exact cohort is the original 498-event manifest minus its 25 recorded exclusions. It evaluates seven tuned predictive candidates with the existing regenerated features and estimates test Top@k%AUC uncertainty using 10,000 paired whole-event bootstrap samples. The notebook starts with a standalone statsmodels `summary2()` diagnostic, excluded from model comparisons. The primary model is selected automatically by the highest validation Top@k%AUC; the current winner is tuned CatBoost. The L2 logistic model uses the same preprocessing and class weights, selecting C from a fixed 25-value grid on validation data only. See [results](docs/results.md) and [cohort provenance](research_snapshots/paper_0167ddd/provenance.json).

## Setup and data

Requires Python 3.13+ and Git. The installer TUI handles environment setup, downloads, Hive conversion, and features:

```bash
./scripts/install.sh
```

The default [event manifest](resources/pumps.json) contains exactly **498 announcements**, byte-for-byte identical to commit `73ba716dfe39be3f2da9b0ccd7d6663ff544b570`. Its [25 recorded exclusions](resources/dropped_pumps.json) leave the **473-event paper cohort**. The installer mounts this checkout's manifest into the feature container.

Existing regenerated features can be reused. Data paths, Docker options, and commands are in the [reproduction guide](docs/reproduction.md).

## Research

```bash
source .venv/bin/activate
jupyter lab notebooks/research_notebook.ipynb
```

- [research_notebook.ipynb](notebooks/research_notebook.ipynb): primary analysis, diagnostic preprocessing, models, comparisons, portfolios, and robustness.
- [analysis_helpers.ipynb](notebooks/analysis_helpers.ipynb): shared plotting, analysis tables, diagnostics, and exports; loaded by the research notebooks.
- [portfolio_1000usdt_slippage.ipynb](notebooks/portfolio_1000usdt_slippage.ipynb): saved-model portfolios and random selections with 1,000 USDT per asset, entry/exit price impact, and 25 bp round-trip costs.
- [visualisations.ipynb](notebooks/visualisations.ipynb): trade-level illustrations.
- [research_refactored_19063b5.ipynb](notebooks/research_refactored_19063b5.ipynb): retained alternative split experiment, with [archived results](docs/results_refactored_19063b5.md).

Run the model fits and bootstrap without portfolios, or run the full notebook:

```bash
MPLBACKEND=Agg .venv/bin/python scripts/run_test_auc_bootstrap.py
MPLBACKEND=Agg .venv/bin/python scripts/run_test_auc_bootstrap.py --full
MPLBACKEND=Agg .venv/bin/python scripts/run_test_auc_bootstrap.py --portfolio
.venv/bin/python scripts/audit_notebook_results.py
.venv/bin/python paper/src/update_results.py
```

Outputs and executed notebooks go under `notebooks/analysis_outputs/paper_cohort_473_l2/`. The paper source is [paper/src/access.tex](paper/src/access.tex); `just paper` builds it. [Build and reproduction details](docs/reproduction.md).

The manuscript uses the completed fixed-seed run for every numerical result. Model comparisons, paired resampling and portfolios share the same saved predictions. The [submitted PDF](paper/baselines/ieee_final_submission/submitted.pdf) is used only as the baseline for highlighting changes.

## Code layout and checks

Reusable training/data pipelines, metrics, resampling algorithms, and execution models live in `src/pumps_and_dumps/`; the data-install TUI is in `src/installer/`. Analysis and plotting live in notebooks. A reduced historical runtime preserves the primary notebook's original model definitions and saved-study protocol.

```bash
.venv/bin/python -m pytest -q
```

[MIT license](LICENSE).
