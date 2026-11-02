# Mitigating Class Imbalance in Pump-and-Dump Detection

Empirical comparison of pump-target prediction models on Binance BTC pairs, with event-level ranking metrics and an execution-sensitivity backtest.

The active experiment restores the **473 pumps analyzed in the paper at `0167ddd`**: **290 train / 103 validation / 80 test**. The exact cohort is the original 498-event manifest minus its 25 recorded exclusions. It evaluates seven tuned predictive candidates with the existing regenerated features and estimates test Top@k%AUC uncertainty using 10,000 paired whole-event bootstrap samples. The notebook starts with a standalone statsmodels `summary2()` diagnostic, excluded from model comparisons. The primary model is selected automatically by the highest validation Top@k%AUC; the current winner is CatBoost + Top@k%AUC early stopping. The L2 logistic comparator uses the same preprocessing and class weights, selecting C from a fixed 25-value grid on validation data only. See [results](docs/results.md) and [cohort provenance](research_snapshots/paper_0167ddd/provenance.json).

## Setup and data

Requires Python 3.13+ and Git. The installer TUI handles environment setup, downloads, Hive conversion, and features:

```bash
./scripts/install.sh
```

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
```

Outputs and executed notebooks go under `notebooks/analysis_outputs/paper_cohort_473_l2/`. The paper source is [paper/src/access.tex](paper/src/access.tex); `just paper` builds it. [Build and reproduction details](docs/reproduction.md).

The manuscript preserves the [submitted model-comparison results](research_snapshots/submitted_results/README.md) and adds the current notebook analyses. A brief reporting note identifies the estimates used by the bootstrap; new intervals use only the new saved predictions. The portfolio analysis uses the validation winner.

## Code layout and checks

Reusable training/data pipelines, metrics, resampling algorithms, and execution models live in `src/pumps_and_dumps/`; the data-install TUI is in `src/installer/`. Analysis and plotting live in notebooks. A reduced historical runtime preserves the primary notebook's original model definitions and saved-study protocol.

```bash
.venv/bin/python -m pytest -q
```

[MIT license](LICENSE).
