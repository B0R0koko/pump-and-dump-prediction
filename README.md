# Mitigating Class Imbalance in Pump-and-Dump Detection

Empirical comparison of pump-target prediction models on Binance BTC pairs, with event-level ranking metrics and an execution-sensitivity backtest.

The active experiment uses **473 events: 290 train / 103 validation / 80 test**, from the 498-event manifest and its 25 recorded exclusions. The manuscript compares six model families across **seeds 1–100**, with **25 Optuna trials per model and seed** using a chronological 232/58 split within training. **CatBoost** has the highest mean validation Top@k%AUC (0.6150). Logistic regression has the highest mean test AUC (0.7030), followed by CatBoost (0.6786). The preselected seed-42 models supply point estimates and 10,000 paired event-bootstrap samples; the selected CatBoost model supplies feature interpretation and all portfolios. See [results](docs/results.md) and [cohort provenance](research_snapshots/paper_0167ddd/provenance.json).

## Setup and data

Requires Python 3.13+ and Git. The installer TUI handles environment setup, downloads, Hive conversion, and features:

```bash
./scripts/install.sh
```

The default [event manifest](resources/pumps.json) contains exactly **498 announcements**, byte-for-byte identical to commit `73ba716dfe39be3f2da9b0ccd7d6663ff544b570`. Its [25 recorded exclusions](resources/dropped_pumps.json) leave the **473-event paper cohort**. The installer mounts this checkout's manifest into the feature container.

Existing regenerated features can be reused. Data paths, Docker options, and commands are in the [reproduction guide](docs/reproduction.md).

The committed [studies.db](studies.db) contains the six 100-trial searches used by the retained fixed-seed notebook. The [search snapshot](research_snapshots/latest_search/README.md) records that archive's provenance. The current manuscript uses the 600 separate 25-trial databases under `notebooks/analysis_outputs/validation_seed_sweep_100x25/studies/`; see the reproduction guide.

## Research

```bash
source .venv/bin/activate
jupyter lab notebooks/research_notebook.ipynb
```

- [research_notebook.ipynb](notebooks/research_notebook.ipynb): retained fixed-seed analysis, diagnostic preprocessing, model comparisons, portfolios, and training-subset checks.
- [validation_seed_sweep.ipynb](notebooks/validation_seed_sweep.ipynb): six model families, 25-trial inner searches for seeds 1–100, and model selection by mean validation Top@k%AUC.
- [test_seed_sweep.ipynb](notebooks/test_seed_sweep.ipynb): the same 600 seed-specific configurations evaluated on the 80 test events, with the validation selection fixed.
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

These fixed-seed notebook outputs remain under `notebooks/analysis_outputs/paper_cohort_473_l2/`. The manuscript now uses `validation_seed_sweep_100x25/`, `test_seed_sweep_100x25/`, and derived artifacts in `seed_sweep_paper/`. Refresh the paper from the completed seed sweeps with `python paper/src/update_results.py`, then audit with `python scripts/audit_seed_sweep_manuscript.py`. The source is [paper/src/access.tex](paper/src/access.tex); `just paper` builds it. [Build and reproduction details](docs/reproduction.md).

The 100-seed distributions describe tuning and training variability on fixed events. Seed-42 point estimates, paired event resampling and portfolios use the same saved predictions. The [submitted PDF](paper/baselines/ieee_final_submission/submitted.pdf) is used only as the baseline for highlighting changes.

## Code layout and checks

Reusable training/data pipelines, metrics, resampling algorithms, and execution models live in `src/pumps_and_dumps/`; the data-install TUI is in `src/installer/`. Analysis and plotting live in notebooks. A reduced historical runtime preserves the primary notebook's original model definitions and saved-study protocol.

```bash
.venv/bin/python -m pytest -q
```

[MIT license](LICENSE).
