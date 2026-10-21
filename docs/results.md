# Research results

[README](../README.md) · [Reproduction guide](reproduction.md)

File paths below are relative to the repository root.

The current manuscript reports **517 events**, split chronologically into **310 training / 52 tuning / 103 comparison validation / 52 test events**. All seven model families use 100 tuning trials on the separate tuning partition. The complete notebook rerun on **2026-10-21** reproduced the manuscript's ranking, bootstrap, portfolio, impact and robustness results. Test events run from 2021-08-24 through 2022-08-14. These historical periods were inspected in earlier research, so this is a retrospective study.

`Top@k` and `Top@k%` measure whether the manipulated asset appears among the highest-ranked k assets or the top percentage of each event's candidates. `Top@k%AUC` is the normalized area under the 0–20% hit curve, evaluated on a 0.5-percentage-point grid.

**Logistic regression is selected on comparison validation**, with Top@k%AUC 0.599 versus 0.597 for CatBoost. This choice is frozen before test evaluation and used for the portfolios. The small validation difference is statistically inconclusive; selection by the largest point estimate does not establish a unique statistical winner. ES below means Top@k%AUC early stopping.

## Validation and test Top@k%AUC

Intervals are **95% pointwise percentile intervals from 10,000 paired whole-event bootstrap draws per partition**, using seed 42. Rows follow validation AUC.

| Model | Validation Top@k%AUC | 95% CI | Test Top@k%AUC | 95% CI |
| --- | --- | --- | --- | --- |
| Logistic regression | 0.599 | [0.524, 0.671] | 0.769 | [0.681, 0.846] |
| CatBoost classifier | 0.597 | [0.521, 0.670] | 0.723 | [0.626, 0.812] |
| CatBoost + Top@k%AUC ES | 0.551 | [0.471, 0.628] | 0.725 | [0.631, 0.813] |
| CatBoost + Top-5 | 0.514 | [0.432, 0.595] | 0.588 | [0.474, 0.702] |
| Random forest | 0.487 | [0.411, 0.561] | 0.688 | [0.598, 0.772] |
| CatBoost ranker | 0.472 | [0.391, 0.553] | 0.591 | [0.479, 0.700] |
| CatBoost + SMOTE | 0.461 | [0.385, 0.537] | 0.612 | [0.507, 0.711] |

## Top@k accuracy (52-event test set)

| Model | Top@1 | Top@2 | Top@5 | Top@10 | Top@20 | Top@30 |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.058 | 0.115 | 0.404 | 0.596 | 0.827 | 0.827 |
| CatBoost classifier | 0.192 | 0.365 | 0.462 | 0.558 | 0.673 | 0.769 |
| CatBoost + Top@k%AUC ES | 0.212 | 0.308 | 0.365 | 0.558 | 0.750 | 0.788 |
| CatBoost + Top-5 | 0.135 | 0.212 | 0.308 | 0.481 | 0.596 | 0.635 |
| Random forest | 0.096 | 0.154 | 0.250 | 0.423 | 0.673 | 0.788 |
| CatBoost ranker | 0.038 | 0.135 | 0.346 | 0.462 | 0.538 | 0.654 |
| CatBoost + SMOTE | 0.096 | 0.192 | 0.327 | 0.385 | 0.596 | 0.654 |

CatBoost + ES has the highest test Top@1/F1 (0.212), PR-AUC (0.164), and balanced accuracy (0.604). CatBoost leads Top@2 (0.365) and Top@5 (0.462); logistic regression leads the broader Top@k%AUC (0.769). These point-estimate rankings alone do not establish statistical superiority.

## Statistical significance

Every bootstrap draw uses the same sampled pumps for all fitted models, preserving complete candidate cross-sections and repeated draws. Two-sided centered-null bootstrap tests use an add-one correction. Holm adjustment covers **21 AUC comparisons on validation** and **all 84 model-pair × metric comparisons on test** (AUC, F1, precision and recall). At Top@1, with one true asset per pump, the three classification metrics coincide but remain in the correction family. Decisions use adjusted p < 0.05, rather than overlap of marginal intervals.

On validation, logistic regression and CatBoost each exceed random forest, SMOTE and the return ranker; the other 15 pairs are inconclusive. On test, only these three AUC differences are significant:

| Higher AUC | Comparator | AUC difference | Holm-adjusted p |
| --- | --- | --- | --- |
| Logistic regression | CatBoost ranker | 0.1781 | 0.0252 |
| Logistic regression | CatBoost + Top-5 | 0.1805 | 0.0415 |
| CatBoost + Top@k%AUC ES | CatBoost ranker | 0.1346 | 0.0492 |

All other test AUC comparisons and all F1/precision/recall comparisons are inconclusive. Non-rejection does not establish equivalence. Inference assumes independent, exchangeable pump events and conditions on fitted models; it excludes retraining and tuning uncertainty and cannot establish live profitability.

## Portfolio performance (validation-selected logistic regression)

Returns use equal-weight Top@k holdings, fixed event notional without reinvestment, first-observed-trade entry/exit and a 25 bp round-trip cost. The table excludes the separate size-dependent impact penalty. Return and volatility columns are decimal fractions.

| Portfolio / benchmark | Avg. event return | Cumulative return | Annualized return | Annualized vol. | Sharpe |
| --- | --- | --- | --- | --- | --- |
| k=1 | 0.0251 | 1.3031 | 1.3360 | 0.8079 | 1.6537 |
| k=2 | 0.0087 | 0.4498 | 0.4612 | 0.4230 | 1.0903 |
| k=5 | 0.0110 | 0.5698 | 0.5842 | 0.2548 | 2.2931 |
| k=10 | 0.0077 | 0.3995 | 0.4096 | 0.1610 | 2.5446 |
| k=20 | 0.0058 | 0.3034 | 0.3111 | 0.1161 | 2.6784 |
| k=30 | 0.0030 | 0.1555 | 0.1594 | 0.0804 | 1.9827 |
| BTCUSDT buy-and-hold | — | -0.4966 | -0.5105 | 0.6794 | -0.7514 |

The primary portfolio size remains **k=5**, with cumulative return **0.570** and Sharpe **2.29**. Other sizes are descriptive sensitivity checks: k=20 has the largest Sharpe (2.68), and k=1 the largest cumulative return (1.303). BTC/USDT loses 49.7% over the same interval, from 2021-08-24 14:45:03 UTC to 2022-08-14 16:01:05 UTC. BTC remains invested between events, so the benchmark matches calendar time but not exposure. These simulated point estimates are not significance tests of profitability.

With fitted square-root impact and k=5 fixed, cumulative ROE falls from **0.486 at 100 USDT per trade** to **−0.261 at 10,000 USDT**. It is 0.195 at 2,000 USDT and −0.020 at 5,000 USDT, crossing zero between those tested sizes. This is conditional execution sensitivity, not a statistically established capacity threshold.

## Robustness, reproducibility and leakage checks

- **Training-subset sensitivity.** Across 25 CatBoost + ES fits on seeded 70% subsets of TRAIN events, test Top@k%AUC has mean 0.740, standard deviation 0.021 and a central 80% range [0.713, 0.766]. This diagnoses ES training sensitivity; it is not a robustness estimate for the selected logistic portfolio model.
- **Temporal sensitivity.** The selected logistic model scores 0.746 on the 46 test events before 2022-06-01. The six later events fail the `min_pumps=10` reporting guard.
- **Reproducibility.** Fixed feature files and hashes, event splits, saved tuning parameters, seeds and the recorded environment support replay. The latest full notebook run reproduced the archived result tables, and 124 targeted research, bootstrap and leakage tests passed. All 15 notebook-generated paper figures match the displayed image files byte for byte; the source notebook omits embedded outputs, with the executed copy and detailed CSVs saved under `notebooks/analysis_outputs/split_60_10_20_10_v1/deterministic_selection/`.
- **Lookahead and regressor leakage.** Automated checks add extreme trades at and after T−15min, future-only listings and later pump announcements, then require candidate membership and all 81 predictors to remain unchanged while forward-looking labels change. Further tests mutate held-out values and future returns, enforce TRAIN-only fallback imputation, verify the regressor allowlist and input order, isolate tuning/early stopping from comparison/test rows, and keep portfolio selection independent of future outcomes. Timestamp audits check that each feature input precedes the cutoff. These checks found no leakage along the tested paths; they do not prove the absence of every possible bias.

The manuscript is in `paper/src/access.tex`, with the compiled [paper](../paper/access.pdf) and [highlighted revisions](../paper/access_highlighted.pdf). Result tables are archived in `paper/src/results_20261016/`. The replay data, source snapshot, verification environment and `replay_test_bootstrap.py` are bundled in `paper/reproducibility_20261016.zip`; the replay script reproduces exported bootstrap metrics, intervals and Holm decisions without retraining. The highlighted PDF compares the manuscript with `paper/archive/before_results_update_20261016/`.
