# Results: 517-event revision

The primary notebook uses the pump set from `0167dddfa6b55819e276812836ac7537c5d3a00b`, existing regenerated features, and the original date boundaries: 321 training, 112 validation, and 84 test events. The 11 historical candidates retain their saved parameters, and an untuned, unweighted, unpenalized statsmodels Logit is fitted as the first baseline. All 12 models enter the comparisons; no new hyperparameter search is run.

**Tuned logistic regression is the primary model**, chosen retrospectively for simplicity, coefficient-level interpretation, and the highest observed test AUC. CatBoost + Top@k%AUC early stopping remains the original validation-AUC winner (0.6283), ahead of tuned CatBoost (0.6158); tuned logistic regression has validation AUC 0.5989. The primary-model choice follows test inspection and requires new holdout validation. Tuned CatBoost leads test Top@1 (0.1667), PR-AUC (0.0973), F1 (0.1667), and balanced accuracy (0.5819).

## Paired test bootstrap

10,000 draws of 84 whole events with replacement, seed 42, shared across fixed models. Top@k%AUC integrates from 0 to 20% in 0.5% increments and is normalised by 0.2. Intervals are pointwise percentile intervals, conditional on the fitted models and assuming independent events.

| Model | Test Top@k%AUC | 95% CI |
|---|---:|---:|
| Logistic regression + Tuned | 0.7126 | [0.6381, 0.7835] |
| CatBoost classifier + Top@k%AUC early stopping | 0.6957 | [0.6186, 0.7683] |
| Logistic regression | 0.6699 | [0.5890, 0.7470] |
| Statsmodels Logit baseline | 0.6679 | [0.5873, 0.7457] |
| CatBoost classifier + Tuned | 0.6582 | [0.5808, 0.7330] |
| CatBoost classifier | 0.6436 | [0.5656, 0.7186] |
| Random forest | 0.6272 | [0.5475, 0.7039] |
| Random forest + Tuned | 0.6186 | [0.5365, 0.6972] |
| CatBoost ranker + Tuned | 0.5841 | [0.4982, 0.6658] |
| CatBoost classifier + SMOTE + Tuned | 0.4875 | [0.4051, 0.5716] |
| CatBoost ranker | 0.4397 | [0.3536, 0.5244] |
| CatBoost classifier + SMOTE | 0.4186 | [0.3373, 0.5030] |

CatBoost + ES minus tuned logistic regression: **−0.0170**, paired 95% CI **[−0.0650, 0.0295]**, two-sided p = **0.4828**, Holm-adjusted p = **1.000**. Against untuned logistic regression: **+0.0257**, CI **[−0.0284, 0.0798]**, adjusted p = **1.000**. Neither comparison establishes superiority or equivalence. Holm correction covers all 66 model pairs. The test period was inspected in earlier research, so these results are exploratory.

## Logistic coefficients and statsmodels

The predictive model uses L1 regularization (`C=0.0018798144`), positive-class weight 72.549639 (negative-class weight 1), and the `liblinear` solver. It has 38 nonzero coefficients out of 81 predictors, with an intercept of zero. [All fitted coefficients](../notebooks/analysis_outputs/pumps_0167ddd/paper_run/tables/logistic_coefficients.csv) include original pipeline units, odds ratios, and coefficients multiplied by training SD for scale comparison. These weighted scores are not calibrated population probabilities.

At the beginning, the notebook prints [statsmodels summary2()](../notebooks/analysis_outputs/pumps_0167ddd/paper_run/tables/logistic_statsmodels_summary2.txt) for the **untuned, unweighted, unpenalized Logit baseline** using the same 49,464 training rows and all 81 predictors. Predictors are centered/scaled using training statistics; standard errors cluster by pump event (321 clusters). This fit converged in 13 iterations: log-likelihood −1612.5432, McFadden pseudo-R² 0.1675, AIC 3389.0865. Its p-values do not apply to the regularized predictive model and do not adjust for earlier research choices. The same fitted baseline is used for validation/test ranking and the all-model bootstrap; it does not replace the tuned primary portfolio model. Its validation AUC is 0.5779 and test AUC 0.6679.

## Reproduction

`notebooks/analysis_outputs/pumps_0167ddd/test_auc_bootstrap/` contains fixed predictions, event AUCs, all bootstrap draws, distributions, pairwise tests, metadata and figures. `paper_run/` contains the full executed analysis, updated tables/figures, feature-file hashes, partition membership and package versions. Run `MPLBACKEND=Agg .venv/bin/python scripts/run_test_auc_bootstrap.py --full` using the existing regenerated features.

The revised paper separates prediction from assumption-dependent execution simulations and limits external claims to Binance BTC pairs. Older results from the alternative four-part split remain in [the archived report](results_refactored_19063b5.md).
