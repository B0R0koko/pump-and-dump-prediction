# Results: restored 473-event paper cohort

The primary notebook uses the exact event identifiers underlying the paper at `0167dddfa6b55819e276812836ac7537c5d3a00b`: **498 original announcements minus 25 historical data-availability exclusions = 473 analyzed events**, split **290 training / 103 validation / 80 test**. The source records and exclusions come from `73ba716`, whose paper PDF is byte-identical to the PDF at `0167ddd`. [Cohort provenance](../research_snapshots/paper_0167ddd/provenance.json) records the source hashes and exact construction. The default `resources/pumps.json` matches the 498-event source manifest exactly. The expanded 517-event list is [archived separately](../research_snapshots/expanded_517/pumps.json) and is not used by default.

Existing regenerated features are reused unchanged. Restoring event membership does not restore old feature values or promise identical historical results. The 11 historical candidates retain saved hyperparameters; an untuned, unweighted, unpenalized statsmodels Logit runs first. All 12 models enter the comparisons; no new hyperparameter search is run.

**Tuned logistic regression remains the primary model**, retaining the earlier retrospective author choice for simplicity, coefficient-level interpretation, and observed test AUC. It also has the highest test AUC point estimate in the restored cohort. CatBoost + Top@k%AUC early stopping remains the validation-AUC winner (0.6464), ahead of tuned CatBoost (0.6398) and tuned logistic regression (0.6104). The primary-model choice follows earlier test inspection and requires new holdout validation. Tuned CatBoost leads test PR-AUC (0.0905) and ties untuned random forest at Top@1/F1 (0.1500) and balanced accuracy (0.5735).

## Paired test bootstrap

10,000 draws of 80 whole events with replacement, seed 42, shared across fixed models. Top@k%AUC integrates from 0 to 20% in 0.5% increments and is normalized by 0.2. Intervals are pointwise percentile intervals, conditional on the fitted models and assuming independent events.

| Model | Test Top@k%AUC | 95% CI |
|---|---:|---:|
| Logistic regression + Tuned | 0.7020 | [0.6247, 0.7759] |
| CatBoost classifier + Top@k%AUC early stopping | 0.6972 | [0.6169, 0.7750] |
| CatBoost classifier + Tuned | 0.6833 | [0.6052, 0.7573] |
| Logistic regression | 0.6703 | [0.5892, 0.7502] |
| Statsmodels Logit baseline | 0.6675 | [0.5878, 0.7452] |
| CatBoost classifier | 0.6223 | [0.5383, 0.7041] |
| Random forest + Tuned | 0.5984 | [0.5133, 0.6816] |
| CatBoost ranker | 0.5705 | [0.4831, 0.6566] |
| Random forest | 0.5595 | [0.4730, 0.6459] |
| CatBoost ranker + Tuned | 0.5323 | [0.4423, 0.6206] |
| CatBoost classifier + SMOTE + Tuned | 0.5175 | [0.4333, 0.6011] |
| CatBoost classifier + SMOTE | 0.3991 | [0.3119, 0.4848] |

CatBoost + ES minus tuned logistic regression: **−0.0048**, paired 95% CI **[−0.0511, 0.0388]**, two-sided p = **0.8368**, Holm-adjusted p = **1.000**. Against untuned logistic regression: **+0.0269**, CI **[−0.0223, 0.0764]**, adjusted p = **1.000**. Neither comparison establishes superiority or equivalence. Holm correction covers all 66 model pairs. The test period was inspected in earlier research, so these results are exploratory.

## Logistic coefficients and statsmodels

The predictive model uses L1 regularization (`C=0.0018798144`), positive-class weight 72.549639 (negative-class weight 1), and the `liblinear` solver. It has 37 nonzero coefficients out of 81 predictors, with an intercept of −0.006527. [All fitted coefficients](../notebooks/analysis_outputs/paper_0167ddd_473/paper_run/tables/logistic_coefficients.csv) include original pipeline units, odds ratios, and coefficients multiplied by training SD for scale comparison. These weighted scores are not calibrated population probabilities.

At the beginning, the notebook prints [statsmodels summary2()](../notebooks/analysis_outputs/paper_0167ddd_473/paper_run/tables/logistic_statsmodels_summary2.txt) for the **untuned, unweighted, unpenalized Logit baseline** using the same 44,868 training rows and all 81 predictors. Predictors are centered/scaled using training statistics; standard errors cluster by pump event (290 clusters). This fit converged in 13 iterations: log-likelihood −1471.1588, McFadden pseudo-R² 0.1599, AIC 3106.3175. Its p-values do not apply to the regularized predictive model and do not adjust for earlier research choices. The same fitted baseline is used for validation/test ranking and the all-model bootstrap; it does not replace the tuned primary portfolio model. Its validation AUC is 0.5939 and test AUC 0.6675.

## Reproduction

`notebooks/analysis_outputs/paper_0167ddd_473/test_auc_bootstrap/` contains fixed predictions, event AUCs, all bootstrap draws, distributions, pairwise tests, metadata and figures. `paper_run/` contains the full executed analysis, updated tables/figures, 473 feature-file hashes, partition membership and package versions. Run `MPLBACKEND=Agg .venv/bin/python scripts/run_test_auc_bootstrap.py --full` using the existing regenerated features.

The revised paper separates prediction from assumption-dependent execution simulations and limits external claims to Binance BTC pairs. The [previous 517-event report](results_517_events.md) and its `pumps_0167ddd` output folder remain archived; the [alternative four-part split report](results_refactored_19063b5.md) is also retained.
