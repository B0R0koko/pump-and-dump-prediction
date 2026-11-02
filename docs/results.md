# Results: restored 473-event paper cohort

This report describes the additional fixed-seed notebook experiment. The manuscript also preserves the original submitted numerical results: tuned L1 test AUC 0.697, CatBoost + ES validation/test AUC 0.667/0.674, and its original robustness results. The original standalone CatBoost portfolio results are retained only in the source archive; active portfolios and impact sensitivity use validation-selected CatBoost + ES. A brief reporting note identifies the results used for resampling. The [submitted-results snapshot](../research_snapshots/submitted_results/README.md) records the source and reported precision. Matching original per-event predictions were not found, so the current bootstrap intervals remain attached to the current estimates below. Differences are not attributed to missing seeds: the model pipelines already set seeds, and the cause has not been established.

The primary notebook uses the exact event identifiers underlying the paper at `0167dddfa6b55819e276812836ac7537c5d3a00b`: **498 original announcements minus 25 historical data-availability exclusions = 473 analyzed events**, split **290 training / 103 validation / 80 test**. The source records and exclusions come from `73ba716`, whose paper PDF is byte-identical to the PDF at `0167ddd`. [Cohort provenance](../research_snapshots/paper_0167ddd/provenance.json) records the source hashes and exact construction. The broader 517-event installation manifest stays available, but is not used by the primary research notebook or its model pipelines.

Existing regenerated features are reused unchanged. Restoring event membership does not restore old feature values or promise identical historical results. The six historical tuned candidates retain saved hyperparameters; a standalone untuned statsmodels fit runs first solely for explanatory diagnostics. The L2 logistic comparator selects C on validation data only; the seven tuned predictive models enter the comparisons, while the historical candidates retain their saved parameters.

**CatBoost + Top@k%AUC early stopping is the primary model**, selected automatically by the highest validation Top@k%AUC (0.6464), ahead of tuned CatBoost (0.6398), L2 logistic regression (0.6112), and tuned L1 logistic regression (0.6104). This fitted validation winner supplies all portfolio, impact and temporal sensitivity results. L2 has the highest descriptive test AUC (0.7100), but test rankings do not change the validation rule. Paired differences between CatBoost + ES and either tuned logistic variant are not significant. All seven tuned models remain in predictive comparisons; the previously inspected test period remains exploratory.

## Paired test bootstrap

10,000 draws of 80 whole events with replacement, seed 42, shared across fixed models. Top@k%AUC integrates from 0 to 20% in 0.5% increments and is normalized by 0.2. Intervals are pointwise percentile intervals, conditional on the fitted models and assuming independent events.

| Model | Test Top@k%AUC | 95% CI |
|---|---:|---:|
| Logistic regression L2 + Tuned | 0.7100 | [0.6327, 0.7839] |
| Logistic regression + Tuned | 0.7020 | [0.6247, 0.7759] |
| CatBoost classifier + Top@k%AUC early stopping | 0.6972 | [0.6169, 0.7750] |
| CatBoost classifier + Tuned | 0.6833 | [0.6052, 0.7573] |
| Random forest + Tuned | 0.5984 | [0.5133, 0.6816] |
| CatBoost ranker + Tuned | 0.5323 | [0.4423, 0.6206] |
| CatBoost classifier + SMOTE + Tuned | 0.5175 | [0.4333, 0.6011] |

CatBoost + ES minus tuned L1 logistic regression: **−0.0048**, paired 95% CI **[−0.0511, 0.0388]**, two-sided p = **0.8368**, Holm-adjusted p = **1.000**. This comparison establishes neither superiority nor equivalence. Holm correction covers all 21 model pairs. The test period was inspected in earlier research, so these results are exploratory.

L2 minus CatBoost + ES: **+0.0128**, paired 95% CI **[−0.0345, 0.0630]**, two-sided p = **0.6063**, Holm-adjusted p = **1.000**.

## L2 logistic comparison

The L2 model uses the same 81 preprocessed predictors and class weights as tuned L1 (positive weight 72.549639, negative weight 1). A fixed 25-point logarithmic grid searches C from 1e-6 to 1e2; fitting uses TRAIN and selection uses only validation Top@k%AUC. Ties choose smaller C. It uses liblinear with tol=1e-8, max_iter=2000 and convergence warnings treated as errors. Selected **C = 4.64159e-05**; **81/81 coefficients remain nonzero**.

Validation AUC: **0.6112**. Test AUC: **0.7100**, 95% CI **[0.6327, 0.7839]**. L1 minus L2: **-0.0080**, paired CI **[-0.0264, 0.0113]**, Holm-adjusted p = **1.0000**. The comparison is exploratory after earlier test inspection; CatBoost + ES is selected on validation; both logistic variants remain comparators.

[Validation path](../notebooks/analysis_outputs/paper_cohort_473_l2/paper_run/tables/logistic_l2_validation_path.csv), [selected parameters](../notebooks/analysis_outputs/paper_cohort_473_l2/paper_run/tables/logistic_l2_selection.json), and [L2 coefficients](../notebooks/analysis_outputs/paper_cohort_473_l2/paper_run/tables/logistic_l2_coefficients.csv) are exported alongside the coefficient comparison plot.

## Logistic coefficients and statsmodels

The L1 comparator uses L1 regularization (`C=0.0018798144`), positive-class weight 72.549639 (negative-class weight 1), and the `liblinear` solver. It has 37 nonzero coefficients out of 81 predictors, with an intercept of −0.006527. [All fitted coefficients](../notebooks/analysis_outputs/paper_cohort_473_l2/paper_run/tables/logistic_coefficients.csv) include original pipeline units, odds ratios, and coefficients multiplied by training SD for scale comparison. These weighted scores are not calibrated population probabilities.

At the beginning, the notebook prints [statsmodels summary2()](../notebooks/analysis_outputs/paper_cohort_473_l2/paper_run/tables/logistic_statsmodels_summary2.txt) for the **standalone untuned, unweighted, unpenalized Logit diagnostic** using the same 44,868 training rows and all 81 predictors. Predictors are centered/scaled using training statistics; standard errors cluster by pump event (290 clusters). This fit converged in 13 iterations: log-likelihood −1471.1588, McFadden pseudo-R² 0.1599, AIC 3106.3175. Its p-values do not apply to the regularized predictive model and do not adjust for earlier research choices. This diagnostic is excluded from validation/test comparisons, ranking plots and the paired bootstrap.

## Random portfolio reference

The main paper curve includes the fixed transaction cost and excludes size-dependent price impact.

The paper and notebook compare validation-selected CatBoost portfolios with 10,000 uniform random asset selections per size (seed 42), keeping the same 80 chronological test events, equal weights, execution rules and 25 bp round-trip cost. At k=5, CatBoost cumulative return is **107.89%**, versus a random mean of **−10.22%** and central 95% selection range **[−23.29%, 20.90%]**. Sharpe is **2.66**, versus a mean random Sharpe of **−1.46** (range **[−3.22, 0.62]**); target inclusion is **45.00%**, versus **1.74%**. Random metrics are computed for each draw before averaging. These conditional selection ranges are separate from event-bootstrap confidence intervals and do not remove the limitation from prior test inspection. [All portfolio comparisons](../notebooks/analysis_outputs/paper_cohort_473_l2/paper_run/tables/portfolio_vs_random.csv) and [random metric ranges](../notebooks/analysis_outputs/paper_cohort_473_l2/paper_run/tables/random_portfolio_metrics.csv) are exported with the exact selections and execution outcomes.

### Slippage at 1,000 USDT per asset

The paper repeats the portfolio comparison after the impact methodology. The [supplemental notebook](../notebooks/portfolio_1000usdt_slippage.ipynb) shows the same portfolios with fitted entry/exit price impact enabled and 25 bp round-trip cost. Each asset receives 1,000 USDT, so k=5 targets 5,000 USDT per event. CatBoost k=5 cumulative return is **71.19%** and Sharpe **1.81**; the random k=5 mean is **-32.99%**, with a pointwise terminal 95% selection range **[-46.11%, -2.22%]**. Returns are additive without reinvestment. The model return matches the existing 1,000-USDT sensitivity result. All 23,015 asset/event execution outcomes are shared by model and random portfolios, and every random selection is identical to the no-impact scenario. See the [figure](../notebooks/analysis_outputs/paper_cohort_473_l2/portfolio_1000usdt_slippage/portfolios_1000usdt_with_slippage.png), [model metrics](../notebooks/analysis_outputs/paper_cohort_473_l2/portfolio_1000usdt_slippage/model_statistics.csv), and [model/random comparisons](../notebooks/analysis_outputs/paper_cohort_473_l2/portfolio_1000usdt_slippage/model_vs_random.csv).

## Sensitivity analyses

Twenty-five 70% training-event subset fits of the L2 comparator retain the selected L2 C and class weights. Mean test AUC is **0.7069**, standard deviation **0.0074**, and central 80% range **[0.6981, 0.7152]**. Subset preprocessing is refitted using each subset; no historical L1 tuning parameters are loaded by this analysis.

The k=5 validation-selected CatBoost impact sweep gives cumulative ROE **0.9626** at 100 USDT, **0.2616** at 5,000 USDT, and **-0.0734** at 10,000 USDT. These are simulation outcomes under the stated fills, costs and impact function, not estimates of executable trading capacity.

## Reproduction

`notebooks/analysis_outputs/paper_cohort_473_l2/test_auc_bootstrap/` contains fixed predictions, event AUCs, all bootstrap draws, distributions, pairwise tests, metadata and figures. `paper_run/` contains the full executed analysis, updated tables/figures, 473 feature-file hashes, partition membership and package versions. Run `MPLBACKEND=Agg .venv/bin/python scripts/run_test_auc_bootstrap.py --full` using the existing regenerated features.

The revised paper separates prediction from assumption-dependent execution simulations and limits external claims to Binance BTC pairs. The [previous 12-model 473-event report](results_473_12_models.md) and its `paper_0167ddd_473` output folder remain archived. The [previous 517-event report](results_517_events.md) and its `pumps_0167ddd` output folder remain archived; the [alternative four-part split report](results_refactored_19063b5.md) is also retained.
