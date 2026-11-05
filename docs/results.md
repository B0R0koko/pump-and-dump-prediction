# Results: fixed-seed notebook run

The manuscript uses this completed run throughout. The submitted PDF is retained only as the highlighting baseline. The cohort contains 473 events from 498 announcements and 25 recorded exclusions: 290 training, 103 validation and 80 test events.

Validation selects **CatBoost** (Top@k%AUC **0.6529**). Its saved predictions supply all portfolio analyses. Each model's point estimates and bootstrap intervals use that model's same saved predictions.

Training and resampling use seed 42 and one CPU thread; Optuna searches are configured for sequential seeded TPE trials. This execution fits the six saved study configurations and reruns the L2 validation grid; it does not rerun the six Optuna searches. The study database and feature inputs are unchanged. Fitted model files, exact feature order, package versions, input hashes and executed notebooks are saved with the run.

## Model comparisons

| Model | Validation AUC | Test AUC | Test 95% CI | Test Top@5 |
|---|---:|---:|---:|---:|
| Logistic regression L2 | 0.6112 | 0.7100 | [0.6327, 0.7839] | 0.4000 |
| Logistic regression L1 | 0.6104 | 0.7020 | [0.6247, 0.7759] | 0.4375 |
| CatBoost + ES | 0.6459 | 0.6942 | [0.6137, 0.7711] | 0.4375 |
| CatBoost | 0.6529 | 0.6887 | [0.6120, 0.7636] | 0.4500 |
| Random forest | 0.5647 | 0.5984 | [0.5133, 0.6816] | 0.3125 |
| CatBoost ranker | 0.5006 | 0.5323 | [0.4423, 0.6206] | 0.2125 |
| CatBoost + SMOTE | 0.5235 | 0.5175 | [0.4333, 0.6011] | 0.2500 |

AUC integrates Top@k% from 0–20% in 0.5% increments and divides by 0.2. The percentile intervals use 10,000 paired whole-event draws shared by all seven fixed models. Two-sided centered tests use an add-one correction and Holm adjustment across 21 pairs. The intervals assume independent events and omit training uncertainty; prior test inspection limits confirmatory interpretation.

## Portfolios

| k | Cumulative return, no size impact | Sharpe | Cumulative return, 1,000 USDT/asset impact | Sharpe | Target events |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7281 | 0.8250 | 0.3496 | 0.3980 | 15/80 |
| 2 | 1.1268 | 1.9258 | 0.7467 | 1.2951 | 22/80 |
| 5 | 0.9746 | 2.8581 | 0.6149 | 1.8693 | 36/80 |
| 10 | 0.4654 | 1.7943 | 0.1113 | 0.4259 | 40/80 |
| 20 | 0.3469 | 2.0796 | 0.0048 | 0.0291 | 53/80 |
| 30 | 0.1998 | 1.7648 | -0.1298 | -1.1131 | 57/80 |

Returns are decimal fractions, summed without reinvestment, with 25 bp round-trip costs. Each portfolio size also has 10,000 uniform random-selection draws. Asset orders are shared across sizes and both execution scenarios. Daily annualization uses a zero-filled common calendar and 365 days per year. Random-selection ranges are conditional benchmarks, not event-bootstrap intervals.

## Sensitivity and verification

CatBoost + ES training-subset AUC: mean 0.688, standard deviation 0.014, central 80% range [0.668, 0.702]. L2: mean 0.707, standard deviation 0.007, range [0.698, 0.715]. Both use 25 random 70% training-event subsets with fixed hyperparameters and validation/test membership.

The selected model's early-period AUC is 0.695 on 76 events. The four-event late period is excluded by the ten-event minimum.

[Prediction and resampling replay audit](notebook_rerun_audit.json) independently reconstructs all point metrics, 10,000 bootstrap draws, 21 paired tests, portfolio returns and target counts, and random selections. [Manuscript audit](manuscript_results_audit.json) checks every generated result table and figure against the exports. [Reproduction commands](reproduction.md) regenerate the outputs and both PDFs.

Run artifacts: `notebooks/analysis_outputs/paper_cohort_473_l2/` (local, excluded from Git). `paper_run/models/` stores all seven fitted models; `paper_run/source_snapshot/` preserves sources used by the execution. `test_auc_bootstrap/` contains predictions and exact event draws, and the two portfolio directories contain execution outcomes and exact random selections.
