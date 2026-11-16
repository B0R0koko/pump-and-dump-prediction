# Results: 100 tuning and training seeds

The manuscript reports six model families across seeds 1–100, with one regularized Logistic regression model. The submitted PDF is retained only as the highlighting baseline. The cohort contains 473 events from 498 announcements and 25 recorded exclusions: 290 training, 103 validation and 80 test events.

Mean validation Top@k%AUC selects **CatBoost** (**0.6150** across 100 seeds). The preselected seed-42 model supplies all portfolio analyses. Each family's seed-42 point estimates and event-bootstrap intervals use the same saved predictions.

Each model and seed has 25 sequential TPE trials using a chronological 232/58 inner split of the 290 training events. The best configuration is fitted on all 290 events, with the inner-selected CatBoost tree count fixed and no evaluation-set early stopping. The 103 outer-validation events compare model families; the same configurations are evaluated on 80 test events without retuning. All 600 repeated final fits reproduced the saved validation and test predictions exactly. Fitted models, parameters, predictions and software/input hashes are saved.

## Seed distributions

| Model | Mean validation AUC | Validation SD | Mean test AUC | Test SD |
|---|---:|---:|---:|---:|
| CatBoost | 0.6150 | 0.0251 | 0.6786 | 0.0218 |
| Logistic regression | 0.6102 | 0.0036 | 0.7030 | 0.0015 |
| CatBoost + ES | 0.5903 | 0.0418 | 0.6576 | 0.0393 |
| Random forest | 0.5635 | 0.0154 | 0.6212 | 0.0205 |
| CatBoost ranker | 0.5189 | 0.0183 | 0.5846 | 0.0290 |
| CatBoost + SMOTE | 0.4849 | 0.0182 | 0.5087 | 0.0228 |

These distributions describe tuning/training randomness conditional on the fixed events. The paired-seed mean difference for logistic regression minus CatBoost on test is 0.0244, with 95% interval [0.0203, 0.0288]. Seed resampling does not measure event-sampling uncertainty or change the validation selection.

## Seed-42 point estimates and event bootstrap

| Model | Validation AUC | Test AUC | Test 95% CI | Test Top@5 |
|---|---:|---:|---:|---:|
| Logistic regression | 0.6120 | 0.7031 | [0.6270, 0.7761] | 0.3875 |
| CatBoost | 0.6285 | 0.6955 | [0.6164, 0.7714] | 0.4250 |
| CatBoost + ES | 0.6232 | 0.6856 | [0.6059, 0.7619] | 0.4250 |
| Random forest | 0.5585 | 0.6297 | [0.5469, 0.7103] | 0.3250 |
| CatBoost ranker | 0.5208 | 0.5709 | [0.4825, 0.6580] | 0.2625 |
| CatBoost + SMOTE | 0.4962 | 0.5308 | [0.4445, 0.6142] | 0.2625 |

AUC integrates Top@k% from 0–20% in 0.5% increments and divides by 0.2. The percentile intervals use 10,000 paired whole-event draws shared by all six fixed models. Two-sided centered tests use an add-one correction and Holm adjustment across the 15 pairs of reported models. The intervals assume independent events and omit training uncertainty; prior test inspection limits confirmatory interpretation.

## Portfolios

| k | Cumulative return, no size impact | Sharpe | Cumulative return, 1,000 USDT/asset impact | Sharpe | Target events |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7582 | 0.8008 | 0.3820 | 0.4051 | 11/80 |
| 2 | 1.2735 | 1.9460 | 0.9013 | 1.4008 | 17/80 |
| 5 | 0.9242 | 2.5322 | 0.5486 | 1.5402 | 34/80 |
| 10 | 0.5416 | 2.5298 | 0.1741 | 0.8329 | 44/80 |
| 20 | 0.2663 | 1.6490 | -0.0828 | -0.4951 | 53/80 |
| 30 | 0.2138 | 1.8467 | -0.1209 | -1.0211 | 60/80 |

Returns are decimal fractions, summed without reinvestment, with 25 bp round-trip costs. Each portfolio size also has 10,000 uniform random-selection draws. Asset orders are shared across sizes and both execution scenarios. Daily annualization uses a zero-filled common calendar and 365 days per year. Random-selection ranges are conditional benchmarks, not event-bootstrap intervals.

## Sensitivity and verification

CatBoost AUC across 100 seeds: mean 0.679, standard deviation 0.022, central 80% range [0.654, 0.700]. Training and evaluation event membership are fixed; each seed controls its inner search and final fit. Feature-importance quantiles use these same 100 CatBoost models.

The selected model's early-period AUC is 0.689 on 76 events. The four-event late period is excluded by the ten-event minimum.

[Seed-sweep audit](test_seed_sweep_results.json) independently verifies all 600 test scores and 15 paired-seed intervals. [Prediction and portfolio audit](seed_sweep_manuscript_audit.json) reconstructs the six seed-42 point metrics, 10,000 event-bootstrap draws, 15 event-bootstrap comparisons, portfolio returns and target counts. [Manuscript audit](manuscript_results_audit.json) checks the six-model comparison, its 15-pair adjustment, and every generated result table and figure against the exports. [Reproduction commands](reproduction.md) regenerate the outputs and both PDFs.

Manuscript artifacts: `notebooks/analysis_outputs/seed_sweep_paper/` (local, excluded from Git). The 600 checkpoints are in `test_seed_sweep_100x25/fits/`; the matching search databases and parameters are in `validation_seed_sweep_100x25/studies/`. The manuscript directory contains point metrics, event-bootstrap draws, seed distributions, feature importances and regenerated portfolio curves. Its provenance records the model-independent execution outcomes and random selections used for the portfolio replay.
