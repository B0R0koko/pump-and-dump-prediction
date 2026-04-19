---
type: module
title: "backtest-pipelines"
created: 2026-04-19
updated: 2026-04-19
tags:
  - module
  - ml
  - backtest
status: seed
related:
  - "[[features]]"
  - "[[backtest-portfolio]]"
  - "[[backtest-utils]]"
  - "[[Cross-Section]]"
sources: []
path: "backtest/pipelines/"
language: python
purpose: "ML model implementations and training orchestration. BasePipeline plus CatboostClassifier(SMOTE/TOPKAUC), CatboostRanker, LogisticRegression, RandomForest. Handles splitting, preprocessing (cross-section standardization), training, and Optuna tuning."
maintainer: borokoko
last_updated: 2026-04-19
depends_on:
  - "[[core]]"
  - "[[features]]"
used_by:
  - "[[backtest-portfolio]]"
  - "[[backtest-utils]]"
---

# backtest-pipelines

## Purpose
Trains ranking models that score assets within each pump-event cross-section.

## Hierarchy
- `BasePipeline` (abstract) → CatboostClassifier, CatboostClassifierSMOTE, CatboostClassifierTOPKAUC, CatboostRanker, LogisticRegression, RandomForest.
- `BaseModel` (abstract) wraps the trained estimator.

## Time splits
- Train: < 2020-09-01
- Validation: 2020-09-01 – 2021-05-01
- Test: > 2021-05-01

## Preprocessing
Cross-section standardization: features are z-scored within each pump event.

## Hyperparameter tuning
Optuna.

## Design notes
- Ranking framing (CatboostRanker) is the headline model; classifiers exist as baselines.
- TOPKAUC variant optimizes the metric the portfolio actually consumes (top-k AUC).

## References

### Tree models
- [[breiman-2001-random-forest]] — RandomForest baseline.
- [[catboost-prokhorenkova-2018-original]] — CatBoost algorithm underpinning the headline classifiers and ranker.
- [[catboost-docs-ranking-objectives]] — CatBoost ranking objectives used by CatboostRanker.
- [[grinsztajn-2022-tree-tabular]] — empirical justification for tree models on tabular data over deep nets.

### Class imbalance
- [[chawla-2002-smote]] — SMOTE oversampling used in CatboostClassifierSMOTE.
- [[blagus-lusa-2013-smote-highdim]] — explains why SMOTE underperformed in our high-dimensional setting.
- [[elkan-2001-cost-sensitive]] — class-weighting alternative to resampling.
- [[lin-2017-focal-loss]] — unexplored alternative loss for extreme imbalance.
- [[liu-ting-zhou-2008-isolation-forest]] — unexplored unsupervised anomaly-detection baseline.
- [[fantazzini-xiao-2023-imbalanced]] — closest prior SMOTE+RF work on P&D detection.

### Ranking
- [[poh-2020-ltr-cross-sectional]] — cross-sectional learning-to-rank rationale informing the ranker framing.
- [[ranking-for-event-prediction]] — internal thesis recommending the ranking objective.

### Tooling
- [[akiba-2019-optuna]] — hyperparameter optimization framework used across pipelines.
