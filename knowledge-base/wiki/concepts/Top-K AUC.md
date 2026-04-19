---
type: concept
title: "Top-K AUC"
created: 2026-04-19
updated: 2026-04-19
tags:
  - concept
  - metric
  - ranking
  - evaluation
status: developing
related:
  - "[[Cross-Section]]"
  - "[[ranking-for-event-prediction]]"
  - "[[backtest-utils]]"
  - "[[backtest-pipelines]]"
sources:
  - "https://arxiv.org/abs/2403.00844"
  - "https://catboost.ai/docs/en/concepts/loss-functions-ranking"
complexity: intermediate
domain: "ranking ML"
aliases:
  - "Partial AUC"
  - "Top-K Percent AUC"
  - "Lower-Left Partial AUC"
---

# Top-K AUC

## Definition
**Top-k AUC** is a per-group ranking metric: within each cross-section it computes the area under the ROC curve restricted to the top-k highest-scored items (equivalently, the lower-left partial AUC). It rewards a model when the positive label is ranked inside the top-k candidates and penalizes false positives that crowd that prefix. It is the metric our portfolio actually consumes: only the top-k assets per pump event are bought.

In the project the implementation lives in `backtest/utils/` as `calculate_topk` and `calculate_topk_percent_auc`, applied per pump event then averaged.

## Why it matters
Standard AUC integrates over the entire ROC curve, so a model that gets the global ordering right but misses the very top is rewarded as much as one that nails the top. For trading strategies that consume only the top-k items per event, this is the wrong signal:

- A 10% gain in mid-rank ordering shows up in AUC but does not change PnL.
- A 1% miss at rank 1 vs rank 2 changes PnL dramatically but barely moves AUC.
- For rare-event detection (one positive per group), AUC's lower-left corner *is* the entire decision-relevant region (Shi et al. 2024, Lower-Left Partial AUC).

Top-k AUC restricts the ROC area to the operating region the strategy actually uses, so optimizing it correlates much more tightly with realized portfolio returns.

## How it appears in this project
- `backtest/utils/`: `calculate_topk_percent_auc` averages a partial-AUC variant per pump event.
- `backtest/pipelines/CatboostClassifierTOPKAUC/`: a CatBoost classifier with sample reweighting designed to push training toward top-k AUC.
- `backtest/pipelines/CatboostRanker/`: a true ranking pipeline whose YetiRank / YetiLoss objective approximates a similar listwise metric directly.
- Evaluation pipeline reports top-k AUC alongside top-k accuracy and portfolio Sharpe.

## Comparison to neighbour metrics
- **Plain AUC**: integrates whole ROC, ignores operating region.
- **Precision@k**: binary, bounded by 1/k for single-positive groups; loses information about ranks 1..k.
- **NDCG@k**: graded relevance; with a single binary positive collapses to 1/log2(rank+1), which is an MRR-like statistic and can be a reasonable substitute.
- **MAP@k**: equivalent to MRR@k when there is one positive per group.
- **Lower-Left Partial AUC (LLPAUC)**: the formal name in the recommender-systems literature; computationally efficient surrogate strongly correlated with top-k metrics (Shi et al. 2024).
- **CatBoost QueryAUC**: per-group AUC averaged across queries; the natural eval-metric handle for top-k AUC inside CatBoost training.

## Examples
- Pump event with 200 candidate assets, k=3. Model A ranks the true target at position 4 (top-k AUC ≈ 0). Model B ranks it at position 1 (top-k AUC ≈ 1). Both can have similar global AUC if Model A's mid-rank ordering is otherwise excellent. Only top-k AUC distinguishes them.

## Open questions
- Exact normalization of `calculate_topk_percent_auc` vs the LLPAUC literature: is the project using a TPR-restricted, FPR-restricted, or hybrid partial AUC? Document and unit-test.
- Whether to swap `CatboostClassifierTOPKAUC` reweighting for a `CatBoostRanker` with `QueryAUC` eval metric (see [[ranking-for-event-prediction]]).
