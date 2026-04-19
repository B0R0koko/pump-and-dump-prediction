---
type: concept
title: "Cross-Section"
created: 2026-04-19
updated: 2026-04-19
tags:
  - concept
  - modeling
status: developing
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[features]]"
  - "[[backtest-pipelines]]"
  - "[[Top-K AUC]]"
  - "[[poh-2020-ltr-cross-sectional]]"
  - "[[catboost-docs-ranking-objectives]]"
  - "[[ranking-for-event-prediction]]"
sources:
  - "https://arxiv.org/abs/2012.07149"
  - "https://catboost.ai/docs/en/concepts/loss-functions-ranking"
complexity: intermediate
domain: "ranking ML"
aliases:
  - "per-event cross-section"
---

# Cross-Section

## Definition
For each pump event, the **cross-section** is the set of all candidate assets active in the same time window. Models rank assets within a cross-section to predict which one will be the manipulation target.

## Why it matters
Per-asset binary classification is the wrong framing: the base rate of "is target" is tiny, classes drift over time, and what we care about is *relative* likelihood within an event, not absolute probability. Ranking inside a cross-section sidesteps all three issues:

- Each cross-section has exactly one (or few) positives, so signal is dense within the unit.
- Cross-section z-scoring removes inter-event regime differences.
- Top-k evaluation matches how a real strategy would size positions.

## How it appears in this project
- [[features]] organizes outputs as (event, asset, offset) tuples.
- [[backtest-pipelines]] applies cross-section standardization in preprocessing and (in `CatboostRanker`) optimizes a learning-to-rank objective.
- `calculate_topk` and `calculate_topk_percent_auc` in [[backtest-utils]] evaluate at the cross-section level.
- [[backtest-portfolio]] picks top-k from each cross-section to build the portfolio.

## Ranking objectives suited to this framing
The cross-section framing is ML-agnostic, but its real value comes from pairing it with a **learning-to-rank** loss that scores assets relative to their group rather than in isolation. Headline options, in order of fit for our 1-positive-per-cross-section setting:

- **QuerySoftMax** (CatBoost) — softmax across the cross-section's documents, optimizing P(asset = best-in-group). Tailor-made for "exactly one positive per group". Currently *not* used by any pipeline in `backtest/pipelines/`; strongest candidate for replacing the existing default.
- **PairLogit** (CatBoost) / `rank:pairwise` (XGBoost) / RankNet — pairwise logistic loss over (positive, negative) pairs constructed within group. With one positive per cross-section every pair has a clear winner, so gradients are dense and stable. Safe default ranker.
- **YetiRank / YetiLoss** (CatBoost, default for `CatBoostRanker`) — listwise loss with stochastic smoothing approximating NDCG, MAP, MRR, ERR, or PFound. Wins on MAP-style metrics per Lyzhin et al. ICML 2023, but more sensitive to small groups and smoothing hyperparameters.
- **LambdaMART** (XGBoost `rank:pairwise` + delta-NDCG, LightGBM `lambdarank`) — pairwise-with-listwise-reweight. Closest published match for cross-sectional momentum [[poh-2020-ltr-cross-sectional]], but designed for multi-level relevance; partly wasted on binary 1-positive groups.
- **Pointwise classifiers** (`CatBoostClassifier`, `LogisticRegression`, `RandomForest`) — ignore the cross-section; only useful as ablation baselines.

Evaluation should track [[Top-K AUC]] (matches the portfolio's operating region) plus MRR@k (= MAP@k = NDCG@k up to log scaling for single-positive groups). See [[ranking-for-event-prediction]] for the full synthesis and recommendation.

## Related
- [[Pump-and-Dump Scheme]] — the event the cross-section is built around.
- [[Top-K AUC]] — evaluation metric matched to the cross-section framing.
- [[poh-2020-ltr-cross-sectional]] — strongest published precedent for LTR on cross-sectional finance.
- [[catboost-docs-ranking-objectives]] — catalogue of ranking losses available to `CatBoostRanker`.
- [[ranking-for-event-prediction]] — synthesis page including objective recommendation for our setup.
