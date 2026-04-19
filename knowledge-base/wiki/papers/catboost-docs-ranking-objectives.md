---
type: paper
title: "CatBoost Ranking Objectives (official docs + Lyzhin et al. 2023)"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - reference
  - catboost
  - learning-to-rank
status: summarized
related:
  - "[[Cross-Section]]"
  - "[[Top-K AUC]]"
  - "[[backtest-pipelines]]"
  - "[[ranking-for-event-prediction]]"
sources:
  - "https://catboost.ai/docs/en/concepts/loss-functions-ranking"
  - "https://arxiv.org/abs/2204.01500"
  - "http://proceedings.mlr.press/v119/ustimenko20a/ustimenko20a.pdf"
year: 2023
authors:
  - "CatBoost team (Yandex)"
  - "Ivan Lyzhin"
  - "Aleksei Ustimenko"
  - "Andrey Gulin"
  - "Liudmila Prokhorenkova"
venue: "Official documentation + ICML 2023"
key_claim: "CatBoost ships pairwise (PairLogit), listwise (YetiRank/YetiLoss), and pointwise-with-grouping (QueryRMSE, QuerySoftMax) ranking losses. YetiLoss with stochastic smoothing is the best available CatBoost ranking objective for arbitrary metrics; PairLogit is the simplest pairwise baseline."
methodology: "Doc summary plus the underlying ICML 2023 'Which Tricks Are Important for Learning to Rank?' paper that introduced YetiLoss as an upgrade to YetiRank."
contradicts: []
supports:
  - "[[ranking-for-event-prediction]]"
url: "https://catboost.ai/docs/en/concepts/loss-functions-ranking"
confidence: high
---

# CatBoost Ranking Objectives

## TL;DR
CatBoost is the ranker we use (`CatboostRanker` in `backtest/pipelines/`). It exposes a small zoo of ranking losses; choice matters most when groups are small or have only one positive label. PairLogit is a clean pairwise baseline, YetiRank/YetiLoss is the workhorse listwise loss with stochastic smoothing, QuerySoftMax is purpose-built for the "exactly one best document per query" setting and is the most natural fit for our P&D cross-sections.

## Objective catalogue (from the official docs)

### Pointwise
- **QueryRMSE** — RMSE computed within groups; cheap and robust but does not directly optimize ranking.
- **QuerySoftMax** — designed for binary targets where each group has one positive (label = 1, rest = 0); optimizes the probability that the positive is the best document. Effectively a softmax cross-entropy across the group's documents. Aligns 1:1 with our P&D setup.

### Pairwise
- **PairLogit** — pairwise logistic loss over (positive, negative) pairs generated independently per group; the standard pairwise baseline. Caveat: when there is only 1 positive per group, the number of useful pairs equals (# negatives), so the loss is well-defined and dense.
- **PairLogitPairwise** — same loss with a more accurate but slower pair-construction routine.

### Listwise
- **YetiRank** — stochastic listwise loss approximating ranking metrics (NDCG, PFound) via stochastic smoothing.
- **YetiRankPairwise** — slower, more accurate variant.
- **YetiLoss** family (post-2022) — extends YetiRank to optimize a *specific* listwise metric chosen by `mode`: `DCG | NDCG | MRR | ERR | MAP`. Lyzhin et al. 2023 show YetiLoss beats LambdaMART on MAP "by a considerable margin" and beats QueryRMSE in most settings.
- **StochasticRank** (Ustimenko & Prokhorenkova 2020) — older stochastic-smoothing approach; superseded by YetiLoss in CatBoost.

### Grouped AUC
- **QueryAUC** — AUC computed per group then averaged. Useful as an *eval metric* for the P&D project's top-k AUC analog.

## Key parameters that matter for our setting
- `group_id` — the pump-event identifier; CatBoost uses this to build pairs/lists.
- `top` — number of top samples used to compute the metric. Setting `top=1..3` aligns the optimizer with our top-k portfolio.
- Pair generation: max pairs per group; with 1 positive per group, capping pairs at `# negatives` is sufficient.
- For YetiRank/YetiLoss, the smoothing temperature controls how aggressively the loss approximates the discrete metric.

## Lyzhin et al. 2023 ("Which Tricks Are Important for Learning to Rank?")
- Compares LambdaMART, YetiRank, StochasticRank, and YetiLoss in a unified GBDT setup.
- Two design axes matter most: (1) whether the loss directly optimizes a smoothed ranking metric vs a convex surrogate, and (2) presence and form of stochastic smoothing of weights.
- YetiLoss (with smoothing) wins on MAP by a wide margin and is generally significantly better than QueryRMSE.
- Stochastic smoothing of pair weights is the key differentiator from LambdaMART, which uses delta-NDCG as a static reweighting.

## Strengths (for our use)
- Native support for grouping; first-class `CatBoostRanker` Python class.
- Pair / list construction is automated from `group_id`, so we do not hand-craft contrastive samples.
- YetiRank/YetiLoss let us directly optimize a metric close to our top-k AUC.

## Weaknesses / Critiques
- Documentation is sparse on edge cases (one positive, very small groups). Some defaults (e.g., max-pairs cap) can silently shrink training signal.
- `top` parameter behavior is implementation-specific; needs validation in our pipeline.
- YetiLoss is newer (post-2022 CatBoost releases); Optuna search spaces in `BasePipeline` may not include it yet.

## Relation to our work
- `CatboostRanker` in `backtest/pipelines/` currently uses YetiRank (default for `CatBoostRanker`). Worth A/B testing PairLogit and QuerySoftMax given our 1-positive-per-group structure.
- `CatboostClassifierTOPKAUC` post-hoc reweights toward top-k AUC; using `QueryAUC` as the in-training eval metric on a true ranker would be more principled.

## Cited concepts
- [[Cross-Section]]
- [[Top-K AUC]]
- YetiRank, PairLogit, QuerySoftMax (folded into this doc; promote to standalone concepts if cap allows later)
