---
type: thesis
title: "Ranking for Event Prediction"
created: 2026-04-19
updated: 2026-04-19
tags:
  - thesis
  - synthesis
  - learning-to-rank
  - ranking
  - cross-section
status: synthesized
related:
  - "[[Cross-Section]]"
  - "[[Top-K AUC]]"
  - "[[poh-2020-ltr-cross-sectional]]"
  - "[[catboost-docs-ranking-objectives]]"
  - "[[backtest-pipelines]]"
confidence: medium
---

# Ranking for Event Prediction

## Overview
The pumps_and_dumps detection task is a **cross-sectional ranking** problem disguised as a binary classification problem. For each pump event, we observe a cross-section of candidate assets and must surface the small set (typically 1) that the manipulator targeted. This synthesis page argues that learning-to-rank (LTR) objectives, evaluated by partial / top-k metrics, dominate per-asset binary classification on imbalanced financial event data, and recommends a concrete objective for our setup.

## Why Ranking Beats Classification

1. **Class imbalance is structural, not bug-fixable.** Per-asset binary classification has a base rate near zero (one target per ~hundreds of candidates per event). Resampling tricks (SMOTE, class weights) distort the marginal distribution but do not fix the fact that *we do not care* about absolute probabilities; we care about which asset wins the cross-section. Confidence: high.
2. **Cross-section regime drift kills calibrated probabilities.** Different events have different volatility, asset universes, and base rates. A ranker conditioned on `group_id` learns purely *relative* signal, sidestepping per-event level shifts. Cross-sectional standardization in [[backtest-pipelines]] is a partial fix on the input side; ranking objectives are the matching fix on the output side. Confidence: high.
3. **The decision is top-k portfolio selection.** The portfolio consumes the top-k items per cross-section; the rest are discarded. A model trained against pointwise log-loss optimizes a metric that is largely irrelevant to PnL ([[Top-K AUC]] is the right loss surface). Confidence: high.
4. **Empirical evidence from finance.** Poh, Lim, Roberts, Zohren (2020) reframed cross-sectional momentum as LTR and reported ~3x Sharpe improvement vs regression baselines, with LambdaMART best [[poh-2020-ltr-cross-sectional]]. The mechanism (rank-correctness > absolute-return-correctness) transfers directly to P&D event prediction. Confidence: high.

## Objective Choices

### Pointwise (logistic / regression with grouping)
- Examples: vanilla `CatBoostClassifier`, `LogisticRegression`, `RandomForest`, `QueryRMSE`.
- Pros: simple, interpretable.
- Cons: optimizes calibrated probabilities, not order. Cross-section regime drift hurts.
- Verdict: weak baseline. Keep in `backtest/pipelines/` for ablations only.

### Pairwise (PairLogit, RankNet, rank:pairwise)
- Loss is a logistic over (positive, negative) pairs constructed within group.
- With 1 positive per cross-section, every pair has a clear winner, giving dense, informative gradients (Lyzhin et al. note this is the LightGBM-issue critique of NDCG-only methods).
- Pros: simplest ranking objective that aligns with our setup; very stable when positives are rare.
- Cons: ignores the difference between "swap top-1 with rank-2" and "swap top-1 with rank-50" unless reweighted (LambdaMART's contribution).
- Verdict: strong default; **PairLogit is the safe choice**.

### Listwise (YetiRank / YetiLoss / NDCG-style LambdaMART)
- Optimizes a smoothed approximation to a listwise metric (NDCG, MAP, MRR, ERR, PFound).
- With 1 positive per group:
  - NDCG@k collapses to 1/log2(rank+1) of the positive: a smoothed MRR. Useful but loses the multi-level relevance signal that motivates NDCG.
  - MAP@k = MRR@k for single-positive groups.
  - Many listwise pairs degenerate to zero-gradient. Lyzhin et al. (Which Tricks 2023) note that *effective pair count* is what determines training signal; YetiLoss with stochastic smoothing recovers this and wins on MAP.
- Pros: directly approximates the metric we score against (Top-K AUC ≈ MRR-like signal at the top of the list).
- Cons: more hyperparameters (smoothing temperature, mode), slower, more sensitive to small groups.
- Verdict: best-case-better than PairLogit but only with careful tuning.

### Group-aware classification (QuerySoftMax, listMLE)
- QuerySoftMax (CatBoost) is *exactly* designed for "one positive document per query": it optimizes the softmax probability that the positive is the best document.
- Mathematically equivalent to fitting cross-entropy across the cross-section as a multinomial.
- Pros: bespoke fit for our setting; no pair construction needed.
- Verdict: **strongest candidate for our specific 1-positive-per-cross-section setup**, currently *not* used by any pipeline in `backtest/pipelines/`.

### Recommendation
For our setup (≥1 positive per cross-section, top-k portfolio):

1. **Primary**: `CatBoostRanker` with `QuerySoftMax` (or `PairLogit` if QuerySoftMax is unstable on small groups). Use `QueryAUC` as the in-training eval metric so it tracks our `calculate_topk_percent_auc`.
2. **Secondary**: `CatBoostRanker` with `YetiLoss(mode=NDCG)` or `YetiLoss(mode=MAP)`, tuned via Optuna. Expect marginal gains over PairLogit at the cost of tuning effort (Lyzhin et al. 2023 evidence).
3. **Avoid as headline model**: pure binary classifiers, including `CatboostClassifierTOPKAUC`'s sample-reweighted classifier. They are useful as ablations but not as the production ranker.

Confidence: medium-high. The QuerySoftMax recommendation is theoretically clean but empirically unverified in our codebase; an A/B against the current YetiRank default is the next concrete experiment.

## Evaluation

### Top-K AUC
The metric the portfolio cares about, see [[Top-K AUC]]. Equivalent to lower-left partial AUC (Shi et al. 2024). Our `calculate_topk_percent_auc` is the operational definition.

### NDCG@k
With 1 binary positive per group: NDCG@k = 1/log2(rank_of_positive + 1) if positive is in top-k else 0. Useful diagnostic but not strictly more informative than MRR@k.

### MAP@k
Equals MRR@k for single-positive groups. Strongly correlated with top-k AUC.

### When to prefer which
- Report **Top-K AUC** as headline (matches PnL).
- Report **MRR@k** (= MAP@k = NDCG@k up to log scaling) as the cleanest single-positive ranking number.
- Plain AUC and Precision@k are misleading defaults; keep them only as auxiliaries.

## Application to P&D
- Each pump event = one query / cross-section.
- Each candidate asset = one document.
- Label = 1 if asset is the manipulation target, 0 otherwise.
- Group features: cross-section z-scored returns, flow imbalance, slippage, multi-offset feature stack from [[features]].
- Portfolio simulation in [[backtest-portfolio]] consumes the top-k ranked items.
- Recommended objective: QuerySoftMax > PairLogit > YetiLoss(NDCG) > pointwise classification.

## Open Questions
- Empirical: does `QuerySoftMax` actually beat YetiRank on our test fold (post-2021-05-01)? File experiment under `gaps/` once run.
- Theoretical: do all members of one cross-section share enough features for cross-section z-scoring to be sufficient, or does the model still need an explicit `group_id` in features?
- Metric: is `calculate_topk_percent_auc` exactly the LLPAUC of Shi et al. 2024, or a different normalization? Documentation gap.
- Engineering: `BasePipeline` Optuna search space currently does not enumerate `QuerySoftMax` / `YetiLoss(mode=...)` modes. Worth wiring in.
- Robustness: very small cross-sections (e.g., events with <20 candidates) may break listwise smoothing. Need a minimum-group-size filter or a fallback to PairLogit.

## Sources
- Poh, D., Lim, B., Roberts, S., Zohren, S. (2020). *Building Cross-Sectional Systematic Strategies By Learning to Rank*. arXiv:2012.07149 / Journal of Financial Data Science 3(2). [[poh-2020-ltr-cross-sectional]]
- Lyzhin, I., Ustimenko, A., Gulin, A., Prokhorenkova, L. (2023). *Which Tricks Are Important for Learning to Rank?* ICML 2023. arXiv:2204.01500. (folded into [[catboost-docs-ranking-objectives]])
- Ustimenko, A., Prokhorenkova, L. (2020). *StochasticRank: Global Optimization of Scale-Free Discrete Functions*. ICML 2020.
- CatBoost docs: Ranking objectives and metrics. https://catboost.ai/docs/en/concepts/loss-functions-ranking. [[catboost-docs-ranking-objectives]]
- Burges, C. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview*. MSR-TR-2010-82.
- Gu, S., Kelly, B., Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning*. Review of Financial Studies 33(5). Cross-sectional rank-mapping of features into [-1, 1] is a precedent for our cross-section z-scoring.
- Shi, W. et al. (2024). *Lower-Left Partial AUC: An Effective and Efficient Optimization Metric for Recommendation*. arXiv:2403.00844. Formalizes top-k AUC.
- LightGBM docs (LGBMRanker), XGBoost docs (Learning to Rank tutorial) for cross-vendor comparison.
