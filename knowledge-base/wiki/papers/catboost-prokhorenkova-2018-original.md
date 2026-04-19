---
type: paper
title: "CatBoost: Unbiased Boosting with Categorical Features"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - ml-method
  - gradient-boosting
  - catboost
status: summarized
related:
  - "[[catboost-docs-ranking-objectives]]"
  - "[[backtest-pipelines]]"
  - "[[Cross-Section]]"
sources:
  - "https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html"
year: 2018
authors:
  - Liudmila Prokhorenkova
  - Gleb Gusev
  - Aleksandr Vorobev
  - Anna Veronika Dorogush
  - Andrey Gulin
venue: "Advances in Neural Information Processing Systems (NeurIPS) 2018"
key_claim: "Standard gradient boosting suffers from a 'prediction shift' bias because residuals are computed using the same examples that trained the model. Ordered boosting and ordered target statistics fix this and yield consistently lower test error."
methodology: "Gradient boosted decision trees with two innovations: (1) ordered target statistics for categorical encoding using a random permutation of the data, (2) ordered boosting where each model in the chain is trained on a disjoint prefix of permuted examples to avoid leakage."
contradicts: []
supports: []
url: "https://arxiv.org/abs/1706.09516"
---

# CatBoost: Unbiased Boosting with Categorical Features (Prokhorenkova et al., 2018)

## TL;DR
The original CatBoost paper. Identifies a subtle bias ("prediction shift") in classical gradient boosting where gradient estimates leak through the same samples used to fit the model. Fixes it with ordered boosting (each tree's gradients are computed from a model trained without that example) and introduces ordered target statistics, a leakage-free way to encode high-cardinality categoricals. Result: a gradient boosting library that is robust to overfitting and handles categoricals natively without manual encoding.

## Key claims
- Greedy target statistics (mean target per category) leak the label and cause overfitting; ordered TS using a random permutation of the training set removes the leakage.
- Standard gradient boosting suffers from "prediction shift": the gradient of `F^{t-1}` evaluated on training point `x_i` is biased because `F^{t-1}` was trained on `x_i`.
- Ordered boosting maintains `n` supporting models, each trained on a different prefix of a single random permutation; gradients for sample `i` come from the model trained on the first `i-1` samples.
- On a battery of public datasets, CatBoost matches or beats XGBoost and LightGBM with default hyperparameters.

## Methodology
- Symmetric (oblivious) decision trees as the weak learner: every node at the same depth uses the same split, making prediction extremely fast.
- Multiple random permutations are kept and rotated across boosting iterations to reduce variance of the ordered TS and ordered gradients.
- Plain mode (classical gradient boosting) and ordered mode are both implemented; ordered mode wins on small/medium datasets, plain on very large ones.

## Strengths
- Native categorical support with no manual encoding.
- Strong out-of-the-box performance with minimal tuning.
- Symmetric trees give very fast inference and are easily exportable.
- Built-in support for ranking, classification, regression, and uncertainty losses.

## Weaknesses
- Ordered boosting is more compute-heavy per iteration than plain GBDT.
- Symmetric trees can underfit in very high-dimensional regimes versus more flexible boosters.
- Library is large; effective tuning still benefits from understanding `border_count`, `l2_leaf_reg`, `bagging_temperature`, etc.

## Relation to our work
- Cited in `paper/access.tex` as `\cite{catboost}` when introducing the CatBoost classifier baseline. Every CatBoost-* pipeline under [[backtest-pipelines]] (CatboostClassifier, CatboostClassifierSMOTE, CatboostClassifierTOPKAUC, CatboostRanker) calls into this library.
- This page covers the *core algorithm* (ordered boosting, ordered TS, oblivious trees). For the ranking-objective menu (`YetiRank`, `PairLogit`, `QuerySoftMax`) we use in `CatboostRanker`, see [[catboost-docs-ranking-objectives]].

## Cited concepts
- [[Cross-Section]]
- [[Top-K AUC]]
