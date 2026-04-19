---
type: paper
title: "Random Forests"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - ml-method
  - tree-ensemble
  - foundational
status: summarized
related:
  - "[[backtest-pipelines]]"
  - "[[Cross-Section]]"
sources:
  - "https://link.springer.com/article/10.1023/A:1010933404324"
year: 2001
authors:
  - Leo Breiman
venue: "Machine Learning, vol. 45, no. 1, pp. 5-32"
key_claim: "An ensemble of decision trees grown on bootstrap samples with a random subset of features at each split achieves low generalization error and is robust to noise without overfitting as more trees are added."
methodology: "Bagging (bootstrap aggregation) of CART trees combined with random feature subsampling at each split; out-of-bag samples used for unbiased error estimation and variable importance."
contradicts: []
supports: []
url: "https://doi.org/10.1023/A:1010933404324"
---

# Random Forests (Breiman, 2001)

## TL;DR
The foundational paper of the random forest algorithm. Combines bagging with random feature selection at each split to produce an ensemble of decorrelated trees. Generalization error converges as the number of trees grows (no overfitting from depth of the forest), and depends on the strength of individual trees and the correlation between them.

## Key claims
- Generalization error of a forest is bounded by `rho * (1 - s^2) / s^2`, where `s` is mean tree strength and `rho` is mean pairwise correlation. Random feature selection lowers `rho` without sacrificing too much `s`.
- Out-of-bag (OOB) samples (~37% of training data per tree) give an unbiased estimate of test error, removing the need for a held-out set.
- Variable importance can be computed from the OOB error increase when a feature's values are permuted.
- Random forests are competitive with AdaBoost and more robust to label noise.

## Methodology
- Draw `B` bootstrap samples from the training set; grow an unpruned CART tree on each.
- At each node, select `m` features uniformly at random from the full set of `p` (typical `m = sqrt(p)` for classification, `p/3` for regression) and choose the best split among them only.
- Aggregate by majority vote (classification) or mean (regression).

## Strengths
- Few hyperparameters; defaults work well across domains.
- Naturally handles mixed feature types, missing values, and high dimensionality.
- OOB error and permutation importance are nearly free.

## Weaknesses
- Memory and prediction cost grow linearly with the number of trees.
- Biased toward features with many possible split values (continuous and high-cardinality categoricals).
- Less competitive than gradient boosting (CatBoost, XGBoost, LightGBM) on most modern tabular benchmarks.

## Relation to our work
- Cited in the modeling section of `paper/access.tex` (`\cite{random_forest_2001}`) as one of the three classification baselines. Our `RandomForest` pipeline under [[backtest-pipelines]] (`backtest/pipelines/RandomForest/`) wraps scikit-learn's implementation and is tuned alongside Logistic Regression and CatBoost via Optuna.
- Random forest serves as a non-boosting tree-ensemble baseline. The CatBoost variants outperform it on Top@K%-AUC, but RF is retained to show the gradient-boosting gain is meaningful.

## Cited concepts
- [[Cross-Section]]
- [[Top-K AUC]]
