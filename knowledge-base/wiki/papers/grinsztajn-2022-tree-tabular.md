---
type: paper
title: "Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - ml-method
  - tabular
  - benchmark
  - tree-vs-dl
status: summarized
related:
  - "[[backtest-pipelines]]"
  - "[[features]]"
sources:
  - "https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html"
year: 2022
authors:
  - Léo Grinsztajn
  - Edouard Oyallon
  - Gaël Varoquaux
venue: "Advances in Neural Information Processing Systems (NeurIPS) 2022, Datasets and Benchmarks track"
key_claim: "On medium-sized tabular datasets, gradient-boosted decision trees and random forests outperform deep architectures (MLP, ResNet, FT-Transformer, SAINT) under matched HPO budgets, driven by three inductive-bias mismatches in neural networks: rotation variance, robustness to uninformative features, and ability to learn irregular target functions."
methodology: "Curated benchmark of 45 tabular datasets (numerical-only and mixed numerical+categorical) with standardized splits and HPO budgets. Compares trees (XGBoost, RF, GBT) vs. neural baselines (MLP, ResNet, FT-Transformer, SAINT). Adds controlled perturbations (random rotations, added uninformative features, smoothed targets) to isolate causes of the gap."
contradicts: []
supports: []
url: "https://arxiv.org/abs/2207.08815"
---

# Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data? (Grinsztajn, Oyallon, Varoquaux, 2022)

## TL;DR
Empirical benchmark showing that on typical mid-sized tabular tasks, tree ensembles still beat the best published deep tabular models even after careful hyperparameter tuning. Identifies three concrete inductive-bias gaps that explain the difference and demonstrates each with controlled perturbation experiments.

## Key claims
- Across 45 datasets, tree ensembles (XGBoost, GBT, RF) outperform neural baselines under matched HPO budgets.
- Cause 1, irregular target functions: NNs are biased toward smooth functions; trees handle piecewise / discontinuous targets better.
- Cause 2, uninformative features: NNs degrade when uninformative columns are added; trees tolerate them via implicit feature selection.
- Cause 3, rotation invariance: NNs are approximately rotation-invariant in feature space, which is the wrong prior when columns have semantic meaning; trees are axis-aligned.
- The gap shrinks with dataset size but does not close in the regime tested (~10k–50k rows).

## Methodology
- Standardized benchmark suite with reproducible preprocessing.
- HPO done with random search, large budget per model.
- Perturbation experiments: rotate features, add Gaussian-noise columns, smooth target via local averaging, and re-measure relative gap.

## Strengths
- Large, well-curated benchmark with public protocol.
- Goes beyond "trees won" to explain *why*, with controlled experiments.
- Fair HPO budgets across model families.

## Weaknesses
- Excludes very large tabular datasets (>1M rows) where DL may close the gap.
- Time-series and panel structure not represented.
- Class imbalance handling is generic; severe-imbalance settings need separate evaluation.

## Relation to our work
- Cited in our paper (`paper/access.tex`, `\cite{grinsztajn_2022}`) to justify our choice of tree-based models (CatBoost classifier, CatBoost ranker, Random Forest) over deep learning for the pump-target prediction task.
- Our setting matches the regime they study: ~70 handcrafted [[features]] per cross-section, mid-sized dataset, mixed numerical features with semantic meaning, irregular target (1 manipulator pick per [[Cross-Section]]).
- Backs the design of [[backtest-pipelines]], which only ships tree models and logistic regression as a linear baseline.

## Cited concepts
- [[Cross-Section]]
- [[Top-K AUC]]
