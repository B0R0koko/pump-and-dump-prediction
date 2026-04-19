---
type: paper
title: "SMOTE: Synthetic Minority Over-sampling Technique"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - ml-method
  - class-imbalance
  - oversampling
  - foundational
status: summarized
related:
  - "[[blagus-lusa-2013-smote-highdim]]"
  - "[[elkan-2001-cost-sensitive]]"
  - "[[backtest-pipelines]]"
  - "[[Cross-Section]]"
sources:
  - "https://www.jair.org/index.php/jair/article/view/10302"
year: 2002
authors:
  - Nitesh V. Chawla
  - Kevin W. Bowyer
  - Lawrence O. Hall
  - W. Philip Kegelmeyer
venue: "Journal of Artificial Intelligence Research, vol. 16, pp. 321-357"
key_claim: "Generating synthetic minority-class samples by linearly interpolating between a minority point and one of its k-nearest minority neighbors outperforms simple random oversampling and undersampling for imbalanced classification."
methodology: "For each minority sample x, find its k-nearest minority neighbors; for each desired synthetic sample, pick one neighbor x' and form x_new = x + lambda * (x' - x) with lambda ~ Uniform(0,1); combine with majority undersampling."
contradicts: []
supports: []
url: "https://doi.org/10.1613/jair.953"
---

# SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)

## TL;DR
The standard reference for synthetic oversampling under class imbalance. Instead of duplicating minority points (which causes overfit decision regions tightly around the existing minorities), SMOTE creates new points along line segments connecting a minority sample to its nearest minority neighbors. Combined with undersampling of the majority class, this expands the decision region for the minority class and improves recall and AUC.

## Key claims
- Random oversampling tightens, rather than enlarges, the minority decision region; SMOTE generalizes it.
- Combining SMOTE with majority undersampling beats either technique alone.
- Demonstrated improvements in AUC across nine UCI datasets using C4.5, Ripper, and Naive Bayes classifiers.

## Methodology
- For each minority example, retrieve its k nearest neighbors among the minority class (typical k=5) in feature space using Euclidean distance.
- To produce one synthetic sample: pick one of the k neighbors at random, then sample uniformly along the line segment in feature space connecting them.
- Repeat to reach the desired oversampling ratio; usually the majority class is also undersampled.

## Strengths
- Conceptually simple, easy to plug in as a preprocessing step.
- Works with any downstream classifier.
- Spawned a large family of variants (Borderline-SMOTE, ADASYN, SMOTE-ENN, SMOTE-Tomek).

## Weaknesses
- Assumes the minority class manifold is locally linear; interpolating between two minorities can produce points inside the majority distribution (especially with bounded or categorical features).
- Distance-based neighbor lookup degrades in high dimensions, see [[blagus-lusa-2013-smote-highdim]].
- Does not account for class overlap; can amplify noise near decision boundaries.
- Operates per-sample, ignoring grouped or panel structure (e.g., cross-sections).

## Relation to our work
- Cited in `paper/access.tex` ([[chawla-2002-smote]]) in two places: (1) describing Fantazzini and Xiao's prior work that combined SMOTE with random forests for pump detection, and (2) in our own modeling section where we test SMOTE as an imbalance remedy.
- Our `CatboostClassifierSMOTE` pipeline under [[backtest-pipelines]] applies SMOTE before training. We find it underperforms class weighting in our setting; the paper attributes this to bounded features, the [[Cross-Section]] normalization breaking neighborhood structure across cross-sections, and high-dimensional sparsity (only 227 positive samples vs 70+ features), echoing [[blagus-lusa-2013-smote-highdim]].

## Cited concepts
- [[Cross-Section]]
- [[Top-K AUC]]
