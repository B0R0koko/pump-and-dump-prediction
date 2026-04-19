---
type: paper
title: "The Foundations of Cost-Sensitive Learning"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - ml-method
  - class-imbalance
  - cost-sensitive
  - foundational
status: summarized
related:
  - "[[chawla-2002-smote]]"
  - "[[blagus-lusa-2013-smote-highdim]]"
  - "[[backtest-pipelines]]"
  - "[[Top-K AUC]]"
sources:
  - "https://cseweb.ucsd.edu/~elkan/rescale.pdf"
year: 2001
authors:
  - Charles Elkan
venue: "Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI), pp. 973-978"
key_claim: "Cost-sensitive classification reduces to choosing a decision threshold that depends on the misclassification cost ratio; equivalently, training-set rebalancing by a known factor can replicate any desired cost matrix under a calibrated probabilistic classifier."
methodology: "Theoretical derivation of the optimal Bayes decision rule under an arbitrary cost matrix; constructive proof that resampling the training distribution by the cost ratio yields the same decision boundary as threshold adjustment on the original distribution."
contradicts: []
supports: []
url: "https://cseweb.ucsd.edu/~elkan/rescale.pdf"
---

# The Foundations of Cost-Sensitive Learning (Elkan, 2001)

## TL;DR
A short, foundational theoretical note. Shows that under a 2x2 cost matrix, the Bayes-optimal decision is a simple threshold on the predicted probability, where the threshold is the cost ratio `c(0,1) / [c(0,1) + c(1,0)]`. As a corollary, rebalancing the training set by exactly the cost ratio (oversampling the minority or undersampling the majority) is mathematically equivalent to leaving the data alone and shifting the decision threshold. Provides the theoretical scaffold most modern imbalanced-learning practice rests on.

## Key claims
- Optimal decision under cost matrix C: predict class 1 iff `p(1|x) >= p_*` where `p_* = C(0,1) / [C(0,1) + C(1,0)]`.
- Rebalancing the training distribution by factor `r = (1 - p_*) / p_*` for the minority class produces a classifier whose 0.5 threshold reproduces the cost-sensitive optimal decision.
- ROC AUC is invariant under class rebalancing; only the threshold (and hence accuracy, F1, precision, recall at fixed threshold) changes.
- Probability calibration matters: rebalancing requires a back-correction to recover the true posterior `p(1|x)` if downstream tasks need calibrated probabilities.

## Methodology
- Decision-theoretic derivation from the Bayes risk integral.
- Worked example on UCI datasets showing that learning algorithms calibrated for 50/50 priors generalize correctly to other priors via rescaling.

## Strengths
- Clean theoretical statement that subsumes oversampling, undersampling, and threshold tuning into a single framework.
- Highlights that AUC and PR curves describe the model independent of the operating point.
- Widely cited as the formal justification for class weighting in modern libraries (sklearn `class_weight`, CatBoost `class_weights`).

## Weaknesses
- Assumes the underlying classifier outputs calibrated probabilities, which many flexible models (e.g., raw boosted trees, deep nets) do not.
- Two-class framing; multiclass extensions need additional assumptions.
- Practical performance under finite-sample, high-dimensional, or non-stationary regimes is not addressed (this is where SMOTE-style critiques like [[blagus-lusa-2013-smote-highdim]] become relevant).

## Relation to our work
- Cited twice in `paper/access.tex` as `\cite{elkan2001cost}`: (1) when justifying PR-AUC over ROC-AUC under extreme imbalance, and (2) when listing cost-sensitive learning (alongside focal loss and deep anomaly detection) as an unexplored alternative to our SMOTE / class-weighting comparison.
- Our [[backtest-pipelines]] use class weighting in the CatBoost variants; Elkan's result is the formal reason that should be (asymptotically) equivalent to SMOTE plus threshold adjustment.

## Cited concepts
- [[Cross-Section]]
- [[Top-K AUC]]
