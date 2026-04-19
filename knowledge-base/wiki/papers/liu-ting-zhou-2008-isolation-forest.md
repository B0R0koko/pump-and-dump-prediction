---
type: paper
title: "Isolation Forest"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - ml-method
  - anomaly-detection
  - unsupervised
status: summarized
related:
  - "[[backtest-pipelines]]"
  - "[[breiman-2001-random-forest]]"
sources:
  - "https://ieeexplore.ieee.org/document/4781136"
year: 2008
authors:
  - Fei Tony Liu
  - Kai Ming Ting
  - Zhi-Hua Zhou
venue: "Proceedings of the IEEE International Conference on Data Mining (ICDM), pp. 413-422"
key_claim: "Anomalies can be detected by isolating points via random axis-aligned partitions; anomalous points have shorter average path lengths to isolation in randomly grown trees, providing a fast unsupervised anomaly score with linear time and constant memory."
methodology: "Build an ensemble of isolation trees: for each tree, recursively select a random feature and a random split value within its range until each sample is in its own leaf. The expected path length to isolation, normalized by the average path length of an unsuccessful BST search, is the anomaly score."
contradicts: []
supports: []
url: "https://doi.org/10.1109/ICDM.2008.17"
---

# Isolation Forest (Liu, Ting, Zhou, 2008)

## TL;DR
A fast, unsupervised anomaly-detection algorithm that flips the usual anomaly-detection logic. Instead of profiling normal data and flagging deviations, isolation forest exploits the fact that anomalies are "few and different" so they require fewer random partitions to isolate from the rest of the data. The anomaly score is a function of the average isolation depth across an ensemble of random trees. No distance computations, sub-linear in the number of samples, and works in high dimensions.

## Key claims
- Anomalies have noticeably shorter expected isolation path lengths than normal points.
- The ensemble works with very small subsamples (256 by default), giving constant memory and O(n) total time.
- Outperforms distance-based methods (LOF, ORCA) on benchmark datasets while being orders of magnitude faster.
- Robust to swamping (normal points labeled anomalous) and masking (anomalies hidden in clusters) when subsampling is used.

## Methodology
- Build `t` isolation trees, each on a subsample of size `psi` (typically 256).
- At each node, select a feature uniformly at random and a split value uniformly in its observed range.
- Recursively partition until each leaf has one sample or the tree reaches a height limit `ceil(log2(psi))`.
- Anomaly score `s(x, n) = 2^(-E[h(x)] / c(n))` where `E[h(x)]` is the mean path length across the ensemble and `c(n)` is the average path length of an unsuccessful BST search of `n` items.

## Strengths
- Linear time, constant memory, embarrassingly parallel.
- Hyperparameter-light: only the number of trees and subsample size matter much.
- No assumptions about data distribution; works on mixed-density and high-dimensional data.

## Weaknesses
- Axis-aligned splits struggle with anomalies that are unusual only along oblique directions; extended isolation forest addresses this with hyperplane splits.
- Score is unitless and not directly probabilistic; threshold selection requires labeled validation data or a contamination prior.
- Doesn't explicitly model temporal or sequential structure; not a time-series anomaly detector out of the box.

## Relation to our work
- Cited in `paper/access.tex` as `\cite{liu2008isolation}` in the discussion section, listed alongside focal loss and cost-sensitive learning ([[elkan-2001-cost-sensitive]]) as imbalance-handling approaches we have not explored. Specifically flagged as representative of "deep anomaly detection / unsupervised baselines" that remain future work for P&D detection.
- Conceptually adjacent to [[breiman-2001-random-forest]] (random tree ensemble) but with the opposite training objective: isolate rather than predict. A natural sanity-check baseline against our supervised [[backtest-pipelines]] models.

## Cited concepts
- [[Cross-Section]]
