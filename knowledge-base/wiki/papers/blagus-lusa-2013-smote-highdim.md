---
type: paper
title: "SMOTE for High-Dimensional Class-Imbalanced Data"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - ml-method
  - class-imbalance
  - smote
  - critique
status: summarized
related:
  - "[[chawla-2002-smote]]"
  - "[[elkan-2001-cost-sensitive]]"
  - "[[backtest-pipelines]]"
  - "[[Cross-Section]]"
sources:
  - "https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-106"
year: 2013
authors:
  - Rok Blagus
  - Lara Lusa
venue: "BMC Bioinformatics, vol. 14, art. 106"
key_claim: "In high-dimensional settings, SMOTE often fails to improve classification performance over simple under-/oversampling, and can hurt classifiers that are not strongly affected by class imbalance to begin with (e.g., regularized linear models)."
methodology: "Simulation study and analysis of fifteen high-dimensional gene-expression datasets; compares no-resampling, random oversampling, random undersampling, and SMOTE across k-NN, SVM, random forest, prediction analysis for microarrays, and penalized logistic regression."
contradicts:
  - "[[chawla-2002-smote]]"
supports: []
url: "https://doi.org/10.1186/1471-2105-14-106"
---

# SMOTE for High-Dimensional Class-Imbalanced Data (Blagus & Lusa, 2013)

## TL;DR
A critical empirical study. SMOTE's interpolation step relies on a meaningful k-nearest-neighbor graph, but in high-dimensional spaces all points are roughly equidistant (the curse of dimensionality), so SMOTE injects synthetic points that are not really "between" minority members. The authors show that for gene-expression data with thousands of features and a few dozen samples, SMOTE rarely beats simple random oversampling and sometimes degrades classifiers (notably k-NN and SVM) that depend on local geometry.

## Key claims
- SMOTE is biased toward the majority class in high dimensions: synthetic minorities have feature means that drift toward the overall data mean.
- For classifiers insensitive to class imbalance (random forest with class weighting, penalized logistic regression), SMOTE provides no benefit and can hurt.
- For k-NN and SVM, SMOTE often hurts because synthetic points distort neighborhood structure.
- Simple random undersampling of the majority class is often competitive with or better than SMOTE in high-dimensional regimes.

## Methodology
- Simulated datasets with controlled dimensionality and class balance, plus 15 publicly available microarray datasets.
- Five classifiers evaluated, four resampling strategies (none, random over, random under, SMOTE).
- Performance measured by AUC, sensitivity, specificity, and predicted class probabilities; compared via cross-validation and bootstrapping.

## Strengths
- Large, well-controlled empirical study with both synthetic and real data.
- Provides a clear mechanistic explanation (centroid shift, broken neighborhood structure) rather than just numbers.
- Influential in shifting practice toward class weighting and cost-sensitive losses for high-dimensional problems.

## Weaknesses
- Restricted to microarray-style data (continuous, dense, modest sample sizes); less direct evidence for sparse or mixed-type features.
- Predates many SMOTE variants designed to address neighborhood-quality issues (Borderline-SMOTE, ADASYN, KMeans-SMOTE).
- Does not study tree-based gradient boosting, the dominant tabular model today.

## Relation to our work
- Cited in `paper/access.tex` in the discussion of why SMOTE underperformed in our experiments ([[blagus-lusa-2013-smote-highdim]]). Our P&D dataset has 70+ features and only ~227 positive training samples per cross-section, exactly the high-dimensional sparse regime they describe.
- This paper anchors our justification for preferring class weighting (and motivates the [[Top-K AUC]] custom metric) over SMOTE for the [[backtest-pipelines]] CatBoost variants. See also [[chawla-2002-smote]] for the original method and [[elkan-2001-cost-sensitive]] for the alternative we lean on.

## Cited concepts
- [[Cross-Section]]
- [[Top-K AUC]]
