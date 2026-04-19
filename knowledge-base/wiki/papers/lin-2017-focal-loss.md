---
type: paper
title: "Focal Loss for Dense Object Detection"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - ml-method
  - class-imbalance
  - loss-function
status: summarized
related:
  - "[[backtest-pipelines]]"
sources:
  - "https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf"
year: 2017
authors:
  - Tsung-Yi Lin
  - Priya Goyal
  - Ross Girshick
  - Kaiming He
  - Piotr Dollár
venue: "Proc. IEEE Int. Conf. Comput. Vis. (ICCV) 2017"
key_claim: "Re-weighting the cross-entropy loss with a (1 - p_t)^gamma factor focuses training on hard, misclassified examples and overcomes the foreground/background imbalance that prevents one-stage detectors from matching two-stage accuracy."
methodology: "Introduce focal loss FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t); train RetinaNet (one-stage detector) on COCO with this loss and benchmark against Faster R-CNN family."
contradicts: []
supports: []
url: "https://arxiv.org/abs/1708.02002"
---

# Focal Loss for Dense Object Detection (Lin et al., 2017)

## TL;DR
Proposes focal loss, a modulated cross-entropy that down-weights well-classified examples by a factor `(1 - p_t)^gamma`, shifting gradient mass onto hard, rare examples. Designed to fix the extreme foreground/background class imbalance (~1:1000) faced by single-stage object detectors. Combined with the RetinaNet architecture, focal loss lets a one-stage detector exceed the accuracy of slower two-stage models on COCO.

## Key claims
- Standard cross-entropy is dominated by the gradient from the abundant easy negatives even when each individual loss is small.
- Focal loss `FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)` smoothly reduces the contribution of confident predictions; gamma in [0, 5], with gamma=2 working best in their experiments.
- A simple one-stage detector (RetinaNet) trained with focal loss matches or beats the accuracy of two-stage detectors at higher speed.
- Focal loss is a drop-in replacement for cross-entropy and adds no inference-time cost.

## Methodology
- Anchor-based dense detector predicts class probabilities at every spatial location.
- Combine balancing prefactor `alpha_t` (class frequency weighting) with focusing factor `(1 - p_t)^gamma`.
- Ablate gamma and alpha jointly on COCO.

## Strengths
- Drop-in loss with one extra hyperparameter.
- General purpose: applies to any classifier facing severe imbalance, not only object detection.
- Clear intuition: easy examples contribute negligible gradient.

## Weaknesses
- Sensitive to gamma choice; optimal value depends on dataset imbalance ratio.
- Not a substitute for resampling when the minority class has too few effective examples.
- Most evidence comes from object detection; tabular and ranking applications need separate validation.

## Relation to our work
- Cited in our paper (`paper/access.tex`, `\cite{lin2017focal}`) in the limitations / future-work discussion of imbalance-handling approaches: we compared SMOTE and class weighting but did not test focal loss, cost-sensitive learning, or anomaly-detection style methods.
- A natural next experiment for our [[backtest-pipelines]] is to swap CatBoost's logloss for a focal-loss objective and re-run the Top@K%-AUC tuning, since pump-event imbalance (~1 target per 200+ same-window listings) is in the regime focal loss was designed for.

## Cited concepts
- [[Top-K AUC]]
- [[Cross-Section]]
- [[Pump-and-Dump Scheme]]
