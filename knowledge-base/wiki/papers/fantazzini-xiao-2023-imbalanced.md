---
type: paper
title: "Detecting Pump-and-Dumps with Crypto-Assets: Dealing with Imbalanced Datasets and Insiders' Anticipated Purchases"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - pump-detection
  - binance
  - smote
  - imbalanced-classification
  - random-forest
status: summarized
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Cross-Section]]"
  - "[[Pre-Pump Accumulation]]"
  - "[[Pump Announcement Window]]"
  - "[[Binance]]"
  - "[[backtest-pipelines]]"
  - "[[lamorgia-2023-doge]]"
  - "[[lamorgia-2020-icccn-realtime]]"
sources:
  - "https://www.mdpi.com/2225-1146/11/3/22"
year: 2023
authors:
  - "Dean Fantazzini"
  - "Yufeng Xiao"
venue: "Econometrics, vol. 11, no. 3, art. 30"
key_claim: "Random-forest classifiers trained on price and volume features with SMOTE oversampling can detect Binance pump-and-dump events one hour in advance, exploiting insiders' anticipated purchases in the pre-announcement window."
methodology: "Curate 2021–2022 Binance Telegram-pump dataset; engineer hourly price/volume features; balance the highly imbalanced training set with SMOTE; train and evaluate random-forest classifiers."
contradicts: []
supports:
  - "[[Pre-Pump Accumulation]]"
---

# Detecting Pump-and-Dumps with Crypto-Assets (Fantazzini & Xiao, 2023)

## TL;DR
Closest prior work to our SMOTE-augmented CatBoost baseline. Fantazzini and Xiao curate a 2021–2022 Binance pump dataset and ask whether insiders accumulating positions before the public announcement leave a detectable footprint one hour ahead of T0. They engineer price and volume features, address the extreme class imbalance with SMOTE, and train random forests to classify pumped vs non-pumped tokens. The headline contribution is methodological: explicit treatment of imbalance via synthetic oversampling, plus a justification rooted in the pre-leak insider-purchase mechanism.

## Key claims
- Pre-announcement insider buying creates a measurable price-volume footprint that random-forest classifiers can pick up one hour before the pump.
- The classification problem is severely imbalanced; SMOTE oversampling materially improves minority-class recall over naive training.
- A simple price + volume feature stack is sufficient to outperform random guessing by a wide margin on this dataset.

## Methodology
- Collect Binance pump events from Telegram channels active in 2021–2022.
- Build hourly price and volume features for each candidate token over the pre-announcement window.
- Apply SMOTE [[chawla-2002-smote]] to the training set to balance the positive class.
- Train random forests; report classification metrics under appropriate splits.

## Strengths
- First study to explicitly tie imbalance handling to the insider-anticipation mechanism: the imbalance is not just a nuisance, it reflects the structural rarity of pump targets.
- Reproducible feature stack on freely available Binance data.
- Clear "predict one hour ahead" framing aligns with practical surveillance.

## Weaknesses / Critiques
- SMOTE's known degradation in high-dimensional feature spaces is not analysed in detail; their feature stack is small enough to avoid this regime.
- No cross-sectional normalisation: each candidate token is scored in isolation rather than ranked within an event's universe.
- Single classifier family (random forest); no comparison with gradient boosting or sequence models.

## Relation to our work
- Cited in `paper/access.tex` (lines 92, 125) as the closest methodological precedent: same one-hour-ahead horizon, same Binance pump corpus era, same SMOTE-based imbalance treatment.
- Directly motivates our `CatboostClassifierSMOTE` pipeline ([[backtest-pipelines]]) as a baseline to beat.
- Our negative SMOTE result (line 457 in `access.tex`) updates this picture: SMOTE helps in their lower-dimensional, non-cross-sectionally-normalised setting, but degrades performance once features are cross-sectionally standardised and dimensionality grows past ~70. We attribute this to (i) SMOTE interpolating across cross-sections that have no shared scale and (ii) high-dimensional sparsity of the nearest-neighbour graph (consistent with [[blagus-lusa-2013-smote-highdim]]).
- Listed in our Table I as the SMOTE + random-forest baseline.

## Cited concepts
- [[Pump-and-Dump Scheme]]
- [[Pre-Pump Accumulation]]
- [[Pump Announcement Window]]
- [[Cross-Section]]
