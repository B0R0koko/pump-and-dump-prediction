---
type: paper
title: "Mid-Price Prediction Based on Machine Learning Methods with Technical and Quantitative Indicators"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - ml-method
  - mid-price-prediction
  - lob
  - feature-engineering
status: summarized
related:
  - "[[features]]"
  - "[[Slippage]]"
sources:
  - "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0234107"
year: 2020
authors:
  - Adamantios Ntakaris
  - Juho Kanniainen
  - Moncef Gabbouj
  - Alexandros Iosifidis
venue: "PLOS ONE, vol. 15, no. 6, art. e0234107"
key_claim: "Carefully engineered technical and quantitative indicators on top of limit-order-book data let classical ML models (RF, SVM, GBM) deliver competitive short-horizon mid-price direction forecasts on the FI-2010 benchmark."
methodology: "Engineer a large bank of technical (moving averages, momentum, volatility) and quantitative (order-flow imbalance, depth, intensity) indicators from LOB snapshots; train RF, SVM, ridge regression, and GBM classifiers to predict next-period mid-price direction; benchmark on FI-2010."
contradicts: []
supports: []
url: "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0234107"
---

# Mid-Price Prediction Based on ML Methods with Technical and Quantitative Indicators (Ntakaris et al., 2020)

## TL;DR
Studies short-horizon mid-price direction forecasting from limit-order-book data. The contribution is feature engineering: a structured library of technical indicators (price-based, momentum, volatility) and quantitative indicators (order-flow imbalance, depth ratios, trade intensity) extracted from LOB snapshots. Standard classical ML models trained on this feature bank reach competitive accuracy on the FI-2010 high-frequency equity benchmark, demonstrating that handcrafted microstructure features remain a strong baseline against end-to-end deep models.

## Key claims
- Dense, hand-engineered LOB indicators outperform raw-LOB inputs for classical (non-deep) models on short-horizon direction prediction.
- Quantitative microstructure features (order-flow imbalance, depth-weighted prices, trade intensity) carry signal beyond purely technical indicators.
- Tree ensembles and SVMs are competitive baselines on FI-2010 once the right features are supplied.
- Feature scaling and stationarization choices materially affect cross-day generalization.

## Methodology
- Source data: FI-2010 high-frequency LOB benchmark (Nordic equities).
- Build a feature bank of technical and quantitative indicators per snapshot.
- Train RF, SVM, ridge, and GBM to classify next-tick mid-price direction (up / stationary / down).
- Evaluate accuracy, F1, and class-balanced metrics across forecast horizons.

## Strengths
- Reproducible on a public benchmark (FI-2010).
- Treats feature engineering as a first-class scientific question.
- Clear ablations across feature families.

## Weaknesses
- Equity LOB regime; results may not transfer directly to thinner crypto books.
- Short-horizon direction prediction, not ranking or PnL; backtest realism is limited.
- Classical models only; no comparison with modern transformer / temporal-CNN baselines.

## Relation to our work
- Cited in our paper (`paper/access.tex`, `\cite{ntakaris_midprice_2022}`, two locations) as a comparable feature-engineering approach for short-horizon market prediction and as an inspiration for the microstructure-based features we add to the [[features]] writer.
- Our feature design borrows the same philosophy: dense handcrafted indicators (asset returns, flow imbalance, [[Slippage]]) computed at multiple time offsets, fed into classical / boosted models rather than end-to-end deep models on raw LOB.

## Cited concepts
- [[Slippage]]
- [[Cross-Section]]
