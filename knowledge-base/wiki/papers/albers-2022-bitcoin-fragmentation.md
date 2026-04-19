---
type: paper
title: "Fragmentation, Price Formation and Cross-Impact in Bitcoin Markets"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - market-microstructure
  - crypto
  - cross-venue
  - bitcoin
status: summarized
related:
  - "[[Temporary vs Permanent Impact]]"
  - "[[Cross-Section]]"
  - "[[backtest-portfolio]]"
  - "[[donier-bonart-2015-bitcoin-metaorder]]"
sources:
  - "https://arxiv.org/abs/2108.09750"
year: 2022
authors:
  - Jakob Albers
  - Mihai Cucuringu
  - Sam Howison
  - Alexander Y. Shestopaloff
venue: "Applied Mathematical Finance, vol. 28, no. 1, pp. 1-48"
key_claim: "Bitcoin price formation is distributed across multiple venues; lead-lag and cross-impact effects between exchanges are quantifiable and material for execution and prediction."
methodology: "High-frequency order-book and trade data from several major Bitcoin exchanges; vector regressions and cross-impact modelling, plus predictive features built from cross-venue order-flow imbalance."
contradicts: []
supports:
  - "[[Cross-Section]]"
url: "https://www.tandfonline.com/doi/full/10.1080/1350486X.2021.1983432"
---

# Fragmentation, Price Formation and Cross-Impact in Bitcoin Markets (Albers, Cucuringu, Howison, Shestopaloff, 2022)

## TL;DR
Empirical microstructure study of fragmented Bitcoin trading across multiple exchanges. The authors collect synchronized high-frequency order-book and trade data from several major venues and quantify how price discovery is shared, how flow on one exchange impacts prices on others (cross-impact), and how predictive signals built from order-flow imbalance across venues forecast short-horizon returns. They show that fragmentation is a first-order feature of crypto markets and that microstructure features generalize from equity-style frameworks while requiring venue-aware modelling.

## Key claims
- Bitcoin price formation is genuinely distributed: no single venue is the unique price leader at all times; leadership rotates with liquidity.
- Cross-impact between exchanges is statistically significant and economically material; ignoring it underestimates true impact and mis-attributes information flow.
- Order-flow imbalance computed across exchanges has predictive power for short-horizon returns, comparable to or stronger than within-venue imbalance.
- Standard equity microstructure tools (price-impact regressions, Hasbrouck-style information shares) extend to crypto but must be reformulated to handle venue heterogeneity, asynchronous trading, and 24/7 operation.

## Methodology
- Multi-venue tick-level data from major Bitcoin exchanges.
- Vector autoregressions and cross-impact regressions of mid-price returns on signed flow per venue.
- Construction and out-of-sample evaluation of cross-venue order-flow predictors.

## Strengths
- One of the most thorough microstructure studies of crypto fragmentation.
- Quantifies cross-impact directly rather than treating venues as independent.
- Provides a template for building cross-venue features in crypto ML pipelines.

## Weaknesses / Critiques
- Bitcoin-only; tiny-cap altcoins (the typical pump targets) likely have very different fragmentation profiles.
- Sample period limited; crypto market structure evolves rapidly with venue entry/exit.
- Assumes synchronized timestamps across venues, which is hard in practice.

## Relation to our work
- Cited twice in `paper/access.tex`: first in the related-work overview ("recent years have seen many articles on predictions using market microstructure, for example [Albers et al.]") and again in our contributions ("we introduce novel microstructure-based features leveraging insights from market prediction"). It anchors the methodological case for treating microstructure features as a serious input to crypto prediction models.
- Motivates the design of our [[Cross-Section]] microstructure features in `features/` (asset returns, flow imbalance, slippage at multiple offsets) and validates the broader strategy of porting equity microstructure tooling to crypto, even though our setting is single-venue (Binance) rather than cross-venue.

## Cited concepts
- [[Cross-Section]]
- [[Temporary vs Permanent Impact]]
