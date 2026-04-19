---
type: paper
title: "The Anatomy of a Cryptocurrency Pump-and-Dump Scheme"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - manipulation
  - telegram
  - foundational
status: summarized
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Telegram Pump Groups]]"
  - "[[Cross-Section]]"
  - "[[telegram-pump-anatomy]]"
sources:
  - "https://arxiv.org/abs/1811.10109"
  - "https://www.usenix.org/system/files/sec19-xu-jiahua_0.pdf"
year: 2019
authors:
  - "Jiahua Xu"
  - "Benjamin Livshits"
venue: "USENIX Security Symposium"
key_claim: "Telegram-coordinated pump events on small-cap altcoins are predictable from pre-pump market features, and a classifier can rank coins by pump likelihood profitably."
methodology: "Empirical analysis of 412 announced pumps + supervised learning to predict pump targets"
contradicts:
  - "[[hamrick-2021-ecosystem]]"
supports:
  - "[[lamorgia-2023-doge]]"
  - "[[hu-2023-sequence-target-prediction]]"
url: "https://arxiv.org/abs/1811.10109"
---

# The Anatomy of a Cryptocurrency Pump-and-Dump Scheme

## TL;DR
Xu and Livshits collected 412 pump events announced in Telegram channels between 17 June 2018 and 26 February 2019, characterised the typical lifecycle (announcement, execution within seconds, dump within minutes), and trained a classifier that predicts which coin will be pumped given pre-pump features. A naive trading strategy on the model's predictions returned around 60% over roughly 2.5 months on small retail capital. Confidence: high.

## Key claims
- Pumps are coordinated in Telegram, target tiny-cap pairs (often `SYM/BTC` on Binance), and typically last only minutes.
- The exchange and exact start time are pre-announced; the target ticker is revealed only at the moment of execution.
- Pre-pump market features (volume, returns, volatility, market cap, age, exchange listings) are informative about which coin will be picked.
- A generalised pump-likelihood model can be used both for detection and for trading.

## Methodology
- Manually monitored Telegram pump channels and confirmed events against on-chain price/volume data.
- Built per-(coin, exchange) feature vectors before each pump.
- Trained classifiers (random forest, generalised linear model) to rank coins; evaluated lift and trading PnL.
- Public material: code in [github.com/xujiahuayz/pumpdump](https://github.com/xujiahuayz/pumpdump). Confidence: high.

## Strengths
- First rigorous, dataset-driven academic treatment of the modern Telegram-coordinated pump.
- Provides both descriptive economics (winners/losers, profit distribution) and a working predictive model.

## Weaknesses / Critiques
- Pre-2019 data; market structure (Binance pair universe, group sophistication) has shifted since.
- Trading strategy assumes the analyst knows the announcement instant; live use depends on monitoring channels in real time.
- Imbalance handling and survivor bias in coin universe are only lightly addressed.

> [!contradiction] Profitability bound
> The headline ~60% return is the upper bound on a model-selected long over the 2.5-month sample. [[hamrick-2021-ecosystem]] reports an unconditional median pump return of only 4.1–7.7%, declining over time. The two figures are reconcilable (Xu = curated, model-selected longs; Hamrick = unconditional baseline) but the 60% should not be read as the typical pump payoff. See [[state-of-detection-2018-2025]] for the full reconciliation.

## Relation to our work
This is the canonical predecessor for the [[Cross-Section]] framing we use in [[backtest-pipelines]]: at each pump event, rank all candidate coins by pump probability. Our feature set in [[features]] echoes Xu and Livshits (volume, returns, volatility) but extends to multi-offset windows and microstructure flow imbalance.

## Cited concepts
- [[Pump-and-Dump Scheme]]
- [[Cross-Section]]
- [[Telegram Pump Groups]]
