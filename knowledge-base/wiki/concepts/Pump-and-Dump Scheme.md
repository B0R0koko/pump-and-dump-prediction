---
type: concept
title: "Pump-and-Dump Scheme"
created: 2026-04-19
updated: 2026-04-19
tags:
  - concept
  - manipulation
status: developing
related:
  - "[[Cross-Section]]"
  - "[[Binance]]"
sources: []
complexity: basic
domain: "market manipulation"
aliases:
  - "P&D"
  - "pump"
---

# Pump-and-Dump Scheme

## Definition
A coordinated market-manipulation pattern in which a group inflates the price of a low-liquidity asset (the **pump**) and then sells into the resulting demand from latecomers (the **dump**), realizing profit at their expense.

In crypto, pumps are typically organized in Telegram or Discord channels and target small-cap altcoins listed on a single major exchange (most often [[Binance]]).

## Anatomy
1. **Accumulation** — organizers quietly build a position over hours or days.
2. **Signal** — the target ticker is announced to a large audience at a precise time.
3. **Pump** — coordinated buying drives price up sharply (often 50–500% in minutes).
4. **Dump** — organizers and early followers sell into late buyers; price collapses.

## Why it matters here
The entire `pumps_and_dumps` project predicts the **target** of an upcoming pump from pre-event microstructure signals. The framing is a [[Cross-Section]] ranking problem: at time T (the announcement), rank all candidate assets by likelihood of being the target.

## How it appears in this project
- `PumpEvent` (in [[core]]) is the foundational record.
- [[features]] computes signals at offsets relative to the pump timestamp.
- [[backtest-pipelines]] and [[backtest-portfolio]] convert rankings into a tradable strategy.

## Related
- [[Cross-Section]] — the per-event ranking unit.
- [[Telegram Pump Groups]] — the social coordination substrate and dataset registry.
- [[telegram-pump-anatomy]] — full synthesis: lifecycle, signals, datasets.
- [[state-of-detection-2018-2025]] — survey of the detection literature this concept anchors.

## Detection literature

The 11 P&D detection papers currently in the vault, grouped by methodological family. One-line annotation per paper.

### Anomaly thresholding (rule-based)
- [[kamps-bennett-2018-to-the-moon]] — first systematic anomaly-detection baseline on volume and price thresholds; cited in every successor paper.

### Pre-event target prediction (ranking / classification)
- [[xu-2019-anatomy]] — foundational 412-event Telegram dataset, Random Forest target ranker, ~60% naive long return.
- [[hu-2023-sequence-target-prediction]] — SIGMOD 23 sequence model with channel-conditioning and positional attention; AUC 0.943, Hit@5 53%.

### Supervised real-time detection (per-second, tree models)
- [[lamorgia-2020-icccn-realtime]] — ICCCN baseline; per-second Binance candles, Random Forest, ~25 second latency.
- [[lamorgia-2023-doge]] — successor "Doge of Wall Street" paper; rush-order features lift F1 to 94.5%, 98.2% precision.
- [[fantazzini-xiao-2023-imbalanced]] — addresses class imbalance via SMOTE/cost-sensitive variants on the La Morgia dataset.

### Deep learning extensions
- [[chadalapaka-2022-deep-learning]] — CNN/RNN re-implementation of the La Morgia setup; confirms tree baseline is hard to beat at this dataset size.

### NLP + microstructure (multimodal)
- [[nghiem-2021-market-social-signals]] — combines social-media signals with market features for early detection.
- [[bolz-2024-bertweet-zscore]] — BERTweet text head plus 20-second order-book z-scores; 2,079 Poloniex pumps, top-5 accuracy 56%.

### Ecosystem characterisation
- [[hamrick-2021-ecosystem]] — ~5,000 events scraped from Telegram and Discord; documents that pumps on average lose late buyers money.

### Microstructure quantification
- [[karbalaii-2025-microstructure]] — most recent corpus (485 Poloniex events, 2024–2025); minute-OHLCV evidence of accumulation in 69.3% of events.
