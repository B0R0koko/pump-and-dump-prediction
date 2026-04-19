---
type: paper
title: "To the Moon: Defining and Detecting Cryptocurrency Pump-and-Dumps"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - pump-detection
  - definitions
  - foundational
status: summarized
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Pump Announcement Window]]"
  - "[[Binance]]"
  - "[[xu-2019-anatomy]]"
  - "[[lamorgia-2023-doge]]"
sources:
  - "https://crimesciencejournal.biomedcentral.com/articles/10.1186/s40163-018-0093-5"
year: 2018
authors:
  - "Josh Kamps"
  - "Bennett Kleinberg"
venue: "Crime Science, vol. 7, art. 18"
key_claim: "Crypto pump-and-dumps can be defined operationally and flagged in OHLCV data using simple anomaly criteria over rolling windows; small-cap altcoins on lightly regulated exchanges are the dominant targets."
methodology: "Operational definition of P&D events plus a parametric anomaly-detection rule over hourly OHLCV across five exchanges; parameter sensitivity analysis."
contradicts: []
supports:
  - "[[Pump-and-Dump Scheme]]"
---

# To the Moon: Defining and Detecting Cryptocurrency Pump-and-Dumps (Kamps & Bennett, 2018)

## TL;DR
Foundational paper that gives crypto pump-and-dump research its working definition and the first reproducible detection rule. Kamps and Bennett crawl 1.5 years of OHLCV data across Binance, Bittrex, Kraken, Kucoin, and Lbank, then flag candle-level anomalies whose magnitude jointly exceeds price and volume thresholds relative to a moving baseline. The paper's lasting contribution is the vocabulary later studies adopt (target, signal, dump phase, criteria for "suspicious" moves), not the classifier itself.

## Key claims
- P&D activity in crypto is concentrated on small-market-cap altcoins listed on lightweight exchanges with modest oversight.
- A simple rule combining a price-percentile breakout and a volume z-score over a rolling lookback flags hundreds of plausible pump candidates that visually match the textbook P&D shape.
- Sensitivity analysis: detection volume scales monotonically with thresholds; tuning is required per exchange because of differing baseline volatility and depth.
- Manual review of flagged events confirms most are organised pumps rather than fundamental news shocks.

## Methodology
- Collect hourly OHLCV from five exchanges over ~1.5 years.
- Define a P&D event by jointly thresholding (i) price increase relative to rolling mean and (ii) volume increase relative to rolling mean over a chosen window.
- Sweep window length and threshold values; report counts and example traces.
- Validate by visual inspection and discussion of false positives (listings, news).

## Strengths
- Establishes the operational vocabulary the field still uses.
- Uses public, reproducible OHLCV; no proprietary data.
- Multi-exchange coverage shows the phenomenon is not Binance-specific.

## Weaknesses / Critiques
- Rule-based, not learned: weak against adversarial obfuscation and regime change.
- No ground-truth labels from Telegram or Discord; "pumps" are defined by the rule itself, so precision/recall are not measured against organiser logs.
- Hourly resolution misses sub-minute pump dynamics that later papers exploit.

## Relation to our work
- Cited in `paper/access.tex` (line 91) as the pioneering study that frames the problem and motivates ML follow-ups including ours.
- Their five-exchange survey establishes the small-cap targeting that we exploit by ranking within an event cross-section ([[Cross-Section]]).
- Our hourly feature offsets in [[features]] partly inherit from the time scale Kamps & Bennett showed is informative.

## Cited concepts
- [[Pump-and-Dump Scheme]]
- [[Pump Announcement Window]]
- [[Binance]]
