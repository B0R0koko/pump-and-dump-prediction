---
type: paper
title: "The Doge of Wall Street: Analysis and Detection of Pump and Dump Cryptocurrency Manipulations"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - manipulation
  - telegram
  - detection
  - real-time
status: summarized
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Telegram Pump Groups]]"
  - "[[xu-2019-anatomy]]"
  - "[[hu-2023-sequence-target-prediction]]"
  - "[[telegram-pump-anatomy]]"
sources:
  - "https://arxiv.org/abs/2105.00733"
  - "https://dl.acm.org/doi/10.1145/3561300"
  - "https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset"
  - "https://github.com/SystemsLab-Sapienza/gme-pump-xrp-telegram"
year: 2023
authors:
  - "Massimo La Morgia"
  - "Alessandro Mei"
  - "Francesco Sassi"
  - "Julinda Stefa"
venue: "ACM Transactions on Internet Technology, Vol. 23, No. 1"
key_claim: "Real-time per-second classification of Binance candles can detect Telegram-organised pumps within ~25 seconds at 94.5% F1, far improving on prior 30-minute windows."
methodology: "3+ years of Telegram/Discord monitoring, second-resolution Binance trades, supervised ML over chunked candles"
contradicts: []
supports:
  - "[[xu-2019-anatomy]]"
url: "https://arxiv.org/abs/2105.00733"
---

# The Doge of Wall Street

## TL;DR
La Morgia et al. monitor 20+ Telegram and Discord pump groups for more than three years, label ~900 confirmed pumps, and train a per-second classifier on Binance trade chunks that detects an active pump within ~25 seconds at 94.5% F1. They also analyse the related "crowd pump" phenomenon (e.g., Dogecoin, XRP, GME-style coordinated retail). Datasets and code are public on GitHub. Confidence: high.

## Key claims
- Telegram pump groups follow a stable lifecycle: pre-announcement (exchange + timestamp + FFA vs Ranked tiering), reveal of ticker at T0, sub-minute pump, multi-minute dump.
- 70% of pre-event volume transacts within one hour of the announcement, with sharp visible footprints in seconds-resolution data.
- Real-time supervised models on 5- or 25-second windows beat earlier 30-minute approaches by an order of magnitude in latency and ~30 F1 points.
- "Crowd pumps" (Dogecoin, XRP, GME) differ from organised pumps: distributed sentiment-driven inflows rather than a single command channel.

## Methodology
- Crawled Telegram channels and Discord servers; manually validated pump events; recorded pre-announced exchange and time.
- Pulled Binance trade-level data and aggregated to per-second candles.
- Engineered chunk-level statistical features (volume bursts, return acceleration, trade count, taker imbalance proxies).
- Compared multiple classifiers (random forest et al.) under realistic streaming protocols.

## Strengths
- Largest publicly described Telegram pump corpus at time of publication.
- Real-time evaluation protocol matches what a trader/exchange compliance team would actually run.
- Datasets released: the [pump-and-dump-dataset](https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset) (CSV of pumps + downloader for Binance trades) and the [gme-pump-xrp-telegram](https://github.com/SystemsLab-Sapienza/gme-pump-xrp-telegram) crowd-pump dataset.

## Weaknesses / Critiques
- Detection assumes the pump is already underway: it does not predict targets pre-announcement (unlike [[xu-2019-anatomy]]).
- Heavy class imbalance handled mainly via sampling; failure modes on similar but non-pump volume spikes (listings, news) are under-explored.
- Group selection bias: 20 monitored groups out of a much larger ecosystem.

## Relation to our work
- The released `pump_telegram.csv` is a candidate ground-truth source for our `PumpEvent` records in [[core]].
- Their per-second feature design motivates our multi-offset framing in [[features]] (5 min down to seconds).
- Their detection-during-pump complements our prediction-of-target setup; the two could be cascaded (predict candidates, then confirm with their classifier).

## Cited concepts
- [[Pump-and-Dump Scheme]]
- [[Telegram Pump Groups]]
- [[Pump Announcement Window]]
