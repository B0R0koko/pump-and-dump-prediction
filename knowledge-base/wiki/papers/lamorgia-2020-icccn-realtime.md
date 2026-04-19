---
type: paper
title: "Pump and Dumps in the Bitcoin Era: Real-Time Detection of Cryptocurrency Market Manipulations"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - pump-detection
  - binance
  - real-time
  - random-forest
status: summarized
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Pump Announcement Window]]"
  - "[[Binance]]"
  - "[[Telegram Pump Groups]]"
  - "[[lamorgia-2023-doge]]"
  - "[[xu-2019-anatomy]]"
sources:
  - "https://ieeexplore.ieee.org/document/9209660"
year: 2020
authors:
  - "Massimo La Morgia"
  - "Alessandro Mei"
  - "Francesco Sassi"
  - "Julinda Stefa"
venue: "Proc. International Conference on Computer Communications and Networks (ICCCN), 2020"
key_claim: "A random-forest classifier on per-second Binance trade aggregates can flag an in-progress Telegram-coordinated pump within seconds of the announcement, well before the dump phase."
methodology: "Telegram-channel monitoring to label ~300 confirmed pumps; per-second feature engineering on Binance trades; supervised random forest with streaming evaluation."
contradicts: []
supports:
  - "[[xu-2019-anatomy]]"
  - "[[lamorgia-2023-doge]]"
---

# Pump and Dumps in the Bitcoin Era (La Morgia et al., 2020)

## TL;DR
Conference precursor to the journal-length [[lamorgia-2023-doge]]. La Morgia and co-authors monitor Telegram pump channels for several months, label roughly 300 confirmed events, and train a random forest over per-second Binance trade features that detects an active pump in near real time. The paper establishes the per-second streaming-detection paradigm that later work scaled up.

## Key claims
- Telegram organisers post a pre-announcement (exchange and timestamp) and reveal the target ticker at T0; the pump completes within minutes.
- Per-second microstructure features (trade count, volume, return, taker-side imbalance proxies) carry strong signal once the announcement fires.
- A random forest trained on these features detects pumps with high precision and recall within seconds, far better than rule-based hourly detectors of the Kamps & Bennett type.
- Realistic streaming evaluation matters: classifiers must be trained and scored on chronologically ordered chunks, not shuffled samples.

## Methodology
- Crawled Telegram pump channels; manually validated each event's exchange and timestamp.
- Pulled Binance trade-level data around each event; aggregated to per-second candles.
- Engineered chunk-level statistics on volume bursts, returns, and trade-count spikes.
- Trained a random forest; reported per-class precision, recall, F1 under streaming evaluation.

## Strengths
- First per-second real-time detector on Binance with Telegram-validated ground truth.
- Clean problem framing (binary "is this second part of a pump?") that downstream work inherits.
- Public release of the labelled pump list later expanded by [[lamorgia-2023-doge]] to ~1100 events; the 2021 update is reused by us and many others.

## Weaknesses / Critiques
- Detection only fires once the pump is underway; no pre-announcement target prediction.
- ~300 events is small relative to the Telegram ecosystem; group selection bias.
- Streaming pipeline assumes the analyst already knows which exchange to watch (the announcement provides this).

## Relation to our work
- Cited in `paper/access.tex` (lines 91, 126, 140, 150) as the canonical ML baseline and as the source of the labelled pump list we incorporate to enlarge our 175-event sample.
- Our problem is complementary: we predict the target asset *before* the announcement, while La Morgia 2020 detects the pump *during* execution.
- Their per-second feature catalogue informs our short-window offsets in [[features]].
- Reference list in our Table I (line 126) lists this work as the random-forest baseline.

## Cited concepts
- [[Pump-and-Dump Scheme]]
- [[Pump Announcement Window]]
- [[Telegram Pump Groups]]
- [[Binance]]
