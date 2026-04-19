---
type: paper
title: "Detecting Cryptocurrency Pump-and-Dump Frauds Using Market and Social Signals"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - pump-detection
  - social-signals
  - deep-learning
  - twitter
status: summarized
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Pump Announcement Window]]"
  - "[[Telegram Pump Groups]]"
  - "[[bolz-2024-bertweet-zscore]]"
  - "[[lamorgia-2023-doge]]"
sources:
  - "https://www.sciencedirect.com/science/article/pii/S0957417421007247"
year: 2021
authors:
  - "Huy Nghiem"
  - "Goran Muric"
  - "Fred Morstatter"
  - "Emilio Ferrara"
venue: "Expert Systems with Applications, vol. 182, art. 115284"
key_claim: "Combining market-data features with Twitter-derived social-signal features in LSTM and CNN models improves prediction of the maximum price move during a known pump event."
methodology: "Per-event LSTM/CNN regression of peak return, using market microstructure features plus Twitter activity and sentiment time series."
contradicts: []
supports: []
---

# Detecting Cryptocurrency Pump-and-Dump Frauds Using Market and Social Signals (Nghiem et al., 2021)

## TL;DR
Nghiem and co-authors frame a different question from most P&D work: given that a pump is announced, can we predict how far the price will move? They build per-event LSTM and CNN models that fuse market microstructure features with Twitter activity and sentiment, and find that adding social signals improves peak-return prediction. They also flag a non-trivial caveat: Twitter snapshots are taken after the fact, so historical sentiment series are biased by deletions, suspended accounts, and unrecoverable real-time content.

## Key claims
- Peak-return regression during a pump benefits from combining market and social signals over only-market or only-social baselines.
- LSTM and CNN architectures both work; the choice matters less than the inclusion of social signal.
- Historical Twitter data introduces a sentiment-snapshot bias: the dataset retrieved months later differs systematically from what was visible in real time.
- Pump magnitudes are predictable to first order from pre-announcement bursts in social and market activity.

## Methodology
- Curate a list of confirmed pumps with exchange and timestamp.
- Pull market data around each event and Twitter posts mentioning the targeted ticker.
- Build per-event time-series feature stacks; train LSTM and CNN regressors against the realised peak return.
- Compare against market-only and social-only ablations.

## Strengths
- One of the first multi-modal (market + social) P&D papers.
- Clearly distinguishes the prediction-of-magnitude task from the detection task.
- Honest discussion of the historical-sentiment-bias problem that later social-signal papers must reckon with.

## Weaknesses / Critiques
- The "predict the peak return" framing assumes the pump and its target are already known; less directly actionable than target prediction.
- Twitter coverage skews toward English-speaking, larger-cap tokens; non-English Telegram pumps are under-represented.
- Sample size and class definitions follow the older small-corpus tradition; newer datasets like [[lamorgia-2023-doge]] now exist.

## Relation to our work
- Cited in `paper/access.tex` (line 94) as a representative non-detection branch of P&D research: predicting move magnitude rather than target identity. We use it to position our work as target-prediction.
- The historical-sentiment-bias warning motivates our decision to stick with market microstructure features and not depend on Twitter-derived features whose past values cannot be reconstructed faithfully.
- See [[bolz-2024-bertweet-zscore]] for a more recent treatment of social-signal P&D detection.

## Cited concepts
- [[Pump-and-Dump Scheme]]
- [[Pump Announcement Window]]
