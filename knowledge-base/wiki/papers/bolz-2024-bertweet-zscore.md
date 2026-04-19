---
type: paper
title: "Machine Learning-Based Detection of Pump-and-Dump Schemes in Real-Time"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - manipulation
  - telegram
  - detection
  - real-time
  - nlp
  - bertweet
  - secondary-exchanges
status: summarized
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Telegram Pump Groups]]"
  - "[[xu-2019-anatomy]]"
  - "[[lamorgia-2023-doge]]"
  - "[[hu-2023-sequence-target-prediction]]"
  - "[[telegram-pump-anatomy]]"
  - "[[state-of-detection-2018-2025]]"
sources:
  - "https://arxiv.org/abs/2412.18848"
  - "https://arxiv.org/html/2412.18848v2"
year: 2024
authors:
  - "Manuel Bolz"
  - "Kevin Brundler"
  - "Liam Kane"
  - "Panagiotis Patsias"
  - "Liam Tessendorf"
  - "Krzysztof Gogol"
  - "Taehoon Kim"
  - "Claudio Tessone"
venue: "arXiv preprint"
key_claim: "A two-stage pipeline (BERTweet for Telegram announcement parsing + Z-score over order book and trade features) can flag pump targets in real time on secondary exchanges, reaching ~56% TOP5 / ~74% TOP10 accuracy on Poloniex within a 20-second window."
methodology: "Fine-tuned BERTweet six-class classifier on 21,092 Telegram messages, plus Z-score statistical model over a three-day window of trade and order-book features at 20 / 40 / 60 second offsets, evaluated on 50-coin random pools."
contradicts: []
supports:
  - "[[xu-2019-anatomy]]"
  - "[[lamorgia-2023-doge]]"
url: "https://arxiv.org/abs/2412.18848"
---

# Machine Learning-Based Detection of Pump-and-Dump Schemes in Real-Time

## TL;DR
Bolz, Brundler, Kane, Patsias, Tessendorf, Gogol, Kim, and Tessone (UZH) build a two-stage real-time pump-and-dump detection pipeline targeting secondary cryptocurrency exchanges. Stage one fine-tunes BERTweet on 21,092 hand-labelled Telegram messages from 43 channels (six message classes), reaching weighted F1 of 0.982. Stage two computes Z-scores over a three-day rolling window of trade and order-book features (order pressure, imbalance, VWAP, trade count, market impact, high-low spread, average order size), and ranks all candidate coins on the announced exchange at the moment of pump start. On 43 Poloniex events the combined Trade+OrderBook ranker reaches TOP5 = 55.81% and TOP10 = 74.42% inside a 20-second window; accuracy collapses to 19% / 29% by 60 seconds. The full historical labelled corpus contains 2,079 distinct events from 2017-12-02 to 2024-10-21 across LATOKEN, XT, Poloniex, KuCoin, MEXC, Binance, LBank, DigiFinex, and Pancakeswap. Confidence: medium-high (single arXiv preprint, sample of 43 evaluated events on one venue).

## Key claims
- BERTweet fine-tuned on six pump-message classes (Pump Announcement, Countdown, Target Coin Release, Pump Results, Delay/Cancellation, Noise) reaches weighted F1 = 0.982. Per-class F1 ranges 0.970 (Pump Results) to 1.000 (Delay/Cancellation). Confidence: high (clean classification problem, large labelled set, balanced reporting).
- A simple per-feature Z-score over a three-day baseline, with no learned model, ranks pump targets at TOP5 = 55.81% on 43 Poloniex events when both trade and order-book features are combined. Confidence: medium (single venue, small evaluation slice; baseline-vs-deep-learning comparison absent).
- Combining trade with order-book Z-scores beats either alone: 55.81% TOP5 (combined) vs 46.51% (trade-only) vs 44.19% (order-book-only) at 20s. Confidence: medium-high.
- Detection signal degrades sharply with offset: TOP5 drops 55.81% -> 41.46% -> 19.05% as window moves from 20s to 40s to 60s post-T0, indicating signals are extremely concentrated around announcement. Confidence: medium-high (consistent with [[lamorgia-2023-doge]]).
- The historical event corpus (2,079 events, 2017-2024) is heavily skewed toward secondary venues: LATOKEN 14.6%, XT 12.9%, Poloniex 11.8%, KuCoin 8.7%, vs Binance only 2.2%. Confidence: high.

## Methodology

### Stage 1: BERTweet announcement parser
- Scraped 91,295 Telegram messages from 43 active pump channels.
- Hand-labelled 21,092 messages into six classes (Pump Announcement, Countdown, Target Coin Release, Pump Results, Delay/Cancellation, Noise).
- Fine-tuned BERTweet (a RoBERTa variant pretrained on English tweets) with an 80/20 train/test split, deduplicated against the test set.
- Output: real-time tagging of incoming Telegram messages so the pipeline knows which announced exchange and approximate T0 to monitor.

### Stage 2: Z-score price-move ranker
- For each candidate coin on the announced exchange, build a three-day rolling window of seven microstructure features: order book pressure, average order size, order imbalance ratio, market order impact, VWAP, high-low spread, trade count.
- Standardize short-window observations against the three-day mean and standard deviation: `z_i = (x_i - mu) / sigma`.
- Aggregate per-coin Z-scores at offsets 20s / 40s / 60s after announced T0; rank within a pool of 50 random coins from the same exchange.
- Evaluation: TOP5 and TOP10 accuracy across 43 held-out Poloniex events.

### Data scope
- 4,643 cryptocurrencies tracked, 365,982 market-cap rows.
- Order book data: 5-44 million rows per exchange. Trade data: 2-5 million rows per exchange.
- Time coverage 2017-12-02 to 2024-10-21; recent OHLCV July-October 2024.

## Strengths
- First sizable, dated Telegram pump corpus (2,079 events) that explicitly extends past 2021 and into the post-FTX period, partially closing a gap flagged in [[telegram-pump-anatomy]].
- Explicit focus on secondary exchanges (LATOKEN, XT, Poloniex, KuCoin, MEXC, LBank, DigiFinex, Pancakeswap), which the Binance-centric prior literature ([[xu-2019-anatomy]], [[lamorgia-2023-doge]]) had largely ignored.
- Clean separation of NLP and price-detection stages makes each independently swappable.
- BERTweet classification numbers are credible and well above earlier rule-based parsers.

## Weaknesses / Critiques
- The price-detection ranker is evaluated on only 43 events from a single venue (Poloniex). Generalization to KuCoin / MEXC / LATOKEN is asserted by dataset construction but not measured.
- Z-score ranking is a baseline, not a learned model. No comparison to logistic regression, random forest, gradient boosting, or to [[hu-2023-sequence-target-prediction]] sequence approach.
- 50-coin random pools are smaller than realistic exchange universes (Poloniex listed >300 spot pairs in 2024); TOP5 numbers are not directly comparable to [[xu-2019-anatomy]] or [[hu-2023-sequence-target-prediction]] which use larger pools.
- No precision / recall / F1 reported for the price-detection stage, only top-k accuracy. No calibration analysis.
- arXiv preprint as of December 2024; not yet through formal peer review at time of writing (April 2026).

## Relation to our work
- Confirms that the [[Cross-Section]] framing (rank candidates within a pool at announced T0) is the right unit even on secondary venues. Our pipeline can ingest the labelled 2,079-event corpus once released.
- Their Z-score baseline is a useful sanity check for our CatBoost ranker: any learned model should beat 55.81% TOP5 on the same Poloniex slice.
- Order-book pressure, imbalance ratio, VWAP, and high-low spread are direct counterparts to our features in [[features]]; their three-day baseline window aligns with our multi-offset framing.
- BERTweet stage suggests a clean way to harvest fresh labels: combine [[lamorgia-2023-doge]]'s SystemsLab corpus with a Bolz-style NLP harvester for post-2021 events.

### Apparent gap with [[lamorgia-2023-doge]]
La Morgia et al. report ~94.5% F1 for per-second pump *detection* on Binance `SYM/BTC` pairs across ~900 events; Bolz et al. report ~55.81% TOP5 ranking accuracy on 43 Poloniex events. The gap is wide but the tasks and venues are not comparable:

- **Task**: La Morgia *detects* whether the current second is mid-pump (binary, after T0); Bolz *ranks* which of 50 coins is the announced target (top-k, around T0). Detection-of-active-pump is a strictly easier task than pre-T0 target ranking, because the price has already moved.
- **Venue**: Binance has tighter spreads, deeper books, higher trade frequency, and stricter market surveillance, so pump footprints are sharper and shorter. Secondary venues (KuCoin, MEXC, LATOKEN, XT, Poloniex) have thinner books, more listings, more sparse trade tapes, so signal-to-noise per second is worse.
- **Channel mix**: La Morgia's 20+ groups skew toward Binance-organising channels; Bolz's 43 channels skew toward LATOKEN / XT / Poloniex / KuCoin organisers (only 2.2% of events are on Binance).
- **Pool size**: Bolz fixes the candidate pool at 50, La Morgia evaluates per-second binary classification on the listed `SYM/BTC` universe (~hundreds of pairs).

The right framing is venue and channel selection effects, not a true contradiction. [[hot]] already adopts this framing. The two papers together suggest that detection performance is roughly monotone in venue depth and channel curation, which is itself a testable prediction.

## Cited concepts
- [[Pump-and-Dump Scheme]]
- [[Telegram Pump Groups]]
- [[Cross-Section]]
- [[Pump Announcement Window]]
- [[Pre-Pump Accumulation]]
