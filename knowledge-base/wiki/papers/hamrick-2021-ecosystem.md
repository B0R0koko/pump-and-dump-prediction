---
type: paper
title: "An examination of the cryptocurrency pump-and-dump ecosystem"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - manipulation
  - telegram
  - discord
  - ecosystem
  - economics
status: summarized
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Telegram Pump Groups]]"
  - "[[xu-2019-anatomy]]"
  - "[[lamorgia-2023-doge]]"
  - "[[telegram-pump-anatomy]]"
  - "[[state-of-detection-2018-2025]]"
sources:
  - "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3303365"
  - "https://www.sciencedirect.com/science/article/abs/pii/S0306457321000169"
  - "https://dl.acm.org/doi/10.1016/j.ipm.2021.102506"
  - "https://discovery.ucl.ac.uk/id/eprint/10133232/"
  - "https://tylermoore.utulsa.edu/ipm21.pdf"
  - "https://tylermoore.utulsa.edu/weis19pump.pdf"
  - "https://cepr.org/voxeu/columns/economics-cryptocurrency-pump-and-dump-schemes"
year: 2021
authors:
  - "JT Hamrick"
  - "Farhang Rouhi"
  - "Arghya Mukherjee"
  - "Amir Feder"
  - "Neil Gandal"
  - "Tyler Moore"
  - "Marie Vasek"
venue: "Information Processing & Management"
key_claim: "Pumps are modestly successful in driving short-term price rises, but the effect has diminished over time; transparent pumps outperform obscured ones (median 7.7% vs 4.1%)."
methodology: "Joined essentially every relevant Telegram and Discord pump channel for the first half of 2018, parsed 4,818 announced signals, and joined them with exchange OHLCV to measure return distributions and compare transparent vs obscured pump styles."
contradicts:
  - "[[xu-2019-anatomy]]"
supports:
  - "[[lamorgia-2023-doge]]"
url: "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3303365"
---

# An examination of the cryptocurrency pump-and-dump ecosystem

## TL;DR
Hamrick, Rouhi, Mukherjee, Feder, Gandal, Moore, and Vasek (IPM 2021) build the largest known catalogue of announced cryptocurrency pump signals at the time: 3,767 distinct signals on Telegram plus 1,051 on Discord (~4,818 total) targeting more than 300 coins, all collected in roughly January through July 2018. This corpus, sometimes referred to as the **Pump-and-Dumpster**, is released in partial form via SSRN supplementary material. The headline economic finding: "pumps are modestly successful in driving short-term price rises, but [...] this effect has diminished over time"; transparent pumps materially outperform obscured ones (median return 7.7% vs 4.1%) and the transparent advantage does not decay. Confidence: high.

## Key claims
- During the first half of 2018, organisers ran ~4,818 announced pumps across hundreds of channels, targeting ~300 coins. Confidence: high.
- The phenomenon is "widespread and often quite profitable" in aggregate, but the per-event lift is modest and shrinks over the sample period. Confidence: high.
- Transparent pumps (organisers openly announce intent and timing) achieve a median return of 7.7%, vs 4.1% for obscured pumps; transparent returns do not erode while obscured returns decline. Confidence: high.
- Pump activity is concentrated in a small number of dominant channels, which the authors argue makes targeted regulatory enforcement viable. Confidence: medium.
- Outsider investors are systematically disadvantaged: realised gains depend critically on how early the signal is received, with late buyers losing on average. Confidence: high.

## Methodology
- **Data collection**: joined the (largely public) Telegram and Discord channels operating in the pump-and-dump ecosystem during early 2018; parsed every pump-signal message, capturing exchange, target coin, scheduled time, and channel metadata.
- **Market data**: matched announcements to OHLCV/trade data from the relevant exchanges; computed short-horizon returns around the announced timestamp.
- **Analysis**: descriptive statistics of channel concentration, coin selection, and lifecycle; regression-style comparison of transparent vs obscured organising styles; longitudinal trend in per-event returns.
- **Release**: corpus partially available via SSRN supplementary material and through correspondence with the authors. A WEIS 2019 precursor (`weis19pump.pdf`) covers the same dataset.

## Strengths
- Largest announced-pump corpus of its era and the only one to cover both Telegram and Discord side-by-side at scale.
- Mixes economic framing (return distributions, market-quality consequences) with measurement, in contrast to detection-focused contemporaries.
- Cleanly identifies a regulatory lever: enforcement against the few dominant transparent channels would dent most of the activity.
- Findings are robust enough that subsequent surveys (Tornes 2023, La Morgia 2023) cite the dataset and the "modestly successful, declining over time" headline as the canonical economic baseline.

## Weaknesses / Critiques
- The corpus is restricted to the first half of 2018: it predates the maturation of Binance-centric `SYM/BTC` pump targeting captured by [[xu-2019-anatomy]] and [[lamorgia-2023-doge]], and well predates the post-FTX market structure.
- "Announced" does not equal "executed": some signals never produce coordinated buying; the paper does not cleanly separate cancelled/failed announcements from successful pumps in the released corpus, which complicates joining with market data.
- Public availability is partial; full per-event records require contacting the authors. This makes exact replication harder than for the Xu or La Morgia GitHub releases.
- Returns are reported as event-level price moves, not net of trading costs or realistic slippage at retail order sizes; the picture would be even bleaker for late buyers under a proper price-impact model.
- Many announcements target tiny exchanges (Yobit, Cryptopia, etc.) that have since closed, limiting forward applicability of the venue mix.

## Relation to our work
- **Settles the central contradiction with [[xu-2019-anatomy]]**. Xu and Livshits report a ~60% return over ~2.5 months for a naive long strategy that buys the predicted pump target; Hamrick et al. instead find pumps are "modestly successful in driving short-term price rises, but [...] this effect has diminished over time", with median per-event returns of only 4.1%–7.7% depending on transparency. The two are not strictly contradictory: Xu and Livshits measure a model-conditioned, intra-pump-window return on a subset of high-confidence picks, while Hamrick et al. measure unconditional event-level returns over a much wider, less curated sample. Together they imply the alpha is real but small and concentrated, and survives only on transparent organisers' picks. Our [[backtest-portfolio]] should treat the Hamrick distribution as the realistic prior, not the Xu headline.
- **Scale of the labelling task**. The 4,818-signal Hamrick corpus is roughly an order of magnitude larger than the [[xu-2019-anatomy]] release (412) and on par with the largest later corpora ([[lamorgia-2023-doge]] ~900 confirmed Binance events, Bolz 2024 ~2,079, Perseus 2025 ~4,101). Even partial access is valuable for cross-validation of pump labels.
- **Channel concentration**. The "few dominant channels run most of the activity" finding supports the channel-conditioned modelling thread (Hu et al. 2023 SIGMOD: per-channel sequence encoder), and it argues for adding a `channel_id` categorical to our feature set.
- **Label hygiene**. The "announced does not always execute" caveat is a direct input to our `PumpEvent` ingestion: events from text-only sources need a market-data sanity check (volume spike at T0+epsilon) before being treated as positive labels. Already noted in [[Telegram Pump Groups]].

## Cited concepts
- [[Pump-and-Dump Scheme]]
- [[Telegram Pump Groups]]
- [[Cross-Section]]
- [[Pump Announcement Window]]
