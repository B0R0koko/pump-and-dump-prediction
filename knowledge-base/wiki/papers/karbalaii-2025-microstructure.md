---
type: paper
title: "Microstructure and Manipulation: Quantifying Pump-and-Dump Dynamics in Cryptocurrency Markets"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - manipulation
  - telegram
  - microstructure
  - poloniex
status: summarized
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Telegram Pump Groups]]"
  - "[[Pre-Pump Accumulation]]"
  - "[[Pump Announcement Window]]"
  - "[[telegram-pump-anatomy]]"
  - "[[xu-2019-anatomy]]"
  - "[[lamorgia-2023-doge]]"
sources:
  - "https://arxiv.org/abs/2504.15790"
  - "https://arxiv.org/html/2504.15790"
year: 2025
authors:
  - "Mahya Karbalaii"
venue: "arXiv preprint"
key_claim: "Minute-OHLCV analysis of 485 Telegram-coordinated Poloniex pumps (Aug 2024 to Feb 2025) shows pre-pump accumulation in 69.3% of events and ~70% of pre-event volume concentrated in the final hour before announcement, with median single-point liquidation profits above 100% and tranche-strategy upper-quartile returns above 2,000%."
methodology: "Algorithmic detection of accumulation phases on minute-level OHLCV across 1,101 Poloniex trading pairs, cross-validated against Telegram announcements; descriptive microstructure statistics."
contradicts: []
supports:
  - "[[xu-2019-anatomy]]"
  - "[[lamorgia-2023-doge]]"
url: "https://arxiv.org/abs/2504.15790"
---

# Microstructure and Manipulation: Quantifying Pump-and-Dump Dynamics in Cryptocurrency Markets

## TL;DR
Karbalaii (2025) extends prior threshold-based pump detection by ingesting minute-level OHLCV bars for 1,101 Poloniex trading pairs over six months (15 August 2024 to 15 February 2025), cross-validating 485 candidate Telegram-announced pumps, and quantifying their microstructure footprints. The paper establishes that 336 of 485 events (69.3%) have non-zero traded volume in the minutes before the announcement, that roughly 70% of pre-event volume is compressed into the final hour before T0, and that insider profits under realistic liquidation strategies range from a median above 100% (single-point exit at 70% of peak) to upper-quartile returns above 2,000% (tranche selling). It is, as of 2026-04-19, the most recent academic corpus on Telegram-organised pumps and the only post-2022 quantitative microstructure study, but it is restricted to Poloniex rather than Binance. Confidence: medium-high.

## Key claims
- 69.3% of confirmed pumps show detectable pre-pump trading activity, partitioning events into "pre-accumulation" and "on-the-spot" archetypes. Confidence: high (direct count from the paper).
- ~70% of pre-event volume transacts within one hour of the pump announcement, indicating that informational leakage and insider positioning are sharply concentrated late. Confidence: medium (claim appears in the abstract but the body provides limited statistical breakdown).
- Insider profitability is highly skewed: median above 100% under a single-point exit at 70% of the peak, and upper-quartile above 2,000% under a tranche schedule (20% at 50% of peak, 30% at 60%, 50% at 80%). Confidence: medium-high.
- A reproducible algorithmic procedure can label accumulation phases purely from minute OHLCV without order-book or trade-tape data, lowering the bar for downstream detection research. Confidence: medium.
- Poloniex remains a non-trivial venue for Telegram-coordinated manipulation in 2024 to 2025, contradicting the assumption that pump activity has fully migrated to a small number of major venues post-FTX. Confidence: medium.

## Methodology
- Source data: Poloniex public REST API, minute OHLCV bars (open, high, low, close, base-asset volume) for 1,101 trading pairs across 15 August 2024 to 15 February 2025.
- Event identification: 1,021 tokens flagged with at least one candidate event; reduced to 485 events with sufficient minute-bar coverage and Telegram cross-validation.
- Accumulation labelling: per-event scan of minute bars before T0; events with any non-zero traded volume in the pre-window are flagged as pre-accumulated.
- Profit modelling: closed-form returns under (a) single-point liquidation at 70% of observed peak and (b) a three-tranche schedule (20% / 30% / 50% at 50% / 60% / 80% of peak).
- Statistics: descriptive only (means, medians, quartiles). No hypothesis tests (Mann-Whitney, t-test, KS) are reported in the paper. Confidence: high.
- Public release: paper states the data are obtained from the free Poloniex API and are publicly accessible; no released dataset CSV or code repository is announced. Confidence: high.

## Strengths
- Most recent post-2022 Telegram-pump corpus with quantitative microstructure analysis.
- Minute-OHLCV-only feature set is realistic: it can be reproduced from any exchange's public REST API without privileged trade-tape access.
- Clear taxonomy ("pre-accumulation" vs "on-the-spot") that is directly actionable for downstream classifiers.
- Six-month window covers the post-FTX, post-spot-ETF macro regime not represented in older corpora.

## Weaknesses / Critiques
- Single venue (Poloniex). Generalisation to Binance, KuCoin, MEXC, or LATOKEN is asserted rather than tested.
- No formal hypothesis testing: the headline 70%-in-one-hour figure is descriptive and the body of the paper does not present the supporting per-event distribution.
- Minute granularity is coarser than La Morgia et al. 2023's per-second resolution and likely understates the ultra-short-horizon dynamics.
- No public release of the labelled event list or detection code; reproducibility depends on re-deriving the candidate set from the Poloniex API.
- Sample size (485) is roughly half of [[lamorgia-2023-doge]] (~900) and an order of magnitude below Hamrick et al. 2021 (~3,800).

## Relation to our work
- Karbalaii 2025 partially closes the lint-flagged "post-2022 Telegram pump corpus" gap: it is the first academic dataset that covers the late-2024 to early-2025 window. **However, the venue is Poloniex, not Binance.** Our pipeline targets Binance `SYM/BTC` and `SYM/USDT` pairs, so Karbalaii's data are not a drop-in `PumpEvent` source for our [[backtest-pipelines]]; the Binance-specific gap remains open and should carry forward as a standalone `wiki/gaps/` entry.
- The two numerical anchors propagated through our prior synthesis ([[telegram-pump-anatomy]]) match the paper exactly:
  - 69.3% pre-pump accumulation: the paper states *"Out of 485 total events, 336 (69.3%) exhibited at least one minute with non-zero traded volume before the pump start."*
  - ~70% pre-event volume in one hour: the abstract states *"70% of pre-event volume transacts within one hour of the pump announcement."* The exact figure carries through but the body offers no supporting per-event histogram, so confidence on this number is medium rather than high.
- The pre-accumulation vs on-the-spot taxonomy maps onto a useful binary feature for our [[Cross-Section]] ranker: events with detectable pre-window volume are likely easier to predict, and stratifying validation by archetype could expose model failure modes.
- Karbalaii's tranche-based profitability model is a candidate sanity check for our [[backtest-portfolio]] (`TOPKPortfolio`) PnL distributions: if our simulated insider-side returns under aggressive exit schedules diverge from the 2,000%+ upper quartile, the divergence bounds either our slippage assumptions or our universe selection.

## Cited concepts
- [[Pump-and-Dump Scheme]]
- [[Telegram Pump Groups]]
- [[Pre-Pump Accumulation]]
- [[Pump Announcement Window]]
- [[Cross-Section]]
