---
type: entity
title: "Telegram Pump Groups"
created: 2026-04-19
updated: 2026-04-19
tags:
  - entity
  - manipulation
  - telegram
  - dataset-source
status: developing
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Binance]]"
  - "[[xu-2019-anatomy]]"
  - "[[lamorgia-2023-doge]]"
  - "[[telegram-pump-anatomy]]"
sources:
  - "https://arxiv.org/abs/1811.10109"
  - "https://arxiv.org/abs/2105.00733"
  - "https://www.sciencedirect.com/science/article/abs/pii/S0306457321000169"
  - "https://arxiv.org/html/2412.18848v2"
  - "https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset"
  - "https://github.com/xujiahuayz/pumpdump"
entity_type: organization
role: "Coordinated communities (and the datasets harvested from them) that organise the pump-and-dump events the project studies."
first_mentioned: "[[xu-2019-anatomy]]"
---

# Telegram Pump Groups

## What it is
A loose ecosystem of public and private Telegram channels (and a smaller set of Discord servers) where organisers schedule coordinated buying of small-cap cryptocurrencies on a chosen exchange at a chosen instant. Membership runs from open free-for-all (FFA) channels with tens of thousands of users to ranked/VIP tiers where higher payers receive the target ticker seconds before everyone else. The groups are the social-coordination substrate of the [[Pump-and-Dump Scheme]]. Confidence: high.

## Operating model
1. **Pre-announcement** (hours to days ahead): organisers post the exchange (overwhelmingly [[Binance]] for BTC pairs, with secondary venues like KuCoin, MEXC, LATOKEN, Poloniex), the exact UTC start time, the rules (FFA vs Ranked), and recruitment messaging. Countdown is repeated with rising frequency.
2. **Insider accumulation**: organisers and tier-1 members quietly build positions. Karbalaii 2025 reports 70% of pre-event volume transacts within one hour of T0, with a mean accumulation span around 36 hours when present.
3. **Reveal**: at T0 the ticker is posted as an image (to defeat copy-paste bots in some groups) or plain text. Ranked tiers receive it earlier.
4. **Pump** (seconds to a few minutes): coordinated market buys lift price 50%–500%+. La Morgia et al. show detectable signatures in per-second Binance candles within ~25 seconds.
5. **Dump** (minutes): organisers and fast followers exit; price collapses to or below pre-pump levels. Late buyers are the marks.

## Why it matters here
- Defines the `PumpEvent` records that feed [[features]] and [[backtest-pipelines]].
- Determines the time-zero anchor for cross-section construction in [[Cross-Section]].
- Their tiered information release explains why pre-T0 microstructure (insider accumulation) is detectable at all.

## Public datasets
For the full dataset registry with coverage dates, pump counts, and public links, see [[telegram-pump-anatomy]].

## Connections
- Coordinates events on [[Binance]] (primary) and secondary low-liquidity venues.
- Studied by [[xu-2019-anatomy]] and [[lamorgia-2023-doge]].
- Their tiered release directly motivates the [[Pre-Pump Accumulation]] signal we look for.

## Notes
- Channel turnover is high; groups rebrand frequently. Long-lived dataset value comes from labelled events, not from re-scraping.
- Some Hamrick et al. pumps did not actually execute (announced but cancelled), which matters for label hygiene when joining text data with market data.

> [!gap] Open questions
> Are there 2022+ public datasets covering the post-FTX/post-CEX-listing-dry-up era? Most public corpora end around 2021.
