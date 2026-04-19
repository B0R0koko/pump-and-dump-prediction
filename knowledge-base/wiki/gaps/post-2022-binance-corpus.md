---
type: gap
title: "Post-2022 Binance Telegram-pump corpus"
created: 2026-04-19
updated: 2026-04-19
tags:
  - gap
  - dataset
  - manipulation
status: developing
related:
  - "[[karbalaii-2025-microstructure]]"
  - "[[lamorgia-2023-doge]]"
  - "[[hamrick-2021-ecosystem]]"
  - "[[telegram-pump-anatomy]]"
  - "[[Telegram Pump Groups]]"
sources: []
priority: high
---

# Post-2022 Binance Telegram-pump corpus

## The gap
All major public Telegram-coordinated pump corpora end on or before 2021:
- [[xu-2019-anatomy]]: Jun 2018 – Feb 2019, 412 events.
- [[hamrick-2021-ecosystem]]: Jan – Jul 2018, 4,818 announced signals.
- [[lamorgia-2023-doge]]: through 2021, ~900 events on Binance `SYM/BTC`.

Karbalaii 2025 closes the temporal gap (Aug 2024 – Feb 2025, 485 events) but only for Poloniex. Bolz et al. 2024 covers 2024 across LATOKEN / XT / Poloniex / KuCoin / MEXC, but Binance is only 2.2% of their event mix.

## Why it matters
Our pipeline (`market_data/`, `preprocessing/`) targets Binance specifically. The time-split convention in `BasePipeline` (test > 2021-05-01) implicitly assumes the Binance pump regime persists past mid-2021, but no public labelled corpus validates that assumption on Binance specifically. Regime drift between 2018-2021 (`SYM/BTC` dominance) and post-2022 (USDT-stablecoin dominance, post-FTX liquidity reshuffle, harsher Binance enforcement) is plausible.

## What "closing it" looks like
Either of these would close the gap:
1. A public 2022+ Telegram-pump dataset on Binance, with announcement timestamps and target tickers.
2. An internally curated Binance corpus built from monitoring known organiser channels in the Bolz/Hamrick channel registries.

## How it shows up in our work
- Our `test > 2021-05-01` split currently has *no out-of-sample data* aligned with the post-FTX regime. Reported test metrics carry an implicit "assuming the regime persisted" caveat.
- Cross-venue transfer (related gap, also filed inline in `gaps/_index.md`) cannot substitute: Bolz reports TOP5=55.81% on secondary venues vs >90% F1 in La Morgia on Binance, so secondary-venue training does not obviously transfer.

## Open sub-questions
- Has Binance enforcement actually killed Telegram-coordinated pumps on its venue, or merely pushed them onto USDT pairs and out of public view?
- Is the post-2022 cross-section of pump-eligible Binance pairs different in market-cap/liquidity distribution from 2019-2021?

## Hand-offs
- developer: monitor for Bolz / Karbalaii follow-ups that release a Binance-annotated extension.
- research: ingest Perseus 2025 (arXiv:2503.01686) — its organiser-wallet identification could bootstrap a Binance corpus from on-chain footprints.
