---
type: concept
title: "Pump Announcement Window"
created: 2026-04-19
updated: 2026-04-19
tags:
  - concept
  - manipulation
  - microstructure
status: seed
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Telegram Pump Groups]]"
  - "[[telegram-pump-anatomy]]"
sources:
  - "https://arxiv.org/abs/1811.10109"
  - "https://arxiv.org/abs/2105.00733"
complexity: basic
domain: market-microstructure
aliases:
  - "Announcement Window"
  - "T0 Window"
---

# Pump Announcement Window

## Definition
The Pump Announcement Window is the moment a pump organizer posts the target ticker, exchange, and start time to the coordination chat. It is the canonical t=0 in our event sequence: every feature offset, every cross-section snapshot, and every label in the `pumps_and_dumps` pipeline is anchored to this instant.

## Why it matters
Before T0 the target asset is not public to the bulk of the channel: only ranked tiers receive the symbol seconds earlier. The seconds around the announcement concentrate the most informative microstructure footprints (order pressure, taker imbalance, rush orders) and define the only window in which a pre-event prediction model can act on a still-cheap price.

## How it appears in this project
- All [[features]] are computed at offsets relative to the pump's announcement timestamp.
- [[Cross-Section]] construction selects all eligible small-cap pairs at T0 to form the per-event ranking pool.
- The [[backtest-portfolio]] simulator assumes execution starts at T0 and unwinds within a short pump window.

## Related
- [[Pump-and-Dump Scheme]]
- [[Telegram Pump Groups]]
- [[telegram-pump-anatomy]]

## Sources
- Xu & Livshits 2019, *The Anatomy of a Cryptocurrency Pump-and-Dump Scheme*, arXiv:1811.10109.
- La Morgia, Mei, Sassi, Stefa 2023, *The Doge of Wall Street*, arXiv:2105.00733.
