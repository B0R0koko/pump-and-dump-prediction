---
type: entity
title: "Binance"
created: 2026-04-19
updated: 2026-04-19
tags:
  - entity
  - exchange
  - data-source
status: developing
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[market_data]]"
sources: []
entity_type: organization
role: "Primary data source and venue for the pump events studied in this project."
first_mentioned: ""
---

# Binance

## What it is
Binance is a centralized cryptocurrency exchange. It hosts spot trading for thousands of assets and publishes historical trade and kline data, which makes it the data source for this project.

## Why it matters here
- All raw trade and kline data ingested by [[market_data]] comes from Binance.
- The pump events studied are typically targeted at low-cap pairs listed on Binance, because that is where coordinated retail liquidity congregates.
- Microstructure assumptions in [[backtest-portfolio]] (price impact, VWAP execution) reflect Binance spot mechanics.

## Connections
- Data pipeline: [[market_data]] → [[preprocessing]] → [[features]]
- Concept: [[Pump-and-Dump Scheme]] (Binance is the dominant venue in our dataset)

## Volume reliability
Binance reported volume on small-cap pairs is unreliable. [[cong-2023-wash-trading]] estimates roughly 70% of reported BTC volume on unregulated exchanges is wash trading. This is a key caveat for the impact-model `Y · σ / √V` calibration in [[backtest-portfolio]], which depends on `V_daily`: feeding raw reported volume into the square-root law underestimates impact at low caps. Any calibration of the impact coefficient should either deflate reported volume or rely on tick-level executed-trade aggregates.

## Open notes

> [!gap] Coverage
> Confirm exact pair universe and date coverage; record here once the dataset is profiled.
