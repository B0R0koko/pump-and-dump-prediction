---
type: module
title: "market_data"
created: 2026-04-19
updated: 2026-04-19
tags:
  - module
  - data-ingestion
status: seed
related:
  - "[[core]]"
  - "[[preprocessing]]"
  - "[[Binance]]"
sources: []
path: "market_data/"
language: python
purpose: "Scrapy-based parsers that download Binance trade and kline data."
maintainer: borokoko
last_updated: 2026-04-19
depends_on:
  - "[[core]]"
used_by:
  - "[[preprocessing]]"
---

# market_data

## Purpose
Ingest layer. Pulls raw trade and kline data from Binance into local storage.

## Public surface
- Scrapy spiders for trades and klines.

## Key files
- `market_data/parsers/binance/BinanceBaseParser.py` — shared base class for all Binance parsers
- `market_data/parsers/binance/BinanceSpotTradesParser.py` — spider for raw spot trade data
- `market_data/parsers/binance/BinanceSpotKlinesParser.py` — spider for OHLCV kline data

## Design notes
- Scrapy chosen for backpressure and resumability.
- Output feeds [[preprocessing]] which normalizes to HIVE-partitioned parquet.
