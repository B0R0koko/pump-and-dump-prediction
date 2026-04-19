---
type: module
title: "preprocessing"
created: 2026-04-19
updated: 2026-04-19
tags:
  - module
  - data-pipeline
status: seed
related:
  - "[[market_data]]"
  - "[[features]]"
sources: []
path: "preprocessing/"
language: python
purpose: "Transforms raw Binance trades into HIVE-partitioned parquet for downstream feature engineering."
maintainer: borokoko
last_updated: 2026-04-19
depends_on:
  - "[[core]]"
  - "[[market_data]]"
used_by:
  - "[[features]]"
---

# preprocessing

## Purpose
Bridge layer. Converts raw scraped data into the columnar, partitioned format that the feature writer can scan efficiently.

## Public surface
- Entry point: `preprocessing/run.py`
- Main pipeline: `preprocessing/pipelines/binance_spot_trades_to_hive.py` — converts raw Binance spot trades to HIVE-partitioned parquet

## Design notes
- HIVE partitioning chosen so partial reads (per-pair, per-day) are cheap during feature generation.
