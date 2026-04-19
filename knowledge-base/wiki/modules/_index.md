---
type: domain
title: "Modules Index"
created: 2026-04-19
updated: 2026-04-19
tags:
  - meta
  - index
  - modules
status: developing
subdomain_of: ""
page_count: 7
---

# Modules

One note per major package in the parent codebase. Each note captures purpose, public surface, key files, dependencies, and design notes that don't live in code.

## Pages

- [[core]] — shared domain types, paths, time utilities
- [[market_data]] — Binance scrapers (trades, klines)
- [[preprocessing]] — raw → HIVE-partitioned parquet
- [[features]] — `PumpsFeatureWriter`, multi-offset feature engineering
- [[backtest-pipelines]] — ML pipelines (CatBoost*, LR, RF, Ranker)
- [[backtest-portfolio]] — top-k portfolio, price impact, VWAP
- [[backtest-utils]] — dataset building, evaluation metrics, robustness
