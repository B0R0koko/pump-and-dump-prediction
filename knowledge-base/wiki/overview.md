---
type: overview
title: "Wiki Overview"
created: 2026-04-19
updated: 2026-04-19
tags:
  - meta
  - overview
status: developing
---

# Overview

This is the persistent knowledge base for the **pumps_and_dumps** project: detecting cryptocurrency pump-and-dump schemes via machine learning on Binance market data.

The wiki spans two layers:

## Code map (Mode B)
Architecture, modules, decisions, and data flows that mirror the parent codebase. See [[modules/_index|Modules]].

## Research (Mode E)
Papers, concepts, and our evolving thesis on what works and what doesn't in market-manipulation detection. See [[papers/_index|Papers]], [[concepts/_index|Concepts]], and [[thesis/_index|Thesis]].

## Pipeline summary

```
Raw Binance trades
  → preprocessing/   (HIVE-partitioned parquet)
  → features/        (returns, flow imbalance, slippage, multi-offset)
  → backtest/pipelines/  (CatBoost / LR / RF; cross-section ranking)
  → backtest/portfolio/  (top-k construction, VWAP, price impact)
  → analysis (PnL, robustness)
```

## How to use this wiki

1. **Add a source** → drop the paper URL or PDF in the chat, say "ingest <name>".
2. **Ask a question** → "what do you know about X" reads `index.md` then relevant pages.
3. **Capture an insight** → "save this" files the current chat snippet into the right folder.
4. **Audit health** → "lint the wiki" finds orphans, dead links, stale claims.
5. **Auto research** → `/autoresearch <topic>` runs an autonomous web research loop and files findings here.
