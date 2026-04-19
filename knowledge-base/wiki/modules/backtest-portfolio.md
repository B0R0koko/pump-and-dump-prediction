---
type: module
title: "backtest-portfolio"
created: 2026-04-19
updated: 2026-04-19
tags:
  - module
  - portfolio
  - backtest
status: seed
related:
  - "[[backtest-pipelines]]"
  - "[[backtest-utils]]"
sources: []
path: "backtest/portfolio/"
language: python
purpose: "Execution simulation: top-k portfolio construction, price-impact modeling, VWAP estimation, PnL calculation. TOPKPortfolio is the main orchestrator."
maintainer: borokoko
last_updated: 2026-04-19
depends_on:
  - "[[core]]"
  - "[[backtest-pipelines]]"
used_by:
  - "[[backtest-utils]]"
---

# backtest-portfolio

## Purpose
Convert model rankings into a tradable portfolio and simulate execution to get a realistic PnL.

## Components
- `BasePortfolio` (abstract)
- `TOPKPortfolio` — selects top-k ranked assets per cross-section
- `PriceImpact` — execution-impact model
- VWAP estimation utilities

## Interfaces
- `QuoteToUSDTProvider` (Protocol) — currency conversion
- `ExecutionImpactModel` (Protocol) — pluggable impact

## Design notes
- Top-k chosen because the model's task is ranking, not absolute scoring.
- Price-impact model exists because raw close-to-close PnL drastically overstates returns at low caps.

## References

### Impact theory
- [[kyle-1985-continuous-auctions]] — foundational impact model, source of Kyle's lambda.
- [[almgren-chriss-2001-optimal-execution]] — optimal execution scheduling under impact.
- [[bouchaud-farmer-lillo-2009-markets-digest]] — review of how markets absorb metaorders.

### Empirical calibration
- [[almgren-thum-hauptmann-2005-direct-estimation]] — direct empirical impact calibration.
- [[toth-2011-anomalous-impact]] — square-root impact law on equities.

### Crypto
- [[donier-bonart-2015-bitcoin-metaorder]] — square-root law confirmed on Bitcoin.
- [[albers-2022-bitcoin-fragmentation]] — cross-venue impact and fragmentation effects.
- [[impact-models-for-lowcap-crypto]] — internal thesis page on calibrating impact for low-cap crypto.

### Concepts used
- [[Square-Root Law]] — used in `PriceImpact.predict_vwap_impact_bps`.
- [[Temporary vs Permanent Impact]] — decomposition assumed by the impact model.
- [[VWAP Execution]] — execution benchmark applied in the simulator.
- [[Slippage]] — realized cost relative to mid/quoted price.
