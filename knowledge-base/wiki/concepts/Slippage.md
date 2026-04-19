---
type: concept
title: "Slippage"
created: 2026-04-19
updated: 2026-04-19
tags:
  - concept
  - execution
  - microstructure
status: seed
related:
  - "[[VWAP Execution]]"
  - "[[Square-Root Law]]"
  - "[[backtest-portfolio]]"
  - "[[Temporary vs Permanent Impact]]"
sources:
  - "https://arxiv.org/abs/1412.4503"
  - "https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf"
complexity: basic
domain: market-microstructure
aliases:
  - "Execution Slippage"
  - "Implementation Shortfall"
---

# Slippage

## Definition
Slippage is the difference between the expected price of an order at decision time and the realized average fill price. It combines the bid-ask spread, market impact (both temporary and permanent), and latency-induced adverse selection: roughly `slippage = spread + impact + adverse selection`.

## Why it matters
Slippage is what turns a model-ranked top-k strategy from a paper-PnL fantasy into a realistic backtest. On Binance majors slippage is 1-3 bps; on low-cap pump-and-dump pairs at $10k-$100k notionals it can run 50-1000 bps, large enough to flip the sign of strategy returns. Distinguishing slippage (the realized cost) from impact (the underlying price-displacement model) is essential when comparing simulator outputs to executable strategies.

## How it appears in this project
- `backtest/portfolio/PriceImpact.py` ([[backtest-portfolio]]) models the impact component via `I(Q) = beta * sqrt(Q)` and applies it to fill prices through `predict_vwap_impact_bps`.
- The [[backtest-portfolio]] PnL accounts for slippage on both entry and exit; without it, raw close-to-close returns overstate strategy edge by 5-10x at low caps.

## Related
- [[VWAP Execution]]
- [[Square-Root Law]]
- [[backtest-portfolio]]
- [[Temporary vs Permanent Impact]]

## Sources
- Donier & Bonart 2015, *A Million Metaorder Analysis of Market Impact on Bitcoin*, arXiv:1412.4503.
- Almgren & Chriss 2001, *Optimal Execution of Portfolio Transactions*, J. Risk 3.
