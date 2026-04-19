---
type: concept
title: "VWAP Execution"
created: 2026-04-19
updated: 2026-04-19
tags:
  - concept
  - execution
  - microstructure
status: seed
related:
  - "[[Square-Root Law]]"
  - "[[backtest-portfolio]]"
  - "[[impact-models-for-lowcap-crypto]]"
  - "[[Slippage]]"
sources:
  - "https://arxiv.org/abs/1412.4503"
  - "https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf"
complexity: intermediate
domain: market-microstructure
aliases:
  - "Volume-Weighted Average Price"
  - "VWAP"
---

# VWAP Execution

## Definition
VWAP execution is an algorithm that splits a parent order over a window targeting the volume-weighted average price `(1/Q) * integral price(q) dq`. It is the standard institutional benchmark for measuring impact and slippage and the natural fill assumption when no better order-by-order schedule is available.

## Why it matters
Under the [[Square-Root Law]] with a linear latent-order-book shape, the VWAP impact has the closed form `I_vwap(Q) = (2/3) * beta * sqrt(Q)`: the average fill is two-thirds of the terminal impact. This is the cost a trader actually pays when liquidating uniformly through the book. For low-liquidity pump windows, the volume profile is severely non-stationary, so VWAP and TWAP both degrade and static deep-learning allocators (Genet 2025) become state of the art.

## How it appears in this project
- `PriceImpact.predict_vwap_impact_bps` in `backtest/portfolio/PriceImpact.py` ([[backtest-portfolio]]) implements `(2/3) * beta * sqrt(Q)` directly.
- The [[backtest-portfolio]] simulator assumes terminal VWAP execution within a single pump bar, which is acceptable at the 5-15 minute horizon.

## Related
- [[Square-Root Law]]
- [[backtest-portfolio]]
- [[impact-models-for-lowcap-crypto]]
- [[Slippage]]

## Sources
- Donier & Bonart 2015, *A Million Metaorder Analysis of Market Impact on Bitcoin*, arXiv:1412.4503.
- Almgren & Chriss 2001, *Optimal Execution of Portfolio Transactions*, J. Risk 3.
