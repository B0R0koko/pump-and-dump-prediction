---
type: concept
title: "Temporary vs Permanent Impact"
created: 2026-04-19
updated: 2026-04-19
tags:
  - concept
  - microstructure
  - execution
status: seed
related:
  - "[[almgren-chriss-2001-optimal-execution]]"
  - "[[Square-Root Law]]"
  - "[[impact-models-for-lowcap-crypto]]"
  - "[[donier-bonart-2015-bitcoin-metaorder]]"
sources:
  - "https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf"
  - "https://arxiv.org/abs/1412.4503"
complexity: intermediate
domain: market-microstructure
aliases:
  - "Transient and Permanent Impact"
  - "Impact Decomposition"
---

# Temporary vs Permanent Impact

## Definition
Market impact decomposes into a temporary component (the slippage paid on each child order, which recovers after execution ends) and a permanent component (a persistent price shift reflecting informational content). Almgren-Chriss separates them additively: total cost = temporary `h(v)` paid per slice plus permanent drift `g(v)` integrated over the trajectory.

## Why it matters
The split governs how aggressively a trader should execute: temporary cost can be amortized by trading slowly, but permanent impact is paid regardless of schedule. For pump-and-dump exits, the question is what fraction of price impact is transient (and so reverts during the dump) versus permanent (and so locks in adverse fills). Donier-Bonart 2015 find Bitcoin metaorder impact is largely transient, decaying close to fully after order completion.

## How it appears in this project
- `backtest/portfolio/PriceImpact.py` ([[backtest-portfolio]]) currently models a single combined impact `I(Q) = beta * sqrt(Q)` and assumes terminal VWAP execution, conflating the two components.
- An Almgren-Chriss-style decomposition is a natural extension to better cost short-horizon pump exits where the temporary part dominates.

## Related
- [[almgren-chriss-2001-optimal-execution]]
- [[Square-Root Law]]
- [[impact-models-for-lowcap-crypto]]
- [[donier-bonart-2015-bitcoin-metaorder]]

## Sources
- Almgren & Chriss 2001, *Optimal Execution of Portfolio Transactions*, J. Risk 3.
- Donier & Bonart 2015, *A Million Metaorder Analysis of Market Impact on Bitcoin*, arXiv:1412.4503.
