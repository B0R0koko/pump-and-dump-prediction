---
type: concept
title: "Kyle's Lambda"
created: 2026-04-19
updated: 2026-04-19
tags:
  - concept
  - microstructure
  - impact
status: seed
related:
  - "[[kyle-1985-continuous-auctions]]"
  - "[[Square-Root Law]]"
  - "[[impact-models-for-lowcap-crypto]]"
---

# Kyle's Lambda

## Definition
The linear price-impact coefficient from Kyle (1985). In Kyle's equilibrium model, a single informed trader submits a market order of size `Q` and the market maker moves the price by:

```
delta_p = lambda * Q
```

where `lambda` (Kyle's lambda) measures price sensitivity to order flow, or equivalently, the illiquidity of the asset. Higher lambda means less liquid: larger price moves per unit of signed order flow.

## Origin
From [[kyle-1985-continuous-auctions]]: "Continuous Auctions and Insider Trading" (Econometrica, 1985). Kyle derives lambda endogenously from the market-maker's inference problem about the informed trader's signal. At equilibrium, lambda depends on the ratio of informed to noise-trader variance.

## Relation to related concepts
- The [[Square-Root Law]] replaces the linear `lambda * Q` form with a concave `beta * sqrt(Q)` form. For large orders (which is the relevant regime for the project's pump events), the square-root form is empirically more accurate.
- [[impact-models-for-lowcap-crypto]] discusses why the linear Kyle model understates impact for the low-cap, low-liquidity pairs targeted in pump events.
- Empirically estimating lambda from order flow data is a common first step in microstructure research ([[albers-2022-bitcoin-fragmentation]] does this for Bitcoin).
