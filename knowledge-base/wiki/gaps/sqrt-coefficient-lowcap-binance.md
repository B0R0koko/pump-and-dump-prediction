---
type: question
title: "What is the square-root impact coefficient for sub-$1M ADV Binance pairs?"
created: 2026-04-19
updated: 2026-04-19
tags:
  - gap
  - question
  - market-microstructure
  - impact-model
status: open
related:
  - "[[Square-Root Law]]"
  - "[[impact-models-for-lowcap-crypto]]"
  - "[[backtest-portfolio]]"
priority: high
---

# What is the square-root impact coefficient for sub-$1M ADV Binance pairs?

## The gap
The Square-Root Law `I(Q) = Y * sigma * sqrt(Q/V)` is empirically validated on equities (Y ~ 0.5-1.5) and on Bitcoin (Donier-Bonart 2015). No peer-reviewed study has measured `Y` (or equivalently the pure-notional `beta = Y * sigma / sqrt(V)`) for low-volume Binance altcoin spot pairs, which is precisely the regime our simulator operates in.

## Why it matters
Our `backtest/portfolio/PriceImpact.py` fits `beta` per-pump from tick data. Without a literature anchor we cannot tell whether fitted values are:
- realistic (consistent with Y ~ O(1)),
- over-fit (noise-driven),
- biased downward by missing pre-pump baseline,
- biased upward by including the manipulated jump itself.

A literature value would let us regularize fits, set sane floors/caps, and report confidence intervals on simulator PnL.

## What we know
- BTC: Y ~ 1, delta ~ 0.5 (Donier-Bonart 2015).
- Equities: Y ~ 0.5-1.5 across studies (Said 2022).
- Practitioner heuristic: low-cap pairs have 3-6% spreads, >10% impact for $100k orders ([source](https://coinbureau.com/guides/low-liquidity-crypto-indicators)). Translating to `beta`: I(100k) ~ 1000 bps => beta ~ 1000 / sqrt(1e5) ~ 3.16, in pure-notional bps-per-sqrt(USDT). For comparison BTC at $1B ADV would be `beta ~ 1 * 0.05 / sqrt(1e9) * 1e4 ~ 0.016`.

## How to close it
1. Compute `beta` distribution across our entire pump dataset, stratify by ADV bucket and pre-pump volatility.
2. Convert to dimensionless `Y = beta * sqrt(V_daily) / sigma_daily`. If `Y` clusters near 1, the universal law transfers.
3. If `Y` deviates systematically (e.g., > 5 for tiny caps), document the new empirical regime and consider publishing.
4. Cross-check against Binance L2 order book snapshots (if available) by integrating depth to compute model-free impact at $10k, $100k, $1M.

## Decision impact
This is the most impactful single calibration decision in the simulator. A factor-of-2 error in `beta` translates to ~factor-2 error in net PnL, since both entry and exit costs scale with sqrt(Q).
