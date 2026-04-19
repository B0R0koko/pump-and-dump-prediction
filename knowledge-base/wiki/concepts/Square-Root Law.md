---
type: concept
title: "Square-Root Law (of Market Impact)"
created: 2026-04-19
updated: 2026-04-19
tags:
  - concept
  - market-microstructure
  - impact-model
status: developing
related:
  - "[[Temporary vs Permanent Impact]]"
  - "[[VWAP Execution]]"
  - "[[Slippage]]"
  - "[[almgren-chriss-2001-optimal-execution]]"
  - "[[donier-bonart-2015-bitcoin-metaorder]]"
sources:
  - "https://arxiv.org/abs/1412.4503"
  - "https://arxiv.org/pdf/2205.07385"
complexity: intermediate
domain: market-microstructure
aliases:
  - "Square-Root Impact Law"
  - "Bouchaud square-root law"
---

# Square-Root Law

## Definition
The Square-Root Law states that the average price impact of a metaorder of total size Q is `I(Q) ~ Y * sigma * sqrt(Q / V)`, where `sigma` is daily volatility, `V` is daily volume, and `Y` is an O(1) coefficient. Equivalently, in pure-notional form, `I(Q) = beta * sqrt(Q)` with beta absorbing the sigma/sqrt(V) constants.

It is one of the most robust empirical regularities in financial microstructure, holding across equities, futures, options, FX, and (per Donier-Bonart 2015) Bitcoin, across roughly 4-6 orders of magnitude in metaorder size.

## Why it matters
- Linear impact (Almgren-Chriss) underestimates cost for large parent orders; square-root is the empirically correct shape above ~0.01-0.1% of ADV.
- Impact is independent of execution duration `T` and of slice count `N` provided participation rate is moderate. So scheduling matters less than total size.
- Coefficient `Y` is approximately constant within an asset class; "approximately 1" is the rule of thumb for large-cap equities.

## How it appears in this project
- `backtest/portfolio/PriceImpact.py` fits `I(Q) = beta * sqrt(Q_usdt)` directly via constrained OLS in sqrt-notional space, with no intercept ([[backtest-portfolio]]).
- `predict_vwap_impact_bps` uses the closed-form integral `I_vwap(Q) = (2/3) * beta * sqrt(Q)`, which assumes a linear order-book shape (consistent with Donier-Bonart's latent order book theory).
- `beta` is fitted per-pump from tick-level Binance data, so we get an event-conditional impact coefficient rather than a universal one.

## Validity range and caveats
- Holds for participation rates `phi = Q/V` roughly in `[1e-4, 0.3]`. Below that, impact is linear in `Q` (Said 2022). Above, it saturates / breaks down.
- For very small daily volumes (<$1M), the "small Q" linear regime barely exists: even a $1k order is ~0.1% of ADV. So square-root is the right asymptotic.
- The coefficient `Y` empirically scales as `~ sigma / sqrt(V)`, which means tiny-cap altcoins (high sigma, low V) are expected to have *much* larger `beta` in the pure-notional form than BTC.

## Related
- [[Temporary vs Permanent Impact]]
- [[VWAP Execution]]
- [[Slippage]]
- [[almgren-chriss-2001-optimal-execution]] — competing linear-impact framework
- [[donier-bonart-2015-bitcoin-metaorder]] — direct crypto evidence
- [[impact-models-for-lowcap-crypto]] — synthesis for our use case

## Examples
- Equity Y ~ 0.5-1.5 across most studies (Almgren-Thum-Hauptmann 2005 finds delta=0.6, close to but not exactly 0.5).
- Bitcoin Y close to 1 with delta ~ 0.5 (Donier-Bonart 2015).
- Low-cap Binance altcoin during a pump: not directly measured in literature; our fitted `beta` per pump-event is the only available estimate. See [[impact-models-for-lowcap-crypto]] and [[sqrt-coefficient-lowcap-binance]].
