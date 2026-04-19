---
type: thesis
title: "Impact Models for Low-Cap Crypto"
created: 2026-04-19
updated: 2026-04-19
tags:
  - thesis
  - synthesis
  - market-microstructure
  - cryptocurrency
  - impact-model
status: developing
related:
  - "[[Square-Root Law]]"
  - "[[almgren-chriss-2001-optimal-execution]]"
  - "[[donier-bonart-2015-bitcoin-metaorder]]"
  - "[[backtest-portfolio]]"
  - "[[Pump-and-Dump Scheme]]"
sources:
  - "https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf"
  - "https://www.worldscientific.com/doi/10.1142/S2382626615500082"
  - "https://arxiv.org/abs/0903.0497"
  - "https://arxiv.org/abs/1402.1448"
  - "https://arxiv.org/abs/2205.07385"
confidence: medium
---

# Impact Models for Low-Cap Crypto

Synthesis page on which market-impact and VWAP-execution model to use inside `backtest/portfolio/PriceImpact.py` and `TOPKPortfolio.py` for sub-$10M-ADV Binance spot pairs during pump-and-dump events.

## Overview
The pump-and-dump simulator must convert a model-ranked top-k portfolio into a realistic PnL. Naive close-to-close fills overstate returns by 5-10x at low caps, because both entry and exit walk the order book. The literature offers four canonical impact frameworks; only one (square-root) generalizes from equities to crypto and to the very-small-cap regime that pumps inhabit.

## Canonical Impact Models

| Model | Form | When valid | Crypto evidence |
|---|---|---|---|
| Kyle (1985) lambda | `I = lambda * Q` | Insider-trading equilibrium, small Q | Estimated for top-5 cryptos (Brauneis et al. 2021) |
| Almgren-Chriss (2001) | linear `g(v)`, linear `h(v)` | Small participation, scheduling problems | Used as scheduling baseline; not the impact shape itself |
| Almgren-Thum-Hauptmann (2005) | `h(v) ~ v^0.6` | US equities, Citigroup desk data | Not crypto |
| Square-root / Bouchaud | `I(Q) = Y * sigma * sqrt(Q/V)` | Universal across asset classes & sizes | Donier-Bonart 2015 confirms on BTC over 4 decades |
| Obizhaeva-Wang (2013) | LOB shape + finite resilience | Microstructural execution | Not directly tested on crypto |

The square-root law is the empirically dominant choice and the only one with primary crypto validation (Bitcoin). Confidence: **high** for general crypto, **medium** for transferring to low-cap altcoins.

## Empirical Evidence (Equities)
- Almgren-Thum-Hauptmann 2005: beta = 0.6 power law on Citigroup US equity executions; coefficient eta scales with `sigma / V^0.6`.
- Equity `Y` coefficient ranges 0.5 to 1.5 across 50+ studies (Said 2022 review).
- Square-root regime kicks in above ~0.01% of ADV; below that impact is linear in Q.

## Empirical Evidence (Crypto)
- **BTC, Mt. Gox 2011-2013** (Donier-Bonart 2015): `delta in [0.4, 0.7]`, square-root holds over 4 decades. Confidence: **high**.
- **BTC, ETH, BNB perpetuals on Binance** (Genet 2025, deep-learning VWAP paper): naive flat-allocation VWAP slippage of 1.6-2.3 basis points on 12-step-ahead executions for liquid majors.
- **Generic Binance spot crypto** (Easley, López de Prado, O'Hara 2024 working paper): Kyle and Amihud measures outperform other liquidity proxies; numerical lambda values not directly portable to USD-notional impact without further normalisation.
- **Low-cap altcoins**: no peer-reviewed direct estimates. Practitioner guidance (Coin Bureau, ECOS) reports 3-6% spreads and >10% impact for $100k orders on illiquid pairs. Confidence: **low**.

## VWAP Execution Mechanics
- VWAP is the standard institutional benchmark: `(1/Q) * integral of price(q) dq`.
- Under `I(Q) = beta * sqrt(Q)` with linear order-book shape, VWAP impact is `I_vwap = (2/3) * beta * sqrt(Q)`. Already correctly implemented in `PriceImpact.predict_vwap_impact_bps`.
- For low liquidity, TWAP can outperform VWAP because the volume curve is too noisy to forecast. For pumps, the volume profile is severely non-stationary so neither is informative; static-allocation deep-learning approaches (Genet 2025) are state of the art.
- Slippage realised in practice for crypto majors on Binance: 1-3 bps on liquid USDT pairs; 50-1000 bps on low-cap pairs at $10k-$100k notionals.

## Implications for Our Simulator (`PriceImpact` / `TOPKPortfolio`)

**Current state (2026-04-19)**:
- `PriceImpact.py` already implements `I(Q) = beta * sqrt(Q_usdt)` with no intercept, fits via constrained OLS, and uses `(2/3)*beta*sqrt(Q)` for VWAP. This is correct in shape and aligned with Bouchaud / Donier-Bonart. Confidence: **high**.
- `beta` is fitted per-pump from tick-level Binance trades, both buy and sell sides pooled. This is more conservative than using a universal coefficient.

**Recommended parameterization**:
1. Keep the `I = beta * sqrt(Q)` shape. It is the empirically correct shape for crypto and the only shape with crypto validation.
2. Use `Y * sigma / sqrt(V_daily)` as a **prior / sanity bound** on per-pump fitted `beta`. With `Y ~ 1`, `sigma_daily ~ 0.1` (10% daily vol typical for tiny cap), `V_daily ~ 1e6 USDT` for a low-cap pair, this gives `beta ~ 0.1 / sqrt(1e6) = 1e-4` in fractional terms, i.e. ~ `1e-4 * 1e4 = 1` bp impact at Q=$1, scaling to ~316 bps at Q=$100k. Verify per-pump fits land in this order of magnitude.
3. Cap `beta` from above (e.g., 95th percentile across pumps) to prevent extreme outliers from making PnL look impossibly bad.
4. Floor `beta` from below using a "minimum bid-ask half-spread" term (Almgren-Chriss `eps` analogue). Suggestion: add a fixed 5-10 bps floor for pumps where tick data is too sparse to fit.
5. Keep the VWAP `2/3 * beta * sqrt(Q)` for the average fill price — this is correct under the latent-order-book assumption.
6. Optional extension: an Almgren-Chriss-style scheduler to pick exit horizon, but at our 5-15 minute pump-window scale, terminal-execution assumption is acceptable.

**Do NOT** switch to linear impact. It will under-cost large fills and overstate top-k strategy returns.

## Open Questions
- See [[sqrt-coefficient-lowcap-binance]]: what is the right value of `Y` (or equivalent `beta` distribution) for sub-$1M ADV Binance pairs in 2017-2022 pump conditions?
- Does impact decay (`I_perm / I_temp` ratio) differ for pump-targeted assets vs ordinary low-cap assets? Pumps may have larger transient/uninformed component.
- Does Binance's tiered fee structure change effective Y for retail flow?

## Sources
- [[almgren-chriss-2001-optimal-execution]] — Almgren & Chriss, J. Risk 2001.
- [[donier-bonart-2015-bitcoin-metaorder]] — Donier & Bonart, Market Microstructure & Liquidity 2015 (arXiv:1412.4503).
- Bouchaud, Bonart, Donier, Gould, *Trades, Quotes and Prices*, Cambridge UP 2018, Chapter 12.
- Almgren, Thum, Hauptmann, Li, *Direct Estimation of Equity Market Impact*, Risk 2005.
- Said 2022, *Market Impact: Empirical Evidence, Theory and Practice*, arXiv:2205.07385.
- Donier, Bonart, Mastromatteo, Bouchaud, *A Fully Consistent, Minimal Model for Non-Linear Market Impact*, Quantitative Finance 2015 (arXiv:1412.0141).
- Obizhaeva & Wang, *Optimal Trading Strategy and Supply/Demand Dynamics*, J. Financial Markets 2013.
- Genet 2025, *Deep Learning for VWAP Execution in Crypto Markets*, arXiv:2502.13722.
- Easley, López de Prado, O'Hara 2024, *Microstructure and Market Dynamics in Crypto Markets*, SSRN 4814346.
