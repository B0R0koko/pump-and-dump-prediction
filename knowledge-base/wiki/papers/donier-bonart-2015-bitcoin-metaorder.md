---
type: paper
title: "A Million Metaorder Analysis of Market Impact on Bitcoin"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - market-microstructure
  - cryptocurrency
  - bitcoin
  - impact-model
status: summarized
related:
  - "[[Square-Root Law]]"
  - "[[backtest-portfolio]]"
  - "[[Binance]]"
sources:
  - "https://arxiv.org/abs/1412.4503"
year: 2015
authors:
  - Jonathan Donier
  - Julius Bonart
venue: "Market Microstructure and Liquidity, World Scientific"
key_claim: "The square-root law of market impact holds on the Bitcoin/USD market over four decades of metaorder size, despite no statistical-arbitrage or market-making infrastructure."
methodology: "Reconstructs ~1M metaorders from Mt. Gox trade-by-trade data with broker tags; fits I(Q) ~ Y * sigma * sqrt(Q/V)."
contradicts: []
supports:
  - "[[Square-Root Law]]"
  - "[[bouchaud-farmer-lillo-2009-markets-digest]]"
url: "https://www.worldscientific.com/doi/10.1142/S2382626615500082"
---

# A Million Metaorder Analysis of Market Impact on Bitcoin (Donier & Bonart, 2015)

## TL;DR
First rigorous empirical confirmation of the square-root impact law in a crypto venue. Using ~1M Mt. Gox metaorders (broker-tagged), they find impact scales as Q^delta with `delta in [0.4, 0.7]`, centred near 0.5, consistent with the equity-market square-root law. Holds across 4 orders of magnitude in metaorder size despite Bitcoin lacking the institutional market-makers and stat-arb that the "Bouchaud universal" theory was originally explained by.

## Key claims
- Square-root law is universal: it does not require modern HFT/market-makers to emerge.
- Impact during a metaorder follows a concave trajectory (square-root in time) and partially reverts after completion.
- Order flow can be split into "informed" (permanent) and "uninformed" (transient) components; the latter decays almost completely.

## Methodology
- Mt. Gox 2011-2013 BTC/USD trade data with anonymized broker IDs lets metaorders be reconstructed.
- Daily volume normalisation `phi = Q / V_daily`; volatility normalisation `sigma_daily`.
- Regress `I(Q) / sigma` on `sqrt(phi)` and on `phi^delta`.

## Strengths
- Crypto-native dataset, and an early one (Mt. Gox era).
- Order-of-magnitude range tested is wider than most equity studies.
- Theoretical match to Donier-Bonart-Bouchaud-Mastromatteo "latent order book" propagator model.

## Weaknesses / Critiques
- Mt. Gox is no longer relevant; modern Binance order books and retail flow look very different.
- Bitcoin only — no altcoin coverage, no information on what happens at <$1M ADV pairs.
- Impact decay is studied at minutes to hours timescale, not the 5-15 minute horizon relevant for pump-and-dump exits.

## Relation to our work
- Strongest direct empirical justification for our `I(Q) = beta * sqrt(Q_usdt)` choice in `backtest/portfolio/PriceImpact.py` ([[backtest-portfolio]]).
- BUT: their dataset is BTC, not low-cap altcoins; our `beta` empirically fitted from pump-event tick data is the more relevant number, and is expected to be much larger than the Donier-Bonart-implied BTC value due to depth scaling.
- See [[impact-models-for-lowcap-crypto]] for parameter recommendations.

## Cited concepts
- [[Square-Root Law]]
- [[Temporary vs Permanent Impact]]
- [[Slippage]]
