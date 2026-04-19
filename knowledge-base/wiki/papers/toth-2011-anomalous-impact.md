---
type: paper
title: "Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - market-microstructure
  - impact-model
  - square-root-law
status: summarized
related:
  - "[[Square-Root Law]]"
  - "[[Temporary vs Permanent Impact]]"
  - "[[VWAP Execution]]"
  - "[[Slippage]]"
  - "[[backtest-portfolio]]"
  - "[[bouchaud-farmer-lillo-2009-markets-digest]]"
  - "[[donier-bonart-2015-bitcoin-metaorder]]"
sources:
  - "https://arxiv.org/abs/1105.1694"
year: 2011
authors:
  - Bence Tóth
  - Yves Lempérière
  - Cyril Deremble
  - Joachim de Lataillade
  - Julien Kockelkoren
  - Jean-Philippe Bouchaud
venue: "Physical Review X, vol. 1, art. 021006"
key_claim: "Metaorder price impact follows a square-root law in size across more than two orders of magnitude in participation, with the exponent close to 0.5 essentially independent of stock, period, or market regime."
methodology: "Empirical analysis of approximately 500,000 institutional metaorders from Capital Fund Management's proprietary execution data; introduces the latent liquidity / latent order book interpretation."
contradicts:
  - "[[kyle-1985-continuous-auctions]]"
supports:
  - "[[Square-Root Law]]"
url: "https://journals.aps.org/prx/abstract/10.1103/PhysRevX.1.021006"
---

# Anomalous Price Impact and the Critical Nature of Liquidity (Tóth et al., 2011)

## TL;DR
The most-cited empirical confirmation of the square-root law of market impact. Using approximately half a million metaorders executed by Capital Fund Management on global equities, the authors show that the average price impact of a metaorder of size `Q` scales as `sqrt(Q / V)` (with `V` average daily volume), with exponent close to 0.5 essentially independent of stock, time period, or market regime. They argue this universality cannot arise from a Kyle-style linear-impact equilibrium and propose the "latent order book" picture: the visible book is the tip of a much larger queue of conditional buy/sell intentions whose density vanishes near the current price (V-shaped), forcing impact to be sub-linear and the market to operate in a self-organized critical state.

## Key claims
- Metaorder impact is well-fitted by `I(Q) = Y * sigma * sqrt(Q / V_daily)` with `Y` of order one.
- Exponent ~0.5 is universal across stocks, market caps, and years of data.
- Linear impact (Kyle) is empirically rejected at the metaorder scale.
- A "latent order book" with V-shaped density at the price explains why impact must be concave.
- Markets operate near criticality: visible liquidity is small and has to be progressively revealed.

## Methodology
- Dataset: roughly 500,000 CFM metaorders, equities, multi-year coverage.
- Aggregate metaorder slippage relative to arrival mid-price, normalize by daily volatility and volume, and bin by participation rate.
- Cross-validate against scrambled controls and across sub-periods to confirm robustness.

## Strengths
- Largest published metaorder dataset of its time; statistical power makes the square-root law hard to dispute.
- Provides a coherent theoretical interpretation (latent order book) tying empirical regularity to market self-organisation.
- Universality result simplifies cross-asset impact modeling.

## Weaknesses / Critiques
- Proprietary data: cannot be replicated independently.
- Latent-order-book argument is suggestive rather than fully derived.
- Equity focus; crypto extension came later (see [[donier-bonart-2015-bitcoin-metaorder]]).

## Relation to our work
- Cited in `paper/access.tex` both in the impact-model motivation ("Tóth et al. demonstrated its universality across stocks and time periods using a large meta-order dataset") and in the impact-regression figure caption ("The concave relationship is consistent with the square-root impact law documented across asset classes").
- Directly justifies our `(2/3) * beta * sqrt(Q)` VWAP impact specification in `PriceImpact.predict_vwap_impact_bps` (`backtest/portfolio/PriceImpact.py`), which assumes the universal exponent 0.5 holds in tiny-cap altcoin markets where pump targets typically trade.

## Cited concepts
- [[Square-Root Law]]
- [[Temporary vs Permanent Impact]]
- [[Slippage]]
