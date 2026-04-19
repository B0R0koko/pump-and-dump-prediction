---
type: paper
title: "Continuous Auctions and Insider Trading"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - market-microstructure
  - impact-model
  - foundational
status: summarized
related:
  - "[[Temporary vs Permanent Impact]]"
  - "[[Square-Root Law]]"
  - "[[backtest-portfolio]]"
  - "[[almgren-chriss-2001-optimal-execution]]"
sources:
  - "https://www.jstor.org/stable/1913210"
year: 1985
authors:
  - Albert S. Kyle
venue: "Econometrica, vol. 53, no. 6, pp. 1315-1335"
key_claim: "A strategic informed insider trades against noise traders and a competitive risk-neutral market maker; the equilibrium price is linear in cumulative order flow with slope lambda, the inverse of market depth."
methodology: "Sequential and continuous-auction equilibrium models with Bayesian updating by the market maker; closed-form derivation of optimal insider trading intensity and price-impact coefficient lambda."
contradicts: []
supports:
  - "[[Temporary vs Permanent Impact]]"
url: "https://doi.org/10.2307/1913210"
---

# Continuous Auctions and Insider Trading (Kyle, 1985)

## TL;DR
Foundational market-microstructure paper. A monopolistic insider with private information about the asset's terminal value trades alongside noise traders; a risk-neutral, competitive market maker observes only aggregate order flow and sets a price equal to the conditional expectation of value. In equilibrium the price is linear in cumulative net flow, with slope `lambda` (now called Kyle's lambda), which measures price impact per unit of order flow and is the inverse of market depth.

## Key claims
- Information is incorporated into prices gradually; the insider spreads trades over time to disguise informational content.
- Equilibrium price impact is linear: `dP = lambda * dQ`, where `lambda = sigma_v / (2 * sigma_u)` (ratio of value to noise-flow volatility).
- Market depth (`1/lambda`) scales with the volume of noise trading: more noise traders means lower impact per unit flow.
- The insider's expected profits equal half the total information value, the rest is captured by reduced noise-trader losses through better prices.

## Methodology
- Two formulations: a single-auction batch model and a continuous-time limit.
- Market maker is Bayesian, risk-neutral, and competitive (zero expected profit).
- Equilibrium derived by solving fixed-point: insider's optimal trading rule given the pricing rule, and pricing rule given the order flow distribution.

## Strengths
- Clean closed-form result that grounded decades of empirical and theoretical microstructure work.
- Provides a microfoundation for permanent (informational) price impact.
- Lambda is directly estimable from regressions of returns on signed order flow.

## Weaknesses / Critiques
- Linear impact is empirically too steep at large sizes; metaorder data show concave (square-root) impact (see [[toth-2011-anomalous-impact]] and [[Square-Root Law]]).
- Single insider, exogenous noise flow, and Gaussian assumptions are stylized.
- No explicit treatment of temporary impact or transient liquidity dynamics.

## Relation to our work
- Cited in `paper/access.tex` as the foundational reference for linear price impact when motivating our square-root impact specification: "Kyle introduced the foundational model of linear price impact from informed order flow; subsequent work demonstrated that the aggregate impact of executed orders follows a concave, approximately square-root, relationship with order size."
- Our `PriceImpact.predict_vwap_impact_bps` in `backtest/portfolio/PriceImpact.py` replaces Kyle's linear `lambda * Q` with a `(2/3) * beta * sqrt(Q)` VWAP shape, motivated by the empirical departure from linearity at the order sizes typical of pump-and-dump entries against thin tiny-cap depth.

## Cited concepts
- [[Temporary vs Permanent Impact]]
- [[Square-Root Law]]
