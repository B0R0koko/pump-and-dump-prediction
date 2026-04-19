---
type: paper
title: "How Markets Slowly Digest Changes in Supply and Demand"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - market-microstructure
  - review
  - impact-model
status: summarized
related:
  - "[[Square-Root Law]]"
  - "[[Temporary vs Permanent Impact]]"
  - "[[VWAP Execution]]"
  - "[[Slippage]]"
  - "[[backtest-portfolio]]"
  - "[[almgren-chriss-2001-optimal-execution]]"
sources:
  - "https://arxiv.org/abs/0809.0822"
year: 2009
authors:
  - Jean-Philippe Bouchaud
  - J. Doyne Farmer
  - Fabrizio Lillo
venue: "Handbook of Financial Markets: Dynamics and Evolution (Hens & Schenk-Hoppé, eds.), Elsevier, ch. 2"
key_claim: "Market impact is concave (approximately square-root) in trade size, order flow is strongly autocorrelated, and the apparent contradiction is resolved by a propagator/transient-impact model in which each trade's impact decays over time."
methodology: "Review and synthesis of empirical regularities from order-book and trade-by-trade data across equity markets, integrated with the propagator model of Bouchaud, Gefen, Potters, Wyart."
contradicts: []
supports:
  - "[[Square-Root Law]]"
  - "[[Temporary vs Permanent Impact]]"
url: "https://arxiv.org/abs/0809.0822"
---

# How Markets Slowly Digest Changes in Supply and Demand (Bouchaud, Farmer, Lillo, 2009)

## TL;DR
A landmark review chapter that synthesizes a decade of empirical microstructure findings. The central puzzle: order flow (sign of market orders) is strongly long-memory autocorrelated, yet returns are nearly uncorrelated and price impact is concave in size. The authors resolve this with the propagator (transient impact) model: each trade exerts an impact that decays as a power law in time, and the decay exponent precisely cancels the autocorrelation of flow, making prices look diffusive at aggregate scales.

## Key claims
- Aggregate impact of metaorders is concave in size, approximately square-root.
- Sign of consecutive market orders is positively autocorrelated, with a slowly decaying power-law (Hurst-like) memory.
- Permanent impact exists but is small; most realized impact is transient and decays over minutes to hours.
- The propagator model `r_t = sum_s G(t - s) * epsilon_s + noise` reconciles concave impact, autocorrelated flow, and diffusive prices.
- Liquidity is dynamic: visible book depth is a small fraction of true latent supply revealed only as price moves.

## Methodology
- Synthesis of empirical results from prior work by the authors and others, mostly on Paris and London stock exchanges.
- Discussion organized around order-flow statistics, impact functions, propagator estimation, and implications for execution cost.

## Strengths
- Definitive review for the square-root law literature; widely cited entry point.
- Bridges theoretical (Kyle-style) and empirical (econophysics) microstructure traditions.
- Frames impact as an inherently dynamic, history-dependent phenomenon.

## Weaknesses / Critiques
- Equity-centric; pre-2009, before high-frequency electronification matured fully.
- Propagator model is phenomenological, not derived from agent optimisation.
- Cryptocurrency markets, fragmented and 24/7, were not in scope (see [[albers-2022-bitcoin-fragmentation]]).

## Relation to our work
- Cited in `paper/access.tex` when establishing the empirical foundation for our impact specification: "Bouchaud et al. surveyed the evidence for the square-root law across asset classes."
- Supports our use of `(2/3) * beta * sqrt(Q)` VWAP impact in `PriceImpact.predict_vwap_impact_bps` (`backtest/portfolio/PriceImpact.py`), and motivates our reliance on a single concave functional form even for very different asset universes (illiquid altcoins).

## Cited concepts
- [[Square-Root Law]]
- [[Temporary vs Permanent Impact]]
- [[Slippage]]
