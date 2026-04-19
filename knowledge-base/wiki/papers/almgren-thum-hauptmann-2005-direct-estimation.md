---
type: paper
title: "Direct Estimation of Equity Market Impact"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - market-microstructure
  - impact-model
  - empirical
status: summarized
related:
  - "[[Square-Root Law]]"
  - "[[Temporary vs Permanent Impact]]"
  - "[[VWAP Execution]]"
  - "[[backtest-portfolio]]"
  - "[[almgren-chriss-2001-optimal-execution]]"
sources:
  - "https://www.ms.mff.cuni.cz/~jaroslav/cizek/cours_marche/almgren_thum_hauptmann_2005.pdf"
year: 2005
authors:
  - Robert Almgren
  - Chee Thum
  - Emmanuel Hauptmann
  - Hong Li
venue: "Risk, vol. 18, no. 7, pp. 58-62"
key_claim: "Permanent impact is approximately linear in trade size, while temporary impact follows a power law with exponent close to 0.6, intermediate between linear and square-root."
methodology: "Nonlinear regression of realized impact on order size, participation rate, daily volatility, and average daily volume across a large dataset of Citigroup institutional equity executions."
contradicts: []
supports:
  - "[[Square-Root Law]]"
url: "https://www.smallake.kr/wp-content/uploads/2017/03/almgren_2005.pdf"
---

# Direct Estimation of Equity Market Impact (Almgren, Thum, Hauptmann, Li, 2005)

## TL;DR
Empirical calibration of the Almgren-Chriss impact framework on a large proprietary dataset of US equity orders executed by Citigroup. By regressing realized post-trade and slippage costs against trade and stock characteristics, the authors estimate functional forms and coefficients for permanent and temporary impact. They find permanent impact roughly linear in size, but temporary impact best fit by a power law with exponent around 0.6, supporting concave (sub-linear) impact and providing one of the most-cited empirical impact-model calibrations.

## Key claims
- Permanent impact `g(v)` is approximately proportional to size, scaled by `sigma * (X / V)` (volatility times participation).
- Temporary impact `h(v)` scales as `(v / V_daily)^beta` with beta ~ 0.6; not the 1.0 of the linear Almgren-Chriss baseline nor the 0.5 of a strict square-root law.
- Universal cross-sectional scaling: once impact is normalized by daily volatility and participation rate, residual stock-specific variation is small.
- Coefficients enable plug-in pre-trade cost estimates for institutional execution.

## Methodology
- Dataset: thousands of Citigroup parent orders in US equities, with detailed child-order timestamps and prices.
- Define realized impact relative to arrival price (temporary) and post-execution decay price (permanent).
- Fit nonlinear regression with stock-level normalisations (`sigma`, `V_daily`).
- Robustness checks across capitalisation buckets, time periods, and trade urgency.

## Strengths
- Rare published calibration on real institutional execution data.
- Establishes power-law exponent empirically rather than assuming it.
- Provides numerical benchmarks (still widely cited) for industry transaction-cost models.

## Weaknesses / Critiques
- Equity-only, US-only, single broker; generalizability to crypto / tiny-caps untested.
- Exponent 0.6 sits between common theoretical predictions (0.5 square-root, 1.0 linear); later work (e.g. [[toth-2011-anomalous-impact]]) argues for closer-to-0.5 universally.
- Functional-form choices (separable permanent + temporary) baked in.

## Relation to our work
- Cited in `paper/access.tex` when motivating the square-root impact specification: "Almgren et al. provided direct empirical estimates of the power-law exponent (~0.6) using institutional equity data."
- Justifies our choice of a sub-linear `beta * sqrt(Q)` form in `PriceImpact.predict_vwap_impact_bps` (`backtest/portfolio/PriceImpact.py`), which is a cleaner exponent-0.5 specialization more conservative than Almgren et al.'s 0.6 fit.

## Cited concepts
- [[Square-Root Law]]
- [[Temporary vs Permanent Impact]]
- [[VWAP Execution]]
