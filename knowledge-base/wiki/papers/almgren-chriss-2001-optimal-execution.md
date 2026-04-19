---
type: paper
title: "Optimal Execution of Portfolio Transactions"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - market-microstructure
  - execution
  - impact-model
status: summarized
related:
  - "[[Square-Root Law]]"
  - "[[Temporary vs Permanent Impact]]"
  - "[[VWAP Execution]]"
  - "[[backtest-portfolio]]"
sources:
  - "https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf"
year: 2001
authors:
  - Robert Almgren
  - Neil Chriss
venue: "Journal of Risk, vol. 3"
key_claim: "Optimal liquidation trades off volatility risk against expected impact cost; under linear temporary and permanent impact, optimal trajectories are hyperbolic-sine in time."
methodology: "Continuous-time mean-variance optimization with linear permanent impact g(v)=gamma*v and linear temporary impact h(v)=epsilon*sgn(v)+eta*v."
contradicts: []
supports:
  - "[[Square-Root Law]]"
url: "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=53501"
---

# Optimal Execution of Portfolio Transactions (Almgren & Chriss, 2001)

## TL;DR
Foundational model of optimal trade scheduling. Splits market impact into permanent (informational) and temporary (liquidity-cost) components. With linear specifications, the cost-risk efficient frontier is closed-form: the optimal trajectory decays exponentially (hyperbolic sine) from initial position to zero over the horizon, with curvature controlled by a risk-aversion parameter `lambda`.

## Key claims
- Total cost of liquidating a position decomposes into permanent impact (price drift) and temporary impact (slippage paid on each child order).
- Linear temporary impact `h(v) = eps*sgn(v) + eta*v` and linear permanent impact `g(v) = gamma*v` yield analytic solutions.
- For risk-neutral traders, VWAP (linear schedule) is optimal. Risk-averse traders trade faster than VWAP.
- Coefficient `eta` (temporary impact slope) scales roughly with bid-ask spread / typical depth; `gamma` (permanent) scales with daily volume.

## Methodology
- Discretize horizon into N slices; trade `n_k` shares per slice.
- Price dynamics: `S_k = S_{k-1} + sigma*sqrt(tau)*xi_k - tau*g(n_k/tau)`. Execution price: `S_k - h(n_k/tau)`.
- Minimize `E[cost] + lambda*Var[cost]`.

## Strengths
- Closed form, easy to calibrate, dominant baseline in industry execution algos.
- Cleanly separates risk and cost.

## Weaknesses / Critiques
- Linear impact is empirically wrong at large sizes: square-root law dominates beyond ~0.1% participation. See [[Square-Root Law]] and Bouchaud et al. 2010 (arXiv:0812.2010).
- Constant volatility, no autocorrelation in flow, no stochastic depth.
- Calibration of `eta` and `gamma` requires proprietary execution data; published equity values use ADV and daily-vol normalisations (see Almgren-Thum-Hauptmann 2005, which finds beta~0.6 closer to 1/2).

## Relation to our work
- Our `PriceImpact.py` ([[backtest-portfolio]]) uses `I(Q) = beta*sqrt(Q_usdt)`, i.e. the [[Square-Root Law]] rather than Almgren-Chriss linear form, because at pump-event sizes vs tiny-cap depth we are well in the non-linear regime.
- Almgren-Chriss is still the right framework for *scheduling* once an impact model is chosen. We currently assume terminal VWAP execution within a single bar; an Almgren-Chriss-style schedule is a natural extension.

## Cited concepts
- [[Temporary vs Permanent Impact]]
- [[Square-Root Law]]
- [[VWAP Execution]]
