---
type: paper
title: "Crypto Wash Trading"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - crypto
  - manipulation
  - volume-estimation
  - wash-trading
status: summarized
related:
  - "[[Binance]]"
  - "[[Pump-and-Dump Scheme]]"
  - "[[backtest-portfolio]]"
sources:
  - "https://pubsonline.informs.org/doi/10.1287/mnsc.2021.02709"
year: 2023
authors:
  - Lin William Cong
  - Xi Li
  - Ke Tang
  - Yang Yang
venue: "Management Science, vol. 69, no. 11, pp. 6427–6454"
key_claim: "Roughly 70 percent of reported BTC trading volume on unregulated centralized exchanges is wash trading; the inflation is detectable using statistical regularities (Benford's law, trade-size bunching, round-number clustering) that legitimate trading flows obey but fabricated flows violate."
methodology: "Combine forensic statistics (first-significant-digit / Benford's law tests, bunching estimators around round trade sizes, intra-trade-size distribution checks) with behavioral and structural cross-section regressions on transaction-level data from a large set of regulated and unregulated crypto exchanges."
contradicts: []
supports: []
url: "https://www.nber.org/papers/w26873"
---

# Crypto Wash Trading (Cong, Li, Tang, Yang, 2023)

## TL;DR
Quantifies the scale of fabricated volume on cryptocurrency exchanges. By applying classical forensic-statistics tools (Benford's law on transaction sizes, bunching tests around round numbers, distributional smoothness) to transaction-level data, the authors estimate that wash trading accounts for roughly 70 percent of reported BTC volume on unregulated centralized exchanges, while regulated venues show patterns consistent with organic flow. Documents structural correlates of wash-trading intensity (exchange age, listing fees, leverage features).

## Key claims
- Unregulated exchanges show systematic deviations from Benford's law, round-number bunching, and other distributional regularities that hold in regulated venue data.
- Implied wash-trading share on unregulated exchanges is on the order of 70 percent of reported volume.
- Wash trading correlates with exchange-level incentives (listing fees, ranking-site exposure, lower regulatory oversight) rather than with underlying coin fundamentals.
- Adjusting for wash trading materially changes cross-exchange volume rankings and apparent market-share statistics.

## Methodology
- Collect transaction-level trade data across a panel of regulated and unregulated crypto exchanges.
- Apply Benford's law tests on first significant digits of trade sizes.
- Use bunching estimators to quantify excess mass at round-number trade sizes.
- Regress estimated wash-trading intensity on exchange characteristics.

## Strengths
- Brings well-established forensic-finance tools into crypto.
- Cross-exchange identification: regulated venues serve as the control group.
- Quantitative magnitudes (70 percent) are headline-ready and have been widely cited in policy discussions.

## Weaknesses
- Detection methods can be evaded by adversaries who learn the test statistics (cat-and-mouse problem).
- Identification rests on the assumption that regulated-venue distributions represent honest flow.
- Limited to BTC and a snapshot of exchanges; extrapolation to altcoins and to current state is non-trivial.

## Relation to our work
- Cited in our paper (`paper/access.tex`, `\cite{cong_washtrading_2023}`, two locations) in two roles: (1) as adjacent crypto-manipulation literature in the related-work survey on broader market manipulation, and (2) as inspiration for wash-trading-aware microstructure features in our [[features]] design.
- Provides essential context for interpreting reported volumes on small-cap [[Binance]] pairs: if a non-trivial share of headline volume is fabricated, the calibration of our square-root impact model `I(Q) = beta * sqrt(Q_usdt)` (in [[backtest-portfolio]]) using `Y * sigma / sqrt(V)` likely understates true execution cost on the thinnest pumped coins.
- Motivates a robustness check: re-run portfolio simulation with adjusted (deflated) volume to bound the impact of inflated reported activity.

## Cited concepts
- [[Pump-and-Dump Scheme]]
- [[Slippage]]
- [[Binance]]
