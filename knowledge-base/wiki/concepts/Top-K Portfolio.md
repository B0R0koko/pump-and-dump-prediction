---
type: concept
title: "Top-K Portfolio"
created: 2026-04-19
updated: 2026-04-19
tags:
  - concept
  - portfolio
  - evaluation
status: seed
related:
  - "[[Top-K AUC]]"
  - "[[Cross-Section]]"
  - "[[backtest-portfolio]]"
---

# Top-K Portfolio

## Definition
A portfolio construction rule that, at each pump event, selects the top-k assets by model score from the current [[Cross-Section]] and enters long positions in all of them with equal weight. No shorting; no position sizing beyond equal weight per selected asset.

## Why it matters here
`TOPKPortfolio` in [[backtest-portfolio]] is the primary evaluation harness for the project. It takes model predictions from [[backtest-pipelines]], selects the top-k ranked candidates, estimates execution cost via the [[Square-Root Law]] impact model, and computes PnL per event.

## Connection to evaluation metrics
[[Top-K AUC]] is the training-time proxy that the top-k portfolio optimizes at test time. High Top-K AUC during validation predicts that the portfolio will find the pump target in its top-k selections most of the time.

## Key parameters
- `portfolio_size` (k): typically 1 to 5 in the project experiments.
- `use_price_impact`: when `True`, execution cost is deducted via VWAP price adjustment. When `False`, only the flat 25 bps exchange fee is deducted.
- Notional sizing: fixed USD or fixed quote-currency notional per position.
