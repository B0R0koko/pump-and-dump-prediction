---
type: module
title: "backtest-utils"
created: 2026-04-19
updated: 2026-04-19
tags:
  - module
  - utils
  - backtest
status: seed
related:
  - "[[backtest-pipelines]]"
  - "[[backtest-portfolio]]"
sources: []
path: "backtest/utils/"
language: python
purpose: "Dataset management (build_dataset, sample), evaluation metrics (calculate_topk, calculate_topk_percent_auc), and robustness testing utilities."
maintainer: borokoko
last_updated: 2026-04-19
depends_on:
  - "[[core]]"
used_by:
  - "[[backtest-pipelines]]"
  - "[[backtest-portfolio]]"
---

# backtest-utils

## Purpose
Cross-cutting helpers for the backtest subsystem.

## Public surface
- `build_dataset`, `sample` — dataset construction
- `calculate_topk`, `calculate_topk_percent_auc` — evaluation metrics
- Robustness testing utilities

## Design notes
- top-k metrics align with the portfolio construction; using vanilla AUC would optimize a different objective than the deployed strategy.
