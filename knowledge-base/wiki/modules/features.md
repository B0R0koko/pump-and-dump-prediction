---
type: module
title: "features"
created: 2026-04-19
updated: 2026-04-19
tags:
  - module
  - feature-engineering
status: seed
related:
  - "[[preprocessing]]"
  - "[[backtest-pipelines]]"
  - "[[Cross-Section]]"
sources: []
path: "features/"
language: python
purpose: "PumpsFeatureWriter computes features (asset returns, flow imbalance, slippage, etc.) at multiple time offsets per pump event."
maintainer: borokoko
last_updated: 2026-04-19
depends_on:
  - "[[core]]"
  - "[[preprocessing]]"
used_by:
  - "[[backtest-pipelines]]"
---

# features

## Purpose
Generates the feature matrix used for ranking. For every (pump event, asset, offset) triple, computes microstructure-derived features.

## Public surface
- `PumpsFeatureWriter`

## Time offsets
Range from 5 minutes up to 14 days relative to the pump event timestamp.

## Feature families
- Returns
- Flow imbalance
- Slippage
- (extend as we ingest)

## Design notes
- Multi-offset is intentional: short windows capture microstructure footprints, long windows capture pre-pump accumulation.

## Design rationale / References
- [[ntakaris-2020-midprice-prediction]] — comparable multi-offset feature-engineering approach for short-horizon market prediction; cited as inspiration in `paper/access.tex`.
- [[grinsztajn-2022-tree-tabular]] — justifies why our 70+ handcrafted features beat learned representations on this kind of tabular data.
- [[hamrick-2021-ecosystem]] — uses similar pre-pump volume features on the same problem domain.
- [[karbalaii-2025-microstructure]] — minute-OHLCV microstructure features comparable to ours.
