---
type: domain
title: "Concepts Index"
created: 2026-04-19
updated: 2026-04-19
tags:
  - meta
  - index
  - concepts
status: developing
subdomain_of: ""
page_count: 12
---

# Concepts

Definitions, frameworks, and patterns: anything that needs a single canonical page so other notes can link `[[concept]]` instead of redefining.

## Pages

- [[Cross-Section]] — per-event ranking framing used by all models
- [[Flow Imbalance]] — signed bid-minus-ask volume fraction; pre-pump accumulation signal
- [[Kyle's Lambda]] — linear price-impact coefficient from Kyle 1985; illiquidity measure
- [[Pre-Pump Accumulation]] — insider buying footprint in the hours before T0
- [[Pump Announcement Window]] — the canonical t=0 of the event sequence
- [[Pump-and-Dump Scheme]] — the manipulation pattern itself
- [[Slippage]] — realized minus expected price; combines spread, impact, adverse selection
- [[Square-Root Law]] — `I(Q) = beta * sqrt(Q)` impact form used in `PriceImpact.py`
- [[Temporary vs Permanent Impact]] — Almgren-Chriss decomposition of execution cost
- [[Top-K AUC]] — partial AUC restricted to the top-k prefix; matches the portfolio's operating region
- [[Top-K Portfolio]] — top-k asset selection per cross-section; the project's main evaluation harness
- [[VWAP Execution]] — volume-weighted benchmark; `(2/3) * beta * sqrt(Q)` closed-form impact
