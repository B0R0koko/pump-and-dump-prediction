---
type: concept
title: "Pre-Pump Accumulation"
created: 2026-04-19
updated: 2026-04-19
tags:
  - concept
  - manipulation
  - signals
status: seed
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Telegram Pump Groups]]"
  - "[[telegram-pump-anatomy]]"
sources:
  - "https://arxiv.org/html/2504.15790"
  - "https://arxiv.org/abs/1811.10109"
complexity: basic
domain: market-microstructure
aliases:
  - "Insider Accumulation"
  - "Pre-Event Accumulation"
---

# Pre-Pump Accumulation

## Definition
Pre-Pump Accumulation is organizer (and tier-1/VIP member) purchasing of the target asset in the hours-to-days before the announcement. On low-cap pairs it is typically detectable as anomalous volume relative to the asset's pre-pump baseline, often with a bid-side imbalance in the final hour.

## Why it matters
Karbalaii 2025 reports detectable accumulation in 69.3% of 485 examined Poloniex events, with mean span ~36 hours and ~70% of pre-event volume compressed into the final hour before T0. Two archetypes emerge: pre-accumulated (visible volume spike) and on-the-spot (no detectable lead). Accumulation is the strongest pre-announcement signal because it shifts the candidate distribution before the channel knows the symbol.

## How it appears in this project
- Volume and flow-imbalance features over the multi-offset windows in `features/PumpsFeatureWriter` are designed to surface accumulation footprints.
- [[Cross-Section]] ranking exploits pre-T0 imbalance asymmetries between the target and other small-cap pairs.

## Related
- [[Pump-and-Dump Scheme]]
- [[Telegram Pump Groups]]
- [[telegram-pump-anatomy]]

## Sources
- Karbalaii 2025, *Microstructure and Manipulation: Quantifying Pump-and-Dump Dynamics*, arXiv:2504.15790.
- Xu & Livshits 2019, *The Anatomy of a Cryptocurrency Pump-and-Dump Scheme*, arXiv:1811.10109.
