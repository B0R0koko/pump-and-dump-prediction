---
type: concept
title: "Flow Imbalance"
created: 2026-04-19
updated: 2026-04-19
tags:
  - concept
  - microstructure
  - feature
status: seed
related:
  - "[[Slippage]]"
  - "[[Pre-Pump Accumulation]]"
  - "[[features]]"
  - "[[albers-2022-bitcoin-fragmentation]]"
---

# Flow Imbalance

## Definition
Signed bid-minus-ask volume fraction over a time window. Formally, if `V_buy` is buyer-initiated volume and `V_sell` is seller-initiated volume over the window, then:

```
flow_imbalance = (V_buy - V_sell) / (V_buy + V_sell)
```

Values near +1 indicate aggressive buying pressure; values near -1 indicate aggressive selling pressure; values near 0 indicate balanced two-sided flow.

## Why it matters here
Flow imbalance is one of the core features computed by [[features]] at multiple time offsets (5min to 14 days) per pump event. It captures the insider accumulation footprint described in [[Pre-Pump Accumulation]]: before a pump, tier-1 members quietly build positions, tilting order flow toward the bid side before the public announcement.

## Relation to related concepts
- [[Slippage]] is downstream of flow imbalance: one-sided flow depletes liquidity and raises transaction costs.
- [[albers-2022-bitcoin-fragmentation]] studies cross-venue flow imbalance and cross-impact in Bitcoin microstructure.
- The [[Square-Root Law]] and [[VWAP Execution]] models assume flow arrives smoothly; a large pre-pump flow imbalance spike violates this assumption.
