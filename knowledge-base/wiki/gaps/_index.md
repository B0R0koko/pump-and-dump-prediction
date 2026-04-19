---
type: domain
title: "Gaps Index"
created: 2026-04-19
updated: 2026-04-19
tags:
  - meta
  - index
  - gaps
status: developing
subdomain_of: ""
page_count: 2
---

# Gaps

Open questions, contradictions between sources, things to investigate.

## Filed pages

- [[sqrt-coefficient-lowcap-binance]] — what is `Y` (or `beta`) for sub-$1M ADV Binance pairs?
- [[post-2022-binance-corpus]] — no public labelled Binance pump corpus exists past 2021; our `test > 2021-05-01` split has no out-of-sample data aligned with the post-FTX regime.

## Open questions (inline, ungrown)

- **Cross-venue feature transfer**: Are features trained on Binance `SYM/BTC` valid for USDT pairs and for KuCoin/MEXC/LATOKEN/Poloniex? Bolz et al. 2024 use the latter venues and report significantly weaker TOP5 ranking accuracy than La Morgia on Binance.
- **Cancelled/failed pumps**: Hamrick et al. report many announced pumps that never executed. Our `PumpEvent` records implicitly assume execution; label hygiene needs an explicit "intent vs execution" distinction.
- **Organiser-wallet leading signal**: Perseus 2025 (arXiv:2503.01686, not yet ingested) claims to identify mastermind wallets. Could provide a true pre-announcement signal.

> [!gap] Larger gaps to file as standalone pages later
> "Are coordinated Telegram pumps still the dominant pattern post-2022?", "How does our impact model compare to Almgren-Chriss?", "Does the time split still hold given regime shifts?".
