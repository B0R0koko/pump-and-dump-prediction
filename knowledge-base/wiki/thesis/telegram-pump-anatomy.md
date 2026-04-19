---
type: thesis
title: "Telegram Pump Anatomy"
created: 2026-04-19
updated: 2026-04-19
tags:
  - thesis
  - synthesis
  - manipulation
  - telegram
status: synthesized
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Cross-Section]]"
  - "[[Telegram Pump Groups]]"
  - "[[Binance]]"
  - "[[xu-2019-anatomy]]"
  - "[[lamorgia-2023-doge]]"
  - "[[hamrick-2021-ecosystem]]"
  - "[[bolz-2024-bertweet-zscore]]"
  - "[[karbalaii-2025-microstructure]]"
sources:
  - "https://arxiv.org/abs/1811.10109"
  - "https://arxiv.org/abs/2105.00733"
  - "https://www.sciencedirect.com/science/article/abs/pii/S0306457321000169"
  - "https://arxiv.org/html/2504.15790"
  - "https://arxiv.org/html/2412.18848v2"
  - "https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset"
  - "https://github.com/xujiahuayz/pumpdump"
---

# Telegram Pump Anatomy

A working synthesis of how Telegram-coordinated cryptocurrency pumps are organised, what footprints they leave, and which datasets exist to study them. This page anchors the literature for the `pumps_and_dumps` codebase.

## Overview
Telegram (and to a lesser extent Discord) is the primary coordination substrate for retail cryptocurrency pump-and-dump schemes targeting small-cap pairs on centralised exchanges, principally [[Binance]]. The phenomenon has a stable, three-phase lifecycle that produces detectable microstructure footprints both before and during the event. A small number of public datasets, all harvested from these channels, dominate the academic literature: [[xu-2019-anatomy|Xu and Livshits 2019]], [[hamrick-2021-ecosystem|Hamrick et al. 2021]] (the "Pump-and-Dumpster"), [[lamorgia-2023-doge|La Morgia et al. 2020/2023]], and a growing set of newer corpora ([[bolz-2024-bertweet-zscore|Bolz et al. 2024]], [[karbalaii-2025-microstructure|Karbalaii 2025]]).

## Lifecycle

### Accumulation (T-days to T-minutes)
- Organisers and tier-1 (VIP) members quietly accumulate positions in the chosen target.
- [[karbalaii-2025-microstructure|Karbalaii 2025]] finds detectable accumulation in 69.3% of 485 examined events; mean span ~36 hours, with the bulk of pre-event volume (~70%) compressed into the final hour before announcement. Confidence: medium (single recent study, single venue).
- Two archetypes: **pre-accumulated** (visible volume spike pre-T0) and **on-the-spot** (no detectable lead).

### Pump Announcement Window (T-hours to T0)
- Pre-announcement messages declare the exchange, the precise UTC start time, and the tiering rules (FFA vs Ranked).
- Countdown messages intensify in the minutes before T0.
- At T0, the ticker is revealed (text or image). Ranked tiers receive it seconds earlier; this asymmetry is the organisers' edge over latecomers.

### Pump (T0 + seconds to minutes)
- Aggressive market buys; price often rises 50%–500% in under a minute.
- La Morgia et al. detect active pumps in per-second Binance candles at 94.5% F1 within ~25 seconds [[lamorgia-2023-doge]]. Confidence: high.

### Dump (T0 + minutes)
- Organisers and earliest followers sell into late demand.
- Price typically reverts to or below pre-pump baseline within 5–20 minutes.
- Profit distribution is sharply skewed: insiders median >100%, latecomers strongly negative ([[karbalaii-2025-microstructure|Karbalaii 2025]]; [[hamrick-2021-ecosystem|Hamrick et al. 2021]]). Confidence: high.

## Pre-Announcement Signals
- **Volume**: [[karbalaii-2025-microstructure|Karbalaii 2025]] quantifies a ~14x volume increase during pump hours vs pre-pump baseline; in the last hour pre-T0 a measurable bid-side imbalance often appears. Confidence: medium-high.
- **Order book**: [[bolz-2024-bertweet-zscore|Bolz et al. 2024]] use bid-ask spread, imbalance ratios, taker-side volume, and order pressure as Z-score features over 20-second windows pre-event; TOP5 ranking accuracy degrades sharply beyond 60s, indicating signals are late-stage. Confidence: medium.
- **Coin selection priors**: [[xu-2019-anatomy|Xu and Livshits 2019]] show coin-level features (low market cap, low daily volume, recent low volatility, listing age, single-exchange concentration) predict which asset will be picked even before any pre-pump trading. Confidence: high.
- **Cross-section ranking**: at any candidate T0, all eligible small-cap pairs form a [[Cross-Section]] and the model ranks them. This is the framing used in the project's [[backtest-pipelines]].

## Datasets Available
| Name | Authors | Window | Events | Channels | Public |
|---|---|---|---|---|---|
| `pumpdump` | [[xu-2019-anatomy|Xu & Livshits 2019]] | Jun 2018 – Feb 2019 | 412 | Telegram | Yes (GitHub) |
| `pump-and-dump-dataset` | La Morgia et al. 2020 | 2018–2020 | hundreds, `SYM/BTC` Binance | Telegram | Yes (GitHub) |
| Doge of Wall Street | [[lamorgia-2023-doge|La Morgia et al. 2023]] | 2018–2021+ | ~900 | 20+ Telegram + Discord | Yes (GitHub) |
| Pump-and-Dumpster | [[hamrick-2021-ecosystem|Hamrick et al. 2021]] | Jan–Jul 2018 | 3,767 Telegram + 1,051 Discord | hundreds, ~300 coins | Partial (SSRN supplementary) |
| Bolz et al. | [[bolz-2024-bertweet-zscore|Bolz et al. 2024]] | 2017–2024 | 2,079 | 43 Telegram | Pending |
| Karbalaii minute-OHLCV | [[karbalaii-2025-microstructure|Karbalaii 2025]] | Aug 2024 – Feb 2025 | 485 (Poloniex) | cross-validated | Described in paper |

Confidence: high on first three releases; medium on Hamrick partial-availability and Bolz pending-release.

## Key Entities
- [[Telegram Pump Groups]] — the social substrate.
- [[Binance]] — dominant venue.

## Key Concepts
- [[Pump-and-Dump Scheme]] — the underlying pattern.
- [[Cross-Section]] — the per-event ranking unit used in our pipelines.
- [[Pre-Pump Accumulation]] — the insider-buying footprint (entity/concept stub; expand later).
- [[Pump Announcement Window]] — the pre-T0 to T0+1min window where signals are most concentrated (stub).

## Open Questions
- Is there a public, post-2022 Telegram pump corpus that captures the post-FTX market structure shift? Most public corpora end around 2021.
- How transferable are features learned on Binance `SYM/BTC` to USDT pairs and to non-Binance venues (KuCoin, MEXC, LATOKEN, Poloniex)?
- What fraction of announced pumps are cancelled or fail to execute, and how should label hygiene handle them?
- Can we identify organiser wallets via on-chain accumulation patterns and use them as a leading signal? (Perseus 2025 work, not yet ingested.)

## Sources
- [[xu-2019-anatomy|Xu & Livshits 2019]], *The Anatomy of a Cryptocurrency Pump-and-Dump Scheme*, USENIX Security. arXiv:1811.10109.
- [[lamorgia-2023-doge|La Morgia, Mei, Sassi, Stefa 2023]], *The Doge of Wall Street*, ACM TOIT 23(1). arXiv:2105.00733.
- La Morgia et al. 2020, *Pump and Dumps in the Bitcoin Era*, ICCCN. arXiv:2005.06610.
- [[hamrick-2021-ecosystem|Hamrick, Rouhi, Mukherjee, Feder, Gandal, Moore, Vasek 2021]], *An examination of the cryptocurrency pump-and-dump ecosystem*, Information Processing & Management.
- [[bolz-2024-bertweet-zscore|Bolz et al. 2024/2025]], *Real-Time ML Detection of Telegram-Based Pump-and-Dump Schemes*, ACM CCS DeFi Workshop. arXiv:2412.18848.
- [[karbalaii-2025-microstructure|Karbalaii 2025]], *Microstructure and Manipulation: Quantifying Pump-and-Dump Dynamics*, arXiv:2504.15790.
