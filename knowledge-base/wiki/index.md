---
type: meta
title: "Wiki Index"
created: 2026-04-19
updated: 2026-04-19
tags:
  - meta
  - index
status: developing
---

# Wiki Index

Master catalog of all pages. Update on every page create / rename / delete.

## Top-level

- [[overview]] — executive summary of the wiki

## Code (Mode B)

- [[modules/_index|Modules]] — major code packages
  - [[core]], [[market_data]], [[preprocessing]], [[features]]
  - [[backtest-pipelines]], [[backtest-portfolio]], [[backtest-utils]]

## Research (Mode E)

### Papers — pump-and-dump detection
- [[kamps-bennett-2018-to-the-moon]] — foundational P&D definitions (Crime Sci 2018)
- [[xu-2019-anatomy]] — Telegram pump dataset and predictive model (USENIX Security 2019)
- [[lamorgia-2020-icccn-realtime]] — predecessor to Doge of Wall Street, real-time RF on Binance (ICCCN 2020)
- [[nghiem-2021-market-social-signals]] — market + social-signal LSTM/CNN (Expert Sys Apps 2021)
- [[hamrick-2021-ecosystem]] — Pump-and-Dumpster corpus, pump returns 4–8% diminishing (IPM 2021)
- [[chadalapaka-2022-deep-learning]] — DL for P&D detection (arXiv 2022)
- [[fantazzini-xiao-2023-imbalanced]] — SMOTE + RF on 2021–2022 Binance pumps (Econometrics 2023)
- [[lamorgia-2023-doge]] — real-time per-second pump detection on Binance (ACM TOIT 2023)
- [[hu-2023-sequence-target-prediction]] — channel-conditioned sequence model, AUC 0.943 (SIGMOD 2023)
- [[bolz-2024-bertweet-zscore]] — BERTweet + Z-score on 5 secondary venues (arXiv 2024)
- [[karbalaii-2025-microstructure]] — minute-OHLCV pre-pump quantification on Poloniex (arXiv 2025)

### Papers — market microstructure / impact
- [[kyle-1985-continuous-auctions]] — "Kyle's lambda" linear-impact insider model (Econometrica 1985)
- [[almgren-chriss-2001-optimal-execution]] — optimal execution + linear impact (J. Risk 2001)
- [[almgren-thum-hauptmann-2005-direct-estimation]] — empirical equity-impact, exponent ≈ 0.6 (Risk 2005)
- [[bouchaud-farmer-lillo-2009-markets-digest]] — market-impact review (Handbook of Financial Markets 2009)
- [[toth-2011-anomalous-impact]] — square-root law on equity meta-orders, latent OB (Phys. Rev. X 2011)
- [[donier-bonart-2015-bitcoin-metaorder]] — square-root law on Bitcoin (Market Microstructure & Liquidity 2015)
- [[albers-2022-bitcoin-fragmentation]] — fragmentation and cross-impact in Bitcoin (Appl. Math. Finance 2022)

### Papers — ranking ML / cross-section
- [[poh-2020-ltr-cross-sectional]] — cross-sectional LTR on momentum (J. Financial Data Sci 2020)
- [[catboost-docs-ranking-objectives]] — CatBoost ranking losses + Lyzhin ICML 2023
- [[ntakaris-2020-midprice-prediction]] — ML mid-price prediction with technical/quant indicators (PLOS ONE 2020)

### Papers — tree-ensemble & boosting foundations
- [[breiman-2001-random-forest]] — foundational random-forest algorithm (Mach. Learn. 2001)
- [[catboost-prokhorenkova-2018-original]] — original CatBoost ordered-boosting paper (NeurIPS 2018)
- [[grinsztajn-2022-tree-tabular]] — why trees still beat DL on tabular data (NeurIPS 2022)

### Papers — class imbalance / cost-sensitive
- [[chawla-2002-smote]] — foundational SMOTE oversampling (JAIR 2002)
- [[blagus-lusa-2013-smote-highdim]] — SMOTE degrades in high-dim (BMC Bioinformatics 2013)
- [[elkan-2001-cost-sensitive]] — foundations of cost-sensitive learning (IJCAI 2001)
- [[lin-2017-focal-loss]] — focal loss for extreme class imbalance (ICCV 2017)
- [[liu-ting-zhou-2008-isolation-forest]] — isolation forest, unsupervised anomaly detection (ICDM 2008)

### Papers — tooling & crypto context
- [[akiba-2019-optuna]] — Optuna hyperparameter optimization framework (KDD 2019)
- [[cong-2023-wash-trading]] — ~70% of unregulated-exchange BTC volume is wash trading (Manage. Sci. 2023)

### Concepts
- [[concepts/_index|Concepts Index]]
- [[Pump-and-Dump Scheme]], [[Cross-Section]]
- [[Pump Announcement Window]], [[Pre-Pump Accumulation]]
- [[Top-K AUC]], [[Square-Root Law]], [[Temporary vs Permanent Impact]], [[VWAP Execution]], [[Slippage]]
- [[Flow Imbalance]], [[Top-K Portfolio]], [[Kyle's Lambda]]

### Entities
- [[entities/_index|Entities Index]]
- [[Binance]], [[Telegram Pump Groups]]

### Thesis (synthesis pages)
- [[thesis/_index|Thesis Index]]
- [[state-of-detection-2018-2025]] — survey synthesis: methods, datasets, contradictions
- [[telegram-pump-anatomy]] — lifecycle, pre-pump signals, public datasets
- [[ranking-for-event-prediction]] — why ranking beats classification, objective recommendations
- [[impact-models-for-lowcap-crypto]] — Almgren-Chriss vs square-root, parameterization for our simulator
- [[code_audit_findings]] — skeptical audit of backtest/ subtree: 1 critical, 2 high, 5 medium, 5 low bugs (IEEE Access R1)

### Gaps
- [[gaps/_index|Gaps Index]]
- [[sqrt-coefficient-lowcap-binance]] — empirical Y for sub-$1M ADV Binance pairs
- [[post-2022-binance-corpus]] — no public labelled Binance pump corpus past 2021
