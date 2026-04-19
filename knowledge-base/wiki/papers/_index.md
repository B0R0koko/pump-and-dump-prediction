---
type: domain
title: "Papers Index"
created: 2026-04-19
updated: 2026-04-19
tags:
  - meta
  - index
  - papers
status: developing
subdomain_of: ""
page_count: 31
---

# Papers

Summaries of academic papers and preprints relevant to crypto market manipulation, pump-and-dump detection, market microstructure, ranking ML, and class-imbalance methods. Pages tagged with `paper`. Items marked **★** are cited in our paper at `paper/access.tex`.

## Pages

### Pump-and-dump detection
- ★ [[kamps-bennett-2018-to-the-moon]] — Kamps & Bennett 2018, foundational P&D definitions + rule-based detection (Crime Sci)
- ★ [[xu-2019-anatomy]] — Xu & Livshits 2019, foundational Telegram pump dataset and predictive model (USENIX Security)
- ★ [[lamorgia-2020-icccn-realtime]] — La Morgia, Mei, Sassi, Stefa 2020, predecessor to Doge of Wall Street, per-second real-time RF on Binance (ICCCN)
- ★ [[nghiem-2021-market-social-signals]] — Nghiem et al. 2021, market + social-signal LSTM/CNN for max-price-move prediction (Expert Sys Apps)
- [[hamrick-2021-ecosystem]] — Hamrick et al. 2021, Pump-and-Dumpster corpus, median return only 4.1–7.7% diminishing over time (IPM)
- ★ [[chadalapaka-2022-deep-learning]] — Chadalapaka et al. 2022, deep learning for P&D detection (arXiv:2205.04646)
- ★ [[fantazzini-xiao-2023-imbalanced]] — Fantazzini & Xiao 2023, SMOTE + RF on 2021–2022 Binance pumps; closest prior to our SMOTE pipeline (Econometrics)
- ★ [[lamorgia-2023-doge]] — La Morgia et al. 2023, real-time per-second pump detection on Binance, ~900 events (ACM TOIT)
- ★ [[hu-2023-sequence-target-prediction]] — Hu et al. 2023, sequence-based neural network with channel-conditioned attention, AUC 0.943 (SIGMOD)
- [[bolz-2024-bertweet-zscore]] — Bolz et al. 2024, BERTweet + Z-score on 5 secondary venues, TOP5=55.81% at 20s (arXiv:2412.18848)
- [[karbalaii-2025-microstructure]] — Karbalaii 2025, minute-OHLCV pre-pump quantification on Poloniex, 485 events (arXiv:2504.15790)

### Market microstructure / impact
- ★ [[kyle-1985-continuous-auctions]] — Kyle 1985, continuous-auction insider model, "Kyle's lambda" linear-impact coefficient (Econometrica)
- ★ [[almgren-chriss-2001-optimal-execution]] — foundational linear-impact + scheduling, J. Risk 2001
- ★ [[almgren-thum-hauptmann-2005-direct-estimation]] — Almgren et al. 2005, empirical equity-impact calibration, exponent ≈ 0.6 (Risk)
- ★ [[bouchaud-farmer-lillo-2009-markets-digest]] — Bouchaud, Farmer, Lillo 2009, market-impact review chapter (Handbook of Financial Markets)
- ★ [[toth-2011-anomalous-impact]] — Tóth et al. 2011, square-root law confirmed on equity meta-orders, latent-order-book interpretation (Phys. Rev. X)
- ★ [[donier-bonart-2015-bitcoin-metaorder]] — square-root law confirmed on Bitcoin, ~1M metaorders
- ★ [[albers-2022-bitcoin-fragmentation]] — Albers et al. 2022, fragmentation, price formation and cross-impact in Bitcoin markets (Appl. Math. Finance)

### Ranking / cross-sectional ML
- ★ [[poh-2020-ltr-cross-sectional]] — Poh et al. 2020, LTR on cross-sectional momentum, ~3x Sharpe vs regression baselines (J. Financial Data Sci)
- ★ [[catboost-docs-ranking-objectives]] — CatBoost ranking-loss reference (PairLogit, YetiRank, QuerySoftMax, QueryAUC) folded with Lyzhin et al. ICML 2023
- ★ [[ntakaris-2020-midprice-prediction]] — Ntakaris et al. 2020, ML mid-price prediction with technical/quantitative indicators on FI-2010 (PLOS ONE)

### Tree-ensemble & boosting foundations
- ★ [[breiman-2001-random-forest]] — Breiman 2001, foundational random-forest algorithm (Mach. Learn.)
- ★ [[catboost-prokhorenkova-2018-original]] — Prokhorenkova et al. 2018, original CatBoost ordered-boosting paper (NeurIPS)
- ★ [[grinsztajn-2022-tree-tabular]] — Grinsztajn et al. 2022, why trees still beat DL on tabular data (NeurIPS)

### Class imbalance / cost-sensitive learning
- ★ [[chawla-2002-smote]] — Chawla et al. 2002, foundational SMOTE oversampling (JAIR)
- ★ [[blagus-lusa-2013-smote-highdim]] — Blagus & Lusa 2013, SMOTE degrades performance in high-dimensional sparse settings (BMC Bioinformatics)
- ★ [[elkan-2001-cost-sensitive]] — Elkan 2001, foundations of cost-sensitive learning (IJCAI)
- ★ [[lin-2017-focal-loss]] — Lin et al. 2017, focal loss for extreme class imbalance (ICCV)
- ★ [[liu-ting-zhou-2008-isolation-forest]] — Liu et al. 2008, isolation forest for unsupervised anomaly detection (ICDM)

### Tooling
- ★ [[akiba-2019-optuna]] — Akiba et al. 2019, Optuna define-by-run hyperparameter optimization framework (KDD)

### Crypto market context
- ★ [[cong-2023-wash-trading]] — Cong et al. 2023, ~70% of unregulated-exchange BTC volume is wash trading (Manage. Sci.)

> [!gap] Still to ingest
> Said 2022 (arXiv:2205.07385) — square-root law review. Donier-Bonart-Mastromatteo-Bouchaud 2015 (arXiv:1412.0141) — fully consistent non-linear model. Easley-López-de-Prado-O'Hara 2024 (SSRN 4814346) — crypto microstructure. Perseus 2025 (arXiv:2503.01686) — mastermind-wallet identification on Telegram.
