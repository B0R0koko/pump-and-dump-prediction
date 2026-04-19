---
type: thesis
title: "State of Cryptocurrency Pump-and-Dump Detection 2018 to 2025"
created: 2026-04-19
updated: 2026-04-19
tags:
  - thesis
  - synthesis
  - manipulation
  - survey
status: synthesized
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Cross-Section]]"
  - "[[Binance]]"
  - "[[xu-2019-anatomy]]"
  - "[[lamorgia-2020-icccn-realtime]]"
  - "[[lamorgia-2023-doge]]"
  - "[[hu-2023-sequence-target-prediction]]"
  - "[[hamrick-2021-ecosystem]]"
  - "[[chadalapaka-2022-deep-learning]]"
  - "[[bolz-2024-bertweet-zscore]]"
  - "[[karbalaii-2025-microstructure]]"
  - "[[nghiem-2021-market-social-signals]]"
  - "[[telegram-pump-anatomy]]"
  - "[[ranking-for-event-prediction]]"
  - "[[impact-models-for-lowcap-crypto]]"
sources:
  - "https://arxiv.org/abs/1811.10109"
  - "https://arxiv.org/abs/2105.00733"
  - "https://arxiv.org/abs/2204.12929"
  - "https://link.springer.com/article/10.1186/s40163-018-0093-5"
  - "https://www.mdpi.com/1999-5903/15/8/267"
  - "https://arxiv.org/abs/2003.06551"
  - "https://arxiv.org/abs/2412.18848"
  - "https://arxiv.org/abs/2503.01686"
  - "https://tylermoore.utulsa.edu/weis19pump.pdf"
confidence: high
period: "2018 to 2025"
---

# State of Cryptocurrency Pump-and-Dump Detection 2018 to 2025

## Overview
Academic work on cryptocurrency pump-and-dump (P&D) detection has matured from anomaly thresholding on volume in 2018 to graph-neural-network-based mastermind tracing by 2025. The literature now splits into three distinct tasks, each with its own benchmarks and best-in-class methods. This page synthesizes seven years of work across 11 detection papers now in the vault, framing the `pumps_and_dumps` project.

## Method Families

| Family | Goal | Representative work | Strengths | Limits |
|---|---|---|---|---|
| Anomaly thresholding | Flag unusual volume or price moves after the fact | [[kamps-bennett-2018-to-the-moon]] | Unsupervised, no labels needed | High false positive rate, slow |
| Supervised real-time detection | Classify rolling time chunks as pump or not | [[lamorgia-2020-icccn-realtime]], [[lamorgia-2023-doge]] | 25 second latency, 98 percent precision | Detects after price has already moved, no per-event ranking |
| Deep-learning detection | Replace tree models with neural sequence/CNN architectures on the same per-second features | [[chadalapaka-2022-deep-learning]] | Extends La Morgia baseline; learns temporal structure | Heavier compute, marginal gains over Random Forest on small datasets |
| Pre-event target prediction | Rank coins before the announcement | [[xu-2019-anatomy]], [[hu-2023-sequence-target-prediction]] | Actionable for trading, naturally cross-sectional | Smaller signal, dataset hungry |
| NLP + microstructure detection | Combine Telegram text models with order-book z-scores | [[bolz-2024-bertweet-zscore]] | Catches both event labelling and target prediction | Late-stage signal (degrades beyond 60s pre-T0) |
| Microstructure quantification | Document accumulation, volume, profit-distribution dynamics empirically | [[karbalaii-2025-microstructure]] | Recent, granular minute-OHLCV evidence | Single venue (Poloniex), descriptive rather than predictive |
| Ecosystem characterisation | Catalogue events and economic harm at scale | [[hamrick-2021-ecosystem|Hamrick et al. 2021]] | Largest scraped corpus, contradicts simple-profitability claims | Less curated, no per-event microstructure modelling |
| Mastermind / network tracing | Identify the human organizers | Perseus 2025 (Fu et al., arXiv:2503.01686, not yet ingested — see [[gaps/_index|gaps]]) | Targets the root cause, useful for regulators | Not a market signal, requires OSN data |
| Hybrid distance / density | Rank exchange pairs by anomaly score | Mansourifar et al. 2020 | Pure-market, no Telegram needed | Coarse, weaker than supervised |
| Social + market fusion | Combine Twitter sentiment with microstructure in LSTM/CNN | [[nghiem-2021-market-social-signals]] | Social signal adds predictive power over market features alone | Twitter-dependent; social layer is noisy |

## Key Findings

1. **Coordinated pumps follow a stereotyped microstructure pattern that is predictable from pre-event features.** The 412-event dataset of [[xu-2019-anatomy]] shows pre-pump signals (volume, returns, age, exchange) carry enough information to rank candidate coins and turn a profit (60 percent over 2.5 months on small capital). Confidence: high.
2. **Rush orders are the dominant microstructure feature for after-the-fact detection.** [[lamorgia-2023-doge]] reports that two rush-order statistics together account for ~37 percent of feature importance and lift 25-second F1 to 94.5 percent on Binance. Confidence: high.
3. **Cross-section ranking with channel-conditioning beats per-coin classification.** [[hu-2023-sequence-target-prediction]] (SIGMOD 23) shows pumped coins display intra-channel homogeneity, and a per-channel sequence encoder with positional attention reaches AUC 0.943 and Hit@5 of 53 percent on a 50-coin pool. Confidence: high.
4. **Supervised methods decisively outperform anomaly-thresholding baselines.** La Morgia et al. report 98.2 percent precision and 91.2 percent recall versus 52.1 percent and 78.8 percent for the Kamps and Kleinberg threshold method, on the same Binance events. Confidence: high.
5. **The economic harm is large and most pumps lose late buyers money on average.** [[hamrick-2021-ecosystem|Hamrick et al. 2021]] catalog ~5,000 pumps across Telegram and Discord; a Wall Street Journal analysis attributes ~$825 million in trading volume to coordinated pumps in H1 2018. Confidence: medium (one mainstream-press estimate, but multiple academic confirmations of the underlying dataset).
6. **The frontier is moving from event detection to network and actor attribution.** Perseus (Fu, Feng, Wu, Xu 2025, arXiv:2503.01686, not yet ingested — see [[gaps/_index|gaps]]) uses GraphSAGE on temporal attributed graphs of Telegram channels to identify 438 mastermind accounts behind 4,101 events between 2018 and 2024, F1 75.2 percent. Confidence: medium (single source, recent).
7. **Modern NLP-based pipelines on Telegram can both label events and predict targets at scale.** [[bolz-2024-bertweet-zscore]] classify 2,079 historical pumps on Poloniex with a BERTweet text head plus order-book z-score features and predict targets at top-5 accuracy of 56 percent over 50-coin pools. Confidence: medium.
8. **Deep-learning sequence models extend, but do not dominate, the per-second detection baseline.** [[chadalapaka-2022-deep-learning]] re-runs the La Morgia 2020 setup with CNN/RNN architectures, confirming that the rush-order microstructure signal is learnable by neural networks but only marginally beating the tree baseline at this dataset size. Confidence: medium.
9. **Granular Poloniex microstructure shows accumulation in most events.** [[karbalaii-2025-microstructure]] examine 485 events on Poloniex (Aug 2024 to Feb 2025) at minute resolution, finding detectable accumulation in 69.3 percent of cases and a ~14x volume spike during the pump hour. Confidence: medium (single venue, recent).

## Datasets

- **[[xu-2019-anatomy|Xu and Livshits (2019)]]**: 412 Telegram pumps, June 2018 to February 2019. Public on `xujiahuayz/pumpdump`.
- **La Morgia et al. ([[lamorgia-2020-icccn-realtime|2020]] / [[lamorgia-2023-doge|2023]])**: 902 confirmed events, 317 on Binance, July 2017 to January 2021. Public on `SystemsLab-Sapienza/pump-and-dump-dataset`. The de-facto benchmark also reused by [[chadalapaka-2022-deep-learning]].
- **[[hu-2023-sequence-target-prediction|Hu et al. (2023)]]**: 709 events, January 2019 to January 2022. Public on `Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency`.
- **[[hamrick-2021-ecosystem|Hamrick et al. (2018, 2021)]]**: ~5,000 events scraped from Telegram and Discord groups (WEIS 2019, IPM 2021). Less curated, broader.
- **[[bolz-2024-bertweet-zscore|Bolz et al. (2024)]]**: 2,079 Poloniex pumps coordinated through 43 Telegram channels, 2017 to 2024. Per-event BERTweet text features plus 20-second order-book z-scores.
- **[[karbalaii-2025-microstructure|Karbalaii (2025)]]**: 485 Poloniex events with minute-OHLCV reconstructions, Aug 2024 to Feb 2025. Most recent corpus.
- **Perseus (Fu et al. 2025, not yet ingested)**: 2,103 Telegram channels and 660 cryptocurrencies, April 2018 to October 2024, 27.4 million messages.

## Key Entities

- **Jiahua Xu** (UCL, formerly Imperial). Co-author of the foundational paper and a senior author on Perseus 2025. Field-shaping researcher.
- **Massimo La Morgia, Alessandro Mei, Julinda Stefa** (SystemsLab, Sapienza University of Rome). Maintain the most-used Binance pump dataset.
- **Sihao Hu and Bingsheng He** (NUS). SNN / SIGMOD 23 work on cross-section target prediction.
- **Tyler Moore, Marie Vasek, Neil Gandal** (economics-of-cybercrime group). Behind the [[hamrick-2021-ecosystem|Hamrick et al. ecosystem studies]].
- **Tobias Kamps and Bennett Kleinberg** (UCL). Author of the 2018 anomaly-detection baseline still cited in every subsequent paper.
- **Binance**: see [[Binance]]. The dominant venue for organized pumps in every dataset since 2019.
- **Telegram and Discord**: the coordination layer; channel attribution is a first-class feature in modern methods.

## Key Concepts

- [[Pump-and-Dump Scheme]] — the manipulation pattern.
- [[Cross-Section]] — the per-event ranking unit; central to [[xu-2019-anatomy]] and [[hu-2023-sequence-target-prediction]] and to our codebase.
- [[ranking-for-event-prediction]] — implementation-rationale companion to this survey: spells out why the project frames target prediction as a per-event ranking problem and which methodological choices follow.
- Rush order — market buy executed at the millisecond, the single most discriminative feature in [[lamorgia-2023-doge]].
- Top-k accuracy / Hit@k — the canonical evaluation metric for the target-prediction task.
- Channel homogeneity — the empirical regularity that pumps from the same Telegram channel target similar coin profiles ([[hu-2023-sequence-target-prediction]]).

## Contradictions

- **Profitability of buying the predicted target.** Xu and Livshits 2019 report a 60 percent return on a naive long strategy; La Morgia et al. characterize this as "maximizing recall at the expense of precision" and [[hamrick-2021-ecosystem|Hamrick et al. 2021]] find that on average, pumps fail to drive durable price increases. The disagreement seems explained by horizon (Xu and Livshits hold for minutes; Hamrick et al. measure days) and by no Xu / Livshits backtest accounting for slippage at retail size. Open question for our project: does the 60 percent edge survive a realistic price-impact model?
- **Whether the right framing is detection or prediction.** La Morgia et al. argue for after-the-fact monitoring as the regulator-friendly target; Xu and Livshits and Hu et al. argue for pre-event prediction. The two communities use different metrics (precision/recall on chunks vs Hit@k on cross-sections) and rarely compare.

## Open Questions

- Does cross-section z-scoring + a Catboost ranker (our setup) match or beat the SNN positional-attention approach without channel embeddings?
- How much of the SNN lift is from channel identity vs from sequence modeling per se?
- What does a proper price-impact-aware backtest do to the Xu and Livshits 60 percent figure on Binance 2021 and later data?
- Are there universal microstructure features (rush orders, depth imbalance) that transfer across exchanges, or are detectors brittle to venue?
- How do mastermind-tracing methods (Perseus) interact with the trading-signal task: can knowing the organizer identity directly improve target prediction?

## Sources

- [Xu and Livshits 2019, arXiv 1811.10109](https://arxiv.org/abs/1811.10109)
- [La Morgia, Mei, Sassi, Stefa 2023, arXiv 2105.00733](https://arxiv.org/abs/2105.00733)
- [Hu, Zhang, Lu, He, Li 2023, arXiv 2204.12929](https://arxiv.org/abs/2204.12929)
- [Kamps and Kleinberg 2018, Crime Science](https://link.springer.com/article/10.1186/s40163-018-0093-5)
- [Survey: Tornes et al. MDPI Future Internet 2023](https://www.mdpi.com/1999-5903/15/8/267)
- [Mansourifar, Chen, Shi 2020, arXiv 2003.06551](https://arxiv.org/abs/2003.06551)
- [[bolz-2024-bertweet-zscore|Bolz et al. 2024, arXiv 2412.18848]]
- [Fu, Feng, Wu, Xu 2025 Perseus, arXiv 2503.01686](https://arxiv.org/abs/2503.01686)
- [[hamrick-2021-ecosystem|Hamrick, Rouhi, Mukherjee, Feder, Gandal, Moore, Vasek 2018/2021]]
- [[lamorgia-2020-icccn-realtime|La Morgia et al. 2020, ICCCN]]
- [[chadalapaka-2022-deep-learning|Chadalapaka et al. 2022, deep-learning extension of La Morgia]]
- [[karbalaii-2025-microstructure|Karbalaii 2025, microstructure quantification on Poloniex]]
- [[nghiem-2021-market-social-signals|Nghiem et al. 2021, market + social signals]]
