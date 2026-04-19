---
type: paper
title: "Sequence-Based Target Coin Prediction for Cryptocurrency Pump-and-Dump"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - target-prediction
  - deep-learning
  - ranking
status: summarized
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Cross-Section]]"
  - "[[Binance]]"
  - "[[xu-2019-anatomy]]"
  - "[[lamorgia-2023-doge]]"
sources:
  - "https://arxiv.org/abs/2204.12929"
  - "https://dl.acm.org/doi/10.1145/3588686"
  - "https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency"
year: 2023
authors:
  - "Sihao Hu"
  - "Zhen Zhang"
  - "Shengliang Lu"
  - "Bingsheng He"
  - "Zhao Li"
venue: "SIGMOD / Proceedings of the ACM on Management of Data"
key_claim: "Pumped coins exhibit intra-channel homogeneity and inter-channel heterogeneity, so encoding a Telegram channel's prior P&D event history with positional attention substantially improves target coin prediction."
methodology: "Cross-section ranking via a sequence-based neural network (SNN). Each candidate coin is scored by attending over the channel's historical pump sequence with a positional attention mechanism."
contradicts: []
supports:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Cross-Section]]"
url: "https://arxiv.org/abs/2204.12929"
confidence: high
---

# Sequence-Based Target Coin Prediction for Cryptocurrency Pump-and-Dump (SNN)

## TL;DR
Hu and colleagues frame pre-pump target prediction as a [[Cross-Section]] ranking problem and observe that the channel that organizes a pump is a strong conditioning signal: pumps from the same channel tend to look alike, while different channels prefer different kinds of coins. They encode each channel's prior pump history as a sequence and attend over it with a positional attention layer. On 709 pumps (Jan 2019 to Jan 2022), the SNN reaches AUC 0.943, Hit@3 43.0 percent, Hit@5 53.2 percent. Confidence: high.

## Key claims
- Channel identity matters: intra-channel homogeneity and inter-channel heterogeneity are strong empirical regularities.
- Modeling the sequence of past pump targets per channel (not just per-coin features) is the key inductive bias.
- A positional attention mechanism handles long, noisy channel histories better than mean-pool or LSTM baselines.
- Hit@k is the right metric for this problem, matching how a strategy would deploy capital across top candidates.

## Methodology
- Data: 709 pumps from Telegram, Jan 2019 to Jan 2022, with per-coin pre-event features and per-channel event histories. Code and dataset on GitHub.
- Architecture: per-coin feature encoder plus channel-history sequence encoder with positional attention; final scorer ranks all candidates per cross-section.
- Baselines: Xu and Livshits style logistic ranker, plus standard sequence baselines.
- Metrics: AUC, Hit@3, Hit@5 (top-k ranking metrics on the cross-section).

## Strengths
- Cleanest formulation of the problem as cross-section ranking with explicit Hit@k evaluation.
- Channel-conditional modeling is novel and aligns with how pumps are actually organized.
- Public dataset and code, larger and more recent than Xu and Livshits.

## Weaknesses / Critiques
- Requires reliable channel attribution per event, which is itself a non-trivial labeling task.
- Pure ML metrics, no portfolio backtest or price-impact analysis.
- Hit@5 of 53 percent on a 50-coin pool is strong but still a long way from a deployable strategy without aggressive position sizing.
- No discussion of robustness when the channel that organized a future pump is unknown at inference time.

## Relation to our work
This is the closest published peer to `pumps_and_dumps`:
- Same cross-section ranking framing (`backtest/pipelines/CatboostRanker`, top-k evaluation in `backtest/utils`).
- Same metric family (top-k accuracy / Hit@k drives `calculate_topk` and `calculate_topk_percent_auc`).
- Their channel-history idea is something our pipeline does not yet exploit. Open question whether adding per-channel embeddings to our Catboost models would replicate their lift.
- Their gap (no portfolio backtest or impact model) is exactly the part our `backtest/portfolio` subsystem fills (see [[impact-models-for-lowcap-crypto]]).

## Cited concepts
- [[Pump-and-Dump Scheme]]
- [[Cross-Section]]
- [[Binance]]
