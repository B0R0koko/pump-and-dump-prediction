---
type: paper
title: "Crypto Pump and Dump Detection via Deep Learning Techniques"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - pump-detection
  - deep-learning
  - lstm
  - transformer
status: summarized
related:
  - "[[Pump-and-Dump Scheme]]"
  - "[[Pump Announcement Window]]"
  - "[[lamorgia-2023-doge]]"
  - "[[lamorgia-2020-icccn-realtime]]"
  - "[[hu-2023-sequence-target-prediction]]"
sources:
  - "https://arxiv.org/abs/2205.04646"
year: 2022
authors:
  - "Viswanath Chadalapaka"
  - "Kyle Chang"
  - "Gireesh Mahajan"
  - "Anuj Vasil"
venue: "arXiv preprint arXiv:2205.04646"
key_claim: "Deep sequence models (LSTMs and Transformers) on per-second market features outperform La Morgia et al.'s random-forest baseline for in-progress pump detection."
methodology: "Reproduce La Morgia 2020 dataset and feature pipeline, then swap the random-forest classifier for LSTM and Transformer architectures; compare under streaming evaluation."
contradicts: []
supports:
  - "[[lamorgia-2020-icccn-realtime]]"
  - "[[lamorgia-2023-doge]]"
---

# Crypto Pump and Dump Detection via Deep Learning Techniques (Chadalapaka et al., 2022)

## TL;DR
Direct deep-learning follow-up to [[lamorgia-2020-icccn-realtime]]. The authors adopt the same per-second Binance trade-feature pipeline and labelled pump list, then replace the random-forest classifier with LSTM and Transformer sequence models. They report gains over the random-forest baseline at the same streaming-detection task and argue that the temporal dynamics of pump windows are better captured by sequence models than by chunk-level summary features.

## Key claims
- Sequence models (LSTM, Transformer) outperform random forests for per-second pump detection on La Morgia's labelled set.
- Modelling the temporal evolution of a pump window, rather than reducing it to a vector of statistics, captures more of the manipulation footprint.
- The improvement holds under streaming evaluation, not just on i.i.d. shuffled splits.

## Methodology
- Reuse the La Morgia 2020 / 2023 dataset of Telegram-confirmed pumps and Binance trade aggregates.
- Engineer per-second features (volume, return, trade-count, taker imbalance proxies).
- Train and compare LSTM and Transformer classifiers against the random-forest baseline.
- Evaluate with chronological splits to mimic real-time deployment.

## Strengths
- First systematic deep-learning treatment of the La Morgia detection benchmark.
- Demonstrates that the field's ceiling is not set by the random-forest choice.
- Clear apples-to-apples comparison: same data, same features, different model class.

## Weaknesses / Critiques
- Workshop / preprint scope; limited hyperparameter search and ablation.
- Same upstream limitations as the underlying dataset (group selection bias, Binance-only).
- Detects in-progress pumps; does not address pre-announcement target prediction.

## Relation to our work
- Cited in `paper/access.tex` (line 91) as the deep-learning follow-up to La Morgia, completing the line of work we contrast with our cross-sectional target-prediction framing.
- Their result that sequence models help in detection does not transfer directly to our setup: our task is single-snapshot ranking of cross-sectional candidates ([[Cross-Section]]) one hour before the announcement, where the temporal-sequence advantage is largely absent.
- See [[hu-2023-sequence-target-prediction]] for a sequence-model approach that does target prediction.

## Cited concepts
- [[Pump-and-Dump Scheme]]
- [[Pump Announcement Window]]
