---
type: paper
title: "Building Cross-Sectional Systematic Strategies By Learning to Rank"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - learning-to-rank
  - cross-sectional
  - finance
status: summarized
related:
  - "[[Cross-Section]]"
  - "[[Top-K AUC]]"
  - "[[ranking-for-event-prediction]]"
  - "[[backtest-pipelines]]"
sources:
  - "https://arxiv.org/abs/2012.07149"
  - "https://jfds.pm-research.com/content/3/2/70"
year: 2020
authors:
  - "Daniel Poh"
  - "Bryan Lim"
  - "Stefan Zohren"
  - "Stephen Roberts"
venue: "The Journal of Financial Data Science (Apr 2021), arXiv:2012.07149"
key_claim: "Replacing regression/classification with learning-to-rank (LambdaMART, ListNet, RankNet) on cross-sectional momentum signals roughly triples Sharpe Ratios versus traditional approaches."
methodology: "Cross-sectional momentum, multi-feature inputs, LTR algorithms compared head-to-head with regression baselines on equities."
contradicts: []
supports:
  - "[[Cross-Section]]"
url: "https://arxiv.org/abs/2012.07149"
confidence: high
---

# Poh et al. 2020: Building Cross-Sectional Systematic Strategies By Learning to Rank

## TL;DR
For cross-sectional financial strategies (long top, short bottom), what matters is the *order* of assets, not absolute return forecasts. The authors reframe portfolio construction as a learning-to-rank (LTR) problem and report ~3x Sharpe improvement over regression-based cross-sectional momentum, with LambdaMART as the top performer.

## Key claims
- Cross-sectional strategy success "depends critically on accurately ranking assets prior to portfolio construction"; pointwise regression/classification objectives are misaligned with this goal.
- Replacing regression with LTR objectives (RankNet, ListNet, LambdaMART) improves both ranking accuracy and trading performance.
- LambdaMART delivers the best returns and ranking accuracy among the methods tested, with lower drawdowns and downside risk than regression baselines.
- The framework is modular: it accepts arbitrary tabular feature inputs, so the LTR layer is a drop-in replacement for the final regression/classification head in any cross-sectional pipeline.

## Methodology
- Demonstrative case study: cross-sectional momentum on equities.
- Compares pointwise (regression), pairwise (RankNet, LambdaRank/LambdaMART), and listwise (ListNet) LTR.
- Evaluation includes ranking accuracy plus standard portfolio metrics (Sharpe, drawdowns).
- Roughly threefold Sharpe boost vs traditional approaches.

## Strengths
- Directly motivates the ranking framing for cross-sectional asset selection.
- Compares pointwise vs pairwise vs listwise on the same cross-sections, isolating the objective effect.
- Quantitative result (~3x Sharpe) is large enough to justify framework adoption.

## Weaknesses / Critiques
- Equities at monthly frequency, not crypto event windows: cross-section size and base rate of "winners" differ.
- Multi-level relevance (top vs bottom decile); our P&D setting is binary with one positive per cross-section, where listwise ranking metrics like NDCG degenerate.
- LambdaMART's edge in their setup may not transfer cleanly when there is only one positive per group (see [[ranking-for-event-prediction]]).

## Relation to our work
This paper is the closest published analog to our setup: a financial cross-sectional ranking problem solved with LTR. It justifies our `CatboostRanker` pipeline over `CatboostClassifier`. Key differences for adaptation:
- We rank candidates *within a single pump event*, not across the full equity universe at one point in time.
- We have ~1 positive per cross-section (the manipulation target), so pairwise objectives like PairLogit / RankNet generate the most informative gradients (every pair has a clear winner). NDCG-driven listwise objectives produce mostly zero-gradient pairs.
- The ~3x Sharpe gap reported here motivates the value of ranking framing even if our absolute multiplier differs.

## Cited concepts
- [[Cross-Section]]
- [[Top-K AUC]]
- Pairwise ranking loss: logistic loss computed over (positive, negative) pairs within a group (e.g. PairLogit in CatBoost, RankNet in neural ranking). See [[catboost-docs-ranking-objectives]].
- LambdaMART, RankNet, ListNet
