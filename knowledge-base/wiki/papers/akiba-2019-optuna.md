---
type: paper
title: "Optuna: A Next-Generation Hyperparameter Optimization Framework"
created: 2026-04-19
updated: 2026-04-19
tags:
  - paper
  - ml-method
  - hyperparameter-optimization
  - tooling
status: summarized
related:
  - "[[backtest-pipelines]]"
  - "[[Top-K AUC]]"
sources:
  - "https://dl.acm.org/doi/10.1145/3292500.3330701"
year: 2019
authors:
  - Takuya Akiba
  - Shotaro Sano
  - Toshihiko Yanase
  - Takeru Ohta
  - Masanori Koyama
venue: "Proc. 25th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining"
key_claim: "Define-by-run search-space construction combined with efficient sampling (TPE, CMA-ES) and pruning of unpromising trials makes hyperparameter optimization both more expressive and more sample-efficient than prior frameworks."
methodology: "Library design paper. Implements TPE / CMA-ES / random sampling, asynchronous successive halving pruners, and a dynamic Python API where the search space is defined inside the objective function."
contradicts: []
supports: []
url: "https://arxiv.org/abs/1907.10902"
---

# Optuna: A Next-Generation Hyperparameter Optimization Framework (Akiba et al., 2019)

## TL;DR
Open-source hyperparameter optimization framework built around three ideas: define-by-run search spaces (parameters declared with Python control flow inside the objective), efficient sampling (Tree-structured Parzen Estimator by default, also CMA-ES, random, grid, Bayesian variants), and trial pruning (asynchronous successive halving, median pruner). Has become the de-facto Python HPO library.

## Key claims
- Define-by-run lets the search space depend on previously sampled values (conditional / hierarchical spaces) without pre-declaring a static grammar.
- TPE-based sequential model-based optimization is sample-efficient on common ML benchmarks vs. random search.
- Pruning unpromising trials early (using intermediate validation scores) gives substantial wall-clock speedups.
- Distributed and asynchronous optimization is supported via a shared storage backend (RDB).

## Methodology
- Frame each hyperparameter sweep as a sequence of trials calling user-supplied `objective(trial)`.
- Trials register parameters via `trial.suggest_*` calls; sampler and pruner are pluggable.
- Benchmark on ML tasks (CIFAR, MovieLens) showing better convergence than Hyperopt and SMAC.

## Strengths
- Pythonic API; integrates cleanly with sklearn, XGBoost, CatBoost, PyTorch.
- Pruning + parallelism dramatically reduce HPO budget.
- Active maintenance, large ecosystem (visualization, integrations).

## Weaknesses
- TPE is a heuristic, not a true Gaussian-process Bayesian optimizer; theoretical guarantees are weak.
- Like all SMBO, sensitive to objective noise; small validation sets give brittle suggestions.
- Pruning correctness depends on monotonic intermediate scores, which is not always the case (e.g., learning-rate-warmup curves).

## Relation to our work
- Cited in our paper (`paper/access.tex`, `\cite{optuna_2019}`) as the optimizer for our 30-trial Bayesian hyperparameter sweep on each model in the [[backtest-pipelines]] subsystem.
- Selection criterion is the [[Top-K AUC]] metric on the validation set; configuration with the best validation Top@K%-AUC is retained for test evaluation.
- Implementation lives in `backtest/pipelines/study.py` and per-model `pipe.py` modules.

## Cited concepts
- [[Top-K AUC]]
- [[Cross-Section]]
