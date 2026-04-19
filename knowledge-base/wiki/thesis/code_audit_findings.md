---
type: thesis
title: "Code Audit: backtest/ subtree (IEEE Access R1 revision)"
created: 2026-04-19
updated: 2026-04-19
tags:
  - thesis
  - revision
  - code-audit
  - empirical
status: developing
related:
  - "[[backtest-portfolio]]"
  - "[[backtest-pipelines]]"
  - "[[backtest-utils]]"
confidence: high
---

# Code Audit: `backtest/` subtree

This is an audit (not an implementation). Findings are enumerated by severity, with `file:line`, fix sketch, and blast radius.

> **Scope**: `backtest/portfolio/`, `backtest/utils/`, `backtest/robust/`, `backtest/pipelines/`, and the `notebooks/research_notebook.ipynb` path that feeds Table 5. Audit only. No code was modified.

## TL;DR

1. **Confirmed (critical):** headline Table 5 runs with `use_price_impact=False`. `backtest/utils/evaluation.py:119-122::get_equity_curve_for_experiment` constructs `TOPKPortfolio(model=..., portfolio_size=portfolio_size)`; `PortfolioExecutionConfig.use_price_impact` defaults to `False`; the flag is never threaded through. `evaluate_topk_pnl_for_quantities` has a separate code path that forces `use_price_impact=True` (`backtest/portfolio/TOPKPortfolio.py:351`) — this is what feeds Fig. 7 (`pnl_vs_quantity`). The two are disjoint.
2. **High:** `num_trades` on a `PriceImpactModel` fitted from klines is the number of *candles*, not trades. Named misleadingly, surfaced in Transaction diagnostics as `entry_impact_num_bars` but called `num_trades` on the model. Mostly cosmetic but risks confusion in reviewer discussions.
3. **High:** no reproducibility seed on any sklearn / CatBoost model (`RandomForestClassifier`, `LogisticRegression`, `CatBoostClassifier`, `CatBoostRanker`, SMOTE). Multi-run robustness is driven by data subsampling only, not retraining stochasticity.
4. **Medium:** SMOTE applied at `create_sample` time on the *entire* train split, not within cross-sections. The cross-section-blending issue is real.
5. **Medium:** CatboostRanker uses `YetiRank:mode=NDCG` with `pct=True` continuous relevance, no `eval_metric` override, so early stopping stops on NDCG rather than Top@K%-AUC. Likely a suboptimal ranker config.
6. **Medium:** `create_dataset` silently drops pumps with no feature file (`backtest/utils/build_dataset.py:51-53`). 25 pumps dropped with a WARNING but the pump list is not surfaced anywhere reproducible.
7. **Medium:** `evaluate_subperiod_metrics` silently produces numbers on n=3 pumps in the late subperiod with no warning — reported directly in `df_subperiod` output.

---

## Bugs

### [CRITICAL] Table 5 headline numbers computed without price impact (confirmed)

- **File**: `backtest/utils/evaluation.py:119-122` (`get_equity_curve_for_experiment`), feeding `compute_equity_curves` → `compute_portfolio_statistics` → Sharpe ratio table.
- **Problem**: `TOPKPortfolio(model=..., portfolio_size=portfolio_size)` instantiates with `use_price_impact` unset, which resolves to `PortfolioExecutionConfig(..., use_price_impact=False)` by default (`backtest/portfolio/config.py:14`). In the VWAP estimator (`backtest/portfolio/vwap_estimator.py:38-48`) this short-circuits the impact branch and returns raw entry/exit prices. The only execution cost applied is the fixed 25 bps round-trip fee in `Transaction.transaction_return` (`backtest/portfolio/BasePortfolio.py:51-58`).
- **Blast radius**: Table 5 / `df_results` in `notebooks/research_notebook.ipynb` cell after `curves[k] = get_equity_curve_for_experiment(...)`, Fig. 6 (equity curves), all Sharpe / annualized return numbers in Section 6.1.
- **Fix sketch**: Add `use_price_impact: bool = False` to `get_equity_curve_for_experiment` and `compute_equity_curves`, thread to `TOPKPortfolio`. For the revision, run both (with/without) and report side-by-side.

### [HIGH] `num_trades` field is candle count, not trade count, when fit from klines

- **File**: `backtest/portfolio/PriceImpact.py:318` (`num_trades=int(samples.shape[0])`), reached from `fit_price_impact_model_from_klines` via `_fit_from_samples`.
- **Problem**: `samples.shape[0]` counts *klines that produced a valid impact sample* (5-min bars, or 5-sec bars for the manipulated provider). The dataclass field is named `num_trades`. The consumer in `TOPKPortfolio._create_transaction_from_intent` (`backtest/portfolio/TOPKPortfolio.py:202-203`) renames it to `entry_impact_num_bars` — correct at the call site, wrong at the source. Manipulated provider uses 5-second bars, so `num_trades` there is on an entirely different scale from the lookback provider's 5-minute bars — not comparable.
- **Fix sketch**: Rename `num_trades` → `num_samples` on `PriceImpactModel`. Add a `sample_frequency` field so the pre-pump (5min) and manipulation (5s) models are self-describing. Update `candidate.num_trades > 0` check to `candidate.num_samples > 0` (same semantics, clearer).

### [HIGH] No random seed on any ML model → non-reproducible retraining

- **Files**:
  - `backtest/pipelines/RandomForest/pipe.py:19` (`_BASE_PARAMS`).
  - `backtest/pipelines/LogisticRegression/pipe.py:29-33` (`_BASE_PARAMS`).
  - `backtest/pipelines/CatboostClassifier/pipe.py:21-27`, `CatboostClassifierSMOTE/pipe.py:31-36`, `CatboostClassifierTOPKAUC/pipe.py:24-30`, `CatboostRanker/pipe.py:27-31`.
  - `CatboostClassifierSMOTE/pipe.py:77` (`SMOTE()` without `random_state`).
- **Problem**: None of the base params dicts include `random_state` / `random_seed`. Rerunning the notebook will not reproduce exact numbers even on the same data. The robustness table (`catboost_topkauc_subset_runs.csv`) varies over data subsets only; model stochasticity noise is invisible.
- **Fix sketch**: Add `random_state=42` (sklearn) / `random_seed=42` (CatBoost) to `_BASE_PARAMS` in every pipeline. For SMOTE: `SMOTE(random_state=42)`. No effect on training semantics; trivial change.

### [MEDIUM] SMOTE ignores cross-section structure

- **File**: `backtest/pipelines/CatboostClassifierSMOTE/pipe.py:72-80` (`apply_smote`).
- **Problem**: SMOTE interpolates in the regressor feature space across the entire train split (after cross-section standardisation). It does **not** respect `pump_hash`; synthetic positives are blended from neighbors in *different* cross-sections. A synthetic minority example has no valid `pump_hash`, `pump_time`, or `currency_pair`. Since only `regressors` and `target` are extracted and concatenated back, the metadata columns simply carry along pre-SMOTE values for the matched rows — in effect, synthetic positives inherit the metadata of whichever minority row was the SMOTE seed. Synthetic samples therefore collide with real cross-sections at scoring time and break the cross-section invariant. This is exactly the issue [[blagus-lusa-2013-smote-highdim]] was probing from a high-dimensional angle.
- **Blast radius**: SMOTE model numbers only. SMOTE is not the champion (CatboostClassifier + TOPKAUC is), but it is reported in Tables 3-4 and the consolidated metrics. Invalid baseline weakens the revision argument.
- **Fix sketch**: Two options:
  - (a) Cross-section-aware SMOTE: resample only within each `pump_hash` group. Problem: most cross-sections have exactly 1 positive, so within-group SMOTE has no minority neighbors.
  - (b) Drop SMOTE from the comparison and explain why: "cross-section structure makes SMOTE ill-defined for this problem". Likely the honest answer.
- **Recommendation**: flag in paper as a limitation, do not fix for the revision. Add a paragraph explaining why SMOTE is not naturally applicable and remove it from the headline comparison, keeping only as an ablation.

### [MEDIUM] CatboostRanker early-stopping metric is NDCG, not Top@K%-AUC

- **File**: `backtest/pipelines/CatboostRanker/pipe.py:27-31` + `backtest/pipelines/CatboostRanker/model.py:21`.
- **Problem**: `_BASE_PARAMS` sets `objective="YetiRank:mode=NDCG"` with `early_stopping_rounds=50, use_best_model=True`. No `eval_metric` override. CatBoost early-stops on NDCG, not on the actual ranking metric we care about (Top@K%-AUC). The target `asset_return_rank` uses `rank(pct=True, ascending=False)`, a continuous 0-1 relevance score — fine for YetiRank but not aligned with the top-of-list metric.
- **Blast radius**: ranker baseline numbers. Ranker is not the champion; still, a misconfigured baseline weakens the "we beat everything including rankers" narrative.
- **Fix sketch**: Add `"eval_metric": TOPKPAUCMetric(df_train, df_val)` to ranker `_BASE_PARAMS`, similar to `CatboostClassifierTOPKAUC`. Alternatively switch to `"objective": "YetiRank:mode=MAP"` or `"YetiRank:mode=MRR"`, which are top-of-list rank-aware.

### [MEDIUM] `create_dataset` silently drops pumps without features

- **File**: `backtest/utils/build_dataset.py:18-33`, `51-54`, `56-58`.
- **Problem**: `_read_cross_section` returns `(pump, None)` in two cases: (1) no parquet file for this pump; (2) pump's own target currency pair is not in the cross-section (feature writer produced a partial cross-section). `create_dataset` collects both under `skipped_pumps` and logs a single `WARNING: No data present for 25 pumps` with no list, then the dataset is built without them. Cases (1) and (2) are not distinguished in the log. The ETHBTC exclusion is **not** codified here; grep shows it is only in the paper text.
- **Fix sketch**: (a) Log per-pump `pump_hash` at `INFO` level when dropping, and persist the list to `resources/dropped_pumps.json` for provenance. (b) Distinguish missing-file from missing-target-pair cases. (c) Add the ETHBTC exclusion to a feature filter or a data-validation stage, with a reason field.

### [MEDIUM] `evaluate_subperiod_metrics` reports n=3 without warning

- **File**: `backtest/robust/robustness.py:142-203`.
- **Problem**: `early_hashes` vs `late_hashes` split returns 55 vs 3 pumps at split date 2022-06-01 (verified from notebook output). The function silently computes Top@K% metrics on 3 pumps and returns a DataFrame with `n_pumps=3`. 3 pumps means the metric can only take values 0.0, 0.333, 0.667, 1.000 — effectively uninterpretable. The function does not filter or warn.
- **Fix sketch**: (a) Add a `min_pumps: int = 10` filter and log a warning when a subperiod is dropped for tiny n. (b) Replace the binary split with a rolling or quarterly analysis, which this function cannot do as written — needs a loop wrapper. (c) Attach a bootstrap CI to each subperiod so tiny-n subperiods are immediately visible as very wide intervals.

### [LOW] Tie-breaking in `TopKPortfolioSelector` is non-deterministic

- **File**: `backtest/portfolio/selector.py:31-33`, same pattern in `backtest/utils/metrics.py:30,55` and `backtest/robust/significance.py:74,99`.
- **Problem**: `sort_values(by=COL_PROBAS_PRED, ascending=False)` without `kind="mergesort"` or a secondary tiebreaker. pandas default `quicksort` is not stable. For tied scores (constant-prediction model, calibrated scores at exactly 0, early rounds of training) the top-k selection depends on pre-sort row order.
- **Fix sketch**: `sort_values(by=[COL_PROBAS_PRED, COL_CURRENCY_PAIR], ascending=[False, True], kind="mergesort")`. Cheap, no behavior change for practical scores.

### [LOW] Transaction `transaction_return` always subtracts 25 bps even under impact

- **File**: `backtest/portfolio/BasePortfolio.py:51-58`.
- **Problem**: Not a bug per se, but the docstring is ambiguous: "fixed 25 bps round-trip trading cost". The 25 bps is subtracted whether or not `use_price_impact=True`, so the effective cost is (fee + slippage from VWAP). The paper Eq. 3 conflates the two in prose.
- **Fix sketch**: Docstring rewrite: "fixed round-trip exchange fee (~20–25 bps at Binance VIP0). Execution slippage is modelled separately through VWAP price adjustment when `use_price_impact=True`."

### [LOW] `BasePortfolio.Portfolio.get_weight` is O(k) per lookup, not cached

- **File**: `backtest/portfolio/BasePortfolio.py:28-30`.
- **Problem**: `self.weights[self.currency_pairs.index(cp)]`. For a top-k portfolio of size 30, per-pump lookup is O(k) and called multiple times per transaction (PnL + impact diagnostics). Not slow at current scale.
- **Fix sketch**: Precompute `self._weights_map: Dict[CurrencyPair, float]` in `__init__`. S effort.

### [LOW] `get_cross_section` mutates dataset view via copy; no memoization

- **File**: `backtest/utils/sample.py:74-79`.
- **Problem**: `get_cross_section` rebuilds a `Pool` every call, even when the same pump is evaluated repeatedly (e.g., in `evaluate_pnl_for_quantities` with multiple quantities). Quadratic-ish cost over quantity grid.
- **Fix sketch**: Memoize cross-section datasets by `(pump.as_pump_hash(), ds_type)`.

### [LOW] `paired_bootstrap_topk_percent_auc_test` p-value convention at exact ties

- **File**: `backtest/robust/significance.py:299-304`.
- **Problem**: For `alternative="greater"`, `p_value = float(np.mean(diffs <= 0))`. At bootstrap samples where `diff == 0` (possible when matrices have integer counts per cross-section and the shared resampled indices produce identical means), the `<=` biases the p-value upward. More standard is `(sum(diff <= 0) + 1) / (n_bootstrap + 1)` (Davison-Hinkley) to avoid zero and integrate the observed statistic.
- **Blast radius**: p = 0.059 for the champion vs CatBoost-Tuned comparison is literally on the 0.05 line.
- **Fix sketch**: Use `(np.sum(diffs <= 0) + 1) / (n_bootstrap + 1)`. S effort.

### [LOW] `ImpactModelProvider` protocol signature mismatch

- **File**: `backtest/portfolio/interfaces.py:20-23` vs `backtest/portfolio/manipulated_impact_provider.py:50-55`.
- **Problem**: The `ImpactModelProvider.get_impact_model(pump, currency_pair)` protocol has two positional args. `ManipulatedImpactModelProvider.get_impact_model` adds `end_exclusive: datetime | None = None`, and `LookbackImpactModelProvider.get_impact_model` does not. The protocol is satisfied only if callers treat the extra kwarg as optional — TOPKPortfolio does at line 172. Static-typing surface is fragile.
- **Fix sketch**: Add `end_exclusive: datetime | None = None` to the protocol, or define two separate protocols. S effort.

---

## Dead code / test drift

### [MEDIUM] Dead test file exercising a removed API

- **File**: `test/analysis/portfolio/TOPKPortfolio.py`.
- **Problem**: Not prefixed with `test_` so pytest does not pick it up. Exercises `portfolio_manager.evaluate_cross_section(...)` — a method that no longer exists on `TOPKPortfolio`. Imports `Portfolio` from `BasePortfolio`, constructs a `Dataset` with an unsupported `FeatureSet(categorical_features=[])` shape, and asserts against hardcoded numeric returns. This file is silently invisible.
- **Fix sketch**: Either delete (if coverage is in `test_price_impact.py`) or rewrite as a proper `test_topk_portfolio.py` calling `evaluate_for_pump`. Recommend rewrite; minimal integration test for the top-k selection path is useful.

### [LOW] Cached `catboost_info/` under every pipeline directory is checked in

- Paths: `backtest/pipelines/CatboostClassifier/catboost_info/`, `.../CatboostClassifierSMOTE/catboost_info/`, etc.
- **Problem**: These are CatBoost training artifacts (tmp logs). Should not be in git.
- **Fix sketch**: Add `catboost_info/` to `.gitignore`; remove from tracked files.

---

## Code smells / improvement opportunities

### [S] Missing `sort=False` in two groupby calls

- `backtest/utils/metrics.py:29, 54` uses `groupby(COL_PUMP_HASH)` without `sort=False`. Not a correctness issue; minor perf.

### [S] `Portfolio.weights: np.ndarray[float]` type annotation is wrong in Python 3.13

- `backtest/portfolio/BasePortfolio.py:22`. `np.ndarray[float]` is legal at runtime but not a proper generic signature. Use `np.ndarray` or `npt.NDArray[np.float64]`.

### [S] `Transaction` dataclass has 13 fields with defaults → long constructor

- `backtest/portfolio/BasePortfolio.py:33-48`. Would be cleaner split into `Transaction` (prices + metadata) and an optional `TransactionExecution` (impact / notional) composed inside it.

### [M] `PriceImpact.py` has three fit functions with different entry points but near-identical internals

- `fit_price_impact_model` (from trades), `fit_price_impact_model_from_klines` (from klines), `_fit_from_samples` (common tail). The trades path is *unused* by the live backtest — the providers go through klines only. Dead code that obscures the actual pipeline.

### [M] `fillna_with_median_by_cross_section` re-fills across cross-sections at the end

- `backtest/pipelines/BasePipeline.py:100-110`. Fills with cross-section median first, then with global median. The fallback to global median violates the cross-section invariant for the handful of features where a whole cross-section is NaN. Small effect, but worth documenting as a deviation from "strict cross-section preprocessing".

### [M] Equity curve aggregation in `compute_portfolio_statistics` reindexes to daily with `fill_value=0`

- `backtest/utils/evaluation.py:165-168`. Filling no-pump days with 0 return (not NaN) biases volatility downward and Sharpe upward — common practice but worth acknowledging. Pumps cluster heavily; the filled zeros dominate the denominator. A realistic Sharpe would compute returns only on days with trades, or use the pump-to-pump interval as the time unit.

### [L] `random_topk_baseline` uses `df_all.groupby("pump_hash")` literal, not the `COL_PUMP_HASH` constant

- `backtest/utils/evaluation.py:30-37`. Cosmetic.

### [L] `NotionalSizer` silently returns 0 when neither quote nor USDT target is positive

- `backtest/portfolio/sizing.py:37`. Combined with `_create_transaction_from_intent`'s `if intent.intended_notional_quote <= 0: return Transaction(..., prices only)` at line 153, this means a misconfigured TOPKPortfolio will silently produce fee-only returns. Worth an explicit validation in `PortfolioExecutionConfig.__post_init__` requiring at least one of the two notionals to be positive.

---

## Correctness questions (for the user)

1. **Does the 25 bps in `Transaction.transaction_return` represent the Binance spot taker fee (0.1% × 2 = 20 bps at VIP0, 25 bps is approximately correct), or a catch-all slippage proxy?** The answer determines whether the "without impact" column in Table 5 is defensible at all, or must be replaced.
2. **SMOTE: was the decision to apply SMOTE post-standardization deliberate?** Applying SMOTE *before* cross-section standardization would interpolate raw features and then zscore within cross-section, which is arguably cleaner but still cross-contaminates.
3. **Is `evaluate_subperiod_metrics`'s split date (2022-06-01) reported anywhere in the paper as a decision, or is it set in the notebook cell?** If it is a notebook-only choice, the revision should either justify the date (FTX collapse around Nov 2022) or move to quarterly splits.
4. **Was the ranker `"asset_return_rank"` target derived from `target_return@5MIN` with `pct=True` intentional?** The continuous relevance score with YetiRank+NDCG is mathematically fine, but if the goal is "rank the pumped asset first" the label could simply be binary (`is_pumped`) and the loss `"QueryRMSE"` or `"YetiRank:mode=MRR"`.
5. **Is there a reason `LookbackImpactModelProvider.get_impact_model` caches only by `(currency_pair.name, pump.time)` and not the lookback-window bounds?** If two runs with different `impact_lookback_days` share a process (e.g., a sweep), the cache will return the model from the first run.

---

## Quick wins (reviewer-adjacent, small effort)

1. **Thread `use_price_impact` through `get_equity_curve_for_experiment`** and rerun Table 5 with impact on, side by side with the current numbers. **Highest leverage.**
2. **Add `random_state=42` to every `_BASE_PARAMS`** and `SMOTE(random_state=42)`. Unlocks clean reproducibility claims and a seed-variance robustness column. ~15 lines total.
3. **Add a `min_pumps: int = 10` guard in `evaluate_subperiod_metrics`** and log a warning for skipped subperiods. Immediately addresses Reviewer 1's "n=3" observation even without switching to quarterly.
4. **Log dropped-pump hashes in `create_dataset`** and persist to `resources/dropped_pumps.json`. Reproducibility quick win and hands off cleanly to the developer agent's data-provenance track.
5. **Rename `PriceImpactModel.num_trades` → `num_samples`** and add a `sample_frequency: str` field. Defuses the "what does 14 bars mean" reviewer question and makes 5s vs 5min scales comparable.

---

## Not flagged as bugs, but deserve a paragraph in the paper Limitations

- **Liquidity filter is not implemented.** `TopKPortfolioSelector` has no concept of per-pair liquidity. The impact model is fit even on near-dead pairs; if `beta=0` (no useful data), impact is zero and the execution is treated as frictionless.
- **Q vs book-depth cap absent.** No order-size cap; impact model extrapolates the sqrt law without bound.
- **Impact model fit on kline aggregates, not tick-by-tick.** Paper (and revision) should acknowledge this is a coarse-grained proxy.
- **ETHBTC exclusion is in paper text only**, not in code. Confirmed by grep. Codify it with a data-validation stage.

---

## Cross-reference map

| Finding | Severity |
|---|---|
| use_price_impact flag dead in Table 5 | CRITICAL |
| num_trades naming / unit | HIGH |
| no model seeds | HIGH |
| SMOTE cross-section blending | MEDIUM |
| Ranker early-stop metric | MEDIUM |
| Silent pump drops | MEDIUM |
| n=3 subperiod | MEDIUM |
| Tie-breaking | LOW |
| Transaction fee docstring | LOW |
| DH-correction p-value | LOW |

---

## What I did not touch

Per audit brief, no fixes applied. All findings above are verification-only.

## Open items for the user to triage

- **Triage severity and cut the revision list.** The minimum viable revision scenario — fixing the seed issue and the impact flag — is ~2 days of work.
- **Decide on SMOTE**: fix it or drop it from the headline comparison. Either is defensible; leaving it as-is is not.
- **Decide on the ranker**: re-tune with a rank-aware eval metric or leave as-is with a known-limitation footnote.
