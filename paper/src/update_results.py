"""Render manuscript results from one completed notebook run and audit the exports.

Run with the project Python after both research notebooks finish. --check checks
the current manuscript without modifying it. The submitted PDF remains the
highlight baseline, not a source of numerical results.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from build_highlighted import floats, normalized_lines, table_body

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "notebooks/analysis_outputs/paper_cohort_473_l2"
RUN = OUTPUT / "paper_run"
BOOT = OUTPUT / "test_auc_bootstrap"
IMPACT = OUTPUT / "portfolio_1000usdt_slippage"
CB = "CatBoost classifier + Top@k%AUC early stopping"
L1 = "Logistic regression + Tuned"
L2 = "Logistic regression L2 + Tuned"
RF = "Random forest + Tuned"
CAT = "CatBoost classifier + Tuned"
SMOTE = "CatBoost classifier + SMOTE + Tuned"
RANKER = "CatBoost ranker + Tuned"
NAMES = {CB: "CatBoost + ES", L1: "Logistic regression L1", L2: "Logistic regression L2",
         RF: "Random forest", CAT: "CatBoost", SMOTE: "CatBoost + SMOTE", RANKER: "CatBoost ranker"}
SIZES = [1, 2, 5, 10, 20, 30]
TABLES = {}
INPUTS = {}
FIGURES = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, **kwargs):
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    return pd.read_csv(path, index_col=0, **kwargs)


def escaped(value):
    return str(value).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fill(text, values):
    return re.sub(r"@@(\w+)@@", lambda match: str(values[match[1]]), text)


def table(label, caption, headers, rows, wide=False):
    kind = "table*" if wide else "table"
    tabular = "\n".join([
        r"    \begin{tabular}{l" + "r" * (len(headers) - 1) + "}",
        r"        \toprule", "        " + " & ".join(headers) + r" \\",
        r"        \midrule",
        *["        " + " & ".join(map(str, row)) + r" \\" for row in rows],
        r"        \bottomrule", r"    \end{tabular}",
    ])
    if wide:
        # Fit wide tables without enlarging compact ones beyond the common font size.
        tabular = r"    \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%" + "\n" + tabular + "\n    }"
    rendered = (f"\\begin{{{kind}}}[!t]\n    \\centering\n    \\small\n"
                "    \\setlength{\\tabcolsep}{5pt}\n    \\renewcommand{\\arraystretch}{1.15}\n"
                f"    \\caption{{{caption}}}\n{tabular}\n    \\label{{{label}}}\n\\end{{{kind}}}\n")
    TABLES[label] = rendered
    return rendered


def figure(filename, label, caption, source=None, wide=True):
    source = source or RUN / "images" / filename
    target = HERE / "images" / filename
    FIGURES[str(target.relative_to(ROOT))] = source
    width, kind = (r"\textwidth", "figure*") if wide else (r"\columnwidth", "figure")
    return (f"\\begin{{{kind}}}[!t]\n    \\centering\n"
            f"    \\includegraphics[width={width}]{{images/{filename}}}\n"
            f"    \\caption{{{caption}}}\n    \\label{{{label}}}\n\\end{{{kind}}}\n")


def paired(frame, left, right):
    row = frame.loc[((frame.model_a == left) & (frame.model_b == right)) |
                    ((frame.model_b == left) & (frame.model_a == right))].iloc[0].copy()
    if row.model_a != left:
        row.observed_diff, row.ci_lower, row.ci_upper = -row.observed_diff, -row.ci_upper, -row.ci_lower
    return (f"${row.observed_diff:.4f}$, paired 95\\% CI $[{row.ci_lower:.4f},\\,{row.ci_upper:.4f}]$, "
            f"two-sided $p={row.p_value:.4f}$ and Holm-adjusted $p={row.p_value_holm:.3f}$")


def replace_float(source, label, replacement):
    lines = normalized_lines(source)
    block = next(b for b in floats(lines) if b["label"] == label)
    lines[block["start"]:block["end"] + 1] = replacement.rstrip().splitlines()
    return "\n".join(lines) + "\n"


def render(source):
    validation = read(RUN / "tables/validation_metrics.csv")
    topk = read(RUN / "tables/test_topk.csv")
    topkp = read(RUN / "tables/test_topkp.csv")
    classification = read(RUN / "tables/test_classification.csv")
    auc = read(BOOT / "auc_summary.csv")
    pairwise = pd.read_csv(BOOT / "auc_pairwise.csv")
    INPUTS[str((BOOT / "auc_pairwise.csv").relative_to(ROOT))] = digest(BOOT / "auc_pairwise.csv")
    robust = read(RUN / "tables/robustness_summary.csv")
    lr_robust = read(RUN / "tables/logistic_robustness_summary.csv")
    importance = read(RUN / "tables/feature_importance_quantiles.csv")
    subperiod = read(RUN / "tables/subperiod_metrics.csv")
    portfolios = read(RUN / "tables/portfolio_statistics.csv")
    impact = read(IMPACT / "model_statistics.csv")
    impact_random = pd.read_csv(IMPACT / "model_vs_random.csv").set_index(["k", "metric"])
    random_metrics = pd.read_csv(RUN / "tables/random_portfolio_metrics.csv").set_index(["k", "metric"])
    impact_summary = pd.read_csv(IMPACT / "random_portfolio/summary.csv").set_index(["k", "metric"])
    sensitivity = read(RUN / "tables/impact_sensitivity.csv")
    manifest = json.loads((RUN / "run_manifest.json").read_text())
    selection = json.loads((RUN / "tables/logistic_l2_selection.json").read_text())
    assert set(validation.index) == set(auc.index) == set(NAMES)
    selected = validation["Top@k%AUC"].idxmax()
    assert selected in (CB, CAT), "Review interpretation for a new model family selected on validation"
    reference = CB if selected != CB else CAT
    selected_name = NAMES[selected]
    portfolio_key = selected.replace(CB, "CatBoost + ES")
    assert auc.point_estimate.idxmax() == L2, "Update test-ranking interpretation for a new leading model"
    assert manifest["reproducibility"]["random_seed"] == 42
    assert manifest["reproducibility"]["model_threads"] == 1
    assert manifest["reproducibility"]["optuna_n_jobs"] == 1
    order = list(validation.sort_values("Top@k%AUC", ascending=False).index)
    c_mantissa, c_exponent = f"{selection['selected_parameters']['C']:.3e}".split("e")
    rb, lr = robust.loc["topk_percent_auc"], lr_robust.loc["topk_percent_auc"]
    hits = int(round(topk.loc[5, selected] * manifest["split_counts"]["TEST"]["events"]))
    v = {"selected": selected_name, "validation_auc": f"{validation.loc[selected, 'Top@k%AUC']:.3f}",
         "cb_auc": f"{auc.loc[selected, 'point_estimate']:.3f}",
         "es_auc": f"{auc.loc[CB, 'point_estimate']:.3f}",
         "l1_auc": f"{auc.loc[L1, 'point_estimate']:.3f}",
         "l2_auc": f"{auc.loc[L2, 'point_estimate']:.3f}",
         "rf_auc": f"{auc.loc[RF, 'point_estimate']:.3f}",
         "cat_auc": f"{auc.loc[CAT, 'point_estimate']:.3f}",
         "smote_auc": f"{auc.loc[SMOTE, 'point_estimate']:.3f}",
         "ranker_auc": f"{auc.loc[RANKER, 'point_estimate']:.3f}",
         "top1": f"{topk.loc[1, selected]:.3f}", "top5": f"{topk.loc[5, selected]:.3f}",
         "top10": f"{topk.loc[10, selected]:.3f}", "hits": hits,
         "hit_percent": f"{100 * topk.loc[5, selected]:.2f}",
         "smote_top1": f"{100 * topk.loc[1, SMOTE]:.2f}",
         "ranker_top1": f"{100 * topk.loc[1, RANKER]:.2f}",
         "robust_mean": f"{rb['mean']:.3f}", "robust_std": f"{rb['std']:.3f}",
         "robust_low": f"{rb['10%']:.3f}", "robust_high": f"{rb['90%']:.3f}",
         "lr_mean": f"{lr['mean']:.3f}", "lr_std": f"{lr['std']:.3f}",
         "lr_low": f"{lr['10%']:.3f}", "lr_high": f"{lr['90%']:.3f}",
         "early_auc": f"{subperiod.loc['early', 'topk_percent_auc']:.3f}",
         "early_events": int(subperiod.loc['early', 'n_pumps']),
         "l2_c": c_mantissa + r"\times10^{" + str(int(c_exponent)) + "}",
         "cb_l1": paired(pairwise, selected, L1), "cb_l2": paired(pairwise, selected, L2),
         "reference": NAMES[reference], "cb_cat": paired(pairwise, selected, reference),
         "selected_rf": paired(pairwise, selected, RF),
         "selected_smote": paired(pairwise, selected, SMOTE),
         "selected_ranker": paired(pairwise, selected, RANKER),
         "l1_l2": paired(pairwise, L1, L2)}
    headers = ["Model", *map(escaped, validation.columns)]
    rows = [[NAMES[m], *[(r"\bfseries " if np.isclose(validation.loc[m, col], validation[col].max()) else "")
                        + f"{validation.loc[m, col]:.3f}" for col in validation.columns]] for m in order]
    v["validation_table"] = table("table:validation_model_selection",
        r"Validation metrics for the seven tuned candidates. Top@k\%AUC selects the portfolio model; maxima are bold. F1 and balanced accuracy use one predicted target per event.", headers, rows, wide=True)
    pieces = [fill(r"""\section{Model Comparison}
\label{section:results}

\subsection{Validation Performance and Model Selection}
Table~\ref{table:validation_model_selection} compares seven tuned candidates using the same chronological split. @@selected@@ has the largest validation Top@k\%AUC ($@@validation_auc@@$). Validation guides model selection and early stopping; it is not an independent performance estimate. The selected model supplies all portfolio and price-impact results.

@@validation_table@@
""", v)]
    for filename, label, caption in [
        ("topk_accuracy_validation.png", "fig:validation_topk", r"Validation Top@k accuracy for the seven tuned candidates and the random-ranking baseline."),
        ("topkp_auc_validation.png", "fig:validation_topkp", r"Validation Top@k\% curves and normalized Top@k\%AUC. Model selection uses this validation criterion."),
        ("classification_metrics_validation.png", "fig:validation_classification", r"Validation PR-AUC, F1 and balanced accuracy for the seven candidates; classification uses one predicted target per event."),
    ]:
        pieces.append(figure(filename, label, caption))
    pieces.append(fill(r"""\subsection{Test Ranking Performance}
Tables~\ref{table:results_topk} and~\ref{table:results_topkp} report target inclusion at fixed portfolio sizes and cross-section fractions. @@selected@@ achieves Top@1 $@@top1@@$, Top@5 $@@top5@@$ (@@hits@@ of 80 events), and Top@10 $@@top10@@$. Test comparisons describe the selected models; they do not change the validation selection rule.
""", v))
    for frame, label, caption, percent in [
        (topk, "table:results_topk", "Test Top@k accuracy for the seven tuned candidates.", False),
        (topkp, "table:results_topkp", r"Test Top@k\% accuracy for the seven tuned candidates.", True),
    ]:
        columns = [1, 2, 5, 10, 20, 30] if not percent else list(frame.index)
        headers = ["Model", *[f"Top@{100 * k:g}\\%" if percent else f"Top@{k}" for k in columns]]
        rows = [[NAMES[m], *[f"{frame.loc[k, m]:.3f}" for k in columns]] for m in order]
        pieces.append(table(label, caption, headers, rows, wide=True))
    pieces.append(r"""The random-ranking baseline averages $\min(k,T_i)/T_i$ across events, where $T_i$ is the number of eligible assets in event $i$. Normalized Top@k\%AUC integrates the inclusion curve from 0 to 20\% in 0.5\% increments, as in Eq.~\ref{eq:topkpauc}.
""")
    pieces.append(figure("topk_accuracy_test.png", "fig:results_topk", "Test Top@k accuracy and the event-specific random-ranking baseline."))
    pieces.append(figure("topkp_auc_test.png", "fig:results_topkp", r"Test Top@k\% accuracy curves and normalized Top@k\%AUC for the same seven fitted candidates."))
    pieces.append(fill(r"""L2 logistic regression has the largest test Top@k\%AUC point estimate ($@@l2_auc@@$), followed by L1 logistic regression ($@@l1_auc@@$), CatBoost + ES ($@@es_auc@@$), CatBoost ($@@cat_auc@@$), and random forest ($@@rf_auc@@$). Statistical comparisons use paired event resampling in Section~\ref{sec:significance}.

\subsection{Classification Metrics}
Fig.~\ref{fig:classification_metrics} reports PR-AUC, F1 and balanced accuracy. F1 and balanced accuracy use the Top@1-per-event rule, so these scores measure the identification of one target within each event rather than an arbitrary probability threshold.
""", v))
    leader = classification["PR-AUC"].idxmax()
    pieces.append(f"{NAMES[leader]} has the highest test PR-AUC (${classification.loc[leader, 'PR-AUC']:.4f}$). "
                  f"{selected_name} has PR-AUC ${classification.loc[selected, 'PR-AUC']:.4f}$, "
                  f"F1 ${classification.loc[selected, 'F1 (Top@1 per cross-section)']:.4f}$ and balanced accuracy "
                  f"${classification.loc[selected, 'Balanced Accuracy (Top@1 per cross-section)']:.4f}$.\n")
    pieces.append(figure("classification_metrics_test.png", "fig:classification_metrics", "Test classification metrics for the seven tuned candidates."))
    pieces.append(fill(r"""\subsection{SMOTE and Ranking Models}
CatBoost with SMOTE attains test Top@1 $@@smote_top1@@\%$ and Top@k\%AUC $@@smote_auc@@$, below the leading class-weighted classifiers. With 81 predictors and 290 positive training events, synthetic interpolation may mix observations with different event-specific reference distributions. This is a plausible explanation, rather than a causal test, and is consistent with known limitations of SMOTE in high-dimensional data~\cite{blagus2013smote}.

The CatBoost ranker attains test Top@1 $@@ranker_top1@@\%$ and Top@k\%AUC $@@ranker_auc@@$. It learns an ordering of post-announcement returns, whereas evaluation asks whether the single manipulation target appears near the top. Noise in the returns of other assets and the mismatch between these objectives may explain its weaker performance. Ranking advantages in other financial applications~\cite{poh_ltr} do not automatically transfer to this sample.

\subsection{Interpretation of CatBoost + ES}
Fig.~\ref{fig:catboost_fi} reports CatBoost PredictionValuesChange for CatBoost + ES fitted on all training events. Table~\ref{table:feature_importance_quantiles} summarizes the 15 largest median importances across the 25 training-subset fits in Section~\ref{sec:robustness}. These describe the fitted prediction function; they are not causal effects or evidence of manipulation by themselves.
""", v))
    pieces.append(figure("feature_importances.png", "fig:catboost_fi", "Feature importance for the CatBoost + ES model fitted on all training events."))
    pieces.append(table("table:feature_importance_quantiles",
        r"CatBoost PredictionValuesChange quantiles across 25 fits on random 70\% training-event subsets. Rows are ordered by median importance.",
        ["Feature", r"10\%", r"25\%", "Median", r"75\%", r"90\%"],
        [[escaped(name), *[f"{row[col]:.2f}" for col in ["p10", "p25", "p50", "p75", "p90"]]]
         for name, row in importance.sort_values("p50", ascending=False).head(15).iterrows()], wide=True))
    pieces.append(r"""\subsection{Statistical Significance}
\label{sec:significance}
We validate the results using 10{,}000 paired bootstrap samples of the 80 whole test events. All candidate assets stay with their event, repeated events retain their multiplicity, and identical event draws are used for all seven fixed models. Models are not retrained within this bootstrap. The percentile intervals quantify sampling uncertainty conditional on the fitted models and assume independent events; they do not account for training uncertainty, temporal dependence or prior test inspection.

Pairwise differences use a two-sided null distribution centered at zero, an add-one Monte Carlo correction and Holm adjustment across the 21 unordered model pairs. We use paired differences rather than marginal-interval overlap to assess significance.
""")
    pieces.append(table("table:significance_ci",
        r"Test sample Top@k\%AUC point estimates and 95\% percentile intervals from 10{,}000 paired event resamples.",
        ["Model", r"Top@k\%AUC", r"95\% CI"],
        [[NAMES[m], f"{row.point_estimate:.3f}", f"$[{row.ci_lower:.3f},\\,{row.ci_upper:.3f}]$"]
         for m, row in auc.sort_values("point_estimate", ascending=False).iterrows()]))
    pieces.append(fill(r"""@@selected@@ minus L1 logistic regression has AUC difference @@cb_l1@@. Against L2 logistic regression, the difference is @@cb_l2@@. Against @@reference@@, the difference is @@cb_cat@@. The L1-minus-L2 difference is @@l1_l2@@. These comparisons do not establish superiority or equivalence. Model selection remains based on validation performance.

The selected model's differences against random forest, CatBoost with SMOTE and the ranker are, respectively: @@selected_rf@@; @@selected_smote@@; and @@selected_ranker@@. These three differences remain significant after Holm correction at the 5\% level, subject to the sampling assumptions and test-inspection limitation above.
""", v))
    pieces.append(figure("auc_distributions.png", "fig:auc_bootstrap", r"Test Top@k\%AUC bootstrap distributions for the seven fixed models: central 50\% and 95\% intervals, medians and test sample point estimates. All models use the same 10{,}000 event draws.", source=BOOT / "auc_distributions.png"))
    pieces.append(figure("significance_bootstrap_ci.png", "fig:bootstrap_ci", selected_name + r" test Top@k and Top@k\% point estimates with 95\% intervals from 10{,}000 whole-event resamples."))
    pieces.append(fill(r"""\subsection{Robustness Analysis}
\label{sec:robustness}
\subsubsection{Training-event sensitivity}
We retrain CatBoost + ES 25 times on random 70\% subsets of the training events, keeping hyperparameters and validation/test membership fixed. Table~\ref{table:robustness_summary} and Fig.~\ref{fig:robustness_distribution} summarize test performance. Mean AUC is $@@robust_mean@@$, its standard deviation is $@@robust_std@@$, and the central 80\% range is $[@@robust_low@@,\,@@robust_high@@]$.
""", v))
    robust_labels = {"topk_percent_auc": r"Top@k\%AUC", **{f"topk_percent@{k}": f"Top@{100*k:g}\\%" for k in [0.01, 0.02, 0.05, 0.1, 0.2]}}
    pieces.append(table("table:robustness_summary", r"Test performance across 25 random 70\% training-event subset fits of CatBoost + ES.",
        ["Metric", "Mean", "Std", r"[10\%, 90\%]"],
        [[robust_labels[name], f"{row['mean']:.3f}", f"{row['std']:.3f}", f"[{row['10%']:.3f}, {row['90%']:.3f}]"]
         for name, row in robust.iterrows() if name in robust_labels]))
    pieces.append(figure("robustness_metrics_distribution.png", "fig:robustness_distribution", r"Distribution of test ranking metrics across 25 CatBoost + ES fits on different 70\% training-event subsets, with validation and test membership fixed."))
    pieces.append(fill(r"""The corresponding L2 logistic regression analysis keeps the selected penalty, $C$, and class weights fixed. Its mean test AUC is $@@lr_mean@@$, standard deviation $@@lr_std@@$, and central 80\% range $[@@lr_low@@,\,@@lr_high@@]$. Preprocessing is fitted using each training subset. These experiments measure conditional training sensitivity and do not provide independent external validation.

\subsubsection{Temporal subperiod analysis}
Using June~1, 2022 as the subperiod boundary, the early test period contains @@early_events@@ events and has @@selected@@ AUC $@@early_auc@@$. The late period contains only four events and is excluded by the minimum of ten events per subperiod. This sample therefore does not support a reliable claim about bear-market or cross-regime performance.
""", v))
    start = source.index(r"\section{Model Comparison}")
    end = source.index(r"\section{Portfolio strategy}")
    source = source[:start] + "\n".join(pieces) + "\n" + source[end:]

    # All subsequent result sections refer to the same validation-selected fit.
    metrics = ["average event return", "cumulative return", "annualized return", "annualized volatility", "Sharpe ratio"]
    portfolio_headers = ["Portfolio", r"Avg. event return", "Cumulative return", "Annualized return", "Annualized volatility", "Sharpe"]
    portfolio_rows = [[f"{selected_name} $k={k}$", *[f"{portfolios.loc[f'{portfolio_key} K={k}', col]:.4f}" for col in metrics]] for k in SIZES]
    portfolio_rows += [[f"Random $k={k}$", *[f"{portfolios.loc[f'Random K={k} (mean)', col]:.4f}" for col in metrics]] for k in SIZES]
    btc = portfolios.loc["BTCUSDT buy-and-hold"]
    btc_row = ["BTCUSDT buy-and-hold", "---", *[f"{btc[col]:.4f}" for col in metrics[1:]]]
    portfolio_rows.append(btc_row)
    portfolio_table = table("table:performance_by_k",
        r"Portfolio performance without size-dependent impact. Random rows average statistics computed separately for 10{,}000 draws. All returns are decimal fractions.", portfolio_headers, portfolio_rows, wide=True)
    model5 = portfolios.loc[f"{portfolio_key} K=5"]
    v.update({"model_return": f"{100*model5['cumulative return']:.2f}", "model_sharpe": f"{model5['Sharpe ratio']:.2f}",
              "impact_return": f"{100*impact.loc[5, 'cumulative return']:.2f}", "impact_sharpe": f"{impact.loc[5, 'Sharpe ratio']:.2f}",
              "random_return": f"{100*random_metrics.loc[(5,'cumulative return'),'mean']:.2f}",
              "random_low": f"{100*random_metrics.loc[(5,'cumulative return'),'lower']:.2f}",
              "random_high": f"{100*random_metrics.loc[(5,'cumulative return'),'upper']:.2f}",
              "random_hits": f"{100*random_metrics.loc[(5,'Top@k hit rate'),'mean']:.2f}",
              "impact_random": f"{100*impact_summary.loc[(5,'cumulative return'),'mean']:.2f}",
              "impact_random_low": f"{100*impact_summary.loc[(5,'cumulative return'),'lower']:.2f}",
              "impact_random_high": f"{100*impact_summary.loc[(5,'cumulative return'),'upper']:.2f}"})
    start = source.index("We use ", source.index(r"\section{Portfolio strategy}"))
    end = source.index(r"\subsection{Execution-Aware Backtesting with Price Impact}")
    portfolio_text = fill(r"""We use the validation-selected @@selected@@ model for every portfolio and price-impact analysis. We compare $k\in\{1,2,5,10,20,30\}$ and retain $k=5$ for order-size sensitivity. At $k=5$, target inclusion is $@@hit_percent@@\%$ (@@hits@@ of 80 events), exactly matching Top@5 in Table~\ref{table:results_topk}.

For each size, we generate 10{,}000 random portfolios. At each event, $k$ different eligible assets are selected uniformly and assigned equal weights, using common random asset orders across sizes. Trading rules, USDT conversion and the 25 bp round-trip cost match the model strategy. Unfilled allocations remain in cash. Events stay in chronological order and returns are summed without reinvestment. Annualized statistics use zero-filled daily returns over the same calendar and 365 days per year; each statistic is computed for each draw before averaging.

At $k=5$, @@selected@@ cumulative return is $@@model_return@@\%$ and Sharpe is $@@model_sharpe@@$. Random selection has mean return $@@random_return@@\%$ with a central 95\% range of $[@@random_low@@\%,\,@@random_high@@\%]$ and mean target inclusion $@@random_hits@@\%$. These ranges describe random selection within the observed test period, not event-bootstrap confidence intervals or confirmatory significance tests.
""", v)
    portfolio_text += portfolio_table + figure("portfolios_vs_btc_test.png", "fig:portfolio_test",
        selected_name + r" portfolio cumulative returns, random selection and BTC/USDT over the same calendar. Returns include 25 bp round-trip costs and exclude size-dependent impact. The dashed curve averages 10{,}000 random $k=5$ portfolios; shading is the pointwise 95\% selection range. The solid black line is BTC/USDT buy-and-hold.")
    portfolio_text += (f"BTC/USDT buy-and-hold has cumulative return ${btc['cumulative return']:.3f}$, "
                       f"annualized return ${btc['annualized return']:.3f}$, annualized volatility "
                       f"${btc['annualized volatility']:.3f}$ and Sharpe ${btc['Sharpe ratio']:.3f}$. "
                       "Its calendar runs from the first strategy decision to the last exit using completed one-minute conversion bars. "
                       "BTC remains invested between events, whereas the model portfolios are exposed only during their event windows.\n\n")
    source = source[:start] + portfolio_text + source[end:]
    start = min(source.index(marker) for marker in (
        r"\subsubsection{Portfolio performance with price impact}",
        r"\subsubsection{PnL sensitivity to order size}",
    ) if marker in source)
    end = source.index(r"\section{Limitations and Future Work}")
    impact_rows = [[f"{selected_name} $k={k}$", *[f"{impact.loc[k,col]:.4f}" for col in metrics]] for k in SIZES]
    impact_rows += [[f"Random $k={k}$", *[f"{impact_random.loc[(k,col),'random mean']:.4f}" for col in metrics]] for k in SIZES]
    impact_rows.append(btc_row)
    tail = fill(r"""\subsubsection{Portfolio performance with price impact}
We apply the fitted impact model to both execution legs with 1{,}000 USDT per selected asset, so a size-$k$ portfolio targets $k\times1{,}000$ USDT per event. Predictions, event order, random selections and other trading rules are identical to the no-impact comparison. At $k=5$, @@selected@@ cumulative return is $@@impact_return@@\%$ and Sharpe is $@@impact_sharpe@@$. Random selection has mean return $@@impact_random@@\%$, with a central 95\% range of $[@@impact_random_low@@\%,\,@@impact_random_high@@\%]$.
""", v)
    tail += table("table:performance_by_k_impact", r"Portfolio performance with 1{,}000 USDT per asset, fitted entry and exit impact, and 25 bp costs. Random rows average 10{,}000 draws; returns are decimal fractions.", portfolio_headers, impact_rows, wide=True)
    tail += figure("portfolios_1000usdt_with_slippage.png", "fig:portfolio_test_impact",
        r"The same portfolios and random selections as Fig.~\ref{fig:portfolio_test}, with fitted price impact on both legs at 1{,}000 USDT per asset. Shading is the pointwise 95\% random-selection range; the BTC/USDT benchmark is unchanged.", source=IMPACT / "portfolios_1000usdt_with_slippage.png")
    sensitivity = sensitivity.set_index("quantity_usdt") if "quantity_usdt" in sensitivity.columns else sensitivity
    tail += r"\subsubsection{PnL sensitivity to order size}" + "\n"
    tail += ("For $k=5$, we vary intended size from 100 to 10{,}000 USDT per asset. "
             f"Cumulative ROE without reinvestment is ${sensitivity.loc[100,'cumulative_roe']:.3f}$ at 100 USDT, "
             f"${sensitivity.loc[2000,'cumulative_roe']:.3f}$ at 2{{,}}000 USDT, "
             f"${sensitivity.loc[5000,'cumulative_roe']:.3f}$ at 5{{,}}000 USDT, and "
             f"${sensitivity.loc[10000,'cumulative_roe']:.3f}$ at 10{{,}}000 USDT. "
             "These conditional simulation outcomes depend on fill and impact assumptions and do not establish executable capacity.\n")
    tail += figure("pnl_vs_quantity_with_impact.png", "fig:pnl_vs_quantity",
        selected_name + r" cumulative ROE over the 80 test events as intended per-asset notional changes. The $k=5$ strategy includes fitted entry and exit impact and 25 bp costs; returns are summed without reinvestment.", wide=False)
    source = source[:start] + tail + "\n" + source[end:]
    source = re.sub(r"^(?:In the model comparison, validation|Validation) Top@k.*$", lambda _: fill(
        r"Validation Top@k\%AUC selects @@selected@@ ($@@validation_auc@@$). It attains test Top@1 $@@top1@@$, Top@10 $@@top10@@$, and Top@k\%AUC $@@cb_auc@@$ on 80 events. L2 logistic regression has the highest descriptive test AUC ($@@l2_auc@@$); paired resampling does not establish a significant AUC difference between these two models.", v), source, flags=re.M)
    source = re.sub(r"^This study compares (?:established|seven tuned).*$", lambda _: fill(
        r"This study compares seven tuned imbalance-aware prediction models on the same Binance BTC-pair events. Validation selects @@selected@@ (Top@k\%AUC $@@validation_auc@@$), which achieves test AUC $@@cb_auc@@$ and Top@5 $@@top5@@$. L2 logistic regression has the largest descriptive test AUC ($@@l2_auc@@$), but paired resampling does not establish its superiority over @@selected@@. The same selected model supplies all portfolio analyses: its $k=5$ cumulative return is $@@model_return@@\%$ without size-dependent impact and $@@impact_return@@\%$ with fitted impact at 1{,}000 USDT per asset. These simulation outcomes are sensitive to execution assumptions and order size.", v), source, flags=re.M)
    source = re.sub(r"^The repository provides numerical tables.*$",
        "The repository provides the event cohort, split membership, software versions, seed settings, feature hashes, model parameters, test predictions, shared bootstrap draws, and random-portfolio selections and execution outcomes. The same saved predictions underlie the test comparison, resampling and portfolios. The event records cannot independently validate Telegram annotations because the underlying messages and URLs are unavailable.", source, flags=re.M)
    method = fill(r"We compare tuned L1 and L2 logistic regression, random forest, CatBoost, CatBoost with SMOTE, CatBoost ranker, and CatBoost + ES. L2 uses the L1 predictors and class weights and selects $C$ on validation Top@k\%AUC from 25 logarithmically spaced values between $10^{-6}$ and $10^2$, breaking ties toward smaller $C$. The selected value is $C=@@l2_c@@$. The standalone unweighted, unpenalized statsmodels diagnostic is excluded from predictive comparisons.", v)
    anchor = "We validated the results using event-level resampling"
    if "% BEGIN MODEL CONFIGURATION" in source:
        source = re.sub(r"% BEGIN MODEL CONFIGURATION.*?% END MODEL CONFIGURATION", lambda _: "% BEGIN MODEL CONFIGURATION\n" + method + "\n% END MODEL CONFIGURATION", source, flags=re.S)
    else:
        source = source.replace(anchor, "% BEGIN MODEL CONFIGURATION\n" + method + "\n% END MODEL CONFIGURATION\n\n" + anchor)
    # Refresh descriptive figures; the sample counts are independently audited.
    for filename in ["cs_size.jpeg", "cs_standardization_impact.png", "impact_regression_example.png", "exit_impact_regression_example.png"]:
        source = source.replace("images/submitted_" + filename, "images/" + filename)
        FIGURES[str((HERE / "images" / filename).relative_to(ROOT))] = RUN / "images" / filename
    splits = manifest["split_counts"]
    dataset_rows = []
    for title, key in [("Positive", "positive"), ("Negative", "negative"), ("Total", "rows"),
                       ("Avg. Cross-section Size", "average_cross_section")]:
        dataset_rows.append([title, *[f"{splits[kind][key]:.2f}" if key == "average_cross_section"
                                      else str(splits[kind][key]) for kind in ["TRAIN", "VALIDATION", "TEST"]]])
    source = replace_float(source, "table:train_val_test_stats", table(
        "table:train_val_test_stats", r"P\&D training, validation and test sample statistics.",
        ["", "Train", "Validation", "Test"], dataset_rows))
    average = sum(part["rows"] for part in splits.values()) / sum(part["events"] for part in splits.values())
    source = re.sub(r"average eligible cross-section contains [\d.]+ instruments",
                    f"average eligible cross-section contains {average:.2f} instruments", source)
    assert "images/submitted_" not in source
    assert "sec:additional_" not in source
    return source, manifest, v


def results_report(manifest, values):
    validation = read(RUN / "tables/validation_metrics.csv")
    summary = read(BOOT / "auc_summary.csv")
    topk = read(RUN / "tables/test_topk.csv")
    portfolio = read(RUN / "tables/portfolio_statistics.csv")
    impact = read(IMPACT / "model_statistics.csv")
    selected = validation["Top@k%AUC"].idxmax()
    key = selected.replace(CB, "CatBoost + ES")
    lines = [
        "# Results: fixed-seed notebook run",
        "",
        "The manuscript uses this completed run throughout. The submitted PDF is retained only as the highlighting baseline. "
        "The cohort contains 473 events from 498 announcements and 25 recorded exclusions: 290 training, 103 validation and 80 test events.",
        "",
        f"Validation selects **{NAMES[selected]}** (Top@k%AUC **{validation.loc[selected, 'Top@k%AUC']:.4f}**). "
        "Its saved predictions supply all portfolio analyses. Each model's point estimates and bootstrap intervals use that model's same saved predictions.",
        "",
        "Training and resampling use seed 42 and one CPU thread; Optuna searches are configured for sequential seeded TPE trials. "
        "This execution fits the six saved study configurations and reruns the L2 validation grid; it does not rerun the six Optuna searches. "
        "The study database and feature inputs are unchanged. Fitted model files, exact feature order, package versions, input hashes and executed notebooks are saved with the run.",
        "",
        "## Model comparisons",
        "",
        "| Model | Validation AUC | Test AUC | Test 95% CI | Test Top@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, row in summary.iterrows():
        lines.append(f"| {NAMES[model]} | {validation.loc[model, 'Top@k%AUC']:.4f} | {row.point_estimate:.4f} | "
                     f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}] | {topk.loc[5, model]:.4f} |")
    lines += ["", "AUC integrates Top@k% from 0–20% in 0.5% increments and divides by 0.2. "
              "The percentile intervals use 10,000 paired whole-event draws shared by all seven fixed models. "
              "Two-sided centered tests use an add-one correction and Holm adjustment across 21 pairs. "
              "The intervals assume independent events and omit training uncertainty; prior test inspection limits confirmatory interpretation.",
              "", "## Portfolios", "",
              "| k | Cumulative return, no size impact | Sharpe | Cumulative return, 1,000 USDT/asset impact | Sharpe | Target events |",
              "|---:|---:|---:|---:|---:|---:|"]
    for k in SIZES:
        row = portfolio.loc[f"{key} K={k}"]
        lines.append(f"| {k} | {row['cumulative return']:.4f} | {row['Sharpe ratio']:.4f} | "
                     f"{impact.loc[k, 'cumulative return']:.4f} | {impact.loc[k, 'Sharpe ratio']:.4f} | "
                     f"{round(topk.loc[k, selected]*80)}/80 |")
    lines += ["", "Returns are decimal fractions, summed without reinvestment, with 25 bp round-trip costs. "
              "Each portfolio size also has 10,000 uniform random-selection draws. Asset orders are shared across sizes and both execution scenarios. "
              "Daily annualization uses a zero-filled common calendar and 365 days per year. Random-selection ranges are conditional benchmarks, not event-bootstrap intervals.",
              "", "## Sensitivity and verification", "",
              f"CatBoost + ES training-subset AUC: mean {values['robust_mean']}, standard deviation {values['robust_std']}, "
              f"central 80% range [{values['robust_low']}, {values['robust_high']}]. "
              f"L2: mean {values['lr_mean']}, standard deviation {values['lr_std']}, range [{values['lr_low']}, {values['lr_high']}]. "
              "Both use 25 random 70% training-event subsets with fixed hyperparameters and validation/test membership.",
              "",
              f"The selected model's early-period AUC is {values['early_auc']} on {values['early_events']} events. "
              "The four-event late period is excluded by the ten-event minimum.",
              "",
              "[Prediction and resampling replay audit](notebook_rerun_audit.json) independently reconstructs all point metrics, "
              "10,000 bootstrap draws, 21 paired tests, portfolio returns and target counts, and random selections. "
              "[Manuscript audit](manuscript_results_audit.json) checks every generated result table and figure against the exports. "
              "[Reproduction commands](reproduction.md) regenerate the outputs and both PDFs.",
              "",
              "Run artifacts: `notebooks/analysis_outputs/paper_cohort_473_l2/` (local, excluded from Git). "
              "`paper_run/models/` stores all seven fitted models; `paper_run/source_snapshot/` preserves sources used by the execution. "
              "`test_auc_bootstrap/` contains predictions and exact event draws, and the two portfolio directories contain execution outcomes and exact random selections.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = HERE / "access.tex"
    source = path.read_text()
    updated, manifest, values = render(source)
    report_path = ROOT / "docs/results.md"
    report_text = results_report(manifest, values)
    if not args.check:
        path.write_text(updated)
        report_path.write_text(report_text)
        for destination, origin in FIGURES.items():
            shutil.copy2(origin, ROOT / destination)
    else:
        assert updated == source, "Manuscript results do not match current run"
        assert report_path.read_text() == report_text, "Results report does not match current run"
    blocks = {b["label"]: b for b in floats(normalized_lines(path.read_text()))}
    cells = []
    for label, expected in TABLES.items():
        expected_block = floats(normalized_lines(expected))[0]
        assert table_body(blocks[label]) == table_body(expected_block), label
        for line in table_body(expected_block):
            if " & " in line:
                cells.append({"table": label, "row": line.strip()})
    for destination, origin in FIGURES.items():
        assert digest(ROOT / destination) == digest(origin), destination
    report = {"status": "PASS", "run_manifest_sha256": digest(RUN / "run_manifest.json"),
              "paper_source_sha256": digest(path), "reproducibility": manifest["reproducibility"],
              "tables": list(TABLES), "table_rows": cells, "key_results": values,
              "input_sha256": INPUTS, "figure_sources": {k: str(v.relative_to(ROOT)) for k, v in FIGURES.items()},
              "previous_submission_used_only_as_highlight_baseline": True}
    # Do not duplicate generated table TeX inside the compact audit summary.
    report["key_results"].pop("validation_table", None)
    (ROOT / "docs/manuscript_results_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(TABLES)} tables and {len(FIGURES)} figures match the completed run.")


if __name__ == "__main__":
    main()
