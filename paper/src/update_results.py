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

import pandas as pd

from build_highlighted import floats, normalized_lines, table_body
from prepare_seed_results import COMPARISON, RUN, IMPACT, OUTPUT, MODEL_NAMES, prepare
from layout import apply_layout, strip_layout
from compact_figures import FILENAMES, SINGLE_COLUMN, OUTPUT as COMPACT_OUTPUT, prepare_compact_figures

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BOOT = OUTPUT / "test_auc_bootstrap"
CB = "CatBoost classifier + Top@k%AUC early stopping"
L1 = "Logistic regression + Tuned"
RF = "Random forest + Tuned"
CAT = "CatBoost classifier + Tuned"
SMOTE = "CatBoost classifier + SMOTE + Tuned"
RANKER = "CatBoost ranker + Tuned"
NAMES = {CB: "CatBoost + ES", L1: "Logistic regression",
         RF: "Random forest", CAT: "CatBoost", SMOTE: "CatBoost + SMOTE", RANKER: "CatBoost ranker"}
SIZES = [1, 2, 5, 10, 20, 30]
TABLES = {}
INPUTS = {}
FIGURES = {}
COMPACT_FIGURES = {
    "topk_accuracy_validation.png", "topkp_auc_validation.png",
    "topk_accuracy_test.png", "topkp_auc_test.png",
    "auc_distributions.png",
    "validation_seed_distributions.png", "test_seed_distributions.png",
}


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
    comparison_images = COMPACT_FIGURES | {"classification_metrics_validation.png", "classification_metrics_test.png"}
    source = source or (COMPARISON if filename in comparison_images else RUN) / "images" / filename
    if filename in FILENAMES:
        source = COMPACT_OUTPUT / "images" / filename
    if filename in SINGLE_COLUMN:
        wide = False
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
    validation = read(COMPARISON / "tables/validation_metrics.csv")
    validation_seeds = read(COMPARISON / "tables/validation_seed_distributions.csv")
    test_seeds = read(COMPARISON / "tables/test_seed_distributions.csv")
    validation_seed_pairs = pd.read_csv(COMPARISON / "tables/validation_paired_seed_differences.csv")
    test_seed_pairs = pd.read_csv(COMPARISON / "tables/test_paired_seed_differences.csv")
    topk = read(COMPARISON / "tables/test_topk.csv")
    topkp = read(COMPARISON / "tables/test_topkp.csv")
    classification = read(COMPARISON / "tables/test_classification.csv")
    auc = read(COMPARISON / "tables/auc_summary.csv")
    pairwise = pd.read_csv(COMPARISON / "tables/auc_pairwise.csv")
    INPUTS[str((COMPARISON / "tables/auc_pairwise.csv").relative_to(ROOT))] = digest(COMPARISON / "tables/auc_pairwise.csv")
    robust = read(RUN / "tables/robustness_summary.csv")
    importance = read(RUN / "tables/feature_importance_quantiles.csv")
    subperiod = read(RUN / "tables/subperiod_metrics.csv")
    portfolios = read(RUN / "tables/portfolio_statistics.csv")
    impact = read(IMPACT / "model_statistics.csv")
    impact_random = pd.read_csv(IMPACT / "model_vs_random.csv").set_index(["k", "metric"])
    random_metrics = pd.read_csv(RUN / "tables/random_portfolio_metrics.csv").set_index(["k", "metric"])
    impact_summary = pd.read_csv(IMPACT / "random_portfolio/summary.csv").set_index(["k", "metric"])
    sensitivity = read(RUN / "tables/impact_sensitivity.csv")
    manifest = json.loads((RUN / "run_manifest.json").read_text())
    assert set(validation.index) == set(auc.index) == set(NAMES)
    selected = MODEL_NAMES[manifest["selection"]["model"]]
    assert manifest["selection"]["model"] == validation_seeds["mean"].idxmax()
    assert selected in (CB, CAT), "Review interpretation for a new model family selected on validation"
    for relative in ("tables/selected_model_analysis.json",):
        provenance_path = RUN / relative
        analysis = json.loads(provenance_path.read_text())
        assert analysis["model"] == selected, "Interpretation and portfolios must use the validation winner"
        assert analysis["parameters"] == manifest["model_parameters"][selected]
        INPUTS[str(provenance_path.relative_to(ROOT))] = digest(provenance_path)
    reference = CB if selected != CB else CAT
    selected_name = NAMES[selected]
    portfolio_key = selected.replace(CB, "CatBoost + ES")
    assert auc.point_estimate.idxmax() == L1, "Update test-ranking interpretation for a new leading model"
    assert manifest["reproducibility"]["random_seed"] == 42
    assert manifest["reproducibility"]["model_threads"] == 1
    assert manifest["reproducibility"]["optuna_n_jobs"] == 1
    order = [MODEL_NAMES[key] for key in validation_seeds.sort_values("mean", ascending=False).index]
    rb = robust.loc["topk_percent_auc"]
    hits = int(round(topk.loc[5, selected] * manifest["split_counts"]["TEST"]["events"]))
    v = {"selected": selected_name, "validation_auc": f"{validation.loc[selected, 'Top@k%AUC']:.3f}",
         "validation_mean": f"{validation_seeds.loc['catboost', 'mean']:.4f}",
         "validation_sd": f"{validation_seeds.loc['catboost', 'std']:.4f}",
         "l1_validation_mean": f"{validation_seeds.loc['logistic', 'mean']:.4f}",
         "cat_test_mean": f"{test_seeds.loc['catboost', 'mean']:.4f}",
         "l1_test_mean": f"{test_seeds.loc['logistic', 'mean']:.4f}",
         "l1_test_sd": f"{test_seeds.loc['logistic', 'std']:.4f}",
         "cb_auc": f"{auc.loc[selected, 'point_estimate']:.3f}",
         "es_auc": f"{auc.loc[CB, 'point_estimate']:.3f}",
         "l1_auc": f"{auc.loc[L1, 'point_estimate']:.3f}",
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
         "early_auc": f"{subperiod.loc['early', 'topk_percent_auc']:.3f}",
         "early_events": int(subperiod.loc['early', 'n_pumps']),
         "cb_l1": paired(pairwise, selected, L1),
         "reference": NAMES[reference], "cb_cat": paired(pairwise, selected, reference),
         "selected_rf": paired(pairwise, selected, RF),
         "selected_smote": paired(pairwise, selected, SMOTE),
         "selected_ranker": paired(pairwise, selected, RANKER)}
    headers = ["Model", "Mean AUC", "SD", r"2.5\%", r"97.5\%", r"Highest AUC (\% of seeds)"]
    rows = [[NAMES[MODEL_NAMES[key]], *[f"{row[col]:.4f}" for col in ["mean", "std", "2.5%", "97.5%"]],
             f"{100*row.win_fraction:.0f}"] for key, row in validation_seeds.sort_values("mean", ascending=False).iterrows()]
    v["validation_table"] = table("table:validation_model_selection",
        r"Validation Top@k\%AUC across 100 seeds per model. Quantiles and standard deviations describe seed variation on the same 103 events. Model selection uses the highest mean AUC.", headers, rows, wide=True)
    pieces = [fill(r"""\section{Model Comparison}
\label{section:results}

\subsection{Validation Performance and Model Selection}
Table~\ref{table:validation_model_selection} compares six model families across seeds 1--100 on the same 103 validation events. @@selected@@ has the largest mean validation Top@k\%AUC ($@@validation_mean@@$, SD $@@validation_sd@@$), followed by logistic regression ($@@l1_validation_mean@@$). Hyperparameters and early stopping are determined within the training sample; the outer validation sample is used only to compare model families. We select the family with the highest mean validation AUC and use its preselected seed-42 model for feature interpretation, event resampling and all portfolio analyses. Seed 42 is fixed independently of its validation or test score.

@@validation_table@@
""", v)]
    val_pair = validation_seed_pairs.query("model_a == 'logistic' and model_b == 'catboost'").iloc[0]
    pieces.append(f"CatBoost minus logistic regression has mean validation AUC difference ${-val_pair.mean_difference:.4f}$, "
                  f"with a paired-seed 95\\% interval $[{-val_pair.ci_upper:.4f},\\,{-val_pair.ci_lower:.4f}]$. "
                  "This interval includes zero: the highest observed mean determines the selection, but does not establish a clear advantage over logistic regression. "
                  r"Fig.~\ref{fig:validation_seed_distribution} shows the full distributions." + "\n")
    pieces.append(figure("validation_seed_distributions.png", "fig:validation_seed_distribution",
                        r"Validation Top@k\%AUC for 100 seeds per model. Each seed uses a 25-trial search within training followed by a fit on all 290 training events. Boxes show the interquartile range and median; triangles show means; dots show individual seeds."))
    pieces.append(fill(r"""For the fixed seed-42 models, @@selected@@ validation AUC is $@@validation_auc@@$. The following threshold curves describe these fixed models; the family selection above uses all 100 seeds.
""", v))
    for filename, label, caption in [
        ("topk_accuracy_validation.png", "fig:validation_topk", r"Validation Top@k accuracy for the six seed-42 models and the random-ranking baseline."),
        ("topkp_auc_validation.png", "fig:validation_topkp", r"Validation Top@k\% curves and normalized Top@k\%AUC for the six seed-42 models."),
    ]:
        descriptions = {
            "fig:validation_topk": r"Fig.~\ref{fig:validation_topk} shows how validation target inclusion changes with the number of selected assets.",
            "fig:validation_topkp": r"Fig.~\ref{fig:validation_topkp} compares the validation ranking curves as a fraction of each event's eligible assets.",
        }
        pieces.append(descriptions[label] + "\n")
        pieces.append(figure(filename, label, caption))
    pieces.append(fill(r"""\subsection{Test Ranking Performance}
Table~\ref{table:test_seed_distribution} and Fig.~\ref{fig:test_seed_distribution} summarize test performance across the same 100 seeds. Each configuration is trained on the same 290 training events and evaluated on the same 80 test events, without retuning or adding validation events to training. Logistic regression has the highest mean test AUC ($@@l1_test_mean@@$, SD $@@l1_test_sd@@$), followed by CatBoost ($@@cat_test_mean@@$). The test ranking does not change the validation-based model selection.
""", v))
    rows = [[NAMES[MODEL_NAMES[key]], *[f"{row[col]:.4f}" for col in ["mean", "std", "2.5%", "97.5%"]],
             f"{100*row.win_fraction:.0f}"] for key, row in test_seeds.sort_values("mean", ascending=False).iterrows()]
    pieces.append(table("table:test_seed_distribution", r"Test Top@k\%AUC across 100 seeds per model on 80 fixed events. Quantiles describe variation across tuning and training seeds, not event-bootstrap confidence intervals.", headers, rows, wide=True))
    pieces.append(figure("test_seed_distributions.png", "fig:test_seed_distribution",
        r"Test Top@k\%AUC for the same 100 seed-specific configurations. The asterisk marks the family selected on validation. Boxes show the interquartile range and median; triangles show means; dots show individual seeds."))
    test_pair = test_seed_pairs.query("model_a == 'logistic' and model_b == 'catboost'").iloc[0]
    pieces.append(f"Logistic regression minus CatBoost has mean test AUC difference ${test_pair.mean_difference:.4f}$, "
                  f"with paired-seed 95\\% interval $[{test_pair.ci_lower:.4f},\\,{test_pair.ci_upper:.4f}]$. "
                  "These intervals use 10{,}000 paired resamples of the 100 seed indices and hold the event sample fixed. "
                  "They measure tuning and training variability; event-sampling uncertainty for fixed models is assessed separately below.\n")
    pieces.append(fill(r"""Tables~\ref{table:results_topk} and~\ref{table:results_topkp} report target inclusion for the seed-42 models at fixed portfolio sizes and cross-section fractions. @@selected@@ achieves Top@1 $@@top1@@$, Top@5 $@@top5@@$ (@@hits@@ of 80 events), and Top@10 $@@top10@@$.
""", v))
    for frame, label, caption, percent in [
        (topk, "table:results_topk", "Test Top@k accuracy for the six seed-42 models.", False),
        (topkp, "table:results_topkp", r"Test Top@k\% accuracy for the six seed-42 models.", True),
    ]:
        columns = [1, 2, 5, 10, 20, 30] if not percent else list(frame.index)
        headers = ["Model", *[f"Top@{100 * k:g}\\%" if percent else f"Top@{k}" for k in columns]]
        rows = [[NAMES[m], *[f"{frame.loc[k, m]:.3f}" for k in columns]] for m in order]
        pieces.append(table(label, caption, headers, rows, wide=True))
    pieces.append(r"""Figs.~\ref{fig:results_topk} and~\ref{fig:results_topkp} show the corresponding test ranking curves. The random-ranking baseline averages $\min(k,T_i)/T_i$ across events, where $T_i$ is the number of eligible assets in event $i$. Normalized Top@k\%AUC integrates the inclusion curve from 0 to 20\% in 0.5\% increments, as in Eq.~\ref{eq:topkpauc}.
""")
    pieces.append(figure("topk_accuracy_test.png", "fig:results_topk", "Test Top@k accuracy and the event-specific random-ranking baseline."))
    pieces.append(figure("topkp_auc_test.png", "fig:results_topkp", r"Test Top@k\% accuracy curves and normalized Top@k\%AUC for the same six fitted candidates."))
    pieces.append(fill(r"""For seed 42, test Top@k\%AUC is $@@l1_auc@@$ for logistic regression, $@@cat_auc@@$ for CatBoost, $@@es_auc@@$ for CatBoost + ES, and $@@rf_auc@@$ for random forest. Statistical comparisons of these fixed models use paired event resampling in Section~\ref{sec:significance}.

\subsection{Classification Metrics}
Fig.~\ref{fig:classification_metrics} reports PR-AUC, F1 and balanced accuracy. F1 and balanced accuracy use the Top@1-per-event rule, so these scores measure the identification of one target within each event rather than an arbitrary probability threshold.
""", v))
    leader = classification["PR-AUC"].idxmax()
    if leader != selected:
        pieces.append(f"{NAMES[leader]} has the highest test PR-AUC (${classification.loc[leader, 'PR-AUC']:.4f}$). ")
    pr_auc_label = "the highest test PR-AUC" if leader == selected else "PR-AUC"
    pieces.append(f"{selected_name} has {pr_auc_label} (${classification.loc[selected, 'PR-AUC']:.4f}$). "
                  f"Its F1 is ${classification.loc[selected, 'F1 (Top@1 per cross-section)']:.4f}$ and balanced accuracy is "
                  f"${classification.loc[selected, 'Balanced Accuracy (Top@1 per cross-section)']:.4f}$.\n")
    pieces.append(figure("classification_metrics_test.png", "fig:classification_metrics", "Test classification metrics for the six seed-42 models."))
    pieces.append(fill(r"""\subsection{SMOTE and Ranking Models}
CatBoost with SMOTE attains test Top@1 $@@smote_top1@@\%$ and Top@k\%AUC $@@smote_auc@@$, below the leading class-weighted classifiers. With 81 predictors and 290 positive training events, synthetic interpolation may mix observations with different event-specific reference distributions. This is a plausible explanation, rather than a causal test, and is consistent with known limitations of SMOTE in high-dimensional data~\cite{blagus2013smote}.

The CatBoost ranker attains test Top@1 $@@ranker_top1@@\%$ and Top@k\%AUC $@@ranker_auc@@$. It learns an ordering of post-announcement returns, whereas evaluation asks whether the single manipulation target appears near the top. Noise in the returns of other assets and the mismatch between these objectives may explain its weaker performance. Ranking advantages in other financial applications~\cite{poh_ltr} do not automatically transfer to this sample.

\subsection{Interpretation of @@selected@@}
Table~\ref{table:feature_importance_quantiles} summarizes the 15 largest median PredictionValuesChange importances across the 100 seed-specific CatBoost models. These describe the fitted prediction functions; they are not causal effects or evidence of manipulation by themselves.
""", v))
    pieces.append(table("table:feature_importance_quantiles",
        selected_name + r" PredictionValuesChange quantiles across 100 seed-specific fits on the same 290 training events. Rows are ordered by median importance.",
        ["Feature", r"10\%", r"25\%", "Median", r"75\%", r"90\%"],
        [[escaped(name), *[f"{row[col]:.2f}" for col in ["p10", "p25", "p50", "p75", "p90"]]]
         for name, row in importance.sort_values("p50", ascending=False).head(15).iterrows()], wide=True))
    pieces.append(r"""\subsection{Statistical Significance}
\label{sec:significance}
We validate the seed-42 point results using 10{,}000 paired bootstrap samples of the 80 whole test events. All candidate assets stay with their event, repeated events retain their multiplicity, and identical event draws are used for all six fixed models. Models are not retrained within this bootstrap. The percentile intervals quantify event-sampling uncertainty conditional on these fits and assume independent events. They do not include training variability, which is summarized by the 100-seed distributions, or account for temporal dependence or prior test inspection.

Pairwise differences use a two-sided null distribution centered at zero, an add-one Monte Carlo correction and Holm adjustment across the 15 unordered pairs of the six reported models. We use paired differences rather than marginal-interval overlap to assess significance.
""")
    pieces.append(table("table:significance_ci",
        r"Seed-42 test sample Top@k\%AUC point estimates and 95\% percentile intervals from 10{,}000 paired event resamples.",
        ["Model", r"Top@k\%AUC", r"95\% CI"],
        [[NAMES[m], f"{row.point_estimate:.3f}", f"$[{row.ci_lower:.3f},\\,{row.ci_upper:.3f}]$"]
         for m, row in auc.sort_values("point_estimate", ascending=False).iterrows()]))
    pieces.append(fill(r"""@@selected@@ minus logistic regression has AUC difference @@cb_l1@@. Against @@reference@@, the difference is @@cb_cat@@. Neither comparison is significant after Holm correction; this does not establish equivalence. Model selection remains based on mean validation performance across seeds.

The selected model's differences against random forest, CatBoost with SMOTE and the ranker are, respectively: @@selected_rf@@; @@selected_smote@@; and @@selected_ranker@@. The differences against SMOTE and the ranker remain significant after Holm correction at the 5\% level; the difference against random forest does not. These conclusions are conditional on the fixed models and the event-sampling assumptions above.
""", v))
    for other, expected in [(L1, False), (reference, False), (RF, False), (SMOTE, True), (RANKER, True)]:
        row = pairwise.loc[((pairwise.model_a == selected) & (pairwise.model_b == other)) |
                           ((pairwise.model_b == selected) & (pairwise.model_a == other))].iloc[0]
        assert bool(row.p_value_holm < .05) == expected, "Review significance interpretation"
    pieces.append(r"Fig.~\ref{fig:auc_bootstrap} summarizes the model-level AUC distributions; Fig.~\ref{fig:bootstrap_ci} shows the selected model's uncertainty at individual ranking thresholds." + "\n")
    pieces.append(figure("auc_distributions.png", "fig:auc_bootstrap", r"Test Top@k\%AUC bootstrap distributions for the six fixed models: central 50\% and 95\% intervals, medians and test sample point estimates. All models use the same 10{,}000 event draws.", source=COMPARISON / "images/auc_distributions.png"))
    pieces.append(figure("significance_bootstrap_ci.png", "fig:bootstrap_ci", selected_name + r" test Top@k point estimates with 95\% intervals from 10{,}000 whole-event resamples."))
    pieces.append(fill(r"""\subsection{Robustness Analysis}
\label{sec:robustness}
\subsubsection{Sensitivity to tuning and training seeds}
Table~\ref{table:robustness_summary} and Fig.~\ref{fig:robustness_distribution} summarize @@selected@@ test ranking metrics across the 100 seed-specific configurations. The training events and evaluation samples are fixed; each seed controls its own 25-trial inner search and final training. Mean AUC is $@@robust_mean@@$, its standard deviation is $@@robust_std@@$, and the central 80\% range is $[@@robust_low@@,\,@@robust_high@@]$.
""", v))
    robust_labels = {"topk_percent_auc": r"Top@k\%AUC", **{f"topk_percent@{k}": f"Top@{100*k:g}\\%" for k in [0.01, 0.02, 0.05, 0.1, 0.2]}}
    pieces.append(table("table:robustness_summary", selected_name + r" test performance across 100 tuning and training seeds, with all event partitions fixed.",
        ["Metric", "Mean", "Std", r"[10\%, 90\%]"],
        [[robust_labels[name], f"{row['mean']:.3f}", f"{row['std']:.3f}", f"[{row['10%']:.3f}, {row['90%']:.3f}]"]
         for name, row in robust.iterrows() if name in robust_labels]))
    pieces.append(figure("robustness_metrics_distribution.png", "fig:robustness_distribution", selected_name + r" test ranking metrics across 100 seed-specific configurations trained on the same 290 events."))
    pieces.append(fill(r"""These experiments measure conditional training sensitivity and do not provide independent external validation.

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
    portfolio_text = fill(r"""We use the selected seed-42 @@selected@@ model for every portfolio and price-impact analysis. We compare $k\in\{1,2,5,10,20,30\}$ and retain $k=5$ for order-size sensitivity. At $k=5$, target inclusion is $@@hit_percent@@\%$ (@@hits@@ of 80 events), exactly matching Top@5 in Table~\ref{table:results_topk}.

For each size, we generate 10{,}000 random portfolios. At each event, $k$ different eligible assets are selected uniformly and assigned equal weights, using common random asset orders across sizes. Trading rules, USDT conversion and the 25 bp round-trip cost match the model strategy. Unfilled allocations remain in cash. Events stay in chronological order and returns are summed without reinvestment. Annualized statistics use zero-filled daily returns over the same calendar and 365 days per year; each statistic is computed for each draw before averaging.

Table~\ref{table:performance_by_k} reports portfolio statistics, and Fig.~\ref{fig:portfolio_test} shows the cumulative return paths. At $k=5$, @@selected@@ cumulative return is $@@model_return@@\%$ and Sharpe is $@@model_sharpe@@$. Random selection has mean return $@@random_return@@\%$ with a central 95\% range of $[@@random_low@@\%,\,@@random_high@@\%]$ and mean target inclusion $@@random_hits@@\%$. These ranges describe random selection within the observed test period, not event-bootstrap confidence intervals or confirmatory significance tests.
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
We apply the fitted impact model to both execution legs with 1{,}000 USDT per selected asset, so a size-$k$ portfolio targets $k\times1{,}000$ USDT per event. Predictions, event order, random selections and other trading rules are identical to the no-impact comparison. Table~\ref{table:performance_by_k_impact} and Fig.~\ref{fig:portfolio_test_impact} report the resulting statistics and return paths. At $k=5$, @@selected@@ cumulative return is $@@impact_return@@\%$ and Sharpe is $@@impact_sharpe@@$. Random selection has mean return $@@impact_random@@\%$, with a central 95\% range of $[@@impact_random_low@@\%,\,@@impact_random_high@@\%]$.
""", v)
    tail += table("table:performance_by_k_impact", r"Portfolio performance with 1{,}000 USDT per asset, fitted entry and exit impact, and 25 bp costs. Random rows average 10{,}000 draws; returns are decimal fractions.", portfolio_headers, impact_rows, wide=True)
    tail += figure("portfolios_1000usdt_with_slippage.png", "fig:portfolio_test_impact",
        r"The same portfolios and random selections as Fig.~\ref{fig:portfolio_test}, with fitted price impact on both legs at 1{,}000 USDT per asset. Shading is the pointwise 95\% random-selection range; the BTC/USDT benchmark is unchanged.", source=IMPACT / "portfolios_1000usdt_with_slippage.png")
    sensitivity = sensitivity.set_index("quantity_usdt") if "quantity_usdt" in sensitivity.columns else sensitivity
    tail += r"\subsubsection{PnL sensitivity to order size}" + "\n"
    tail += (r"Fig.~\ref{fig:pnl_vs_quantity} shows the order-size sensitivity for $k=5$, varying intended size from 100 to 10{,}000 USDT per asset. "
             f"Cumulative ROE without reinvestment is ${sensitivity.loc[100,'cumulative_roe']:.3f}$ at 100 USDT, "
             f"${sensitivity.loc[2000,'cumulative_roe']:.3f}$ at 2{{,}}000 USDT, "
             f"${sensitivity.loc[5000,'cumulative_roe']:.3f}$ at 5{{,}}000 USDT, and "
             f"${sensitivity.loc[10000,'cumulative_roe']:.3f}$ at 10{{,}}000 USDT. "
             "These conditional simulation outcomes depend on fill and impact assumptions and do not establish executable capacity.\n")
    tail += figure("pnl_vs_quantity_with_impact.png", "fig:pnl_vs_quantity",
        selected_name + r" cumulative ROE over the 80 test events as intended per-asset notional changes. The $k=5$ strategy includes fitted entry and exit impact and 25 bp costs; returns are summed without reinvestment.", wide=False)
    source = source[:start] + tail + "\n" + source[end:]
    source = re.sub(r"^(?:In the model comparison, validation|Validation Top@k|Across 100 seeds).*$", lambda _: fill(
        r"Across 100 seeds per model, each with 25 hyperparameter trials within training, @@selected@@ has the highest mean validation Top@k\%AUC ($@@validation_mean@@$). On 80 test events, mean AUC is $@@cat_test_mean@@$ for CatBoost and $@@l1_test_mean@@$ for logistic regression. The selected seed-42 CatBoost model attains test AUC $@@cb_auc@@$ and Top@5 $@@top5@@$; paired event resampling does not establish a significant AUC difference from the seed-42 logistic model.", v), source, flags=re.M)
    source = re.sub(r"^This study compares (?:established|seven tuned|six tuned).*$", lambda _: fill(
        r"This study compares six tuned imbalance-aware model families across 100 seeds on the same Binance BTC-pair events. CatBoost has the highest mean validation Top@k\%AUC ($@@validation_mean@@$), while logistic regression has the highest mean test AUC ($@@l1_test_mean@@$, compared with CatBoost's $@@cat_test_mean@@$). The preselected seed-42 CatBoost model achieves test AUC $@@cb_auc@@$ and Top@5 $@@top5@@$; paired event resampling does not establish its AUC superiority over logistic regression or CatBoost + ES after multiplicity correction. Its $k=5$ cumulative return is $@@model_return@@\%$ without size-dependent impact and $@@impact_return@@\%$ with fitted impact at 1{,}000 USDT per asset. These simulation outcomes are sensitive to execution assumptions and order size.", v), source, flags=re.M)
    source = re.sub(r"^The repository provides.*$",
        "The repository provides the event cohort, split membership, software versions, seeds 1--100, feature hashes, all 600 fitted configurations, validation and test predictions, shared bootstrap draws, and random-portfolio selections and execution outcomes. Repeating all 600 final fits reproduces their saved validation and test predictions exactly. The seed-42 point estimates, event resampling and portfolios use the same saved predictions. The event records cannot independently validate Telegram annotations because the underlying messages and URLs are unavailable.", source, flags=re.M)
    training_start = source.index(r"\subsection{Training}")
    training_end = source.index("% BEGIN MODEL CONFIGURATION", training_start)
    source = source[:training_start] + r"""\subsection{Training}
We run 25 Optuna TPE trials for each of six model families and each seed from 1 to 100: 600 studies and 15{,}000 trials. Within the 290 training events, a chronological split at March 18, 2020, 19:00 UTC gives 232 inner-training events and 58 inner-tuning events. Simultaneous events remain together. Hyperparameters maximize inner-tuning Top@k\%AUC; preprocessing priors use only the corresponding training partition.

Tree depth is tuned for random forest and CatBoost. During the inner search, CatBoost variants use early stopping with 50 rounds of patience; CatBoost + ES uses Top@k\%AUC as its stopping metric. For each seed, we retain the best trial's parameters and fitted tree count, then train on all 290 training events with no validation-controlled stopping. These models are evaluated on the 103 outer-validation events and the 80 test events. Neither evaluation sample is used for hyperparameter tuning or final training.

The same seed controls TPE, stochastic model training and SMOTE within each configuration. Trials run sequentially and each model uses one CPU thread; independent configurations run in parallel. We select the family with the highest mean validation AUC across the 100 seeds, breaking exact ties by lower standard deviation and then model key. Seed 42 is chosen in advance for the fixed-model analyses, independently of performance. Event and seed resampling also use a fixed random seed. Run records preserve parameters, fitted models, feature order, input hashes and software versions.

""" + source[training_end:]
    source = source.replace("The reported intervals condition on the fitted models and assume independent pump events; correlated campaigns and test-set reuse may make them optimistic.",
        "Event-bootstrap intervals condition on the fixed seed-42 models and assume independent pump events; correlated campaigns and test-set reuse may make them optimistic. The seed distributions measure a different source of variability and do not enlarge the event sample.")
    method = r"We compare tuned logistic regression, random forest, CatBoost, CatBoost with SMOTE, CatBoost ranker, and CatBoost + ES. Logistic regression uses L1 regularization and class weights. The standalone unweighted, unpenalized statsmodels diagnostic is excluded from predictive comparisons."
    source = source.replace("L2 grid-search ties favor smaller $C$; the unpenalized logistic diagnostic", "The unpenalized logistic diagnostic")
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
    validation = read(COMPARISON / "tables/validation_metrics.csv")
    validation_seeds = read(COMPARISON / "tables/validation_seed_distributions.csv")
    test_seeds = read(COMPARISON / "tables/test_seed_distributions.csv")
    summary = read(COMPARISON / "tables/auc_summary.csv")
    topk = read(COMPARISON / "tables/test_topk.csv")
    portfolio = read(RUN / "tables/portfolio_statistics.csv")
    impact = read(IMPACT / "model_statistics.csv")
    selected = MODEL_NAMES[manifest["selection"]["model"]]
    key = selected.replace(CB, "CatBoost + ES")
    lines = [
        "# Results: 100 tuning and training seeds",
        "",
        "The manuscript reports six model families across seeds 1–100, with one regularized Logistic regression model. The submitted PDF is retained only as the highlighting baseline. "
        "The cohort contains 473 events from 498 announcements and 25 recorded exclusions: 290 training, 103 validation and 80 test events.",
        "",
        f"Mean validation Top@k%AUC selects **{NAMES[selected]}** (**{validation_seeds.loc['catboost', 'mean']:.4f}** across 100 seeds). "
        "The preselected seed-42 model supplies all portfolio analyses. Each family's seed-42 point estimates and event-bootstrap intervals use the same saved predictions.",
        "",
        "Each model and seed has 25 sequential TPE trials using a chronological 232/58 inner split of the 290 training events. "
        "The best configuration is fitted on all 290 events, with the inner-selected CatBoost tree count fixed and no evaluation-set early stopping. "
        "The 103 outer-validation events compare model families; the same configurations are evaluated on 80 test events without retuning. "
        "All 600 repeated final fits reproduced the saved validation and test predictions exactly. Fitted models, parameters, predictions and software/input hashes are saved.",
        "",
        "## Seed distributions",
        "",
        "| Model | Mean validation AUC | Validation SD | Mean test AUC | Test SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, row in validation_seeds.iterrows():
        test = test_seeds.loc[model]
        lines.append(f"| {NAMES[MODEL_NAMES[model]]} | {row['mean']:.4f} | {row['std']:.4f} | {test['mean']:.4f} | {test['std']:.4f} |")
    lines += [
        "",
        "These distributions describe tuning/training randomness conditional on the fixed events. "
        "The paired-seed mean difference for logistic regression minus CatBoost on test is 0.0244, with 95% interval [0.0203, 0.0288]. "
        "Seed resampling does not measure event-sampling uncertainty or change the validation selection.",
        "",
        "## Seed-42 point estimates and event bootstrap",
        "",
        "| Model | Validation AUC | Test AUC | Test 95% CI | Test Top@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, row in summary.iterrows():
        lines.append(f"| {NAMES[model]} | {validation.loc[model, 'Top@k%AUC']:.4f} | {row.point_estimate:.4f} | "
                     f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}] | {topk.loc[5, model]:.4f} |")
    lines += ["", "AUC integrates Top@k% from 0–20% in 0.5% increments and divides by 0.2. "
              "The percentile intervals use 10,000 paired whole-event draws shared by all six fixed models. "
              "Two-sided centered tests use an add-one correction and Holm adjustment across the 15 pairs of reported models. "
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
              f"{NAMES[selected]} AUC across 100 seeds: mean {values['robust_mean']}, standard deviation {values['robust_std']}, "
              f"central 80% range [{values['robust_low']}, {values['robust_high']}]. "
              "Training and evaluation event membership are fixed; each seed controls its inner search and final fit. Feature-importance quantiles use these same 100 CatBoost models.",
              "",
              f"The selected model's early-period AUC is {values['early_auc']} on {values['early_events']} events. "
              "The four-event late period is excluded by the ten-event minimum.",
              "",
              "[Seed-sweep audit](test_seed_sweep_results.json) independently verifies all 600 test scores and 15 paired-seed intervals. "
              "[Prediction and portfolio audit](seed_sweep_manuscript_audit.json) reconstructs the six seed-42 point metrics, "
              "10,000 event-bootstrap draws, 15 event-bootstrap comparisons, portfolio returns and target counts. "
              "[Manuscript audit](manuscript_results_audit.json) checks the six-model comparison, its 15-pair adjustment, and every generated result table and figure against the exports. "
              "[Reproduction commands](reproduction.md) regenerate the outputs and both PDFs.",
              "",
              "Manuscript artifacts: `notebooks/analysis_outputs/seed_sweep_paper/` (local, excluded from Git). "
              "The 600 checkpoints are in `test_seed_sweep_100x25/fits/`; the matching search databases and parameters are in `validation_seed_sweep_100x25/studies/`. "
              "The manuscript directory contains point metrics, event-bootstrap draws, seed distributions, feature importances and regenerated portfolio curves. "
              "Its provenance records the model-independent execution outcomes and random selections used for the portfolio replay.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = HERE / "access.tex"
    source = path.read_text()
    comparison = prepare(NAMES, check=args.check)
    compact = prepare_compact_figures(NAMES, MODEL_NAMES, check=args.check)
    updated, manifest, values = render(strip_layout(source))
    updated = apply_layout(updated)
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
              "comparison": comparison, "reported_models": list(NAMES), "paired_comparisons": 15,
              "compact_figure_provenance": compact,
              "previous_submission_used_only_as_highlight_baseline": True}
    # Do not duplicate generated table TeX inside the compact audit summary.
    report["key_results"].pop("validation_table", None)
    (ROOT / "docs/manuscript_results_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(TABLES)} tables and {len(FIGURES)} figures match the completed run.")


if __name__ == "__main__":
    main()
