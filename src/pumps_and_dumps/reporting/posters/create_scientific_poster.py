#!/usr/bin/env python3
"""Render the English A1 poster from saved results; no models are retrained.

Run through create_poster.py with uv. Text, equations and figures are vector.
"""

import json
from pathlib import Path

if __package__:
    from .poster_data import load_poster_data
else:
    from poster_data import load_poster_data

mm = 72 / 25.4
W, H = 841 * mm, 594 * mm
BLUE, INK, MUTED = "#245291", "#172A3D", "#53616D"
LIGHT, GRID, WHITE = "#F2F5F9", "#D8E0E8", "#FFFFFF"
COL = (805 - 34) / 3
X1, X2, X3 = 18, 18 + COL + 17, 18 + 2 * (COL + 17)
COLORS = {
    "Logistic regression": "#D26A27", "Random forest": "#758936",
    "CatBoost classifier": "#2575B8", "CatBoost classifier + SMOTE": "#AC477A",
    "CatBoost ranker": "#8B6E57",
    "CatBoost classifier + Top@k%AUC early stopping": "#007F7C",
    "CatBoost + QuerySoftMax": "#7555AE", "CatBoost + Top-5": "#596A83",
    "uniform": "#A1A8AE",
}
QUERY = "CatBoost + QuerySoftMax"
SMOTE = "CatBoost classifier + SMOTE"
PLOT_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 16, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "svg.fonttype": "path",
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "mathtext.fontset": "stix",
}


def _register_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for name, file in {
        "Body": "dejavu/DejaVuSans.ttf", "Bold": "dejavu/DejaVuSans-Bold.ttf",
        "Display": "roboto/unhinted/RobotoCondensed-Bold.ttf",
    }.items():
        pdfmetrics.registerFont(TTFont(name, f"/usr/share/fonts/truetype/{file}"))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold", italic="Body", boldItalic="Bold")


def _create_canvas(output):
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(output), pagesize=(W, H), pageCompression=1,
                      initialFontName="Body", lang="en-GB")
    c.setTitle("Predicting pump-and-dump targets in cryptocurrency markets")
    c.setAuthor("M. Mironov; D. Bogutsky; P. Lukianchenko — HSE University")
    c.setSubject("Weighted portfolio selection, paired event bootstrap, and probability-based capital allocation")
    c.setCreator("ReportLab / Matplotlib; create_scientific_poster.py")
    return c


class ScientificPosterRenderer:
    """Draw an explicit result snapshot, independently of experiment execution."""

    def __init__(self, canvas, data, assets):
        self.canvas, self.data, self.assets = canvas, data, assets
        self.models = {row["name"]: row for row in data["models"]}

    def rect(self, x, y, w, h, color, stroke=None):
        from reportlab.lib.colors import HexColor

        self.canvas.setFillColor(HexColor(color))
        self.canvas.setStrokeColor(HexColor(stroke or color))
        self.canvas.rect(x * mm, H - (y + h) * mm, w * mm, h * mm, fill=1, stroke=bool(stroke))

    def line(self, x1, y1, x2, y2, color=GRID, width=1):
        from reportlab.lib.colors import HexColor

        self.canvas.setStrokeColor(HexColor(color))
        self.canvas.setLineWidth(width)
        self.canvas.line(x1 * mm, H - y1 * mm, x2 * mm, H - y2 * mm)

    def txt(self, text, x, y, size=22, font="Body", color=INK, align="left"):
        from reportlab.lib.colors import HexColor
        from reportlab.pdfbase import pdfmetrics

        width = pdfmetrics.stringWidth(text, font, size)
        left = x * mm - (width / 2 if align == "center" else width if align == "right" else 0)
        if left < 0 or left + width > W + 0.01:
            raise ValueError(f"Text extends beyond the page: {text}")
        self.canvas.setFillColor(HexColor(color))
        self.canvas.setFont(font, size)
        self.canvas.drawString(left, H - y * mm - size * 0.82, text)

    def para(self, text, x, y, w, max_h, size=21, leading=27, color=INK):
        from reportlab.lib.colors import HexColor
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph

        style = ParagraphStyle("body", fontName="Body", fontSize=size, leading=leading,
                               textColor=HexColor(color), spaceAfter=0)
        paragraph = Paragraph(text, style)
        _, height = paragraph.wrap(w * mm, max_h * mm)
        if height > max_h * mm + 0.1:
            raise ValueError(f"Text overflow ({height / mm:.1f} > {max_h} mm): {text}")
        paragraph.drawOn(self.canvas, x * mm, H - y * mm - height)

    def heading(self, title, x, y, size=30):
        self.txt(title, x, y, size, "Display", BLUE)
        self.line(x, y + 14, x + COL, y + 14, BLUE, 1.8)

    def svg(self, path, x, y, w, h):
        from reportlab.graphics import renderPDF
        from svglib.svglib import svg2rlg

        drawing = svg2rlg(str(path))
        scale = min(w * mm / drawing.width, h * mm / drawing.height)
        width, height = drawing.width * scale, drawing.height * scale
        drawing.scale(scale, scale)
        renderPDF.draw(drawing, self.canvas, x * mm + (w * mm - width) / 2,
                       H - y * mm - height - (h * mm - height) / 2)

    def equation(self, name, expression, x, y, w, h):
        from matplotlib import mathtext
        from matplotlib.font_manager import FontProperties

        output = self.assets / f"{name}.svg"
        mathtext.math_to_image(f"${expression}$", str(output), format="svg",
                               prop=FontProperties(size=25), color=INK)
        self.svg(output, x, y, w, h)

    def _plot_portfolios(self):
        from matplotlib.figure import Figure
        from matplotlib.ticker import PercentFormatter

        fig = Figure(figsize=(9, 4.1))
        ax = fig.subplots()
        records = self.data["allocation"]["test"]["probability"]["sampled"]
        for name, color in COLORS.items():
            rows = [row for row in records if row["strategy"] == name]
            ax.scatter([r["max_drawdown"] for r in rows], [r["final_return"] for r in rows],
                       s=12, color=color, alpha=0.30, edgecolors="none")
            frontier = sorted((r for r in rows if r["is_pareto"]), key=lambda r: r["max_drawdown"])
            ax.plot([r["max_drawdown"] for r in frontier], [r["final_return"] for r in frontier],
                    color=color, linewidth=2.6)
        ax.axhline(0, color=GRID, linewidth=1)
        ax.set(xlabel="Median maximum drawdown", ylabel="Median final return")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.text(0.98, 0.96, "Upper left is better\nLines: model Pareto frontiers",
                transform=ax.transAxes, va="top", ha="right", fontsize=12, color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        fig.subplots_adjust(left=0.14, bottom=0.23, right=0.99, top=0.98)
        fig.savefig(self.assets / "probability_portfolios_test.svg", transparent=True)

    def _draw_header(self):
        self.rect(0, 0, 841, 80, BLUE)
        self.txt("STOCHASTICS, ARTIFICIAL INTELLIGENCE AND MATHEMATICAL FINANCE", 18, 12, 17, "Bold", "#DEE8F7")
        self.txt("Predicting pump-and-dump targets in cryptocurrency markets", 18, 26, 58, "Display", WHITE)
        self.txt("M. Mironov · D. Bogutsky · P. Lukianchenko", 18, 52, 26, "Bold", WHITE)
        self.txt("HSE University · Faculty of Computer Science · Laboratory of AI in Mathematical Finance", 18, 66, 20, color=WHITE)
        self.rect(18, 89, 805, 29, LIGHT)
        self.rect(18, 89, 1.4, 29, BLUE)
        self.para("<b>Research question:</b> can we identify the target among hundreds of assets 15 minutes before a scheduled manipulation?"
                  "<br/><b>Main result:</b> QuerySoftMax and SMOTE lead weighted selection; at sampled K=5, probability allocation raises median return and drawdown.",
                  24, 94, 790, 23, size=23.5, leading=30)

    def _draw_split_table(self):
        self.rect(X1, 288, COL, 12, LIGHT)
        for title, offset, align in [("Partition", 3, "left"), ("Event dates", 80, "left"), ("Pumps", COL - 3, "right")]:
            self.txt(title, X1 + offset, 291, 18, "Bold", BLUE, align)
        labels = {"TRAIN": "Training", "TUNING": "Tuning", "VALIDATION": "Comparison", "TEST": "Test"}
        for i, row in enumerate(self.data["splits"]):
            y = 303 + i * 12
            start, end = row["start"][:10], row["end"][:10]
            self.txt(labels[row["split"]], X1 + 3, y, 19, "Bold" if i == 2 else "Body")
            self.txt(f"{start} – {end}", X1 + 80, y + 0.5, 16.5)
            self.txt(str(row["events"]), X1 + COL - 3, y, 20, "Bold", align="right")
        self.line(X1, 352, X1 + COL, 352)

    def _draw_data(self):
        run = self.data["run"]
        test = run["splits"]["TEST"]
        self.heading("Problem and prediction time", X1, 130)
        self.para("Pump-and-dump groups announce a scheduled time, then reveal the target ticker. "
                  "We rank eligible assets using information available <b>15 minutes before the announcement</b>. "
                  "Each event has one announced target.", X1, 151, COL, 42)
        self.equation("prediction", r"s_{e,i}=f_\theta(x_{e,i}),\qquad t_e=T_e-15\,\mathrm{min}", X1 + 6, 199, COL - 12, 14)
        self.heading("Data and chronological separation", X1, 225)
        self.para(f"<b>{run['n_events']}</b> usable Binance/BTC events, January 2018–August 2022; "
                  f"<b>{run['n_candidates']:,}</b> candidate-event rows. "
                  "Chronological <b>60 / 10 / 20 / 10%</b> split; no event crosses partitions.",
                  X1, 246, COL, 34)
        self._draw_split_table()
        accuracy = 1 - test["events"] / test["rows"]
        self.para(f"Test: <b>{test['events']} targets among {test['rows']:,} candidates</b> "
                  f"(about 1:{test['rows'] / test['events']:.0f}). "
                  f"Always predicting ‘not the target’ gives {accuracy:.2%} accuracy, so we measure target inclusion instead.",
                  X1, 361, COL, 30, size=20, leading=25)
        self.heading(f"{run['n_features']} features, available before the event", X1, 393, size=28)
        self.para("<b>Flow:</b> buyer share, buy/sell imbalance and trade count.<br/>"
                  "<b>Price and liquidity:</b> returns, standardized returns and volume, slippage imbalance, and order-size tails.<br/>"
                  "<b>History:</b> the asset’s previous pump announcements.", X1, 414, COL, 49, size=20.5, leading=26)
        self.para("Ten lookbacks: 5 minutes–14 days; 30-day history for normalization. "
                  "Assets must have traded in the 24 hours before the decision time.",
                  X1, 466, COL, 23, size=19, leading=23)
        self.rect(X1, 492, COL, 24, LIGHT)
        self.para("<b>Causal inputs:</b> features and prior-pump counts stop before T − 15 min; "
                  "imputation uses training data only.",
                  X1 + 4, 496, COL - 8, 17, size=18.5, leading=23)
        self.para("<b>Execution:</b> enter the first trade in [T − 15 min, T); after the announcement, "
                  "exit other assets at T and the target at T + 1 min, using first available trades.",
                  X1, 523, COL, 29, size=18.5, leading=23)

    def _draw_methods(self):
        splits = self.data["run"]["splits"]
        self.heading("Models and evaluation protocol", X2, 130)
        width, gap = 78, (COL - 3 * 78) / 2
        for i, (title, subtitle) in enumerate([("Trades", "before T − 15 min"), ("Features", "event normalization"), ("Scores", "rank / sample")]):
            x = X2 + i * (width + gap)
            self.rect(x, 152, width, 28, LIGHT, BLUE)
            self.txt(title, x + width / 2, 157, 23, "Bold", BLUE, "center")
            self.txt(subtitle, x + width / 2, 169, 16, color=MUTED, align="center")
            if i < 2:
                self.txt("→", x + width + gap / 2, 161, 24, color=MUTED, align="center")
        self.para("<b>Eight models:</b> logistic regression, random forest, four CatBoost classifier/ranker variants, "
                  "QuerySoftMax and a Top-5 margin loss. QuerySoftMax rewards the target’s within-event softmax probability; "
                  "Top-5 rewards clearing the fifth-negative score. SMOTE modifies training data only.",
                  X2, 190, COL, 45, size=20, leading=25)
        self.para("<b>800 fresh Optuna trials:</b> 100 per model. Each retains its tuning objective. "
                  f"Tune and stop on <b>{splits['TUNING']['events']}</b> events; compare frozen models on "
                  f"<b>{splits['VALIDATION']['events']}</b> later events, then evaluate "
                  f"<b>{splits['TEST']['events']}</b> test events with the choice fixed.",
                  X2, 237, COL, 34, size=19.5, leading=24)
        self.heading("Model selection by sampled inclusion", X2, 278, size=28)
        self.equation("sampled_auc", r"A_{\mathrm{sampled}}=\frac{1}{0.20}\int_0^{0.20}H_{\mathrm{sampled}}(q)\,dq",
                      X2 + 12, 298, COL - 24, 18)
        self.para("H(q) averages target inclusion when sampling a fraction q of each event’s assets. "
                  "Use <b>1,000 weighted samples/event</b>: classifier probabilities or softmax ranking scores, "
                  "without replacement. Shared random numbers pair models/K. Area: 0–20%; not ROC-AUC.",
                  X2, 322, COL, 36, size=19, leading=24)
        self._draw_results_table()
        self._draw_significance()

    def _draw_results_table(self):
        self.rect(X2, 363, COL, 11, LIGHT)
        self.txt("Model", X2 + 3, 366, 17.5, "Bold", BLUE)
        self.txt("Compare", X2 + COL - 48, 366, 17, "Bold", BLUE, "right")
        self.txt("Test", X2 + COL - 3, 366, 17, "Bold", BLUE, "right")
        best_compare = max(self.models, key=lambda name: self.models[name]["comparison_auc"])
        best_test = max(self.models, key=lambda name: self.models[name]["test_auc"])
        for i, model in enumerate(self.data["models"]):
            y = 375 + i * 9.5
            self.rect(X2 + 3, y + 3, 2, 2, COLORS[model["name"]])
            self.txt(model["label"], X2 + 9, y + 1, 17)
            self.txt(f"{model['comparison_auc']:.4f}", X2 + COL - 48, y + 1, 18,
                     "Bold" if model["name"] == best_compare else "Body", align="right")
            self.txt(f"{model['test_auc']:.4f}", X2 + COL - 3, y + 1, 18,
                     "Bold" if model["name"] == best_test else "Body", align="right")
        self.line(X2, 453, X2 + COL, 453)
        self.txt("Sampled Top@k%AUC; model colors match the portfolio plot.", X2, 456, 15.5, color=MUTED)

    def _draw_significance(self):
        pairs = self.data["pairwise"]
        between = {
            split: next(row for row in records if {row["model_a"], row["model_b"]} == {QUERY, SMOTE})
            for split, records in pairs.items()
        }
        self.para("<b>10,000 paired whole-event bootstraps</b>, fixed models and portfolio draws; "
                  "two-sided tests, Holm over 28 pairs (α = 0.05). "
                  "<b>QuerySoftMax and SMOTE each beat the other six models</b> on both samples. "
                  f"Their difference is inconclusive: adjusted p = {between['comparison']['p_value_holm']:.3f} "
                  f"(comparison), {between['test']['p_value_holm']:.3f} (test).",
                  X2, 470, COL, 52, size=19, leading=24)
        self.rect(X2, 528, COL, 24, LIGHT)
        self.para("<b>QuerySoftMax is selected on comparison AUC.</b> "
                  "SMOTE leads the test point estimate; neither is a uniquely established winner.",
                  X2 + 4, 532, COL - 8, 18, size=18.5, leading=23)

    def _draw_results(self):
        self.heading("Probability-based capital allocation", X3, 130, size=29)
        self.para("Keep each sampled K-asset portfolio unchanged; replace equal capital weights with "
                  "weights proportional to classifier probabilities or exponentiated ranking scores. "
                  "Normalize within the selected set S:", X3, 151, COL, 35, size=20, leading=25)
        self.equation("allocation", r"w_i=\frac{q_i}{\sum_{j\in S}q_j},\qquad |S|=K", X3 + 22, 188, COL - 44, 19)
        self.txt("Test portfolios · probability allocation", X3, 217, 24, "Display", BLUE)
        self.svg(self.assets / "probability_portfolios_test.svg", X3 - 2, 228, COL + 3, 117)
        self.txt("K = 1–30 · 1,000 full test paths per model / K · gray: uniform", X3, 346, 15.5, color=MUTED)
        self._draw_allocation_table()
        self.para("<b>Sampled K=5:</b> QuerySoftMax has the highest median return; SMOTE has slightly lower return "
                  "and lower drawdown. Allocation changes capital exposure, not target inclusion or its bootstrap results.",
                  X3, 487, COL, 29, size=18.5, leading=23)
        self.para("Fixed notional, no compounding; USDT returns, 25 bp round-trip costs, <b>no price impact</b>. "
                  "Unfilled weights stay in cash. Drawdown uses event closes. Return and drawdown are separate medians; "
                  "simulation distributions are <b>not confidence intervals</b>.",
                  X3, 520, COL, 32, size=18, leading=22)

    def _draw_allocation_table(self):
        self.txt("Sampled K=5 · equal vs probability allocation", X3, 362, 23, "Display", BLUE)
        self.rect(X3, 377, COL, 11, LIGHT)
        self.txt("Model", X3 + 3, 380, 17, "Bold", BLUE)
        self.txt("Median return", X3 + COL - 72, 380, 17, "Bold", BLUE, "right")
        self.txt("Median drawdown", X3 + COL - 3, 380, 17, "Bold", BLUE, "right")
        rows = {row["strategy"]: row for row in self.data["allocation"]["test"]["k5"]["sampled"]}
        order = [model["name"] for model in self.data["models"]] + ["uniform"]
        for i, name in enumerate(order):
            row, y = rows[name], 391 + i * 10
            label = self.models[name]["label"] if name in self.models else "Uniform sampling"
            self.rect(X3 + 3, y + 2, 2, 2, COLORS[name])
            self.txt(label, X3 + 9, y, 16)
            self.txt(f"{row['final_return_equal']:+.1%} → {row['final_return_probability']:+.1%}",
                     X3 + COL - 72, y, 16, align="right")
            self.txt(f"{row['max_drawdown_equal']:.1%} → {row['max_drawdown_probability']:.1%}",
                     X3 + COL - 3, y, 16, align="right")
        self.line(X3, 481, X3 + COL, 481)

    def _qr(self, url, x, y, size):
        import qrcode

        code = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, border=4)
        code.add_data(url)
        code.make(fit=True)
        matrix = code.get_matrix()
        unit = size / len(matrix)
        for row, values in enumerate(matrix):
            for col, value in enumerate(values):
                if value:
                    self.rect(x + col * unit, y + row * unit, unit + 0.005, unit + 0.005, INK)
        self.canvas.linkURL(url, (x * mm, H - (y + size) * mm, (x + size) * mm, H - y * mm), relative=0)

    def _draw_footer(self):
        self.line(18, 560, 823, 560, BLUE, 1.2)
        self.txt("Paper: Mironov, Bogutsky & Lukianchenko, Mitigating Class Imbalance in Pump-and-Dump Detection. Results updated 12 October 2026.",
                 18, 574, 14.5, color=MUTED)
        self.txt("Code: github.com/B0R0koko/pump-and-dump-prediction · dbogucki@hse.ru · Support: Ministry of Economic Development of the Russian Federation;",
                 18, 581, 13.5, color=MUTED)
        self.txt("agreements 000000C313925P4E0002 and 139-15-2025-009 with HSE University.", 18, 587, 12.5, color=MUTED)
        self._qr(self.data["sources"]["repository"], 794, 563, 26)
        self.txt("Code & data", 786, 576, 14, "Bold", BLUE, "right")

    def render(self):
        self._plot_portfolios()
        self._draw_header()
        self._draw_data()
        self._draw_methods()
        self._draw_results()
        self._draw_footer()


def _default_directory():
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent / "poster"
    raise FileNotFoundError("Cannot locate project root; pass directory to build_poster()")


def build_poster(directory=None, *, data=None, output_path=None, assets_dir=None):
    """Build an English edition while retaining the original Russian PDF."""
    import matplotlib

    directory = Path(directory) if directory is not None else _default_directory()
    output = Path(output_path) if output_path else directory / "poster_saimf_en_A1_scientific.pdf"
    assets = Path(assets_dir) if assets_dir else directory / "assets" / "scientific_en"
    results = data if data is not None else load_poster_data(directory.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    _register_fonts()
    canvas = _create_canvas(output)
    with matplotlib.rc_context(PLOT_STYLE):
        ScientificPosterRenderer(canvas, results, assets).render()
    canvas.showPage()
    canvas.save()
    (directory / "results_en.json").write_text(json.dumps(results, indent=2) + "\n")
    return output


def main():
    print(build_poster())


if __name__ == "__main__":
    main()
