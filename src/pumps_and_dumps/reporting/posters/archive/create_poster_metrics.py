#!/usr/bin/env python3
"""Build a one-page A1 landscape scientific poster in Russian.

Run from any directory: python src/pumps_and_dumps/reporting/posters/create_poster.py
Dependencies: reportlab, qrcode. Fonts: Debian DejaVu Sans, Roboto Condensed.
Data provenance and the exact notebook impact values are in results.json.
All text, plots, diagrams and QR modules are vector objects in the PDF.
"""

import json
import math
from pathlib import Path
from typing import Any, Mapping

mm = 72.0 / 25.4
W, H = 841 * mm, 594 * mm
INK, MUTED = "#142E46", "#526575"
TEAL, ORANGE, PURPLE = "#007D80", "#AA5D29", "#886F99"
GRID, LIGHT, PALE = "#D9E2E7", "#F2F6F8", "#E6F3F1"
WHITE = "#FFFFFF"
COL = (805 - 36) / 3
X1, X2, X3 = 18, 18 + COL + 18, 18 + 2 * (COL + 18)


def _register_fonts() -> None:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    pdfmetrics.registerFont(
        TTFont("Body", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )
    pdfmetrics.registerFont(
        TTFont("Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    )
    pdfmetrics.registerFont(
        TTFont(
            "Display",
            "/usr/share/fonts/truetype/roboto/unhinted/RobotoCondensed-Bold.ttf",
        )
    )
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="Bold", italic="Body", boldItalic="Bold"
    )


def _create_canvas(output: Path):
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(
        str(output),
        pagesize=(W, H),
        pageCompression=1,
        initialFontName="Body",
        lang="ru-RU",
    )
    c.setTitle("Прогнозирование целей pump-and-dump на рынке криптовалют")
    c.setAuthor("М. Миронов; Д. Богутский; П. Лукьянченко — НИУ ВШЭ")
    c.setSubject(
        "Стохастика, искусственный интеллект и финансовая математика. Постер А1, русский язык."
    )
    c.setCreator(
        "ReportLab; source: src/pumps_and_dumps/reporting/posters/create_poster.py"
    )
    return c


def ru(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


class MetricsPosterRenderer:
    """Draw the fixed poster layout using supplied data, assets, and PDF canvas."""

    def __init__(self, canvas, data: Mapping[str, Any], assets: Path):
        self.canvas = canvas
        self.data = data
        self.assets = assets

    def rect(self, x, y, w, h, fill, stroke=None, radius=0):
        from reportlab.lib.colors import HexColor

        self.canvas.setFillColor(HexColor(fill))
        self.canvas.setStrokeColor(HexColor(stroke or fill))
        if radius:
            self.canvas.roundRect(
                x * mm,
                H - (y + h) * mm,
                w * mm,
                h * mm,
                radius * mm,
                fill=1,
                stroke=bool(stroke),
            )
        else:
            self.canvas.rect(
                x * mm, H - (y + h) * mm, w * mm, h * mm, fill=1, stroke=bool(stroke)
            )

    def line(self, x1, y1, x2, y2, color=GRID, width=1, dash=None):
        from reportlab.lib.colors import HexColor

        self.canvas.saveState()
        self.canvas.setStrokeColor(HexColor(color))
        self.canvas.setLineWidth(width)
        if dash:
            self.canvas.setDash(dash)
        self.canvas.line(x1 * mm, H - y1 * mm, x2 * mm, H - y2 * mm)
        self.canvas.restoreState()

    def txt(self, text, x, y, size=22, font="Body", color=INK, align="left"):
        """Text at approximate top-left; validate against the physical page."""
        from reportlab.lib.colors import HexColor
        from reportlab.pdfbase import pdfmetrics

        missing = [
            ch
            for ch in text
            if ord(ch) not in pdfmetrics.getFont(font).face.charToGlyph
        ]
        if missing:
            raise ValueError(f"Missing glyphs in {font}: {missing}")
        width = pdfmetrics.stringWidth(text, font, size)
        left = x * mm - (
            width / 2 if align == "center" else width if align == "right" else 0
        )
        if left < 0 or left + width > W + 0.01:
            raise ValueError(f"Text extends beyond the page: {text}")
        self.canvas.setFillColor(HexColor(color))
        self.canvas.setFont(font, size)
        self.canvas.drawString(left, H - y * mm - size * 0.82, text)

    def para(self, text, x, y, w, max_h, size=22, leading=29, color=INK, bold=False):
        from reportlab.lib.colors import HexColor
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph

        style = ParagraphStyle(
            "p",
            fontName="Bold" if bold else "Body",
            fontSize=size,
            leading=leading,
            textColor=HexColor(color),
            spaceAfter=0,
        )
        p = Paragraph(text, style)
        pw, ph = p.wrap(w * mm, max_h * mm)
        if ph > max_h * mm + 0.1:
            raise ValueError(f"Text overflow {ph / mm:.1f} > {max_h}: {text}")
        p.drawOn(self.canvas, x * mm, H - y * mm - ph)
        return ph / mm

    def circle(self, x, y, r, fill, stroke=None, width=1):
        from reportlab.lib.colors import HexColor

        self.canvas.setFillColor(HexColor(fill))
        self.canvas.setStrokeColor(HexColor(stroke or fill))
        self.canvas.setLineWidth(width)
        self.canvas.circle(x * mm, H - y * mm, r * mm, fill=1, stroke=bool(stroke))

    def heading(self, n, text, x, y, w):
        self.txt(n, x, y + 1, 23, "Bold", TEAL)
        self.txt(text, x + 17, y, 31, "Display")
        self.line(x, y + 16, x + w, y + 16, INK, 1.1)

    def path(self, points, color, width=3, dash=None):
        from reportlab.lib.colors import HexColor

        self.canvas.saveState()
        self.canvas.setStrokeColor(HexColor(color))
        self.canvas.setLineWidth(width)
        self.canvas.setLineJoin(1)
        if dash:
            self.canvas.setDash(dash)
        p = self.canvas.beginPath()
        p.moveTo(points[0][0] * mm, H - points[0][1] * mm)
        for x, y in points[1:]:
            p.lineTo(x * mm, H - y * mm)
        self.canvas.drawPath(p)
        self.canvas.restoreState()

    def qr(self, url, x, y, size):
        import qrcode

        code = qrcode.QRCode(
            error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=1, border=4
        )
        code.add_data(url)
        code.make(fit=True)
        matrix = code.get_matrix()
        unit = size / len(matrix)
        for row, entries in enumerate(matrix):
            for col, value in enumerate(entries):
                if value:
                    self.rect(
                        x + col * unit, y + row * unit, unit + 0.006, unit + 0.006, INK
                    )
        self.canvas.linkURL(
            url, (x * mm, H - (y + size) * mm, (x + size) * mm, H - y * mm), relative=0
        )

    def topk_chart(self, x, y, w, h):
        self.txt("Доля событий с целью среди первых k активов", x, y, 19, color=MUTED)
        left, right, top, bottom = x + 21, x + w - 5, y + 17, y + h - 33
        sx = lambda k: left + (k - 1) / 29 * (right - left)
        sy = lambda p: bottom - p / 0.8 * (bottom - top)
        for p in [0, 0.2, 0.4, 0.6, 0.8]:
            self.line(left, sy(p), right, sy(p), GRID, 0.8)
            self.txt(f"{p:.0%}", left - 5, sy(p) - 3.1, 18, color=MUTED, align="right")
        for k in [1, 5, 10, 20, 30]:
            self.txt(str(k), sx(k), bottom + 4, 18, color=MUTED, align="center")
        self.txt(
            "k — число отобранных активов",
            (left + right) / 2,
            bottom + 15,
            18,
            color=MUTED,
            align="center",
        )
        models = [(3, ORANGE, [8, 4]), (5, PURPLE, [2, 4]), (0, TEAL, None)]
        for idx, color, dash in models:
            pts = [
                (sx(k), sy(v))
                for k, v in zip(self.data["top_k"], self.data["models"][idx]["top_k"])
            ]
            self.path(pts, color, 3.8 if idx == 0 else 2.8, dash)
            for px, py in pts:
                if idx == 3:
                    self.rect(px - 1.5, py - 1.5, 3, 3, WHITE, color)
                elif idx == 0:
                    self.circle(px, py, 1.8, color)
        self.rect(sx(10) - 13, sy(0.563) + 5, 31, 10, WHITE)
        self.txt("56,3%", sx(10) + 2.5, sy(0.563) + 6, 21, "Bold", TEAL, "center")
        for idx, (label, color, dash) in enumerate(
            [
                ("CatBoost + ES", TEAL, None),
                ("Логист. регрессия", ORANGE, [8, 4]),
                ("CatBoost + SMOTE", PURPLE, [2, 4]),
            ]
        ):
            lx = x + (idx % 2) * 132
            ly = y + h + (idx // 2) * 10
            self.line(lx, ly + 3, lx + 11, ly + 3, color, 3, dash)
            self.txt(label, lx + 15, ly, 17.5)

    def impact_chart(self, x, y, w, h):
        self.txt(
            "Накопленная доходность с учётом влияния на цену", x, y, 19, color=MUTED
        )
        left, right, top, bottom = x + 25, x + w - 9, y + 22, y + h - 30
        sx = lambda q: left + (math.log10(q) - 2) / 2 * (right - left)
        sy = lambda p: bottom - (p + 0.3) / 1.3 * (bottom - top)
        for p in [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]:
            self.line(
                left,
                sy(p),
                right,
                sy(p),
                MUTED if p == 0 else GRID,
                1.4 if p == 0 else 0.8,
            )
            self.txt(
                f"{p:.0%}".replace("-", "−"),
                left - 5,
                sy(p) - 3,
                18,
                color=MUTED,
                align="right",
            )
        for q, label in [
            (100, "100"),
            (500, "500"),
            (1000, "1 000"),
            (5000, "5 000"),
            (10000, "10 000"),
        ]:
            self.txt(label, sx(q), bottom + 4, 17.5, color=MUTED, align="center")
        self.txt(
            "Размер одной сделки, USDT · логарифмическая шкала",
            (left + right) / 2,
            bottom + 16,
            17.5,
            color=MUTED,
            align="center",
        )
        pts = [
            (sx(d["quantity_usdt"]), sy(d["cumulative_roe"]))
            for d in self.data["impact"]
        ]
        self.path(pts, TEAL, 4)
        for px, py in pts:
            self.circle(px, py, 1.9, WHITE, TEAL, 2.2)
        self.txt("+84,0%", pts[0][0] + 3, pts[0][1] - 15, 24, "Bold", TEAL)
        self.txt("+13,5%", pts[-2][0] - 22, pts[-2][1] - 15, 23, "Bold", TEAL)
        self.txt(
            "−20,1%", pts[-1][0] - 5, pts[-1][1] + 1.5, 24, "Bold", ORANGE, "right"
        )

    def _draw_header(self) -> None:
        # Header and the single, explicit test-result banner.
        self.rect(0, 0, 841, 594, WHITE)
        self.rect(18, 16, 3, 95, TEAL)
        self.txt(
            "КОНФЕРЕНЦИЯ «СТОХАСТИКА, ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ И ФИНАНСОВАЯ МАТЕМАТИКА»",
            27,
            16,
            19,
            "Bold",
            MUTED,
        )
        self.txt("Прогнозирование целей pump-and-dump", 26, 32, 72, "Display")
        self.txt("на рынке криптовалют", 26, 59, 72, "Display")
        self.txt(
            "Микроструктурные признаки и обучение при сильном дисбалансе классов",
            28,
            88,
            28,
            color=TEAL,
        )
        self.txt("М. Миронов · Д. Богутский · П. Лукьянченко", 28, 103, 25, "Bold")
        self.txt(
            "НИУ ВШЭ · Факультет компьютерных наук · Лаборатория «Искусственный интеллект в математических финансах»",
            28,
            116,
            20.5,
            color=MUTED,
        )
        self.txt(
            "CatBoost + ES · модель зафиксирована до тестирования",
            821,
            126,
            17,
            "Bold",
            TEAL,
            "right",
        )

        self.rect(18, 134, 805, 43, INK, radius=3)
        metrics = [
            ("15,0%", "Top@1: 12 из 80 событий"),
            ("56,3%", "Top@10: 45 из 80 событий"),
            ("0,674", "Top@k%AUC · тест"),
            ("1 : 287", "Дисбаланс классов · тест"),
        ]
        for i, (value, caption) in enumerate(metrics):
            xx = 28 + i * 202
            if i:
                self.line(xx - 7, 141, xx - 7, 169, "#385068", 0.8)
            self.txt(value, xx, 140, 52, "Display", WHITE)
            self.txt(caption, xx, 163, 20, color="#D8E9ED")

        self.line(X2 - 9, 191, X2 - 9, 518, GRID, 0.8)
        self.line(X3 - 9, 191, X3 - 9, 518, GRID, 0.8)

    def _draw_data(self) -> None:
        # Column 1: question, sample construction, chronological protocol, method.
        self.heading("01", "Задача и данные", X1, 190, COL)
        self.para(
            "Pump-and-dump — разгон цены и последующая распродажа. Время объявления известно заранее. "
            "<b>Задача: за 15 минут до него</b> ранжировать пары Binance к BTC и определить единственную цель манипуляции.",
            X1,
            213,
            COL,
            43,
        )

        self.rect(X1, 263, COL, 36, LIGHT, radius=2)
        self.line(X1 + 14, 280, X1 + COL - 14, 280, GRID, 3)
        self.line(X1 + 14, 280, X1 + 125, 280, TEAL, 4)
        self.circle(X1 + 125, 280, 2.4, TEAL)
        self.circle(X1 + COL - 22, 280, 2.4, INK)
        self.txt("Только прошлые сделки", X1 + 9, 267, 18, color=TEAL)
        self.txt("T − 15 мин: прогноз", X1 + 125, 287, 18, "Bold", TEAL, "center")
        self.txt("T: объявление", X1 + COL - 9, 267, 18, "Bold", INK, "right")

        self.txt("498 объявлений · 473 доступных события", X1, 307, 24, "Display")
        self.txt(
            "Январь 2018 — август 2022 · разбиение по времени",
            X1,
            319,
            18.5,
            color=MUTED,
        )
        for i, (n, label, dates) in enumerate(
            [
                (290, "Обучение", "до 09.2020"),
                (103, "Валидация", "09.2020–04.2021"),
                (80, "Тест", "05.2021–08.2022"),
            ]
        ):
            xx = X1 + i * (COL / 3)
            self.rect(xx, 334, COL / 3 - 3, 38, PALE if i == 2 else LIGHT, radius=2)
            self.txt(str(n), xx + 5, 337, 38, "Display", TEAL if i == 2 else INK)
            self.txt(label, xx + 5, 352, 19, "Bold")
            self.txt(dates, xx + 5, 363, 15.5, color=MUTED)
        self.para(
            "25 событий исключены из-за отсутствия истории. Доходность после объявления не влияет на отбор.",
            X1,
            379,
            COL,
            22,
            size=20,
            leading=26,
            color=MUTED,
        )

        self.heading("02", "Метод", X1, 408, COL)
        self.para(
            "<b>1. Признаки.</b> Поток покупок и продаж, проскальзывание, доходности, объёмы, число сделок и прошлых пампов.",
            X1,
            431,
            COL,
            24,
            size=21,
            leading=27,
        )
        self.para(
            "<b>2. Нормализация.</b> z-оценки по истории; стандартизация части признаков по активам события.",
            X1,
            458,
            COL,
            24,
            size=21,
            leading=27,
        )
        self.para(
            "<b>3. Обучение.</b> Веса классов; CatBoost с ранней остановкой (ES) по Top@k%AUC. Выбор модели — только на валидации.",
            X1,
            486,
            COL,
            30,
            size=21,
            leading=27,
        )

    def _draw_quality(self) -> None:
        # Column 2: readable fixed-k curves, then normalized model-selection metric.
        self.heading("03", "Качество прогнозирования", X2, 190, COL)
        self.topk_chart(X2, 214, COL, 122)
        self.txt("Top@k%AUC: площадь под Top@k% на 0–20%", X2, 363, 23, "Display")
        self.txt(
            "Нормирована к [0; 1]; k% — доля активов в отборе.",
            X2,
            374,
            18.5,
            color=MUTED,
        )
        self.txt(
            "Базовые модели — с подбором гиперпараметров.", X2, 383, 15, color=MUTED
        )

        table_y = 389
        self.rect(X2, table_y, COL, 15, INK)
        self.txt("Модель", X2 + 3, table_y + 4, 19, "Bold", WHITE)
        self.txt("Валидация", X2 + COL - 51, table_y + 4, 18, "Bold", WHITE, "right")
        self.txt("Тест", X2 + COL - 4, table_y + 4, 18, "Bold", WHITE, "right")
        for i, model in enumerate(self.data["models"]):
            yy = table_y + 15 + i * 12
            self.rect(
                X2, yy, COL, 12, PALE if i == 0 else (LIGHT if i % 2 == 0 else WHITE)
            )
            font = "Bold" if i == 0 else "Body"
            self.txt(model["name"], X2 + 3, yy + 3, 18, font)
            self.txt(
                ru(model["validation_auc"]),
                X2 + COL - 51,
                yy + 3,
                19,
                font,
                TEAL if i == 0 else INK,
                "right",
            )
            self.txt(
                ru(model["test_auc"]),
                X2 + COL - 4,
                yy + 3,
                19,
                "Bold" if i in (0, 3) else "Body",
                INK,
                "right",
            )
        self.line(X2, 476, X2 + COL, 476, GRID, 1)
        self.para(
            "<b>Неопределённость.</b> 95% ДИ для CatBoost + ES: [0,592; 0,755]. "
            "Разница с CatBoost: 0,0116; парный bootstrap, односторонний p = 0,174. Преимущество статистически не подтверждено.",
            X2,
            482,
            COL,
            35,
            size=20,
            leading=24.5,
        )

    def _draw_capacity(self) -> None:
        # Column 3: economic result and material limits from the manuscript.
        self.heading("04", "Исполнение и ёмкость", X3, 190, COL)
        self.para(
            "Портфель из <b>5 активов</b>, равные веса. Вход после прогноза; выход из цели через 1 минуту после объявления, "
            "из остальных — при объявлении. Комиссия за круг — 0,25%; без реинвестирования.",
            X3,
            213,
            COL,
            44,
            size=21,
            leading=27,
        )
        self.impact_chart(X3, 263, COL, 139)

        self.rect(X3, 411, COL, 29, PALE, radius=2)
        self.txt(
            "Ёмкость сигнала ограничена ликвидностью", X3 + 6, 416, 25, "Display", TEAL
        )
        self.txt(
            "Результат меняет знак между 5 000 и 10 000 USDT на сделку.",
            X3 + 6,
            429,
            17.5,
            color=INK,
        )
        self.para(
            "Влияние сделки на цену: <b>I(Q) = β√Q</b>; среднее проскальзывание исполнения: "
            "<b>I<sub>VWAP</sub>(Q) = ⅔ β√Q</b>. Коэффициент β оценивается по рыночным данным до сделки.",
            X3,
            450,
            COL,
            33,
            size=20,
            leading=26,
        )
        self.para(
            "<b>Границы результата.</b> Только Binance и 80 тестовых событий. Модель исполнения не учитывает задержки API, "
            "исчерпание глубины стакана и конкурирующие заявки.",
            X3,
            485,
            COL,
            31,
            size=20,
            leading=26,
        )

    def _draw_conclusion(self) -> None:
        # A concise takeaway connects target prediction to the execution constraint.
        self.rect(18, 530, 805, 25, INK, radius=2)
        self.txt("ВЫВОД", 26, 538, 25, "Display", "#8DDECE")
        self.txt(
            "Микроструктура помогает выявлять цели заранее; влияние сделок на цену ограничивает масштаб стратегии.",
            72,
            539,
            25,
            "Display",
            WHITE,
        )

    def _draw_footer(self) -> None:
        # Human-readable provenance and a vector QR to the published research code.
        self.txt(
            "Исследование: Mironov, Bogutsky, Lukianchenko. Mitigating Class Imbalance in Pump-and-Dump Detection (рукопись).",
            18,
            564,
            16,
            color=MUTED,
        )
        self.txt(
            "Контакт: dbogucki@hse.ru   ·   Код и данные: github.com/B0R0koko/pump-and-dump-prediction",
            18,
            573,
            16,
            color=MUTED,
        )
        self.txt(
            "Поддержка: Минэкономразвития России · соглашения 000000C313925P4E0002 и № 139-15-2025-009 с НИУ ВШЭ.",
            18,
            582,
            13.5,
            color=MUTED,
        )
        self.canvas.linkURL(
            "mailto:dbogucki@hse.ru",
            (18 * mm, H - 581 * mm, 136 * mm, H - 571 * mm),
            relative=0,
        )
        self.canvas.linkURL(
            self.data["sources"]["repository"],
            (160 * mm, H - 581 * mm, 540 * mm, H - 571 * mm),
            relative=0,
        )
        self.canvas.linkURL(
            self.data["sources"]["seminar"],
            (18 * mm, H - 129 * mm, 560 * mm, H - 115 * mm),
            relative=0,
        )
        self.qr(self.data["sources"]["repository"], 790, 560, 29)
        self.txt("Код и данные", 777, 573, 15, "Bold", TEAL, "right")

    def render(self) -> None:
        self._draw_header()
        self._draw_data()
        self._draw_quality()
        self._draw_capacity()
        self._draw_conclusion()
        self._draw_footer()


def _default_directory() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent / "poster/archive"
    raise FileNotFoundError(
        "Cannot locate poster inputs; pass directory to build_poster()"
    )


def build_poster(
    directory: Path | None = None,
    data: Mapping[str, Any] | None = None,
    output_path: Path | None = None,
    assets_dir: Path | None = None,
) -> Path:
    """Load saved results and build this edition only when explicitly requested."""
    directory = Path(directory) if directory is not None else _default_directory()
    output = (
        Path(output_path)
        if output_path is not None
        else directory / "poster_saimf_ru_A1.pdf"
    )
    assets = Path(assets_dir) if assets_dir is not None else directory / "assets"
    results = (
        data
        if data is not None
        else json.loads((directory / "results.json").read_text())
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _register_fonts()
    canvas = _create_canvas(output)
    MetricsPosterRenderer(canvas, results, assets).render()
    canvas.showPage()
    canvas.save()
    return output


def main() -> None:
    print(build_poster())


if __name__ == "__main__":
    main()
