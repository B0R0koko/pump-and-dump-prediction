#!/usr/bin/env python3
"""Build a one-page A1 landscape scientific poster in Russian.

Run from any directory: python src/pumps_and_dumps/reporting/posters/create_poster.py
Dependencies: reportlab, qrcode. Fonts: Debian DejaVu Sans, Roboto Condensed.
This visual edition explains the manipulation mechanism and its importance.
The real event chart is copied unchanged from paper/src/images.
Illustration prompt and provenance are saved in assets/illustration_prompt.txt.
Text and QR are vector objects; the two illustrations are embedded raster assets.
"""

import json
from pathlib import Path
from typing import Any, Mapping

mm = 72.0 / 25.4
W, H = 841 * mm, 594 * mm
INK, MUTED = "#142E46", "#526575"
TEAL, ORANGE, PURPLE = "#007D80", "#AA5D29", "#886F99"
GRID, LIGHT, PALE = "#D9E2E7", "#F2F6F8", "#E6F3F1"
WHITE = "#FFFFFF"


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


class VisualPosterRenderer:
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

    def illustration(self, filename, x, y, w, h):
        """Fit the original raster in a box without cropping, overlays or alteration."""
        from reportlab.lib.utils import ImageReader

        im = ImageReader(str(self.assets / filename))
        iw, ih = im.getSize()
        scale = min(w / iw, h / ih)
        dw, dh = iw * scale, ih * scale
        xx, yy = x + (w - dw) / 2, y + (h - dh) / 2
        self.canvas.drawImage(
            im, xx * mm, H - (yy + dh) * mm, dw * mm, dh * mm, mask="auto"
        )
        print(
            f"{filename}: {iw}x{ih} pixels; print resolution {iw / (dw / 25.4):.0f} dpi"
        )

    def _draw_header(self) -> None:
        # Header: preserve the paper's topic, authors and affiliation.
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
            "Почему важно заметить манипуляцию до массовых покупок",
            28,
            88,
            29,
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

    def _draw_question(self) -> None:
        # Plain-language context and the actual prediction task; no result metrics.
        self.rect(18, 134, 805, 45, INK, radius=3)
        self.txt("ЧТО ПРОИСХОДИТ", 28, 140, 19, "Bold", "#8DDECE")
        self.para(
            "Организаторы заранее покупают актив, создают ажиотаж, а затем продают его новым участникам. "
            "Рост цены сменяется обвалом.",
            28,
            151,
            498,
            24,
            size=24,
            leading=30,
            color=WHITE,
        )
        self.line(553, 141, 553, 172, "#385068", 1)
        self.txt("НАША ЗАДАЧА", 566, 140, 19, "Bold", "#8DDECE")
        self.para(
            "За <b>15 минут</b> до известного времени объявления определить актив-цель по истории сделок.",
            566,
            151,
            244,
            25,
            size=22,
            leading=28,
            color=WHITE,
        )

    def _draw_illustrations(self) -> None:
        # Imagery: generated conceptual diptych and an unchanged historical chart.
        self.heading("01", "Разгон и обвал: две стороны одной схемы", 18, 192, 535)
        self.heading("02", "Реальный пример", 583, 192, 240)
        self.illustration("pump_dump_story.png", 18, 216, 535, 175)
        self.illustration("vibbtc_real_event.png", 583, 216, 240, 175)
        self.txt("PUMP · РАЗГОН ЦЕНЫ", 26, 398, 29, "Display", TEAL)
        self.txt("DUMP · РАСПРОДАЖА", 301, 398, 29, "Display", ORANGE)
        self.para(
            "Координированные покупки и сообщения в чатах привлекают новых покупателей.",
            26,
            412,
            247,
            23,
            size=22,
            leading=28,
        )
        self.para(
            "Ранние участники продают. Поздние покупатели остаются с падающим активом.",
            301,
            412,
            247,
            23,
            size=22,
            leading=28,
        )
        self.para(
            "<b>VIB/BTC · Binance · 5 сентября 2021.</b><br/>Всплеск и падение цены разворачиваются в течение минуты. Источник: рисунок 1 статьи.",
            583,
            398,
            240,
            36,
            size=20,
            leading=26,
            color=MUTED,
        )

    def _draw_applications(self) -> None:
        # Why the topic matters: three concise applications, not claims of deployed impact.
        self.txt("Почему раннее выявление важно", 18, 447, 32, "Display")
        card_w = (805 - 24) / 3
        cards = [
            (
                "Защита инвесторов",
                "Поздний вход на ажиотаже может привести к убыткам. Ранний сигнал даёт время оценить риск до массовых покупок.",
                TEAL,
            ),
            (
                "Рыночный надзор",
                "Согласованные сделки искажают картину спроса. Сигнал может помочь бирже направить подозрительный актив на проверку.",
                ORANGE,
            ),
            (
                "Поиск ранних признаков",
                "Мы обучаем ИИ на истории сделок: изучаем объёмы, движение цены и активность покупателей и продавцов до объявления.",
                TEAL,
            ),
        ]
        for i, (title, body, accent) in enumerate(cards):
            x = 18 + i * (card_w + 12)
            self.rect(x, 465, card_w, 61, LIGHT, radius=2)
            self.rect(x, 465, 2, 61, accent)
            self.txt(title, x + 8, 473, 29, "Display", accent)
            self.para(body, x + 8, 489, card_w - 16, 33, size=22, leading=28)

        self.rect(18, 537, 805, 19, INK, radius=2)
        self.txt(
            "Быстрый рост цены может быть манипуляцией. Цель исследования — дать время на проверку.",
            28,
            543,
            27,
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
            "Поддержка: Минэкономразвития России · 000000C313925P4E0002 и № 139-15-2025-009. Иллюстрация схемы создана с помощью ИИ.",
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
        self._draw_question()
        self._draw_illustrations()
        self._draw_applications()
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
    *,
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
    VisualPosterRenderer(canvas, results, assets).render()
    canvas.showPage()
    canvas.save()
    return output


def main() -> None:
    print(build_poster())


if __name__ == "__main__":
    main()
