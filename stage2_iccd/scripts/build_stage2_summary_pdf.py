from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SOURCE_MD = ROOT / "results_summary_zh.md"
OUTPUT_PDF = REPO_ROOT / "output" / "pdf" / "stage2_iccd_summary.pdf"


def main() -> None:
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    story = build_story(SOURCE_MD.read_text(encoding="utf-8"), styles)
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=1.45 * cm,
        rightMargin=1.45 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.55 * cm,
        title="Stage2 可微 ICCD 技术总结",
        author="Codex",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT_PDF)


def make_styles() -> dict[str, ParagraphStyle]:
    register_font("MicrosoftYaHei", [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc"])
    register_font("MicrosoftYaHei-Bold", [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"])
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=19,
            leading=26,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1f2a44"),
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=13.6,
            leading=20,
            textColor=colors.HexColor("#1f2a44"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="MicrosoftYaHei-Bold",
            fontSize=11.2,
            leading=16,
            textColor=colors.HexColor("#243b53"),
            spaceBefore=6,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=9.25,
            leading=14.0,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=5.3,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=9.0,
            leading=13.4,
            leftIndent=14,
            firstLineIndent=-8,
            textColor=colors.HexColor("#1f2933"),
            spaceAfter=3.8,
        ),
        "table": ParagraphStyle(
            "table",
            parent=base["BodyText"],
            fontName="MicrosoftYaHei",
            fontSize=6.9,
            leading=9.2,
            textColor=colors.HexColor("#1f2933"),
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "code",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=7.4,
            leading=9.6,
            backColor=colors.HexColor("#f8fafc"),
            borderColor=colors.HexColor("#d9e2ec"),
            borderWidth=0.25,
            borderPadding=4,
            textColor=colors.HexColor("#102a43"),
            spaceAfter=5,
        ),
    }


def register_font(name: str, candidates: list[str]) -> None:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))
            return
    raise FileNotFoundError(f"Cannot find a font for {name}: {candidates}")


def build_story(markdown: str, styles: dict[str, ParagraphStyle]) -> list:
    story: list = [Paragraph("Stage2 可微 ICCD 技术总结", styles["title"])]
    lines = markdown.splitlines()
    idx = 0
    in_code = False
    code_lines: list[str] = []
    while idx < len(lines):
        line = lines[idx].rstrip()
        if line.strip().startswith("```"):
            if in_code:
                story.append(Paragraph("<br/>".join(escape_inline(item) for item in code_lines), styles["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            idx += 1
            continue
        if in_code:
            code_lines.append(line)
            idx += 1
            continue
        if not line.strip():
            story.append(Spacer(1, 3))
            idx += 1
            continue
        if is_table_start(lines, idx):
            table_lines = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                table_lines.append(lines[idx].strip())
                idx += 1
            story.append(markdown_table(table_lines, styles))
            story.append(Spacer(1, 5))
            continue
        if line.startswith("## "):
            story.append(Paragraph(escape_inline(line[3:].strip()), styles["h1"]))
        elif line.startswith("### "):
            story.append(Paragraph(escape_inline(line[4:].strip()), styles["h2"]))
        elif line.startswith("# "):
            story.append(Paragraph(escape_inline(line[2:].strip()), styles["h1"]))
        elif line.lstrip().startswith("- "):
            story.append(Paragraph("• " + escape_inline(line.lstrip()[2:].strip()), styles["bullet"]))
        elif re.match(r"^\s*\d+\.\s+", line):
            text = re.sub(r"^\s*(\d+\.)\s+", r"\1 ", line)
            story.append(Paragraph(escape_inline(text), styles["bullet"]))
        else:
            paragraph_lines = [line]
            idx += 1
            while idx < len(lines):
                nxt = lines[idx].rstrip()
                if not nxt.strip() or nxt.startswith("#") or nxt.lstrip().startswith("- ") or re.match(r"^\s*\d+\.\s+", nxt) or nxt.strip().startswith("|") or nxt.strip().startswith("```"):
                    break
                paragraph_lines.append(nxt)
                idx += 1
            story.append(Paragraph(escape_inline("".join(paragraph_lines)), styles["body"]))
            continue
        idx += 1
    if code_lines:
        story.append(Paragraph("<br/>".join(escape_inline(item) for item in code_lines), styles["code"]))
    return story


def is_table_start(lines: list[str], idx: int) -> bool:
    return (
        idx + 1 < len(lines)
        and lines[idx].strip().startswith("|")
        and lines[idx + 1].strip().startswith("|")
        and set(lines[idx + 1].replace("|", "").replace(":", "").replace(" ", "").strip()) <= {"-"}
    )


def markdown_table(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    rows = []
    for pos, line in enumerate(lines):
        if pos == 1:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append([Paragraph(escape_inline(cell), styles["table"]) for cell in cells])
    col_count = max(len(row) for row in rows)
    for row in rows:
        while len(row) < col_count:
            row.append(Paragraph("", styles["table"]))
    available = A4[0] - 2.9 * cm
    col_widths = [available / col_count] * col_count
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6f0ff")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bcccdc")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 3.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
            ]
        )
    )
    return table


def escape_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = text.replace("**", "")
    return text


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("MicrosoftYaHei", 8)
    canvas.setFillColor(colors.HexColor("#627d98"))
    canvas.drawString(1.45 * cm, 1.0 * cm, "Stage2 可微 ICCD 技术总结")
    canvas.drawRightString(A4[0] - 1.45 * cm, 1.0 * cm, f"第 {doc.page} 页")
    canvas.restoreState()


if __name__ == "__main__":
    main()
