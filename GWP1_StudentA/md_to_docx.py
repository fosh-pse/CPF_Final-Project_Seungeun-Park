"""
md_to_docx.py
=============
Minimal, dependency-light Markdown -> .docx converter tailored to the GWP1
reports. Handles: ATX headings (#..####), pipe tables, bullet/numbered lists,
fenced code blocks (monospace), blockquotes, bold (**...**) and inline `code`,
horizontal rules, and embedded images via a ![alt](path) syntax.

Usage:
    python3 md_to_docx.py INPUT.md OUTPUT.docx [--figures figures]
"""
import re
import sys
import os
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH


def add_runs(paragraph, text):
    """Render **bold** and `code` inline spans."""
    for part in re.split(r"(\*\*.+?\*\*|`.+?`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = paragraph.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = paragraph.add_run(part[1:-1]); r.font.name = "Consolas"
            r.font.size = Pt(9.5); r.font.color.rgb = RGBColor(0xB0, 0x30, 0x30)
        else:
            paragraph.add_run(part)


BODY_FONT = "Arial"


def _force_font(doc):
    """Apply BODY_FONT to every style and run (code spans keep Consolas)."""
    for st in ["Normal", "Heading 1", "Heading 2", "Heading 3", "Heading 4",
               "Title", "Intense Quote", "List Bullet", "List Number"]:
        try:
            doc.styles[st].font.name = BODY_FONT
        except KeyError:
            pass
    paras = list(doc.paragraphs)
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                paras.extend(cell.paragraphs)
    for p in paras:
        for r in p.runs:
            if r.font.name != "Consolas":
                r.font.name = BODY_FONT


def convert(md_path, docx_path, fig_dir="figures"):
    doc = Document()
    doc.styles["Normal"].font.name = BODY_FONT
    doc.styles["Normal"].font.size = Pt(11)

    lines = open(md_path, encoding="utf-8").read().splitlines()
    i, n = 0, len(lines)
    base = os.path.dirname(md_path)

    while i < n:
        ln = lines[i]

        # fenced code block ---------------------------------------------------
        if ln.strip().startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            p = doc.add_paragraph()
            r = p.add_run("\n".join(buf))
            r.font.name = "Consolas"; r.font.size = Pt(9)
            p.paragraph_format.left_indent = Inches(0.2)
            continue

        # horizontal rule -----------------------------------------------------
        if re.match(r"^---+\s*$", ln):
            doc.add_paragraph("_" * 60).alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue

        # image ---------------------------------------------------------------
        m = re.match(r"!\[(.*?)\]\((.+?)\)", ln.strip())
        if m:
            path = m.group(2)
            if not os.path.isabs(path):
                path = os.path.join(base, path)
            if os.path.exists(path):
                doc.add_picture(path, width=Inches(6.2))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue

        # table ---------------------------------------------------------------
        if ln.strip().startswith("|") and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|", lines[i + 1]):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            header, body = rows[0], rows[2:]
            t = doc.add_table(rows=1, cols=len(header))
            t.style = "Light Grid Accent 1"
            for j, c in enumerate(header):
                cell = t.rows[0].cells[j]
                cell.paragraphs[0].add_run(re.sub(r"\*\*", "", c)).bold = True
            for br in body:
                cells = t.add_row().cells
                for j, c in enumerate(br[:len(header)]):
                    add_runs(cells[j].paragraphs[0], c)
            doc.add_paragraph()
            continue

        # headings ------------------------------------------------------------
        h = re.match(r"^(#{1,4})\s+(.*)", ln)
        if h:
            level = len(h.group(1))
            doc.add_heading(re.sub(r"\*\*", "", h.group(2)), level=min(level, 4))
            i += 1
            continue

        # blockquote ----------------------------------------------------------
        if ln.strip().startswith(">"):
            p = doc.add_paragraph(style="Intense Quote")
            add_runs(p, ln.strip()[1:].strip())
            i += 1
            continue

        # bullet / numbered list ----------------------------------------------
        b = re.match(r"^\s*[-*]\s+(.*)", ln)
        o = re.match(r"^\s*\d+\.\s+(.*)", ln)
        if b:
            add_runs(doc.add_paragraph(style="List Bullet"), b.group(1)); i += 1; continue
        if o:
            add_runs(doc.add_paragraph(style="List Number"), o.group(1)); i += 1; continue

        # blank / paragraph ---------------------------------------------------
        if ln.strip() == "":
            i += 1
            continue
        add_runs(doc.add_paragraph(), ln)
        i += 1

    _force_font(doc)
    doc.save(docx_path)
    print(f"wrote {docx_path} (font: {BODY_FONT})")


if __name__ == "__main__":
    md, out = sys.argv[1], sys.argv[2]
    convert(md, out)
