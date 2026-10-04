"""Layout-aware PDF text extraction.

Produces, for every page, a list of text lines in reading order. Handles:
  * letter-spaced fonts ("M U S H R O O M") by rebuilding words from glyph gaps
  * multi-column pages and ingredient/amount tables (recursive XY-cut)
  * repeated headers / footers / page numbers
  * scanned pages (optional Tesseract OCR fallback)
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import pymupdf

from src.config import settings

MIN_GUTTER = 10.0  # minimum horizontal whitespace (pt) to treat as a column gap


@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    bold: bool = False
    page: int = 0


@dataclass
class Page:
    number: int  # 1-based
    width: float
    height: float
    lines: list[Line] = field(default_factory=list)
    ocr_used: bool = False


@dataclass
class Document:
    path: Path
    pages: list[Page]
    body_size: float
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name


# ---------------------------------------------------------------- raw lines

def _spaced_out(chars: list[dict], min_glyphs: int) -> bool:
    """True for "M U S H R O O M": every glyph but the last precedes a space."""
    glyphs = [i for i, c in enumerate(chars) if c["c"] != " "]
    return len(glyphs) >= min_glyphs and sum(
        1 for i in glyphs[:-1] if chars[i + 1]["c"] == " "
    ) >= 0.8 * (len(glyphs) - 1)


def _span_segments(span: dict, spaced_fonts: set[str]) -> list[tuple[str, float, float]]:
    """Split one span into (text, x0, x1) segments, repairing letter-spacing
    and breaking where a wide horizontal gap separates table cells."""
    chars = span["chars"]
    if not chars:
        return []
    size = span["size"]
    glyph_gaps = [
        b["bbox"][0] - a["bbox"][2]
        for a, b in zip(chars, chars[1:]) if a["c"] != " " and b["c"] != " "
    ]
    tracked = len(glyph_gaps) >= 3 and median(glyph_gaps) > 0.08 * size
    # short spans only count when their font is letter-spaced elsewhere on the page
    min_glyphs = 2 if span["font"] in spaced_fonts else 4
    letter_spaced = _spaced_out(chars, min_glyphs) or tracked
    cell_gap = max(2.2 * size, 18.0)

    segments: list[tuple[str, float, float]] = []
    text, x0, x1 = "", None, None
    prev = None  # previous glyph, spaces included
    for c in chars:
        cx0, cx1 = c["bbox"][0], c["bbox"][2]
        if c["c"] == " ":
            if letter_spaced:
                # an explicit space only marks a word break when extra
                # advance was inserted before it
                if prev is not None and cx0 - prev["bbox"][2] > 0.1 * size and text:
                    text += " "
            elif text:
                text += " "
            prev = c
            continue
        if x1 is not None and cx0 - x1 > cell_gap:
            segments.append((text, x0, x1))
            text, x0 = "", None
        if x0 is None:
            x0 = cx0
        text += c["c"]
        x1 = cx1
        prev = c
    if text.strip():
        segments.append((text, x0, x1))
    return segments


def _raw_lines(page: pymupdf.Page) -> list[Line]:
    lines: list[Line] = []
    blocks = [b for b in page.get_text("rawdict")["blocks"] if b["type"] == 0]
    spaced_fonts = {
        span["font"] for b in blocks for ln in b["lines"] for span in ln["spans"]
        if _spaced_out(span["chars"], 6)
    }
    for block in blocks:
        for ln in block["lines"]:
            if abs(ln["dir"][1]) > 0.1:  # skip rotated text
                continue
            parts: list[tuple[str, float, float, float, bool]] = []
            for span in ln["spans"]:
                bold = bool(span["flags"] & 16) or "bold" in span["font"].lower()
                for text, x0, x1 in _span_segments(span, spaced_fonts):
                    parts.append((text, x0, x1, span["size"], bold))
            if not parts:
                continue
            y0, y1 = ln["bbox"][1], ln["bbox"][3]
            # glue adjacent spans of the same line, split on wide gaps
            cur = list(parts[0])
            for text, x0, x1, size, bold in parts[1:]:
                gap = x0 - cur[2]
                if gap > max(2.2 * size, 18.0):
                    lines.append(_mk(cur, y0, y1, page.number + 1))
                    cur = [text, x0, x1, size, bold]
                else:
                    sep = " " if gap > 0.25 * size and not cur[0].endswith(" ") else ""
                    cur[0] += sep + text
                    cur[2] = max(cur[2], x1)
                    cur[3] = max(cur[3], size)
                    cur[4] = cur[4] and bold
            lines.append(_mk(cur, y0, y1, page.number + 1))
    return [l for l in lines if l.text and not _BULLET_ONLY.match(l.text)]


_BULLET_ONLY = re.compile(r"^[•●▪◦‣·\-\*\s]+$")
_LEADING_BULLET = re.compile(r"^[•●▪◦‣·]\s*")


def _mk(cur: list, y0: float, y1: float, page: int) -> Line:
    text = _LEADING_BULLET.sub("", re.sub(r"\s+", " ", cur[0]).strip())
    pad = 0.15 * (y1 - y0)  # font line boxes include leading; keep the ink
    return Line(text, cur[1], y0 + pad, cur[2], y1 - pad, round(cur[3], 1), cur[4], page)


# ------------------------------------------------------------ reading order

def _merged_intervals(spans: list[tuple[float, float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _merge_rows(lines: list[Line]) -> list[Line]:
    """Join lines that share a baseline into one line (left to right)."""
    rows: list[list[Line]] = []
    for ln in sorted(lines, key=lambda l: ((l.y0 + l.y1) / 2, l.x0)):
        mid = (ln.y0 + ln.y1) / 2
        if rows:
            ref = rows[-1][0]
            same_scale = max(ln.size, ref.size) <= 1.5 * min(ln.size, ref.size)
            if same_scale and abs(mid - (ref.y0 + ref.y1) / 2) < 0.6 * (ref.y1 - ref.y0):
                rows[-1].append(ln)
                continue
        rows.append([ln])
    out = []
    for row in rows:
        row.sort(key=lambda l: l.x0)
        out.append(Line(
            " ".join(l.text for l in row),
            row[0].x0, min(l.y0 for l in row), max(l.x1 for l in row),
            max(l.y1 for l in row), max(l.size for l in row),
            all(l.bold for l in row), row[0].page,
        ))
    return out


def _is_table(left: list[Line], right: list[Line]) -> bool:
    """A narrow right-hand column of short cells aligned with the rows on the
    left is a table (ingredient | amount), not a second text column."""
    if len(right) < 2 or median(len(l.text) for l in right) > 12:
        return False
    aligned = sum(
        1 for r in right
        if any(min(r.y1, l.y1) - max(r.y0, l.y0) > 0.5 * (r.y1 - r.y0) for l in left)
    )
    return aligned >= 0.6 * len(right)


def order_lines(lines: list[Line]) -> list[Line]:
    """Recursive XY-cut: columns are read top-to-bottom, left column first."""
    if len(lines) <= 1:
        return list(lines)

    # 1) a clear vertical gap between blocks (title / body / columns)
    ys = _merged_intervals([(l.y0, l.y1) for l in lines])
    h_gaps = [(b[0] - a[1], (a[1] + b[0]) / 2) for a, b in zip(ys, ys[1:])]
    line_h = median(l.y1 - l.y0 for l in lines)
    xs = _merged_intervals([(l.x0, l.x1) for l in lines])
    gaps = [(b[0] - a[1], (a[1] + b[0]) / 2) for a, b in zip(xs, xs[1:])]
    gaps = sorted((g for g in gaps if g[0] >= MIN_GUTTER), reverse=True)

    if h_gaps and max(h_gaps)[0] >= 1.5 * line_h:
        cut_y = max(h_gaps)[1]
        # ...unless the block above is one row of column headings
        # ("Ingredients | Method"): then each heading stays with its column
        top = [l for l in lines if (l.y0 + l.y1) / 2 < cut_y]
        one_row = len(top) >= 2 and max(l.y0 for l in top) < min(l.y1 for l in top)
        headed = one_row and gaps and any(l.x1 <= gaps[0][1] for l in top)             and any(l.x0 >= gaps[0][1] for l in top)
        if not headed:
            return _split_rows(lines, cut_y)

    # 2) a column gutter
    for _, cut in gaps:
        left = [l for l in lines if l.x1 <= cut]
        right = [l for l in lines if l.x0 >= cut]
        if len(left) >= 2 and len(right) >= 2:
            if _is_table(left, right):
                return _merge_rows(lines)
            return order_lines(left) + order_lines(right)

    # 3) any remaining vertical gap, else the lines share one row
    if h_gaps:
        return _split_rows(lines, max(h_gaps)[1])
    return _merge_rows(lines)


def _split_rows(lines: list[Line], cut: float) -> list[Line]:
    top = [l for l in lines if (l.y0 + l.y1) / 2 < cut]
    bottom = [l for l in lines if (l.y0 + l.y1) / 2 >= cut]
    if not top or not bottom:
        return _merge_rows(lines)
    return order_lines(top) + order_lines(bottom)


def _join_label_values(lines: list[Line]) -> list[Line]:
    """Attach a value to the "LABEL:" sitting left of it on the same baseline
    (e.g. "PREP TIME:" + "10 MIN"), wherever the two were split apart."""
    used: set[int] = set()
    out: list[Line] = []
    for i, ln in enumerate(lines):
        if i in used:
            continue
        if ln.text.endswith(":") and len(ln.text) <= 20:
            mid = (ln.y0 + ln.y1) / 2
            cands = [
                (o.x0 - ln.x1, j) for j, o in enumerate(lines)
                if j != i and j not in used and o.y0 < mid < o.y1
                and 0 <= o.x0 - ln.x1 < 120 and len(o.text) <= 25
                and not o.text.endswith(":")
            ]
            if cands:
                _, j = min(cands)
                o = lines[j]
                used.add(j)
                ln = Line(f"{ln.text} {o.text}", ln.x0, min(ln.y0, o.y0), o.x1,
                          max(ln.y1, o.y1), ln.size, ln.bold, ln.page)
        out.append(ln)
    return out


# --------------------------------------------------------- headers / footers

_PAGE_NO = re.compile(r"^(page\s*)?\d{1,3}(\s*(/|of)\s*\d{1,3})?$", re.I)
_NOISE = re.compile(
    r"^(\.{2,}\s*continued|continue[sd]\s*\.{2,}|[\w-]+(\.[\w-]+)*\.(com|org|net)(\.\w{2})?)$",
    re.I,
)
# never strip lines that look like section headings even if they repeat
_KEEP = re.compile(r"^(ingredients?|method|instructions?|directions?|notes?|tips?|steps?)\b", re.I)


def _strip_boilerplate(pages: list[Page]) -> None:
    def norm(t: str) -> str:
        return re.sub(r"\d+", "#", t.lower()).strip()

    seen: Counter[str] = Counter()
    for p in pages:
        seen.update({norm(l.text) for l in p.lines})
    # repetition is only meaningful on longer documents
    limit = max(3, 0.4 * len(pages)) if len(pages) >= 5 else float("inf")

    for p in pages:
        kept = []
        for l in p.lines:
            edge = l.y1 < 0.10 * p.height or l.y0 > 0.88 * p.height
            repeated = seen[norm(l.text)] >= limit and not _KEEP.match(l.text)
            if _NOISE.match(l.text):
                continue
            if edge and _PAGE_NO.match(l.text):
                continue
            if repeated and (edge or len(l.text) > 25):
                continue
            kept.append(l)
        p.lines = kept


# ---------------------------------------------------------------------- OCR

def _ocr_page(page: pymupdf.Page) -> list[Line]:
    """OCR fallback for scanned pages. Returns [] when Tesseract is missing."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return []
    try:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    scale = 72 / 200
    rows: dict[tuple, list[int]] = {}
    for i, word in enumerate(data["text"]):
        if word.strip() and float(data["conf"][i]) >= 40:
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            rows.setdefault(key, []).append(i)
    lines = []
    for idx in rows.values():
        x0 = min(data["left"][i] for i in idx) * scale
        y0 = min(data["top"][i] for i in idx) * scale
        x1 = max(data["left"][i] + data["width"][i] for i in idx) * scale
        y1 = max(data["top"][i] + data["height"][i] for i in idx) * scale
        text = " ".join(data["text"][i] for i in idx)
        lines.append(Line(text, x0, y0, x1, y1, round(y1 - y0, 1), False, page.number + 1))
    return lines


def _needs_ocr(page: pymupdf.Page, lines: list[Line]) -> bool:
    if sum(len(l.text) for l in lines) >= settings.ocr_min_chars:
        return False
    area = abs(page.rect)
    for img in page.get_images(full=True):
        for rect in page.get_image_rects(img[0]):
            if abs(rect & page.rect) > 0.5 * area:
                return True
    return False


# --------------------------------------------------------------------- main

def extract_document(path: Path) -> Document:
    warnings: list[str] = []
    with pymupdf.open(path) as doc:
        if doc.is_encrypted and not doc.authenticate(""):
            raise ValueError("PDF is password protected")
        pages: list[Page] = []
        for pg in doc:
            lines = _raw_lines(pg)
            ocr_used = False
            if _needs_ocr(pg, lines):
                ocr = _ocr_page(pg)
                native = sum(len(l.text) for l in lines)
                if sum(len(l.text) for l in ocr) >= max(settings.ocr_min_chars, 3 * native):
                    lines, ocr_used = ocr, True
            pages.append(Page(pg.number + 1, pg.rect.width, pg.rect.height, lines, ocr_used))

    _strip_boilerplate(pages)
    sizes: Counter[float] = Counter()
    for p in pages:
        p.lines = order_lines(_join_label_values(p.lines))
        for l in p.lines:
            sizes[l.size] += len(l.text)
        if not p.lines:
            warnings.append(f"page {p.number}: no extractable text")
        if p.ocr_used:
            warnings.append(f"page {p.number}: text obtained through OCR")
    body_size = sizes.most_common(1)[0][0] if sizes else 0.0
    return Document(path, pages, body_size, warnings)
