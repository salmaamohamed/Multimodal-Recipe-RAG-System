"""Turn extracted pages into recipes with named sections.

A recipe starts at a title (a line set clearly larger than the body text).
Pages without a title continue the previous recipe; pages holding only a
title are category dividers ("CURRIES", "DESSERTS", ...).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.ingestion.pdf_extractor import Document, Line

SECTIONS = ("description", "ingredients", "instructions", "cooking_info", "nutrition", "notes")

# heading text (lower-cased, trailing colon removed) -> section
_HEADERS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(the )?ingredients?( amounts?| for .{1,30}| list)?$|^amounts?$|^shopping list.*$"), "ingredients"),
    (re.compile(r"^(methods?|instructions?|directions?|preparation|procedure|steps?|"
                r"let[’'`]?s create|how to (make|cook|prepare).{0,30})$"), "instructions"),
    (re.compile(r"^(notes?|tips?|pro tips?|cook[’']?s notes?|variations?|serving( suggestions?)?)$"), "notes"),
    (re.compile(r"^nutrition(al)?( facts| information| info| values?)?$"), "nutrition"),
    (re.compile(r"^overview$"), "cooking_info"),
]
# "Ingredients: flour, salt ..." / "Note: ..." on a single line
_INLINE = re.compile(
    r"^(ingredients?|method|instructions?|directions?|preparation|steps?|"
    r"nutrition(?:al)?(?: facts)?|notes?|tips?|source)\s*:\s*(\S.*)$", re.I)
_INLINE_SECTION = {"ingr": "ingredients", "meth": "instructions", "inst": "instructions",
                   "dire": "instructions", "prep": "instructions", "step": "instructions",
                   "nutr": "nutrition", "note": "notes", "tip": "notes", "tips": "notes",
                   "sour": "notes"}

_INFO_LABEL = re.compile(
    r"\b(prep(?:aration)?\s*time|cook(?:ing)?\s*time|total\s*time|time|serves|servings?|"
    r"yield|makes|difficulty)\s*:\s*", re.I)
_INFO_KEY = {"prep": "prep_time", "cook": "cook_time", "tota": "total_time", "time": "total_time",
             "serv": "servings", "yiel": "servings", "make": "servings", "diff": "difficulty"}
INFO_LABELS = {"prep_time": "Preparation time", "cook_time": "Cooking time",
               "total_time": "Total time", "servings": "Servings", "difficulty": "Difficulty"}


@dataclass
class Recipe:
    name: str
    source_pdf: str
    category: str = ""
    recipe_id: str = ""
    # section -> [(text, page_number)]
    sections: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    info: dict[str, str] = field(default_factory=dict)
    pages: list[int] = field(default_factory=list)

    def text(self, section: str) -> str:
        return "\n".join(t for t, _ in self.sections.get(section, []))


def slugify(text: str, limit: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit].strip("-")


def _header_section(text: str) -> str | None:
    key = text.lower().strip().rstrip(":").strip()
    if len(key) > 45:
        return None
    for pattern, section in _HEADERS:
        if pattern.match(key):
            return section
    return None


def _clean_title(lines: list[str]) -> str:
    title = re.sub(r"\s+", " ", " ".join(lines)).strip(" :-")
    title = re.sub(r"\s+recipes?$", "", title, flags=re.I)
    if title.isupper():
        small = {"and", "of", "with", "in", "the", "for", "a", "ka", "ki"}
        words = title.lower().split()
        title = " ".join(w if i and w in small else w.capitalize() for i, w in enumerate(words))
    return title


def _parse_info(text: str) -> tuple[dict[str, str], bool]:
    """Extract "Prep time: 10 min"-style facts. The flag tells whether the
    line held nothing else (so it can be left out of the section text)."""
    labels = list(_INFO_LABEL.finditer(text))
    info: dict[str, str] = {}
    for m, nxt in zip(labels, labels[1:] + [None]):
        value = text[m.end(): nxt.start() if nxt else len(text)].strip(" ,;|")
        if value and len(value) <= 40:
            info.setdefault(_INFO_KEY[m.group(1).lower()[:4]], value)
    only_info = bool(info) and labels[0].start() == 0 and len(info) == len(labels)
    return info, only_info


def _title_flags(lines: list[Line], body_size: float) -> list[bool]:
    flags = [
        l.size >= 1.25 * body_size and len(l.text) <= 80
        and any(ch.isalpha() for ch in l.text)
        and _header_section(l.text) is None and not _INLINE.match(l.text)
        and not _INFO_LABEL.match(l.text)
        for l in lines
    ]
    if any(flags):  # keep only the dominant heading size of the page
        top = max(l.size for l, f in zip(lines, flags) if f)
        flags = [f and l.size >= 0.8 * top for l, f in zip(lines, flags)]
    return flags


def _reseat_headers(lines: list[Line]) -> list[Line]:
    """Move a section heading directly above the text it visually heads.

    Column ordering can leave e.g. "INSTRUCTIONS" (sitting under a photo) in
    front of an ingredient list from the neighbouring column."""
    lines = list(lines)
    for head in [l for l in lines if _header_section(l.text)]:
        i = lines.index(head)
        for j in range(i + 1, len(lines)):
            o = lines[j]
            if o.y0 >= head.y1 - 1 and o.x0 < head.x1 and head.x0 < o.x1:
                if j > i + 1:
                    lines.insert(j - 1, lines.pop(i))
                break
    return lines


def parse_recipes(doc: Document) -> tuple[list[Recipe], dict[int, Recipe], list[str]]:
    """Returns (recipes, page -> owning recipe, notes about skipped content)."""
    notes: list[str] = []
    drafts: list[tuple[Recipe, list[Line]]] = []
    category = ""

    def start(name: str) -> None:
        drafts.append((Recipe(name=name, source_pdf=doc.name, category=category), []))

    for page in doc.pages:
        lines = _reseat_headers(page.lines)
        if not lines:
            continue
        flags = _title_flags(lines, doc.body_size)
        body_chars = sum(len(l.text) for l, f in zip(lines, flags) if not f)

        if any(flags) and body_chars < 20:  # divider page
            category = _clean_title([l.text for l, f in zip(lines, flags) if f])
            notes.append(f"page {page.number}: category divider '{category}'")
            continue

        i = 0
        first_title = True
        while i < len(lines):
            if not flags[i]:
                if not drafts:
                    start("")  # document without a recognisable title
                drafts[-1][1].append(lines[i])
                i += 1
                continue
            j = i
            while j < len(lines) and flags[j]:
                j += 1
            title = _clean_title([l.text for l in lines[i:j]])
            same = drafts and slugify(drafts[-1][0].name) == slugify(title)
            if not same:
                # a title sitting above every heading owns the whole page, even
                # when the column order placed a few lines before it
                headers_above = any(
                    _header_section(l.text) and l.y0 < lines[i].y0 for l in lines[:i])
                stolen: list[Line] = []
                if first_title and drafts and not headers_above:
                    prev = drafts[-1][1]
                    while prev and prev[-1].page == page.number:
                        stolen.insert(0, prev.pop())
                start(title)
                drafts[-1][1].extend(stolen)
            first_title = False
            i = j

    recipes: list[Recipe] = []
    seen_ids: dict[str, int] = {}
    stem = re.sub(r"\.pdf$", "", doc.name, flags=re.I)
    for recipe, lines in drafts:
        if not lines:
            continue
        if not recipe.name:
            first = lines[0].text
            if len(first) <= 60 and _header_section(first) is None:
                recipe.name = _clean_title([first])
                lines = lines[1:]
            else:
                recipe.name = stem
                notes.append(f"no title found, using file name for '{stem}'")
        _assign_sections(recipe, lines)
        if not recipe.sections.get("ingredients") and not recipe.sections.get("instructions"):
            notes.append(f"skipped non-recipe content '{recipe.name}' "
                         f"(page {lines[0].page}): no ingredients or instructions")
            continue
        base = f"{slugify(stem, 40)}__{slugify(recipe.name)}"
        seen_ids[base] = seen_ids.get(base, 0) + 1
        recipe.recipe_id = base if seen_ids[base] == 1 else f"{base}-{seen_ids[base]}"
        recipes.append(recipe)

    owner: dict[int, Recipe] = {}
    for recipe in recipes:
        counts: dict[int, int] = {}
        for items in recipe.sections.values():
            for _, pg in items:
                counts[pg] = counts.get(pg, 0) + 1
        recipe.pages = sorted(counts)
        for pg in counts:  # a page shared by two recipes goes to the first one
            owner.setdefault(pg, recipe)
    return recipes, owner, notes


def _assign_sections(recipe: Recipe, lines: list[Line]) -> None:
    section = "description"
    current: Line | None = None  # heading that opened the current section
    headings = [(l, _header_section(l.text)) for l in lines]
    headings = [(l, sec) for l, sec in headings if sec]

    def overlaps(head: Line, line: Line) -> bool:
        return head.page == line.page and head.x0 < line.x1 and line.x0 < head.x1

    for line in lines:
        text = line.text
        header = _header_section(text)
        if header:
            section, current = header, line
            continue
        # Text placed under another heading than the one reading order gave it
        # (a full-width step list resuming below a side-by-side ingredient
        # column) follows the nearer heading that actually sits above it.
        if current is not None and current.page == line.page and not overlaps(current, line):
            above = [(h, sec) for h, sec in headings
                     if overlaps(h, line) and current.y0 < h.y0 <= line.y0]
            if above:
                current, section = max(above, key=lambda hs: hs[0].y0)
        inline = _INLINE.match(text)
        if inline:
            key = inline.group(1).lower()
            target = _INLINE_SECTION.get(key[:4], _INLINE_SECTION.get(key))
            if target in ("notes",):
                # one-line remark: file it under notes without switching section
                recipe.sections.setdefault("notes", []).append((text, line.page))
                continue
            if target:
                section, text = target, inline.group(2)
        info, only_info = _parse_info(text)
        for k, v in info.items():
            recipe.info.setdefault(k, v)
        if only_info:
            continue
        if section == "cooking_info":
            section = "description"  # free text after the overview block
        recipe.sections.setdefault(section, []).append((text, line.page))
