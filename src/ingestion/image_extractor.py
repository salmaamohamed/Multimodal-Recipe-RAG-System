"""Extract embedded recipe photos from a PDF and tie them to recipes."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from src.config import ROOT, settings
from src.ingestion.recipe_parser import Recipe


@dataclass
class ExtractedImage:
    recipe: Recipe
    page_number: int
    path: Path
    caption: str = ""

    @property
    def rel_path(self) -> str:
        try:
            return self.path.relative_to(ROOT).as_posix()
        except ValueError:
            return self.path.as_posix()


def extract_images(pdf_path: Path, owner: dict[int, Recipe], recipes: list[Recipe],
                   out_dir: Path) -> tuple[list[ExtractedImage], list[str]]:
    """Save every usable embedded image of `pdf_path` into `out_dir`.

    Skipped: logos / rules (tiny or extreme aspect ratio), artwork repeated
    across pages (templates, watermarks), full-page backgrounds, and images on
    pages that belong to no recipe (covers, category dividers).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    images: list[ExtractedImage] = []
    notes: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        usage = Counter(x for pg in doc for x in {img[0] for img in pg.get_images(full=True)})
        repeat_limit = 3
        saved: dict[tuple[str, int], Path] = {}  # (recipe_id, xref) -> file
        per_page: Counter[tuple[str, int]] = Counter()
        last_owner: Recipe | None = None

        for pg in doc:
            number = pg.number + 1
            has_text = bool(pg.get_text().strip())
            recipe = owner.get(number)
            if recipe is None and not has_text:
                # picture-only page: belongs to the recipe just before it
                recipe = last_owner or (recipes[0] if len(recipes) == 1 else None)
            if recipe is not None:
                last_owner = recipe
            page_area = abs(pg.rect)

            for img in pg.get_images(full=True):
                xref, width, height = img[0], img[2], img[3]
                if min(width, height) < settings.min_image_side:
                    continue
                if max(width, height) / min(width, height) > settings.max_image_aspect:
                    continue
                if usage[xref] >= repeat_limit:
                    continue
                rects = pg.get_image_rects(xref)
                placed = [r for r in rects if abs(r & pg.rect) < 0.7 * page_area]
                if has_text and rects and not placed:
                    continue  # page background
                if recipe is None:
                    notes.append(f"page {number}: image not linked to any recipe (skipped)")
                    continue
                if (recipe.recipe_id, xref) in saved:
                    continue
                try:
                    pix = pymupdf.Pixmap(doc, xref)
                    if pix.colorspace is None or pix.colorspace.n != 3:
                        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                    if pix.alpha:
                        pix = pymupdf.Pixmap(pix, 0)
                    per_page[(recipe.recipe_id, number)] += 1
                    name = (f"{recipe.recipe_id}_page_{number}"
                            f"_image_{per_page[(recipe.recipe_id, number)]}.jpg")
                    path = out_dir / name
                    pix.save(path, jpg_quality=90)
                except Exception as exc:  # corrupt / unsupported stream
                    notes.append(f"page {number}: could not extract image xref {xref} ({exc})")
                    continue
                saved[(recipe.recipe_id, xref)] = path
                images.append(ExtractedImage(recipe, number, path))
    return images, notes
