"""Recipe-aware chunking: one chunk per recipe section, never across recipes."""
from __future__ import annotations

from src.config import settings
from src.ingestion.image_extractor import ExtractedImage
from src.ingestion.recipe_parser import INFO_LABELS, SECTIONS, Recipe

SECTION_TITLES = {
    "description": "Description", "ingredients": "Ingredients",
    "instructions": "Instructions", "cooking_info": "Cooking Information",
    "nutrition": "Nutrition", "notes": "Tips / Notes",
}


def info_text(info: dict[str, str]) -> str:
    return "\n".join(f"{label}: {info[key]}" for key, label in INFO_LABELS.items() if key in info)


def _split(items: list[tuple[str, int]], limit: int) -> list[list[tuple[str, int]]]:
    """Split a long section on line boundaries."""
    parts, cur, size = [], [], 0
    for text, page in items:
        if cur and size + len(text) > limit:
            parts.append(cur)
            cur, size = [], 0
        cur.append((text, page))
        size += len(text) + 1
    if cur:
        parts.append(cur)
    return parts


def _base_meta(recipe: Recipe) -> dict:
    return {
        "recipe_id": recipe.recipe_id,
        "recipe_name": recipe.name,
        "source_pdf": recipe.source_pdf,
        "category": recipe.category,
        **{k: recipe.info.get(k, "") for k in INFO_LABELS},
    }


def text_records(recipe: Recipe) -> list[dict]:
    """Vector records ({id, content, metadata}) for the text of one recipe."""
    records: list[dict] = []
    header = f"Recipe: {recipe.name}"
    if recipe.category:
        header += f"\nCategory: {recipe.category}"

    def add(section: str, body: str, page: int, pages: list[int], index: int) -> None:
        records.append({
            "id": f"{recipe.recipe_id}::text::{section}::{index}",
            "content": f"{header}\nSection: {SECTION_TITLES[section]}\n{body}".strip(),
            "metadata": {
                **_base_meta(recipe), "modality": "text", "section": section,
                "chunk_index": index, "page_number": page,
                "pages": ",".join(map(str, pages)),
            },
        })

    first_page = recipe.pages[0] if recipe.pages else 0
    for section in SECTIONS:
        if section == "cooking_info":
            body = info_text(recipe.info)
            if body:
                add(section, body, first_page, [first_page], 0)
            continue
        items = recipe.sections.get(section, [])
        if not items and section == "description":
            # every recipe gets an identity chunk, even without a blurb
            add(section, f"Source document: {recipe.source_pdf}", first_page, [first_page], 0)
            continue
        for index, part in enumerate(_split(items, settings.max_chunk_chars)):
            pages = sorted({pg for _, pg in part})
            add(section, "\n".join(t for t, _ in part), pages[0], pages, index)
    return records


def image_records(images: list[ExtractedImage]) -> list[dict]:
    """Vector records for image captions. The caption is the embedded content."""
    records = []
    for n, img in enumerate(images, 1):
        records.append({
            "id": f"{img.recipe.recipe_id}::image::{img.path.stem}",
            "content": img.caption,
            "metadata": {
                **_base_meta(img.recipe), "modality": "image",
                "page_number": img.page_number, "pages": str(img.page_number),
                "image_path": img.rel_path, "caption": img.caption,
            },
        })
    return records
