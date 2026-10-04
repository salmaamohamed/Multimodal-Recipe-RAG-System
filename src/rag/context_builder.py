"""Build the structured, source-attributed context handed to the LLM."""
from __future__ import annotations

from src.config import settings
from src.ingestion.chunker import SECTION_TITLES
from src.ingestion.recipe_parser import SECTIONS
from src.retrieval.catalog import content_terms
from src.retrieval.hybrid_retriever import RecipeResult


def _body(content: str) -> str:
    """Strip the "Recipe: / Category: / Section:" header stored with a chunk."""
    lines = content.split("\n")
    while lines and lines[0].startswith(("Recipe:", "Category:", "Section:")):
        lines.pop(0)
    return "\n".join(lines).strip()


def recipe_sections(store, recipe_id: str) -> tuple[dict[str, str], list[dict]]:
    """All stored text of a recipe by section, plus its image records."""
    records = store.recipe_records(recipe_id)
    text = sorted((r for r in records if r["metadata"]["modality"] == "text"),
                  key=lambda r: (SECTIONS.index(r["metadata"]["section"]),
                                 r["metadata"]["chunk_index"]))
    sections: dict[str, str] = {}
    for r in text:
        key = r["metadata"]["section"]
        sections[key] = (sections.get(key, "") + "\n" + _body(r["content"])).strip()
    images = sorted((r for r in records if r["metadata"]["modality"] == "image"),
                    key=lambda r: r["id"])
    return sections, images


def build_context(results: list[RecipeResult], store) -> str:
    """One block per ranked recipe: complete sections once, never repeated chunks."""
    blocks = []
    results = results[: settings.max_context_recipes]
    for n, result in enumerate(results, 1):
        sections, images = recipe_sections(store, result.recipe_id)
        title = "===== RECIPE =====" if len(results) == 1 else f"===== RECIPE {n} ====="
        pages = ", ".join(map(str, result.pages)) or "unknown"
        parts = [title, f"Recipe: {result.recipe_name}", f"Source PDF: {result.source_pdf}",
                 f"Page: {pages}"]
        if result.category:
            parts.append(f"Category: {result.category}")
        matched = [f"{kind} similarity {score:.2f}" for kind, score in
                   (("text", result.text_score), ("image", result.image_score)) if score]
        parts.append(f"Matched by: {', '.join(matched)}")

        budget = settings.max_chars_per_recipe
        for key in SECTIONS:
            body = sections.get(key, "")
            if key == "description" and body.startswith("Source document:"):
                continue
            if not body or budget <= 0:
                continue
            if len(body) > budget:
                body = body[:budget].rsplit("\n", 1)[0] + "\n[...truncated]"
            budget -= len(body)
            parts.append(f"\n{SECTION_TITLES[key]}:\n{body}")

        # matched images first, then one more of the recipe's own pictures
        shown = {h["metadata"]["image_path"] for h in result.image_results}
        chosen = [h["metadata"] for h in result.image_results]
        chosen += [r["metadata"] for r in images if r["metadata"]["image_path"] not in shown][:1]
        for meta in chosen[:3]:
            parts.append(f"\nImage Description (automatic caption, approximate): {meta['caption']}"
                         f"\nImage: {meta['image_path']} (page {meta['page_number']})")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def build_keyword_index(question: str | None, catalog, max_recipes: int = 20) -> str:
    """List every recipe whose ingredient list mentions a word of the question.

    Vector search only surfaces a handful of recipes; this lets "which recipes
    contain potatoes?" be answered across the whole knowledge base."""
    lines = []
    for term in content_terms(question or ""):
        hits = catalog.with_ingredient(term)
        if not hits or len(hits) > max_recipes:  # absent, or too common to be useful
            continue
        listed = "; ".join(
            f"{r['recipe_name']} ({r['source_pdf']}, page {', '.join(map(str, r['pages']))})"
            for r in hits)
        lines.append(f'Recipes whose ingredient list mentions "{term}" ({len(hits)}): {listed}')
    return "===== INGREDIENT INDEX =====\n" + "\n".join(lines) if lines else ""
