"""End-to-end query pipeline: validate -> retrieve -> rank -> context -> answer."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from src.config import NOT_RECIPE_MESSAGE, settings
from src.rag import generator
from src.rag.context_builder import build_context, build_keyword_index, recipe_sections
from src.rag.kb_detector import RECIPE_FOUND, RECIPE_NOT_FOUND, Evidence, detect
from src.rag.llm import get_llm
from src.retrieval.catalog import content_terms, get_catalog
from src.retrieval.hybrid_retriever import HybridRetriever, RecipeResult
from src.validation.query_validator import NOT_RECIPE, Validation, validate_query


@dataclass
class QueryResult:
    status: str                       # RECIPE_FOUND | RECIPE_NOT_FOUND | NOT_RECIPE
    answer: str
    validation: Validation | None = None
    evidence: Evidence | None = None
    image_caption: str | None = None
    recipes: list[RecipeResult] = field(default_factory=list)   # ranked retrieval results
    used: list[RecipeResult] = field(default_factory=list)      # recipes the answer relies on
    context: str = ""
    answered_by: str = ""             # llm | extractive | rule

    @property
    def sources(self) -> list[dict]:
        return [{"recipe_name": r.recipe_name, "source_pdf": r.source_pdf, "pages": r.pages}
                for r in self.used]

    @property
    def images(self) -> list[str]:
        return [r.images[0] for r in self.used if r.images]


_LISTING = re.compile(
    r"\b(which|what|any|all|list|show|find|suggest|recommend)\b.{0,40}\b(recipes?|dishes|cook|make)\b"
    r"|\brecipes? (with|that|using|containing)\b", re.I)


def _listing_addendum(question: str | None, answer: str, catalog) -> str:
    """For "which recipes contain X" questions, name the matching recipes the
    answer left out. Vector search and the LLM only see a few recipes; the
    ingredient index covers the whole knowledge base."""
    if not question or not _LISTING.search(question):
        return ""
    lines = []
    for term in content_terms(question):
        hits = catalog.with_ingredient(term)
        if not hits or len(hits) > 20:
            continue
        missing = [r for r in hits if r["recipe_name"].lower() not in answer.lower()]
        if missing:
            listed = "\n".join(
                f"- {r['recipe_name']} — {r['source_pdf']}, page {', '.join(map(str, r['pages']))}"
                for r in missing)
            lines.append(f"**Other recipes whose ingredient list mentions “{term}”:**\n{listed}")
    return "\n\n" + "\n\n".join(lines) if lines else ""


class KnowledgeBaseMissing(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    from src.embeddings.embedder import get_embedder
    from src.vectorstore.chroma_store import RecipeStore
    store = RecipeStore()
    catalog = get_catalog()
    if store.count() == 0 or len(catalog) == 0:
        raise KnowledgeBaseMissing(
            "The recipe knowledge base is empty. Run `python scripts/ingest.py` first.")
    if store.embedding_model and store.embedding_model != settings.embedding_model:
        raise KnowledgeBaseMissing(
            f"The vector store was built with '{store.embedding_model}' but EMBEDDING_MODEL is "
            f"'{settings.embedding_model}'. Re-run `python scripts/ingest.py`.")
    return HybridRetriever(store, get_embedder(), catalog)


def process_query(question: str | None, image=None) -> QueryResult:
    """Answer a recipe question, optionally accompanied by an image
    (PIL image or path)."""
    question = (question or "").strip() or None
    llm = get_llm()
    catalog = get_catalog()

    # 1-2) validate; reject before any retrieval
    validation = validate_query(question, image is not None, llm, catalog)
    if not validation.is_recipe:
        return QueryResult(NOT_RECIPE, NOT_RECIPE_MESSAGE, validation, answered_by="rule")

    # 3-5) image -> BLIP caption
    caption = None
    if image is not None:
        from src.ingestion.captioner import caption_image
        caption = caption_image(image)

    # 6-9) embed, hybrid retrieval, group and rank per recipe
    retriever = get_retriever()
    results = retriever.retrieve(question, caption)
    evidence = detect(question, results, catalog)
    top = results[: settings.max_context_recipes]
    if evidence.status == RECIPE_NOT_FOUND:
        return QueryResult(RECIPE_NOT_FOUND,
                           generator.not_found_message(question, evidence, results),
                           validation, evidence, caption, results, answered_by="rule")

    # 10-11) context + grounded answer
    context = build_context(top, retriever.store)
    index = build_keyword_index(question, catalog)
    if index:
        context = index + "\n\n" + context
    note = ""
    if llm.available:
        try:
            status, answer, used = generator.generate(llm, question, caption, context, len(top))
            used_recipes = [top[i] for i in used]
            if status == RECIPE_FOUND and not used_recipes:
                used_recipes = top[:1]
            if status == RECIPE_FOUND:
                answer += _listing_addendum(question, answer, catalog)
            return QueryResult(status, answer, validation, evidence, caption, results,
                               used_recipes, context, "llm")
        except Exception as exc:
            note = f"\n\n_(LLM unavailable: {type(exc).__name__}; showing stored recipe text.)_"
    # no LLM: quote the best-matching recipe verbatim
    sections, _ = recipe_sections(retriever.store, top[0].recipe_id)
    answer = generator.extractive_answer(top, sections) + note
    return QueryResult(RECIPE_FOUND, answer, validation, evidence, caption, results,
                       top[:1], context, "extractive")
