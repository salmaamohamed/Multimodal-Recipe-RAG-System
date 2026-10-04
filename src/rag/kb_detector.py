"""Decide whether the knowledge base actually holds evidence for a request.

Similar wording in an unrelated recipe is not evidence: the dish / ingredient
words of the question have to occur in the retrieved recipes, or the semantic
match has to be strong.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import settings
from src.retrieval.catalog import Catalog, content_terms
from src.retrieval.hybrid_retriever import RecipeResult

RECIPE_FOUND = "RECIPE_FOUND"
RECIPE_NOT_FOUND = "RECIPE_NOT_FOUND"


@dataclass
class Evidence:
    status: str
    reason: str
    terms: list[str] = field(default_factory=list)      # dish / ingredient words asked for
    matched: list[str] = field(default_factory=list)    # ...found in the retrieved recipes
    missing: list[str] = field(default_factory=list)    # ...absent from the whole KB


def detect(question: str | None, results: list[RecipeResult], catalog: Catalog) -> Evidence:
    if not results:
        return Evidence(RECIPE_NOT_FOUND, "nothing retrieved")
    top = results[: settings.max_context_recipes]
    best = top[0].score
    if best < settings.min_recipe_score:
        return Evidence(RECIPE_NOT_FOUND, f"best score {best:.2f} below threshold")

    terms = content_terms(question or "")
    if not terms:  # image-only or fully generic question: rely on similarity
        return Evidence(RECIPE_FOUND, f"best score {best:.2f}")

    matched = [t for t in terms if any(catalog.has_term(r.recipe_id, t) for r in top)]
    missing = [t for t in terms if not catalog.known(t)]
    if not matched and best < settings.strong_recipe_score:
        return Evidence(RECIPE_NOT_FOUND,
                        f"none of {terms} occur in the retrieved recipes", terms, matched, missing)
    return Evidence(RECIPE_FOUND, f"best score {best:.2f}, matched {matched}",
                    terms, matched, missing)
