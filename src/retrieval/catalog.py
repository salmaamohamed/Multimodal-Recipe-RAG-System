"""Recipe catalog written by ingestion: names, sources and vocabulary of every
recipe. Used for keyword evidence checks without touching the vector store."""
from __future__ import annotations

import json
import re
from functools import lru_cache

from src.config import settings

STOPWORDS = set("""
a about all also an and any are as at be been but by can could did do does for from get give
had has have how i if in into is it its just like many me more most much my need needs of on
or our out please should show so some tell than that the their them then there these they
this those to too us use used using want was we what when where which who why will with
would you your one two three don doesn isn not can cannot here there now really very
""".split())

# words that say "this is about cooking" but do not identify a dish or ingredient
GENERIC = set("""
recipe recipes dish dishes food foods meal meals make making made cook cooking cooked prepare
prepared preparing preparation ingredient ingredients step steps instruction instructions
method methods time long minutes minute hours hour serve serves serving servings portion
portions kitchen knowledge base available contain contains containing include includes
included closest similar look looks image picture photo shown substitute substituted
substitution replace replacement instead alternative compare comparison difference between
versus best good easy quick simple list find something anything cuisine eat eating tasty
vegetarian vegan healthy spicy sweet hot cold fresh
""".split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())


def forms(token: str) -> set[str]:
    """Token plus naive singular forms, so 'potatoes' meets 'potato'."""
    out = {token}
    if token.endswith("ies") and len(token) > 4:
        out |= {token[:-3] + "y", token[:-1]}
    if token.endswith("es") and len(token) > 4:
        out.add(token[:-2])
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        out.add(token[:-1])
    return out


def content_terms(text: str) -> list[str]:
    """Query words that could name a dish or an ingredient."""
    seen, terms = set(), []
    for tok in tokenize(text):
        if tok in STOPWORDS or forms(tok) & GENERIC or tok in seen:
            continue
        seen.add(tok)
        terms.append(tok)
    return terms


class Catalog:
    def __init__(self, path=None):
        path = path or settings.catalog_path
        self.recipes: list[dict] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        self.by_id = {r["recipe_id"]: r for r in self.recipes}
        self._forms: dict[str, set[str]] = {}
        self._ingredient_forms: dict[str, set[str]] = {}
        self._name_terms: dict[str, list[str]] = {}
        for r in self.recipes:
            expanded: set[str] = set()
            for tok in r["tokens"]:
                expanded |= forms(tok)
            self._forms[r["recipe_id"]] = expanded
            self._ingredient_forms[r["recipe_id"]] = {
                f for tok in r.get("ingredient_tokens", []) for f in forms(tok)}
            self._name_terms[r["recipe_id"]] = [
                t for t in tokenize(r["recipe_name"]) if t not in STOPWORDS]
        self.vocabulary: set[str] = set().union(*self._forms.values()) if self._forms else set()
        self.name_vocabulary: set[str] = {
            f for terms in self._name_terms.values() for t in terms for f in forms(t)}

    def __len__(self) -> int:
        return len(self.recipes)

    def has_term(self, recipe_id: str, term: str) -> bool:
        return bool(forms(term) & self._forms.get(recipe_id, set()))

    def with_ingredient(self, term: str) -> list[dict]:
        """Recipes whose ingredient list mentions the term."""
        wanted = forms(term)
        return [r for r in self.recipes if wanted & self._ingredient_forms[r["recipe_id"]]]

    def known(self, term: str) -> bool:
        return bool(forms(term) & self.vocabulary)

    def name_match(self, recipe_id: str, query_terms: list[str]) -> float:
        """Share of the recipe-name words that occur in the query (0..1)."""
        name = self._name_terms.get(recipe_id, [])
        if not name:
            return 0.0
        query = set()
        for t in query_terms:
            query |= forms(t)
        return sum(1 for t in name if forms(t) & query) / len(name)


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    return Catalog()
