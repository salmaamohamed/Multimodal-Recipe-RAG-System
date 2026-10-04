"""Hybrid retrieval over text chunks and image-caption records, ranked per recipe."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.config import settings
from src.retrieval.catalog import Catalog, content_terms

MAX_TEXT_PER_RECIPE = 3
MAX_IMAGES_PER_RECIPE = 2


@dataclass
class RecipeResult:
    recipe_id: str
    recipe_name: str
    source_pdf: str
    score: float = 0.0
    text_score: float = 0.0
    image_score: float = 0.0
    name_match: float = 0.0
    category: str = ""
    text_results: list[dict] = field(default_factory=list)
    image_results: list[dict] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def modality_weights(has_question: bool, has_image: bool) -> tuple[float, float]:
    """(text weight, image weight) for the kind of query at hand."""
    if has_question and has_image:
        return settings.text_weight, settings.image_weight
    if has_image:
        return settings.image_only_text_weight, settings.image_only_image_weight
    return settings.text_only_text_weight, settings.text_only_image_weight


class HybridRetriever:
    def __init__(self, store, embedder, catalog: Catalog):
        self.store = store
        self.embedder = embedder
        self.catalog = catalog

    def retrieve(self, question: str | None, image_caption: str | None) -> list[RecipeResult]:
        queries = [q for q in (question, image_caption) if q]
        if not queries:
            return []

        # 1) search both modalities with every query, keep the best hit per record
        hits: dict[str, dict] = {}
        for query in queries:
            vector = self.embedder.embed_query(query)
            for modality, k in (("text", settings.top_k_text), ("image", settings.top_k_image)):
                for hit in self.store.search(vector, k, modality):
                    if hit["id"] not in hits or hit["score"] > hits[hit["id"]]["score"]:
                        hits[hit["id"]] = hit

        # 2) group by recipe
        grouped: dict[str, RecipeResult] = {}
        for hit in sorted(hits.values(), key=lambda h: -h["score"]):
            meta = hit["metadata"]
            result = grouped.setdefault(meta["recipe_id"], RecipeResult(
                meta["recipe_id"], meta["recipe_name"], meta["source_pdf"],
                category=meta.get("category", "")))
            bucket = result.image_results if meta["modality"] == "image" else result.text_results
            limit = MAX_IMAGES_PER_RECIPE if meta["modality"] == "image" else MAX_TEXT_PER_RECIPE
            if len(bucket) < limit:  # strongest chunks only, no flooding
                bucket.append(hit)

        # 3) recipe-level score
        w_text, w_image = modality_weights(bool(question), bool(image_caption))
        terms = content_terms(" ".join(queries))
        for result in grouped.values():
            result.text_score = result.text_results[0]["score"] if result.text_results else 0.0
            result.image_score = result.image_results[0]["score"] if result.image_results else 0.0
            if result.text_results and result.image_results:
                score = (w_text * result.text_score + w_image * result.image_score) / (w_text + w_image)
            else:  # a modality without hits must not drag the recipe down
                score = result.text_score or result.image_score
            result.name_match = self.catalog.name_match(result.recipe_id, terms)
            if result.name_match >= 0.5:
                score += settings.name_match_bonus * result.name_match
            result.score = round(min(score, 1.0), 4)

            entry = self.catalog.by_id.get(result.recipe_id, {})
            pages = {int(p) for h in result.text_results + result.image_results
                     for p in str(h["metadata"].get("pages", "")).split(",") if p}
            result.pages = sorted(pages) or entry.get("pages", [])
            matched = [h["metadata"]["image_path"] for h in result.image_results]
            result.images = matched + [p for p in entry.get("images", []) if p not in matched]

        return sorted(grouped.values(), key=lambda r: -r.score)
