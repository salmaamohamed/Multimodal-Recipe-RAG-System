"""Text embeddings. Recipe text and BLIP image captions share this one space."""
from __future__ import annotations

from functools import lru_cache

from src.config import settings


class Embedder:
    def __init__(self, model_name: str | None = None, query_prefix: str | None = None,
                 device: str | None = None):
        from sentence_transformers import SentenceTransformer
        from src.ingestion.captioner import resolve_device

        self.model_name = model_name or settings.embedding_model
        self.query_prefix = settings.embedding_query_prefix if query_prefix is None else query_prefix
        self.model = SentenceTransformer(self.model_name, device=device or resolve_device())

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                    show_progress_bar=len(texts) > 200)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([self.query_prefix + text], normalize_embeddings=True)
        return vector[0].tolist()


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
