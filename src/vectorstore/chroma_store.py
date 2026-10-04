"""Persistent ChromaDB collection holding text chunks and image captions."""
from __future__ import annotations

import chromadb

from src.config import settings


class RecipeStore:
    def __init__(self, path=None, collection: str | None = None):
        self.path = path or settings.vectorstore_dir
        self.name = collection or settings.collection_name
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                self.name, metadata={"hnsw:space": "cosine"})
        return self._collection

    def reset(self, embedding_model: str) -> None:
        """Drop and recreate the collection (full rebuild)."""
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        self._collection = self.client.create_collection(
            self.name, metadata={"hnsw:space": "cosine", "embedding_model": embedding_model})

    @property
    def embedding_model(self) -> str:
        return (self.collection.metadata or {}).get("embedding_model", "")

    def count(self) -> int:
        return self.collection.count()

    def upsert(self, records: list[dict], embeddings: list[list[float]], batch: int = 256) -> None:
        for i in range(0, len(records), batch):
            chunk = records[i:i + batch]
            self.collection.upsert(
                ids=[r["id"] for r in chunk],
                documents=[r["content"] for r in chunk],
                metadatas=[r["metadata"] for r in chunk],
                embeddings=embeddings[i:i + batch],
            )

    def search(self, embedding: list[float], k: int, modality: str) -> list[dict]:
        """Nearest records of one modality, as {id, content, metadata, score}."""
        res = self.collection.query(
            query_embeddings=[embedding], n_results=k, where={"modality": modality},
            include=["documents", "metadatas", "distances"])
        return [
            {"id": i, "content": d, "metadata": m, "score": 1.0 - dist}
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                     res["metadatas"][0], res["distances"][0])
        ]

    def recipe_records(self, recipe_id: str) -> list[dict]:
        """Every stored record of one recipe."""
        res = self.collection.get(where={"recipe_id": recipe_id},
                                  include=["documents", "metadatas"])
        return [{"id": i, "content": d, "metadata": m}
                for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
