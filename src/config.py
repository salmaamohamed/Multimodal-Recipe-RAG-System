"""Central configuration. Every value can be overridden through environment
variables (or a .env file in the project root)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
if load_dotenv:
    load_dotenv(ROOT / ".env")


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else ROOT / p


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # --- data locations ---
    pdf_dir: Path = _path("PDF_DIR", "data/pdfs")
    image_dir: Path = _path("IMAGE_DIR", "data/images")
    vectorstore_dir: Path = _path("VECTORSTORE_DIR", "data/vectorstore")
    collection_name: str = os.getenv("COLLECTION_NAME", "recipes")

    # --- models ---
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    # bge models expect this instruction on the query side only
    embedding_query_prefix: str = os.getenv(
        "EMBEDDING_QUERY_PREFIX",
        "Represent this sentence for searching relevant passages: ",
    )
    caption_model: str = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
    device: str = os.getenv("DEVICE", "auto")  # auto | cpu | cuda

    # --- LLM (any OpenAI-compatible endpoint) ---
    llm_model: str = os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")
    llm_api_key: str = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    # auto = LLM when a key is configured, otherwise keyword heuristic
    validator_mode: str = os.getenv("VALIDATOR_MODE", "auto")  # auto | llm | heuristic

    # --- ingestion ---
    max_chunk_chars: int = _int("MAX_CHUNK_CHARS", 1400)
    min_image_side: int = _int("MIN_IMAGE_SIDE", 120)
    max_image_aspect: float = _float("MAX_IMAGE_ASPECT", 3.0)
    ocr_min_chars: int = _int("OCR_MIN_CHARS", 30)

    # --- retrieval ---
    top_k_text: int = _int("TOP_K_TEXT", 20)
    top_k_image: int = _int("TOP_K_IMAGE", 8)
    # weights used when both a question and an image are supplied
    text_weight: float = _float("TEXT_WEIGHT", 0.6)
    image_weight: float = _float("IMAGE_WEIGHT", 0.4)
    # text-only questions lean on text records
    text_only_text_weight: float = _float("TEXT_ONLY_TEXT_WEIGHT", 0.8)
    text_only_image_weight: float = _float("TEXT_ONLY_IMAGE_WEIGHT", 0.2)
    # image-only queries lean on image-caption records
    image_only_text_weight: float = _float("IMAGE_ONLY_TEXT_WEIGHT", 0.4)
    image_only_image_weight: float = _float("IMAGE_ONLY_IMAGE_WEIGHT", 0.6)
    name_match_bonus: float = _float("NAME_MATCH_BONUS", 0.15)

    # --- context / KB detection ---
    max_context_recipes: int = _int("MAX_CONTEXT_RECIPES", 4)
    max_chars_per_recipe: int = _int("MAX_CHARS_PER_RECIPE", 4500)
    # below this top recipe score the KB is considered to have no evidence
    min_recipe_score: float = _float("MIN_RECIPE_SCORE", 0.60)
    # above this score a missing keyword match no longer forces NOT_FOUND
    strong_recipe_score: float = _float("STRONG_RECIPE_SCORE", 0.80)

    @property
    def catalog_path(self) -> Path:
        return self.vectorstore_dir / "catalog.json"


settings = Settings()

NOT_RECIPE_MESSAGE = (
    "I can only answer questions related to the recipes in the knowledge base.\n"
    "Please ask me about a recipe, ingredient, cooking method, or preparation step."
)
