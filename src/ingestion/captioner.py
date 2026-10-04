"""BLIP image captioning. Captions are a semantic handle for retrieval, not a
reliable ingredient detector."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from PIL import Image

from src.config import settings


def resolve_device() -> str:
    if settings.device != "auto":
        return settings.device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load():
    from transformers import BlipForConditionalGeneration, BlipProcessor
    processor = BlipProcessor.from_pretrained(settings.caption_model)
    model = BlipForConditionalGeneration.from_pretrained(settings.caption_model)
    model.to(resolve_device()).eval()
    return processor, model


def caption_image(image: Image.Image | str | Path) -> str:
    """Return a BLIP caption for a PIL image or an image path."""
    import torch
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    processor, model = _load()
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # the penalties stop BLIP from looping on unfamiliar dishes ("chick chick chick")
        out = model.generate(**inputs, max_new_tokens=40, num_beams=3,
                             repetition_penalty=1.5, no_repeat_ngram_size=2)
    return processor.decode(out[0], skip_special_tokens=True).strip()


class CaptionCache:
    """Captions keyed by image content, so re-ingesting skips BLIP."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, str] = {}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    def _key(self, image_path: Path) -> str:
        digest = hashlib.sha1(image_path.read_bytes()).hexdigest()
        return f"{settings.caption_model}:v2:{digest}"

    def caption(self, image_path: Path) -> str:
        key = self._key(image_path)
        if key not in self.data:
            self.data[key] = caption_image(image_path)
        return self.data[key]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
