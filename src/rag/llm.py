"""Thin wrapper around an OpenAI-compatible chat endpoint."""
from __future__ import annotations

from functools import lru_cache

from src.config import settings


class LLMClient:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None):
        self.model = model or settings.llm_model
        self.api_key = settings.llm_api_key if api_key is None else api_key
        self.base_url = settings.llm_base_url if base_url is None else base_url
        self._client = None

    @property
    def available(self) -> bool:
        # local OpenAI-compatible servers usually need no key
        return bool(self.api_key or self.base_url)

    def complete(self, system: str, user: str, max_tokens: int = 1200,
                 temperature: float = 0.0) -> str:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key or "not-needed",
                                  base_url=self.base_url or None, timeout=60)
        response = self._client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return (response.choices[0].message.content or "").strip()


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()
