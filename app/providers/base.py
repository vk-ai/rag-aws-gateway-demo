from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import Settings


class GenerationProvider(ABC):
    @abstractmethod
    def generate(self, question: str, context_chunks: list[str]) -> str:
        raise NotImplementedError


def get_provider(settings: Settings) -> GenerationProvider:
    name = (settings.rag_provider or "mock").strip().lower()
    if name == "mock":
        from app.providers.mock import MockProvider

        return MockProvider()
    if name == "openai":
        from app.providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(settings)
    if name == "bedrock":
        from app.providers.bedrock import BedrockProvider

        return BedrockProvider(settings)
    raise ValueError(f"Unknown RAG_PROVIDER={settings.rag_provider!r}; use mock|openai|bedrock")
