from __future__ import annotations

import json
import urllib.error
import urllib.request

from app.config import Settings
from app.providers.base import GenerationProvider
from app.providers.mock import MockProvider


class OpenAICompatProvider(GenerationProvider):
    """Optional OpenAI-compatible chat completions call; falls back to mock without a key."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._fallback = MockProvider()

    def generate(self, question: str, context_chunks: list[str]) -> str:
        if not self.settings.openai_api_key:
            return (
                "[openai-stub] OPENAI_API_KEY not set; using mock fallback.\n"
                + self._fallback.generate(question, context_chunks)
            )

        context = "\n\n".join(context_chunks) if context_chunks else "(no context)"
        payload = {
            "model": self.settings.openai_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful RAG assistant. Answer using only the "
                        "provided context. If context is insufficient, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            "temperature": 0.2,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.openai_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            return (
                f"[openai-stub] request failed ({exc}); using mock fallback.\n"
                + self._fallback.generate(question, context_chunks)
            )
