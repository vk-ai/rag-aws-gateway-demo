from __future__ import annotations

from app.providers.base import GenerationProvider


class MockProvider(GenerationProvider):
    """Deterministic offline generator — no network, no API keys."""

    def generate(self, question: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return (
                f"[mock] No relevant context found for: {question.strip() or '(empty)'}. "
                "Try rephrasing or adding documents to the corpus."
            )
        joined = "\n---\n".join(context_chunks)
        preview = joined if len(joined) <= 600 else joined[:600] + "…"
        return (
            f"[mock] Answer grounded in {len(context_chunks)} retrieved chunk(s).\n"
            f"Question: {question.strip()}\n"
            f"Context summary:\n{preview}"
        )
