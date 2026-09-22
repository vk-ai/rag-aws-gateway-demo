from __future__ import annotations

from app.providers.base import GenerationProvider


class MockProvider(GenerationProvider):
    """Deterministic offline generator — no network, no API keys.

    Emits ``[n]`` citation markers (1-based) so clients can bind claims to
    retrieved chunks. Teaching demo only — not a production LLM.
    """

    def generate(self, question: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return (
                f"[mock] No relevant context found for: {question.strip() or '(empty)'}. "
                "Try rephrasing or adding documents to the corpus."
            )
        lines: list[str] = [
            f"[mock] Answer grounded in {len(context_chunks)} retrieved chunk(s).",
            f"Question: {question.strip()}",
            "Citations:",
        ]
        for i, chunk in enumerate(context_chunks, start=1):
            preview = chunk if len(chunk) <= 180 else chunk[:180] + "…"
            lines.append(f"[{i}] {preview}")
        return "\n".join(lines)
