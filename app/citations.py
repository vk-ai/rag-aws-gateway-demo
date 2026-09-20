"""Citations + lexical grounding_score (OSS/learning — not RAGAS / Check Grounding)."""

from __future__ import annotations

import re
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
# Citation markers like [1], [2] in generated answers (1-based index into retrieved)
_MARKER_RE = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Citation:
    """Claim→chunk binding: which retrieved chunk supports which quoted span."""

    chunk_id: str
    source: str
    quote: str
    marker: int  # 1-based [n] index in the answer
    start: int  # char offset of quote in chunk text
    end: int


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def grounding_score(answer: str, context: str) -> float:
    """Lexical token-overlap: fraction of answer tokens found in context (0–1).

    Teaching metric only — not NLI entailment, not Google Check Grounding,
    not RAGAS faithfulness. Same *shape* as those APIs for interview literacy.
    """
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0
    context_tokens = set(tokenize(context))
    hit = sum(1 for t in answer_tokens if t in context_tokens)
    return round(hit / len(answer_tokens), 4)


def _best_quote_span(chunk_text: str, question: str, *, max_len: int = 120) -> tuple[str, int, int]:
    """Pick a short quote span: prefer a sentence overlapping query tokens, else prefix."""
    text = chunk_text.strip()
    if not text:
        return "", 0, 0
    q_tokens = set(tokenize(question))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    best: tuple[int, str, int, int] | None = None
    cursor = 0
    for sent in sentences:
        start = text.find(sent, cursor)
        if start < 0:
            start = cursor
        end = start + len(sent)
        cursor = end
        overlap = len(q_tokens & set(tokenize(sent)))
        if best is None or overlap > best[0]:
            best = (overlap, sent.strip(), start, end)
    if best and best[0] > 0:
        quote = best[1]
        if len(quote) > max_len:
            quote = quote[: max_len - 1] + "…"
            return quote, best[2], best[2] + len(quote)
        return quote, best[2], best[3]
    quote = text if len(text) <= max_len else text[: max_len - 1] + "…"
    return quote, 0, len(quote)


def build_citations(
    answer: str,
    chunks: list,  # list[RetrievedChunk] — duck-typed to avoid circular import
    question: str = "",
) -> list[Citation]:
    """Bind [n] markers in the answer to retrieved chunks; else cite all top hits."""
    citations: list[Citation] = []
    if not chunks:
        return citations

    markers = sorted({int(m) for m in _MARKER_RE.findall(answer)})
    indices: list[int]
    if markers:
        indices = [m - 1 for m in markers if 1 <= m <= len(chunks)]
    else:
        # No markers — still expose chunk bindings so clients can render sources
        indices = list(range(len(chunks)))

    seen: set[int] = set()
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        hit = chunks[idx]
        chunk = hit.chunk
        quote, start, end = _best_quote_span(chunk.text, question)
        citations.append(
            Citation(
                chunk_id=chunk.doc_id,
                source=chunk.source,
                quote=quote,
                marker=idx + 1,
                start=start,
                end=end,
            )
        )
    return citations
