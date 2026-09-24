"""Deterministic mock query rewriter (offline teaching stub).

Expands common abbreviations and strips filler phrases before retrieval.
Not production HyDE / live-LLM rewrite — same *stage shape* only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Lowercase abbreviation → expansion (whole-word, case-insensitive).
_ABBREVS: dict[str, str] = {
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "llms": "large language models",
    "api": "application programming interface",
    "apis": "application programming interfaces",
    "faq": "frequently asked questions",
    "sku": "stock keeping unit",
    "rrf": "reciprocal rank fusion",
    "bm25": "best match 25",
    "nlp": "natural language processing",
    "kb": "knowledge base",
}

# Filler phrases stripped from the query (order: longer first).
_FILLERS: tuple[str, ...] = (
    "can you please",
    "could you please",
    "can you",
    "could you",
    "please tell me",
    "please explain",
    "please",
    "tell me about",
    "tell me",
    "i want to know",
    "i need to know",
    "i was wondering",
    "what about",
)


@dataclass(frozen=True)
class RewriteResult:
    original: str
    rewritten: str
    ops: list[str] = field(default_factory=list)


def rewrite_query(question: str) -> RewriteResult:
    """Apply deterministic filler-strip + abbreviation expand. Idempotent-ish."""
    original = question
    text = question.strip()
    ops: list[str] = []
    if not text:
        return RewriteResult(original=original, rewritten=original, ops=ops)

    lower = text.lower()
    for filler in _FILLERS:
        if lower.startswith(filler):
            text = text[len(filler) :].lstrip(" ,:-")
            ops.append(f"strip_filler:{filler}")
            lower = text.lower()
            break
        # also strip mid-sentence "please"
        pattern = re.compile(rf"\b{re.escape(filler)}\b", re.I)
        if pattern.search(text):
            text = pattern.sub(" ", text)
            text = re.sub(r"\s{2,}", " ", text).strip(" ,")
            ops.append(f"strip_filler:{filler}")
            lower = text.lower()

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        key = token.lower()
        expansion = _ABBREVS[key]
        ops.append(f"expand:{key}->{expansion}")
        # Preserve simple capitalization of first letter if original was Title-ish
        if token[0].isupper() and token[1:].islower():
            return expansion.capitalize()
        if token.isupper() and len(token) > 1:
            return expansion.upper()
        return expansion

    # Whole-word abbrev expand (skip already-expanded long forms by matching short tokens only)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted(_ABBREVS, key=len, reverse=True)) + r")\b",
        re.I,
    )
    rewritten = pattern.sub(repl, text)
    rewritten = re.sub(r"\s{2,}", " ", rewritten).strip()
    if not rewritten:
        rewritten = original.strip()
    return RewriteResult(original=original, rewritten=rewritten, ops=ops)
