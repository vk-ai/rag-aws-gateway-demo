"""Offline bag-of-words hashing embeddings (no API keys, no heavy deps)."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _hash_token(token: str, dim: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dim


class HashingEmbedder:
    """Signed hashing trick → dense float vectors for cosine search."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            idx = _hash_token(token, self.dim)
            sign = 1.0 if (hashlib.md5(token.encode()).digest()[0] % 2 == 0) else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_many(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts])
