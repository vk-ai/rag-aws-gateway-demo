"""Tiny Okapi BM25 over in-memory chunks (offline, no deps beyond numpy)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from app.embeddings import tokenize


@dataclass
class BM25Index:
    """Okapi BM25 over a fixed corpus of tokenized documents."""

    docs: list[list[str]]
    doc_lens: list[int]
    avgdl: float
    df: dict[str, int]
    n_docs: int
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, texts: list[str], *, k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        docs = [tokenize(t) for t in texts]
        doc_lens = [len(d) for d in docs]
        n_docs = len(docs)
        avgdl = (sum(doc_lens) / n_docs) if n_docs else 0.0
        df: dict[str, int] = {}
        for tokens in docs:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        return cls(
            docs=docs,
            doc_lens=doc_lens,
            avgdl=avgdl,
            df=df,
            n_docs=n_docs,
            k1=k1,
            b=b,
        )

    def _idf(self, term: str) -> float:
        # Robertson–Sparck Jones IDF with +0.5 smoothing
        n_q = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - n_q + 0.5) / (n_q + 0.5))

    def scores(self, query: str) -> list[float]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.docs:
            return [0.0] * self.n_docs
        q_tf = Counter(q_tokens)
        out = [0.0] * self.n_docs
        for i, tokens in enumerate(self.docs):
            if not tokens:
                continue
            tf = Counter(tokens)
            dl = self.doc_lens[i]
            score = 0.0
            for term, qf in q_tf.items():
                if term not in tf:
                    continue
                idf = self._idf(term)
                freq = tf[term]
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score += idf * (freq * (self.k1 + 1.0) / denom) * qf
            out[i] = score
        return out
