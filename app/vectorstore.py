"""In-memory hybrid retrieval: hashing-cosine + BM25 fused with RRF."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.bm25 import BM25Index
from app.embeddings import HashingEmbedder


@dataclass
class Chunk:
    doc_id: str
    text: str
    source: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float  # fused RRF score (primary ranking key)
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    channel_ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    *,
    k: int = 60,
) -> dict[int, float]:
    """RRF: score(d) = sum_i 1 / (k + rank_i(d)), ranks are 1-based."""
    scores: dict[int, float] = {}
    for ranking in ranked_lists:
        for rank, doc_idx in enumerate(ranking, start=1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return scores


class NumpyVectorStore:
    def __init__(
        self,
        embedder: HashingEmbedder | None = None,
        *,
        rrf_k: int = 60,
        channel_pool: int = 20,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None
        self._bm25: BM25Index | None = None
        self.rrf_k = rrf_k
        self.channel_pool = channel_pool

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        vectors = self.embedder.embed_many(c.text for c in chunks)
        if self.matrix is None:
            self.matrix = vectors
        else:
            self.matrix = np.vstack([self.matrix, vectors])
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self._bm25 = BM25Index.build([c.text for c in self.chunks])

    def _dense_scores(self, query: str) -> np.ndarray:
        if not self.chunks or self.matrix is None:
            return np.array([], dtype=np.float64)
        q = self.embedder.embed(query)
        return self.matrix @ q  # cosine: both sides L2-normalized

    def search_dense(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        dense = self._dense_scores(query)
        if dense.size == 0:
            return []
        k = min(top_k, len(self.chunks))
        idxs = np.argsort(-dense)[:k]
        return [
            RetrievedChunk(
                chunk=self.chunks[i],
                score=float(dense[i]),
                dense_score=float(dense[i]),
                rrf_score=float(dense[i]),
                channel_ranks={"dense": rank},
            )
            for rank, i in enumerate(idxs, start=1)
        ]

    def search_bm25(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not self.chunks or self._bm25 is None:
            return []
        bm25 = self._bm25.scores(query)
        order = sorted(range(len(bm25)), key=lambda i: -bm25[i])[: min(top_k, len(bm25))]
        return [
            RetrievedChunk(
                chunk=self.chunks[i],
                score=float(bm25[i]),
                bm25_score=float(bm25[i]),
                rrf_score=float(bm25[i]),
                channel_ranks={"bm25": rank},
            )
            for rank, i in enumerate(order, start=1)
        ]

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Hybrid BM25 + hashing-cosine fused with Reciprocal Rank Fusion."""
        if not self.chunks or self.matrix is None or self._bm25 is None:
            return []

        dense = self._dense_scores(query)
        bm25 = self._bm25.scores(query)
        n = len(self.chunks)
        pool = min(self.channel_pool, n)

        dense_order = list(np.argsort(-dense)[:pool])
        bm25_order = sorted(range(n), key=lambda i: -bm25[i])[:pool]

        dense_rank = {idx: r for r, idx in enumerate(dense_order, start=1)}
        bm25_rank = {idx: r for r, idx in enumerate(bm25_order, start=1)}

        rrf = reciprocal_rank_fusion([dense_order, bm25_order], k=self.rrf_k)
        fused = sorted(rrf.items(), key=lambda kv: -kv[1])
        k = min(top_k, len(fused))

        results: list[RetrievedChunk] = []
        for idx, rrf_score in fused[:k]:
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(rrf_score),
                    dense_score=float(dense[idx]),
                    bm25_score=float(bm25[idx]),
                    rrf_score=float(rrf_score),
                    channel_ranks={
                        **({"dense": dense_rank[idx]} if idx in dense_rank else {}),
                        **({"bm25": bm25_rank[idx]} if idx in bm25_rank else {}),
                    },
                )
            )
        return results


def load_corpus(corpus_dir: str | Path) -> list[Chunk]:
    root = Path(corpus_dir)
    chunks: list[Chunk] = []
    if not root.exists():
        return chunks
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        # Simple paragraph chunking
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not parts:
            parts = [text]
        for i, part in enumerate(parts):
            chunks.append(
                Chunk(
                    doc_id=f"{path.stem}#{i}",
                    text=part,
                    source=str(path.as_posix()),
                )
            )
    return chunks
