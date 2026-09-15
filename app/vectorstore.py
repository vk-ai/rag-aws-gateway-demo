"""In-memory numpy cosine vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.embeddings import HashingEmbedder


@dataclass
class Chunk:
    doc_id: str
    text: str
    source: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class NumpyVectorStore:
    def __init__(self, embedder: HashingEmbedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        vectors = self.embedder.embed_many(c.text for c in chunks)
        if self.matrix is None:
            self.matrix = vectors
        else:
            self.matrix = np.vstack([self.matrix, vectors])

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not self.chunks or self.matrix is None:
            return []
        q = self.embedder.embed(query)
        scores = self.matrix @ q  # cosine: both sides L2-normalized
        k = min(top_k, len(self.chunks))
        idxs = np.argsort(-scores)[:k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]))
            for i in idxs
        ]


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
