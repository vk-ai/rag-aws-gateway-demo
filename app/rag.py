from __future__ import annotations

from dataclasses import dataclass

from app.citations import Citation, build_citations, grounding_score
from app.config import Settings, get_settings
from app.providers.base import GenerationProvider, get_provider
from app.vectorstore import Chunk, NumpyVectorStore, RetrievedChunk, load_corpus


@dataclass
class QueryResult:
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    provider: str
    citations: list[Citation]
    grounding_score: float


class RagService:
    def __init__(
        self,
        store: NumpyVectorStore,
        provider: GenerationProvider,
        settings: Settings,
    ) -> None:
        self.store = store
        self.provider = provider
        self.settings = settings

    def query(self, question: str) -> QueryResult:
        hits = self.store.search(question, top_k=self.settings.rag_top_k)
        context = [h.chunk.text for h in hits]
        answer = self.provider.generate(question, context)
        context_blob = "\n".join(context)
        score = grounding_score(answer, context_blob) if context else 0.0
        citations = build_citations(answer, hits, question=question)
        return QueryResult(
            question=question,
            answer=answer,
            chunks=hits,
            provider=self.settings.rag_provider,
            citations=citations,
            grounding_score=score,
        )


def build_service(settings: Settings | None = None) -> RagService:
    settings = settings or get_settings()
    store = NumpyVectorStore()
    store.add(load_corpus(settings.rag_corpus_dir))
    provider = get_provider(settings)
    return RagService(store=store, provider=provider, settings=settings)


# Re-export for convenience
__all__ = ["Chunk", "QueryResult", "RagService", "build_service"]
