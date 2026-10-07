from __future__ import annotations

from dataclasses import dataclass, field

from app.abstain import (
    ABSTAIN_MESSAGE,
    STATUS_ANSWERED,
    STATUS_INSUFFICIENT,
    evidence_score,
)
from app.citations import Citation, build_citations, grounding_score
from app.config import Settings, get_settings
from app.providers.base import GenerationProvider, get_provider
from app.rewrite import RewriteResult, rewrite_query
from app.vectorstore import Chunk, NumpyVectorStore, RetrievedChunk, load_corpus


@dataclass
class QueryResult:
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    provider: str
    citations: list[Citation]
    grounding_score: float
    rewritten_query: str = ""
    rewrite_ops: list[str] = field(default_factory=list)
    status: str = STATUS_ANSWERED
    abstained: bool = False
    evidence_score: float = 0.0
    evidence_threshold: float = 0.0


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

    def query(
        self,
        question: str,
        *,
        rewrite: bool | None = None,
        abstain: bool | None = None,
    ) -> QueryResult:
        do_rewrite = self.settings.rag_rewrite if rewrite is None else rewrite
        if do_rewrite:
            rw: RewriteResult = rewrite_query(question)
            retrieve_q = rw.rewritten
            rewritten = rw.rewritten
            ops = list(rw.ops)
        else:
            retrieve_q = question
            rewritten = question
            ops = []

        hits = self.store.search(retrieve_q, top_k=self.settings.rag_top_k)
        evidence = evidence_score(
            self.store._bm25, retrieve_q, (h.bm25_score for h in hits)
        )
        threshold = float(self.settings.rag_min_evidence)
        do_abstain = self.settings.rag_abstain if abstain is None else abstain
        if do_abstain and evidence < threshold:
            # Abstention gate: do not call the provider on weak evidence.
            return QueryResult(
                question=question,
                answer=ABSTAIN_MESSAGE,
                chunks=hits,
                provider=self.settings.rag_provider,
                citations=[],
                grounding_score=0.0,
                rewritten_query=rewritten,
                rewrite_ops=ops,
                status=STATUS_INSUFFICIENT,
                abstained=True,
                evidence_score=round(evidence, 4),
                evidence_threshold=threshold,
            )
        context = [h.chunk.text for h in hits]
        # Generate still sees the user's original question for answer framing
        answer = self.provider.generate(question, context)
        context_blob = "\n".join(context)
        score = grounding_score(answer, context_blob) if context else 0.0
        citations = build_citations(answer, hits, question=retrieve_q)
        return QueryResult(
            question=question,
            answer=answer,
            chunks=hits,
            provider=self.settings.rag_provider,
            citations=citations,
            grounding_score=score,
            rewritten_query=rewritten,
            rewrite_ops=ops,
            evidence_score=round(evidence, 4),
            evidence_threshold=threshold,
        )


def build_service(settings: Settings | None = None) -> RagService:
    settings = settings or get_settings()
    store = NumpyVectorStore()
    store.add(load_corpus(settings.rag_corpus_dir))
    provider = get_provider(settings)
    return RagService(store=store, provider=provider, settings=settings)


# Re-export for convenience
__all__ = ["Chunk", "QueryResult", "RagService", "build_service"]
