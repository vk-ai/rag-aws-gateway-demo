from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.rag import RagService, build_service

_service: RagService | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _service
    _service = build_service(get_settings())
    yield
    _service = None


app = FastAPI(
    title="RAG AWS Gateway Demo",
    description=(
        "OSS/learning retrieve-then-generate RAG service. "
        "Not production software and not affiliated with any employer deployment."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is RAG?"])


class ChunkOut(BaseModel):
    doc_id: str
    source: str
    text: str
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    channel_ranks: dict[str, int] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    question: str
    answer: str
    provider: str
    retrieved: list[ChunkOut]


def _chunk_out(h) -> ChunkOut:
    return ChunkOut(
        doc_id=h.chunk.doc_id,
        source=h.chunk.source,
        text=h.chunk.text,
        score=h.score,
        dense_score=h.dense_score,
        bm25_score=h.bm25_score,
        rrf_score=h.rrf_score,
        channel_ranks=h.channel_ranks,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest) -> QueryResponse:
    if _service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    result = _service.query(body.question)
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        provider=result.provider,
        retrieved=[_chunk_out(h) for h in result.chunks],
    )


def create_app_with_service(service: RagService) -> FastAPI:
    """Test helper: bind a prebuilt service without lifespan race."""
    test_app = FastAPI(title="RAG AWS Gateway Demo (test)")

    @test_app.post("/query", response_model=QueryResponse)
    def _query(body: QueryRequest) -> Any:
        result = service.query(body.question)
        return QueryResponse(
            question=result.question,
            answer=result.answer,
            provider=result.provider,
            retrieved=[_chunk_out(h) for h in result.chunks],
        )

    return test_app
