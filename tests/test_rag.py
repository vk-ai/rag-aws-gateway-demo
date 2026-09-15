from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import HashingEmbedder
from app.main import create_app_with_service
from app.providers.mock import MockProvider
from app.providers.bedrock import BedrockProvider
from app.rag import RagService, build_service
from app.vectorstore import Chunk, NumpyVectorStore, load_corpus

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"


def test_load_corpus_has_docs():
    chunks = load_corpus(CORPUS)
    assert len(chunks) >= 4
    assert all(c.text for c in chunks)


def test_retrieve_relevant_chunk():
    store = NumpyVectorStore(HashingEmbedder(dim=256))
    store.add(load_corpus(CORPUS))
    hits = store.search("What is retrieval augmented generation?", top_k=3)
    assert hits
    joined = " ".join(h.chunk.text.lower() for h in hits)
    assert "retrieval" in joined or "rag" in joined or "generate" in joined


def test_mock_generate_path():
    settings = Settings(
        rag_provider="mock",
        rag_top_k=2,
        rag_corpus_dir=str(CORPUS),
    )
    service = build_service(settings)
    result = service.query("How does vector similarity search work?")
    assert result.provider == "mock"
    assert result.chunks
    assert "[mock]" in result.answer
    assert "Question:" in result.answer


def test_query_endpoint_mock():
    settings = Settings(
        rag_provider="mock",
        rag_top_k=3,
        rag_corpus_dir=str(CORPUS),
    )
    service = build_service(settings)
    client = TestClient(create_app_with_service(service))
    resp = client.post("/query", json={"question": "Explain FastAPI for RAG services"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "mock"
    assert data["answer"]
    assert isinstance(data["retrieved"], list)
    assert len(data["retrieved"]) >= 1
    assert "score" in data["retrieved"][0]


def test_bedrock_stub_without_creds():
    settings = Settings(
        rag_provider="bedrock",
        rag_corpus_dir=str(CORPUS),
        aws_access_key_id="",
        aws_secret_access_key="",
    )
    provider = BedrockProvider(settings)
    answer = provider.generate("hi", ["ctx"])
    assert "bedrock-stub" in answer
    assert "[mock]" in answer


def test_empty_store_still_answers():
    store = NumpyVectorStore()
    service = RagService(store=store, provider=MockProvider(), settings=Settings())
    result = service.query("anything")
    assert "No relevant context" in result.answer
    assert result.chunks == []
