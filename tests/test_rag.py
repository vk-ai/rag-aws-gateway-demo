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
        rag_bedrock_live=False,
    )
    provider = BedrockProvider(settings)
    answer = provider.generate("hi", ["ctx"])
    assert "bedrock-stub" in answer
    assert "[mock]" in answer


def test_bedrock_stub_with_creds_but_live_disabled():
    """Even with fake AWS keys, default RAG_BEDROCK_LIVE=false must stay stubbed."""
    settings = Settings(
        rag_provider="bedrock",
        rag_corpus_dir=str(CORPUS),
        aws_access_key_id="AKIA_FAKE_FOR_TEST",
        aws_secret_access_key="fake_secret_for_offline_test",
        aws_region="us-west-2",
        bedrock_model_id="anthropic.claude-3-haiku-20240307-v1:0",
        rag_bedrock_live=False,
    )
    provider = BedrockProvider(settings)
    answer = provider.generate("hi", ["ctx"])
    assert "bedrock-stub" in answer
    assert "RAG_BEDROCK_LIVE=false" in answer
    assert "[mock]" in answer
    assert "bedrock-live" not in answer


def test_bedrock_live_true_without_creds_falls_back():
    settings = Settings(
        rag_provider="bedrock",
        rag_corpus_dir=str(CORPUS),
        aws_access_key_id="",
        aws_secret_access_key="",
        rag_bedrock_live=True,
    )
    provider = BedrockProvider(settings)
    answer = provider.generate("hi", ["ctx"])
    assert "bedrock-stub" in answer
    assert "credentials" in answer.lower()
    assert "[mock]" in answer
    assert "bedrock-live" not in answer


def test_bedrock_live_true_without_boto3_falls_back(monkeypatch):
    """Simulate missing boto3 even if somehow imported elsewhere."""
    import builtins
    import sys

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "boto3" or name.startswith("boto3."):
            raise ImportError("boto3 deliberately unavailable in offline test")
        return real_import(name, *args, **kwargs)

    # Ensure a real boto3 is not already cached as usable for this path
    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("boto3", None)

    settings = Settings(
        rag_provider="bedrock",
        rag_corpus_dir=str(CORPUS),
        aws_access_key_id="AKIA_FAKE_FOR_TEST",
        aws_secret_access_key="fake_secret_for_offline_test",
        rag_bedrock_live=True,
    )
    provider = BedrockProvider(settings)
    answer = provider.generate("hi", ["ctx"])
    assert "bedrock-stub" in answer
    assert "boto3" in answer.lower()
    assert "[mock]" in answer
    assert "bedrock-live" not in answer


def test_empty_store_still_answers():
    store = NumpyVectorStore()
    service = RagService(store=store, provider=MockProvider(), settings=Settings())
    result = service.query("anything")
    assert "No relevant context" in result.answer
    assert result.chunks == []


def test_rrf_fusion_prefers_shared_top_hits():
    from app.vectorstore import reciprocal_rank_fusion

    # Doc 0 ranks high in both channels → highest RRF
    scores = reciprocal_rank_fusion([[0, 1, 2], [0, 2, 1]], k=60)
    assert max(scores, key=scores.get) == 0


def test_hybrid_helps_exact_sku_id():
    """Exact inventory ID: BM25 / hybrid beat thematic dense distractor."""
    store = NumpyVectorStore(HashingEmbedder(dim=256))
    store.add(load_corpus(CORPUS))
    query = "SKU-7F3A-9910 inventory identifier"

    dense = store.search_dense(query, top_k=5)
    bm25 = store.search_bm25(query, top_k=5)
    hybrid = store.search(query, top_k=5)

    assert bm25, "BM25 should retrieve something"
    assert any("SKU-7F3A-9910" in h.chunk.text for h in bm25[:2])

    # Dense-only often elevates the thematic returns-policy distractor
    dense_top_ids = [h.chunk.doc_id for h in dense]
    hybrid_top_ids = [h.chunk.doc_id for h in hybrid]
    sku_hits = [h for h in hybrid if "SKU-7F3A-9910" in h.chunk.text]
    assert sku_hits, f"hybrid missed SKU doc; dense={dense_top_ids} hybrid={hybrid_top_ids}"
    sku = sku_hits[0]
    assert sku.bm25_score > 0
    assert sku.rrf_score == sku.score
    assert "bm25" in sku.channel_ranks

    # Hybrid must surface the SKU doc at least as high as dense-only
    dense_sku_rank = next(
        (i for i, h in enumerate(dense) if "SKU-7F3A-9910" in h.chunk.text),
        99,
    )
    hybrid_sku_rank = next(
        i for i, h in enumerate(hybrid) if "SKU-7F3A-9910" in h.chunk.text
    )
    assert hybrid_sku_rank <= dense_sku_rank


def test_query_endpoint_returns_channel_scores():
    settings = Settings(
        rag_provider="mock",
        rag_top_k=3,
        rag_corpus_dir=str(CORPUS),
    )
    service = build_service(settings)
    client = TestClient(create_app_with_service(service))
    resp = client.post("/query", json={"question": "SKU-7F3A-9910"})
    assert resp.status_code == 200
    hit = resp.json()["retrieved"][0]
    assert "dense_score" in hit and "bm25_score" in hit and "rrf_score" in hit
    assert "channel_ranks" in hit
    assert hit["score"] == hit["rrf_score"]


def test_grounding_score_lexical():
    from app.citations import grounding_score

    ctx = "Retrieval augmented generation retrieves documents then generates an answer."
    good = "RAG retrieves documents then generates an answer from context."
    bad = "The weather in Antarctica is unusually warm today."
    assert grounding_score(good, ctx) > 0.4
    assert grounding_score(bad, ctx) < 0.2
    assert grounding_score("", ctx) == 0.0


def test_citations_from_markers():
    from app.citations import build_citations
    from app.vectorstore import Chunk, RetrievedChunk

    chunks = [
        RetrievedChunk(
            chunk=Chunk(doc_id="a", text="Alpha chunk about RAG retrieval.", source="a.md"),
            score=0.9,
        ),
        RetrievedChunk(
            chunk=Chunk(doc_id="b", text="Beta chunk about BM25 keywords.", source="b.md"),
            score=0.8,
        ),
    ]
    answer = "RAG uses retrieval [1] and keywords [2]."
    cites = build_citations(answer, chunks, question="RAG retrieval")
    assert len(cites) == 2
    assert cites[0].chunk_id == "a" and cites[0].marker == 1
    assert cites[1].chunk_id == "b" and cites[1].marker == 2
    assert cites[0].quote


def test_query_endpoint_citations_and_grounding():
    settings = Settings(
        rag_provider="mock",
        rag_top_k=3,
        rag_corpus_dir=str(CORPUS),
    )
    service = build_service(settings)
    client = TestClient(create_app_with_service(service))
    resp = client.post("/query", json={"question": "What is retrieval augmented generation?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "citations" in data and isinstance(data["citations"], list)
    assert len(data["citations"]) >= 1
    cite = data["citations"][0]
    assert "chunk_id" in cite and "quote" in cite and "marker" in cite
    assert cite["marker"] >= 1
    assert "grounding_score" in data
    assert 0.0 <= data["grounding_score"] <= 1.0
    # Mock provider emits [n] markers
    assert "[1]" in data["answer"]
    # Lexical overlap should be non-trivial for mock answers that echo context
    assert data["grounding_score"] > 0.1
