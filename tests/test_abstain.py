"""Insufficient-evidence abstention gate: signal, calibration, service + API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.abstain import (
    ABSTAIN_MESSAGE,
    DEFAULT_MIN_EVIDENCE,
    calibrate_threshold,
    evidence_score,
    gate_rates,
    max_possible_bm25,
)
from app.bm25 import BM25Index
from app.config import Settings
from app.main import create_app_with_service
from app.providers.base import GenerationProvider
from app.rag import RagService, build_service
from evals.retrieval_eval import abstention_eval, check_abstention, load_qrels, main

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"


# --- evidence signal ---------------------------------------------------------


def test_evidence_score_bounded_and_monotone():
    idx = BM25Index.build(["alpha beta gamma", "delta epsilon", "zeta eta theta iota"])
    full = evidence_score(idx, "alpha beta", idx.scores("alpha beta"))
    half = evidence_score(idx, "alpha omega", idx.scores("alpha omega"))
    none = evidence_score(idx, "omega", idx.scores("omega"))
    assert 0.0 < half < full <= 1.0
    assert none == 0.0
    assert max_possible_bm25(idx, "alpha alpha") == pytest.approx(
        2 * max_possible_bm25(idx, "alpha")
    )


def test_evidence_score_handles_empty_inputs():
    idx = BM25Index.build(["alpha"])
    assert evidence_score(None, "alpha", [1.0]) == 0.0
    assert evidence_score(idx, "", []) == 0.0
    assert evidence_score(idx, "alpha", []) == 0.0


# --- calibration ---------------------------------------------------------------


def test_gate_rates():
    fa, fr = gate_rates([0.5, 0.2, 0.05], [0.01, 0.3], threshold=0.1)
    assert fa == pytest.approx(0.5)  # 0.3 gets through
    assert fr == pytest.approx(1 / 3)  # 0.05 refused


def test_calibrate_separable_picks_gap_midpoint():
    cal = calibrate_threshold([0.4, 0.5, 0.6], [0.1, 0.2], max_false_refusal=0.0)
    assert cal.threshold == pytest.approx(0.3)
    assert cal.false_answer_rate == 0.0
    assert cal.false_refusal_rate == 0.0


def test_calibrate_respects_refusal_budget():
    pos = [0.05, 0.4, 0.5, 0.6]
    neg = [0.1, 0.2]
    strict = calibrate_threshold(pos, neg, max_false_refusal=0.0)
    assert strict.false_refusal_rate == 0.0
    assert strict.threshold <= 0.05  # cannot refuse the 0.05 answerable query
    assert strict.false_answer_rate == 1.0
    loose = calibrate_threshold(pos, neg, max_false_refusal=0.25)
    assert loose.false_refusal_rate == pytest.approx(0.25)
    assert loose.false_answer_rate == 0.0
    assert loose.threshold == pytest.approx(0.3)


def test_calibrate_requires_both_classes():
    with pytest.raises(ValueError):
        calibrate_threshold([0.5], [])


# --- calibration on the round-4 retrieval eval set -----------------------------


@pytest.fixture(scope="module")
def abst():
    return abstention_eval()


def test_eval_set_has_answerable_and_unanswerable(abst):
    qrels = load_qrels()
    assert sum(q.answerable for q in qrels.queries) >= 20
    assert sum(not q.answerable for q in qrels.queries) >= 10


def test_shipped_threshold_matches_calibration(abst):
    """Re-labelling the eval set must come with a conscious threshold update."""
    assert DEFAULT_MIN_EVIDENCE == pytest.approx(abst.calibration.threshold, abs=1e-3)
    assert Settings().rag_min_evidence == DEFAULT_MIN_EVIDENCE


def test_abstention_floors_hold(abst):
    assert check_abstention(abst) == []
    assert abst.shipped_false_refusal_rate <= 0.05


def test_off_topic_probes_abstain_near_miss_documented(abst):
    qrels = {q.id: q for q in load_qrels().queries}
    off_topic = [i for i, q in qrels.items() if "off-topic" in q.tags]
    assert off_topic
    for i in off_topic:
        assert abst.scores[i] < abst.shipped_threshold, i
    # A lexical gate cannot catch every near-miss; the eval must surface them.
    assert set(abst.false_answers) <= {i for i, q in qrels.items() if "near-miss" in q.tags}


def test_cli_calibrate(capsys):
    assert main(["--calibrate", "--check"]) == 0
    out = capsys.readouterr().out
    assert "insufficient_evidence" in out
    assert "false-answer rate" in out


# --- service + API -----------------------------------------------------------------


class CountingProvider(GenerationProvider):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question: str, context_chunks: list[str]) -> str:
        self.calls += 1
        return "[counting] " + (context_chunks[0] if context_chunks else "")


def _service(**kw) -> tuple[RagService, CountingProvider]:
    base = build_service(Settings(rag_corpus_dir=str(CORPUS), **kw))
    provider = CountingProvider()
    return RagService(store=base.store, provider=provider, settings=base.settings), provider


def test_off_topic_question_abstains_without_calling_provider():
    service, provider = _service()
    result = service.query("How do I bake sourdough bread?")
    assert result.status == "insufficient_evidence"
    assert result.abstained is True
    assert result.answer == ABSTAIN_MESSAGE
    assert result.citations == []
    assert result.grounding_score == 0.0
    assert result.evidence_score < result.evidence_threshold
    assert provider.calls == 0
    assert result.chunks  # retrieved hits are still returned for debugging


def test_on_topic_question_answers_and_reports_evidence():
    service, provider = _service()
    result = service.query("What is retrieval augmented generation?")
    assert result.status == "answered"
    assert result.abstained is False
    assert result.evidence_score >= result.evidence_threshold
    assert provider.calls == 1


def test_gate_can_be_disabled_by_setting_or_request():
    service, provider = _service(rag_abstain=False)
    assert service.query("How do I bake sourdough bread?").status == "answered"
    service, provider = _service()
    assert service.query("How do I bake sourdough bread?", abstain=False).status == "answered"
    assert provider.calls == 1


def test_query_endpoint_flags_abstained():
    service, _ = _service()
    client = TestClient(create_app_with_service(service))
    off = client.post("/query", json={"question": "How do I bake sourdough bread?"}).json()
    assert off["status"] == "insufficient_evidence"
    assert off["abstained"] is True
    assert off["citations"] == []
    assert 0.0 <= off["evidence_score"] < off["evidence_threshold"]
    on = client.post("/query", json={"question": "What is RAG?"}).json()
    assert on["status"] == "answered"
    assert on["abstained"] is False
    forced = client.post(
        "/query", json={"question": "How do I bake sourdough bread?", "abstain": False}
    ).json()
    assert forced["status"] == "answered"
