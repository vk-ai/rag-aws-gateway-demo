"""Retrieval eval harness: metric math, qrels sanity, ablation, and the CI floor gate."""

from __future__ import annotations

import json
import math

import pytest

from evals.retrieval_eval import (
    MODES,
    Qrels,
    QrelQuery,
    check_floors,
    evaluate,
    hit_at_k,
    load_floors,
    load_qrels,
    main,
    markdown_table,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.vectorstore import load_corpus


# --- metric math on hand-computed examples ---------------------------------

RANKED = ["d3", "d1", "d7", "d2"]
REL = {"d1": 2, "d2": 1}


def test_hit_at_k():
    assert hit_at_k(RANKED, REL, 1) == 0.0
    assert hit_at_k(RANKED, REL, 2) == 1.0


def test_recall_at_k():
    assert recall_at_k(RANKED, REL, 1) == 0.0
    assert recall_at_k(RANKED, REL, 3) == 0.5
    assert recall_at_k(RANKED, REL, 4) == 1.0
    assert recall_at_k(RANKED, {}, 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, REL) == 0.5
    assert reciprocal_rank(["x", "y"], REL) == 0.0


def test_ndcg_graded_matches_hand_calc():
    # actual: d3(0) d1(2) d7(0) d2(1) ; gain = 2^g - 1 ; discount log2(i+2)
    dcg = (3 / math.log2(3)) + (1 / math.log2(5))
    idcg = (3 / math.log2(2)) + (1 / math.log2(3))
    assert ndcg_at_k(RANKED, REL, 4) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["d1", "d2"], REL, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["d2", "d1"], REL, 2) < 1.0


# --- qrels / corpus sanity --------------------------------------------------

def test_qrels_reference_real_chunks_and_are_small_but_real():
    qrels = load_qrels()
    ids = {c.doc_id for c in load_corpus(qrels.corpus_dir)}
    assert 15 <= len(qrels.queries) <= 40
    for q in qrels.queries:
        assert set(q.relevant) <= ids, q.id
    assert any(not q.answerable for q in qrels.queries), "keep >=1 unanswerable probe"
    tags = {t for q in qrels.queries for t in q.tags}
    assert {"identifier", "acronym", "paraphrase"} <= tags


def test_unknown_chunk_id_in_qrels_raises():
    qrels = load_qrels()
    bad = Qrels(
        corpus_dir=qrels.corpus_dir,
        queries=[QrelQuery(id="bad", query="x", relevant={"nope#9": 2})],
    )
    with pytest.raises(ValueError, match="unknown chunk_ids"):
        evaluate(bad)


# --- ablation + CI floor gate ----------------------------------------------

@pytest.fixture(scope="module")
def report():
    return evaluate()


def test_ablation_covers_all_modes(report):
    assert tuple(report.modes) == MODES
    for res in report.modes.values():
        for v in res.means.values():
            assert 0.0 <= v <= 1.0
    assert report.probes[0] == "probe_unanswerable"
    assert len(report.probes) >= 10  # unanswerable probes also calibrate the abstention gate


def test_retrieval_floors_hold(report):
    """CI gate: fails if any mode drops below evals/retrieval_floors.json."""
    violations = check_floors(report)
    assert not violations, "\n" + "\n".join(violations) + "\n\n" + markdown_table(report)


def test_hybrid_not_worse_than_dense_on_recall_at_3(report):
    assert report.modes["hybrid"].means["recall@3"] >= report.modes["dense"].means["recall@3"]


def test_rewrite_can_help_and_hurt_per_query(report):
    """The lesson: a global mean hides per-query wins and losses."""
    base = report.modes["hybrid"].per_query
    rw = report.modes["hybrid+rewrite"].per_query
    assert rw["faq_refund"]["mrr"] > base["faq_refund"]["mrr"]  # FAQ -> frequently asked questions
    assert rw["bm25_rank"]["mrr"] < base["bm25_rank"]["mrr"]  # BM25 -> "best match 25" drops the token


def test_floor_gate_catches_a_broken_retriever():
    """A retriever that returns the corpus in reverse file order must fail the floors."""
    qrels = load_qrels()
    order = [c.doc_id for c in load_corpus(qrels.corpus_dir)][::-1]
    broken = {mode: (lambda q, _o=order: list(_o)) for mode in MODES}
    violations = check_floors(evaluate(qrels, broken))
    assert violations
    assert any("hybrid recall@3" in v for v in violations)


def test_relative_gate_reports_regression():
    qrels = load_qrels()
    good = evaluate(qrels)
    floors = {"min": {}, "no_worse_than": [["dense", "bm25", "mrr"]]}
    # On this toy set BM25 beats dense, so asserting dense >= bm25 must fail.
    assert check_floors(good, floors)


def test_floors_file_is_valid():
    floors = load_floors()
    assert set(floors["min"]) <= set(MODES)
    for better, worse, metric in floors["no_worse_than"]:
        assert better in MODES and worse in MODES and metric


def test_cli_json_and_check(capsys):
    assert main(["--json", "--check"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data["metrics"]) == set(MODES)
    assert data["floor_violations"] == []


def test_cli_markdown_table(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "| mode | hit@1 |" in out
    assert "hybrid+rewrite" in out
