"""'Insufficient evidence' abstention gate (say "I don't know" before generating).

If the best retrieved chunk does not lexically support the question, ``/query``
returns ``status="insufficient_evidence"`` and skips generation instead of
letting the provider improvise from weak context.

The signal is a **normalized BM25 evidence score** in [0, 1]::

    evidence = max_i bm25(query, hit_i) / max_possible_bm25(query)

where ``max_possible_bm25 = sum_t idf(t) * (k1 + 1) * qf(t)`` is the BM25
upper bound for that query on this corpus (every term saturated). It reads as
"how much of the query's IDF-weighted vocabulary the best hit actually covers".
It is bounded, so a threshold has a chance to transfer across corpora, unlike raw
BM25 scores or RRF scores (RRF is rank-based and always looks confident).

The toy hashing-cosine channel is deliberately *not* used: with 256 buckets,
hash collisions give off-topic questions cosine scores as high as on-topic
ones (see the README table).

The threshold is calibrated offline on answerable vs unanswerable queries in
``evals/qrels.json`` (``python -m evals.retrieval_eval --calibrate``) and
shipped as :data:`DEFAULT_MIN_EVIDENCE`. A lexical gate cannot catch near-miss
questions whose words *are* in the corpus but whose answer is not; the eval
reports those as false answers rather than hiding them.

OSS/learning only — not an NLI sufficiency judge, not RAGAS.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from app.bm25 import BM25Index
from app.embeddings import tokenize

# Calibrated by `python -m evals.retrieval_eval --calibrate` on evals/qrels.json
# (hybrid+rewrite, top_k=3, max_false_refusal=0.05). A test pins this value to
# the calibration output so re-labelling the eval set forces a conscious update.
DEFAULT_MIN_EVIDENCE = 0.087

ABSTAIN_MESSAGE = (
    "I don't have enough evidence in the indexed documents to answer that. "
    "Try rephrasing, or add a document that covers it."
)

STATUS_ANSWERED = "answered"
STATUS_INSUFFICIENT = "insufficient_evidence"


def max_possible_bm25(index: BM25Index, query: str) -> float:
    """BM25 upper bound for ``query``: each term at saturation (tf -> inf)."""
    q_tf = Counter(tokenize(query))
    return sum(index._idf(t) * (index.k1 + 1.0) * qf for t, qf in q_tf.items())


def evidence_score(index: BM25Index | None, query: str, bm25_scores: Iterable[float]) -> float:
    """Normalized best-hit BM25 in [0, 1] (0.0 when there is nothing to score)."""
    if index is None:
        return 0.0
    upper = max_possible_bm25(index, query)
    best = max(bm25_scores, default=0.0)
    if upper <= 0.0 or best <= 0.0:
        return 0.0
    return min(best / upper, 1.0)


@dataclass(frozen=True)
class Calibration:
    threshold: float
    false_answer_rate: float  # unanswerable queries that passed the gate
    false_refusal_rate: float  # answerable queries that were refused
    n_answerable: int
    n_unanswerable: int
    max_false_refusal: float

    def to_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 4),
            "false_answer_rate": round(self.false_answer_rate, 4),
            "false_refusal_rate": round(self.false_refusal_rate, 4),
            "n_answerable": self.n_answerable,
            "n_unanswerable": self.n_unanswerable,
            "max_false_refusal": self.max_false_refusal,
        }


def gate_rates(
    answerable: Sequence[float], unanswerable: Sequence[float], threshold: float
) -> tuple[float, float]:
    """(false_answer_rate, false_refusal_rate) for ``abstain if score < threshold``."""
    fa = sum(1 for s in unanswerable if s >= threshold) / max(len(unanswerable), 1)
    fr = sum(1 for s in answerable if s < threshold) / max(len(answerable), 1)
    return fa, fr


def calibrate_threshold(
    answerable: Sequence[float],
    unanswerable: Sequence[float],
    *,
    max_false_refusal: float = 0.05,
) -> Calibration:
    """Pick the threshold with the fewest false answers within a refusal budget.

    Candidates are midpoints between adjacent distinct scores (plus 0.0).
    Among thresholds whose false-refusal rate is <= ``max_false_refusal``,
    choose the lowest false-answer rate; ties go to the *lowest* threshold
    (refuse as little as possible). Deterministic, stdlib only.
    """
    if not answerable or not unanswerable:
        raise ValueError("need at least one answerable and one unanswerable score")
    pts = sorted(set(answerable) | set(unanswerable))
    candidates = [0.0] + [round((a + b) / 2, 4) for a, b in zip(pts, pts[1:])]
    best: tuple[float, float, float] | None = None  # (fa, threshold, fr)
    for t in candidates:
        fa, fr = gate_rates(answerable, unanswerable, t)
        if fr > max_false_refusal + 1e-12:
            continue
        if best is None or fa < best[0] - 1e-12:
            best = (fa, t, fr)
    assert best is not None  # threshold 0.0 always has fr == 0
    fa, t, fr = best
    return Calibration(
        threshold=t,
        false_answer_rate=fa,
        false_refusal_rate=fr,
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        max_false_refusal=max_false_refusal,
    )
