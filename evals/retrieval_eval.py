"""Offline retrieval eval: hand-labelled qrels -> hit@k / recall@k / MRR / nDCG@k.

Ablation over four retrieval modes on a small synthetic corpus (``evals/corpus``):

- ``dense``           hashing-cosine only
- ``bm25``            Okapi BM25 only
- ``hybrid``          BM25 + dense fused with RRF (what ``/query`` uses)
- ``hybrid+rewrite``  mock query rewrite, then hybrid

Run::

    python -m evals.retrieval_eval            # markdown table
    python -m evals.retrieval_eval --json     # machine-readable
    python -m evals.retrieval_eval --check    # exit 1 if any floor is violated

OSS/learning only: toy corpus + hand labels. The numbers teach the *method*
(measure retrieval before tuning rerank/generation); they are not a benchmark,
not RAGAS, not BEIR, and involve no LLM judge.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from app.embeddings import HashingEmbedder
from app.rewrite import rewrite_query
from app.vectorstore import NumpyVectorStore, load_corpus

ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_QRELS = EVALS_DIR / "qrels.json"
DEFAULT_FLOORS = EVALS_DIR / "retrieval_floors.json"

MODES: tuple[str, ...] = ("dense", "bm25", "hybrid", "hybrid+rewrite")
KS: tuple[int, ...] = (1, 3, 5)
DEPTH = 10  # ranked-list depth used for MRR

# Retriever signature: (query) -> ranked chunk_ids (best first)
Retriever = Callable[[str], list[str]]


# ---------------------------------------------------------------------------
# Metrics (stdlib only). `ranked` = retrieved chunk_ids best-first;
# `relevant` = {chunk_id: grade>0}. Binary metrics treat any grade > 0 as relevant.
# ---------------------------------------------------------------------------

def hit_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """1.0 if any relevant chunk is in the top k, else 0.0."""
    return 1.0 if any(doc in relevant for doc in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Fraction of all relevant chunks found in the top k."""
    if not relevant:
        return 0.0
    found = sum(1 for doc in set(ranked[:k]) if doc in relevant)
    return found / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Mapping[str, int]) -> float:
    """1 / rank of the first relevant chunk (0.0 if none retrieved)."""
    for rank, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Graded nDCG@k with gain = 2^grade - 1 and log2(rank + 1) discount."""
    if not relevant:
        return 0.0

    def dcg(grades: Iterable[int]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))

    actual = dcg(relevant.get(doc, 0) for doc in ranked[:k])
    ideal = dcg(sorted(relevant.values(), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def metric_names(ks: Sequence[int] = KS) -> list[str]:
    names: list[str] = []
    for k in ks:
        names += [f"hit@{k}", f"recall@{k}", f"ndcg@{k}"]
    names.append("mrr")
    return names


def score_query(ranked: Sequence[str], relevant: Mapping[str, int], ks: Sequence[int] = KS) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(ranked, relevant, k)
        out[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
    out["mrr"] = reciprocal_rank(ranked, relevant)
    return out


# ---------------------------------------------------------------------------
# Qrels + retrievers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QrelQuery:
    id: str
    query: str
    relevant: dict[str, int]
    tags: list[str] = field(default_factory=list)

    @property
    def answerable(self) -> bool:
        return bool(self.relevant)


@dataclass(frozen=True)
class Qrels:
    corpus_dir: Path
    queries: list[QrelQuery]


def load_qrels(path: Path | str = DEFAULT_QRELS) -> Qrels:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    corpus_dir = Path(data.get("corpus_dir", "evals/corpus"))
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir
    queries = [
        QrelQuery(
            id=q["id"],
            query=q["query"],
            relevant={str(k): int(v) for k, v in (q.get("relevant") or {}).items() if int(v) > 0},
            tags=list(q.get("tags") or []),
        )
        for q in data["queries"]
    ]
    return Qrels(corpus_dir=corpus_dir, queries=queries)


def build_store(corpus_dir: Path | str) -> NumpyVectorStore:
    store = NumpyVectorStore(HashingEmbedder(dim=256))
    store.add(load_corpus(corpus_dir))
    return store


def make_retrievers(store: NumpyVectorStore, depth: int = DEPTH) -> dict[str, Retriever]:
    def ids(hits) -> list[str]:
        return [h.chunk.doc_id for h in hits]

    return {
        "dense": lambda q: ids(store.search_dense(q, top_k=depth)),
        "bm25": lambda q: ids(store.search_bm25(q, top_k=depth)),
        "hybrid": lambda q: ids(store.search(q, top_k=depth)),
        "hybrid+rewrite": lambda q: ids(store.search(rewrite_query(q).rewritten, top_k=depth)),
    }


# ---------------------------------------------------------------------------
# Eval + report
# ---------------------------------------------------------------------------

@dataclass
class ModeResult:
    mode: str
    means: dict[str, float]
    per_query: dict[str, dict[str, float]]
    rankings: dict[str, list[str]]


@dataclass
class EvalReport:
    n_queries: int
    n_answerable: int
    probes: list[str]
    modes: dict[str, ModeResult]

    def to_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "n_answerable": self.n_answerable,
            "unanswerable_probes": self.probes,
            "metrics": {m: {k: round(v, 4) for k, v in r.means.items()} for m, r in self.modes.items()},
        }


def evaluate(
    qrels: Qrels | None = None,
    retrievers: Mapping[str, Retriever] | None = None,
    *,
    ks: Sequence[int] = KS,
) -> EvalReport:
    qrels = qrels or load_qrels()
    if retrievers is None:
        corpus_ids = {c.doc_id for c in load_corpus(qrels.corpus_dir)}
        unknown = sorted({d for q in qrels.queries for d in q.relevant} - corpus_ids)
        if unknown:
            raise ValueError(f"qrels reference unknown chunk_ids: {unknown}")
        retrievers = make_retrievers(build_store(qrels.corpus_dir))
    answerable = [q for q in qrels.queries if q.answerable]
    probes = [q.id for q in qrels.queries if not q.answerable]
    names = metric_names(ks)
    modes: dict[str, ModeResult] = {}
    for mode, retrieve in retrievers.items():
        per_query: dict[str, dict[str, float]] = {}
        rankings: dict[str, list[str]] = {}
        for q in qrels.queries:
            ranked = retrieve(q.query)
            rankings[q.id] = ranked
            if q.answerable:
                per_query[q.id] = score_query(ranked, q.relevant, ks)
        n = max(len(answerable), 1)
        means = {name: sum(pq[name] for pq in per_query.values()) / n for name in names}
        modes[mode] = ModeResult(mode=mode, means=means, per_query=per_query, rankings=rankings)
    return EvalReport(
        n_queries=len(qrels.queries),
        n_answerable=len(answerable),
        probes=probes,
        modes=modes,
    )


TABLE_COLUMNS: tuple[str, ...] = ("hit@1", "hit@3", "recall@3", "recall@5", "mrr", "ndcg@3", "ndcg@5")


def markdown_table(report: EvalReport, columns: Sequence[str] = TABLE_COLUMNS) -> str:
    header = "| mode | " + " | ".join(columns) + " |"
    sep = "|---|" + "|".join("---:" for _ in columns) + "|"
    rows = [
        f"| {mode} | " + " | ".join(f"{res.means[c]:.3f}" for c in columns) + " |"
        for mode, res in report.modes.items()
    ]
    note = (
        f"\n{report.n_answerable} answerable queries"
        + (f"; {len(report.probes)} unanswerable probe(s) excluded from means: {', '.join(report.probes)}" if report.probes else "")
        + f". MRR over top-{DEPTH}."
    )
    return "\n".join([header, sep, *rows]) + "\n" + note


# ---------------------------------------------------------------------------
# Floors (CI gate)
# ---------------------------------------------------------------------------

def load_floors(path: Path | str = DEFAULT_FLOORS) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_floors(report: EvalReport, floors: Mapping | None = None) -> list[str]:
    """Return human-readable violations (empty list == pass).

    ``floors`` shape::

        {"min": {"hybrid": {"recall@3": 0.8}},
         "no_worse_than": [["hybrid", "dense", "recall@3"]]}
    """
    floors = floors if floors is not None else load_floors()
    violations: list[str] = []
    for mode, metrics in (floors.get("min") or {}).items():
        if mode not in report.modes:
            violations.append(f"floor references unknown mode {mode!r}")
            continue
        for metric, floor in metrics.items():
            actual = report.modes[mode].means[metric]
            if actual + 1e-9 < float(floor):
                violations.append(f"{mode} {metric}={actual:.3f} < floor {float(floor):.3f}")
    for better, worse, metric in floors.get("no_worse_than") or []:
        a = report.modes[better].means[metric]
        b = report.modes[worse].means[metric]
        if a + 1e-9 < b:
            violations.append(f"{better} {metric}={a:.3f} < {worse} {metric}={b:.3f}")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qrels", default=str(DEFAULT_QRELS))
    parser.add_argument("--floors", default=str(DEFAULT_FLOORS))
    parser.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    parser.add_argument("--check", action="store_true", help="exit 1 if any floor is violated")
    args = parser.parse_args(argv)

    report = evaluate(load_qrels(args.qrels))
    violations = check_floors(report, load_floors(args.floors)) if args.check else []
    if args.json:
        out = report.to_dict()
        if args.check:
            out["floor_violations"] = violations
        print(json.dumps(out, indent=2))
    else:
        print(markdown_table(report))
        if args.check:
            print("\nfloors: " + ("OK" if not violations else "FAILED"))
            for v in violations:
                print(f"  - {v}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
