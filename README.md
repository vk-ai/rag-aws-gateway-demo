# rag-aws-gateway-demo

Minimal **retrieve-then-generate** RAG service for learning and OSS demos.

> **Important:** This is an educational / open-source demo only. It is **not**
> production software and does **not** represent any employer's (including
> Lowe's) production systems, gateways, or architectures.

## What it does

1. Loads a small local markdown/txt corpus (`data/corpus/`)
2. Retrieves with **hybrid** search: Okapi BM25 (keyword) + hashing-cosine (dense) fused by **Reciprocal Rank Fusion (RRF)**
3. Returns per-hit `dense_score`, `bm25_score`, `rrf_score`, and `channel_ranks` in `retrieved`
4. Generates an answer via a pluggable provider (`mock` by default) with `[n]` citation markers
5. Returns `citations[]` (chunk_id + quote span + marker) and a lexical `grounding_score` (0–1)
6. **Abstains** with `status: "insufficient_evidence"` (no generation) when retrieval evidence is too weak
7. Optional **mock query rewrite** (expand abbreviations / strip filler) before retrieve; response includes `rewritten_query` (+ `rewrite_ops`); disable via `RAG_REWRITE=false` or `{"rewrite": false}`

Hybrid retrieval stays fully offline (no vector DB, no live Bedrock invoke). Hashing
embeddings alone can under-rank exact IDs/acronyms; BM25 + RRF is the teaching fix.

`grounding_score` is **token-overlap** of answer vs retrieved context — a cheap teaching
signal with the same *shape* as industry “check grounding” APIs. It is **not** RAGAS
faithfulness, not NLI entailment, and not Google Check Grounding.

No API keys or AWS credentials are required to run or test.


## Mock query rewrite

Hybrid RRF cannot fix a bad query. Before retrieve, a deterministic rewriter:

- strips filler ("can you please…", "tell me…")
- expands abbreviations (`RAG` → `retrieval augmented generation`, `SKU`, `API`, …)
- exposes `rewritten_query` and `rewrite_ops[]` on the `/query` response

```bash
# Default: rewrite on
curl -s -X POST http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"What is RAG?"}' | python -m json.tool

# Per-request disable
curl -s -X POST http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"What is RAG?","rewrite":false}' | python -m json.tool
```

Env toggle: `RAG_REWRITE=false` (see `.env.example`).

> **Honesty:** Rule/heuristic mock rewriter — **not** production HyDE, not live LLM
> rewrite cost, not employer search stack. Community signal: r/Rag query rewriting /
> pronoun+acronym threads and hybrid-RAG READMEs that list rewrite after hybrid+rerank.


## Retrieval eval (hit@k / recall@k / MRR / nDCG) + CI floor

Hybrid and rewrite are only "improvements" if you can measure them. `evals/` holds a
small **hand-labelled synthetic** eval set and a stdlib harness that scores four modes:

| Mode | What runs |
|---|---|
| `dense` | hashing-cosine only |
| `bm25` | Okapi BM25 only |
| `hybrid` | BM25 + dense fused with RRF (what `/query` uses) |
| `hybrid+rewrite` | mock query rewrite → hybrid |

- `evals/corpus/`: 30 synthetic chunks (retrieval notes, gateway ops, a returns desk with
  near-duplicate part numbers and bin codes as distractors). This is separate from `data/corpus/`.
- `evals/qrels.json`: 40 queries → `{chunk_id: grade}` (2 = answers, 1 = partial), tagged
  `identifier` / `acronym` / `paraphrase` / `distractor` / …. It includes 12 unanswerable probes
  (`off-topic` and `near-miss`), which are excluded from the means and calibrate the
  abstention gate below.
- `evals/retrieval_floors.json`: per-mode minimums plus relative gates
  (`hybrid ≥ dense` on recall@3 and MRR; `hybrid+rewrite ≥ hybrid` on MRR).

```bash
python -m evals.retrieval_eval            # markdown table
python -m evals.retrieval_eval --json     # machine-readable
python -m evals.retrieval_eval --check    # exit 1 if a floor is violated
pytest tests/test_retrieval_eval.py -q    # same gate in CI
```

Current numbers (28 answerable queries, MRR over top-10):

| mode | hit@1 | hit@3 | recall@3 | recall@5 | mrr | ndcg@3 | ndcg@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.714 | 0.893 | 0.839 | 0.839 | 0.790 | 0.766 | 0.766 |
| bm25 | 0.857 | 0.929 | 0.911 | 0.911 | 0.897 | 0.889 | 0.889 |
| hybrid | 0.750 | 0.893 | 0.875 | 0.875 | 0.819 | 0.819 | 0.819 |
| hybrid+rewrite | 0.750 | 0.929 | 0.875 | 0.875 | 0.838 | 0.827 | 0.827 |

**What the table teaches:**
- Hybrid beats the hashing "dense" channel, but on this set **BM25 alone wins**. The toy dense
  channel is also lexical (no IDF) and pulls RRF toward distractors, so `hybrid ≥ bm25` is
  deliberately *not* gated.
- The rewrite results are mixed per query. `FAQ` → "frequently asked questions" helps
  (`faq_refund`), while `BM25` → "best match 25" drops the exact token and hurts (`bm25_rank`).
  A single mean hides both, so the tests pin both.

> **Honesty:** toy corpus plus hand labels. The numbers teach the *method* ("measure retrieval
> before touching rerank or generation"), not a benchmark. This is not RAGAS, not BEIR, and has no LLM judge.
> The motivation is the community thread [r/Rag: "Hybrid search and reranking made my RAG worse"](https://www.reddit.com/r/Rag/comments/1v7g3oe/hybrid_search_and_reranking_made_my_rag_worse/)
> and [The Neural Base on retriever vs reranker evaluation](https://theneuralbase.com/ragas/learn/intermediate/retriever-vs-reranker-evaluation/).

## Abstention gate: `insufficient_evidence`

Saying "I don't know" is cheaper than a confident answer built on weak context. Before
generation, `/query` computes an **evidence score** in [0, 1]: the best retrieved chunk's BM25
score divided by the maximum BM25 score that query could reach on this corpus. Roughly, it is
the share of the query's IDF-weighted words that the best hit covers. If the score is below
`RAG_MIN_EVIDENCE`, the provider is **not called** and the response says so:

```json
{"status": "insufficient_evidence", "abstained": true, "evidence_score": 0.0277,
 "evidence_threshold": 0.087, "citations": [], "grounding_score": 0.0,
 "answer": "I don't have enough evidence in the indexed documents to answer that. ..."}
```

Answered responses carry `status: "answered"`, `abstained: false` and their `evidence_score`.
You can turn the gate off with `RAG_ABSTAIN=false`, or per request with `{"abstain": false}`.

**Calibrated with the round-4 retrieval eval.** The 28 answerable qrels queries and 12
unanswerable probes are scored with the same signal. The threshold is the one with the fewest
false answers inside a 5% false-refusal budget:

```bash
python -m evals.retrieval_eval --calibrate     # prints the table below
python -m evals.retrieval_eval --check         # CI: also caps abstention error rates
pytest tests/test_abstain.py -q
```

| threshold | false-answer rate | false-refusal rate |
|---|---:|---:|
| calibrated / shipped 0.087 | 0.167 (2/12) | 0.036 (1/28) |

**What it teaches:**
- RRF scores are rank-based and look confident even for nonsense, and the toy hashing-cosine
  channel collides: "What is the capital of Australia?" gets a cosine of 0.52. Neither can gate.
  The normalized BM25 score is bounded and IDF-aware, so it can.
- The 2 false answers are both `near-miss` probes ("warranty period for the cordless drill",
  "Kubernetes version"). Their words *are* in the corpus but the answer is not. A lexical gate
  cannot catch these; that needs a sufficiency judge (NLI or LLM). The eval lists them instead
  of hiding them. The 1 false refusal is a paraphrase ("bring back an opened laptop").
- A test pins the shipped default to the calibration output, so re-labelling the eval set forces
  a deliberate threshold update. The threshold is calibrated on `evals/corpus`; on the small
  `data/corpus` fixture it transfers only roughly (for example "capital of Australia" scores 0.09
  there and is answered), so recalibrate on your own corpus.

> **Honesty:** a lexical teaching gate. It is not NLI or RAGAS context sufficiency, and has no LLM
> judge. Motivation: [unrag docs: topK, thresholds and "no good match"](https://unrag.dev/docs/rag/04-retrieval/01-topk-thresholds-and-no-good-match),
> [QASkills 2026 RAG testing guide](https://qaskills.sh/blog/rag-evaluation-metrics-complete-2026)
> (abstention as a machine-readable status; measure under- and over-abstention), and
> [tianpan.co "RAG's Dirty Secret"](https://tianpan.co/blog/2026/04/10/rag-context-sufficiency-problem).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run API
uvicorn app.main:app --reload --port 8000

# Query
curl -s -X POST http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"What is RAG?"}' | python -m json.tool
```

## Providers

Set `RAG_PROVIDER` (see `.env.example`):

| Value     | Behavior |
|-----------|----------|
| `mock`    | Default. Deterministic offline answer from retrieved context. |
| `openai`  | OpenAI-compatible chat completions if `OPENAI_API_KEY` is set; otherwise mock fallback. |
| `bedrock` | **Stub by default** (`[bedrock-stub]`). Never calls AWS unless you explicitly opt in (below). |

### Optional live Bedrock (opt-in)

The default `bedrock` path is offline and clearly labeled `[bedrock-stub]` — even if
`AWS_*` credentials exist in the environment. That keeps `pytest` and local demos
credential-free.

To attempt a real Bedrock `InvokeModel` (Anthropic Messages-style body, e.g. Claude Haiku):

1. `RAG_PROVIDER=bedrock`
2. Set `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `BEDROCK_MODEL_ID`
3. Set **`RAG_BEDROCK_LIVE=true`** (default is `false`)
4. Install the optional SDK: `pip install boto3` (or `pip install '.[bedrock]'`)

Successful live answers are prefixed `[bedrock-live]`. If `boto3` is missing, creds are
incomplete, or invoke fails, the provider falls back to `[bedrock-stub]` + mock — it does
**not** crash the API.

This optional path is for learning how a gateway-shaped RAG service *could* call Bedrock.
It is **not** a production Bedrock integration and makes no claims about any employer's systems.

## Tests

```bash
pytest -q
```

Runs fully offline (no network, no AWS). Live Bedrock is never exercised in CI.

## CI

Workflow definition: [`ci/github-actions.yml`](ci/github-actions.yml).
To enable GitHub Actions, copy it to `.github/workflows/ci.yml` (requires a token
with the `workflow` scope to push that path).

## Project layout

```
app/
  main.py          # FastAPI POST /query, GET /health
  rag.py           # retrieve-then-generate orchestration (+ abstention gate)
  abstain.py       # evidence score + threshold calibration
  citations.py     # citations[] + lexical grounding_score
  embeddings.py    # hashing embedder
  bm25.py          # Okapi BM25
  vectorstore.py   # hybrid BM25 + cosine + RRF + corpus loader
  providers/       # mock | openai | bedrock (stub + optional live)
data/corpus/       # fixture documents
evals/             # retrieval eval: synthetic corpus, qrels, floors, harness
tests/             # pytest
.github/workflows/ # CI
```

## License

MIT — see [LICENSE](LICENSE).
