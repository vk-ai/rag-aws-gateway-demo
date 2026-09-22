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

Hybrid retrieval stays fully offline (no vector DB, no live Bedrock invoke). Hashing
embeddings alone can under-rank exact IDs/acronyms; BM25 + RRF is the teaching fix.

`grounding_score` is **token-overlap** of answer vs retrieved context — a cheap teaching
signal with the same *shape* as industry “check grounding” APIs. It is **not** RAGAS
faithfulness, not NLI entailment, and not Google Check Grounding.

No API keys or AWS credentials are required to run or test.

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
| `bedrock` | Stub that mocks unless AWS keys are present; still no live invoke in this learning demo. |

## Tests

```bash
pytest -q
```

Runs fully offline (no network, no AWS).

## CI

Workflow definition: [`ci/github-actions.yml`](ci/github-actions.yml).
To enable GitHub Actions, copy it to `.github/workflows/ci.yml` (requires a token
with the `workflow` scope to push that path).

## Project layout

```
app/
  main.py          # FastAPI POST /query, GET /health
  rag.py           # retrieve-then-generate orchestration
  citations.py     # citations[] + lexical grounding_score
  embeddings.py    # hashing embedder
  bm25.py          # Okapi BM25
  vectorstore.py   # hybrid BM25 + cosine + RRF + corpus loader
  providers/       # mock | openai | bedrock
data/corpus/       # fixture documents
tests/             # pytest
.github/workflows/ # CI
```

## License

MIT — see [LICENSE](LICENSE).
