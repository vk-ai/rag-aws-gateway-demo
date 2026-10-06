from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.abstain import DEFAULT_MIN_EVIDENCE


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    rag_provider: str = "mock"
    rag_top_k: int = 3
    # Deterministic mock query rewrite before retrieve (set false to disable)
    rag_rewrite: bool = True
    rag_corpus_dir: str = "data/corpus"
    # "Insufficient evidence" abstention gate: skip generation when the
    # normalized BM25 evidence score of the best hit is below the threshold.
    # Default threshold is calibrated by `python -m evals.retrieval_eval --calibrate`.
    rag_abstain: bool = True
    rag_min_evidence: float = DEFAULT_MIN_EVIDENCE

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    # Explicit opt-in for live Bedrock InvokeModel. Default false keeps the
    # demo fully offline even when AWS_* creds are present in the environment.
    rag_bedrock_live: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
