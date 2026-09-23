from __future__ import annotations

import json
from typing import Any

from app.config import Settings
from app.providers.base import GenerationProvider
from app.providers.mock import MockProvider


class BedrockProvider(GenerationProvider):
    """
    Optional AWS Bedrock generator for this OSS/learning demo.

    Default path is a clearly labeled stub (no AWS calls) so ``pytest`` and
    local runs stay offline. Live ``InvokeModel`` only runs when
    ``RAG_BEDROCK_LIVE=true`` *and* AWS credentials are set *and* optional
    ``boto3`` is installed. Failures fall back to the stub — never crash the API.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._fallback = MockProvider()

    def _has_credentials(self) -> bool:
        return bool(self.settings.aws_access_key_id and self.settings.aws_secret_access_key)

    def _stub(self, reason: str, question: str, context_chunks: list[str]) -> str:
        return (
            f"[bedrock-stub] {reason}\n"
            + self._fallback.generate(question, context_chunks)
        )

    def generate(self, question: str, context_chunks: list[str]) -> str:
        live = bool(self.settings.rag_bedrock_live)

        if not live:
            if self._has_credentials():
                return self._stub(
                    f"Credentials detected for region={self.settings.aws_region} "
                    f"model={self.settings.bedrock_model_id}, but live Bedrock invoke "
                    "is disabled via RAG_BEDROCK_LIVE=false. Mock answer follows.",
                    question,
                    context_chunks,
                )
            return self._stub(
                "AWS credentials not configured; live invoke requires "
                "RAG_BEDROCK_LIVE=true plus AWS keys. Using mock fallback.",
                question,
                context_chunks,
            )

        if not self._has_credentials():
            return self._stub(
                "RAG_BEDROCK_LIVE=true but AWS credentials are missing "
                "(set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY). Using mock fallback.",
                question,
                context_chunks,
            )

        try:
            import boto3  # optional hard dep — not in default requirements.txt
        except ImportError:
            return self._stub(
                "RAG_BEDROCK_LIVE=true but boto3 is not installed "
                "(pip install boto3). Using mock fallback.",
                question,
                context_chunks,
            )

        try:
            text = self._invoke_live(boto3, question, context_chunks)
        except Exception as exc:  # noqa: BLE001 — demo must not crash the API
            return self._stub(
                f"Live Bedrock InvokeModel failed ({exc!s}); using mock fallback.",
                question,
                context_chunks,
            )

        return f"[bedrock-live] {text}"

    def _invoke_live(self, boto3: Any, question: str, context_chunks: list[str]) -> str:
        context = "\n\n".join(context_chunks) if context_chunks else "(no context)"
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.2,
            "system": (
                "You are a helpful RAG assistant for a learning demo. "
                "Answer using only the provided context. If context is insufficient, say so."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                }
            ],
        }
        client = boto3.client(
            "bedrock-runtime",
            region_name=self.settings.aws_region,
            aws_access_key_id=self.settings.aws_access_key_id,
            aws_secret_access_key=self.settings.aws_secret_access_key,
        )
        response = client.invoke_model(
            modelId=self.settings.bedrock_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body).encode("utf-8"),
        )
        raw = response["body"].read()
        parsed = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
        # Anthropic Messages-style on Bedrock: content is a list of blocks
        blocks = parsed.get("content") or []
        texts = [
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type", "text") == "text"
        ]
        joined = "\n".join(t for t in texts if t).strip()
        if not joined:
            raise ValueError(f"empty or unexpected Bedrock response keys={list(parsed.keys())}")
        return joined
