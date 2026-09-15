from __future__ import annotations

from app.config import Settings
from app.providers.base import GenerationProvider
from app.providers.mock import MockProvider


class BedrockProvider(GenerationProvider):
    """
    Bedrock stub for learning demos.

    Without AWS credentials this no-ops to the mock provider so tests and local
    runs never require AWS. With credentials present it still mocks unless boto3
    and a live invoke are intentionally wired later — keep the demo dependency-free.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._fallback = MockProvider()

    def _has_credentials(self) -> bool:
        return bool(self.settings.aws_access_key_id and self.settings.aws_secret_access_key)

    def generate(self, question: str, context_chunks: list[str]) -> str:
        if not self._has_credentials():
            return (
                "[bedrock-stub] AWS credentials not configured; using mock fallback.\n"
                + self._fallback.generate(question, context_chunks)
            )
        # Intentional demo stub: do not call AWS from this OSS learning repo by default.
        return (
            f"[bedrock-stub] Credentials detected for region={self.settings.aws_region} "
            f"model={self.settings.bedrock_model_id}, but live Bedrock invoke is disabled "
            "in this learning demo. Mock answer follows.\n"
            + self._fallback.generate(question, context_chunks)
        )
