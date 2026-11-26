import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    data_dir: Path = PROJECT_ROOT / "data"
    cache_dir: Path = PROJECT_ROOT / ".cache"
    groq_model: str = "openai/gpt-oss-20b"
    llm_provider: Literal["groq", "openrouter"] = "groq"
    openrouter_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    openrouter_output_method: str = "function_calling"
    output_method: str = "json_schema"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=8, ge=1)
    track_embedding_column: str = "metadata-qwen3_embedding_0.6b"
    dimension: int = 1024
    candidate_count: int = 24
    neighbor_count: int = 8
    cf_history_mode: Literal["official_test", "internal_validation"] = "official_test"
    max_output_tokens: int = 1200

    def api_key(self) -> str:
        """Read credentials only when needed; never store them in run manifests."""
        if self.llm_provider == "openrouter":
            return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY") or ""
        return os.getenv("GROQ_API_KEY", "")

    @property
    def model_name(self) -> str:
        return self.openrouter_model if self.llm_provider == "openrouter" else self.groq_model

    @property
    def structured_output_method(self) -> str:
        return (
            self.openrouter_output_method
            if self.llm_provider == "openrouter"
            else self.output_method
        )

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")
        result = cls(
            llm_provider=os.getenv("LLM_PROVIDER") or "groq",
            openrouter_model=os.getenv("OPENROUTER_MODEL")
            or os.getenv("OPEN_ROUTER_MODEL")
            or "nvidia/nemotron-3-ultra-550b-a55b:free",
            openrouter_output_method=os.getenv("OPENROUTER_OUTPUT_METHOD") or "function_calling",
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            output_method=os.getenv("GROQ_OUTPUT_METHOD", "json_schema"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
            cf_history_mode=os.getenv("CF_HISTORY_MODE", "official_test"),
            track_embedding_column=os.getenv(
                "TRACK_EMBEDDING_COLUMN", "metadata-qwen3_embedding_0.6b"
            ),
        )
        # Keep model/data downloads within this project, not the user's global cache.
        os.environ.setdefault("HF_HOME", str(result.cache_dir / "huggingface"))
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        return result
