import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    data_dir: Path = PROJECT_ROOT / "data"
    cache_dir: Path = PROJECT_ROOT / ".cache"
    groq_model: str = "openai/gpt-oss-20b"
    output_method: str = "json_schema"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=8, ge=1)
    track_embedding_column: str = "metadata-qwen3_embedding_0.6b"
    dimension: int = 1024
    candidate_count: int = 24
    neighbor_pool: int = 80
    neighbor_count: int = 8
    max_output_tokens: int = 1200

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")
        result = cls(
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            output_method=os.getenv("GROQ_OUTPUT_METHOD", "json_schema"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
            track_embedding_column=os.getenv(
                "TRACK_EMBEDDING_COLUMN", "metadata-qwen3_embedding_0.6b"
            ),
        )
        # Keep model/data downloads within this project, not the user's global cache.
        os.environ.setdefault("HF_HOME", str(result.cache_dir / "huggingface"))
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        return result
