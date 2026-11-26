import os
from contextlib import asynccontextmanager
from importlib import import_module
from pathlib import Path

from fastapi import FastAPI

from music_recommender.agent import build_agent, create_llm
from music_recommender.config import Settings
from music_recommender.embeddings import LocalEmbedder
from music_recommender.retrieval import Retriever
from music_recommender.router import router
from music_recommender.store import CatalogStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    catalog_name = os.getenv("TALKPLAY_CATALOG", "catalog")
    if catalog_name not in {"catalog", "sample"}:
        raise ValueError("TALKPLAY_CATALOG must be either 'catalog' or 'sample'.")
    settings.data_dir = Path(settings.data_dir) / catalog_name
    catalog_path = settings.data_dir / "catalog.sqlite3"
    if not catalog_path.exists():
        raise RuntimeError(
            f"Catalog not found at {catalog_path}. Run `python -m music_recommender.ingest` first."
        )

    store = CatalogStore(settings)
    if not store.state("ready"):
        raise RuntimeError(
            f"Catalog at {catalog_path} is not ready. Complete ingestion before starting the API."
        )

    retriever = Retriever(settings, store, LocalEmbedder(settings))
    api_key = settings.api_key()
    llm = create_llm(settings, api_key) if api_key else None
    memory_checkpoint = import_module("langgraph.checkpoint.memory")
    saver_class = getattr(memory_checkpoint, "InMemorySaver", None) or memory_checkpoint.MemorySaver
    app.state.agent = build_agent(
        settings,
        retriever,
        llm,
        checkpointer=saver_class(),
    )
    yield
    del app.state.agent


app = FastAPI(title="TalkPlay Recommendation API", lifespan=lifespan)
app.include_router(router)
