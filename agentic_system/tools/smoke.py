"""Exercise the actual catalog and local Qwen model without running the API."""

import argparse
import json

from music_recommender.agent import build_agent, create_llm
from music_recommender.config import Settings
from music_recommender.embeddings import LocalEmbedder
from music_recommender.retrieval import Retriever
from music_recommender.schemas import RecommendationRequest
from music_recommender.store import CatalogStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true")
    providers = parser.add_mutually_exclusive_group()
    providers.add_argument("--groq", action="store_true")
    providers.add_argument("--openrouter", action="store_true")
    parser.add_argument("--query", default="Mellow instrumental music for a rainy evening")
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.data_dir /= "sample" if args.sample else "catalog"
    store = CatalogStore(settings)
    llm = None
    if args.groq or args.openrouter:
        settings.llm_provider = "openrouter" if args.openrouter else "groq"
        if not settings.api_key():
            parser.error(f"Configure the {settings.llm_provider} API key in .env.")
        llm = create_llm(settings, settings.api_key())
    graph = build_agent(settings, Retriever(settings, store, LocalEmbedder(settings)), llm)
    response = graph.invoke({"request": RecommendationRequest(message=args.query), "notices": []})[
        "response"
    ]
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
