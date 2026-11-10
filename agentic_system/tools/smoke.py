"""Exercise the actual catalog and local Qwen model without opening Streamlit."""

import argparse
import json
import os

from music_recommender.config import Settings
from music_recommender.embeddings import LocalEmbedder
from music_recommender.graph import build_graph, create_llm
from music_recommender.retrieval import Retriever
from music_recommender.schemas import RecommendationRequest
from music_recommender.store import CatalogStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--groq", action="store_true")
    parser.add_argument("--query", default="Mellow instrumental music for a rainy evening")
    args = parser.parse_args()
    settings = Settings.from_env()
    settings.data_dir /= "sample" if args.sample else "catalog"
    store = CatalogStore(settings)
    llm = None
    if args.groq:
        if not os.getenv("GROQ_API_KEY"):
            parser.error("Set GROQ_API_KEY in .env before using --groq.")
        llm = create_llm(settings, os.environ["GROQ_API_KEY"])
    graph = build_graph(settings, Retriever(settings, store, LocalEmbedder(settings)), llm)
    response = graph.invoke({"request": RecommendationRequest(message=args.query), "notices": []})[
        "response"
    ]
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
