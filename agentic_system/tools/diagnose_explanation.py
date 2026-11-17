"""Replay the first evaluation turn and retain explanation diagnostics locally."""

import json
from pathlib import Path

from music_recommender import agent
from music_recommender.config import Settings
from music_recommender.embeddings import LocalEmbedder
from music_recommender.evaluation.protocol import extract_cases
from music_recommender.retrieval import Retriever
from music_recommender.store import CatalogStore

output = Path("data/evaluations/openrouter-nemotron-test-100/diagnostics")
output.mkdir(parents=True, exist_ok=True)
bundle = json.loads((output.parent / "inputs.json").read_text(encoding="utf-8"))
request = extract_cases(bundle["sessions"][0])[0].request
settings = Settings.from_env()
settings.llm_provider = "openrouter"
settings.data_dir /= "catalog"
settings.max_output_tokens = 4096


def save(name, value):
    (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


original_parse = agent.parse_explanation_call
original_validate = agent.validate_explanations


def parse(message):
    save(
        "provider_response.json",
        {
            "tool_calls": message.tool_calls,
            "invalid_tool_calls": message.invalid_tool_calls,
            "response_metadata": message.response_metadata,
            "usage_metadata": message.usage_metadata,
        },
    )
    try:
        return original_parse(message)
    except ValueError as exc:
        save("error.json", {"stage": "parse", "error": str(exc)})
        print("Explanation parse:", str(exc), flush=True)
        raise


def validate(draft, ranked):
    save(
        "validation_input.json",
        {"draft": draft.model_dump(), "ranked": [r.model_dump() for r in ranked]},
    )
    try:
        return original_validate(draft, ranked)
    except ValueError as exc:
        save("error.json", {"stage": "validate", "error": str(exc)})
        print("Explanation validation:", str(exc), flush=True)
        raise


agent.parse_explanation_call = parse
agent.validate_explanations = validate
graph = agent.build_agent(
    settings,
    Retriever(settings, CatalogStore(settings), LocalEmbedder(settings)),
    agent.create_llm(settings, settings.api_key()),
)
for state in graph.stream({"request": request, "notices": []}, stream_mode="values"):
    if "ranked" in state:
        save("ranked.json", [item.model_dump() for item in state["ranked"]])
    if "response" in state:
        print("Result:", state["response"].mode, state["response"].notices, flush=True)
