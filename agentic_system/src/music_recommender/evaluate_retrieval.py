"""Score retrieval+ranking alone, bypassing the LLM/agent entirely.

Lets us ablate retrieval sources directly: run the same protocol and metrics as
`music_recommender.evaluate`, but build each turn's candidates from an explicit,
chosen subset of {text, user_to_user, user_to_item} instead of the agent
deciding which tools to call. No LLM is involved, so this is free and fast, and
isolates retrieval/ranking quality from the agent's tool-selection behavior.

`user_to_user` is CF-BPR neighbor-pooled evidence (retrieve_cf): tracks played
after similar listeners' positive feedback. `user_to_item` is a direct
inner-product ranking of the catalog against the listener's own CF-BPR factor.
`item_to_item` is intentionally not offered here: it needs a reference track
the agent would normally pick mid-conversation, which has no equivalent in a
single blind retrieval call.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from .config import Settings
from .evaluate import append_checkpoint, digest, load_inputs, read_checkpoint, write_json
from .evaluation.metrics import summarize, validate_predictions
from .evaluation.protocol import extract_cases
from .ranking import rank_candidates

SOURCES = ("text", "user_to_user", "user_to_item")


def retrieve_selected(retriever, request, sources: list[str]):
    results = []
    if "text" in sources:
        results.append(retriever.retrieve_text(request, request.message))
    if "user_to_user" in sources:
        results.append(retriever.retrieve_cf(request))
    if "user_to_item" in sources:
        results.append(retriever.retrieve_user_to_item(request))
    return retriever.merge_results(request, request.message, results)


def run(args):
    settings = Settings.from_env()
    turns_per_session = args.turns_per_session
    if not 1 <= turns_per_session <= 8:
        raise ValueError("Turn limit per session must be between 1 and 8.")
    sources = sorted(set(args.sources))
    unknown = set(sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"Unknown retrieval source(s): {sorted(unknown)}; choose from {SOURCES}.")
    if not sources:
        raise ValueError("At least one retrieval source is required.")
    settings.cf_history_mode = "official_test"
    settings.data_dir = (args.data_dir or settings.data_dir / "catalog").resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_inputs(settings, output, args.sessions, args.seed)
    cases = [
        case
        for row in bundle["sessions"]
        for case in extract_cases(row)[:turns_per_session]
    ]
    catalog_ids = set(bundle["catalog_ids"])
    predictions_path = output / "checkpoint.jsonl"
    predictions = read_checkpoint(predictions_path)
    _, completed = validate_predictions(cases, predictions, catalog_ids)
    manifest_path = output / "manifest.json"
    existing_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    )
    if not args.score_only:
        from .cf_data import load_history
        from .embeddings import LocalEmbedder
        from .retrieval import Retriever
        from .store import CatalogStore

        if not (settings.data_dir / "catalog.sqlite3").exists():
            raise ValueError("Full catalog is missing. Run music_recommender.ingest first.")
        store = CatalogStore(settings)
        if not store.state("ready") or (store.state("scope") or {}).get("sample") is not False:
            raise ValueError("Evaluation requires a ready full catalog, not a sample index.")
        prepared_dir = args.prepared_dir.resolve()
        history_report = load_history(store, prepared_dir, "official_test")
        if "user_to_item" in sources and not store.state("cf_query_vectors_ready"):
            raise ValueError("Load published test query vectors with music_recommender.cf_data first.")
        source_root = Path(__file__).parent
        code_hash = digest(
            {
                str(p.relative_to(source_root)): p.read_text(encoding="utf-8")
                for p in sorted(source_root.rglob("*.py"))
            }
        )
        manifest = {
            "protocol": "retrieval-only-ablation-v1",
            "retrieval_sources": sources,
            "prepared_history": {
                "source": str(prepared_dir / "conversation_interactions.parquet"),
                "sha256": hashlib.sha256(
                    (prepared_dir / "conversation_interactions.parquet").read_bytes()
                ).hexdigest(),
                "eligible_rows": history_report,
                "query_vectors_hash": digest(store.records("cf_query_vectors")),
            },
            "input_hash": digest(bundle),
            "turns_per_session": turns_per_session,
            "settings": settings.model_dump(mode="json"),
            "source_hash": code_hash,
            "catalog_identity": store.state("identity"),
            "catalog_counts": store.counts(),
            "catalog_scope": store.state("scope"),
        }
        if existing_manifest is not None and existing_manifest != manifest:
            raise ValueError(
                "Run configuration/code changed. Use a new output directory to avoid mixing results."
            )
        if predictions and existing_manifest is None:
            raise ValueError("Checkpoint has no manifest; cannot safely resume.")
        write_json(manifest_path, manifest)
        retriever = Retriever(settings, store, LocalEmbedder(settings))
        mode_label = "+".join(sources)
        for case in cases:
            if case.key in completed:
                continue
            started = time.monotonic()
            merged = retrieve_selected(retriever, case.request, sources)
            ranked = rank_candidates(merged, case.request, [])
            row = {
                "session_id": case.session_id,
                "user_id": case.user_id,
                "turn_number": case.turn_number,
                "predicted_track_ids": [item.candidate.track.track_id for item in ranked],
                "predicted_response": "",
                "mode": mode_label,
                "notices": [],
                "seconds": round(time.monotonic() - started, 3),
            }
            validate_predictions([case], [row], catalog_ids)
            append_checkpoint(predictions_path, row)
            predictions.append(row)
            write_json(
                output / "run_status.json",
                {"status": "running", "completed_turns": len(predictions), "total_turns": len(cases)},
            )
            print(f"Completed {len(predictions)}/{len(cases)} turns ({mode_label}).", flush=True)
    elif existing_manifest is None or existing_manifest["input_hash"] != digest(bundle):
        raise ValueError("Scoring requires the original manifest and matching input snapshot.")
    scores = summarize(cases, predictions, catalog_ids, seed=args.seed)
    scores["retrieval_sources"] = sources
    scores["note"] = (
        "Retrieval-only ablation: no LLM, no agent, no explanations. predicted_response is "
        "always empty, so lexical_diversity and any judge score are not meaningful here; "
        "compare nDCG@k and catalog_diversity against the full-agent run instead."
    )
    write_json(
        output / "predictions.json",
        [
            {k: row[k] for k in ("session_id", "user_id", "turn_number", "predicted_track_ids")}
            for row in predictions
        ],
    )
    write_json(output / "scores.json", scores)
    print(json.dumps(scores, indent=2, ensure_ascii=False), flush=True)
    write_json(
        output / "run_status.json",
        {"status": "complete", "completed_turns": len(predictions), "total_turns": len(cases)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", nargs="+", choices=list(SOURCES), required=True,
        help="Retrieval sources to combine for this run, e.g. --sources user_to_user "
        "or --sources user_to_user user_to_item.",
    )
    parser.add_argument("--sessions", type=int, default=100, help="Random complete sessions; 0 means all.")
    parser.add_argument(
        "--turns-per-session", type=int, default=8,
        help="Evaluate only the first 1-8 turns per sampled session.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--prepared-dir", type=Path,
        default=Path(__file__).resolve().parents[3] / "prepared-dataset-eda" / "data",
        help="Prepared dataset directory; only positive labeled training interactions are used.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--score-only", action="store_true", help="Score saved predictions without regenerating them."
    )
    args = parser.parse_args()
    if args.sessions < 0:
        parser.error("Sessions must be nonnegative.")
    if not 1 <= args.turns_per_session <= 8:
        parser.error("Turns per session must be between 1 and 8.")
    try:
        run(args)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Evaluation stopped: {exc}\n")


if __name__ == "__main__":
    main()
