"""Generate/resume TalkPlay test predictions and score them against held-out labels."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from .config import Settings
from .evaluation.metrics import prediction_key, summarize, validate_predictions
from .evaluation.protocol import CATALOG, DATASET, extract_cases, sample_sessions


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except ValueError:
            raise ValueError(
                f"Invalid checkpoint JSON at {path.name}:{number}; repair before resuming."
            ) from None
    return rows


def append_checkpoint(path: Path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_inputs(settings, output, count, seed):
    """Snapshot public evaluation inputs once, pinning both dataset revisions."""
    path = output / "inputs.json"
    if path.exists():
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if (bundle["sample_count"], bundle["seed"]) != (count, seed):
            raise ValueError(
                "This output directory uses a different sample. Choose another directory."
            )
        return bundle
    from datasets import load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    revision = api.dataset_info(DATASET).sha
    catalog_revision = api.dataset_info(CATALOG).sha
    rows = list(
        load_dataset(
            DATASET,
            split="test",
            revision=revision,
            streaming=True,
            cache_dir=str(settings.cache_dir / "datasets"),
        )
    )
    selected = sample_sessions(rows, count, seed)
    # Catalog denominator includes tracks lacking embeddings in our local index.
    catalog_ids = [
        str(row["track_id"])
        for row in load_dataset(
            CATALOG,
            split="all_tracks",
            revision=catalog_revision,
            streaming=True,
            cache_dir=str(settings.cache_dir / "datasets"),
        )
    ]
    if not catalog_ids or len(catalog_ids) != len(set(catalog_ids)):
        raise ValueError("Official catalog IDs are empty or duplicated.")
    # Strip hidden thoughts, goals and assessments even from the saved input snapshot.
    safe_rows = [
        {
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "user_profile": row.get("user_profile") or {},
            "conversations": [
                {key: entry[key] for key in ("role", "turn_number", "content")}
                for entry in row["conversations"]
            ],
        }
        for row in selected
    ]
    bundle = {
        "dataset": DATASET,
        "split": "test",
        "revision": revision,
        "catalog": CATALOG,
        "catalog_revision": catalog_revision,
        "total_test_sessions": len(rows),
        "sample_count": count,
        "seed": seed,
        "sessions": safe_rows,
        "catalog_ids": sorted(catalog_ids),
    }
    write_json(path, bundle)
    return bundle


def run(args):
    settings = Settings.from_env()
    settings.llm_provider = args.mode
    settings.data_dir = (args.data_dir or settings.data_dir / "catalog").resolve()
    # Twenty explanations need more space than the interactive five-track default.
    settings.max_output_tokens = 4096
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_inputs(settings, output, args.sessions, args.seed)
    cases = [case for row in bundle["sessions"] for case in extract_cases(row)]
    catalog_ids = set(bundle["catalog_ids"])
    predictions_path = output / "checkpoint.jsonl"
    predictions = read_checkpoint(predictions_path)
    _, completed = validate_predictions(cases, predictions, catalog_ids)
    manifest_path = output / "manifest.json"
    existing_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if not args.score_only:
        from .agent import build_agent, create_llm
        from .embeddings import LocalEmbedder
        from .retrieval import Retriever
        from .store import CatalogStore

        if not (settings.data_dir / "catalog.sqlite3").exists():
            raise ValueError("Full catalog is missing. Run music_recommender.ingest first.")
        store = CatalogStore(settings)
        if not store.state("ready") or (store.state("scope") or {}).get("sample") is not False:
            raise ValueError("Evaluation requires a ready full catalog, not a sample index.")
        source_root = Path(__file__).parent
        source_hash = digest(
            {
                str(p.relative_to(source_root)): p.read_text(encoding="utf-8")
                for p in sorted(source_root.rglob("*.py"))
            }
        )
        manifest = {
            "protocol": "public-test-teacher-forced-v1",
            "input_hash": digest(bundle),
            "mode": args.mode,
            "settings": settings.model_dump(mode="json"),
            "allow_fallbacks": args.allow_fallbacks,
            "source_hash": source_hash,
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
        if not settings.api_key():
            raise ValueError(f"API key is required for --mode {args.mode}.")
        llm = create_llm(settings, settings.api_key())
        agent = build_agent(settings, Retriever(settings, store, LocalEmbedder(settings)), llm)
        indexed = set(store.records("tracks"))
        write_json(
            output / "coverage.json",
            {
                "official_catalog_size": len(catalog_ids),
                "indexed_catalog_size": len(indexed),
                "target_turns_missing_from_index": sum(
                    c.target_track_id not in indexed for c in cases
                ),
                "note": "Targets absent from the searchable index remain in evaluation and count as misses.",
            },
        )
        for case in cases:
            if case.key in completed:
                continue
            if args.mode == "openrouter" and settings.openrouter_model.endswith(":free"):
                from .evaluation.openrouter import require_turn_budget

                try:
                    allowance = require_turn_budget(settings.api_key(), required=5)
                except RuntimeError as exc:
                    write_json(
                        output / "run_status.json",
                        {
                            "status": "stopped",
                            "reason": str(exc),
                            "completed_turns": len(predictions),
                            "total_turns": len(cases),
                        },
                    )
                    raise
                write_json(output / "allowance.json", allowance)
            started = time.monotonic()
            response = agent.invoke({"request": case.request, "notices": []})["response"]
            if response.mode == "fallback" and not args.allow_fallbacks:
                write_json(
                    output / "last_failure.json",
                    {
                        "session_id": case.session_id,
                        "turn_number": case.turn_number,
                        "notices": response.notices,
                    },
                )
                raise RuntimeError(
                    f"{args.mode} fell back locally. Run paused; inspect last_failure.json and rerun to resume."
                )
            row = {
                "session_id": case.session_id,
                "user_id": case.user_id,
                "turn_number": case.turn_number,
                "predicted_track_ids": [r.track.track_id for r in response.recommendations],
                "predicted_response": "\n".join(
                    f"{r.track.title} by {', '.join(r.track.artists)}: {r.explanation}"
                    for r in response.recommendations
                ),
                "mode": response.mode,
                "notices": response.notices,
                "seconds": round(time.monotonic() - started, 3),
            }
            validate_predictions([case], [row], catalog_ids)
            append_checkpoint(predictions_path, row)
            predictions.append(row)
            write_json(
                output / "run_status.json",
                {
                    "status": "running",
                    "completed_turns": len(predictions),
                    "total_turns": len(cases),
                },
            )
            print(f"Completed {len(predictions)}/{len(cases)} turns ({row['mode']}).", flush=True)
            if args.delay:
                time.sleep(args.delay)
    elif existing_manifest is None or existing_manifest["input_hash"] != digest(bundle):
        raise ValueError("Scoring requires the original manifest and matching input snapshot.")
    scores = summarize(cases, predictions, catalog_ids, seed=args.seed)
    write_json(
        output / "predictions.json",
        [
            {
                key: row[key]
                for key in (
                    "session_id",
                    "user_id",
                    "turn_number",
                    "predicted_track_ids",
                    "predicted_response",
                )
            }
            for row in predictions
        ],
    )
    scores["evaluation_scope"] = (
        "full_public_test"
        if len(bundle["sessions"]) == bundle["total_test_sessions"]
        else "sampled_public_test"
    )
    scores["protocol"] = (
        "Teacher-forced previous turns; no goals, specificity, thoughts or future answers."
    )
    scores["judge_note"] = (
        "Official Gemini judge is private. No judge score or full composite computed."
    )
    write_json(output / "scores.json", scores)
    if args.judge_model:
        from .evaluation.judge import JUDGE_PROMPT, judge_response

        key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPEN_ROUTER_API_KEY")
        if not key:
            raise ValueError("Metrics saved. Set OPEN_ROUTER_API_KEY for the Nemotron judge.")
        judge_manifest = {
            "provider": "openrouter",
            "model": args.judge_model,
            "prompt": JUDGE_PROMPT,
            "predictions_hash": digest(predictions),
            "inputs_hash": digest(bundle),
            "normalization": "(score - 1) / 4; equal dimension weights",
        }
        path = output / "judge_manifest.json"
        if path.exists() and json.loads(path.read_text()) != judge_manifest:
            raise ValueError("Judge configuration changed; use a separate run directory.")
        write_json(path, judge_manifest)
        judged = read_checkpoint(output / "judge_checkpoint.jsonl")
        judgments = {prediction_key(row): row["score"] for row in judged}
        if len(judgments) != len(judged) or not set(judgments) <= {case.key for case in cases}:
            raise ValueError("Judge checkpoint contains duplicate or unexpected turns.")
        by_key = {prediction_key(row): row for row in predictions}
        for case in cases:
            if case.key not in judgments:
                score = judge_response(
                    case.request, by_key[case.key]["predicted_response"], args.judge_model, key
                )
                judgments[case.key] = score.model_dump()
                append_checkpoint(
                    output / "judge_checkpoint.jsonl",
                    {
                        "session_id": case.session_id,
                        "turn_number": case.turn_number,
                        "score": score.model_dump(),
                    },
                )
                print(f"Judged {len(judgments)}/{len(cases)} turns (unofficial proxy).", flush=True)
        scores.update(summarize(cases, predictions, catalog_ids, judgments, seed=args.seed))
        scores["judge_note"] = (
            "Unofficial OpenRouter judge with a public local prompt; not the official Gemini blind-set score."
        )
        write_json(output / "scores.json", scores)
    print(json.dumps(scores, indent=2), flush=True)
    write_json(
        output / "run_status.json",
        {
            "status": "complete",
            "completed_turns": len(predictions),
            "total_turns": len(cases),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions", type=int, default=100, help="Random complete sessions; 0 means all."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["groq", "openrouter"], default="groq")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--delay", type=float, default=0, help="Seconds between recommendation turns."
    )
    parser.add_argument(
        "--allow-fallbacks",
        action="store_true",
        help="Permit explicitly labeled mixed-mode results.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Score saved predictions without regenerating them.",
    )
    parser.add_argument(
        "--judge-model",
        default=Settings.from_env().openrouter_model,
        help="OpenRouter judge model; defaults to configured Nemotron.",
    )
    parser.add_argument(
        "--no-judge",
        dest="judge_model",
        action="store_const",
        const=None,
        help="Compute retrieval/diversity metrics only.",
    )
    args = parser.parse_args()
    if args.sessions < 0 or args.delay < 0:
        parser.error("Sessions and delay must be nonnegative.")
    try:
        run(args)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Evaluation stopped: {exc}\n")


if __name__ == "__main__":
    main()
