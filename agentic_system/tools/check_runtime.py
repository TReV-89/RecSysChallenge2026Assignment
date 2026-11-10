"""Check native-library imports and basic operations before downloading datasets."""

import argparse
import importlib
import json
import sys
from importlib.metadata import version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    modules = {"xxhash": "xxhash", "pandas": "pandas", "torch": "torch"}
    if not args.native_only:
        modules.update(
            {
                "datasets": "datasets",
                "chromadb": "chromadb",
                "langchain_groq": "langchain-groq",
                "langgraph.graph": "langgraph",
                "sentence_transformers": "sentence-transformers",
                "streamlit": "streamlit",
            }
        )
    print(f"Interpreter: {sys.executable}", flush=True)
    failed = False
    for module_name, distribution in modules.items():
        try:
            module = importlib.import_module(module_name)
            if module_name == "xxhash":
                module.xxh64(b"runtime-check").hexdigest()
            elif module_name == "pandas":
                assert module.DataFrame({"value": [1, 2]}).value.sum() == 3
            elif module_name == "torch":
                assert module.ones(2, 2).matmul(module.ones(2, 2)).sum().item() == 8
            result = {"module": module_name, "version": version(distribution), "status": "OK"}
        except Exception as exc:  # noqa: BLE001 - report each independent import failure
            failed = True
            result = {"module": module_name, "status": "FAILED", "error": str(exc)}
        print(json.dumps(result), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
