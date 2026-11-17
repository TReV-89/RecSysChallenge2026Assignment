"""Read OpenRouter's daily free-request allowance without exposing credentials."""

import json
import urllib.error
import urllib.request


def free_request_allowance(api_key: str) -> dict | None:
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)["data"]
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenRouter allowance check failed (HTTP {exc.code}).") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("OpenRouter allowance check could not connect.") from None
    counter = data.get("free_model_daily_requests")
    if not isinstance(counter, dict):
        return None
    return {name: counter[name] for name in ("used", "limit", "remaining") if name in counter}


def require_turn_budget(api_key: str, required: int = 4) -> dict | None:
    allowance = free_request_allowance(api_key)
    if allowance is not None and allowance.get("remaining", required) < required:
        raise RuntimeError(
            f"OpenRouter has {allowance['remaining']} free requests remaining today; "
            f"a complete turn may need {required}. Completed predictions are saved. "
            "Rerun the same command after the allowance resets."
        )
    return allowance
