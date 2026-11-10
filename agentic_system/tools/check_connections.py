"""Verify configured provider credentials without printing secrets or account details."""

import json
import os
import urllib.error
import urllib.request

from music_recommender.config import Settings


def request_json(url, token, payload=None):
    headers = {"Authorization": f"Bearer {token}", "User-Agent": "TalkPlay-Prototype/0.1"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, headers=headers, data=data)
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.load(response), None
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError):
        return None, "Network connection failed"


def main():
    settings = Settings.from_env()
    reports = {}
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        _, error = request_json("https://huggingface.co/api/whoami-v2", hf_token)
        reports["hugging_face_auth"] = error or "OK"
    else:
        reports["hugging_face_auth"] = "HF_TOKEN is not configured"

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        payload = {
            "model": settings.groq_model,
            "messages": [{"role": "user", "content": "Return the status ok."}],
            "temperature": 0,
            "max_completion_tokens": 512,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "connection_status",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"status": {"type": "string", "enum": ["ok"]}},
                        "required": ["status"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        if settings.groq_model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = "low"
        response, error = request_json(
            "https://api.groq.com/openai/v1/chat/completions", groq_key, payload
        )
        if error:
            reports["groq_structured_output"] = error
        else:
            try:
                content = json.loads(response["choices"][0]["message"]["content"])
                reports["groq_structured_output"] = (
                    "OK" if content == {"status": "ok"} else "Unexpected response"
                )
            except (KeyError, IndexError, TypeError, ValueError):
                reports["groq_structured_output"] = "Invalid structured response"
        reports["groq_model"] = settings.groq_model
    else:
        reports["groq_structured_output"] = "GROQ_API_KEY is not configured"
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
