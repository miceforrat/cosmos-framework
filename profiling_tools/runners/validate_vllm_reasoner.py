"""Issue deterministic streaming requests to a Reasoner-only vLLM server."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_PROMPT = (
    "A robot must move a fragile red cup from a table to a shelf. "
    "Give a concise three-step plan. Do not use Markdown."
)


def _request_json(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    parsed = urlparse(base_url)
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=600)
    body = None if payload is None else json.dumps(payload)
    headers = {} if body is None else {"Content-Type": "application/json"}
    connection.request(method, f"{parsed.path.rstrip('/')}{path}", body=body, headers=headers)
    response = connection.getresponse()
    data = response.read()
    connection.close()
    if response.status >= 300:
        raise RuntimeError(f"HTTP {response.status}: {data.decode(errors='replace')}")
    return json.loads(data)


def _model_id(base_url: str) -> str:
    response = _request_json(base_url, "GET", "/models")
    models = response.get("data", [])
    if not models:
        raise RuntimeError("vLLM /models returned no models")
    return str(models[0]["id"])


def _stream_once(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    parsed = urlparse(base_url)
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=600)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise planning assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    connection.request(
        "POST",
        f"{parsed.path.rstrip('/')}/chat/completions",
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status >= 300:
        data = response.read()
        connection.close()
        raise RuntimeError(f"HTTP {response.status}: {data.decode(errors='replace')}")

    first_content_time: float | None = None
    pieces: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    while True:
        line = response.readline()
        if not line:
            break
        decoded = line.decode("utf-8", errors="replace").strip()
        if not decoded.startswith("data:"):
            continue
        data = decoded[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                if first_content_time is None:
                    first_content_time = time.perf_counter()
                pieces.append(str(content))
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
    end = time.perf_counter()
    connection.close()

    text = "".join(pieces)
    completion_tokens = usage.get("completion_tokens")
    decode_seconds = None if first_content_time is None else end - first_content_time
    tokens_per_second = None
    if isinstance(completion_tokens, int) and completion_tokens > 1 and decode_seconds and decode_seconds > 0:
        tokens_per_second = (completion_tokens - 1) / decode_seconds
    return {
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "finish_reason": finish_reason,
        "usage": usage,
        "ttft_seconds": None if first_content_time is None else first_content_time - start,
        "e2e_seconds": end - start,
        "decode_seconds_after_first_content": decode_seconds,
        "decode_tokens_per_second": tokens_per_second,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--mode", choices=("eager", "compile", "graph"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be >= 0 and --repeats must be > 0")

    model = _model_id(args.base_url)
    warmup_records = [
        _stream_once(args.base_url, model, args.prompt, args.max_tokens, args.seed)
        for _ in range(args.warmup)
    ]
    measured = [
        _stream_once(args.base_url, model, args.prompt, args.max_tokens, args.seed)
        for _ in range(args.repeats)
    ]
    hashes = {record["text_sha256"] for record in measured}
    if len(hashes) != 1:
        raise RuntimeError(f"Measured requests were not deterministic: {sorted(hashes)}")

    ttft_values = [record["ttft_seconds"] for record in measured if record["ttft_seconds"] is not None]
    rate_values = [
        record["decode_tokens_per_second"]
        for record in measured
        if record["decode_tokens_per_second"] is not None
    ]
    result = {
        "schema_version": 1,
        "mode": args.mode,
        "base_url": args.base_url,
        "model": model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "warmup": warmup_records,
        "measured": measured,
        "summary": {
            "output_sha256": measured[0]["text_sha256"],
            "median_ttft_seconds": statistics.median(ttft_values) if ttft_values else None,
            "median_e2e_seconds": statistics.median(record["e2e_seconds"] for record in measured),
            "median_decode_tokens_per_second": statistics.median(rate_values) if rate_values else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print("output:", args.output)
    print("text:", measured[0]["text"])


if __name__ == "__main__":
    main()
