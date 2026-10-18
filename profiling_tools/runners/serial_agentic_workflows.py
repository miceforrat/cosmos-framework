"""Prepare files for serial prompt-upsampling and agentic T2I workflows.

This utility never loads a model.  It only connects the file outputs of one
Cosmos inference process to the inputs of the next process.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


def _extract_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE))
    for candidate in candidates:
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("Reasoner output did not contain a JSON object")


def _human_prompt(sample: dict[str, Any]) -> str:
    extra = sample.get("extra")
    if isinstance(extra, dict) and isinstance(extra.get("prompt"), str):
        return extra["prompt"]
    prompt = sample.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("Base T2I input has no string prompt")
    return prompt


def make_upsample_request(args: argparse.Namespace) -> None:
    from cosmos_framework.inference.prompt_upsampling import build_t2i_prompt_text

    base = _read_json(args.base_input)
    request = {
        "model_mode": "reasoner",
        "name": args.name,
        "prompt": build_t2i_prompt_text(
            _human_prompt(base), resolution=args.resolution, aspect_ratio=args.aspect_ratio
        ),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    _write_json(args.output, request)


def make_t2i_from_upsample(args: argparse.Namespace) -> None:
    base = _read_json(args.base_input)
    expanded = _extract_json(args.reasoner_text.read_text(encoding="utf-8"))
    base["name"] = args.name
    base["model_mode"] = "text2image"
    base["prompt"] = json.dumps(expanded, ensure_ascii=True, separators=(",", ":"))
    base.setdefault("extra", {})["prompt"] = _human_prompt(base)
    _write_json(args.output, base)


def make_refinement_request(args: argparse.Namespace) -> None:
    base = _read_json(args.base_input)
    original = _human_prompt(base)
    structured = base.get("prompt", original)
    prompt = f"""You are evaluating an image generated from a text-to-image request.

Original user request:
{original}

Current structured generation prompt:
{structured}

Inspect the attached generated image. Return exactly one JSON object with:
- pass: boolean, true only if the image satisfies the important requirements;
- score: number from 0 to 1;
- violations: array of concise strings;
- revised_prompt: a complete revised prompt for the next text-to-image generation.

Preserve correct content and composition. Correct only observed omissions,
attribute errors, counting errors, spatial-relation errors, and rendering
defects. Do not include Markdown fences or commentary outside the JSON object.
"""
    request = {
        "model_mode": "reasoner",
        "name": args.name,
        "prompt": prompt,
        "vision_path": str(args.image),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    _write_json(args.output, request)


def make_refined_t2i(args: argparse.Namespace) -> None:
    base = _read_json(args.base_input)
    critique = _extract_json(args.reasoner_text.read_text(encoding="utf-8"))
    revised = critique.get("revised_prompt")
    if isinstance(revised, dict):
        revised_text = json.dumps(revised, ensure_ascii=True, separators=(",", ":"))
    elif isinstance(revised, str) and revised.strip():
        revised_text = revised.strip()
    else:
        raise ValueError("Reasoner JSON has no non-empty revised_prompt")
    base["name"] = args.name
    base["model_mode"] = "text2image"
    base["prompt"] = revised_text
    base.setdefault("extra", {})["prompt"] = revised_text
    _write_json(args.output, base)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-upsample-request")
    p.add_argument("--base-input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="upsample_reasoner")
    p.add_argument("--resolution", default="480")
    p.add_argument("--aspect-ratio", default="1,1")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.set_defaults(func=make_upsample_request)

    p = sub.add_parser("make-t2i-from-upsample")
    p.add_argument("--base-input", type=Path, required=True)
    p.add_argument("--reasoner-text", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="t2i_upsampled")
    p.set_defaults(func=make_t2i_from_upsample)

    p = sub.add_parser("make-refinement-request")
    p.add_argument("--base-input", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="t2i_critique")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.set_defaults(func=make_refinement_request)

    p = sub.add_parser("make-refined-t2i")
    p.add_argument("--base-input", type=Path, required=True)
    p.add_argument("--reasoner-text", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="t2i_refined")
    p.set_defaults(func=make_refined_t2i)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
