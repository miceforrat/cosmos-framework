"""Run the minimal local Cosmos prompt-upsampling -> T2I workflow.

This first implementation deliberately uses two sequential inference
subprocesses.  It is a correctness/profiling bridge that can later grow a
critic/rewrite loop without changing the on-disk stage contracts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from string import Template
from typing import Any

RESOLUTION_RATIO_DICT: dict[str, dict[str, dict[str, int]]] = {
    "256": {
        "1,1": {"W": 256, "H": 256},
        "4,3": {"W": 320, "H": 256},
        "3,4": {"W": 256, "H": 320},
        "16,9": {"W": 320, "H": 192},
        "9,16": {"W": 192, "H": 320},
    },
    "480": {
        "1,1": {"W": 640, "H": 640},
        "4,3": {"W": 736, "H": 544},
        "3,4": {"W": 544, "H": 736},
        "16,9": {"W": 832, "H": 480},
        "9,16": {"W": 480, "H": 832},
    },
}

REQUIRED_T2I_KEYS = {
    "subjects",
    "subject_details",
    "background_setting",
    "lighting",
    "text_and_signage_elements",
    "resolution",
    "aspect_ratio",
    "comprehensive_t2i_caption",
}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first decodable JSON object, tolerating Markdown fences."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Reasoner output did not contain a valid JSON object")


def validate_t2i_prompt(value: dict[str, Any], *, compact: bool = False) -> None:
    if compact:
        caption = value.get("comprehensive_t2i_caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("Compact upsampler output has an empty comprehensive_t2i_caption")
        return
    missing = sorted(REQUIRED_T2I_KEYS - value.keys())
    if missing:
        raise ValueError(f"Upsampled prompt is missing required keys: {missing}")
    if not isinstance(value["subjects"], list):
        raise ValueError("Upsampled prompt field 'subjects' must be a list")
    if not isinstance(value["text_and_signage_elements"], list):
        raise ValueError("Upsampled prompt field 'text_and_signage_elements' must be a list")
    caption = value["comprehensive_t2i_caption"]
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("Upsampled prompt has an empty comprehensive_t2i_caption")


def render_compact_upsampling_prompt(*, prompt: str, resolution: str, aspect_ratio: str) -> str:
    """Request only the dense caption consumed by the downstream T2I generator."""
    return f"""You are a prompt upsampler for a text-to-image model.
Expand the input into one concrete, visually executable caption of 80-140 words.
Preserve the subject and action, and add coherent composition, lighting, camera,
material, color, background, and mood details. Do not add conflicting objects.
Return exactly one valid JSON object with exactly one key and no prose or fences:
{{"comprehensive_t2i_caption": "your expanded caption"}}

Input: {prompt.strip()}
Output parameters: resolution {resolution}, aspect_ratio {aspect_ratio}.
"""


def render_reasoner_upsampling_prompt(
    *, repo_root: Path, prompt: str, resolution: str, aspect_ratio: str
) -> tuple[str, dict[str, int]]:
    try:
        size = RESOLUTION_RATIO_DICT[resolution][aspect_ratio]
    except KeyError as exc:
        raise ValueError(f"Unsupported Edge resolution/aspect ratio: {resolution}/{aspect_ratio}") from exc

    template_dir = repo_root / "cosmos_framework/inference/prompting_templates/reasoner"
    prompt_template = Template((template_dir / "reasoner_t2i_prompt.txt").read_text(encoding="utf-8"))
    json_template = (template_dir / "reasoner_t2i_json_schema.json").read_text(encoding="utf-8").rstrip()
    description = f"{prompt.strip()}\n\nOutput parameters: resolution {resolution}, aspect_ratio {aspect_ratio}."
    rendered = prompt_template.substitute(nl_description=description, json_template=json_template)
    rendered = (
        rendered.replace("{aspect_ratio}", aspect_ratio)
        .replace("{resolution_w}", str(size["W"]))
        .replace("{resolution_h}", str(size["H"]))
    )
    unresolved = ("$nl_description", "$json_template", "{resolution_w}", "{resolution_h}", "{aspect_ratio}")
    if any(token in rendered for token in unresolved):
        raise ValueError("Official Reasoner T2I template still contains unresolved placeholders")
    return rendered, size


def inference_command(*, input_path: Path, output_dir: Path, checkpoint_path: str, seed: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "profiling_tools.runners.run_profiled",
        "--parallelism-preset=latency",
        "--no-guardrails",
        "-i",
        str(input_path),
        "-o",
        str(output_dir),
        "--checkpoint-path",
        checkpoint_path,
        f"--seed={seed}",
    ]


def run_stage(
    command: list[str],
    *,
    profile_path: Path | None,
    repo_root: Path,
    green_components: tuple[str, ...] = (),
    green_sm_count: int | None = None,
) -> None:
    env = os.environ.copy()
    # Cosmos setup documentation recommends clearing inherited host CUDA paths.
    env["LD_LIBRARY_PATH"] = ""
    if profile_path is not None:
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        env["COSMOS_COMPONENT_PROFILE_OUTPUT"] = str(profile_path)
    if green_sm_count is not None:
        if green_sm_count <= 0:
            raise ValueError("Green Context SM count must be positive")
        if not green_components:
            raise ValueError("Green Context components must be specified with an SM count")
        env["COSMOS_GREEN_COMPONENTS"] = ",".join(green_components)
        env["COSMOS_GREEN_SM_COUNT"] = str(green_sm_count)
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="One natural-language T2I prompt.")
    source.add_argument("--prompt-file", type=Path, help="Text file containing one prompt.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", default="Cosmos3-Edge")
    parser.add_argument("--base-input", type=Path, default=Path("inputs/omni/t2i.json"))
    parser.add_argument("--resolution", choices=sorted(RESOLUTION_RATIO_DICT), default="480")
    parser.add_argument("--aspect-ratio", default="1,1")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--compact-upsampling",
        action="store_true",
        help="Generate only an 80-140 word comprehensive T2I caption.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--upsampler-sm-count", type=int)
    parser.add_argument("--generator-sm-count", type=int)
    parser.add_argument("--no-component-profile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = args.output_dir.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new run directory: {output_root}")
    input_dir = output_root / "inputs"
    stage_dir = output_root / "stages"
    profile_dir = output_root / "profiles"
    input_dir.mkdir(parents=True, exist_ok=True)

    prompt = args.prompt
    if args.prompt_file is not None:
        prompts = [line.strip() for line in args.prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(prompts) != 1:
            raise ValueError(f"--prompt-file must contain exactly one non-empty line, found {len(prompts)}")
        prompt = prompts[0]
    if not prompt:
        raise ValueError("The input prompt is empty")

    size = RESOLUTION_RATIO_DICT[args.resolution][args.aspect_ratio]
    if args.compact_upsampling:
        rendered = render_compact_upsampling_prompt(
            prompt=prompt, resolution=args.resolution, aspect_ratio=args.aspect_ratio
        )
    else:
        rendered, size = render_reasoner_upsampling_prompt(
            repo_root=repo_root,
            prompt=prompt,
            resolution=args.resolution,
            aspect_ratio=args.aspect_ratio,
        )
    max_new_tokens = args.max_new_tokens or (768 if args.compact_upsampling else 4096)
    request_path = input_dir / "upsample_reasoner.json"
    write_json(
        request_path,
        {
            "model_mode": "reasoner",
            "name": "upsample_reasoner",
            "prompt": rendered,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        },
    )
    write_json(
        output_root / "request.json",
        {
            "prompt": prompt,
            "compact_upsampling": args.compact_upsampling,
            "max_new_tokens": max_new_tokens,
        },
    )

    upsampler_output = stage_dir / "01_upsampler"
    run_stage(
        inference_command(
            input_path=request_path,
            output_dir=upsampler_output,
            checkpoint_path=args.checkpoint_path,
            seed=args.seed,
        ),
        profile_path=None if args.no_component_profile else profile_dir / "upsampler.json",
        repo_root=repo_root,
        green_components=("reasoner_e2e", "reasoner_prefill", "reasoner_decode"),
        green_sm_count=args.upsampler_sm_count,
    )

    reasoner_text_path = upsampler_output / "upsample_reasoner/reasoner_text.txt"
    prompt_json = extract_json_object(reasoner_text_path.read_text(encoding="utf-8"))
    prompt_json["resolution"] = {"H": size["H"], "W": size["W"]}
    prompt_json["aspect_ratio"] = args.aspect_ratio
    validate_t2i_prompt(prompt_json, compact=args.compact_upsampling)
    write_json(output_root / "upsampled_prompt.json", prompt_json)

    base_input = args.base_input if args.base_input.is_absolute() else repo_root / args.base_input
    t2i_request = read_json(base_input)
    t2i_request.update(
        {
            "model_mode": "text2image",
            "name": "t2i_upsampled",
            "prompt": json.dumps(prompt_json, ensure_ascii=True, separators=(",", ":")),
            "resolution": args.resolution,
            "aspect_ratio": args.aspect_ratio,
        }
    )
    t2i_request["extra"] = {"prompt": prompt}
    t2i_input_path = input_dir / "t2i_upsampled.json"
    write_json(t2i_input_path, t2i_request)

    generator_output = stage_dir / "02_generator"
    run_stage(
        inference_command(
            input_path=t2i_input_path,
            output_dir=generator_output,
            checkpoint_path=args.checkpoint_path,
            seed=args.seed,
        ),
        profile_path=None if args.no_component_profile else profile_dir / "generator.json",
        repo_root=repo_root,
        green_components=("condition_prepare", "diffusion_sampling", "vae_decode"),
        green_sm_count=args.generator_sm_count,
    )

    image_path = generator_output / "t2i_upsampled/vision.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(f"T2I completed without the expected image: {image_path}")
    write_json(
        output_root / "summary.json",
        {
            "status": "completed",
            "prompt": prompt,
            "upsampled_prompt": str(output_root / "upsampled_prompt.json"),
            "image": str(image_path),
        },
    )
    print(f"Completed: {image_path}", flush=True)


if __name__ == "__main__":
    main()
