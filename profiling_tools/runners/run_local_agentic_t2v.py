"""Run the minimal local Cosmos prompt-upsampling -> T2V workflow.

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

REQUIRED_T2V_KEYS = {
    "subjects",
    "subject_details",
    "background_setting",
    "lighting",
    "text_and_signage_elements",
    "resolution",
    "aspect_ratio",
    "temporal_caption",
}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def print_timing_summary(profile_dir: Path) -> None:
    """Print measured GPU time by stage/component after a successful run."""
    stages = (("Upsampler", "upsampler.json"), ("Critic", "critic.json"), ("Generator", "generator.json"))
    grand_total_ms = 0.0
    print("\nTiming summary (measured GPU time):", flush=True)
    for stage_name, filename in stages:
        path = profile_dir / filename
        if not path.is_file():
            continue
        totals: dict[str, float] = {}
        for record in read_json(path).get("records", []):
            if record.get("phase") not in (None, "measured") or record.get("gpu_ms") is None:
                continue
            component = str(record["component"])
            totals[component] = totals.get(component, 0.0) + float(record["gpu_ms"])
        if not totals:
            continue
        print(f"  {stage_name}:", flush=True)
        for component, milliseconds in totals.items():
            print(f"    {component:20s} {milliseconds / 1000:9.3f} s", flush=True)
        stage_total = totals.get("reasoner_e2e", sum(totals.values()))
        if "reasoner_e2e" not in totals:
            print(f"    {'stage_total':20s} {stage_total / 1000:9.3f} s", flush=True)
        grand_total_ms += stage_total
    print(f"  {'profiled_pipeline_total':24s} {grand_total_ms / 1000:9.3f} s", flush=True)


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


def extract_json_string_field(text: str, field: str) -> str:
    """Extract one completed JSON string even if the surrounding object is truncated."""
    decoder = json.JSONDecoder()
    marker = json.dumps(field)
    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            break
        colon = text.find(":", index + len(marker))
        if colon < 0:
            break
        try:
            value, _ = decoder.raw_decode(text[colon + 1 :].lstrip())
        except json.JSONDecodeError:
            start = index + len(marker)
            continue
        if isinstance(value, str) and value.strip():
            return value
        start = index + len(marker)
    raise ValueError(f"Reasoner output did not contain a completed {field!r} string")



def validate_t2v_prompt(value: dict[str, Any], *, compact: bool = False) -> None:
    if compact:
        caption = value.get("temporal_caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("Compact upsampler output has an empty temporal_caption")
        return
    missing = sorted(REQUIRED_T2V_KEYS - value.keys())
    if missing:
        raise ValueError(f"Upsampled prompt is missing required keys: {missing}")
    if not isinstance(value["subjects"], list):
        raise ValueError("Upsampled prompt field 'subjects' must be a list")
    if not isinstance(value["text_and_signage_elements"], list):
        raise ValueError("Upsampled prompt field 'text_and_signage_elements' must be a list")


def render_compact_upsampling_prompt(
    *,
    prompt: str,
    resolution: str,
    aspect_ratio: str,
    duration: str,
    fps: int,
    extended_planning: bool,
    rich_upsampling: bool,
) -> str:
    """Request a compact or rich temporal plan for downstream T2V."""
    if rich_upsampling:
        planning = (
            "Write temporal_caption first. Then generate planning_notes, exactly six timestamped motion_plan entries, "
            "exactly six timestamped camera_plan entries, exactly eight concise "
            "continuity_constraints. Make every entry specific and non-redundant. "
        )
        schema = (
            '{"temporal_caption": "timestamped caption", '
            '"planning_notes": "detailed scene and physics plan", '
            '"motion_plan": [{"time": "0:00", "description": "motion beat"}], '
            '"camera_plan": [{"time": "0:00", "description": "camera beat"}], '
            '"continuity_constraints": ["constraint"]}'
        )
    elif extended_planning:
        planning = (
            "First write a 600-700 word planning_notes string covering shot-by-shot motion, "
            "physical consistency, subject state, camera path, lighting continuity, composition, "
            "materials, background dynamics, and failure cases. "
        )
        schema = '{"planning_notes": "detailed plan", "temporal_caption": "timestamped caption"}'
    else:
        planning = ""
        schema = '{"temporal_caption": "timestamped caption"}'
    return f"""You are a prompt upsampler for a text-to-video model.
{planning}Write a concrete 100-180 word temporal_caption using timestamped beats from
0:00 through 0:05. Describe subject motion, physical state changes, camera motion,
composition, lighting, materials, background, and atmosphere. Preserve the requested
subject/action and avoid conflicting objects. Return exactly one valid JSON object
with exactly the requested keys and no prose or fences:
{schema}

Input: {prompt.strip()}
Output parameters: duration {duration}, fps {fps}, resolution {resolution},
aspect_ratio {aspect_ratio}.
"""


def render_reasoner_upsampling_prompt(
    *, repo_root: Path, prompt: str, resolution: str, aspect_ratio: str
) -> tuple[str, dict[str, int]]:
    try:
        size = RESOLUTION_RATIO_DICT[resolution][aspect_ratio]
    except KeyError as exc:
        raise ValueError(f"Unsupported Edge resolution/aspect ratio: {resolution}/{aspect_ratio}") from exc

    template_dir = repo_root / "cosmos_framework/inference/prompting_templates/reasoner"
    prompt_template = Template((template_dir / "reasoner_t2v_prompt.txt").read_text(encoding="utf-8"))
    json_template = (template_dir / "reasoner_t2v_json_schema.json").read_text(encoding="utf-8").rstrip()
    description = f"{prompt.strip()}\n\nOutput parameters: resolution {resolution}, aspect_ratio {aspect_ratio}."
    rendered = prompt_template.substitute(nl_description=description, json_template=json_template)
    rendered = (
        rendered.replace("{aspect_ratio}", aspect_ratio)
        .replace("{resolution_w}", str(size["W"]))
        .replace("{resolution_h}", str(size["H"]))
    )
    unresolved = ("$nl_description", "$json_template", "{resolution_w}", "{resolution_h}", "{aspect_ratio}")
    if any(token in rendered for token in unresolved):
        raise ValueError("Official Reasoner T2V template still contains unresolved placeholders")
    return rendered, size


def render_critic_prompt(temporal_caption: str) -> str:
    """Ask the local Reasoner for a concise, minimally edited refinement."""
    return f"""You are the final caption refiner in a text-to-video pipeline.
Make only necessary edits to improve timestamp consistency, physical plausibility,
subject continuity, and camera coherence. Preserve the candidate's content and length.
Do not perform or describe an exhaustive checklist. Return exactly one valid JSON
object with one key and no fences or surrounding prose:
{{"temporal_caption": "the minimally revised timestamped caption"}}

Candidate temporal caption:
{temporal_caption}
"""


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
    source.add_argument("--prompt", help="One natural-language T2V prompt.")
    source.add_argument("--prompt-file", type=Path, help="Text file containing one prompt.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", default="Cosmos3-Edge")
    parser.add_argument("--base-input", type=Path, default=Path("inputs/omni/t2v.json"))
    parser.add_argument("--resolution", choices=sorted(RESOLUTION_RATIO_DICT), default="480")
    parser.add_argument("--aspect-ratio", default="16,9")
    parser.add_argument("--duration", default="5s")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--extended-planning",
        action="store_true",
        help="Generate detailed planning notes, then strip them before T2V generation.",
    )
    parser.add_argument("--rich-upsampling", action="store_true")
    parser.add_argument("--with-critic", action="store_true")
    parser.add_argument("--critic-max-new-tokens", type=int, default=400)
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
    rendered = render_compact_upsampling_prompt(
        prompt=prompt,
        resolution=args.resolution,
        aspect_ratio=args.aspect_ratio,
        duration=args.duration,
        fps=args.fps,
        extended_planning=args.extended_planning,
        rich_upsampling=args.rich_upsampling,
    )
    max_new_tokens = args.max_new_tokens or (1350 if args.rich_upsampling else 1536 if args.extended_planning else 1024)
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
            "compact_upsampling": True,
            "extended_planning": args.extended_planning,
            "rich_upsampling": args.rich_upsampling,
            "with_critic": args.with_critic,
            "critic_max_new_tokens": args.critic_max_new_tokens,
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
    reasoner_text_path = upsampler_output / "upsample_reasoner/reasoner_text.txt"
    reasoner_text = reasoner_text_path.read_text(encoding="utf-8")
    try:
        prompt_json = extract_json_object(reasoner_text)
    except ValueError:
        prompt_json = {}
    if not isinstance(prompt_json.get("temporal_caption"), str):
        prompt_json = {"temporal_caption": extract_json_string_field(reasoner_text, "temporal_caption")}
    upsampling_details = {
        key: prompt_json.pop(key)
        for key in ("planning_notes", "motion_plan", "camera_plan", "continuity_constraints")
        if key in prompt_json
    }
    if args.with_critic:
        critic_request_path = input_dir / "critic_reasoner.json"
        write_json(
            critic_request_path,
            {
                "model_mode": "reasoner",
                "name": "critic_reasoner",
                "prompt": render_critic_prompt(prompt_json["temporal_caption"]),
                "max_new_tokens": args.critic_max_new_tokens,
                "do_sample": False,
            },
        )
        critic_output = stage_dir / "01b_critic"
        run_stage(
            inference_command(
                input_path=critic_request_path,
                output_dir=critic_output,
                checkpoint_path=args.checkpoint_path,
                seed=args.seed,
            ),
            profile_path=None if args.no_component_profile else profile_dir / "critic.json",
            repo_root=repo_root,
            green_components=("reasoner_e2e", "reasoner_prefill", "reasoner_decode"),
            green_sm_count=args.upsampler_sm_count,
        )
        critic_text = critic_output / "critic_reasoner/reasoner_text.txt"
        try:
            revised_prompt_json = extract_json_object(critic_text.read_text(encoding="utf-8"))
            validate_t2v_prompt(revised_prompt_json, compact=True)
            prompt_json = revised_prompt_json
        except ValueError:
            print(
                "Critic exhausted its decode budget before producing valid JSON; "
                "keeping the upsampler temporal caption.",
                flush=True,
            )
    prompt_json["resolution"] = {"H": size["H"], "W": size["W"]}
    prompt_json["aspect_ratio"] = args.aspect_ratio
    prompt_json["duration"] = args.duration
    prompt_json["fps"] = args.fps
    validate_t2v_prompt(prompt_json, compact=True)
    write_json(output_root / "upsampled_prompt.json", prompt_json)
    if upsampling_details:
        write_json(output_root / "upsampling_details.json", upsampling_details)

    base_input = args.base_input if args.base_input.is_absolute() else repo_root / args.base_input
    t2v_request = read_json(base_input)
    t2v_request.update(
        {
            "model_mode": "text2video",
            "name": "t2v_upsampled",
            "prompt": json.dumps(prompt_json, ensure_ascii=True, separators=(",", ":")),
            "resolution": args.resolution,
            "aspect_ratio": args.aspect_ratio,
            "fps": args.fps,
        }
    )
    t2v_request["extra"] = {"prompt": prompt}
    t2v_input_path = input_dir / "t2v_upsampled.json"
    write_json(t2v_input_path, t2v_request)

    generator_output = stage_dir / "02_generator"
    run_stage(
        inference_command(
            input_path=t2v_input_path,
            output_dir=generator_output,
            checkpoint_path=args.checkpoint_path,
            seed=args.seed,
        ),
        profile_path=None if args.no_component_profile else profile_dir / "generator.json",
        repo_root=repo_root,
        green_components=("condition_prepare", "diffusion_sampling", "vae_decode"),
        green_sm_count=args.generator_sm_count,
    )

    video_path = next(generator_output.rglob("*.mp4"), None)
    if video_path is None or not video_path.is_file():
        raise FileNotFoundError(f"T2V completed without the expected video: {video_path}")
    write_json(
        output_root / "summary.json",
        {
            "status": "completed",
            "prompt": prompt,
            "upsampled_prompt": str(output_root / "upsampled_prompt.json"),
            "video": str(video_path),
        },
    )
    if not args.no_component_profile:
        print_timing_summary(profile_dir)
    print(f"Completed: {video_path}", flush=True)


if __name__ == "__main__":
    main()
