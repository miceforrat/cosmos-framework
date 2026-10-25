#!/usr/bin/env python3
"""Compare matched native and vLLM decode x diffusion-sampling experiments."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NATIVE_FIELDS = (
    "decode_alone",
    "decode_colocated",
    "decode_slowdown",
    "diffusion_sampling_alone",
    "diffusion_sampling_colocated",
    "diffusion_sampling_slowdown",
)


def parse_native_log(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    result: dict[str, float] = {}
    for field in NATIVE_FIELDS:
        match = re.findall(rf"^\s*{re.escape(field)}=([0-9.]+)(?: s|x)?\s*$", text, re.MULTILINE)
        if not match:
            raise ValueError(f"{path}: missing {field}")
        result[field] = float(match[-1])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--vllm-json", type=Path, required=True)
    args = parser.parse_args()

    native = parse_native_log(args.native_log)
    manifest = json.loads(args.native_manifest.read_text(encoding="utf-8"))
    vllm = json.loads(args.vllm_json.read_text(encoding="utf-8"))
    identity = vllm["matched_workload_identity"]
    summary = vllm["summary"]

    checks = {
        "reasoner_prompt_token_ids_sha256": (
            manifest["reasoner"]["prompt_token_ids_sha256"]
            == identity["reasoner_prompt_token_ids_sha256"]
        ),
        "fixed_completion_tokens": (
            manifest["reasoner"]["fixed_completion_tokens"]
            == identity["reasoner_fixed_completion_tokens_per_request"]
        ),
        "generator_canonical_workload_sha256": (
            manifest["generator"]["canonical_workload_sha256"]
            == identity["generator_canonical_workload_sha256"]
        ),
        "region_is_diffusion_sampling": (
            summary["measured_generator_stage"] == "diffusion_sampling"
        ),
        "fixed_token_runs_valid": bool(summary["reasoner_token_workload_valid"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    print("Matched workload checks:")
    for name, passed in checks.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    if failed:
        raise SystemExit(f"Workload mismatch: {', '.join(failed)}")

    rows = [
        (
            "native Cosmos graph",
            native["decode_alone"],
            native["decode_colocated"],
            native["decode_slowdown"],
            native["diffusion_sampling_alone"],
            native["diffusion_sampling_colocated"],
            native["diffusion_sampling_slowdown"],
        ),
        (
            "vLLM graph + Omni",
            summary["reasoner_alone_seconds"],
            summary["reasoner_concurrent_seconds"],
            summary["reasoner_slowdown"],
            summary["diffusion_sampling_alone_seconds"],
            summary["diffusion_sampling_concurrent_seconds"],
            summary["diffusion_sampling_slowdown"],
        ),
    ]
    print("\nimplementation              decode alone  colocated  slowdown   sampling alone  colocated  slowdown")
    for name, da, dc, ds, sa, sc, ss in rows:
        print(f"{name:27s} {da:11.3f}  {dc:9.3f}  {ds:7.3f}x  {sa:14.3f}  {sc:9.3f}  {ss:7.3f}x")
    print("\nNote: throughput gain is intentionally omitted for component-only overlap; compare the two directional slowdowns.")


if __name__ == "__main__":
    main()
