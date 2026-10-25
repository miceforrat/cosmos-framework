"""Compare deterministic eager and CUDA Graph Reasoner validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    args = parser.parse_args()
    eager = json.loads(args.eager.read_text())
    graph = json.loads(args.graph.read_text())
    eager_summary = eager["summary"]
    graph_summary = graph["summary"]
    same_hash = eager_summary["output_sha256"] == graph_summary["output_sha256"]
    print("output_hash_equal:", same_hash)
    print("eager_sha256:", eager_summary["output_sha256"])
    print("graph_sha256:", graph_summary["output_sha256"])
    for key in ("median_ttft_seconds", "median_e2e_seconds", "median_decode_tokens_per_second"):
        eager_value = eager_summary[key]
        graph_value = graph_summary[key]
        if eager_value is None or graph_value is None:
            print(f"{key}: eager={eager_value} graph={graph_value}")
        else:
            print(f"{key}: eager={eager_value:.6f} graph={graph_value:.6f}")
    if not same_hash:
        raise SystemExit("Eager and graph output text hashes differ")


if __name__ == "__main__":
    main()
