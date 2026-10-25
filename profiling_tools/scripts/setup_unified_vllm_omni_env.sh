#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
runtime_root=${COSMOS_UNIFIED_RUNTIME_ROOT:-"$repo_root/.runtime/vllm_omni_tf517"}
source_dir="$runtime_root/vllm-omni"
venv_dir="$runtime_root/venv"

omni_commit=${VLLM_OMNI_COMMIT:-03c334818305d3b1666da01cd0f67d48a4f3e7a2}
omni_version=${VLLM_OMNI_PRETEND_VERSION:-0.31.0rc2.dev288}
vllm_version=${VLLM_VERSION:-0.31.0}
transformers_version=${TRANSFORMERS_VERSION:-5.17.0}
python_version=${PYTHON_VERSION:-3.13}
torch_backend=${UV_TORCH_BACKEND:-cu130}

export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
export UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-20}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}

mkdir -p "$runtime_root"
if [[ ! -d "$source_dir/.git" ]]; then
  git clone https://github.com/vllm-project/vllm-omni.git "$source_dir"
fi
git -C "$source_dir" fetch origin "$omni_commit"
git -C "$source_dir" checkout --detach "$omni_commit"

# The pinned Omni commit was tested only with Transformers <5.15. Edge support
# is upstream in Transformers 5.17. Gate -1 validated this source at runtime
# with 5.17, so keep the experiment isolated and change only package metadata.
python3 - "$source_dir/requirements/common.txt" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "transformers >= 5.13.0, < 5.15"
new = "transformers >= 5.17.0, < 5.18"
if old in text:
    path.write_text(text.replace(old, new, 1))
elif new not in text:
    raise SystemExit(f"unexpected Transformers constraint in {path}")
PY

uv venv --python "$python_version" "$venv_dir"
SETUPTOOLS_SCM_PRETEND_VERSION="$omni_version" \
  uv pip install \
    --python "$venv_dir/bin/python" \
    --torch-backend "$torch_backend" \
    --editable "$source_dir" \
    "vllm==$vllm_version" \
    "transformers==$transformers_version"

"$venv_dir/bin/python" - <<'PY'
import importlib.metadata as metadata

from transformers import AutoConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
    Cosmos3OmniDiffusersPipeline,
)

for package in ("vllm", "vllm-omni", "transformers", "torch"):
    print(f"{package}={metadata.version(package)}")
config = AutoConfig.from_pretrained("nvidia/Cosmos3-Edge", local_files_only=True)
assert type(config).__name__ == "Cosmos3EdgeConfig"
assert "Cosmos3EdgeForConditionalGeneration" in ModelRegistry.get_supported_archs()
print(f"edge_config={type(config).__module__}.{type(config).__name__}")
print(f"omni_pipeline={Cosmos3OmniDiffusersPipeline.__module__}.{Cosmos3OmniDiffusersPipeline.__name__}")
print("unified environment check: OK")
PY

echo "Unified environment: $venv_dir"
