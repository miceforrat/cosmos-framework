"""NVTX, CUDA-event timing, and optional Green Context component execution."""

from __future__ import annotations

import atexit
import contextvars
import functools
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable

import torch

from profiling_tools.core.green_context import GreenContextStream

_OUTPUT = os.environ.get("COSMOS_COMPONENT_PROFILE_OUTPUT")
_GREEN_COMPONENT = os.environ.get("COSMOS_GREEN_COMPONENT")
_GREEN_COMPONENTS = {
    value.strip() for value in os.environ.get("COSMOS_GREEN_COMPONENTS", "").split(",") if value.strip()
}
if _GREEN_COMPONENT:
    _GREEN_COMPONENTS.add(_GREEN_COMPONENT)
_GREEN_SM_COUNT = int(os.environ.get("COSMOS_GREEN_SM_COUNT", "0"))
_WARMUP_ITERS = int(os.environ.get("COSMOS_COMPONENT_WARMUP_ITERS", "0"))
_PASS_INDEX: contextvars.ContextVar[int | None] = contextvars.ContextVar("cosmos_component_profile_pass", default=None)
_next_pass_index = 0
_records: list[dict[str, object]] = []
_owner: GreenContextStream | None = None
_stream: torch.cuda.ExternalStream | None = None


def _get_green_stream() -> tuple[GreenContextStream, torch.cuda.ExternalStream]:
    global _owner, _stream
    if _owner is None:
        if _GREEN_SM_COUNT <= 0:
            raise ValueError("COSMOS_GREEN_SM_COUNT must be positive")
        device = torch.cuda.current_device()
        _owner = GreenContextStream(_GREEN_SM_COUNT, device)
        _stream = torch.cuda.ExternalStream(_owner.stream_ptr, device=device)
    assert _stream is not None
    return _owner, _stream


def wrap_component(component: str, function: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a real Cosmos function without changing its inputs or return value."""

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        use_green = component in _GREEN_COMPONENTS
        owner: GreenContextStream | None = None
        stream: torch.cuda.Stream | None = None
        if torch.cuda.is_available():
            if use_green:
                owner, stream = _get_green_stream()
                torch.cuda.synchronize()
            else:
                stream = torch.cuda.current_stream()
        start = torch.cuda.Event(enable_timing=True) if stream is not None else None
        end = torch.cuda.Event(enable_timing=True) if stream is not None else None
        cpu_start = time.perf_counter()
        with ExitStack() as stack:
            if torch.cuda.is_available():
                stack.enter_context(torch.cuda.nvtx.range(f"cosmos.component.{component}"))
            if use_green:
                assert stream is not None
                stack.enter_context(torch.cuda.stream(stream))
            if start is not None:
                start.record(stream)
            result = function(*args, **kwargs)
            if end is not None:
                end.record(stream)
        if end is not None:
            end.synchronize()
            gpu_ms: float | None = start.elapsed_time(end) if start is not None else None
        else:
            gpu_ms = None
        if use_green:
            torch.cuda.synchronize()
        pass_index = _PASS_INDEX.get()
        if pass_index is None:
            phase = "unclassified"
            iteration_index = None
        elif _WARMUP_ITERS == 0:
            phase = "measured"
            iteration_index = pass_index
        elif pass_index == 0:
            phase = "cold"
            iteration_index = 0
        elif pass_index < _WARMUP_ITERS:
            phase = "warmup"
            iteration_index = pass_index
        else:
            phase = "measured"
            iteration_index = pass_index - _WARMUP_ITERS
        _records.append(
            {
                "component": component,
                "cpu_ms": (time.perf_counter() - cpu_start) * 1000.0,
                "gpu_ms": gpu_ms,
                "requested_sm_count": owner.requested_sm_count if owner else None,
                "actual_sm_count": owner.actual_sm_count if owner else None,
                "pass_index": pass_index,
                "phase": phase,
                "iteration_index": iteration_index,
            }
        )
        return result

    return wrapped


def wrap_reasoner_forward(function: Callable[..., Any]) -> Callable[..., Any]:
    """Classify the shared AR forward as prompt prefill or one-token decode."""

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 1:
            input_ids = args[1]
        inputs_embeds = kwargs.get("inputs_embeds")
        is_prefill = inputs_embeds is not None or (
            isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2 and input_ids.shape[1] > 1
        )
        component = "reasoner_prefill" if is_prefill else "reasoner_decode"
        return wrap_component(component, function)(*args, **kwargs)

    return wrapped


def install() -> None:
    """Install aligned wrappers around the shared Cosmos generation components."""
    from cosmos_framework.data.generator.action.utils.action_processing import ActionProcessor
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.model.generator.diffusion.samplers.edm import EDMSampler
    from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler
    from cosmos_framework.model.generator.mot.unified_mot import (
        Nemotron3DenseVLTextModel,
        Qwen3VLMoeTextModel,
        Qwen3VLTextModel,
    )
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.vision_siglip2 import (
        NemotronSiglip2VisionEncoder,
    )
    from cosmos_framework.model.generator.reasoner.qwen3_vl import utils as qwen3_vl_utils

    original_generate_batch = OmniInference.generate_batch

    @functools.wraps(original_generate_batch)
    def generate_batch_with_profile_pass(*args: Any, **kwargs: Any) -> Any:
        global _next_pass_index
        pass_index = _next_pass_index
        _next_pass_index += 1
        token = _PASS_INDEX.set(pass_index)
        try:
            return original_generate_batch(*args, **kwargs)
        finally:
            _PASS_INDEX.reset(token)

    OmniInference.generate_batch = generate_batch_with_profile_pass

    # Shared names deliberately stay identical across T2I, T2V, and Action.
    OmniMoTModel._prepare_inference_data = wrap_component("condition_prepare", OmniMoTModel._prepare_inference_data)
    OmniMoTModel.decode = wrap_component("vae_decode", OmniMoTModel.decode)
    ActionProcessor.postprocess_action = staticmethod(
        wrap_component("action_postprocess", ActionProcessor.postprocess_action)
    )
    for sampler_type in (UniPCSampler, EDMSampler, FixedStepSampler):
        sampler_type.__call__ = wrap_component("diffusion_sampling", sampler_type.__call__)

    # If native prompt upsampling is enabled, it becomes an aligned text stage.
    OmniMoTModel.upsample_captions = wrap_component("text_reasoner", OmniMoTModel.upsample_captions)

    for model_type in (Qwen3VLTextModel, Qwen3VLMoeTextModel, Nemotron3DenseVLTextModel):
        model_type.reasoner_forward = wrap_reasoner_forward(model_type.reasoner_forward)

    NemotronSiglip2VisionEncoder.get_image_features = wrap_component(
        "vision_encoder", NemotronSiglip2VisionEncoder.get_image_features
    )
    qwen3_vl_utils.get_image_features = wrap_component("vision_encoder", qwen3_vl_utils.get_image_features)
    OmniMoTModel.generate_reasoner_text = wrap_component("reasoner_e2e", OmniMoTModel.generate_reasoner_text)


def _shutdown() -> None:
    if _OUTPUT and _records:
        output = Path(_OUTPUT)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"records": _records}, indent=2) + "\n")
    if _owner is not None:
        _owner.close()


atexit.register(_shutdown)
