"""Profiling-only stream injection at serving-runtime execution boundaries."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable
import importlib

import torch


def get_vllm_gpu_model_runner(reasoner):
    """Return the in-process V1 GPUModelRunner owned by an offline LLM."""
    core_client = reasoner.llm_engine.engine_core
    engine_core = core_client.engine_core
    executor = engine_core.model_executor
    wrapper = executor.driver_worker
    worker = wrapper.worker
    return worker.model_runner


def bind_vllm_runner_stream(reasoner, stream: torch.cuda.Stream) -> dict[str, int | None]:
    """Synchronously rebind vLLM eager compute and output copy to ``stream``."""
    stream.synchronize()
    runner = get_vllm_gpu_model_runner(reasoner)
    previous_main = runner.__dict__.get("main_stream")
    previous_copy = runner.output_copy_stream
    runner.__dict__["main_stream"] = stream
    runner.output_copy_stream = stream
    return {
        "previous_main_stream": (
            None if previous_main is None else int(previous_main.cuda_stream)
        ),
        "previous_copy_stream": int(previous_copy.cuda_stream),
        "new_main_stream": int(stream.cuda_stream),
        "new_copy_stream": int(stream.cuda_stream),
    }


@dataclass
class RuntimeStreamPatch:
    owner: type
    method_name: str
    original: Callable[..., Any]
    replacement: Callable[..., Any]
    calls: int = 0
    stream: torch.cuda.Stream | None = None

    def switch_stream(self, stream: torch.cuda.Stream) -> dict[str, int]:
        """Synchronously move a mutable runtime patch to another stream."""
        if self.stream is None:
            raise RuntimeError("This runtime patch has no mutable stream")
        self.stream.synchronize()
        previous = int(self.stream.cuda_stream)
        self.stream = stream
        return {"previous_stream": previous, "new_stream": int(stream.cuda_stream)}

    def restore(self) -> None:
        current = getattr(self.owner, self.method_name)
        if current is self.replacement:
            setattr(self.owner, self.method_name, self.original)
        elif current is not self.original:
            raise RuntimeError(
                f"{self.owner.__name__}.{self.method_name} changed after patch installation"
            )


@dataclass
class GraphCaptureStreamPatch:
    replacements: list[RuntimeStreamPatch]
    calls: int = 0
    manager_capture_stream_calls: int = 0
    compiled_capture_stream_calls: int = 0

    def restore(self) -> None:
        for replacement in reversed(self.replacements):
            replacement.restore()


@dataclass
class OmniRuntimeStreamPatch:
    execute_patch: RuntimeStreamPatch
    pack_patch: RuntimeStreamPatch
    postprocess_patch: RuntimeStreamPatch
    stream: torch.cuda.Stream
    calls: int = 0
    pack_calls: int = 0
    postprocess_calls: int = 0

    def switch_stream(self, stream: torch.cuda.Stream) -> dict[str, int]:
        self.stream.synchronize()
        previous = int(self.stream.cuda_stream)
        self.stream = stream
        self.execute_patch.stream = stream
        self.pack_patch.stream = stream
        self.postprocess_patch.stream = stream
        return {"previous_stream": previous, "new_stream": int(stream.cuda_stream)}

    def restore(self) -> None:
        self.postprocess_patch.restore()
        self.pack_patch.restore()
        self.execute_patch.restore()


def install_omni_diffusion_worker_stream_patch(
    stream: torch.cuda.Stream,
) -> OmniRuntimeStreamPatch:
    """Run the current Gate 3 single-request Omni worker path on ``stream``."""
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

    method_name = "execute_model"
    original = getattr(DiffusionWorker, method_name)
    if getattr(original, "_cosmos_gc_stream_patch", False):
        raise RuntimeError("Omni DiffusionWorker stream patch is already installed")

    patch: OmniRuntimeStreamPatch

    def execute_model_on_external_stream(self, *args, **kwargs):
        patch.calls += 1
        active_stream = patch.stream
        with torch.cuda.stream(active_stream), torch.cuda.nvtx.range(
            "gate3.omni_worker.generator_gc"
        ):
            return original(self, *args, **kwargs)

    execute_model_on_external_stream._cosmos_gc_stream_patch = True
    execute_patch = RuntimeStreamPatch(
        owner=DiffusionWorker,
        method_name=method_name,
        original=original,
        replacement=execute_model_on_external_stream,
        stream=stream,
    )
    diffusion_worker_module = importlib.import_module(
        "vllm_omni.diffusion.worker.diffusion_worker"
    )
    pack_name = "pack_diffusion_output_shm"
    original_pack = getattr(diffusion_worker_module, pack_name)

    def pack_output_on_external_stream(output, d2h_stream=None, **kwargs):
        patch.pack_calls += 1
        active_stream = patch.stream
        # WorkerProc's persistent ordinary side stream has already waited on
        # the producer event. Drain it before moving postprocessing and D2H to
        # the current Green Context stream.
        if d2h_stream is not None and d2h_stream.cuda_stream != active_stream.cuda_stream:
            d2h_stream.synchronize()
        with torch.cuda.stream(active_stream), torch.cuda.nvtx.range(
            "gate3.omni_worker.output_gc"
        ):
            return original_pack(output, d2h_stream=active_stream, **kwargs)

    pack_output_on_external_stream._cosmos_gc_stream_patch = True
    pack_patch = RuntimeStreamPatch(
        owner=diffusion_worker_module,
        method_name=pack_name,
        original=original_pack,
        replacement=pack_output_on_external_stream,
        stream=stream,
    )

    # In the single-process engine path, Cosmos3 video normalization is not
    # performed by WorkerProc's SHM packer. DiffusionEngine invokes the model
    # postprocessor after execute_model() returns; without an explicit binding
    # its per-frame mul/clamp/add kernels land on PyTorch's ordinary stream.
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    postprocess_name = "postprocess_output"
    original_postprocess = getattr(DiffusionEngine, postprocess_name)
    if getattr(original_postprocess, "_cosmos_gc_stream_patch", False):
        raise RuntimeError("Omni DiffusionEngine postprocess stream patch is already installed")

    def postprocess_output_on_external_stream(self, *args, **kwargs):
        patch.postprocess_calls += 1
        active_stream = patch.stream
        with torch.cuda.stream(active_stream), torch.cuda.nvtx.range(
            "gate3.omni_engine.output_postprocess_gc"
        ):
            return original_postprocess(self, *args, **kwargs)

    postprocess_output_on_external_stream._cosmos_gc_stream_patch = True
    postprocess_patch = RuntimeStreamPatch(
        owner=DiffusionEngine,
        method_name=postprocess_name,
        original=original_postprocess,
        replacement=postprocess_output_on_external_stream,
        stream=stream,
    )
    patch = OmniRuntimeStreamPatch(
        execute_patch=execute_patch,
        pack_patch=pack_patch,
        postprocess_patch=postprocess_patch,
        stream=stream,
    )
    setattr(DiffusionWorker, method_name, execute_model_on_external_stream)
    setattr(diffusion_worker_module, pack_name, pack_output_on_external_stream)
    setattr(DiffusionEngine, postprocess_name, postprocess_output_on_external_stream)
    return patch


def install_vllm_graph_capture_stream_patch(
    stream: torch.cuda.Stream,
) -> GraphCaptureStreamPatch:
    """Make vLLM graph capture use a supplied Green Context stream.

    GPUModelRunner imports ``graph_capture`` by value, so both the source
    module and that bound module global must be replaced. Group communicator
    capture hooks remain unchanged.
    """
    from vllm.distributed import parallel_state
    from vllm.v1.worker import gpu_model_runner
    from vllm.v1.worker.gpu import cudagraph_utils
    from vllm.compilation import cuda_graph as compilation_cuda_graph

    patch: GraphCaptureStreamPatch

    @contextmanager
    def graph_capture_on_external_stream(device):
        del device
        patch.calls += 1
        context = parallel_state.GraphCaptureContext(stream)
        with (
            parallel_state.get_tp_group().graph_capture(context),
            parallel_state.get_pp_group().graph_capture(context),
            parallel_state.get_dp_group().graph_capture(context),
        ):
            yield context

    replacements = []
    for owner in (parallel_state, gpu_model_runner, cudagraph_utils):
        original = getattr(owner, "graph_capture")
        setattr(owner, "graph_capture", graph_capture_on_external_stream)
        replacements.append(
            RuntimeStreamPatch(
                owner=owner,
                method_name="graph_capture",
                original=original,
                replacement=graph_capture_on_external_stream,
            )
        )

    capture_stream_owner = cudagraph_utils.ModelCudaGraphManager
    capture_stream_name = "_capture_stream"
    original_capture_stream = getattr(capture_stream_owner, capture_stream_name)

    def capture_stream_on_external_green_context(self, desc):
        del self, desc
        patch.manager_capture_stream_calls += 1
        return stream

    setattr(
        capture_stream_owner,
        capture_stream_name,
        capture_stream_on_external_green_context,
    )
    replacements.append(
        RuntimeStreamPatch(
            owner=capture_stream_owner,
            method_name=capture_stream_name,
            original=original_capture_stream,
            replacement=capture_stream_on_external_green_context,
        )
    )

    compiled_stream_name = "current_stream"
    original_compiled_stream = getattr(compilation_cuda_graph, compiled_stream_name)

    def compiled_capture_stream_on_external_green_context():
        patch.compiled_capture_stream_calls += 1
        return stream

    setattr(
        compilation_cuda_graph,
        compiled_stream_name,
        compiled_capture_stream_on_external_green_context,
    )
    replacements.append(
        RuntimeStreamPatch(
            owner=compilation_cuda_graph,
            method_name=compiled_stream_name,
            original=original_compiled_stream,
            replacement=compiled_capture_stream_on_external_green_context,
        )
    )
    patch = GraphCaptureStreamPatch(replacements=replacements)
    return patch
