from __future__ import annotations


import os
import sys
import threading
import hashlib
import json
import time
import functools
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro

from cosmos_framework.runtime.regions import (
    install_expected_cuda_stream_ptr,
    install_region_factory,
    reset_expected_cuda_stream_ptr,
    reset_region_factory,
)
from profiling_tools.core.green_context import GreenContextPair

REPO_ROOT = Path(__file__).resolve().parents[2]

@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

@contextmanager
def nsys_capture():
    enabled = os.environ.get("COSMOS_NSYS_CAPTURE") == "1"
    if enabled:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        yield
    finally:
        if enabled:
            torch.cuda.cudart().cudaProfilerStop()

class ColocationResources:
    def __init__(
        self,
        generator_sm_count: int,
        device: int = 0,
    ) -> None:
        # Split the generator partition first so its requested capacity is the
        # hard constraint. Reasoner decode receives the driver's exact remainder.
        self.owner = GreenContextPair(
            generator_sm_count,
            device,
            ignore_sm_coscheduling=False,
        )
        self.generator_stream = torch.cuda.ExternalStream(
            self.owner.first_stream_ptr,
            device=device,
        )
        self.reasoner_stream = torch.cuda.ExternalStream(
            self.owner.second_stream_ptr,
            device=device,
        )
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return

        self.reasoner_stream.synchronize()
        self.generator_stream.synchronize()
        self.owner.close()
        self._closed = True
        
class DecodeRegionController:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        expected_regions: int,
        barrier: threading.Barrier | None = None,
        barrier_timeout_s: float = 300.0,
        nvtx_range_name: str = "reasoner_decode_region",
        measure_host_region: bool = False,
    ) -> None:
        self.stream = stream
        self.expected_regions = expected_regions
        self.barrier = barrier
        self.barrier_timeout_s = barrier_timeout_s
        self.nvtx_range_name = nvtx_range_name
        self.measure_host_region = measure_host_region
        self.enter_count = 0
        self.start_event: torch.cuda.Event | None = None
        self.end_event: torch.cuda.Event | None = None
        self.host_start_s: float | None = None
        self.host_end_s: float | None = None

    def factory(self, name: str):
        if name != "reasoner_decode":
            return nullcontext()
        return self.decode_region()

    @contextmanager
    def decode_region(self):
        self.enter_count += 1
        if self.enter_count > self.expected_regions:
            raise RuntimeError(
                f"reasoner_decode entered {self.enter_count} times; "
                f"expected {self.expected_regions}"
            )

        is_first = self.enter_count == 1
        original_stream = torch.cuda.current_stream()
        prefill_done = torch.cuda.Event()
        current_end_event = torch.cuda.Event(enable_timing=True)
        if is_first:
            self.start_event = torch.cuda.Event(enable_timing=True)

        prefill_done.record(original_stream)
        self.stream.wait_event(prefill_done)

        with torch.cuda.stream(self.stream):
            if is_first and self.barrier is not None:
                self.barrier.wait(timeout=self.barrier_timeout_s)
            if is_first:
                self.host_start_s = time.perf_counter()
                assert self.start_event is not None
                self.start_event.record(self.stream)
            expected_stream_token = install_expected_cuda_stream_ptr(
                int(self.stream.cuda_stream)
            )
            try:
                with nvtx_range(self.nvtx_range_name):
                    yield
            finally:
                reset_expected_cuda_stream_ptr(expected_stream_token)
                current_end_event.record(self.stream)
                self.end_event = current_end_event
                if self.measure_host_region:
                    current_end_event.synchronize()
                    self.host_end_s = time.perf_counter()

        original_stream.wait_event(current_end_event)


class GeneratorStageController:
    """Gate and time one generator component without changing its implementation."""

    def __init__(self, target: str, barrier: threading.Barrier | None) -> None:
        self.target = target
        self.barrier = barrier
        self.enter_count = 0
        self.start_event: torch.cuda.Event | None = None
        self.end_event: torch.cuda.Event | None = None

    def wrap(self, component: str, function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if component != self.target:
                return function(*args, **kwargs)
            self.enter_count += 1
            if self.enter_count != 1:
                raise RuntimeError(f"Generator stage {component!r} entered more than once")
            if self.barrier is not None:
                self.barrier.wait(timeout=300.0)
            stream = torch.cuda.current_stream()
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record(stream)
            with nvtx_range(f"colocated_generator_{component}"):
                result = function(*args, **kwargs)
            self.end_event.record(stream)
            return result
        return wrapped

    def elapsed_seconds(self) -> float:
        if self.enter_count != 1 or self.start_event is None or self.end_event is None:
            raise RuntimeError(f"Generator stage {self.target!r} was not observed exactly once")
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event) / 1000.0


@contextmanager
def install_generator_stage_wrappers(controller: GeneratorStageController):
    from cosmos_framework.model.generator.diffusion.samplers.edm import EDMSampler
    from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    if controller.target == "diffusion_sampling":
        targets = [(cls, "__call__") for cls in (UniPCSampler, EDMSampler, FixedStepSampler)]
    elif controller.target == "vae_decode":
        targets = [(OmniMoTModel, "decode")]
    else:
        raise ValueError("Generator region must be full, diffusion_sampling, or vae_decode")

    originals = []
    try:
        for owner, attribute in targets:
            original = getattr(owner, attribute)
            originals.append((owner, attribute, original))
            setattr(owner, attribute, controller.wrap(controller.target, original))
        yield
    finally:
        for owner, attribute, original in reversed(originals):
            setattr(owner, attribute, original)

@dataclass
class LoadedWorkloads:
    pipe: Any
    reasoner_sample_args: Any
    generator_sample_args: Any

    def fresh_reasoner_batch(
        self,
    ) -> tuple[list[Any], dict[str, Any]]:
        batches = self.pipe.create_batches(
            [self.reasoner_sample_args]
        )
        return next(batches)

    def fresh_generator_batch(
        self,
    ) -> tuple[list[Any], dict[str, Any]]:
        batches = self.pipe.create_batches(
            [self.generator_sample_args]
        )
        return next(batches)

def parse_official_args(
    *,
    reasoner_input: Path,
    generator_input: Path,
    output_dir: Path,
    checkpoint_path: str,
):
    from cosmos_framework.inference.common.args import (
        tyro_cli,
    )
    from cosmos_framework.scripts.inference import (
        InferenceArgs,
    )

    argv = [
        "cosmos_framework.scripts.inference",
        "--parallelism-preset=latency",
        "--no-guardrails",
        "-i",
        str(reasoner_input),
        str(generator_input),
        "-o",
        str(output_dir),
        "--checkpoint-path",
        checkpoint_path,
        "--seed=0",
    ]

    previous_argv = sys.argv
    sys.argv = argv      
    try:
        return tyro_cli(
            InferenceArgs,
            description="Reasoner decode and T2V colocation",
            config=(
                tyro.conf.OmitArgPrefixes,
                tyro.conf.CascadeSubcommandArgs,
                tyro.conf.OmitSubcommandPrefixes,
            ),
        )
    finally:
        sys.argv = previous_argv

def load_workloads(
    *,
    reasoner_input: Path,
    generator_input: Path,
    output_dir: Path,
    checkpoint_path: str,
) -> LoadedWorkloads:
    from cosmos_framework.inference.common.init import (
        init_output_dir,
    )

    inference_args = parse_official_args(
        reasoner_input=reasoner_input,
        generator_input=generator_input,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
    )

    setup_args = inference_args.setup.build_setup()
    init_output_dir(setup_args.output_dir)

    sample_overrides_list = (
        setup_args
        .get_sample_overrides_cls()
        .from_files(
            inference_args.input_files,
            overrides=setup_args.sample_overrides,
        )
    )

    if len(sample_overrides_list) != 2:
        raise RuntimeError(
            "Expected one Reasoner and one T2V request, "
            f"got {len(sample_overrides_list)}"
        )

    for overrides in sample_overrides_list:
        if not overrides.name:
            raise ValueError(
                "Every colocation input must have a name"
            )

        overrides.output_dir = (
            setup_args.output_dir / overrides.name
        )
        overrides.download(
            overrides.output_dir / "inputs"
        )

    # 因为输入同时含 Reasoner 和 T2V，
    # 这里不会进入 reasoner-only 加载模式。
    pipe = setup_args.get_inference_cls().create(
        setup_args
    )

    if pipe.vae_decode_stream is not None:
        raise RuntimeError(
            "Colocation requires pipe.vae_decode_stream "
            "to be None"
        )

    sample_args_list = [
        overrides.build_sample(
            model_config=pipe.model_config
        )
        for overrides in sample_overrides_list
    ]

    reasoner_args = [
        args
        for args in sample_args_list
        if args.model_mode.is_reasoner
    ]
    generator_args = [
        args
        for args in sample_args_list
        if not args.model_mode.is_reasoner
    ]

    if len(reasoner_args) != 1:
        raise RuntimeError(
            f"Expected one Reasoner request, got "
            f"{len(reasoner_args)}"
        )

    if len(generator_args) != 1:
        raise RuntimeError(
            f"Expected one generator request, got "
            f"{len(generator_args)}"
        )

    return LoadedWorkloads(
        pipe=pipe,
        reasoner_sample_args=reasoner_args[0],
        generator_sample_args=generator_args[0],
    )

def run_colocated_once(
    *,
    workloads: LoadedWorkloads,
    resources: ColocationResources,
    label: str,
    reasoner_repeats: int,
    generator_repeats: int,
    generator_region: str,
) -> tuple[float, float, float, float | None]:
    nvtx_label = label.lower().replace(" ", "_").replace("/", "_")
    barrier = threading.Barrier(3 if generator_region == "full" else 2)
    stage_controller = None if generator_region == "full" else GeneratorStageController(generator_region, barrier)
    controller = DecodeRegionController(
        stream=resources.reasoner_stream,
        expected_regions=reasoner_repeats,
        barrier=barrier,
        nvtx_range_name=f"reasoner_decode_{nvtx_label}",
    )

    def reasoner_worker():
        torch.cuda.set_device(resources.owner.device)
        token = install_region_factory(controller.factory)
        try:
            output = None
            for request_idx in range(reasoner_repeats):
                sample_args, data_batch = workloads.fresh_reasoner_batch()
                with nvtx_range(f"{nvtx_label}_reasoner_request_{request_idx + 1}"):
                    output = workloads.pipe.generate_batch(
                        sample_args,
                        data_batch,
                        save_outputs=False,
                    )
            return output
        except BaseException:
            barrier.abort()
            raise
        finally:
            reset_region_factory(token)

    def generator_worker():
        torch.cuda.set_device(resources.owner.device)
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(resources.generator_stream):
                if generator_region == "full":
                    barrier.wait(timeout=300.0)
                start.record(resources.generator_stream)
                output = None
                for request_idx in range(generator_repeats):
                    sample_args, data_batch = workloads.fresh_generator_batch()
                    with nvtx_range(
                        f"{nvtx_label}_generator_request_{request_idx + 1}"
                    ):
                        output = workloads.pipe.generate_batch(
                            sample_args,
                            data_batch,
                            save_outputs=False,
                        )
                end.record(resources.generator_stream)
            return output, start, end
        except BaseException:
            barrier.abort()
            raise
    wrapper_context = nullcontext() if stage_controller is None else install_generator_stage_wrappers(stage_controller)
    with wrapper_context, ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="decode-t2v-colocation",
    ) as executor:
        reasoner_future = executor.submit(
            reasoner_worker
        )
        generator_future = executor.submit(
            generator_worker
        )

        if generator_region == "full":
            try:
                barrier.wait(timeout=300.0)
            except BaseException as barrier_error:
                barrier.abort()
                raise RuntimeError("Start barrier broke") from barrier_error

        wall_start = time.perf_counter()


        reasoner_output = reasoner_future.result()
        generator_output, generator_start, generator_end = generator_future.result()

    del reasoner_output, generator_output
    if controller.enter_count != reasoner_repeats:
        raise RuntimeError(
            f"Expected {reasoner_repeats} reasoner_decode regions, "
            f"got {controller.enter_count}"
        )
    if controller.start_event is None or controller.end_event is None:
        raise RuntimeError("Reasoner timing events were not recorded")

    controller.end_event.synchronize()
    generator_end.synchronize()
    reasoner_ms = controller.start_event.elapsed_time(controller.end_event)
    generator_ms = generator_start.elapsed_time(generator_end)
    stage_seconds = stage_controller.elapsed_seconds() if stage_controller is not None else None
    wall_ms = (time.perf_counter() - wall_start) * 1000.0

    print(
        f"\nColocation result [{label}]:",
        f"\n  reasoner_decode={reasoner_ms / 1000:.3f} s",
        f"\n  generator={generator_ms / 1000:.3f} s",
        f"\n  gpu_makespan={max(reasoner_ms, generator_ms) / 1000:.3f} s",
        f"\n  host_wall={wall_ms / 1000:.3f} s",
        *([f"\n  {generator_region}={stage_seconds:.3f} s"] if stage_seconds is not None else []),
        flush=True,
    )
    return reasoner_ms / 1000.0, generator_ms / 1000.0, wall_ms / 1000.0, stage_seconds


def run_reasoner_alone(
    *,
    workloads: LoadedWorkloads,
    resources: ColocationResources,
    reasoner_repeats: int,
    nvtx_range_name: str = "reasoner_decode_alone",
    stream: torch.cuda.Stream | None = None,
    display_name: str = "Partitioned-alone decode",
    emit_result: bool = True,
) -> tuple[float, float, float, float]:
    decode_stream = resources.reasoner_stream if stream is None else stream
    controller = DecodeRegionController(
        stream=decode_stream,
        expected_regions=reasoner_repeats,
        nvtx_range_name=nvtx_range_name,
        measure_host_region=True,
    )
    token = install_region_factory(controller.factory)
    try:
        for _ in range(reasoner_repeats):
            sample_args, data_batch = workloads.fresh_reasoner_batch()
            output = workloads.pipe.generate_batch(
                sample_args,
                data_batch,
                save_outputs=False,
            )
            del output
    finally:
        reset_region_factory(token)

    if controller.enter_count != reasoner_repeats:
        raise RuntimeError(
            f"Expected {reasoner_repeats} reasoner_decode regions, "
            f"got {controller.enter_count}"
        )
    if controller.start_event is None or controller.end_event is None:
        raise RuntimeError("Reasoner-alone timing events were not recorded")
    if controller.host_start_s is None or controller.host_end_s is None:
        raise RuntimeError("Reasoner-alone host region timestamps were not recorded")
    controller.end_event.synchronize()
    host_seconds = controller.host_end_s - controller.host_start_s
    seconds = controller.start_event.elapsed_time(controller.end_event) / 1000.0
    if emit_result:
        print(
            f"\n{display_name} ({reasoner_repeats} requests): "
            f"gpu={seconds:.3f} s host_region={host_seconds:.3f} s",
            flush=True,
        )
    return seconds, host_seconds, controller.host_start_s, controller.host_end_s


def run_generator_alone(
    *,
    workloads: LoadedWorkloads,
    resources: ColocationResources,
    generator_region: str,
    generator_repeats: int,
    stream: torch.cuda.Stream | None = None,
    display_name: str = "Partitioned-alone generator",
    emit_result: bool = True,
) -> tuple[float, float | None, float, float, float]:
    generator_stream = resources.generator_stream if stream is None else stream
    stage_controller = None if generator_region == "full" else GeneratorStageController(generator_region, None)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wrapper_context = nullcontext() if stage_controller is None else install_generator_stage_wrappers(stage_controller)
    with wrapper_context, torch.cuda.stream(generator_stream):
        host_start_s = time.perf_counter()
        start.record(generator_stream)
        output = None
        for _ in range(generator_repeats):
            sample_args, data_batch = workloads.fresh_generator_batch()
            output = workloads.pipe.generate_batch(
                sample_args,
                data_batch,
                save_outputs=False,
            )
        end.record(generator_stream)
    end.synchronize()
    host_end_s = time.perf_counter()
    host_seconds = host_end_s - host_start_s
    del output
    seconds = start.elapsed_time(end) / 1000.0
    if emit_result:
        print(
            f"{display_name} ({generator_repeats} requests): "
            f"gpu={seconds:.3f} s host_region={host_seconds:.3f} s",
            flush=True,
        )
    stage_seconds = stage_controller.elapsed_seconds() if stage_controller is not None else None
    if stage_seconds is not None:
        print(f"Partitioned-alone {generator_region}: {stage_seconds:.3f} s", flush=True)
    return seconds, stage_seconds, host_seconds, host_start_s, host_end_s


def main() -> None:
    requested_generator_sm_count = int(os.environ.get("COSMOS_GENERATOR_SM_COUNT", "136"))
    reasoner_repeats = int(os.environ.get("COSMOS_REASONER_REPEATS", "1"))
    generator_repeats = int(os.environ.get("COSMOS_GENERATOR_REPEATS", "1"))
    warmup_runs = int(os.environ.get("COSMOS_WARMUP_RUNS", "1"))
    generator_region = os.environ.get("COSMOS_COLOCATE_GENERATOR_REGION", "full")
    generator_workload = os.environ.get("COSMOS_GENERATOR_WORKLOAD", "t2v")
    full_sm_decode_only = os.environ.get("COSMOS_FULL_SM_DECODE_ONLY") == "1"
    full_sm_generator_only = os.environ.get("COSMOS_FULL_SM_GENERATOR_ONLY") == "1"
    full_sm_serial_only = os.environ.get("COSMOS_FULL_SM_SERIAL_BASELINE_ONLY") == "1"
    gc_decode_only = os.environ.get("COSMOS_GC_DECODE_ONLY") == "1"
    use_cuda_graph = os.environ.get("COSMOS_REASONER_CUDA_GRAPH") == "1"
    use_static_kv_eager = os.environ.get("COSMOS_REASONER_STATIC_KV_EAGER") == "1"
    if use_cuda_graph and use_static_kv_eager:
        raise ValueError("COSMOS_REASONER_CUDA_GRAPH and COSMOS_REASONER_STATIC_KV_EAGER are mutually exclusive")
    decode_implementation = "graph" if use_cuda_graph else "static_eager" if use_static_kv_eager else "eager"
    if sum((full_sm_decode_only, full_sm_generator_only, full_sm_serial_only, gc_decode_only)) > 1:
        raise ValueError("COSMOS_FULL_SM_DECODE_ONLY, COSMOS_FULL_SM_GENERATOR_ONLY, COSMOS_FULL_SM_SERIAL_BASELINE_ONLY, and COSMOS_GC_DECODE_ONLY are mutually exclusive")
    if generator_region not in {"full", "diffusion_sampling", "vae_decode"}:
        raise ValueError("COSMOS_COLOCATE_GENERATOR_REGION must be full, diffusion_sampling, or vae_decode")
    if reasoner_repeats <= 0 or generator_repeats <= 0 or warmup_runs < 0:
        raise ValueError(
            "COSMOS_REASONER_REPEATS and COSMOS_GENERATOR_REPEATS must be > 0; "
            "COSMOS_WARMUP_RUNS must be >= 0"
        )
    if generator_region != "full" and generator_repeats != 1:
        raise ValueError("Multiple generator requests are currently supported only for the full region")
    if generator_workload not in {"t2i", "t2v"}:
        raise ValueError("COSMOS_GENERATOR_WORKLOAD must be t2i or t2v")
    default_reasoner_input = (
        REPO_ROOT / "outputs/local_agentic_t2v/rich02/inputs/upsample_reasoner.json"
    )
    default_generator_inputs = {
        "t2v": REPO_ROOT / "outputs/local_agentic_t2v/rich02/inputs/t2v_upsampled.json",
        "t2i": REPO_ROOT / "outputs/local_agentic_t2i/smoke01/inputs/t2i_upsampled.json",
    }
    reasoner_input = Path(
        os.environ.get("COSMOS_REASONER_INPUT", str(default_reasoner_input))
    )
    generator_input = Path(
        os.environ.get("COSMOS_GENERATOR_INPUT", str(default_generator_inputs[generator_workload]))
    )
    output_dir = (
        REPO_ROOT
        / "outputs/decode_t2v_colocation"
        / (
            f"{generator_workload}_generator_sm{requested_generator_sm_count:03d}_"
            f"{generator_region}_repeat{generator_repeats:02d}"
        )
    )
    workloads = load_workloads(
        reasoner_input=reasoner_input,
        generator_input=generator_input,
        output_dir=output_dir,
        checkpoint_path="Cosmos3-Edge",
    )

    reasoner_input_data = json.loads(reasoner_input.read_text(encoding="utf-8"))
    prompt_token_ids = workloads.pipe.model.tokenize_text(
        reasoner_input_data["prompt"], device=None
    ).reshape(-1).tolist()
    prompt_token_ids_json = json.dumps(prompt_token_ids, separators=(",", ":"))
    generator_sample = workloads.generator_sample_args.model_dump(mode="json")
    generator_prompt = generator_sample["prompt"]
    resolution_payload = json.loads(generator_prompt).get("resolution", {})
    canonical_generator_workload = {
        "prompt": generator_prompt,
        "negative_prompt": generator_sample["negative_prompt"],
        "height": int(resolution_payload["H"]),
        "width": int(resolution_payload["W"]),
        "num_frames": int(generator_sample["num_frames"]),
        "num_inference_steps": int(generator_sample["num_steps"]),
        "guidance_scale": float(generator_sample["guidance"]),
        "flow_shift": float(generator_sample["shift"]),
        "seed": int(generator_sample["seed"]),
    }
    canonical_generator_json = json.dumps(
        canonical_generator_workload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    workload_identity = {
        "reasoner": {
            "raw_prompt_sha256": hashlib.sha256(
                reasoner_input_data["prompt"].encode("utf-8")
            ).hexdigest(),
            "prompt_token_count": len(prompt_token_ids),
            "prompt_token_ids_sha256": hashlib.sha256(
                prompt_token_ids_json.encode("utf-8")
            ).hexdigest(),
            "fixed_completion_tokens": int(reasoner_input_data["max_new_tokens"]),
        },
        "generator": {
            "canonical_workload": canonical_generator_workload,
            "canonical_workload_sha256": hashlib.sha256(
                canonical_generator_json.encode("utf-8")
            ).hexdigest(),
            "prompt_sha256": hashlib.sha256(
                generator_prompt.encode("utf-8")
            ).hexdigest(),
            "negative_prompt_sha256": hashlib.sha256(
                generator_sample["negative_prompt"].encode("utf-8")
            ).hexdigest(),
        },
    }
    workload_identity_path = output_dir / "matched_workload_identity.json"
    workload_identity_path.parent.mkdir(parents=True, exist_ok=True)
    workload_identity_path.write_text(
        json.dumps(workload_identity, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Matched workload identity:",
        f"reasoner_prompt_tokens={len(prompt_token_ids)}",
        f"reasoner_token_hash={workload_identity['reasoner']['prompt_token_ids_sha256']}",
        f"generator_hash={workload_identity['generator']['canonical_workload_sha256']}",
        f"manifest={workload_identity_path}",
        flush=True,
    )

    resources = ColocationResources(
        generator_sm_count=requested_generator_sm_count,
        device=0,
    )

    print(
        "GC partition:",
        f"requested_generator={resources.owner.requested_first_sm_count}",
        f"actual_generator={resources.owner.first_sm_count}",
        f"actual_reasoner_remainder={resources.owner.second_sm_count}",
        f"total={resources.owner.total_sm_count}",
        f"reasoner_repeats={reasoner_repeats}",
        f"generator_workload={generator_workload}",
        f"generator_repeats={generator_repeats}",
        f"decode_implementation={decode_implementation}",
        flush=True,
    )
    try:
        if full_sm_decode_only:
            # A regular non-default CUDA stream is unrestricted by either GC
            # partition. CUDA Graph capture cannot use the legacy default
            # stream, so use this primary-context side stream explicitly.
            full_sm_stream = torch.cuda.Stream(device=resources.owner.device)
            print(f"Priming full-SM reasoner decode ({decode_implementation})...", flush=True)
            run_reasoner_alone(
                workloads=workloads,
                resources=resources,
                reasoner_repeats=1,
                nvtx_range_name=f"reasoner_decode_full_sm_{decode_implementation}_prime",
                stream=full_sm_stream,
                display_name=f"Full-SM {decode_implementation} prime",
            )
            with nvtx_range("NSYS_REASONER_FULL_SM_ALONE"):
                full_sm_s, full_sm_host_s, _, _ = run_reasoner_alone(
                    workloads=workloads,
                    resources=resources,
                    reasoner_repeats=reasoner_repeats,
                    nvtx_range_name="reasoner_decode_full_sm_alone",
                    stream=full_sm_stream,
                    display_name="Full-SM decode",
                )
            print(f"\nFull-SM decode baseline: gpu={full_sm_s:.3f} s host_wall={full_sm_host_s:.3f} s", flush=True)
            return

        if full_sm_generator_only:
            full_sm_stream = torch.cuda.Stream(device=resources.owner.device)
            print(f"Priming full-SM {generator_workload} generator...", flush=True)
            run_generator_alone(
                workloads=workloads,
                resources=resources,
                generator_region=generator_region,
                generator_repeats=1,
                stream=full_sm_stream,
                display_name="Full-SM generator prime",
            )
            with nvtx_range("NSYS_GENERATOR_FULL_SM_ALONE"):
                full_sm_s, full_sm_stage_s, full_sm_host_s, _, _ = run_generator_alone(
                    workloads=workloads,
                    resources=resources,
                    generator_region=generator_region,
                    generator_repeats=generator_repeats,
                    stream=full_sm_stream,
                    display_name="Full-SM generator",
                )
            print(
                "\nFull-SM generator baseline:",
                f"workload={generator_workload}",
                f"repeats={generator_repeats}",
                f"gpu={full_sm_s:.3f} s",
                f"host_wall={full_sm_host_s:.3f} s",
                flush=True,
            )
            if full_sm_stage_s is not None:
                print(f"Full-SM {generator_region}: {full_sm_stage_s:.3f} s", flush=True)
            return

        if full_sm_serial_only:
            full_sm_stream = torch.cuda.Stream(device=resources.owner.device)
            print(
                f"Priming matched full-SM serial baseline ({decode_implementation} + "
                f"{generator_workload} x{generator_repeats})...",
                flush=True,
            )
            run_reasoner_alone(
                workloads=workloads,
                resources=resources,
                reasoner_repeats=1,
                nvtx_range_name="reasoner_decode_full_sm_serial_prime",
                stream=full_sm_stream,
                display_name="Full-SM serial decode prime",
            )
            run_generator_alone(
                workloads=workloads,
                resources=resources,
                generator_region=generator_region,
                generator_repeats=1,
                stream=full_sm_stream,
                display_name="Full-SM serial generator prime",
            )
            with nvtx_range("NSYS_FULL_SM_SERIAL_BASELINE"):
                decode_s, decode_host_s, serial_start_s, _ = run_reasoner_alone(
                    workloads=workloads,
                    resources=resources,
                    reasoner_repeats=reasoner_repeats,
                    nvtx_range_name="reasoner_decode_full_sm_serial",
                    stream=full_sm_stream,
                    emit_result=False,
                )
                generator_s, generator_stage_s, generator_host_s, _, serial_end_s = run_generator_alone(
                    workloads=workloads,
                    resources=resources,
                    generator_region=generator_region,
                    generator_repeats=generator_repeats,
                    stream=full_sm_stream,
                    emit_result=False,
                )
            serial_host_wall_s = serial_end_s - serial_start_s
            print(
                "\nFull-SM serial baseline:",
                f"\n  decode_implementation={decode_implementation}",
                f"\n  generator_workload={generator_workload}",
                f"\n  generator_repeats={generator_repeats}",
                f"\n  decode_gpu={decode_s:.3f} s",
                f"\n  decode_host_region={decode_host_s:.3f} s",
                f"\n  generator_gpu={generator_s:.3f} s",
                f"\n  generator_host_region={generator_host_s:.3f} s",
                f"\n  transition_and_python_overhead={serial_host_wall_s - decode_host_s - generator_host_s:.6f} s",
                f"\n  full_sm_serial_host_wall={serial_host_wall_s:.3f} s",
                flush=True,
            )
            if generator_stage_s is not None:
                print(f"  {generator_region}={generator_stage_s:.3f} s", flush=True)
            return

        if gc_decode_only:
            print(f"Priming reasoner decode on the GC stream ({decode_implementation})...", flush=True)
            run_reasoner_alone(
                workloads=workloads,
                resources=resources,
                reasoner_repeats=1,
                nvtx_range_name="reasoner_decode_gc_graph_prime",
                display_name="GC graph prime",
            )
            with nvtx_range("NSYS_REASONER_GC_ALONE"):
                gc_decode_s, _, _, _ = run_reasoner_alone(
                    workloads=workloads,
                    resources=resources,
                    reasoner_repeats=reasoner_repeats,
                    nvtx_range_name="reasoner_decode_gc_alone",
                    display_name="GC decode",
                )
            print(
                "\nGC decode baseline:",
                f"requested_generator={resources.owner.requested_first_sm_count}",
                f"actual_generator={resources.owner.first_sm_count}",
                f"actual_reasoner={resources.owner.second_sm_count}",
                f"seconds={gc_decode_s:.6f}",
                flush=True,
            )
            return

        if os.environ.get("COSMOS_REASONER_CUDA_GRAPH_PERSISTENT") == "1":
            print("Priming persistent reasoner CUDA Graph without T2V concurrency...", flush=True)
            run_reasoner_alone(
                workloads=workloads,
                resources=resources,
                reasoner_repeats=1,
                nvtx_range_name="reasoner_decode_graph_prime",
            )

        for warmup_idx in range(warmup_runs):
            run_colocated_once(
                workloads=workloads,
                resources=resources,
                label=f"warmup {warmup_idx + 1}/{warmup_runs}",
                reasoner_repeats=reasoner_repeats,
                generator_repeats=generator_repeats,
                generator_region=generator_region,
            )

        with nsys_capture():
            serial_end_to_end_start = time.perf_counter()
            with nvtx_range("NSYS_REASONER_ALONE"):
                alone_reasoner_s, alone_reasoner_host_s, _, _ = run_reasoner_alone(
                    workloads=workloads,
                    resources=resources,
                    reasoner_repeats=reasoner_repeats,
                )
            with nvtx_range("NSYS_GENERATOR_ALONE"):
                alone_generator_s, alone_stage_s, alone_generator_host_s, _, _ = run_generator_alone(
                    workloads=workloads,
                    resources=resources,
                    generator_region=generator_region,
                    generator_repeats=generator_repeats,
                )
            # Keep the old end-to-end diagnostic, but do not use it for
            # throughput: it starts before Reasoner prefill, whereas the full
            # colocation wall starts after prefill at the decode/T2V barrier.
            torch.cuda.synchronize()
            serial_end_to_end_wall_s = time.perf_counter() - serial_end_to_end_start
            with nvtx_range("NSYS_BALANCED_MEASURED"):
                coloc_reasoner_s, coloc_generator_s, coloc_wall_s, coloc_stage_s = run_colocated_once(
                    workloads=workloads,
                    resources=resources,
                    label="balanced measured warm",
                    reasoner_repeats=reasoner_repeats,
                    generator_repeats=generator_repeats,
                    generator_region=generator_region,
                )

        serial_gpu_sum_s = alone_reasoner_s + alone_generator_s
        # Strictly matched to the full-colocation wall boundary: only the
        # Reasoner decode region plus the full T2V request are included.
        serial_host_wall_s = alone_reasoner_host_s + alone_generator_host_s
        makespan_s = (
            max(coloc_reasoner_s, coloc_generator_s)
            if generator_region == "full"
            else coloc_wall_s
        )
        print(
            "\nBalanced interference summary:",
            f"\n  decode_alone={alone_reasoner_s:.3f} s",
            f"\n  decode_colocated={coloc_reasoner_s:.3f} s",
            f"\n  decode_slowdown={coloc_reasoner_s / alone_reasoner_s:.3f}x",
            f"\n  generator_workload={generator_workload}",
            f"\n  generator_repeats={generator_repeats}",
            f"\n  generator_alone={alone_generator_s:.3f} s",
            f"\n  generator_colocated={coloc_generator_s:.3f} s",
            f"\n  generator_slowdown={coloc_generator_s / alone_generator_s:.3f}x",
            f"\n  serial_gpu_sum={serial_gpu_sum_s:.3f} s",
            f"\n  partition_serial_host_wall={serial_host_wall_s:.3f} s",
            f"\n  serial_end_to_end_wall_diagnostic={serial_end_to_end_wall_s:.3f} s",
            f"\n  colocated_makespan={makespan_s:.3f} s",
            f"\n  throughput_gain_gpu_sum={serial_gpu_sum_s / makespan_s:.3f}x",
            f"\n  partition_overlap_efficiency_host_wall={serial_host_wall_s / coloc_wall_s:.3f}x",
            flush=True,
        )
        if alone_stage_s is not None and coloc_stage_s is not None:
            print(
                f"  {generator_region}_alone={alone_stage_s:.3f} s",
                f"\n  {generator_region}_colocated={coloc_stage_s:.3f} s",
                f"\n  {generator_region}_slowdown={coloc_stage_s / alone_stage_s:.3f}x",
                flush=True,
            )
    finally:
        resources.close()


if __name__ == "__main__":
    main()
