"""Single-rank compatibility layer for co-resident vLLM and Omni engines.

vLLM and vLLM-Omni both store model-parallel coordinators in vLLM process-global
variables.  For TP=DP=PP=1 the diffusion engine can safely borrow vLLM's tensor
parallel coordinator while owning only its diffusion-specific groups.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch


def _install_single_rank_graph_capture_compat() -> None:
    """Add vLLM's capture-stream contract to Omni's rank-1 coordinators.

    Omni's diffusion GroupCoordinator predates the ``graph_capture`` method
    required by vLLM 0.31.  There is no collective/communicator capture work
    at world size one; only the shared capture stream must be entered.
    """
    from vllm.distributed import parallel_state as vllm_ps

    @contextmanager
    def graph_capture(group, graph_capture_context=None):
        if group.world_size != 1:
            raise RuntimeError("Omni graph-capture compatibility is restricted to world_size=1")
        if graph_capture_context is None:
            graph_capture_context = vllm_ps.GraphCaptureContext(torch.cuda.Stream())
        stream = graph_capture_context.stream
        current = torch.cuda.current_stream()
        if current != stream:
            stream.wait_stream(current)
        with torch.cuda.stream(stream):
            yield graph_capture_context

    seen: set[int] = set()
    for group in (vllm_ps._TP, vllm_ps._DP, vllm_ps._PP):
        if group is None or id(group) in seen:
            continue
        seen.add(id(group))
        if group.world_size != 1:
            raise RuntimeError("shared graph capture requires TP=DP=PP=1")
        if not hasattr(group, "graph_capture"):
            group.graph_capture = MethodType(graph_capture, group)


def install_reasoner_context_groups_from_omni() -> None:
    """Fill vLLM's missing rank-1 PCP/DCP/ETP slots from Omni's TP group."""
    from vllm.distributed import parallel_state as vllm_ps

    if vllm_ps._TP is None or vllm_ps._PP is None:
        raise RuntimeError("Omni must initialize vLLM TP/PP before the Reasoner")
    _install_single_rank_graph_capture_compat()
    for name in ("_PCP", "_DCP", "_ETP"):
        current = getattr(vllm_ps, name)
        if current is not None and current is not vllm_ps._TP:
            raise RuntimeError(f"cannot replace existing vLLM {name} group")
        setattr(vllm_ps, name, vllm_ps._TP)


def clear_reasoner_context_group_aliases() -> None:
    """Drop borrowed aliases before Omni destroys its owned TP coordinator."""
    from vllm.distributed import parallel_state as vllm_ps

    for name in ("_PCP", "_DCP", "_ETP"):
        if getattr(vllm_ps, name) is vllm_ps._TP:
            setattr(vllm_ps, name, None)


def shutdown_reasoner_preserving_shared_parallel_state(reasoner) -> None:
    """Release one in-process Reasoner without destroying Omni-owned groups.

    ``EngineCore.shutdown`` unconditionally calls its module-bound
    ``cleanup_dist_env_and_memory``, which destroys vLLM's process-global
    TP/DP/PP and default process group.  In this harness those objects existed
    before the Reasoner and are still required by the resident Omni Generator.
    Suppress only that final distributed teardown; worker/model/KV/graph
    shutdown still runs normally.
    """
    import gc

    from vllm.distributed import parallel_state as vllm_ps
    from vllm.v1.engine import core as engine_core_module

    saved_groups = {name: getattr(vllm_ps, name) for name in ("_TP", "_DP", "_PP", "_WORLD")}
    if any(value is None for value in saved_groups.values()):
        raise RuntimeError("Cannot preserve missing vLLM shared parallel state")

    clear_reasoner_context_group_aliases()
    original_cleanup = engine_core_module.cleanup_dist_env_and_memory

    def cleanup_memory_without_distributed_teardown(*args, **kwargs):
        del args, kwargs
        gc.collect()
        torch.cuda.empty_cache()

    engine_core_module.cleanup_dist_env_and_memory = cleanup_memory_without_distributed_teardown
    try:
        reasoner.llm_engine.engine_core.shutdown()
    finally:
        engine_core_module.cleanup_dist_env_and_memory = original_cleanup

    changed = [name for name, value in saved_groups.items() if getattr(vllm_ps, name) is not value]
    if changed:
        raise RuntimeError(f"Reasoner shutdown changed borrowed parallel groups: {changed}")


def install_single_rank_shared_parallel_state() -> None:
    from vllm.distributed import parallel_state as vllm_ps
    from vllm_omni.diffusion.distributed import parallel_state as omni_ps
    from vllm_omni.diffusion.worker import diffusion_worker

    if getattr(diffusion_worker, "_cosmos_shared_parallel_installed", False):
        return

    def initialize_shared(
        data_parallel_size: int | None = None,
        cfg_parallel_size: int = 1,
        sequence_parallel_size: int | None = None,
        ulysses_degree: int = 1,
        ring_degree: int = 1,
        allgather_degree: int = 1,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        fully_shard_degree: int = 1,
        enable_expert_parallel: bool = False,
        use_hsdp: bool = False,
        backend: str | None = None,
    ) -> None:
        sizes = {
            "data_parallel_size": data_parallel_size or 1,
            "cfg_parallel_size": cfg_parallel_size,
            "sequence_parallel_size": sequence_parallel_size or 1,
            "ulysses_degree": ulysses_degree,
            "ring_degree": ring_degree,
            "allgather_degree": allgather_degree,
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "fully_shard_degree": fully_shard_degree,
        }
        if torch.distributed.get_world_size() != 1 or any(value != 1 for value in sizes.values()):
            raise RuntimeError(f"shared parallel compatibility is restricted to world_size=1, got {sizes}")
        if enable_expert_parallel or use_hsdp:
            raise RuntimeError("shared parallel compatibility does not support EP or HSDP")
        if any(group is not None for group in (omni_ps._DP, omni_ps._CFG, omni_ps._SP, omni_ps._PP)):
            raise RuntimeError("Omni diffusion parallel groups are already initialized")
        if any(group is None for group in (vllm_ps._TP, vllm_ps._DP, vllm_ps._PP)):
            raise RuntimeError("Reasoner vLLM groups must exist before shared Omni initialization")

        local_rank = omni_ps.get_world_group().local_rank
        backend_name = backend or torch.distributed.get_backend(omni_ps.get_world_group().device_group)
        singleton = [[0]]
        omni_ps._DP = omni_ps.init_model_parallel_group(singleton, local_rank, backend_name, "data")
        omni_ps._CFG = omni_ps.init_model_parallel_group(
            singleton, local_rank, backend_name, "classifier_free_guidance"
        )
        omni_ps._PP = omni_ps.init_model_parallel_group(singleton, local_rank, backend_name, "pipeline")
        ulysses_pg, ring_pg, allgather_pg = omni_ps.set_seq_parallel_pg(
            sp_ulysses_degree=1,
            sp_ring_degree=1,
            sp_allgather_degree=1,
            rank=omni_ps.get_world_group().rank_in_group,
            world_size=1,
            sp_group_ranks=singleton,
        )
        omni_ps._SP = omni_ps.init_model_parallel_group(
            singleton,
            local_rank,
            backend_name,
            "sequence",
            ulysses_group=ulysses_pg,
            ring_group=ring_pg,
            allgather_group=allgather_pg,
        )
        omni_ps._EXPERT_PARALLEL_GROUP_RANKS = singleton

    def destroy_shared() -> None:
        # These four groups were created above and are owned by the diffusion
        # engine. vLLM's TP/DP/PP/EP/PCP and default process group are borrowed.
        for name in ("_SP", "_CFG", "_PP", "_DP"):
            group = getattr(omni_ps, name)
            if group is not None:
                group.destroy()
                setattr(omni_ps, name, None)
        omni_ps._EXPERT_PARALLEL_GROUP_RANKS = None
        # Omni's world coordinator wraps the already-active torch/vLLM
        # distributed environment. Do not destroy its underlying process group
        # while the Reasoner is alive; dropping the reference is sufficient for
        # this process-lifetime Gate-1 harness.
        omni_ps._WORLD = None

    # DiffusionWorker imported these functions directly, so patch its bound
    # names rather than only changing the defining module.
    diffusion_worker.initialize_model_parallel = initialize_shared
    diffusion_worker.destroy_distributed_env = destroy_shared
    diffusion_worker._cosmos_shared_parallel_installed = True
