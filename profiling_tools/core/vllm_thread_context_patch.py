"""Temporary Gate-3 isolation for vLLM IR-op dispatch priority.

vLLM 0.31 stores the active implementation list directly on each process-wide
``IrOp`` instance.  Nested context managers are correct in one thread, but two
concurrent model threads can overwrite and restore each other's priority.  This
profiling-only patch moves scoped overrides into ``ContextVar`` while retaining
the original process-wide list as the default.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class VllmIrPriorityContextPatch:
    owner: type
    original_dispatch: Callable[..., Any]
    original_get_priority: Callable[..., Any]
    original_set_priority: Callable[..., Any]
    dispatch_calls: int = 0
    override_entries: int = 0

    def restore(self) -> None:
        global _active_patch
        expected = {
            "dispatch": _dispatch_with_context_priority,
            "get_priority": _get_context_priority,
            "set_priority": _set_context_priority,
        }
        originals = {
            "dispatch": self.original_dispatch,
            "get_priority": self.original_get_priority,
            "set_priority": self.original_set_priority,
        }
        for name, replacement in expected.items():
            current = getattr(self.owner, name)
            if current is replacement:
                setattr(self.owner, name, originals[name])
            elif current is not originals[name]:
                raise RuntimeError(f"{self.owner.__name__}.{name} changed after patch installation")
        if _active_patch is self:
            _active_patch = None


_active_patch: VllmIrPriorityContextPatch | None = None


def _priority_context(op) -> ContextVar:
    context = getattr(op, "_cosmos_priority_context", None)
    if context is None:
        context = ContextVar(f"vllm_ir_priority_{op.name}_{id(op)}", default=None)
        setattr(op, "_cosmos_priority_context", context)
    return context


def _active_priority(op):
    override = _priority_context(op).get()
    return op._priority_impls if override is None else override


def _dispatch_with_context_priority(self, *args, **kwargs):
    import torch
    from vllm.ir import op as op_module

    assert _active_patch is not None
    _active_patch.dispatch_calls += 1
    priority_impls = _active_priority(self)
    if not priority_impls:
        if not torch.compiler.is_compiling():
            op_module.logger.warning_once(
                "Priority not set for op %s, using native implementation.", self.name
            )
        return self.impls["native"]

    for impl in priority_impls:
        if not impl.supported:
            raise ValueError(
                f"Implementation {impl.provider} for op {self.name} not supported. "
                "All implementations in priority list must be supported."
            )
        if impl.supports_args(*args, **kwargs):
            return impl
        if not torch.compiler.is_compiling():
            op_module.logger.debug(
                "Skipping provider %s because it does not support %s with args=%s kwargs=%s",
                impl.provider,
                self.name,
                op_module.lazy(lambda: op_module.tensors_str_no_data(args)),
                op_module.lazy(lambda: op_module.tensors_str_no_data(kwargs)),
            )
    raise RuntimeError(
        "Priority set incorrectly: the last implementation must support all args "
        "(can be native). This is likely an internal bug"
    )


def _get_context_priority(self) -> list[str]:
    return [impl.provider for impl in _active_priority(self)]


@contextmanager
def _set_context_priority(self, priority: list[str]):
    assert _active_patch is not None
    filtered = self._filter_priority_impls(priority)
    token = _priority_context(self).set(filtered)
    _active_patch.override_entries += 1
    try:
        yield
    finally:
        _priority_context(self).reset(token)


def install_vllm_ir_priority_context_patch() -> VllmIrPriorityContextPatch:
    """Install a reversible, thread/task-local scoped priority override."""
    global _active_patch
    if _active_patch is not None:
        raise RuntimeError("vLLM IR priority ContextVar patch is already installed")

    from vllm.ir.op import IrOp

    for name in ("dispatch", "get_priority", "set_priority"):
        if getattr(getattr(IrOp, name), "_cosmos_context_priority_patch", False):
            raise RuntimeError(f"vLLM IrOp.{name} is already patched")

    patch = VllmIrPriorityContextPatch(
        owner=IrOp,
        original_dispatch=IrOp.dispatch,
        original_get_priority=IrOp.get_priority,
        original_set_priority=IrOp.set_priority,
    )
    for replacement in (
        _dispatch_with_context_priority,
        _get_context_priority,
        _set_context_priority,
    ):
        replacement._cosmos_context_priority_patch = True
    IrOp.dispatch = _dispatch_with_context_priority
    IrOp.get_priority = _get_context_priority
    IrOp.set_priority = _set_context_priority
    _active_patch = patch
    return patch
