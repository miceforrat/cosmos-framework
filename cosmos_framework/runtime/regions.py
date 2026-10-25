from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar, Token
from typing import Callable, ContextManager, TypeAlias

RegionFactory: TypeAlias = Callable[[str], ContextManager[None]]

_REGION_FACTORY: ContextVar[RegionFactory | None] = ContextVar(
    "cosmos_runtime_region_factory",
    default=None,
)


# Optional CUDA stream identity supplied by a runtime region controller.
# This is context-local so concurrent workload threads cannot overwrite each
# other's expected stream. Model code can use it to verify CUDA Graph capture.
_EXPECTED_CUDA_STREAM_PTR: ContextVar[int | None] = ContextVar(
    "cosmos_expected_cuda_stream_ptr",
    default=None,
)


def runtime_region(name: str) -> ContextManager[None]:
    factory = _REGION_FACTORY.get()
    if factory is None:
        return nullcontext()
    return factory(name)


def install_region_factory(
    factory: RegionFactory,
) -> Token[RegionFactory | None]:
    return _REGION_FACTORY.set(factory)


def reset_region_factory(token: Token[RegionFactory | None]) -> None:
    _REGION_FACTORY.reset(token)


def expected_cuda_stream_ptr() -> int | None:
    return _EXPECTED_CUDA_STREAM_PTR.get()


def install_expected_cuda_stream_ptr(stream_ptr: int) -> Token[int | None]:
    return _EXPECTED_CUDA_STREAM_PTR.set(int(stream_ptr))


def reset_expected_cuda_stream_ptr(token: Token[int | None]) -> None:
    _EXPECTED_CUDA_STREAM_PTR.reset(token)
