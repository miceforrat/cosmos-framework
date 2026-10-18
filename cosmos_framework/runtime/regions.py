from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar, Token
from typing import Callable, ContextManager, TypeAlias

RegionFactory: TypeAlias = Callable[[str], ContextManager[None]]

_REGION_FACTORY: ContextVar[RegionFactory | None] = ContextVar(
    "cosmos_runtime_region_factory",
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