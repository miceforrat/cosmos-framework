from contextlib import contextmanager

from cosmos_framework.runtime.regions import (
    install_region_factory,
    reset_region_factory,
    runtime_region,
)

def test_runtime_region_defaults_to_noop() -> None:
    with runtime_region("reasoner_decode"):
        value = 1

    assert value == 1




def test_runtime_region_uses_installed_factory() -> None:
    events: list[str] = []

    @contextmanager
    def factory(name: str):
        events.append(f"enter:{name}")
        try:
            yield
        finally:
            events.append(f"exit:{name}")

    token = install_region_factory(factory)
    try:
        with runtime_region("reasoner_decode"):
            events.append("body")
    finally:
        reset_region_factory(token)

    assert events == [
        "enter:reasoner_decode",
        "body",
        "exit:reasoner_decode",
    ]

def test_reset_restores_previous_factory() -> None:
    calls: list[str] = []

    @contextmanager
    def factory(name: str):
        calls.append(name)
        yield

    token = install_region_factory(factory)
    with runtime_region("first"):
        pass
    reset_region_factory(token)

    with runtime_region("second"):
        pass

    assert calls == ["first"]