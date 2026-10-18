"""Run Cosmos inference after installing component-level profiling wrappers."""

from __future__ import annotations

import runpy

from profiling_tools.core.component_runtime import install


if __name__ == "__main__":
    install()
    runpy.run_module("cosmos_framework.scripts.inference", run_name="__main__")

