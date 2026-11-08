"""Baseline and agent benchmark entry points under the test package."""

from importlib import import_module

__all__ = [
    "AgentRunner", "BaselineRunner", "ComparisonConfig", "RunResult",
    "build_runner", "build_runners", "JsonlResultWriter",
]


def __getattr__(name):
    # Scoring saved outputs must not initialize agent/retriever components.
    modules = {"ComparisonConfig": ".config", "JsonlResultWriter": ".results"}
    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module(modules.get(name, ".runners"), __name__), name)
    globals()[name] = value
    return value
