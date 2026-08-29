"""Public ABLE interfaces, loaded only when callers request them."""

from importlib import import_module
from typing import TYPE_CHECKING


__all__ = [
    "RunnerABLE",
    "TextDataLoader",
    "ABLEDataManager",
    "LoggerABLE",
    "ABLECalculator",
]

_LAZY_IMPORTS = {
    "RunnerABLE": (".runner", "RunnerABLE"),
    "TextDataLoader": (".io.text_data", "TextDataLoader"),
    "ABLEDataManager": (".io.able_data", "ABLEDataManager"),
    "LoggerABLE": (".logging.logger", "LoggerABLE"),
    "ABLECalculator": (".calculator.able", "ABLECalculator"),
}

if TYPE_CHECKING:
    from .calculator.able import ABLECalculator
    from .io.able_data import ABLEDataManager
    from .io.text_data import TextDataLoader
    from .logging.logger import LoggerABLE
    from .runner import RunnerABLE


def __getattr__(name: str):
    """Load a public interface on first access and cache it in this module."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
