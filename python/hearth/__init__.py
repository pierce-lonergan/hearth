"""Hearth: frontier-scale Mixture-of-Experts models on ordinary hardware.

`import hearth` is cheap and never loads the native library or optional
dependencies; the names below resolve lazily on first use.
"""
from __future__ import annotations

import importlib

__version__ = "0.1.0"

_LAZY = {
    "Engine": "hearth.engine",
    "HearthError": "hearth.engine",
    "HearthLibNotFound": "hearth._native",
    "Sampler": "hearth.generate",
    "GenStats": "hearth.generate",
    "Conversation": "hearth.chat",
    "Tokenizer": "hearth.chat",
    "ChatTemplate": "hearth.chat",
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'hearth' has no attribute {name!r}")
    value = getattr(importlib.import_module(mod), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
