"""bot_core package — number-sequence clicking bot engine (Windows-first, fallbacks everywhere)."""
from __future__ import annotations

import importlib
from typing import Any

__all__ = ["NumberBot", "GridDetector", "NumberRecognizer", "Clicker",
           "wait_for_clear", "RunStats", "load_config"]
__version__ = "1.1.0"

# Lazy symbol map: resolved on first access, never at import time, so
# `import bot_core` stays side-effect-free while siblings are still written.
_LAZY = {
    "NumberBot": ("bot_core.bot", "NumberBot"),
    "load_config": ("bot_core.bot", "load_config"),
    "GridDetector": ("bot_core.detector", "GridDetector"),
    "NumberRecognizer": ("bot_core.detector", "NumberRecognizer"),
    "scan_roi": ("bot_core.detector", "scan_roi"),
    "build_number_map": ("bot_core.detector", "build_number_map"),
    "Clicker": ("bot_core.clicker", "Clicker"),
    "cell_snapshot": ("bot_core.verifier", "cell_snapshot"),
    "is_cleared": ("bot_core.verifier", "is_cleared"),
    "wait_for_clear": ("bot_core.verifier", "wait_for_clear"),
    "ocr_gone_check": ("bot_core.verifier", "ocr_gone_check"),
    "RunStats": ("bot_core.stats", "RunStats"),
}
# Alternate homes probed when the primary module lacks the attribute.
_FALLBACK = {
    "GridDetector": [("bot_core.bot", "GridDetector")],
    "NumberRecognizer": [("bot_core.bot", "NumberRecognizer")],
    "Clicker": [("bot_core.bot", "Clicker")],
    "wait_for_clear": [("bot_core.bot", "wait_for_clear")],
    "load_config": [("bot_core.gui", "load_config")],
    "RunStats": [("bot_core.gui", "RunStats")],
}


def _probe(mod: str, attr: str) -> Any:
    """Import attr from mod; None when unavailable (concurrent-write safe)."""
    try:
        return getattr(importlib.import_module(mod), attr, None)
    except Exception:
        return None


def __getattr__(name: str) -> Any:  # PEP 562 lazy exports; never raises for known names
    if name in _LAZY:
        mod, attr = _LAZY[name]
        val = _probe(mod, attr)
        if val is None:
            for mod2, attr2 in _FALLBACK.get(name, []):
                val = _probe(mod2, attr2)
                if val is not None:
                    break
        if val is not None:
            globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
