"""Fast left-click output via Win32 SendInput (ctypes), fallback to pyautogui.

All third-party imports are guarded so ``import bot_core.clicker`` never fails
on a machine missing optional dependencies.
"""
from __future__ import annotations

import ctypes
import random
import time
from typing import Dict, Optional

# Guarded optional deps.
try:
    import pyautogui  # type: ignore
except Exception:  # pragma: no cover
    pyautogui = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# Win32 SendInput plumbing (Windows only; everything else uses fallback).
# ---------------------------------------------------------------------------
try:
    _user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    _IS_WINDOWS = True
except Exception:  # pragma: no cover - non-Windows interpreter.
    _user32 = None  # type: ignore
    _IS_WINDOWS = False

# SendInput constants.
_INPUT_MOUSE = 0
_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_ABSOLUTE = 0x8000


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("mi", _MOUSEINPUT)]


def _send_input(*inputs: "_INPUT") -> int:
    """Thin wrapper around Win32 SendInput; returns events accepted."""
    if _user32 is None:
        return 0
    n = len(inputs)
    arr = (_INPUT * n)(*inputs)
    cb = ctypes.sizeof(_INPUT)
    try:
        return int(_user32.SendInput(n, arr, cb))
    except Exception:
        return 0


def _screen_size() -> tuple[int, int]:
    """Primary monitor size via Win32 (fallback 1920x1080)."""
    if _user32 is not None:
        try:
            return (int(_user32.GetSystemMetrics(0)), int(_user32.GetSystemMetrics(1)))
        except Exception:
            pass
    return (1920, 1080)


def _move_absolute(x: int, y: int) -> None:
    """Move cursor to absolute screen coords via SendInput (0..65535 scale)."""
    sw, sh = _screen_size()
    ax = max(0, min(65535, int(x * 65535 / max(1, sw - 1))))
    ay = max(0, min(65535, int(y * 65535 / max(1, sh - 1))))
    extra = ctypes.pointer(ctypes.c_ulong(0))
    mi = _MOUSEINPUT(ax, ay, 0, _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE, 0, extra)
    _send_input(_INPUT(_INPUT_MOUSE, mi))


def _left_down_up() -> None:
    """Emit left-down + left-up as two SendInput events."""
    extra = ctypes.pointer(ctypes.c_ulong(0))
    down = _INPUT(_INPUT_MOUSE, _MOUSEINPUT(0, 0, 0, _MOUSEEVENTF_LEFTDOWN, 0, extra))
    up = _INPUT(_INPUT_MOUSE, _MOUSEINPUT(0, 0, 0, _MOUSEEVENTF_LEFTUP, 0, extra))
    # Keep a reference to `extra` alive for the duration of the call.
    _send_input(down)
    _send_input(up)


# ---------------------------------------------------------------------------
# Clicker
# ---------------------------------------------------------------------------
class Clicker:
    """High-speed clicker with humanization + smooth-move options.

    Attributes:
        clicks: total successful click calls.
        start_time: ``time.perf_counter()`` of first click (None until then).
    """

    def __init__(
        self,
        click_delay_ms: int = 30,
        humanize: bool = False,
        smooth_move: bool = False,
        jitter_px: int = 1,
    ) -> None:
        """Args:
            click_delay_ms: post-click sleep (lets the game register input).
            humanize: add tiny random jitter + delay variance (slower, safer).
            smooth_move: interpolate cursor in 2-3 micro-steps (slower).
            jitter_px: max pixel offset applied when ``humanize`` is on.
        """
        self.click_delay_ms = max(0, int(click_delay_ms))
        self.humanize = bool(humanize)
        self.smooth_move = bool(smooth_move)
        self.jitter_px = max(0, int(jitter_px))
        self.clicks: int = 0
        self.start_time: Optional[float] = None
        self._last_pos: Optional[tuple[int, int]] = None
        # Detect fastest available backend once.
        self._use_sendinput = bool(_IS_WINDOWS and _user32 is not None)
        if pyautogui is not None:
            try:
                pyautogui.FAILSAFE = False
                pyautogui.PAUSE = 0.0  # we manage our own pacing.
            except Exception:
                pass

    # -- internals -----------------------------------------------------------
    def _apply_jitter(self, x: int, y: int) -> tuple[int, int]:
        """Optionally offset the target by +/- jitter_px (human-like noise)."""
        if not self.humanize or self.jitter_px <= 0:
            return (int(x), int(y))
        return (
            int(x) + random.randint(-self.jitter_px, self.jitter_px),
            int(y) + random.randint(-self.jitter_px, self.jitter_px),
        )

    def move(self, x: int, y: int) -> None:
        """Move cursor to (x, y) with no click and no pacing sleep.

        Splitting move from press lets the bot snapshot the cell AFTER the
        cursor arrives (so the pointer is inside the 'before' image) and
        measure click-to-clear latency without a built-in sleep.
        """
        self._move(int(x), int(y))

    def press(self) -> None:
        """Emit one left click at the current cursor position (no move/sleep)."""
        if self.start_time is None:
            self.start_time = time.perf_counter()
        self._press()
        self.clicks += 1

    def _move(self, x: int, y: int) -> None:
        """Move cursor to (x, y); interpolated when smooth_move is enabled."""
        if self.smooth_move and self._last_pos is not None:
            # 3-step linear interpolation: cheap and looks human enough.
            x0, y0 = self._last_pos
            for t in (0.33, 0.66, 1.0):
                xi = int(x0 + (x - x0) * t)
                yi = int(y0 + (y - y0) * t)
                self._move_once(xi, yi)
        else:
            self._move_once(int(x), int(y))
        self._last_pos = (int(x), int(y))

    def _move_once(self, x: int, y: int) -> None:
        """Single cursor move via best backend (SendInput -> pyautogui -> noop)."""
        if self._use_sendinput:
            try:
                _move_absolute(int(x), int(y))
                return
            except Exception:
                self._use_sendinput = False  # degrade gracefully, retry fallback.
        if pyautogui is not None:
            try:
                pyautogui.moveTo(int(x), int(y), _pause=False)
                return
            except Exception:
                pass
        # Last resort: SetCursorPos via ctypes (still Win32, minimal deps).
        if _user32 is not None:
            try:
                _user32.SetCursorPos(int(x), int(y))
            except Exception:
                pass

    def _press(self) -> None:
        """Emit a left click at the current cursor position."""
        if self._use_sendinput:
            try:
                if _send_input is not None:
                    _left_down_up()
                    return
            except Exception:
                self._use_sendinput = False
        if pyautogui is not None:
            try:
                pyautogui.click(_pause=False)
                return
            except Exception:
                pass
        if _user32 is not None:
            try:
                _user32.mouse_event(_MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                _user32.mouse_event(_MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                return
            except Exception as exc:
                raise RuntimeError(f"Clicker: no usable click backend: {exc}") from exc
        raise RuntimeError("Clicker: no usable click backend (need Windows API or pyautogui).")

    # -- public API ----------------------------------------------------------
    def click(self, x: int, y: int) -> None:
        """Move to (x, y) and left-click (no pacing sleep; bot owns pacing).

        Kept as a convenience/compat wrapper. ``NumberBot`` uses
        :meth:`move` + :meth:`press` so the inter-click gap is measured from
        the previous press instead of delaying every verification.
        """
        tx, ty = self._apply_jitter(int(x), int(y))
        self.move(tx, ty)
        self.press()

    def stats(self) -> Dict[str, float]:
        """Return ``{"clicks", "elapsed_s", "cps"}`` snapshot."""
        elapsed = (time.perf_counter() - self.start_time) if self.start_time else 0.0
        cps = (self.clicks / elapsed) if elapsed > 1e-9 else 0.0
        return {"clicks": float(self.clicks), "elapsed_s": float(elapsed), "cps": float(cps)}

    def reset_stats(self) -> None:
        """Zero counters (keeps configuration)."""
        self.clicks = 0
        self.start_time = None
        self._last_pos = None


__all__ = ["Clicker"]
