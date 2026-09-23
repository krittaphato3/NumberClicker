"""Clear-verification: confirm a clicked number actually disappeared.

Speed budget: 1..50 in <20s => ~400ms/number worst case, target 50-150ms.
So :func:`wait_for_clear` polls a tiny (44x44) grayscale ROI every ~8ms and
returns the instant the mean-absolute-difference exceeds the threshold.

Performance note: a *persistent, thread-local* ``mss`` instance is reused
across grabs. Creating one per grab costs ~5-10ms (socket setup), which at
an 8ms poll cadence would dominate the 20s budget.

All heavy imports are guarded; the module imports even with zero deps. Calls
that need a missing dep raise RuntimeError only when invoked.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

# Guarded imports.
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import mss  # type: ignore
except Exception:  # pragma: no cover
    mss = None  # type: ignore

try:
    from PIL import ImageGrab  # type: ignore
except Exception:  # pragma: no cover
    ImageGrab = None  # type: ignore


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("numpy is required for verification but not installed.")


# Thread-local persistent mss instance: mss objects are NOT thread-safe, but
# reusing one per thread removes per-grab setup cost in the polling hot loop.
_tls = threading.local()


def _get_sct():
    """Return this thread's persistent mss instance (created on first use)."""
    sct = getattr(_tls, "sct", None)
    if sct is None:
        sct = mss.mss()
        _tls.sct = sct
    return sct


def close_thread_capture() -> None:
    """Close this thread's persistent mss instance (call on thread exit)."""
    sct = getattr(_tls, "sct", None)
    if sct is not None:
        try:
            sct.close()
        except Exception:
            pass
        _tls.sct = None


def cell_snapshot(cx: int, cy: int, size: int = 44):
    """Grab a small square ROI centered at (cx, cy), returned as grayscale.

    Args:
        cx, cy: screen-space center of the clicked cell.
        size: side length in px (default 44 — covers one digit cell).

    Returns:
        2D uint8 grayscale numpy array of shape (size, size).

    Backend: persistent ``mss`` is fastest; falls back to PIL ImageGrab.
    Raises RuntimeError only when no backend/numpy is available.
    """
    _require_numpy()
    half = max(1, int(size) // 2)
    left = int(cx) - half
    top = int(cy) - half
    w = h = int(size)
    # --- mss fast path (persistent per-thread instance) ----------------------
    if mss is not None:
        try:
            sct = _get_sct()
            shot = sct.grab({"left": left, "top": top, "width": w, "height": h})
            bgra = np.array(shot, dtype=np.uint8)
            bgr = bgra[:, :, :3]
            if cv2 is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # Manual BGR->gray (BT.601) when cv2 is missing.
            b = bgr[:, :, 0].astype(np.float32)
            g = bgr[:, :, 1].astype(np.float32)
            r = bgr[:, :, 2].astype(np.float32)
            return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
        except Exception:
            # Broken persistent instance: drop it so the next call recreates.
            _tls.sct = None  # fall through to PIL fallback.
    # --- PIL fallback -------------------------------------------------------
    if ImageGrab is not None:
        try:
            pil_img = ImageGrab.grab(bbox=(left, top, left + w, top + h)).convert("L")
            arr = np.array(pil_img, dtype=np.uint8)
            if arr.shape != (h, w):
                # Clamp/pad defensively near screen edges.
                import numpy as _np  # reuse guarded np.

                out = _np.zeros((h, w), dtype=_np.uint8)
                hh, ww = arr.shape[:2]
                out[: min(h, hh), : min(w, ww)] = arr[: min(h, hh), : min(w, ww)]
                return out
            return arr
        except Exception as exc:
            raise RuntimeError(f"cell_snapshot failed: {exc}") from exc
    raise RuntimeError("No screen-capture backend available (install `mss` or `pillow`).")


def is_cleared(before_gray, after_gray, diff_threshold: float = 12.0) -> bool:
    """Return True when two grayscale snapshots differ enough to mean "gone".

    Uses mean absolute difference (MAD):
        ``mean(|after - before|) > diff_threshold`` -> cleared.

    The game grays out / removes the digit on a valid click, which reliably
    shifts a 44x44 cell by >>12 gray levels on average, while an unregistered
    click leaves the cell nearly identical (MAD ~0-3 from capture noise).
    """
    _require_numpy()
    try:
        a = np.asarray(before_gray, dtype=np.float32)
        b = np.asarray(after_gray, dtype=np.float32)
        if a.shape != b.shape:
            # Resize `b` to `a` when cv2 exists; else compare overlap region.
            if cv2 is not None:
                b = cv2.resize(b, (a.shape[1], a.shape[0]))
            else:
                hh = min(a.shape[0], b.shape[0])
                ww = min(a.shape[1], b.shape[1])
                a, b = a[:hh, :ww], b[:hh, :ww]
        mad = float(np.mean(np.abs(b - a)))
        return mad > float(diff_threshold)
    except Exception:
        return False  # never crash the hot loop on a bad frame.


def wait_for_clear(
    cx: int,
    cy: int,
    before_img,
    timeout_ms: int = 1200,
    poll_ms: int = 8,
    diff_threshold: float = 12.0,
) -> bool:
    """Poll ``cell_snapshot`` until the cell visibly changes or timeout hits.

    Args:
        cx, cy: screen-space center of the clicked cell.
        before_img: grayscale snapshot taken *before* the click.
        timeout_ms: max wait (default 1200ms; normal clears land in 50-150ms).
        poll_ms: sleep between polls (~8ms keeps 1..50 under the 20s budget).
        diff_threshold: MAD threshold forwarded to :func:`is_cleared`.

    Returns:
        True as soon as ``diff > threshold`` (number gone/grayed),
        False on timeout (caller should retry click / rescan).

    Notes:
        * Uses ``time.perf_counter`` deadline (monotonic, not wall-clock).
        * Any single grab failure is ignored — polling continues until deadline.
        * ``poll_ms`` sleep uses high-resolution ``time.sleep``; on Windows the
          effective granularity is ~1-2ms+ which is fine for an 8ms cadence.
    """
    deadline = time.perf_counter() + max(0.0, float(timeout_ms) / 1000.0)
    interval = max(0.0, float(poll_ms) / 1000.0)
    # Fast-path: if before_img is missing, we cannot diff — take one fresh
    # frame and report change relative to nothing as False (caller rescans).
    if before_img is None:
        return False
    while True:
        try:
            after = cell_snapshot(int(cx), int(cy))
            if is_cleared(before_img, after, diff_threshold):
                return True
        except Exception:
            pass  # transient grab error -> keep polling until deadline.
        if time.perf_counter() >= deadline:
            return False
        if interval > 0:
            time.sleep(interval)


def ocr_gone_check(cx: int, cy: int, target: int, size: int = 44,
                   recognizer=None) -> Optional[bool]:
    """Re-OCR a single cell; cleared when its text != target.

    Args:
        cx, cy: screen-space center of the cell.
        target: the number that should no longer be visible.
        size: snapshot side length in px.
        recognizer: optional shared NumberRecognizer (constructed per call
            when omitted, which is fine — this path is a rare tie-breaker).

    Returns:
        True  -> cell no longer shows ``target`` (cleared).
        False -> cell still shows ``target`` (click missed).
        None  -> OCR unavailable/uncertain (caller should use diff result).

    This is *optional* (slower than diffing) and only used as a tiebreaker
    after a diff-timeout in ``bot.py``.
    """
    if recognizer is None:
        try:
            # Local guarded import: detector tolerates missing tesseract itself.
            from .detector import NumberRecognizer  # type: ignore
        except Exception:
            try:
                from bot_core.detector import NumberRecognizer  # type: ignore
            except Exception:
                return None
        recognizer = NumberRecognizer()
    try:
        after = cell_snapshot(int(cx), int(cy), size=int(size))
    except Exception:
        return None
    try:
        parsed = recognizer.ocr_cell(after)
    except Exception:
        return None
    if parsed is None:
        return None  # unreadable != proof of cleared; let diff decide.
    value, _conf = parsed
    try:
        return bool(int(value) != int(target))
    except Exception:
        return None


__all__ = ["cell_snapshot", "is_cleared", "wait_for_clear", "ocr_gone_check",
           "close_thread_capture"]
