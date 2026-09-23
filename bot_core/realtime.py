"""Realtime bot: scan the board, click the lowest number, repeat forever.

Strategy (user-requested simplification):
    loop:
      1. grab the ROI, detect digit blobs -> cells
      2. OCR every cell -> {number: (x, y)}
      3. click the smallest number found, then VERIFY the click registered
         (the cell changed on screen). If not, retry the click a few times,
         then blacklist that cell briefly so a dead tile can't stall the run.
      4. rescan immediately.

Board-refill aware: in many games a NEW number appears (often in the same
place) shortly after a click, and the board can look empty for a moment
during the swap. Therefore the loop NEVER treats "empty scan" as done by
default — it keeps auto-scanning until the stop hotkey, Ctrl+C, or the
configured target is clicked. ``exit_on_empty`` re-enables the old behavior
with an ``empty_timeout_ms`` grace period for the refill animation.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from .detector import (
    ClusterBoardDetector,
    GridDetector,
    TemplateRecognizer,
    build_number_map,
    ensure_dpi_awareness,
    scan_roi,
    tesseract_available,
)
from .verifier import cell_snapshot, wait_for_clear

Point = Tuple[int, int]
LogFn = Callable[[str], None]


class RealtimeBot:
    """Scan -> click lowest (verified) -> rescan. Thread-safe stop/pause."""

    def __init__(self, config: Optional[Dict] = None, log: Optional[LogFn] = None,
                 debug: bool = False) -> None:
        self.config: Dict = dict(config or {})
        self.log: LogFn = log if log is not None else (lambda m: print(f"[bot] {m}", flush=True))
        self.debug = bool(debug)
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.clicks = 0
        self.misses = 0
        self.verified = 0
        self.scans = 0
        self.t_start: Optional[float] = None
        self.t_end: Optional[float] = None
        self.last_map: Dict[int, Point] = {}
        self.history: List[Tuple[int, int, int, bool]] = []  # (n, x, y, verified)
        # (number, qx, qy) -> blacklist-until timestamp (perf_counter seconds)
        self._blacklist: Dict[Tuple[int, int, int], float] = {}

    # -- control --------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    @property
    def is_paused(self) -> bool:
        return self._pause.is_set()

    @property
    def is_stopped(self) -> bool:
        return self._stop.is_set()

    # -- config helpers -------------------------------------------------------
    def _cfg(self, key: str, default):
        return self.config.get(key, default)

    def _roi_dict(self) -> Optional[Dict[str, int]]:
        roi = self.config.get("roi")
        if isinstance(roi, (list, tuple)) and len(roi) == 4:
            x, y, w, h = (int(v) for v in roi)
            if w > 0 and h > 0:
                return {"left": x, "top": y, "width": w, "height": h}
        if isinstance(roi, dict):
            try:
                rd = {"left": int(roi["left"]), "top": int(roi["top"]),
                      "width": int(roi["width"]), "height": int(roi["height"])}
                if rd["width"] > 0 and rd["height"] > 0:
                    return rd
            except Exception:
                pass
        return None

    def _make_recognizer(self):
        """Template recognizer by default; tesseract when available."""
        conf = float(self._cfg("template_confidence", 0.3) or 0.3)
        rec = TemplateRecognizer(confidence_threshold=conf)
        if tesseract_available(str(self._cfg("tesseract_cmd", "") or "")):
            try:
                from .detector import NumberRecognizer
                return NumberRecognizer(
                    confidence_threshold=float(self._cfg("confidence_threshold", 0.6) or 0.6),
                    tesseract_cmd=str(self._cfg("tesseract_cmd", "") or ""),
                )
            except Exception:
                return rec
        return rec

    # -- one scan -------------------------------------------------------------
    def scan_once(self, recognizer=None, detector=None):
        """Grab ROI -> cells -> {number: (x,y)} in absolute screen coords."""
        rec = recognizer or self._make_recognizer()
        det = detector or GridDetector()
        roi = self._roi_dict()
        frame = scan_roi(roi)
        cells = None
        try:
            cells = ClusterBoardDetector().detect(frame)
        except Exception:
            cells = None
        if not cells:
            cells = det.find_cells(frame)
        nmap = build_number_map(frame, cells, rec,
                                workers=int(self._cfg("ocr_workers", 4) or 4))
        ox, oy = (roi["left"], roi["top"]) if roi else (0, 0)
        out = {int(n): (int(x) + ox, int(y) + oy) for n, (x, y) in nmap.items()}
        if self.debug:
            self._show_debug(frame, cells, nmap)
        return out

    def _show_debug(self, frame, cells, nmap_roi) -> None:
        """Draw detections on a copy of the frame and show it (cv2 window)."""
        try:
            import cv2

            vis = frame.copy()
            for (x, y, w, h) in (cells or []):
                cv2.rectangle(vis, (x, y), (x + w, y + h), (120, 120, 120), 1)
            for n, (x, y) in nmap_roi.items():
                cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
                cv2.putText(vis, str(n), (int(x) + 4, int(y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("NumberBot live scan (green=detected, red=next click)",
                       vis)
            cv2.waitKey(1)
        except Exception:
            pass

    # -- blacklist (dead cells after failed verification) ----------------------
    def _blacklist_add(self, n: int, x: int, y: int) -> None:
        ms = float(self._cfg("click_blacklist_ms", 1200) or 1200)
        until = time.perf_counter() + max(50.0, ms) / 1000.0
        key = (int(n), int(x) // 16, int(y) // 16)
        self._blacklist[key] = until
        # prune expired
        now = time.perf_counter()
        self._blacklist = {k: u for k, u in self._blacklist.items() if u > now}

    def _is_blacklisted(self, n: int, x: int, y: int) -> bool:
        key = (int(n), int(x) // 16, int(y) // 16)
        until = self._blacklist.get(key)
        return until is not None and time.perf_counter() < until

    # -- click + verify --------------------------------------------------------
    def click_and_verify(self, n: int, x: int, y: int, clicker) -> bool:
        """Click (x, y) and wait until the cell visibly changes on screen.

        A missing/failed verification is retried up to ``max_click_attempts``
        times. Returns True when the game visibly reacted (or verification is
        impossible), False when the cell refused to change — the caller then
        blacklists the cell so the loop moves on instead of spam-clicking.
        """
        attempts = max(1, int(self._cfg("max_click_attempts", 3) or 3))
        timeout_ms = int(self._cfg("clear_timeout_ms", 900) or 900)
        poll_ms = int(self._cfg("clear_poll_ms", 8) or 8)
        diff_th = float(self._cfg("clear_diff_threshold", 12.0) or 12.0)
        reaction_ms = int(self._cfg("reaction_buffer_ms", 15) or 15)

        before = None
        try:
            clicker.move(int(x), int(y))
            before = cell_snapshot(int(x), int(y), size=44)
        except Exception:
            before = None  # cannot verify; rely on the next scan to correct.

        for attempt in range(1, attempts + 1):
            try:
                clicker.press()
                self.clicks += 1
            except Exception as exc:
                self.log(f"click {n} failed: {exc}")
                return False
            if before is None:
                self.history.append((int(n), int(x), int(y), False))
                return True
            if reaction_ms > 0:
                time.sleep(reaction_ms / 1000.0)
            try:
                if wait_for_clear(int(x), int(y), before, timeout_ms, poll_ms, diff_th):
                    self.verified += 1
                    self.history.append((int(n), int(x), int(y), True))
                    return True
            except Exception:
                pass
            # Cell did not change: refresh the baseline and click again.
            self.log(f"[{n}] click not registered (attempt {attempt}/{attempts}), retrying")
            try:
                before = cell_snapshot(int(x), int(y), size=44)
            except Exception:
                before = None
        self.misses += 1
        self.history.append((int(n), int(x), int(y), False))
        return False

    # -- main loop ------------------------------------------------------------
    def run(self, target: Optional[int] = None) -> Dict:
        """Run until stop is requested, target is clicked, or (optionally,
        with ``exit_on_empty``) the board stays empty for ``empty_timeout_ms``.
        """
        ensure_dpi_awareness()
        from .clicker import Clicker

        clicker = Clicker(
            click_delay_ms=int(self._cfg("click_delay_ms", 30) or 0),
            humanize=bool(self._cfg("humanize", False)),
            smooth_move=bool(self._cfg("smooth_move", False)),
        )
        rec = self._make_recognizer()
        det = GridDetector()
        target = int(target) if target else None
        exit_on_empty = bool(self._cfg("exit_on_empty", False))
        empty_timeout_ms = float(self._cfg("empty_timeout_ms", 2500) or 2500)
        delay = float(self._cfg("realtime_scan_delay_ms", 30) or 30) / 1000.0

        self.t_start = time.perf_counter()
        self.t_end = None
        last_nonempty_t = time.perf_counter()

        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.05)
                continue
            try:
                nmap = self.scan_once(rec, det)
            except Exception as exc:
                self.log(f"scan failed: {exc}")
                time.sleep(0.2)
                continue
            self.scans += 1
            self.last_map = nmap

            if not nmap:
                # Board may be between the click and the number replacement —
                # keep scanning; only exit when explicitly configured to.
                if exit_on_empty and (time.perf_counter() - last_nonempty_t) * 1000.0 >= empty_timeout_ms:
                    self.log(f"board empty for {empty_timeout_ms:.0f} ms -> done")
                    break
                time.sleep(delay)
                continue
            last_nonempty_t = time.perf_counter()

            # Lowest number that isn't on the failed-click blacklist.
            candidates = sorted(
                (n, xy) for n, xy in nmap.items()
                if not self._is_blacklisted(n, xy[0], xy[1])
            )
            if not candidates:
                time.sleep(delay)  # everything blacklisted; wait for refresh
                continue
            n, (x, y) = candidates[0]
            ok = self.click_and_verify(n, x, y, clicker)
            if not ok:
                self._blacklist_add(n, x, y)
            if target is not None and n >= target:
                self.log(f"reached target {target} -> done")
                break
            time.sleep(delay)

        self.t_end = time.perf_counter()
        if self.debug:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass
        elapsed = (self.t_end - self.t_start) if self.t_start and self.t_end else 0.0
        return {
            "clicks": self.clicks,
            "verified": self.verified,
            "misses": self.misses,
            "scans": self.scans,
            "elapsed_s": round(elapsed, 3),
            "cps": round(self.clicks / elapsed, 2) if elapsed > 0 else 0.0,
            "last_map_size": len(self.last_map),
        }

    # -- background run --------------------------------------------------------
    def start_async(self, target: Optional[int] = None) -> threading.Thread:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("already running")
        self._thread = threading.Thread(target=self.run, kwargs={"target": target},
                                        name="RealtimeBot", daemon=True)
        self._thread.start()
        return self._thread


__all__ = ["RealtimeBot"]
