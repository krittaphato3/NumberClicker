"""NumberBot: orchestrate scan -> click -> verify-clear for numbers 1..50.

Algorithm flow (a-h):
    a. calibrate(): grab ROI, detect grid, full OCR scan -> number_map.
    b. run(): iterate target = 1..50 in order.
    c. lookup coord (targeted re-scan only for missing numbers).
    d. move cursor, snapshot cell, click, then wait_for_clear BEFORE
       incrementing to the next number.
    e. on timeout: OCR tie-break, retry click, then full rescan.
    f. stats (clicks, cps, accuracy, elapsed) snapshot via stats().
    g. pause / resume / stop via thread-safe events (own + external).
    h. start_async(): run loop on a background thread.

STRICT SEQUENCING RULE (the core requirement):
    The bot NEVER advances to number N+1 until number N is *proven* gone:
      1. pixel-diff verification of the clicked cell (fast path), or
      2. OCR tie-break proves the digit no longer reads as N, or
      3. N consecutive full-board rescans cannot find N anywhere
         (only possible when it is genuinely no longer on the board).
    If a number refuses to clear after ``stuck_limit`` step attempts, the run
    ABORTS (or skips that number when ``allow_skip`` is true) instead of
    clicking further numbers out of order.

Speed budget: 1..50 in <20s => ~400ms/number worst case. Clean run:
    click(~1ms) + reaction(20ms) + clear detect(~10-80ms) + pacing(30ms)
    ~= 60-130ms per number => 3-7s total including calibration.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Guarded sibling imports -- bot.py imports even when sibling deps are missing.
# ---------------------------------------------------------------------------
try:
    from .detector import GridDetector, NumberRecognizer, build_number_map, scan_roi
except Exception:  # pragma: no cover - fallback for script-style execution.
    try:
        from bot_core.detector import (  # type: ignore
            GridDetector, NumberRecognizer, build_number_map, scan_roi,
        )
    except Exception:
        GridDetector = None  # type: ignore
        NumberRecognizer = None  # type: ignore
        build_number_map = None  # type: ignore
        scan_roi = None  # type: ignore

try:
    from .clicker import Clicker
except Exception:  # pragma: no cover
    try:
        from bot_core.clicker import Clicker  # type: ignore
    except Exception:
        Clicker = None  # type: ignore

try:
    from .verifier import cell_snapshot, is_cleared, ocr_gone_check, wait_for_clear
except Exception:  # pragma: no cover
    try:
        from bot_core.verifier import (  # type: ignore
            cell_snapshot, is_cleared, ocr_gone_check, wait_for_clear,
        )
    except Exception:
        cell_snapshot = None  # type: ignore
        is_cleared = None  # type: ignore
        ocr_gone_check = None  # type: ignore
        wait_for_clear = None  # type: ignore

Point = Tuple[int, int]
LogFn = Callable[[str], None]

# Step outcomes returned by step_number().
CLEARED = "cleared"      # number verified gone -> safe to advance.
STUCK = "stuck"          # clicked but could not verify clear (game may be lagging).
NOT_FOUND = "not_found"  # number absent from the map even after a targeted rescan.

DEFAULTS: Dict = {
    "click_delay_ms": 30,          # pacing gap between successive clicks.
    "reaction_buffer_ms": 20,      # settle time after click before verifying.
    "confidence_threshold": 0.6,   # tesseract mean-confidence gate.
    "template_confidence": 0.3,    # template-OCR correlation gate (raw TM score).
    "grid_mode": "auto",           # "auto" or "manual" (fixed rows x cols).
    "grid_rows": 5,
    "grid_cols": 10,
    "roi": None,                   # None | [x, y, w, h] | {left, top, width, height}.
    "start_hotkey": "F1",
    "stop_hotkey": "F2",
    "pause_hotkey": "F3",
    "scan_frequency_hz": 0,        # 0 = watcher off (fastest); >0 = background re-scan rate.
    "max_verify_attempts": 2,      # clicks per step_number call before giving up.
    "stuck_limit": 4,              # consecutive failed steps for one number -> abort/skip.
    "absent_limit": 2,             # consecutive full rescans absent -> treat as cleared.
    "allow_skip": False,           # True = skip stuck numbers instead of aborting.
    "ocr_workers": 4,              # parallel tesseract workers for full-board scans.
    "verify_size": 44,             # side of the square cell snapshot used for diffing.
    "clear_timeout_ms": 1200,      # max wait for one click to visually clear.
    "clear_poll_ms": 8,            # poll interval while waiting for clear.
    "clear_diff_threshold": 12.0,  # mean-abs-diff gray levels that means "gone".
    "humanize": False,             # jitter + delay variance (slower, safer).
    "smooth_move": False,          # interpolated cursor movement (slower).
    "tesseract_cmd": "",           # explicit tesseract.exe path (Windows); "" = PATH.
    "debug_overlay": True,         # GUI overlay draws detected numbers.
    "phase_wait_timeout_ms": 15000,  # max wait for the next board phase to appear.
    "phase_poll_interval_s": 0.35,   # rescan interval while waiting for board swap.
}


def _default_log(msg: str) -> None:
    print(f"[bot] {msg}", flush=True)


def load_config(path: str = "config.json") -> Dict:
    """Load JSON config with sane defaults for every known key."""
    cfg = dict(DEFAULTS)
    try:
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                user = json.load(fh)
            if isinstance(user, dict):
                defaults = dict(DEFAULTS)
                defaults.update(user)
                return defaults
    except Exception:
        pass  # corrupt config -> safe defaults.
    return cfg


class NumberBot:
    """Stateful 1..50 clicking bot. Reusable across runs via :meth:`reset`."""

    def __init__(self, config: Optional[Dict] = None, log: Optional[LogFn] = None) -> None:
        """Args:
            config: settings dict (see config.json); defaults when None.
            log: injectable logger (defaults to stdout; None = silent).
        """
        self.config: Dict = dict(config) if config else load_config()
        self.log: LogFn = log if log is not None else _default_log
        # --- subsystems (constructed tolerantly of missing deps) ------------
        rows = int(self.config.get("grid_rows", 5) or 5)
        cols = int(self.config.get("grid_cols", 10) or 10)
        mode = str(self.config.get("grid_mode", "auto") or "auto")
        if GridDetector is not None:
            self.detector = GridDetector(rows=rows, cols=cols, mode=mode)
        else:  # pragma: no cover
            self.detector = None  # type: ignore
        conf_th = float(self.config.get("confidence_threshold", 0.6) or 0.6)
        tess_cmd = str(self.config.get("tesseract_cmd", "") or "")
        if NumberRecognizer is not None:
            self.recognizer = NumberRecognizer(confidence_threshold=conf_th, tesseract_cmd=tess_cmd)
        else:  # pragma: no cover
            self.recognizer = None  # type: ignore
        if Clicker is not None:
            self.clicker = Clicker(
                click_delay_ms=int(self.config.get("click_delay_ms", 30) or 0),
                humanize=bool(self.config.get("humanize", False)),
                smooth_move=bool(self.config.get("smooth_move", False)),
            )
        else:  # pragma: no cover
            self.clicker = None  # type: ignore
        # --- thread-safe control flags --------------------------------------
        self._stop = threading.Event()
        self._pause = threading.Event()  # set() == paused.
        self._step_busy = threading.Event()  # True while a click+verify is in flight.
        self._map_lock = threading.RLock()
        self._watch_stop = threading.Event()
        self._watcher: Optional[threading.Thread] = None
        self._thread: Optional[threading.Thread] = None
        # --- recognizer fallback state ----------------------------------------
        self._tess_checked: bool = False
        self._tess_ok: bool = False
        self._template_rec = None
        # --- run state -------------------------------------------------------
        self.number_map: Dict[int, Point] = {}
        self.cells: List[Tuple[int, int, int, int]] = []
        self.roi_offset: Tuple[int, int] = (0, 0)
        self.grid_image = None
        self.target: int = 1
        self.cleared: List[int] = []
        self.misses: int = 0
        self.retries: int = 0
        self.rescans: int = 0
        self.aborted: bool = False
        self._start_num: int = 1
        self._end_num: int = 50
        self._last_press_t: Optional[float] = None
        self.t_start: Optional[float] = None
        self.t_end: Optional[float] = None

    # -- config helpers -------------------------------------------------------
    def _cfg_int(self, key: str, default: int) -> int:
        try:
            return int(self.config.get(key, default) or default)
        except Exception:
            return default

    def _cfg_float(self, key: str, default: float) -> float:
        try:
            return float(self.config.get(key, default) or default)
        except Exception:
            return default

    # -- control flags ---------------------------------------------------------
    def stop(self) -> None:
        """Signal the run loop to stop at the next safe checkpoint."""
        self._stop.set()

    def pause(self) -> None:
        """Pause after the current number finishes verifying."""
        self._pause.set()

    def resume(self) -> None:
        """Resume a paused run."""
        self._pause.clear()

    @property
    def is_paused(self) -> bool:
        """True while paused."""
        return self._pause.is_set()

    @property
    def is_stopped(self) -> bool:
        """True after :meth:`stop` was called."""
        return self._stop.is_set()

    # -- ROI -------------------------------------------------------------------
    def _roi_dict(self) -> Optional[Dict[str, int]]:
        """Normalize config roi ([x,y,w,h] list or dict) -> mss-style dict."""
        roi = self.config.get("roi")
        if isinstance(roi, dict) and {"left", "top", "width", "height"} <= set(roi.keys()):
            try:
                rd = {
                    "left": int(roi["left"]),
                    "top": int(roi["top"]),
                    "width": int(roi["width"]),
                    "height": int(roi["height"]),
                }
                if rd["width"] > 0 and rd["height"] > 0:
                    return rd
            except Exception:
                return None
        if isinstance(roi, (list, tuple)) and len(roi) == 4:
            try:
                x, y, w, h = (int(v) for v in roi)
                if w > 0 and h > 0:
                    return {"left": x, "top": y, "width": w, "height": h}
            except Exception:
                return None
        return None  # None == full screen.

    # -- recognizer backends ---------------------------------------------------
    def _tesseract_ok(self) -> bool:
        """True when a working tesseract binary is present (cached per bot)."""
        if self._tess_checked:
            return self._tess_ok
        self._tess_checked = True
        try:
            from .detector import tesseract_available
        except Exception:
            try:
                from bot_core.detector import tesseract_available  # type: ignore
            except Exception:
                self._tess_ok = False
                return False
        try:
            self._tess_ok = bool(tesseract_available(str(self.config.get("tesseract_cmd", "") or "")))
        except Exception:
            self._tess_ok = False
        if not self._tess_ok:
            self.log("tesseract not found -> using built-in font-template OCR")
        return self._tess_ok

    def _template_recognizer(self):
        """Lazily-built font-template recognizer (tesseract-free fallback)."""
        if self._template_rec is None:
            try:
                from .detector import TemplateRecognizer
            except Exception:
                try:
                    from bot_core.detector import TemplateRecognizer  # type: ignore
                except Exception:
                    return None
            conf = self._cfg_float("template_confidence", 0.3)
            self._template_rec = TemplateRecognizer(confidence_threshold=conf)
        return self._template_rec

    def _cluster_cells(self, frame) -> Optional[List[Tuple[int, int, int, int]]]:
        """Try digit-blob cell detection (handles uneven tiles); adopts on success."""
        try:
            from .detector import ClusterBoardDetector
        except Exception:
            try:
                from bot_core.detector import ClusterBoardDetector  # type: ignore
            except Exception:
                return None
        try:
            cells = ClusterBoardDetector().detect(frame)
        except Exception:
            return None
        if cells and len(cells) >= 4:
            self.cells = list(cells)
            try:
                self.detector.set_cells(list(cells))  # keep rescan cache in sync.
            except Exception:
                pass
            return self.cells
        return None

    # -- calibration -------------------------------------------------------------
    def calibrate(self) -> Dict[int, Point]:
        """(a) Grab ROI -> detect grid -> full OCR scan -> number_map.

        Returns the fresh number_map (absolute screen coords). Raises
        RuntimeError when capture/detection backends are unavailable.
        """
        if scan_roi is None or self.detector is None or build_number_map is None:
            raise RuntimeError("calibrate: detector/capture backend unavailable.")
        try:
            from .detector import ensure_dpi_awareness
        except Exception:
            try:
                from bot_core.detector import ensure_dpi_awareness  # type: ignore
            except Exception:
                ensure_dpi_awareness = None  # type: ignore
        if ensure_dpi_awareness is not None:
            ensure_dpi_awareness()  # physical-pixel coords on scaled displays.
        roi = self._roi_dict()
        self.roi_offset = (roi["left"], roi["top"]) if roi else (0, 0)
        frame = scan_roi(roi)  # BGR numpy image, ROI-local coords.
        self.grid_image = frame
        cluster_cells = None
        if str(self.config.get("grid_mode", "auto")).lower() == "auto":
            # Digit-blob detection first: robust to rounded/uneven tiles.
            cluster_cells = self._cluster_cells(frame)
        if cluster_cells is None:
            if str(self.config.get("grid_mode", "auto")).lower() == "auto":
                try:
                    self.cells = self.detector.auto_detect_grid(frame)
                except Exception:
                    self.cells = self.detector.find_cells(frame)
            else:
                self.cells = self.detector.find_cells(frame)
        workers = self._cfg_int("ocr_workers", 4)
        recog = self.recognizer if self._tesseract_ok() else (self._template_recognizer() or self.recognizer)
        raw_map: Dict[int, Point] = {}
        try:
            raw_map = build_number_map(frame, self.cells, recog, workers=workers) or {}
        except Exception as exc:
            self.log(f"calibrate: OCR scan failed: {exc}")
        # Under-detection fallbacks, cheap -> expensive:
        if len(raw_map) < 4 and cluster_cells is None:
            # Grid path found no readable digits: retry with digit-blob cells.
            if self._cluster_cells(frame) is not None:
                try:
                    raw_map = build_number_map(frame, self.cells, recog, workers=workers) or {}
                except Exception as exc:
                    self.log(f"calibrate: cluster-rescan OCR failed: {exc}")
        if len(raw_map) < 4 and recog is self.recognizer:
            # Tesseract active but (almost) nothing read -> try templates too.
            alt = self._template_recognizer()
            if alt is not None:
                try:
                    alt_map = build_number_map(frame, self.cells, alt, workers=workers) or {}
                    if len(alt_map) > len(raw_map):
                        raw_map = alt_map
                        self.log("calibrate: template OCR outperformed tesseract")
                except Exception as exc:
                    self.log(f"calibrate: template fallback failed: {exc}")
        ox, oy = self.roi_offset
        with self._map_lock:
            self.number_map = {n: (x + ox, y + oy) for n, (x, y) in raw_map.items()}
            # Already-cleared numbers must never come back on the map.
            for n in self.cleared:
                self.number_map.pop(int(n), None)
        self.log(f"calibrate: {len(self.number_map)} numbers mapped, {len(self.cells)} cells")
        return dict(self.number_map)

    def _calibration_healthy(self, min_numbers: int = 4) -> bool:
        """True when calibration found enough numbers to trust the run.

        A real board shows most of its numbers at once, so the expectation
        scales with detected cell count (>=35%, bounded to 4..10). Reading far
        fewer means capture/OCR is broken (wrong ROI, DPI mismatch, missing
        fonts); proceeding would let the absent->cleared rule fake a perfect
        run with zero clicks, so callers abort instead.
        """
        with self._map_lock:
            found = len(self.number_map)
        cells = len(self.cells)
        expected = max(min_numbers, int(round(0.35 * cells))) if cells else min_numbers
        expected = min(expected, 10)
        if found >= expected:
            return True
        self.log(
            f"calibration unhealthy: only {found} number(s) mapped from {cells} "
            f"cells (need >= {expected}) — check ROI, game window and OCR"
        )
        return False

    # -- targeted rescan ----------------------------------------------------------
    def rescan_number(self, target: int) -> Optional[Point]:
        """Re-OCR the board to relocate one missing number (cheap targeted scan).

        Re-grabs the board and re-splits using cached geometry when the frame
        shape is unchanged; OCR each cell (parallel) and refresh the whole map
        with anything newly recognised. Returns the target coord or None.
        """
        if scan_roi is None or self.detector is None or build_number_map is None:
            with self._map_lock:
                return self.number_map.get(int(target))
        try:
            roi = self._roi_dict()
            frame = scan_roi(roi)
            self.grid_image = frame
            shape = getattr(frame, "shape", None)
            cached = self.detector.cached_cells
            if cached and shape is not None and len(cached) == len(self.cells):
                cells = cached
            else:
                cells = self.detector.find_cells(frame)
                self.cells = cells
            workers = self._cfg_int("ocr_workers", 4)
            recog = self.recognizer if self._tesseract_ok() else (self._template_recognizer() or self.recognizer)
            fresh = build_number_map(frame, cells, recog, workers=workers) or {}
            ox, oy = self.roi_offset
            with self._map_lock:
                for n, (x, y) in fresh.items():
                    self.number_map[int(n)] = (x + ox, y + oy)
                for n in self.cleared:
                    self.number_map.pop(int(n), None)
                coord = self.number_map.get(int(target))
            self.rescans += 1
            return coord
        except Exception as exc:
            self.log(f"rescan_number({target}) failed: {exc}")
            with self._map_lock:
                return self.number_map.get(int(target))

    def full_rescan(self) -> Dict[int, Point]:
        """Full-board rescan (used after timeout-retry exhaustion / absence)."""
        self.rescans += 1
        try:
            return self.calibrate()
        except Exception as exc:
            self.log(f"full_rescan failed: {exc}")
            with self._map_lock:
                return dict(self.number_map)

    # -- board watcher (optional, scan_frequency_hz > 0) ---------------------------
    def _watcher_loop(self, stop: threading.Event) -> None:
        """Background thread that refreshes number_map between clicks."""
        hz = self._cfg_float("scan_frequency_hz", 0.0)
        period = 1.0 / max(0.1, hz)
        while not stop.wait(period):
            if self._step_busy.is_set() or self._pause.is_set() or self._stop.is_set():
                continue
            try:
                self._refresh_map()
            except Exception as exc:
                self.log(f"watcher: {exc}")

    def _refresh_map(self) -> None:
        """One watcher pass: re-scan board and merge fresh coords (no state reset)."""
        if scan_roi is None or self.detector is None or build_number_map is None:
            return
        roi = self._roi_dict()
        frame = scan_roi(roi)
        cached = self.detector.cached_cells
        cells = cached if cached else self.detector.find_cells(frame)
        workers = self._cfg_int("ocr_workers", 4)
        recog = self.recognizer if self._tesseract_ok() else (self._template_recognizer() or self.recognizer)
        fresh = build_number_map(frame, cells, recog, workers=workers) or {}
        ox, oy = self.roi_offset
        with self._map_lock:
            merged = {n: (x + ox, y + oy) for n, (x, y) in fresh.items()}
            for n in self.cleared:
                merged.pop(int(n), None)
            self.number_map = merged

    # -- single-number step ---------------------------------------------------------
    def step_number(self, target: int, max_attempts: Optional[int] = None) -> str:
        """Click + verify exactly one number. Returns CLEARED / STUCK / NOT_FOUND.

        Guarantees: returns CLEARED only after the cell verified cleared
        (pixel diff, OCR tie-break, or post-hoc diff). The caller's counter
        must NOT advance on STUCK/NOT_FOUND.
        """
        target = int(target)
        if max_attempts is None:
            max_attempts = max(1, self._cfg_int("max_verify_attempts", 2))
        self._step_busy.set()
        try:
            with self._map_lock:
                coord = self.number_map.get(target)
            if coord is None:
                coord = self.rescan_number(target)  # targeted rescan, not full loop.
            if coord is None:
                self.misses += 1
                return NOT_FOUND
            if self.clicker is None or wait_for_clear is None or cell_snapshot is None:
                raise RuntimeError("step_number: click/verify backend unavailable.")
            timeout_ms = self._cfg_int("clear_timeout_ms", 1200)
            poll_ms = self._cfg_int("clear_poll_ms", 8)
            diff_th = self._cfg_float("clear_diff_threshold", 12.0)
            reaction_ms = self._cfg_int("reaction_buffer_ms", 20)
            verify_size = self._cfg_int("verify_size", 44)
            cx, cy = int(coord[0]), int(coord[1])

            for attempt in range(1, max_attempts + 1):
                if self._stop.is_set():
                    return STUCK
                # Enforce the minimum inter-click gap (click_delay_ms) measured
                # from the previous press, so pacing survives move/press split.
                pacing_ms = self._cfg_int("click_delay_ms", 30)
                if pacing_ms > 0 and self._last_press_t is not None:
                    since = (time.perf_counter() - self._last_press_t) * 1000.0
                    if since < pacing_ms:
                        time.sleep((pacing_ms - since) / 1000.0)
                # Move cursor FIRST so the pointer is inside the "before"
                # snapshot; otherwise the cursor itself would fake a diff.
                try:
                    self.clicker.move(cx, cy)
                except Exception as exc:
                    self.log(f"[{target}] move failed: {exc}")
                    return STUCK
                try:
                    before = cell_snapshot(cx, cy, size=verify_size)
                except Exception as exc:
                    self.log(f"[{target}] snapshot failed: {exc}")
                    before = None
                try:
                    self.clicker.press()
                    self._last_press_t = time.perf_counter()
                except Exception as exc:
                    self.log(f"[{target}] click failed: {exc}")
                    return STUCK
                if reaction_ms > 0:
                    time.sleep(reaction_ms / 1000.0)
                cleared = False
                try:
                    cleared = bool(wait_for_clear(cx, cy, before, timeout_ms, poll_ms, diff_th))
                except Exception:
                    cleared = False
                if cleared:
                    return self._mark_cleared(target)
                # --- timeout path: prove what happened ----------------------
                if ocr_gone_check is not None:
                    gone: Optional[bool] = None
                    tie_recog = self.recognizer if self._tesseract_ok() else self._template_recognizer()
                    try:
                        gone = ocr_gone_check(cx, cy, target, size=verify_size, recognizer=tie_recog)
                    except Exception:
                        gone = None
                    if gone is True:
                        # Diff missed it (e.g. subtle gray-out) but OCR proves gone.
                        return self._mark_cleared(target)
                    if gone is False:
                        # Definitely still showing `target` -> retry if allowed.
                        if attempt < max_attempts:
                            self.retries += 1
                            self.log(f"[{target}] still present, retrying (attempt {attempt + 1})")
                            continue
                        break
                # OCR unavailable/uncertain: late-change diff check.
                if before is not None and is_cleared is not None:
                    try:
                        after = cell_snapshot(cx, cy, size=verify_size)
                        if bool(is_cleared(before, after, diff_th)):
                            return self._mark_cleared(target)
                    except Exception:
                        pass
                if attempt < max_attempts:
                    self.retries += 1
                    continue
            # All attempts exhausted and the cell never verified cleared.
            self.full_rescan()  # board may have re-shuffled; refresh coords.
            return STUCK
        finally:
            self._step_busy.clear()

    def _mark_cleared(self, target: int) -> str:
        """Record a verified-cleared number (idempotent) and return CLEARED."""
        target = int(target)
        if target not in self.cleared:
            self.cleared.append(target)
        with self._map_lock:
            self.number_map.pop(target, None)  # must not click it again.
        return CLEARED

    # -- progress reporting ------------------------------------------------------
    def mapping_boxes(self) -> Dict[int, Tuple[int, int, int, int]]:
        """number -> (x, y, w, h) boxes for the overlay, from cached cell size."""
        w = h = 48
        if self.cells:
            ws = sorted(int(c[2]) for c in self.cells)
            hs = sorted(int(c[3]) for c in self.cells)
            w, h = ws[len(ws) // 2], hs[len(hs) // 2]
        with self._map_lock:
            return {
                int(n): (int(x) - w // 2, int(y) - h // 2, w, h)
                for n, (x, y) in self.number_map.items()
            }

    def _report(self, report: Optional[Callable[[dict], None]], target: int,
                stats_obj=None) -> None:
        """Push one progress snapshot to the caller's callback (best-effort)."""
        if report is None:
            return
        try:
            payload = {
                "target": int(target),
                "cleared": len(self.cleared),
                "detected": len(self.number_map),
                "mapping": self.mapping_boxes(),
                "stats": stats_obj.to_dict() if stats_obj is not None and hasattr(stats_obj, "to_dict") else self.stats(),
            }
            report(payload)
        except Exception:
            pass

    # -- strict per-number clearing ------------------------------------------------
    def _clear_target(self, target: int, stopped: Callable[[], bool],
                      paused_wait: Callable[[], bool], report=None,
                      stats_obj=None) -> str:
        """Drive one number to a proven-cleared state. Returns:
            "ok"    -> number cleared (or skipped with allow_skip); advance.
            "stop"  -> stop requested; halt the run.
            "abort" -> stuck beyond stuck_limit with strict sequencing.
        """
        absent = 0
        stuck = 0
        absent_limit = max(1, self._cfg_int("absent_limit", 2))
        stuck_limit = max(1, self._cfg_int("stuck_limit", 4))
        while True:
            if stopped():
                return "stop"
            if not paused_wait():
                return "stop"  # stop arrived while we were paused.
            status = self.step_number(target)
            if status == CLEARED:
                if stats_obj is not None and hasattr(stats_obj, "record_hit"):
                    try:
                        stats_obj.record_hit()
                    except Exception:
                        pass
                self._report(report, target, stats_obj)
                return "ok"
            if status == NOT_FOUND:
                self.full_rescan()
                with self._map_lock:
                    found = int(target) in self.number_map
                    board_visible = len(self.number_map) > 0
                if found:
                    absent = 0  # rescan found it -> go click it.
                    continue
                if not board_visible:
                    # Capture is blind (bad ROI, window moved, OCR all-failed):
                    # treating numbers as "absent -> cleared" here would fake a
                    # perfect run, so abort with an actionable message instead.
                    self.log(
                        f"[{target}] board not visible during rescan -> ABORTING "
                        "(check ROI / game window / tesseract)"
                    )
                    return "abort"
                absent += 1
                self._report(report, target, stats_obj)
                if absent >= absent_limit:
                    # Not on the board after N consecutive full rescans: the
                    # only consistent explanation is that it is already gone.
                    self.log(
                        f"[{target}] absent from {absent} consecutive rescans -> treating as cleared"
                    )
                    self._mark_cleared(target)
                    if stats_obj is not None and hasattr(stats_obj, "record_hit"):
                        try:
                            stats_obj.record_hit()
                        except Exception:
                            pass
                    return "ok"
                continue
            # STUCK: clicked but could not prove clear.
            stuck += 1
            self.misses += 1
            if stats_obj is not None and hasattr(stats_obj, "record_miss"):
                try:
                    stats_obj.record_miss(retried=True)
                except Exception:
                    pass
            self._report(report, target, stats_obj)
            if stuck >= stuck_limit:
                if bool(self.config.get("allow_skip", False)):
                    self.log(f"[{target}] stuck x{stuck} -> skipping (allow_skip=true)")
                    return "ok"
                self.log(
                    f"[{target}] stuck x{stuck} -> ABORTING run "
                    "(strict sequence: refusing to advance past an unverified number)"
                )
                return "abort"    # -- phase waiting (board refill) --------------------------------------------
    def _wait_for_next_phase(self, next_start: int, stop_check: Callable[[], bool],
                             max_wait_ms: Optional[int] = None) -> bool:
        """Poll the board until numbers >= next_start appear (board refill).

        Many games swap the whole board after a phase: a 5x5 board shows 1..25,
        then (after the last click) replaces everything with 26..50. When the
        current phase finishes, the board can be empty for a while — this
        method rescans until the next phase is visible so the run can continue
        instead of stopping at 25.

        Returns True when numbers >= next_start were detected (number_map is
        refreshed), False on stop/timeout.
        """
        if max_wait_ms is None:
            max_wait_ms = self._cfg_int("phase_wait_timeout_ms", 15000)
        delay = max(0.05, self._cfg_float("phase_poll_interval_s", 0.35) or 0.35)
        deadline = time.perf_counter() + max(1.0, max_wait_ms / 1000.0)
        self.log(f"waiting for next board phase (numbers >= {next_start})...")
        while time.perf_counter() < deadline:
            if stop_check():
                return False
            try:
                self.calibrate()
            except Exception as exc:
                self.log(f"phase-wait rescan failed: {exc}")
            with self._map_lock:
                ready = any(n >= next_start for n in self.number_map)
            if ready:
                return True
            time.sleep(delay)
        self.log(f"next board phase did not appear within {max_wait_ms} ms")
        return False

    # -- main loop --------------------------------------------------------------------
    def run(self, target: Optional[int] = None, start: int = 1, end: Optional[int] = None,
            stop_event: Optional[threading.Event] = None,
            pause_event: Optional[threading.Event] = None,
            report: Optional[Callable[[dict], None]] = None, stats=None) -> Dict:
        """(b-h) Click targets start..end in order, verifying each before next.

        Phases: if the board holds fewer numbers than the target (e.g. a 5x5
        board shows 1..25 but the target is 50), the run is split into
        phases: after clearing up to the board's max, the bot AUTO-RESCANS and
        WAITS for the swapped-in board (26..50) and continues — no manual
        restart needed.

        Args:
            target: final number (used when ``end`` is None). ``run(50)" = 1..50.
            start: first number (default 1).
            end: explicit final number; overrides ``target``.
            stop_event: external stop signal (GUI/hotkeys); checked alongside our own.
            pause_event: external pause signal.
            report: callback receiving progress dicts (for GUI queues).
            stats: optional RunStats-like object; hit/miss recorded into it.

        Returns a stats dict: ``{cleared, total, accuracy, clicks, cps,
        elapsed_s, misses, retries, rescans, target, aborted}``.
        """
        if self.clicker is None:
            raise RuntimeError("run: Clicker backend unavailable.")
        if end is None:
            end = int(target) if target is not None else 50
        self._start_num = int(start)
        self._end_num = int(end)
        self._stop.clear()
        self._pause.clear()
        self.aborted = False
        self.target = int(start)
        self.cleared = []
        self.misses = 0
        self.retries = 0
        self._last_press_t = None

        own_stop, own_pause = self._stop, self._pause

        def stopped() -> bool:
            return own_stop.is_set() or (stop_event is not None and stop_event.is_set())

        def paused() -> bool:
            return own_pause.is_set() or (pause_event is not None and pause_event.is_set())

        def paused_wait() -> bool:
            """Block while paused. Returns True when clear to proceed."""
            while paused():
                if stopped():
                    return False
                time.sleep(0.05)
            return True

        if stats is not None and hasattr(stats, "reset"):
            try:
                stats.reset()
            except Exception:
                pass
        if not self.number_map:
            try:
                self.calibrate()
            except Exception as exc:
                self.log(f"run: initial calibrate failed: {exc}")
        if not self._calibration_healthy():
            self.aborted = True
            self.t_end = time.perf_counter()
            if report is not None:
                try:
                    report({"target": 0, "cleared": 0, "detected": len(self.number_map),
                            "mapping": {}, "state": "CalibrationFailed", "stats": self.stats()})
                except Exception:
                    pass
            return self.stats()
        if self.clicker is not None and hasattr(self.clicker, "reset_stats"):
            try:
                self.clicker.reset_stats()
            except Exception:
                pass
        # Optional background board-watcher (detection thread alongside clicking).
        self._watch_stop.clear()
        if self._cfg_float("scan_frequency_hz", 0.0) > 0:
            self._watcher = threading.Thread(
                target=self._watcher_loop, args=(self._watch_stop,),
                name="NumberBot-watcher", daemon=True,
            )
            self._watcher.start()

        self.t_start = time.perf_counter()
        self.t_end = None
        try:
            phase_start = int(start)
            while phase_start <= int(end) and not stopped():
                # Phase-local end: this board can only contain what it shows now.
                with self._map_lock:
                    max_seen = max(self.number_map.keys()) if self.number_map else 0
                    covered = sum(1 for n in self.number_map if n <= max_seen)
                completeness = (covered / float(max_seen)) if max_seen else 0.0
                phase_end = int(end)
                if max_seen and max_seen < int(end) and completeness >= 0.8:
                    phase_end = min(phase_end, int(max_seen))
                    self.log(
                        f"phase: board max {max_seen} ({covered}/{max_seen} seen) "
                        f"-> clearing {phase_start}..{phase_end}, then auto-rescan"
                    )
                self._end_num = int(phase_end)
                for n in range(phase_start, phase_end + 1):
                    if stopped():
                        break
                    self.target = int(n)
                    outcome = self._clear_target(int(n), stopped, paused_wait, report, stats)
                    if outcome == "stop":
                        break
                    if outcome == "abort":
                        self.aborted = True
                        break
                    # outcome "ok" -> number proven cleared -> advance.
                if self.aborted or stopped():
                    break
                phase_done = phase_end
                if phase_done >= int(end):
                    break  # final phase finished -> run complete.
                # Phase finished but numbers remain: WAIT FOR THE BOARD SWAP.
                next_start = phase_done + 1
                if not self._wait_for_next_phase(next_start, stopped):
                    self.log(f"could not find numbers >= {next_start}; ending run")
                    break
                phase_start = next_start
        finally:
            self.t_end = time.perf_counter()
            self._watch_stop.set()
        if report is not None:
            final = {
                "target": int(self.target),
                "cleared": len(self.cleared),
                "detected": len(self.number_map),
                "mapping": self.mapping_boxes(),
                "state": "Stopped",
                "stats": stats.to_dict() if stats is not None and hasattr(stats, "to_dict") else self.stats(),
            }
            try:
                report(final)
            except Exception:
                pass
        return self.stats()

    # -- async -----------------------------------------------------------------------
    def start_async(self, start: int = 1, end: int = 50) -> threading.Thread:
        """Run :meth:`run` on a daemon thread; returns the Thread object."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("start_async: a run is already in progress.")
        self._thread = threading.Thread(
            target=self.run, kwargs={"start": int(start), "end": int(end)},
            name="NumberBot-run", daemon=True,
        )
        self._thread.start()
        return self._thread

    def join(self, timeout: Optional[float] = None) -> None:
        """Block until the async run thread finishes (noop when not started)."""
        if self._thread is not None:
            self._thread.join(timeout)

    # -- lifecycle ----------------------------------------------------------------------
    def reset(self) -> None:
        """Clear run state for reuse (keeps config + subsystems)."""
        self._stop.clear()
        self._pause.clear()
        self._watch_stop.set()
        with self._map_lock:
            self.number_map = {}
        self.cells = []
        self.target = 1
        self.cleared = []
        self.misses = 0
        self.retries = 0
        self.rescans = 0
        self.aborted = False
        self._start_num, self._end_num = 1, 50
        self._last_press_t = None
        self.t_start = None
        self.t_end = None
        if self.detector is not None:
            try:
                self.detector.clear_cache()
            except Exception:
                pass
        if self.clicker is not None and hasattr(self.clicker, "reset_stats"):
            try:
                self.clicker.reset_stats()
            except Exception:
                pass

    def stats(self) -> Dict:
        """Snapshot of accuracy + speed for the current/last run."""
        total = max(1, self._end_num - self._start_num + 1)
        done = len(self.cleared)
        elapsed = 0.0
        if self.t_start is not None:
            end_t = self.t_end if self.t_end is not None else time.perf_counter()
            elapsed = max(0.0, end_t - self.t_start)
        cps = 0.0
        clicks = 0
        if self.clicker is not None and hasattr(self.clicker, "stats"):
            try:
                cs = self.clicker.stats()
                clicks = int(cs.get("clicks", 0))
                cps = float(cs.get("cps", 0.0))
                if cs.get("elapsed_s", 0) and not elapsed:
                    elapsed = float(cs["elapsed_s"])
            except Exception:
                pass
        return {
            "cleared": done,
            "cleared_list": list(self.cleared),
            "total": total,
            "accuracy": round(float(done) / float(total), 4) if total else 0.0,
            "clicks": clicks,
            "cps": round(float(cps), 3),
            "elapsed_s": round(float(elapsed), 3),
            "misses": int(self.misses),
            "retries": int(self.retries),
            "rescans": int(self.rescans),
            "target": int(self.target),
            "aborted": bool(self.aborted),
        }


__all__ = ["NumberBot", "load_config", "CLEARED", "STUCK", "NOT_FOUND", "DEFAULTS"]
