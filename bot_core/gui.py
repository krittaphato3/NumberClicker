"""Tkinter settings panel + click-through overlay for the number-clicking bot.

Wiring:
    * Start/Pause/Stop buttons and F1/F2/F3 (window bindings + global hooks
      when `keyboard` is installed) all drive the same threading.Events that
      NumberBot watches, so hotkeys work while the game window has focus.
    * The bot runs on a worker thread and reports progress dicts into a
      queue; the Tk main thread drains the queue at STATUS_POLL_MS.
    * Config keys are canonical (bot_core.bot.DEFAULTS): click_delay_ms,
      reaction_buffer_ms, confidence_threshold, clear_timeout_ms, ...
    * The overlay is a transparent fullscreen window. On Windows we use
      -transparentcolor so transparent pixels are CLICK-THROUGH (the bot's
      own clicks must land on the game, not on our overlay). On other
      platforms a small window with -alpha is used and users may need to
      hide the overlay during runs.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time

# --- Guarded stdlib GUI imports ---
try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except Exception:  # headless env without Tk
    tk = None  # type: ignore
    messagebox = ttk = None  # type: ignore

try:
    import keyboard as _keyboard  # global hotkeys (optional)
except Exception:
    _keyboard = None

from bot_core.stats import RunStats

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json"
)

# Canonical defaults mirror bot_core.bot.DEFAULTS (kept in sync there).
GUI_DEFAULTS = {
    "click_delay_ms": 30,
    "reaction_buffer_ms": 20,
    "confidence_threshold": 0.6,
    "grid_mode": "auto",
    "grid_rows": 5,
    "grid_cols": 10,
    "roi": None,                # [x, y, w, h] or None
    "scan_frequency_hz": 0,     # background re-scan rate; 0 = off (fastest)
    "clear_timeout_ms": 1200,
    "clear_poll_ms": 8,
    "clear_diff_threshold": 12.0,
    "max_verify_attempts": 2,
    "stuck_limit": 4,
    "absent_limit": 2,
    "allow_skip": False,
    "humanize": False,
    "smooth_move": False,
    "tesseract_cmd": "",
    "debug_overlay": True,
    "target": 50,
}
GRID_CHOICES = ("auto", "manual")
STATUS_POLL_MS = 100


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load config.json defensively; missing/corrupt -> defaults copy."""
    cfg = dict(GUI_DEFAULTS)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cfg.update(data)
    except Exception:
        pass
    return cfg


def save_config(cfg: dict, path: str = CONFIG_PATH) -> None:
    """Write config.json preserving unknown keys already on disk."""
    merged: dict = {}
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, dict):
                merged.update(old)
    except Exception:
        pass
    merged.update(cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)


def _require_tk() -> None:
    if tk is None:
        raise RuntimeError("tkinter is not available in this environment")


# --- ROI drag-select -------------------------------------------------------
def select_roi(parent=None) -> tuple | None:
    """Fullscreen drag-select; returns (x, y, w, h) or None if cancelled."""
    _require_tk()
    result: dict = {}
    sel = tk.Toplevel(parent) if parent is not None else tk.Toplevel()
    sel.attributes("-fullscreen", True)
    sel.attributes("-alpha", 0.35)
    sel.attributes("-topmost", True)
    sel.configure(cursor="crosshair")
    canvas = tk.Canvas(sel, highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    start: dict = {}
    rect_id = [None]

    def on_down(e):
        start["x"], start["y"] = e.x_root, e.y_root
        rect_id[0] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=2)

    def on_drag(e):
        if rect_id[0] is not None:
            x0 = sel.winfo_rootx()
            y0 = sel.winfo_rooty()
            canvas.coords(rect_id[0], start["x"] - x0, start["y"] - y0, e.x, e.y)

    def on_up(e):
        x0, y0 = min(start["x"], e.x_root), min(start["y"], e.y_root)
        w, h = abs(e.x_root - start["x"]), abs(e.y_root - start["y"])
        result["roi"] = (x0, y0, w, h) if w > 4 and h > 4 else None
        sel.destroy()

    def on_esc(_e=None):
        result["roi"] = None
        sel.destroy()

    sel.bind("<ButtonPress-1>", on_down)
    sel.bind("<B1-Motion>", on_drag)
    sel.bind("<ButtonRelease-1>", on_up)
    sel.bind("<Escape>", on_esc)
    sel.grab_set()
    sel.wait_window(sel)
    return result.get("roi")


# --- Transparent click-through overlay --------------------------------------
class Overlay:
    """Topmost overlay drawing detected-cell boxes + target highlight.

    Windows: fullscreen window whose background color is fully transparent
    AND click-through (-transparentcolor). Canvas boxes/labels are drawn in
    opaque colors, so they show but clicks pass through the empty areas.
    Non-Windows: tiny alpha window centered on the mapped cells (may catch
    clicks; use the Overlay checkbox to disable during runs).
    """

    def __init__(self, master=None):
        self._master = master
        self._win = None
        self._canvas = None
        self._pending = None  # (mapping, target)
        self.enabled = True
        self._geom_cache = ""
        self._is_windows = os.name == "nt"

    def _ensure(self):
        if tk is None or self._master is None:
            return False
        if self._win is None:
            try:
                self._win = tk.Toplevel(self._master)
                self._win.overrideredirect(True)
                self._win.attributes("-topmost", True)
                if self._is_windows:
                    self._win.config(bg="cyan")  # the magic transparent key
                    self._win.attributes("-transparentcolor", "cyan")
                    self._canvas = tk.Canvas(self._win, bg="cyan", highlightthickness=0)
                    # Cover the whole virtual screen so any cell is drawable.
                    self._win.geometry(f"{self._master.winfo_screenwidth()}x"
                                       f"{self._master.winfo_screenheight()}+0+0")
                else:
                    self._win.attributes("-alpha", 0.35)
                    self._win.configure(bg="black")
                    self._canvas = tk.Canvas(self._win, bg="black", highlightthickness=0)
                self._canvas.pack(fill="both", expand=True)
                self._win.withdraw()
            except Exception:
                self._win = None
                return False
        return True

    def show(self, mapping, target=None) -> None:
        """Queue a redraw; mapping: {num:(x,y,w,h)} (absolute screen coords)."""
        self._pending = (mapping, target)
        if self._master is not None and tk is not None:
            try:
                self._master.after(0, self._render)
            except Exception:
                pass

    def hide(self) -> None:
        self._pending = None
        try:
            if self._win is not None:
                self._win.withdraw()
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self._win is not None:
                self._win.destroy()
        except Exception:
            pass
        self._win = None

    def _render(self) -> None:
        if not self.enabled or self._pending is None:
            return
        if not self._ensure():
            return
        mapping, target = self._pending
        try:
            items = [(int(n), int(x), int(y), int(w), int(h))
                     for n, (x, y, w, h) in dict(mapping or {}).items()]
        except Exception:
            items = []
        try:
            if not items:
                if self._win is not None:
                    self._win.withdraw()
                return
            if not self._is_windows:
                # Non-Windows fallback: small window hugging the mapped cells.
                xs = [i[1] for i in items]; ys = [i[2] for i in items]
                xe = [i[1] + i[3] for i in items]; ye = [i[2] + i[4] for i in items]
                x0, y0 = min(xs) - 10, min(ys) - 10
                geom = f"{max(xe) - x0 + 10}x{max(ye) - y0 + 10}+{x0}+{y0}"
                if geom != self._geom_cache:
                    self._win.geometry(geom)
                    self._geom_cache = geom
            self._win.deiconify()
            self._win.attributes("-topmost", True)
            c = self._canvas
            c.delete("all")
            for num, x, y, w, h in items:
                color = "lime" if (target is not None and num == target) else "#ff3333"
                if self._is_windows:
                    ox, oy = x, y  # absolute screen coords; window is full-screen.
                else:
                    ox, oy = x - (min(xs) - 10), y - (min(ys) - 10)
                c.create_rectangle(ox, oy, ox + w, oy + h, outline=color, width=2)
                c.create_text(ox + 3, oy + 3, text=str(num), fill=color, anchor="nw")
        except Exception:
            pass


# --- Main GUI --------------------------------------------------------------
class BotGui:
    """Settings panel; bot runs in a worker thread, GUI polls a queue."""

    def __init__(self, root, config: dict | None = None):
        _require_tk()
        self.root = root
        self.cfg = config or load_config()
        self.queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.bot = None
        self.state = "Stopped"
        self.stats = RunStats()
        self.overlay = Overlay(root)
        self.overlay.enabled = bool(self.cfg.get("debug_overlay", True))
        self._build_widgets()
        self._bind_hotkeys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(STATUS_POLL_MS, self._poll)

    # -- widgets --
    def _build_widgets(self) -> None:
        self.root.title("Number Click Bot (1-50)")
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill="both", expand=True)
        self.vars: dict = {}
        row = 0

        def add_scale_entry(label, key, lo, hi, is_float=False):
            nonlocal row
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w")
            default = GUI_DEFAULTS.get(key, 0)
            if is_float:
                var = tk.DoubleVar(value=float(self.cfg.get(key, default)))
            else:
                var = tk.IntVar(value=int(self.cfg.get(key, default) or 0))
            ttk.Scale(frm, from_=lo, to=hi, variable=var, orient="horizontal",
                      length=170).grid(row=row, column=1, sticky="ew")
            ttk.Entry(frm, textvariable=var, width=8).grid(row=row, column=2, padx=4)
            self.vars[key] = var
            row += 1

        add_scale_entry("Click delay (ms)", "click_delay_ms", 0, 2000)
        add_scale_entry("Reaction buffer (ms)", "reaction_buffer_ms", 0, 1000)
        add_scale_entry("OCR confidence", "confidence_threshold", 0.0, 1.0, is_float=True)
        add_scale_entry("Clear timeout (ms)", "clear_timeout_ms", 100, 5000)
        add_scale_entry("Clear diff threshold", "clear_diff_threshold", 1, 60, is_float=True)
        add_scale_entry("Verify size (px)", "verify_size", 20, 120)
        add_scale_entry("Watcher scan (Hz, 0=off)", "scan_frequency_hz", 0, 10, is_float=True)

        ttk.Label(frm, text="Grid").grid(row=row, column=0, sticky="w")
        gvar = tk.StringVar(value=str(self.cfg.get("grid_mode", "auto")))
        ttk.Combobox(frm, textvariable=gvar, values=list(GRID_CHOICES),
                     state="readonly", width=10).grid(row=row, column=1, sticky="w")
        self.vars["grid_mode"] = gvar
        row += 1

        for key, label in (("humanize", "Humanize (jitter+delay variance)"),
                           ("smooth_move", "Smooth cursor move"),
                           ("debug_overlay", "Show overlay")):
            v = tk.BooleanVar(value=bool(self.cfg.get(key, False)))
            ttk.Checkbutton(frm, text=label, variable=v).grid(
                row=row, column=0, columnspan=3, sticky="w")
            self.vars[key] = v
            row += 1

        ttk.Label(frm, text="Target").grid(row=row, column=0, sticky="w")
        tvar = tk.IntVar(value=int(self.cfg.get("target", 50) or 50))
        ttk.Entry(frm, textvariable=tvar, width=8).grid(row=row, column=1, sticky="w")
        self.vars["target"] = tvar
        row += 1

        self.roi_label = ttk.Label(frm, text=f"ROI: {self.cfg.get('roi')}")
        self.roi_label.grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=3, pady=6)
        ttk.Button(btns, text="Select ROI", command=self.on_calibrate).pack(side="left", padx=2)
        ttk.Button(btns, text="Scan board", command=self.on_scan).pack(side="left", padx=2)
        ttk.Button(btns, text="Start (F1)", command=self.on_start).pack(side="left", padx=2)
        ttk.Button(btns, text="Pause/Resume (F3)", command=self.on_pause).pack(side="left", padx=2)
        ttk.Button(btns, text="Stop (F2)", command=self.on_stop).pack(side="left", padx=2)
        ttk.Button(btns, text="Save Config", command=self.on_save).pack(side="left", padx=2)
        row += 1

        self.status_vars = {k: tk.StringVar(value="-") for k in
                            ("Status", "Current target", "Numbers detected",
                             "CPS", "Elapsed", "Accuracy", "Cleared", "Note")}
        for k, v in self.status_vars.items():
            f = ttk.Frame(frm)
            f.grid(row=row, column=0, columnspan=3, sticky="w")
            ttk.Label(f, text=f"{k}: ").pack(side="left")
            ttk.Label(f, textvariable=v).pack(side="left")
            row += 1

        frm.columnconfigure(1, weight=1)

    # -- hotkeys --
    def _bind_hotkeys(self) -> None:
        try:
            self.root.bind("<F1>", lambda _e: self.on_start())
            self.root.bind("<F2>", lambda _e: self.on_stop())
            self.root.bind("<F3>", lambda _e: self.on_pause())
        except Exception:
            pass
        if _keyboard is not None:  # global hooks fire even when game is focused
            try:
                _keyboard.add_hotkey("f1", self.on_start)
                _keyboard.add_hotkey("f2", self.on_stop)
                _keyboard.add_hotkey("f3", self.on_pause)
            except Exception:
                pass

    # -- config helpers --
    def _collect(self) -> dict:
        cfg = dict(self.cfg)
        try:
            cfg["click_delay_ms"] = int(float(self.vars["click_delay_ms"].get()))
            cfg["reaction_buffer_ms"] = int(float(self.vars["reaction_buffer_ms"].get()))
            cfg["confidence_threshold"] = max(0.0, min(1.0, float(self.vars["confidence_threshold"].get())))
            cfg["clear_timeout_ms"] = int(float(self.vars["clear_timeout_ms"].get()))
            cfg["clear_diff_threshold"] = float(self.vars["clear_diff_threshold"].get())
            cfg["verify_size"] = int(float(self.vars["verify_size"].get()))
            cfg["scan_frequency_hz"] = float(self.vars["scan_frequency_hz"].get())
            cfg["grid_mode"] = str(self.vars["grid_mode"].get())
            cfg["humanize"] = bool(self.vars["humanize"].get())
            cfg["smooth_move"] = bool(self.vars["smooth_move"].get())
            cfg["debug_overlay"] = bool(self.vars["debug_overlay"].get())
            cfg["target"] = max(1, int(float(self.vars["target"].get())))
        except Exception:
            pass
        return cfg

    # -- buttons --
    def on_calibrate(self) -> None:
        """Drag-select the game grid ROI (screen coords)."""
        try:
            self.root.withdraw()
            time.sleep(0.2)
            roi = select_roi(None)
        finally:
            try:
                self.root.deiconify()
            except Exception:
                pass
        if roi:
            self.cfg["roi"] = list(roi)
            self.roi_label.config(text=f"ROI: {list(roi)}")
            try:
                save_config(self._collect())
            except Exception:
                pass

    def on_scan(self) -> None:
        """One full OCR sweep to preview what the bot can see."""
        def _scan():
            try:
                from bot_core.bot import NumberBot
                bot = NumberBot(self._collect())
                mapping = bot.calibrate()
                self.queue.put({"mapping": {n: (x, y, 48, 48) for n, (x, y) in mapping.items()},
                                "target": None, "detected": len(mapping),
                                "note": f"scan: {len(mapping)} numbers, {len(bot.cells)} cells"})
            except Exception as e:
                self.queue.put({"error": f"scan failed: {e}"})
        threading.Thread(target=_scan, daemon=True).start()

    def on_save(self) -> None:
        self.cfg = self._collect()
        try:
            save_config(self.cfg)
            if messagebox is not None:
                messagebox.showinfo("Saved", "config.json saved.")
        except Exception as e:
            if messagebox is not None:
                messagebox.showerror("Error", f"Save failed: {e}")

    def on_start(self) -> None:
        if self.state == "Running":
            return
        if self.state == "Paused":
            self.on_pause()  # resume
            return
        self.cfg = self._collect()
        self.overlay.enabled = bool(self.cfg.get("debug_overlay", True))
        self.stop_event.clear()
        self.pause_event.clear()
        self.stats = RunStats()
        self.state = "Running"
        self.status_vars["Note"].set("")
        self.worker = threading.Thread(target=self._run_bot, daemon=True)
        self.worker.start()

    def on_pause(self) -> None:
        if self.state == "Running":
            self.pause_event.set()
            if self.bot is not None:
                try:
                    self.bot.pause()
                except Exception:
                    pass
            self.state = "Paused"
        elif self.state == "Paused":
            self.pause_event.clear()
            if self.bot is not None:
                try:
                    self.bot.resume()
                except Exception:
                    pass
            self.state = "Running"

    def on_stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        if self.bot is not None:
            try:
                self.bot.stop()
            except Exception:
                pass
        self.state = "Stopped"
        self.overlay.hide()

    # -- worker --
    def _run_bot(self) -> None:
        try:
            from bot_core.bot import NumberBot
            bot = NumberBot(dict(self.cfg))
        except Exception as e:
            self.queue.put({"error": f"NumberBot init failed: {e}"})
            self.queue.put({"state": "Stopped"})
            return
        self.bot = bot
        target = int(self.cfg.get("target", 50) or 50)
        try:
            bot.run(target=target, stop_event=self.stop_event,
                    pause_event=self.pause_event, report=self.queue.put,
                    stats=self.stats)
            self.queue.put({"note": f"run finished: {len(bot.cleared)}/{target} cleared"
                             + (" (ABORTED: stuck number)" if bot.aborted else "")})
        except Exception as e:
            self.queue.put({"error": str(e)})
        finally:
            self.queue.put({"state": "Stopped"})
            self.bot = None

    # -- polling (GUI thread only) --
    def _poll(self) -> None:
        try:
            while True:
                msg = self.queue.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass
        except Exception:
            pass
        self.status_vars["Status"].set(self.state)
        self.status_vars["CPS"].set(f"{self.stats.cps():.2f}")
        self.status_vars["Elapsed"].set(f"{self.stats.elapsed():.1f}s")
        self.status_vars["Accuracy"].set(f"{self.stats.accuracy() * 100:.1f}%")
        try:
            self.root.after(STATUS_POLL_MS, self._poll)
        except Exception:
            pass

    def _handle_msg(self, msg: dict) -> None:
        if not isinstance(msg, dict):
            return
        if "error" in msg:
            if messagebox is not None:
                try:
                    messagebox.showerror("Bot error", str(msg["error"]))
                except Exception:
                    pass
            self.state = "Stopped"
        if "note" in msg:
            self.status_vars["Note"].set(str(msg["note"]))
        if "stats" in msg and isinstance(msg["stats"], dict):
            s = msg["stats"]
            for k in ("clicks", "hits", "misses", "retries"):
                if k in s:
                    try:
                        setattr(self.stats, k, int(s[k]))
                    except Exception:
                        pass
        if "target" in msg and msg["target"] is not None:
            self.status_vars["Current target"].set(str(msg["target"]))
        if "detected" in msg:
            self.status_vars["Numbers detected"].set(str(msg["detected"]))
        if "cleared" in msg:
            self.status_vars["Cleared"].set(str(msg["cleared"]))
        mapping = msg.get("mapping")
        if mapping and self.overlay.enabled and tk is not None:
            try:
                self.overlay.show(mapping, msg.get("target"))
            except Exception:
                pass
        if msg.get("state") in ("Stopped", "Running", "Paused"):
            self.state = msg["state"]
            if self.state == "Stopped":
                self.overlay.hide()

    def _on_close(self) -> None:
        try:
            self.on_stop()
        except Exception:
            pass
        try:
            if self.overlay is not None:
                self.overlay.close()
        except Exception:
            pass
        self.root.destroy()


def launch(config: dict | None = None) -> None:
    """Open the settings panel (blocking)."""
    _require_tk()
    root = tk.Tk()
    root.withdraw()  # hide the bare root; BotGui manages its own widgets on root
    try:
        root.deiconify()
    except Exception:
        pass
    BotGui(root, config)
    root.mainloop()


if __name__ == "__main__":
    launch()
