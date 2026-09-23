"""Entry point: Tkinter GUI (default), realtime mode, or headless loop.

Usage:
    python main.py                     # GUI
    python main.py --realtime          # scan->click-lowest loop (default bot)
    python main.py --realtime --debug-view   # + live scan window
    python main.py --headless          # strict 1..N sequencer (legacy)
    python main.py --headless --target 25 --click-delay-ms 10
    python main.py --calibrate-only --roi 100,100,800,500

Hotkeys (headless/realtime, when `keyboard` is installed):
    F1 = start/resume   F2 = stop   F3 = pause/resume
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import keyboard as _keyboard  # optional global hotkeys
except Exception:
    _keyboard = None


def parse_roi(s: str | None):
    """Parse 'x,y,w,h' -> [x, y, w, h]; None in -> None out."""
    if s is None:
        return None
    try:
        parts = [int(float(p)) for p in s.split(",")]
        if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
            raise ValueError("need x,y,w,h with positive w,h")
        return parts
    except Exception as e:
        raise argparse.ArgumentTypeError(f"invalid --roi {s!r}: {e}")


def build_parser() -> argparse.ArgumentParser:
    """CLI: --gui default, --headless, --target, --click-delay-ms, --roi, --calibrate-only."""
    p = argparse.ArgumentParser(description="Number-sequence clicking bot (1..50)")
    p.add_argument("--gui", action="store_true", help="launch Tkinter GUI (default)")
    p.add_argument("--headless", action="store_true", help="run strict 1..N sequencer without GUI")
    p.add_argument("--realtime", action="store_true",
                   help="realtime mode: rescan constantly, click the lowest number visible")
    p.add_argument("--debug-view", action="store_true",
                   help="show a live window of what the bot sees (realtime mode)")
    p.add_argument("--target", type=int, default=None, help="final number (default 50)")
    p.add_argument("--start", type=int, default=None, help="first number (default 1)")
    p.add_argument("--click-delay-ms", type=int, default=None, help="min gap between clicks (ms)")
    p.add_argument("--allow-skip", action="store_true",
                   help="skip stuck numbers instead of aborting the strict sequence")
    p.add_argument("--roi", type=parse_roi, default=None, help="capture region 'x,y,w,h'")
    p.add_argument("--calibrate-only", action="store_true", help="calibrate ROI, print map, exit")
    return p


def load_config() -> dict:
    """Single source of truth: bot_core.bot.load_config (defaults + config.json)."""
    try:
        from bot_core.bot import load_config as _load
        return _load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))
    except Exception:
        return {"target": 50, "click_delay_ms": 30}


def make_bot(cfg: dict):
    """Instantiate NumberBot(cfg) with TypeError-tolerant fallback."""
    from bot_core.bot import NumberBot
    try:
        return NumberBot(cfg)
    except TypeError:
        return NumberBot()


def calibrate_and_exit(bot, cfg: dict) -> int:
    """Run ROI calibration, print the number map, exit 0."""
    fn = getattr(bot, "calibrate", None)
    if not callable(fn):
        print("calibrate not supported; set --roi x,y,w,h instead.")
        return 1
    try:
        mapping = fn()
    except Exception as e:
        print(f"calibration failed: {e}", file=sys.stderr)
        return 1
    print(f"calibrated: {len(mapping)} numbers mapped")
    for n in sorted(mapping):
        print(f"  {n:>2} -> {mapping[n]}")
    return 0


def run_headless(bot, cfg: dict) -> int:
    """Headless 1..target loop; returns process exit code."""
    stop = threading.Event()
    pause = threading.Event()
    target = int(cfg.get("target", 50))
    start = int(cfg.get("start", 1) or 1)

    # Global hotkeys (best-effort). They flip the same events the bot watches.
    if _keyboard is not None:
        try:
            _keyboard.add_hotkey(str(cfg.get("start_hotkey", "F1")), pause.clear)
            _keyboard.add_hotkey(str(cfg.get("stop_hotkey", "F2")), stop.set)
            _keyboard.add_hotkey(str(cfg.get("pause_hotkey", "F3")),
                                 lambda: pause.set() if not pause.is_set() else pause.clear())
            print(f"Hotkeys: {cfg.get('start_hotkey', 'F1')}=resume "
                  f"{cfg.get('stop_hotkey', 'F2')}=stop {cfg.get('pause_hotkey', 'F3')}=pause")
        except Exception as e:
            print(f"[warn] hotkey registration failed: {e}", file=sys.stderr)
    else:
        print("[info] 'keyboard' not installed; hotkeys disabled (pip install keyboard).")

    def on_progress(msg: dict) -> None:
        try:
            print(f"\r[target {msg.get('target'):>2}] cleared={msg.get('cleared')} "
                  f"detected={msg.get('detected')}  ", end="", flush=True)
        except Exception:
            pass

    rc = 0
    t0 = time.perf_counter()
    try:
        bot.run(target=target, start=start, stop_event=stop, pause_event=pause,
                report=on_progress)
        print()
    except KeyboardInterrupt:
        print("\ninterrupted.")
        bot.stop()
    except Exception as e:
        print(f"\nheadless run failed: {e}", file=sys.stderr)
        rc = 1
    finally:
        try:
            s = bot.stats()
            print(
                f"cleared {s.get('cleared', 0)}/{s.get('total', target)} "
                f"in {s.get('elapsed_s', time.perf_counter() - t0):.2f}s | "
                f"cps={s.get('cps', 0):.1f} accuracy={s.get('accuracy', 0) * 100:.1f}% | "
                f"misses={s.get('misses', 0)} retries={s.get('retries', 0)} "
                f"rescans={s.get('rescans', 0)} aborted={s.get('aborted', False)}"
            )
        except Exception:
            print(f"done target={target} elapsed={time.perf_counter() - t0:.1f}s")
    return rc


def main(argv=None) -> int:
    """Parse args; default (no flags) launches the GUI."""
    args = build_parser().parse_args(argv)
    cfg = load_config()
    if args.target is not None:
        cfg["target"] = args.target
    if args.start is not None:
        cfg["start"] = args.start
    if args.click_delay_ms is not None:
        cfg["click_delay_ms"] = args.click_delay_ms
    if args.allow_skip:
        cfg["allow_skip"] = True
    if args.roi is not None:
        cfg["roi"] = list(args.roi)

    if args.calibrate_only:
        bot = make_bot(cfg)
        return calibrate_and_exit(bot, cfg)

    if args.realtime:
        from bot_core.realtime import RealtimeBot
        import keyboard as _kb
        stop = threading.Event()
        if _kb is not None:
            try:
                _kb.add_hotkey(str(cfg.get("stop_hotkey", "F2")), stop.set)
                _kb.add_hotkey(str(cfg.get("pause_hotkey", "F3")),
                               lambda: None)  # pause handled in-run
                print(f"Hotkeys: {cfg.get('stop_hotkey', 'F2')}=stop "
                      f"{cfg.get('pause_hotkey', 'F3')}=pause")
            except Exception as e:
                print(f"[warn] hotkeys unavailable: {e}", file=sys.stderr)
        bot = RealtimeBot(cfg, debug=args.debug_view)
        # Realtime sessions run until F2/Ctrl+C by default (refill games never
        # end). A target only applies when explicitly given via --target N.
        tgt = args.target
        print(f"realtime mode: ROI={cfg.get('roi')} "
              f"target={tgt if tgt is not None else 'none (runs until stop)'}")
        try:
            stats = bot.run(target=tgt)
            print(f"done: {stats}")
        except KeyboardInterrupt:
            bot.stop()
            print("stopped.")
        return 0

    if args.headless:
        bot = make_bot(cfg)
        return run_headless(bot, cfg)

    # Default: GUI.
    try:
        from bot_core.gui import launch
    except Exception as e:
        print(f"ERROR: GUI unavailable ({e})", file=sys.stderr)
        return 1
    try:
        launch(cfg)
    except RuntimeError as e:  # e.g. no tkinter
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
