"""End-to-end live test on this Windows machine.

Puts a real 25-tile game window on screen (Tk), then runs the REAL NumberBot:
real screen capture, real OCR, real SendInput clicks, real clear-verification.
Fails if any number is clicked out of order or any click fails to clear.
"""
import os
import sys
import threading
import time
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot_core.bot import NumberBot
from bot_core.detector import ensure_dpi_awareness

ensure_dpi_awareness()

BOARD = [13, 19, 6, 1, 15,
         11, 20, 4, 16, 18,
         10, 3, 5, 24, 21,
         12, 14, 9, 17, 2,
         25, 7, 8, 23, 22]

TILE, GAP, MARGIN = 64, 10, 14
COLS, ROWS = 5, 5
W = MARGIN * 2 + COLS * TILE + (COLS - 1) * GAP
H = MARGIN * 2 + ROWS * TILE + (ROWS - 1) * GAP

clicked_order = []
lock = threading.Lock()
root = None
tiles = []


def make_window():
    global root, tiles
    root = tk.Tk()
    root.title("Simulated Number Game")
    root.geometry(f"{W}x{H}+60+60")
    root.configure(bg="#f4f1ea")
    root.attributes("-topmost", True)
    for i, num in enumerate(BOARD):
        r, c = divmod(i, COLS)
        x = MARGIN + c * (TILE + GAP)
        y = MARGIN + r * (TILE + GAP)
        btn = tk.Label(root, text=str(num), font=("Segoe UI", 20, "bold"),
                       bg="#ffffff", fg="#1c1c1c", width=TILE // 8, height=3,
                       relief="ridge", bd=1)
        btn.place(x=x, y=y, width=TILE, height=TILE)
        btn.bind("<Button-1>", lambda e, n=num, b=btn: on_click(n, b))
        tiles.append(btn)


def on_click(num, btn):
    with lock:
        clicked_order.append(num)
    btn.configure(bg="#dfe8d8", fg="#9aa79a", text=str(num))  # game 'clears' it


def bot_worker(result, roi_getter):
    time.sleep(1.2)  # let the window render
    rx, ry, rw, rh = roi_getter()
    cfg = {
        "roi": [rx, ry, rw, rh],
        "grid_mode": "auto",
        "click_delay_ms": 15,
        "reaction_buffer_ms": 15,
        "clear_timeout_ms": 900,
        "clear_poll_ms": 8,
        "humanize": False,
        "smooth_move": False,
        "allow_skip": False,
        "stuck_limit": 3,
        "absent_limit": 2,
        "ocr_workers": 4,
        "confidence_threshold": 0.3,
    }
    bot = NumberBot(cfg, log=lambda m: print(f"  [bot] {m}"))
    try:
        stats = bot.run(start=1, end=25)
        result["stats"] = stats
    except Exception as exc:
        result["error"] = str(exc)
    result["cleared_list"] = list(bot.cleared)


def main():
    make_window()
    # Real client-area geometry (excludes the title bar), valid after layout.
    root.update_idletasks()
    root.update()

    def roi_getter():
        return (root.winfo_rootx(), root.winfo_rooty(),
                root.winfo_width(), root.winfo_height())

    result = {}
    t = threading.Thread(target=bot_worker, args=(result, roi_getter), daemon=True)
    t.start()
    # Run Tk for at most 60s; closes itself when the bot finishes or errors.
    deadline = time.time() + 60

    def poll():
        if result.get("stats") or result.get("error") or time.time() > deadline:
            root.destroy()
            return
        root.after(100, poll)
    root.after(100, poll)
    root.mainloop()
    t.join(timeout=5)

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return 1
    order = clicked_order
    expected = list(range(1, 26))
    cleared = result.get("cleared_list", [])
    stats = result.get("stats", {})
    print(f"\nclick order ({len(order)}): {order}")
    print(f"cleared list: {cleared}")
    print(f"stats: {stats}")
    ok = order == expected and cleared == expected and not stats.get("aborted")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    if not ok:
        if order != expected:
            print("  click order mismatch")
        if cleared != expected:
            print("  cleared list mismatch")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
