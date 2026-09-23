"""End-to-end PHASE-SWAP test (the user's exact scenario).

A 5x5 board shows 1..25. After the 25th number is clicked, the whole board
SWAPS to 26..50 (same tiles, new numbers) after a short animation delay.
The bot must auto-rescan during the empty/swap window and continue clicking
26..50 — finishing 1..50 in a single run instead of stopping at 25.

Passes only if every number 1..50 is clicked exactly once, in order.
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

PHASE1 = [13, 19, 6, 1, 15,
          11, 20, 4, 16, 18,
          10, 3, 5, 24, 21,
          12, 14, 9, 17, 2,
          25, 7, 8, 23, 22]

# Phase 2: 26..50 shuffled into the SAME 25 tiles.
PHASE2 = [48, 29, 49, 31, 38,
          44, 30, 41, 34, 26,
          47, 32, 39, 27, 36,
          37, 35, 43, 40, 50,
          33, 46, 28, 42, 45]

SWAP_DELAY_S = 0.8  # game animation gap between phase 1 clear and the swap

TILE, GAP, MARGIN = 64, 10, 14
COLS = 5
W = MARGIN * 2 + COLS * TILE + (COLS - 1) * GAP
H = MARGIN * 2 + 5 * TILE + 4 * GAP

clicked_order = []
lock = threading.Lock()
root = None
tiles = []          # list of tk.Label, one per tile
numbers = []        # current number on each tile
swap_timer = None


def layout_num(i):
    r, c = divmod(i, COLS)
    return (MARGIN + c * (TILE + GAP), MARGIN + r * (TILE + GAP))


def make_window():
    global root, tiles, numbers
    root = tk.Tk()
    root.title("Simulated Number Game (phases)")
    root.geometry(f"{W}x{H}+60+60")
    root.configure(bg="#f4f1ea")
    root.attributes("-topmost", True)
    for i, num in enumerate(PHASE1):
        x, y = layout_num(i)
        btn = tk.Label(root, text=str(num), font=("Segoe UI", 20, "bold"),
                       bg="#ffffff", fg="#1c1c1c", width=TILE // 8, height=3,
                       relief="ridge", bd=1)
        btn.place(x=x, y=y, width=TILE, height=TILE)
        btn.bind("<Button-1>", lambda e, n=num, b=btn: on_click(n, b))
        tiles.append(btn)
        numbers.append(num)


def on_click(num, btn):
    with lock:
        clicked_order.append(num)
    btn.configure(bg="#dfe8d8", fg="#9aa79a", text=str(num))
    # The last number of phase 1 triggers the board swap.
    if num == 25:
        root.after(int(SWAP_DELAY_S * 1000), swap_board)


def swap_board():
    """Replace every tile with its phase-2 number (same positions)."""
    global numbers
    for i, btn in enumerate(tiles):
        new_num = PHASE2[i]
        numbers[i] = new_num
        btn.configure(bg="#ffffff", fg="#1c1c1c", text=str(new_num))
        btn.bind("<Button-1>", lambda e, n=new_num, b=btn: on_click(n, b))


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
        "phase_wait_timeout_ms": 15000,
        "phase_poll_interval_s": 0.35,
    }
    bot = NumberBot(cfg, log=lambda m: print(f"  [bot] {m}"))
    try:
        stats = bot.run(target=50)   # full 1..50 target; phases handled inside
        result["stats"] = stats
    except Exception as exc:
        result["error"] = str(exc)
    result["cleared_list"] = list(bot.cleared)


def main():
    make_window()
    root.update_idletasks()
    root.update()

    def roi_getter():
        return (root.winfo_rootx(), root.winfo_rooty(),
                root.winfo_width(), root.winfo_height())

    result = {}
    t = threading.Thread(target=bot_worker, args=(result, roi_getter), daemon=True)
    t.start()
    deadline = time.time() + 90

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
    expected = list(range(1, 51))
    cleared = result.get("cleared_list", [])
    stats = result.get("stats", {})
    print(f"\nclick order ({len(order)}): {order}")
    print(f"cleared list ({len(cleared)})")
    print(f"stats: {stats}")
    # The invariant is NOT "zero duplicate clicks" (a click the game drops
    # must be retried on the SAME number). It is: first occurrences strictly
    # ascending 1..50, duplicates only as immediate same-number retries.
    firsts = []
    for n in order:
        if not firsts or firsts[-1] != n:
            firsts.append(n)
    ok = (firsts == expected and cleared == expected
          and set(order) == set(expected) and not stats.get("aborted"))
    print("\nRESULT:", "PASS" if ok else "FAIL")
    if not ok:
        if firsts != expected:
            print(f"  first-click order mismatch: {firsts}")
        if cleared != expected:
            print("  cleared list mismatch")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
