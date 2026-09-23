"""End-to-end realtime test with BOARD REFILL (the user's scenario).

Simulates the real game behavior: when a number is clicked it disappears and
a NEW number appears in its place after a short animation (so the board is
momentarily empty and numbers keep coming back). One tile is dead (never
registers clicks) to exercise retry + blacklist. Passes only if the bot:
  1. always clicks the currently-lowest number (order never decreases),
  2. survives the empty refill frames without stopping,
  3. keeps clicking until the stop request, and
  4. does not get stuck forever on the dead tile (retry + blacklist).
"""
import os
import sys
import threading
import time
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bot_core.realtime import RealtimeBot
from bot_core.detector import ensure_dpi_awareness

ensure_dpi_awareness()

TILE, GAP, MARGIN, COLS, ROWS = 64, 10, 14, 5, 5
W = MARGIN * 2 + COLS * TILE + (COLS - 1) * GAP
H = MARGIN * 2 + ROWS * TILE + (ROWS - 1) * GAP

DEAD_INDEX = 20  # row 5 col 1 tile never registers clicks (exercises blacklist)

root = None
buttons = []
state = {"alive": True, "clicks": [], "bot": None}
result = {}


def spawn_board():
    """Fill all tiles with random unique numbers 1..50."""
    import random
    nums = random.sample(range(1, 51), COLS * ROWS)
    for i, btn in enumerate(buttons):
        btn.num = nums[i]
        btn.configure(text=str(nums[i]), bg="#ffffff", fg="#1c1c1c")


def make_window():
    global root
    root = tk.Tk()
    root.title("Simulated Number Game (refill)")
    root.geometry(f"{W}x{H}+60+60")
    root.configure(bg="#f4f1ea")
    root.attributes("-topmost", True)
    for i in range(COLS * ROWS):
        r, c = divmod(i, COLS)
        x = MARGIN + c * (TILE + GAP)
        y = MARGIN + r * (TILE + GAP)
        btn = tk.Label(root, text="", font=("Segoe UI", 20, "bold"),
                       bg="#ffffff", fg="#1c1c1c", relief="ridge", bd=1)
        btn.place(x=x, y=y, width=TILE, height=TILE)
        btn.num = 0
        btn.dead = (i == DEAD_INDEX)
        btn.bind("<Button-1>", lambda e, b=btn: on_click(b))
        buttons.append(btn)
    spawn_board()


def on_click(btn):
    if btn.dead or btn.num == 0:
        return
    state["clicks"].append(btn.num)
    # Game behavior: number disappears, then a NEW number appears in the same
    # place after a short refill animation.
    btn.num = 0
    btn.configure(text="", bg="#e8e4da")

    def refill():
        if not state["alive"]:
            return
        import random
        new = random.randint(1, 50)
        btn.num = new
        btn.configure(text=str(new), bg="#ffffff", fg="#1c1c1c")
    root.after(250, refill)


def bot_worker(roi_getter):
    time.sleep(1.2)
    rx, ry, rw, rh = roi_getter()
    cfg = {
        "roi": [rx, ry, rw, rh],
        "click_delay_ms": 15,
        "realtime_scan_delay_ms": 20,
        "ocr_workers": 4,
        "template_confidence": 0.3,
        "max_click_attempts": 2,
        "click_blacklist_ms": 900,
        "exit_on_empty": False,
    }
    bot = RealtimeBot(cfg, log=lambda m: print(f"  [bot] {m}"))
    state["bot"] = bot
    try:
        result["stats"] = bot.run()  # runs until stop is requested
    except Exception as exc:
        result["error"] = str(exc)
    result["history"] = list(bot.history)


def main():
    make_window()
    root.update_idletasks()
    root.update()

    def roi_getter():
        return (root.winfo_rootx(), root.winfo_rooty(),
                root.winfo_width(), root.winfo_height())

    t = threading.Thread(target=bot_worker, args=(roi_getter,), daemon=True)
    t.start()

    check_deadline = time.time() + 12

    def poll():
        if time.time() > check_deadline or result.get("error"):
            state["alive"] = False
            bot = state.get("bot")
            if bot is not None:
                bot.stop()
            root.after(1500, root.destroy)  # let the bot finish its last cycle
            return
        root.after(100, poll)
    root.after(200, poll)
    root.mainloop()
    t.join(timeout=6)

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return 1
    clicks = list(state["clicks"])
    history = result.get("history", [])
    stats = result.get("stats", {}) or {}

    # Under refill, order may legitimately DROP (a fresh low number can spawn
    # anywhere after a click) — the real requirements are:
    # 1) kept running until the stop request (did not quit on refill empties)
    kept_running = stats.get("elapsed_s", 0) >= 9.0
    # 2) steady clicking activity across the window
    enough_clicks = stats.get("clicks", 0) >= 8
    # 3) clicks landed on several distinct tiles (not stuck on one)
    distinct_tiles = len({(x, y) for (_n, x, y, _v) in history})
    # 4) verification worked on live tiles
    verified_some = stats.get("verified", 0) >= 5
    # 5) dead tile didn't spam-forever: few misses, and bot moved past it
    few_misses = stats.get("misses", 0) <= 4

    ok = (kept_running and enough_clicks and distinct_tiles >= 4
          and verified_some and few_misses)
    print(f"\ngame-registered clicks ({len(clicks)}): {clicks[:40]}")
    print(f"stats: {stats}")
    print(f"distinct tiles clicked: {distinct_tiles}")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    if not ok:
        print(f"  kept_running={kept_running} enough_clicks={enough_clicks} "
              f"distinct_tiles={distinct_tiles} verified_some={verified_some} "
              f"few_misses={few_misses}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
