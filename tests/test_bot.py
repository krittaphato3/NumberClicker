"""End-to-end fake-backend tests for NumberBot's strict sequencing.

These tests inject a fully fake environment (capture, clicker, verifier) so
the real click-verify loop runs without a screen, tesseract, or Windows.
"""
from __future__ import annotations

import sys
import threading
import time
import unittest
from typing import Dict, Optional, Tuple

sys.path.insert(0, ".")

from bot_core.bot import CLEARED, NOT_FOUND, STUCK, NumberBot  # noqa: E402


class FakeClicker:
    """Clicker stand-in that applies clicks to a virtual board."""

    def __init__(self, board):
        self.board = board
        self.clicks = 0
        self.start_time = time.perf_counter()
        self.positions = []
        self.move = lambda x, y: self.positions.append((x, y))
        self.stats = lambda: {"clicks": self.clicks, "elapsed_s": 0.0, "cps": 0.0}
        self.reset_stats = lambda: None

    def press(self):
        self.clicks += 1
        self.board.click(self.positions[-1] if self.positions else (0, 0))


class FakeBoard:
    """Virtual 1..N game: a dict number -> state ('present'/'gone')."""

    def __init__(self, n=50, auto_clear: bool = True):
        self.numbers: Dict[int, str] = {i: "present" for i in range(1, n + 1)}
        self.auto_clear = auto_clear
        self.stuck: set = set()  # numbers that never clear no matter what
        self.click_log = []

    def click(self, pos: Tuple[int, int]) -> None:
        for n, state in self.numbers.items():
            if state == "present" and abs(pos[0] - n) < 1 and abs(pos[1] - 100) < 1:
                if self.auto_clear and n not in self.stuck:
                    self.numbers[n] = "gone"
                self.click_log.append(n)
                return

    def coord_for(self, n: int) -> Tuple[int, int]:
        return (n, 100)


class FakeEnv:
    """Wires a FakeBoard into a NumberBot by replacing subsystem methods."""

    def __init__(self, n: int = 50, auto_clear: bool = True, **cfg):
        self.board = FakeBoard(n, auto_clear)
        cfg.setdefault("grid_mode", "manual")
        cfg.setdefault("grid_rows", n)
        cfg.setdefault("grid_cols", 1)
        cfg.setdefault("click_delay_ms", 0)
        cfg.setdefault("reaction_buffer_ms", 0)
        cfg.setdefault("clear_timeout_ms", 20)
        cfg.setdefault("clear_poll_ms", 2)
        cfg.setdefault("scan_frequency_hz", 0)
        cfg.setdefault("confidence_threshold", 0.1)
        self.bot = NumberBot(cfg, log=lambda _m: None)
        # Pretend calibration already happened.
        self.bot.number_map = {i: self.board.coord_for(i) for i in self.board.numbers}
        self.bot.detector = None
        self.bot.clicker = FakeClicker(self.board)
        # Patch verifier functions used by step_number.
        self._snap_of = lambda n: bytes([self.board.numbers[n] == "gone" and 255 or 10])
        bot_core_bot = sys.modules["bot_core.bot"]
        self._orig = {k: getattr(bot_core_bot, k) for k in
                      ("cell_snapshot", "wait_for_clear", "is_cleared", "ocr_gone_check")}

        import bot_core.bot as bb

        def fake_cell_snapshot(cx, cy, size=44):
            return self._snap_of(cx)

        def fake_wait_for_clear(cx, cy, before, timeout_ms, poll_ms, diff_th):
            deadline = time.perf_counter() + timeout_ms / 1000.0
            while time.perf_counter() < deadline:
                if self.board.numbers.get(cx) == "gone":
                    return True
                time.sleep(poll_ms / 1000.0)
            return False

        def fake_is_cleared(before, after, diff_th):
            return before != after

        def fake_ocr_gone_check(cx, cy, target, size=44, recognizer=None):
            st = self.board.numbers.get(target)
            return None if st is None else (st == "gone")

        bb.cell_snapshot = fake_cell_snapshot
        bb.wait_for_clear = fake_wait_for_clear
        bb.is_cleared = fake_is_cleared
        bb.ocr_gone_check = fake_ocr_gone_check
        self.bb = bb

    def restore(self):
        for k, v in self._orig.items():
            setattr(self.bb, k, v)

    def close(self):
        self.restore()


class TestStrictSequencing(unittest.TestCase):
    """The core rule: never advance until the current number is proven gone."""

    def setUp(self):
        self.envs = []

    def tearDown(self):
        for e in self.envs:
            e.close()

    def _env(self, *a, **k):
        env = FakeEnv(*a, **k)
        self.envs.append(env)
        return env

    def test_full_run_cleans_all_in_order(self):
        env = self._env(50, auto_clear=True)
        bot = env.bot
        s = bot.run(target=50)
        self.assertEqual(s["cleared"], 50)
        self.assertFalse(s["aborted"])
        self.assertEqual(env.board.click_log, list(range(1, 51)),
                         "numbers must be clicked strictly in ascending order")

    def test_never_advances_past_unclicked_number(self):
        # Number 7 refuses to clear -> bot must abort at 7, not click 8..50.
        env = self._env(50, auto_clear=True)
        env.board.stuck = {7}  # 1-6 clear normally; 7 never does
        bot = env.bot
        s = bot.run(target=50)
        self.assertTrue(s["aborted"])
        self.assertNotIn(8, env.board.click_log)
        self.assertNotIn(50, env.board.click_log)
        self.assertIn(7, env.board.click_log)

    def test_allow_skip_continues_past_stuck(self):
        env = self._env(50, auto_clear=True)
        env.board.stuck = {7}
        env.bot.config["allow_skip"] = True
        s = env.bot.run(target=50)
        self.assertFalse(s["aborted"])
        self.assertIn(8, env.board.click_log)
        self.assertIn(50, env.board.click_log)
        self.assertEqual(s["cleared"], 49)  # 7 skipped, not counted cleared

    def test_missing_number_treated_as_already_cleared(self):
        # 12 vanishes without a click (already cleared externally).
        env = self._env(50, auto_clear=True)
        del env.bot.number_map[12]
        orig_rescan = env.bot.rescan_number

        def rescan_without_12(target):
            if target == 12:
                return None
            return orig_rescan(target)

        env.bot.rescan_number = rescan_without_12
        s = env.bot.run(target=50)
        self.assertEqual(s["cleared"], 50)
        self.assertFalse(s["aborted"])
        self.assertNotIn(12, env.board.click_log)

    def test_blind_capture_aborts_instead_of_faking_clear(self):
        """If rescans return an EMPTY board, the bot must abort (not treat
        every number as 'absent -> cleared') — otherwise a wrong ROI would
        produce a perfect-looking run with zero real clicks."""
        env = self._env(50, auto_clear=True)
        bot = env.bot
        # Make number 7 unfindable AND make full rescans blind (empty map).
        del bot.number_map[7]
        bot.full_rescan = lambda: bot.number_map.clear()
        s = bot.run(target=50)
        self.assertTrue(s["aborted"])
        self.assertNotIn(8, env.board.click_log)  # no fake advance past 7

    def test_stop_event_halts_run(self):
        env = self._env(50, auto_clear=True)
        bot = env.bot
        threading.Timer(0.05, bot.stop).start()
        s = bot.run(target=50)
        self.assertLess(s["cleared"], 50)
        self.assertFalse(s["aborted"])

    def test_pause_resume_completes(self):
        env = self._env(10, auto_clear=True)
        bot = env.bot

        def mid_run_pause():
            bot.pause()
            time.sleep(0.05)
            bot.resume()

        threading.Timer(0.02, mid_run_pause).start()
        s = bot.run(target=10)
        self.assertEqual(s["cleared"], 10)

    def test_step_number_states(self):
        env = self._env(5, auto_clear=True)
        bot = env.bot
        self.assertEqual(bot.step_number(1), CLEARED)
        self.assertEqual(bot.step_number(2), CLEARED)
        # Unknown number -> NOT_FOUND.
        self.assertEqual(bot.step_number(999), NOT_FOUND)

    def test_stuck_number_reports_stuck(self):
        env = self._env(3, auto_clear=True)
        env.board.stuck = {2}
        bot = env.bot
        self.assertEqual(bot.step_number(2), STUCK)

    def test_run_accepts_target_positionally(self):
        """bot.run(50) semantics: target=50 -> 1..50 (regression for old run(start))."""
        env = self._env(50, auto_clear=True)
        s = env.bot.run(50)
        self.assertEqual(s["cleared"], 50)

    def test_external_stop_event_respected(self):
        env = self._env(50, auto_clear=True)
        ext_stop = threading.Event()
        ext_pause = threading.Event()
        threading.Timer(0.03, ext_stop.set).start()
        s = env.bot.run(target=50, stop_event=ext_stop, pause_event=ext_pause)
        self.assertLess(s["cleared"], 50)

    def test_phase_swap_auto_rescans_to_target(self):
        """5x5-board scenario: board shows 1..25, then swaps to 26..50 after
        the 25th click. run(target=50) must auto-rescan and finish 1..50
        in ONE run instead of stopping at the board's max."""
        env = self._env(50, auto_clear=True)
        bot = env.bot
        # Phase 1: only 1..25 on the board.
        with bot._map_lock:
            bot.number_map = {i: bot.number_map[i] for i in range(1, 26)}
        state = {"swapped": False}
        real_step = bot.step_number

        def swapping_step(n):
            res = real_step(n)
            # After 25 clears, the game replaces the whole board in place.
            if n == 25 and res == CLEARED and not state["swapped"]:
                state["swapped"] = True
                with bot._map_lock:
                    for i in range(1, 26):
                        bot.number_map.pop(i, None)
                    for i in range(26, 51):
                        bot.number_map[i] = env.board.coord_for(i)
            return res

        bot.step_number = swapping_step
        s = bot.run(target=50)
        self.assertTrue(state["swapped"])
        self.assertFalse(s["aborted"])
        self.assertEqual(s["cleared"], 50)
        self.assertEqual(env.board.click_log, list(range(1, 51)),
                         "phase 2 (26..50) must be clicked after the swap")

    def test_phase_wait_timeout_ends_run_cleanly(self):
        """If the next phase never appears, the run ends (not hung forever).
        Phase 1 (1..25) completes; no swap ever comes; target 50 unreachable."""
        env = self._env(50, auto_clear=True)
        bot = env.bot
        with bot._map_lock:
            bot.number_map = {i: bot.number_map[i] for i in range(1, 26)}
        bot.config["phase_wait_timeout_ms"] = 300
        bot.config["phase_poll_interval_s"] = 0.05
        s = bot.run(target=50)
        self.assertFalse(s["aborted"])
        self.assertEqual(s["cleared"], 25)  # phase 1 done, phase 2 never came
        self.assertEqual(env.board.click_log, list(range(1, 26)))

    def test_stats_shape(self):
        env = self._env(5, auto_clear=True)
        s = env.bot.run(target=5)
        for key in ("cleared", "total", "accuracy", "clicks", "cps", "elapsed_s",
                    "misses", "retries", "rescans", "target", "aborted"):
            self.assertIn(key, s)
        self.assertEqual(s["total"], 5)
        self.assertAlmostEqual(s["accuracy"], 1.0, places=3)

    def test_roi_dict_normalization(self):
        env = self._env(5, auto_clear=True)
        cases = [
            ([100, 200, 800, 600], {"left": 100, "top": 200, "width": 800, "height": 600}),
            ({"left": 1, "top": 2, "width": 3, "height": 4},
             {"left": 1, "top": 2, "width": 3, "height": 4}),
            (None, None),
            ([0, 0, 0, 0], None),  # degenerate -> None
        ]
        for roi, want in cases:
            env.bot.config["roi"] = roi
            self.assertEqual(env.bot._roi_dict(), want, f"roi={roi!r}")

    def test_mapping_boxes(self):
        env = self._env(4, auto_clear=True)
        boxes = env.bot.mapping_boxes()
        self.assertEqual(set(boxes.keys()), {1, 2, 3, 4})
        for n, (x, y, w, h) in boxes.items():
            cx, cy = env.board.coord_for(n)
            # mapping_boxes returns the box TOP-LEFT; the center must match.
            self.assertAlmostEqual(x + w / 2, cx, delta=1)
            self.assertAlmostEqual(y + h / 2, cy, delta=1)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)

    def test_cleared_numbers_removed_from_map(self):
        env = self._env(3, auto_clear=True)
        bot = env.bot
        bot.step_number(1)
        self.assertNotIn(1, bot.number_map)
        self.assertIn(1, bot.cleared)


class TestRunStats(unittest.TestCase):
    def test_record_and_derive(self):
        from bot_core.stats import RunStats
        st = RunStats()
        st.record_hit()
        st.record_hit()
        st.record_miss(retried=True)
        self.assertEqual((st.clicks, st.hits, st.misses, st.retries), (3, 2, 1, 1))
        self.assertAlmostEqual(st.accuracy(), 2 / 3, places=3)
        d = st.to_dict()
        self.assertEqual(d["clicks"], 3)
        self.assertIn("cps", d)
        self.assertIn("elapsed", d)

    def test_reset(self):
        from bot_core.stats import RunStats
        st = RunStats()
        st.record_hit()
        st.reset()
        self.assertEqual(st.clicks, 0)
        self.assertEqual(st.accuracy(), 0.0)


class TestConfigLoad(unittest.TestCase):
    def test_defaults_applied(self):
        from bot_core.bot import DEFAULTS, load_config
        cfg = load_config(path="config.json")
        self.assertEqual(cfg["click_delay_ms"], 30)
        self.assertEqual(cfg["confidence_threshold"], 0.6)
        for k in DEFAULTS:
            self.assertIn(k, cfg)

    def test_corrupt_config_falls_back(self):
        from bot_core.bot import load_config
        cfg = load_config(path="definitely_missing_config.json")
        self.assertEqual(cfg["click_delay_ms"], 30)


class TestIsCleared(unittest.TestCase):
    def test_diff_threshold_semantics(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        from bot_core.verifier import is_cleared
        a = np.full((20, 20), 100, dtype=np.uint8)
        same = np.full((20, 20), 102, dtype=np.uint8)   # MAD 2 < 12 -> not cleared
        diff = np.full((20, 20), 200, dtype=np.uint8)   # MAD 100 > 12 -> cleared
        self.assertFalse(is_cleared(a, same, 12.0))
        self.assertTrue(is_cleared(a, diff, 12.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
