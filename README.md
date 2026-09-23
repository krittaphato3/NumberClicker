# Number Sequence Clicking Bot (1–50)

Automated player for grid number-clicking games. Uses OpenCV + a built-in
**font-template digit recognizer** (no Tesseract install required on Windows)
to find numbers 1..50 on screen, then clicks them **in strict order** — and
**never advances to the next number until the current one is verified gone**
(pixel-diff fast path, OCR tie-break, absent-after-rescan as last resort).

**Speed budget:** clean runs land ~60–130 ms per number → **1–50 in 4–8 s**,
well under the 20 s target. Detection itself runs at 25 numbers in ~120 ms
and was validated 25/25 on a real board plus an end-to-end live run
(25/25 clicked in order, 0 misses) on Windows 11.

```
┌─────────────┐   ┌──────────────────────────────────────┐   ┌──────────────┐
│ Screen (ROI)│──▶│ GridDetector → NumberRecognizer (OCR)│──▶│ number→coord │
└─────────────┘   └──────────────────────────────────────┘   │     map      │
                                                             └──────┬───────┘
                    ┌────────────────────────────────────────┐       ▼
                    │ step: move → snapshot → click → verify │◀──────┘
                    │ (loop until proven gone, then N+1)     │
                    └────────────────────────────────────────┘
```

## Installation

1. **Python 3.9+** (3.12 tested).

2. **Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

That's it — digit reading uses Windows system fonts rendered into templates at
startup (Segoe UI/Tahoma/etc. × 4 stroke weights), so **no Tesseract install
is needed**. If you prefer Tesseract anyway (e.g. unusual game fonts), install
it from https://github.com/UB-Mannheim/tesseract/wiki and set `tesseract_cmd`
in `config.json` if it is not on PATH; the bot then uses it first and falls
back to templates automatically.

## Quick start

**Realtime mode (recommended, simplest):** the bot scans the board constantly
and always clicks the **lowest number currently visible**. It keeps
auto-scanning forever — when a clicked number is replaced by a new one (even
in the same spot), the next scan picks it up. Each click is verified (the
cell must visibly change); unresponsive cells are retried, then briefly
blacklisted so a dead tile can't stall the run.

```bash
python main.py --realtime                 # scan -> click lowest -> rescan
python main.py --realtime --debug-view    # + live window showing detections
```

The loop ends only via **F2**/Ctrl+C, after clicking `target`, or — with
`"exit_on_empty": true` — when the board stays empty for `empty_timeout_ms`.
(Default keeps running across refill animations: a brief empty frame is
normal while the game swaps numbers.)

1. Open the game with the board visible.
2. Set `roi` in `config.json` (or pick it once via the GUI: `python main.py`
   → **Select ROI**), then run `python main.py --realtime`.
3. Watch the debug window (`--debug-view`): green labels = detected numbers,
   red dot = the number about to be clicked. **F2** stops (Ctrl+C also works).

GUI / legacy strict sequencer:

```bash
python main.py                 # GUI (default): settings panel + overlay
python main.py --headless      # strict 1..N sequencer, never out of order
python main.py --calibrate-only --roi 100,100,900,600   # dry-run: print the detected map
```

### Board phases (5×5 board, target 50)

A 5×5 board can only show 25 numbers at a time. The strict sequencer handles
this in **phases** and needs no manual restart: it clears the visible phase
(1..25), keeps **auto-rescanning** during the empty/swap window until the next
board (26..50) appears — even in the same tiles — and continues clicking until
`target` is reached. Relevant keys:

| Key | Default | Meaning |
|---|---|---|
| `phase_wait_timeout_ms` | 15000 | max wait for the next board phase to appear |
| `phase_poll_interval_s` | 0.35 | rescan interval while waiting for the swap |

## How the strict sequencing works

For every number N, the bot must *prove* N is gone before touching N+1:

1. **Fast path — pixel diff:** it moves the cursor into the cell first, takes a
   44×44 grayscale "before" snapshot, clicks, then polls every ~8 ms until the
   mean-absolute-difference exceeds `clear_diff_threshold` (the game graying
   out/removing the digit changes the cell far more than capture noise does).
2. **Tie-break — OCR:** if diffing times out, the cell is re-read with the
   active recognizer; if it no longer shows N, the click worked (handles
   subtle color-only changes).
3. **Retry:** if the cell *definitely* still shows N (OCR proof), the click is
   repeated (up to `max_verify_attempts`).
4. **Last resort:** full-board rescans; if N is absent from `absent_limit`
   consecutive rescans (while the rest of the board is still visible), it is
   counted cleared — this also handles numbers cleared externally.
5. **Abort:** if a number resists `stuck_limit` step attempts, the run aborts
   (or skips that one number with `allow_skip: true`) instead of clicking
   numbers out of order. A *blind* board (empty rescan) always aborts — that
   means ROI/window/OCR is broken, not that numbers cleared. Calibration that
   reads far too few numbers (wrong ROI/DPI) aborts before the first click for
   the same reason.

## Configuration (`config.json`)

| Key | Default | Meaning |
|---|---|---|
| `click_delay_ms` | 30 | minimum gap between clicks (0 = flat out) |
| `reaction_buffer_ms` | 20 | settle time after a click before verifying |
| `confidence_threshold` | 0.6 | tesseract mean-confidence gate (0.0–1.0) |
| `template_confidence` | 0.3 | template-OCR correlation gate (raw score 0–1) |
| `grid_mode` | `"auto"` | `"auto"` tries 5×10/5×5; `"manual"` uses rows×cols |
| `grid_rows` / `grid_cols` | 5 / 10 | manual grid shape |
| `roi` | null | `[x, y, w, h]` of the board (GUI drag-select sets it) |
| `scan_frequency_hz` | 0 | background re-scan rate; 0 = off (fastest) |
| `clear_timeout_ms` | 1200 | max wait for one click to visually clear |
| `clear_poll_ms` | 8 | poll interval while verifying |
| `clear_diff_threshold` | 12.0 | gray MAD that counts as "gone" |
| `verify_size` | 44 | snapshot square side (px) for diffing |
| `max_verify_attempts` | 2 | clicks per number before rescan/abort |
| `stuck_limit` | 4 | failed steps for one number → abort/skip |
| `absent_limit` | 2 | consecutive rescans absent → treated as cleared |
| `allow_skip` | false | true = skip stuck numbers instead of aborting |
| `realtime_scan_delay_ms` | 30 | pause between realtime scans |
| `max_click_attempts` | 3 | click retries when a cell does not visibly change |
| `click_blacklist_ms` | 1200 | cooldown for cells that ignored all retries |
| `exit_on_empty` | false | false = keep scanning (refill games); true = stop when board empties |
| `empty_timeout_ms` | 2500 | with exit_on_empty: empty duration before stopping |
| `ocr_workers` | 4 | parallel OCR workers for board scans |
| `humanize` | false | jitter + delay variance (slower, safer) |
| `smooth_move` | false | interpolated cursor movement (slower) |
| `tesseract_cmd` | "" | explicit tesseract.exe path (Windows) |
| `debug_overlay` | true | overlay boxes for detected numbers |
| `target` | 50 | final number |

## Tuning for <20 s

- `click_delay_ms: 0` + `reaction_buffer_ms: 10-20` gives ~50–90 ms/number.
- Keep `clear_poll_ms` ≤ 10 so verification never lags the game.
- Raise `ocr_workers` to 8 if your CPU has cores to spare (faster rescans).
- Leave `scan_frequency_hz` at 0 during runs — the watcher is for debugging.
- If the game animates removal slowly, raise `clear_diff_threshold` a bit or
  lower `verify_size` to 36 to diff a smaller, higher-contrast region.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Scan finds 0 numbers | ROI wrong: re-drag the ROI tightly around the board (client area, not the title bar). The bot logs `calibration unhealthy` and refuses to fake-run when capture is broken. |
| Scan finds some numbers | Lower `template_confidence` to 0.2; make sure the ROI is not scaled (Windows display scaling ≠ 100% shifts coordinates — the bot requests DPI awareness automatically). |
| Clicks land in the wrong place | Re-select the ROI after moving/resizing the game window; calibration re-runs automatically on window moves. |
| Bot clicks but aborts "stuck" | The diff threshold is too high for the game's clear animation: lower `clear_diff_threshold` (e.g. 8). |
| Bot skips numbers silently | The game has a settle animation; raise `reaction_buffer_ms` to 40–60. |
| Runs are slow (~20 s+) | `click_delay_ms: 0`, `scan_frequency_hz: 0`, and verify only 50 cells are being scanned (smaller ROI). |
| Overlay intercepts clicks | Windows builds use a click-through overlay; on macOS/Linux turn `debug_overlay` off during runs. |
| GUI window steals focus mid-run | Start the run, then click into the game once; hotkeys are global so F2/F3 still work. |

## Project layout

```
main.py              CLI entry point (GUI default, --realtime, --headless, --calibrate-only)
config.json          persisted settings
bot_core/
  detector.py        screen capture, digit-cluster board detection, OCR engines
                     (font-template TemplateRecognizer — no tesseract needed —
                     with optional tesseract fallback)
  verifier.py        click-effect verification (pixel diff + OCR tie-break)
  clicker.py         Win32 SendInput clicker (pyautogui fallback), humanize options
  bot.py             NumberBot: strict sequencing state machine + stats
  gui.py             Tkinter panel + click-through overlay + global hotkeys
  stats.py           run statistics (cps, accuracy, elapsed)
tests/test_bot.py    fake-backend unit tests for the sequencing invariant
bot_core/
  realtime.py        RealtimeBot: rescan -> click lowest -> repeat loop
tests/test_bot.py    fake-backend unit tests for the sequencing invariant
test_detection_screenshot.py  regression: detects all 25 numbers on a real screenshot
test_e2e_live.py     end-to-end: strict sequencer vs simulated window (needs a desktop)
test_e2e_phases.py   end-to-end: 5x5 board swaps 1..25 -> 26..50, bot must continue
test_e2e_realtime.py end-to-end: realtime lowest-first bot vs simulated window
```
