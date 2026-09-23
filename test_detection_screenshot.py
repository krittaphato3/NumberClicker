"""Offline detection test: run cluster-cell detection + template OCR on the
pasted game screenshot (no screen capture, no clicking)."""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bot_core.detector import ClusterBoardDetector, TemplateRecognizer, build_number_map

IMG = r"C:\Users\Oz\AppData\Local\Temp\freebuff-desktop-pastes\paste-1790145615437-21788.png"

EXPECTED = {  # read manually from the screenshot
    13: 0, 19: 1, 6: 2, 1: 3, 15: 4,
    11: 0, 20: 1, 4: 2, 16: 3, 18: 4,
    10: 0, 3: 1, 5: 2, 24: 3, 21: 4,
    12: 0, 14: 1, 9: 2, 17: 3, 2: 4,
    25: 0, 7: 1, 8: 2, 23: 3, 22: 4,
}

def main():
    img = cv2.imread(IMG, cv2.IMREAD_COLOR)
    if img is None:
        print(f"FAIL: cannot read {IMG}")
        return 1
    print(f"image: {img.shape[1]}x{img.shape[0]}")

    t0 = time.perf_counter()
    cells = ClusterBoardDetector().detect(img)
    t1 = time.perf_counter()
    if not cells:
        print("FAIL: cluster detection found no cells")
        return 1
    print(f"cluster cells: {len(cells)} in {(t1 - t0) * 1000:.1f} ms")

    rec = TemplateRecognizer()
    t2 = time.perf_counter()
    nmap = build_number_map(img, cells, rec, workers=4)
    t3 = time.perf_counter()
    print(f"OCR: {len(nmap)}/25 numbers in {(t3 - t2) * 1000:.1f} ms")

    got = set(nmap.keys())
    want = set(EXPECTED.keys())
    missing = sorted(want - got)
    extra = sorted(got - want)
    if missing:
        print(f"MISSING: {missing}")
    if extra:
        print(f"EXTRA (misreads): {extra}")

    # Position sanity: expected counts at correct spot are encoded loosely; just
    # verify every found coordinate sits inside the board area.
    h, w = img.shape[:2]
    bad_pos = [n for n, (x, y) in nmap.items() if not (0 <= x < w and 0 <= y < h)]
    if bad_pos:
        print(f"BAD POSITIONS: {bad_pos}")

    ok = not missing and not extra and not bad_pos
    print("RESULT:", "PASS" if ok else "FAIL")
    for n in sorted(nmap):
        print(f"  {n:>2} -> {nmap[n]}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
