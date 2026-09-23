"""Grid detection + digit OCR for the 1..50 number clicking game.

Windows-first, dependency-tolerant: every third-party import is guarded so the
module imports cleanly even when opencv/numpy/mss/pillow/pytesseract are absent.
Functions that need a missing dependency raise a clear RuntimeError only when
actually called (never at import time).
"""
from __future__ import annotations

import math
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Lazy / guarded third-party imports — never crash at import time.
# ---------------------------------------------------------------------------
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore  # noqa: N816

try:
    import mss  # type: ignore
except Exception:  # pragma: no cover
    mss = None  # type: ignore

try:
    from PIL import ImageGrab  # type: ignore
except Exception:  # pragma: no cover
    try:
        from PIL import Image as _PILImage  # noqa: F401
        ImageGrab = None  # type: ignore
    except Exception:
        ImageGrab = None  # type: ignore

try:
    import pytesseract  # type: ignore
    from pytesseract import Output as _TessOutput  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None  # type: ignore
    _TessOutput = None  # type: ignore


# Type aliases (loose so file type-checks without numpy installed).
BBox = Tuple[int, int, int, int]  # (x, y, w, h) in image-local coords.
Point = Tuple[int, int]  # (cx, cy) center coords.

# OCR config: digits only, single uniform textline/block for one cell.
_TESS_CONFIG = "--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789"


def ensure_dpi_awareness() -> None:
    """Make capture/input coordinates match physical pixels (Windows, best-effort).

    Without this, mss/SendInput see *virtualized* coordinates while Tk windows
    report *physical* ones on displays with scaling != 100%, silently shifting
    every ROI and click by the scale factor. Idempotent; no-op off Windows.
    """
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # system-DPI-aware
            return
        except Exception:
            pass
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_cv2() -> None:
    """Raise a helpful error if OpenCV is unavailable."""
    if cv2 is None:
        raise RuntimeError("opencv-python (cv2) is required but not installed.")


def _require_numpy() -> None:
    """Raise a helpful error if numpy is unavailable."""
    if np is None:
        raise RuntimeError("numpy is required but not installed.")


def tesseract_available(tesseract_cmd: str = "") -> bool:
    """Return True when pytesseract + a tesseract binary appear usable.

    Never raises; returns False when anything is missing/broken.
    """
    if pytesseract is None:
        return False
    try:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        # get_tesseract_version shells out; cheap enough to call on demand.
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def scan_roi(roi: Optional[Dict[str, int]] = None):
    """Grab a screen region and return it as a BGR numpy image.

    Args:
        roi: Optional dict with keys ``left``/``top``/``width``/``height``.
            ``None`` means full primary monitor.

    Returns:
        BGR ``numpy.ndarray`` of shape (H, W, 3), dtype uint8.

    Strategy:
        1. Prefer ``mss`` (fastest, no temp files).
        2. Fall back to ``PIL.ImageGrab`` and convert RGB -> BGR.
    """
    _require_numpy()
    # --- Fast path: mss -----------------------------------------------------
    if mss is not None:
        try:
            with mss.mss() as sct:
                monitor = dict(sct.monitors[0]) if roi is None else {
                    "left": int(roi.get("left", 0)),
                    "top": int(roi.get("top", 0)),
                    "width": int(roi.get("width", 0)),
                    "height": int(roi.get("height", 0)),
                }
                shot = sct.grab(monitor)
                # mss returns BGRA; drop alpha -> BGR.
                img = np.array(shot, dtype=np.uint8)[:, :, :3]
                return img
        except Exception:
            pass  # fall through to PIL fallback below.
    # --- Fallback: PIL ImageGrab (Windows/macOS) ----------------------------
    if ImageGrab is not None:
        try:
            _require_cv2()
            if roi is None:
                pil_img = ImageGrab.grab()
            else:
                x0 = int(roi.get("left", 0))
                y0 = int(roi.get("top", 0))
                x1 = x0 + int(roi.get("width", 0))
                y1 = y0 + int(roi.get("height", 0))
                pil_img = ImageGrab.grab(bbox=(x0, y0, x1, y1))
            rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
        except Exception as exc:
            raise RuntimeError(f"scan_roi failed (mss + ImageGrab both failed): {exc}") from exc
    raise RuntimeError("No screen-capture backend available (install `mss` or `pillow`).")


# ---------------------------------------------------------------------------
# Grid detection
# ---------------------------------------------------------------------------
class GridDetector:
    """Locate the game grid and split it into clickable cells.

    Pipeline (per frame):
        1. Grayscale + blur + adaptive threshold -> binary board mask.
        2. Largest external contour ~= outer game board rectangle.
        3. If contour detection fails or yields a degenerate box, fall back
           to the full ROI as the grid.
        4. Subdivide the grid rect into ``rows x cols`` cells.

    The detector caches the last grid rect + cell list so repeated clicks in
    ``bot.py`` do not pay contour-detection cost on every number.
    """

    def __init__(
        self,
        rows: int = 5,
        cols: int = 10,
        mode: str = "auto",
        min_board_area_ratio: float = 0.02,
    ) -> None:
        """Args:
            rows/cols: expected grid shape (5x10 for 1..50, 5x5 for 1..25).
            mode: ``"auto"`` tries 5x10 then 5x5 heuristics; otherwise fixed.
            min_board_area_ratio: smallest acceptable board area as a fraction
                of the whole image (guards against noise contours).
        """
        self.rows = int(rows)
        self.cols = int(cols)
        self.mode = str(mode or "auto").lower()
        self.min_board_area_ratio = float(min_board_area_ratio)
        # Cache: last successful detection.
        self._cached_rect: Optional[BBox] = None
        self._cached_cells: List[BBox] = []
        self._cached_shape: Optional[Tuple[int, int]] = None

    # -- public API ----------------------------------------------------------
    @property
    def cached_cells(self) -> List[BBox]:
        """Return the last detected cell list (copy)."""
        return list(self._cached_cells)

    def clear_cache(self) -> None:
        """Invalidate cached grid geometry (call after window moves/resizes)."""
        self._cached_rect = None
        self._cached_cells = []
        self._cached_shape = None

    def find_grid_rect(self, image) -> BBox:
        """Find the outer board rectangle ``(x, y, w, h)`` in image coords.

        Falls back to the full image ``(0, 0, W, H)`` when OpenCV is missing
        or no plausible contour is found — the caller can still proceed with
        a uniform subdivision.
        """
        h, w = image.shape[:2] if hasattr(image, "shape") else (0, 0)
        if cv2 is None or np is None or h == 0 or w == 0:
            return (0, 0, int(w), int(h))
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Adaptive threshold handles uneven in-game shading better than OTSU here.
            binary = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                51, 9,
            )
            # Close small gaps in grid lines so the board becomes one contour.
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return (0, 0, w, h)
            img_area = float(w * h)
            # Largest contour by area is normally the board frame.
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours[:5]:
                area = cv2.contourArea(cnt)
                if area < img_area * self.min_board_area_ratio:
                    continue
                x, y, bw, bh = cv2.boundingRect(cnt)
                # Reject degenerate slivers / full-frame noise.
                if bw < w * 0.2 or bh < h * 0.2:
                    continue
                if bw >= w and bh >= h:
                    continue
                # Slight inset so cell sampling avoids the border line itself.
                pad_x = max(1, int(bw * 0.005))
                pad_y = max(1, int(bh * 0.005))
                return (x + pad_x, y + pad_y, bw - 2 * pad_x, bh - 2 * pad_y)
        except Exception:
            pass  # any failure -> full-ROI fallback below.
        return (0, 0, w, h)

    def subdivide(self, rect: BBox, rows: int, cols: int) -> List[BBox]:
        """Split ``rect`` into ``rows x cols`` equal cells (row-major order)."""
        x, y, w, h = rect
        rows = max(1, int(rows))
        cols = max(1, int(cols))
        cells: List[BBox] = []
        for r in range(rows):
            y0 = y + (r * h) // rows
            y1 = y + ((r + 1) * h) // rows
            for c in range(cols):
                x0 = x + (c * w) // cols
                x1 = x + ((c + 1) * w) // cols
                cells.append((x0, y0, x1 - x0, y1 - y0))
        return cells

    def set_cells(self, cells: List[BBox]) -> None:
        """Adopt externally-detected cells (e.g. digit-cluster boxes) into the cache."""
        self._cached_cells = list(cells)
        self._cached_rect = None

    def find_cells(self, image, rows: Optional[int] = None, cols: Optional[int] = None) -> List[BBox]:
        """Detect grid then return cell bboxes; caches the result.

        Args:
            image: BGR/gray numpy frame.
            rows/cols: override the constructor shape for this call.
        """
        r = int(rows or self.rows)
        c = int(cols or self.cols)
        rect = self.find_grid_rect(image)
        cells = self.subdivide(rect, r, c)
        # Cache for hot-loop reuse.
        self._cached_rect = rect
        self._cached_cells = list(cells)
        self._cached_shape = (image.shape[0], image.shape[1]) if hasattr(image, "shape") else None
        return cells

    def auto_detect_grid(self, image, layouts: Sequence[Tuple[int, int]] = ((5, 10), (5, 5))) -> List[BBox]:
        """Try candidate ``(rows, cols)`` layouts and keep the best-scoring one.

        Scoring heuristic: run a cheap edge-density check per layout — the
        layout whose cells contain the most internal contour content (i.e.
        looks like digits, not blank space) wins. Falls back to constructor
        ``rows x cols`` when scoring is impossible (missing cv2/numpy).
        """
        if cv2 is None or np is None:
            return self.find_cells(image)
        layouts = list(layouts) if layouts else [(self.rows, self.cols)]
        if self.mode != "auto" or len(layouts) == 1:
            r, c = layouts[0]
            return self.find_cells(image, r, c)
        rect = self.find_grid_rect(image)
        best: Optional[List[BBox]] = None
        best_score = -1.0
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 60, 160)
        except Exception:
            r, c = layouts[0]
            return self.find_cells(image, r, c)
        for (r, c) in layouts:
            try:
                cells = self.subdivide(rect, int(r), int(c))
                # Score = mean edge pixels per cell interior (shrunk 15% margin
                # to ignore grid lines themselves).
                densities: List[float] = []
                for (x, y, w, h) in cells:
                    mx, my = int(w * 0.15), int(h * 0.15)
                    x0, y0 = x + mx, y + my
                    cw, ch = max(1, w - 2 * mx), max(1, h - 2 * my)
                    crop = edges[y0:y0 + ch, x0:x0 + cw]
                    densities.append(float(np.mean(crop > 0)) if crop.size else 0.0)
                score = float(np.mean(densities)) if densities else 0.0
                # Prefer the fuller 5x10 board unless 5x5 scores clearly higher
                # (avoids misclassifying sparse late-game boards).
                if best is None or score > best_score:
                    best_score = score
                    best = cells
            except Exception:
                continue
        cells = best if best else self.subdivide(rect, self.rows, self.cols)
        self._cached_rect = rect
        self._cached_cells = list(cells)
        self._cached_shape = (image.shape[0], image.shape[1]) if hasattr(image, "shape") else None
        return cells


# ---------------------------------------------------------------------------
# Digit OCR
# ---------------------------------------------------------------------------
class NumberRecognizer:
    """Preprocess + OCR single number cells.

    Designed for game digits: bold, centered, dark-on-light (or light-on-dark).
    Missing tesseract is *not* fatal — :meth:`ocr_cell` returns ``None`` so the
    bot can fall back to template/diff verification instead of crashing.
    """

    def __init__(self, confidence_threshold: float = 0.6, tesseract_cmd: str = "") -> None:
        """Args:
            confidence_threshold: minimum mean char confidence in [0, 1].
            tesseract_cmd: explicit path to ``tesseract.exe`` (Windows); when
                empty the system PATH is used.
        """
        self.confidence_threshold = float(confidence_threshold)
        self.tesseract_cmd = str(tesseract_cmd or "")
        if pytesseract is not None and self.tesseract_cmd:
            try:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            except Exception:
                pass

    # -- preprocessing -------------------------------------------------------
    def preprocess_cell(self, cell_img):
        """Upscale 2x + grayscale + OTSU binarization for OCR.

        Returns a binary (uint8, 0/255) image ~2x the input size. Also tries
        inverted polarity handling implicitly: OTSU threshold is computed on
        the grayscale image and the caller (ocr) retries inversion when the
        first pass yields nothing.
        """
        _require_cv2()
        _require_numpy()
        img = cell_img
        # Accept file path or PIL image defensively.
        if isinstance(img, str):
            img = cv2.imread(img, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"preprocess_cell: cannot read image path: {cell_img!r}")
        elif not hasattr(img, "shape"):
            try:
                from PIL import Image as _Img  # local import, pillow optional.

                if isinstance(img, _Img.Image):
                    img = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
            except Exception:
                pass
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # 2x upscale with cubic interpolation: dramatically helps psm 8 on small cells.
        h, w = gray.shape[:2]
        up = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        # Light denoise before thresholding.
        up = cv2.medianBlur(up, 3)
        _, binary = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    # -- OCR -----------------------------------------------------------------
    @property
    def available(self) -> bool:
        """True when pytesseract + binary seem usable."""
        return tesseract_available(self.tesseract_cmd)

    def ocr_cell(self, cell_img) -> Optional[Tuple[int, float]]:
        """OCR one cell image -> ``(number, confidence)`` or ``None``.

        Filtering rules:
            * keeps digits only (``[^0-9]`` stripped),
            * accepts 1-2 digit values in 1..99 (game uses 1..50),
            * requires mean tesseract word confidence >= threshold,
            * retries with inverted polarity if the first pass is empty,
            * returns ``None`` (never raises) when tesseract is missing or
              the cell is unreadable — the bot treats this as "unknown cell".

        Performance: uses DICT output (no pandas dependency) — the old
        DATAFRAME path raised on every call when pandas was absent, paying
        the exception cost per cell on every scan.
        """
        if pytesseract is None:
            return None  # graceful fallback: tesseract not installed.
        try:
            _require_cv2()
        except RuntimeError:
            return None
        try:
            binary = self.preprocess_cell(cell_img)
        except Exception:
            return None
        for attempt in range(2):
            try:
                img_try = binary if attempt == 0 else cv2.bitwise_not(binary)
                number: Optional[int] = None
                conf = 0.0
                try:
                    data = pytesseract.image_to_data(
                        img_try, config=_TESS_CONFIG, output_type=_TessOutput.DICT
                    )
                    texts = [str(t) for t in (data.get("text") or [])]
                    digits = "".join(ch for t in texts for ch in t if ch.isdigit())
                    raw_confs: List[float] = []
                    for c in (data.get("conf") or []):
                        try:
                            cval = float(str(c))
                        except Exception:
                            continue
                        if cval >= 0:
                            raw_confs.append(cval)
                    conf = (sum(raw_confs) / len(raw_confs) / 100.0) if raw_confs else 0.0
                    if digits and digits.isdigit():
                        number = int(digits)
                except Exception:
                    # Fallback path: plain string, uncalibrated confidence.
                    try:
                        raw = pytesseract.image_to_string(img_try, config=_TESS_CONFIG) or ""
                    except Exception:
                        raw = ""
                    digits = "".join(ch for ch in str(raw) if ch.isdigit())
                    if digits and digits.isdigit():
                        number = int(digits)
                        conf = 0.5  # uncalibrated but actionable.
                if number is not None and 1 <= number <= 99 and conf >= self.confidence_threshold:
                    return (int(number), float(conf))
                # Low-confidence digits are treated as unknown (safer than misclick).
            except Exception:
                continue  # try inverted polarity, then give up -> None.
        return None

    def ocr_text(self, cell_img) -> str:
        """Best-effort raw digit string ("" when unavailable). Convenience helper."""
        res = self.ocr_cell(cell_img)
        return str(res[0]) if res else ""


# ---------------------------------------------------------------------------
# Font-template digit recognition (tesseract-free fallback)
# ---------------------------------------------------------------------------
# Rendering digit templates from Windows fonts at process start is ~5-20ms per
# font size, cached for the life of the process. Template matching per cell is
# ~0.3ms — orders of magnitude faster than tesseract's ~50-150ms per cell.
_TEMPLATE_LOCK = threading.Lock()
_TEMPLATE_CACHE: Dict[Tuple, Dict[int, object]] = {}


def _candidate_fonts() -> List[str]:
    """Windows fonts likely to match game UI digits, best first."""
    return [
        "segoeuib.ttf",  # Windows UI default bold
        "segoeuib.ttf",  # keep first match cheap
        "arialbd.ttf",
        "tahomabd.ttf",
        "calibrib.ttf",
        "verdanab.ttf",
        "arial.ttf",
        "segoeui.ttf",
    ]


def _load_font(size: int):
    """Load the first available candidate font at ``size`` (None when none)."""
    for name in _candidate_fonts()[:6]:
        try:
            from PIL import ImageFont

            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return None


def _render_digit_templates(size: int = 96, strokes: Sequence[int] = (0, 2, 4, 6)) -> Dict[int, List]:
    """Render grayscale digit templates at several stroke weights (cached).

    Returns ``{digit: [gray_uint8 crops, white-on-black]}``. Multiple stroke
    widths (PIL ``stroke_width``) let the matcher imitate thin, bold, and
    extra-bold game fonts without image morphology guesswork. Rendering at
    96px keeps curves smooth when templates are resized down to glyph boxes.
    """
    key = (int(size), tuple(int(s) for s in strokes))
    with _TEMPLATE_LOCK:
        cached = _TEMPLATE_CACHE.get(key)
        if cached is not None:
            return cached
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return {}
    per_digit: Dict[int, List] = {d: [] for d in range(10)}
    for name in _candidate_fonts()[:3]:
        try:
            font = ImageFont.truetype(name, size)
        except Exception:
            continue
        for sw in strokes:
            for d in range(10):
                im = Image.new("L", (size * 3, size * 2), 0)
                ImageDraw.Draw(im).text((size, size // 2), str(d), font=font,
                                        fill=255, stroke_width=int(sw), stroke_fill=255)
                arr = np.array(im)
                ink = arr > 100
                ys, xs = np.where(ink)
                if len(xs) == 0:
                    continue
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                # TIGHT crop, no padding: glyphs are tight-cropped at match
                # time too, and any border shrinks the digit when the template
                # is resized into the glyph box (correlation drops 0.97->0.13).
                per_digit[d].append(arr[y0:y1, x0:x1])
        break  # one font x stroke variants is enough; multi-font averaging hurts '1'.
    with _TEMPLATE_LOCK:
        _TEMPLATE_CACHE[key] = per_digit
    return per_digit


class TemplateRecognizer:
    """Fast digit reader: grayscale template matching vs font-rendered glyphs.

    Why: tesseract.exe is a heavyweight install that many players lack; this
    recognizer needs only opencv+pillow and Windows system fonts. It is also
    ~10x faster per cell (~3-6ms vs ~50-150ms), which matters for rescans.

    Method (validated 25/25 on a real game screenshot):
        1. Binarize the cell (OTSU, auto polarity) -> glyph boxes.
        2. Per glyph: crop the GRAYSCALE box, inverted to match template
           polarity, and score vs every (digit, stroke-weight) template —
           PIL stroke_width variants imitate thin/medium/extra-bold fonts.
        3. Hole-count filter (0=none, 4/6/9=one, 8=two) kills the classic
           5/6 and 2/3 confusions before scoring.
        4. Concatenate glyph digits left-to-right -> (number, mean score).
    """

    # Typical enclosed-hole counts per digit in UI fonts.
    _HOLES = {0: 1, 4: 1, 6: 1, 8: 2, 9: 1, 1: 0, 2: 0, 3: 0, 5: 0, 7: 0}

    def __init__(self, confidence_threshold: float = 0.3) -> None:
        # NOTE: template-match confidence is a raw correlation in [0, 1] and
        # correct game digits commonly land in 0.2-0.7 (font mismatch), so the
        # default gate is much lower than tesseract's calibrated 0.6.
        self.confidence_threshold = float(confidence_threshold)

    @staticmethod
    def _count_holes(binimg) -> int:
        """Enclosed background regions inside the (white-ink) glyph mask."""
        try:
            inv = cv2.bitwise_not(binimg)
            n, _labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
            holes = 0
            gh, gw = binimg.shape[:2]
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                # Holes never touch the crop border; junk specks are tiny.
                if x <= 0 or y <= 0 or x + w >= gw or y + h >= gh:
                    continue
                if area >= max(6, (gh * gw) // 200):
                    holes += 1
            return holes
        except Exception:
            return -1  # unknown -> no filtering.

    # -- core ----------------------------------------------------------------
    @staticmethod
    def _binarize_digits_white(gray):
        """Return a binary image where digit ink is 255 (white on black)."""
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binimg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Ensure digits are white: sample the border (background) — if the
        # border is white, the ink must be black -> invert.
        border = np.concatenate([
            binimg[0, :].ravel(), binimg[-1, :].ravel(),
            binimg[:, 0].ravel(), binimg[:, -1].ravel(),
        ])
        if float(np.mean(border > 0)) > 0.5:
            binimg = cv2.bitwise_not(binimg)
        return binimg

    @staticmethod
    def _glyph_boxes(binimg, min_area: int = 12):
        """Connected components -> list of glyph bboxes sorted left-to-right."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binimg, connectivity=8)
        boxes = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area or h < 6:
                continue
            boxes.append([int(x), int(y), int(w), int(h)])
        # Merge horizontally overlapping boxes (broken glyphs, dotted '1').
        boxes.sort(key=lambda b: b[0])
        merged: List[List[int]] = []
        for b in boxes:
            if merged and b[0] <= merged[-1][0] + merged[-1][2] - 2:
                m = merged[-1]
                nx0 = min(m[0], b[0])
                ny0 = min(m[1], b[1])
                nx1 = max(m[0] + m[2], b[0] + b[2])
                ny1 = max(m[1] + m[3], b[1] + b[3])
                merged[-1] = [nx0, ny0, nx1 - nx0, ny1 - ny0]
            else:
                merged.append(list(b))
        return merged

    def _match_glyph(self, glyph_gray, glyph_bin) -> Tuple[Optional[int], float]:
        """Best (digit, adjusted score) for one tight-boxed glyph.

        Every digit is scored (a hard hole filter misfires when anti-aliased
        binarization closes or opens loops); the hole-count prior only shifts
        scores by a small bonus/penalty afterwards.
        """
        templates = _render_digit_templates()
        if not templates:
            return (None, 0.0)
        ys, xs = np.where(glyph_bin > 0)
        if len(xs) == 0:
            return (None, 0.0)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        g = glyph_gray[y0:y1, x0:x1]
        g = 255 - g  # game digits dark-on-light -> white-on-black like templates
        gh, gw = g.shape[:2]
        if gw < 3 or gh < 5:
            return (None, 0.0)
        gb = cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)
        holes = self._count_holes(glyph_bin[y0:y1, x0:x1])
        per_digit: List[Tuple[float, int]] = []
        for digit, variants in templates.items():
            best = -1.0
            for tmpl in variants:
                try:
                    t = cv2.resize(tmpl, (gw, gh), interpolation=cv2.INTER_AREA)
                    tb = cv2.GaussianBlur(t, (3, 3), 0).astype(np.float32)
                    s = float(cv2.matchTemplate(gb, tb, cv2.TM_CCOEFF_NORMED)[0, 0])
                except Exception:
                    continue
                if s > best:
                    best = s
            if best > -1.0:
                # Soft hole prior: reward agreement, mildly punish mismatch.
                if holes >= 0:
                    hp = self._HOLES.get(digit, 0)
                    best += 0.06 if hp == holes else -0.06
                per_digit.append((best, int(digit)))
        if not per_digit:
            return (None, 0.0)
        per_digit.sort(reverse=True)
        return (per_digit[0][1], per_digit[0][0])

    def ocr_cell(self, cell_img) -> Optional[Tuple[int, float]]:
        """Read one cell image -> (number, confidence 0..1) or None.

        Never raises; returns None for blank/unclear cells. Accepts 1..99,
        rejecting anything with >2 glyphs or glyphs that failed to match.
        """
        if cv2 is None or np is None:
            return None
        try:
            if cell_img is None or getattr(cell_img, "size", 0) == 0:
                return None
            gray = cell_img
            if gray.ndim == 3:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            # Upscale the cell so glyphs approach the 96px template scale —
            # correlation quality (and 5-vs-6 discrimination) drops sharply
            # when glyphs stay under ~50px tall.
            h, w = gray.shape[:2]
            f = max(2, int(round(132.0 / max(h, w))))
            if f > 1:
                gray = cv2.resize(gray, (w * f, h * f), interpolation=cv2.INTER_CUBIC)
            binimg = self._binarize_digits_white(gray)
            glyphs = self._glyph_boxes(binimg)
            if not glyphs or len(glyphs) > 2:
                return None
            digits: List[int] = []
            scores: List[float] = []
            for (gx, gy, gw, gh) in glyphs:
                crop_bin = binimg[gy:gy + gh, gx:gx + gw]
                crop_gray = gray[gy:gy + gh, gx:gx + gw]
                digit, score = self._match_glyph(crop_gray, crop_bin)
                if digit is None or score < 0.3:
                    return None  # unknown glyph -> refuse to guess.
                digits.append(int(digit))
                scores.append(float(score))
            if not digits:
                return None
            number = int("".join(str(d) for d in digits))
            conf = sum(scores) / len(scores)
            if not (1 <= number <= 99):
                return None
            if conf < self.confidence_threshold:
                return None
            return (number, float(min(1.0, conf)))
        except Exception:
            return None


class ClusterBoardDetector:
    """Detect cells by finding digit blobs instead of assuming a uniform grid.

    The screenshot's board has 6 rows x 5 cols with big rounded tiles and
    variable spacing — a fixed 5x10 subdivision mis-centers every click.
    This detector:
        1. Binarizes (auto polarity) and morphs digit pixels into blobs.
        2. Groups blobs into rows then clusters columns by x.
        3. Derives a uniform tile size from medians; each tile box is centered
           on its digit with side ~= median row pitch * 0.8 (click-safe).
    Falls back to None when fewer than 4 blobs are found (caller keeps the
    grid-based path).
    """

    def detect(self, image) -> Optional[List[BBox]]:
        """Return per-tile boxes (row-major), or None when no lattice is found.

        Robust to: cleared/empty tiles (lattice fill), uneven tile spacing,
        header/score text above the board (row filtering), and both dark- and
        light-themed boards (polarity auto-detection).
        """
        if cv2 is None or np is None:
            return None
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if getattr(image, "ndim", 2) == 3 else image
            _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # Polarity: keep ink white. Dark text on light bg -> INV gives white
            # text on light-gray bg; flip so bg is black.
            border = np.concatenate([binimg[0, :].ravel(), binimg[-1, :].ravel(),
                                     binimg[:, 0].ravel(), binimg[:, -1].ravel()])
            if float(np.mean(border > 0)) > 0.5:
                binimg = cv2.bitwise_not(binimg)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            dil = cv2.dilate(binimg, kernel, iterations=2)
            n, _labels, stats, centroids = cv2.connectedComponentsWithStats(dil, connectivity=8)
            blobs = []
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if area < 40 or h < 8 or w < 4:
                    continue
                blobs.append((float(centroids[i][0]), float(centroids[i][1]), int(x), int(y), int(w), int(h)))
            if len(blobs) < 4:
                return None
            # --- row grouping by cy proximity -----------------------------------
            med_h = float(np.median([b[5] for b in blobs]))
            y_tol = max(8.0, med_h * 0.6)
            blobs.sort(key=lambda b: b[1])
            y_groups: List[List] = []
            for b in blobs:
                if y_groups and abs(b[1] - float(np.mean([g[1] for g in y_groups[-1]]))) <= y_tol:
                    y_groups[-1].append(b)
                else:
                    y_groups.append([b])
            # Board rows have ~equal blob counts; UI junk is sparse rows.
            counts = [len(g) for g in y_groups]
            if not counts:
                return None
            modal_count = max(set(counts), key=counts.count)
            if modal_count < 3:
                return None
            rows = [g for g in y_groups if len(g) >= max(3, modal_count - 2)]
            if len(rows) < 2:
                return None
            # --- column x-positions: union across rows, snapped -----------------
            xs: List[float] = []
            for g in rows:
                xs.extend(b[0] for b in g)
            xs_sorted = sorted(xs)
            med_w = float(np.median([b[4] for b in blobs]))
            x_tol = max(10.0, med_w * 0.9)
            col_centers: List[float] = []
            for x in xs_sorted:
                if col_centers and x - col_centers[-1][-1] <= x_tol:
                    col_centers[-1].append(x)
                else:
                    col_centers.append([x])
            col_x = [float(np.mean(c)) for c in col_centers]
            # --- lattice fill: emit a box for every (row, col) pair -------------
            pitch_y = float(np.median([
                rows[i + 1][0][1] - rows[i][0][1]
                for i in range(len(rows) - 1)
                if rows[i + 1][0][1] - rows[i][0][1] > 0
            ])) if len(rows) > 1 else med_h * 2.5
            pitch_x = (float(np.median(np.diff(col_x))) if len(col_x) > 1 else med_w * 2.5)
            side = int(np.clip(0.9 * min(pitch_x, pitch_y), 28, 160))
            cells: List[BBox] = []
            for row in rows:
                ry = float(np.mean([b[1] for b in row]))
                for cxc in col_x:
                    # Snap each column center to a nearby blob when present.
                    near = [b for b in row if abs(b[0] - cxc) <= x_tol]
                    cx = float(np.mean([b[0] for b in near])) if near else cxc
                    cells.append((int(cx - side // 2), int(ry - side // 2), side, side))
            if len(cells) < 4:
                return None
            return cells
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Full-board scan
# ---------------------------------------------------------------------------
def build_number_map(
    grid_image,
    cells: Sequence[BBox],
    recognizer: Optional[NumberRecognizer] = None,
    workers: int = 4,
) -> Dict[int, Point]:
    """OCR every cell and map ``number -> (center_x, center_y)`` (image coords).

    Args:
        grid_image: full-board BGR frame that ``cells`` index into.
        cells: bboxes in the same coordinate frame as ``grid_image``.
        recognizer: reused when provided (avoids re-probing tesseract). Its
            ``ocr_cell`` may also be a TemplateRecognizer (same signature).
        workers: parallel OCR workers (tesseract is a subprocess per cell, so
            threads overlap OS wait; template OCR is in-process and fast).

    Returns:
        Dict for numbers found (subset of 1..50). On duplicates the higher-
        confidence reading wins; unreadable cells are skipped silently.
    """
    recog = recognizer or NumberRecognizer()
    number_map: Dict[int, Point] = {}
    conf_map: Dict[int, float] = {}
    if grid_image is None or cells is None:
        return number_map
    # Crop once on this thread; OCR crops (possibly in parallel) below.
    crops: List[Tuple[int, int, int, int, object]] = []
    for (x, y, w, h) in cells:
        try:
            x0, y0 = max(0, int(x)), max(0, int(y))
            crop = grid_image[y0:y0 + max(1, int(h)), x0:x0 + max(1, int(w))]
            if crop is None or getattr(crop, "size", 0) == 0:
                continue
            crops.append((x0, y0, int(w), int(h), crop))
        except Exception:
            continue  # one bad cell must never kill the whole scan.

    def _ocr_one(item: Tuple[int, int, int, int, object]) -> Optional[Tuple[int, Point, float]]:
        x0, y0, w, h, crop = item
        try:
            parsed = recog.ocr_cell(crop)
        except Exception:
            return None
        if parsed is None:
            return None
        value, conf = parsed
        if 1 <= int(value) <= 50:
            return (int(value), (x0 + w // 2, y0 + h // 2), float(conf))
        return None

    results = []
    try:
        n_workers = max(1, min(int(workers or 1), len(crops) or 1))
    except Exception:
        n_workers = 1
    if n_workers > 1 and len(crops) > 1:
        try:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(_ocr_one, crops))
        except Exception:
            results = [r for r in (_ocr_one(c) for c in crops) if r is not None]
    else:
        results = [r for r in (_ocr_one(c) for c in crops) if r is not None]
    for res in results:
        if res is None:
            continue
        value, pt, conf = res
        prev = conf_map.get(value)
        if prev is None or conf > prev:
            number_map[value] = pt
            conf_map[value] = conf
    return number_map


__all__ = [
    "GridDetector",
    "NumberRecognizer",
    "TemplateRecognizer",
    "ClusterBoardDetector",
    "scan_roi",
    "build_number_map",
    "tesseract_available",
    "ensure_dpi_awareness",
]
