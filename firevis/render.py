"""Turn arrays into images: stretches, colour ramps, overlays and PNG bytes.

Only numpy is required. Pillow encodes PNGs when it is installed (it is in
Colab); otherwise a small built-in encoder does.
"""

from __future__ import annotations

import io
import struct
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MISSING = (229, 229, 234)  # NaN pixels
PLAIN = (246, 246, 248)  # overlay background when there is none

# Colour ramps as evenly spaced anchors, light/early first.
RAMPS: Dict[str, List[str]] = {
    "values": ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"],  # viridis
    "fire": ["#ffe066", "#ffa62b", "#f3722c", "#d62828", "#8b1e3f", "#4a0d2a"],
    "ice": ["#a5d8ff", "#4dabf7", "#1c7ed6", "#1864ab", "#0b3d91", "#081c4a"],
    "ember": ["#fff1a8", "#fcbf49", "#f77f00", "#c1121f", "#780000", "#3d0000"],
}

# Land cover classes: value -> (name, colour).
PALETTES: Dict[str, Dict[int, Tuple[str, str]]] = {
    "esri": {  # Esri 10 m land use / land cover
        1: ("Water", "#419bdf"), 2: ("Trees", "#397d49"), 4: ("Flooded vegetation", "#7a87c6"),
        5: ("Crops", "#e49635"), 7: ("Built area", "#c4281b"), 8: ("Bare ground", "#a59b8f"),
        9: ("Snow / ice", "#a8ebff"), 10: ("Clouds", "#616161"), 11: ("Rangeland", "#e3e2c3"),
    },
    "igbp": {  # MODIS LC_Type1
        1: ("Evergreen needleleaf", "#05450a"), 2: ("Evergreen broadleaf", "#086a10"),
        3: ("Deciduous needleleaf", "#54a708"), 4: ("Deciduous broadleaf", "#78d203"),
        5: ("Mixed forest", "#009900"), 6: ("Closed shrubland", "#c6b044"),
        7: ("Open shrubland", "#dcd159"), 8: ("Woody savanna", "#dade48"),
        9: ("Savanna", "#fbff13"), 10: ("Grassland", "#b6ff05"),
        11: ("Wetland", "#27ff87"), 12: ("Cropland", "#c24f44"), 13: ("Urban", "#a5a5a5"),
        14: ("Cropland / natural mosaic", "#ff6d4c"), 15: ("Snow / ice", "#69fff8"),
        16: ("Barren", "#f9ffa4"), 17: ("Water", "#1c0dff"),
    },
}
_SPARE = ["#8dd3c7", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
          "#bc80bd", "#ccebc5", "#ffed6f", "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00"]


def hex_rgb(colour: str) -> Tuple[int, int, int]:
    colour = colour.lstrip("#")
    return tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4))


def ramp_colour(name: str, t: float) -> np.ndarray:
    """Colour at position t (0..1) along a ramp, as float RGB 0..255."""
    anchors = np.array([hex_rgb(c) for c in RAMPS[name]], dtype=np.float32)
    x = float(np.clip(t, 0.0, 1.0)) * (len(anchors) - 1)
    i = min(int(x), len(anchors) - 2)
    return anchors[i] + (anchors[i + 1] - anchors[i]) * (x - i)


def ramp_css(name: str) -> str:
    """CSS linear-gradient for a ramp, for legends."""
    return f"linear-gradient(90deg, {', '.join(RAMPS[name])})"


def stretch(array: np.ndarray, low: float = 2, high: float = 98):
    """Scale finite values to 0..1 by percentiles. Returns (scaled, valid, (lo, hi))."""
    array = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(array)
    if not valid.any():
        return np.zeros(array.shape, np.float32), valid, (float("nan"), float("nan"))
    lo, hi = np.percentile(array[valid], [low, high])
    if hi <= lo:
        lo, hi = float(array[valid].min()), float(array[valid].max())
    scaled = np.zeros(array.shape, np.float32)
    if hi > lo:
        scaled[valid] = np.clip((array[valid] - lo) / (hi - lo), 0, 1)
    return scaled, valid, (float(lo), float(hi))


def values_rgb(array: np.ndarray, ramp: str = "values", span: Optional[Tuple[float, float]] = None):
    """A continuous band on a colour ramp. Returns (rgb uint8, (lo, hi)).

    `span` fixes the (lo, hi) range, so several images share one colour scale;
    by default each image is stretched to its own 2nd-98th percentile.
    """
    if span is None:
        scaled, valid, span = stretch(array)
    else:
        array = np.asarray(array, dtype=np.float32)
        valid = np.isfinite(array)
        lo, hi = span
        scaled = np.zeros(array.shape, np.float32)
        if hi > lo:
            scaled[valid] = np.clip((array[valid] - lo) / (hi - lo), 0, 1)
        span = (float(lo), float(hi))
    anchors = np.array([hex_rgb(c) for c in RAMPS[ramp]], dtype=np.float32)
    x = scaled * (len(anchors) - 1)
    i = np.minimum(x.astype(int), len(anchors) - 2)
    frac = (x - i)[..., None]
    rgb = anchors[i] + (anchors[i + 1] - anchors[i]) * frac
    rgb[~valid] = MISSING
    return rgb.astype(np.uint8), span


def true_rgb(array: np.ndarray) -> np.ndarray:
    """(3, H, W) or (H, W, 3) bands -> RGB with one shared percentile stretch."""
    array = np.asarray(array, dtype=np.float32)
    if array.shape[0] == 3 and array.ndim == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    scaled, valid, _ = stretch(array, 1, 99)
    rgb = (scaled * 255).astype(np.uint8)
    rgb[~valid.all(axis=-1)] = MISSING
    return rgb


def classes_rgb(array: np.ndarray, palette: Optional[str] = None):
    """Class values -> colours. Returns (rgb uint8, {value: (name, colour)} present)."""
    array = np.asarray(array)
    valid = np.isfinite(array)
    codes = np.zeros(array.shape, dtype=np.int64)
    codes[valid] = np.rint(array[valid]).astype(np.int64)
    known = PALETTES.get(palette or "", {})
    rgb = np.empty(array.shape + (3,), dtype=np.uint8)
    rgb[...] = MISSING
    legend = {}
    for n, value in enumerate(np.unique(codes[valid])):
        name, colour = known.get(int(value), (f"Class {int(value)}", _SPARE[n % len(_SPARE)]))
        rgb[valid & (codes == value)] = hex_rgb(colour)
        legend[int(value)] = (name, colour)
    return rgb, legend


def mask_of(array: np.ndarray) -> np.ndarray:
    """Fire-style mask: finite and above zero (NaN never counts)."""
    array = np.asarray(array)
    return np.isfinite(array) & (array > 0)


def outline(mask: np.ndarray) -> np.ndarray:
    """Pixels of a mask that touch a pixel outside it."""
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner


def background(shape, rgb: Optional[np.ndarray] = None, dim: float = 0.55) -> np.ndarray:
    """Float RGB base for overlays: a faded image, or a plain sheet."""
    if rgb is None:
        out = np.empty(tuple(shape) + (3,), np.float32)
        out[...] = PLAIN
        return out
    grey = rgb.astype(np.float32).mean(axis=-1, keepdims=True)
    faded = 0.5 * rgb.astype(np.float32) + 0.5 * grey  # half desaturated
    return faded * dim + 255 * (1 - dim)


def paint(canvas: np.ndarray, where: np.ndarray, colour, alpha: float) -> None:
    """Blend `colour` into `canvas` (float RGB) where `where` is True, in place."""
    if np.any(where):
        canvas[where] = canvas[where] * (1 - alpha) + np.asarray(colour, np.float32) * alpha


def overlay_masks(masks: Sequence[np.ndarray], positions: Sequence[float], ramp: str,
                  canvas: np.ndarray, alpha: float = 0.65, cumulative: bool = False,
                  outlined: bool = False) -> np.ndarray:
    """Draw one mask per day on `canvas`, each tinted by its place in time (0..1).

    Later days are drawn on top. For a cumulative mask (one that only grows),
    each pixel is coloured by the first day it appears. Outlined masks (like
    perimeters) get a light fill and a solid edge.
    """
    seen = np.zeros(canvas.shape[:2], dtype=bool)
    for mask, t in zip(masks, positions):
        mask = np.asarray(mask, dtype=bool)
        if cumulative:
            mask, seen = mask & ~seen, seen | mask
        colour = ramp_colour(ramp, t)
        if outlined:
            paint(canvas, mask, colour, alpha * 0.35)
            paint(canvas, outline(mask), colour, 0.95)
        else:
            paint(canvas, mask, colour, alpha)
    return canvas


def overlay_values(frames: Sequence[np.ndarray], positions: Sequence[float], ramp: str,
                   canvas: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Overlay continuous frames: each day tinted by time, opacity by value."""
    stacked = np.stack([np.asarray(f, np.float32) for f in frames])
    scaled, valid, _ = stretch(stacked)
    for i, t in enumerate(positions):
        weight = (scaled[i] * alpha)[..., None] * valid[i][..., None]
        colour = ramp_colour(ramp, t)
        canvas[:] = canvas * (1 - weight) + colour * weight
    return canvas


def to_uint8(canvas: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)


def png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 image as PNG."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    try:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(buffer, format="PNG", compress_level=3)
        return buffer.getvalue()
    except ImportError:
        return _png_fallback(rgb)


def _png_fallback(rgb: np.ndarray) -> bytes:
    height, width = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
