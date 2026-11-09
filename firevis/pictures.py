"""Draw a layer as an RGB picture plus a legend description.

Shared by the explorer's viewer (which turns the legend into HTML) and the
Colab plotting objects in firevis/fire.py (which turn it into matplotlib
colorbars and patches), so both show the same image for the same data.

    frame_picture(stack, layer, i)    one date
    overlay_picture(stack, layer)     every date at once, shaded by date
    view_box(stack, window, view)     "fire" area, model "crop", or "whole"
    crop(rgb, box)                    cut a picture to a view box
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from . import render
from .stacks import CLASSES, MASK, RGB, VALUES, Layer, Stack

BACKGROUNDS = ("first", "last", "none")
VIEWS = ("fire", "crop", "whole")


@dataclass
class Legend:
    """What a picture's colours mean.

    kind: "values"  continuous band: lo/hi on `ramp`
          "time"    overlaid dates: `first` → `last` on `ramp`
          "classes" land cover: `classes` {value: (name, colour)}
          "mask"    one date of a fire mask: `colour`, `name`, `count` pixels
          "empty"   nothing to draw: `message`
          "none"    no legend (RGB images)
    """

    kind: str
    lo: float = float("nan")
    hi: float = float("nan")
    ramp: str = "values"
    first: str = ""
    last: str = ""
    note: str = ""
    classes: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    colour: Tuple[int, int, int] = (0, 0, 0)
    name: str = ""
    count: int = 0
    message: str = ""


@dataclass
class Picture:
    rgb: np.ndarray  # (H, W, 3) uint8
    legend: Legend


def can_overlay(layer: Layer) -> bool:
    """Overlays need several dates of a mask or a continuous band."""
    return layer.count > 1 and layer.style in (MASK, VALUES)


def shared_span(arrays: Sequence[np.ndarray], low: float = 2, high: float = 98):
    """One (lo, hi) stretch for several frames, so their colours compare."""
    values = [np.asarray(a, np.float32)[np.isfinite(a)] for a in arrays]
    values = [v for v in values if v.size]
    if not values:
        return None
    lo, hi = np.percentile(np.concatenate(values), [low, high])
    return (float(lo), float(hi))


def _background_name(background: str) -> str:
    name = (background or "none").lower().split()[0]  # "First day" -> "first"
    if name not in BACKGROUNDS:
        raise ValueError(f"background must be one of {BACKGROUNDS}, not {background!r}")
    return name


def base_canvas(stack: Stack, layer: Layer, window: int, frame: Optional[int],
                background: str = "first") -> np.ndarray:
    """Faded false colour under fire masks (or a plain sheet).

    frame=None (overlays) uses the first or last day; a frame uses that day's
    own image when the dates line up.
    """
    shape = np.shape(layer.read(0))[-2:]
    bg = stack.background(window)
    choice = _background_name(background)
    if bg is None or choice == "none":
        return render.background(shape)
    if frame is None:
        frame = 0 if choice == "first" else bg.count - 1
    else:
        date = layer.dates[frame]
        frame = bg.dates.index(date) if date in bg.dates else 0
    image = render.true_rgb(bg.read(frame))
    if image.shape[:2] != tuple(shape):
        return render.background(shape)
    return render.background(shape, image)


def frame_picture(stack: Stack, layer: Layer, frame: int, window: int = 0,
                  background: str = "first", span: Optional[Tuple[float, float]] = None) -> Picture:
    """One date of a layer. `span` fixes the value range of continuous bands."""
    array = layer.read(frame)
    if layer.style == RGB:
        return Picture(render.true_rgb(array), Legend("none"))
    if layer.style == CLASSES:
        rgb, classes = render.classes_rgb(array, layer.palette)
        return Picture(rgb, Legend("classes", classes=classes))
    if layer.style == MASK:
        canvas = base_canvas(stack, layer, window, frame, background)
        colour = render.ramp_colour(layer.ramp, 0.55)
        mask = np.asarray(array, bool)
        if layer.outlined:
            render.paint(canvas, mask, colour, 0.3)
            render.paint(canvas, render.outline(mask), colour, 0.95)
        else:
            render.paint(canvas, mask, colour, 0.85)
        return Picture(render.to_uint8(canvas), Legend(
            "mask", colour=tuple(int(c) for c in colour), name=layer.name, count=int(mask.sum())))
    rgb, (lo, hi) = render.values_rgb(array, span=span)
    return Picture(rgb, Legend("values", lo=lo, hi=hi, note="2nd–98th percentile"))


def overlay_picture(stack: Stack, layer: Layer, window: int = 0,
                    background: str = "first") -> Picture:
    """Every date of a layer at once, slightly transparent, light (first) to dark (last)."""
    if not can_overlay(layer):
        raise ValueError(f"Can't overlay {layer.name}: overlays need several dates of a "
                         "fire mask or a continuous band")
    n = layer.count
    positions = np.linspace(0, 1, n)
    frames = [layer.read(i) for i in range(n)]
    if layer.style == MASK:
        canvas = base_canvas(stack, layer, window, None, background)
        render.overlay_masks(frames, positions, layer.ramp, canvas,
                             cumulative=layer.cumulative, outlined=layer.outlined)
        pixels = int(np.logical_or.reduce([np.asarray(f, bool) for f in frames]).sum())
        if not pixels:
            return Picture(render.to_uint8(canvas), Legend(
                "empty", message=f"No {layer.name} pixels on any of the {n} days."))
        note = (f"colour = first day burned · {pixels:,} pixels" if layer.cumulative
                else f"later days on top · {pixels:,} pixels")
    else:
        canvas = render.background(np.shape(frames[0]))  # values: their own colours only
        render.overlay_values(frames, positions, layer.ramp, canvas)
        note = "later days on top · stronger = higher value"
    return Picture(render.to_uint8(canvas), Legend(
        "time", ramp=layer.ramp, first=layer.dates[0], last=layer.dates[-1], note=note))


def view_box(stack: Stack, window: int = 0, view: str = "fire"):
    """(top, bottom, left, right) fractions for a view, or None for the whole image.

    fire:  the fire's area (when the stack knows it)
    crop:  the dataset's 256 x 256 model crop (fire folders; an .npz already is it)
    whole: everything
    """
    view = (view or "whole").lower()
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, not {view!r}")
    if view == "fire":
        return stack.focus(window)
    if view == "crop":
        return stack.crop_box(window)
    return None


def crop(rgb: np.ndarray, box) -> np.ndarray:
    """Cut an image to (top, bottom, left, right) fractions (None = unchanged)."""
    if box is None:
        return rgb
    height, width = rgb.shape[:2]
    top, bottom, left, right = box
    rows = slice(int(round(top * height)), max(int(round(bottom * height)), int(round(top * height)) + 1))
    cols = slice(int(round(left * width)), max(int(round(right * width)), int(round(left * width)) + 1))
    return rgb[rows, cols]
