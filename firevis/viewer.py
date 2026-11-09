"""The image viewer shown in the explorer's preview pane.

    ┌ [layer ▾] [window ▾] [background ▾] ──────────┐
    │ ◧ overlay                               ⤢ size │
    │                  image                         │
    ├ legend ───────────────────────────────────────┤
    │        ◀        2017-04-25 · 3 / 8        ▶    │
    └───────────────────────────────────────────────┘

One day at a time: the arrows step through the layer's dates, and the date sits
bottom-middle. Overlaid: every day is drawn at once, slightly transparent and
shaded from light (first day) to dark (last day); the bottom shows the range.
Fire layers start overlaid; the top-left corner toggles it for any layer that
can be overlaid. The top-right corner makes the image big or small.

Everything is drawn into widgets (an Image and a few HTML labels), so it shows
up in Colab as well as Jupyter.
"""

from __future__ import annotations

import html
from typing import Callable, List, Optional

import ipywidgets as widgets
import numpy as np

from . import pictures, render
from .stacks import MASK, Layer, Stack

BACKGROUNDS = ("First day", "Last day", "None")
VIEWS = ("Fire area", "Whole image")


def _button(text: str, *classes: str, tooltip: str = "") -> widgets.Button:
    button = widgets.Button(description=text, tooltip=tooltip, layout=widgets.Layout(width="auto"))
    for name in classes:
        button.add_class(name)
    return button


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "–"
    if value != 0 and (abs(value) >= 10000 or abs(value) < 0.01):
        return f"{value:.3g}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


class StackViewer:
    """Step through or overlay the layers of a Stack.

    on_size: called with True/False when the size corner is toggled, so the
    container can make room.
    """

    def __init__(self, stack: Stack, on_size: Optional[Callable[[bool], None]] = None):
        self.stack = stack
        self.on_size = on_size
        self.window = 0
        self.layers: List[Layer] = stack.layers(0)
        if not self.layers:
            raise ValueError(f"Nothing to show in {stack.title}")
        self.index = 0
        self.frame = 0
        self.overlaid = self.layers[0].overlay
        self.big = False
        self._chose_overlay = False  # the user switched overlay on themselves

        self._layer_dd = widgets.Dropdown(
            options=[(layer.label, i) for i, layer in enumerate(self.layers)],
            layout=widgets.Layout(flex="1 1 220px", width="auto"))
        self._window_dd = widgets.Dropdown(
            options=[(label, w) for w, label in enumerate(stack.windows)],
            layout=widgets.Layout(flex="1 1 220px", width="auto",
                                  display="" if len(stack.windows) > 1 else "none"))
        self._bg_dd = widgets.Dropdown(
            options=[(f"Background: {b}", b) for b in BACKGROUNDS],
            layout=widgets.Layout(flex="0 1 190px", width="auto"))
        self._view_dd = widgets.Dropdown(
            options=[(f"View: {v}", v) for v in VIEWS],
            layout=widgets.Layout(flex="0 1 170px", width="auto"))
        for dropdown in (self._layer_dd, self._window_dd, self._bg_dd, self._view_dd):
            dropdown.add_class("fx-select")
        self._layer_dd.observe(self._on_layer, "value")
        self._window_dd.observe(self._on_window, "value")
        self._bg_dd.observe(lambda _: self.render(), "value")
        self._view_dd.observe(lambda _: self.render(), "value")
        controls = widgets.HBox([self._layer_dd, self._window_dd, self._bg_dd, self._view_dd])
        controls.add_class("fx-controls")

        self._image = widgets.Image(format="png", layout=widgets.Layout(width="100%"))
        self._image.add_class("fx-image")
        self._overlay_btn = _button("◧ Overlay", "fx-corner", "fx-corner-left",
                                    tooltip="Overlay every day / one day at a time")
        self._size_btn = _button("⤢", "fx-corner", "fx-corner-right", tooltip="Big / small")
        self._overlay_btn.on_click(lambda _: self.toggle_overlay())
        self._size_btn.on_click(lambda _: self.toggle_size())
        self._stage = widgets.Box([self._image, self._overlay_btn, self._size_btn])
        self._stage.add_class("fx-stage")

        self._legend = widgets.HTML()
        self._legend.add_class("fx-legend")
        self._prev = _button("◀", "fx-arrow", tooltip="Previous day")
        self._next = _button("▶", "fx-arrow", tooltip="Next day")
        self._prev.on_click(lambda _: self.step(-1))
        self._next.on_click(lambda _: self.step(1))
        self._date = widgets.HTML()
        self._date.add_class("fx-date")
        nav = widgets.HBox([self._prev, self._date, self._next])
        nav.add_class("fx-nav")

        self.widget = widgets.VBox([controls, self._stage, self._legend, nav])
        self.widget.add_class("fx-viewer")
        self.render()

    # ----- state ------------------------------------------------------------

    @property
    def layer(self) -> Layer:
        return self.layers[self.index]

    def can_overlay(self, layer: Optional[Layer] = None) -> bool:
        return pictures.can_overlay(layer or self.layer)

    def show_layer(self, index: int) -> None:
        """Switch layer, staying on the same date when the new layer has it."""
        old_date = self.layer.dates[self.frame] if self.layer.dates else None
        self.index = index
        dates = self.layer.dates
        self.frame = dates.index(old_date) if old_date in dates else min(self.frame, len(dates) - 1)
        self.overlaid = self.can_overlay() and (self.layer.overlay or self._chose_overlay)
        if self._layer_dd.value != index:
            self._layer_dd.value = index  # fires _on_layer, which renders
        else:
            self.render()

    def step(self, delta: int) -> None:
        self.frame = int(np.clip(self.frame + delta, 0, self.layer.count - 1))
        self.render()

    def toggle_overlay(self) -> None:
        if self.can_overlay():
            self.overlaid = not self.overlaid
            self._chose_overlay = self.overlaid
            self.render()

    def toggle_size(self) -> None:
        self.big = not self.big
        if self.on_size:
            self.on_size(self.big)
        self.render()

    def _on_layer(self, change) -> None:
        if change["new"] != self.index:
            self.show_layer(change["new"])
        else:
            self.render()

    def _on_window(self, change) -> None:
        self.window = change["new"]
        self.layers = self.stack.layers(self.window)
        self.index = min(self.index, len(self.layers) - 1)
        self.frame = min(self.frame, self.layer.count - 1)
        self.render()

    # ----- drawing ----------------------------------------------------------

    def render(self) -> None:
        layer = self.layer
        overlay_ok = self.can_overlay()
        self.overlaid = self.overlaid and overlay_ok
        self._overlay_btn.disabled = not overlay_ok
        self._overlay_btn.description = "◧ Overlay: on" if self.overlaid else "◧ Overlay: off"
        self._size_btn.description = "⤡ Small" if self.big else "⤢ Big"
        _set(self._stage, "fx-big", self.big)
        _set(self.widget, "fx-viewer-big", self.big)
        _set(self._overlay_btn, "fx-on", self.overlaid)
        has_bg = self.stack.background(self.window) is not None
        self._bg_dd.layout.display = "" if has_bg and layer.style == MASK else "none"
        try:
            box = self.stack.focus(self.window)
            self._view_dd.layout.display = "" if box else "none"
            background = self._bg_dd.value
            picture = (pictures.overlay_picture(self.stack, layer, self.window, background)
                       if self.overlaid else
                       pictures.frame_picture(self.stack, layer, self.frame, self.window, background))
            rgb = picture.rgb
            if box and self._view_dd.value == "Fire area":
                rgb = pictures.crop(rgb, box)
        except Exception as error:  # unreadable band, missing library...
            self._legend.value = f'<span class="fx-legend-error">⚠️ {html.escape(str(error))}</span>'
            self._image.value = b""
            return
        self._image.value = render.png_bytes(rgb)
        self._legend.value = legend_html(picture.legend)
        self._update_nav()

    def _update_nav(self) -> None:
        layer = self.layer
        if self.overlaid:
            text = f"{layer.dates[0]} → {layer.dates[-1]} · {layer.count} days overlaid"
            self._prev.layout.visibility = self._next.layout.visibility = "hidden"
        else:
            date = layer.dates[self.frame] or "—"
            text = f"<b>{html.escape(date)}</b> · {self.frame + 1} / {layer.count}"
            visible = "visible" if layer.count > 1 else "hidden"
            self._prev.layout.visibility = self._next.layout.visibility = visible
            self._prev.disabled = self.frame == 0
            self._next.disabled = self.frame >= layer.count - 1
        self._date.value = text

    def close(self) -> None:
        self.widget.close()
        close = getattr(self.stack, "close", None)
        if close:
            close()


def _set(widget, name: str, on: bool) -> None:
    (widget.add_class if on else widget.remove_class)(name)


def _bar(left: str, gradient: str, right: str, note: str = "") -> str:
    note_html = f'<span class="fx-legend-note">{html.escape(note)}</span>' if note else ""
    return (f"<span>{html.escape(left)}</span>"
            f'<span class="fx-bar" style="background:{gradient}"></span>'
            f"<span>{html.escape(right)}</span>{note_html}")


def legend_html(legend: pictures.Legend) -> str:
    """The viewer's legend line for a picture."""
    if legend.kind == "values":
        return _bar(_fmt(legend.lo), render.ramp_css(legend.ramp), _fmt(legend.hi), legend.note)
    if legend.kind == "time":
        return _bar(legend.first, render.ramp_css(legend.ramp), legend.last, legend.note)
    if legend.kind == "classes":
        return _class_legend(legend.classes)
    if legend.kind == "mask":
        count = legend.count
        return (f'<span class="fx-swatch" style="background:rgb{legend.colour}"></span>'
                f"{html.escape(legend.name)} · {count:,} pixel{'s' if count != 1 else ''}")
    if legend.kind == "empty":
        return f'<span class="fx-legend-note">{html.escape(legend.message)}</span>'
    return ""


def _class_legend(classes) -> str:
    return "".join(
        f'<span class="fx-class"><span class="fx-swatch" style="background:{colour}"></span>'
        f"{html.escape(name)}</span>"
        for _, (name, colour) in sorted(classes.items()))
