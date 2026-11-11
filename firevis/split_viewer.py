"""The fire split viewer shown in the explorer's preview pane.

    ┌ 4,379 burned pixels → 13 fires · 3 splits · 2 merges ────────┐
    │ [Before] [After · 6 fires]   [background ▾] [view ▾]  ⤢ Big │
    │                                                              │
    │   Before: one image          After: one image per fire       │
    │   every burned pixel         ┌────────┐ ┌────────┐           │
    │                              │ Fire 1 │ │ Fire 4 │ ...       │
    │                              └────────┘ └────────┘           │
    ├ legend: first target day ▬▬▬▬▬▬▬ last target day ───────────┤
    │ ◀ Layers                                                     │
    └──────────────────────────────────────────────────────────────┘

Every image is the burned pixels overlaid over time: light (first target day)
to dark (last), the same day always the same colour, on one crop, so a fire's
pixels sit in the same place in the before and in its after.
"""

from __future__ import annotations

import html
from typing import Callable, List, Optional, Tuple

import ipywidgets as widgets

from . import pictures, render
from .split import FireSplit, SplitStack
from .stacks import Layer
from .viewer import BACKGROUNDS, VIEWS, _button, _set, legend_html

BEFORE, AFTER = "before", "after"


def layer_picture(stack: SplitStack, layer: Layer, background: str) -> pictures.Picture:
    """All of a layer's days at once (or its one day, for a one-window archive)."""
    if pictures.can_overlay(layer):
        return pictures.overlay_picture(stack, layer, 0, background)
    return pictures.frame_picture(stack, layer, 0, 0, background)


class SplitViewer:
    """Before and after of one FireSplit.

    on_size: called with True/False when ⤢ is toggled, so the container can make room.
    on_back: called by the ◀ Layers button (the explorer reopens the layer viewer).
    """

    def __init__(self, split: FireSplit, on_size: Optional[Callable[[bool], None]] = None,
                 on_back: Optional[Callable[[], None]] = None, mode: str = AFTER):
        self.split = split
        self.stack = split.stack()
        self.on_size = on_size
        self.mode = mode if mode in (BEFORE, AFTER) else AFTER
        self.big = False

        self._summary = widgets.HTML(self._summary_html())
        self._summary.add_class("fx-split-summary")

        count = len(self.stack.fire_layers)
        self._before_btn = _button("Before", "fx-tab", tooltip="Every burned pixel of the archive")
        self._after_btn = _button(f"After · {count} fire{'s' if count != 1 else ''}", "fx-tab",
                                  tooltip="Each fire's own burned pixels")
        self._before_btn.on_click(lambda _: self.show(BEFORE))
        self._after_btn.on_click(lambda _: self.show(AFTER))
        self._bg_dd = widgets.Dropdown(options=[(f"Background: {b}", b) for b in BACKGROUNDS],
                                       layout=widgets.Layout(flex="0 1 190px", width="auto"))
        self._view_dd = widgets.Dropdown(options=[(f"View: {v}", v) for v in VIEWS],
                                         layout=widgets.Layout(flex="0 1 170px", width="auto"))
        for dropdown in (self._bg_dd, self._view_dd):
            dropdown.add_class("fx-select")
            dropdown.observe(lambda _: self.render(), "value")
        self._bg_dd.layout.display = "" if self.stack.background() is not None else "none"
        self._view_dd.layout.display = "" if self.stack.focus() else "none"
        self._size_btn = _button("⤢ Big", "fx-tab", tooltip="Big / small")
        self._size_btn.on_click(lambda _: self.toggle_size())
        controls = widgets.HBox([self._before_btn, self._after_btn, self._bg_dd, self._view_dd,
                                 self._size_btn])
        controls.add_class("fx-controls")

        # Before: one image.
        self._before_image = widgets.Image(format="png", layout=widgets.Layout(width="100%"))
        self._before_image.add_class("fx-image")
        self._before_stage = widgets.Box([self._before_image])
        self._before_stage.add_class("fx-stage")
        self._before_note = widgets.HTML()
        self._before_note.add_class("fx-split-note")

        # After: one card per fire, then the pixels of no fire (id -1).
        afters = list(self.stack.fire_layers.items())
        if self.stack.no_fire is not None:
            afters.append((-1, self.stack.no_fire))
        self._cards: List[Tuple[Layer, widgets.Image]] = []
        cards = []
        for fire_id, layer in afters:
            image = widgets.Image(format="png", layout=widgets.Layout(width="100%"))
            image.add_class("fx-image")
            caption = widgets.HTML(self._caption(fire_id))
            caption.add_class("fx-split-caption")
            card = widgets.VBox([image, caption])
            card.add_class("fx-split-card")
            self._cards.append((layer, image))
            cards.append(card)
        self._grid = widgets.Box(cards)
        self._grid.add_class("fx-split-grid")
        self._after_note = widgets.HTML(self._after_note_html())
        self._after_note.add_class("fx-split-note")

        self._legend = widgets.HTML()
        self._legend.add_class("fx-legend")
        back = _button("◀ Layers", "fx-arrow", "fx-split-back", tooltip="Back to the archive's layers")
        back.on_click(lambda _: on_back and on_back())
        back.layout.display = "" if on_back else "none"
        self._back_btn = back

        self._body = widgets.VBox()
        self.widget = widgets.VBox([self._summary, controls, self._body, self._legend, back])
        self.widget.add_class("fx-viewer")
        self.widget.add_class("fx-split")
        self.render()

    # ----- state ------------------------------------------------------------

    def show(self, mode: str) -> None:
        """Switch to "before" or "after"."""
        if mode not in (BEFORE, AFTER):
            raise ValueError(f"mode must be {BEFORE!r} or {AFTER!r}, not {mode!r}")
        self.mode = mode
        self.render()

    def toggle_size(self) -> None:
        self.big = not self.big
        if self.on_size:
            self.on_size(self.big)
        self.render()

    # ----- drawing ----------------------------------------------------------

    def render(self) -> None:
        before = self.mode == BEFORE
        _set(self._before_btn, "fx-on", before)
        _set(self._after_btn, "fx-on", not before)
        _set(self._size_btn, "fx-on", self.big)
        self._size_btn.description = "⤡ Small" if self.big else "⤢ Big"
        _set(self._before_stage, "fx-big", self.big)
        _set(self.widget, "fx-viewer-big", self.big)
        background = self._bg_dd.value
        box = self.stack.focus() if self._view_dd.value == "Fire area" else None
        try:
            if before:
                picture = layer_picture(self.stack, self.stack.before, background)
                self._before_image.value = render.png_bytes(pictures.crop(picture.rgb, box))
                self._before_note.value = self._before_note_html()
                self._body.children = [self._before_stage, self._before_note]
                self._legend.value = legend_html(picture.legend)
            else:
                for layer, image in self._cards:
                    picture = layer_picture(self.stack, layer, background)
                    image.value = render.png_bytes(pictures.crop(picture.rgb, box))
                self._body.children = [self._grid, self._after_note]
                self._legend.value = self._time_legend()
        except Exception as error:  # unreadable archive, missing library...
            self._legend.value = f'<span class="fx-legend-error">⚠️ {html.escape(str(error))}</span>'

    def close(self) -> None:
        self.widget.close()

    # ----- text -------------------------------------------------------------

    def _summary_html(self) -> str:
        split = self.split
        fires = len(split.events)
        settings = split.settings
        line = (f"<b>{split.burned:,}</b> burned pixel{'s' if split.burned != 1 else ''} in "
                f"{split.windows} window{'s' if split.windows != 1 else ''} → "
                f"<b>{fires}</b> fire{'s' if fires != 1 else ''} · "
                f"<b>{split.splits}</b> split{'s' if split.splits != 1 else ''} · "
                f"{split.merges} merge{'s' if split.merges != 1 else ''}")
        detail = (f"Fire splitting at {settings.scale_km:g} km: pixels within {settings.link_pixels:g} px "
                  f"are one fire, a fire is remembered {settings.memory_days} days, and a front that "
                  f"stays in parts {settings.min_growth_days} days splits.")
        return f'<div>{line}</div><div class="fx-split-detail">{html.escape(detail)}</div>'

    def _caption(self, fire_id: int) -> str:
        """Title, pixels, target days and story of one after (-1: the pixels of no fire)."""
        split = self.split
        if fire_id < 0:
            title = "No fire"
            what = "New ignitions no fire reaches: left out of every fire's training loss."
        else:
            title = f"Fire {fire_id}"
            event = next(e for e in split.events if e.event_id == fire_id)
            what = split.describe(event)
            what = what[0].upper() + what[1:] + "."
        pixels = split.pixels(fire_id)
        windows = split.windows_of(fire_id)
        dates = split.target_dates
        span = (f"{dates[windows[0]]} → {dates[windows[-1]]}" if len(windows) > 1
                else dates[windows[0]] if windows else "")
        share = f" ({pixels / split.burned:.0%})" if split.burned else ""
        return (f"<b>{html.escape(title)}</b> · {pixels:,} px{share}"
                f'<div class="fx-split-when">{html.escape(span)}</div>'
                f"<div>{html.escape(what)}</div>")

    def _before_note_html(self) -> str:
        split = self.split
        dates = split.target_dates
        span = f"{dates[0]} → {dates[-1]}" if len(dates) > 1 else dates[0]
        return (f"Every window's target (newly burned the next day), {html.escape(span)}: "
                f"{split.burned:,} pixels, before fire splitting gives each one to a fire.")

    def _after_note_html(self) -> str:
        split = self.split
        if not split.events and not split.unowned:
            return "No fires: the archive has no burned pixels."
        parts = []
        if not self.stack.fire_layers:
            parts.append("No fire owns a burned pixel.")
        quiet = split.quiet
        if quiet:
            ids = ", ".join(str(e.event_id) for e in quiet)
            parts.append(f"{len(quiet)} more fire{'s' if len(quiet) != 1 else ''} own no burned pixel "
                         f"(fire{'s' if len(quiet) != 1 else ''} {ids}): split branches that never grew "
                         "into a target, or heat before the first target.")
        parts.append("The afters add up to the before: every burned pixel is in exactly one.")
        return html.escape(" ".join(parts))

    def _time_legend(self) -> str:
        dates = self.split.target_dates
        if len(dates) < 2:
            return legend_html(pictures.Legend("none"))
        return legend_html(pictures.Legend(
            "time", ramp="fire", first=dates[0], last=dates[-1],
            note="colour = day the pixel burned, the same in every image"))
