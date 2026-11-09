"""Load one fire from a link and plot it from notebook cells.

    import firevis
    fire = firevis.load("https://huggingface.co/datasets/SamuelWu318/ts-satfire/tree/main/ts-satfire/20562846")
    win = firevis.load("SamuelWu318/ts-satfire-6-1/val/p20562846.npz")

    fire.channels()                    # every channel name
    fire.dates("I4")                   # its dates
    fire.show("I4", "2017-04-25")      # one date (a date string or an index)
    fire.overlay("Active fire")        # every date at once, light (first) to dark (last)
    fire.grid("I4")                    # every date as small pictures

    both = firevis.compare(fire, win)  # side by side: same functions
    both.channels(); both.show("I4", "2017-04-25"); both.overlay("I4"); both.grid("I4")

A link is a fire folder (VIIRS_Day/ plus FirePred/ or VIIRS_Night/) or an .npz,
as a Hugging Face URL, "owner/dataset/path", an explorer Entry, or a local path.
Files download once into the mirror folder (default hf_mirror/), like in the
explorer. Plots are matplotlib and return their Axes; save one with
ax.figure.savefig("fire.png").
"""

from __future__ import annotations

import difflib
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import unquote

import numpy as np

from . import pictures, render
from .hub import Entry, HubSource
from .stacks import (VALUES, FireStack, Layer, NpzStack, Stack, date_of, is_fire_folder,
                     same_channel_names)

Date = Union[int, str]
_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ----- links -------------------------------------------------------------------


def parse_link(link) -> Tuple[str, str, str]:
    """A Hub link -> (repo_id, path inside it, revision).

    Accepts https://huggingface.co/datasets/<owner>/<repo>/{tree,blob,resolve}/<rev>/<path>,
    hf://datasets/..., datasets/<owner>/<repo>/<path> and <owner>/<repo>/<path>.
    """
    if isinstance(link, Entry):
        return link.repo_id, link.path, "main"
    text = str(link).strip()
    text = re.sub(r"^hf://", "", text)
    text = re.sub(r"^(https?://)?(www\.)?huggingface\.co/", "", text)
    text = text.split("?", 1)[0].split("#", 1)[0]
    parts = [unquote(p) for p in text.split("/") if p]
    if parts and parts[0] == "datasets":
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError(f"Not a dataset link: {link!r} (expected owner/dataset/path)")
    repo_id, rest, revision = "/".join(parts[:2]), parts[2:], "main"
    if len(rest) >= 2 and rest[0] in ("tree", "blob", "resolve"):
        revision, rest = rest[1], rest[2:]
    return repo_id, "/".join(rest), revision


_SOURCES: Dict[tuple, HubSource] = {}


def _source_for(owner: str, mirror_dir: str, revision: str) -> HubSource:
    """One shared HubSource per owner/mirror/revision, so names are listed once."""
    key = (owner, str(Path(mirror_dir).expanduser().resolve()), revision)
    if key not in _SOURCES:
        _SOURCES[key] = HubSource(owner, revision=revision, mirror_dir=mirror_dir)
    return _SOURCES[key]


def _find_entry(source: HubSource, repo_id: str, path: str) -> Entry:
    if not path:
        raise ValueError(f"{repo_id} is a whole dataset: link to a fire folder or an .npz inside it")
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    try:
        siblings = source.list((repo_id, parent))
    except FileNotFoundError:
        raise FileNotFoundError(f"No folder {parent!r} in {repo_id}") from None
    for entry in siblings:
        if entry.path == path:
            return entry
    close = difflib.get_close_matches(path.rsplit("/", 1)[-1], [e.name for e in siblings],
                                      n=5, cutoff=0.4)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise FileNotFoundError(f"No {path!r} in {repo_id}.{hint}")


def npz_date_lookup(source: HubSource, entry: Entry):
    """(window, day) -> date for a window archive, from its fire's raw VIIRS_Day files.

    Archive p<id>.npz holds windows of fire <id>; window w starts on day
    w * ts_interval (window_config.json at the dataset root, default 1).
    Returns None when the raw fire folder isn't in this account.
    """
    fire_id = entry.name[:-4] if entry.name.lower().endswith(".npz") else entry.name
    fire_id = fire_id[1:] if fire_id.startswith("p") else fire_id
    raw_repo = f"{source.owner}/ts-satfire"
    try:
        source.ensure(raw_repo)
    except Exception:
        return None
    days_at = (raw_repo, f"ts-satfire/{fire_id}/VIIRS_Day")
    if not source.exists(days_at):
        return None
    days = [date_of(e.name) for e in source.list(days_at) if not e.is_dir]
    interval = 1
    try:
        config = [e for e in source.list((entry.repo_id, "")) if e.name == "window_config.json"]
        if config:
            interval = int(json.loads(source.download(config[0]).read_text()).get("ts_interval", 1))
    except Exception:
        pass

    def lookup(window: int, day: int) -> Optional[str]:
        index = window * interval + day
        return days[index] if 0 <= index < len(days) else None

    return lookup


def load(link, *, window: int = 0, mirror_dir: str = "hf_mirror",
         source: Optional[HubSource] = None) -> "FireData":
    """Download (once) and open a fire folder or .npz; returns a FireData.

    link: a Hugging Face URL, "owner/dataset/path", an explorer Entry, or a
    local folder/.npz. A fire folder downloads every GeoTIFF in it.
    """
    if not isinstance(link, Entry) and os.path.exists(os.path.expanduser(str(link))):
        return _load_local(Path(os.path.expanduser(str(link))), window)
    repo_id, path, revision = parse_link(link)
    source = source or _source_for(repo_id.split("/")[0], mirror_dir, revision)
    entry = link if isinstance(link, Entry) else _find_entry(source, repo_id, path)

    if entry.is_dir:
        folders = [e.name for e in source.list(entry.location) if e.is_dir]
        if not is_fire_folder(folders):
            raise ValueError(f"{entry.path} isn't a fire folder: it needs VIIRS_Day plus FirePred "
                             f"or VIIRS_Night (it has {', '.join(folders) or 'no folders'})")
        tifs = [e for e in source.files_under(entry.location) if e.name.lower().endswith(".tif")]
        if any(not source.is_local(e) for e in tifs):
            source.download_many(tifs, progress=_printer(entry.name))
        stack: Stack = FireStack(source.local_path(entry))
        kind = "fire folder"
    elif entry.name.lower().endswith(".npz"):
        if not source.is_local(entry):
            from .explorer import format_size

            print(f"Downloading {entry.name} ({format_size(entry.size)})…", flush=True)
        stack = NpzStack(source.download(entry), dates=npz_date_lookup(source, entry))
        kind = "window archive" if stack.window_members else "npz"
    else:
        raise ValueError(f"{entry.name}: load() takes a fire folder or an .npz file")
    return FireData(stack, name=entry.name, kind=kind, link=entry.url,
                    path=source.local_path(entry), window=window)


def _load_local(path: Path, window: int) -> "FireData":
    if path.is_dir():
        if not is_fire_folder([p.name for p in path.iterdir() if p.is_dir()]):
            raise ValueError(f"{path} isn't a fire folder (needs VIIRS_Day plus FirePred or VIIRS_Night)")
        return FireData(FireStack(path), name=path.name, kind="fire folder", path=path, window=window)
    if path.suffix.lower() == ".npz":
        stack = NpzStack(path)
        kind = "window archive" if stack.window_members else "npz"
        return FireData(stack, name=path.name, kind=kind, path=path, window=window)
    raise ValueError(f"{path}: load() takes a fire folder or an .npz file")


def _printer(name: str):
    def report(done: int, total: int, done_bytes: int, total_bytes: int) -> None:
        from .explorer import format_size

        print(f"\rDownloading {name}: {done}/{total} files · {format_size(done_bytes)} of "
              f"{format_size(total_bytes)}", end="\n" if done == total else "", flush=True)

    return report


# ----- channels ------------------------------------------------------------------


class ChannelError(LookupError):
    """An unknown or ambiguous channel name."""

    def __str__(self) -> str:  # no quotes around the message, unlike KeyError
        return self.args[0]


def find_layer(layers: Sequence[Layer], channel: Union[str, Layer]) -> Layer:
    """A channel by name: exact (any case), then raw ↔ npz alias, then a unique substring."""
    if isinstance(channel, Layer):
        return channel
    names = same_channel_names(channel)
    wanted = channel.strip().lower()
    exact = [l for l in layers if l.name.lower() in names or l.label.lower() == wanted]
    if len(exact) == 1:
        return exact[0]
    partial = exact or [l for l in layers if any(n in l.name.lower() for n in names)]
    if len(partial) == 1:
        return partial[0]
    if partial:
        raise ChannelError(f"{channel!r} matches several channels: "
                           f"{', '.join(l.name for l in partial)}. Use the full name.")
    close = difflib.get_close_matches(channel, [l.name for l in layers], n=5, cutoff=0.4)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ChannelError(f"No channel {channel!r}.{hint} See .channels() for all of them.")


def _has(layers: Sequence[Layer], channel: str) -> bool:
    try:
        find_layer(layers, channel)
        return True
    except ChannelError:
        return False


# ----- matplotlib helpers ----------------------------------------------------------


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("Plotting needs matplotlib. In Colab run: %pip install matplotlib") from None
    return plt


def _axes(ax, figsize):
    """Use the caller's Axes, or make a figure with one."""
    if ax is not None:
        return ax.figure, ax
    fig, ax = _plt().subplots(figsize=figsize)
    return fig, ax


def _cmap(ramp: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(ramp, render.RAMPS[ramp])


def _image(ax, rgb: np.ndarray, title: str) -> None:
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def _blank(ax, message: str, title: str = "") -> None:
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#6e6e73", wrap=True)


def _legend(fig, axes: List, legend: pictures.Legend, label: str = "") -> None:
    """Draw a picture's legend: colorbar for values/time, patches for classes/masks."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    if legend.kind == "values" and math.isfinite(legend.lo) and math.isfinite(legend.hi):
        bar = fig.colorbar(ScalarMappable(Normalize(legend.lo, legend.hi), _cmap(legend.ramp)),
                           ax=axes, shrink=0.8, pad=0.02)
        bar.set_label(label or legend.note, fontsize=8)
    elif legend.kind == "time":
        bar = fig.colorbar(ScalarMappable(Normalize(0, 1), _cmap(legend.ramp)),
                           ax=axes, shrink=0.8, pad=0.02)
        bar.set_ticks([0, 1])
        bar.set_ticklabels([legend.first, legend.last], fontsize=8)
        bar.set_label(legend.note, fontsize=8)
    elif legend.kind == "classes":
        handles = [Patch(facecolor=colour, edgecolor="#00000030", label=name)
                   for _, (name, colour) in sorted(legend.classes.items())]
        axes[-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
                        fontsize=8, frameon=False)
    elif legend.kind == "mask":
        colour = tuple(c / 255 for c in legend.colour)
        axes[-1].legend(handles=[Patch(facecolor=colour, label=f"{legend.name} · {legend.count:,} px")],
                        loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=False)
    elif legend.kind == "empty":
        axes[-1].text(0.5, -0.06, legend.message, transform=axes[-1].transAxes,
                      ha="center", va="top", fontsize=8, color="#6e6e73")


def _index(layer: Layer, date: Date) -> int:
    """A date string or index -> frame index in a layer."""
    if isinstance(date, str):
        if date in layer.dates:
            return layer.dates.index(date)
        raise ValueError(f"{layer.name} has no date {date!r}; it has {layer.count}: "
                         f"{layer.dates[0]} … {layer.dates[-1]}")
    i = int(date)
    if i < 0:
        i += layer.count
    if not 0 <= i < layer.count:
        raise IndexError(f"{layer.name} has {layer.count} dates (0 … {layer.count - 1}); got {date}")
    return i


# ----- one fire -------------------------------------------------------------------


class FireData:
    """One fire folder or .npz, ready to list and plot. Make one with firevis.load(link)."""

    def __init__(self, stack: Stack, *, name: str, kind: str, link: Optional[str] = None,
                 path: Optional[Path] = None, window: int = 0):
        self.stack = stack
        self.name = name
        self.kind = kind
        self.link = link
        self.path = Path(path) if path else None
        self._window = 0
        self._layers = stack.layers(0)
        self.window = window

    # -- info

    @property
    def windows(self) -> List[str]:
        """Window labels of a window archive (one blank label otherwise)."""
        return list(self.stack.windows)

    @property
    def window(self) -> int:
        """The window in use (window archives); set it to switch."""
        return self._window

    @window.setter
    def window(self, value: int) -> None:
        count = len(self.stack.windows)
        if not 0 <= value < count:
            raise IndexError(f"{self.name} has {count} window(s): 0 … {count - 1}")
        if value != self._window:
            self._layers = self.stack.layers(value)
        self._window = value

    @property
    def layers(self) -> List[Layer]:
        return self._layers

    def channels(self, group: Optional[str] = None) -> List[str]:
        """Every channel name (or one group's: "Fire", "VIIRS day", "FirePred", ...)."""
        layers = self._layers
        if group:
            layers = [l for l in layers if l.group.lower() == group.strip().lower()]
            if not layers:
                raise ChannelError(f"No group {group!r}; groups: {', '.join(self.groups())}")
        return [l.name for l in layers]

    def groups(self) -> List[str]:
        return list(dict.fromkeys(l.group for l in self._layers))

    def dates(self, channel: Optional[str] = None) -> List[str]:
        """A channel's dates (default: the channel with the most dates)."""
        layer = self.layer(channel) if channel else max(self._layers, key=lambda l: l.count)
        return list(layer.dates)

    def layer(self, channel: Union[str, Layer]) -> Layer:
        return find_layer(self._layers, channel)

    def __repr__(self) -> str:
        dates = self.dates()
        span = f"{dates[0]} → {dates[-1]}" if len(dates) > 1 else (dates[0] if dates else "")
        lines = [f"<FireData {self.name} · {self.kind} · {len(dates)} date(s) {span}"]
        if len(self.stack.windows) > 1:
            lines.append(f"  .window = {self._window} → {self.stack.windows[self._window]} "
                         f"({len(self.stack.windows)} windows: 0 … {len(self.stack.windows) - 1})")
        for group in self.groups():
            names = self.channels(group)
            shown = ", ".join(names[:8]) + (f", … (+{len(names) - 8})" if len(names) > 8 else "")
            lines.append(f"  {group} ({len(names)}): {shown}")
        return "\n".join(lines) + ">"

    # -- plots

    def show(self, channel: str, date: Date = 0, *, view: str = "fire",
             background: str = "first", ax=None):
        """One channel on one date. Returns the matplotlib Axes."""
        layer = self.layer(channel)
        i = _index(layer, date)
        picture = pictures.frame_picture(self.stack, layer, i, self._window, background)
        fig, ax = _axes(ax, (6, 5.5))
        _image(ax, pictures.crop(picture.rgb, self.box(view)),
               f"{self.name} · {layer.name} · {layer.dates[i]}")
        _legend(fig, [ax], picture.legend)
        return ax

    def overlay(self, channel: str, *, view: str = "fire", background: str = "first", ax=None):
        """Every date of a channel at once, slightly transparent, light (first) to dark (last)."""
        layer = self.layer(channel)
        picture = pictures.overlay_picture(self.stack, layer, self._window, background)
        fig, ax = _axes(ax, (6, 5.5))
        _image(ax, pictures.crop(picture.rgb, self.box(view)),
               f"{self.name} · {layer.name} · {layer.dates[0]} → {layer.dates[-1]}")
        _legend(fig, [ax], picture.legend)
        return ax

    def grid(self, channel: str, dates: Union[None, slice, Sequence[Date]] = None, *,
             cols: int = 4, view: str = "fire", background: str = "first"):
        """One small picture per date (all dates, a slice, or a list). Returns the Axes grid.

        Continuous channels share one colour scale across the pictures.
        """
        layer = self.layer(channel)
        indices = _indices(layer, dates)
        span = _span([layer], [indices])
        rows = max(1, math.ceil(len(indices) / cols))
        fig, axes = _plt().subplots(rows, cols, figsize=(3 * cols, 3.1 * rows), squeeze=False)
        box = self.box(view)
        legend = None
        for ax, i in zip(axes.flat, indices):
            picture = pictures.frame_picture(self.stack, layer, i, self._window, background, span)
            title = layer.dates[i]
            if picture.legend.kind == "mask":
                title += f" · {picture.legend.count:,} px"
            _image(ax, pictures.crop(picture.rgb, box), title)
            legend = _merge(legend, picture.legend)
        for ax in list(axes.flat)[len(indices):]:
            ax.set_axis_off()
        fig.suptitle(f"{self.name} · {layer.name}", fontsize=11)
        if legend is not None:
            _legend(fig, list(axes.flat), legend)
        return axes

    def compare(self, *others: "FireData") -> "Comparison":
        """Side by side with other fires: same functions, one panel each."""
        return Comparison(self, *others)

    def box(self, view: str):
        return pictures.view_box(self.stack, self._window, view)


def _indices(layer: Layer, dates) -> List[int]:
    if dates is None:
        return list(range(layer.count))
    if isinstance(dates, slice):
        return list(range(layer.count))[dates]
    return [_index(layer, d) for d in dates]


def _span(layers: Sequence[Layer], indices: Sequence[Sequence[Optional[int]]]):
    """One colour scale for every chosen frame of continuous layers (else None)."""
    if not all(l.style == VALUES for l in layers):
        return None
    frames = [l.read(i) for l, idx in zip(layers, indices) for i in idx if i is not None]
    return pictures.shared_span(frames)


def _merge(legend: Optional[pictures.Legend], new: pictures.Legend) -> pictures.Legend:
    """Combine panel legends: class lists union; mask counts add up."""
    if legend is None:
        return new
    if legend.kind == "classes" and new.kind == "classes":
        legend.classes.update(new.classes)
    elif legend.kind == "mask" and new.kind == "mask":
        legend.count += new.count
    return legend


# ----- side by side --------------------------------------------------------------


class Comparison:
    """Several fires side by side. Make one with firevis.compare(a, b) or a.compare(b).

    Channel names work across a raw fire folder and its window archives ("I4"
    finds "VIIRS I4 (running max)" in an .npz). A date string picks that date in
    each fire; an index picks each fire's own i-th date.
    """

    def __init__(self, *fires):
        fires = [f if isinstance(f, FireData) else load(f) for f in fires]
        if len(fires) < 2:
            raise ValueError("compare() needs at least two fires")
        self.fires = fires

    def __repr__(self) -> str:
        return "<Comparison " + " | ".join(f"{f.name} ({f.kind})" for f in self.fires) + ">"

    def channels(self) -> List[str]:
        """Channels every fire has, in the first fire's naming."""
        first, rest = self.fires[0], self.fires[1:]
        return [l.name for l in first.layers if all(_has(f.layers, l.name) for f in rest)]

    def _view(self, view: Optional[str]) -> str:
        """Default: the model crop when raw folders meet window archives (same ground)."""
        if view:
            return view
        kinds = {f.kind == "fire folder" for f in self.fires}
        return "crop" if len(kinds) == 2 else "fire"

    @staticmethod
    def _pick(layer: Layer, date: Date) -> Optional[int]:
        if isinstance(date, str):
            return layer.dates.index(date) if date in layer.dates else None
        return _index(layer, date)

    def show(self, channel: str, date: Date = 0, *, view: Optional[str] = None,
             background: str = "first"):
        """One date of a channel in each fire, side by side. Returns the Axes."""
        view = self._view(view)
        layers = [f.layer(channel) for f in self.fires]
        picks = [self._pick(l, date) for l in layers]
        span = _span(layers, [[i] for i in picks])
        fig, axes = _plt().subplots(1, len(self.fires), figsize=(5.2 * len(self.fires), 5.4),
                                    squeeze=False)
        axes = list(axes[0])
        legend = None
        for ax, fire, layer, i in zip(axes, self.fires, layers, picks):
            if i is None:
                _blank(ax, f"No {date} in {fire.name}", f"{fire.name}\n{layer.name}")
                continue
            picture = pictures.frame_picture(fire.stack, layer, i, fire.window, background, span)
            _image(ax, pictures.crop(picture.rgb, fire.box(view)),
                   f"{fire.name}\n{layer.name} · {layer.dates[i]}")
            if span is None:  # masks, classes, colour: each panel's own legend
                _legend(fig, [ax], picture.legend)
            else:
                legend = picture.legend
        if legend is not None:
            _legend(fig, axes, legend, "shared colour scale")
        return axes

    def overlay(self, channel: str, *, view: Optional[str] = None, background: str = "first"):
        """Every date of a channel, overlaid, in each fire side by side. Returns the Axes."""
        view = self._view(view)
        fig, axes = _plt().subplots(1, len(self.fires), figsize=(5.2 * len(self.fires), 5.4),
                                    squeeze=False)
        axes = list(axes[0])
        for ax, fire in zip(axes, self.fires):
            layer = fire.layer(channel)
            picture = pictures.overlay_picture(fire.stack, layer, fire.window, background)
            _image(ax, pictures.crop(picture.rgb, fire.box(view)),
                   f"{fire.name}\n{layer.name} · {layer.dates[0]} → {layer.dates[-1]}")
            _legend(fig, [ax], picture.legend)
        return axes

    def grid(self, channel: str, dates: Union[None, slice, Sequence[Date]] = None, *,
             view: Optional[str] = None, background: str = "first"):
        """One row per fire, one column per date. Returns the Axes grid.

        Default dates: every calendar date any fire has (when they all have real
        dates), else day 1, 2, ... of each.
        """
        view = self._view(view)
        layers = [f.layer(channel) for f in self.fires]
        if dates is None:
            if all(all(_ISO.match(d) for d in l.dates) for l in layers):
                dates = sorted({d for l in layers for d in l.dates})
            else:
                dates = list(range(max(l.count for l in layers)))
        elif isinstance(dates, slice):
            dates = sorted({d for l in layers for d in l.dates})[dates]
        picks = [[self._pick_or_none(l, d) for d in dates] for l in layers]
        span = _span(layers, picks)
        rows, cols = len(self.fires), len(dates)
        fig, axes = _plt().subplots(rows, cols, figsize=(2.8 * cols, 3.0 * rows), squeeze=False)
        legend = None
        for r, (fire, layer) in enumerate(zip(self.fires, layers)):
            for c, d in enumerate(dates):
                ax, i = axes[r][c], picks[r][c]
                label = d if isinstance(d, str) else f"day {d + 1}"
                if i is None:
                    _blank(ax, "—", label)
                    continue
                picture = pictures.frame_picture(fire.stack, layer, i, fire.window, background, span)
                _image(ax, pictures.crop(picture.rgb, fire.box(view)), layer.dates[i])
                legend = _merge(legend, picture.legend)
            axes[r][0].annotate(fire.name, xy=(-0.08, 0.5), xycoords="axes fraction",
                                rotation=90, ha="right", va="center", fontsize=10)
        fig.suptitle(" | ".join(f"{f.name}: {l.name}" for f, l in zip(self.fires, layers)),
                     fontsize=11)
        if legend is not None:
            _legend(fig, list(axes.flat), legend)
        return axes

    @staticmethod
    def _pick_or_none(layer: Layer, date: Date) -> Optional[int]:
        try:
            return Comparison._pick(layer, date)
        except IndexError:
            return None


def compare(*fires) -> Comparison:
    """Side by side: firevis.compare(fire_a, fire_b) (FireData objects or links)."""
    return Comparison(*fires)
