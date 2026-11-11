"""Split a window archive into its fires with the fire-split code, ready to draw.

    from firevis.split import split_archive
    result = split_archive("hf_mirror/SamuelWu318/ts-satfire-6-1/train/p21998264.npz")
    print(result.summary())     # 13 fires: 3 splits, 2 merges ...
    result.fires                # the fires that own burned pixels, in id order

Fire splitting (firevis/firesplit, copied from firepred's fire-split-zones
branch) rebuilds the day each pixel first burned from the archive's next-day
targets, follows every fire day by day (new ignitions, splits, merges) and
gives each target pixel to one fire. That is how training breaks an archive
apart: every window becomes one sample per fire. Here it is drawn as burned
pixels over time:

    before   every window's target, coloured by the day it burned
    after    each fire's own target pixels, on the same days and colours,
             plus the pixels no fire reaches (new ignitions)

Every target pixel is in exactly one "after", so the afters add up to the before.

Targets only date every burn when a window starts every day, so format-2
archives must be built with -it 1 (ts-satfire-6-1, not ts-satfire-10-3).
Daily archives (format 3) are read at -it 1 whatever they record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .firesplit.config import FIRE_SEPARATION_KM, NIGHT_I4_CHANNEL, SEED_NIGHT_I4_KELVIN
from .firesplit.fire_records import FireEvent, SplitSettings, ZoneFile
from .firesplit.fire_tracks import BurnHistory, FireTracks, burn_history, split_settings, track_fire
from .firesplit.fire_zones import zone_file
from .stacks import MASK, RGB, STORED_CHANNEL_NAMES, Layer, Npz, Stack, fire_box

DateLookup = Callable[[int, int], Optional[str]]  # (window, day) -> date, as for NpzStack

FALSE_COLOUR = [2, 1, 0]  # VIIRS I3, I2, I1 in a 27-channel archive
DAILY_MARKER = "num_days"  # only daily archives (format 3) have it


class SplitError(ValueError):
    """The archive can't be split: no windows, or windows that don't start every day."""


# ----- Reading what fire splitting needs ----------------------------------------


@dataclass(frozen=True)
class ArchiveDays:
    """What fire splitting reads from one archive (the same as firepred's zones_master).

    labels:       (windows, H, W) bool next-day targets; window w's target is day length + w
    night_heat:   (length, H, W) night I4 of the days before the first target (heat seeds)
    length:       input days per window, which is the day of window 0's target
    label_sel:    the label rule that built the targets (-1 if unrecorded)
    first_colour: (3, H, W) VIIRS I3, I2, I1 of the first input day, or None
    last_colour:  the same for the last input day (the day before the last target)
    """

    labels: np.ndarray
    night_heat: np.ndarray
    length: int
    label_sel: int
    first_colour: Optional[np.ndarray] = None
    last_colour: Optional[np.ndarray] = None

    @property
    def windows(self) -> int:
        return len(self.labels)


def can_split(path) -> bool:
    """A window archive (data_### and label_###) or a daily archive, judged by its member names."""
    try:
        npz = Npz(Path(path))
    except Exception:  # not a zip at all
        return False
    try:
        names = set(npz.files)
    finally:
        npz.close()
    return ("data_000" in names and "label_000" in names) or DAILY_MARKER in names


def read_archive(path) -> ArchiveDays:
    """Read an archive's targets, early night heat and two false-colour days."""
    npz = Npz(Path(path))
    try:
        if DAILY_MARKER in npz.files:
            return _read_daily(npz, Path(path).name)
        return _read_windows(npz, Path(path).name)
    finally:
        npz.close()


def _label_sel(npz: Npz) -> int:
    return int(np.asarray(npz["label_sel"]).item()) if "label_sel" in npz.files else -1


def _target(label: np.ndarray) -> np.ndarray:
    """One (H, W) bool target; older exports have a leading singleton axis."""
    label = np.asarray(label)
    if label.ndim == 3 and label.shape[0] == 1:
        label = label[0]
    if label.ndim != 2:
        raise SplitError(f"Fire splitting needs one (H, W) target per window, not {label.shape}")
    return label > 0


def window_step(first: np.ndarray, second: np.ndarray) -> Optional[int]:
    """Days between the starts of two consecutive windows, from the day they share.

    Returns k when window 1's first day is window 0's day k, or None when they
    share no day (they start `length` or more days apart).
    """
    start = second[:, 0].tobytes()
    for k in range(1, first.shape[1]):
        if first[:, k].tobytes() == start:
            return k
    return None


def _read_windows(npz: Npz, name: str) -> ArchiveDays:
    """A format-2 archive: data_### (C, T, H, W) and label_### per window."""
    members = sorted(n for n in npz.files if re.fullmatch(r"data_\d+", n))
    count = len(members)
    if not count:
        raise SplitError(f"{name} has no windows to split")
    missing = [f"label_{w:03d}" for w in range(count) if f"label_{w:03d}" not in npz.files]
    if missing or members != [f"data_{w:03d}" for w in range(count)]:
        raise SplitError(f"{name} doesn't have a data_### and label_### for every window")
    first = npz["data_000"]
    if first.ndim != 4:
        raise SplitError(f"data_000 is {first.shape}, not (channels, days, H, W)")
    channels, length = first.shape[:2]
    last = first
    if count > 1:
        second = npz["data_001"]
        step = window_step(first, second) if length > 1 else 1
        if step != 1:
            every = f"every {step} days" if step else f"{length} or more days apart"
            raise SplitError(
                "Fire splitting needs a window starting every day (built with -it 1): it dates "
                f"each burn by the first target it is in. These windows start {every}; open this "
                "fire in a dataset ending in -1, such as ts-satfire-6-1.")
        last = second if count == 2 else npz[members[-1]]
    labels = np.stack([_target(npz[f"label_{w:03d}"]) for w in range(count)])
    shape = first.shape[2:]
    heat = (np.asarray(first[NIGHT_I4_CHANNEL], np.float32) if channels > NIGHT_I4_CHANNEL
            else np.zeros((0,) + shape, np.float32))
    colour = channels == len(STORED_CHANNEL_NAMES)
    return ArchiveDays(
        labels=labels, night_heat=heat, length=int(length), label_sel=_label_sel(npz),
        first_colour=first[FALSE_COLOUR, 0] if colour else None,
        last_colour=last[FALSE_COLOUR, length - 1] if colour else None)


def _read_daily(npz: Npz, name: str) -> ArchiveDays:
    """A daily archive (format 3): every day once, in blocks, plus packed targets.

    Read as the windows that start every day at the archive's own length, like
    firepred's open_daily_windows does for zones.
    """
    read = lambda key: int(np.asarray(npz[key]).item())  # noqa: E731
    length, num_days, block_days = read("ts_length"), read(DAILY_MARKER), read("block_days")
    channels, height, width = (int(n) for n in npz["frame_shape"])
    static = {int(c) for c in npz["static_channels"]}
    target_days = [int(d) for d in npz["target_days"]]
    index = {day: i for i, day in enumerate(target_days)}
    wanted = list(range(length, max(target_days, default=-1) + 1))
    if not wanted:
        raise SplitError(f"{name} has no windows to split")
    missing = [day for day in wanted if day not in index]
    if missing or wanted[-1] > num_days:
        raise SplitError(f"{name} can't give a window starting every day: it lacks target days "
                         f"{missing[:5]} (built with -ts {length} -it {read('ts_interval')})")
    packed = npz["targets"]
    labels = np.stack([np.unpackbits(packed[index[day]], axis=-1, count=width) > 0 for day in wanted])

    def days(channel: int, start: int, stop: int) -> np.ndarray:
        if channel in static:
            frame = npz[f"static_c{channel:02d}"]
            return np.repeat(frame[None], stop - start, axis=0).astype(np.float32)
        out = np.empty((stop - start, height, width), np.float32)
        day = start
        while day < stop:
            first_day = day - day % block_days
            block = npz[f"days_c{channel:02d}_{first_day:03d}"]
            end = min(first_day + len(block), stop)
            out[day - start:end - start] = block[day - first_day:end - first_day]
            day = end
        return out

    heat = (days(NIGHT_I4_CHANNEL, 0, length) if channels > NIGHT_I4_CHANNEL
            else np.zeros((0, height, width), np.float32))
    colour = channels == len(STORED_CHANNEL_NAMES)
    last_day = wanted[-1] - 1  # the last window's last input day

    def false_colour(day: int) -> np.ndarray:
        return np.stack([days(c, day, day + 1)[0] for c in FALSE_COLOUR])

    return ArchiveDays(
        labels=labels, night_heat=heat, length=length, label_sel=_label_sel(npz),
        first_colour=false_colour(0) if colour else None,
        last_colour=false_colour(last_day) if colour else None)


# ----- The split ------------------------------------------------------------------


@dataclass
class FireSplit:
    """One archive before and after fire splitting.

    zones.target_owners[w] says which fire each new pixel of window w's target
    belongs to (-1: no fire reaches it, 0: not a new pixel); zones.events lists
    every fire, with where it came from and how it ended.
    """

    title: str
    days: ArchiveDays
    history: BurnHistory
    tracks: FireTracks
    zones: ZoneFile
    dates: Optional[DateLookup] = None

    @property
    def settings(self) -> SplitSettings:
        return self.zones.settings

    @property
    def events(self) -> tuple:
        return self.zones.events

    @property
    def windows(self) -> int:
        return self.days.windows

    @property
    def length(self) -> int:
        return self.days.length

    def day_label(self, day: int) -> str:
        """The date of a day of the archive (0 = window 0's first input day)."""
        found = self.dates(0, day) if self.dates else None
        return found or f"Day {day + 1}"

    @property
    def target_dates(self) -> List[str]:
        """The date of every window's target, the dates burned pixels are drawn on."""
        return [self.day_label(self.length + w) for w in range(self.windows)]

    @property
    def burned(self) -> int:
        """Burned pixels before splitting: every new pixel of every target."""
        return int(self.days.labels.sum())

    def pixels(self, fire_id: int) -> int:
        """Target pixels a fire owns (-1: the pixels no fire reaches)."""
        owners = self.zones.target_owners
        return int((owners < 0).sum() if fire_id < 0 else (owners == fire_id).sum())

    @property
    def unowned(self) -> int:
        return self.pixels(-1)

    @property
    def fires(self) -> List[FireEvent]:
        """The fires that own at least one burned pixel: the afters."""
        counts = self._counts()
        return [e for e in self.events if counts.get(e.event_id, 0)]

    @property
    def quiet(self) -> List[FireEvent]:
        """Fires that own no target pixel (a branch that stopped growing, an early seed fire)."""
        counts = self._counts()
        return [e for e in self.events if not counts.get(e.event_id, 0)]

    @property
    def splits(self) -> int:
        """How many times a fire broke apart (fires that have split branches)."""
        return len({e.parent_id for e in self.events if e.kind == "split"})

    @property
    def merges(self) -> int:
        return sum(1 for e in self.events if e.merged_into)

    def branches(self, fire_id: int) -> List[int]:
        return [e.event_id for e in self.events if e.kind == "split" and e.parent_id == fire_id]

    def windows_of(self, fire_id: int) -> List[int]:
        """The windows whose target has pixels of this fire."""
        owners = self.zones.target_owners
        mine = owners < 0 if fire_id < 0 else owners == fire_id
        return [int(w) for w in np.flatnonzero(mine.reshape(len(mine), -1).any(axis=1))]

    def describe(self, event: FireEvent) -> str:
        """Where a fire came from and how it ended, in a few words."""
        start = self.day_label(event.start_day)
        born = (f"split from fire {event.parent_id} on {start}" if event.kind == "split"
                else f"new fire from {start}")
        if event.end_day < 0:
            return f"{born}; still burning at the end"
        end = self.day_label(event.end_day)
        if event.merged_into:
            return f"{born}; merged into fire {event.merged_into} on {end}"
        branches = self.branches(event.event_id)
        if branches:
            return f"{born}; split on {end} into fires {', '.join(map(str, branches))}"
        return f"{born}; ended {end}"

    def summary(self) -> str:
        """One line: how many fires the archive became."""
        events = len(self.events)
        text = (f"{self.title}: {self.burned:,} burned pixels in {self.windows} windows became "
                f"{events} fire{'s' if events != 1 else ''} ({self.splits} split"
                f"{'s' if self.splits != 1 else ''}, {self.merges} merge{'s' if self.merges != 1 else ''})")
        if self.burned:
            text += f"; {self.unowned:,} pixels ({self.unowned / self.burned:.1%}) belong to no fire"
        return text

    def stack(self) -> "SplitStack":
        return SplitStack(self)

    def _counts(self) -> Dict[int, int]:
        ids, counts = np.unique(self.zones.target_owners, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, counts) if i > 0}


def split_archive(path, scale_km: float = FIRE_SEPARATION_KM, dates: Optional[DateLookup] = None,
                  seed_kelvin: float = SEED_NIGHT_I4_KELVIN) -> FireSplit:
    """Split one archive into its fires, exactly as firepred's `dataset zones` does.

    path:     a window archive (built with -it 1) or a daily archive
    scale_km: the separation distance every threshold comes from (default 2.25 km)
    dates:    optional (window, day) -> date lookup, as the explorer has for Hub archives
    """
    days = read_archive(path)
    settings = split_settings(scale_km)
    try:
        history = burn_history(days.labels, days.night_heat, days.label_sel, days.length, seed_kelvin)
    except ValueError as error:  # targets that repeat pixels
        raise SplitError(str(error)) from None
    tracks = track_fire(history, settings)
    return FireSplit(Path(path).name, days, history, tracks, zone_file(tracks, history, settings), dates)


# ----- Layers to draw -------------------------------------------------------------


class SplitStack(Stack):
    """Before and after as mask layers on the target dates, for firevis.pictures.

    Every layer has one frame per window (its target day), so overlays of
    different fires share one colour per day.
    """

    def __init__(self, split: FireSplit):
        super().__init__(title=split.title)
        self.split = split
        dates = split.target_dates
        labels, owners = split.days.labels, split.zones.target_owners
        self.before = Layer("burned", "Before", MASK, dates,
                            lambda i: labels[i], overlay=True, ramp="fire")
        self.fire_layers: Dict[int, Layer] = {
            e.event_id: Layer(f"Fire {e.event_id}", "After", MASK, dates,
                              lambda i, f=e.event_id: owners[i] == f, overlay=True, ramp="fire")
            for e in split.fires}
        self.no_fire = (Layer("No fire", "After", MASK, dates, lambda i: owners[i] < 0,
                              overlay=True, ramp="fire") if split.unowned else None)
        first, last = split.days.first_colour, split.days.last_colour
        self._background = None
        if first is not None and last is not None:
            frames = [first, last]
            self._background = Layer(
                "False colour (I3, I2, I1)", "Channels", RGB,
                [split.day_label(0), split.day_label(split.length + split.windows - 2)],
                lambda i: frames[i])
        self._focus = fire_box(labels.any(axis=0)) if labels.size else None

    @property
    def afters(self) -> List[Layer]:
        """One layer per fire with burned pixels, then the pixels of no fire."""
        return list(self.fire_layers.values()) + ([self.no_fire] if self.no_fire else [])

    def layers(self, window: int = 0) -> List[Layer]:
        return [self.before] + self.afters

    def background(self, window: int = 0) -> Optional[Layer]:
        return self._background

    def focus(self, window: int = 0):
        return self._focus
