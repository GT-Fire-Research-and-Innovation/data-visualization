"""Fire imagery as layers of dated frames, for the viewer.

A *layer* is one thing to look at (active fire, VIIRS I4, wind speed, ...) with
one frame per date. A *stack* is all the layers of one fire folder or one .npz
file. The viewer steps through a layer's frames with arrows, or overlays them
all at once, coloured by date.

Two sources:

* FireStack: a TS-SatFire fire folder in the local mirror, with daily GeoTIFFs
  in VIIRS_Day/, VIIRS_Night/, FirePred/ and (usually) ESRI_LULC/. File names
  start with their date: 2017-04-25_VIIRS_Day.tif. Reading needs rasterio.
* NpzStack: an .npz file. Window archives (data_000, label_000, ...) become one
  window per data_### member, with a layer per channel and the next-day
  target; any other .npz shows each of its arrays.
"""

from __future__ import annotations

import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .render import mask_of

# Layer styles.
MASK = "mask"  # 0/1 fire pixels
VALUES = "values"  # continuous band
CLASSES = "classes"  # land cover classes
RGB = "rgb"  # three bands as colour

FIRE_SUBFOLDERS = ("VIIRS_Day", "VIIRS_Night", "FirePred", "ESRI_LULC")

# Window archive channels (firepred/config/dataset.py STORED_CHANNEL_NAMES).
STORED_CHANNEL_NAMES = (
    "VIIRS I1", "VIIRS I2", "VIIRS I3", "VIIRS I4 (running max)", "VIIRS I5 (running max)",
    "VIIRS M11", "VIIRS I4 night (running max)", "VIIRS I5 night (running max)", "NDVI", "EVI",
    "total precipitation", "wind speed", "wind direction", "min temperature", "max temperature",
    "energy release component", "specific humidity", "slope", "aspect", "elevation", "PDSI",
    "land cover", "total precipitation surface", "forecast wind speed",
    "forecast wind direction", "forecast temperature", "forecast specific humidity",
)
LANDCOVER_CHANNEL = 21

VIIRS_DAY_NAMES = {"b1": "I1", "b2": "I2", "b3": "I3", "b4": "I4", "b5": "I5", "m11": "M11"}
_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def is_fire_folder(child_folders: Sequence[str]) -> bool:
    """A TS-SatFire fire folder: daily VIIRS_Day images plus FirePred or VIIRS_Night."""
    names = set(child_folders)
    return "VIIRS_Day" in names and bool(names & {"FirePred", "VIIRS_Night"})


def fire_box(mask: np.ndarray, margin: float = 0.4, smallest: Optional[int] = None):
    """Square around most of a mask's pixels, padded by `margin` x its size, as image fractions.

    The box spans the 5th-95th percentile of pixel rows and columns, so a few
    stray detections far from the fire don't stretch it to the whole image.
    The square is at least `smallest` pixels (default: 64, or a quarter of a
    small image). Returns (top, bottom, left, right) in 0..1, or None for an
    empty mask.
    """
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    height, width = mask.shape
    if smallest is None:
        smallest = min(64, max(height, width) // 4)
    spread = (5, 95) if len(rows) >= 20 else (0, 100)
    r0, r1 = np.percentile(rows, spread)
    c0, c1 = np.percentile(cols, spread)
    cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
    side = max(r1 - r0 + 1, c1 - c0 + 1)
    side = min(max(side * (1 + 2 * margin), smallest), height, width)
    top = int(np.clip(round(cy - side / 2), 0, height - side))
    left = int(np.clip(round(cx - side / 2), 0, width - side))
    side = int(side)
    return (top / height, (top + side) / height, left / width, (left + side) / width)


def date_of(name: str) -> str:
    """The YYYY-MM-DD a file name starts with (or the name itself)."""
    match = _DATE.match(name)
    return match.group(1) if match else Path(name).stem


@dataclass
class Layer:
    """One thing to look at, with one frame per date."""

    name: str
    group: str
    style: str
    dates: List[str]
    read: Callable[[int], np.ndarray]  # frame index -> (H, W) or (3, H, W) array
    overlay: bool = False  # start with every frame overlaid
    cumulative: bool = False  # the mask only grows: colour pixels by first day
    outlined: bool = False  # draw masks as filled shapes with an edge
    ramp: str = "fire"  # colour ramp over time when overlaid
    palette: Optional[str] = None  # class palette for CLASSES ("esri", "igbp")

    @property
    def count(self) -> int:
        return len(self.dates)

    @property
    def label(self) -> str:
        return f"{self.group} · {self.name}"


@dataclass
class Stack:
    """All layers of one fire folder or file."""

    title: str
    windows: List[str] = field(default_factory=lambda: [""])  # labels; one = no picker

    def layers(self, window: int = 0) -> List[Layer]:
        raise NotImplementedError

    def background(self, window: int = 0) -> Optional[Layer]:
        """An RGB layer to draw overlays on, if there is one."""
        return None

    def focus(self, window: int = 0) -> Optional[Tuple[float, float, float, float]]:
        """The fire's area as fractions (top, bottom, left, right) of the image, if known."""
        return None


# ----- GeoTIFFs ---------------------------------------------------------------


@dataclass
class Raster:
    array: np.ndarray  # (bands, H, W) float32
    names: Tuple[str, ...]
    transform: object
    crs: object


def _rasterio():
    try:
        import rasterio

        return rasterio
    except ImportError:
        raise RuntimeError(
            "Showing GeoTIFFs needs rasterio. In Colab run: %pip install rasterio"
        ) from None


def read_tif(path: Path) -> Raster:
    rasterio = _rasterio()
    with rasterio.open(path) as src:
        names = tuple(d or f"band {i + 1}" for i, d in enumerate(src.descriptions))
        return Raster(src.read().astype(np.float32, copy=False), names, src.transform, src.crs)


def read_on_grid(path: Path, like: Raster) -> np.ndarray:
    """Band 1 of a (large) GeoTIFF resampled onto another raster's grid."""
    rasterio = _rasterio()
    from rasterio.warp import Resampling, reproject

    out = np.full(like.array.shape[1:], np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), out, dst_transform=like.transform, dst_crs=like.crs,
                  resampling=Resampling.nearest, dst_nodata=np.nan)
    return out


class FireStack(Stack):
    """Layers from a fire folder of daily GeoTIFFs (all files already local)."""

    def __init__(self, folder: Path, cache_size: int = 16):
        folder = Path(folder)
        super().__init__(title=folder.name)
        self.folder = folder
        self.files: Dict[str, List[Path]] = {
            kind: sorted((folder / kind).glob("*.tif")) for kind in FIRE_SUBFOLDERS
        }
        if not self.files["VIIRS_Day"]:
            raise FileNotFoundError(f"No VIIRS_Day GeoTIFFs in {folder}")
        self._cache: "OrderedDict[Path, Raster]" = OrderedDict()
        self._cache_size = cache_size
        self._masks: Dict[Tuple[Path, int], np.ndarray] = {}
        self._focus: Optional[tuple] = None
        self._layers = self._build()

    def layers(self, window: int = 0) -> List[Layer]:
        return self._layers

    def background(self, window: int = 0) -> Optional[Layer]:
        return self._false_colour

    def focus(self, window: int = 0) -> Optional[Tuple[float, float, float, float]]:
        """Square around every active-fire and NIFC pixel of every day, with margin."""
        if self._focus is None:
            day, (af, nifc) = self.files["VIIRS_Day"], self._fire_bands
            burned = np.zeros(self.raster(day[0]).array.shape[1:], bool)
            for path in day:
                burned |= self._mask(path, af) | self._mask(path, nifc)
            self._focus = fire_box(burned) or ()
        return self._focus or None

    # -- reading, with a small cache (a VIIRS_Day day is ~11 MB in memory)

    def raster(self, path: Path) -> Raster:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        raster = read_tif(path)
        self._cache[path] = raster
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return raster

    def _band(self, path: Path, band: int) -> np.ndarray:
        array = self.raster(path).array
        if band >= array.shape[0]:  # e.g. a day without the NIFC band
            return np.full(array.shape[1:], np.nan, np.float32)
        return array[band]

    def _mask(self, path: Path, band: int) -> np.ndarray:
        key = (path, band)
        if key not in self._masks:  # masks are small: keep them all for overlays
            self._masks[key] = mask_of(self._band(path, band))
        return self._masks[key]

    def _build(self) -> List[Layer]:
        day = self.files["VIIRS_Day"]
        day_dates = [date_of(p.name) for p in day]
        first = self.raster(day[0])
        names = [n.lower() for n in first.names]
        af = names.index("af") if "af" in names else 6
        nifc = names.index("first") if "first" in names else 7
        self._fire_bands = (af, nifc)

        def cumulative(i: int) -> np.ndarray:
            burned = np.zeros(first.array.shape[1:], bool)
            for path in day[: i + 1]:
                burned |= self._mask(path, af) | self._mask(path, nifc)
            return burned

        layers = [
            Layer("Active fire", "Fire", MASK, day_dates,
                  lambda i: self._mask(day[i], af), overlay=True, ramp="fire"),
            Layer("NIFC perimeter", "Fire", MASK, day_dates,
                  lambda i: self._mask(day[i], nifc), overlay=True, outlined=True, ramp="ice"),
            Layer("Burned area (active fire + NIFC)", "Fire", MASK, day_dates, cumulative,
                  overlay=True, cumulative=True, ramp="ember"),
        ]
        self._false_colour = Layer(
            "False colour (I3, I2, I1)", "VIIRS day", RGB, day_dates,
            lambda i: self.raster(day[i]).array[[2, 1, 0]])
        layers.append(self._false_colour)
        for band, name in enumerate(first.names):
            label = VIIRS_DAY_NAMES.get(name.lower())
            if label:
                layers.append(Layer(label, "VIIRS day", VALUES, day_dates,
                                    lambda i, b=band: self._band(day[i], b)))

        night = self.files["VIIRS_Night"]
        if night:
            dates = [date_of(p.name) for p in night]

            def night_band(i: int, which: int) -> np.ndarray:
                array = self.raster(night[i]).array
                return array[-2 + which] if array.shape[0] >= 2 else np.full(
                    array.shape[1:], np.nan, np.float32)  # 5-band files: I4, I5 are last

            for which, name in enumerate(("I4 night", "I5 night")):
                layers.append(Layer(name, "VIIRS night", VALUES, dates,
                                    lambda i, w=which: night_band(i, w)))

        pred = self.files["FirePred"]
        if pred:
            dates = [date_of(p.name) for p in pred]
            for band, name in enumerate(self.raster(pred[0]).names):
                is_class = name.lower().startswith("lc_type")
                layers.append(Layer(
                    "land cover (MODIS)" if is_class else name, "FirePred",
                    CLASSES if is_class else VALUES, dates,
                    lambda i, b=band: self._band(pred[i], b),
                    palette="igbp" if is_class else None))

        lulc = self.files["ESRI_LULC"]
        if lulc:
            cache: Dict[int, np.ndarray] = {}

            def esri(i: int) -> np.ndarray:
                if i not in cache:  # 10 m source: resample onto the VIIRS grid once
                    cache[i] = read_on_grid(lulc[i], first)
                return cache[i]

            layers.append(Layer("land cover (ESRI)", "Land cover", CLASSES,
                                [date_of(p.name) for p in lulc], esri, palette="esri"))
        return layers


# ----- .npz -------------------------------------------------------------------


def _zipfile():
    if sys.version_info >= (3, 14):
        import zipfile
    else:  # window archives use Zstandard members, new in Python 3.14
        try:
            from backports.zstd import zipfile
        except ImportError:
            import zipfile
    return zipfile


class Npz:
    """Read .npz members one at a time, including Zstandard-compressed ones."""

    def __init__(self, path: Path):
        self._zip = _zipfile().ZipFile(path)
        self.files = [n[:-4] for n in self._zip.namelist() if n.endswith(".npy")]

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            with self._zip.open(f"{name}.npy") as member:
                return np.lib.format.read_array(member, allow_pickle=False)
        except NotImplementedError as error:  # "That compression method is not supported"
            raise RuntimeError(
                f"{error}. This archive uses Zstandard; in Colab run: %pip install backports.zstd"
            ) from None

    def close(self) -> None:
        self._zip.close()


DateLookup = Callable[[int, int], Optional[str]]  # (window, day) -> date or None


class NpzStack(Stack):
    """Layers from an .npz file.

    dates: optional (window, day) -> "YYYY-MM-DD" lookup; without it frames are
    labelled "Day 1", "Day 2", ...
    """

    def __init__(self, path: Path, dates: Optional[DateLookup] = None):
        path = Path(path)
        self.path = path
        self.npz = Npz(path)
        self._dates = dates
        self._window_cache: Tuple[int, Optional[np.ndarray]] = (-1, None)
        self.window_members = sorted(n for n in self.npz.files if re.fullmatch(r"data_\d+", n))
        windows = [self._window_label(w) for w in range(len(self.window_members))]
        super().__init__(title=path.name, windows=windows or [""])

    # -- window archives

    def _shape(self, w: int) -> Tuple[int, ...]:
        return self._window(w).shape

    def _window(self, w: int) -> np.ndarray:
        if self._window_cache[0] != w:
            self._window_cache = (w, self.npz[self.window_members[w]])
        return self._window_cache[1]

    def _date(self, w: int, day: int) -> str:
        found = self._dates(w, day) if self._dates else None
        return found or f"Day {day + 1}"

    def _window_label(self, w: int) -> str:
        member = self.window_members[w]
        if self._dates:
            first, last = self._dates(w, 0), None
            days = self._header_days(member)
            if days:
                last = self._dates(w, days - 1)
            if first and last:
                return f"Window {w + 1}: {first} → {last}"
        return f"Window {w + 1}"

    def _header_days(self, member: str) -> Optional[int]:
        """Days per window from the .npy header, without reading the data."""
        try:
            with self.npz._zip.open(f"{member}.npy") as f:
                major, _ = np.lib.format.read_magic(f)
                read_header = (np.lib.format.read_array_header_1_0 if major == 1
                               else np.lib.format.read_array_header_2_0)
                shape = read_header(f)[0]
            return shape[1] if len(shape) == 4 else None
        except Exception:
            return None

    def background(self, window: int = 0) -> Optional[Layer]:
        """False colour from the window's own VIIRS I3, I2, I1 channels."""
        if not self.window_members:
            return None
        shape = self._shape(window)
        if shape[0] != len(STORED_CHANNEL_NAMES) or len(shape) != 4:
            return None
        dates = [self._date(window, d) for d in range(shape[1])]
        return Layer("False colour (I3, I2, I1)", "Channels", RGB, dates,
                     lambda i: self._window(window)[[2, 1, 0], i])

    def layers(self, window: int = 0) -> List[Layer]:
        if not self.window_members:
            return self._plain_layers()
        data_shape = self._shape(window)
        channels, days = data_shape[0], data_shape[1] if len(data_shape) == 4 else 1
        dates = [self._date(window, d) for d in range(days)]
        layers = []
        label = self.window_members[window].replace("data_", "label_")
        if label in self.npz.files:
            target_date = self._dates(window, days) if self._dates else None

            def target(i: int) -> np.ndarray:
                array = self.npz[label]
                return array[min(i, array.shape[0] - 1)] if array.ndim == 3 else array

            layers.append(Layer("Target: newly burned next day", "Label", MASK,
                                [target_date or "Next day"], target, ramp="fire"))
        for c in range(channels):
            name = STORED_CHANNEL_NAMES[c] if channels == len(STORED_CHANNEL_NAMES) else f"Channel {c}"
            is_class = channels == len(STORED_CHANNEL_NAMES) and c == LANDCOVER_CHANNEL
            layers.append(Layer(
                name, "Channels", CLASSES if is_class else VALUES, dates,
                (lambda i, c=c: self._window(window)[c, i]) if len(data_shape) == 4
                else (lambda i, c=c: self._window(window)[c]),
                palette="igbp" if is_class else None, ramp="fire"))
        return layers

    # -- any other .npz: each array is a layer

    def _plain_layers(self) -> List[Layer]:
        layers = []
        for name in self.npz.files:
            array = self.npz[name]
            if array.ndim < 2 or not np.issubdtype(array.dtype, np.number):
                continue
            if array.ndim == 2:
                layers.append(Layer(name, "Arrays", VALUES, [""], lambda i, a=array: a))
            else:
                flat = array.reshape((-1,) + array.shape[-2:])
                layers.append(Layer(name, "Arrays", VALUES,
                                    [f"Slice {k + 1}" for k in range(len(flat))],
                                    lambda i, a=flat: a[i]))
        return layers

    def close(self) -> None:
        self.npz.close()
