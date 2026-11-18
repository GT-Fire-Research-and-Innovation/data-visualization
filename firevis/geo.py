"""Vector data for the preview pane: GeoJSON features and CSV points, drawn as a map.

    read_geojson(path)          features (all, or the first ones of a huge file) + a summary
    feature_times(features)     the date of each feature, from the property that holds dates
    csv_points(frame)           latitude/longitude columns of a table, and a date per row
    map_png(...)                polygons, lines and points on lon/lat axes, coloured by date

GeoJSON coordinates are longitude/latitude (RFC 7946); no projection is
needed. Features are coloured light (earliest) to dark (latest) on the
explorer's fire ramp, so perimeters that grow day by day read like the
overlays of the image viewer. Drawing uses matplotlib (Colab has it).
"""

from __future__ import annotations

import io
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import render

FULL_READ_BYTES = 64_000_000  # bigger GeoJSON is streamed: only its first features are read
STREAM_FEATURES = 5_000  # features read from a streamed file
STREAM_BYTES = 200_000_000  # ... or this much of it, whichever comes first
MAX_VERTICES = 2_000_000  # longer rings are thinned before drawing
MAX_POINTS = 100_000  # points drawn from a table

# Properties that hold a feature's date, best first (compared lower-case). A fire's
# own date (perimeter time, fire year) beats when a record was edited (date current).
TIME_KEYS = (
    "perimeterdatetime", "polygondatetime", "linedatetime", "pointdatetime", "acq_date",
    "fire_year_int", "fire_year", "fireyear", "year",
    "date", "datetime", "time", "datecurrent", "date_cur", "gdb_from_date", "createdate",
)
LAT_NAMES = ("latitude", "lat")
LON_NAMES = ("longitude", "lon", "lng", "long")


@dataclass
class GeoJson:
    """The features read from a GeoJSON file and what is known about the rest."""

    features: List[dict]
    total_bytes: int
    read_bytes: int
    complete: bool  # every feature was read
    geometry_types: List[str] = field(default_factory=list)

    @property
    def note(self) -> str:
        if self.complete:
            return ""
        return (f"Showing the first {len(self.features):,} features "
                f"({self.read_bytes / 1e6:,.0f} of {self.total_bytes / 1e6:,.0f} MB read).")


def read_geojson(path, full_read_bytes: int = FULL_READ_BYTES, stream_features: int = STREAM_FEATURES,
                 stream_bytes: int = STREAM_BYTES) -> GeoJson:
    """Read a GeoJSON file: whole when it is small, otherwise its first features only.

    A FeatureCollection streamed feature by feature never needs the whole file
    in memory, so even the largest FAH files (about 900 MB) preview quickly.
    A single Feature or a bare geometry counts as one feature.
    """
    path = Path(path)
    total = path.stat().st_size
    if total <= full_read_bytes:
        data = json.loads(path.read_text(encoding="utf-8"))
        features = _as_features(data)
        return GeoJson(features, total, total, True, _geometry_types(features))
    features, read, complete = [], 0, False
    for feature, read in _stream_features(path):
        features.append(feature)
        if len(features) >= stream_features or read >= stream_bytes:
            break
    else:
        complete = True
    return GeoJson(features, total, read if not complete else total, complete, _geometry_types(features))


def _as_features(data) -> List[dict]:
    kind = data.get("type") if isinstance(data, dict) else None
    if kind == "FeatureCollection":
        return [f for f in data.get("features") or [] if isinstance(f, dict)]
    if kind == "Feature":
        return [data]
    if kind == "GeometryCollection" or (isinstance(data, dict) and "coordinates" in data):
        return [{"type": "Feature", "geometry": data, "properties": {}}]
    raise ValueError("Not GeoJSON: expected a FeatureCollection, Feature or geometry")


def _geometry_types(features) -> List[str]:
    return sorted({(f.get("geometry") or {}).get("type") or "none" for f in features})


def _stream_features(path: Path, chunk: int = 4_000_000) -> Iterator[Tuple[dict, int]]:
    """Yield (feature, bytes read so far) from a FeatureCollection without loading it whole."""
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buffer, offset, read, eof = "", 0, 0, False

        def more() -> bool:
            nonlocal buffer, offset, read, eof
            text = f.read(chunk)
            if not text:
                eof = True
                return False
            read += len(text)  # characters; GeoJSON is nearly all ASCII, so about bytes
            buffer = buffer[offset:] + text
            offset = 0
            return True

        # Find the opening bracket of "features".
        while True:
            match = re.search(r'"features"\s*:\s*\[', buffer)
            if match:
                offset = match.end()
                break
            if not more():
                raise ValueError("Not a GeoJSON FeatureCollection: no \"features\" array found")
            if len(buffer) > 50 * chunk:
                raise ValueError("No \"features\" array near the start of this file")
        while True:
            while True:  # skip separators
                while offset < len(buffer) and buffer[offset] in " \t\r\n,":
                    offset += 1
                if offset < len(buffer) or not more():
                    break
            if offset >= len(buffer) or buffer[offset] == "]":
                return
            try:
                feature, end = decoder.raw_decode(buffer, offset)
            except json.JSONDecodeError:
                if eof or not more():
                    raise ValueError("The file ends inside a feature") from None
                continue
            offset = end
            if isinstance(feature, dict):
                yield feature, read


# ----- dates ------------------------------------------------------------------


def parse_time(value) -> Optional[np.datetime64]:
    """One property value as a date: epoch ms/s, YYYYMMDD[hhmmss], a year, or ISO text."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        text = str(int(value))
    else:
        text = str(value).strip()
    if not text or text.lower() in ("none", "null", "nan", "nat"):
        return None
    try:
        if text.isdigit():
            if len(text) == 4:
                year = int(text)
                return np.datetime64(f"{year:04d}-01-01") if 1800 <= year <= 2100 else None
            if len(text) == 8:
                return np.datetime64(f"{text[:4]}-{text[4:6]}-{text[6:]}")
            if len(text) == 14:
                return np.datetime64(f"{text[:4]}-{text[4:6]}-{text[6:8]}T{text[8:10]}:{text[10:12]}:{text[12:]}")
            if len(text) == 13:
                return np.datetime64(int(text), "ms")
            if len(text) == 10:
                return np.datetime64(int(text), "s")
            return None
        return np.datetime64(text.replace(" ", "T")[:19].rstrip("Z"))
    except ValueError:
        return None


def feature_times(features: Sequence[dict]) -> Tuple[Optional[str], List[Optional[np.datetime64]]]:
    """The property that dates most features, and each feature's date (None where it has none).

    Keys are tried in TIME_KEYS order; the first that dates at least half the
    features wins. A feature without it takes the next date property it has
    (perimeter records without a perimeter time still have a date current).
    (None, all None) when no property does.
    """
    if not features:
        return None, []
    keys = {}
    for feature in features[:200]:
        for key in (feature.get("properties") or {}):
            keys.setdefault(key.lower(), key)
    present = [keys[k] for k in TIME_KEYS if k in keys]
    for position, key in enumerate(present):
        times = [parse_time((f.get("properties") or {}).get(key)) for f in features]
        if sum(t is not None for t in times) * 2 < len(features):
            continue
        for fallback in present[position + 1:]:
            missing = [i for i, t in enumerate(times) if t is None]
            if not missing:
                break
            for i in missing:
                times[i] = parse_time((features[i].get("properties") or {}).get(fallback))
        return key, times
    return None, [None] * len(features)


def format_time(value: np.datetime64) -> str:
    """A date as text, with the time of day only when it is not midnight."""
    text = str(np.datetime64(value, "m"))
    return text[:10] if text.endswith("T00:00") else text.replace("T", " ")


# ----- tables with coordinates ----------------------------------------------


def csv_points(frame) -> Optional[Tuple[np.ndarray, np.ndarray, List[Optional[np.datetime64]], str]]:
    """(lon, lat, dates, date column) of a table with latitude/longitude columns, else None.

    The date comes from a date column; FIRMS tables add their HHMM acq_time.
    """
    columns = {str(c).lower(): c for c in frame.columns}
    lat = next((columns[n] for n in LAT_NAMES if n in columns), None)
    lon = next((columns[n] for n in LON_NAMES if n in columns), None)
    if lat is None or lon is None:
        return None
    lats = np.asarray(frame[lat], dtype=float)
    lons = np.asarray(frame[lon], dtype=float)
    ok = np.isfinite(lats) & np.isfinite(lons) & (np.abs(lats) <= 90) & (np.abs(lons) <= 360)
    if not ok.any():
        return None
    date_col = next((columns[k] for k in TIME_KEYS if k in columns), None)
    if date_col is None:
        date_col = next((c for key, c in columns.items() if "date" in key or "time" in key), None)
    times: List[Optional[np.datetime64]] = [None] * len(frame)
    label = ""
    if date_col is not None:
        times = [parse_time(v) for v in frame[date_col]]
        label = str(date_col)
        clock = columns.get("acq_time")
        if clock is not None and str(date_col).lower() == "acq_date":
            hhmm = np.asarray(frame[clock])
            times = [t + np.timedelta64(int(h) // 100, "h") + np.timedelta64(int(h) % 100, "m")
                     if t is not None and np.isfinite(h) else t for t, h in zip(times, hhmm.astype(float))]
            label = f"{date_col} + {clock}"
    keep = np.flatnonzero(ok)
    return lons[keep], lats[keep], [times[i] for i in keep], label


# ----- drawing ----------------------------------------------------------------


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise RuntimeError("Maps need matplotlib. In Colab it is installed; elsewhere: pip install matplotlib") from None


def time_colours(times: Sequence[Optional[np.datetime64]], ramp: str = "fire", missing=(0.55, 0.55, 0.6)):
    """An RGB (0..1) colour per date on a ramp (earliest light, latest dark), and the date range."""
    known = [t for t in times if t is not None]
    if not known:
        return [missing] * len(times), None
    first, last = min(known), max(known)
    span = (last - first) / np.timedelta64(1, "s") if last > first else 0.0
    colours = []
    for t in times:
        if t is None:
            colours.append(missing)
            continue
        position = ((t - first) / np.timedelta64(1, "s")) / span if span else 0.5
        colours.append(tuple(render.ramp_colour(ramp, position) / 255.0))
    return colours, (first, last)


def _rings(geometry) -> Iterator[Tuple[str, List[np.ndarray]]]:
    """(kind, rings) per part: "polygon" (exterior + holes), "line" or "point"."""
    if not geometry:
        return
    kind, coords = geometry.get("type"), geometry.get("coordinates")
    if kind == "GeometryCollection":
        for part in geometry.get("geometries") or []:
            yield from _rings(part)
    elif kind == "Polygon":
        yield "polygon", [np.asarray(r, float)[:, :2] for r in coords if len(r) >= 3]
    elif kind == "MultiPolygon":
        for polygon in coords:
            yield "polygon", [np.asarray(r, float)[:, :2] for r in polygon if len(r) >= 3]
    elif kind == "LineString":
        yield "line", [np.asarray(coords, float)[:, :2]]
    elif kind == "MultiLineString":
        yield "line", [np.asarray(c, float)[:, :2] for c in coords if len(c) >= 2]
    elif kind == "Point":
        yield "point", [np.asarray([coords], float)[:, :2]]
    elif kind == "MultiPoint":
        if coords:
            yield "point", [np.asarray(coords, float)[:, :2]]


def map_png(features: Sequence[dict] = (), colours: Sequence = (), points=None, point_colours=(),
            title: str = "", width_px: int = 760, draw_order=None) -> bytes:
    """Draw features (and/or points) on lon/lat axes and return PNG bytes.

    features/colours: GeoJSON features and an RGB colour each. Polygons are
    filled lightly with a solid edge, so overlapping perimeters stay visible.
    draw_order: feature indexes, first drawn at the bottom (default: file order).
    points: (lon, lat) arrays drawn as dots in point_colours.
    """
    plt = _plt()
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    order = range(len(features)) if draw_order is None else draw_order
    parts = [(kind, rings, colours[i]) for i in order for kind, rings in _rings(features[i].get("geometry"))]
    vertices = sum(len(r) for _, rings, _ in parts for r in rings)
    step = max(1, math.ceil(vertices / MAX_VERTICES))
    patches, patch_colours, lines, line_colours, dots, dot_colours = [], [], [], [], [], []
    for kind, rings, colour in parts:
        rings = [r[::step] if len(r) > 8 * step else r for r in rings]
        if kind == "polygon" and rings:
            verts = np.concatenate([np.vstack([r, r[:1]]) for r in rings])
            codes = np.concatenate([[MplPath.MOVETO] + [MplPath.LINETO] * (len(r) - 1) + [MplPath.CLOSEPOLY]
                                    for r in rings])
            patches.append(PathPatch(MplPath(verts, codes)))
            patch_colours.append(colour)
        elif kind == "line":
            lines.extend(rings)
            line_colours.extend([colour] * len(rings))
        elif kind == "point":
            dots.extend(rings)
            dot_colours.extend([colour] * sum(len(r) for r in rings))
    if points is not None:
        lon, lat = points
        dots.append(np.column_stack([lon, lat]))
        dot_colours.extend(point_colours)

    fig, ax = plt.subplots(figsize=(width_px / 100, width_px / 100 * 0.78), dpi=100)
    try:
        if patches:
            ax.add_collection(PatchCollection(patches, facecolors=[(*c, 0.28) for c in patch_colours],
                                              edgecolors=patch_colours, linewidths=0.9, match_original=False))
        if lines:
            ax.add_collection(LineCollection(lines, colors=line_colours, linewidths=1.2))
        if dots:
            xy = np.concatenate(dots)
            ax.scatter(xy[:, 0], xy[:, 1], s=8 if len(xy) < 5000 else 2, c=dot_colours,
                       linewidths=0, zorder=3)
        ax.autoscale_view()
        y0, y1 = ax.get_ylim()
        mid = math.radians(max(-80.0, min(80.0, (y0 + y1) / 2)))
        ax.set_aspect(1 / max(math.cos(mid), 0.1))  # a degree of longitude is shorter than one of latitude
        ax.set_xlabel("longitude", fontsize=8)
        ax.set_ylabel("latitude", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.4, color="#d9d9de")
        ax.ticklabel_format(useOffset=False, style="plain")
        if title:
            ax.set_title(title, fontsize=9)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor="white")
        return buffer.getvalue()
    finally:
        plt.close(fig)


def latest_first(times: Sequence[Optional[np.datetime64]]) -> List[int]:
    """Feature indexes in drawing order: undated, then latest to earliest, so a fire's earlier
    (smaller) perimeters are drawn over its later ones, like growth rings."""
    undated = [i for i, t in enumerate(times) if t is None]
    dated = sorted((i for i, t in enumerate(times) if t is not None), key=lambda i: times[i], reverse=True)
    return undated + dated
