"""Turn a downloaded file into HTML for the explorer's preview pane.

Everything here returns an HTML string that the explorer puts into a widget
inside its own window. That works the same in Colab and Jupyter, unlike
printing or `display()` from a click handler, which Colab may not show.

Text files (txt, json, md, csv, yaml, ...) are shown as you'd expect: JSON
pretty-printed, Markdown rendered, CSV (also .csv.gz) as a table, everything
else as plain text. A table with latitude/longitude columns (FIRMS) also gets a
point map. GeoJSON is drawn as a map coloured by date with its attribute
table; OpenStreetMap .osm.pbf extracts as a map of roads, buildings and water;
ZIP archives as a list of their members. PNG files are shown as images.
NetCDF, GeoTIFF and .npz files open in the image viewer instead (explorer.py).
Other binary files have a saved-file note.
"""

from __future__ import annotations

import base64
import html
import json
import zipfile
from datetime import datetime
from pathlib import Path

MAX_CHARS = 100_000  # longest text shown; the rest is cut with a note
CSV_ROWS = 200  # rows of a CSV shown as a table
JSON_PARSE_LIMIT = 5_000_000  # bytes; bigger JSON is shown as plain text

NUMPY = {".npz", ".npy"}
NETCDF = {".nc", ".nc4", ".cdf"}
GEOTIFF = {".tif", ".tiff"}
GEOJSON = {".geojson"}
OSM = {".pbf"}
ZIP = {".zip"}
MARKDOWN = {".md", ".markdown"}
TABLES = {".csv", ".tsv", ".csv.gz", ".tsv.gz"}
COMPOUND = (".csv.gz", ".tsv.gz", ".osm.pbf")  # read as one extension
MAP_ROWS = 500_000  # rows of a table read for its point map
OSM_DECODE_BYTES = 64_000_000  # bigger .osm.pbf files are summarized from their header only
TEXT = {
    ".txt", ".json", ".yaml", ".yml", ".log", ".py", ".cfg", ".ini", ".toml",
    ".xml", ".sh", ".gitattributes", ".gitignore",
} | MARKDOWN | TABLES
BINARY = {
    ".tif", ".tiff", ".jpg", ".jpeg", ".zip", ".gz", ".zst", ".tar",
    ".nc", ".h5", ".hdf5", ".parquet", ".pt", ".pth", ".bin",
} | NUMPY


def suffix(path: Path) -> str:
    """Lower-case extension; a dotfile like ".gitattributes" counts as its own."""
    name = path.name.lower()
    if name.startswith(".") and name.count(".") == 1:
        return name
    for compound in COMPOUND:
        if name.endswith(compound):
            return ".pbf" if compound == ".osm.pbf" else compound
    return path.suffix.lower()


def looks_like_text(path: Path, sample: int = 4096) -> bool:
    """Guess whether a file with an unknown extension is UTF-8 text."""
    with open(path, "rb") as f:
        chunk = f.read(sample)
    if b"\0" in chunk:
        return False
    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError as error:
        return error.start >= len(chunk) - 3  # a character cut off by the sample
    return True


def file_kind(path: Path) -> str:
    """One of "numpy", "netcdf", "geotiff", "geojson", "osm", "zip", "png", "markdown", "json",
    "table", "text" or "binary"."""
    ext = suffix(path)
    if ext in NUMPY:
        return "numpy"
    for kind, extensions in (("netcdf", NETCDF), ("geotiff", GEOTIFF), ("geojson", GEOJSON),
                             ("osm", OSM), ("zip", ZIP)):
        if ext in extensions:
            return kind
    if ext == ".png":
        return "png"
    if ext in MARKDOWN:
        return "markdown"
    if ext == ".json":
        return "json"
    if ext in TABLES:
        return "table"
    if ext in TEXT:
        return "text"
    if ext in BINARY:
        return "binary"
    return "text" if looks_like_text(path) else "binary"


def to_html(path: Path) -> str:
    """HTML showing the contents of a downloaded file."""
    path = Path(path)
    kind = file_kind(path)
    if kind == "png":
        return _png(path)
    if kind == "numpy":
        return note(
            "NumPy archive saved. No preview yet.<br>Load it with "
            f"<code>np.load({html.escape(repr(str(path)))})</code>",
            card=True,
        )
    if kind in ("netcdf", "geotiff"):
        return note("Opened in the image viewer (explorer.open_file).", card=True)
    if kind == "geojson":
        return geojson_html(path)
    if kind == "osm":
        return osm_html(path)
    if kind == "zip":
        return zip_html(path)
    if kind == "binary":
        what = f"{html.escape(suffix(path))} files" if suffix(path) else "this file type"
        return note(f"Saved. No preview for {what} yet.", card=True)
    if kind == "table":
        return _table(path)
    if kind == "json":
        return _json(path)
    text, cut = _read_text(path)
    if kind == "markdown":
        return f'<div class="fx-md">{markdown_to_html(text)}</div>' + _cut_note(cut)
    return _pre(text) + _cut_note(cut)


def note(message: str, card: bool = False, error: bool = False) -> str:
    """A short message (already HTML) for the preview pane."""
    classes = "fx-note" + (" fx-card" if card else "") + (" fx-note-error" if error else "")
    return f'<div class="{classes}">{message}</div>'


def markdown_to_html(text: str) -> str:
    """Render Markdown with whichever library is installed (Colab has `markdown`)."""
    text = _front_matter_as_code(text)
    try:
        import markdown

        return markdown.markdown(text, extensions=["fenced_code", "tables"])
    except ImportError:
        pass
    try:
        import mistune

        render = getattr(mistune, "html", None) or mistune.markdown
        return render(text)
    except ImportError:
        return _pre(text)


def _pre(text: str) -> str:
    return f'<pre class="fx-code">{html.escape(text, quote=False)}</pre>'


def _cut_note(cut: str) -> str:
    return note(cut) if cut else ""


def _png(path: Path) -> str:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("File does not have a PNG signature")
    encoded = base64.b64encode(data).decode("ascii")
    return (f'<img class="fx-preview-png" src="data:image/png;base64,{encoded}" '
            f'alt="{html.escape(path.name)}">')


def _read_text(path: Path):
    """The file's text (at most MAX_CHARS) and a note if it was cut short."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read(MAX_CHARS + 1)
    if len(text) <= MAX_CHARS:
        return text, ""
    total = path.stat().st_size
    return text[:MAX_CHARS], f"Showing the first {MAX_CHARS:,} characters of {total:,} bytes."


def _json(path: Path) -> str:
    if path.stat().st_size <= JSON_PARSE_LIMIT:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            pass  # not valid JSON: show it as it is
        else:
            text = json.dumps(data, indent=2, ensure_ascii=False)
            if len(text) > MAX_CHARS:
                return _pre(text[:MAX_CHARS]) + note(f"Showing the first {MAX_CHARS:,} characters.")
            return _pre(text)
    text, cut = _read_text(path)
    return _pre(text) + _cut_note(cut)


def _table(path: Path) -> str:
    try:
        import pandas as pd

        sep = "\t" if suffix(path).startswith(".tsv") else ","
        frame = pd.read_csv(path, sep=sep, nrows=MAP_ROWS + 1, low_memory=False)  # .gz is inferred
    except Exception:  # no pandas, or not really a table
        if suffix(path).endswith(".gz"):
            return note("Saved. Showing a compressed table needs pandas.", card=True)
        text, cut = _read_text(path)
        return _pre(text) + _cut_note(cut)
    more = len(frame) > CSV_ROWS
    table = frame.head(CSV_ROWS).to_html(index=False, border=0, classes="fx-table")
    rows = (f"Showing the first {CSV_ROWS} rows of {len(frame):,}{'+' if len(frame) > MAP_ROWS else ''}."
            if more else f"{len(frame)} rows.")
    return _points_map(frame.head(MAP_ROWS)) + f'<div class="fx-table-wrap">{table}</div>' + note(rows)


# ----- maps -------------------------------------------------------------------


def _png_img(data: bytes, alt: str, css: str = "fx-map") -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f'<img class="{css}" src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'


def _time_legend(span, label: str) -> str:
    """first ▬▬ last bar for colours by date, as the image viewer draws it."""
    from . import geo, render

    if span is None:
        return ""
    first, last = (geo.format_time(t) for t in span)
    return (f'<div class="fx-map-legend"><span>{html.escape(first)}</span>'
            f'<span class="fx-bar" style="background:{render.ramp_css("fire")}"></span>'
            f'<span>{html.escape(last)}</span>'
            f'<span class="fx-legend-note">colour = {html.escape(label)}, earliest light to latest dark</span></div>')


def _points_map(frame) -> str:
    """A map of a table's latitude/longitude rows (FIRMS detections), coloured by date; "" without them."""
    from . import geo

    found = geo.csv_points(frame)
    if found is None:
        return ""
    lon, lat, times, label = found
    if len(lon) > geo.MAX_POINTS:
        keep = slice(None, None, -(-len(lon) // geo.MAX_POINTS))
        lon, lat, times = lon[keep], lat[keep], times[keep]
    colours, span = geo.time_colours(times)
    order = geo.latest_first(times)[::-1]  # latest on top: the newest detections stay visible
    image = geo.map_png(points=(lon[order], lat[order]), point_colours=[colours[i] for i in order])
    return (_png_img(image, "point map") + (_time_legend(span, label) if label else "")
            + note(f"{len(lon):,} points from the latitude and longitude columns."))


def geojson_html(path: Path) -> str:
    """A GeoJSON file as a map coloured by date, its geometry types and attribute table."""
    from . import geo

    data = geo.read_geojson(path)
    features = data.features
    if not features:
        return note("No features: the FeatureCollection is empty.", card=True)
    key, times = geo.feature_times(features)
    colours, span = geo.time_colours(times)
    parts = [f"<b>{len(features):,}</b> feature{'s' if len(features) != 1 else ''}"
             f"{'' if data.complete else ' (first ones)'} · {html.escape(', '.join(data.geometry_types))}"]
    if key:
        parts.append(f"coloured by <code>{html.escape(key)}</code>")
    head = f'<div class="fx-map-head">{" · ".join(parts)}</div>'
    image = geo.map_png(features, colours, draw_order=geo.latest_first(times))
    body = head + _png_img(image, path.name) + (_time_legend(span, key) if key else "")
    if data.note:
        body += note(html.escape(data.note))
    return body + _properties_table(features)


def _properties_table(features) -> str:
    rows = [dict(f.get("properties") or {}) for f in features[:CSV_ROWS]]
    if not rows or not any(rows):
        return ""
    try:
        import pandas as pd

        table = pd.DataFrame(rows).to_html(index=False, border=0, classes="fx-table", na_rep="")
    except ImportError:
        columns = list(dict.fromkeys(k for row in rows for k in row))
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns) + "</tr>"
                       for row in rows)
        table = f'<table class="fx-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
    more = f"Properties of the first {CSV_ROWS} features." if len(features) > CSV_ROWS else "Feature properties."
    return f'<div class="fx-table-wrap">{table}</div>' + note(more)


def osm_html(path: Path) -> str:
    """An OpenStreetMap extract: what it holds, and a map of its roads, buildings and water."""
    from . import geo, pbf

    decode = path.stat().st_size <= OSM_DECODE_BYTES
    summary = pbf.read_pbf(path, decode=decode)
    facts = [f"written by {html.escape(summary.writing_program or 'unknown')}"]
    if summary.replication_timestamp:
        facts.append(f"data as of {summary.replication_timestamp:%Y-%m-%d %H:%M} UTC")
    if summary.decoded:
        facts.insert(0, f"<b>{summary.nodes:,}</b> nodes · <b>{summary.ways:,}</b> ways · "
                        f"<b>{summary.relations:,}</b> relations")
    else:
        facts.insert(0, f"{summary.data_blocks:,} data blocks (too large to decode here)")
    head = f'<div class="fx-map-head">{" · ".join(facts)}</div>'
    if not summary.decoded or not summary.drawn:
        empty = "No roads, buildings or water in this extract." if summary.decoded else ""
        return head + (note(empty) if empty else "") + _osm_hint()
    features, colours, chips = [], [], []
    for key, (label, colour) in pbf.CATEGORIES.items():
        ways = summary.drawn.get(key, [])
        if not ways:
            continue
        rgb = tuple(c / 255 for c in _hex(colour))
        for xy in ways:
            closed = key in pbf.AREAS and len(xy) >= 4 and np_equal(xy[0], xy[-1])
            geometry = ({"type": "Polygon", "coordinates": [xy.tolist()]} if closed
                        else {"type": "LineString", "coordinates": xy.tolist()})
            features.append({"geometry": geometry})
            colours.append(rgb)
        chips.append(f'<span class="fx-class"><span class="fx-swatch" style="background:{colour}"></span>'
                     f"{html.escape(label)} {len(ways):,}</span>")
    image = geo.map_png(features, colours)
    keys = "".join(f"<tr><td>{html.escape(k)}</td><td>{n:,}</td></tr>" for k, n in summary.way_keys.most_common(15))
    table = (f'<div class="fx-table-wrap"><table class="fx-table"><thead><tr><th>way tag</th><th>ways</th></tr>'
             f"</thead><tbody>{keys}</tbody></table></div>" if keys else "")
    return (head + _png_img(image, path.name) + f'<div class="fx-map-legend">{"".join(chips)}</div>'
            + table + _osm_hint())


def _osm_hint() -> str:
    return note("Every tag and relation: <code>pyrosm</code> or <code>osmium</code> "
                "(<code>%pip install osmium</code>) on the saved file.")


def _hex(colour: str):
    colour = colour.lstrip("#")
    return tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4))


def np_equal(a, b) -> bool:
    return bool((a == b).all())


def zip_html(path: Path, limit: int = 500) -> str:
    """The members of a ZIP archive: name, size, packed size and date."""
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
    total = sum(m.file_size for m in members)
    rows = "".join(
        f"<tr><td>{html.escape(m.filename)}</td><td>{m.file_size:,}</td><td>{m.compress_size:,}</td>"
        f"<td>{datetime(*m.date_time):%Y-%m-%d %H:%M}</td></tr>" for m in members[:limit])
    table = (f'<div class="fx-table-wrap"><table class="fx-table"><thead><tr><th>member</th><th>bytes</th>'
             f"<th>packed</th><th>modified</th></tr></thead><tbody>{rows}</tbody></table></div>")
    more = f" Showing the first {limit}." if len(members) > limit else ""
    return (f'<div class="fx-map-head"><b>{len(members):,}</b> files, {total:,} bytes unpacked.{more}</div>'
            + table + note("Unpack with <code>zipfile.ZipFile(path).extractall()</code>."))


def _front_matter_as_code(text: str) -> str:
    """Show a Hub README's leading YAML block as code instead of stray text."""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            yaml_block = text[4:end]
            rest = text[end + 4:].lstrip("\n")
            return f"```yaml\n{yaml_block}\n```\n\n{rest}"
    return text
