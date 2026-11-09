"""Show a downloaded file in the notebook.

Text files (txt, json, md, csv, yaml, ...) are shown as you'd expect: JSON
pretty-printed, Markdown rendered, CSV as a table, everything else as plain
text. NumPy archives and other binary files are only downloaded for now; the
preview says where the file was saved.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import List, Optional

from IPython.display import HTML, Markdown

from .hub import Entry

MAX_CHARS = 100_000  # longest text shown; the rest is cut with a note
CSV_ROWS = 200  # rows of a CSV shown as a table
JSON_PARSE_LIMIT = 5_000_000  # bytes; bigger JSON is shown as plain text

NUMPY = {".npz", ".npy"}
MARKDOWN = {".md", ".markdown"}
TABLES = {".csv", ".tsv"}
TEXT = {
    ".txt", ".json", ".yaml", ".yml", ".log", ".py", ".cfg", ".ini", ".toml",
    ".xml", ".sh", ".gitattributes", ".gitignore",
} | MARKDOWN | TABLES
BINARY = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".zip", ".gz", ".zst", ".tar",
    ".nc", ".h5", ".hdf5", ".parquet", ".pt", ".pth", ".bin",
} | NUMPY

_BOX = (
    "max-width:980px;margin:6px 0;padding:10px 12px;border:1px solid #d9d9de;"
    "border-radius:8px;background:#f7f7f9;color:#1d1d1f;"
)
_PRE = (
    _BOX + "max-height:480px;overflow:auto;white-space:pre-wrap;word-break:break-word;"
    "font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
)
_NOTE = "font:12px -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#6e6e73;"


def suffix(path: Path) -> str:
    """Lower-case extension; a dotfile like ".gitattributes" counts as its own."""
    name = path.name.lower()
    if name.startswith(".") and name.count(".") == 1:
        return name
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
    """One of "numpy", "markdown", "json", "table", "text" or "binary"."""
    ext = suffix(path)
    if ext in NUMPY:
        return "numpy"
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


def render(path: Path, entry: Optional[Entry] = None) -> List[object]:
    """Displayable objects for a file: a header line, then its contents."""
    path = Path(path)
    kind = file_kind(path)
    items: List[object] = [_header(path, entry)]
    if kind == "numpy":
        items.append(_note(
            "NumPy archive saved. No preview yet. Load it with "
            f"<code>np.load({html.escape(repr(str(path)))})</code>."
        ))
    elif kind == "binary":
        items.append(_note(f"Saved. No preview for {html.escape(suffix(path) or 'this')} files yet."))
    elif kind == "table":
        items.extend(_table(path))
    elif kind == "json":
        items.extend(_json(path))
    elif kind == "markdown":
        text, cut = _read_text(path)
        items.append(Markdown(_front_matter_as_code(text)))
        if cut:
            items.append(_note(cut))
    else:
        items.extend(_text(*_read_text(path)))
    return items


def _header(path: Path, entry: Optional[Entry]) -> HTML:
    from .explorer import format_size, icon_and_kind

    name = entry.name if entry else path.name
    size = entry.size if entry and entry.size is not None else path.stat().st_size
    icon = icon_and_kind(entry)[0] if entry else "📄"
    return HTML(
        f'<div style="{_NOTE}margin-top:8px">{icon} <b style="color:inherit">'
        f"{html.escape(name)}</b> · {format_size(size)} · saved at "
        f"<code>{html.escape(str(path))}</code></div>"
    )


def _note(message: str) -> HTML:
    return HTML(f'<div style="{_BOX}{_NOTE}">{message}</div>')


def _read_text(path: Path):
    """The file's text (at most MAX_CHARS) and a note if it was cut short."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read(MAX_CHARS + 1)
    if len(text) <= MAX_CHARS:
        return text, ""
    total = path.stat().st_size
    return text[:MAX_CHARS], f"Showing the first {MAX_CHARS:,} characters of {total:,} bytes."


def _text(text: str, cut: str = "") -> List[object]:
    items: List[object] = [HTML(f'<pre style="{_PRE}">{html.escape(text, quote=False)}</pre>')]
    if cut:
        items.append(_note(cut))
    return items


def _json(path: Path) -> List[object]:
    if path.stat().st_size <= JSON_PARSE_LIMIT:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            pass  # not valid JSON: show it as it is
        else:
            text = json.dumps(data, indent=2, ensure_ascii=False)
            cut = ""
            if len(text) > MAX_CHARS:
                text, cut = text[:MAX_CHARS], f"Showing the first {MAX_CHARS:,} characters."
            return _text(text, cut)
    return _text(*_read_text(path))


def _table(path: Path) -> List[object]:
    try:
        import pandas as pd

        sep = "\t" if suffix(path) == ".tsv" else ","
        frame = pd.read_csv(path, sep=sep, nrows=CSV_ROWS + 1)
    except Exception:  # no pandas, or not really a table
        return _text(*_read_text(path))
    if len(frame) <= CSV_ROWS:
        return [frame]
    return [frame.head(CSV_ROWS), _note(f"Showing the first {CSV_ROWS} rows.")]


def _front_matter_as_code(text: str) -> str:
    """Show a Hub README's leading YAML block as code instead of stray text."""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            yaml_block = text[4:end]
            rest = text[end + 4:].lstrip("\n")
            return f"```yaml\n{yaml_block}\n```\n\n{rest}"
    return text
