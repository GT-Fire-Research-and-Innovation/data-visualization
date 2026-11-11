"""Turn a downloaded file into HTML for the explorer's preview pane.

Everything here returns an HTML string that the explorer puts into a widget
inside its own window. That works the same in Colab and Jupyter, unlike
printing or `display()` from a click handler, which Colab may not show.

Text files (txt, json, md, csv, yaml, ...) are shown as you'd expect: JSON
pretty-printed, Markdown rendered, CSV as a table, everything else as plain
text. PNG files are shown as images. Other binary files have a saved-file note.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

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
    ".tif", ".tiff", ".jpg", ".jpeg", ".zip", ".gz", ".zst", ".tar",
    ".nc", ".h5", ".hdf5", ".parquet", ".pt", ".pth", ".bin",
} | NUMPY


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
    """One of "numpy", "png", "markdown", "json", "table", "text" or "binary"."""
    ext = suffix(path)
    if ext in NUMPY:
        return "numpy"
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

        sep = "\t" if suffix(path) == ".tsv" else ","
        frame = pd.read_csv(path, sep=sep, nrows=CSV_ROWS + 1)
    except Exception:  # no pandas, or not really a table
        text, cut = _read_text(path)
        return _pre(text) + _cut_note(cut)
    more = len(frame) > CSV_ROWS
    table = frame.head(CSV_ROWS).to_html(index=False, border=0, classes="fx-table")
    rows = f"Showing the first {CSV_ROWS} rows." if more else f"{len(frame)} rows."
    return f'<div class="fx-table-wrap">{table}</div>' + note(rows)


def _front_matter_as_code(text: str) -> str:
    """Show a Hub README's leading YAML block as code instead of stray text."""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            yaml_block = text[4:end]
            rest = text[end + 4:].lstrip("\n")
            return f"```yaml\n{yaml_block}\n```\n\n{rest}"
    return text
