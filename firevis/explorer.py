"""A Finder / File Explorer style browser for Hub datasets, built on ipywidgets.

    from firevis import open_explorer
    explorer = open_explorer()          # root: https://huggingface.co/SamuelWu318

Every file and folder name loads when the explorer opens, so browsing needs no
further requests. Clicking a folder opens it. Clicking a file opens it: the
first time, it is downloaded into a local mirror of the Hub layout
(`hf_mirror/<owner>/<dataset>/<path>`); later clicks use that saved copy. Text
files (txt, json, md, csv, ...), PNG images and NumPy archives are shown in a
preview pane inside the window. Other binary files are saved. Files over 1 GB
ask first.

A window archive's preview has a "✂ Visualize split" button: it splits the
archive into its fires with the fire-split code (firevis/split.py) and shows
the burned pixels before, and each fire's after, overlaid over time.

Functions registered with `explorer.on_file_click(fn)` run after that, with the
file's `Entry`; anything they print or plot shows under the window.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional

import ipywidgets as widgets
from IPython.display import display

from . import viewers
from .split import FireSplit, SplitError, can_split, split_archive
from .split_viewer import SplitViewer
from .stacks import FireStack, NpzStack, is_fire_folder
from .viewer import StackViewer
from .fire import npz_date_lookup
from .hub import DATASET, DEFAULT_ROOT, FILE, FOLDER, ROOT, Entry, HubSource, Location

FileCallback = Callable[[Entry], None]

# Extension -> (icon, kind label) for the Kind column.
_FILE_KINDS = {
    ".npz": ("🧮", "NumPy archive"),
    ".npy": ("🧮", "NumPy array"),
    ".tif": ("🖼️", "GeoTIFF image"),
    ".tiff": ("🖼️", "GeoTIFF image"),
    ".png": ("🖼️", "PNG image"),
    ".jpg": ("🖼️", "JPEG image"),
    ".jpeg": ("🖼️", "JPEG image"),
    ".csv": ("📊", "CSV table"),
    ".parquet": ("📊", "Parquet table"),
    ".json": ("📄", "JSON"),
    ".md": ("📝", "Markdown"),
    ".txt": ("📝", "Text"),
    ".zip": ("📦", "ZIP archive"),
    ".zst": ("📦", "Zstandard archive"),
    ".gz": ("📦", "Gzip archive"),
    ".tar": ("📦", "Tar archive"),
    ".nc": ("🌐", "NetCDF"),
    ".h5": ("🌐", "HDF5"),
    ".hdf5": ("🌐", "HDF5"),
}


def icon_and_kind(entry: Entry):
    if entry.kind == DATASET:
        return "🗂️", "Dataset"
    if entry.kind == FOLDER:
        return "📁", "Folder"
    ext = os.path.splitext(entry.name)[1].lower()
    if ext in _FILE_KINDS:
        return _FILE_KINDS[ext]
    return "📄", f"{ext[1:].upper()} file" if ext else "File"


def format_size(size: Optional[int]) -> str:
    """Bytes as Finder shows them (1000-based): 6.9 KB, 1.8 GB."""
    if size is None:
        return "—"
    if size < 1000:
        return "1 byte" if size == 1 else f"{size} bytes"
    value = float(size)
    for unit in ("KB", "MB", "GB", "TB"):
        value /= 1000
        if value < 1000 or unit == "TB":
            return f"{value:.1f} {unit}"
    return f"{size} bytes"  # unreachable


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_CSS = """
<style>
.fx-window {
  --fx-text: #1d1d1f; --fx-muted: #6e6e73; --fx-line: #d9d9de;
  --fx-chrome: #f3f3f5; --fx-side: #ececef; --fx-zebra: #f7f7f9;
  --fx-hover: #e7eefc; --fx-accent: #0a64d6; --fx-active: #dcdce2;
  max-width: 1400px; background: #fff; border: 1px solid var(--fx-line);
  border-radius: 10px; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,.10);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 13px; color: var(--fx-text);
}
.fx-window .jupyter-widgets { margin: 0; }
.fx-window .widget-html, .fx-window .widget-html-content {
  height: auto; line-height: normal; min-width: 0; color: var(--fx-text);
}
.fx-window .jupyter-button {
  font-family: inherit; font-size: 13px; color: var(--fx-text) !important;
  background: transparent !important; border: 0 !important; box-shadow: none !important;
  outline: none !important; border-radius: 5px; height: 26px; line-height: 26px;
  overflow: hidden; white-space: nowrap; text-overflow: ellipsis;
}
.fx-window .jupyter-button:disabled { opacity: .35; }

/* title bar */
.fx-title { display: flex; align-items: center; height: 30px; padding: 0 12px;
  background: var(--fx-chrome); }
.fx-title-text { flex: 1; text-align: center; font-weight: 600;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* toolbar + address bar */
.fx-toolbar { align-items: center; gap: 2px; padding: 4px 10px 8px;
  background: var(--fx-chrome); border-bottom: 1px solid var(--fx-line); }
.fx-window .jupyter-button.fx-tool { width: 30px; padding: 0; font-size: 15px; }
.fx-window .jupyter-button.fx-tool:hover:enabled { background: var(--fx-active) !important; }
.fx-crumbs { flex: 1 1 auto; align-items: center; margin-left: 8px !important;
  background: #fff; border: 1px solid var(--fx-line); border-radius: 6px;
  height: 28px; padding: 0 4px; overflow: hidden; }
.fx-window .jupyter-button.fx-crumb { width: auto; padding: 0 4px; flex: 0 0 auto; }
.fx-window .jupyter-button.fx-crumb:hover { background: var(--fx-hover) !important; }
.fx-window .jupyter-button.fx-crumb + .fx-crumb::before {
  content: "›"; color: var(--fx-muted); margin-right: 8px; }
.fx-window .jupyter-button.fx-crumb.fx-current { font-weight: 600; }

/* sidebar */
.fx-sidebar { flex: 0 0 200px; padding: 8px; background: var(--fx-side);
  border-right: 1px solid var(--fx-line); overflow-y: auto; }
.fx-side-label { font-size: 11px; font-weight: 600; color: var(--fx-muted);
  padding: 10px 8px 4px; }
.fx-window .jupyter-button.fx-side { width: 100%; text-align: left; padding: 0 8px; }
.fx-window .jupyter-button.fx-side:hover { background: var(--fx-active) !important; }
.fx-window .jupyter-button.fx-side.fx-active { background: var(--fx-active) !important;
  font-weight: 600; }

/* file list */
.fx-main { flex: 1 1 auto; min-width: 0; }
.fx-head { display: flex; height: 26px; align-items: center; padding: 0 8px;
  border-bottom: 1px solid var(--fx-line); color: var(--fx-muted); font-size: 12px;
  font-weight: 600; }
.fx-col-name { flex: 1 1 auto; padding-left: 8px; }
.fx-size { display: inline-block; width: 80px; text-align: right; padding-right: 18px;
  box-sizing: border-box; color: var(--fx-muted); }
.fx-kind { display: inline-block; width: 130px; overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; color: var(--fx-muted); vertical-align: top; }
.fx-list { overflow-y: auto; padding: 4px 8px; }
.fx-row { height: 26px; align-items: center; border-radius: 5px; flex: 0 0 auto; }
.fx-row:nth-child(even) { background: var(--fx-zebra); }
.fx-row:hover { background: var(--fx-hover); }
.fx-window .jupyter-button.fx-name { flex: 1 1 auto; width: auto; min-width: 0;
  text-align: left; padding: 0 8px; }
.fx-meta { flex: 0 0 228px; line-height: 26px; }
.fx-local { display: inline-block; width: 18px; text-align: center; color: #1a7f37;
  font-weight: 700; }
.fx-row.fx-selected { background: var(--fx-accent) !important; }
.fx-row.fx-selected .jupyter-button, .fx-row.fx-selected .fx-size,
.fx-row.fx-selected .fx-kind, .fx-row.fx-selected .fx-local { color: #fff !important; }
.fx-empty { color: var(--fx-muted); text-align: center; padding: 40px 0; }
.fx-window .jupyter-button.fx-more { width: auto; color: var(--fx-accent) !important;
  margin: 6px auto !important; padding: 0 10px; }

/* preview pane: right of the list, or below it when the window is narrow */
.fx-body { flex-wrap: wrap; }
.fx-main { flex: 1 1 380px; }
.fx-preview { flex: 1 1 440px; min-width: 0; border-left: 1px solid var(--fx-line);
  background: #fff; }
.fx-preview-head, .fx-preview-path, .fx-preview-ask { flex: 0 0 auto; }
.fx-preview-head { align-items: center; height: 34px; padding: 0 6px 0 12px;
  background: var(--fx-chrome); border-bottom: 1px solid var(--fx-line); }
.fx-preview-title { flex: 1 1 auto; min-width: 0; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; font-weight: 600; }
.fx-preview-title span { font-weight: 400; color: var(--fx-muted); }
.fx-preview-path { padding: 6px 12px; font-size: 11px; color: var(--fx-muted);
  word-break: break-all; border-bottom: 1px solid var(--fx-line); }
.fx-preview-body { flex: 1 1 auto; min-height: 0; overflow: auto; padding: 10px 12px; }
.fx-preview-png { display: block; max-width: 100%; height: auto; }
.fx-preview-ask { padding: 0 12px 12px; justify-content: center; }
.fx-window .jupyter-button.fx-primary { width: auto; padding: 0 14px;
  background: var(--fx-accent) !important; color: #fff !important; }
.fx-code { margin: 0; padding: 8px 10px; background: var(--fx-zebra);
  border: 1px solid var(--fx-line); border-radius: 6px; color: var(--fx-text);
  font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre-wrap; word-break: break-word; }
.fx-md { line-height: 1.55; }
.fx-md h1 { font-size: 20px; margin: 4px 0 10px; }
.fx-md h2 { font-size: 16px; margin: 16px 0 8px; }
.fx-md h3, .fx-md h4 { font-size: 14px; margin: 14px 0 6px; }
.fx-md p, .fx-md ul, .fx-md ol { margin: 0 0 10px; }
.fx-md code { font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  background: var(--fx-zebra); padding: 1px 4px; border-radius: 4px; }
.fx-md pre { margin: 0 0 10px; padding: 8px 10px; background: var(--fx-zebra);
  border: 1px solid var(--fx-line); border-radius: 6px; white-space: pre-wrap; }
.fx-md pre code { padding: 0; background: none; }
.fx-md a { color: var(--fx-accent); }
.fx-table-wrap { overflow-x: auto; }
.fx-md table, .fx-table { border-collapse: collapse; font-size: 12px; }
.fx-md th, .fx-md td, .fx-table th, .fx-table td { border: 1px solid var(--fx-line);
  padding: 3px 8px; text-align: left; white-space: nowrap; }
.fx-md th, .fx-table th { background: var(--fx-chrome); }
.fx-note { margin: 8px 0; color: var(--fx-muted); font-size: 12px; }

/* image viewer (firevis/viewer.py) */
.fx-preview.fx-preview-wide { flex: 1 1 100%; border-left: 0;
  border-top: 1px solid var(--fx-line); }
.fx-preview-body { align-items: flex-start; }  /* content keeps its height; the body scrolls */
.fx-viewer { width: 100%; gap: 8px; }
.fx-viewer > * { flex: 0 0 auto !important; }
.fx-controls { flex-wrap: wrap; gap: 6px; }
.fx-window .fx-select select { font-size: 12px; height: 26px; }
.fx-window .fx-stage { position: relative; display: block !important; width: 100%; max-width: 560px;
  margin: 0 auto; border: 1px solid var(--fx-line); border-radius: 6px; overflow: hidden;
  background: var(--fx-zebra); }
.fx-stage.fx-big { max-width: 960px; }
.fx-stage img, .fx-image { display: block; width: 100% !important; height: auto !important;
  image-rendering: pixelated; }
.fx-window .jupyter-button.fx-corner { position: absolute; top: 6px; height: 24px;
  line-height: 24px; padding: 0 8px; font-size: 12px;
  background: rgba(255,255,255,.88) !important; border: 1px solid var(--fx-line) !important;
  border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.15) !important; }
.fx-window .jupyter-button.fx-corner-left { left: 6px; }
.fx-window .jupyter-button.fx-corner-right { right: 6px; }
.fx-window .jupyter-button.fx-corner.fx-on { background: var(--fx-accent) !important;
  color: #fff !important; border-color: var(--fx-accent) !important; }
.fx-legend .widget-html-content { display: flex; flex-wrap: wrap; align-items: center;
  justify-content: center; gap: 6px 10px; font-size: 12px; color: var(--fx-muted); }
.fx-bar { display: inline-block; width: 160px; height: 10px; border-radius: 5px;
  border: 1px solid var(--fx-line); }
.fx-legend-note { flex-basis: 100%; text-align: center; }
.fx-legend-error { color: #c4314b; }
.fx-swatch { display: inline-block; width: 11px; height: 11px; border-radius: 3px;
  margin-right: 4px; vertical-align: -1px; border: 1px solid rgba(0,0,0,.15); }
.fx-class { white-space: nowrap; }
.fx-nav { align-items: center; justify-content: space-between; max-width: 560px;
  width: 100%; margin: 0 auto; }
.fx-viewer-big .fx-nav { max-width: 960px; }
.fx-window .jupyter-button.fx-arrow { width: 44px; font-size: 16px;
  background: var(--fx-chrome) !important; border: 1px solid var(--fx-line) !important; }
.fx-window .jupyter-button.fx-arrow:hover:enabled { background: var(--fx-hover) !important; }
.fx-date { flex: 1 1 auto; text-align: center; }
.fx-date .widget-html-content { justify-content: center; text-align: center; }
.fx-card { margin: 40px 16px; text-align: center; font-size: 13px; line-height: 1.6; }
.fx-note-error { color: #c4314b; }

/* fire split (firevis/split_viewer.py) */
.fx-preview-actions { flex: 0 0 auto; padding: 8px 12px 0; gap: 6px; }
.fx-split-summary .widget-html-content { display: block; font-size: 13px; line-height: 1.5; }
.fx-split-detail, .fx-split-note .widget-html-content { font-size: 12px; color: var(--fx-muted);
  line-height: 1.5; }
.fx-split-note .widget-html-content { display: block; text-align: center; max-width: 560px;
  margin: 0 auto; }
.fx-window .jupyter-button.fx-tab { width: auto; padding: 0 12px;
  background: var(--fx-chrome) !important; border: 1px solid var(--fx-line) !important; }
.fx-window .jupyter-button.fx-tab:hover { background: var(--fx-hover) !important; }
.fx-window .jupyter-button.fx-tab.fx-on { background: var(--fx-accent) !important;
  color: #fff !important; border-color: var(--fx-accent) !important; }
.fx-split-grid { display: flex !important; flex-wrap: wrap; gap: 10px; width: 100%; }
.fx-split-card { flex: 1 1 200px !important; max-width: 280px; min-width: 0;
  border: 1px solid var(--fx-line); border-radius: 6px; overflow: hidden; background: #fff; }
.fx-viewer-big .fx-split-card { max-width: 320px; }
.fx-split-card img { display: block; width: 100% !important; height: auto !important;
  image-rendering: pixelated; background: var(--fx-zebra); }
.fx-split-caption .widget-html-content { display: block; padding: 6px 8px; font-size: 12px;
  line-height: 1.45; white-space: normal; }
.fx-split-when { color: var(--fx-muted); }
.fx-window .jupyter-button.fx-split-back { width: auto; padding: 0 12px; font-size: 13px; }

/* status bar */
.fx-status { height: 24px; line-height: 24px; padding: 0 12px; font-size: 12px;
  color: var(--fx-muted); background: var(--fx-chrome);
  border-top: 1px solid var(--fx-line); overflow: hidden; white-space: nowrap;
  text-overflow: ellipsis; }
.fx-status.fx-error { color: #c4314b; }
</style>
"""


def _button(description: str = "", *classes: str, tooltip: str = "", **layout) -> widgets.Button:
    button = widgets.Button(
        description=description, tooltip=tooltip, layout=widgets.Layout(**layout)
    )
    for name in classes:
        button.add_class(name)
    return button


def _set_class(widget: widgets.DOMWidget, name: str, on: bool) -> None:
    if on:
        widget.add_class(name)
    else:
        widget.remove_class(name)


class DatasetExplorer:
    """File-explorer window over a Hub user's datasets.

    All file and folder names load once, up front, so browsing is instant.
    Clicking a file opens it: the first time, it is downloaded into the local
    mirror (`mirror_dir`); after that the saved copy is used. Text files are
    shown in the window's preview pane; other files are just saved for now.

    root:          Hub link or user/org name whose datasets form the top level.
    source:        A `HubSource` (or anything with the same methods); defaults
                   to `HubSource(root, mirror_dir=mirror_dir)`.
    mirror_dir:    Folder that downloaded files are saved into. Use a mounted
                   Google Drive folder to keep them across Colab runtimes.
    on_file_click: Optional function called with the `Entry` of a clicked file.
    open_files:    Download files and show them in the preview pane when
                   clicked (default True).
    confirm_size:  Files bigger than this many bytes ask before downloading.
    height:        Height of the list area, in pixels.
    page_size:     Rows shown before a "Show more" button appears.
    autoload:      Load the names now. `open_explorer` passes False so it can
                   show the window first and fill it in.
    """

    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        *,
        source=None,
        mirror_dir: str = "hf_mirror",
        on_file_click: Optional[FileCallback] = None,
        open_files: bool = True,
        confirm_size: int = 1_000_000_000,
        height: int = 420,
        page_size: int = 200,
        autoload: bool = True,
    ):
        self.source = source if source is not None else HubSource(root, mirror_dir=mirror_dir)
        self.open_files = open_files
        self.confirm_size = confirm_size
        self.page_size = page_size
        self.location: Location = ROOT
        self.entries: List[Entry] = []
        self.selected: Optional[Entry] = None
        self._callbacks: List[FileCallback] = []
        self._back: List[Location] = []
        self._forward: List[Location] = []
        self._limit = page_size
        self._rows = []  # reusable (row, name button, meta html) triples
        self._crumbs: List[widgets.Button] = []
        self._side_buttons = {}  # repo_id (None = root) -> Button

        if on_file_click is not None:
            self.on_file_click(on_file_click)

        self._build(height)
        if autoload:
            self.load()

    # ----- public API -------------------------------------------------------

    def load(self) -> bool:
        """Fetch every file and folder name from the Hub and show the root."""
        self._set_status(f"Loading file list for {self.source.owner}…")
        try:
            self.source.load()
        except Exception as error:
            self._set_status(f"⚠️ Couldn't load {self.source.owner}: {error}", error=True)
            return False
        self._back.clear()
        self._forward.clear()
        self._load_sidebar()
        return self.navigate(ROOT, record=False)

    @property
    def mirror_dir(self) -> Path:
        """Folder that opened files are saved into."""
        return self.source.mirror_dir

    def download(self, entry: Optional[Entry] = None) -> Path:
        """Path to a file in the mirror (the selected file by default).

        The first call fetches it from the Hub; later calls, even in a new
        explorer, return the saved copy without downloading.
        """
        entry = self._file(entry)
        if not self.source.is_local(entry):
            self._set_status(f"Downloading {entry.name} ({format_size(entry.size)})…")
        try:
            path = self.source.download(entry)
        finally:
            self._update_status()
        self._refresh_row(entry)
        return path

    def open_file(self, entry: Optional[Entry] = None, confirmed: bool = False) -> Optional[Path]:
        """Download a file if needed and show it in the preview pane.

        Returns the file's path in the mirror, or None if it couldn't be opened
        or is waiting for the Download button. Files over `confirm_size` ask
        first, unless `confirmed` is True or they are already saved.
        """
        entry = self._file(entry)
        too_big = entry.size is not None and entry.size > self.confirm_size
        if too_big and not confirmed and not self.source.is_local(entry):
            self._show_preview(entry, viewers.note(
                f"<b>{_escape(entry.name)}</b> is {format_size(entry.size)}.<br>"
                f"Download it into <code>{_escape(str(self.mirror_dir))}</code>?",
                card=True,
            ), ask=lambda: self.open_file(entry, confirmed=True))
            return None
        if not self.source.is_local(entry):
            self._show_preview(entry, viewers.note(
                f"Downloading {_escape(entry.name)} ({format_size(entry.size)})…", card=True))
        try:
            path = self.download(entry)
        except Exception as error:  # gated dataset, no network, disk full...
            body = self._error_html("Couldn't download", entry, error)
            if type(error).__name__ == "GatedRepoError":
                body += viewers.note(
                    f"{_escape(entry.repo_id)} is gated: accept its terms at "
                    f'<a href="https://huggingface.co/datasets/{entry.repo_id}" target="_blank">'
                    f"{_escape(entry.repo_id)}</a>, "
                    "then set an <code>HF_TOKEN</code> (in Colab: a secret) and reopen."
                )
            self._show_preview(entry, body)
            return None
        try:
            if path.suffix.lower() == ".npz":
                self._show_viewer(entry, NpzStack(path, dates=self._npz_dates(entry)),
                                  split=can_split(path))
                return path
            body = viewers.to_html(path)
        except Exception as error:  # unreadable file: say so instead of a blank pane
            body = self._error_html("Saved, but couldn't show", entry, error)
        self._show_preview(entry, body)
        return path

    def open_fire(self, folder: Optional[Entry] = None, confirmed: bool = False) -> Optional[Path]:
        """Download every GeoTIFF of a fire folder and show them in the viewer.

        A fire folder holds VIIRS_Day/ plus FirePred/ or VIIRS_Night/ (and
        usually ESRI_LULC/). Saved files are reused; folders whose missing
        files total more than `confirm_size` ask first. Returns the folder's
        path in the mirror, or None if it couldn't be opened (or is waiting).
        """
        folder = folder or self._entry_for(self.location)
        if folder is None or not self.is_fire_folder(folder.location):
            raise ValueError("Not a fire folder (needs VIIRS_Day plus FirePred or VIIRS_Night)")
        tifs = [e for e in self.source.files_under(folder.location) if e.name.lower().endswith(".tif")]
        missing = sum(e.size or 0 for e in tifs if not self.source.is_local(e))
        if missing > self.confirm_size and not confirmed:
            self._show_preview(folder, viewers.note(
                f"<b>{_escape(folder.name)}</b> has {len(tifs)} GeoTIFFs; "
                f"{format_size(missing)} still to download.<br>"
                f"Download them into <code>{_escape(str(self.mirror_dir))}</code>?", card=True,
            ), ask=lambda: self.open_fire(folder, confirmed=True), kind="Fire folder")
            return None

        def progress(done: int, total: int, done_bytes: int, total_bytes: int) -> None:
            text = (f"Downloading {folder.name}: {done} / {total} files · "
                    f"{format_size(done_bytes)} of {format_size(total_bytes)}")
            self._set_status(text + "…")
            self._show_preview(folder, viewers.note(_escape(text) + "…", card=True), kind="Fire folder")

        try:
            self.source.download_many(tifs, progress=progress)
        except Exception as error:
            self._show_preview(folder, self._error_html("Couldn't download", folder, error),
                               kind="Fire folder")
            return None
        path = self.source.local_path(folder)
        try:
            self._show_viewer(folder, FireStack(path), kind="Fire folder")
        except Exception as error:
            self._show_preview(folder, self._error_html("Downloaded, but couldn't show", folder, error),
                               kind="Fire folder")
            return None
        self._update_status()
        for entry in self.entries:
            if not entry.is_dir:
                self._refresh_row(entry)
        return path

    def open_split(self, entry: Optional[Entry] = None, scale_km: Optional[float] = None,
                   mode: str = "after") -> Optional[FireSplit]:
        """Split a window archive into its fires and show before / after in the pane.

        The archive is downloaded first if needed. Splitting reads only its
        targets, the first window's night I4 and two false-colour days, then
        runs firevis/firesplit (fire-split-zones). Returns the FireSplit, or
        None when the archive can't be split (the pane says why).
        `scale_km` overrides the 2.25 km separation distance.
        """
        entry = self._file(entry)
        if not entry.name.lower().endswith(".npz"):
            raise ValueError(f"{entry.name} isn't an .npz archive")
        try:
            path = self.download(entry)
        except Exception as error:
            self._show_preview(entry, self._error_html("Couldn't download", entry, error))
            return None
        options = {} if scale_km is None else {"scale_km": scale_km}
        key = (path, scale_km)
        if self._last_split is not None and self._last_split[0] == key:
            result = self._last_split[1]  # back from the layers: no need to split again
        else:
            self._set_status(f"Splitting {entry.name} into fires…")
            self._show_preview(entry, viewers.note(
                f"Splitting <b>{_escape(entry.name)}</b> into fires…", card=True), kind="Fire split")
            try:
                result = split_archive(path, dates=self._npz_dates(entry), **options)
            except SplitError as error:
                self._set_status(f"⚠️ Can't split {entry.name}: {error}", error=True)
                self._show_preview(entry, viewers.note(
                    f"⚠️ Can't split <b>{_escape(entry.name)}</b> into fires:<br>{_escape(str(error))}",
                    card=True, error=True), ask=lambda: self.open_file(entry), ask_label="◀ Layers",
                    kind="Fire split")
                return None
            except Exception as error:
                self._show_preview(entry, self._error_html("Couldn't split", entry, error),
                                   ask=lambda: self.open_file(entry), ask_label="◀ Layers",
                                   kind="Fire split")
                return None
            self._last_split = (key, result)
        viewer = SplitViewer(result, on_size=self._wide, on_back=lambda: self.open_file(entry), mode=mode)
        self._show_preview(entry, viewer=viewer, kind="Fire split")
        self._set_status(result.summary())
        return result

    def is_fire_folder(self, location: Location) -> bool:
        if location[0] is None or not self.source.exists(location):
            return False
        return is_fire_folder([e.name for e in self.source.list(location) if e.is_dir])

    def close_preview(self) -> None:
        """Hide the preview pane."""
        self._preview.layout.display = "none"
        self._previewing = None
        self._set_viewer(None)
        self._wide(False)

    def on_file_click(self, callback: FileCallback) -> FileCallback:
        """Register `callback(entry)` to run when a file is clicked.

        Returns the callback, so this also works as a decorator.
        """
        self._callbacks.append(callback)
        return callback

    def navigate(self, location: Location, record: bool = True) -> bool:
        """Open a folder. Returns False (and stays put) if it can't be opened."""
        try:
            entries = self.source.list(location)
        except Exception as error:  # network, auth, missing path...
            self._set_status(f"⚠️ Couldn't open {self._label(location)}: {error}", error=True)
            return False
        if record and location != self.location:
            self._back.append(self.location)
            self._forward.clear()
        self.location = location
        self.entries = list(entries)
        self.selected = None
        self._limit = self.page_size
        self._render()
        return True

    def back(self) -> None:
        self._step(self._back, self._forward)

    def forward(self) -> None:
        self._step(self._forward, self._back)

    def up(self) -> None:
        parent = self._parent(self.location)
        if parent is not None:
            self.navigate(parent)

    def refresh(self) -> None:
        """Re-fetch names from the Hub: the current dataset, or all at the root."""
        repo_id = self.location[0]
        if repo_id is None:
            self.load()
            return
        self._set_status(f"Reloading {repo_id}…")
        try:
            self.source.load(repo_id)
        except Exception as error:
            self._set_status(f"⚠️ Couldn't reload {repo_id}: {error}", error=True)
            return
        target = self.location
        while target != ROOT and not self.source.exists(target):
            target = self._parent(target)  # the folder was deleted on the Hub
        self.navigate(target, record=False)

    @property
    def widget(self) -> widgets.VBox:
        """The whole explorer (window plus the output area under it)."""
        return self._root

    def show(self) -> None:
        """Display the explorer in the current notebook cell."""
        display(self._root)

    def __repr__(self) -> str:
        # Plain text on purpose: `open_explorer()` as a cell's last line already
        # displayed the window, and a rich repr would draw it a second time.
        return f"<DatasetExplorer {self._label(self.location)}>"

    # ----- layout -----------------------------------------------------------

    def _build(self, height: int) -> None:
        style = widgets.HTML(_CSS, layout=widgets.Layout(display="none"))
        self._title = widgets.HTML()

        self._back_btn = _button("◀", "fx-tool", tooltip="Back")
        self._fwd_btn = _button("▶", "fx-tool", tooltip="Forward")
        self._up_btn = _button("▲", "fx-tool", tooltip="Enclosing folder")
        self._refresh_btn = _button("⟳", "fx-tool", tooltip="Refresh")
        self._back_btn.on_click(lambda _: self.back())
        self._fwd_btn.on_click(lambda _: self.forward())
        self._up_btn.on_click(lambda _: self.up())
        self._refresh_btn.on_click(lambda _: self.refresh())
        self._crumb_box = widgets.HBox()
        self._crumb_box.add_class("fx-crumbs")
        toolbar = widgets.HBox(
            [self._back_btn, self._fwd_btn, self._up_btn, self._refresh_btn, self._crumb_box]
        )
        toolbar.add_class("fx-toolbar")

        self._sidebar = widgets.VBox(layout=widgets.Layout(height=f"{height + 27}px"))
        self._sidebar.add_class("fx-sidebar")

        header = widgets.HTML(
            '<div class="fx-head"><span class="fx-col-name">Name</span>'
            '<span class="fx-size">Size</span><span class="fx-kind">Kind</span>'
            '<span class="fx-local"></span></div>'
        )
        self._list = widgets.VBox(layout=widgets.Layout(height=f"{height}px"))
        self._list.add_class("fx-list")
        self._empty = widgets.HTML('<div class="fx-empty">This folder is empty.</div>')
        self._more_btn = _button("", "fx-more")
        self._more_btn.on_click(lambda _: self._show_more())
        main = widgets.VBox([header, self._list])
        main.add_class("fx-main")

        # Preview pane: opened files are drawn here, inside the window.
        self._preview_title = widgets.HTML()
        self._preview_title.add_class("fx-preview-title")
        close = _button("✕", "fx-tool", tooltip="Close preview")
        close.on_click(lambda _: self.close_preview())
        head = widgets.HBox([self._preview_title, close])
        head.add_class("fx-preview-head")
        self._preview_path = widgets.HTML()
        self._preview_path.add_class("fx-preview-path")
        self._preview_body = widgets.HTML()  # text, tables, messages
        self._preview_content = widgets.Box([self._preview_body])  # or an image viewer
        self._preview_content.add_class("fx-preview-body")
        self._download_btn = _button("Download", "fx-primary")
        self._download_btn.on_click(lambda _: self._ask_action and self._ask_action())
        self._preview_ask = widgets.HBox([self._download_btn])
        self._preview_ask.add_class("fx-preview-ask")
        # Shown above a window archive's viewer: split it into its fires.
        self._split_btn = _button("✂ Visualize split", "fx-primary",
                                  tooltip="Split this archive into its fires (fire-split-zones)")
        self._split_btn.on_click(lambda _: self._previewing and self.open_split(self._previewing))
        self._preview_actions = widgets.HBox([self._split_btn], layout=widgets.Layout(display="none"))
        self._preview_actions.add_class("fx-preview-actions")
        self._pane_height = f"{height + 27}px"
        self._preview = widgets.VBox(
            [head, self._preview_path, self._preview_actions, self._preview_content,
             self._preview_ask],
            layout=widgets.Layout(height=self._pane_height, display="none"),
        )
        self._preview.add_class("fx-preview")
        self._previewing: Optional[Entry] = None
        self._viewer = None  # a StackViewer or SplitViewer
        self._last_split = None  # ((path, scale_km), FireSplit) of the last split shown
        self._ask_action: Optional[Callable[[], object]] = None

        body = widgets.HBox([self._sidebar, main, self._preview])
        body.add_class("fx-body")
        self._status = widgets.HTML()
        self._status.add_class("fx-status")

        window = widgets.VBox([style, self._title, toolbar, body, self._status])
        window.add_class("fx-window")
        self.output = widgets.Output()
        self._root = widgets.VBox([window, self.output])

    def _load_sidebar(self) -> None:
        for button in self._side_buttons.values():
            button.close()
        self._side_buttons = {}

        home = _button(f"🏠 {self.source.owner}", "fx-side")
        home.on_click(lambda _: self.navigate(ROOT))
        self._side_buttons[None] = home
        label = widgets.HTML('<div class="fx-side-label">Datasets</div>')
        try:
            datasets = self.source.list(ROOT)
        except Exception:
            datasets = []  # the main list reports the error
        children = [home, label]
        for entry in datasets:
            button = _button(f"🗂️ {entry.name}", "fx-side", tooltip=entry.repo_id)
            button.on_click(lambda _, loc=entry.location: self.navigate(loc))
            self._side_buttons[entry.repo_id] = button
            children.append(button)
        self._sidebar.children = children

    # ----- rendering --------------------------------------------------------

    def _render(self) -> None:
        repo_id, _ = self.location
        title = self.location[1].rsplit("/", 1)[-1] or (
            repo_id.split("/", 1)[-1] if repo_id else self.source.owner
        )
        self._title.value = (
            '<div class="fx-title">'
            f'<span class="fx-title-text">{_escape(title)}</span></div>'
        )
        self._back_btn.disabled = not self._back
        self._fwd_btn.disabled = not self._forward
        self._up_btn.disabled = self.location == ROOT
        for key, button in self._side_buttons.items():
            _set_class(button, "fx-active", key == repo_id)
        self._render_crumbs()
        self._render_rows()
        self._update_status()

    def _render_crumbs(self) -> None:
        for button in self._crumbs:
            button.close()
        trail = [(f"🏠 {self.source.owner}", ROOT)]
        repo_id, path = self.location
        if repo_id is not None:
            trail.append((repo_id.split("/", 1)[-1], (repo_id, "")))
            parts = path.split("/") if path else []
            for i, part in enumerate(parts):
                trail.append((part, (repo_id, "/".join(parts[: i + 1]))))
        self._crumbs = []
        for i, (text, loc) in enumerate(trail):
            button = _button(text, "fx-crumb", tooltip=text)
            if i == len(trail) - 1:
                button.add_class("fx-current")
            button.on_click(lambda _, loc=loc: self.navigate(loc))
            self._crumbs.append(button)
        self._crumb_box.children = self._crumbs

    def _render_rows(self) -> None:
        shown = self.entries[: self._limit]
        while len(self._rows) < len(shown):
            self._rows.append(self._make_row(len(self._rows)))
        for i, entry in enumerate(shown):
            row, name, meta = self._rows[i]
            icon, kind = icon_and_kind(entry)
            name.description = f"{icon}  {entry.name}"
            name.tooltip = entry.path or entry.repo_id
            meta.value = self._meta_html(entry, kind)
            _set_class(row, "fx-selected", entry == self.selected)
        children = [row for row, _, _ in self._rows[: len(shown)]]
        hidden = len(self.entries) - len(shown)
        if hidden > 0:
            self._more_btn.description = f"Show {min(hidden, self.page_size)} more of {hidden}…"
            children.append(self._more_btn)
        if not self.entries:
            children.append(self._empty)
        self._list.children = children

    def _make_row(self, index: int):
        name = _button("", "fx-name")
        name.on_click(lambda _: self._click(index))
        meta = widgets.HTML()
        meta.add_class("fx-meta")
        row = widgets.HBox([name, meta])
        row.add_class("fx-row")
        return row, name, meta

    def _meta_html(self, entry: Entry, kind: Optional[str] = None) -> str:
        if kind is None:
            kind = icon_and_kind(entry)[1]
        saved = not entry.is_dir and self.source.is_local(entry)
        mark = '<span class="fx-local" title="Saved locally">✓</span>' if saved else (
            '<span class="fx-local"></span>'
        )
        return (
            f'<span class="fx-size">{format_size(entry.size)}</span>'
            f'<span class="fx-kind">{_escape(kind)}</span>{mark}'
        )

    def _refresh_row(self, entry: Entry) -> None:
        """Redraw one row's size/kind/saved mark (after a download)."""
        for i, shown in enumerate(self.entries[: min(self._limit, len(self._rows))]):
            if shown == entry:
                self._rows[i][2].value = self._meta_html(entry)

    def _update_status(self) -> None:
        count = len(self.entries)
        text = f"{count} item{'s' if count != 1 else ''}"
        total = sum(entry.size or 0 for entry in self.entries)
        if total:
            text += f", {format_size(total)}"
        if self.selected is not None:
            text += f"  ·  Selected: {self.selected.name} ({format_size(self.selected.size)})"
            if self.source.is_local(self.selected):
                text += ", saved locally"
        self._set_status(text)

    def _set_status(self, text: str, error: bool = False) -> None:
        self._status.value = _escape(text)
        _set_class(self._status, "fx-error", error)

    # ----- events -----------------------------------------------------------

    def _click(self, index: int) -> None:
        if index >= len(self.entries):
            return
        entry = self.entries[index]
        if entry.is_dir:
            if self.navigate(entry.location) and self.open_files and self.is_fire_folder(entry.location):
                self.open_fire(entry)
            return
        previous = self.selected
        self.selected = entry
        for i, (row, _, _) in enumerate(self._rows[: len(self.entries)]):
            if self.entries[i] in (previous, entry):
                _set_class(row, "fx-selected", self.entries[i] == entry)
        self._update_status()
        if self.open_files:
            self.open_file(entry)
        if self._callbacks:
            self.output.clear_output()
            with self.output:
                for callback in self._callbacks:
                    callback(entry)

    def _show_preview(self, entry: Entry, body: Optional[str] = None,
                      ask: Optional[Callable[[], object]] = None,
                      viewer=None, kind: Optional[str] = None,
                      split: bool = False, ask_label: str = "Download") -> None:
        """Fill the pane: a title, where the file lives, then HTML or a viewer.

        split: show the "✂ Visualize split" button (window archives).
        """
        icon, default_kind = icon_and_kind(entry)
        self._preview_title.value = (
            f"{icon} {_escape(entry.name)} <span>· {format_size(entry.size)} · "
            f"{_escape(kind or default_kind)}</span>"
        )
        where = self.source.local_path(entry)
        done = viewer is not None if entry.is_dir else self.source.is_local(entry)
        saved = "Saved at" if done else "Saves to"
        self._preview_path.value = f"{saved} <code>{_escape(str(where))}</code>"
        self._set_viewer(viewer)
        if viewer is None:
            self._preview_body.value = body or ""
        self._ask_action = ask
        self._download_btn.description = ask_label
        self._preview_ask.layout.display = "" if ask else "none"
        self._preview_actions.layout.display = "" if split else "none"
        self._previewing = entry
        self._preview.layout.display = ""

    def _show_viewer(self, entry: Entry, stack, kind: Optional[str] = None, split: bool = False) -> None:
        self._show_preview(entry, viewer=StackViewer(stack, on_size=self._wide), kind=kind, split=split)

    def _set_viewer(self, viewer) -> None:
        """Swap what the pane's body holds, closing the previous viewer's widgets."""
        if self._viewer is not None and self._viewer is not viewer:
            self._viewer.close()
            self._wide(False)
        self._viewer = viewer
        self._preview_content.children = [viewer.widget if viewer else self._preview_body]
        self._fit_pane()

    def _wide(self, on: bool) -> None:
        """Big viewer: the pane takes the window's full width."""
        _set_class(self._preview, "fx-preview-wide", on)
        self._fit_pane()

    def _fit_pane(self) -> None:
        """Text keeps the list's height and scrolls; an image viewer gets the room it needs."""
        grow = self._viewer is not None or "fx-preview-wide" in self._preview._dom_classes
        self._preview.layout.height = None if grow else self._pane_height

    def _entry_for(self, location: Location) -> Optional[Entry]:
        repo_id, path = location
        if repo_id is None:
            return None
        for entry in self.source.list(self._parent(location) or ROOT):
            if entry.location == location:
                return entry
        return None

    def _npz_dates(self, entry: Entry):
        """(window, day) -> date for a window archive (see fire.npz_date_lookup)."""
        return npz_date_lookup(self.source, entry)

    def _error_html(self, what: str, entry: Entry, error: Exception) -> str:
        self._set_status(f"⚠️ {what} {entry.name}: {error}", error=True)
        return viewers.note(
            f"⚠️ {what} <b>{_escape(entry.name)}</b>:<br>{_escape(str(error))}",
            card=True, error=True,
        )

    def _file(self, entry: Optional[Entry]) -> Entry:
        entry = entry if entry is not None else self.selected
        if entry is None:
            raise ValueError("No file selected")
        if entry.is_dir:
            raise IsADirectoryError(f"{entry.name} is a folder")
        return entry

    def _show_more(self) -> None:
        self._limit += self.page_size
        self._render_rows()

    # ----- helpers ----------------------------------------------------------

    def _step(self, source: List[Location], dest: List[Location]) -> None:
        """Move one step through history (source -> dest), undoing it on failure."""
        if not source:
            return
        current, target = self.location, source.pop()
        dest.append(current)  # before navigate() renders, so the buttons are right
        if not self.navigate(target, record=False):
            dest.pop()
            source.append(target)

    @staticmethod
    def _parent(location: Location) -> Optional[Location]:
        repo_id, path = location
        if repo_id is None:
            return None
        if not path:
            return ROOT
        return (repo_id, path.rsplit("/", 1)[0] if "/" in path else "")

    def _label(self, location: Location) -> str:
        repo_id, path = location
        if repo_id is None:
            return self.source.owner
        return f"{repo_id}/{path}" if path else repo_id


def open_explorer(
    root: str = DEFAULT_ROOT, on_file_click: Optional[FileCallback] = None, **options
) -> DatasetExplorer:
    """Show a dataset explorer in the notebook and return it.

    The window appears right away and fills in once every file and folder name
    has loaded (a few seconds for SamuelWu318's ~13,000 entries).
    """
    explorer = DatasetExplorer(root, on_file_click=on_file_click, autoload=False, **options)
    explorer.show()
    explorer.load()
    return explorer
