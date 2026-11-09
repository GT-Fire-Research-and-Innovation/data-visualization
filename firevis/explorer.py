"""A Finder / File Explorer style browser for Hub datasets, built on ipywidgets.

    from firevis import open_explorer
    explorer = open_explorer()          # root: https://huggingface.co/SamuelWu318

Clicking a folder opens it. Clicking a file selects it and calls every function
registered with `explorer.on_file_click(fn)`; `fn` gets the file's `Entry`.
Anything those functions print or plot shows up under the explorer window.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional

import ipywidgets as widgets
from IPython.display import display

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
  max-width: 980px; background: #fff; border: 1px solid var(--fx-line);
  border-radius: 10px; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,.10);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 13px; color: var(--fx-text);
}
.fx-window .jupyter-widgets { margin: 0; }
.fx-window .widget-html, .fx-window .widget-html-content {
  height: auto; line-height: normal; min-width: 0;
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
.fx-lights { display: flex; gap: 8px; width: 60px; }
.fx-lights i { width: 12px; height: 12px; border-radius: 50%; display: block; }
.fx-lights i:nth-child(1) { background: #ff5f57; }
.fx-lights i:nth-child(2) { background: #febc2e; }
.fx-lights i:nth-child(3) { background: #28c840; }
.fx-title-text { flex: 1; text-align: center; font-weight: 600; margin-right: 60px;
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
.fx-meta { flex: 0 0 210px; line-height: 26px; }
.fx-row.fx-selected { background: var(--fx-accent) !important; }
.fx-row.fx-selected .jupyter-button, .fx-row.fx-selected .fx-size,
.fx-row.fx-selected .fx-kind { color: #fff !important; }
.fx-empty { color: var(--fx-muted); text-align: center; padding: 40px 0; }
.fx-window .jupyter-button.fx-more { width: auto; color: var(--fx-accent) !important;
  margin: 6px auto !important; padding: 0 10px; }

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

    root:          Hub link or user/org name whose datasets form the top level.
    source:        Anything with `owner`, `list(location)` and `refresh(location)`;
                   defaults to `HubSource(root)`. Swap in a fake for tests.
    on_file_click: Optional function called with the `Entry` of a clicked file.
    height:        Height of the list area, in pixels.
    page_size:     Rows shown before a "Show more" button appears.
    """

    def __init__(
        self,
        root: str = DEFAULT_ROOT,
        *,
        source=None,
        on_file_click: Optional[FileCallback] = None,
        height: int = 420,
        page_size: int = 200,
    ):
        self.source = source if source is not None else HubSource(root)
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
        self._load_sidebar()
        self.navigate(ROOT, record=False)

    # ----- public API -------------------------------------------------------

    def on_file_click(self, callback: FileCallback) -> FileCallback:
        """Register `callback(entry)` to run when a file is clicked.

        Returns the callback, so this also works as a decorator.
        """
        self._callbacks.append(callback)
        return callback

    def navigate(self, location: Location, record: bool = True) -> bool:
        """Open a folder. Returns False (and stays put) if listing it fails."""
        self._set_status("Loading…")
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
        """Re-fetch the current folder from the Hub."""
        self.source.refresh(self.location)
        if self.location == ROOT:
            self._load_sidebar()
        self.navigate(self.location, record=False)

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
            '<span class="fx-size">Size</span><span class="fx-kind">Kind</span></div>'
        )
        self._list = widgets.VBox(layout=widgets.Layout(height=f"{height}px"))
        self._list.add_class("fx-list")
        self._empty = widgets.HTML('<div class="fx-empty">This folder is empty.</div>')
        self._more_btn = _button("", "fx-more")
        self._more_btn.on_click(lambda _: self._show_more())
        main = widgets.VBox([header, self._list])
        main.add_class("fx-main")

        body = widgets.HBox([self._sidebar, main])
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
            '<div class="fx-title"><span class="fx-lights"><i></i><i></i><i></i></span>'
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
            meta.value = (
                f'<span class="fx-size">{format_size(entry.size)}</span>'
                f'<span class="fx-kind">{_escape(kind)}</span>'
            )
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

    def _update_status(self) -> None:
        count = len(self.entries)
        text = f"{count} item{'s' if count != 1 else ''}"
        if self.selected is not None:
            text += f"  ·  Selected: {self.selected.name} ({format_size(self.selected.size)})"
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
            self.navigate(entry.location)
            return
        previous = self.selected
        self.selected = entry
        for i, (row, _, _) in enumerate(self._rows[: len(self.entries)]):
            if self.entries[i] in (previous, entry):
                _set_class(row, "fx-selected", self.entries[i] == entry)
        self._update_status()
        with self.output:
            for callback in self._callbacks:
                callback(entry)

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
    """Show a dataset explorer in the notebook and return it."""
    explorer = DatasetExplorer(root, on_file_click=on_file_click, **options)
    explorer.show()
    return explorer
