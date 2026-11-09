"""Read the Hugging Face Hub as a folder tree.

The root is a Hub user (or org) and its datasets are the top-level folders.
Every file and folder name is fetched up front (one recursive listing per
dataset, run in parallel), so moving around the tree needs no network. File
contents are the lazy part: `HubSource.download(entry)` fetches a file only
when it is asked for, and the Hub cache keeps it for next time.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

DEFAULT_ROOT = "https://huggingface.co/SamuelWu318"

DATASET = "dataset"
FOLDER = "folder"
FILE = "file"

# A place in the tree: (dataset repo id, path inside it). The root is (None, "").
Location = Tuple[Optional[str], str]
ROOT: Location = (None, "")


@dataclass(frozen=True)
class Entry:
    """One row in the explorer: a dataset, a folder, or a file."""

    name: str
    kind: str  # DATASET, FOLDER or FILE
    repo_id: str
    path: str = ""  # path inside the dataset repo; "" for the dataset itself
    size: Optional[int] = None  # bytes; for folders and datasets, everything inside

    @property
    def is_dir(self) -> bool:
        return self.kind != FILE

    @property
    def location(self) -> Location:
        return (self.repo_id, self.path)

    @property
    def url(self) -> str:
        """Link to this entry on the Hub website."""
        base = f"https://huggingface.co/datasets/{self.repo_id}"
        if self.kind == DATASET:
            return base
        return f"{base}/{'tree' if self.is_dir else 'blob'}/main/{self.path}"


def parse_owner(root: str) -> str:
    """Turn a Hub link or bare name into the owner name.

    Accepts "SamuelWu318", "https://huggingface.co/SamuelWu318" and
    "https://huggingface.co/datasets/SamuelWu318".
    """
    text = root.strip()
    if "://" in text or text.startswith("huggingface.co"):
        if "://" not in text:
            text = "https://" + text
        parts = [p for p in urlparse(text).path.split("/") if p]
        if parts and parts[0] in ("datasets", "models", "spaces"):
            parts = parts[1:]
        if not parts:
            raise ValueError(f"No user or org name in {root!r}")
        return parts[0]
    return text.strip("/").split("/")[0]


def _parent(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _ancestors(path: str):
    """Folders above `path`, nearest first: "a/b/c.npz" -> "a/b", "a"."""
    while "/" in path:
        path = path.rsplit("/", 1)[0]
        yield path


def _sort_key(entry: Entry):
    return (not entry.is_dir, entry.name.lower())


def build_tree(repo_id: str, items) -> Tuple[Dict[Location, List[Entry]], int]:
    """Turn a flat recursive listing into {folder location: sorted children}.

    `items` are RepoFile / RepoFolder objects (anything with `.path`, plus
    `.size` for files). Folders implied by file paths are filled in. Returns the
    tree and the dataset's total size in bytes.
    """
    files: Dict[str, int] = {}
    folders = set()
    for item in items:
        size = getattr(item, "size", None)  # RepoFolder has no size
        if size is None:
            folders.add(item.path)
        else:
            files[item.path] = size
    for path in list(files) + list(folders):
        folders.update(_ancestors(path))

    totals = dict.fromkeys(folders, 0)
    for path, size in files.items():
        for folder in _ancestors(path):
            totals[folder] += size

    tree: Dict[Location, List[Entry]] = {(repo_id, ""): []}
    tree.update({(repo_id, folder): [] for folder in folders})
    for folder in folders:
        name = folder.rsplit("/", 1)[-1]
        tree[(repo_id, _parent(folder))].append(
            Entry(name, FOLDER, repo_id, folder, totals[folder])
        )
    for path, size in files.items():
        name = path.rsplit("/", 1)[-1]
        tree[(repo_id, _parent(path))].append(Entry(name, FILE, repo_id, path, size))
    for children in tree.values():
        children.sort(key=_sort_key)
    return tree, sum(files.values())


class HubSource:
    """All file and folder names for one Hub owner's datasets, held in memory.

    The first `list()` loads everything; `load()` re-fetches it all, or one
    dataset with `load(repo_id)`. `api` defaults to `huggingface_hub.HfApi()`,
    which uses a token from `HF_TOKEN` or `huggingface_hub.login()` if present
    (needed for private datasets, and for downloading gated ones).
    """

    def __init__(self, root: str = DEFAULT_ROOT, api=None, revision: str = "main",
                 workers: int = 8):
        self.owner = parse_owner(root)
        self.revision = revision
        self.workers = workers
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        self.api = api
        self.loaded = False
        self._repo_ids: List[str] = []
        self._tree: Dict[Location, List[Entry]] = {}
        self._totals: Dict[str, Optional[int]] = {}
        self._errors: Dict[str, Exception] = {}

    # ----- names (loaded up front) ------------------------------------------

    def load(self, repo_id: Optional[str] = None) -> None:
        """Fetch every dataset's file and folder names (or just one dataset's)."""
        if repo_id is None:
            repo_ids = [d.id for d in self.api.list_datasets(author=self.owner)]
            self._repo_ids = sorted(repo_ids, key=lambda r: r.split("/", 1)[-1].lower())
            self._tree, self._totals, self._errors = {}, {}, {}
            with ThreadPoolExecutor(max(1, min(self.workers, len(repo_ids)))) as pool:
                for rid, result in zip(repo_ids, pool.map(self._fetch, repo_ids)):
                    self._store(rid, result)
        else:
            self._tree = {loc: v for loc, v in self._tree.items() if loc[0] != repo_id}
            self._errors.pop(repo_id, None)
            self._store(repo_id, self._fetch(repo_id))
        self._tree[ROOT] = [
            Entry(rid.split("/", 1)[-1], DATASET, rid, "", self._totals.get(rid))
            for rid in self._repo_ids
        ]
        self.loaded = True

    def list(self, location: Location) -> List[Entry]:
        """Entries directly inside `location`, folders first, then by name."""
        if not self.loaded:
            self.load()
        repo_id, path = location
        if repo_id in self._errors:
            raise self._errors[repo_id]
        try:
            return self._tree[location]
        except KeyError:
            where = f"{repo_id}/{path}" if path else str(repo_id)
            raise FileNotFoundError(f"No folder {where}") from None

    def exists(self, location: Location) -> bool:
        return location in self._tree

    def _fetch(self, repo_id: str):
        try:
            items = self.api.list_repo_tree(
                repo_id, repo_type="dataset", revision=self.revision, recursive=True
            )
            return build_tree(repo_id, items)
        except Exception as error:  # one bad dataset shouldn't hide the others
            return error

    def _store(self, repo_id: str, result) -> None:
        if isinstance(result, Exception):
            self._errors[repo_id] = result
            self._totals[repo_id] = None
        else:
            tree, total = result
            self._tree.update(tree)
            self._totals[repo_id] = total

    # ----- file contents (lazy) ---------------------------------------------

    def download(self, entry: Entry) -> str:
        """Local path to a file's contents, downloading it only the first time.

        Uses the Hugging Face cache (~/.cache/huggingface), so a file already
        fetched in this runtime comes straight from disk.
        """
        if entry.kind != FILE:
            raise IsADirectoryError(f"{entry.repo_id}/{entry.path} is a folder")
        return self.api.hf_hub_download(
            entry.repo_id, entry.path, repo_type="dataset", revision=self.revision
        )
