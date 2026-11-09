"""Read the Hugging Face Hub as a folder tree.

The root is a Hub user (or org). Its datasets are the top-level folders, and
inside a dataset the repo's own directories and files are listed one level at a
time, so opening a folder only fetches that folder.
"""

from __future__ import annotations

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
    size: Optional[int] = None  # bytes, files only

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


def _sort_key(entry: Entry):
    return (not entry.is_dir, entry.name.lower())


class HubSource:
    """Lists datasets and their contents for one Hub owner, with caching.

    `api` defaults to `huggingface_hub.HfApi()`, which picks up a token from
    `HF_TOKEN` or a prior `huggingface_hub.login()` (needed for private repos).
    """

    def __init__(self, root: str = DEFAULT_ROOT, api=None, revision: str = "main"):
        self.owner = parse_owner(root)
        self.revision = revision
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        self.api = api
        self._cache: Dict[Location, List[Entry]] = {}

    def list(self, location: Location) -> List[Entry]:
        """Entries directly inside `location`, folders first, then by name."""
        if location not in self._cache:
            repo_id, path = location
            if repo_id is None:
                entries = self._list_datasets()
            else:
                entries = self._list_tree(repo_id, path)
            self._cache[location] = sorted(entries, key=_sort_key)
        return self._cache[location]

    def refresh(self, location: Optional[Location] = None) -> None:
        """Forget cached listings (one location, or all of them)."""
        if location is None:
            self._cache.clear()
        else:
            self._cache.pop(location, None)

    def _list_datasets(self) -> List[Entry]:
        return [
            Entry(name=d.id.split("/", 1)[-1], kind=DATASET, repo_id=d.id)
            for d in self.api.list_datasets(author=self.owner)
        ]

    def _list_tree(self, repo_id: str, path: str) -> List[Entry]:
        items = self.api.list_repo_tree(
            repo_id,
            path_in_repo=path or None,
            repo_type="dataset",
            revision=self.revision,
            recursive=False,
        )
        entries = []
        for item in items:
            size = getattr(item, "size", None)
            # RepoFolder has no size; RepoFile always has one.
            kind = FILE if size is not None else FOLDER
            entries.append(
                Entry(
                    name=item.path.rsplit("/", 1)[-1],
                    kind=kind,
                    repo_id=repo_id,
                    path=item.path,
                    size=size,
                )
            )
        return entries
