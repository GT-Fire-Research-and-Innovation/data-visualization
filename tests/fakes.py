"""A stand-in for huggingface_hub.HfApi with a small fixed set of datasets."""

from pathlib import Path
from types import SimpleNamespace

# Flat recursive listings, like list_repo_tree(recursive=True). "x/" is a folder,
# "path:size" a file. "deep/a" is never listed itself: it is implied by its file.
TREE = {
    "me/alpha": ["roi/", "roi/fire_all.csv:2048", "train/"]
    + [f"train/p{i}.npz:{1000 * i}" for i in range(1, 6)]
    + ["README.md:6865", "deep/a/b.txt:5"],
    "me/beta": [],
    "me/broken": PermissionError("401 Unauthorized"),
}

# What downloads contain, padded with newlines to the listed size. Other files
# get zero bytes.
CONTENTS = {
    "README.md": b"# Alpha\n\nFire data.\n",
    "roi/fire_all.csv": b"id,year\n1,2017\n2,2018\n",
    "deep/a/b.txt": b"hello",
}


class FakeApi:
    def __init__(self, tree=TREE, contents=CONTENTS):
        self.tree = dict(tree)
        self.contents = dict(contents)
        self.calls = []
        self.download_error = None  # set to an exception to make downloads fail

    def list_datasets(self, author):
        self.calls.append(("datasets", author))
        return [SimpleNamespace(id=repo) for repo in reversed(list(self.tree))]

    def list_repo_tree(self, repo_id, repo_type=None, revision=None, recursive=False,
                       path_in_repo=None):
        self.calls.append(("tree", repo_id))
        assert repo_type == "dataset" and recursive and path_in_repo is None
        items = self.tree[repo_id]
        if isinstance(items, Exception):
            raise items
        for item in items:
            if item.endswith("/"):
                yield SimpleNamespace(path=item[:-1])  # like RepoFolder: no size
            else:
                path, size = item.rsplit(":", 1)
                yield SimpleNamespace(path=path, size=int(size))

    def size_of(self, repo_id, filename):
        for item in self.tree[repo_id]:
            if item.startswith(filename + ":"):
                return int(item.rsplit(":", 1)[1])
        raise FileNotFoundError(filename)

    def hf_hub_download(self, repo_id, filename, repo_type=None, revision=None, local_dir=None):
        self.calls.append(("download", repo_id, filename))
        assert repo_type == "dataset" and local_dir
        if self.download_error is not None:
            raise self.download_error
        size = self.size_of(repo_id, filename)
        data = self.contents.get(filename, b"").ljust(size, b"\n" if filename in self.contents else b"\0")
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data[:size])
        return str(target)
