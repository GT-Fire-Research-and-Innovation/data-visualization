"""A stand-in for huggingface_hub.HfApi with a small fixed tree."""

from types import SimpleNamespace

TREE = {
    "me/alpha": {
        "": ["roi/", "train/", "README.md:6865"],
        "roi": ["roi/fire_all.csv:2048"],
        "train": [f"train/p{i}.npz:{1000 * i}" for i in range(1, 6)],
    },
    "me/beta": {"": []},
}


class FakeApi:
    def __init__(self, tree=TREE):
        self.tree = tree
        self.calls = []

    def list_datasets(self, author):
        self.calls.append(("datasets", author))
        return [SimpleNamespace(id=repo) for repo in reversed(list(self.tree))]

    def list_repo_tree(self, repo_id, path_in_repo=None, repo_type=None, revision=None,
                       recursive=False):
        self.calls.append(("tree", repo_id, path_in_repo))
        assert repo_type == "dataset" and not recursive
        try:
            items = self.tree[repo_id][path_in_repo or ""]
        except KeyError:
            raise FileNotFoundError(f"{repo_id}/{path_in_repo} not found")
        for item in items:
            if item.endswith("/"):
                yield SimpleNamespace(path=item[:-1])  # like RepoFolder: no size
            else:
                path, size = item.rsplit(":", 1)
                yield SimpleNamespace(path=path, size=int(size))
