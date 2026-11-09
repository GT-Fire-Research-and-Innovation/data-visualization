import pytest

from firevis.hub import DATASET, FILE, FOLDER, ROOT, Entry, HubSource, build_tree, parse_owner

from fakes import FakeApi


@pytest.mark.parametrize("root", [
    "SamuelWu318",
    "https://huggingface.co/SamuelWu318",
    "https://huggingface.co/SamuelWu318/",
    "huggingface.co/SamuelWu318",
    "https://huggingface.co/datasets/SamuelWu318",
    "https://huggingface.co/datasets/SamuelWu318/ts-satfire",
])
def test_parse_owner(root):
    assert parse_owner(root) == "SamuelWu318"


def test_parse_owner_rejects_bare_site():
    with pytest.raises(ValueError):
        parse_owner("https://huggingface.co/")


def test_build_tree_fills_in_folders_and_sizes():
    items = [type("F", (), {"path": p, "size": s})() for p, s in
             [("a/b/c.npz", 10), ("a/d.csv", 5), ("e.md", 1)]]
    tree, total = build_tree("me/x", items)
    assert total == 16
    assert [(e.name, e.kind, e.size) for e in tree[("me/x", "")]] == [
        ("a", FOLDER, 15), ("e.md", FILE, 1)]
    assert [(e.name, e.size) for e in tree[("me/x", "a")]] == [("b", 10), ("d.csv", 5)]
    assert tree[("me/x", "a/b")] == [Entry("c.npz", FILE, "me/x", "a/b/c.npz", 10)]


def test_everything_loads_once_up_front():
    api = FakeApi()
    source = HubSource("me", api=api)
    root = source.list(ROOT)  # first use loads all datasets
    assert [e.name for e in root] == ["alpha", "beta", "broken"]
    assert all(e.kind == DATASET for e in root)
    assert sorted(c for c in api.calls if c[0] == "tree") == [
        ("tree", "me/alpha"), ("tree", "me/beta"), ("tree", "me/broken")]
    calls = len(api.calls)
    source.list(("me/alpha", "train"))
    source.list(("me/alpha", "deep/a"))
    assert len(api.calls) == calls  # browsing needs no more requests


def test_listing_contents():
    source = HubSource("me", api=FakeApi())
    assert [(e.name, e.kind, e.size) for e in source.list(("me/alpha", ""))] == [
        ("deep", FOLDER, 5), ("roi", FOLDER, 2048), ("train", FOLDER, 15000),
        ("README.md", FILE, 6865)]
    assert source.list(ROOT)[0].size == 2048 + 15000 + 6865 + 5
    assert source.list(("me/beta", "")) == []


def test_broken_dataset_is_listed_but_cannot_open():
    source = HubSource("me", api=FakeApi())
    broken = source.list(ROOT)[2]
    assert broken.name == "broken" and broken.size is None
    with pytest.raises(PermissionError):
        source.list(broken.location)
    with pytest.raises(FileNotFoundError):
        source.list(("me/alpha", "nope"))


def test_reload_one_dataset():
    api = FakeApi()
    source = HubSource("me", api=api)
    source.load()
    api.tree["me/beta"] = ["new.csv:7"]
    api.calls.clear()
    source.load("me/beta")
    assert api.calls == [("tree", "me/beta")]
    assert [e.name for e in source.list(("me/beta", ""))] == ["new.csv"]
    assert source.list(ROOT)[1].size == 7
    assert source.exists(("me/alpha", "train"))  # other datasets untouched


def test_download_is_lazy_and_files_only():
    api = FakeApi()
    source = HubSource("me", api=api)
    readme = source.list(("me/alpha", ""))[-1]
    assert not any(c[0] == "download" for c in api.calls)
    assert source.download(readme) == "/cache/me/alpha/README.md"
    assert api.calls[-1] == ("download", "me/alpha", "README.md")
    with pytest.raises(IsADirectoryError):
        source.download(source.list(("me/alpha", ""))[0])


def test_entry_urls():
    assert Entry("alpha", DATASET, "me/alpha").url == "https://huggingface.co/datasets/me/alpha"
    assert Entry("roi", FOLDER, "me/alpha", "roi").url.endswith("/tree/main/roi")
    assert Entry("a.csv", FILE, "me/alpha", "roi/a.csv", 3).url.endswith("/blob/main/roi/a.csv")
