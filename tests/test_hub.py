import pytest

from firevis.hub import DATASET, FILE, FOLDER, ROOT, Entry, HubSource, parse_owner

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


def test_root_lists_datasets_by_name():
    source = HubSource("me", api=FakeApi())
    entries = source.list(ROOT)
    assert [e.name for e in entries] == ["alpha", "beta"]
    assert all(e.kind == DATASET and e.repo_id.startswith("me/") for e in entries)


def test_tree_lists_folders_first_with_sizes():
    api = FakeApi()
    entries = HubSource("me", api=api).list(("me/alpha", ""))
    assert [(e.name, e.kind, e.size) for e in entries] == [
        ("roi", FOLDER, None),
        ("train", FOLDER, None),
        ("README.md", FILE, 6865),
    ]
    assert api.calls[-1] == ("tree", "me/alpha", None)  # repo root passes no path
    assert entries[1].location == ("me/alpha", "train")


def test_listings_are_cached_until_refresh():
    api = FakeApi()
    source = HubSource("me", api=api)
    source.list(("me/alpha", "roi"))
    source.list(("me/alpha", "roi"))
    assert len(api.calls) == 1
    source.refresh(("me/alpha", "roi"))
    source.list(("me/alpha", "roi"))
    assert len(api.calls) == 2


def test_entry_urls():
    assert Entry("alpha", DATASET, "me/alpha").url == "https://huggingface.co/datasets/me/alpha"
    assert Entry("roi", FOLDER, "me/alpha", "roi").url.endswith("/tree/main/roi")
    assert Entry("a.csv", FILE, "me/alpha", "roi/a.csv", 3).url.endswith("/blob/main/roi/a.csv")
