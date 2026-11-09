import pytest

from firevis.explorer import DatasetExplorer, format_size, icon_and_kind
from firevis.hub import FILE, ROOT, Entry, HubSource

from fakes import FakeApi


@pytest.fixture
def explorer():
    return DatasetExplorer(source=HubSource("me", api=FakeApi()), page_size=3)


def names(explorer):
    """Names of the visible rows, as the user sees them."""
    rows = [w for w in explorer._list.children if "fx-row" in w._dom_classes]
    return [row.children[0].description.split("  ", 1)[1] for row in rows]


def click(explorer, name):
    for row in explorer._list.children:
        if "fx-row" in row._dom_classes and row.children[0].description.endswith(name):
            row.children[0].click()
            return
    raise AssertionError(f"no row named {name}")


def test_starts_at_root_with_datasets(explorer):
    assert explorer.location == ROOT
    assert names(explorer) == ["alpha", "beta"]
    assert explorer._up_btn.disabled and explorer._back_btn.disabled
    assert explorer._status.value == "2 items"
    assert [b.description for b in explorer._crumbs] == ["🏠 me"]


def test_clicking_folders_opens_them(explorer):
    click(explorer, "alpha")
    assert explorer.location == ("me/alpha", "")
    assert names(explorer) == ["roi", "train", "README.md"]
    click(explorer, "roi")
    assert explorer.location == ("me/alpha", "roi")
    assert [b.description for b in explorer._crumbs] == ["🏠 me", "alpha", "roi"]
    assert "fx-active" in explorer._side_buttons["me/alpha"]._dom_classes
    assert "fx-active" not in explorer._side_buttons[None]._dom_classes


def test_back_forward_and_up(explorer):
    click(explorer, "alpha")
    click(explorer, "roi")
    explorer._back_btn.click()
    assert explorer.location == ("me/alpha", "")
    assert not explorer._fwd_btn.disabled
    explorer._fwd_btn.click()
    assert explorer.location == ("me/alpha", "roi")
    assert explorer._fwd_btn.disabled
    explorer._up_btn.click()
    assert explorer.location == ("me/alpha", "")
    explorer._up_btn.click()
    assert explorer.location == ROOT
    assert explorer._up_btn.disabled


def test_breadcrumb_jumps_back(explorer):
    click(explorer, "alpha")
    click(explorer, "roi")
    explorer._crumbs[0].click()
    assert explorer.location == ROOT


def test_clicking_a_file_selects_it_and_calls_back(explorer):
    clicked = []
    explorer.on_file_click(clicked.append)
    click(explorer, "alpha")
    click(explorer, "README.md")
    assert explorer.location == ("me/alpha", "")  # files don't navigate
    assert explorer.selected == clicked[0] == Entry("README.md", FILE, "me/alpha", "README.md", 6865)
    selected = [r for r in explorer._list.children if "fx-selected" in r._dom_classes]
    assert len(selected) == 1 and selected[0].children[0].description.endswith("README.md")
    assert "Selected: README.md (6.9 KB)" in explorer._status.value


def test_rows_are_reused_and_selection_cleared(explorer):
    click(explorer, "alpha")
    click(explorer, "README.md")
    pool = len(explorer._rows)
    explorer._up_btn.click()
    assert explorer.selected is None
    assert len(explorer._rows) == pool
    assert not any("fx-selected" in r._dom_classes for r in explorer._list.children)


def test_long_folders_page(explorer):
    click(explorer, "alpha")
    click(explorer, "train")
    assert names(explorer) == ["p1.npz", "p2.npz", "p3.npz"]
    assert explorer._more_btn in explorer._list.children
    assert explorer._more_btn.description == "Show 2 more of 2…"
    explorer._more_btn.click()
    assert len(names(explorer)) == 5
    assert explorer._more_btn not in explorer._list.children


def test_empty_folder(explorer):
    click(explorer, "beta")
    assert names(explorer) == []
    assert explorer._empty in explorer._list.children
    assert explorer._status.value == "0 items"


def test_failed_listing_keeps_place(explorer):
    click(explorer, "alpha")
    assert explorer.navigate(("me/alpha", "missing")) is False
    assert explorer.location == ("me/alpha", "")
    assert "fx-error" in explorer._status._dom_classes
    click(explorer, "roi")
    assert "fx-error" not in explorer._status._dom_classes


def test_format_size():
    assert format_size(None) == "—"
    assert format_size(1) == "1 byte"
    assert format_size(999) == "999 bytes"
    assert format_size(6865) == "6.9 KB"
    assert format_size(1_779_597_989) == "1.8 GB"


def test_kinds():
    assert icon_and_kind(Entry("x.npz", FILE, "r", "x.npz", 1))[1] == "NumPy archive"
    assert icon_and_kind(Entry("x.tif", FILE, "r", "x.tif", 1))[1] == "GeoTIFF image"
    assert icon_and_kind(Entry("x.weird", FILE, "r", "x.weird", 1))[1] == "WEIRD file"
    assert icon_and_kind(Entry(".gitattributes", FILE, "r", ".gitattributes", 1))[1] == "File"
