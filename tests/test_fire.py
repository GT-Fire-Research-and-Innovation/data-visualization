"""firevis.load / FireData / Comparison: links, channels, plots, side by side."""

import json

import numpy as np
import pytest

import firevis
from firevis import ChannelError, Comparison, FireData, HubSource, parse_link
from firevis.hub import FOLDER, Entry

from fakes import FakeApi
from imagery import DAYS, fire_folder, window_npz

pytest.importorskip("rasterio")
plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def api():
    files = fire_folder()
    npz = window_npz(windows=2, days=2)
    config = json.dumps({"ts_length": 2, "ts_interval": 1}).encode()
    contents = dict(files, **{"val/p42.npz": npz, "window_config.json": config})
    tree = {
        "me/ts-satfire": [f"{path}:{len(data)}" for path, data in files.items()],
        "me/windows": [f"val/p42.npz:{len(npz)}", f"window_config.json:{len(config)}"],
        "me/other": ["README.md:3"],
    }
    return FakeApi(tree=tree, contents=contents)


@pytest.fixture
def source(api, mirror):
    return HubSource("me", api=api, mirror_dir=mirror)


@pytest.fixture
def fire(source):
    return firevis.load("https://huggingface.co/datasets/me/ts-satfire/tree/main/ts-satfire/42",
                        source=source)


@pytest.fixture
def win(source):
    return firevis.load("me/windows/val/p42.npz", source=source)


def downloads(api):
    return [c[2] for c in api.calls if c[0] == "download"]


# ----- links -------------------------------------------------------------------


@pytest.mark.parametrize("link, expected", [
    ("https://huggingface.co/datasets/me/ts-satfire/tree/main/ts-satfire/42",
     ("me/ts-satfire", "ts-satfire/42", "main")),
    ("https://huggingface.co/datasets/me/w/blob/v2/val/p42.npz?download=true",
     ("me/w", "val/p42.npz", "v2")),
    ("huggingface.co/datasets/me/w/resolve/main/val/p42.npz", ("me/w", "val/p42.npz", "main")),
    ("hf://datasets/me/w/val/p42.npz", ("me/w", "val/p42.npz", "main")),
    ("datasets/me/w/val/p42.npz", ("me/w", "val/p42.npz", "main")),
    ("me/w/val/p%2042.npz", ("me/w", "val/p 42.npz", "main")),
    (Entry("42", FOLDER, "me/ts-satfire", "ts-satfire/42"), ("me/ts-satfire", "ts-satfire/42", "main")),
])
def test_parse_link(link, expected):
    assert parse_link(link) == expected


def test_parse_link_needs_a_dataset():
    with pytest.raises(ValueError):
        parse_link("https://huggingface.co/me")


# ----- loading -----------------------------------------------------------------


def test_load_fire_folder_downloads_once(fire, api, source):
    assert isinstance(fire, FireData) and fire.kind == "fire folder" and fire.name == "42"
    assert len(downloads(api)) == 10
    assert ("datasets", "me") not in api.calls  # only the one dataset was listed
    assert fire.path.is_dir() and fire.link.endswith("/tree/main/ts-satfire/42")
    again = firevis.load("me/ts-satfire/ts-satfire/42", source=source)
    assert len(downloads(api)) == 10 and again.channels() == fire.channels()


def test_load_npz_with_dates_from_its_fire(win):
    assert win.kind == "window archive"
    assert win.windows == ["Window 1: 2020-07-01 → 2020-07-02", "Window 2: 2020-07-02 → 2020-07-03"]
    assert win.dates("I1") == DAYS[:2]
    win.window = 1
    assert win.dates("I1") == DAYS[1:]
    with pytest.raises(IndexError):
        win.window = 5


def test_load_errors(source):
    with pytest.raises(ValueError, match="isn't a fire folder"):
        firevis.load("me/ts-satfire/ts-satfire/42/VIIRS_Day", source=source)
    with pytest.raises(FileNotFoundError, match="Did you mean: 42"):
        firevis.load("me/ts-satfire/ts-satfire/43", source=source)
    with pytest.raises(ValueError, match="whole dataset"):
        firevis.load("me/ts-satfire", source=source)
    with pytest.raises(ValueError, match="fire folder or an .npz"):
        firevis.load("me/other/README.md", source=source)


def test_load_local_paths(fire, win):
    local = firevis.load(fire.path)
    assert local.kind == "fire folder" and local.channels() == fire.channels()
    local_npz = firevis.load(str(win.path))
    assert local_npz.kind == "window archive" and local_npz.dates("I1") == ["Day 1", "Day 2"]


# ----- channels and dates ------------------------------------------------------


def test_channels_dates_and_summary(fire, win):
    assert fire.channels()[:3] == ["Active fire", "NIFC perimeter", "Burned area (active fire + NIFC)"]
    assert fire.channels("firepred") == ["NDVI_last", "wind speed", "land cover (MODIS)"]
    assert "VIIRS night" in fire.groups()
    assert fire.dates("I4") == DAYS and fire.dates() == DAYS
    assert "False colour (I3, I2, I1)" in win.channels()
    text = repr(fire)
    assert "42 · fire folder · 3 date(s) 2020-07-01 → 2020-07-03" in text and "FirePred (3)" in text
    assert ".window = 0" in repr(win)
    with pytest.raises(ChannelError, match="No group"):
        fire.channels("nope")


def test_channel_names_resolve_across_raw_and_npz(fire, win):
    assert win.layer("I4").name == "VIIRS I4 (running max)"
    assert win.layer("i4 night").name == "VIIRS I4 night (running max)"
    assert fire.layer("VIIRS I4 (running max)").name == "I4"
    assert fire.layer("land cover").name == "land cover (MODIS)"
    assert fire.layer("FirePred · wind speed").name == "wind speed"
    assert fire.layer("esri").name == "land cover (ESRI)"  # unique part of a name
    with pytest.raises(ChannelError, match="several channels"):
        fire.layer("I")
    with pytest.raises(ChannelError, match="Did you mean: wind speed"):
        fire.layer("wind sped")


# ----- plots -------------------------------------------------------------------


def test_show_values_has_a_colorbar(fire):
    ax = fire.show("I4", "2020-07-02")
    assert ax.get_title() == "42 · I4 · 2020-07-02"
    assert len(ax.figure.axes) == 2  # the image and its colorbar
    assert fire.show("I4", -1).get_title().endswith("2020-07-03")


def test_show_masks_and_classes_have_legends(fire):
    ax = fire.show("Active fire", 1)
    assert ax.get_legend().get_texts()[0].get_text() == "Active fire · 4 px"
    ax = fire.show("land cover (MODIS)")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Open shrubland", "Grassland"]


def test_bad_dates(fire):
    with pytest.raises(ValueError, match="no date '2021-01-01'"):
        fire.show("I4", "2021-01-01")
    with pytest.raises(IndexError):
        fire.show("I4", 9)


def test_overlay(fire):
    ax = fire.overlay("Active fire")
    assert ax.get_title() == "42 · Active fire · 2020-07-01 → 2020-07-03"
    colorbar = ax.figure.axes[1]
    assert [t.get_text() for t in colorbar.get_yticklabels()] == ["2020-07-01", "2020-07-03"]
    with pytest.raises(ValueError, match="Can't overlay"):
        fire.overlay("False colour")


def test_overlay_empty_layer_says_so(fire, monkeypatch):
    layer = fire.layer("NIFC perimeter")
    monkeypatch.setattr(layer, "read", lambda i: np.zeros((32, 32), bool))
    ax = fire.overlay("NIFC perimeter")
    assert any("No NIFC perimeter pixels" in t.get_text() for t in ax.texts)


def test_grid(fire):
    axes = fire.grid("I4", cols=2)
    assert axes.shape == (2, 2)
    assert [a.get_title() for a in axes.flat[:3]] == DAYS
    assert not axes.flat[3].axison and not axes.flat[3].images  # spare panel left empty
    assert axes.flat[0].figure._suptitle.get_text() == "42 · I4"
    assert fire.grid("Active fire", dates=slice(1, 3)).flat[0].get_title() == "2020-07-02 · 4 px"
    assert fire.grid("I4", dates=["2020-07-03"]).flat[0].get_title() == "2020-07-03"


def test_plot_into_your_own_axes(fire):
    fig, (left, right) = plt.subplots(1, 2)
    assert fire.show("I1", ax=left) is left
    assert fire.overlay("Active fire", ax=right) is right


# ----- comparison --------------------------------------------------------------


def test_compare_channels_and_default_view(fire, win):
    both = firevis.compare(fire, win)
    assert isinstance(both, Comparison) and fire.compare(win).fires == [fire, win]
    shared = both.channels()
    assert "I4" in shared and "wind speed" in shared and "Active fire" not in shared
    assert both._view(None) == "crop" and firevis.compare(fire, fire)._view(None) == "fire"
    with pytest.raises(ValueError):
        Comparison(fire)


def test_compare_show_same_date(fire, win):
    left, right = firevis.compare(fire, win).show("I4", "2020-07-02")
    assert left.get_title() == "42\nI4 · 2020-07-02"
    assert right.get_title() == "p42.npz\nVIIRS I4 (running max) · 2020-07-02"
    assert len(left.figure.axes) == 3  # two images, one shared colorbar
    _, missing = firevis.compare(fire, win).show("I4", "2020-07-03")
    assert not missing.images and "No 2020-07-03 in p42.npz" in missing.texts[0].get_text()


def test_compare_overlay_and_grid(fire, win):
    axes = firevis.compare(fire, win).overlay("I4")
    assert len(axes) == 2 and axes[1].get_title().endswith("2020-07-01 → 2020-07-02")
    grid = firevis.compare(fire, win).grid("I4")
    assert grid.shape == (2, 3)  # every date either has, one row each
    assert grid[1][2].texts[0].get_text() == "—"  # the window has no 2020-07-03
    assert [a.get_title() for a in grid[0]] == DAYS
