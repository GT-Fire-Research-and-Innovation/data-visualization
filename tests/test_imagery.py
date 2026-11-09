"""Rendering, stacks and the image viewer, on small real GeoTIFFs and archives."""

import io

import numpy as np
import pytest

from firevis import render
from firevis.stacks import (CLASSES, MASK, RGB, VALUES, FireStack, Layer, NpzStack, Stack,
                            date_of, fire_box, is_fire_folder)
from firevis.viewer import StackViewer

from imagery import DAYS, fire_folder, window_npz

rasterio = pytest.importorskip("rasterio")


def write_files(root, files):
    for path, data in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root


@pytest.fixture
def fire(tmp_path):
    write_files(tmp_path, fire_folder())
    return FireStack(tmp_path / "ts-satfire" / "42")


# ----- helpers ---------------------------------------------------------------


def test_fire_folder_detection_and_dates():
    assert is_fire_folder(["ESRI_LULC", "FirePred", "VIIRS_Day", "VIIRS_Night"])
    assert is_fire_folder(["FirePred", "VIIRS_Day", "VIIRS_Night"])  # 28 fires lack ESRI
    assert not is_fire_folder(["ESRI_LULC"]) and not is_fire_folder(["VIIRS_Day"])
    assert date_of("2017-04-25_VIIRS_Day.tif") == "2017-04-25"
    assert date_of("odd.tif") == "odd"


def test_fire_box_is_a_padded_square():
    mask = np.zeros((100, 100), bool)
    mask[40:50, 60:64] = True
    top, bottom, left, right = fire_box(mask, margin=1.0, smallest=8)
    assert (bottom - top) == pytest.approx(right - left)  # square
    assert top <= 0.40 and bottom >= 0.50 and left <= 0.60 and right >= 0.64
    assert fire_box(np.zeros((5, 5), bool)) is None


def test_overlay_colours_follow_time():
    canvas = render.background((4, 4))
    day1 = np.zeros((4, 4), bool); day1[0, 0] = True
    day2 = np.zeros((4, 4), bool); day2[0, 0] = day2[1, 1] = True
    render.overlay_masks([day1, day2], [0.0, 1.0], "fire", canvas, cumulative=True, alpha=1.0)
    early, late = render.ramp_colour("fire", 0), render.ramp_colour("fire", 1)
    assert np.allclose(canvas[0, 0], early, atol=1)  # first burned on day 1, stays day-1 colour
    assert np.allclose(canvas[1, 1], late, atol=1)
    assert np.allclose(canvas[3, 3], render.PLAIN)


def test_classes_and_png():
    rgb, legend = render.classes_rgb(np.array([[1, 2], [np.nan, 99]], np.float32), "esri")
    assert legend[1][0] == "Water" and legend[2][0] == "Trees" and legend[99][0] == "Class 99"
    assert tuple(rgb[1, 0]) == render.MISSING
    for encode in (render.png_bytes, render._png_fallback):
        data = encode(rgb)
        from PIL import Image
        assert Image.open(io.BytesIO(data)).size == (2, 2)


# ----- fire folders ----------------------------------------------------------


def test_fire_stack_layers(fire):
    layers = {layer.name: layer for layer in fire.layers()}
    assert [l.name for l in fire.layers()[:3]] == [
        "Active fire", "NIFC perimeter", "Burned area (active fire + NIFC)"]
    assert all(layers[n].overlay for n in ("Active fire", "NIFC perimeter"))
    assert layers["Active fire"].dates == DAYS
    assert [layers[n].style for n in ("I1", "M11", "I4 night", "wind speed")] == [VALUES] * 4
    assert layers["land cover (MODIS)"].style == CLASSES and layers["land cover (MODIS)"].palette == "igbp"
    assert layers["land cover (ESRI)"].dates == ["2020-01-01"]
    assert layers["land cover (ESRI)"].read(0).shape == (32, 32)  # resampled onto VIIRS
    assert layers["Active fire"].read(1).sum() == 4  # one 4-pixel column per day
    burned = layers["Burned area (active fire + NIFC)"]
    assert burned.read(0).sum() == 4 and burned.read(2).sum() == 24  # NIFC block covers all AF
    assert layers["NIFC perimeter"].read(0).sum() == 0 and layers["NIFC perimeter"].read(1).sum() == 24
    assert fire.background().style == RGB
    assert fire.focus() is not None


def test_viewer_starts_overlaid_on_fire_layers(fire):
    viewer = StackViewer(fire)
    assert viewer.layer.name == "Active fire" and viewer.overlaid
    assert bytes(viewer._image.value).startswith(b"\x89PNG")
    assert "2020-07-01 → 2020-07-03 · 3 days overlaid" in viewer._date.value
    assert viewer._prev.layout.visibility == "hidden"
    assert "2020-07-01" in viewer._legend.value and "12 pixels" in viewer._legend.value


def test_viewer_steps_one_day_at_a_time(fire):
    viewer = StackViewer(fire)
    viewer.show_layer([l.name for l in viewer.layers].index("I1"))
    assert not viewer.overlaid and "<b>2020-07-01</b> · 1 / 3" in viewer._date.value
    assert viewer._prev.disabled and not viewer._next.disabled
    viewer._next.click()
    assert "<b>2020-07-02</b> · 2 / 3" in viewer._date.value
    viewer._next.click(); viewer._next.click()  # stops at the end
    assert viewer.frame == 2 and viewer._next.disabled
    viewer.show_layer([l.name for l in viewer.layers].index("wind speed"))
    assert viewer.frame == 2  # same date on the new layer


def test_viewer_corners(fire):
    sizes = []
    viewer = StackViewer(fire, on_size=sizes.append)
    viewer._overlay_btn.click()
    assert not viewer.overlaid and "1 / 3" in viewer._date.value
    viewer._size_btn.click()
    assert sizes == [True] and "fx-big" in viewer._stage._dom_classes
    viewer.show_layer([l.name for l in viewer.layers].index("land cover (ESRI)"))
    assert viewer._overlay_btn.disabled  # one date, classes: nothing to overlay
    assert viewer._prev.layout.visibility == "hidden"


def test_empty_overlay_says_so(fire):
    viewer = StackViewer(fire)
    viewer.show_layer(1)  # NIFC perimeter: only day 2 has pixels
    assert "24 pixels" in viewer._legend.value
    layer = Layer("Nothing", "Fire", MASK, DAYS, lambda i: np.zeros((4, 4), bool), overlay=True)

    class Empty(Stack):
        def layers(self, window=0):
            return [layer]

    assert "No Nothing pixels on any of the 3 days" in StackViewer(Empty("e"))._legend.value


def test_fire_area_view_crops(fire):
    viewer = StackViewer(fire)
    from PIL import Image

    cropped = Image.open(io.BytesIO(bytes(viewer._image.value))).size
    viewer._view_dd.value = "Whole image"
    whole = Image.open(io.BytesIO(bytes(viewer._image.value))).size
    assert whole == (32, 32) and cropped[0] < 32


# ----- npz -------------------------------------------------------------------


@pytest.mark.parametrize("zstd", [False, True])
def test_npz_windows(tmp_path, zstd):
    path = tmp_path / "p42.npz"
    path.write_bytes(window_npz(zstd=zstd))
    stack = NpzStack(path, dates=lambda w, d: DAYS[d] if w + d < 3 else None)
    assert stack.windows == ["Window 1: 2020-07-01 → 2020-07-03", "Window 2"]
    layers = stack.layers(0)
    assert layers[0].name == "Target: newly burned next day" and layers[0].style == MASK
    assert layers[1].name == "VIIRS I1" and layers[1].dates == DAYS
    assert [l.name for l in layers].count("land cover") == 1
    assert layers[1].read(2).shape == (8, 8)
    assert stack.background(0).read(0).shape == (3, 8, 8)


def test_npz_viewer_toggles_and_windows(tmp_path):
    path = tmp_path / "p42.npz"
    path.write_bytes(window_npz())
    viewer = StackViewer(NpzStack(path))
    viewer.show_layer(1)  # VIIRS I1
    assert not viewer.overlaid and "<b>Day 1</b> · 1 / 3" in viewer._date.value
    assert not viewer._overlay_btn.disabled
    viewer._overlay_btn.click()
    assert viewer.overlaid and "3 days overlaid" in viewer._date.value
    viewer._window_dd.value = 1
    assert viewer.window == 1 and viewer._window_dd.layout.display == ""
    viewer.close()


def test_other_npz_shows_each_array(tmp_path):
    path = tmp_path / "tessera.npz"
    np.savez(path, embedding=np.zeros((4, 6, 6), np.float16), coverage=np.ones((6, 6)), note=np.array("x"))
    layers = NpzStack(path).layers()
    assert [l.name for l in layers] == ["embedding", "coverage"]
    assert layers[0].dates == ["Slice 1", "Slice 2", "Slice 3", "Slice 4"]


# ----- explorer --------------------------------------------------------------


@pytest.fixture
def imagery_api():
    import json

    from fakes import FakeApi

    files = fire_folder()
    npz = window_npz(windows=2, days=2)
    config = json.dumps({"ts_length": 2, "ts_interval": 1}).encode()
    contents = dict(files, **{"train/p42.npz": npz, "window_config.json": config})
    tree = {
        "me/ts-satfire": [f"{path}:{len(data)}" for path, data in files.items()],
        "me/windows": [f"train/p42.npz:{len(npz)}", f"window_config.json:{len(config)}"],
    }
    return FakeApi(tree=tree, contents=contents)


def explorer_for(api, mirror, **options):
    from firevis import DatasetExplorer, HubSource

    return DatasetExplorer(source=HubSource("me", api=api, mirror_dir=mirror), **options)


def open_row(explorer, name):
    for row in explorer._list.children:
        if "fx-row" in row._dom_classes and row.children[0].description.endswith(name):
            row.children[0].click()
            return
    raise AssertionError(name)


def test_clicking_a_fire_folder_downloads_and_shows(imagery_api, mirror):
    explorer = explorer_for(imagery_api, mirror)
    open_row(explorer, "ts-satfire")
    open_row(explorer, "ts-satfire")
    open_row(explorer, "42")
    downloaded = [c[2] for c in imagery_api.calls if c[0] == "download"]
    assert len(downloaded) == 10 and all(p.endswith(".tif") for p in downloaded)
    assert explorer.location == ("me/ts-satfire", "ts-satfire/42")  # navigated as usual
    viewer = explorer._viewer
    assert viewer is not None and viewer.layer.name == "Active fire"
    assert explorer._preview_content.children == (viewer.widget,)
    assert "Fire folder" in explorer._preview_title.value
    assert "Saved at" in explorer._preview_path.value
    open_row(explorer, "VIIRS_Day")  # an ordinary folder: the viewer stays open
    assert explorer._viewer is viewer
    explorer._back_btn.click()
    open_row(explorer, "VIIRS_Day")
    explorer._crumbs[3].click()  # back in the fire folder (the crumbs don't reopen it)
    explorer.open_fire()  # again, from code: nothing new to download
    assert len([c for c in imagery_api.calls if c[0] == "download"]) == 10


def test_big_fire_folder_asks_first(imagery_api, mirror):
    explorer = explorer_for(imagery_api, mirror, confirm_size=1000)
    open_row(explorer, "ts-satfire"); open_row(explorer, "ts-satfire"); open_row(explorer, "42")
    assert explorer._viewer is None and explorer._preview_ask.layout.display == ""
    assert "10 GeoTIFFs" in explorer._preview_body.value
    explorer._download_btn.click()
    assert explorer._viewer is not None


def test_size_corner_widens_the_pane(imagery_api, mirror):
    explorer = explorer_for(imagery_api, mirror)
    open_row(explorer, "ts-satfire"); open_row(explorer, "ts-satfire"); open_row(explorer, "42")
    explorer._viewer._size_btn.click()
    assert "fx-preview-wide" in explorer._preview._dom_classes
    assert explorer._preview.layout.height is None
    explorer.close_preview()
    assert "fx-preview-wide" not in explorer._preview._dom_classes and explorer._viewer is None


def test_npz_gets_dates_from_its_fire(imagery_api, mirror):
    explorer = explorer_for(imagery_api, mirror)
    open_row(explorer, "windows"); open_row(explorer, "train"); open_row(explorer, "p42.npz")
    viewer = explorer._viewer
    assert viewer is not None
    assert viewer.stack.windows == ["Window 1: 2020-07-01 → 2020-07-02",
                                    "Window 2: 2020-07-02 → 2020-07-03"]
    viewer.show_layer(1)
    assert "<b>2020-07-01</b> · 1 / 2" in viewer._date.value
    assert viewer.layers[0].dates == ["2020-07-03"]  # window 1's target is the next day


def test_fire_box_ignores_stray_detections():
    mask = np.zeros((600, 600), bool)
    mask[290:310, 250:340] = True  # the fire
    mask[180, 210] = mask[558, 465] = True  # two stray detections
    top, bottom, left, right = fire_box(mask)
    assert (bottom - top) * 600 < 200 and top * 600 > 180 and right * 600 < 465
