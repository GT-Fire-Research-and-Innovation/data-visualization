"""Fire splitting: reading archives, the split itself, the split viewer and the explorer's button."""

import numpy as np
import pytest

from firevis.split import SplitError, can_split, read_archive, split_archive
from firevis.split_viewer import SplitViewer
from firevis.stacks import NpzStack
from firevis.viewer import StackViewer
from imagery import diverging_fire, split_daily_npz, split_window_npz, window_npz


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "p7.npz"
    path.write_bytes(split_window_npz(diverging_fire()))
    return path


# ----- reading ------------------------------------------------------------------


def test_window_archive_is_read_like_zones_master(archive):
    days = read_archive(archive)
    assert days.length == 6 and days.windows == 7 and days.label_sel == -1
    assert days.labels.dtype == bool and days.labels.shape == (7, 64, 64)
    assert days.night_heat.shape == (6, 64, 64) and (days.night_heat < 310).all()
    assert days.first_colour.shape == days.last_colour.shape == (3, 64, 64)


def test_only_window_and_daily_archives_can_split(tmp_path, archive):
    other = tmp_path / "tessera.npz"
    np.savez(other, embedding=np.zeros((4, 6, 6), np.float16))
    text = tmp_path / "notes.npz"
    text.write_text("not a zip")
    assert can_split(archive) and not can_split(other) and not can_split(text)


def test_windows_that_skip_days_are_refused(tmp_path):
    path = tmp_path / "p7.npz"
    path.write_bytes(split_window_npz(diverging_fire(), interval=3))
    with pytest.raises(SplitError, match="every 3 days.*-1"):
        split_archive(path)


def test_daily_archive_gives_the_same_split(tmp_path, archive):
    daily = tmp_path / "daily.npz"
    daily.write_bytes(split_daily_npz(diverging_fire()))
    a, b = split_archive(archive), split_archive(daily)
    assert np.array_equal(a.days.labels, b.days.labels)
    assert np.array_equal(a.days.night_heat, b.days.night_heat)
    assert np.array_equal(a.days.last_colour, b.days.last_colour)
    assert np.array_equal(a.zones.target_owners, b.zones.target_owners)
    assert a.events == b.events


# ----- the split ----------------------------------------------------------------


def test_diverging_front_splits_in_two(archive):
    split = split_archive(archive)
    assert [(e.event_id, e.kind, e.parent_id, e.start_day, e.end_day) for e in split.events] == [
        (1, "ignition", 0, 6, 10), (2, "split", 1, 10, -1), (3, "split", 1, 10, -1)]
    assert split.splits == 1 and split.merges == 0
    assert [e.event_id for e in split.fires] == [1, 2, 3] and split.quiet == []
    # Days 6 and 7 burn before the fire is confirmed (it must grow on 2 days): no fire owns them.
    assert split.windows_of(-1) == [0, 1] and split.unowned == 64
    # Day 11 comes after the split: each end goes to its own branch.
    assert split.windows_of(1) == [2, 3, 4] and split.windows_of(2) == split.windows_of(3) == [5]
    owners = split.zones.target_owners
    assert owners[5, 31, 9] == 2 and owners[5, 31, 53] == 3


def test_afters_add_up_to_the_before(archive):
    split = split_archive(archive)
    total = sum(split.pixels(e.event_id) for e in split.fires) + split.unowned
    assert total == split.burned == int((diverging_fire() >= 0).sum())


def test_story_and_dates(archive):
    dates = [f"2020-08-{d:02d}" for d in range(1, 14)]
    split = split_archive(archive, dates=lambda w, d: dates[w + d] if w + d < len(dates) else None)
    assert split.target_dates[0] == "2020-08-07" and split.target_dates[-1] == "2020-08-13"
    first, second = split.events[:2]
    assert split.describe(first) == "new fire from 2020-08-07; split on 2020-08-11 into fires 2, 3"
    assert split.describe(second) == "split from fire 1 on 2020-08-11; still burning at the end"
    assert "→ 3 fires (1 split, 0 merges)" not in split.summary()
    assert "became 3 fires (1 split, 0 merges)" in split.summary()


def test_far_speckle_belongs_to_no_fire(tmp_path):
    days = diverging_fire()
    days[5, 60] = 9  # one lone pixel far from the fire
    path = tmp_path / "p7.npz"
    path.write_bytes(split_window_npz(days))
    split = split_archive(path)
    assert split.unowned == 64 + 1 and split.zones.target_owners[3, 5, 60] == -1
    assert split.stack().no_fire is not None
    assert split_archive(archive_without(days, tmp_path)).windows_of(-1) == [0, 1]


def archive_without(days, tmp_path):
    days = days.copy()
    days[5, 60] = -1
    path = tmp_path / "p8.npz"
    path.write_bytes(split_window_npz(days))
    return path


def test_scale_changes_the_split(archive):
    assert split_archive(archive, scale_km=8).splits == 0  # ends never 8 km apart: one fire


# ----- viewer -------------------------------------------------------------------


def test_viewer_shows_after_then_before(archive):
    back = []
    viewer = SplitViewer(split_archive(archive), on_back=lambda: back.append(True))
    assert viewer.mode == "after" and "fx-on" in viewer._after_btn._dom_classes
    assert viewer._after_btn.description == "After · 3 fires"
    assert viewer._body.children[0] is viewer._grid and len(viewer._grid.children) == 4  # + no fire
    assert all(bytes(image.value).startswith(b"\x89PNG") for _, image in viewer._cards)
    captions = [card.children[1].value for card in viewer._grid.children]
    assert "<b>Fire 2</b>" in captions[1] and "Split from fire 1 on Day 11" in captions[1]
    assert "<b>No fire</b> · 64 px" in captions[3]
    assert "Day 7" in viewer._legend.value and "Day 13" in viewer._legend.value
    assert "<b>3</b> fires · <b>1</b> split" in viewer._summary.value
    viewer._before_btn.click()
    assert viewer.mode == "before" and viewer._body.children[0] is viewer._before_stage
    assert bytes(viewer._before_image.value).startswith(b"\x89PNG")
    viewer._view_dd.value = "Whole image"
    viewer._bg_dd.value = "None"
    assert "fx-legend-error" not in viewer._legend.value
    viewer._back_btn.click()
    assert back == [True]
    viewer.close()


def test_fire_area_crops_every_image_alike(tmp_path):
    import io

    from PIL import Image

    path = tmp_path / "p7.npz"
    path.write_bytes(split_window_npz(diverging_fire(size=128)))
    viewer = SplitViewer(split_archive(path))
    sizes = {Image.open(io.BytesIO(image.value)).size for _, image in viewer._cards}
    viewer.show("before")
    sizes.add(Image.open(io.BytesIO(viewer._before_image.value)).size)
    assert len(sizes) == 1 and next(iter(sizes))[0] < 128  # one crop for all, smaller than the tile
    viewer._view_dd.value = "Whole image"
    assert Image.open(io.BytesIO(viewer._before_image.value)).size == (128, 128)


def test_viewer_without_fires(tmp_path):
    path = tmp_path / "p7.npz"
    path.write_bytes(split_window_npz(np.full((64, 64), -1, np.int16)))
    viewer = SplitViewer(split_archive(path))
    assert viewer._after_btn.description == "After · 0 fires"
    assert "no burned pixels" in viewer._after_note.value
    viewer.show("before")
    assert "No burned pixels on any of the 7 days" in viewer._legend.value


# ----- explorer -----------------------------------------------------------------


@pytest.fixture
def split_api():
    from fakes import FakeApi

    good, skipping = split_window_npz(diverging_fire()), split_window_npz(diverging_fire(), interval=3)
    plain = window_npz(windows=1)
    contents = {"train/p7.npz": good, "train/p8.npz": skipping, "train/p9.npz": plain}
    tree = {"me/ts-satfire-6-1": [f"{path}:{len(data)}" for path, data in contents.items()]}
    return FakeApi(tree=tree, contents=contents)


def open_row(explorer, name):
    for row in explorer._list.children:
        if "fx-row" in row._dom_classes and row.children[0].description.endswith(name):
            row.children[0].click()
            return
    raise AssertionError(name)


def open_archive(api, mirror, name):
    from firevis import DatasetExplorer, HubSource

    explorer = DatasetExplorer(source=HubSource("me", api=api, mirror_dir=mirror))
    open_row(explorer, "ts-satfire-6-1"); open_row(explorer, "train"); open_row(explorer, name)
    return explorer


def test_npz_preview_offers_the_split(split_api, mirror):
    explorer = open_archive(split_api, mirror, "p7.npz")
    assert isinstance(explorer._viewer, StackViewer)
    assert explorer._preview_actions.layout.display == ""
    explorer._split_btn.click()
    viewer = explorer._viewer
    assert isinstance(viewer, SplitViewer) and viewer.split.splits == 1
    assert "Fire split" in explorer._preview_title.value
    assert explorer._preview_actions.layout.display == "none"
    assert "became 3 fires" in explorer._status.value
    viewer._back_btn.click()  # back to the layers, with the button again
    assert isinstance(explorer._viewer, StackViewer) and explorer._preview_actions.layout.display == ""
    explorer._split_btn.click()
    assert explorer._viewer.split is viewer.split  # not split a second time


def test_archive_that_cannot_split_says_why(split_api, mirror):
    explorer = open_archive(split_api, mirror, "p8.npz")
    explorer._split_btn.click()
    assert explorer._viewer is None and "every 3 days" in explorer._preview_body.value
    assert explorer._download_btn.description == "◀ Layers"
    explorer._download_btn.click()
    assert isinstance(explorer._viewer, StackViewer)


def test_one_window_archive_still_splits(split_api, mirror):
    explorer = open_archive(split_api, mirror, "p9.npz")
    assert explorer.open_split() is not None


def test_other_npz_has_no_split_button(tmp_path, mirror):
    from fakes import FakeApi

    other = tmp_path / "t.npz"
    np.savez(other, embedding=np.zeros((2, 4, 4), np.float32))
    data = other.read_bytes()
    api = FakeApi(tree={"me/ts-satfire-6-1": [f"train/t.npz:{len(data)}"]}, contents={"train/t.npz": data})
    explorer = open_archive(api, mirror, "t.npz")
    assert isinstance(explorer._viewer, StackViewer) and explorer._preview_actions.layout.display == "none"
    assert isinstance(explorer._viewer.stack, NpzStack)
