"""Small real files for viewer tests: GeoTIFF fire folders and window archives."""

import io
import zipfile

import numpy as np

DAYS = ["2020-07-01", "2020-07-02", "2020-07-03"]
SIZE = 32


def window_npz(windows=2, days=3, size=8, channels=27, zstd=False) -> bytes:
    """A window archive like firepred writes: data_### (C, T, H, W) + label_###."""
    rng = np.random.default_rng(0)
    arrays = {"format_version": np.array(2), "num_windows": np.array(windows)}
    for w in range(windows):
        arrays[f"data_{w:03d}"] = rng.random((channels, days, size, size), dtype=np.float32)
        label = np.zeros((size, size), np.float32)
        label[2 + w, 3:5] = 1
        arrays[f"label_{w:03d}"] = label
    buffer = io.BytesIO()
    method = zipfile.ZIP_ZSTANDARD if zstd else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(buffer, "w", compression=method) as z:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, array)
            z.writestr(f"{name}.npy", member.getvalue())
    return buffer.getvalue()


def geotiff(bands: np.ndarray, names, pixel: float = 0.0034) -> bytes:
    """A small EPSG:4326 GeoTIFF with named bands."""
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    count, height, width = bands.shape
    profile = dict(driver="GTiff", count=count, height=height, width=width, dtype="float32",
                   crs="EPSG:4326", transform=from_origin(-110, 35, pixel, pixel))
    with MemoryFile() as memory:
        with memory.open(**profile) as dst:
            dst.write(bands.astype(np.float32))
            for i, name in enumerate(names, start=1):
                dst.set_band_description(i, name)
        return memory.read()


def fire_folder(fire_id="42", esri=True):
    """{path: bytes} for one fire: 3 days of VIIRS_Day/Night/FirePred, plus ESRI land cover.

    Active fire grows by a column each day; the NIFC band is set on day 2 only.
    """
    rng = np.random.default_rng(1)
    files = {}
    for d, date in enumerate(DAYS):
        day = rng.random((8, SIZE, SIZE)).astype(np.float32) * 50
        day[6:] = np.nan
        day[6, 10:14, 10 + d] = 300.0  # active fire
        if d == 1:
            day[7, 9:15, 9:13] = 1.0  # NIFC perimeter
        files[f"ts-satfire/{fire_id}/VIIRS_Day/{date}_VIIRS_Day.tif"] = geotiff(
            day, ["b1", "b2", "b3", "b4", "b5", "m11", "af", "first"])
        files[f"ts-satfire/{fire_id}/VIIRS_Night/{date}_VIIRS_Night.tif"] = geotiff(
            rng.random((2, SIZE, SIZE)) * 300, ["b1", "b2"])
        pred = rng.random((3, SIZE, SIZE)).astype(np.float32)
        pred[2] = np.where(pred[2] > 0.5, 7, 10)
        files[f"ts-satfire/{fire_id}/FirePred/{date}_FirePred.tif"] = geotiff(
            pred, ["NDVI_last", "wind speed", "LC_Type1"])
    if esri:
        files[f"ts-satfire/{fire_id}/ESRI_LULC/2020-01-01_ESRI_LULC.tif"] = geotiff(
            np.where(rng.random((1, SIZE * 4, SIZE * 4)) > 0.5, 2.0, 11.0), ["lulc_classification"],
            pixel=0.0034 / 4)  # finer grid over the same area, like the 10 m originals
    return files


# ----- archives for fire splitting --------------------------------------------


def _npz_bytes(arrays, zstd=False) -> bytes:
    buffer = io.BytesIO()
    method = zipfile.ZIP_ZSTANDARD if zstd else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(buffer, "w", compression=method) as z:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(array))
            z.writestr(f"{name}.npy", member.getvalue())
    return buffer.getvalue()


def diverging_fire(size=64):
    """The burn-day map of firepred's split test: a front that breaks in two on day 10.

    Days 6..11 burn (window length 6, so the targets are days 6..12): one fire
    spreading both ways from columns 28..35, whose two ends end up far apart.
    """
    days = np.full((size, size), -1, np.int16)
    rows = slice(30, 34)
    for day, (first, last) in {6: (28, 35), 7: (24, 27), 8: (20, 23), 9: (16, 19), 10: (12, 15),
                               11: (8, 11)}.items():
        days[rows, first:last + 1] = day
    for day, (first, last) in {7: (36, 39), 8: (40, 43), 9: (44, 47), 10: (48, 51), 11: (52, 55)}.items():
        days[rows, first:last + 1] = day
    return days


def _frames(burn_day, num_days, channels=27):
    """One (channels, H, W) frame per day; night I4 stays cool, so there are no heat seeds."""
    rng = np.random.default_rng(2)
    frames = rng.random((channels, num_days) + burn_day.shape, dtype=np.float32)
    frames[6] = 290.0
    return frames


def split_window_npz(burn_day, length=6, last_day=12, interval=1, zstd=False) -> bytes:
    """A format-2 window archive whose targets are the pixels of each day of `burn_day`."""
    starts = range(0, last_day - length + 1, interval)
    frames = _frames(burn_day, last_day)
    arrays = {"format_version": np.array(2), "num_windows": np.array(len(starts))}
    for w, start in enumerate(starts):
        arrays[f"data_{w:03d}"] = frames[:, start:start + length]
        arrays[f"label_{w:03d}"] = (burn_day == start + length).astype(np.float32)
    return _npz_bytes(arrays, zstd)


def split_daily_npz(burn_day, length=6, last_day=12, block_days=6) -> bytes:
    """The same fire as a daily archive (format 3), as firepred's convert-daily writes it."""
    frames = _frames(burn_day, last_day)
    channels, num_days = frames.shape[:2]
    target_days = np.arange(length, last_day + 1, dtype=np.int16)
    arrays = {
        "format_version": np.array(3, np.int16), "num_windows": np.array(len(target_days), np.int32),
        "ts_length": np.array(length, np.int16), "ts_interval": np.array(1, np.int16),
        "num_days": np.array(num_days, np.int32), "block_days": np.array(block_days, np.int16),
        "frame_shape": np.array((channels,) + burn_day.shape, np.int32),
        "static_channels": np.array([19], np.int16),
    }
    frames[19] = frames[19, :1]  # elevation: the same every day, stored once
    for c in range(channels):
        if c == 19:
            arrays["static_c19"] = frames[19, 0]
            continue
        for first in range(0, num_days, block_days):
            arrays[f"days_c{c:02d}_{first:03d}"] = frames[c, first:first + block_days]
    arrays["target_days"] = target_days
    arrays["targets"] = np.packbits(np.stack([burn_day == d for d in target_days]), axis=-1)
    arrays["target_dtype"] = np.array("float32")
    return _npz_bytes(arrays)
