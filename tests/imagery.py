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
