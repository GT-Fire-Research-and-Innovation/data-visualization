"""The FAH-2014-2025 file formats: GeoJSON, CSV(.gz) points, OSM PBF, ZIP, NetCDF and single GeoTIFFs."""

import gzip
import io
import json
import struct
import zipfile
import zlib

import numpy as np
import pytest

from firevis import geo, pbf, viewers
from firevis.viewers import file_kind, to_html

pytest.importorskip("matplotlib")


def square(x, y, d=0.01):
    return [[[x, y], [x + d, y], [x + d, y + d], [x, y + d], [x, y]]]


def collection(features):
    return {"type": "FeatureCollection", "features": features}


def perimeter(x, when_ms, **props):
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": square(x, 44.5)},
            "properties": {"perimeterdatetime": str(when_ms), "datecurrent": "1564444800000", **props}}


# ----- GeoJSON -------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [
    ("1564454400000", "2019-07-30T02:40"),        # epoch milliseconds (NIFC / ODA)
    (1564454400, "2019-07-30T02:40"),             # epoch seconds
    ("20140102000000", "2014-01-02T00:00"),       # YYYYMMDDhhmmss (DATE_CUR)
    ("20190711", "2019-07-11T00:00"),
    ("1935", "1935-01-01T00:00"),                 # a fire year
    ("2019-07-11", "2019-07-11T00:00"),
    ("2019-07-11 03:00:00", "2019-07-11T03:00"),
])
def test_parse_time(value, expected):
    assert str(np.datetime64(geo.parse_time(value), "m")) == expected


@pytest.mark.parametrize("value", [None, "", " ", "None", "nan", "12", "abc", True])
def test_parse_time_rejects(value):
    assert geo.parse_time(value) is None


def test_feature_times_prefers_the_fires_own_date_and_falls_back():
    features = [perimeter(0, 1562814000000), perimeter(1, 1564454400000)]
    del features[1]["properties"]["perimeterdatetime"]  # this one only has its "date current"
    key, times = geo.feature_times(features)
    assert key == "perimeterdatetime"
    assert [geo.format_time(t) for t in times] == ["2019-07-11 03:00", "2019-07-30"]
    year_key, _ = geo.feature_times([{"properties": {"DATE_CUR": "20140102000000", "FIRE_YEAR_INT": "1935"}}])
    assert year_key == "FIRE_YEAR_INT"  # when it burned beats when the record was edited
    assert geo.feature_times([{"properties": {"name": "x"}}]) == (None, [None])


def test_large_geojson_is_streamed(tmp_path):
    path = tmp_path / "event_polygons.geojson"
    features = [perimeter(i * 0.02, 1562814000000 + i * 3_600_000, n=i) for i in range(50)]
    path.write_text(json.dumps(collection(features), indent=1))
    whole = geo.read_geojson(path)
    assert whole.complete and len(whole.features) == 50 and whole.note == ""
    first = geo.read_geojson(path, full_read_bytes=0, stream_features=7)
    assert not first.complete and [f["properties"]["n"] for f in first.features] == list(range(7))
    assert "first 7 features" in first.note
    everything = geo.read_geojson(path, full_read_bytes=0, stream_features=1000)
    assert everything.complete and len(everything.features) == 50
    small_chunks = list(geo._stream_features(path, chunk=97))  # features cut across chunk borders
    assert [f["properties"]["n"] for f, _ in small_chunks] == list(range(50))


def test_geojson_preview_draws_a_dated_map_and_table(tmp_path):
    path = tmp_path / "records.geojson"
    features = [perimeter(0, 1562814000000, gisacres="10"), perimeter(0.005, 1564454400000, gisacres="900"),
                {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[0, 44.4], [0.02, 44.45]]},
                 "properties": {"perimeterdatetime": "1563000000000"}},
                {"type": "Feature", "geometry": {"type": "MultiPoint", "coordinates": [[0.01, 44.51], [0.012, 44.52]]},
                 "properties": {}}]
    path.write_text(json.dumps(collection(features)))
    assert file_kind(path) == "geojson"
    body = to_html(path)
    assert "<b>4</b> features" in body and "LineString, MultiPoint, Polygon" in body
    assert 'class="fx-map"' in body and "data:image/png;base64," in body
    assert "2019-07-11 03:00" in body and "2019-07-30 02:40" in body and "perimeterdatetime" in body
    assert "<th>gisacres</th>" in body and "<td>900</td>" in body


def test_empty_and_bare_geojson(tmp_path):
    empty = tmp_path / "event_lines.geojson"
    empty.write_text(json.dumps(collection([])))
    assert "No features" in to_html(empty)
    bare = tmp_path / "point.geojson"
    bare.write_text(json.dumps({"type": "Point", "coordinates": [-115, 44.5]}))
    assert "<b>1</b> feature " in to_html(bare)


# ----- tables with coordinates --------------------------------------------------


FIRMS = ("latitude,longitude,bright_ti4,acq_date,acq_time,satellite\n"
         "44.51485,-115.01574,296.65,2019-07-11,1012,N\n"
         "44.51596,-115.02042,310.66,2019-07-11,1012,N\n"
         "44.53000,-115.03000,330.10,2019-08-02,957,N\n")


def test_gzipped_csv_is_a_table_with_a_point_map(tmp_path):
    pytest.importorskip("pandas")
    path = tmp_path / "VIIRS_SNPP_SP.csv.gz"
    path.write_bytes(gzip.compress(FIRMS.encode()))
    assert file_kind(path) == "table" and viewers.suffix(path) == ".csv.gz"
    body = to_html(path)
    assert "<th>bright_ti4</th>" in body and "3 rows." in body
    assert 'class="fx-map"' in body and "3 points" in body
    assert "2019-07-11 10:12" in body and "2019-08-02 09:57" in body and "acq_date + acq_time" in body


def test_csv_without_coordinates_has_no_map(tmp_path):
    pytest.importorskip("pandas")
    path = tmp_path / "incidents.csv"
    path.write_text("INCIDENT_ID,FINAL_ACRES\nA,10\nB,20\n")
    body = to_html(path)
    assert "fx-map" not in body and "<td>20</td>" in body


# ----- OpenStreetMap PBF ----------------------------------------------------------


def _varint(value):
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _zig(value):
    return (value << 1) ^ (value >> 63)


def _field(number, value):
    if isinstance(value, int):
        return _varint(number << 3) + _varint(value)
    return _varint(number << 3 | 2) + _varint(len(value)) + value


def _packed(values, signed=False):
    return b"".join(_varint(_zig(v) if signed else v) for v in values)


def _deltas(values):
    return [b - a for a, b in zip([0] + list(values[:-1]), values)]


def _block(kind, message):
    blob = _field(2, len(message)) + _field(3, zlib.compress(message))
    header = _field(1, kind.encode()) + _field(3, len(blob))
    return struct.pack(">I", len(header)) + header + blob


def osm_pbf(roads=True):
    """A tiny extract: 4 nodes, a road (2 nodes) and a closed building (3 nodes + first)."""
    header = (_field(1, _field(1, _zig(-115_100_000_000)) + _field(2, _zig(-115_000_000_000))
                     + _field(3, _zig(44_600_000_000)) + _field(4, _zig(44_500_000_000)))
              + _field(4, b"OsmSchema-V0.6") + _field(4, b"DenseNodes") + _field(16, b"test-writer")
              + _field(32, 1483228800))
    ids, lat, lon = [1, 2, 3, 4], [445100000, 445200000, 445150000, 445150000], [-1150500000, -1150400000,
                                                                               -1150450000, -1150460000]
    dense = _field(1, _packed(_deltas(ids), True)) + _field(8, _packed(_deltas(lat), True)) + \
        _field(9, _packed(_deltas(lon), True))
    strings = _field(1, b"") + _field(1, b"highway") + _field(1, b"residential") + _field(1, b"building") + \
        _field(1, b"yes")
    road = _field(1, 10) + _field(2, _packed([1])) + _field(3, _packed([2])) + _field(8, _packed(_deltas([1, 2]), True))
    house = _field(1, 11) + _field(2, _packed([3])) + _field(3, _packed([4])) + \
        _field(8, _packed(_deltas([2, 3, 4, 2]), True))
    groups = _field(2, dense) + (_field(3, road) if roads else b"") + _field(3, house) + _field(4, _field(1, 99))
    data = _field(1, strings) + _field(2, groups)
    return _block("OSMHeader", header) + _block("OSMData", data)


def test_pbf_is_decoded_like_osmium(tmp_path):
    path = tmp_path / "osm_2017-01-01.osm.pbf"
    path.write_bytes(osm_pbf())
    summary = pbf.read_pbf(path)
    assert summary.bbox == (-115.1, 44.5, -115.0, 44.6)
    assert summary.writing_program == "test-writer" and summary.required_features == ["OsmSchema-V0.6", "DenseNodes"]
    assert summary.replication_timestamp.year == 2017
    assert (summary.nodes, summary.ways, summary.relations, summary.data_blocks) == (4, 2, 1, 1)
    assert summary.way_keys == {"highway": 1, "building": 1}
    assert np.allclose(summary.drawn["highway"][0], [[-115.05, 44.51], [-115.04, 44.52]])
    assert len(summary.drawn["building"][0]) == 4
    header_only = pbf.read_pbf(path, decode=False)
    assert not header_only.decoded and header_only.nodes == 0 and header_only.data_blocks == 1


def test_packed_varints_match_protobuf():
    values = [0, 1, 127, 128, 300, 2 ** 31, 2 ** 40 + 5, 2 ** 62]
    assert pbf.packed_varints(_packed(values)).tolist() == values
    signed = [0, -1, 1, -2, 123456789, -987654321]
    assert pbf.packed_varints(_packed(signed, True), signed=True).tolist() == signed


def test_pbf_preview(tmp_path):
    path = tmp_path / "osm_2017-01-01.osm.pbf"
    path.write_bytes(osm_pbf())
    assert file_kind(path) == "osm"
    body = to_html(path)
    assert "<b>4</b> nodes · <b>2</b> ways · <b>1</b> relations" in body and "test-writer" in body
    assert "Roads 1" in body and "Buildings 1" in body and 'class="fx-map"' in body
    assert "data as of 2017-01-01 00:00 UTC" in body
    with pytest.raises(ValueError, match="OSMHeader"):
        bad = tmp_path / "bad.osm.pbf"
        bad.write_bytes(_block("OSMData", b""))
        pbf.read_pbf(bad)


# ----- ZIP --------------------------------------------------------------------------


def test_zip_lists_its_members(tmp_path):
    path = tmp_path / "ics209plus-wildfire.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("ics209plus-wildfire/ics209-plus-wf_incidents_1999to2020.csv", "a,b\n" * 1000)
        z.writestr("ics209plus-wildfire/README.txt", "hello")
    assert file_kind(path) == "zip"
    body = to_html(path)
    assert "<b>2</b> files" in body and "ics209-plus-wf_incidents_1999to2020.csv" in body and "4,000" in body


# ----- NetCDF and GeoTIFF -----------------------------------------------------------


def gridmet_nc(path, lat_ascending=False, days=5):
    """A GRIDMET-like NetCDF3 file: day x lat x lon wind speed, packed int16, plus a code variable."""
    from scipy.io import netcdf_file

    with netcdf_file(path, "w") as f:
        f.createDimension("day", days)
        f.createDimension("lat", 3)
        f.createDimension("lon", 4)
        day = f.createVariable("day", "f8", ("day",))
        day[:] = np.arange(days) + 43641  # days since 1900-01-01 -> 2019-06-27..
        day.units = b"days since 1900-01-01 00:00:00"
        lat = f.createVariable("lat", "f8", ("lat",))
        lat[:] = [44.5, 44.6, 44.7] if lat_ascending else [44.7, 44.6, 44.5]
        lon = f.createVariable("lon", "f8", ("lon",))
        lon[:] = [-115.2, -115.1, -115.0, -114.9]
        speed = f.createVariable("daily_mean_wind_speed", "i2", ("day", "lat", "lon"))
        speed.scale_factor = 0.1
        speed.units = b"m/s"
        speed._FillValue = np.int16(32767)
        raw = (np.arange(days * 12, dtype=np.int16) % 50).reshape(days, 3, 4)
        raw[0, 0, 0] = 32767
        raw[:, 0, 1] = 200  # the first stored row holds a marker value
        speed[:] = raw
        flags = f.createVariable("flag", "i2", ("day", "lat", "lon"))
        flags[:] = np.tile(np.array([0, 1, 2, 3], np.int16), (days, 3, 1))


def test_netcdf_stack_dates_units_classes_and_north_up(tmp_path):
    pytest.importorskip("xarray")
    from firevis.stacks import NetcdfStack

    path = tmp_path / "gridmet_vs.nc"
    gridmet_nc(path)
    assert file_kind(path) == "netcdf"
    stack = NetcdfStack(path)
    speed, flag = stack.layers()
    assert speed.label == "Variables · daily_mean_wind_speed (m/s)" and speed.style == "values"
    assert speed.dates == ["2019-06-27", "2019-06-28", "2019-06-29", "2019-06-30", "2019-07-01"]
    first = speed.read(0)
    assert np.isnan(first[0, 0]) and np.isclose(first[0, 1], 20.0)  # fill value -> NaN, scale applied
    assert flag.style == "classes" and "day 5, lat 3, lon 4" in stack.description
    ascending = tmp_path / "south_first.nc"
    gridmet_nc(ascending, lat_ascending=True)
    assert np.isclose(NetcdfStack(ascending).layers()[0].read(0)[-1, 1], 20.0)  # flipped: north at the top


def test_geotiff_stack_names_fuel_models_and_downsamples(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from firevis.stacks import TifStack

    def write(name, array, dtype, nodata=None):
        path = tmp_path / name
        profile = dict(driver="GTiff", count=1, height=array.shape[0], width=array.shape[1], dtype=dtype,
                       crs="EPSG:5070", transform=from_origin(0, 0, 30, 30), nodata=nodata)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array.astype(dtype), 1)
        return path

    fuels = np.array([[91, 101, 102], [165, 165, -9999]])
    stack = TifStack(write("landfire_LF2016_FBFM40.tif", fuels, "int16", nodata=-9999))
    layer = stack.layers()[0]
    assert layer.style == "classes" and layer.palette == "fbfm40" and np.isnan(layer.read(0)[1, 2])
    from firevis import pictures

    legend = pictures.frame_picture(stack, layer, 0).legend.classes
    assert legend[165][0] == "TU5 Very high load dry timber-shrub" and legend[91][0] == "NB1 Urban / developed"
    cover = TifStack(write("landfire_LF2016_CC.tif", np.array([[0, 15], [25, 35]]), "int16"))
    assert cover.layers()[0].style == "values"  # canopy cover is a measurement, not classes
    dem = TifStack(write("dem_glo30.tif", np.random.default_rng(0).random((40, 2000)) * 3000, "float32"),
                   max_side=500)
    assert dem.layers()[0].read(0).shape == (10, 500) and "shown at 500 x 10" in dem.description


def test_slider_steps_long_series(tmp_path):
    pytest.importorskip("xarray")
    from firevis.stacks import NetcdfStack
    from firevis.viewer import StackViewer

    path = tmp_path / "gridmet_vs.nc"
    gridmet_nc(path, days=30)
    viewer = StackViewer(NetcdfStack(path))
    assert viewer._slider.layout.display == "" and viewer._slider.max == 29
    viewer._slider.value = 12
    assert viewer.frame == 12 and "<b>2019-07-09</b> · 13 / 30" in viewer._date.value
    viewer.step(1)
    assert viewer._slider.value == 13
    viewer.toggle_overlay()
    assert viewer.overlaid and viewer._slider.layout.display == "none"
    assert "day 30, lat 3, lon 4" in viewer._about.value
    viewer.close()


# ----- explorer -----------------------------------------------------------------------


def test_explorer_opens_every_fah_format(tmp_path, mirror):
    pytest.importorskip("xarray")
    pytest.importorskip("rasterio")
    pytest.importorskip("pandas")
    from fakes import FakeApi

    from firevis import DatasetExplorer, HubSource
    from firevis.stacks import NetcdfStack
    from firevis.viewer import StackViewer

    nc = tmp_path / "gridmet_vs.nc"
    gridmet_nc(nc)
    contents = {
        "fires/F1/gridmet/gridmet_vs.nc": nc.read_bytes(),
        "fires/F1/perimeters/records.geojson": json.dumps(collection([perimeter(0, 1562814000000)])).encode(),
        "fires/F1/firms/VIIRS_SNPP_SP.csv.gz": gzip.compress(FIRMS.encode()),
        "fires/F1/osm/osm_2017-01-01.osm.pbf": osm_pbf(),
    }
    api = FakeApi(tree={"me/FAH": [f"{p}:{len(d)}" for p, d in contents.items()]}, contents=contents)
    explorer = DatasetExplorer(source=HubSource("me", api=api, mirror_dir=mirror))
    from test_imagery import open_row  # noqa: E402  (imported here: it needs rasterio)

    open_row(explorer, "FAH"); open_row(explorer, "fires"); open_row(explorer, "F1")
    rows = {row.children[0].description.split("  ", 1)[1]: row.children[1].value
            for row in explorer._list.children if "fx-row" in row._dom_classes}
    assert set(rows) == {"firms", "gridmet", "osm", "perimeters"}
    open_row(explorer, "gridmet"); open_row(explorer, "gridmet_vs.nc")
    assert isinstance(explorer._viewer, StackViewer) and isinstance(explorer._viewer.stack, NetcdfStack)
    explorer.up(); open_row(explorer, "perimeters"); open_row(explorer, "records.geojson")
    assert explorer._viewer is None and "fx-map" in explorer._preview_body.value
    assert "GeoJSON map" in explorer._preview_title.value
    assert explorer._preview.layout.height is None  # a map gets the room it needs, like the image viewer
    explorer.up(); open_row(explorer, "firms")
    assert "CSV table (gzip)" in explorer._list.children[0].children[1].value
    open_row(explorer, "VIIRS_SNPP_SP.csv.gz")
    assert "3 points" in explorer._preview_body.value
    explorer.up(); open_row(explorer, "osm"); open_row(explorer, "osm_2017-01-01.osm.pbf")
    assert "Roads 1" in explorer._preview_body.value and "OpenStreetMap extract" in explorer._preview_title.value
