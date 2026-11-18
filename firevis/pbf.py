"""Read an OpenStreetMap .osm.pbf file without osmium or pyrosm: its header, counts and human features.

An .osm.pbf file is a run of blocks, each a 4-byte length, a BlobHeader and a
Blob (protobuf). The first block's Blob ("OSMHeader") holds the box, the
writing program and the replication timestamp; every later block ("OSMData")
is a zlib-compressed PrimitiveBlock of nodes, ways or relations. read_pbf
decodes them all (packed varints are decoded with numpy, so a 25 MB extract
takes seconds) and keeps the ways a fire map needs: roads, railways, power
lines, buildings and water, with their coordinates.
"""

from __future__ import annotations

import struct
import zlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# Ways drawn on the map, by their first matching tag key: (label, colour). Closed
# buildings and water are drawn as areas, everything else as lines.
CATEGORIES = {
    "highway": ("Roads", "#6e6e73"),
    "railway": ("Railways", "#3d3d40"),
    "power": ("Power lines", "#c4314b"),
    "building": ("Buildings", "#8b1e3f"),
    "waterway": ("Waterways", "#1c7ed6"),
    "natural": ("Water", "#4dabf7"),  # only natural=water
}
AREAS = ("building", "natural")


@dataclass
class PbfSummary:
    """What an .osm.pbf file holds: its header, counts, common tags and the ways to draw."""

    bbox: Optional[Tuple[float, float, float, float]] = None  # west, south, east, north (degrees)
    required_features: List[str] = field(default_factory=list)
    optional_features: List[str] = field(default_factory=list)
    writing_program: str = ""
    source: str = ""
    replication_timestamp: Optional[datetime] = None
    data_blocks: int = 0
    data_bytes: int = 0
    decoded: bool = False  # data blocks were decoded (counts and ways below)
    nodes: int = 0
    ways: int = 0
    relations: int = 0
    way_keys: Counter = field(default_factory=Counter)  # tag key -> ways carrying it
    drawn: Dict[str, List[np.ndarray]] = field(default_factory=dict)  # category -> (n, 2) lon/lat arrays

    @property
    def extent(self) -> Optional[Tuple[float, float, float, float]]:
        """west, south, east, north of the drawn ways (the header box when there is one)."""
        if self.bbox:
            return self.bbox
        arrays = [a for ways in self.drawn.values() for a in ways]
        if not arrays:
            return None
        xy = np.concatenate(arrays)
        return float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())


def _varint(data: bytes, pos: int) -> Tuple[int, int]:
    result = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _fields(data: bytes) -> Iterator[Tuple[int, int, object]]:
    """(field number, wire type, value) of a protobuf message: ints for varints, bytes for strings."""
    pos = 0
    while pos < len(data):
        key, pos = _varint(data, pos)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(data, pos)
        elif wire == 2:
            length, pos = _varint(data, pos)
            value, pos = data[pos:pos + length], pos + length
        elif wire == 1:
            value, pos = data[pos:pos + 8], pos + 8
        elif wire == 5:
            value, pos = data[pos:pos + 4], pos + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield number, wire, value


def _zigzag(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def _blobs(path: Path, read_all: bool = False) -> Iterator[Tuple[str, int, Optional[bytes]]]:
    """(blob type, blob size, blob bytes or None) for every block; only the first is read unless read_all."""
    with open(path, "rb") as f:
        first = True
        while True:
            head = f.read(4)
            if not head:
                return
            if len(head) < 4:
                raise ValueError("Truncated .osm.pbf: a block length is cut short")
            (size,) = struct.unpack(">I", head)
            if size > 64 * 1024:
                raise ValueError("Not an .osm.pbf file: block header too large")
            header = dict((n, v) for n, _, v in _fields(f.read(size)))
            kind = header.get(1, b"").decode("utf-8", "replace")
            datasize = int(header.get(3, 0))
            if first or read_all:
                yield kind, datasize, f.read(datasize)
                first = False
            else:
                f.seek(datasize, 1)
                yield kind, datasize, None


def _blob_data(blob: bytes) -> bytes:
    parts: Dict[int, object] = dict((n, v) for n, _, v in _fields(blob))
    if 1 in parts:
        return parts[1]  # raw
    if 3 in parts:
        return zlib.decompress(parts[3])
    raise ValueError("This .osm.pbf header is compressed with something other than zlib")


def read_pbf(path, decode: bool = True) -> PbfSummary:
    """Decode an .osm.pbf file: its header, then (decode=True) every data block's nodes and ways."""
    summary = PbfSummary()
    state = _State()
    for index, (kind, size, blob) in enumerate(_blobs(Path(path), read_all=decode)):
        if index == 0:
            if kind != "OSMHeader":
                raise ValueError(f"Not an .osm.pbf file: first block is {kind!r}, not 'OSMHeader'")
            for number, _, value in _fields(_blob_data(blob)):
                if number == 1:  # HeaderBBox, nanodegrees, sint64
                    box = {n: _zigzag(v) / 1e9 for n, _, v in _fields(value)}
                    summary.bbox = (box.get(1, 0.0), box.get(4, 0.0), box.get(2, 0.0), box.get(3, 0.0))
                elif number == 4:
                    summary.required_features.append(value.decode("utf-8", "replace"))
                elif number == 5:
                    summary.optional_features.append(value.decode("utf-8", "replace"))
                elif number == 16:
                    summary.writing_program = value.decode("utf-8", "replace")
                elif number == 17:
                    summary.source = value.decode("utf-8", "replace")
                elif number == 32:
                    summary.replication_timestamp = datetime.fromtimestamp(value, tz=timezone.utc)
        elif kind == "OSMData":
            summary.data_blocks += 1
            summary.data_bytes += size
            if decode:
                _decode_block(_blob_data(blob), summary, state)
    if decode:
        summary.decoded = True
        _resolve_ways(summary, state)
    return summary


# ----- data blocks --------------------------------------------------------------


def packed_varints(data: bytes, signed: bool = False) -> np.ndarray:
    """Decode a packed run of protobuf varints at once with numpy (zigzag when signed)."""
    raw = np.frombuffer(data, dtype=np.uint8)
    if not raw.size:
        return np.zeros(0, dtype=np.int64)
    last = (raw & 0x80) == 0
    ends = np.flatnonzero(last)
    starts = np.concatenate(([0], ends[:-1] + 1))
    position = np.arange(raw.size) - np.repeat(starts, ends - starts + 1)
    parts = (raw & 0x7F).astype(np.uint64) << (7 * position).astype(np.uint64)
    values = np.add.reduceat(parts, starts)
    if signed:
        return ((values >> np.uint64(1)).astype(np.int64)) ^ -((values & np.uint64(1)).astype(np.int64))
    return values.astype(np.int64)


class _State:
    """Node coordinates and tagged ways collected over the blocks, resolved at the end."""

    def __init__(self):
        self.node_ids: List[np.ndarray] = []
        self.node_lonlat: List[np.ndarray] = []
        self.ways: List[Tuple[str, np.ndarray]] = []  # (category, node ids)


def _category(keys: List[str], tags: Dict[str, str]) -> Optional[str]:
    for key in CATEGORIES:
        if key in tags and (key != "natural" or tags[key] == "water"):
            return key
    return None


def _decode_block(data: bytes, summary: PbfSummary, state: _State) -> None:
    strings: List[str] = []
    groups: List[bytes] = []
    granularity, lat_offset, lon_offset = 100, 0, 0
    for number, _, value in _fields(data):
        if number == 1:
            strings = [s.decode("utf-8", "replace") for n, _, s in _fields(value) if n == 1]
        elif number == 2:
            groups.append(value)
        elif number == 17:
            granularity = value
        elif number in (19, 20):  # int64: a negative offset arrives as a 64-bit two's complement
            offset = value - (1 << 64) if value >= 1 << 63 else value
            lat_offset, lon_offset = (offset, lon_offset) if number == 19 else (lat_offset, offset)

    def scale(raw, offset):
        return (offset + granularity * raw.astype(np.float64)) * 1e-9

    for group in groups:
        for number, _, value in _fields(group):
            if number == 2:  # DenseNodes: delta-coded ids and coordinates
                fields = {n: v for n, _, v in _fields(value)}
                ids = np.cumsum(packed_varints(fields.get(1, b""), signed=True))
                lat = np.cumsum(packed_varints(fields.get(8, b""), signed=True))
                lon = np.cumsum(packed_varints(fields.get(9, b""), signed=True))
                summary.nodes += len(ids)
                state.node_ids.append(ids)
                state.node_lonlat.append(np.column_stack([scale(lon, lon_offset), scale(lat, lat_offset)]))
            elif number == 1:  # a plain Node
                fields = {n: v for n, _, v in _fields(value)}
                summary.nodes += 1
                state.node_ids.append(np.array([_zigzag(fields.get(1, 0))]))
                state.node_lonlat.append(np.array([[scale(np.array(_zigzag(fields.get(9, 0))), lon_offset),
                                                    scale(np.array(_zigzag(fields.get(8, 0))), lat_offset)]]))
            elif number == 3:  # Way: tags and delta-coded node references
                summary.ways += 1
                fields = {n: v for n, _, v in _fields(value)}
                keys = [strings[k] for k in packed_varints(fields.get(2, b""))]
                vals = [strings[v] for v in packed_varints(fields.get(3, b""))]
                summary.way_keys.update(keys)
                category = _category(keys, dict(zip(keys, vals)))
                if category:
                    state.ways.append((category, np.cumsum(packed_varints(fields.get(8, b""), signed=True))))
            elif number == 4:
                summary.relations += 1


def _resolve_ways(summary: PbfSummary, state: _State) -> None:
    """Turn the kept ways' node ids into lon/lat arrays (ways with missing nodes are dropped)."""
    if not state.ways or not state.node_ids:
        return
    ids = np.concatenate(state.node_ids)
    lonlat = np.concatenate(state.node_lonlat)
    order = np.argsort(ids, kind="stable")
    ids, lonlat = ids[order], lonlat[order]
    for category, refs in state.ways:
        where = np.searchsorted(ids, refs)
        where = np.clip(where, 0, len(ids) - 1)
        if len(refs) < 2 or not np.array_equal(ids[where], refs):
            continue
        summary.drawn.setdefault(category, []).append(lonlat[where])
