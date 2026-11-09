import html
import json

import pytest
from IPython.display import HTML, Markdown

from firevis import viewers
from firevis.hub import FILE, Entry
from firevis.viewers import file_kind, render


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


@pytest.mark.parametrize("name, data, kind", [
    ("a.npz", b"PK\x03\x04", "numpy"),
    ("a.json", "{}", "json"),
    ("README.md", "# hi", "markdown"),
    ("fire_all.csv", "a,b\n1,2\n", "table"),
    ("notes.txt", "hi", "text"),
    (".gitattributes", "*.npz filter=lfs", "text"),
    ("x.tif", b"II*\x00", "binary"),
    ("data", "some text", "text"),          # no extension: sniffed
    ("blob", b"\x00\x01\x02", "binary"),
    ("odd.xyz", "plain words", "text"),
])
def test_file_kind(tmp_path, name, data, kind):
    assert file_kind(write(tmp_path, name, data)) == kind


def test_json_is_pretty_printed(tmp_path):
    items = render(write(tmp_path, "c.json", json.dumps({"window": [1, 2], "name": "x"})))
    assert isinstance(items[0], HTML) and "c.json" in items[0].data
    assert '"window": [\n    1,' in items[1].data


def test_invalid_json_shown_as_text(tmp_path):
    items = render(write(tmp_path, "bad.json", "{not json"))
    assert "{not json" in items[1].data


def test_text_is_escaped(tmp_path):
    items = render(write(tmp_path, "t.txt", "<script>alert(1)</script>"))
    assert "<script>" not in items[1].data and "&lt;script&gt;" in items[1].data


def test_long_text_is_cut(tmp_path, monkeypatch):
    monkeypatch.setattr(viewers, "MAX_CHARS", 10)
    items = render(write(tmp_path, "long.txt", "x" * 50))
    assert items[1].data.split(">", 1)[1].count("x") == 10  # skip the style attribute
    assert "first 10 characters of 50 bytes" in items[2].data


def test_markdown_front_matter_becomes_code(tmp_path):
    items = render(write(tmp_path, "README.md", "---\nlicense: cc-by-4.0\n---\n# TS-SatFire\n"))
    assert isinstance(items[1], Markdown)
    assert items[1].data == "```yaml\nlicense: cc-by-4.0\n```\n\n# TS-SatFire\n"


def test_csv_is_a_table(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    path = write(tmp_path, "f.csv", "id,year\n" + "".join(f"{i},2017\n" for i in range(5)))
    frame = render(path)[1]
    assert isinstance(frame, pd.DataFrame) and list(frame.columns) == ["id", "year"]
    monkeypatch.setattr(viewers, "CSV_ROWS", 3)
    items = render(path)
    assert len(items[1]) == 3 and "first 3 rows" in items[2].data


def test_npz_and_binary_are_only_saved(tmp_path):
    npz = render(write(tmp_path, "p1.npz", b"PK\x03\x04"))
    assert "No preview yet" in npz[1].data
    assert html.escape(repr(str(tmp_path / "p1.npz"))) in npz[1].data
    tif = render(write(tmp_path, "x.tif", b"II*\x00"))
    assert "No preview for .tif files yet" in tif[1].data


def test_header_uses_entry(tmp_path):
    path = write(tmp_path, "b.txt", "hello")
    header = render(path, Entry("b.txt", FILE, "me/alpha", "deep/a/b.txt", 5))[0]
    assert "5 bytes" in header.data and str(path) in header.data
