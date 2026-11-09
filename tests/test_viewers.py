import json

import pytest

from firevis import viewers
from firevis.viewers import file_kind, markdown_to_html, to_html


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
    body = to_html(write(tmp_path, "c.json", json.dumps({"window": [1, 2], "name": "x"})))
    assert body.startswith('<pre class="fx-code">')
    assert '"window": [\n    1,' in body


def test_invalid_json_shown_as_text(tmp_path):
    assert "{not json" in to_html(write(tmp_path, "bad.json", "{not json"))


def test_text_is_escaped(tmp_path):
    body = to_html(write(tmp_path, "t.txt", "<script>alert(1)</script>"))
    assert "<script>" not in body and "&lt;script&gt;" in body


def test_long_text_is_cut(tmp_path, monkeypatch):
    monkeypatch.setattr(viewers, "MAX_CHARS", 10)
    body = to_html(write(tmp_path, "long.txt", "x" * 50))
    shown = body.split(">", 1)[1].split("</pre>", 1)[0]
    assert shown == "x" * 10
    assert "first 10 characters of 50 bytes" in body


def test_markdown_is_rendered_with_front_matter_as_code(tmp_path):
    body = to_html(write(tmp_path, "README.md",
                         "---\nlicense: cc-by-4.0\n---\n# TS-SatFire\n\nSome *text*.\n"))
    assert body.startswith('<div class="fx-md">')
    assert "<h1>TS-SatFire</h1>" in body and "<em>text</em>" in body
    assert "license: cc-by-4.0" in body and "<pre>" in body


def test_markdown_without_a_library_falls_back_to_text(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_markdown(name, *args, **kwargs):
        if name in ("markdown", "mistune"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_markdown)
    assert markdown_to_html("# hi") == '<pre class="fx-code"># hi</pre>'


def test_csv_is_a_table(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    path = write(tmp_path, "f.csv", "id,year\n" + "".join(f"{i},2017\n" for i in range(5)))
    body = to_html(path)
    assert "<th>id</th>" in body and "<td>2017</td>" in body and "5 rows." in body
    monkeypatch.setattr(viewers, "CSV_ROWS", 3)
    body = to_html(path)
    assert body.count("<tr") == 4 and "first 3 rows" in body  # header + 3 rows


def test_npz_and_binary_are_only_saved(tmp_path):
    body = to_html(write(tmp_path, "p1.npz", b"PK\x03\x04"))
    assert "No preview yet" in body and "np.load(" in body and "p1.npz" in body
    assert "No preview for .tif files yet" in to_html(write(tmp_path, "x.tif", b"II*\x00"))
    empty_zip = to_html(write(tmp_path, "data", b"PK\x05\x06" + b"\x00" * 18))  # real Hub file
    assert "No preview for this file type yet" in empty_zip
