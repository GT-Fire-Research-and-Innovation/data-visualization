# data-visualization
repository for visualizing ts-satfire data before and after processing.

## Dataset explorer

A Finder / File Explorer style browser for the datasets at
[huggingface.co/SamuelWu318](https://huggingface.co/SamuelWu318). The root is the Hub
account and each dataset is a top-level folder.

Every file and folder name loads once when the explorer opens (about 13,000 entries,
a few seconds), so browsing after that is instant and folder sizes are totals. File
contents load lazily: nothing is downloaded until code asks for a file.

The code lives in the `firevis` package here; a Colab notebook only needs:

```python
%pip install -q --force-reinstall --no-deps git+https://github.com/GT-Fire-Research-and-Innovation/data-visualization.git

from firevis import open_explorer
explorer = open_explorer("https://huggingface.co/SamuelWu318")
```

- Click a folder (or a dataset in the sidebar, or a part of the path bar) to open it.
- ◀ ▶ go back and forward, ▲ goes to the enclosing folder, ⟳ re-fetches the names
  (the current dataset, or everything at the root).
- Click a file to select it. Nothing else happens yet; to hook something up later:

```python
@explorer.on_file_click
def show(entry):                   # entry.repo_id, entry.path, entry.size, entry.url
    path = explorer.download(entry)    # lazy: fetched on first use, then cached locally
    print(path)                        # prints and plots appear under the explorer
```

Gated datasets (`ts-satfire-processed`) list without a token but need `HF_TOKEN` (a Colab
secret works) and accepted terms to download.

## Layout

| Path | What |
| --- | --- |
| `firevis/hub.py` | Loads every dataset's file and folder names; downloads files on demand |
| `firevis/explorer.py` | The explorer window (ipywidgets) |
| `tests/` | `pip install -e '.[test]' && pytest` |

Links: [visualization notebook](https://colab.research.google.com/drive/1eGlHT7b1RYUqGIe4Wtf9xmc2DQlJlQ7N?usp=sharing) ·
[datasets](https://huggingface.co/SamuelWu318) (also in `notebooks`).
