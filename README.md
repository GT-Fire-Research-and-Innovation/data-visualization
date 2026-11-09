# data-visualization
repository for visualizing ts-satfire data before and after processing.

## Dataset explorer

A Finder / File Explorer style browser for the datasets at
[huggingface.co/SamuelWu318](https://huggingface.co/SamuelWu318). The root is the Hub
account, each dataset is a top-level folder, and folders load from the Hub as you open them.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GT-Fire-Research-and-Innovation/data-visualization/blob/main/colab/explorer.ipynb)

The code lives in the `firevis` package here; a Colab notebook only needs:

```python
%pip install -q --force-reinstall --no-deps git+https://github.com/GT-Fire-Research-and-Innovation/data-visualization.git

from firevis import open_explorer
explorer = open_explorer("https://huggingface.co/SamuelWu318")
```

- Click a folder (or a dataset in the sidebar, or a part of the path bar) to open it.
- ◀ ▶ go back and forward, ▲ goes to the enclosing folder, ⟳ re-fetches the folder.
- Click a file to select it. Nothing else happens yet; to hook something up later:

```python
@explorer.on_file_click
def show(entry):          # entry.repo_id, entry.path, entry.size, entry.url
    print(entry.url)      # prints and plots appear under the explorer
```

## Layout

| Path | What |
| --- | --- |
| `firevis/hub.py` | Lists Hub datasets and their folders one level at a time (cached) |
| `firevis/explorer.py` | The explorer window (ipywidgets) |
| `colab/explorer.ipynb` | Colab caller: install + open the explorer |
| `tests/` | `pip install -e '.[test]' && pytest` |

Links: [visualization notebook](https://colab.research.google.com/drive/1eGlHT7b1RYUqGIe4Wtf9xmc2DQlJlQ7N?usp=sharing) ·
[datasets](https://huggingface.co/SamuelWu318) (also in `notebooks`).
