# data-visualization
repository for visualizing ts-satfire data before and after processing.

## Dataset explorer

A Finder / File Explorer style browser for the datasets at
[huggingface.co/SamuelWu318](https://huggingface.co/SamuelWu318). The root is the Hub
account and each dataset is a top-level folder.

Every file and folder name loads once when the explorer opens (about 13,000 entries,
a few seconds), so browsing after that is instant and folder sizes are totals.

File contents load lazily. The first time a file is clicked it is downloaded into a
local mirror of the Hub layout, `hf_mirror/<owner>/<dataset>/<path>` (in Colab:
`/content/hf_mirror/SamuelWu318/ts-satfire/README.md`, ...). After that the saved copy
is used, and saved files get a ✓ in the list. Opened files show in a preview pane
inside the explorer window (right of the list, or below it in a narrow window); ✕ closes it.

| Clicked file | What happens |
| --- | --- |
| `.txt`, `.json`, `.md`, `.csv`, `.gitattributes`, other text | Saved, then shown in the preview pane (JSON pretty-printed, Markdown rendered, CSV as a table) |
| `.npz` | Saved, then opened in the image viewer (below) |
| `.tif`, other binary | Saved only (open the fire folder to see its GeoTIFFs) |
| Anything over 1 GB (the `.zip` archives, up to 43.7 GB) | The pane asks first, with a Download button |

### Fire folders and window archives

Clicking a **fire folder** (one with `VIIRS_Day/` plus `FirePred/` or `VIIRS_Night/`, like
`ts-satfire/ts-satfire/20562846`) opens it as usual and also downloads every GeoTIFF in it
(in parallel, with a count in the pane; folders over 1 GB ask first). The pane then shows
an image viewer:

- **Fire layers are overlaid**: active fire, NIFC perimeters and burned area (active fire +
  NIFC, coloured by the first day each pixel burned). Every day is drawn at once, slightly
  transparent, from light (first day) to dark (last day), over a faded VIIRS false-colour
  image (Background: first day / last day / none).
- **Everything else steps day by day**: VIIRS bands and false colour, night bands, all 19
  FirePred bands, ESRI and MODIS land cover. ◀ ▶ at the bottom change the day; the date is
  in the bottom middle.
- **View: Fire area** zooms to the fire (ignoring stray detections); **Whole image** shows
  the full tile.

Clicking a **window `.npz`** opens the same viewer with every channel (named, for the
27-channel archives) plus the next-day target. Every layer gets arrows and a date (taken
from the fire's raw VIIRS_Day files when the fire is in this account; otherwise Day 1, 2,
...), and a window picker when the archive has several windows. Any other `.npz` shows
each of its arrays.

Two corners on every image: **top-left** turns overlay on/off (all days at once vs one
day at a time), **top-right** makes the image big (the pane then spans the window) or small.

The code lives in the `firevis` package here; a Colab notebook only needs:

```python
%pip install -q rasterio backports.zstd
%pip install -q --force-reinstall --no-deps git+https://github.com/SamuelWu318/data-visualization.git

from firevis import open_explorer
explorer = open_explorer("https://huggingface.co/SamuelWu318")
```

After installing a new version into a runtime that already imported `firevis`, use
**Runtime > Restart session** and run both cells again; otherwise the old code keeps running.

- Click a folder (or a dataset in the sidebar, or a part of the path bar) to open it.
- ◀ ▶ go back and forward, ▲ goes to the enclosing folder, ⟳ re-fetches the names
  (the current dataset, or everything at the root).
- Click a file to open it (see the table above). To add your own handling:

```python
@explorer.on_file_click
def show(entry):                   # entry.repo_id, entry.path, entry.size, entry.url
    path = explorer.download(entry)    # the mirror copy; downloads only the first time
    print(path)                        # prints and plots appear under the explorer
```

From code, `explorer.open_file(entry)` opens a file in the pane and returns its path.

Colab clears `/content` when the runtime is deleted. To keep downloads between runtimes,
put the mirror on Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
explorer = open_explorer(mirror_dir="/content/drive/MyDrive/hf_mirror")
```

Gated datasets (`ts-satfire-processed`) list without a token but need `HF_TOKEN` (a Colab
secret works) and accepted terms to download.

## Layout

| Path | What |
| --- | --- |
| `firevis/hub.py` | Loads every dataset's file and folder names; downloads files into the mirror |
| `firevis/viewers.py` | Turns an opened file into HTML for the preview pane |
| `firevis/stacks.py` | Reads fire folders (GeoTIFFs) and `.npz` files into layers of dated frames |
| `firevis/render.py` | Colour ramps, overlays, land-cover palettes, PNG encoding (numpy only) |
| `firevis/viewer.py` | The image viewer: layer picker, corners, legend, ◀ date ▶ |
| `firevis/explorer.py` | The explorer window (ipywidgets) |
| `tests/` | `pip install -e '.[test]' && pytest` |

Links: [visualization notebook](https://colab.research.google.com/drive/1eGlHT7b1RYUqGIe4Wtf9xmc2DQlJlQ7N?usp=sharing) ·
[datasets](https://huggingface.co/SamuelWu318) (also in `notebooks`).
