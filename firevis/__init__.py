"""Visualization tools for TS-SatFire data on the Hugging Face Hub.

    firevis.open_explorer()        browse every dataset like a file explorer
    firevis.load(link)             one fire folder or .npz, to list and plot
    firevis.compare(a, b)          several fires side by side
    firevis.split_archive(path)    one window archive split into its fires (fire-split-zones)
"""

from .explorer import DatasetExplorer, open_explorer
from .fire import ChannelError, Comparison, FireData, compare, load, parse_link
from .hub import DEFAULT_ROOT, Entry, HubSource
from .split import FireSplit, SplitError, split_archive
from .split_viewer import SplitViewer

__all__ = ["DatasetExplorer", "open_explorer", "load", "compare", "FireData", "Comparison",
           "ChannelError", "parse_link", "DEFAULT_ROOT", "Entry", "HubSource",
           "split_archive", "FireSplit", "SplitError", "SplitViewer"]
