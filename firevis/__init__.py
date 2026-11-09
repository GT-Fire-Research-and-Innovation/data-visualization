"""Visualization tools for TS-SatFire data on the Hugging Face Hub.

    firevis.open_explorer()        browse every dataset like a file explorer
    firevis.load(link)             one fire folder or .npz, to list and plot
    firevis.compare(a, b)          several fires side by side
"""

from .explorer import DatasetExplorer, open_explorer
from .fire import ChannelError, Comparison, FireData, compare, load, parse_link
from .hub import DEFAULT_ROOT, Entry, HubSource

__all__ = ["DatasetExplorer", "open_explorer", "load", "compare", "FireData", "Comparison",
           "ChannelError", "parse_link", "DEFAULT_ROOT", "Entry", "HubSource"]
