import funcnodes as fn

from .normalization import NORM_NODE_SHELF as NORM
from .smoothing import SMOOTH_NODE_SHELF as SMOOTH
from .peak_analysis import PEAKS_NODE_SHELF as PEAK
from .baseline import BASELINE_NODE_SHELF as BASELINE

__version__ = "0.2.5"

NODE_SHELF = fn.Shelf(
    name="Spectral Analysis",
    description="Spectral analysis for funcnodes",
    nodes=[],
    subshelves=[NORM, SMOOTH, BASELINE, PEAK],
)
