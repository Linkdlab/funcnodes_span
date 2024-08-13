import funcnodes as fn

from .normalization import NORM_NODE_SHELF as NORM
# from .basics import SMOOTH_NODE_SHELF as SMOOTH


__version__ = "0.1.0"

NODE_SHELF = fn.Shelf(
    name="Spectral Analysis",
    description="Spectral analysis for funcnodes",
    nodes=[],
    subshelves=[
        NORM,
    ],
)
