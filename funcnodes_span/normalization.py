from funcnodes import NodeDecorator, Shelf
import numpy as np
from enum import Enum


class NormMode(Enum):
    ZERO_ONE = "zero_one"
    MINUS_ONE_ONE = "minus_one_one"
    SUM_ABS = "sum_abs"
    SUM = "sum"
    EUCLIDEAN = "euclidean"
    MEAN_STD = "mean_std"
    MAX = "max"

    @classmethod
    def default(cls) -> "NormMode":
        """Returns the default normalization mode."""
        return cls.ZERO_ONE


@NodeDecorator(id="span.basics.norm", name="Normalizations")
def _norm(array: np.ndarray, mode: NormMode = NormMode.default()) -> np.ndarray:
    # """
    # Apply different normalizations to the array.

    # Args:
    #     array (np.ndarray): The input array to be normalized.
    #     mode (NormMode): The normalization mode to apply. Defaults to NormMode.ZERO_ONE.

    # Returns:
    #     np.ndarray: The normalized array.

    # Raises:
    #     ValueError: If an unsupported normalization mode is provided.
    # """
    normalization_methods = {
        NormMode.ZERO_ONE: lambda x: (x - np.amin(x)) / (np.amax(x) - np.amin(x)),
        NormMode.MINUS_ONE_ONE: lambda x: 2
        * ((x - np.amin(x)) / (np.amax(x) - np.amin(x)))
        - 1,
        NormMode.SUM_ABS: lambda x: x / np.abs(x).sum(),
        NormMode.SUM: lambda x: x / x.sum(),
        NormMode.EUCLIDEAN: lambda x: x / np.sqrt((x**2).sum()),
        NormMode.MEAN_STD: lambda x: (x - x.mean()) / x.std(),
        NormMode.MAX: lambda x: x / x.max(),
    }
    if mode not in normalization_methods:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    return normalization_methods[mode](array)


NORM_NODE_SHELF = Shelf(
    nodes=[_norm],
    subshelves=[],
    name="Normalization",
    description="Normalization of the spectra",
)
