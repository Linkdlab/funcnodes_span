from typing import Dict, Callable, Union
from funcnodes import NodeDecorator, Shelf
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d, median_filter
import warnings
import funcnodes as fn

warnings.filterwarnings("ignore")


class SmoothMode(fn.DataEnum):
    SAVITZKY_GOLAY = "savgol"
    GAUSSIAN = "gaussian"
    MOVING_AVERAGE = "ma"
    EXPONENTIAL_MOVING_AVERAGE = "ema"
    MEDIAN = "median"
    SPIKE_REMOVE = "remove_spike"


def smooth_savgol(x: np.ndarray, window: int) -> np.ndarray:
    return savgol_filter(x, window, 2)


def smooth_gaussian(x: np.ndarray, window: int) -> np.ndarray:
    return gaussian_filter1d(x, window)


def smooth_ma(x: np.ndarray, window: int) -> np.ndarray:
    if x.ndim > 1:
        n, m = x.shape
        result = np.zeros((n, m))
        for i in range(n):
            result[i, :] = np.convolve(x[i, :], np.ones(window) / window, mode="same")
        return result
    else:
        return np.convolve(x, np.ones(window) / window, mode="same")


def smooth_ema(x: np.ndarray, window: int) -> np.ndarray:
    if x.ndim > 1:
        n, m = x.shape
        result = np.zeros((n, m))
        for i in range(n):
            result[i, :] = pd.Series(x[i, :]).ewm(span=window).mean().values
        return result
    else:
        return pd.Series(x).ewm(span=window).mean().values


def smooth_median(x: np.ndarray, window: int) -> np.ndarray:
    return medfilt(x, window)


def _remove_spikes_1d(y: np.ndarray, threshold: float, window: int) -> np.ndarray:
    """
    Whitaker & Hayes (2018) modified-z-score cosmic spike removal.

    1. Compute first differences: delta_i = y_i - y_{i-1}
    2. Modified z-score on delta: z_i = 0.6745 * (delta_i - median(delta)) / MAD(delta)
    3. Flag points where |z_i| > threshold as spikes
    4. Replace flagged points with the median of a local window of
       neighbouring NON-spike values.

    Parameters
    ----------
    y         : 1D spectrum
    threshold : modified z-score cutoff (typically 5-10; lower = more aggressive)
    window    : half-width of the window used to compute the replacement value
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    delta = np.diff(y)
    median_delta = np.median(delta)
    mad = np.median(np.abs(delta - median_delta))
    if mad == 0:
        return y.copy()  # nothing to do — perfectly smooth diff

    z = 0.6745 * (delta - median_delta) / mad
    # a spike at index i produces a large |z| at delta[i-1] (jump up)
    # AND a large |z| at delta[i] (jump back down) -> flag both endpoints
    spike_mask = np.zeros(n, dtype=bool)
    flagged = np.where(np.abs(z) > threshold)[0]
    for idx in flagged:
        spike_mask[idx] = True  # point before the jump
        spike_mask[idx + 1] = True  # point after the jump

    cleaned = y.copy()
    spike_positions = np.where(spike_mask)[0]
    for idx in spike_positions:
        lo = max(0, idx - window)
        hi = min(n, idx + window + 1)
        local = y[lo:hi]
        local_mask = ~spike_mask[lo:hi]
        if local_mask.sum() == 0:
            continue  # entire window is spikes — leave as-is (rare, widen window)
        cleaned[idx] = np.median(local[local_mask])

    return cleaned


def smooth_remove_cosmic_spikes(x: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Remove cosmic-ray spikes via residual deviation from a robust local median.

    A robust self-estimate of each spectrum is computed using a median filter
    of width ``window``. The residual (input - smoothed) is then converted to
    a modified z-score (Iglewicz & Hoaglin, using the 0.6745 MAD scaling
    factor), and points with |z| > 3.5 — the standard outlier cutoff for the
    modified z-score — are replaced by their corresponding median-filtered
    value.

    This is threshold-free in the sense that the spike cutoff (3.5) is a
    fixed, statistically-derived constant rather than a user-tuned parameter;
    the only remaining parameter is the median-filter window size, which
    controls how much real spectral structure is treated as "local trend"
    versus "anomaly."

    Parameters
    ----------
    x : np.ndarray
        Input spectrum/spectra.
        - 1D, shape ``(n_points,)``: a single spectrum.
        - 2D, shape ``(n_spectra, n_points)``: a batch of spectra. The median
          filter is applied along the last axis only (``size=(1, window)``),
          so each spectrum is processed independently — a spike in one row
          cannot affect the smoothing of another.
    window : int, default=5
        Width of the median filter used to compute the robust local
        self-estimate. Must be odd and >= 3. Larger windows treat broader
        features as "trend" (more tolerant of wide real peaks but slower to
        react to genuine sharp features); smaller windows are more sensitive
        but risk flagging steep real peak edges as spikes.

    Returns
    -------
    np.ndarray
        Array of the same shape as ``x``, with spike points replaced by their
        local median-filtered value. Rows (in the 2D case) or the whole
        array (in the 1D case) with zero residual MAD — i.e. perfectly flat
        signals — are returned unchanged, since no statistically meaningful
        z-score can be computed.

    Raises
    ------
    ValueError
        If ``x`` is not 1D or 2D, or if ``window`` is not odd and >= 3.

    Notes
    -----
    - This method operates on a single spectrum/acquisition. If repeated
      acquisitions of the same sample are available, a median-across-scans
      approach is strictly more robust (a cosmic ray is non-reproducible
      across acquisitions) and should be preferred when feasible.
    - The 3.5 cutoff is a standard convention for the modified z-score, not
      an arbitrary default; it is not currently exposed as a parameter.


    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        size = window
    elif x.ndim == 2:
        size = (1, window)
    else:
        raise ValueError(f"Expected 1D or 2D input, got shape {x.shape}")

    if window < 3 or window % 2 == 0:
        raise ValueError(f"window must be odd and >= 3, got {window}")

    smooth = median_filter(x, size=size)
    resid = x - smooth
    if x.ndim == 1:
        mad = np.median(np.abs(resid - np.median(resid)))
        if mad == 0:
            return x.copy()
        z = 0.6745 * (resid - np.median(resid)) / mad
        spike_mask = np.abs(z) > 3.5  # statistical, not arbitrary, cutoff
        cleaned = x.copy()
        cleaned[spike_mask] = smooth[spike_mask]
        return cleaned

    # 2D: per-row median/MAD, vectorized
    med = np.median(resid, axis=1, keepdims=True)
    mad = np.median(np.abs(resid - med), axis=1, keepdims=True)

    cleaned = x.copy()
    safe_mad = np.where(
        mad == 0, 1.0, mad
    )  # avoid div-by-zero; rows with mad==0 get no spikes flagged
    z = 0.6745 * (resid - med) / safe_mad
    spike_mask = (np.abs(z) > 3.5) & (mad != 0)
    cleaned[spike_mask] = smooth[spike_mask]
    return cleaned


_SMOOTHING_MAPPER: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    SmoothMode.SAVITZKY_GOLAY.value: smooth_savgol,
    SmoothMode.GAUSSIAN.value: smooth_gaussian,
    SmoothMode.MOVING_AVERAGE.value: smooth_ma,
    SmoothMode.EXPONENTIAL_MOVING_AVERAGE.value: smooth_ema,
    SmoothMode.MEDIAN.value: smooth_median,
    SmoothMode.SPIKE_REMOVE.value: smooth_remove_cosmic_spikes,
}


@NodeDecorator(
    "span.basics.smooth",
    name="Smoothing",
    outputs=[{"name": "smoothed"}],
)
def _smooth(
    y: np.ndarray,
    mode: SmoothMode = SmoothMode.SAVITZKY_GOLAY,
    window: Union[float, int] = 5,
    x: np.ndarray = None,
) -> np.ndarray:
    # """
    # Apply different smoothing techniques to the input array.
    # the window is the number of points to consider for the smoothing.
    # If x is provided, the window is in x units and is converted to points using the median x difference.

    # Args:
    #     y (np.ndarray): The input array to be smoothed.
    #     mode (SmoothMode): The smoothing mode to apply. Defaults to SmoothMode.SAVITZKY_GOLAY.
    #     window (int): The window size for the smoothing function. Defaults to 5.
    #     x (np.ndarray): The x values of the input array. Defaults to None.

    # Returns:
    #     np.ndarray: The smoothed array.

    # Raises:
    #     ValueError: If an unsupported smoothing mode is provided.
    # """
    mode = SmoothMode.v(mode)

    if mode not in _SMOOTHING_MAPPER.keys():
        raise ValueError(f"Unsupported smoothing mode: {mode}")
    y = np.asarray(y)
    if x is not None:
        x = np.asarray(x)
        med_xdiff = np.nanmedian(np.diff(x))
        window = window / med_xdiff
    window = int(window)
    if window == 0:
        return y.copy()
    return _SMOOTHING_MAPPER[mode](y, window)


SMOOTH_NODE_SHELF = Shelf(
    nodes=[_smooth],
    subshelves=[],
    name="Smoothing",
    description="Smoothing of the spectra",
)
