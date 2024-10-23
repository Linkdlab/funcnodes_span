from funcnodes import NodeDecorator, Shelf
import funcnodes as fn
import numpy as np
from exposedfunctionality import controlled_wrapper
from typing import Optional, List, Tuple, Union
from scipy.signal import find_peaks
from scipy.stats import norm
from scipy import signal, interpolate
from scipy.ndimage import gaussian_filter1d
import copy
import plotly.graph_objs as go
from .normalization import density_normalization
from .peaks import PeakProperties

from .fitting import fit_peaks, AUTOMODELMAP, fit_local_peak


@NodeDecorator(
    id="span.basics.peaks",
    name="Peak finder",
    outputs=[{"name": "peaks"}, {"name": "norm_x"}, {"name": "norm_y"}],
)
@controlled_wrapper(find_peaks, wrapper_attribute="__fnwrapped__")
def peak_finder(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    on: Optional[np.ndarray] = None,
    noise_level: Optional[int] = None,
    height: Optional[float] = None,
    threshold: Optional[float] = None,
    distance: Optional[float] = None,
    prominence: Optional[float] = None,
    width: Optional[float] = None,
    wlen: Optional[int] = None,
    rel_height: float = 0.05,
    width_at_rel_height: float = 0.5,
    plateau_size: Optional[int] = None,
) -> Tuple[List[PeakProperties], np.ndarray, np.ndarray]:
    """ """
    peak_lst = []

    y = np.array(y, dtype=float)
    _y = y
    if on is not None:
        y = on

    noise_level = int(noise_level) if noise_level is not None else None
    height = float(height) if height is not None else None
    threshold = float(threshold) if threshold is not None else None
    distance = float(distance) if distance is not None else None
    prominence = float(prominence) if prominence is not None else None
    width = float(width) if width is not None else None
    wlen = float(wlen) if wlen is not None else None
    width_at_rel_height = float(width_at_rel_height)
    plateau_size = float(plateau_size) if plateau_size is not None else None

    height = rel_height * np.max(y) if height is None else height
    noise_level = 5000 if noise_level is None else noise_level

    if x is not None:
        ox = x = np.array(x, dtype=float)
        x, y = density_normalization(
            x,
            y,
        )
        if on is not None:
            _, _y = density_normalization(
                ox,
                _y,
            )
        else:
            _y = y

        xdiff = x[1] - x[0]
        # if x is given width is based on the x scale and has to be converted to index
        if width is not None:
            width = width / xdiff

        # same for distance
        if distance is not None:
            distance = distance / xdiff

        # same for wlen
        if wlen is not None:
            wlen = wlen / xdiff

        # same for plateau_size
        if plateau_size is not None:
            plateau_size = plateau_size / xdiff
    else:
        x = np.arange(len(y))
    # Find the peaks in the copy of the input array
    peaks, _ = find_peaks(
        y,
        threshold=threshold,
        prominence=prominence,
        height=height,
        distance=distance,
        width=max(1, width) if width is not None else None,
        wlen=int(wlen) if wlen is not None else None,
        rel_height=width_at_rel_height,
        plateau_size=plateau_size,
    )

    # Calculate the standard deviation of peak prominences
    rnd = np.random.RandomState(42)
    # Fit a normal distribution to the input array
    mu, std = norm.fit(y)
    if peaks is not None:
        try:
            # Add noise to the input array
            noise = rnd.normal(mu / noise_level, std / noise_level, np.shape(y))
            y = y + noise

            # Find the minimums in the copy of the input array
            mins, _ = find_peaks(-1 * y)

            # Iterate over the peaks
            for peak in peaks:
                # Calculate the prominences of the peak
                # Find the right minimum of the peak
                right_min = mins[np.argmax(mins > peak)]
                if right_min < peak:
                    right_min = len(y) - 1

                try:
                    # Find the left minimum of the peak
                    left_min = np.array(mins)[np.where(np.array(mins) < peak)][-1]
                except IndexError:
                    left_min = 0

                if height is None:
                    # If no height is specified, append the peak bounds to the peak list
                    peak_lst.append([left_min, peak, right_min])

                else:
                    # If a height is specified, append the peak bounds to the peak list
                    # if the peak's value is greater than the height
                    if y[peak] > height:
                        peak_lst.append([left_min, peak, right_min])

        except ValueError:
            # If an error occurs when adding noise to the input array, add stronger noise and try again
            noise = rnd.normal(mu / 100, std / 100, np.shape(y))
            y = y + noise
            mins, _ = find_peaks(-1 * y)
            for peak in peaks:
                right_min = mins[np.argmax(mins > peak)]
                if right_min < peak:
                    right_min = len(y) - 1
                try:
                    left_min = np.array(mins)[np.where(np.array(mins) < peak)][-1]
                except IndexError:
                    left_min = 0
                if height is None:
                    # If no height is specified, append the peak bounds to the peak list
                    peak_lst.append([left_min, peak, right_min])
                else:
                    # If a height is specified, append the peak bounds to the peak list
                    # if the peak's value is greater than the height
                    if y[peak] > height:
                        peak_lst.append([left_min, peak, right_min])

    peak_properties_list = []

    for peak_nr, peak in enumerate(peak_lst):
        i_index, index, f_index = peak
        peak_properties = PeakProperties(
            id=str(peak_nr + 1),
            i_index=i_index,
            index=index,
            f_index=f_index,
            xfull=x,
            yfull=_y,
        )
        peak_properties_list.append(peak_properties)

    return peak_properties_list, x, y


@NodeDecorator(
    "span.basics.interpolation_1d",
    name="Interpolation 1D",
    outputs=[
        {
            "name": "x_interpolated",
        },
        {"name": "y_interpolated"},
    ],
)
def interpolation_1d(
    x: np.array, y: np.array, multipled_by: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate the given 1D data to increase the resolution.

    Parameters:
    - x (np.array): The x-values of the data.
    - y (np.array): The y-values of the data.
    - multipled_by (int): The factor by which to multiply the number of points.

    Returns:
    - np.array: The interpolated x-values.
    - np.array: The interpolated y-values.
    """

    f_interpol = interpolate.interp1d(x, y)
    x_interpolated = np.linspace(x[0], x[-1], num=len(x) * multipled_by, endpoint=True)
    y_interpolated = f_interpol(x_interpolated)
    return x_interpolated, y_interpolated


@NodeDecorator(
    "span.basics.force_fit",
    name="Advanced peak finder",
)
def force_peak_finder(
    x: np.array,
    y: np.array,
    basic_peaks: Union[List[PeakProperties], PeakProperties],
) -> List[PeakProperties]:
    # """
    # Identify and return the two peaks around the main peak in the given peaks dictionary.

    # Parameters:
    # - peaks (dict): A dictionary containing peak information.
    #                 It should have the keys 'peaks' and 'data'.
    #                 'peaks' should contain a list of dictionaries with keys 'Initial index', 'Index',
    #                  and 'Ending index'.
    #                 'data' should contain arrays 'x' and 'y'.

    # Returns:
    # - dict: A dictionary containing information about the two identified peaks.
    # """
    if isinstance(basic_peaks, (list, np.ndarray, tuple)):
        if len(basic_peaks) != 1:
            raise ValueError(
                "This method accepts one and only one main peak as an input."
            )
        basic_peaks = basic_peaks[0]

    peaks = copy.deepcopy(basic_peaks)
    main_peak_i_index = peaks.i_index
    main_peak_r_index = peaks.index
    main_peak_f_index = peaks.f_index
    y_array = y
    x_array = x
    # Calculate first and second derivatives
    y_array_p = np.gradient(y_array, x, axis=-1)
    y_array_pp = np.gradient(y_array_p, x, axis=-1)
    # Smooth derivatives using Gaussian filter
    y_array_p = gaussian_filter1d(y_array_p, 5)
    y_array_pp = gaussian_filter1d(y_array_pp, 5)

    # maxx = [main_peak_r_index]
    # minn = [main_peak_i_index, main_peak_f_index]
    # Find local maxima and minima of derivatives
    max_p = signal.argrelmax(y_array_p)[0]
    min_p = signal.argrelmin(y_array_p)[0]
    max_pp = signal.argrelmax(y_array_pp)[0]
    # min_pp = signal.argrelmin(y_array_pp)[0]

    # main_peak_i_index = peaks.i_index
    # main_peak_r_index = peaks.index
    # main_peak_f_index = peaks.f_index

    # Determine which peak is on the left and right side of the main peak
    if (
        x_array[main_peak_r_index] - x_array[main_peak_i_index]
        > x_array[main_peak_f_index] - x_array[main_peak_r_index]
    ):  # seond peak is in the leftside of the max peak #TODO: fix this
        common_point = max([num for num in max_pp if num < main_peak_r_index])

        # print("Left convoluted")
        peak1 = {
            "I.Index": main_peak_i_index,
            "R.Index": max(
                [num for num in min_p if num < main_peak_r_index]
            ),  # TODO: fix this
            "F.Index": common_point,
        }
        peak2 = {
            "I.Index": common_point,
            "R.Index": main_peak_r_index,
            "F.Index": main_peak_f_index,
        }
    else:
        common_point = next((x for x in max_pp if x > main_peak_r_index), None)
        # print("Right convoluted")
        peak1 = {
            "I.Index": main_peak_i_index,
            "R.Index": main_peak_r_index,
            "F.Index": common_point,
        }
        peak2 = {
            "I.Index": common_point,
            "R.Index": next((x for x in max_p if x > main_peak_r_index), None),
            "F.Index": main_peak_f_index,
        }
    peak_lst = []
    peak_lst.append([peak1["I.Index"], peak1["R.Index"], peak1["F.Index"]])
    peak_lst.append([peak2["I.Index"], peak2["R.Index"], peak2["F.Index"]])
    peak_properties_list = []
    for peak_nr, peak in enumerate(peak_lst):
        peak_properties = PeakProperties(
            id=basic_peaks.id + f"_{peak_nr + 1}",
            i_index=peak[0],
            index=peak[1],
            f_index=peak[2],
            xfull=x,
            yfull=y,
        )

        peak_properties_list.append(peak_properties)

    return peak_properties_list


@NodeDecorator(
    id="span.basics.peaks.plot",
    name="Plot peaks",
    default_render_options={"data": {"src": "figure"}},
    outputs=[{"name": "figure"}],
)
def plot_peaks(x: np.array, y: np.array, peaks: List[PeakProperties]) -> go.Figure:
    fig = go.Figure()

    # Set up line plot
    plot_trace = {"x": x, "y": y, "mode": "lines", "name": "data"}

    fig.add_trace(go.Scatter(**plot_trace))

    # Define a list of colors for the peaks
    peaks_colors = ["orange", "green", "red", "blue"]

    # Add rectangle shapes for each peak
    for index, peak in enumerate(peaks):
        peak_height = peak.y_at_index
        plot_y_min = min(peak.y_at_i_index, peak.y_at_f_index)

        # Create a scatter trace that simulates a rectangle
        fig.add_trace(
            go.Scatter(
                x=[
                    peak.x_at_i_index,
                    peak.x_at_f_index,
                    peak.x_at_f_index,
                    peak.x_at_i_index,
                ],
                y=[plot_y_min, plot_y_min, peak_height, peak_height],
                fill="toself",
                fillcolor=peaks_colors[index % len(peaks_colors)],
                opacity=0.3,
                line=dict(width=0),
                mode="lines",
                name=f"Peak {peak.id}",
                legendgroup=f"Peak {peak.id}",  # Group by Peak id
                showlegend=True,
            )
        )
        # Add an X marker at the exact peak position
        fig.add_trace(
            go.Scatter(
                x=[peak.x_at_index],
                y=[peak.y_at_index],
                mode="markers",
                marker=dict(symbol="x", size=10, color="black"),
                legendgroup=f"Peak {peak.id}",  # Same group as rectangle
                showlegend=False,
            )
        )

        if hasattr(peak, "model") and peak.model is not None:
            model = peak.model
            y_fit = model.eval(x=peak.xrange, params=model.make_params())
            fig.add_trace(
                go.Scatter(
                    x=peak.xrange,
                    y=y_fit,
                    mode="lines",
                    name=f"Peak {peak.id} fit",
                    line=dict(
                        dash="dash", color=peaks_colors[index % len(peaks_colors)]
                    ),
                    legendgroup=f"Peak {peak.id}",
                ),
            )

    # Customize layout (axes labels and title can be added here if needed)
    fig.update_layout(
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
    )

    return fig


color_map = {
    "C0": "blue",
    "C1": "orange",
    "C2": "green",
    "C3": "red",
    "C4": "purple",
    "C5": "brown",
    "C6": "pink",
    "C7": "gray",
    "C8": "olive",
    "C9": "cyan",
}


@NodeDecorator(
    id="span.basics.fit.plot",
    name="Plot fit 1D",
    default_render_options={"data": {"src": "figure"}},
    outputs=[{"name": "figure"}],
)
def plot_fitted_peaks(peaks: List[PeakProperties]) -> go.Figure:
    peak = peaks[0]
    if not peak.model:
        raise ValueError("No fitting information is available.")

    x_diff = min([np.diff(peak.xfull).mean()])
    minx = min([peak.xfull.min()])
    max_x = max([peak.xfull.max()])
    x_range = np.arange(minx, max_x + x_diff, x_diff)
    y_raw = np.array(
        [
            np.interp(x_range, peak.xfull, peak.yfull, left=np.nan, right=np.nan)
            for peak in peaks
        ]
    )
    y_raw = np.nanmean(y_raw, axis=0)
    # interrpolate nan values
    is_nan = np.isnan(y_raw)
    y_raw[is_nan] = np.interp(x_range[is_nan], x_range[~is_nan], y_raw[~is_nan])

    # Create a subplot with 1 row, 1 column, and a secondary y-axis
    fig = go.Figure()

    # Add the original data trace
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_raw,
            mode="lines",
            name="original",
        ),
    )

    total_y = np.zeros_like(x_range)

    for peak in peaks:
        # add the peak trace
        fig.add_trace(
            go.Scatter(
                x=peak.xrange,
                y=peak.yrange,
                mode="lines",
                name=peak.id,
                legendgroup=peak.id,
                legendgrouptitle={"text": f"Peak {peak.id}"},
            ),
        )

        model = peak.model

        ypeak = model.eval(x=x_range, params=model.make_params())
        total_y += ypeak

        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=ypeak,
                mode="lines",
                name=peak.id + " fit",
                line=dict(dash="dash"),
                legendgroup=peak.id,
            ),
        )

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=total_y,
            mode="lines",
            name="total fit",
            line=dict(dash="dash"),
        ),
    )

    # callculate r2 between the total fit and the original data
    r2 = 1 - np.sum((y_raw - total_y) ** 2) / np.sum((y_raw - np.mean(y_raw)) ** 2)

    # Update axes labels and legend
    fig.update_layout(
        title={
            "text": f"Fitted peaks score = {np.round(r2, 4)}",
            "x": 0.5,  # Center the title
            "xanchor": "center",
        },
    )

    return fig


@NodeDecorator(
    "span.basics.plot_peak",
    name="Plot Peak",
    outputs=[{"name": "figure"}],
    default_render_options={"data": {"src": "figure"}},
)
def plot_peak(
    peak: PeakProperties,
    x: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
) -> go.Figure:
    if x is None:
        x = peak.xrange
    else:
        x = x[peak.i_index : peak.f_index]
    if y is None:
        y = peak.yrange
    else:
        y = y[peak.i_index : peak.f_index]

    if x is None or y is None:
        raise ValueError("x and y must be provided or peak must have xfull and yfull")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Signal"))
    fig.add_trace(
        go.Scatter(
            x=[x[peak.index - peak.i_index]],
            y=[y[peak.index - peak.i_index]],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Peak",
        )
    )
    fig.update_layout(title=f"Peak {peak.id}")
    return fig


fit_peak_node = fn.NodeDecorator(
    id="span.peaks.fit_peak",
    name="Fit Peak",
    outputs=[{"name": "fitted_peak"}, {"name": "model"}, {"name": "fit_result"}],
    # separate_process=True,
)(fit_local_peak)

fit_peaks_node = fn.NodeDecorator(
    id="span.peaks.fit_peaks",
    name="Fit Peaks",
    outputs=[{"name": "fitted_peaks"}, {"name": "model"}, {"name": "fit_results"}],
    default_io_options={
        "modelname": {"value_options": {"options": list(AUTOMODELMAP.keys())}},
    },
    separate_process=True,
)(fit_peaks)

PEAKS_NODE_SHELF = Shelf(
    nodes=[
        peak_finder,
        interpolation_1d,
        force_peak_finder,
        plot_peaks,
        fit_peaks_node,
        fit_peak_node,
        plot_fitted_peaks,
        plot_peak,
    ],
    subshelves=[],
    name="Peak analysis",
    description="Tools for the peak analysis of the spectra",
)
