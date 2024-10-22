from typing import List, Type
import numpy as np
from lmfit import Model
from lmfit.models import SkewedGaussianModel, GaussianModel
from .peak_analysis import PeakProperties


def group_signals(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[PeakProperties],
    baseline_factor: float = 0.001,
) -> List[List[PeakProperties]]:
    """
    Groups peaks based on the baseline factor and the signal array.

    Args:
        x (np.ndarray): Array of x-values (independent variable).
        y (np.ndarray): Array of y-values (dependent variable).
        peaks (List[PeakProperties]): List of detected peaks.
        baseline_factor (float): Factor to determine the baseline cutoff.

    Returns:
        List[List[PeakProperties]]: List of grouped peaks.
    """

    is_baseline = (np.abs(y) >= np.abs(baseline_factor * y.max())).astype(int)
    baseline_cut = np.unique(
        [0] + np.where(np.diff(is_baseline) != 0)[0].tolist() + [len(is_baseline) - 1]
    )

    peak_groups = [[] for _ in baseline_cut]
    for pi, p in enumerate(peaks):
        ingroup = (baseline_cut >= p.i_index) & (baseline_cut <= p.f_index)
        if ingroup.sum() == 0:
            ingroup[(baseline_cut >= p.i_index).argmax()] = True

        for i in range(len(ingroup)):
            if ingroup[i]:
                peak_groups[i].append(pi)

        connected_peaks = []
        current_peak_group = set()
        for pg in peak_groups:
            if len(pg) == 0:
                if len(current_peak_group) > 0:
                    connected_peaks.append(list(current_peak_group))
                current_peak_group = set()
            else:
                current_peak_group.update(pg)

    return [[peaks[pi] for pi in cp] for cp in connected_peaks if len(cp) > 0]


def fit_local_peak(
    x: np.ndarray,
    y: np.ndarray,
    peak: PeakProperties,
    model_class: Type[Model] = SkewedGaussianModel,
    filter_negatives: bool = True,
    incomplete_peak_model_class: Type[Model] = GaussianModel,
    incomplete_threshold: float = 0.8,
    incomplete_x_extend: float = 2.0,
):
    """
    Fits a local peak with the provided model class.

    Args:
        x (np.ndarray): Array of x-values.
        y (np.ndarray): Array of y-values.
        peak (PeakProperties): Peak properties for the specific peak.
        model_class: Model class for fitting, defaults to SkewedGaussianModel.
        filter_negatives (bool): Whether to filter negative y-values.
        incomplete_peak_model_class: Model class for incomplete peaks.
        incomplete_threshold (float): Threshold to consider a peak as incomplete.
        incomplete_x_extend (float): Factor to extend the x-range for incomplete peaks.

    Returns:
        Model: The fitted model for the peak.
    """

    pf = f"p{peak.id}_"
    m = model_class(prefix=pf)
    left = peak.i_index
    right = peak.f_index + 1

    yf = y[left:right]
    xf = x[left:right]

    local_min = yf.min()
    local_max = yf.max()

    if np.abs(yf[-1] - yf[0]) > incomplete_threshold * (local_max - local_min):
        # incomplete peak
        # complete peak by adding a gaussian
        m_complete = incomplete_peak_model_class()
        guess = m_complete.guess(data=yf, x=xf)
        fr_complete = m_complete.fit(data=yf, x=xf, params=guess)

        fwhm = 2 * np.sqrt(2 * np.log(2)) * fr_complete.params["sigma"].value
        center = fr_complete.params["center"].value
        xmeandiff = np.diff(xf).mean()
        xnew = np.arange(
            min(xf.min(), center - incomplete_x_extend * fwhm),
            max(xf.max(), center + incomplete_x_extend * fwhm),
            xmeandiff,
        )

        yf = np.interp(xnew, xf, yf, left=np.nan, right=np.nan)
        xf = xnew

        # fill nan with gaussian
        nan_filter = np.isnan(yf)
        yf[nan_filter] = fr_complete.eval(x=xf[nan_filter], params=fr_complete.params)

    if filter_negatives:
        negativ_filter = yf >= 0
        yf = yf[negativ_filter]
        xf = xf[negativ_filter]

    guess = m.guess(data=yf, x=xf)
    fr = m.fit(data=yf, x=xf, params=guess)

    for pname, param in fr.params.items():
        v = param.value
        if v != 0:
            m.set_param_hint(pname, value=v)
        else:
            m.set_param_hint(pname, value=v)

    return m


def fit_peak_group(
    x,
    y,
    peaks: List[PeakProperties],
    model_class: Type[Model] = SkewedGaussianModel,
    filter_negatives: bool = True,
    incomplete_threshold: float = 0.8,
    incomplete_x_extend: float = 2.0,
    incomplete_peak_model_class: Type[Model] = GaussianModel,
):
    """
    Fits a group of peaks using a model class.

    Args:
        x (np.ndarray): Array of x-values.
        y (np.ndarray): Array of y-values.
        peaks (List[PeakProperties]): List of peaks to fit.
        model_class: Model class for fitting, defaults to SkewedGaussianModel.
        filter_negatives (bool): Whether to filter negative y-values.
        incomplete_threshold (float): Threshold to consider a peak as incomplete.
        incomplete_x_extend (float): Factor to extend the x-range for incomplete peaks.
        incomplete_peak_model_class: Model class for incomplete peaks.

    Returns:
        Model: The fitted model for the group of peaks.
    """
    groupmodel = None
    most_left = min([p.i_index for p in peaks])
    most_right = max([p.f_index for p in peaks])
    for peak in peaks:
        peakmodel = fit_local_peak(
            x=x,
            y=y,
            peak=peak,
            model_class=model_class,
            filter_negatives=filter_negatives,
            incomplete_threshold=incomplete_threshold,
            incomplete_x_extend=incomplete_x_extend,
            incomplete_peak_model_class=incomplete_peak_model_class,
        )
        if groupmodel is None:
            groupmodel = peakmodel
        else:
            groupmodel += peakmodel

    xgroup = x[most_left : most_right + 1]
    ygroup = y[most_left : most_right + 1]

    groupfit = groupmodel.fit(data=ygroup, x=xgroup)

    for pname, param in groupfit.params.items():
        v = param.value
        groupmodel.set_param_hint(pname, value=v)

    return groupmodel


def fit_signals_1D(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[PeakProperties],
    model_class: Type[Model] = SkewedGaussianModel,
    filter_negatives: bool = True,
    baseline_factor: float = 0.001,
    incomplete_threshold: float = 0.8,
    incomplete_x_extend: float = 2.0,
    incomplete_peak_model_class: Type[Model] = GaussianModel,
):
    """
    Fits signals in 1D by first grouping the peaks and then fitting each group.

    Args:
        x (np.ndarray): Array of x-values.
        y (np.ndarray): Array of y-values.
        peaks (List[PeakProperties]): List of detected peaks.
        model_class: Model class for fitting, defaults to SkewedGaussianModel.
        filter_negatives (bool): Whether to filter negative y-values.
        baseline_factor (float): Factor to determine the baseline cutoff.
        incomplete_threshold (float): Threshold to consider a peak as incomplete.
        incomplete_x_extend (float): Factor to extend the x-range for incomplete peaks.
        incomplete_peak_model_class: Model class for incomplete peaks.

    Returns:
        Model: The fitted model for the global signal.
    """
    connected_peaks = group_signals(x, y, peaks, baseline_factor=baseline_factor)

    global_model = None
    for pg in connected_peaks:
        m = fit_peak_group(
            x,
            y,
            pg,
            model_class=model_class,
            filter_negatives=filter_negatives,
            incomplete_threshold=incomplete_threshold,
            incomplete_x_extend=incomplete_x_extend,
            incomplete_peak_model_class=incomplete_peak_model_class,
        )
        if global_model is None:
            global_model = m
        else:
            global_model += m

    global_fit = global_model.fit(data=y, x=x)

    for pname, param in global_fit.params.items():
        v = param.value
        global_model.set_param_hint(pname, value=v)

    return global_model
