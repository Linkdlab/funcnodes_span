from funcnodes import NodeDecorator, Shelf
from exposedfunctionality import controlled_wrapper
import numpy as np
from enum import Enum
from typing import Optional, Tuple
import pybaselines


class CostFunction(Enum):
    asymmetric_indec = "asymmetric_indec"
    asymmetric_truncated_quadratic = "asymmetric_truncated_quadratic"
    asymmetric_huber = "asymmetric_huber"

    @classmethod
    def default(cls):
        return cls.asymmetric_indec.value


@NodeDecorator(
    "pybaselines.polynomial.goldindec",
    name="goldindec",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.polynomial.goldindec, wrapper_attribute="__fnwrapped__")
def _goldindec(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    poly_order: int = 2,
    tol: float = 0.001,
    max_iter: int = 250,
    weights: Optional[np.ndarray] = None,
    cost_function: CostFunction = CostFunction.default(),
    peak_ratio: float = 0.5,
    alpha_factor: float = 0.99,
    tol_2: float = 0.001,
    tol_3: float = 1e-06,
    max_iter_2: int = 100,
    return_coef: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    if isinstance(cost_function, CostFunction):
        cost_function = cost_function.value
    baseline, params = pybaselines.polynomial.goldindec(
        data,
        x_data=x_data,
        poly_order=poly_order,
        tol=tol,
        max_iter=max_iter,
        weights=weights,
        cost_function=cost_function,
        peak_ratio=peak_ratio,
        alpha_factor=alpha_factor,
        tol_2=tol_2,
        tol_3=tol_3,
        return_coef=return_coef,
        max_iter_2=max_iter_2,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.polynomial.imodpoly",
    name="imodpoly",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.polynomial.imodpoly, wrapper_attribute="__fnwrapped__")
def _imodpoly(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    poly_order: int = 2,
    max_iter: int = 250,
    tol: float = 1e-3,
    weights: Optional[np.ndarray] = None,
    num_std: float = 1.0,
    use_original: bool = False,
    mask_initial_peaks: bool = False,
    return_coef: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.polynomial.imodpoly(
        data,
        x_data=x_data,
        poly_order=poly_order,
        max_iter=max_iter,
        tol=tol,
        weights=weights,
        num_std=num_std,
        use_original=use_original,
        return_coef=return_coef,
        mask_initial_peaks=mask_initial_peaks,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.polynomial.loess",
    name="loess",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.polynomial.loess, wrapper_attribute="__fnwrapped__")
def _loess(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    fraction: float = 0.2,
    total_points: Optional[int] = None,
    scale: float = 3.0,
    poly_order: int = 1,
    max_iter: int = 10,
    tol: float = 1e-3,
    symmetric_weights: bool = False,
    use_threshold: bool = False,
    weights: Optional[np.ndarray] = None,
    num_std: float = 1.0,
    use_original: bool = False,
    conserve_memory: bool = False,
    delta: float = 0.0,
    return_coef: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.polynomial.loess(
        data,
        x_data=x_data,
        fraction=fraction,
        total_points=total_points,
        scale=scale,
        poly_order=poly_order,
        max_iter=max_iter,
        tol=tol,
        weights=weights,
        num_std=num_std,
        use_original=use_original,
        symmetric_weights=symmetric_weights,
        use_threshold=use_threshold,
        conserve_memory=conserve_memory,
        return_coef=return_coef,
        delta=delta,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.polynomial.modpoly",
    name="modpoly",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.polynomial.modpoly, wrapper_attribute="__fnwrapped__")
def _modpoly(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    poly_order: int = 2,
    max_iter: int = 250,
    tol: float = 1e-3,
    weights: Optional[np.ndarray] = None,
    use_original: bool = False,
    mask_initial_peaks: bool = False,
    return_coef: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.polynomial.modpoly(
        data,
        x_data=x_data,
        poly_order=poly_order,
        max_iter=max_iter,
        tol=tol,
        weights=weights,
        use_original=use_original,
        mask_initial_peaks=mask_initial_peaks,
        return_coef=return_coef,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


class PenalizedPolyCostFunction(Enum):
    asymmetric_truncated_quadratic = "asymmetric_truncated_quadratic"
    symmetric_truncated_quadratic = "symmetric_truncated_quadratic"
    asymmetric_huber = "asymmetric_huber"
    symmetric_huber = "symmetric_huber"
    asymmetric_indec = "asymmetric_indec"
    symmetric_indec = "symmetric_indec"

    @classmethod
    def default(cls):
        return cls.asymmetric_truncated_quadratic.value


@NodeDecorator(
    "pybaselines.polynomial.penalized_poly",
    name="penalized_poly",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(
    pybaselines.polynomial.penalized_poly, wrapper_attribute="__fnwrapped__"
)
def _penalized_poly(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    poly_order: int = 2,
    max_iter: int = 250,
    tol: float = 1e-3,
    weights: Optional[np.ndarray] = None,
    cost_function: PenalizedPolyCostFunction = PenalizedPolyCostFunction.default(),
    threshold: Optional[float] = None,
    alpha_factor: float = 0.99,
    return_coef: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    if isinstance(cost_function, PenalizedPolyCostFunction):
        cost_function = cost_function.value
    baseline, params = pybaselines.polynomial.penalized_poly(
        data,
        x_data=x_data,
        poly_order=poly_order,
        max_iter=max_iter,
        tol=tol,
        weights=weights,
        cost_function=cost_function,
        threshold=threshold,
        alpha_factor=alpha_factor,
        return_coef=return_coef,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.polynomial.poly",
    name="poly",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.polynomial.poly, wrapper_attribute="__fnwrapped__")
def _poly(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    poly_order: int = 2,
    weights: Optional[np.ndarray] = None,
    return_coef: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.polynomial.poly(
        data,
        x_data=x_data,
        poly_order=poly_order,
        weights=weights,
        return_coef=return_coef,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.polynomial.quant_reg",
    name="quant_reg",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.polynomial.quant_reg, wrapper_attribute="__fnwrapped__")
def _quant_reg(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    poly_order: int = 2,
    quantile: float = 0.05,
    max_iter: int = 250,
    tol: float = 1e-6,
    weights: Optional[np.ndarray] = None,
    eps: Optional[float] = None,
    return_coef: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.polynomial.quant_reg(
        data,
        x_data=x_data,
        poly_order=poly_order,
        max_iter=max_iter,
        tol=tol,
        weights=weights,
        quantile=quantile,
        eps=eps,
        return_coef=return_coef,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


POLYNOMIAL_NODE_SHELF = Shelf(
    nodes=[_goldindec, _imodpoly, _loess, _modpoly, _penalized_poly, _poly, _quant_reg],
    subshelves=[],
    name="Polynomial",
    description="Fits a polynomial baseline",
)


@NodeDecorator(
    "pybaselines.whittaker.airpls",
    name="airpls",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.whittaker.airpls, wrapper_attribute="__fnwrapped__")
def _airpls(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    lam: float = 1000000.0,
    diff_order: int = 2,
    max_iter: int = 50,
    tol: float = 1e-3,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.whittaker.airpls(
        data,
        x_data=x_data,
        lam=lam,
        max_iter=max_iter,
        tol=tol,
        weights=weights,
        diff_order=diff_order,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.whittaker.arpls",
    name="arpls",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.whittaker.arpls, wrapper_attribute="__fnwrapped__")
def _arpls(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    lam: float = 100000.0,
    diff_order: int = 2,
    max_iter: int = 50,
    tol: float = 1e-3,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.whittaker.arpls(
        data,
        x_data=x_data,
        lam=lam,
        max_iter=max_iter,
        tol=tol,
        weights=weights,
        diff_order=diff_order,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.whittaker.asls",
    name="asls",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.whittaker.asls, wrapper_attribute="__fnwrapped__")
def _asls(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    lam: float = 1000000.0,
    p: float = 0.01,
    diff_order: int = 2,
    max_iter: int = 50,
    tol: float = 1e-3,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.whittaker.asls(
        data,
        x_data=x_data,
        lam=lam,
        p=p,
        max_iter=max_iter,
        tol=tol,
        weights=weights,
        diff_order=diff_order,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.whittaker.aspls",
    name="aspls",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.whittaker.aspls, wrapper_attribute="__fnwrapped__")
def _aspls(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    lam: float = 100000.0,
    alpha: Optional[np.ndarray] = None,
    diff_order: int = 2,
    max_iter: int = 100,
    tol: float = 1e-3,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.whittaker.aspls(
        data,
        x_data=x_data,
        lam=lam,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        weights=weights,
        diff_order=diff_order,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.whittaker.derpsalsa",
    name="derpsalsa",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.whittaker.derpsalsa, wrapper_attribute="__fnwrapped__")
def _derpsalsa(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    lam: float = 1000000.0,
    p: float = 0.01,
    k: Optional[float] = None,
    diff_order: int = 2,
    max_iter: int = 50,
    weights: Optional[np.ndarray] = None,
    smooth_half_window: Optional[int] = None,
    num_smooths: int = 16,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.whittaker.derpsalsa(
        data,
        x_data=x_data,
        lam=lam,
        p=p,
        max_iter=max_iter,
        k=k,
        weights=weights,
        diff_order=diff_order,
        smooth_half_window=smooth_half_window,
        num_smooths=num_smooths,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.whittaker.drpls",
    name="drpls",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.whittaker.drpls, wrapper_attribute="__fnwrapped__")
def _drpls(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    lam: float = 100000.0,
    eta: float = 0.5,
    max_iter: int = 50,
    tol: float = 1e-3,
    diff_order: int = 2,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.whittaker.drpls(
        data,
        x_data=x_data,
        lam=lam,
        eta=eta,
        max_iter=max_iter,
        weights=weights,
        diff_order=diff_order,
        tol=tol,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.whittaker.iarpls",
    name="iarpls",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.whittaker.iarpls, wrapper_attribute="__fnwrapped__")
def _iarpls(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    lam: float = 100000.0,
    max_iter: int = 50,
    tol: float = 1e-3,
    diff_order: int = 2,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.whittaker.iarpls(
        data,
        x_data=x_data,
        lam=lam,
        max_iter=max_iter,
        weights=weights,
        diff_order=diff_order,
        tol=tol,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.whittaker.iasls",
    name="iasls",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.whittaker.iasls, wrapper_attribute="__fnwrapped__")
def _iasls(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    lam: float = 100000.0,
    p: float = 0.5,
    lam_1: float = 0.0001,
    max_iter: int = 50,
    tol: float = 1e-3,
    diff_order: int = 2,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.whittaker.iasls(
        data,
        x_data=x_data,
        lam=lam,
        max_iter=max_iter,
        lam_1=lam_1,
        p=p,
        weights=weights,
        diff_order=diff_order,
        tol=tol,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.whittaker.psalsa",
    name="psalsa",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.whittaker.psalsa, wrapper_attribute="__fnwrapped__")
def _psalsa(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    lam: float = 100000.0,
    p: float = 0.5,
    k: Optional[float] = None,
    max_iter: int = 50,
    tol: float = 1e-3,
    diff_order: int = 2,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.whittaker.psalsa(
        data,
        x_data=x_data,
        lam=lam,
        max_iter=max_iter,
        p=p,
        k=k,
        weights=weights,
        diff_order=diff_order,
        tol=tol,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


WHITTAKER_NODE_SHELF = Shelf(
    nodes=[
        _airpls,
        _arpls,
        _asls,
        _aspls,
        _derpsalsa,
        _drpls,
        _iarpls,
        _iasls,
        _psalsa,
    ],
    subshelves=[],
    name="Whittaker",
    description="Fits a Whittaker baseline",
)


@NodeDecorator(
    "pybaselines.morphological.amormol",
    name="amormol",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(
    pybaselines.morphological.amormol, wrapper_attribute="__fnwrapped__"
)
def _amormol(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    half_window: Optional[int] = None,
    tol: float = 1e-3,
    max_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.morphological.amormol(
        data, x_data=x_data, max_iter=max_iter, tol=tol, half_window=half_window
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.morphological.imor",
    name="imor",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.morphological.imor, wrapper_attribute="__fnwrapped__")
def _imor(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    half_window: Optional[int] = None,
    tol: float = 1e-3,
    max_iter: int = 200,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.morphological.imor(
        data, x_data=x_data, max_iter=max_iter, tol=tol, half_window=half_window
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.morphological.jbcd",
    name="jbcd",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.morphological.jbcd, wrapper_attribute="__fnwrapped__")
def _jbcd(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    half_window: Optional[int] = None,
    alpha: float = 0.1,
    beta: float = 10.0,
    gamma: float = 1.0,
    beta_mult: float = 1.1,
    gamma_mult: float = 0.909,
    diff_order: int = 1,
    tol: float = 1e-2,
    tol_2: float = 1e-3,
    max_iter: int = 20,
    robust_opening: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.morphological.jbcd(
        data,
        x_data=x_data,
        max_iter=max_iter,
        tol=tol,
        half_window=half_window,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        beta_mult=beta_mult,
        gamma_mult=gamma_mult,
        diff_order=diff_order,
        tol_2=tol_2,
        robust_opening=robust_opening,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.morphological.mor",
    name="mor",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.morphological.mor, wrapper_attribute="__fnwrapped__")
def _mor(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    half_window: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.morphological.mor(
        data, x_data=x_data, half_window=half_window
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.morphological.mormol",
    name="mormol",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.morphological.mormol, wrapper_attribute="__fnwrapped__")
def _mormol(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    half_window: Optional[int] = None,
    tol: float = 1e-3,
    max_iter: int = 200,
    smooth_half_window: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.morphological.mormol(
        data,
        x_data=x_data,
        max_iter=max_iter,
        tol=tol,
        half_window=half_window,
        smooth_half_window=smooth_half_window,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.morphological.mpls",
    name="mpls",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.morphological.mpls, wrapper_attribute="__fnwrapped__")
def _mpls(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    half_window: Optional[int] = None,
    lam: float = 1000000.0,
    p: float = 0.0,
    diff_order: int = 2,
    tol: float = 1e-3,
    max_iter: int = 50,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.morphological.mpls(
        data,
        x_data=x_data,
        max_iter=max_iter,
        tol=tol,
        half_window=half_window,
        lam=lam,
        p=p,
        diff_order=diff_order,
        weights=weights,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.morphological.mpspline",
    name="mpspline",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(
    pybaselines.morphological.mpspline, wrapper_attribute="__fnwrapped__"
)
def _mpspline(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    half_window: Optional[int] = None,
    lam: float = 10000.0,
    lam_smooth: float = 0.01,
    p: float = 0.0,
    num_knots: int = 100,
    spline_degree: int = 3,
    diff_order: int = 2,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.morphological.mpspline(
        data,
        x_data=x_data,
        half_window=half_window,
        lam=lam,
        lam_smooth=lam_smooth,
        num_knots=num_knots,
        spline_degree=spline_degree,
        p=p,
        diff_order=diff_order,
        weights=weights,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.morphological.mwmv",
    name="mwmv",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.morphological.mwmv, wrapper_attribute="__fnwrapped__")
def _mwmv(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    half_window: Optional[int] = None,
    smooth_half_window: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.morphological.mwmv(
        data,
        x_data=x_data,
        smooth_half_window=smooth_half_window,
        half_window=half_window,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.morphological.rolling_ball",
    name="rolling_ball",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(
    pybaselines.morphological.rolling_ball, wrapper_attribute="__fnwrapped__"
)
def _rolling_ball(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    half_window: Optional[int] = None,
    smooth_half_window: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.morphological.rolling_ball(
        data,
        x_data=x_data,
        smooth_half_window=smooth_half_window,
        half_window=half_window,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


@NodeDecorator(
    "pybaselines.morphological.tophat",
    name="tophat",
    outputs=[
        {"name": "baseline_corrected"},
        {"name": "baseline"},
        {"name": "params"},
    ],
)
@controlled_wrapper(pybaselines.morphological.tophat, wrapper_attribute="__fnwrapped__")
def _tophat(
    data: np.ndarray,
    x_data: Optional[np.ndarray] = None,
    half_window: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    baseline, params = pybaselines.morphological.tophat(
        data,
        x_data=x_data,
        half_window=half_window,
    )
    baseline_corrected = data - baseline
    return baseline_corrected, baseline, params


MORPHOLOGICAL_NODE_SHELF = Shelf(
    nodes=[
        _amormol,
        _imor,
        _jbcd,
        _mor,
        _mormol,
        _mpls,
        _mpspline,
        _mwmv,
        _rolling_ball,
        _tophat,
    ],
    subshelves=[],
    name="Morphological",
    description="Fits a morphological baseline",
)

BASELINE_NODE_SHELF = Shelf(
    nodes=[],
    subshelves=[POLYNOMIAL_NODE_SHELF, WHITTAKER_NODE_SHELF],
    name="Baseline correction",
    description="Provides different techniques for fitting baselines to experimental data using pybaselines.",
)
