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

BASELINE_NODE_SHELF = Shelf(
    nodes=[],
    subshelves=[POLYNOMIAL_NODE_SHELF],
    name="Baseline coorection",
    description="Provides different techniques for fitting baselines to experimental data using pybaselines.",
)
# @NodeDecorator(
#     "pybaselines.spline.corner_cutting",
#     name="corner_cutting",
#     outputs=[
#         {"name": "baseline_corrected"},
#         {"name": "baseline"},
#         {"name": "params"},
#     ],
# )
# @controlled_wrapper(
#     pybaselines.spline.corner_cutting, wrapper_attribute="__fnwrapped__"
# )
# def _corner_cutting(
#     data: np.ndarray,
#     max_iter: Optional[int] = None,
#     x_data: Optional[np.ndarray] = None,
# ) -> Tuple[np.ndarray, dict]:
#     baseline, params = pybaselines.spline.corner_cutting(
#         data, x_data=x_data, max_iter=max_iter
#     )
#     baseline_corrected = data - baseline
#     return baseline_corrected, baseline, params


# @NodeDecorator(
#     "pybaselines.spline.irsqr",
#     name="irsqr",
#     outputs=[
#         {"name": "baseline_corrected"},
#         {"name": "baseline"},
#         {"name": "params"},
#     ],
# )
# @controlled_wrapper(pybaselines.spline.irsqr, wrapper_attribute="__fnwrapped__")
# def _irsqr(
#     data: np.ndarray,
#     lam: Optional[float] = None,
#     quantile: Optional[float] = None,
#     num_knots: Optional[int] = None,
#     spline_degree: Optional[int] = None,
#     diff_order: Optional[int] = None,
#     max_iter: Optional[int] = None,
#     tol: Optional[float] = None,
#     weights: Optional[np.ndarray] = None,
#     eps: Optional[float] = None,
#     x_data: Optional[np.ndarray] = None,
# ) -> Tuple[np.ndarray, dict]:
#     baseline, params = pybaselines.spline.irsqr(
#         data,
#         lam=lam,
#         quantile=quantile,
#         num_knots=num_knots,
#         spline_degree=spline_degree,
#         diff_order=diff_order,
#         max_iter=max_iter,
#         tol=tol,
#         weights=weights,
#         eps=eps,
#         x_data=x_data,
#     )
#     baseline_corrected = data - baseline
#     return baseline_corrected, baseline, params


# @NodeDecorator(
#     "pybaselines.spline.mixture_model",
#     name="mixture_model",
#     outputs=[
#         {"name": "baseline_corrected"},
#         {"name": "baseline"},
#         {"name": "params"},
#     ],
# )
# @controlled_wrapper(pybaselines.spline.mixture_model, wrapper_attribute="__fnwrapped__")
# def _mixture_model(
#     data: np.ndarray,
#     lam: Optional[float] = None,
#     p: Optional[float] = None,
#     num_knots: Optional[int] = None,
#     spline_degree: Optional[int] = None,
#     diff_order: Optional[int] = None,
#     max_iter: Optional[int] = None,
#     tol: Optional[float] = None,
#     weights: Optional[np.ndarray] = None,
#     symmetric: bool = False,
#     x_data: Optional[np.ndarray] = None,
# ) -> Tuple[np.ndarray, dict]:
#     baseline, params = pybaselines.spline.mixture_model(
#         data,
#         lam=lam,
#         p=p,
#         num_knots=num_knots,
#         spline_degree=spline_degree,
#         diff_order=diff_order,
#         max_iter=max_iter,
#         tol=tol,
#         weights=weights,
#         symmetric=symmetric,
#         x_data=x_data,
#     )
#     baseline_corrected = data - baseline
#     return baseline_corrected, baseline, params


# @NodeDecorator(
#     "pybaselines.spline.pspline_airpls",
#     name="pspline_airpls",
#     outputs=[
#         {"name": "baseline_corrected"},
#         {"name": "baseline"},
#         {"name": "params"},
#     ],
# )
# @controlled_wrapper(
#     pybaselines.spline.pspline_airpls, wrapper_attribute="__fnwrapped__"
# )
# def _pspline_airpls(
#     data: np.ndarray,
#     lam: Optional[float] = None,
#     num_knots: Optional[int] = None,
#     spline_degree: Optional[int] = None,
#     diff_order: Optional[int] = None,
#     max_iter: Optional[int] = None,
#     tol: Optional[float] = None,
#     weights: Optional[np.ndarray] = None,
#     x_data: Optional[np.ndarray] = None,
# ) -> Tuple[np.ndarray, dict]:
#     baseline, params = pybaselines.spline.pspline_airpls(
#         data,
#         lam=lam,
#         num_knots=num_knots,
#         spline_degree=spline_degree,
#         diff_order=diff_order,
#         max_iter=max_iter,
#         tol=tol,
#         weights=weights,
#         x_data=x_data,
#     )
#     baseline_corrected = data - baseline
#     return baseline_corrected, baseline, params


# @NodeDecorator(
#     "pybaselines.spline.pspline_arpls",
#     name="pspline_arpls",
#     outputs=[
#         {"name": "baseline_corrected"},
#         {"name": "baseline"},
#         {"name": "params"},
#     ],
# )
# @controlled_wrapper(pybaselines.spline.pspline_arpls, wrapper_attribute="__fnwrapped__")
# def _pspline_arpls(
#     data: np.ndarray,
#     lam: Optional[float] = None,
#     num_knots: Optional[int] = None,
#     spline_degree: Optional[int] = None,
#     diff_order: Optional[int] = None,
#     max_iter: Optional[int] = None,
#     tol: Optional[float] = None,
#     weights: Optional[np.ndarray] = None,
#     x_data: Optional[np.ndarray] = None,
# ) -> Tuple[np.ndarray, dict]:
#     baseline, params = pybaselines.spline.pspline_arpls(
#         data,
#         lam=lam,
#         num_knots=num_knots,
#         spline_degree=spline_degree,
#         diff_order=diff_order,
#         max_iter=max_iter,
#         tol=tol,
#         weights=weights,
#         x_data=x_data,
#     )
#     baseline_corrected = data - baseline
#     return baseline_corrected, baseline, params


# @NodeDecorator(
#     "pybaselines.spline.pspline_asls",
#     name="pspline_asls",
#     outputs=[
#         {"name": "baseline_corrected"},
#         {"name": "baseline"},
#         {"name": "params"},
#     ],
# )
# @controlled_wrapper(pybaselines.spline.pspline_asls, wrapper_attribute="__fnwrapped__")
# def _pspline_asls(
#     data: np.ndarray,
#     lam: Optional[float] = None,
#     p: Optional[float] = None,
#     num_knots: Optional[int] = None,
#     spline_degree: Optional[int] = None,
#     diff_order: Optional[int] = None,
#     max_iter: Optional[int] = None,
#     tol: Optional[float] = None,
#     weights: Optional[np.ndarray] = None,
#     x_data: Optional[np.ndarray] = None,
# ) -> Tuple[np.ndarray, dict]:
#     baseline, params = pybaselines.spline.pspline_asls(
#         data,
#         lam=lam,
#         p=p,
#         num_knots=num_knots,
#         spline_degree=spline_degree,
#         diff_order=diff_order,
#         max_iter=max_iter,
#         tol=tol,
#         weights=weights,
#         x_data=x_data,
#     )
#     baseline_corrected = data - baseline
#     return baseline_corrected, baseline, params


# @NodeDecorator(
#     "pybaselines.spline.pspline_aspls",
#     name="pspline_aspls",
#     outputs=[
#         {"name": "baseline_corrected"},
#         {"name": "baseline"},
#         {"name": "params"},
#     ],
# )
# @controlled_wrapper(pybaselines.spline.pspline_aspls, wrapper_attribute="__fnwrapped__")
# def _pspline_aspls(
#     data: np.ndarray,
#     lam: Optional[float] = None,
#     num_knots: Optional[int] = None,
#     spline_degree: Optional[int] = None,
#     diff_order: Optional[int] = None,
#     max_iter: Optional[int] = None,
#     tol: Optional[float] = None,
#     weights: Optional[np.ndarray] = None,
#     alpha: Optional[np.ndarray] = None,
#     x_data: Optional[np.ndarray] = None,
# ) -> Tuple[np.ndarray, dict]:
#     baseline, params = pybaselines.spline.pspline_aspls(
#         data,
#         lam=lam,
#         num_knots=num_knots,
#         spline_degree=spline_degree,
#         diff_order=diff_order,
#         max_iter=max_iter,
#         tol=tol,
#         weights=weights,
#         alpha=alpha,
#         x_data=x_data,
#     )
#     baseline_corrected = data - baseline
#     return baseline_corrected, baseline, params


# @NodeDecorator(
#     "pybaselines.spline.pspline_derpsalsa",
#     name="pspline_derpsalsa",
#     outputs=[
#         {"name": "baseline_corrected"},
#         {"name": "baseline"},
#         {"name": "params"},
#     ],
# )
# @controlled_wrapper(
#     pybaselines.spline.pspline_derpsalsa, wrapper_attribute="__fnwrapped__"
# )
# def _pspline_derpsalsa(
#     data: np.ndarray,
#     lam: Optional[float] = None,
#     p: Optional[float] = None,
#     k: Optional[float] = None,
#     num_knots: Optional[int] = None,
#     spline_degree: Optional[int] = None,
#     diff_order: Optional[int] = None,
#     max_iter: Optional[int] = None,
#     tol: Optional[float] = None,
#     weights: Optional[np.ndarray] = None,
#     smooth_half_window: Optional[int] = None,
#     num_smooths: Optional[int] = None,
#     x_data: Optional[np.ndarray] = None,
# ) -> Tuple[np.ndarray, dict]:
#     baseline, params = pybaselines.spline.pspline_derpsalsa(
#         data,
#         lam=lam,
#         num_knots=num_knots,
#         spline_degree=spline_degree,
#         diff_order=diff_order,
#         max_iter=max_iter,
#         tol=tol,
#         weights=weights,
#         smooth_half_window=smooth_half_window,
#         num_smooths=num_smooths,
#         x_data=x_data,
#     )
#     baseline_corrected = data - baseline
#     return baseline_corrected, baseline, params
