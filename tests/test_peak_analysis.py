import funcnodes as fn
import unittest
import numpy as np
from funcnodes_span.peak_analysis import (
    peak_finder,
    PeakProperties,
    interpolation_1d,
    fit_1D,
    FittingModel,
    BaselineModel,
    plot_peaks,
    plot_fitted_peaks,
    force_peak_finder,
)
from scipy.datasets import electrocardiogram
import plotly.graph_objects as go

fn.config.IN_NODE_TEST = True


class TestPeakFinder(unittest.IsolatedAsyncioTestCase):
    async def test_default(self):
        peaks: fn.Node = peak_finder()
        peaks.inputs["y"].value = electrocardiogram()[2000:4000]
        peaks.inputs["x"].value = np.arange(len(electrocardiogram()[2000:4000]))
        peaks.inputs["height"].value = 2
        self.assertIsInstance(peaks, fn.Node)
        await peaks
        out = peaks.outputs["out"]
        self.assertIsInstance(out.value[0], PeakProperties)
        self.assertEqual(len(out.value), 1)

    async def test_plot_peaks(self):
        peaks: fn.Node = peak_finder()
        peaks.inputs["y"].value = electrocardiogram()[2000:4000]
        peaks.inputs["x"].value = np.arange(len(electrocardiogram()[2000:4000]))
        peaks.inputs["height"].value = 2

        plotter = plot_peaks()

        plotter.inputs["peaks_dict"].connect(peaks.outputs["out"])
        plotter.inputs["y"].value = electrocardiogram()[2000:4000]
        plotter.inputs["x"].value = np.arange(len(electrocardiogram()[2000:4000]))

        await fn.run_until_complete(peaks, plotter)

        self.assertIsInstance(plotter.outputs["figure"].value, go.Figure)


class TestInterpolation(unittest.IsolatedAsyncioTestCase):
    async def test_default(self):
        inter1d: fn.Node = interpolation_1d()
        inter1d.inputs["y"].value = electrocardiogram()[2000:4000]
        inter1d.inputs["x"].value = np.arange(len(electrocardiogram()[2000:4000]))
        inter1d.inputs["multipled_by"].value = 2
        self.assertIsInstance(inter1d, fn.Node)
        await inter1d
        y_interpolated = inter1d.outputs["y_interpolated"]
        x_interpolated = inter1d.outputs["x_interpolated"]
        self.assertIsInstance(y_interpolated.value, np.ndarray)
        self.assertIsInstance(x_interpolated.value, np.ndarray)
        self.assertEqual(
            len(y_interpolated.value), 2 * len(electrocardiogram()[2000:4000])
        )


class TestFit1D(unittest.IsolatedAsyncioTestCase):
    async def test_default(self):
        peaks: fn.Node = peak_finder()
        peaks.inputs["y"].value = electrocardiogram()[2000:4000]
        peaks.inputs["x"].value = np.arange(len(electrocardiogram()[2000:4000]))
        peaks.inputs["height"].value = 2
        self.assertIsInstance(peaks, fn.Node)
        await peaks
        self.assertIsInstance(peaks.outputs["out"].value[0], PeakProperties)
        fit: fn.Node = fit_1D()
        fit.inputs["y"].value = electrocardiogram()[2000:4000]
        fit.inputs["x"].value = np.arange(len(electrocardiogram()[2000:4000]))
        fit.inputs["basic_peaks"].connect(peaks.outputs["out"])
        fit.inputs["main_model"].value = FittingModel.Gaussian
        fit.inputs["baseline_model"].value = BaselineModel.Linear
        self.assertIsInstance(fit, fn.Node)
        await fn.run_until_complete(fit, peaks)
        out = fit.outputs["out"]
        self.assertIsInstance(out.value[0], PeakProperties)

    async def test_plot_fitted_peaks(self):
        peaks: fn.Node = peak_finder()
        peaks.inputs["y"].value = electrocardiogram()[2000:4000]
        peaks.inputs["x"].value = np.arange(len(electrocardiogram()[2000:4000]))
        peaks.inputs["height"].value = 2

        fit: fn.Node = fit_1D()
        fit.inputs["y"].value = electrocardiogram()[2000:4000]
        fit.inputs["x"].value = np.arange(len(electrocardiogram()[2000:4000]))
        fit.inputs["basic_peaks"].connect(peaks.outputs["out"])
        fit.inputs["main_model"].value = FittingModel.Gaussian
        fit.inputs["baseline_model"].value = BaselineModel.Linear

        plotter = plot_fitted_peaks()

        plotter.inputs["peaks"].connect(fit.outputs["out"])

        await fn.run_until_complete(fit, peaks, plotter)

        self.assertIsInstance(plotter.outputs["figure"].value, go.Figure)


class TestForcePeakFinder(unittest.IsolatedAsyncioTestCase):
    async def test_force_peak_finder(self):
        x = np.linspace(0, 20, 1000)
        # gaussian distribution
        y = np.exp(-((x - 10) ** 2)) + np.exp(-(((x - 12) / 3) ** 2))
        peaks: fn.Node = peak_finder()
        peaks.inputs["y"].value = y
        peaks.inputs["x"].value = x

        await peaks

        force_peaks: fn.Node = force_peak_finder()

        force_peaks.inputs["basic_peaks"].value = peaks.outputs["out"].value

        force_peaks.inputs["y"].value = y
        force_peaks.inputs["x"].value = x

        await force_peaks

        self.assertIsInstance(force_peaks.outputs["out"].value[0], PeakProperties)
        self.assertEqual(len(force_peaks.outputs["out"].value), 2)
