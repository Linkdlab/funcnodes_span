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
)
from scipy.datasets import electrocardiogram


class TestPeakFinder(unittest.IsolatedAsyncioTestCase):
    async def test_default(self):
        peaks: fn.Node = peak_finder()
        peaks.inputs["y_array"].value = electrocardiogram()[2000:4000]
        peaks.inputs["x_array"].value = np.arange(len(electrocardiogram()[2000:4000]))
        peaks.inputs["height"].value = 2
        self.assertIsInstance(peaks, fn.Node)
        await peaks
        out = peaks.outputs["out"]
        self.assertIsInstance(out.value[0], PeakProperties)
        self.assertEqual(len(out.value), 1)


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
        peaks.inputs["y_array"].value = electrocardiogram()[2000:4000]
        peaks.inputs["x_array"].value = np.arange(len(electrocardiogram()[2000:4000]))
        peaks.inputs["height"].value = 2
        self.assertIsInstance(peaks, fn.Node)
        await peaks
        fit: fn.Node = fit_1D()
        fit.inputs["y_array"].value = electrocardiogram()[2000:4000]
        fit.inputs["x_array"].value = np.arange(len(electrocardiogram()[2000:4000]))
        fit.inputs["basic_peaks"].connect(peaks.outputs["out"])
        fit.inputs["main_model"].value = FittingModel.Gaussian
        fit.inputs["baseline_model"].value = BaselineModel.Linear
        self.assertIsInstance(fit, fn.Node)
        await fn.run_until_complete(fit, peaks)
        out = fit.outputs["out"]
        self.assertIsInstance(out.value[0], PeakProperties)
