import funcnodes as fn
import unittest
import numpy as np
from funcnodes_span.peak_analysis import peak_finder
from scipy.datasets import electrocardiogram


class TestPeakFinder(unittest.IsolatedAsyncioTestCase):
    async def test_default(self):
        peaks: fn.Node = peak_finder()
        peaks.inputs["y_array"].value = electrocardiogram()[2000:4000]
        peaks.inputs["x_array"].value = np.arange(len(electrocardiogram()[2000:4000]))
        peaks.inputs["height"].value = 0
        self.assertIsInstance(peaks, fn.Node)
        await peaks
        out = peaks.outputs["out"]
        self.assertIsInstance(out.value[0], dict)
        self.assertEqual(len(out.value), 138)

    # async def test_non_default_mode(self):
    #     norm: fn.Node = _norm()
    #     norm.inputs["array"].value = electrocardiogram()[2000:4000]
    #     norm.inputs["mode"].value = NormMode.SUM_ABS
    #     self.assertIsInstance(norm, fn.Node)
    #     await norm
    #     out = norm.outputs["out"]
    #     self.assertIsInstance(out.value, np.ndarray)