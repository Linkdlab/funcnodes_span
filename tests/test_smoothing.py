import funcnodes as fn
import unittest
import numpy as np
from funcnodes_span.smoothing import _smooth, SmoothMode
from scipy.datasets import electrocardiogram


# x = electrocardiogram()[2000:4000]
class TestSmoothing(unittest.IsolatedAsyncioTestCase):
    async def test_default(self):
        smooth: fn.Node = _smooth()
        smooth.inputs["array"].value = electrocardiogram()[2000:4000]
        smooth.inputs["mode"].value = SmoothMode.SAVITZKY_GOLAY
        self.assertIsInstance(smooth, fn.Node)
        await smooth
        out = smooth.outputs["out"]
        self.assertIsInstance(out.value, np.ndarray)

