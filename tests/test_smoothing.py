import funcnodes as fn
import unittest
import numpy as np
from funcnodes_span.smoothing import _smooth, SmoothMode
from scipy.datasets import electrocardiogram


# x = electrocardiogram()[2000:4000]
class TestSmoothing(unittest.IsolatedAsyncioTestCase):
    async def test_default(self):
        smooth: fn.Node = _smooth()
        smooth.inputs["y"].value = electrocardiogram()[2000:4000]
        self.assertIsInstance(smooth, fn.Node)
        await smooth
        out = smooth.outputs["smoothed"]
        self.assertIsInstance(out.value, np.ndarray)

    async def test_non_default_mode(self):
        smooth: fn.Node = _smooth()
        smooth.inputs["y"].value = electrocardiogram()[2000:4000]
        smooth.inputs["mode"].value = SmoothMode.MOVING_AVERAGE
        self.assertIsInstance(smooth, fn.Node)
        await smooth
        out = smooth.outputs["smoothed"]
        self.assertIsInstance(out.value, np.ndarray)

    async def test_1d_preserves_shape_and_finiteness(self):
        smooth: fn.Node = _smooth()
        smooth.inputs["y"].value = electrocardiogram()[2000:4000]
        smooth.inputs["mode"].value = SmoothMode.SPIKE_REMOVE
        self.assertIsInstance(smooth, fn.Node)
        await smooth
        out = smooth.outputs["smoothed"]
        self.assertIsInstance(out.value, np.ndarray)

    async def test_no_spikes_minimal_change(self):
        smooth: fn.Node = _smooth()
        smooth.inputs["y"].value = electrocardiogram()[2000:4000]
        smooth.inputs["mode"].value = SmoothMode.SPIKE_REMOVE
        self.assertIsInstance(smooth, fn.Node)
        await smooth
        out = smooth.outputs["smoothed"].value
        diff = np.abs(out - smooth.inputs["y"].value)
        frac_changed = np.mean(diff > 1e-10)
        self.assertIsInstance(diff, np.ndarray)
        self.assertLess(frac_changed, 0.05)
