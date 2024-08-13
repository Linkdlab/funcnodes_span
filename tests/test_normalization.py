import funcnodes as fn
import unittest
import numpy as np
from funcnodes_span.normalization import _norm, NormMode
from scipy.datasets import electrocardiogram


# x = electrocardiogram()[2000:4000]
class TestNormalization(unittest.IsolatedAsyncioTestCase):
    async def test_default(self):
        norm: fn.Node = _norm()
        norm.inputs["array"].value = electrocardiogram()[2000:4000]
        norm.inputs["mode"].value = NormMode.ZERO_ONE
        self.assertIsInstance(norm, fn.Node)
        await norm
        out = norm.outputs["out"]
        self.assertIsInstance(out.value, np.ndarray)

