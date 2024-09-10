import numpy as np
from pybaselines.utils import gaussian
import funcnodes as fn
import unittest
from funcnodes_span.baseline import (
    _goldindec,
    _imodpoly,
    _loess,
    _modpoly,
    _penalized_poly,
    _poly,
    _quant_reg,
)


x = np.linspace(0, 1000, 1000)
signal = (
    gaussian(x, 9, 100, 12)
    + gaussian(x, 6, 180, 5)
    + gaussian(x, 8, 350, 11)
    + gaussian(x, 15, 400, 18)
    + gaussian(x, 6, 550, 6)
    + gaussian(x, 13, 700, 8)
    + gaussian(x, 9, 800, 9)
    + gaussian(x, 9, 880, 7)
)
baseline = 5 + 10 * np.exp(-x / 600)

noise = np.random.default_rng(0).normal(0, 0.1, len(x))
y = signal + baseline + noise


class TestBaselinePolynomial(unittest.IsolatedAsyncioTestCase):
    async def test_goldindec(self):
        bl: fn.Node = _goldindec()
        bl.inputs["data"].value = y
        bl.inputs["poly_order"].value = 3
        self.assertIsInstance(bl, fn.Node)
        await bl
        baseline_corrected = bl.outputs["baseline_corrected"]
        baseline = bl.outputs["baseline"]
        params = bl.outputs["params"]
        self.assertIsInstance(baseline_corrected.value, np.ndarray)
        self.assertIsInstance(baseline.value, np.ndarray)
        self.assertIsInstance(params.value, dict)

    async def test_imodpoly(self):
        bl: fn.Node = _imodpoly()
        bl.inputs["data"].value = y
        bl.inputs["poly_order"].value = 3
        self.assertIsInstance(bl, fn.Node)
        await bl
        baseline_corrected = bl.outputs["baseline_corrected"]
        baseline = bl.outputs["baseline"]
        params = bl.outputs["params"]
        self.assertIsInstance(baseline_corrected.value, np.ndarray)
        self.assertIsInstance(baseline.value, np.ndarray)
        self.assertIsInstance(params.value, dict)

    async def test_loess(self):
        bl: fn.Node = _loess()
        bl.inputs["data"].value = y
        bl.inputs["poly_order"].value = 3
        self.assertIsInstance(bl, fn.Node)
        await bl
        baseline_corrected = bl.outputs["baseline_corrected"]
        baseline = bl.outputs["baseline"]
        params = bl.outputs["params"]
        self.assertIsInstance(baseline_corrected.value, np.ndarray)
        self.assertIsInstance(baseline.value, np.ndarray)
        self.assertIsInstance(params.value, dict)

    async def test_modpoly(self):
        bl: fn.Node = _modpoly()
        bl.inputs["data"].value = y
        bl.inputs["poly_order"].value = 3
        self.assertIsInstance(bl, fn.Node)
        await bl
        baseline_corrected = bl.outputs["baseline_corrected"]
        baseline = bl.outputs["baseline"]
        params = bl.outputs["params"]
        self.assertIsInstance(baseline_corrected.value, np.ndarray)
        self.assertIsInstance(baseline.value, np.ndarray)
        self.assertIsInstance(params.value, dict)

    async def test_penalized_poly(self):
        bl: fn.Node = _penalized_poly()
        bl.inputs["data"].value = y
        bl.inputs["poly_order"].value = 3
        self.assertIsInstance(bl, fn.Node)
        await bl
        baseline_corrected = bl.outputs["baseline_corrected"]
        baseline = bl.outputs["baseline"]
        params = bl.outputs["params"]
        self.assertIsInstance(baseline_corrected.value, np.ndarray)
        self.assertIsInstance(baseline.value, np.ndarray)
        self.assertIsInstance(params.value, dict)

    async def test_poly(self):
        bl: fn.Node = _poly()
        bl.inputs["data"].value = y
        bl.inputs["poly_order"].value = 3
        self.assertIsInstance(bl, fn.Node)
        await bl
        baseline_corrected = bl.outputs["baseline_corrected"]
        baseline = bl.outputs["baseline"]
        params = bl.outputs["params"]
        self.assertIsInstance(baseline_corrected.value, np.ndarray)
        self.assertIsInstance(baseline.value, np.ndarray)
        self.assertIsInstance(params.value, dict)

    async def test_quant_reg(self):
        bl: fn.Node = _quant_reg()
        bl.inputs["data"].value = y
        bl.inputs["poly_order"].value = 3
        self.assertIsInstance(bl, fn.Node)
        await bl
        baseline_corrected = bl.outputs["baseline_corrected"]
        baseline = bl.outputs["baseline"]
        params = bl.outputs["params"]
        self.assertIsInstance(baseline_corrected.value, np.ndarray)
        self.assertIsInstance(baseline.value, np.ndarray)
        self.assertIsInstance(params.value, dict)
