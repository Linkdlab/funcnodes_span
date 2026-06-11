import funcnodes as fn
import unittest
import numpy as np
from funcnodes_span.normalization import _norm, NormMode, density_normalization_node
from scipy.datasets import electrocardiogram


# x = electrocardiogram()[2000:4000]
class TestNormalization(unittest.IsolatedAsyncioTestCase):
    async def test_default(self):
        norm: fn.Node = _norm()
        norm.inputs["array"].value = electrocardiogram()[2000:4000]
        self.assertIsInstance(norm, fn.Node)
        await norm
        out = norm.outputs["out"]
        self.assertIsInstance(out.value, np.ndarray)

    async def test_non_default_mode(self):
        norm: fn.Node = _norm()
        norm.inputs["array"].value = electrocardiogram()[2000:4000]
        norm.inputs["mode"].value = NormMode.SUM_ABS
        self.assertIsInstance(norm, fn.Node)
        await norm
        out = norm.outputs["out"]
        self.assertIsInstance(out.value, np.ndarray)

    async def test_desity_norm(self):
        norm: fn.Node = density_normalization_node()
        norm.inputs["x"].value = np.log10(np.arange(1, 100))
        norm.inputs["y"].value = 2 * norm.inputs["x"].value
        self.assertIsInstance(norm, fn.Node)
        await norm
        x_new = norm.outputs["x_new"].value
        self.assertIsInstance(x_new, np.ndarray)
        self.assertEqual(len(x_new), 229)
        np.testing.assert_equal(x_new * 2, norm.outputs["y_new"].value)
        np.testing.assert_almost_equal(np.diff(x_new), x_new[1] - x_new[0])


class TestNorm(unittest.IsolatedAsyncioTestCase):
    async def test_zero_one(self):
        node = _norm()
        node.inputs["array"].value = np.array([0.0, 2.0, 5.0, 10.0])
        node.inputs["mode"].value = NormMode.ZERO_ONE
        await node
        result = node.outputs["out"].value
        self.assertIsInstance(result, np.ndarray)
        self.assertAlmostEqual(result.min(), 0.0)
        self.assertAlmostEqual(result.max(), 1.0)

    async def test_minus_one_one(self):
        node = _norm()
        node.inputs["array"].value = np.array([0.0, 2.0, 5.0, 10.0])
        node.inputs["mode"].value = NormMode.MINUS_ONE_ONE
        await node
        result = node.outputs["out"].value
        self.assertIsInstance(result, np.ndarray)
        self.assertAlmostEqual(result.min(), -1.0)
        self.assertAlmostEqual(result.max(), 1.0)

    async def test_sum_abs(self):
        node = _norm()
        node.inputs["array"].value = np.array([0.0, 2.0, 5.0, 10.0])
        node.inputs["mode"].value = NormMode.SUM_ABS
        await node
        result = node.outputs["out"].value
        self.assertAlmostEqual(float(np.abs(result).sum()), 1.0, places=9)

    async def test_euclidean(self):
        node = _norm()
        node.inputs["array"].value = np.array([3.0, 4.0])
        node.inputs["mode"].value = NormMode.EUCLIDEAN
        await node
        result = node.outputs["out"].value
        self.assertAlmostEqual(float(np.sqrt((result**2).sum())), 1.0, places=9)

    async def test_mean_std(self):
        node = _norm()
        node.inputs["array"].value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        node.inputs["mode"].value = NormMode.MEAN_STD
        await node
        result = node.outputs["out"].value
        self.assertAlmostEqual(float(result.mean()), 0.0, places=9)
        self.assertAlmostEqual(float(result.std()), 1.0, places=9)

    async def test_max(self):
        node = _norm()
        node.inputs["array"].value = np.array([1.0, 2.0, 4.0])
        node.inputs["mode"].value = NormMode.MAX
        await node
        result = node.outputs["out"].value
        self.assertAlmostEqual(float(result.max()), 1.0)


# ── density_normalization_node tests ─────────────────────────────────────────


class TestDensityNormalization(unittest.IsolatedAsyncioTestCase):
    async def test_evenly_spaced_returns_unchanged(self):
        node = density_normalization_node()
        node.inputs["x"].value = np.array([0.0, 1.0, 2.0, 3.0])
        node.inputs["y"].value = np.array([1.0, 2.0, 3.0, 4.0])
        await node
        np.testing.assert_array_equal(node.outputs["x_new"].value, [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(node.outputs["y_new"].value, [1.0, 2.0, 3.0, 4.0])

    async def test_uneven_spacing_auto_num_points(self):
        node = density_normalization_node()
        node.inputs["x"].value = np.array([0.0, 1.0, 3.0, 10.0])
        node.inputs["y"].value = np.array([0.0, 1.0, 2.0, 3.0])
        await node
        x_new = node.outputs["x_new"].value
        diffs = np.diff(x_new)
        self.assertTrue(np.allclose(diffs, diffs[0]), "x_new should be evenly spaced")

    async def test_single_unique_x_point(self):
        node = density_normalization_node()
        node.inputs["x"].value = np.array([5.0, 5.0, 5.0])
        node.inputs["y"].value = np.array([1.0, 2.0, 3.0])
        await node
        self.assertEqual(len(node.outputs["x_new"].value), 3)

    async def test_explicit_num_points(self):
        node = density_normalization_node()
        node.inputs["x"].value = np.array([0.0, 1.0, 3.0, 10.0])
        node.inputs["y"].value = np.array([0.0, 1.0, 2.0, 3.0])
        node.inputs["num_points"].value = 50
        await node
        self.assertEqual(len(node.outputs["x_new"].value), 50)

    async def test_distance_estimation_median(self):
        node = density_normalization_node()
        node.inputs["x"].value = np.array([0.0, 1.0, 3.0, 10.0])
        node.inputs["y"].value = np.array([0.0, 1.0, 2.0, 3.0])
        node.inputs["distance_estimation"].value = "median"
        await node
        self.assertGreater(len(node.outputs["x_new"].value), 0)

    async def test_distance_estimation_mean(self):
        node = density_normalization_node()
        node.inputs["x"].value = np.array([0.0, 1.0, 3.0, 10.0])
        node.inputs["y"].value = np.array([0.0, 1.0, 2.0, 3.0])
        node.inputs["distance_estimation"].value = "mean"
        await node
        self.assertGreater(len(node.outputs["x_new"].value), 0)

    async def test_distance_estimation_min(self):
        node = density_normalization_node()
        node.inputs["x"].value = np.array([0.0, 1.0, 3.0, 10.0])
        node.inputs["y"].value = np.array([0.0, 1.0, 2.0, 3.0])
        node.inputs["distance_estimation"].value = "min"
        await node
        self.assertGreater(len(node.outputs["x_new"].value), 0)

    async def test_distance_estimation_max(self):
        node = density_normalization_node()
        node.inputs["x"].value = np.array([0.0, 1.0, 3.0, 10.0])
        node.inputs["y"].value = np.array([0.0, 1.0, 2.0, 3.0])
        node.inputs["distance_estimation"].value = "max"
        await node
        self.assertGreater(len(node.outputs["x_new"].value), 0)
