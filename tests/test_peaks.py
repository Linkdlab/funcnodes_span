import unittest
import numpy as np
from funcnodes_span.peaks import (
    PeakProperties,
    calculate_peak_area,
    calculate_peak_symmetricity,
)


# ── shared fixture data ───────────────────────────────────────────────────────


def make_gaussian(n=100, center=50, width=10, amplitude=1.0):
    """A simple gaussian peak for testing."""
    x = np.linspace(0, 100, n)
    y = amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
    return x, y


def make_peak(i=30, idx=50, f=70, n=100):
    """A PeakProperties with x and y arrays attached."""
    x, y = make_gaussian(n=n)
    return PeakProperties(id="test", i_index=i, index=idx, f_index=f, xfull=x, yfull=y)


# ── PeakProperties construction ───────────────────────────────────────────────


class TestPeakPropertiesConstruction(unittest.TestCase):
    def test_basic_construction(self):
        p = PeakProperties(id="p1", i_index=0, index=5, f_index=10)
        self.assertEqual(p.id, "p1")
        self.assertEqual(p.i_index, 0)
        self.assertEqual(p.index, 5)
        self.assertEqual(p.f_index, 10)

    def test_negative_indices_raise(self):
        # line 26
        with self.assertRaises(ValueError):
            PeakProperties(id="p", i_index=-1, index=5, f_index=10)

    def test_index_order_violated_raises(self):
        # line 32 — index not between i_index and f_index
        with self.assertRaises(ValueError):
            PeakProperties(id="p", i_index=0, index=15, f_index=10)

    def test_extra_properties_stored(self):
        # lines 46, 56-60 — add_serializable_property and __getattribute__
        p = PeakProperties(id="p", i_index=0, index=5, f_index=10, custom_prop=42)
        self.assertEqual(p.custom_prop, 42)

    def test_missing_attribute_raises(self):
        p = PeakProperties(id="p", i_index=0, index=5, f_index=10)
        with self.assertRaises(AttributeError):
            _ = p.nonexistent_attribute


# ── xfull / yfull setters ─────────────────────────────────────────────────────


class TestPeakPropertiesArraySetters(unittest.TestCase):
    def test_xfull_already_set_raises(self):
        # line 92
        x, y = make_gaussian()
        p = PeakProperties(id="p", i_index=30, index=50, f_index=70, xfull=x)
        with self.assertRaises(ValueError, msg="The full x-array has already been set"):
            p.xfull = x

    def test_yfull_already_set_raises(self):
        # line 107
        x, y = make_gaussian()
        p = PeakProperties(id="p", i_index=30, index=50, f_index=70, yfull=y)
        with self.assertRaises(ValueError, msg="The full y-array has already been set"):
            p.yfull = y

    def test_xfull_length_mismatch_raises(self):
        # line 94-95
        x, y = make_gaussian(n=100)
        p = PeakProperties(id="p", i_index=30, index=50, f_index=70, yfull=y)
        with self.assertRaises(ValueError):
            p.xfull = np.linspace(0, 100, 50)  # wrong length

    def test_yfull_length_mismatch_raises(self):
        # line 108
        x, y = make_gaussian(n=100)
        p = PeakProperties(id="p", i_index=30, index=50, f_index=70, xfull=x)
        with self.assertRaises(ValueError):
            p.yfull = np.linspace(0, 1, 50)  # wrong length

    def test_xfull_too_short_raises(self):
        p = PeakProperties(id="p", i_index=30, index=50, f_index=70)
        with self.assertRaises(ValueError):
            p.xfull = np.linspace(0, 1, 10)  # shorter than f_index+1

    def test_xfull_order_violated_raises(self):
        # line 100 — x_at_index not between x_at_i_index and x_at_f_index
        x = np.linspace(0, 100, 100)
        x_reversed = x[::-1].copy()  # descending: x[50] < x[30]
        p = PeakProperties(id="p", i_index=30, index=50, f_index=70)
        with self.assertRaises(ValueError):
            p.xfull = x_reversed


# ── properties returning None when arrays not set ─────────────────────────────


class TestPeakPropertiesWithoutArrays(unittest.TestCase):
    def test_xrange_none_without_xfull(self):
        # line 119
        p = PeakProperties(id="p", i_index=0, index=5, f_index=10)
        self.assertIsNone(p.xrange)

    def test_yrange_none_without_yfull(self):
        # line 122
        p = PeakProperties(id="p", i_index=0, index=5, f_index=10)
        self.assertIsNone(p.yrange)

    def test_yrange_corrected_none_without_arrays(self):
        # line 127
        p = PeakProperties(id="p", i_index=0, index=5, f_index=10)
        self.assertIsNone(p.yrange_corrected)

    def test_x_at_i_index_raises_without_xfull(self):
        # line 150
        p = PeakProperties(id="p", i_index=0, index=5, f_index=10)
        with self.assertRaises(ValueError):
            _ = p.x_at_i_index

    def test_x_at_index_raises_without_xfull(self):
        # line 156
        p = PeakProperties(id="p", i_index=0, index=5, f_index=10)
        with self.assertRaises(ValueError):
            _ = p.x_at_index

    def test_x_at_f_index_raises_without_xfull(self):
        # line 162
        p = PeakProperties(id="p", i_index=0, index=5, f_index=10)
        with self.assertRaises(ValueError):
            _ = p.x_at_f_index

    def test_y_at_i_index_raises_without_yfull(self):
        # lines 165-168
        p = PeakProperties(id="p", i_index=0, index=5, f_index=10)
        with self.assertRaises(ValueError):
            _ = p.y_at_i_index

    def test_y_at_index_raises_without_yfull(self):
        # line 173
        p = PeakProperties(id="p", i_index=0, index=5, f_index=10)
        with self.assertRaises(ValueError):
            _ = p.y_at_index

    def test_y_at_f_index_raises_without_yfull(self):
        # line 179
        p = PeakProperties(id="p", i_index=0, index=5, f_index=10)
        with self.assertRaises(ValueError):
            _ = p.y_at_f_index


# ── positive property ─────────────────────────────────────────────────────────


class TestPeakPositive(unittest.TestCase):
    def test_positive_peak(self):
        # line 64 — gaussian is positive (max in middle)
        p = make_peak()
        self.assertTrue(p.positive)

    def test_negative_peak(self):
        # line 64 — inverted gaussian
        x, y = make_gaussian()
        p = PeakProperties(id="p", i_index=30, index=50, f_index=70, xfull=x, yfull=-y)
        self.assertFalse(p.positive)


# ── get_width_at_height ───────────────────────────────────────────────────────


class TestGetWidthAtHeight(unittest.TestCase):
    def test_fwhm(self):
        # lines 229-259, 270
        p = make_peak()
        fwhm = p.fwhm
        self.assertGreater(fwhm, 0)

    def test_width_base(self):
        # line 274
        p = make_peak()
        self.assertGreater(p.width_base, 0)

    def test_invalid_height_raises(self):
        # height < 0
        p = make_peak()
        with self.assertRaises(ValueError):
            p.get_width_at_height(-0.1)

    def test_height_above_one_raises(self):
        # height > 1
        p = make_peak()
        with self.assertRaises(ValueError):
            p.get_width_at_height(1.1)

    def test_corrected_width(self):
        p = make_peak()
        w, *_ = p.get_width_at_height(0.5, corrected=True)
        self.assertGreater(w, 0)

    def test_relative_height(self):
        p = make_peak()
        w, *_ = p.get_width_at_height(0.5, absolute=False)
        self.assertGreater(w, 0)


# ── __str__ and to_dict ───────────────────────────────────────────────────────


class TestPeakStrAndDict(unittest.TestCase):
    def test_str(self):
        # line 277
        p = make_peak()
        s = str(p)
        self.assertIn("PeakProperties", s)
        self.assertIn("test", s)

    def test_to_dict(self):
        # lines 280-285
        p = make_peak()
        d = p.to_dict()
        self.assertIn("id", d)
        self.assertIn("x_at_index", d)
        self.assertIn("area", d)
        self.assertIn("symmetricity", d)

    def test_to_dict_no_calc(self):
        p = make_peak()
        d = p.to_dict(calc_props=False)
        self.assertIn("id", d)
        self.assertNotIn("area", d)


# ── calculate_peak_area ───────────────────────────────────────────────────────


class TestCalculatePeakArea(unittest.TestCase):
    def test_trapz(self):
        # lines 323-355
        p = make_peak()
        area = calculate_peak_area(p, method="trapz")
        self.assertGreater(area, 0)

    def test_simps(self):
        p = make_peak()
        area = calculate_peak_area(p, method="simps")
        self.assertGreater(area, 0)

    def test_absolute(self):
        x, y = make_gaussian()
        p = PeakProperties(id="p", i_index=30, index=50, f_index=70, xfull=x, yfull=-y)
        area = calculate_peak_area(p, absolute=True)
        self.assertGreater(area, 0)

    def test_relative_to_baseline(self):
        p = make_peak()
        area = calculate_peak_area(p, relative_to_baseline=True)
        self.assertGreater(area, 0)

    def test_unsupported_method_raises(self):
        p = make_peak()
        with self.assertRaises(ValueError):
            calculate_peak_area(p, method="unknown")

    def test_no_add_property(self):
        p = make_peak()
        calculate_peak_area(p, add_property=False)
        self.assertNotIn("area", p._serdata)

    # def test_negative_peak_area(self): #TODO
    #     x, y = make_gaussian()
    #     p = PeakProperties(id="p", i_index=30, index=50, f_index=70,
    #                        xfull=x, yfull=-y)
    #     area = calculate_peak_area(p)
    #     self.assertGreater(area, 0)   # sign is adjusted for negative peaks


# ── calculate_peak_symmetricity ───────────────────────────────────────────────


class TestCalculatePeakSymmetricity(unittest.TestCase):
    # def test_area_method(self): #TODO
    #     # lines 363-416
    #     p = make_peak()
    #     s = calculate_peak_symmetricity(p, method="area")
    #     self.assertGreater(s, 0)
    #     self.assertLessEqual(s, 1.0)

    # def test_area_simps_method(self): #TODO
    #     p = make_peak()
    #     s = calculate_peak_symmetricity(p, method="area_simps")
    #     self.assertGreater(s, 0)

    def test_h5p_method(self):
        p = make_peak()
        s = calculate_peak_symmetricity(p, method="h5p")
        self.assertGreater(s, 0)

    def test_fwhm_method(self):
        p = make_peak()
        s = calculate_peak_symmetricity(p, method="fwhm")
        self.assertGreater(s, 0)

    def test_skewness_method(self):
        p = make_peak()
        s = calculate_peak_symmetricity(p, method="skewness")
        self.assertIsInstance(s, float)

    def test_skewness_uneven_x(self):
        # hits the interpolation branch inside skewness
        x = np.array(
            [
                0,
                1,
                3,
                6,
                10,
                15,
                21,
                28,
                36,
                45,
                50,
                55,
                63,
                70,
                76,
                81,
                85,
                88,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
                100,
            ],
            dtype=float,
        )
        y = np.exp(-0.5 * ((x - 50) / 10) ** 2)
        p = PeakProperties(id="p", i_index=3, index=10, f_index=25, xfull=x, yfull=y)
        s = calculate_peak_symmetricity(p, method="skewness")
        self.assertIsInstance(s, float)

    def test_unsupported_method_raises(self):
        p = make_peak()
        with self.assertRaises(ValueError):
            calculate_peak_symmetricity(p, method="unknown")

    def test_no_add_property(self):
        p = make_peak()
        calculate_peak_symmetricity(p, add_property=False)
        self.assertNotIn("symmetricity", p._serdata)
