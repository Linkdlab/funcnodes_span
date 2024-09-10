import sys
import os
import unittest
import funcnodes_span as fnmodule  # noqa
from funcnodes_span import peak_analysis

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from all_nodes_test_base import TestAllNodesBase  # noqa E402

from . import (  # noqa E402
    test_normalization,
    test_smoothing,
    test_peak_analysis,
    test_baseline,
)

sub_test_classes = []

for mod in (
    test_normalization,
    test_smoothing,
    test_peak_analysis,
    test_baseline,
):
    for cls in mod.__dict__.values():
        if isinstance(cls, type) and issubclass(cls, unittest.IsolatedAsyncioTestCase):
            sub_test_classes.append(cls)


class TestAllNodes(TestAllNodesBase):
    # in this test class all nodes should be triggered at least once to mark them as testing

    sub_test_classes = sub_test_classes

    ignore_nodes = [
        peak_analysis.plot_peaks,
        peak_analysis.plot_fitted_peaks,
        peak_analysis.force_peak_finder,
    ]