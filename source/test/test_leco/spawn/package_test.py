import unittest

import numpy as np
import leco.spawn as spawn
from leco.spawn.configuration import as_string, expand_range, expand_set


class PackageTest(unittest.TestCase):
    def test_all_attribute(self):
        self.assertTrue(hasattr(spawn, "__all__"))
        self.assertEqual(spawn.__all__, ["cluster", "default_max_nr_workers", "run"])

    def test_default_max_nr_workers(self):
        self.assertGreater(spawn.default_max_nr_workers(), 0)

    def test_expand_range(self):
        parameter = {"range": [1, 5, 1]}
        self.assertEqual(list(expand_range(parameter)), [1, 2, 3, 4])

        parameter = {"range": [1.0, 1.6, 0.1]}
        np.testing.assert_almost_equal(list(expand_range(parameter)), [1.0, 1.1, 1.2, 1.3, 1.4, 1.5])

    def test_expand_set(self):
        parameter = {"set": [1, 5, 1]}
        self.assertEqual(list(expand_set(parameter)), [1, 5])

        parameter = {"set": [1.0, 1.6, 0.1]}
        np.testing.assert_almost_equal(list(expand_set(parameter)), [0.1, 1.0, 1.6])

    def test_as_string(self):
        self.assertEqual(as_string(1), "1")
        self.assertEqual(int("1"), 1)

        self.assertEqual(as_string(0.1), "0.1")
        self.assertEqual(float("0.1"), 0.1)
