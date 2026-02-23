import unittest

import numpy as np
from leco.model.interaction import nearest_neighbors


class TestNearestNeighbors(unittest.TestCase):
    def test_self_not_in_neighbors(self):
        """Test that agents are not their own neighbors."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        radius = 5.0
        result = nearest_neighbors(positions, radius)

        for i, neighbor_list in enumerate(result):
            self.assertNotIn(i, neighbor_list)

    def test_correct_number_of_neighbor_lists(self):
        """Test that we get one neighbor list per agent."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        radius = 2.0
        result = nearest_neighbors(positions, radius)

        self.assertEqual(len(result), len(positions))
