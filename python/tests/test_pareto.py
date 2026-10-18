import unittest
import numpy as np
from gcs.pareto import crowding_distance, dominates, hypervolume, igd, non_dominated_sort


class ParetoTests(unittest.TestCase):
    def test_dominates(self):
        self.assertTrue(dominates([1, 1], [2, 2]))
        self.assertFalse(dominates([1, 2], [2, 1]))
        self.assertFalse(dominates([1, 1], [1, 1]))

    def test_sort_fronts(self):
        F = [[1, 4], [2, 2], [4, 1], [3, 3], [5, 5]]
        self.assertEqual([sorted(f) for f in non_dominated_sort(F)], [[0, 1, 2], [3], [4]])

    def test_crowding(self):
        d = crowding_distance([[0, 4], [1, 2], [4, 0]])
        self.assertTrue(np.isinf(d[0]) and np.isinf(d[2]))
        self.assertAlmostEqual(d[1], (4 - 0) / 4 + (4 - 0) / 4)

    def test_hypervolume_2d(self):
        self.assertAlmostEqual(hypervolume([[1, 2], [2, 1]], [3, 3]), 3.0)

    def test_hypervolume_3d_single_box(self):
        self.assertAlmostEqual(hypervolume([[1, 1, 1]], [3, 3, 3]), 8.0)

    def test_hypervolume_ignores_points_beyond_reference(self):
        self.assertAlmostEqual(hypervolume([[1, 1], [5, 0.5]], [3, 3]), 4.0)

    def test_igd(self):
        self.assertAlmostEqual(igd([[0, 0]], [[3, 4], [0, 0]]), 2.5)


if __name__ == "__main__":
    unittest.main()
