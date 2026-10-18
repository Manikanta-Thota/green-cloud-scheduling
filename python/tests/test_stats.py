import unittest
import numpy as np
from gcs.stats import bootstrap_ci, cliffs_delta, holm_adjust, practical_acceptance


class StatsTests(unittest.TestCase):
    def test_cliffs_delta_extremes(self):
        self.assertEqual(cliffs_delta([4, 5, 6], [1, 2, 3]), 1.0)
        self.assertEqual(cliffs_delta([1, 2, 3], [4, 5, 6]), -1.0)
        self.assertEqual(cliffs_delta([1, 2, 3], [1, 2, 3]), 0.0)

    def test_holm(self):
        np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_bootstrap_ci_brackets_median(self):
        v = np.arange(1, 32, dtype=float)
        lo, hi = bootstrap_ci(v, resamples=2000, seed=1)
        self.assertLess(lo, np.median(v)); self.assertGreater(hi, np.median(v))

    def test_practical_acceptance(self):
        rules = {"max_makespan_degradation": 0.05, "max_cost_increase": 0.05, "max_sla_increase_pp": 1.0}
        base = {"makespan": 100.0, "cost": 10.0, "sla": 0.02}
        ok = practical_acceptance({"makespan": 104.0, "cost": 10.4, "sla": 0.025}, base, rules)
        bad = practical_acceptance({"makespan": 106.0, "cost": 10.0, "sla": 0.02}, base, rules)
        self.assertTrue(ok["within_tolerance"]); self.assertFalse(bad["within_tolerance"])


if __name__ == "__main__":
    unittest.main()
