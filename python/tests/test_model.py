import os
import unittest
import numpy as np
from gcs.model import evaluate, load_case, simulate
from gcs.baselines import fcfs, min_min

CASE = os.path.join(os.path.dirname(__file__), "..", "..", "testdata", "tiny_case.txt")


class HandCheckedCase(unittest.TestCase):
    """Proposal Section 3.4: small hand-checkable examples for time, energy, cost, utilisation and SLA."""

    def setUp(self):
        self.inst, self.assign, self.expect = load_case(CASE)

    def test_times(self):
        sim = simulate(self.inst, self.assign)
        np.testing.assert_allclose(sim["finish"], [2.0, 2.0, 3.0])
        np.testing.assert_allclose(sim["busy"], [3.0, 2.0])

    def test_objectives(self):
        obj = evaluate(self.inst, self.assign)
        for key in ("makespan", "energy", "cost", "imbalance", "sla"):
            self.assertAlmostEqual(obj[key], self.expect[key], places=6, msg=key)

    def test_fcfs(self):
        self.assertEqual(list(fcfs(self.inst)), self.expect["expect_fcfs"])

    def test_minmin(self):
        self.assertEqual(list(min_min(self.inst)), self.expect["expect_minmin"])

    def test_invalid_assignment(self):
        with self.assertRaises(ValueError):
            evaluate(self.inst, [0, 1, 5])


class Properties(unittest.TestCase):
    def test_energy_at_least_idle_floor(self):
        inst, assign, _ = load_case(CASE)
        obj = evaluate(inst, assign)
        floor = float(inst.p_idle.sum()) * obj["makespan"]
        self.assertGreaterEqual(obj["energy"], floor)

    def test_single_vm_has_zero_imbalance(self):
        inst, _, _ = load_case(CASE)
        inst.mips, inst.price, inst.host_of = inst.mips[:1], inst.price[:1], inst.host_of[:1]
        inst.p_idle, inst.p_max = inst.p_idle[:1], inst.p_max[:1]
        self.assertAlmostEqual(evaluate(inst, [0, 0, 0])["imbalance"], 0.0)


if __name__ == "__main__":
    unittest.main()
