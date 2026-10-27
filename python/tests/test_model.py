import os
import unittest
import numpy as np
from gcs.model import evaluate, load_case, simulate
from gcs.baselines import fcfs, min_min

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "testdata")
CASE = os.path.join(DATA, "tiny_case.txt")
PARITY = os.path.join(DATA, "parity_case.txt")
PARITY_WF = os.path.join(DATA, "parity_workflow_case.txt")


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


class ParityCases(unittest.TestCase):
    """The parity files hold Python's results when they were exported; the Java SelfCheck reads the same files."""

    def test_reproduce_exported_results(self):
        for case in (PARITY, PARITY_WF):
            inst, assign, expect = load_case(case)
            obj = evaluate(inst, assign)
            for key in ("makespan", "energy", "cost", "imbalance", "sla"):
                self.assertAlmostEqual(obj[key], expect[key], delta=1e-9 * max(1.0, abs(expect[key])), msg=key)
            self.assertEqual(list(fcfs(inst)), expect["expect_fcfs"])
            self.assertEqual(list(min_min(inst)), expect["expect_minmin"])


class Properties(unittest.TestCase):
    def test_energy_matches_closed_form(self):
        """Event-by-event integration equals p_idle * on-time + (p_max - p_idle) / capacity * MI per host
        (the closed form the Java evaluator uses), on the hand case and on both parity cases."""
        for case in (CASE, PARITY, PARITY_WF):
            inst, assign, _ = load_case(case)
            sim = simulate(inst, assign)
            host = inst.host_of[sim["assign"]]
            cap = np.bincount(inst.host_of, weights=inst.mips, minlength=len(inst.p_idle))
            closed = 0.0
            for h in np.unique(host):
                on = sim["finish"][host == h].max() - sim["start"][host == h].min()
                closed += inst.p_idle[h] * on + (inst.p_max[h] - inst.p_idle[h]) / cap[h] * inst.length[host == h].sum()
            self.assertAlmostEqual(evaluate(inst, assign)["energy"], closed, delta=1e-9 * closed)

    def test_unused_host_draws_no_power(self):
        inst, _, _ = load_case(CASE)
        # everything on VM 0 (host 0): 4000 MI at 1000 MIPS -> on 0-4 s, 100 W idle + 100 W x 4 s dynamic
        self.assertAlmostEqual(evaluate(inst, [0, 0, 0])["energy"], 100 * 4 + 100 * 4)

    def test_single_vm_has_zero_imbalance(self):
        inst, _, _ = load_case(CASE)
        inst.mips, inst.price, inst.host_of = inst.mips[:1], inst.price[:1], inst.host_of[:1]
        inst.p_idle, inst.p_max = inst.p_idle[:1], inst.p_max[:1]
        self.assertAlmostEqual(evaluate(inst, [0, 0, 0])["imbalance"], 0.0)


if __name__ == "__main__":
    unittest.main()
