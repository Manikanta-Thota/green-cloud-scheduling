import unittest
import numpy as np
from gcs.baselines import fcfs, min_min
from gcs.model import evaluate
from gcs.scenarios import instance_seed, load_config, scenario_matrix
from gcs.workload import generate


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.matrix = scenario_matrix(self.cfg)

    def test_matrix_size(self):
        self.assertEqual(len(self.matrix), 81)
        self.assertEqual(len({m["scenario_id"] for m in self.matrix}), 81)

    def test_same_seed_same_instance(self):
        sc = self.matrix[0]
        a = generate(self.cfg, sc, instance_seed(self.cfg, 0, 7))
        b = generate(self.cfg, sc, instance_seed(self.cfg, 0, 7))
        np.testing.assert_array_equal(a.length, b.length)
        np.testing.assert_array_equal(a.arrival, b.arrival)

    def test_different_replicates_differ(self):
        sc = self.matrix[0]
        a = generate(self.cfg, sc, instance_seed(self.cfg, 0, 0))
        b = generate(self.cfg, sc, instance_seed(self.cfg, 0, 1))
        self.assertFalse(np.array_equal(a.length, b.length))

    def test_arrivals_sorted_and_deadlines_after_arrival(self):
        for sc in self.matrix:
            if sc["arrival"] == "workflow" or sc["tasks"] != 500:
                continue
            inst = generate(self.cfg, sc, instance_seed(self.cfg, sc["scenario_id"], 0))
            self.assertTrue(np.all(np.diff(inst.arrival) >= 0))
            self.assertTrue(np.all(inst.deadline > inst.arrival))

    def test_workflow_not_implemented_yet(self):
        sc = next(m for m in self.matrix if m["arrival"] == "workflow")
        with self.assertRaises(NotImplementedError):
            generate(self.cfg, sc, instance_seed(self.cfg, sc["scenario_id"], 0))

    def test_baselines_produce_valid_objectives(self):
        sc = next(m for m in self.matrix if m["tasks"] == 500 and m["vms"] == 20 and m["arrival"] == "bursty")
        inst = generate(self.cfg, sc, instance_seed(self.cfg, sc["scenario_id"], 0))
        for fn in (fcfs, min_min):
            obj = evaluate(inst, fn(inst))
            self.assertTrue(all(np.isfinite(v) and v >= 0 for v in obj.values()))
            self.assertLessEqual(obj["sla"], 1.0)


if __name__ == "__main__":
    unittest.main()
