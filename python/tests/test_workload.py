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

    def test_workflow_instances_are_valid_dags(self):
        for sc in self.matrix:
            if sc["arrival"] != "workflow" or sc["vms"] != 20 or sc["heterogeneity"] != "medium":
                continue
            inst = generate(self.cfg, sc, instance_seed(self.cfg, sc["scenario_id"], 0))
            self.assertEqual(inst.n_tasks, sc["tasks"])
            pos = np.empty(inst.n_tasks, dtype=int)
            pos[inst.order()] = np.arange(inst.n_tasks)
            n_edges = 0
            for t, ps in enumerate(inst.preds):
                for p in ps:   # predecessors come first in the execution order, same workflow arrival, lower rank
                    self.assertLess(pos[p], pos[t])
                    self.assertEqual(inst.arrival[p], inst.arrival[t])
                    self.assertLess(inst.rank[p], inst.rank[t])
                    n_edges += 1
            self.assertGreater(n_edges, inst.n_tasks // 2)
            self.assertTrue(np.all(inst.deadline > inst.arrival))
            self.assertAlmostEqual(inst.length.mean(), self.cfg["fixed_parameters"]["task_length_mean_mi"], places=6)

    def test_job_log_instances_keep_offered_load_and_mean_length(self):
        p = self.cfg["fixed_parameters"]
        for arrival in ("steady", "bursty"):
            sc = next(m for m in self.matrix if m["arrival"] == arrival and m["tasks"] == 5000 and m["vms"] == 20)
            inst = generate(self.cfg, sc, instance_seed(self.cfg, sc["scenario_id"], 0))
            self.assertEqual(inst.n_tasks, 5000)
            self.assertAlmostEqual(inst.length.mean(), p["task_length_mean_mi"], places=6)
            load = inst.length.sum() / inst.arrival[-1] / inst.mips.sum()
            self.assertAlmostEqual(load, p["datacentre_load"], places=9)

    def test_bursty_windows_more_irregular_than_steady(self):
        cv = {}
        for arrival in ("steady", "bursty"):
            sc = next(m for m in self.matrix if m["arrival"] == arrival and m["tasks"] == 2000 and m["vms"] == 50)
            vals = []
            for r in range(10):
                gaps = np.diff(generate(self.cfg, sc, instance_seed(self.cfg, sc["scenario_id"], r)).arrival)
                vals.append(gaps.std() / gaps.mean())
            cv[arrival] = np.median(vals)
        self.assertGreater(cv["bursty"], cv["steady"])

    def test_job_lengths_are_one_rescaled_window_of_a_log(self):
        from gcs.workload import job_log, log_catalogue
        n = 500
        sc = next(m for m in self.matrix if m["arrival"] == "steady" and m["tasks"] == n)
        inst = generate(self.cfg, sc, instance_seed(self.cfg, sc["scenario_id"], 3))
        rel = inst.length / inst.length[0]
        found = False
        for log, _ in log_catalogue():
            submit, run = job_log(log)
            self.assertTrue(np.all(run > 0) and np.all(np.diff(submit) >= 0))
            cand = np.flatnonzero(np.isclose(run[1:len(run) - n + 1] / run[:len(run) - n], rel[1], rtol=1e-9))
            found = found or any(np.allclose(run[s:s + n] / run[s], rel, rtol=1e-9) for s in cand)
        self.assertTrue(found, "instance lengths are not one rescaled window of a job log")

    def test_price_grows_faster_than_speed(self):
        inst = generate(self.cfg, self.matrix[2], instance_seed(self.cfg, 2, 0))
        per_mi = inst.price / inst.mips
        self.assertTrue(np.all(np.diff(per_mi[np.argsort(inst.mips)]) > 0))

    def test_mini_yaml_matches_structure(self):
        from gcs.scenarios import DEFAULT_CONFIG, _mini_yaml
        with open(DEFAULT_CONFIG) as f:
            cfg = _mini_yaml(f.read())
        self.assertEqual(cfg["scenario_factors"]["tasks"], [500, 2000, 5000])
        self.assertEqual(cfg["power_profiles"]["specpower_2008"]["p_idle_w"], [86.0, 93.7])
        self.assertIn("hpeas", cfg["algorithms"])
        self.assertEqual(cfg["decision_rules"]["max_cost_increase"], 0.05)

    def test_baselines_produce_valid_objectives(self):
        sc = next(m for m in self.matrix if m["tasks"] == 500 and m["vms"] == 20 and m["arrival"] == "bursty")
        inst = generate(self.cfg, sc, instance_seed(self.cfg, sc["scenario_id"], 0))
        for fn in (fcfs, min_min):
            obj = evaluate(inst, fn(inst))
            self.assertTrue(all(np.isfinite(v) and v >= 0 for v in obj.values()))
            self.assertLessEqual(obj["sla"], 1.0)


if __name__ == "__main__":
    unittest.main()
