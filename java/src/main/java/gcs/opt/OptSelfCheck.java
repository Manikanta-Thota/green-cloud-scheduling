package gcs.opt;

import gcs.core.*;
import java.nio.file.Path;
import java.util.*;

/**
 * Unit checks for the optimisation layer (milestone M3), run by `make test-java`:
 * dominance rules, archive invariants, indicator values against python/gcs/pareto.py, exact budget
 * accounting, determinism under a fixed seed, and validity of every algorithm's output.
 * Usage: java -cp java/out gcs.opt.OptSelfCheck testdata/parity_workflow_case.txt
 */
public final class OptSelfCheck {
    static int failures = 0;

    public static void main(String[] args) throws Exception {
        // dominance and feasibility-first dominance
        check("dominates", Pareto.dominates(new double[] {1, 1}, new double[] {2, 2}));
        check("no self-dominance", !Pareto.dominates(new double[] {1, 1}, new double[] {1, 1}));
        Solution feas = new Solution(null, new double[] {9, 9}, 0), infeas = new Solution(null, new double[] {1, 1}, 0.2),
                 lessInfeas = new Solution(null, new double[] {5, 5}, 0.1);
        check("feasible beats infeasible", Pareto.cdominates(feas, infeas) && !Pareto.cdominates(infeas, feas));
        check("smaller violation wins", Pareto.cdominates(lessInfeas, infeas));

        // indicators against values computed by python/gcs/pareto.py
        double[][] pts = {{0.805, 0.808, 0.515, 0.286, 0.054}, {0.383, 0.408, 0.045, 0.049, 0.999},
            {0.652, 0.235, 0.435, 0.974, 0.898}, {0.844, 0.392, 0.493, 0.677, 0.061}, {0.556, 0.271, 0.88, 0.064, 0.679},
            {0.87, 0.227, 0.895, 0.872, 0.019}, {0.707, 0.001, 0.503, 0.437, 0.203}, {0.325, 0.806, 0.316, 0.149, 0.699},
            {0.449, 0.799, 0.236, 0.32, 0.8}, {0.507, 0.506, 0.236, 0.015, 0.933}, {0.086, 0.845, 0.368, 0.951, 0.399},
            {0.936, 0.556, 0.24, 0.741, 0.674}};
        List<double[]> all = Arrays.asList(pts);
        double[] ref = {1.1, 1.1, 1.1, 1.1, 1.1};
        close("hypervolume 5-d", Indicators.hypervolume(all, ref), 0.28660899058027023);
        close("igd", Indicators.igd(all.subList(0, 4), all), 0.4029289349786758);
        close("hypervolume 2-d", Indicators.hypervolume(List.of(new double[] {1, 2}, new double[] {2, 1}), new double[] {3, 3}), 3.0);

        // archive: members mutually non-dominated, capacity respected
        SplittableRandom r = new SplittableRandom(1);
        Archive arch = new Archive(10);
        for (int i = 0; i < 500; i++) arch.add(new Solution(null, new double[] {r.nextDouble(), r.nextDouble(), r.nextDouble()}, 0));
        check("archive capacity", arch.size() <= 10);
        boolean nd = true;
        for (Solution a : arch.members()) for (Solution b : arch.members()) if (a != b && Pareto.cdominates(a, b)) nd = false;
        check("archive non-dominated", nd);

        // every algorithm: exact budget, valid assignment, determinism
        Instance in = CaseFile.read(Path.of(args[0])).instance;
        int budget = 600;
        for (String algo : List.of("ga", "ga_seeded", "mogwo", "mogwo_seeded", "hpeas", "hpeas_noseed", "hpeas_nols",
                                   "hpeas_static", "nsga2")) {
            Map<String, String> prm = Map.of("pop", "20", "archive", "30", "ls_steps", "10");
            List<Solution> a = run(algo, in, budget, prm, 42), b = run(algo, in, budget, prm, 42);
            boolean valid = true;
            for (Solution s : a) {
                for (int v : s.x) if (v < 0 || v >= in.nVms()) valid = false;
                Objectives o = Evaluator.evaluate(in, s.x);   // stored objectives are the evaluator's
                if (Math.abs(o.energy() - s.f[Evaluator.ENERGY]) > 1e-6 * o.energy()) valid = false;
            }
            check(algo + " valid output", valid && !a.isEmpty());
            check(algo + " deterministic", a.size() == b.size() && Arrays.equals(Selection.minEnergy(a).f, Selection.minEnergy(b).f));
        }
        Problem p = new Problem(in, budget, 0.05, 0.05, 0.01);
        RunExperiment.make("hpeas", new RunExperiment.Params(Map.of("pop", "20"))).run(p, new SplittableRandom(3));
        check("budget spent exactly", p.evaluations() == budget);
        check("seeded HPEAS keeps the reference envelope", p.trace[Problem.TRACE_POINTS - 1] <= p.ref.f[Evaluator.ENERGY]);

        System.out.println(failures == 0 ? "OptSelfCheck: all checks passed" : "OptSelfCheck: " + failures + " FAILED");
        System.exit(failures == 0 ? 0 : 1);
    }

    static List<Solution> run(String algo, Instance in, int budget, Map<String, String> prm, long seed) {
        Problem p = new Problem(in, budget, 0.05, 0.05, 0.01);
        List<Solution> out = RunExperiment.make(algo, new RunExperiment.Params(prm)).run(p, new SplittableRandom(seed));
        if (p.evaluations() != budget) { System.out.println("  " + algo + " used " + p.evaluations() + " evaluations"); failures++; }
        return out;
    }

    static void check(String name, boolean ok) {
        System.out.printf("  %-45s %s%n", name, ok ? "ok" : "FAILED");
        if (!ok) failures++;
    }

    static void close(String name, double got, double want) {
        check(name + String.format(" (%.12f)", got), Math.abs(got - want) <= 1e-9 * Math.max(1, Math.abs(want)));
    }
}
