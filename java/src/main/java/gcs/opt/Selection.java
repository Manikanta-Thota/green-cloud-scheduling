package gcs.opt;

import gcs.core.Evaluator;
import java.util.List;

/** Rules that pick the one schedule to deploy from a set of trade-off solutions (decided before the experiment). */
public final class Selection {
    private Selection() {}

    /**
     * Primary rule, "tolerance-constrained minimum energy": the lowest-energy solution inside the
     * tolerance envelope; if none is feasible, the one with the smallest violation (then lowest energy).
     */
    public static Solution minEnergy(List<Solution> sols) {
        Solution best = null;
        for (Solution s : sols) {
            if (best == null) { best = s; continue; }
            if (s.feasible() != best.feasible()) { if (s.feasible()) best = s; continue; }
            if (!s.feasible() && s.cv != best.cv) { if (s.cv < best.cv) best = s; continue; }
            if (s.f[Evaluator.ENERGY] < best.f[Evaluator.ENERGY]) best = s;
        }
        return best;
    }

    /**
     * Sensitivity rule, "knee": among feasible solutions (all, if none is feasible), the one closest to
     * the ideal point after min-max normalising each objective over the set.
     */
    public static Solution knee(List<Solution> sols) {
        boolean anyFeasible = sols.stream().anyMatch(Solution::feasible);
        List<Solution> c = sols.stream().filter(s -> !anyFeasible || s.feasible()).toList();
        int d = c.get(0).f.length;
        double[] lo = new double[d], hi = new double[d];
        for (int k = 0; k < d; k++) { lo[k] = Double.POSITIVE_INFINITY; hi[k] = Double.NEGATIVE_INFINITY; }
        for (Solution s : c) for (int k = 0; k < d; k++) { lo[k] = Math.min(lo[k], s.f[k]); hi[k] = Math.max(hi[k], s.f[k]); }
        Solution best = null;
        double bestD = Double.POSITIVE_INFINITY;
        for (Solution s : c) {
            double dist = 0;
            for (int k = 0; k < d; k++) {
                double z = hi[k] > lo[k] ? (s.f[k] - lo[k]) / (hi[k] - lo[k]) : 0;
                dist += z * z;
            }
            if (dist < bestD) { bestD = dist; best = s; }
        }
        return best;
    }
}
