package gcs.core;

import java.util.Arrays;

/**
 * Energy-preferred greedy heuristic (HPEAS seed). Tasks in execution order; each goes to the VM with the
 * smallest marginal energy (extra host on-time at idle power plus dynamic energy) among the VMs that
 * still meet the task's deadline, preferring the earlier finish on ties. If no VM meets the deadline
 * the task goes to the VM that finishes it earliest. This consolidates work onto few, efficient hosts.
 */
public final class EnergyGreedyScheduler implements Scheduler {
    @Override public String name() { return "energy_greedy"; }

    @Override public int[] schedule(Instance in) {
        int n = in.nTasks(), m = in.nVms(), nh = in.nHosts();
        double[] free = new double[m], finish = new double[n];
        double[] on = new double[nh], off = new double[nh];
        Arrays.fill(on, Double.POSITIVE_INFINITY);
        Arrays.fill(off, Double.NEGATIVE_INFINITY);
        int[] assign = new int[n];
        for (int t : in.order) {
            double ready = in.arrival[t];
            for (int p : in.preds[t]) ready = Math.max(ready, finish[p]);
            int best = -1, fastest = 0;
            double bestDe = 0, bestF = 0, fastestF = Double.POSITIVE_INFINITY;
            for (int v = 0; v < m; v++) {
                double s = Math.max(ready, free[v]), f = s + in.exec(t, v);
                if (f < fastestF) { fastestF = f; fastest = v; }
                if (f > in.deadline[t]) continue;
                int h = in.hostOf[v];
                double idle = off[h] >= on[h] ? Math.max(off[h], f) - Math.min(on[h], s) - (off[h] - on[h]) : f - s;
                double de = in.pIdle[h] * idle + in.joulesPerMi[h] * in.length[t];
                if (best < 0 || de < bestDe || (de == bestDe && f < bestF)) { best = v; bestDe = de; bestF = f; }
            }
            if (best < 0) best = fastest;
            double s = Math.max(ready, free[best]), f = s + in.exec(t, best);
            int h = in.hostOf[best];
            on[h] = Math.min(on[h], s); off[h] = Math.max(off[h], f);
            assign[t] = best;
            finish[t] = free[best] = f;
        }
        return assign;
    }
}
