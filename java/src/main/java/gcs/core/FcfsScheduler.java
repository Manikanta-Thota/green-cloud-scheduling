package gcs.core;

import java.util.Arrays;
import java.util.Comparator;

/** Tasks in arrival order; each goes to the VM that can start it earliest (lowest index on ties). */
public final class FcfsScheduler implements Scheduler {
    @Override public String name() { return "fcfs"; }

    @Override public int[] schedule(Instance in) {
        int n = in.nTasks(), m = in.nVms();
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, Comparator.<Integer>comparingDouble(i -> in.arrival[i]).thenComparingInt(i -> i));
        double[] free = new double[m];
        int[] assign = new int[n];
        for (int t : order) {
            int best = 0;
            double bestStart = Math.max(in.arrival[t], free[0]);
            for (int v = 1; v < m; v++) {
                double s = Math.max(in.arrival[t], free[v]);
                if (s < bestStart) { bestStart = s; best = v; }
            }
            assign[t] = best;
            free[best] = bestStart + in.exec(t, best);
        }
        return assign;
    }
}
