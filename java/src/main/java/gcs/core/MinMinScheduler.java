package gcs.core;

/** Min-Min with ready times: repeatedly schedule the task with the smallest best completion time. */
public final class MinMinScheduler implements Scheduler {
    @Override public String name() { return "minmin"; }

    @Override public int[] schedule(Instance in) {
        int n = in.nTasks(), m = in.nVms();
        double[] free = new double[m];
        int[] assign = new int[n];
        boolean[] done = new boolean[n];
        for (int round = 0; round < n; round++) {
            int bestT = -1, bestV = -1;
            double bestCt = Double.POSITIVE_INFINITY;
            for (int t = 0; t < n; t++) {
                if (done[t]) continue;
                int v = 0;
                double ct = Math.max(in.arrival[t], free[0]) + in.exec(t, 0);
                for (int k = 1; k < m; k++) {
                    double c = Math.max(in.arrival[t], free[k]) + in.exec(t, k);
                    if (c < ct) { ct = c; v = k; }
                }
                if (ct < bestCt) { bestCt = ct; bestT = t; bestV = v; }   // strict < keeps the lowest task index
            }
            assign[bestT] = bestV;
            free[bestV] = bestCt;
            done[bestT] = true;
        }
        return assign;
    }
}
