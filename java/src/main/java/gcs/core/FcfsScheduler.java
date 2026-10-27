package gcs.core;

/**
 * Tasks in execution order (arrival, rank, index); each goes to the VM that can start it earliest
 * (lowest index on ties). A workflow task becomes ready when its predecessors have finished.
 */
public final class FcfsScheduler implements Scheduler {
    @Override public String name() { return "fcfs"; }

    @Override public int[] schedule(Instance in) {
        int n = in.nTasks(), m = in.nVms();
        double[] free = new double[m], finish = new double[n];
        int[] assign = new int[n];
        for (int t : in.order) {
            double ready = in.arrival[t];
            for (int p : in.preds[t]) ready = Math.max(ready, finish[p]);
            int best = 0;
            double bestStart = Math.max(ready, free[0]);
            for (int v = 1; v < m; v++) {
                double s = Math.max(ready, free[v]);
                if (s < bestStart) { bestStart = s; best = v; }
            }
            assign[t] = best;
            finish[t] = free[best] = bestStart + in.exec(t, best);
        }
        return assign;
    }
}
