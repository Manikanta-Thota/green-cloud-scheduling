package gcs.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Min-Min with ready times: repeatedly schedule, among the tasks whose predecessors are all scheduled,
 * the one with the smallest best completion time (lowest task index, then lowest VM index on ties).
 * Incremental: when VM v's free time grows only tasks whose best VM was v are re-scored, which gives
 * the same result as re-scoring everything because free times never decrease.
 */
public final class MinMinScheduler implements Scheduler {
    @Override public String name() { return "minmin"; }

    @Override public int[] schedule(Instance in) {
        int n = in.nTasks(), m = in.nVms();
        double[] free = new double[m], ct = new double[n], ready = new double[n], bestCt = new double[n];
        int[] assign = new int[n], bestV = new int[n], predsLeft = new int[n];
        boolean[] isReady = new boolean[n], done = new boolean[n];
        List<List<Integer>> succ = new ArrayList<>();
        for (int t = 0; t < n; t++) succ.add(new ArrayList<>());
        for (int t = 0; t < n; t++) {
            predsLeft[t] = in.preds[t].length;
            for (int p : in.preds[t]) succ.get(p).add(t);
        }
        for (int t = 0; t < n; t++) if (predsLeft[t] == 0) { ready[t] = in.arrival[t]; isReady[t] = true; score(in, t, ready, free, bestCt, bestV); }
        for (int round = 0; round < n; round++) {
            int bt = -1;
            for (int t = 0; t < n; t++)
                if (isReady[t] && !done[t] && (bt < 0 || bestCt[t] < bestCt[bt])) bt = t;   // strict < keeps the lowest index
            int v = bestV[bt];
            assign[bt] = v;
            ct[bt] = free[v] = bestCt[bt];
            done[bt] = true;
            for (int t = 0; t < n; t++)
                if (isReady[t] && !done[t] && bestV[t] == v) score(in, t, ready, free, bestCt, bestV);
            for (int s : succ.get(bt))
                if (--predsLeft[s] == 0) {
                    double r = in.arrival[s];
                    for (int p : in.preds[s]) r = Math.max(r, ct[p]);
                    ready[s] = r; isReady[s] = true;
                    score(in, s, ready, free, bestCt, bestV);
                }
        }
        return assign;
    }

    private static void score(Instance in, int t, double[] ready, double[] free, double[] bestCt, int[] bestV) {
        int v = 0;
        double c = Math.max(ready[t], free[0]) + in.exec(t, 0);
        for (int k = 1; k < free.length; k++) {
            double x = Math.max(ready[t], free[k]) + in.exec(t, k);
            if (x < c) { c = x; v = k; }
        }
        bestCt[t] = c; bestV[t] = v;
    }
}
