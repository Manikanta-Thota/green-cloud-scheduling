package gcs.core;

import java.util.Arrays;
import java.util.Comparator;

/**
 * Objective evaluation, kept identical to python/gcs/model.py (see the conventions there).
 * One task at a time per VM, queued in (arrival, index) order; hosts on during [0, makespan];
 * linear host power in the MIPS utilisation of the host's VMs.
 */
public final class Evaluator {
    private Evaluator() {}

    public static final class Sim {
        public final double[] start, finish, busy;
        Sim(int nTasks, int nVms) { start = new double[nTasks]; finish = new double[nTasks]; busy = new double[nVms]; }
    }

    public static Sim simulate(Instance in, int[] assign) {
        int n = in.nTasks(), m = in.nVms();
        if (assign.length != n) throw new IllegalArgumentException("invalid assignment");
        for (int v : assign) if (v < 0 || v >= m) throw new IllegalArgumentException("invalid assignment");
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, Comparator.<Integer>comparingDouble(i -> in.arrival[i]).thenComparingInt(i -> i));
        double[] free = new double[m];
        Sim sim = new Sim(n, m);
        for (int t : order) {
            int v = assign[t];
            double s = Math.max(in.arrival[t], free[v]);
            double d = in.exec(t, v);
            sim.start[t] = s; sim.finish[t] = s + d;
            free[v] = s + d; sim.busy[v] += d;
        }
        return sim;
    }

    public static Objectives evaluate(Instance in, int[] assign) {
        Sim sim = simulate(in, assign);
        int n = in.nTasks(), m = in.nVms();
        double makespan = 0;
        for (double f : sim.finish) makespan = Math.max(makespan, f);

        double sumU = 0;
        double[] util = new double[m];
        for (int v = 0; v < m; v++) { util[v] = makespan > 0 ? sim.busy[v] / makespan : 0; sumU += util[v]; }
        double meanU = sumU / m, var = 0;
        for (double u : util) var += (u - meanU) * (u - meanU);
        double imbalance = meanU > 0 ? Math.sqrt(var / m) / meanU : 0;

        double cost = 0;
        for (int v = 0; v < m; v++) cost += in.price[v] * sim.busy[v];

        int late = 0;
        for (int t = 0; t < n; t++) if (sim.finish[t] > in.deadline[t] + 1e-9) late++;

        return new Objectives(energy(in, assign, sim, makespan), makespan, cost, imbalance, (double) late / n);
    }

    private static double energy(Instance in, int[] assign, Sim sim, double makespan) {
        int nHosts = in.pIdle.length;
        double[] cap = new double[nHosts];
        for (int v = 0; v < in.nVms(); v++) cap[in.hostOf[v]] += in.mips[v];
        // events per host: {time, +/- mips}
        java.util.List<java.util.List<double[]>> events = new java.util.ArrayList<>();
        for (int h = 0; h < nHosts; h++) events.add(new java.util.ArrayList<>());
        for (int t = 0; t < in.nTasks(); t++) {
            int v = assign[t], h = in.hostOf[v];
            events.get(h).add(new double[] {sim.start[t], +in.mips[v]});
            events.get(h).add(new double[] {sim.finish[t], -in.mips[v]});
        }
        double total = 0;
        for (int h = 0; h < nHosts; h++) {
            events.get(h).sort(Comparator.<double[]>comparingDouble(e -> e[0]).thenComparingDouble(e -> e[1]));
            double busyMips = 0, prev = 0, span = in.pMax[h] - in.pIdle[h];
            for (double[] e : events.get(h)) {
                if (e[0] > prev) {
                    double u = cap[h] > 0 ? busyMips / cap[h] : 0;
                    total += (in.pIdle[h] + span * u) * (e[0] - prev);
                    prev = e[0];
                }
                busyMips += e[1];
            }
            if (makespan > prev) total += in.pIdle[h] * (makespan - prev);
        }
        return total;
    }
}
