package gcs.core;

import java.util.Arrays;

/**
 * Objective evaluation, kept identical to python/gcs/model.py (see the conventions there).
 * One task at a time per VM, queued in (arrival, rank, index) order; a task also waits for its
 * predecessors. A host is on from the first start to the last finish of its tasks (hosts with no
 * tasks stay off) and draws linear power in the MIPS utilisation of its VMs. Because the power model
 * is linear, a host's energy is exactly pIdle * (on time) + joulesPerMi * (MI executed on it); the
 * Python oracle integrates the same quantity event by event, which cross-checks this closed form.
 */
public final class Evaluator {
    private Evaluator() {}

    public static final int ENERGY = 0, MAKESPAN = 1, COST = 2, IMBALANCE = 3, SLA = 4, N_OBJ = 5;

    /** Reusable buffers holding the simulation state of the last evaluated assignment. */
    public static final class Workspace {
        public final double[] start, finish;        // per task
        public final double[] free, busy;           // per VM: time it becomes free, busy seconds
        public final double[] hostOn, hostOff, hostWork;   // per host
        public Workspace(Instance in) {
            start = new double[in.nTasks()]; finish = new double[in.nTasks()];
            free = new double[in.nVms()]; busy = new double[in.nVms()];
            hostOn = new double[in.nHosts()]; hostOff = new double[in.nHosts()]; hostWork = new double[in.nHosts()];
        }
    }

    /** Evaluate into {@code out} (length N_OBJ) using {@code w} as scratch; w keeps the simulation state. */
    public static void evaluate(Instance in, int[] assign, Workspace w, double[] out) {
        int n = in.nTasks(), m = in.nVms(), nh = in.nHosts();
        Arrays.fill(w.free, 0.0);
        Arrays.fill(w.busy, 0.0);
        Arrays.fill(w.hostOn, Double.POSITIVE_INFINITY);
        Arrays.fill(w.hostOff, Double.NEGATIVE_INFINITY);
        Arrays.fill(w.hostWork, 0.0);
        double makespan = 0;
        int late = 0;
        for (int t : in.order) {
            int v = assign[t];
            double ready = in.arrival[t];
            for (int p : in.preds[t]) if (w.finish[p] > ready) ready = w.finish[p];
            double s = Math.max(ready, w.free[v]);
            double d = in.length[t] / in.mips[v];
            double f = s + d;
            w.start[t] = s; w.finish[t] = f;
            w.free[v] = f; w.busy[v] += d;
            int h = in.hostOf[v];
            if (s < w.hostOn[h]) w.hostOn[h] = s;
            if (f > w.hostOff[h]) w.hostOff[h] = f;
            w.hostWork[h] += in.length[t];
            if (f > makespan) makespan = f;
            if (f > in.deadline[t] + 1e-9) late++;
        }
        double energy = 0;
        for (int h = 0; h < nh; h++)
            if (w.hostWork[h] > 0) energy += in.pIdle[h] * (w.hostOff[h] - w.hostOn[h]) + in.joulesPerMi[h] * w.hostWork[h];
        double cost = 0, sumB = 0;
        for (int v = 0; v < m; v++) { cost += in.price[v] * w.busy[v]; sumB += w.busy[v]; }
        // imbalance = coefficient of variation of utilisation busy/makespan; the makespan cancels out
        double meanB = sumB / m, var = 0;
        for (int v = 0; v < m; v++) var += (w.busy[v] - meanB) * (w.busy[v] - meanB);
        out[ENERGY] = energy;
        out[MAKESPAN] = makespan;
        out[COST] = cost;
        out[IMBALANCE] = meanB > 0 ? Math.sqrt(var / m) / meanB : 0;
        out[SLA] = (double) late / n;
    }

    public static Objectives evaluate(Instance in, int[] assign) {
        int m = in.nVms();
        if (assign.length != in.nTasks()) throw new IllegalArgumentException("invalid assignment");
        for (int v : assign) if (v < 0 || v >= m) throw new IllegalArgumentException("invalid assignment");
        double[] o = new double[N_OBJ];
        evaluate(in, assign, new Workspace(in), o);
        return new Objectives(o[ENERGY], o[MAKESPAN], o[COST], o[IMBALANCE], o[SLA]);
    }
}
