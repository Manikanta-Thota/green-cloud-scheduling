package gcs.core;

import java.util.Arrays;
import java.util.Comparator;

/** A scheduling instance: tasks, VMs and hosts. Mirrors python/gcs/model.py (Instance). */
public final class Instance {
    public final double[] length, arrival, deadline;   // per task
    public final int[] rank;                           // per task: DAG depth (0 for independent tasks)
    public final int[][] preds;                        // per task: predecessor task indices (empty if none)
    public final double[] mips, price;                 // per VM
    public final int[] hostOf;                         // per VM
    public final double[] pIdle, pMax;                 // per host

    /** Execution order: (arrival, rank, index). Every VM queue runs its tasks in this order. */
    public final int[] order;
    /** Total MIPS of each host's VMs, and dynamic joules per MI on each host: (pMax - pIdle) / cap. */
    public final double[] hostCap, joulesPerMi;

    public Instance(double[] length, double[] arrival, double[] deadline, int[] rank, int[][] preds,
                    double[] mips, double[] price, int[] hostOf, double[] pIdle, double[] pMax) {
        this.length = length; this.arrival = arrival; this.deadline = deadline;
        this.rank = rank; this.preds = preds;
        this.mips = mips; this.price = price; this.hostOf = hostOf;
        this.pIdle = pIdle; this.pMax = pMax;

        Integer[] o = new Integer[length.length];
        for (int i = 0; i < o.length; i++) o[i] = i;
        Arrays.sort(o, Comparator.<Integer>comparingDouble(i -> arrival[i])
                .thenComparingInt(i -> rank[i]).thenComparingInt(i -> i));
        order = new int[o.length];
        for (int i = 0; i < o.length; i++) order[i] = o[i];

        hostCap = new double[pIdle.length];
        for (int v = 0; v < mips.length; v++) hostCap[hostOf[v]] += mips[v];
        joulesPerMi = new double[pIdle.length];
        for (int h = 0; h < pIdle.length; h++) joulesPerMi[h] = hostCap[h] > 0 ? (pMax[h] - pIdle[h]) / hostCap[h] : 0;
    }

    public int nTasks() { return length.length; }
    public int nVms() { return mips.length; }
    public int nHosts() { return pIdle.length; }
    public double exec(int task, int vm) { return length[task] / mips[vm]; }

    public boolean hasPrecedence() {
        for (int[] p : preds) if (p.length > 0) return true;
        return false;
    }
}
