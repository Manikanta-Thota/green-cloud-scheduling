package gcs.core;

/** A scheduling instance: tasks, VMs and hosts. Mirrors python/gcs/model.py (Instance). */
public final class Instance {
    public final double[] length, arrival, deadline;   // per task
    public final double[] mips, price;                 // per VM
    public final int[] hostOf;                         // per VM
    public final double[] pIdle, pMax;                 // per host

    public Instance(double[] length, double[] arrival, double[] deadline,
                    double[] mips, double[] price, int[] hostOf, double[] pIdle, double[] pMax) {
        this.length = length; this.arrival = arrival; this.deadline = deadline;
        this.mips = mips; this.price = price; this.hostOf = hostOf;
        this.pIdle = pIdle; this.pMax = pMax;
    }

    public int nTasks() { return length.length; }
    public int nVms() { return mips.length; }
    public double exec(int task, int vm) { return length[task] / mips[vm]; }
}
