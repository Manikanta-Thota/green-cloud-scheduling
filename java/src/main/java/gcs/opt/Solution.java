package gcs.opt;

/** An evaluated assignment: objective vector (all minimised) and constraint violation (0 = feasible). */
public final class Solution {
    public final int[] x;
    public final double[] f;
    public final double cv;
    int rank;
    double crowd;
    boolean explored;    // local search already started from this archive member

    Solution(int[] x, double[] f, double cv) { this.x = x; this.f = f; this.cv = cv; }

    public boolean feasible() { return cv <= 0; }
}
