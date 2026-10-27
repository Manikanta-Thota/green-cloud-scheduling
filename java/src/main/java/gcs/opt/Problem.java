package gcs.opt;

import gcs.core.*;

/**
 * One optimisation run on one instance: counts objective evaluations against a fixed budget and
 * applies the tolerance envelope of the protocol as a constraint. The reference schedule is the
 * deterministic baseline (FCFS or Min-Min) with the smaller makespan; computing both costs two
 * evaluations, charged to every metaheuristic. A solution is feasible when its makespan and cost are
 * within (1 + tolerance) of the reference and its SLA violation rate is at most reference + tolerance.
 */
public final class Problem {
    public final Instance in;
    public final int budget;
    public final double tolMakespan, tolCost, tolSla;
    public final Solution fcfs, minmin, ref;
    final Evaluator.Workspace ws;
    private final double[] buf = new double[Evaluator.N_OBJ];
    private int evals;

    // convergence trace: best feasible energy seen after each 5% of the budget
    public static final int TRACE_POINTS = 20;
    public final double[] trace = new double[TRACE_POINTS];
    private double bestFeasibleEnergy = Double.POSITIVE_INFINITY;
    private int nextTrace = 0;

    public Problem(Instance in, int budget, double tolMakespan, double tolCost, double tolSla) {
        this.in = in; this.budget = budget;
        this.tolMakespan = tolMakespan; this.tolCost = tolCost; this.tolSla = tolSla;
        ws = new Evaluator.Workspace(in);
        int[] xf = new FcfsScheduler().schedule(in), xm = new MinMinScheduler().schedule(in);
        double[] ff = new double[Evaluator.N_OBJ], fm = new double[Evaluator.N_OBJ];
        Evaluator.evaluate(in, xf, ws, ff);
        Evaluator.evaluate(in, xm, ws, fm);
        evals = 2;
        double[] r = ff[Evaluator.MAKESPAN] <= fm[Evaluator.MAKESPAN] ? ff : fm;
        ref = new Solution(null, r.clone(), 0);
        fcfs = new Solution(xf, ff, violation(ff));
        minmin = new Solution(xm, fm, violation(fm));
        record(fcfs);
        record(minmin);
    }

    public double violation(double[] f) {
        double[] r = ref.f;
        return Math.max(0, f[Evaluator.MAKESPAN] / r[Evaluator.MAKESPAN] - 1 - tolMakespan)
             + Math.max(0, f[Evaluator.COST] / r[Evaluator.COST] - 1 - tolCost)
             + Math.max(0, f[Evaluator.SLA] - r[Evaluator.SLA] - tolSla);
    }

    /** Evaluate a new candidate (one unit of budget). The workspace then holds its simulation state. */
    public Solution evaluate(int[] x) {
        Evaluator.evaluate(in, x, ws, buf);
        evals++;
        Solution s = new Solution(x, buf.clone(), violation(buf));
        record(s);
        return s;
    }

    /** Re-simulate an already evaluated solution to refresh the workspace (no budget charged). */
    void resimulate(Solution s) { Evaluator.evaluate(in, s.x, ws, buf); }

    private void record(Solution s) {
        if (s.feasible() && s.f[Evaluator.ENERGY] < bestFeasibleEnergy) bestFeasibleEnergy = s.f[Evaluator.ENERGY];
        while (nextTrace < TRACE_POINTS && evals >= (long) budget * (nextTrace + 1) / TRACE_POINTS)
            trace[nextTrace++] = bestFeasibleEnergy;
    }

    public boolean exhausted() { return evals >= budget; }
    public int evaluations() { return evals; }

    /** Fill any trace points not reached (algorithms that stop before the budget). */
    public void closeTrace() { while (nextTrace < TRACE_POINTS) trace[nextTrace++] = bestFeasibleEnergy; }
}
