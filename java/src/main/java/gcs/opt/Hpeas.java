package gcs.opt;

import gcs.core.*;
import java.util.*;

/**
 * Hybrid Pareto Energy-Aware Scheduler (proposal Section 3.2). Four components:
 * <ol>
 *   <li>Heuristic seeding: FCFS, Min-Min and energy-greedy schedules plus perturbed copies of them;
 *       the rest of the population is random.</li>
 *   <li>Global search: NSGA-II generations (crowded tournament, uniform crossover, uniform mutation,
 *       feasibility-first non-dominated sorting with crowding distance).</li>
 *   <li>Adaptive local search: when the best feasible energy has not improved for {@code stagnation}
 *       generations, problem-specific moves are applied to the archive's lowest-energy member and to
 *       other archive members not yet explored. Operators are chosen by probability matching on
 *       their recent success (entry into the archive).</li>
 *   <li>A bounded external archive (feasibility-first dominance, crowding truncation).</li>
 * </ol>
 * Ablation switches turn off seeding, local search, or the adaptive parts (static schedule and uniform
 * operator choice). All variants stop at the same number of objective evaluations.
 */
public final class Hpeas implements Optimiser {
    public enum Ls { NONE, ADAPTIVE, STATIC }

    final String name;
    final boolean seeding;
    final Ls ls;
    final int popSize, archiveSize, stagnation, lsMembers, lsSteps, staticPeriod;
    final double pc, pm;
    static final double P_MIN = 0.05, ADAPT_RATE = 0.3, MIN_GAIN = 1e-4;

    public Hpeas(String name, boolean seeding, Ls ls, int popSize, int archiveSize, double pc, double pm,
                 int stagnation, int lsMembers, int lsSteps, int staticPeriod) {
        this.name = name; this.seeding = seeding; this.ls = ls; this.popSize = popSize; this.archiveSize = archiveSize;
        this.pc = pc; this.pm = pm; this.stagnation = stagnation; this.lsMembers = lsMembers; this.lsSteps = lsSteps;
        this.staticPeriod = staticPeriod;
    }

    @Override public String name() { return name; }

    // ---------------------------------------------------------------- main loop

    @Override public List<Solution> run(Problem p, SplittableRandom rng) {
        int n = p.in.nTasks(), m = p.in.nVms();
        double rate = pm / n;
        Archive archive = new Archive(archiveSize);
        List<Solution> pop = Seeding.initial(p, popSize, seeding, rng);
        for (Solution s : pop) archive.add(s);
        Pareto.sort(pop);

        LocalSearch local = new LocalSearch(p, rng);
        List<Solution> injected = new ArrayList<>();
        double best = progress(archive);
        int gen = 0, lastGain = 0;
        while (!p.exhausted()) {
            List<Solution> pool = new ArrayList<>(pop);
            for (int k = 0; k < popSize && !p.exhausted(); k++) {
                Solution a = Pareto.tournament(pop, rng), b = Pareto.tournament(pop, rng);
                int[] child = rng.nextDouble() < pc ? Optimiser.uniformCrossover(a.x, b.x, rng) : a.x.clone();
                Optimiser.mutate(child, m, rate, rng);
                Solution s = p.evaluate(child);
                archive.add(s);
                pool.add(s);
            }
            pool.addAll(injected);
            injected.clear();
            pop = Pareto.select(pool, popSize);
            gen++;
            double now = progress(archive);
            if (now < best * (1 - MIN_GAIN)) { best = now; lastGain = gen; }
            boolean trigger = switch (ls) {
                case NONE -> false;
                case ADAPTIVE -> gen - lastGain >= stagnation;
                case STATIC -> gen % staticPeriod == 0;
            };
            if (trigger && !p.exhausted()) {
                local.run(archive, injected, ls == Ls.ADAPTIVE);
                double after = progress(archive);
                if (after < best) best = after;
                lastGain = gen;    // restart the stagnation counter after a local-search phase
            }
        }
        return new ArrayList<>(archive.members());
    }

    /** Search progress: best feasible energy, or (no feasible member yet) 1e300 times the smallest violation. */
    static double progress(Archive a) {
        double e = Double.POSITIVE_INFINITY, cv = Double.POSITIVE_INFINITY;
        for (Solution s : a.members()) {
            if (s.feasible()) e = Math.min(e, s.f[Evaluator.ENERGY]);
            else cv = Math.min(cv, s.cv);
        }
        return e < Double.POSITIVE_INFINITY ? e : 1e300 * Math.min(1, cv);
    }

    // ---------------------------------------------------------------- local search

    /** Problem-specific neighbourhood moves with probability-matching operator selection. */
    final class LocalSearch {
        static final int MAKESPAN = 0, CONSOLIDATE = 1, EFFICIENCY = 2, SLA_FIX = 3, COST = 4, SWAP = 5, N_OPS = 6;
        static final int[] TARGET = {Evaluator.MAKESPAN, Evaluator.ENERGY, Evaluator.ENERGY, Evaluator.SLA,
                                     Evaluator.COST, Evaluator.IMBALANCE};
        final Problem p;
        final Instance in;
        final SplittableRandom rng;
        final double[] quality = new double[N_OPS];
        // state of the current local-search point
        final double[] finish, free, busy, hostWork;
        final int[][] onVm;
        final int[] onVmCount;

        LocalSearch(Problem p, SplittableRandom rng) {
            this.p = p; this.in = p.in; this.rng = rng;
            Arrays.fill(quality, 1.0);
            finish = new double[in.nTasks()]; free = new double[in.nVms()]; busy = new double[in.nVms()];
            hostWork = new double[in.nHosts()];
            onVm = new int[in.nVms()][];
            onVmCount = new int[in.nVms()];
        }

        void run(Archive archive, List<Solution> injected, boolean adaptive) {
            List<Solution> starts = new ArrayList<>();
            Solution low = Selection.minEnergy(archive.members());
            if (low != null) starts.add(low);
            List<Solution> fresh = new ArrayList<>();
            for (Solution s : archive.members()) if (!s.explored && s != low) fresh.add(s);
            while (starts.size() < lsMembers && !fresh.isEmpty()) starts.add(fresh.remove(rng.nextInt(fresh.size())));

            for (Solution start : starts) {
                start.explored = true;
                Solution cur = start;
                p.resimulate(cur);
                load(cur);
                for (int step = 0; step < lsSteps && !p.exhausted(); step++) {
                    int op = adaptive ? choose() : rng.nextInt(N_OPS);
                    int[] x = move(op, cur.x);
                    if (x == null) continue;
                    Solution cand = p.evaluate(x);
                    boolean entered = archive.add(cand);
                    boolean better = Pareto.cdominates(cand, cur);
                    double reward = entered ? 1.0 : better ? 0.5 : 0.0;
                    quality[op] += ADAPT_RATE * (reward - quality[op]);
                    if (entered) injected.add(cand);
                    if (better || entered && cand.f[TARGET[op]] < cur.f[TARGET[op]]) {
                        cur = cand;
                        load(cur);   // workspace still holds cand's simulation state
                    }
                }
            }
        }

        int choose() {
            double sum = 0;
            for (double q : quality) sum += q;
            double r = rng.nextDouble();
            for (int k = 0; k < N_OPS; k++) {
                double prob = P_MIN + (1 - N_OPS * P_MIN) * (sum > 0 ? quality[k] / sum : 1.0 / N_OPS);
                r -= prob;
                if (r <= 0) return k;
            }
            return N_OPS - 1;
        }

        /** Copy the workspace (holding {@code s}'s simulation) and index the tasks of every VM. */
        void load(Solution s) {
            Evaluator.Workspace w = p.ws;
            System.arraycopy(w.finish, 0, finish, 0, finish.length);
            System.arraycopy(w.free, 0, free, 0, free.length);
            System.arraycopy(w.busy, 0, busy, 0, busy.length);
            System.arraycopy(w.hostWork, 0, hostWork, 0, hostWork.length);
            Arrays.fill(onVmCount, 0);
            for (int v : s.x) onVmCount[v]++;
            for (int v = 0; v < onVm.length; v++) onVm[v] = new int[onVmCount[v]];
            Arrays.fill(onVmCount, 0);
            for (int t = 0; t < s.x.length; t++) { int v = s.x[t]; onVm[v][onVmCount[v]++] = t; }
        }

        int randomTaskOn(int v) { return onVmCount[v] == 0 ? -1 : onVm[v][rng.nextInt(onVmCount[v])]; }

        /** VM (other than {@code except}) with the earliest estimated completion of task t, restricted by {@code allowed}. */
        int earliestVm(int t, int except, java.util.function.IntPredicate allowed) {
            int best = -1;
            double bestC = Double.POSITIVE_INFINITY;
            for (int v = 0; v < free.length; v++) {
                if (v == except || !allowed.test(v)) continue;
                double c = free[v] + in.exec(t, v);
                if (c < bestC) { bestC = c; best = v; }
            }
            return best;
        }

        int[] move(int op, int[] x) {
            int m = in.nVms();
            switch (op) {
                case MAKESPAN: {   // move a task off the VM that finishes last
                    int vmax = argmax(free);
                    int t = randomTaskOn(vmax);
                    if (t < 0) return null;
                    return reassign(x, t, earliestVm(t, vmax, v -> true));
                }
                case CONSOLIDATE: {   // empty the least-loaded active host so it can be switched off
                    int hmin = -1, active = 0;
                    for (int h = 0; h < hostWork.length; h++)
                        if (hostWork[h] > 0) { active++; if (hmin < 0 || hostWork[h] < hostWork[hmin]) hmin = h; }
                    if (active < 2) return null;
                    final int hh = hmin;
                    int t = randomTaskOnHost(hh);
                    if (t < 0) return null;
                    return reassign(x, t, earliestVm(t, -1, v -> in.hostOf[v] != hh && hostWork[in.hostOf[v]] > 0));
                }
                case EFFICIENCY: {   // shift work from the least to a more energy-efficient active host
                    int hworst = -1;
                    for (int h = 0; h < hostWork.length; h++)
                        if (hostWork[h] > 0 && (hworst < 0 || in.joulesPerMi[h] > in.joulesPerMi[hworst])) hworst = h;
                    if (hworst < 0) return null;
                    final int hw = hworst;
                    int t = randomTaskOnHost(hw);
                    if (t < 0) return null;
                    return reassign(x, t, earliestVm(t, -1, v -> in.joulesPerMi[in.hostOf[v]] < in.joulesPerMi[hw]
                                                              && hostWork[in.hostOf[v]] > 0));
                }
                case SLA_FIX: {   // move a late task to the VM that should finish it earliest
                    int late = 0, pick = -1;
                    for (int t = 0; t < finish.length; t++)
                        if (finish[t] > in.deadline[t] + 1e-9 && rng.nextInt(++late) == 0) pick = t;   // reservoir sample
                    if (pick < 0) return null;
                    return reassign(x, pick, earliestVm(pick, x[pick], v -> true));
                }
                case COST: {   // move a task to a cheaper-per-MI VM that does not extend the makespan
                    int t = rng.nextInt(x.length);
                    double makespan = free[argmax(free)];
                    int cur = x[t], best = -1;
                    double bestPrice = in.price[cur] / in.mips[cur];
                    for (int v = 0; v < m; v++) {
                        double unit = in.price[v] / in.mips[v];
                        if (unit < bestPrice && free[v] + in.exec(t, v) <= makespan) { bestPrice = unit; best = v; }
                    }
                    return reassign(x, t, best);
                }
                default: {   // SWAP: exchange a long task on the busiest VM with a shorter one on the idlest
                    int vmax = argmax(busy), vmin = 0;
                    for (int v = 1; v < m; v++) if (busy[v] < busy[vmin]) vmin = v;
                    int a = randomTaskOn(vmax);
                    if (a < 0 || vmax == vmin) return null;
                    int b = -1, seen = 0;
                    for (int i = 0; i < onVmCount[vmin]; i++) {
                        int t = onVm[vmin][i];
                        if (in.length[t] < in.length[a] && rng.nextInt(++seen) == 0) b = t;
                    }
                    int[] y = x.clone();
                    y[a] = vmin;
                    if (b >= 0) y[b] = vmax;
                    return y;
                }
            }
        }

        int randomTaskOnHost(int h) {
            int total = 0;
            for (int v = 0; v < free.length; v++) if (in.hostOf[v] == h) total += onVmCount[v];
            if (total == 0) return -1;
            int k = rng.nextInt(total);
            for (int v = 0; v < free.length; v++)
                if (in.hostOf[v] == h) { if (k < onVmCount[v]) return onVm[v][k]; k -= onVmCount[v]; }
            return -1;
        }

        int[] reassign(int[] x, int t, int v) {
            if (v < 0 || v == x[t]) return null;
            int[] y = x.clone();
            y[t] = v;
            return y;
        }
    }

    static int argmax(double[] a) {
        int k = 0;
        for (int i = 1; i < a.length; i++) if (a[i] > a[k]) k = i;
        return k;
    }
}
