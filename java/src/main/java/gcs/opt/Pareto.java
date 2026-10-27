package gcs.opt;

import java.util.*;

/** Feasibility-first (constrained) dominance, non-dominated sorting and crowding distance (Deb et al., 2002). */
public final class Pareto {
    private Pareto() {}

    public static boolean dominates(double[] a, double[] b) {
        boolean better = false;
        for (int k = 0; k < a.length; k++) {
            if (a[k] > b[k]) return false;
            if (a[k] < b[k]) better = true;
        }
        return better;
    }

    /** Feasible beats infeasible; two infeasible: smaller violation wins; two feasible: Pareto dominance. */
    public static boolean cdominates(Solution a, Solution b) {
        if (a.feasible() != b.feasible()) return a.feasible();
        if (!a.feasible()) return a.cv < b.cv;
        return dominates(a.f, b.f);
    }

    /** Sets {@code rank} (0 = first front) and {@code crowd} on every solution; returns the fronts. */
    public static List<List<Solution>> sort(List<Solution> pop) {
        int n = pop.size();
        int[] count = new int[n];
        List<List<Integer>> dominated = new ArrayList<>();
        for (int i = 0; i < n; i++) dominated.add(new ArrayList<>());
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                if (cdominates(pop.get(i), pop.get(j))) { dominated.get(i).add(j); count[j]++; }
                else if (cdominates(pop.get(j), pop.get(i))) { dominated.get(j).add(i); count[i]++; }
            }
        List<List<Solution>> fronts = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        for (int i = 0; i < n; i++) if (count[i] == 0) cur.add(i);
        int r = 0;
        while (!cur.isEmpty()) {
            List<Solution> front = new ArrayList<>();
            List<Integer> next = new ArrayList<>();
            for (int i : cur) {
                pop.get(i).rank = r;
                front.add(pop.get(i));
                for (int j : dominated.get(i)) if (--count[j] == 0) next.add(j);
            }
            crowding(front);
            fronts.add(front);
            cur = next;
            r++;
        }
        return fronts;
    }

    public static void crowding(List<Solution> front) {
        int n = front.size();
        for (Solution s : front) s.crowd = 0;
        if (n <= 2) { for (Solution s : front) s.crowd = Double.POSITIVE_INFINITY; return; }
        int d = front.get(0).f.length;
        Solution[] arr = front.toArray(new Solution[0]);
        for (int k = 0; k < d; k++) {
            final int kk = k;
            Arrays.sort(arr, Comparator.comparingDouble(s -> s.f[kk]));
            double span = arr[n - 1].f[k] - arr[0].f[k];
            arr[0].crowd = arr[n - 1].crowd = Double.POSITIVE_INFINITY;
            if (span <= 0) continue;
            for (int i = 1; i < n - 1; i++) arr[i].crowd += (arr[i + 1].f[k] - arr[i - 1].f[k]) / span;
        }
    }

    /** Crowded-comparison binary tournament (rank first, then larger crowding distance). */
    public static Solution tournament(List<Solution> pop, SplittableRandom rng) {
        Solution a = pop.get(rng.nextInt(pop.size())), b = pop.get(rng.nextInt(pop.size()));
        if (a.rank != b.rank) return a.rank < b.rank ? a : b;
        return a.crowd >= b.crowd ? a : b;
    }

    /** NSGA-II environmental selection of {@code size} survivors. */
    public static List<Solution> select(List<Solution> pool, int size) {
        List<Solution> out = new ArrayList<>(size);
        for (List<Solution> front : sort(pool)) {
            if (out.size() + front.size() <= size) { out.addAll(front); continue; }
            front.sort(Comparator.comparingDouble((Solution s) -> s.crowd).reversed());
            out.addAll(front.subList(0, size - out.size()));
            break;
        }
        return out;
    }
}
