package gcs.opt;

import java.util.List;
import java.util.SplittableRandom;

/** A search algorithm run on one {@link Problem} until its evaluation budget is spent. */
public interface Optimiser {
    String name();

    /** @return the final set of solutions (an archive, or a single best solution). */
    List<Solution> run(Problem p, SplittableRandom rng);

    // ---- variation operators shared by the population-based algorithms ----

    static int[] randomAssignment(int n, int m, SplittableRandom rng) {
        int[] x = new int[n];
        for (int i = 0; i < n; i++) x[i] = rng.nextInt(m);
        return x;
    }

    static int[] uniformCrossover(int[] a, int[] b, SplittableRandom rng) {
        int[] c = a.clone();
        for (int i = 0; i < c.length; i++) if (rng.nextBoolean()) c[i] = b[i];
        return c;
    }

    /** Reassign each gene to a random VM with probability {@code rate} (at least one gene changes). */
    static void mutate(int[] x, int m, double rate, SplittableRandom rng) {
        boolean changed = false;
        for (int i = 0; i < x.length; i++)
            if (rng.nextDouble() < rate) { x[i] = rng.nextInt(m); changed = true; }
        if (!changed) x[rng.nextInt(x.length)] = rng.nextInt(m);
    }
}
