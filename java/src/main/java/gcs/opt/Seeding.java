package gcs.opt;

import gcs.core.EnergyGreedyScheduler;
import java.util.*;

/** Initial populations shared by every population-based algorithm, so seeding is identical where it is used. */
public final class Seeding {
    private Seeding() {}

    /**
     * Seeded: the FCFS and Min-Min schedules (already evaluated by {@link Problem}), the energy-greedy
     * schedule, perturbed copies of these three (0.5-10% of genes reassigned) up to half the population,
     * then random assignments. Unseeded: random assignments only.
     */
    public static List<Solution> initial(Problem p, int popSize, boolean seeded, SplittableRandom rng) {
        int n = p.in.nTasks(), m = p.in.nVms();
        List<Solution> pop = new ArrayList<>();
        if (seeded) {
            List<Solution> seeds = new ArrayList<>(List.of(p.fcfs, p.minmin));
            if (!p.exhausted()) seeds.add(p.evaluate(new EnergyGreedyScheduler().schedule(p.in)));
            pop.addAll(seeds);
            for (int i = 0; pop.size() < popSize / 2 && !p.exhausted(); i++) {
                int[] x = seeds.get(i % seeds.size()).x.clone();
                Optimiser.mutate(x, m, 0.005 + 0.095 * rng.nextDouble(), rng);
                pop.add(p.evaluate(x));
            }
        }
        while (pop.size() < popSize && !p.exhausted())
            pop.add(p.evaluate(Optimiser.randomAssignment(n, m, rng)));
        return pop;
    }
}
