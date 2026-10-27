package gcs.opt;

import gcs.core.Evaluator;
import java.util.*;

/**
 * Single-objective genetic algorithm (the "single global metaheuristic" baseline). Objectives are
 * scalarised with equal weights after dividing by the reference schedule's values (with small floors
 * for imbalance and SLA, which can be zero), plus a penalty on constraint violation. Random initial
 * population (or the shared heuristic seeding, for the fairness variant), binary tournament, uniform
 * crossover, uniform mutation and elitism.
 */
public final class Ga implements Optimiser {
    final String name;
    final boolean seeded;
    final int popSize, elites;
    final double pc, pm, penalty;

    public Ga(String name, boolean seeded, int popSize, int elites, double pc, double pm, double penalty) {
        this.name = name; this.seeded = seeded;
        this.popSize = popSize; this.elites = elites; this.pc = pc; this.pm = pm; this.penalty = penalty;
    }

    @Override public String name() { return name; }

    static double fitness(Problem p, Solution s, double penalty) {
        double[] r = p.ref.f, f = s.f;
        return 0.2 * (f[Evaluator.ENERGY] / r[Evaluator.ENERGY] + f[Evaluator.MAKESPAN] / r[Evaluator.MAKESPAN]
                + f[Evaluator.COST] / r[Evaluator.COST] + f[Evaluator.IMBALANCE] / Math.max(r[Evaluator.IMBALANCE], 0.05)
                + f[Evaluator.SLA] / Math.max(r[Evaluator.SLA], 0.01)) + penalty * s.cv;
    }

    @Override public List<Solution> run(Problem p, SplittableRandom rng) {
        int n = p.in.nTasks(), m = p.in.nVms();
        double rate = pm / n;
        List<Solution> pop = Seeding.initial(p, popSize, seeded, rng);
        Map<Solution, Double> fit = new HashMap<>();
        for (Solution s : pop) fit.put(s, fitness(p, s, penalty));
        Comparator<Solution> byFit = Comparator.comparingDouble(fit::get);
        while (!p.exhausted()) {
            pop.sort(byFit);
            List<Solution> next = new ArrayList<>(pop.subList(0, Math.min(elites, pop.size())));
            while (next.size() < popSize && !p.exhausted()) {
                Solution a = tournament(pop, fit, rng), b = tournament(pop, fit, rng);
                int[] child = rng.nextDouble() < pc ? Optimiser.uniformCrossover(a.x, b.x, rng) : a.x.clone();
                Optimiser.mutate(child, m, rate, rng);
                Solution s = p.evaluate(child);
                fit.put(s, fitness(p, s, penalty));
                next.add(s);
            }
            fit.keySet().retainAll(new HashSet<>(next));
            pop = next;
        }
        return List.of(Collections.min(pop, byFit));
    }

    private static Solution tournament(List<Solution> pop, Map<Solution, Double> fit, SplittableRandom rng) {
        Solution a = pop.get(rng.nextInt(pop.size())), b = pop.get(rng.nextInt(pop.size()));
        return fit.get(a) <= fit.get(b) ? a : b;
    }
}
