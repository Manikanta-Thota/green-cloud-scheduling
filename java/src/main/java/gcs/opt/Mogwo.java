package gcs.opt;

import java.util.*;

/**
 * Multi-objective grey wolf optimiser (Mirjalili et al., 2016) adapted to task-to-VM assignment as in
 * Khan et al. (2024): the "recent multi-objective baseline". Each wolf holds a continuous position in
 * [0, m) per task, decoded to a VM by truncation; the initial pack is random, or (fairness variant)
 * the shared heuristic seeding. Alpha, beta and delta leaders are drawn from a grid
 * archive by roulette favouring sparsely populated hypercubes; when the archive overflows, a member of
 * a crowded hypercube is removed. The archive uses the same feasibility-first dominance as HPEAS so the
 * comparison isolates the search engine.
 */
public final class Mogwo implements Optimiser {
    final String name;
    final boolean seeded;
    final int packSize, archiveSize, nGrid;
    final double alphaInflate = 0.1, beta = 4.0, gamma = 2.0;

    public Mogwo(String name, boolean seeded, int packSize, int archiveSize, int nGrid) {
        this.name = name; this.seeded = seeded;
        this.packSize = packSize; this.archiveSize = archiveSize; this.nGrid = nGrid;
    }

    @Override public String name() { return name; }

    @Override public List<Solution> run(Problem p, SplittableRandom rng) {
        int n = p.in.nTasks(), m = p.in.nVms();
        double[][] pos = new double[packSize][n];
        List<Solution> archive = new ArrayList<>();
        Solution[] wolves = new Solution[packSize];
        List<Solution> init = Seeding.initial(p, packSize, seeded, rng);
        int alive = init.size();
        for (int w = 0; w < alive; w++) {
            wolves[w] = init.get(w);
            for (int j = 0; j < n; j++) pos[w][j] = wolves[w].x[j] + rng.nextDouble();   // a point inside the decoded VM's cell
            offer(archive, wolves[w], rng);
        }
        int maxIter = Math.max(1, (p.budget - p.evaluations()) / packSize);
        for (int it = 0; !p.exhausted(); it++) {
            double a = 2.0 - 2.0 * Math.min(it, maxIter) / maxIter;
            for (int w = 0; w < alive && !p.exhausted(); w++) {
                Solution[] lead = leaders(archive, wolves, alive, rng);
                double[] x = pos[w];
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (Solution l : lead) {
                        double lj = l.x[j] + 0.5;
                        double A = 2 * a * rng.nextDouble() - a, C = 2 * rng.nextDouble();
                        sum += lj - A * Math.abs(C * lj - x[j]);
                    }
                    x[j] = Math.min(m - 1e-9, Math.max(0, sum / 3));
                }
                wolves[w] = p.evaluate(decode(x, m));
                offer(archive, wolves[w], rng);
            }
        }
        return archive;
    }

    private static int[] decode(double[] x, int m) {
        int[] v = new int[x.length];
        for (int j = 0; j < x.length; j++) v[j] = Math.min(m - 1, (int) x[j]);
        return v;
    }

    private void offer(List<Solution> archive, Solution s, SplittableRandom rng) {
        for (Solution a : archive)
            if (Pareto.cdominates(a, s) || (Arrays.equals(a.f, s.f) && a.cv == s.cv)) return;
        archive.removeIf(a -> Pareto.cdominates(s, a));
        archive.add(s);
        if (archive.size() > archiveSize) {
            Map<Long, List<Solution>> cells = grid(archive);
            List<List<Solution>> occ = new ArrayList<>(cells.values());
            double[] wts = new double[occ.size()];
            for (int i = 0; i < wts.length; i++) wts[i] = Math.pow(occ.get(i).size(), gamma);
            List<Solution> cell = occ.get(roulette(wts, rng));
            archive.remove(cell.get(rng.nextInt(cell.size())));
        }
    }

    private Solution[] leaders(List<Solution> archive, Solution[] wolves, int alive, SplittableRandom rng) {
        List<Solution> pool = new ArrayList<>(archive);
        Solution[] lead = new Solution[3];
        Map<Long, List<Solution>> cells = grid(pool);
        List<List<Solution>> occ = new ArrayList<>(cells.values());
        for (int k = 0; k < 3; k++) {
            if (occ.isEmpty()) {   // archive smaller than three: fall back to the best remaining wolves
                List<Solution> rest = new ArrayList<>(Arrays.asList(wolves).subList(0, alive));
                Pareto.sort(rest);
                rest.sort(Comparator.comparingInt((Solution s) -> s.rank).thenComparing(s -> -s.crowd));
                lead[k] = rest.get(Math.min(k, rest.size() - 1));
                continue;
            }
            double[] wts = new double[occ.size()];
            for (int i = 0; i < wts.length; i++) wts[i] = Math.pow(occ.get(i).size(), -beta);
            int c = roulette(wts, rng);
            List<Solution> cell = occ.get(c);
            lead[k] = cell.remove(rng.nextInt(cell.size()));   // distinct leaders when possible
            if (cell.isEmpty()) occ.remove(c);
        }
        return lead;
    }

    /** Hypercube index of every archive member over the archive's inflated objective ranges. */
    private Map<Long, List<Solution>> grid(List<Solution> sols) {
        int d = sols.get(0).f.length;
        double[] lo = new double[d], hi = new double[d];
        Arrays.fill(lo, Double.POSITIVE_INFINITY);
        Arrays.fill(hi, Double.NEGATIVE_INFINITY);
        for (Solution s : sols) for (int k = 0; k < d; k++) { lo[k] = Math.min(lo[k], s.f[k]); hi[k] = Math.max(hi[k], s.f[k]); }
        Map<Long, List<Solution>> cells = new LinkedHashMap<>();
        for (Solution s : sols) {
            long key = 0;
            for (int k = 0; k < d; k++) {
                double span = (hi[k] - lo[k]) * (1 + 2 * alphaInflate);
                double base = lo[k] - alphaInflate * (hi[k] - lo[k]);
                int idx = span > 0 ? (int) Math.min(nGrid - 1, (s.f[k] - base) / span * nGrid) : 0;
                key = key * nGrid + idx;
            }
            cells.computeIfAbsent(key, z -> new ArrayList<>()).add(s);
        }
        return cells;
    }

    private static int roulette(double[] w, SplittableRandom rng) {
        double sum = 0;
        for (double x : w) sum += x;
        double r = rng.nextDouble() * sum;
        for (int i = 0; i < w.length; i++) { r -= w[i]; if (r <= 0) return i; }
        return w.length - 1;
    }
}
