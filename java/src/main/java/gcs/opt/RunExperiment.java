package gcs.opt;

import gcs.core.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import java.util.concurrent.*;
import java.util.zip.GZIPOutputStream;

/**
 * Runs a set of algorithms on every case file in a directory and writes one CSV row per
 * (instance, algorithm, selection rule), plus convergence traces and, optionally, the selected
 * schedules. Called by python/gcs/experiment.py. With --order-seed, instances and algorithms run in a
 * seeded random order; the output keeps file and algorithm order either way.
 *
 * <pre>
 * java -cp java/out gcs.opt.RunExperiment --in DIR --out runs.csv --trace trace.csv
 *      --algos fcfs,minmin,ga,mogwo,hpeas --budget 10000 --threads 8 [--param key=value ...] [--fronts DIR]
 *      [--schedules schedules.csv.gz --config-hash HEX] [--order-seed N]
 * </pre>
 */
public final class RunExperiment {
    static final String[] OBJ = {"energy", "makespan", "cost", "imbalance", "sla"};

    public static void main(String[] args) throws Exception {
        Map<String, String> a = new HashMap<>();
        Map<String, String> prm = new HashMap<>();
        for (int i = 0; i < args.length; i += 2) {
            if (args[i].equals("--param")) { String[] kv = args[i + 1].split("=", 2); prm.put(kv[0], kv[1]); }
            else a.put(args[i].substring(2), args[i + 1]);
        }
        List<Path> files = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(Path.of(a.get("in")), "*.txt")) { ds.forEach(files::add); }
        Collections.sort(files);
        List<String> algos = List.of(a.getOrDefault("algos", "fcfs,minmin,ga,mogwo,hpeas").split(","));
        int budget = Integer.parseInt(a.getOrDefault("budget", "10000"));
        int threads = Integer.parseInt(a.getOrDefault("threads", String.valueOf(Runtime.getRuntime().availableProcessors())));
        Path frontsDir = a.containsKey("fronts") ? Path.of(a.get("fronts")) : null;
        if (frontsDir != null) Files.createDirectories(frontsDir);
        Params p = new Params(prm);
        Long orderSeed = a.containsKey("order-seed") ? Long.parseLong(a.get("order-seed")) : null;
        String configHash = a.getOrDefault("config-hash", "");

        List<String> metaKeys = new ArrayList<>(CaseFile.read(files.get(0)).meta.keySet());
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(Path.of(a.get("out"))));
             PrintWriter trace = new PrintWriter(Files.newBufferedWriter(Path.of(a.get("trace"))));
             PrintWriter sched = a.containsKey("schedules") ? new PrintWriter(new OutputStreamWriter(
                     new GZIPOutputStream(Files.newOutputStream(Path.of(a.get("schedules")))), StandardCharsets.UTF_8)) : null) {
            out.println(String.join(",", metaKeys) + ",algorithm,selection," + String.join(",", OBJ)
                    + ",cv,feasible,evaluations,runtime_s,front_size,hv,igd,spacing");
            StringBuilder th = new StringBuilder(String.join(",", metaKeys) + ",algorithm");
            for (int k = 1; k <= Problem.TRACE_POINTS; k++) th.append(",t").append(k);
            trace.println(th);
            if (sched != null) sched.println("scenario_id,rep,algorithm,selection,seed,config_hash,schedule_sha256,assignment");

            List<Path> submitOrder = new ArrayList<>(files);
            if (orderSeed != null) Collections.shuffle(submitOrder, new Random(orderSeed));
            ExecutorService pool = Executors.newFixedThreadPool(threads);
            Map<Path, Future<String[]>> jobs = new HashMap<>();
            for (Path f : submitOrder)
                jobs.put(f, pool.submit(() -> runCase(f, metaKeys, algos, budget, p, frontsDir, orderSeed, configHash)));
            for (Path f : files) {
                String[] res = jobs.get(f).get();
                out.print(res[0]);
                trace.print(res[1]);
                out.flush();
                trace.flush();
                if (sched != null) sched.print(res[2]);
            }
            pool.shutdown();
        }
    }

    static String[] runCase(Path file, List<String> metaKeys, List<String> algos, int budget, Params p,
                            Path frontsDir, Long orderSeed, String configHash) throws IOException {
        CaseFile c = CaseFile.read(file);
        StringBuilder metaCols = new StringBuilder();
        for (String k : metaKeys) metaCols.append(c.meta.getOrDefault(k, "")).append(',');
        Map<String, List<Solution>> results = new HashMap<>();
        Map<String, double[]> infos = new HashMap<>();
        Map<String, double[]> traceOf = new HashMap<>();
        Map<String, Long> seedOf = new HashMap<>();
        List<String> execOrder = new ArrayList<>(algos);
        if (orderSeed != null) Collections.shuffle(execOrder, new Random(orderSeed ^ file.getFileName().toString().hashCode()));
        for (String algo : execOrder) {
            long seed = c.seeds.getOrDefault(algo, (long) (file.getFileName().toString() + algo).hashCode());
            SplittableRandom rng = new SplittableRandom(seed);
            long t0 = System.nanoTime();
            Problem prob = new Problem(c.instance, budget, p.d("tol_makespan", 0.05), p.d("tol_cost", 0.05), p.d("tol_sla", 0.01));
            List<Solution> result = switch (algo) {
                case "fcfs" -> List.of(prob.fcfs);
                case "minmin" -> List.of(prob.minmin);
                case "energy_greedy" -> List.of(prob.evaluate(new EnergyGreedyScheduler().schedule(c.instance)));
                default -> make(algo, p).run(prob, rng);
            };
            double secs = (System.nanoTime() - t0) / 1e9;
            prob.closeTrace();
            results.put(algo, result);
            seedOf.put(algo, seed);
            boolean heuristic = algo.equals("fcfs") || algo.equals("minmin") || algo.equals("energy_greedy");
            infos.put(algo, new double[] {heuristic ? (algo.equals("energy_greedy") ? 1 : 0) : prob.evaluations(), secs});
            if (!heuristic) traceOf.put(algo, prob.trace);
        }
        // from here on, configured algorithm order (the reference front and output rows depend on it)
        Map<String, List<Solution>> fronts = new LinkedHashMap<>();
        Map<String, double[]> runInfo = new LinkedHashMap<>();
        StringBuilder traces = new StringBuilder();
        for (String algo : algos) {
            fronts.put(algo, results.get(algo));
            runInfo.put(algo, infos.get(algo));
            if (traceOf.containsKey(algo)) {
                traces.append(metaCols).append(algo);
                for (double v : traceOf.get(algo)) traces.append(',').append(v);
                traces.append('\n');
            }
        }
        StringBuilder schedules = new StringBuilder();

        // reference front: non-dominated union of every algorithm's points; normalise by its ideal/nadir
        List<double[]> all = new ArrayList<>();
        for (List<Solution> s : fronts.values()) for (Solution x : s) all.add(x.f);
        List<double[]> refFront = Indicators.nonDominated(all, OBJ.length);
        double[] lo = new double[OBJ.length], hi = new double[OBJ.length];
        Arrays.fill(lo, Double.POSITIVE_INFINITY);
        Arrays.fill(hi, Double.NEGATIVE_INFINITY);
        for (double[] x : refFront) for (int k = 0; k < lo.length; k++) { lo[k] = Math.min(lo[k], x[k]); hi[k] = Math.max(hi[k], x[k]); }
        List<double[]> refNorm = normalise(refFront, lo, hi);
        double[] refPoint = new double[OBJ.length];
        Arrays.fill(refPoint, 1.1);

        StringBuilder rows = new StringBuilder();
        for (String algo : algos) {
            List<Solution> s = fronts.get(algo);
            List<double[]> pts = new ArrayList<>();
            for (Solution x : s) pts.add(x.f);
            List<double[]> norm = normalise(pts, lo, hi);
            double hv = Indicators.hypervolume(norm, refPoint), igd = Indicators.igd(norm, refNorm), sp = Indicators.spacing(norm);
            double[] info = runInfo.get(algo);
            Map<String, Solution> picks = new LinkedHashMap<>();
            if (s.size() == 1 && (algo.equals("fcfs") || algo.equals("minmin") || algo.equals("energy_greedy") || algo.startsWith("ga")))
                picks.put("single", s.get(0));
            else { picks.put("tcme", Selection.minEnergy(s)); picks.put("knee", Selection.knee(s)); }
            for (var e : picks.entrySet()) {
                Solution x = e.getValue();
                rows.append(metaCols).append(algo).append(',').append(e.getKey());
                for (double v : x.f) rows.append(',').append(v);
                rows.append(',').append(x.cv).append(',').append(x.feasible() ? 1 : 0)
                    .append(',').append((long) info[0]).append(',').append(info[1])
                    .append(',').append(s.size()).append(',').append(hv).append(',').append(igd).append(',').append(sp).append('\n');
                String assignment = assignment(x.x);
                schedules.append(c.meta.getOrDefault("scenario_id", "")).append(',').append(c.meta.getOrDefault("rep", ""))
                    .append(',').append(algo).append(',').append(e.getKey()).append(',').append(seedOf.get(algo))
                    .append(',').append(configHash).append(',').append(sha256(assignment)).append(',').append(assignment).append('\n');
            }
            if (frontsDir != null) {
                try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(
                        frontsDir.resolve(file.getFileName().toString().replace(".txt", "") + "_" + algo + ".csv")))) {
                    w.println(String.join(",", OBJ) + ",cv");
                    for (Solution x : s) {
                        StringBuilder b = new StringBuilder();
                        for (double v : x.f) b.append(v).append(',');
                        w.println(b.append(x.cv));
                    }
                }
            }
        }
        System.err.println("done " + file.getFileName());
        return new String[] {rows.toString(), traces.toString(), schedules.toString()};
    }

    /** VM index of each task, space-separated (task order of the case file). */
    static String assignment(int[] x) {
        StringBuilder b = new StringBuilder(x.length * 3);
        for (int i = 0; i < x.length; i++) { if (i > 0) b.append(' '); b.append(x[i]); }
        return b.toString();
    }

    static String sha256(String s) {
        try {
            byte[] h = MessageDigest.getInstance("SHA-256").digest(s.getBytes(StandardCharsets.US_ASCII));
            return HexFormat.of().formatHex(h);
        } catch (java.security.NoSuchAlgorithmException e) { throw new IllegalStateException(e); }
    }

    static List<double[]> normalise(List<double[]> pts, double[] lo, double[] hi) {
        List<double[]> out = new ArrayList<>();
        for (double[] x : pts) {
            double[] z = new double[x.length];
            for (int k = 0; k < x.length; k++) z[k] = hi[k] > lo[k] ? (x[k] - lo[k]) / (hi[k] - lo[k]) : 0;
            out.add(z);
        }
        return out;
    }

    static Optimiser make(String algo, Params p) {
        int pop = p.i("pop", 40), arch = p.i("archive", 100);
        double pc = p.d("pc", 0.9), pm = p.d("pm", 5);
        int stag = p.i("stagnation", 3), members = p.i("ls_members", 3), steps = p.i("ls_steps", 20), period = p.i("static_period", 5);
        return switch (algo) {
            case "ga" -> new Ga(algo, false, pop, 2, p.d("ga_pc", pc), p.d("ga_pm", pm), 10.0);
            case "ga_seeded" -> new Ga(algo, true, pop, 2, p.d("ga_pc", pc), p.d("ga_pm", pm), 10.0);
            case "mogwo" -> new Mogwo(algo, false, pop, arch, 10);
            case "mogwo_seeded" -> new Mogwo(algo, true, pop, arch, 10);
            case "hpeas" -> new Hpeas(algo, true, Hpeas.Ls.ADAPTIVE, pop, arch, pc, pm, stag, members, steps, period);
            case "hpeas_noseed" -> new Hpeas(algo, false, Hpeas.Ls.ADAPTIVE, pop, arch, pc, pm, stag, members, steps, period);
            case "hpeas_nols" -> new Hpeas(algo, true, Hpeas.Ls.NONE, pop, arch, pc, pm, stag, members, steps, period);
            case "hpeas_static" -> new Hpeas(algo, true, Hpeas.Ls.STATIC, pop, arch, pc, pm, stag, members, steps, period);
            case "nsga2" -> new Hpeas(algo, false, Hpeas.Ls.NONE, pop, arch, pc, pm, stag, members, steps, period);
            default -> throw new IllegalArgumentException("unknown algorithm " + algo);
        };
    }

    record Params(Map<String, String> m) {
        int i(String k, int def) { return m.containsKey(k) ? Integer.parseInt(m.get(k)) : def; }
        double d(String k, double def) { return m.containsKey(k) ? Double.parseDouble(m.get(k)) : def; }
    }
}
