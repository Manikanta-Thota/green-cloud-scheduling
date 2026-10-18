package gcs.core;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Dependency-free parity check: reads testdata/tiny_case.txt (the same file the Python tests use)
 * and verifies the evaluator and both baselines against the hand-computed values.
 * Exit code 0 = all checks passed.
 */
public final class SelfCheck {
    public static void main(String[] args) throws IOException {
        List<double[]> vms = new ArrayList<>(), tasks = new ArrayList<>();
        Map<Integer, double[]> hosts = new TreeMap<>();
        Map<String, Double> expect = new HashMap<>();
        int[] assign = null, expFcfs = null, expMinMin = null;
        for (String raw : Files.readAllLines(Path.of(args[0]))) {
            String line = raw.contains("#") ? raw.substring(0, raw.indexOf('#')) : raw;
            line = line.trim();
            if (line.isEmpty()) continue;
            String[] tok = line.split("\\s+");
            switch (tok[0]) {
                case "vm" -> vms.add(new double[] {d(tok[1]), d(tok[2]), d(tok[3])});
                case "host" -> hosts.put(Integer.parseInt(tok[1]), new double[] {d(tok[2]), d(tok[3])});
                case "task" -> tasks.add(new double[] {d(tok[1]), d(tok[2]), d(tok[3])});
                case "assign" -> assign = ints(tok);
                case "expect" -> expect.put(tok[1], d(tok[2]));
                case "expect_fcfs" -> expFcfs = ints(tok);
                case "expect_minmin" -> expMinMin = ints(tok);
                default -> throw new IllegalArgumentException("unknown key " + tok[0]);
            }
        }
        List<Integer> hostIds = new ArrayList<>(hosts.keySet());
        int n = tasks.size(), m = vms.size(), h = hostIds.size();
        double[] len = new double[n], arr = new double[n], dl = new double[n];
        for (int i = 0; i < n; i++) { len[i] = tasks.get(i)[0]; arr[i] = tasks.get(i)[1]; dl[i] = tasks.get(i)[2]; }
        double[] mips = new double[m], price = new double[m];
        int[] hostOf = new int[m];
        for (int i = 0; i < m; i++) { mips[i] = vms.get(i)[0]; price[i] = vms.get(i)[1]; hostOf[i] = hostIds.indexOf((int) vms.get(i)[2]); }
        double[] pIdle = new double[h], pMax = new double[h];
        for (int i = 0; i < h; i++) { pIdle[i] = hosts.get(hostIds.get(i))[0]; pMax[i] = hosts.get(hostIds.get(i))[1]; }
        Instance in = new Instance(len, arr, dl, mips, price, hostOf, pIdle, pMax);

        int failures = 0;
        Objectives o = Evaluator.evaluate(in, assign);
        failures += check("makespan", o.makespan(), expect.get("makespan"));
        failures += check("energy", o.energy(), expect.get("energy"));
        failures += check("cost", o.cost(), expect.get("cost"));
        failures += check("imbalance", o.imbalance(), expect.get("imbalance"));
        failures += check("sla", o.sla(), expect.get("sla"));
        failures += checkAssign("fcfs", new FcfsScheduler().schedule(in), expFcfs);
        failures += checkAssign("minmin", new MinMinScheduler().schedule(in), expMinMin);
        System.out.println(failures == 0 ? "SelfCheck: all checks passed" : "SelfCheck: " + failures + " FAILED");
        System.exit(failures == 0 ? 0 : 1);
    }

    private static double d(String s) { return Double.parseDouble(s); }

    private static int[] ints(String[] tok) {
        int[] r = new int[tok.length - 1];
        for (int i = 1; i < tok.length; i++) r[i - 1] = Integer.parseInt(tok[i]);
        return r;
    }

    private static int check(String name, double got, double want) {
        boolean ok = Math.abs(got - want) <= 1e-6 * Math.max(1.0, Math.abs(want));   // relative tolerance
        System.out.printf("  %-10s got %.6f want %.6f  %s%n", name, got, want, ok ? "ok" : "MISMATCH");
        return ok ? 0 : 1;
    }

    private static int checkAssign(String name, int[] got, int[] want) {
        boolean ok = Arrays.equals(got, want);
        String shown = got.length <= 20 ? "got " + Arrays.toString(got) + " want " + Arrays.toString(want)
                                        : "(" + got.length + " assignments compared)";
        System.out.printf("  %-10s %s  %s%n", name, shown, ok ? "ok" : "MISMATCH");
        return ok ? 0 : 1;
    }
}
