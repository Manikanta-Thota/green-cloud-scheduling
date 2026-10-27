package gcs.core;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Reader for the shared text format written by python/gcs/export_case.py:
 * <pre>
 * vm &lt;mips&gt; &lt;price_per_s&gt; &lt;host_id&gt;     host &lt;id&gt; &lt;p_idle_w&gt; &lt;p_max_w&gt;
 * task &lt;length_mi&gt; &lt;arrival_s&gt; &lt;deadline_s&gt; [rank]     pred &lt;task&gt; &lt;p1&gt; &lt;p2&gt; ...
 * assign ...   expect &lt;objective&gt; &lt;value&gt;   expect_&lt;scheduler&gt; ...   meta &lt;key&gt; &lt;value&gt;   seed &lt;algorithm&gt; &lt;long&gt;
 * </pre>
 */
public final class CaseFile {
    public Instance instance;
    public int[] assign;
    public final Map<String, Double> expect = new HashMap<>();
    public final Map<String, int[]> expectAssign = new HashMap<>();
    public final Map<String, String> meta = new LinkedHashMap<>();
    public final Map<String, Long> seeds = new HashMap<>();

    public static CaseFile read(Path path) throws IOException {
        CaseFile c = new CaseFile();
        List<double[]> vms = new ArrayList<>(), tasks = new ArrayList<>();
        Map<Integer, double[]> hosts = new TreeMap<>();
        Map<Integer, int[]> preds = new HashMap<>();
        for (String raw : Files.readAllLines(path)) {
            String line = raw.contains("#") ? raw.substring(0, raw.indexOf('#')) : raw;
            line = line.trim();
            if (line.isEmpty()) continue;
            String[] tok = line.split("\s+");
            switch (tok[0]) {
                case "vm" -> vms.add(new double[] {d(tok[1]), d(tok[2]), d(tok[3])});
                case "host" -> hosts.put(Integer.parseInt(tok[1]), new double[] {d(tok[2]), d(tok[3])});
                case "task" -> tasks.add(new double[] {d(tok[1]), d(tok[2]), d(tok[3]), tok.length > 4 ? d(tok[4]) : 0});
                case "pred" -> preds.put(Integer.parseInt(tok[1]), Arrays.copyOfRange(ints(tok), 1, tok.length - 1));
                case "assign" -> c.assign = ints(tok);
                case "expect" -> c.expect.put(tok[1], d(tok[2]));
                case "meta" -> c.meta.put(tok[1], tok[2]);
                case "seed" -> c.seeds.put(tok[1], Long.parseLong(tok[2]));
                default -> {
                    if (tok[0].startsWith("expect_")) c.expectAssign.put(tok[0].substring(7), ints(tok));
                    else throw new IllegalArgumentException("unknown key " + tok[0]);
                }
            }
        }
        List<Integer> hostIds = new ArrayList<>(hosts.keySet());
        int n = tasks.size(), m = vms.size(), h = hostIds.size();
        double[] len = new double[n], arr = new double[n], dl = new double[n];
        int[] rank = new int[n];
        int[][] pr = new int[n][];
        for (int i = 0; i < n; i++) {
            double[] t = tasks.get(i);
            len[i] = t[0]; arr[i] = t[1]; dl[i] = t[2]; rank[i] = (int) t[3];
            pr[i] = preds.getOrDefault(i, new int[0]);
        }
        double[] mips = new double[m], price = new double[m];
        int[] hostOf = new int[m];
        for (int i = 0; i < m; i++) { mips[i] = vms.get(i)[0]; price[i] = vms.get(i)[1]; hostOf[i] = hostIds.indexOf((int) vms.get(i)[2]); }
        double[] pIdle = new double[h], pMax = new double[h];
        for (int i = 0; i < h; i++) { pIdle[i] = hosts.get(hostIds.get(i))[0]; pMax[i] = hosts.get(hostIds.get(i))[1]; }
        c.instance = new Instance(len, arr, dl, rank, pr, mips, price, hostOf, pIdle, pMax);
        return c;
    }

    private static double d(String s) { return Double.parseDouble(s); }

    private static int[] ints(String[] tok) {
        int[] r = new int[tok.length - 1];
        for (int i = 1; i < tok.length; i++) r[i - 1] = Integer.parseInt(tok[i]);
        return r;
    }
}
