package gcs.opt;

import java.util.*;

/**
 * Pareto-front quality indicators on objective vectors (all minimised): exact hypervolume by slicing
 * objectives (HSO; While et al., 2006), inverted generational distance and Schott's spacing.
 * Mirrors python/gcs/pareto.py, which the Python tests use to cross-check this implementation.
 */
public final class Indicators {
    private Indicators() {}

    public static double hypervolume(List<double[]> pts, double[] ref) {
        List<double[]> p = new ArrayList<>();
        for (double[] x : pts) {
            boolean inside = true;
            for (int k = 0; k < ref.length; k++) if (x[k] >= ref[k]) { inside = false; break; }
            if (inside) p.add(x);
        }
        return hv(nonDominated(p, ref.length), ref, ref.length);
    }

    private static double hv(List<double[]> p, double[] ref, int d) {
        if (p.isEmpty()) return 0;
        if (d == 1) {
            double mn = Double.POSITIVE_INFINITY;
            for (double[] x : p) mn = Math.min(mn, x[0]);
            return ref[0] - mn;
        }
        if (d == 2) {
            List<double[]> s = new ArrayList<>(p);
            s.sort(Comparator.comparingDouble(x -> x[0]));
            double vol = 0, prevY = ref[1];
            for (double[] x : s) if (x[1] < prevY) { vol += (ref[0] - x[0]) * (prevY - x[1]); prevY = x[1]; }
            return vol;
        }
        List<double[]> s = new ArrayList<>(p);
        s.sort(Comparator.comparingDouble(x -> x[d - 1]));
        double vol = 0;
        List<double[]> slice = new ArrayList<>();
        for (int i = 0; i < s.size(); i++) {
            slice.add(s.get(i));
            double upper = i + 1 < s.size() ? s.get(i + 1)[d - 1] : ref[d - 1];
            double h = upper - s.get(i)[d - 1];
            if (h > 0) vol += hv(nonDominated(slice, d - 1), ref, d - 1) * h;
        }
        return vol;
    }

    /** Non-dominated subset considering only the first {@code d} objectives (duplicates collapse to one). */
    static List<double[]> nonDominated(List<double[]> p, int d) {
        List<double[]> out = new ArrayList<>();
        outer:
        for (int i = 0; i < p.size(); i++) {
            double[] a = p.get(i);
            for (int j = 0; j < p.size(); j++) {
                if (i == j) continue;
                double[] b = p.get(j);
                boolean leq = true, lt = false;
                for (int k = 0; k < d; k++) { if (b[k] > a[k]) { leq = false; break; } if (b[k] < a[k]) lt = true; }
                if (leq && (lt || j < i)) continue outer;   // dominated, or an earlier duplicate exists
            }
            out.add(a);
        }
        return out;
    }

    public static double igd(List<double[]> pts, List<double[]> refFront) {
        double sum = 0;
        for (double[] r : refFront) {
            double best = Double.POSITIVE_INFINITY;
            for (double[] x : pts) {
                double d2 = 0;
                for (int k = 0; k < r.length; k++) d2 += (r[k] - x[k]) * (r[k] - x[k]);
                best = Math.min(best, d2);
            }
            sum += Math.sqrt(best);
        }
        return sum / refFront.size();
    }

    /** Schott's spacing: standard deviation of nearest-neighbour L1 distances (0 for fewer than two points). */
    public static double spacing(List<double[]> pts) {
        int n = pts.size();
        if (n < 2) return 0;
        double[] dmin = new double[n];
        for (int i = 0; i < n; i++) {
            dmin[i] = Double.POSITIVE_INFINITY;
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                double d = 0;
                for (int k = 0; k < pts.get(i).length; k++) d += Math.abs(pts.get(i)[k] - pts.get(j)[k]);
                dmin[i] = Math.min(dmin[i], d);
            }
        }
        double mean = Arrays.stream(dmin).average().orElse(0), var = 0;
        for (double d : dmin) var += (d - mean) * (d - mean);
        return Math.sqrt(var / (n - 1));
    }
}
