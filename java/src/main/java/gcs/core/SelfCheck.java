package gcs.core;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

/**
 * Dependency-free parity check: reads a case file in testdata/ (the same files the Python tests use)
 * and verifies the evaluator and the baseline schedulers against the expected values in the file.
 * Exit code 0 = all checks passed.
 */
public final class SelfCheck {
    public static void main(String[] args) throws IOException {
        CaseFile c = CaseFile.read(Path.of(args[0]));
        Instance in = c.instance;
        int failures = 0;
        Objectives o = Evaluator.evaluate(in, c.assign);
        failures += check("makespan", o.makespan(), c.expect.get("makespan"));
        failures += check("energy", o.energy(), c.expect.get("energy"));
        failures += check("cost", o.cost(), c.expect.get("cost"));
        failures += check("imbalance", o.imbalance(), c.expect.get("imbalance"));
        failures += check("sla", o.sla(), c.expect.get("sla"));
        Map<String, Scheduler> schedulers = Map.of("fcfs", new FcfsScheduler(), "minmin", new MinMinScheduler());
        for (var e : c.expectAssign.entrySet())
            failures += checkAssign(e.getKey(), schedulers.get(e.getKey()).schedule(in), e.getValue());
        System.out.println(failures == 0 ? "SelfCheck: all checks passed" : "SelfCheck: " + failures + " FAILED");
        System.exit(failures == 0 ? 0 : 1);
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
