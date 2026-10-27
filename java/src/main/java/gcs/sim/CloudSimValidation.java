package gcs.sim;

import ch.qos.logback.classic.Level;
import gcs.core.*;
import java.nio.file.Path;
import java.util.*;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.power.models.PowerModelHostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerSpaceShared;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.util.Log;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

/**
 * Replays a schedule in CloudSim Plus (Silva Filho et al., 2017) and compares it with the evaluator
 * (proposal Section 3.4, validation step 3). Each VM gets one PE and a space-shared cloudlet scheduler
 * (one task at a time, FIFO); hosts carry their VMs at the instance's MIPS; tasks are cloudlets bound to
 * their VM and submitted with a delay equal to their arrival time. Energy is recomputed from CloudSim's
 * cloudlet start/finish times with CloudSim's own linear host power model (PowerModelHostSimple).
 * Independent-task instances only: CloudSim Plus has no native task-precedence support.
 *
 * Usage: java -cp "java/out-sim;java/lib/*" gcs.sim.CloudSimValidation testdata/parity_case.txt
 */
public final class CloudSimValidation {
    /**
     * Accepted relative difference in makespan and energy. CloudSim Plus finishes every cloudlet 0.01 s
     * after its exact completion time, which accumulates along a VM's busy period, and its space-shared
     * scheduler can start a cloudlet that arrives at the same event as a completion ahead of cloudlets
     * already waiting; both are simulator mechanics, not differences in the scheduling model.
     */
    static final double TOLERANCE = 1e-3;

    public static void main(String[] args) throws Exception {
        Log.setLevel(Level.WARN);
        CaseFile c = CaseFile.read(Path.of(args[0]));
        Instance in = c.instance;
        if (in.hasPrecedence()) throw new IllegalArgumentException("precedence-constrained instances are not supported");
        Map<String, int[]> schedules = new LinkedHashMap<>();
        schedules.put("fcfs", new FcfsScheduler().schedule(in));
        schedules.put("minmin", new MinMinScheduler().schedule(in));
        schedules.put("energy_greedy", new EnergyGreedyScheduler().schedule(in));
        int worst = 0;
        for (var e : schedules.entrySet()) worst += compare(e.getKey(), in, e.getValue()) ? 0 : 1;
        System.out.println(worst == 0 ? "CloudSimValidation: all schedules agree" : "CloudSimValidation: " + worst + " DISAGREE");
        System.exit(worst == 0 ? 0 : 1);
    }

    static boolean compare(String name, Instance in, int[] assign) {
        int n = in.nTasks(), m = in.nVms(), nh = in.nHosts();
        final double S = 1e3;   // cloudlet lengths are integers: count thousandths of MI, and scale speeds alike
        CloudSimPlus sim = new CloudSimPlus(1e-6);
        List<Host> hosts = new ArrayList<>();
        for (int h = 0; h < nh; h++) {
            List<Pe> pes = new ArrayList<>();
            for (int v = 0; v < m; v++) if (in.hostOf[v] == h) pes.add(new PeSimple(in.mips[v] * S));
            Host host = new HostSimple(1_000_000, 100_000_000, 1_000_000, pes).setVmScheduler(new VmSchedulerTimeShared());
            host.setPowerModel(new PowerModelHostSimple(in.pMax[h], in.pIdle[h]));
            hosts.add(host);
        }
        var policy = new VmAllocationPolicySimple();
        policy.setFindHostForVmFunction((pol, vm) -> Optional.of(hosts.get(in.hostOf[(int) vm.getId()])));
        new DatacenterSimple(sim, hosts, policy);
        var broker = new DatacenterBrokerSimple(sim);
        List<Vm> vms = new ArrayList<>();
        for (int v = 0; v < m; v++) {
            Vm vm = new VmSimple(v, in.mips[v] * S, 1).setRam(10).setBw(1_000_000).setSize(10);
            vm.setCloudletScheduler(new CloudletSchedulerSpaceShared());
            vms.add(vm);
        }
        List<Cloudlet> cls = new ArrayList<>();
        int[] order = in.order;
        for (int k = 0; k < n; k++) {   // submit in execution order so equal arrival times keep that order
            int t = order[k];
            Cloudlet cl = new CloudletSimple(t, Math.round(in.length[t] * S), 1);
            cl.setUtilizationModelCpu(new UtilizationModelFull())
              .setUtilizationModelRam(new UtilizationModelDynamic(0)).setUtilizationModelBw(new UtilizationModelDynamic(0));
            cl.setSubmissionDelay(in.arrival[t]);
            cls.add(cl);
        }
        broker.submitVmList(vms);
        broker.submitCloudletList(cls);
        for (Cloudlet cl : cls) broker.bindCloudletToVm(cl, vms.get(assign[(int) cl.getId()]));
        sim.start();

        Evaluator.Workspace w = new Evaluator.Workspace(in);
        double[] obj = new double[Evaluator.N_OBJ];
        Evaluator.evaluate(in, assign, w, obj);
        double maxErr = 0, makespan = 0;
        double[] start = new double[n], finish = new double[n];
        int unfinished = 0;
        for (Cloudlet cl : cls) {
            int t = (int) cl.getId();
            if (!cl.isFinished()) { unfinished++; continue; }
            start[t] = cl.getStartTime();
            finish[t] = cl.getFinishTime();
            makespan = Math.max(makespan, finish[t]);
            maxErr = Math.max(maxErr, Math.abs(finish[t] - w.finish[t]));
            if (System.getenv("GCS_DEBUG") != null && Math.abs(finish[t] - w.finish[t]) > 1e-3)
                System.out.printf("    task %d vm %d arr %.4f len %.1f | ours %.4f-%.4f | cloudsim %.4f-%.4f (exec start %.4f)%n", t, assign[t],
                        in.arrival[t], in.length[t], w.start[t], w.finish[t], start[t], finish[t], cl.getStartTime());
        }
        // energy from CloudSim's timeline and power model, with the protocol's host on/off rule
        double energy = 0;
        for (int h = 0; h < nh; h++) {
            List<double[]> ev = new ArrayList<>();
            for (int t = 0; t < n; t++) {
                int v = assign[t];
                if (in.hostOf[v] != h) continue;
                ev.add(new double[] {start[t], in.mips[v]});
                ev.add(new double[] {finish[t], -in.mips[v]});
            }
            if (ev.isEmpty()) continue;
            ev.sort(Comparator.<double[]>comparingDouble(e -> e[0]).thenComparingDouble(e -> e[1]));
            var pm = (PowerModelHostSimple) hosts.get(h).getPowerModel();
            double busy = 0, prev = ev.get(0)[0];
            for (double[] e : ev) {
                if (e[0] > prev) { energy += pm.getPower(Math.max(0, Math.min(1, busy / in.hostCap[h]))) * (e[0] - prev); prev = e[0]; }
                busy += e[1];
            }
        }
        double relMk = Math.abs(makespan - obj[Evaluator.MAKESPAN]) / obj[Evaluator.MAKESPAN];
        double relE = Math.abs(energy - obj[Evaluator.ENERGY]) / obj[Evaluator.ENERGY];
        boolean ok = unfinished == 0 && relMk < TOLERANCE && relE < TOLERANCE;
        System.out.printf("  %-14s tasks %d  unfinished %d  max |finish diff| %.3f s  makespan %.3f vs %.3f (rel %.1e)"
                + "  energy %.0f vs %.0f J (rel %.1e)  %s%n", name, n, unfinished, maxErr, makespan, obj[Evaluator.MAKESPAN],
                relMk, energy, obj[Evaluator.ENERGY], relE, ok ? "ok" : "MISMATCH");
        System.out.printf(Locale.ROOT, "CSV,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.3e,%.3e,%d%n", name, n, unfinished, maxErr, makespan,
                obj[Evaluator.MAKESPAN], energy, obj[Evaluator.ENERGY], relMk, relE, ok ? 1 : 0);
        return ok;
    }
}
