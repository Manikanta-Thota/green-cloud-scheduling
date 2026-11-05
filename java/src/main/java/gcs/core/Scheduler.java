package gcs.core;

/** A deterministic scheduler: maps every task to a VM index. */
public interface Scheduler {
    String name();
    int[] schedule(Instance instance);
}
