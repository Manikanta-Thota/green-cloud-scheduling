package gcs.core;

/** A scheduler maps every task to a VM index. Stochastic schedulers take their seed in the constructor. */
public interface Scheduler {
    String name();
    int[] schedule(Instance instance);
}
