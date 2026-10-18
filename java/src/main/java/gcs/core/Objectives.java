package gcs.core;

/** Objective vector; all components are minimised. */
public record Objectives(double energy, double makespan, double cost, double imbalance, double sla) {
    public double[] toArray() { return new double[] {energy, makespan, cost, imbalance, sla}; }
}
