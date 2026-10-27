package gcs.opt;

import java.util.*;

/**
 * Bounded external archive of mutually non-dominated solutions under feasibility-first dominance.
 * Solutions whose objective vector is already in the archive are rejected; when the archive overflows,
 * the member with the smallest crowding distance is removed (extremes are never removed).
 */
public final class Archive {
    public final int capacity;
    final List<Solution> members = new ArrayList<>();

    public Archive(int capacity) { this.capacity = capacity; }

    /** @return true if {@code s} entered the archive. */
    public boolean add(Solution s) {
        for (Solution a : members)
            if (Pareto.cdominates(a, s) || (Arrays.equals(a.f, s.f) && a.cv == s.cv)) return false;
        members.removeIf(a -> Pareto.cdominates(s, a));
        members.add(s);
        if (members.size() > capacity) {
            Pareto.crowding(members);
            Solution worst = members.get(0);
            for (Solution a : members) if (a.crowd < worst.crowd) worst = a;
            members.remove(worst);
            return worst != s;
        }
        return true;
    }

    public List<Solution> members() { return Collections.unmodifiableList(members); }
    public int size() { return members.size(); }
}
