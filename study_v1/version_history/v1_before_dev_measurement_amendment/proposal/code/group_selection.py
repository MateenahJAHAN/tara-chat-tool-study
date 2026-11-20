"""Original deterministic exact-quota selection with one row per joint group.

Input-only helper. A bipartite matching prevents a mixed-label group from being
counted as available separately for both labels, the draft prepare failure.
"""
from collections import defaultdict
import hashlib


def _rank(seed, *parts):
    return hashlib.sha256((str(seed) + ":" + ":".join(map(str, parts))).encode()).hexdigest()


def select_balanced_unique_groups(records, per_class, seed):
    if per_class < 0:
        raise ValueError("Negative quota")
    options = {0: defaultdict(list), 1: defaultdict(list)}
    seen_ids = set()
    for record in records:
        if record["label"] not in options:
            raise ValueError("Expected binary benchmark label")
        if record["uid"] in seen_ids:
            raise ValueError("Duplicate uid")
        seen_ids.add(record["uid"])
        options[record["label"]][record["group"]].append(record)
    ordered = {
        label: sorted(groups, key=lambda group: _rank(seed, label, group))
        for label, groups in options.items()
    }
    group_to_slot, slot_to_group = {}, {}

    def augment(slot, visited):
        label, _ = slot
        for group in ordered[label]:
            if group in visited:
                continue
            visited.add(group)
            old_slot = group_to_slot.get(group)
            if old_slot is None or augment(old_slot, visited):
                group_to_slot[group] = slot
                slot_to_group[slot] = group
                return True
        return False

    slots = [(label, i) for label in (0, 1) for i in range(per_class)]
    for slot in slots:
        if not augment(slot, set()):
            raise ValueError(
                "Exact group-balanced quota infeasible: "
                f"target={per_class}/class, candidate_groups="
                f"{ {label: len(groups) for label, groups in options.items()} }, "
                f"joint_groups={len(set(options[0]) | set(options[1]))}"
            )
    chosen = [
        min(options[label][slot_to_group[(label, i)]],
            key=lambda row: _rank(seed, "row", row["uid"]))
        for label, i in slots
    ]
    assert len(chosen) == 2 * per_class
    assert len({row["group"] for row in chosen}) == len(chosen)
    assert all(sum(row["label"] == label for row in chosen) == per_class for label in (0, 1))
    return sorted(chosen, key=lambda row: row["uid"])
