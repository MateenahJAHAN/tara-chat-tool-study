from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import unittest
from group_selection import select_balanced_unique_groups as select


def row(uid, label, group):
    return {"uid": uid, "label": label, "group": group}


class GroupSelectionChecks(unittest.TestCase):
    def test_mixed_group_not_double_counted(self):
        with self.assertRaisesRegex(ValueError, "infeasible"):
            select([row("a", 0, "g"), row("b", 1, "g")], 1, 1)

    def test_matching_reassigns_shared_group(self):
        data = [row("a", 0, "shared"), row("b", 0, "exclusive"), row("c", 1, "shared")]
        out = select(data, 1, 1)
        self.assertEqual({x["uid"] for x in out}, {"b", "c"})

    def test_group_cap_global_and_exact_balance(self):
        data = [row(f"{g}-{label}-{i}", label, str(g))
                for g in range(8) for label in (0, 1) for i in range(3)]
        out = select(data, 3, 20261119)
        self.assertEqual(len(out), 6)
        self.assertEqual(len({x["group"] for x in out}), 6)
        self.assertEqual([sum(x["label"] == label for x in out) for label in (0, 1)], [3, 3])

    def test_order_invariance(self):
        data = [row(f"{g}-{label}", label, str(g)) for g in range(8) for label in (0, 1)]
        self.assertEqual(select(data, 3, 4), select(list(reversed(data)), 3, 4))

    def test_duplicate_uid_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select([row("same", 0, "a"), row("same", 1, "b")], 1, 1)

    def test_zero_quota(self):
        self.assertEqual(select([], 0, 1), [])


if __name__ == "__main__":
    unittest.main()
