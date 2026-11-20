"""No silent use of old development scores; synthetic files only."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import common
from run import dev_score_directory


class ExplicitScoreVersion(unittest.TestCase):
    def test_old_scores_require_complete_hashed_derivation(self):
        cfg = common.load_config()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dev = root / "runs/dev"; dev.mkdir(parents=True)
            for condition in common.CONDITIONS:
                common.append_jsonl(dev / f"generations_baseline_{condition}.jsonl", {"historical": True})
            with self.assertRaisesRegex(RuntimeError, "historical scores"):
                dev_score_directory(root, cfg)
            for condition in common.CONDITIONS:
                p = dev / f"generations_baseline_{condition}.jsonl"; p.unlink()
                common.append_jsonl(p, {"score_version": cfg["measurement_version"]})
            self.assertEqual(dev_score_directory(root, cfg), dev)
            derived = dev / cfg["dev_scores_subdirectory"]; derived.mkdir()
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                dev_score_directory(root, cfg)
            files = {}
            for condition in common.CONDITIONS:
                p = derived / f"generations_baseline_{condition}.jsonl"
                common.append_jsonl(p, {"score_version": cfg["measurement_version"]})
                files[p.name] = {"derived_sha256": common.sha256_file(p)}
            common.write_json(derived / "RESCORING_MANIFEST.json", {
                "status": "PASS", "score_version": cfg["measurement_version"],
                "scorer_sha256": common.sha256_file(common.CODE_DIR / "parser.py"), "files": files})
            self.assertEqual(dev_score_directory(root, cfg), derived)
            next(derived.glob("*.jsonl")).write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                dev_score_directory(root, cfg)


if __name__ == "__main__":
    unittest.main()
