"""Pre-heldout artifact mutation regressions; synthetic bytes only."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import common
import gate


class FittedArtifactGate(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ["frozen", "runs/dev", "manifests"]:
            (self.root / name).mkdir(parents=True)
        files = {"probe_selection_sha256": "probe_selection.json", "probe_weights_sha256": "probe_weights.npz",
                 "baseline_models_sha256": "baseline_models.json", "control_weights_sha256": "control_weights.npz",
                 "steering_vectors_sha256": "steering_vectors.npz"}
        selected = {}
        for field, filename in files.items():
            p = self.root / "frozen" / filename
            p.write_bytes(b"synthetic fixed artifact " + filename.encode())
            selected[field] = common.sha256_file(p)
        common.write_json(self.root / "frozen/selected_params.json", selected)
        common.write_json(self.root / "manifests/split_manifest.json", {"synthetic": True})
        (self.root / "runs/dev/raw.jsonl").write_text("{}\n")
        common.write_json(self.root / "runs/dev/DEV_RECORD.json", {
            "files": {"raw.jsonl": common.sha256_file(self.root / "runs/dev/raw.jsonl")}})

    def tearDown(self):
        self.temp.cleanup()

    def test_fixed_artifact_tree_hashes(self):
        hashes = gate.compute_hashes(self.root, verify_model=False)
        self.assertEqual(set(hashes), set(gate.REQUIRED))

    def test_mutated_fitted_weights_rejected(self):
        (self.root / "frozen/control_weights.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "fitted artifact changed"):
            gate.compute_hashes(self.root, verify_model=False)

    def test_mutated_dev_measurement_rejected(self):
        (self.root / "runs/dev/raw.jsonl").write_text('{"changed":true}\n')
        with self.assertRaisesRegex(RuntimeError, "development artifact changed"):
            gate.compute_hashes(self.root, verify_model=False)

    def test_missing_or_changed_root_approval_rejected(self):
        hashes = gate.compute_hashes(self.root, verify_model=False)
        approval = {"approved": True, "approved_for": "TEST", "phases": ["HELDOUT_BASELINE"], "hashes": hashes.copy()}
        self.assertEqual(gate.check_approval(approval, hashes, "HELDOUT_BASELINE", "TEST"), [])
        approval["hashes"]["selected_params_sha256"] = "wrong"
        self.assertIn("mismatch selected_params_sha256", gate.check_approval(approval, hashes, "HELDOUT_BASELINE", "TEST"))


if __name__ == "__main__":
    unittest.main()
