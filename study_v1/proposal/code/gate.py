"""Heldout gate. `--write-hash-record` writes frozen/hash_record.json for root review. Root (only) writes
approvals/root-heldout-approval.json containing the same hashes. Heldout code calls require_approval()."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

REQUIRED = ("code_tree_sha256", "config_sha256", "split_manifest_sha256", "model_manifest_sha256",
            "parser_sha256", "selected_params_sha256", "dev_record_sha256")
HELDOUT_PHASES = ("HELDOUT_BASELINE", "HELDOUT_PROBE_EVAL", "HELDOUT_STEERING")


def compute_hashes(study_root, verify_model=True):
    root = Path(study_root)
    cfg = common.load_config()
    if verify_model:
        common.verify_staged(root, cfg, include_model=True)
    selected = common.read_json(root / "frozen/selected_params.json")
    artifacts = {"probe_selection_sha256": "probe_selection.json", "probe_weights_sha256": "probe_weights.npz",
                 "baseline_models_sha256": "baseline_models.json", "control_weights_sha256": "control_weights.npz",
                 "steering_vectors_sha256": "steering_vectors.npz"}
    for key, filename in artifacts.items():
        if selected.get(key) != common.sha256_file(root / "frozen" / filename):
            raise RuntimeError("STOP: selected fitted artifact changed: " + filename)
    dev_record = common.read_json(root / "runs/dev/DEV_RECORD.json")
    for rel, digest in dev_record["files"].items():
        if common.sha256_file(root / "runs/dev" / rel) != digest:
            raise RuntimeError("STOP: development artifact changed: " + rel)
    return {
        "code_tree_sha256": common.tree_sha256(common.code_files(), common.PROPOSAL_DIR),
        "config_sha256": common.sha256_file(common.CONFIG_PATH),
        "split_manifest_sha256": common.sha256_file(root / "manifests" / "split_manifest.json"),
        "model_manifest_sha256": common.sha256_text(common.canonical_json(cfg["model"]["files_sha256"])),
        "parser_sha256": common.sha256_file(common.CODE_DIR / "parser.py"),
        "selected_params_sha256": common.sha256_file(root / "frozen" / "selected_params.json"),
        "dev_record_sha256": common.sha256_file(root / "runs" / "dev" / "DEV_RECORD.json"),
    }


def check_approval(approval, current, phase, design_id):
    problems = []
    if approval.get("approved") is not True:
        problems.append("approved!=true")
    if approval.get("approved_for") != design_id:
        problems.append("approved_for mismatch")
    if phase not in approval.get("phases", []):
        problems.append(f"phase {phase} not approved")
    hashes = approval.get("hashes", {})
    for k in REQUIRED:
        if k not in hashes:
            problems.append("missing " + k)
        elif hashes[k] != current[k]:
            problems.append("mismatch " + k)
    return problems


def require_approval(study_root, phase):
    if phase not in HELDOUT_PHASES:
        raise ValueError(phase)
    root = Path(study_root)
    path = root / "approvals" / "root-heldout-approval.json"
    if not path.exists():
        raise RuntimeError("STOP: no root-heldout-approval.json; heldout is locked")
    problems = check_approval(common.read_json(path), compute_hashes(root), phase, common.load_config()["design_id"])
    if problems:
        raise RuntimeError("STOP: heldout approval invalid: " + "; ".join(problems))
    return common.sha256_file(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", required=True)
    ap.add_argument("--write-hash-record", action="store_true")
    args = ap.parse_args()
    hashes = compute_hashes(args.study_root)
    record = {"design_id": common.load_config()["design_id"], "computed_utc": common.now_utc(), "hashes": hashes,
              "approval_template": {"approved_for": common.load_config()["design_id"], "approved": "ROOT_SETS_true",
                                    "approver_role": "root", "approved_at_utc": "ROOT_FILLS", "phases": list(HELDOUT_PHASES),
                                    "hashes": hashes, "notes": "ROOT_FILLS"}}
    if args.write_hash_record:
        common.write_json(Path(args.study_root) / "frozen" / "hash_record.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
