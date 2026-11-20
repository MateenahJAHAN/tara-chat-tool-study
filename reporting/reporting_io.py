"""Original, phase-authorized report IO. No model imports or inference calls."""
import hashlib
import json
from pathlib import Path
import re

CONDITIONS = ("TEXT_CATALOGUE", "NATIVE_TOOLS")
ENDPOINTS = ("false_call", "broad_false_call", "operational_abstention", "strict_abstention",
             "call_attempt", "broad_call_attempt", "valid_call", "correct_call", "format_compliant",
             "parse_error", "unknown_name", "multiple_calls", "empty_output", "truncated", "errored",
             "invalid_or_empty", "unknown_output", "protocol_format_error", "invalid_stop_reason")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def ensure_public_payload(value):
    """Refuse machine-home paths in shareable JSON; authorization stays local."""
    serialized = json.dumps(value, ensure_ascii=False)
    if re.search(r"/Users/[^/\s\"']+|/home/[^/\s\"']+|[A-Za-z]:\\\\Users\\\\", serialized):
        raise RuntimeError("Shareable payload contains a machine-home path")
    return value


def local_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise RuntimeError("Input path escapes main root")
    return path


def authorize(main_root, phase, authorization):
    """Read authorization before any actual outcome/report/raw generation file."""
    root = Path(main_root).resolve()
    auth_path = Path(authorization).resolve()
    auth = read_json(auth_path)
    if not (isinstance(auth.get("main_root"), str) and auth["main_root"]
            and auth.get("approved") is True and auth.get("completed") is True
            and auth.get("phase") == phase and Path(auth.get("main_root", "")).resolve() == root):
        raise RuntimeError("Explicit matching completed-phase reporting authorization is required")
    if phase not in {"DEV", "HELDOUT"}:
        raise RuntimeError("Unknown reporting phase")
    if (not isinstance(auth.get("design_id"), str) or not auth["design_id"]
            or not re.fullmatch(r"[0-9a-f]{64}", auth.get("split_manifest_sha256", ""))):
        raise RuntimeError("Authorization must bind design ID and exact split-manifest SHA256")
    reports = auth.get("approved_reports", {})
    behaviour = auth.get("behaviour_report", "reports/behaviour_" + phase.lower() + ".json")
    if behaviour not in reports:
        raise RuntimeError("Authorization must name/hash the completed behaviour report")
    for relative in reports:
        if Path(relative).parent != Path("reports") or Path(relative).suffix != ".json":
            raise RuntimeError("Only main_v1/reports JSON outcomes can be approved")
    if phase == "DEV" and any("heldout" in relative.lower() for relative in reports):
        raise RuntimeError("DEV authorization cannot expose heldout reports")
    if phase == "HELDOUT":
        # These metadata checks precede every heldout outcome read.
        gate = local_path(root, "approvals/root-heldout-approval.json")
        if read_json(gate).get("approved") is not True:
            raise RuntimeError("Root heldout approval is missing")
        if auth.get("heldout_gate_approval_sha256") != sha(gate):
            raise RuntimeError("Reporting authorization does not bind the root heldout approval")
        complete = local_path(root, "runs/heldout/HELDOUT_COMPLETE.json")
        if not complete.is_file() or auth.get("heldout_complete_sha256") != sha(complete):
            raise RuntimeError("Heldout completion metadata missing or mismatched")
    return root, auth, sha(auth_path)


def approved_report(root, auth, relative):
    if relative not in auth["approved_reports"]:
        raise RuntimeError("Report not explicitly authorized: " + relative)
    path = local_path(root, relative)
    if sha(path) != auth["approved_reports"][relative]:
        raise RuntimeError("Report hash mismatch: " + relative)
    return read_json(path)


def paired_counts(a, b):
    """All matched scored rows retained; unknown/missing endpoints are explicit."""
    if set(a) != set(b):
        raise RuntimeError("Incomplete paired bank")
    result = {}
    for label in (0, 1):
        ids = [uid for uid in sorted(a) if a[uid]["label"] == label]
        for uid in ids:
            if a[uid]["group"] != b[uid]["group"] or a[uid]["label"] != b[uid]["label"]:
                raise RuntimeError("Paired group/label mismatch")
        endpoints = sorted(set(ENDPOINTS) | {key for uid in ids for rec in (a[uid], b[uid])
                                           for key, value in rec["score"].items() if isinstance(value, bool)})
        result["label" + str(label)] = {}
        for endpoint in endpoints:
            counts = {"a0_b0": 0, "a0_b1": 0, "a1_b0": 0, "a1_b1": 0,
                      "not_applicable_or_missing_pairs": 0, "n_scheduled": len(ids)}
            for uid in ids:
                av, bv = a[uid]["score"].get(endpoint), b[uid]["score"].get(endpoint)
                if isinstance(av, bool) and isinstance(bv, bool):
                    counts["a" + str(int(av)) + "_b" + str(int(bv))] += 1
                else:
                    counts["not_applicable_or_missing_pairs"] += 1
            result["label" + str(label)][endpoint] = counts
    return result
