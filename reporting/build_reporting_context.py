"""Summarize retained scorer endpoints only after explicit completed-phase permission.

Default prints requirements, reading no results. Does not fit/select/test anything.
Actual generation text, arguments and activations are not copied to the report.
"""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reporting_io import CONDITIONS, ENDPOINTS, approved_report, authorize, ensure_public_payload, local_path, paired_counts, read_json, sha


def summarize(records):
    result = {}
    for label in (0, 1):
        rows = [rec for rec in records if rec["label"] == label]
        keys = sorted(set(ENDPOINTS) | {key for rec in rows for key, value in rec["score"].items() if isinstance(value, bool)})
        endpoints = {}
        for key in keys:
            values = [rec["score"].get(key) for rec in rows]
            known = [value for value in values if isinstance(value, bool)]
            endpoints[key] = {"k": sum(known), "n_boolean": len(known), "n_scheduled": len(rows),
                              "not_applicable": sum(key in rec["score"] and rec["score"][key] is None for rec in rows),
                              "missing_or_nonboolean": sum(key not in rec["score"] or rec["score"][key] is not None
                                                           and not isinstance(rec["score"][key], bool) for rec in rows)}
        result["label" + str(label)] = {
            "n_scheduled": len(rows), "n_groups": len({rec["group"] for rec in rows}), "endpoints": endpoints,
            "stop_reason_counts": dict(Counter(rec.get("stop_reason", "MISSING") for rec in rows)),
            "error_records": sum(rec.get("error") is not None for rec in rows),
            "scorer_error_records": sum(rec["score"].get("scorer_error") is not None for rec in rows),
            "n_attempts_distribution": dict(Counter(str(rec["score"].get("n_attempts", "MISSING")) for rec in rows)),
            "validity_issue_counts": dict(Counter(issue for rec in rows for issue in rec["score"].get("validity_issues") or [])),
            "parser_diagnostic_counts": dict(Counter(issue for rec in rows for issue in rec["score"].get("parser_diagnostics") or [])),
            "format_diagnostic_counts": dict(Counter(issue for rec in rows for issue in rec["score"].get("format_diagnostics") or [])),
            "attempt_kind_counts": dict(Counter(kind for rec in rows for kind in rec["score"].get("attempt_kinds") or [])),
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main-root", type=Path)
    ap.add_argument("--phase", choices=["DEV", "HELDOUT"])
    ap.add_argument("--authorization", type=Path)
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if not args.build:
        print(json.dumps({"results_read": False, "context_written": False,
                          "requires": ["--build", "--main-root", "--phase", "--authorization",
                                       "parent approved+completed exact phase and behaviour report hash",
                                       "HELDOUT also binds actual root approval and completion marker"]}, indent=2))
        return
    if not all((args.main_root, args.phase, args.authorization)):
        ap.error("Actual mode needs main root, phase and explicit authorization")
    root, auth, auth_sha = authorize(args.main_root, args.phase, args.authorization)
    behaviour_rel = auth.get("behaviour_report", "reports/behaviour_" + args.phase.lower() + ".json")
    behaviour = approved_report(root, auth, behaviour_rel)
    if behaviour.get("split") != args.phase:
        raise RuntimeError("Behaviour report phase mismatch")
    manifest_path = local_path(root, "manifests/split_manifest.json")
    manifest = read_json(manifest_path)
    if manifest.get("design_id") != auth["design_id"] or sha(manifest_path) != auth["split_manifest_sha256"]:
        raise RuntimeError("Manifest design/hash does not match local phase authorization")
    expected = {item["uid"]: item for item in manifest["items"] if item["split"] == args.phase}
    if not expected:
        raise RuntimeError("No scheduled rows in phase")
    sources = {"manifests/split_manifest.json": sha(manifest_path), behaviour_rel: sha(local_path(root, behaviour_rel))}
    freeze_path = local_path(root, "PROSPECTIVE_FREEZE_RECORD.json")
    freeze = read_json(freeze_path)
    if freeze["files"]["manifests/split_manifest.json"] != sha(manifest_path):
        raise RuntimeError("Split manifest differs from prospective freeze")
    sources["PROSPECTIVE_FREEZE_RECORD.json"] = sha(freeze_path)
    banks = {}
    for condition in CONDITIONS:
        bank_spec = auth.get("scored_banks", {}).get(condition)
        relative = bank_spec["path"] if bank_spec else "runs/" + args.phase.lower() + "/generations_baseline_" + condition + ".jsonl"
        if args.phase == "DEV" and "heldout" in str(relative).lower():
            raise RuntimeError("DEV context cannot read a heldout scored bank")
        path = local_path(root, relative)
        if bank_spec and sha(path) != bank_spec["sha256"]:
            raise RuntimeError("Authorized scored bank hash mismatch")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if any(rec.get("split") != args.phase for rec in rows):
            raise RuntimeError("Raw baseline contains unexpected phase")
        bank = {rec["uid"]: rec for rec in rows}
        if len(bank) != len(rows) or set(bank) != set(expected):
            raise RuntimeError("Incomplete/duplicate scheduled scored bank")
        for uid, rec in bank.items():
            if any(rec.get(key) != expected[uid][key] for key in ("label", "group", "split")) or rec.get("condition") != condition:
                raise RuntimeError("Scored row metadata does not match frozen manifest")
            if not isinstance(rec.get("score"), dict):
                raise RuntimeError("Missing scored output is retained failure; no completion report")
        banks[condition] = bank
        sources[relative] = sha(path)
    summaries = {condition: summarize(list(bank.values())) for condition, bank in banks.items()}
    for condition in CONDITIONS:
        for label, endpoints in behaviour["rates"][condition].items():
            for key, value in endpoints.items():
                if isinstance(value, dict) and "k" in value and "n" in value:
                    raw = summaries[condition][label]["endpoints"].get(key)
                    if raw is None or raw["k"] != value["k"] or raw["n_boolean"] != value["n"]:
                        raise RuntimeError("Existing report and raw scored counts disagree")
    selected_path = local_path(root, "frozen/selected_params.json")
    probe_path = local_path(root, "frozen/probe_selection.json")
    selected = read_json(selected_path) if selected_path.is_file() else None
    probes = read_json(probe_path) if probe_path.is_file() else None
    for relative, path in (("frozen/selected_params.json", selected_path), ("frozen/probe_selection.json", probe_path)):
        if path.is_file():
            sources[relative] = sha(path)
    rows = list(expected.values())
    context = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "phase": args.phase,
        "design_id": manifest["design_id"], "split_manifest_sha256": sha(manifest_path), "authorization_sha256": auth_sha,
        "behaviour_report": behaviour_rel, "scoring_version": auth.get("scoring_version", "as recorded in authorized report and scored banks"),
        "builder_sha256": sha(Path(__file__)),
        "status": "ACTUAL_COMPLETED_PHASE_DESCRIPTIVE_COUNTS", "no_new_selection_or_tests": True,
        "statistical_unit": "Selected BFCL item/component; paired interfaces are repeated observations, not additional independent items",
        "scope": "Native DEV is selection-used development; live HELDOUT is a separate selected finite bank. No population/safety inference.",
        "scheduled_counts": {"rows": len(rows), "groups": len({row["group"] for row in rows}),
                             "label0": sum(row["label"] == 0 for row in rows), "label1": sum(row["label"] == 1 for row in rows)},
        "scheduled_steering_subset": {"rows": sum(bool(row.get("steering_subset")) for row in rows),
                                      "label0": sum(row["label"] == 0 and bool(row.get("steering_subset")) for row in rows),
                                      "label1": sum(row["label"] == 1 and bool(row.get("steering_subset")) for row in rows),
                                      "scope": "Frozen subset schedule, not a claim that every optional steering generation ran"},
        "condition_summaries": summaries,
        "paired_counts": paired_counts(banks[CONDITIONS[0]], banks[CONDITIONS[1]]),
        "probe_selection": probes.get("selected") if probes else None,
        "surface_selection": probes.get("baselines") if probes else None,
        "dev_transfer_matrix_selection_used": probes.get("dev_transfer_matrix_optimistic") if probes else None,
        "steering_status": selected.get("steering") if selected else None,
        "unavailable": ([] if probes else ["probe_selection not yet produced"]) + ([] if selected else ["steering selection/status not yet produced"]),
        "input_sha256": sources,
        "counts_definition": "Raw retained boolean scorer values. n_scheduled always includes errors, unknown and truncated outputs; nonapplicable/missing values are separate, never silently dropped.",
    }
    output = local_path(root, "reports/reporting_context_" + args.phase.lower() + ".json")
    ensure_public_payload(context)
    with output.open("x") as handle:
        json.dump(context, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"context": str(output.relative_to(root)), "design_id": manifest["design_id"], "sha256": sha(output), "phase": args.phase,
                      "rows": len(rows), "figures_created": 0, "model_calls": 0}))


if __name__ == "__main__":
    main()
