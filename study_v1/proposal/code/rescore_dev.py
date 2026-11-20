"""Explicit DEV-discovered V2 measurement amendment; no inference or raw overwrite.

Re-score every original DEV output identically. Original file/line hashes and all
generation fields are preserved; the primary/strict-format endpoints must agree.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import parser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", required=True)
    args = ap.parse_args()
    root = Path(args.study_root).resolve()
    cfg = common.load_config()
    original = root / "runs/dev"
    output = original / cfg["dev_scores_subdirectory"]
    if output.exists():
        raise RuntimeError("STOP: derived score version already exists; no overwrite")
    issue = common.read_json(root / "DEV_SCORING_ISSUE.json")
    amendment = common.read_json(root / "MEASUREMENT_AMENDMENT_V2.json")
    if common.sha256_file(common.CODE_DIR / "parser.py") != amendment["candidate_parser_sha256"]:
        raise RuntimeError("STOP: approved candidate scorer mismatch")
    items = {x["uid"]: x for x in common.read_json(root / "manifests/split_manifest.json")["items"] if x["split"] == "DEV"}
    output.mkdir(parents=True, exist_ok=False)
    invariant = ["call_attempt", "false_call", "broad_false_call", "strict_abstention", "operational_abstention",
                 "format_compliant", "invalid_or_empty", "unknown_name", "empty_output", "truncated", "errored", "multiple_calls"]
    record = {"phase": "DEV_ONLY_DERIVED_RESCORING", "created_at_utc": common.now_utc(),
              "score_version": cfg["measurement_version"], "scorer_sha256": amendment["candidate_parser_sha256"],
              "source_script_sha256": common.sha256_file(Path(__file__)), "raw_outputs_overwritten": False,
              "inference_repeated": False, "heldout_read": False, "files": {}, "checked_invariant_fields": invariant}
    total = 0
    for condition in common.CONDITIONS:
        filename = f"generations_baseline_{condition}.jsonl"
        source = original / filename
        source_sha = common.sha256_file(source)
        if issue["original_raw_scored_file_sha256"][filename] != source_sha:
            raise RuntimeError("STOP: original raw/scored output changed")
        lines = source.read_text().splitlines(keepends=True)
        records, changes = [], []
        for index, line in enumerate(lines, 1):
            old = json.loads(line)
            item = items[old["uid"]]
            new = copy.deepcopy(old)
            new["score"] = parser.score(old["generated_text"], old["stop_reason"], item["tool"], item["label"], item["possible_answer"])
            for key in invariant:
                if old["score"][key] != new["score"][key]:
                    raise RuntimeError("STOP: invariant endpoint changed: " + old["uid"] + " " + key)
            if any(old[key] != new[key] for key in old if key != "score"):
                raise RuntimeError("STOP: generation fields changed")
            new["score_version"] = cfg["measurement_version"]
            new["score_derivation"] = {"original_file_sha256": source_sha,
                "original_line_sha256": common.sha256_text(line), "original_line_number": index,
                "scorer_sha256": amendment["candidate_parser_sha256"], "inference_repeated": False}
            changed = [key for key in old["score"] if old["score"][key] != new["score"].get(key)]
            if changed:
                changes.append({"uid": old["uid"], "changed_fields": changed})
            common.append_jsonl(output / filename, new)
            records.append(new)
        if len(records) != len(items) or {x["uid"] for x in records} != set(items):
            raise RuntimeError("STOP: derived bank differs from frozen DEV bank")
        total += len(records)
        record["files"][filename] = {"original_sha256": source_sha, "derived_sha256": common.sha256_file(output / filename),
                                     "records": len(records), "semantically_changed_records": changes}
    if total != 64:
        raise RuntimeError("STOP: expected all64 development outputs")
    record.update(total_records=total, invariants_all_pass=True, status="PASS")
    common.write_json(output / "RESCORING_MANIFEST.json", record)
    print(json.dumps({"status": "PASS", "records": total, "inference_repeated": False,
                      "primary_and_strict_format_unchanged": True}))


if __name__ == "__main__":
    main()
