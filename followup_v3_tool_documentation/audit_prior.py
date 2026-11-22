"""Verify the cited frozen archive and recount the two public studies."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("repo", type=Path)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
v2 = a.repo / "followup_v2_tool_applicability"
sys.path.insert(0, str(v2.resolve()))
from pilot import score

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def readlines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]

files = json.loads((a.repo / "COMPLETED_PUBLIC_MANIFEST.json").read_text())["files"]
fail = [k for k,v in files.items() if digest(a.repo / k) != v]
res = {"commit": subprocess.check_output(["git","-C",str(a.repo),"rev-parse","HEAD"], text=True).strip(),
       "manifest_files": len(files), "manifest_mismatches": fail}
assert not fail
counts = {}
for mode in ["NATIVE_TOOLS","TEXT_CATALOGUE"]:
    records = readlines(a.repo / f"study_v1/runs/heldout/generations_baseline_{mode}.jsonl")
    neg = [r for r in records if r["source"] == "BFCL_v4_live_irrelevance.json"]
    assert len(records) == 64 and len(neg) == 32 and len({r["id"] for r in records}) == 64
    counts[mode] = {"all":len(records), "out_of_scope":len(neg), "attempts":sum(r["score"]["call_attempt"] for r in neg)}
res["bfcl_from_saved_scores"] = counts
v = json.loads((v2 / "verification.json").read_text())
assert digest(v2/"bank.json") == v["bank_sha256"] and digest(v2/"PROTOCOL.md") == v["protocol_sha256"]
assert all(digest(v2/k) == val for k,val in v["sha256"].items())
c = {}
for k in ["results_qwen25/qwen2.5-1.5b_raw.jsonl", "results/qwen3.5-4b_raw.jsonl"]:
    rows = readlines(v2/k)
    assert len(rows) == 84
    for r in rows:
        assert score(r["response"]["choices"][0]["message"],r["mode"],r["expected"]) == r["score"]
    for mode in ["text","native"]:
        neg = [r for r in rows if r["mode"] == mode and r["phase"] != "control" and r["expected"] is None]
        assert len(neg) == 20
        c[f"{rows[0]['model']} {mode}"] = sum(r["score"]["call_attempt"] for r in neg)
res["matched_scores_reparsed_including_controls"] = 168
res["matched_false_calls_out_of_20"] = c
a.out.write_text(json.dumps(res, indent=2) + "\n")
