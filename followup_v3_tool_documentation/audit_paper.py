"""Reconstruct both checkpoints' call and abstention outcomes from full text."""
from collections import Counter
from hashlib import sha256
from itertools import product
from pathlib import Path
import csv
import json
import re

ROOT = Path(__file__).resolve().parent
CELL = ("vocabulary", "office_first", "service_match", "office_match")
KEY = ("family", "slot") + CELL

def write_table(name, rows):
    with (ROOT/name).open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    checkpoints = {"Qwen2.5": "transfer_01", "SmolLM2": "replication_01"}
    results, paired, cells, per_tool = {}, [], [], []
    reference = {}
    for model, path in checkpoints.items():
        manifest = json.loads((ROOT/path/"run.json").read_text())
        raw = (ROOT/path/"raw.jsonl").read_bytes()
        assert sha256(raw).hexdigest() == manifest["raw_sha256"]
        for file, expected in manifest.get("files", manifest.get("code_plan_sha256", {})).items():
            assert sha256((ROOT/file).read_bytes()).hexdigest() == expected
        records = {}
        forms = Counter()
        for line in raw.splitlines():
            r = json.loads(line)
            key = tuple(r[k] for k in KEY)
            assert key not in records
            text = r["text"].strip()
            obj = None if text=="NO_CALL" else json.loads(text)
            wanted = {"district": r["district"]}
            if obj is None:
                form = "literal abstention"
            elif obj == {"name": r["family"], "arguments": wanted}:
                form = "exact JSON"
            else:
                assert obj == {"name": r["family"], "arguments": {"properties": wanted}}, text
                form = "district nested under properties"
            assert r["score"] == dict(attempt=obj is not None,exact=form=="exact JSON",abstain=obj is None,other=False)
            assert r["finish"] == "stop" and r["generated_ids"][-1] in manifest["generation"]["eos_token_id"]
            r["form"] = form
            if model == "Qwen2.5":
                reference[key] = r["prompt"]
            else:
                assert r["prompt"] == reference[key]
            records[key] = r
            forms[form] += 1
        assert len(records)==512
        assert set(records) == set(product({r["family"] for r in records.values()}, range(4), ("compass","tree"), (False,True), (False,True), (False,True)))
        for (v,o,s,d) in product(("compass","tree"), (False,True), (False,True), (False,True)):
            group = [r for r in records.values() if tuple(r[k] for k in CELL)==(v,o,s,d)]
            assert len(group)==32
            tally = {s:sum(r["score"][s] for r in group) for s in ("attempt","exact","abstain","other")}
            cells.append(dict(model=model,vocabulary=v,office_first=o,service_match=s,office_match=d,n=len(group),**tally))
            for family in sorted({r["family"] for r in group}):
                subset = [r for r in group if r["family"]==family]
                assert len(subset)==4
                per_tool.append(dict(model=model,vocabulary=v,office_first=o,service_match=s,office_match=d,family=family,
                                     n=len(subset),attempt=sum(r["score"]["attempt"] for r in subset),exact=sum(r["score"]["exact"] for r in subset)))
        for v,o,score in product(("compass","tree"),(False,True),("attempt","exact")):
            change = Counter()
            for (f,slot,v2,o2,s,d), r in records.items():
                if v==v2 and o==o2 and not s and not d:
                    alt=records[f,slot,v,o,False,True]
                    change[int(alt["score"][score])-int(r["score"][score])] += 1
            assert sum(change.values())==32
            paired.append(dict(model=model,vocabulary=v,office_first=o,endpoint=score,pairs=32,
                               positive=change[1],negative=change[-1],same=change[0],difference=(change[1]-change[-1])/32))
        results[model] = {"full_responses":len(records),"format_counts":dict(forms),"capped":0,"unclassified":0,
                          "in_scope_exact":sum(r["score"]["exact"] for r in records.values() if r["service_match"]),
                          "out_of_scope_exact":sum(r["score"]["exact"] for r in records.values() if not r["service_match"]),
                          "out_of_scope_attempts":sum(r["score"]["attempt"] for r in records.values() if not r["service_match"]),
                          "source_sha256":manifest["raw_sha256"]}
    for m in checkpoints:
        for endpoint in ("attempt","exact"):
            strata=[p for p in paired if p["model"]==m and p["endpoint"]==endpoint]
            results[m][endpoint+"_paired"] = {k:sum(p[k] for p in strata) for k in ("pairs","positive","negative","same")}
    results["matched_rendered_prompts"] = len(reference)
    write_table("paper_cells.csv",cells)
    write_table("paper_pairs.csv",paired)
    write_table("paper_tools.csv",per_tool)
    (ROOT/"paper_behavior_audit.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))

if __name__=="__main__":
    main()
