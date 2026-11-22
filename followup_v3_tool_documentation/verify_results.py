"""Independent recount using the four-cell means, full strings and raw score files."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean

ROOT = Path(__file__).resolve().parent
def lines(path):
    return [json.loads(x) for x in (ROOT/path).read_text().splitlines()]
def checkclose(a,b):
    assert abs(a-b)<1e-8, [a,b]
j = lines("run_01/raw.jsonl")
raw = lines("actions_01/raw.jsonl")
results = json.loads((ROOT/"results.json").read_text())
assert len(j)==512 and len(raw)==256
assert not any(r["scores"]["baseline"]["margin"] == 0 for r in j)
means = defaultdict(list)
patches = defaultdict(list)
for r in j:
    s,o = r["service_match"],r["office_match"]
    sign = 1 if r["positive_letter"]=="A" else -1
    def margin(name):
        m = r["scores"][name]
        return sign*(m["logit_a"]-m["logit_b"])
    b = margin("baseline")
    means[s,o].append(b)
    patches["service"].append((b-margin("patch_service"))*(1 if s else -1))
    patches["office"].append((b-margin("patch_office"))*(1 if o else -1))
assert set(map(len, means.values())) == {128}
service = mean([mean(means[1,1])-mean(means[0,1]), mean(means[1,0])-mean(means[0,0])])
office = mean([mean(means[1,1])-mean(means[1,0]), mean(means[0,1])-mean(means[0,0])])
checkclose(service-office,results["means"]["primary"])
checkclose(mean(patches["service"])-mean(patches["office"]),results["means"]["mid_primary"])
forms = Counter()
unclassified = Counter()
for r in raw:
    text = r["text"].strip()
    stripped = text
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)[:-3].strip()
    if stripped.startswith("<tool_call>") and stripped.endswith("</tool_call>"):
        stripped = stripped[len("<tool_call>"):-len("</tool_call>")].strip()
    try:
        obj = json.loads(stripped)
    except ValueError:
        obj = None
    valid = isinstance(obj,dict) and set(obj)=={"name","arguments"} and obj["name"]==r["family"] and obj["arguments"]=={"district":r["district"]}
    assert valid == r["score"]["exact"]
    normalized = "".join(x for x in text if not x.isspace() and x!="_").casefold()
    assert (normalized == "nocall") == r["score"]["abstain"]
    assert sum([r["score"]["attempt"],r["score"]["abstain"],r["score"]["other"]]) == 1
    assert (r["finish"] == "stop") == (r["generated_ids"][-1] in [151645,151643])
    if r["score"]["attempt"] and not valid:
        unclassified[text]+=1
    if r["score"]["other"]:
        unclassified[text]+=1
    if valid:
        forms["exact JSON or tagged JSON"]+=1
    elif r["score"]["abstain"]:
        forms["literal abstention"]+=1
    else:
        forms["for manual review"]+=1
counts = Counter()
for r in raw:
    group = (r["mode"],r["service_match"],r["office_match"])
    counts[group,"n"]+=1
    for k in ["attempt","exact","abstain","other"]:
        counts[group,k]+=r["score"][k]
for row in results["actions_by_cell"]:
    group = (row["mode"],row["service_match"],row["office_match"])
    for key in ["n","exact","abstain","other"]:
        assert row[key] == counts[group,key]
    assert row["free_attempt"] == counts[group,"attempt"]
out={"judgment_rows_recounted":len(j),"action_rows_recounted":len(raw),"baseline_contrast_recomputed_from_cell_means":service-office,
     "patch_contrast_recomputed_from_logits":mean(patches["service"])-mean(patches["office"]),
     "generated_forms":dict(forms),"other_or_inexact_strings":[{"text":t,"count":n} for t,n in unclassified.most_common()]}
(ROOT/"independent_checks.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({k:v for k,v in out.items() if k!="other_or_inexact_strings"},indent=2))
