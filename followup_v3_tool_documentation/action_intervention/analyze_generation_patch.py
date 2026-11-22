"""Recount free generations, audit every control and plot conditional calls."""
from collections import Counter
from hashlib import sha256
from itertools import product
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
CELLS=list(product((False,True),repeat=2))
CONDITIONS=("baseline","identity","block_service","block_office","embed_service","embed_office")
OUT=HERE/"run_01"

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def parse(r):
    text=r["text"].strip()
    if text=="NO_CALL":
        return {"attempt":False,"abstain":True,"exact":False,"other":False}
    # This independent decoder knows both accepted response formats. If either
    # changes, the original scoring and the response have to be inspected.
    try:
        obj=json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj,dict) and "name" in obj:
        return {"attempt":True, "abstain":False, "other":False,
                "exact":obj=={"name":r["family"],"arguments":{"district":r["district"]}}}
    return None

def plot(table):
    fig,ax=plt.subplots(figsize=(7.7,3.65),layout="constrained")
    cond=("baseline","block_office","embed_office","block_service","embed_service")
    data=[[table[c,s,o]["attempts"] for s,o in CELLS] for c in cond]
    cm=LinearSegmentedColormap.from_list("scope",["#f1f5f2","#9ec5b9","#12676c"])
    ax.imshow(data,cmap=cm,vmin=0,vmax=32,aspect="auto")
    for i,row in enumerate(data):
        for j,value in enumerate(row):
            ax.text(j,i,str(value),ha="center",va="center",weight="bold",fontsize=12,color="white" if value>=20 else "#203436")
    ax.set_yticks(range(len(cond)),["Unmodified","Office state, block 14","Office input embedding","Service state, block 14","Service input embedding"])
    ax.set_xticks(range(4),["Neither","Office only","Service only","Both"])
    ax.set_xlabel("District field matching the request in the original prompt")
    ax.tick_params(length=0)
    ax.axvline(1.5,color="white",lw=3)
    ax.spines[:].set_visible(False)
    fig.savefig(HERE/"calls_after_patch.png",dpi=200,bbox_inches="tight",pad_inches=.1)
    plt.close(fig)

def main():
    meta=json.loads((OUT/"run.json").read_text())
    assert meta["completed_utc"] and meta["raw_sha256"]==digest(OUT/"raw.jsonl")
    for f,d in meta["sources"].items(): assert digest(ROOT/f)==d
    raw=[json.loads(x) for x in (OUT/"raw.jsonl").read_text().splitlines()]
    originals=[json.loads(x) for x in (ROOT/"actions_01/raw.jsonl").read_text().splitlines()]
    archive={(r["family"],r["district"],r["service_match"],r["office_match"]):r for r in originals if r["mode"]=="text"}
    records={}
    unparsed=[]
    for r in raw:
        key=(r["condition"],r["family"],r["district"],r["service_match"],r["office_match"])
        assert key not in records and key[1:] in archive
        records[key]=r
        independent=parse(r)
        if independent is None:
            unparsed.append({"key":key,"text":r["text"],"score":r["score"]})
        else:
            assert independent==r["score"]
        assert r["finish"]=="stop" if r["generated_ids"][-1] in [151645,151643] else r["finish"]=="length"
    assert len(records)==768 and set(records)=={(c,*k) for c in CONDITIONS for k in archive}
    mismatch=[]
    changes=Counter()
    tables={}
    for c in CONDITIONS:
        for s,o in CELLS:
            rows=[r for r in raw if (r["condition"],r["service_match"],r["office_match"])==(c,s,o)]
            assert len(rows)==32
            tally=Counter()
            for r in rows:
                k=(r["family"],r["district"],s,o)
                a=archive[k]
                b=records["baseline",*k]
                if c in ("baseline","identity","embed_service","embed_office"):
                    if c=="baseline":
                        ref=a
                    elif c=="identity":
                        ref=b
                    else:
                        ns,no=r["donor"]
                        ref=records["baseline",r["family"],r["district"],ns,no]
                        expected=(not s,o) if c=="embed_service" else (s,not o)
                        assert (ns,no)==expected
                    for field in ("generated_ids","text","score","finish"):
                        if ref[field]!=r[field]: mismatch.append(dict(condition=c,recipient=k,field=field))
                if c.startswith("block_"):
                    expected=(not s,o) if c=="block_service" else (s,not o)
                    assert tuple(r["donor"])==expected
                    donor=archive[r["family"],r["district"],*expected]
                    diffs=[i for i,(x,y) in enumerate(zip(a["input_ids"],donor["input_ids"])) if x!=y]
                    assert len(a["input_ids"])==len(donor["input_ids"]) and diffs==[r["site"]]
                for label,flag in {"attempts":r["score"]["attempt"],"exact":r["score"]["exact"],
                    "abstain":r["score"]["abstain"],"other":r["score"]["other"],"cap":r["finish"]=="length",
                    "attempt_gained":r["score"]["attempt"] and not b["score"]["attempt"],
                    "attempt_lost":b["score"]["attempt"] and not r["score"]["attempt"],
                    "exact_gained":r["score"]["exact"] and not b["score"]["exact"],
                    "exact_lost":b["score"]["exact"] and not r["score"]["exact"]}.items():
                    tally[label]+=flag
                if r["generated_ids"]!=b["generated_ids"]:changes[c,"changed_sequence"]+=1
                if r["first_token_top_two_logits"]!=b["first_token_top_two_logits"]:changes[c,"changed_logits"]+=1
            tables[c,s,o]=dict(condition=c,service_match=s,office_match=o,n=32,**tally)
    report={"raw_records":len(records),"controls":{"archive_baseline":128,"identity_to_recipient":128,
              "service_embedding_to_donor":128,"office_embedding_to_donor":128,"mismatches":mismatch},
            "independent_parser_unclassified":unparsed,
            "complete_sequences_changed_against_recipient":{c:changes[c,"changed_sequence"] for c in CONDITIONS},
            "first_token_top_two_logits_changed_against_recipient":{c:changes[c,"changed_logits"] for c in CONDITIONS},
            "seconds":meta["elapsed_seconds"],"raw_sha256":meta["raw_sha256"]}
    with (HERE/"cells.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(next(iter(tables.values()))))
        writer.writeheader();writer.writerows(tables.values())
    (HERE/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    plot(tables)
    print(json.dumps(report,indent=2))
    print('counts',[(c,[tables[c,s,o]["attempts"] for s,o in CELLS]) for c in CONDITIONS])

if __name__=="__main__":
    main()
