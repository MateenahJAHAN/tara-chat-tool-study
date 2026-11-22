"""Verify and analyze every prompt and intervention from the two complete runs."""
import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import csv
import math
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FAMILIES = [f[0] for f in json.loads((ROOT / "bank_spec.json").read_text())]
DISTRICTS = ["East","West","North","South"]
MODES = ["text","native"]
KEY = ["family","district","mode","positive_letter","service_match","office_match"]
BASE = ["family","district","mode","service_match","office_match"]


def groupkey(row, fields):
    return tuple(row[k] for k in fields)


def read_run(folder, keys, expected):
    path = folder
    meta = json.loads((path/"run.json").read_text())
    assert meta["completed_utc"] and meta["raw_sha256"] == sha256((path/"raw.jsonl").read_bytes()).hexdigest()
    for p,h in meta["sources_sha256"].items():
        assert sha256((ROOT/p).read_bytes()).hexdigest() == h, p
    rows = [json.loads(x) for x in (path/"raw.jsonl").read_text().splitlines()]
    by_key = {groupkey(r, keys): r for r in rows}
    assert len(rows) == len(by_key) and set(by_key) == expected
    return rows, by_key


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summaries(rows):
    return {k: mean(r[k] for r in rows) for k in ["service", "office", "primary", "mid_service", "mid_office", "mid_primary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, default=ROOT/"run_01")
    parser.add_argument("--actions", type=Path, default=ROOT/"actions_01")
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    vals = [False, True]
    expected = set(product(FAMILIES,DISTRICTS,MODES,"AB",vals,vals))
    rows, index = read_run(args.judgments, KEY, expected)
    controls = {"self": [], "embed_service": [], "embed_office": []}
    measurements = []
    for r in rows:
        key = groupkey(r, KEY)
        b = r["scores"]["baseline"]["margin"]
        for score in r["scores"].values():
            assert all(math.isfinite(score[k]) for k in ["logit_a","logit_b","margin","ab_probability"])
            assert 0 <= score["ab_probability"] <= 1.00001
        out = {k:r[k] for k in KEY}
        for field, idx in [("service",4),("office",5)]:
            donor_key = list(key)
            donor_key[idx] = not donor_key[idx]
            donor = index[tuple(donor_key)]
            differences = [i for i,(x,y) in enumerate(zip(r["tokens"],donor["tokens"])) if x != y]
            assert len(r["tokens"]) == len(donor["tokens"]) and differences == [r[field+"_token_index"]]
            donor_margin = donor["scores"]["baseline"]["margin"]
            orientation = -1 if key[idx] else 1
            out[field] = (donor_margin-b)*orientation
            out["mid_"+field] = (r["scores"]["patch_"+field]["margin"]-b)*orientation
            if "embed_"+field in r["scores"]:
                controls["embed_"+field].append(abs(r["scores"]["embed_"+field]["margin"]-donor_margin))
        if "self" in r["scores"]:
            controls["self"].append(abs(r["scores"]["self"]["margin"]-b))
        out["primary"] = out["service"]-out["office"]
        out["mid_primary"] = out["mid_service"]-out["mid_office"]
        measurements.append(out)
    assert all(len(v) == 32 and max(v)<=0.05 for v in controls.values())
    ag = {}
    fams = [{ "family":f, **summaries([r for r in measurements if r["family"]==f])} for f in FAMILIES]
    write_csv(args.out/"family_effects.csv", fams)
    write_csv(args.out/"prompt_effects.csv", measurements)
    for key in ["primary","mid_primary"]:
        leave = [mean([r[key] for r in fams if r["family"]!=f]) for f in FAMILIES]
        ag[key] = {"mean":mean(r[key] for r in fams), "family_range":[min(r[key] for r in fams),max(r[key] for r in fams)],
                   "leave_one_family_out_range":[min(leave),max(leave)]}
    def diag(rs):
        assert all(r["scores"]["baseline"]["margin"] != 0 for r in rs)
        return {"n":len(rs), "correct_ab":sum((r["scores"]["baseline"]["margin"]>0)==r["service_match"] for r in rs),
                "argmax_is_letter":sum(r["scores"]["baseline"]["top_token"] in ("A","B") for r in rs),
                "argmax_correct_letter":sum(r["scores"]["baseline"]["top_token"] == (r["positive_letter"] if r["service_match"] else ("B" if r["positive_letter"]=="A" else "A")) for r in rs),
                "mean_ab_probability":mean(r["scores"]["baseline"]["ab_probability"] for r in rs),
                "median_ab_probability":median(r["scores"]["baseline"]["ab_probability"] for r in rs)}
    res = {"planned_contrasts":ag, "means":summaries(measurements), "behavior":diag(rows),
           "qc_max_absolute_margin_error":{k:max(v) for k,v in controls.items()}, "qc_per_type":32,
           "by_interface":{m:{**diag([r for r in rows if r["mode"]==m]),**summaries([r for r in measurements if r["mode"]==m])} for m in MODES}}
    res["by_label_and_office_order"] = [{"positive_letter":lab,"office_first":first,**diag([r for r in rows if r["positive_letter"]==lab and r["office_first"]==first])} for lab,first in product("AB",vals)]
    expected_actions = set(product(FAMILIES,DISTRICTS,MODES,vals,vals))
    actions, act_index = read_run(args.actions, BASE, expected_actions)
    action_report = []
    cross = []
    for r in actions:
        a,b = [index[(r["family"],r["district"],r["mode"],lab,r["service_match"],r["office_match"])]["scores"]["baseline"]["margin"] for lab in "AB"]
        assert a+b != 0
        cross.append({**{k:r[k] for k in BASE},"judged_covers":a+b>0,"free_attempt":r["score"]["attempt"],
                      "exact":r["score"]["exact"],"abstain":r["score"]["abstain"],"other":r["score"]["other"],"capped":r["finish"]=="length"})
    for mode in MODES:
        for s,o in product(vals,vals):
            subset = [r for r in cross if r["mode"]==mode and r["service_match"]==s and r["office_match"]==o]
            assert len(subset)==32
            action_report.append({"mode":mode,"service_match":s,"office_match":o,"n":len(subset),**{k:sum(r[k] for r in subset) for k in ["judged_covers","free_attempt","exact","abstain","other","capped"]},
                "attempt_despite_judged_inapplicable":sum(r["free_attempt"] and not r["judged_covers"] for r in subset),
                "judgment_action_agree":sum(r["free_attempt"]==r["judged_covers"] for r in subset)})
    write_csv(args.out/"actions_by_cell.csv",action_report)
    write_csv(args.out/"judgment_and_action.csv",cross)
    res["actions_by_cell"] = action_report
    res["n"]={"judgments":len(rows),"block_14_patches":2*len(rows),"numerical_controls":sum(len(v) for v in controls.values()),"greedy_responses":len(actions)}
    (args.out/"results.json").write_text(json.dumps(res,indent=2)+"\n")
    (args.out/"other_responses.json").write_text(json.dumps([r for r in actions if r["score"]["other"]],indent=2)+"\n")
    fig, axes = plt.subplots(1,3,figsize=(11.8,3.9),gridspec_kw={"width_ratios":[1,1,1.5]})
    for ax,x,y,title in zip(axes[:2], ["service","mid_service"],["office","mid_office"],["A   Change the card","B   Patch block 14"]):
        for j,mode in enumerate(MODES):
            color = ["#b65b42","#075e68"][j]
            coords = [v+(j-.5)*.08 for v in range(2)]
            m_fams = [summaries([r for r in measurements if r["mode"]==mode and r["family"]==f]) for f in FAMILIES]
            for r in m_fams:
                ax.plot(coords,[r[x],r[y]],color=color,alpha=.25,marker="o",ms=2,lw=.65)
            ax.plot(coords,[mean(r[x] for r in m_fams),mean(r[y] for r in m_fams)],color=color,lw=2,
                    marker="o",ms=4,label=["Inline","Native"][j])
        ax.axhline(0,color="#777777",lw=.8,ls="--")
        ax.set(xlim=(-.25,1.25),xticks=[0,1],xticklabels=["Service\ndistrict","Office\ndistrict"])
        ax.set_title(title,loc="left",fontsize=11,fontweight="bold")
        ax.set_ylabel("Change in applicability log-odds",fontsize=9)
    ax = axes[2]
    for j,m in enumerate(MODES):
        c = sorted((r for r in action_report if r["mode"]==m),key=lambda r:(r["service_match"],r["office_match"]))
        pos = [v+(j-.5)*.32 for v in range(4)]
        bars = ax.bar(pos,[r["free_attempt"]/r["n"] for r in c],width=.31,color=["#b65b42","#075e68"][j],label=["Inline","Native"][j])
        ax.bar_label(bars,labels=[str(r["free_attempt"]) for r in c],fontsize=8,padding=2)
    ax.set(xlim=(-.55,3.55),ylim=(0,1.16),xticks=[0,1,2,3],xticklabels=["Neither","Office\nonly","Service\nonly","Both"],ylabel="Fraction attempting a call")
    ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(.48,-.15),ncol=2,fontsize=8)
    ax.set_title("C   Generate a response",loc="left",fontsize=11,fontweight="bold")
    for ax in axes:
        ax.spines[["right","top"]].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.tight_layout(w_pad=2)
    fig.savefig(args.out/"figure.png",dpi=200,bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(res,indent=2))


if __name__ == "__main__":
    main()
