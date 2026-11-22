"""Greedy tool responses for the prespecified order-by-vocabulary transfer grid."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from run import CELLS, DISTRICTS, HERE, WEIGHTS, REVISION, dump_line, file_digest
from run_actions import classify

NAMES = [("Cedar", "Maple"), ("Maple", "Cedar"), ("Birch", "Pine"), ("Pine", "Birch")]


def description(service, office, office_first):
    parts = [f"Service district: {service}.", f"Documentation office district: {office}."]
    return " ".join(reversed(parts) if office_first else parts)


def render(source, slot, new_names, office_first):
    target, alt = DISTRICTS[slot]
    target_new, alt_new = NAMES[slot] if new_names else DISTRICTS[slot]
    service, office = source["service_match"], source["office_match"]
    old_desc = description(target if service else alt, target if office else alt, source["office_first"])
    new_desc = description(target_new if service else alt_new, target_new if office else alt_new, office_first)
    old_query = f"in the {target} district."
    assert source["prompt"].count(old_desc) == source["prompt"].count(old_query) == 1
    return source["prompt"].replace(old_desc, new_desc).replace(old_query, f"in the {target_new} district.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source", type=Path, default=HERE/"actions_01/raw.jsonl")
    p.add_argument("--device", default="mps", choices=["cpu","mps"])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    assert file_digest(args.model/"model.safetensors") == WEIGHTS
    records = [json.loads(line) for line in args.source.read_text().splitlines()]
    original = {(r["family"],r["district"],r["service_match"],r["office_match"]):r for r in records if r["mode"]=="text"}
    assert len(original)==128
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager",
                                                local_files_only=True).to(args.device).eval()
    conf = GenerationConfig(do_sample=False, max_new_tokens=72, use_cache=True, eos_token_id=[151645,151643], pad_token_id=151643)
    stamp = lambda: datetime.now(timezone.utc).isoformat()
    meta={"started_utc":stamp(),"weights_sha256":WEIGHTS,"revision":REVISION,"device":args.device,"dtype":"float32","attention":"eager",
          "source_raw_sha256":file_digest(args.source), "generation":conf.to_diff_dict(),"code_plan_sha256":{f:file_digest(HERE/f) for f in ["run.py","run_actions.py","run_transfer.py","TRANSFER_PROTOCOL.md","bank_spec.json"]}}
    (args.out/"run.json").write_text(json.dumps(meta,indent=2)+"\n")
    grid=[]
    for family, *_ in json.loads((HERE/"bank_spec.json").read_text()):
        for slot, new, office_first in product(range(4),[False,True],[False,True]):
            group=[]
            for s,o in CELLS:
                r=original[family,DISTRICTS[slot][0],s,o]
                prompt=render(r,slot,new,office_first)
                ids=tok.encode(prompt,add_special_tokens=False)
                repeat = not new and r["office_first"]==office_first
                assert repeat == (prompt==r["prompt"])
                if repeat:
                    assert ids==r["input_ids"]
                group.append({"family":family,"district":(NAMES if new else DISTRICTS)[slot][0], "slot":slot,
                              "vocabulary":"tree" if new else "compass","office_first":office_first,
                              "service_match":s,"office_match":o,"prompt":prompt,"input_ids":ids,"repeated":repeat})
            grid.append(group)
    assert len({r["prompt"] for group in grid for r in group})==512
    assert sum(r["repeated"] for group in grid for r in group)==128
    with (args.out/"raw.jsonl").open("x") as fp, torch.inference_mode():
        for i,group in enumerate(grid):
            buckets=defaultdict(list)
            for r in group:
                buckets[len(r["input_ids"])].append(r)
            for length,rows in buckets.items():
                ids=torch.tensor([r["input_ids"] for r in rows],device=args.device)
                out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),generation_config=conf)[:,length:].cpu().tolist()
                for r,seq in zip(rows,out):
                    end=next((j for j,t in enumerate(seq) if t in conf.eos_token_id),None)
                    used=seq[:end+1] if end is not None else seq
                    text=tok.decode(used[:-1] if end is not None else used,skip_special_tokens=False)
                    r.update(generated_ids=used,finish="stop" if end is not None else "length",text=text,score=classify(text,r["family"],r["district"]))
                    dump_line(fp,r)
            fp.flush()
            if (i+1)%16==0:
                print(f"{(i+1)//16}/8 services complete",flush=True)
    meta.update(completed_utc=stamp(),raw_sha256=file_digest(args.out/"raw.jsonl"))
    (args.out/"run.json").write_text(json.dumps(meta,indent=2)+"\n")


if __name__=="__main__":
    main()
