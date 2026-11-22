"""Generate the full free-response tool scope bank under six fixed interventions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

LOCAL = Path(__file__).resolve().parent
ROOT = LOCAL.parent
sys.path.insert(0, str(ROOT))
from run import CELLS, DISTRICTS, REVISION, WEIGHTS, diff_position
from run_actions import classify

CONDITIONS = ("baseline", "identity", "block_service", "block_office", "embed_service", "embed_office")
REFERENCE = "bf7dd6d046eba2b4bd61191b26d3b6030430bd4ab1ba7230798927bc5f511dbe"

def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()

def utc():
    return datetime.now(timezone.utc).isoformat()

def make_hook(edits, counter):
    def hook(_module, _args, hidden):
        if hidden.shape[1] == 1:
            return hidden
        counter.append(tuple(hidden.shape))
        patched = hidden.clone()
        for recipient, donor, index in edits:
            patched[recipient,index] = hidden[donor,index]
        return patched
    return hook

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    one = ROOT/"actions_01/raw.jsonl"
    assert digest(one)==REFERENCE and digest(args.model/"model.safetensors")==WEIGHTS
    records=[json.loads(line) for line in one.read_text().splitlines()]
    archived={(r["family"],r["district"],r["service_match"],r["office_match"]):r for r in records if r["mode"]=="text"}
    assert len(archived)==128
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float32,attn_implementation="eager",local_files_only=True).to("mps").eval()
    assert len(model.model.layers)==28
    conf=GenerationConfig(do_sample=False,max_new_tokens=72,use_cache=True,eos_token_id=[151645,151643],pad_token_id=151643)
    args.out.mkdir(exist_ok=False)
    manifest={"started_utc":utc(), "device":"mps","weights_sha256":WEIGHTS,"revision":REVISION,"dtype":"float32",
              "attention":"eager","versions":{"python":platform.python_version(),"torch":str(torch.__version__),"transformers":transformers.__version__},
              "generation":conf.to_diff_dict(),"batch":24,"archive_sha256":REFERENCE,
              "sources":{str(f.relative_to(ROOT)):digest(f) for f in [LOCAL/"PROTOCOL.md",Path(__file__),ROOT/"run.py",ROOT/"run_actions.py"]}}
    summary=args.out/"run.json"
    summary.write_text(json.dumps(manifest,indent=2)+"\n")
    start=time.monotonic()
    families=json.loads((ROOT/"bank_spec.json").read_text())
    with (args.out/"raw.jsonl").open("x") as out, torch.inference_mode():
        for family_spec in families:
            family=family_spec[0]
            for district, alternative in DISTRICTS:
                quartet=[archived[(family,district,s,o)] for s,o in CELLS]
                tokens=[r["input_ids"] for r in quartet]
                assert all(tok.encode(r["prompt"],add_special_tokens=False)==r["input_ids"] for r in quartet)
                embed_edits=[]
                block_edits=[]
                rows=[]
                for condition in CONDITIONS:
                    for i,(s,o) in enumerate(CELLS):
                        r=quartet[i]
                        dst=len(rows)
                        donor=pos=None
                        if condition!="baseline":
                            service = condition=="identity" or condition.endswith("service")
                            x=CELLS.index((not s,o) if service else (s,not o))
                            pos=diff_position(tokens[i],tokens[x])
                            matched=s if service else o
                            assert tok.decode([tokens[i][pos]]).strip()==(district if matched else alternative)
                            assert tok.decode([tokens[x][pos]]).strip()==(alternative if matched else district)
                            donor=i if condition=="identity" else x
                            edits=embed_edits if condition.startswith("embed") else block_edits
                            edits.append((dst,donor,pos))
                        rows.append(dict(condition=condition,service_match=s,office_match=o,family=family,district=district,
                                         office_first=r["office_first"],site=pos,donor=None if donor is None else list(CELLS[donor])))
                ids=torch.tensor(tokens*len(CONDITIONS),dtype=torch.long,device="mps")
                seen_embed, seen_block=[],[]
                handles=[model.model.embed_tokens.register_forward_hook(make_hook(embed_edits,seen_embed)),
                         model.model.layers[13].register_forward_hook(make_hook(block_edits,seen_block))]
                try:
                    res=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),generation_config=conf,
                                       return_dict_in_generate=True,output_scores=True)
                finally:
                    for h in handles:
                        h.remove()
                assert len(seen_embed)==len(seen_block)==1
                generated=res.sequences[:,ids.shape[1]:].cpu().tolist()
                vals,ids_top=res.scores[0].topk(2,dim=-1)
                for i,(row,raw_ids) in enumerate(zip(rows,generated)):
                    end=next((j for j,t in enumerate(raw_ids) if t in conf.eos_token_id),None)
                    used=raw_ids if end is None else raw_ids[:end+1]
                    text=tok.decode(used if end is None else used[:-1],skip_special_tokens=False)
                    row.update(text=text,generated_ids=used,finish="length" if end is None else "stop",score=classify(text,family,district),
                               first_token_top_two_ids=ids_top[i].cpu().tolist(),first_token_top_two_logits=vals[i].cpu().tolist())
                    out.write(json.dumps(row,separators=(",",":"))+"\n")
                out.flush()
            print(f"{family} complete ({time.monotonic()-start:.2f}s)",flush=True)
    manifest.update(completed_utc=utc(),elapsed_seconds=time.monotonic()-start,raw_sha256=digest(args.out/"raw.jsonl"))
    summary.write_text(json.dumps(manifest,indent=2)+"\n")

if __name__=="__main__":
    main()
