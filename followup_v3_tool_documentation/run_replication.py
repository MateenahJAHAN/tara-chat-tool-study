"""Apply the original 512 inline decision cards to a second model family."""
from argparse import ArgumentParser
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from run_actions import classify

ROOT = Path(__file__).resolve().parent
REVISION = "31b70e2e869a7173562077fd711b654946d38674"
WEIGHTS = "f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8"
BANK = "566a152abf7584126452bd2f7b6e35f68fa0df263a4d0049888e7decc5864c54"

def digest(file):
    with file.open("rb") as reader:
        return hashlib.file_digest(reader,"sha256").hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def main():
    ap = ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    source = ROOT/"transfer_01/raw.jsonl"
    assert digest(source) == BANK and digest(args.model/"model.safetensors") == WEIGHTS
    args.out.mkdir(exist_ok=False)
    tok = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,torch_dtype=torch.float32,attn_implementation="eager").to("mps").eval()
    eos = model.generation_config.eos_token_id
    eoses = [eos] if isinstance(eos,int) else list(eos)
    conf = GenerationConfig(do_sample=False,max_new_tokens=72,use_cache=True,pad_token_id=tok.pad_token_id,eos_token_id=eoses)
    manifest = {"start_utc":now(),"weights_sha256":WEIGHTS,"revision":REVISION,"source_sha256":BANK,
                "torch":torch.__version__,"transformers":transformers.__version__,"device":"mps","dtype":"float32","attention":"eager",
                "batch_size":12,"generation":conf.to_diff_dict(),"files":{f:digest(ROOT/f) for f in ["REPLICATION_PROTOCOL.md","run_replication.py","run_actions.py"]},
                "tokenizer_files":{f:digest(args.model/f) for f in ["config.json","tokenizer.json","tokenizer_config.json"]}}
    (args.out/"run.json").write_text(json.dumps(manifest,indent=2)+"\n")
    grid = []
    for line in source.read_text().splitlines():
        row = json.loads(line)
        match = re.fullmatch(r"<\|im_start\|>system\n(.*?)<\|im_end\|>\s*<\|im_start\|>user\n(.*?)<\|im_end\|>\s*<\|im_start\|>assistant\n",row["prompt"],flags=re.S)
        assert match
        system,user = match.groups()
        messages = [{"role":"system","content":system},{"role":"user","content":user}]
        prompt = tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        assert prompt.count(system)==prompt.count(user)==1
        ids = tok.encode(prompt,add_special_tokens=False)
        record = {k:row[k] for k in ("family","district","slot","vocabulary","office_first","service_match","office_match")}
        grid.append(dict(record,prompt=prompt,input_ids=ids))
    assert len({r["prompt"] for r in grid})==len(grid)==512
    grid.sort(key=lambda r:len(r["input_ids"]))
    with (args.out/"raw.jsonl").open("x") as out, torch.inference_mode():
        for start in range(0,len(grid),12):
            group = grid[start:start+12]
            width = max(len(r["input_ids"]) for r in group)
            ids = torch.full((len(group),width),tok.pad_token_id,dtype=torch.long,device="mps")
            mask = torch.zeros_like(ids)
            for i,r in enumerate(group):
                n=len(r["input_ids"])
                ids[i,-n:]=torch.tensor(r["input_ids"],device="mps")
                mask[i,-n:]=1
            generated = model.generate(input_ids=ids,attention_mask=mask,generation_config=conf)[:,width:].cpu().tolist()
            for r,seq in zip(group, generated):
                end=next((i for i,t in enumerate(seq) if t in eoses),None)
                used=seq if end is None else seq[:end+1]
                text=tok.decode(used if end is None else used[:-1],skip_special_tokens=False)
                r.update(text=text,generated_ids=used,finish="length" if end is None else "stop",score=classify(text,r["family"],r["district"]))
                out.write(json.dumps(r,separators=(",",":"))+"\n")
            out.flush()
            if start%120==0: print(start+len(group),"/512",flush=True)
    manifest.update(end_utc=now(),raw_sha256=digest(args.out/"raw.jsonl"))
    (args.out/"run.json").write_text(json.dumps(manifest,indent=2)+"\n")

if __name__=="__main__":
    main()
