import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import sys
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

ROOT=Path(__file__).resolve().parent
MODELS={
    'qwen':('Qwen--Qwen2.5-1.5B-Instruct','989aa7980e4cf806f80c7fef2b1adb7bc71aa306','dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'),
    'smol':('HuggingFaceTB--SmolLM2-1.7B-Instruct','31b70e2e869a7173562077fd711b654946d38674','f55217be716b6a997b97b9d8d7eb6fad02e00858f5010ec24f64603c3a98a0e8'),
}
INSTRUCTION='Use the listed function if it directly covers the user request. Return one JSON object with fields "name" and "arguments". If the listed function cannot fulfill the request, reply NO_CALL. Do not execute anything.\nFunction catalogue (JSON):\n'

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def now():return datetime.now(timezone.utc).isoformat()

def main(label):
    name,revision,weight=MODELS[label]
    model_path=Path.home()/'.cache/huggingface/hub'/('models--'+name)/'snapshots'/revision
    assert sha(model_path/'model.safetensors')==weight
    freeze=json.loads((ROOT/'freeze.json').read_text())
    for p,h in freeze['files'].items(): assert sha(ROOT/p)==h,p
    out=ROOT/label
    out.mkdir(exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=torch.float32,attn_implementation='eager').to('mps').eval()
    eos=model.generation_config.eos_token_id
    ends=[eos] if isinstance(eos,int) else list(eos)
    gen=GenerationConfig(do_sample=False,max_new_tokens=96,use_cache=True,pad_token_id=tokenizer.pad_token_id,eos_token_id=ends)
    meta={'start':now(),'model':name.replace('--','/'),'revision':revision,'weights_sha256':weight,'freeze_sha256':sha(ROOT/'freeze.json'),'torch':torch.__version__,'transformers':transformers.__version__,'device':'mps','dtype':'float32','attention':'eager','batch_size':12,'generation':gen.to_diff_dict(),'tokenizer_files':{f:sha(model_path/f) for f in ('tokenizer.json','tokenizer_config.json','config.json')}}
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    rows=[]
    for line in (ROOT/'bank.jsonl').read_text().splitlines():
        r=json.loads(line)
        tool=r['source_tool'].copy()
        if r['naming']=='masked':tool['name']='lookup_function'
        system=INSTRUCTION+json.dumps(tool,ensure_ascii=False,separators=(',',':'))
        chat=tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':r['question']}],tokenize=False,add_generation_prompt=True)
        tokens=tokenizer.encode(chat,add_special_tokens=False)
        rows.append({'id':r['id'],'prompt':chat,'input_ids':tokens})
    assert len({r['prompt'] for r in rows})==len(rows)
    rows.sort(key=lambda r:len(r['input_ids']))
    with (out/'raw.jsonl').open('x') as f,torch.inference_mode():
        for i in range(0,len(rows),12):
            batch=rows[i:i+12]
            width=max(len(r['input_ids']) for r in batch)
            ids=torch.full((len(batch),width),tokenizer.pad_token_id,dtype=torch.long,device='mps')
            mask=torch.zeros_like(ids)
            for j,r in enumerate(batch):
                n=len(r['input_ids'])
                ids[j,-n:]=torch.tensor(r['input_ids'],device='mps')
                mask[j,-n:]=1
            seqs=model.generate(input_ids=ids,attention_mask=mask,generation_config=gen)[:,width:].cpu().tolist()
            for r,seq in zip(batch,seqs):
                end=next((k for k,t in enumerate(seq) if t in ends),None)
                used=seq if end is None else seq[:end+1]
                r.update(text=tokenizer.decode(used if end is None else used[:-1],skip_special_tokens=False),generated_ids=used,finish='length' if end is None else 'stop')
                f.write(json.dumps(r,separators=(',',':'))+'\n')
            f.flush()
            print(label,i+len(batch),'/',len(rows),flush=True)
    meta.update(end=now(),raw_sha256=sha(out/'raw.jsonl'),n=len(rows))
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')

if __name__=='__main__':main(sys.argv[1])
