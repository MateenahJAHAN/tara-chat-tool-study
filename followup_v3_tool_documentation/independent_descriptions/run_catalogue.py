"""Run the frozen external-catalogue city crossover without invoking any functions."""
from pathlib import Path
import sys
import json
import torch
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
from run import ROOT,MODELS,sha,now

instruction='Use one listed function if it directly covers the user request. Return one JSON object with fields "name" and "arguments". If no listed function can fulfill the request, reply NO_CALL. Do not execute anything. The documentation office is where the catalogue was written.\n'
def main(label):
    manifest=json.loads((ROOT/'catalogue_freeze.json').read_text())
    for p,h in manifest['files'].items():assert sha(ROOT/p)==h,p
    name,revision,weight=MODELS[label]
    path=Path.home()/'.cache/huggingface/hub'/('models--'+name)/'snapshots'/revision
    assert sha(path/'model.safetensors')==weight
    out=ROOT/('catalogue_'+label)
    out.mkdir(exist_ok=False)
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float32,attn_implementation='eager').to('mps').eval()
    eos=model.generation_config.eos_token_id
    eoses=[eos] if isinstance(eos,int) else list(eos)
    cfg=GenerationConfig(do_sample=False,max_new_tokens=96,eos_token_id=eoses,pad_token_id=tok.pad_token_id,use_cache=True)
    meta={'start':now(),'revision':revision,'weights_sha256':weight,'freeze_sha256':sha(ROOT/'catalogue_freeze.json'),'batch_size':4,'device':'mps','dtype':'float32','attention':'eager','generation':cfg.to_diff_dict(),'torch':torch.__version__,'transformers':transformers.__version__}
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    cards=[]
    for line in (ROOT/'catalogue_bank.jsonl').read_text().splitlines():
        item=json.loads(line)
        system=instruction+'Documentation office: '+item['office']+'.\nFunction catalogue (JSON):\n'+json.dumps(item['tools'],separators=(',',':'),ensure_ascii=False)
        prompt=tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':item['question']}],tokenize=False,add_generation_prompt=True)
        cards.append(dict(id=item['id'],prompt=prompt,input_ids=tok.encode(prompt,add_special_tokens=False)))
    assert len(cards)==len({c['prompt'] for c in cards})
    cards.sort(key=lambda c:len(c['input_ids']))
    with (out/'raw.jsonl').open('x') as f,torch.inference_mode():
        for i in range(0,len(cards),4):
            batch=cards[i:i+4];w=max(len(c['input_ids']) for c in batch)
            ids=torch.full((len(batch),w),tok.pad_token_id,dtype=torch.long,device='mps')
            mask=torch.zeros_like(ids)
            for j,c in enumerate(batch):
                n=len(c['input_ids']);ids[j,-n:]=torch.tensor(c['input_ids'],device='mps');mask[j,-n:]=1
            outputs=model.generate(input_ids=ids,attention_mask=mask,generation_config=cfg)[:,w:].cpu().tolist()
            for c,seq in zip(batch,outputs):
                last=next((j for j,t in enumerate(seq) if t in eoses),None)
                used=seq if last is None else seq[:last+1]
                c.update(generated_ids=used,finish='length' if last is None else 'stop',text=tok.decode(used if last is None else used[:-1],skip_special_tokens=False))
                f.write(json.dumps(c,separators=(',',':'))+'\n')
            f.flush()
            print(label,i+len(batch),'/',len(cards),flush=True)
    meta.update(end=now(),raw_sha256=sha(out/'raw.jsonl'),n=len(cards))
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main(sys.argv[1])
