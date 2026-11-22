"""Recount transfer responses and paired switches directly from the raw text."""
import csv
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
def load(p): return [json.loads(x) for x in (HERE/p).read_text().splitlines()]
def digest(p): return sha256((HERE/p).read_bytes()).hexdigest()
def write(p, rows):
    with (HERE/p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def classify(r):
    # All current responses are exact JSON calls or literal abstentions.
    text=r['text'].strip()
    raw=re.sub(r'^\s*```(?:json)?\s*(.*?)\s*```\s*$',r'\1',text,flags=re.S)
    raw=re.sub(r'^\s*<tool_call>\s*(.*?)\s*</tool_call>\s*$',r'\1',raw,flags=re.S)
    try: obj=json.loads(raw)
    except ValueError: obj=None
    exact=obj=={'name':r['family'],'arguments':{'district':r['district']}}
    abst=re.sub(r'[\s_]','',text).upper()=='NOCALL'
    if exact: attempted=True
    elif abst: attempted=False
    else: attempted=r['score']['attempt']
    assert (exact,abst,attempted)==tuple(r['score'][k] for k in ['exact','abstain','attempt'])
    assert (r['finish']=='stop')==(r['generated_ids'][-1] in [151645,151643])
    return attempted

r=load('transfer_01/raw.jsonl'); original=load('actions_01/raw.jsonl'); meta=json.loads((HERE/'transfer_01/run.json').read_text())
assert meta['completed_utc'] and digest('transfer_01/raw.jsonl')==meta['raw_sha256']
assert digest('actions_01/raw.jsonl')==meta['source_raw_sha256']
assert all(digest(p)==v for p,v in meta['code_plan_sha256'].items())
orig={ (x['family'],x['district'],x['service_match'],x['office_match']):x for x in original if x['mode']=='text'}
vals=[False, True]; vocs=['compass','tree']; fs=[x[0] for x in json.loads((HERE/'bank_spec.json').read_text())]
fields=['family','slot','vocabulary','office_first','service_match','office_match']
index={tuple(x[k] for k in fields):x for x in r}
assert len(r)==len(index)==512 and set(index)==set(product(fs,range(4),vocs,vals,vals,vals))
assert len({x['prompt'] for x in r})==512
for x in r: x['a']=classify(x)
repeat=[x for x in r if x['repeated']]
assert len(repeat)==128
repeats={}
for key in ['prompt','input_ids','generated_ids','text','score','finish']:
    repeats[key]=sum(x[key]==orig[x['family'],x['district'],x['service_match'],x['office_match']][key] for x in repeat)
assert repeats['prompt']==repeats['input_ids']==128
cells=[]
for v,order,s,o in product(vocs,vals,vals,vals):
    sel=[x for x in r if (x['vocabulary'],x['office_first'],x['service_match'],x['office_match'])==(v,order,s,o)]
    assert len(sel)==32
    cells.append({'vocabulary':v,'office_first':order,'service_match':s,'office_match':o,'n':len(sel),'calls':sum(x['a'] for x in sel),'exact':sum(x['score']['exact'] for x in sel),'abstain':sum(x['score']['abstain'] for x in sel),'other':sum(x['score']['other'] for x in sel),'cap':sum(x['finish']=='length' for x in sel)})
write('transfer_cells.csv',cells)
paired=[]
for name in ['office matched versus neither','office first versus service first','tree versus compass']:
    factors = [(v,order) for v in vocs for order in vals] if name.startswith('office matched') else [(v,None) for v in vocs] if name.startswith('office first') else [(None,order) for order in vals]
    for v,order in factors:
        pairs=[]
        for f,d in product(fs,range(4)):
            if name.startswith('office matched'): keys=[(f,d,v,order,False,m) for m in vals]
            elif name.startswith('office first'): keys=[(f,d,v,ord_,False,True) for ord_ in vals]
            else: keys=[(f,d,voc,order,False,True) for voc in vocs]
            pairs.append(tuple(index[k]['a'] for k in keys))
        paired.append({'contrast':name,'vocabulary':v,'office_first':order,'n':len(pairs),'no_to_call':pairs.count((False,True)),'call_to_no':pairs.count((True,False)),'both_call':pairs.count((True,True)),'neither_call':pairs.count((False,False))})
write('transfer_paired.csv',paired)
fam=[]
for f,v,ord_ in product(fs,vocs,vals):
    counts=[sum(index[f,d,v,ord_,False,o]['a'] for d in range(4)) for o in vals]
    fam.append({'family':f,'vocabulary':v,'office_first':ord_,'neither_calls':counts[0],'office_only_calls':counts[1],'denominator':4})
write('transfer_families.csv',fam)
res={'n':len(r),'unique_prompts':len(index),'replicated':repeats,'novel_prompts':512-128,'generated_calls':sum(x['a'] for x in r),'normalized_abstentions':sum(x['score']['abstain'] for x in r),'exact_json_calls':sum(x['score']['exact'] for x in r),'unclassified':sum(x['score']['other'] for x in r),'capped':sum(x['finish']=='length' for x in r),'cells':cells,'paired':paired}
(HERE/'transfer_results.json').write_text(json.dumps(res,indent=2)+'\n')
print('n, repeat',len(r),repeats,'forms',[(k,res[k]) for k in ['generated_calls','exact_json_calls','normalized_abstentions','unclassified','capped']])
for v,ord_ in product(vocs,vals):
    print(v,ord_,[x['calls'] for x in cells if x['vocabulary']==v and x['office_first']==ord_])
print([(x['contrast'],x['vocabulary'],x['office_first'],x['no_to_call'],x['call_to_no']) for x in paired])
