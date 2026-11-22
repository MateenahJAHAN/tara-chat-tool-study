"""Check if the published value list permits at least one valid target call."""
from collections import Counter
from itertools import product
import json
from pathlib import Path
from score import ROOT,load,valid_schema,reference_match
items=[r for r in load(ROOT/'bank.jsonl') if r['role']=='target' and r['naming']=='original']
out=[]
for r in items:
    ref=r['reference_arguments']; schema=r['source_tool']['parameters']
    feasible=[]
    for proposed in product(*ref.values()):
        # The BFCL empty string in the accepted list allows omission.
        args={k:v for k,v in zip(ref,proposed) if v!=''}
        if valid_schema(args,schema) and reference_match(args,ref,schema):feasible.append(args)
    out.append(dict(id=r['source_id'],target=r['target'],n_jointly_valid=len(feasible),example=feasible[:1]))
(ROOT/'reference_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(len(items),'reference items;',sum(bool(x['n_jointly_valid']) for x in out),'allow a jointly valid call.')
