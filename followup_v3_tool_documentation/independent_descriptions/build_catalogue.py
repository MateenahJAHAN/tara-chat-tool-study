"""Freeze the geographic office manipulation on the external city-bearing questions."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent
CITIES='Sydney|San Bruno|London|Los Angeles|Sonoma|Philadelphia|Berkeley|Portland|New York|Austin|Fremont'.split('|')

def digest(s):return hashlib.sha256(s.encode()).hexdigest()
groups=defaultdict(list)
for line in (ROOT/'bank.jsonl').read_text().splitlines():
    r=json.loads(line)
    if r['naming']=='original':groups[r['source_id']].append(r)
geographic=[]
for sid,items in sorted(groups.items()):
    cities=[c for c in CITIES if c.casefold() in items[0]['question'].casefold()]
    if len(cities)==1:geographic.append((sid,items,cities[0]))
    else:assert not cities
q_cities=[c for _,_,c in geographic]
offset=next(n for n in range(1,len(q_cities)) if all(c!=q_cities[(i+n)%len(q_cities)] for i,c in enumerate(q_cities)))
out=[]
for i,(sid,items,city) in enumerate(geographic):
    ordered=sorted(items,key=lambda r:digest('external-catalogue-v1/'+sid+'/'+r['source_tool']['name']))
    for present in (False,True):
        offered=[r['source_tool'] for r in ordered if present or r['role']!='target']
        for matches in (False,True):
            office=city if matches else q_cities[(i+offset)%len(q_cities)]
            out.append(dict(id=f"{sid}/present{int(present)}/match{int(matches)}",source_id=sid,question=items[0]['question'],target=items[0]['target'],tools=offered,city=city,office=office,present=present,matches=matches))
(ROOT/'catalogue_bank.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in out))
print(len(out),'cards;',len(geographic),'questions; mismatch rotation',offset)
