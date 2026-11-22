"""Make a local reviewer-only packet and a separate administrator key."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import secrets

ROOT=Path(__file__).resolve().parent

def records(p):return [json.loads(l) for l in p.read_text().splitlines()]

def write_csv(path,data):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def main():
    blind=ROOT/'reviewer_packet'
    admin=ROOT/'coordinator'
    blind.mkdir(exist_ok=True)
    admin.mkdir(exist_ok=True)
    seed_file=admin/'shuffle_seed.txt'
    if not seed_file.exists():seed_file.write_text(secrets.token_hex(24)+'\n')
    seed=seed_file.read_text().strip()
    bank={r['id']:r for r in records(ROOT/'bank.jsonl')}
    geo={r['id']:r for r in records(ROOT/'catalogue_bank.jsonl')}
    data=[(m,r,False) for m in ('qwen','smol') for r in records(ROOT/m/'raw.jsonl')]
    data.extend((m,r,True) for m in ('qwen','smol') for r in records(ROOT/('catalogue_'+m)/'raw.jsonl'))
    assert len(data)==2*(len(bank)+len(geo))
    data.sort(key=lambda x:(x[0],x[2],x[1]['id']))
    random.Random(seed).shuffle(data)
    reviews=[];key=[]
    for m,r,is_geo in data:
        if is_geo:
            b=geo[r['id']]
            offered=b['tools']
            context='The documentation office is where the catalogue was written. Documentation office: '+b['office']+'.'
            condition='office_match' if b['matches'] else 'office_mismatch'
            role='reference_present' if b['present'] else 'reference_absent'
        else:
            b=bank[r['id']]
            tool=b['source_tool'].copy()
            if b['naming']=='masked': tool['name']='lookup_function'
            offered=[tool]
            context=''
            condition=b['naming']
            role=b['role']
        rid=hashlib.sha256((seed+m+b['id']).encode()).hexdigest()[:12]
        reviews.append(dict(review_id=rid,question=b['question'],catalogue_context=context,offered_functions=json.dumps(offered,ensure_ascii=False),reply=r['text'],tool_covers_request='',reply_action='',arguments_and_values='',notes=''))
        key.append(dict(review_id=rid,model=m,condition=condition,role=role,source_id=b['source_id'],item_id=b['id'],finish=r['finish']))
    assert len({r['review_id'] for r in reviews})==len(reviews)
    write_csv(blind/'responses.csv',reviews)
    write_csv(admin/'review_key.csv',key)
    (blind/'INSTRUCTIONS.txt').write_text("""Each row contains the actual user request, the catalogue offered to the model and its verbatim reply. Judge four fields independently before seeing aggregate results.

tool_covers_request: yes / no / ambiguous. Can any offered function directly provide what the user asks for? A nearby service that needs extra information or performs an unsolicited action may be unsuitable; explain borderline cases.
reply_action: call / abstain / other / unclear. Consider what the reply actually did. Describe any malformed call in notes.
arguments_and_values: adequate / inadequate / ambiguous / not applicable. Judge whether a proposed call uses the offered function and supplies the needed information without invention. For no attempted call, mark not applicable.
notes: identify the concrete issue or plausible alternate reading if there is one.

The model identity, benchmark answer and machine score are withheld. Function names, optional office address and complete tools remain visible, since they are part of the prompt. Complete one row at a time. Work from an independent copy. Return a scored copy only after finishing the packet.
""")
if __name__=='__main__':main()
