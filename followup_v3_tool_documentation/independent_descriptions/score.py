"""Score the complete external-tool bank, preserving ambiguous and truncated replies."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent

def load(path): return [json.loads(line) for line in path.read_text().splitlines()]

def parse(text):
    s=text.strip()
    if s=='NO_CALL': return 'abstain',None
    without_fence=re.sub(r'^\x60\x60\x60(?:json|python)?\s*|\s*\x60\x60\x60$', '', s, flags=re.I)
    if without_fence.startswith('<tool_call>'):
        without_fence=without_fence.removeprefix('<tool_call>').removesuffix('</tool_call>').strip()
    try: obj=json.loads(without_fence)
    except ValueError: obj=None
    if isinstance(obj,dict) and ('name' in obj or 'arguments' in obj):return 'call',obj
    if '<tool_call>' in s or re.match(r'^[\w.]+\s*\(',s) or re.match(r'^\s*\{\s*"(?:arguments|name)"\s*:',without_fence):return 'call',None
    return 'other',None

def match(value,expected):
    if isinstance(value,str) and isinstance(expected,str):
        return value.strip().casefold()==expected.strip().casefold()
    if isinstance(value,bool) or isinstance(expected,bool): return type(value)==type(expected) and value==expected
    if isinstance(value,(float,int)) and isinstance(expected,(float,int)):return value==expected
    return False

def valid_schema(args,schema):
    props=schema['properties']
    if not isinstance(args,dict) or not set(schema.get('required',[]))<=args.keys()<=props.keys():return False
    types={'string':str,'boolean':bool,'integer':int,'float':(int,float),'number':(int,float),'array':list,'dict':dict,'object':dict}
    for k,val in args.items():
        spec=props[k]
        if not isinstance(val,types[spec['type']]) or (spec['type'] in ('integer','float','number') and isinstance(val,bool)):return False
        if 'enum' in spec and val not in spec['enum']:return False
    return True

def reference_match(args,ref,schema):
    if not isinstance(args,dict): return False
    for k,values in ref.items():
        if k not in args:
            if '' not in values:return False
        elif not any(match(args[k],v) for v in values):return False
    return all(k in ref or ('default' in schema['properties'].get(k,{}) and match(val,schema['properties'][k]['default'])) for k,val in args.items())

def write_csv(name,rows):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def main():
    bank={r['id']:r for r in load(ROOT/'bank.jsonl')}
    scored=[]
    for model in ('qwen','smol'):
        data=load(ROOT/model/'raw.jsonl')
        assert len(data)==len(bank)==len({r['id'] for r in data}) and set(bank)=={r['id'] for r in data}
        for row in data:
            entry=bank[row['id']]
            kind,obj=parse(row['text'])
            source=entry['source_tool']
            expected=source['name'] if entry['naming']=='original' else 'lookup_function'
            selected=bool(obj and obj.get('name')==expected and isinstance(obj.get('arguments'),dict))
            good=selected and valid_schema(obj['arguments'],source['parameters'])
            ref=bool(selected and entry['role']=='target' and reference_match(obj['arguments'],entry['reference_arguments'],source['parameters']))
            scoped = source['name'].rsplit('_',1)[0]==entry['target'].rsplit('_',1)[0]
            scored.append(dict(model=model,id=row['id'],source_id=entry['source_id'],naming=entry['naming'],role=entry['role'],same_namespace=scoped,kind=kind,expected_name_and_object=selected,schema_valid=good,reference_match=ref,schema_and_reference=ref and good,finish=row['finish'],reply=row['text']))
    write_csv('responses_scored.csv',scored)
    cells=[]
    groups=[]
    differences=[]
    for model in ('qwen','smol'):
        by_model=[r for r in scored if r['model']==model]
        for naming in ('original','masked'):
            for role in ('target','distractor'):
                rs=[r for r in by_model if r['naming']==naming and r['role']==role]
                per=defaultdict(list)
                for r in rs:per[r['source_id']].append(r)
                def weighted(key):return mean(mean(func(x) for x in group) for group in per.values() for func in [key])
                cells.append(dict(model=model,naming=naming,role=role,n=len(rs),questions=len(per),n_call=sum(r['kind']=='call' for r in rs),n_no_call=sum(r['kind']=='abstain' for r in rs),n_other=sum(r['kind']=='other' for r in rs),n_named=sum(r['expected_name_and_object'] for r in rs),n_schema=sum(r['schema_valid'] for r in rs),n_ref=sum(r['reference_match'] for r in rs),n_schema_plus_ref=sum(r['schema_and_reference'] for r in rs),n_length=sum(r['finish']=='length' for r in rs),question_weighted_attempt=weighted(lambda r:r['kind']=='call'),question_weighted_attempt_or_other=weighted(lambda r:r['kind']!='abstain')))
                if role=='distractor':
                    for qid,sub in per.items():
                        a=[r for r in sub if r['same_namespace']]
                        b=[r for r in sub if not r['same_namespace']]
                        if a and b:groups.append(dict(model=model,naming=naming,source_id=qid,same_n=len(a),same_fraction=mean(r['kind']=='call' for r in a),other_n=len(b),other_fraction=mean(r['kind']=='call' for r in b)))
        lookup={r['id']:r for r in by_model}
        for r in by_model:
            if r['naming']!='original':continue
            s=lookup[r['id'].rsplit('/',1)[0]+'/masked']
            differences.append(dict(model=model,source_id=r['source_id'],tool=r['id'].split('/')[1],role=r['role'],original=r['kind'],masked=s['kind'],change=int(s['kind']=='call')-int(r['kind']=='call')))
    write_csv('cells.csv',cells)
    write_csv('namespace_within_question.csv',groups)
    write_csv('name_masking_pairs.csv',differences)
    (ROOT/'results.json').write_text(json.dumps(cells,indent=2)+'\n')
if __name__=='__main__':main()
