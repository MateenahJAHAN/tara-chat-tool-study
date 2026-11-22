from collections import Counter,defaultdict
import json
from math import comb
from statistics import mean
from score import ROOT,load,parse,valid_schema,reference_match,write_csv

def exact_p(up,down):
    n=up+down
    return min(1,2*sum(comb(n,i) for i in range(min(up,down)+1))/2**n) if n else None

def main():
    bank={r['id']:r for r in load(ROOT/'catalogue_bank.jsonl')}
    ref={r['source_id']:r['reference_arguments'] for r in load(ROOT/'bank.jsonl') if r['role']=='target'}
    results=[];pairs=[];selection=[];cells=[]
    for model in ('qwen','smol'):
        actual=load(ROOT/('catalogue_'+model)/'raw.jsonl')
        assert len(actual)==len(bank)==len({r['id'] for r in actual}) and {r['id'] for r in actual}==set(bank)
        for r in actual:
            c=bank[r['id']]
            kind,obj=parse(r['text'])
            tools={t['name']:t for t in c['tools']}
            name=obj.get('name') if isinstance(obj,dict) else None
            named=name in tools and isinstance(obj.get('arguments'),dict) if isinstance(name,str) else False
            target=named and name==c['target']
            shape=bool(named and valid_schema(obj['arguments'],tools[name]['parameters']))
            exact=bool(target and shape and reference_match(obj['arguments'],ref[c['source_id']],tools[name]['parameters']))
            results.append(dict(model=model,id=r['id'],source_id=c['source_id'],present=c['present'],matches=c['matches'],kind=kind,selected_offered=named,selected_target=target,schema_valid=shape,target_schema_and_ref=exact,finish=r['finish'],reply=r['text']))
        for present in (False,True):
            subset=[r for r in results if r['model']==model and r['present']==present]
            for matched in (False,True):
                part=[r for r in subset if r['matches']==matched]
                cells.append(dict(model=model,present=present,matches=matched,n=len(part),calls=sum(r['kind']=='call' for r in part),abstain=sum(r['kind']=='abstain' for r in part),other=sum(r['kind']=='other' for r in part),selected_offered=sum(r['selected_offered'] for r in part),selected_reference=sum(r['selected_target'] for r in part),schema_and_reference=sum(r['target_schema_and_ref'] for r in part),length=sum(r['finish']=='length' for r in part)))
            by_q=defaultdict(dict)
            for r in subset:by_q[r['source_id']][r['matches']]=r
            delta=[]
            for q,comp in sorted(by_q.items()):
                assert set(comp)=={False,True}
                metric=lambda r:r['selected_target'] if present else r['kind']=='call'
                diff=int(metric(comp[True]))-int(metric(comp[False]))
                pairs.append(dict(model=model,source_id=q,present=present,mismatch=comp[False]['kind'],match=comp[True]['kind'],mismatch_target=comp[False]['selected_target'],match_target=comp[True]['selected_target'],endpoint_change=diff))
                delta.append(diff)
            selection.append(dict(model=model,present=present,pairs=len(delta),gains=delta.count(1),losses=delta.count(-1),ties=delta.count(0),mean_change=mean(delta),exact_mcnemar_p=exact_p(delta.count(1),delta.count(-1))))
    write_csv('catalogue_responses_scored.csv',results)
    write_csv('catalogue_pairs.csv',pairs)
    write_csv('catalogue_cells.csv',cells)
    (ROOT/'catalogue_results.json').write_text(json.dumps(dict(cells=cells,paired_office_change=selection),indent=2)+'\n')
if __name__=='__main__':main()
