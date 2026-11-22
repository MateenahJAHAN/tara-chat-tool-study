"""Re-decode every generation and recount the action endpoints without importing their scorer."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent

def read_jsonl(path):return [json.loads(x) for x in path.read_text().splitlines()]
def h(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def tab(file):
    with file.open() as f:return list(csv.DictReader(f))

def main():
    allids=set()
    discrep=[]
    cells=defaultdict(Counter)
    expected_rows=Counter()
    for geo in (False,True):
        bank={r['id']:r for r in read_jsonl(ROOT/('catalogue_bank.jsonl' if geo else 'bank.jsonl'))}
        score=tab(ROOT/('catalogue_responses_scored.csv' if geo else 'responses_scored.csv'))
        reports={(r['model'],r['id']):r for r in score}
        for model in ('qwen','smol'):
            folder=ROOT/(('catalogue_' if geo else '')+model)
            run=json.loads((folder/'run.json').read_text())
            data=read_jsonl(folder/'raw.jsonl')
            assert run['raw_sha256']==h(folder/'raw.jsonl') and len(data)==run['n']==len(bank)
            hub='models--Qwen--Qwen2.5-1.5B-Instruct' if model=='qwen' else 'models--HuggingFaceTB--SmolLM2-1.7B-Instruct'
            tokenizer=AutoTokenizer.from_pretrained(Path.home()/'.cache/huggingface/hub'/hub/'snapshots'/run['revision'],local_files_only=True)
            eoses=run['generation']['eos_token_id']
            eoses={eoses} if isinstance(eoses,int) else set(eoses)
            assert len({r['id'] for r in data})==len(data)
            for r in data:
                b=bank[r['id']]
                assert r['input_ids']==tokenizer.encode(r['prompt'],add_special_tokens=False)
                assert b['question'] in r['prompt']
                split=r['prompt'].split('Function catalogue (JSON):\n',1)[1]
                shown=json.JSONDecoder().raw_decode(split)[0]
                if geo:
                    assert b['tools']==shown and ('Documentation office: '+b['office']+'.') in r['prompt']
                    target=b['target'] if b['present'] else None
                    cell=('catalogue',model,b['present'],b['matches'])
                else:
                    exp=b['source_tool'].copy()
                    if b['naming']=='masked':exp['name']='lookup_function'
                    assert shown==exp
                    target=exp['name']
                    cell=('single',model,b['role'],b['naming'])
                context = ('The documentation office is where the catalogue was written. Documentation office: ' + b['office'] + '.') if geo else ''
                offered = b['tools'] if geo else [exp]
                expected_rows[(b['question'],context,json.dumps(offered,sort_keys=True),r['text'])] += 1
                tokens=r['generated_ids']
                expected_len=r['finish']=='length'
                assert (len(tokens)==96 and not eoses.intersection(tokens)) if expected_len else (tokens[-1] in eoses and not eoses.intersection(tokens[:-1]))
                reply=tokenizer.decode(tokens if expected_len else tokens[:-1],skip_special_tokens=False)
                assert reply==r['text']
                parsed=None
                # Locate the first JSON object independently of the optional Markdown/code wrapper.
                if '{' in reply:
                    try:parsed,_=json.JSONDecoder().raw_decode(reply[reply.index('{'):])
                    except ValueError:pass
                if reply.strip()=='NO_CALL':kind='abstain'
                elif isinstance(parsed,dict) and {'name','arguments'}.intersection(parsed):kind='call'
                elif '<tool_call>' in reply or re.search(r'^\s*[a-zA-Z][\w.]*\s*\(',reply) or re.search(r'^\s*(?:\x60\x60\x60\w*\s*)?\{\s*"(?:name|arguments)"\s*:',reply):kind='call'
                else:kind='other'
                selected=bool(target and isinstance(parsed,dict) and parsed.get('name')==target and isinstance(parsed.get('arguments'),dict))
                official=reports[model,r['id']]
                field='selected_target' if geo else 'expected_name_and_object'
                if (official['kind'],official[field]=='True')!=(kind,selected):
                    discrep.append({'model':model,'id':r['id'],'primary':official['kind'],'permissive':kind,'primary_named':official[field],'permissive_named':selected,'reply':reply})
                cells[cell][kind]+=1
                cells[cell]['target_name_and_object']+=selected
                allids.add((model,r['id']))
    packet=tab(ROOT/'reviewer_packet/responses.csv')
    assert len(allids)==len(packet)==len({r['review_id'] for r in packet})
    assert set(packet[0])==set('review_id question catalogue_context offered_functions reply tool_covers_request reply_action arguments_and_values notes'.split())
    observed_rows=Counter((r['question'],r['catalogue_context'],json.dumps(json.loads(r['offered_functions']),sort_keys=True),r['reply']) for r in packet)
    assert observed_rows==expected_rows
    assert all(not any(r[s] for s in ['tool_covers_request','reply_action','arguments_and_values','notes']) for r in packet)
    literal_model_mentions=[r['review_id'] for r in packet if re.search(r'\b(?:qwen(?:2(?:\.5)?)?|smollm(?:2)?)\b',r['reply'],re.I)]
    out={'literal_model_name_mentions_in_blind_replies':literal_model_mentions,'raw_records_checked':len(allids),'reviewer_rows_verified':len(packet),'independent_parse_counts':{'/'.join(map(str,k)):v for k,v in cells.items()},'strict_and_permissive_differences':discrep}
    (ROOT/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print('Verified',len(allids),'raw outputs and blinded rows;',len(discrep),'permissive/strict parser differences.')
if __name__=='__main__':main()
