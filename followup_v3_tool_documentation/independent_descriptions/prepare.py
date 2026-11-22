"""Build all eligible presentations from fixed external source revisions."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import urllib.request

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
BFCL = 'https://raw.githubusercontent.com/ShishirPatil/gorilla/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data/'
W2C = 'https://raw.githubusercontent.com/NVIDIA/When2Call/ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc/data/test/when2call_test_llm_judge.jsonl'
FILES = [
    ('maj_tara_external_when2call.jsonl', W2C, '0b710578e3b02479e5acf5688140ec5edf71383fa5c093481c7536b57fd13b25'),
    ('maj_bfcl_live_multiple_tara.json', BFCL + 'BFCL_v4_live_multiple.json', 'fd8ccfad4d911420d0e3341dbe2fff77d1d341da934248b9bb2bda24ab3a10c8'),
    ('maj_bfcl_live_multiple_answers_tara.json', BFCL + 'possible_answer/BFCL_v4_live_multiple.json', '97e90d59c5bd76c55a2920ce93e5566e9046307d3f558578f085f9d3a56c3084'),
]
TARGETS = set('Movies_1_FindMovies Movies_3_FindMovies Media_3_FindMovies Music_3_LookupMusic Events_3_FindEvents Travel_1_FindAttractions Hotels_2_SearchHouse Hotels_4_SearchHotel Weather_1_GetWeather Services_1_FindProvider'.split())
PREFIXES = tuple('Movies_1_ Movies_3_ Media_3_ Music_3_ Events_3_ Travel_1_ Restaurants_2_ Hotels_2_ Hotels_4_ Flights_4_ Buses_3_ Trains_1_ RentalCars_3_ Services_1_ Weather_1_'.split())
# Mixed play and search requests cannot unambiguously label a play-only distractor as a false call.
AMBIGUOUS = {'live_multiple_343-133-7', 'live_multiple_840-178-15'}

def norm(text):
    return re.sub(r'[^a-z0-9]+','',text.casefold())

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def seen():
    names, descs, questions = set(), set(), set()
    def visit(obj):
        if isinstance(obj, list):
            for item in obj: visit(item)
        elif isinstance(obj, dict):
            if {'name','description','parameters'} <= obj.keys():
                names.add(norm(obj['name']))
                descs.add(norm(obj['description']))
            for key, val in obj.items():
                if key in {'query','question','content'} and isinstance(val,str): questions.add(norm(val))
                if isinstance(val,(list,dict)): visit(val)
    study=WORK/'tara-chat-tool-study'
    files=list(study.glob('study_v1/**/split_manifest.json'))+list(study.glob('development_calibration/data/BFCL*.json'))+[study/'followup_v2_tool_applicability/bank.json']
    for p in files:
        try: data=json.loads(p.read_text())
        except json.JSONDecodeError: data=rows(p)
        visit(data)
    for name,desc,_ in json.loads((HERE.parent/'bank_spec.json').read_text()):
        names.add(norm(name))
        descs.add(norm(desc))
    transfer = HERE.parent/'transfer_01/raw.jsonl'
    for r in rows(transfer):
        questions.add(norm(re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>',r['prompt'],re.S).group(1)))
    files.extend([transfer,HERE.parent/'bank_spec.json'])
    return names,descs,questions,{str(p.relative_to(WORK)):digest(p) for p in files}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',type=Path,default=Path('/tmp'))
    cache=ap.parse_args().cache
    cache.mkdir(exist_ok=True,parents=True)
    for f,url,expected in FILES:
        p=cache/f
        if not p.exists(): urllib.request.urlretrieve(url,p)
        assert digest(p)==expected, f
    external,original,answers=[rows(cache/f) for f,_,_ in FILES]
    bf={r['id']:r for r in original}
    truth={r['id']:r['ground_truth'] for r in answers}
    names,descs,qs,prior=seen()
    counts=Counter()
    items=[]
    for src in sorted(external,key=lambda r:r['source_id']):
        if src['source']!='BFCL v2 Live Multiple' or src['correct_answer']!='tool_call':continue
        counts['upstream_positive_live_multiple']+=1
        key=src['source_id']
        target=json.loads(src['target_tool'])['name']
        if target not in TARGETS:continue
        counts['target_in_prespecified_domains']+=1
        if key in AMBIGUOUS:
            counts['mixed_search_and_play_excluded']+=1
            continue
        row=bf[key]
        q=row['question'][0][0]['content']
        if q != src['question']:
            counts['version_changed_question_excluded']+=1
            continue
        if norm(q) in qs:
            counts['previous_question_excluded']+=1
            continue
        assert list(truth[key][0])==[target] and len(truth[key])==1
        external_tools={t['name']:t for t in map(json.loads,src['tools'])}
        tools={t['name']:t for t in row['function']}
        if any(name not in external_tools or t['description']!=external_tools[name]['description'] for name,t in tools.items()):
            counts['version_changed_tool_description_excluded']+=1
            continue
        def old(t):return norm(t['name']) in names or norm(t['description']) in descs
        if old(tools[target]):
            counts['previous_target_excluded']+=1
            continue
        alternatives=[t for t in tools.values() if t['name']!=target and t['name'].startswith(PREFIXES) and not old(t)]
        if not alternatives:
            counts['no_unused_distractor']+=1
            continue
        counts['eligible_questions']+=1
        for t in [tools[target],*alternatives]:
            role='target' if t['name']==target else 'distractor'
            counts[role+'_pairings']+=1
            for naming in ('original','masked'):
                items.append({'id':f"{key}/{t['name']}/{naming}",'source_id':key,'when2call_uuid':src['uuid'],'question':q,'target':target,'role':role,'naming':naming,'source_tool':t,'reference_arguments':truth[key][0][target] if role=='target' else None})
    assert len(items)==len({r['id'] for r in items})
    (HERE/'bank.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
    meta={'counts':dict(counts),'distinct_schemas':len({json.dumps(r['source_tool'],sort_keys=True) for r in items}),'source_files':FILES,'prior_study_files':prior,'bank_sha256':digest(HERE/'bank.jsonl')}
    (HERE/'sources.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta['counts']), len(items),meta['distinct_schemas'])
if __name__=='__main__':main()
