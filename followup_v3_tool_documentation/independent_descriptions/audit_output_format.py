"""Distinguish a requested action from a schema echo or an unusable reply."""
from collections import Counter,defaultdict
import csv,json,re,sys
from score import ROOT,load,parse,valid_schema,write_csv
def main():
    rows=[]
    for catalogue in ([False] if "--single-only" in sys.argv else [False,True]):
        bank={r['id']:r for r in load(ROOT/('catalogue_bank.jsonl' if catalogue else 'bank.jsonl'))}
        for model in ('qwen','smol'):
            scope=('catalogue_' if catalogue else '')+model
            for r in load(ROOT/scope/'raw.jsonl'):
                item=bank[r['id']]
                if catalogue:
                    offered={t['name']:t for t in item['tools']}
                    naming='same_city' if item['matches'] else 'other_city'
                    role='target' if item['present'] else 'distractor'
                else:
                    t=item['source_tool']
                    name='lookup_function' if item['naming']=='masked' else t['name']
                    offered={name:t}
                    naming=item['naming']
                    role=item['role']
                text=r['text'].strip();kind,obj=parse(text)
                try:plain=json.loads(text)
                except ValueError:plain=None
                extracted=None;extraneous=False
                if '{' in text:
                    try:
                        begin=text.find('{')
                        extracted,stop=json.JSONDecoder().raw_decode(text[begin:])
                        extraneous=bool((text[:begin]+text[begin+stop:]).strip())
                    except ValueError:pass
                det=extracted if isinstance(extracted,dict) else {}
                named=isinstance(det.get('name'),str) and det['name'] in offered
                args_ok=named and isinstance(det.get('arguments'),dict)
                valid=bool(args_ok and valid_schema(det['arguments'],offered[det['name']]['parameters']))
                # This is presence of an echo of the tool documentation, distinct from a call with invented args.
                # Exact source string revisits are checked separately from echo shape, which also covers clipped JSON.
                echo=not re.search(r'"arguments"\s*:',text) and bool(re.search(r'"name"\s*:',text)) and bool(re.search(r'"(?:description|parameters)"\s*:',text))
                echo_text=echo and any(json.dumps(t['description'],ensure_ascii=False)[1:-1] in text for t in offered.values())
                exact=(isinstance(plain,dict) and set(plain)=={'name','arguments'} and valid)
                suggested=bool(re.search(r'"name"\s*:\s*"(?:NO_CALL|none|null)"',text,re.I))
                rows.append(dict(stage='catalogue' if catalogue else 'single',model=model,role=role,condition=naming,id=r['id'],primary_kind=kind,length=r['finish']=='length',plain_json=isinstance(plain,dict),schema_echo=echo,verbatim_description_echo=echo_text,complete_object=isinstance(extracted,dict),contains_extraneous_text=extraneous,offered_name_plus_arguments=bool(args_ok),offered_and_schema_valid=valid,exact_dispatch_format=exact,no_call_name_marker=suggested))
    prefix='single_' if '--single-only' in sys.argv else ''
    write_csv(prefix+'output_format_by_reply.csv',rows)
    aggs=[]
    for key in sorted({(r['stage'],r['model'],r['role'],r['condition']) for r in rows}):
        group=[r for r in rows if (r['stage'],r['model'],r['role'],r['condition'])==key]
        aggs.append(dict(zip(('stage','model','role','condition'),key))|dict(n=len(group),attempt=sum(r['primary_kind']=='call' for r in group),unclassified=sum(r['primary_kind']=='other' for r in group),attempts_with_incomplete_json=sum(r['primary_kind']=='call' and not r['complete_object'] for r in group),n_length=sum(r['length'] for r in group),schema_echo=sum(r['schema_echo'] for r in group),echo_incomplete=sum(r['schema_echo'] and not r['complete_object'] for r in group),echo_verbatim=sum(r['verbatim_description_echo'] for r in group),named_plus_args=sum(r['offered_name_plus_arguments'] for r in group),schema_valid=sum(r['offered_and_schema_valid'] for r in group),plain_json_dispatchable=sum(r['exact_dispatch_format'] for r in group),no_call_name_marker=sum(r['no_call_name_marker'] for r in group)))
    write_csv(prefix+'output_format_summary.csv',aggs)
if __name__=='__main__':main()
