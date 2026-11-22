"""Recount text overlap, including the fully rendered earlier scope substitutions."""
from pathlib import Path
from prepare import HERE,WORK,norm,rows,seen
import json

old_names,old_desc,old_questions,files=seen()
two=json.loads((WORK/'tara-chat-tool-study/followup_v2_tool_applicability/bank.json').read_text())
for p in two['pairs']:
    for k in ['in_scope','out_of_scope']:old_questions.add(norm(p['query'].format(scope=p[k])))
bank=rows(HERE/'bank.jsonl')
new_names={norm(r['source_tool']['name']) for r in bank}
new_desc={norm(r['source_tool']['description']) for r in bank}
new_queries={norm(r['question']) for r in bank}
result={k:{'new_strings':len(new),'exact_overlap':sorted(new&old)} for k,new,old in [('source_name',new_names,old_names),('top_level_description',new_desc,old_desc),('question',new_queries,old_questions)]}
assert all(not d['exact_overlap'] for d in result.values())
(HERE/'holdout_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(result)
