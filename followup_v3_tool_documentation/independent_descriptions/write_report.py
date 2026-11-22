"""Render the research note from complete, verified outputs."""
import csv
import json
from pathlib import Path
from statistics import mean

from audit_output_format import main as audit_formats
ROOT=Path(__file__).resolve().parent
audit_formats()
def data(name):
    with (ROOT/name).open() as f:return list(csv.DictReader(f))
def table(head,body):
    return '| '+' | '.join(head)+' |\n|'+'|'.join('---' for h in head)+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in body)
model_names={'qwen':'Qwen2.5-1.5B','smol':'SmolLM2-1.7B'}
cells=json.loads((ROOT/'results.json').read_text()); catalog=json.loads((ROOT/'catalogue_results.json').read_text())
audit=json.loads((ROOT/'verification.json').read_text())
rows=[];pairrows=[];cityrows=[];formatrows=[]
for r in data('output_format_summary.csv'):
    if r['stage']=='single' and r['role']=='distractor':
        formatrows.append([model_names[r['model']],r['condition'],r['attempt'],r['named_plus_args'],r['schema_valid'],r['plain_json_dispatchable'],r['schema_echo'],r['attempts_with_incomplete_json'],r['n_length']])
for model,name in model_names.items():
    for naming,label in [('original','Source'),('masked','Masked')]:
        t,d=[next(r for r in cells if (r['model'],r['naming'],r['role'])==(model,naming,k)) for k in ['target','distractor']]
        rows.append([name,label,f"{t['n_named']}/23",f"{t['n_schema_plus_ref']}/23",f"{d['n_call']}/60",f"{d['n_named']}/60",f"{d['question_weighted_attempt']:.1%}",f"{d['n_no_call']} / {d['n_other']}"])
        match=[r for r in data('namespace_within_question.csv') if r['model']==model and r['naming']==naming]
        pairrows.append([name,label,len(match),f"{mean(float(r['same_fraction']) for r in match):.1%}",f"{mean(float(r['other_fraction']) for r in match):.1%}"])
    for pres in (False,True):
        p=next(r for r in catalog['paired_office_change'] if (r['model'],r['present'])==(model,pres))
        a,b=[next(r for r in catalog['cells'] if (r['model'],r['present'],r['matches'])==(model,pres,v)) for v in [False,True]]
        k='selected_reference' if pres else 'calls'
        cityrows.append([name,'Reference tool selected' if pres else 'Any call with target removed',f"{a[k]}/12",f"{b[k]}/12",p['gains'],p['losses'],p['ties']])
qwen=next(r for r in cells if r['model']=='qwen' and r['role']=='distractor' and r['naming']=='original')
smol=next(r for r in cells if r['model']=='smol' and r['role']=='distractor' and r['naming']=='original')
geo= {r['model']:r for r in catalog['paired_office_change'] if not r['present']}
total_length=sum(r['n_length'] for r in cells)+sum(r['length'] for r in catalog['cells'])
total_other=sum(r['n_other'] for r in cells)+sum(r['other'] for r in catalog['cells'])
report=f"""# Do language models call a tool that cannot answer the question?

With one unrelated tool offered under its published name, Qwen2.5-1.5B attempted a call on {qwen['n_call']} of 60 presentations and SmolLM2-1.7B on {smol['n_call']} of 60. The questions asked about movies, music, events, lodging, attractions and salons. Each came from the public Berkeley Function Calling Leaderboard (BFCL) and appeared with the same wording in When2Call. Across the 12 questions naming a city, making the documentation office match the requested city gained {geo['qwen']['gains']} false calls and lost {geo['qwen']['losses']} for Qwen, and gained {geo['smol']['gains']} and lost {geo['smol']['losses']} for SmolLM2.

## Calling a single offered function

We gave each model the actual reference function or one of the distractors that appeared with it in the source catalogue. Each presentation was repeated with the function name replaced by `lookup_function`. This isolates the contribution of the name while keeping the description and argument schema fixed. Function names, top-level descriptions and questions were screened against the earlier TARA manifests. None matched after case and punctuation normalization, including the rendered earlier scope substitutions. [Source manifests](sources.json) · [Overlap checks](holdout_check.json). The final bank includes 23 distinct questions, 23 function names, 63 distinct description texts, and 65 distinct full schemas. Twelve of the 23 questions concern movies. Each model completed 166 single-tool presentations. The published name and masked name are two presentations of the same example.

{table(['Checkpoint','Name','Reference selected','Reference + valid args¹','Distractor attempts','Named distractor','Distractor rate²','Exact NO_CALL / other, distractors'],rows)}

¹ Requires a correctly named call with an argument object, valid supplied field names, types, enumerations and required fields, plus values accepted by our disclosed comparison against the BFCL reference. Date formats, omitted values and city abbreviations can affect this strict count. This measures agreement with the reference under our comparison rule. One source item about actor Madge Brindley requires the value `Any` for genre, but its function enum excludes `Any`. This caps the joint score at 22/23. [Reference consistency audit](reference_audit.json). No functions were executed.

² For each of the 23 questions, divide its distractor calls by its number of distractors, then average across questions.

![Reference and distractor calls, by checkpoint and function naming](tool_name_effect.png)

### Could a strict tool runtime dispatch the reply?

The primary attempt endpoint counts identifiable function-shaped output. A model that repeats the tool schema has emitted documentation. It has supplied no argument values. The table separates these cases from replies that supply a real function name and arguments. The strict dispatch column requires a standalone JSON object with exactly the two requested fields and arguments satisfying the offered schema. All columns refer to the same 60 distractor presentations per row and may overlap.

{table(['Checkpoint','Name','Broad attempts','Name + arguments','Valid against schema','Strict dispatch','Schema echoes','Attempts with incomplete JSON','Hit token cap'],formatrows)}

For one Los Angeles movie request, the user gave “Supernatural” as the genre and asked for a regular screening. Offered only the show-times lookup, Qwen supplied `Supernatural` as a movie title and supplied the unrequested date `2023-04-05`. The arguments passed the offered schema. A schema check alone would miss both the change in the user's request and the unsupported date.

With the name masked, SmolLM2 went from abstention to a call-shaped response on 10 of the 60 distractor pairs, with zero reverse switches. Qwen gained two and lost one. Its equal-question-weighted distractor rate stayed at 4.3% in both naming conditions. [Paired name changes](name_masking_pairs.csv).

The format audit searches for an explicit description or parameters field with a name and no arguments field. An incomplete prefix can satisfy that test. The dispatch score measures format and schema only. A well-formed call can still be inappropriate for the question. [Per-reply format audit](output_format_by_reply.csv) · [All format counts](output_format_summary.csv).

Ten questions offered both a distractor from the target API namespace and one from another namespace. A ticket-purchase function and a search function, for example, can both concern events yet have different effects. The following rates give each of those ten questions equal weight. Membership in an API namespace also changes wording, schemas and functions, so these differences are descriptive.

{table(['Checkpoint','Name','Questions','Calls, same namespace','Calls, other namespace'],pairrows)}

## Does the documentation-office effect transfer?

For each of 12 questions naming one city, we presented the eligible catalogue with and without the source reference function. Within each pair, the only change was the catalogue's documentation-office city. A cyclic reassignment preserved the city frequency across both conditions. For example, a request to find music events in Philadelphia could be shown a ticket-purchase function alone. Its documentation office could be Philadelphia or another city. Neither address turns a purchase function into a search function.

{table(['Checkpoint','Outcome','Other office city','Same office city','Gained','Lost','Tied'],cityrows)}

Both checkpoints made one fewer false attempt in aggregate when the office matched. The paired, two-sided exact McNemar test gives p = 1.0 for each checkpoint. This small selected sample gives no evidence that a matching office increased tool use. Qwen’s earlier switch on 64/64 locally written compass-card pairs did not recur here.

The primary parser left nine SmolLM2 catalogue replies unclassified. Six contained an actual call inside a single-element JSON array. Two were natural-language refusals, and one gave a truncated example call inside an explanatory paragraph. Counting array-wrapped calls increases SmolLM2’s reference-absent counts from 7 to 9 (other city) and 6 to 8 (same city). The office contrast stays at one fewer call. Counting every unclassified negative as an attempt bounds those cells at 7–10/12 and 6–9/12. The strict standalone, schema-valid counts were 4→3 for SmolLM2 and 4→2 for Qwen. Positive target selection for SmolLM2 was unaffected by accepting arrays because the wrapped positive calls selected a house-booking distractor.

Three of the 12 reference-absent catalogues contained a single distractor. The other nine had two to seven. All reference-present catalogues contained at least two functions. The source selection rule and outcomes were frozen before reading either model's replies. [Matched rows](catalogue_pairs.csv) · [Four cells per model, including unclassified and truncated replies](catalogue_cells.csv).

![Each question across office conditions](catalogue_office_pairs.png)

## Interpretation and limits

The previous TARA experiment changed a service district and a documentation-office field on locally authored cards. Here scope comes from real functions with different operations. The office belongs to a catalogue rather than an individual service. The present test measures transfer of the incidental cue to this setting. Its 12 city-bearing queries give limited precision. Repeated schemas and shared question templates also limit independence. The external examples predate this study, and both external releases share BFCL provenance. Prior model-training exposure is unknown. These two small checkpoints ran with inline catalogues, greedy decoding and their own instruction templates.

Both stages produced {audit['raw_records_checked']} generations. {total_length} reached the 96-token limit and {total_other} lacked both an exact abstention and a recognizable call under the prespecified action definition. The independent verifier reconstructed token sequences for every prompt and reply and compared a more permissive JSON read with the primary parser. It recorded {len(audit['strict_and_permissive_differences'])} differing action or named-selection judgments. [Verifier and differences](verification.json) · [Single-tool raw replies](qwen/raw.jsonl) · [Second checkpoint](smol/raw.jsonl) · [Catalogue replies for Qwen](catalogue_qwen/raw.jsonl) · [Catalogue replies for SmolLM2](catalogue_smol/raw.jsonl).

## Review and reproduction

Give reviewers the [instructions](reviewer_packet/INSTRUCTIONS.txt) and [randomized responses](reviewer_packet/responses.csv) together. The review file contains the questions, tool definitions, verbatim replies and blank rating columns. The model and expected answer are withheld. Function names and office metadata remain visible to support independent judgments. No human ratings have been entered.

[Sources, authors and licences](ATTRIBUTION.md) · [Single-tool protocol](PROTOCOL.md) · [Office protocol](CATALOGUE_PROTOCOL.md) · [Frozen single-tool inputs](freeze.json) · [Frozen office inputs](catalogue_freeze.json) · [All scored single-tool responses](responses_scored.csv) · [All scored catalogues](catalogue_responses_scored.csv).

To rebuild both question banks from the pinned releases, run `python rebuild_banks.py` from this directory. It checks the reconstructed files against the original design hashes, including all prior-study prompts used in the overlap check. [Original selection and input checksums](sources.json) · [Verbatim selection code](prepare.py).

To reproduce results from the saved model outputs, run `python score.py`, `python score_catalogue.py`, `python check_references.py`, `python audit_output_format.py`, and `python verify.py`. The verifier uses the pinned tokenizers and checks all 428 archived responses against the reviewer packet. Run `python plot.py` and `python plot_catalogue.py` for the figures. For fresh inference, use the pinned model snapshots in an empty output copy and run `run.py qwen`, `run.py smol`, `run_catalogue.py qwen` and `run_catalogue.py smol` with Python. The original run manifests record hardware, versions and output checksums.
"""
(ROOT/'README.md').write_text(report)
