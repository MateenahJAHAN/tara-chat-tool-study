"""Show each matched question, including neutral pairs and opposite switches."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
ROOT=Path(__file__).resolve().parent
by_id={}
for r in map(json.loads,(ROOT/'catalogue_bank.jsonl').open()):
    by_id[r['source_id']]=r
order=sorted(by_id,key=lambda k:(by_id[k]['city'],k))
models={'qwen':'Qwen2.5-1.5B','smol':'SmolLM2-1.7B'}
with (ROOT/'catalogue_responses_scored.csv').open() as f:lookup={(r['model'],r['source_id'],r['present']=='True',r['matches']=='True'):r for r in csv.DictReader(f)}
fig,axes=plt.subplots(1,4,figsize=(9.3,5.6),layout='constrained',sharey=True)
colors={'reference':'#226968','wrong':'#c86c40','abstain':'#d9e0e4','other':'#85709d'}
for ax,(model,present) in zip(axes,[(m,p) for m in models for p in [False,True]]):
    for row,sid in enumerate(order):
        for col,match in enumerate([False,True]):
            r=lookup[model,sid,present,match]
            tag='reference' if r['selected_target']=='True' else 'wrong' if r['kind']=='call' else r['kind']
            ax.add_patch(Rectangle((col-.48,row-.46),.96,.92,facecolor=colors[tag],edgecolor='none'))
    ax.set_xlim(-.5,1.5)
    ax.set_ylim(len(order)-.5,-.5)
    ax.set_xticks([0,1],['Other city','Same city'],rotation=25,ha='right')
    ax.set_title(models[model]+'\n'+('Reference included' if present else 'Reference removed'),fontsize=10)
    ax.tick_params(length=0,labelsize=8)
    ax.set_yticks(range(len(order)),[by_id[s]['city']+' · '+by_id[s]['target'].split('_')[-1] for s in order],fontsize=8)
    for spine in ax.spines.values():spine.set_visible(False)
fig.suptitle('Office-address crossover on the 12 city questions',fontsize=15,weight='bold')
labels={'reference':'Called the reference tool','wrong':'Other attempted call','abstain':'Exact NO_CALL','other':'Other response'}
fig.legend(handles=[Patch(facecolor=v,label=labels[k]) for k,v in colors.items()],ncols=2,loc='outside lower center',frameon=False)
fig.savefig(ROOT/'catalogue_office_pairs.png',dpi=220)
