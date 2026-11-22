import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

root=Path(__file__).resolve().parent
with (root/'cells.csv').open() as f: cells={(r['model'],r['naming'],r['role']):r for r in csv.DictReader(f)}
fig,axes=plt.subplots(1,2,figsize=(9.1,3.7),layout='constrained',sharex=True)
palette={'original':'#1b5f75','masked':'#b4532f'}
for ax,(role,title) in zip(axes,[('target','Reference tool available'),('distractor','Only a distractor available')]):
    for index,model in enumerate(('qwen','smol')):
        ys=[]
        for i,naming in enumerate(('original','masked')):
            r=cells[model,naming,role]
            value=float(r['question_weighted_attempt']) if role=='distractor' else int(r['n_named'])/int(r['n'])
            y=3-index*2-i*.56
            ys.append((value,y))
            ax.scatter(value,y,s=66,color=palette[naming],edgecolor='white',zorder=3)
            ax.annotate(f"{value:.0%}",(value,y),xytext=(-8 if value>.83 else 8,0),textcoords='offset points',ha='right' if value>.83 else 'left',va='center',fontsize=9)
        ax.plot([p[0] for p in ys],[p[1] for p in ys],color='#b4b9bf',lw=1.5,zorder=1)
    ax.set_title(title,fontsize=13,loc='left',pad=10,weight='bold')
    ax.set_yticks([2.72,.72],['Qwen2.5\n1.5B','SmolLM2\n1.7B'])
    ax.set_xlim(-.06,1.06);ax.set_ylim(0.09,3.45)
    ax.set_xticks([0,.25,.5,.75,1])
    ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.grid(axis='x',alpha=.25);ax.tick_params(axis='both',length=0)
    for side in ['top','right','left']:ax.spines[side].set_visible(False)
    ax.set_xlabel('Called the named function' if role=='target' else 'Attempted a call, mean across questions',fontsize=10)
fig.suptitle('Tool use on 23 externally authored questions',x=.07,ha='left',weight='bold',fontsize=16)
for label,color in palette.items():
    axes[0].scatter([],[],color=color,label='Original name' if label=='original' else 'Masked name')
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncols=2,frameon=False)
fig.savefig(root/'tool_name_effect.png',dpi=220)
