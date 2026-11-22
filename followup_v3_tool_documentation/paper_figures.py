"""Draw the task design, cross-checkpoint response counts and site intervention."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
blue, rust, ink = "#116a70", "#a35536", "#182d36"

def save(f,name):
    f.savefig(ROOT/(name + ".png"),dpi=200,bbox_inches="tight",pad_inches=.12,facecolor="white")
    plt.close(f)

def methods():
    f, ax = plt.subplots(figsize=(8.4,3.35))
    ax.set(xlim=(0,10),ylim=(0,5)); ax.axis("off")
    ax.text(0,4.72,"A   Fixed request",weight="bold",fontsize=13,color=ink)
    ax.text(.05,4.18,"“Please find the library opening hours in the East district.”",fontsize=10.5)
    ax.text(0,3.42,"B   Change one field at a time",weight="bold",fontsize=13,color=ink)
    for j,(pos,lab) in enumerate([(0,"Service district"),(2.13,"Manual office"),(4.28,"Correct action")]):
        ax.text(pos,2.89,lab,weight="bold",fontsize=10)
    for i,(s,o,a) in enumerate([("West","West","Abstain"),("West","East","Abstain"),("East","West","Call"),("East","East","Call")]):
        y = 2.39-.55*i
        ax.text(0,y,s,color=blue,weight="bold");ax.text(2.13,y,o,color=rust,weight="bold");ax.text(4.28,y,a)
    ax.plot([6.19,6.19],[.6,3.74],lw=.8,color="#b1c0c1")
    ax.text(6.43,3.42,"C   Readouts",weight="bold",fontsize=13,color=ink)
    ax.text(6.43,2.78,"Both checkpoints",weight="bold",color=blue)
    ax.text(6.43,2.37,"Generate a call or abstain.")
    ax.text(6.43,1.76,"Qwen only",weight="bold",color=blue)
    ax.text(6.43,1.35,"Score the two answer letters.")
    ax.text(6.43,.87,"Swap a district state, then")
    ax.text(6.43,.48,"score letters or generate a call.")
    save(f,"methods_scope")

def behavior():
    rows = pd.read_csv(ROOT/"paper_cells.csv")
    grid = [(v,o) for v in ("compass","tree") for o in (False,True)]
    cols = [(s,d) for s in (False,True) for d in (False,True)]
    f, axes = plt.subplots(1,2,figsize=(8.7,3.45),sharey=True,layout="constrained")
    palette=LinearSegmentedColormap.from_list("calls",["#f3f6f5","#b6d2c8",blue])
    for ax, model in zip(axes,("Qwen2.5","SmolLM2")):
        dat=rows.loc[rows.model==model].set_index(["vocabulary","office_first","service_match","office_match"])
        array=np.array([[dat.loc[(v,o,s,d),"attempt"] for s,d in cols] for v,o in grid])
        ax.imshow(array,vmin=0,vmax=32,cmap=palette,aspect="auto")
        for i,j in np.ndindex(array.shape):
            ax.text(j,i,str(array[i,j]),ha="center",va="center",weight="bold",fontsize=13,color="white" if array[i,j]>19 else ink)
        ax.set_xticks(range(4),["Neither","Office\nonly","Service\nonly","Both"])
        ax.set_yticks(range(4),["Compass / service first","Compass / office first","Trees / service first","Trees / office first"])
        ax.set_title(model + ("-1.5B" if model=="Qwen2.5" else "-1.7B"),loc="left",weight="bold")
        ax.axvline(1.5,color="white",lw=3)
        ax.tick_params(length=0); ax.spines[:].set_visible(False)
    axes[0].set_xlabel("District field matching request");axes[1].set_xlabel("District field matching request")
    save(f,"calls_by_scope")

def intervention():
    tables=pd.read_csv(ROOT/"prompt_effects.csv")
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.1),layout="constrained")
    for ax,(mode,title) in zip(axes,[("text","Inline catalog"),("native","Native template")]):
        dat=tables[tables["mode"]==mode].groupby("family")[["service","office","mid_service","mid_office"]].mean()
        for x,field in enumerate(("service","office","mid_service","mid_office")):
            vals = dat[field]
            # Deterministic offsets distinguish the same eight services in each column.
            ax.scatter(x+np.linspace(-.09,.09,len(vals)),vals,s=15,alpha=.65,color=(blue if "service" in field else rust))
            ax.plot([x-.18,x+.18],[vals.mean()]*2,color=ink,lw=2.4)
        ax.axhline(0,color="#a6b2b4",lw=.75)
        ax.axvline(1.5,ls=":",color="#9da7a7",lw=.8)
        ax.set_xticks(range(4),["Service\nword","Office\nword","Service\npatch","Office\npatch"],fontsize=9)
        ax.set_title(title,loc="left",weight="bold")
        ax.set_ylabel("Shift in A/B log odds")
        ax.margins(x=.1,y=.14)
    save(fig,"field_and_patch")

def formats():
    data=pd.read_csv(ROOT/"paper_cells.csv")
    vals={m:[] for m in ["Qwen2.5","SmolLM2"]}
    for m in vals:
        for scope in (False,True):
            block=data[(data.model==m)&(data.service_match==scope)]
            vals[m].append([block.exact.sum(),(block.attempt-block.exact).sum(),block.abstain.sum()])
    fig,ax=plt.subplots(figsize=(8.1,2.65),layout="constrained")
    categories = ["Exact call","Invalid arguments","NO_CALL"]
    rows=np.array([*vals["Qwen2.5"],*vals["SmolLM2"]]);start=np.zeros(4)
    for j,c in enumerate([blue,rust,"#dae6e3"]):
        ax.barh(range(4),rows[:,j],left=start,height=.65,color=c,label=categories[j])
        for i,x in enumerate(rows[:,j]):
            if x>0: ax.text(start[i]+x/2,i,str(x),ha="center",va="center",color=(ink if j==2 else "white"),weight="bold")
        start+=rows[:,j]
    ax.set_yticks(range(4),["Qwen / out of scope","Qwen / in scope","SmolLM2 / out of scope","SmolLM2 / in scope"])
    ax.invert_yaxis();ax.set_xlim(0,256);ax.set_xlabel("Responses (256 per bar)");ax.legend(loc="lower center",bbox_to_anchor=(.5,1),ncol=3,frameon=False)
    save(fig,"argument_validity")

if __name__=="__main__":
    methods();behavior();intervention();formats()
