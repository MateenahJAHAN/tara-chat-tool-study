"""Plot actual authorized reports only. Default previews requirements, no data/figures.

Use an existing plotting interpreter; no inference imports, installs or refitting.
All table rows/endpoints, including zeros, unknowns, errors and unrun status, remain.
"""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reporting_io import CONDITIONS, ENDPOINTS, approved_report, authorize, ensure_public_payload, local_path, read_json, sha

NAMES = {"TEXT_CATALOGUE": "Text catalogue", "NATIVE_TOOLS": "Native tools", "QUERY_ONLY_CONTROL": "Query only"}
COLORS = ("#2563A6", "#D47B22")
CAUTION = "Selected finite bank; exploratory descriptives. Intervals have no population-coverage guarantee."
PAIRED = (("false_call", "label1", "False call", "primary_false_call_native_minus_text"),
          ("strict_abstention", "label1", "Explicit NO_CALL", "S1_operational_abstention"),
          ("correct_call", "label0", "Reference-correct call", "S2_correct_call"))


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def textnum(value, digits=3):
    return format(value, "." + str(digits) + "f") if numeric(value) else "unavailable"


def error_point(ax, x, y, interval, color, label=None):
    if not numeric(y):
        return
    if interval and all(numeric(v) for v in interval):
        lo, hi = interval
        ax.plot([x,x],[lo,hi],color=color)
        ax.plot([x-.035,x+.035],[lo,lo],color=color)
        ax.plot([x-.035,x+.035],[hi,hi],color=color)
        ax.plot(x,y,"o",color=color,label=label)
    else:
        ax.plot(x, y, "o", color=color, label=label)


def save(fig, out, name, files):
    for suffix in ("png", "pdf", "svg"):
        path = out / (name + "." + suffix)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        files.append(path.name)


def export_tables(context, out):
    endpoints, statuses, pairs = [], [], []
    for condition, groups in context["condition_summaries"].items():
        for label, summary in groups.items():
            for key, value in sorted(summary["endpoints"].items()):
                endpoints.append({"phase": context["phase"], "condition": condition, "label": label,
                                  "endpoint": key, **value, "n_groups": summary["n_groups"]})
            for family in ("stop_reason_counts", "n_attempts_distribution", "validity_issue_counts", "parser_diagnostic_counts",
                           "format_diagnostic_counts", "attempt_kind_counts"):
                for key, count in summary[family].items():
                    statuses.append({"phase":context["phase"], "condition":condition, "label":label,
                                     "status_family":family, "status":key, "count":count})
            for key in ("error_records", "scorer_error_records"):
                statuses.append({"phase":context["phase"], "condition":condition, "label":label,
                                 "status_family":"record_status", "status":key, "count":summary[key]})
    for label, rows in context["paired_counts"].items():
        for endpoint, counts in rows.items():
            pairs.append({"phase":context["phase"], "label":label, "endpoint":endpoint, **counts})
    for name, rows in (("all_endpoints.csv", endpoints), ("all_statuses.csv", statuses), ("all_paired_counts.csv", pairs)):
        if rows:
            with (out / name).open("x", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return endpoints


def plot_behaviour(plt, behaviour, context, out, files):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    for ax, (endpoint, label, title, _) in zip(axes, PAIRED):
        for index, condition in enumerate(CONDITIONS):
            value = behaviour["rates"][condition][label].get(endpoint)
            if value is None:
                ax.text(index, .5, "Not reported", ha="center", rotation=90)
                continue
            k, n = value["k"], value["n"]
            if not isinstance(k, int) or not isinstance(n, int) or not 0 <= k <= n or n == 0:
                raise RuntimeError("Invalid reported count/denominator")
            error_point(ax, index, k/n, value.get("descriptive_wilson_binomial_reference95"), COLORS[index])
            ax.annotate(f"{k}/{n}", (index, k/n), xytext=(0, 10), textcoords="offset points", ha="center")
        ax.set(title=title, ylim=(-.04, 1.12), xlim=(-.5, 1.5), xticks=[0,1],
               xticklabels=["Text", "Native"])
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Observed proportion")
    counts = context["scheduled_counts"]
    scope = " (selection-used)" if context["phase"] == "DEV" else ""
    fig.suptitle(context["phase"] + scope + f" - {counts['rows']} selected items / {counts['groups']} components")
    protocol=[]
    for condition in CONDITIONS:
        value=context['condition_summaries'][condition]['label0']['endpoints']['protocol_format_error']
        protocol.append(NAMES[condition]+": "+str(value['k'])+"/"+str(value['n_boolean']))
    fig.text(.5, -.025, "95% descriptive Wilson binomial-reference intervals. " + CAUTION
             +"\nSemantic correctness is separate from protocol compliance. Relevant-call protocol errors: "+"; ".join(protocol)+".", ha="center", fontsize=8)
    save(fig, out, "01_behaviour_rates", files)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    annotations = []
    for index, (endpoint, label, title, report_key) in enumerate(PAIRED):
        paired = context["paired_counts"][label][endpoint]
        n = paired["n_scheduled"] - paired["not_applicable_or_missing_pairs"]
        if n == 0:
            annotations.append(title + ": unavailable")
            continue
        difference = (paired["a0_b1"] - paired["a1_b0"]) / n
        result = behaviour.get(report_key) if report_key.startswith("primary_") else behaviour.get("secondary", {}).get(report_key)
        interval = result.get("ci_b_minus_a") if result else None
        if interval and abs(interval["mean"] - difference) > 1e-8:
            raise RuntimeError("Existing paired estimate and raw scored pair counts disagree")
        lohi = [interval["lo"], interval["hi"]] if interval else None
        if lohi and all(numeric(v) for v in lohi):
            ax.plot(lohi,[index,index],color=COLORS[1]);ax.plot(difference,index,"o",color=COLORS[1])
        else:
            ax.plot(difference, index, "o", color=COLORS[1])
        annotations.append(f"{title}: n={n}; pairs 00/01/10/11=" + "/".join(str(paired[k]) for k in ("a0_b0","a0_b1","a1_b0","a1_b1"))
                           + ("; degenerate resampling" if interval and interval.get("degenerate") else ""))
    ax.axvline(0, color=".5", lw=1)
    ax.set(xlim=(-1.05,1.05), yticks=range(3), yticklabels=[row[2] for row in PAIRED],
           xlabel="Native minus text (observed proportion)", title=context["phase"]+" paired differences")
    ax.grid(axis="x", alpha=.2)
    fig.text(.02,-.08,"\n".join(annotations)+"\n95% group-resampling descriptors; "+CAUTION,fontsize=8)
    save(fig,out,"02_paired_differences",files)
    plt.close(fig)


def plot_endpoint_pages(plt, PdfPages, rows, context, out, files):
    # Include every endpoint, both labels and both conditions, including zeros
    # and unavailable fields. CSV preserves complete nonboolean status tallies.
    path = out / "03_all_endpoint_counts.pdf"
    with PdfPages(path) as pdf:
        for start in range(0,len(rows),28):
            page=rows[start:start+28]
            fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis("off")
            cells=[[NAMES[r["condition"]],r["label"],r["endpoint"],f"{r['k']}/{r['n_boolean']}",str(r["n_scheduled"]),
                    str(r["not_applicable"]),str(r["missing_or_nonboolean"])] for r in page]
            table=ax.table(cellText=cells,colLabels=["Interface","Source label","Endpoint","True / boolean n","Scheduled","N/A","Missing"],
                           colWidths=[.17,.11,.27,.16,.1,.075,.075],loc="upper center",cellLoc="left")
            table.auto_set_font_size(False);table.set_fontsize(7.5);table.scale(1,1.35)
            ax.set_title(context["phase"]+" — all retained scorer endpoints (no zero/status filtering)",pad=20)
            fig.text(.03,.025,"label0: simple/relevant source; label1: irrelevance source. These are benchmark labels, not human ratings.\n"+CAUTION,fontsize=8)
            fig.text(.94,.025,f"Page {start//28+1}/{math.ceil(len(rows)/28)}",ha="right",fontsize=8)
            pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
    files.append(path.name)


def plot_probes(plt, report, context, out, files):
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    image=None
    for ax,kind in zip(axes,("MM","LR")):
        values=[]
        for src in CONDITIONS:
            values.append([report["matrix"][kind+":"+src+"->"+tgt]["auroc"] for tgt in CONDITIONS])
        if any(not numeric(value) for row in values for value in row):
            raise RuntimeError("Undefined selected probe AUROC retained; no fabricated heatmap")
        image=ax.imshow(values,vmin=0,vmax=1,cmap="Blues")
        for i,src in enumerate(CONDITIONS):
            for j,tgt in enumerate(CONDITIONS):
                r=report["matrix"][kind+":"+src+"->"+tgt];ci=r.get("ci",{})
                layer=(context.get("probe_selection") or {}).get(kind+"__"+src,{}).get("layer","not reported")
                ax.text(j,i,f"AUC {textnum(r['auroc'])}\n[{textnum(ci.get('lo'))}, {textnum(ci.get('hi'))}]\nBA {textnum(r.get('bacc'))}; n={r.get('n','not reported')}\nselected layer {layer}",ha="center",va="center",fontsize=8,color="white" if r['auroc']>.65 else "black")
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Text","Native"],yticklabels=["Text","Native"],xlabel="Test interface",ylabel="Train interface",title=kind)
    axes[1].set_ylabel("")
    fig.subplots_adjust(wspace=.25)
    fig.colorbar(image,ax=axes,label="AUROC",fraction=.035,pad=.04)
    fig.suptitle("HELDOUT selected probe transfer — exploratory")
    fig.text(.02,-.01,"Layers/C selected on native DEV only. Intervals are descriptive group-resampling; skipped single-class draws retained in audit.\n"+CAUTION,fontsize=8)
    save(fig,out,"04_probe_transfer",files);plt.close(fig)

    labels,estimates,intervals,notes=[],[],[],[]
    for name,r in sorted(report.get("baselines",{}).items()):
        labels.append("Surface: "+name);estimates.append(r.get("auroc"));intervals.append(r.get("ci"));notes.append("DEV-selected C; n="+str(context["scheduled_counts"]["rows"]))
    for kind in ("MM","LR"):
        key=kind+":QUERY_ONLY_CONTROL->QUERY_ONLY_CONTROL"
        if key in report.get("matrix",{}):
            r=report["matrix"][key];labels.append(kind+" query-only activation");estimates.append(r.get("auroc"));intervals.append(r.get("ci"));notes.append("n="+str(r.get("n","not reported")))
    for key,r in sorted(report.get("controls",{}).items()):
        for family in ("permuted_label","random_direction"):
            val=r.get(family)
            labels.append(key+" "+family);estimates.append(val.get("median") if val else None);intervals.append(None)
            notes.append("not run/unavailable" if not val else f"n={val['n']}; median; max={textnum(val['max'])}; frac≥observed={textnum(val.get('frac_ge_observed'))}")
    fig,ax=plt.subplots(figsize=(12,max(6,len(labels)*.32)))
    for i,(value,ci,note) in enumerate(zip(estimates,intervals,notes)):
        if numeric(value):
            if ci and numeric(ci.get("lo")) and numeric(ci.get("hi")):
                ax.plot([ci['lo'],ci['hi']],[i,i],color=COLORS[0]);ax.plot(value,i,"o",color=COLORS[0])
            else:ax.plot(value,i,"o",color=COLORS[1])
        ax.text(1.025,i,note,va="center",fontsize=7)
    ax.axvline(.5,color=".5",ls="--",lw=1);ax.set(xlim=(0,1),yticks=range(len(labels)),yticklabels=labels,xlabel="Heldout AUROC (controls: median, not confidence interval)",title="Query/surface and all reported permutation/random controls")
    ax.invert_yaxis();fig.text(.02,-.01,CAUTION+" Benchmark source/style confounding remains.",fontsize=8)
    save(fig,out,"05_query_surface_controls",files);plt.close(fig)


def plot_dev_probes(plt, context, out, files):
    matrix=context.get('dev_transfer_matrix_selection_used')
    selected=context.get('probe_selection') or {}
    if not matrix:
        return 'DEV transfer metadata unavailable; no transfer figure generated.'
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8));image=None
    for ax,kind in zip(axes,('MM','LR')):
        values=[[matrix[kind+':'+src+'->'+tgt] for tgt in CONDITIONS] for src in CONDITIONS]
        if any(not numeric(v) for row in values for v in row):
            raise RuntimeError('Undefined DEV AUROC retained; cannot invent heatmap values')
        image=ax.imshow(values,vmin=0,vmax=1,cmap='Blues')
        for i,src in enumerate(CONDITIONS):
            meta=selected.get(kind+'__'+src,{})
            for j,tgt in enumerate(CONDITIONS):
                ax.text(j,i,f"AUROC {values[i][j]:.3f}\nselected layer {meta.get('layer','not reported')}\nC={meta.get('C')}" if kind=='LR'
                        else f"AUROC {values[i][j]:.3f}\nselected layer {meta.get('layer','not reported')}",ha='center',va='center',fontsize=9,
                        color='white' if values[i][j]>.65 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Text','Native'],yticklabels=['Text','Native'],
               xlabel='Evaluate interface',ylabel='Train interface',title=kind)
    axes[1].set_ylabel("")
    fig.subplots_adjust(wspace=.25)
    fig.colorbar(image,ax=axes,label='AUROC',fraction=.035,pad=.04)
    n=context['scheduled_counts'];fig.suptitle('Native DEV selected transfer - selection-used and optimistic')
    fig.text(.02,-.025,f"Actual DEV: {n['rows']} items / {n['groups']} components; {n['label0']} relevant-source + {n['label1']} irrelevant-source.\n"
             +'Layer/C selection used this same DEV bank. No uncertainty interval is supplied for these selection-used summaries.\n'+CAUTION,fontsize=8)
    save(fig,out,'04_dev_selected_transfer',files);plt.close(fig)
    labels,values,notes=[],[],[]
    for name,meta in sorted((context.get('surface_selection') or {}).items()):
        labels.append('Surface: '+name);values.append(meta.get('dev_auroc'));notes.append('DEV-selected C='+str(meta.get('C')))
    for kind in ('MM','LR'):
        key=kind+'__QUERY_ONLY_CONTROL';meta=selected.get(key)
        if meta:
            labels.append(kind+' query-only activation');values.append(meta.get('dev_auroc'))
            notes.append('DEV-selected layer '+str(meta.get('layer'))+'; C='+str(meta.get('C')))
    fig,ax=plt.subplots(figsize=(11,5))
    for i,(value,note) in enumerate(zip(values,notes)):
        if numeric(value):ax.plot(value,i,'o',color=COLORS[0])
        ax.text(1.025,i,note,va='center',fontsize=8)
    ax.axvline(.5,color='.5',ls='--',lw=1);ax.set(xlim=(0,1),yticks=range(len(labels)),yticklabels=labels,
                  xlabel='Actual DEV AUROC (selection-used; no interval supplied)',title='DEV query/surface controls - selection-used and optimistic')
    ax.invert_yaxis();fig.text(.02,-.07,f"n={n['rows']} selected items; groups={n['groups']}. Source/style confounding remains.\n"
             +'Permutation/random heldout evaluations are not authorized or shown in this DEV report.\n'+CAUTION,fontsize=8)
    save(fig,out,'05_dev_query_surface_controls',files);plt.close(fig)
    return None


def plot_steering(plt, behaviour, context, out, files):
    statuses=context.get("steering_status")
    if statuses is None:
        return "Steering status metadata not yet produced; no steering figure generated."
    fig,axes=plt.subplots(1,2,figsize=(13,9.2 if context["phase"] == "HELDOUT" else 6))
    for ax,condition in zip(axes,CONDITIONS):
        ax.axis("off");record=statuses.get(condition,{})
        lines=[NAMES[condition],"Actual DEV status: "+str(record.get("status","not recorded")),
               "DEV-selected probe layer (0-index): "+str(record.get('layer','not recorded')),
               "Entry gate pass: "+str(record.get("entry_pass","not recorded")),
               "Entry metrics: "+json.dumps(record.get("entry",{}),sort_keys=True),
               "Selected configuration: "+json.dumps(record.get("selected"),sort_keys=True)]
        sweep=record.get("sweep",[])
        lines.append("Actual evaluated DEV candidates: "+str(len(sweep)))
        if record.get('status')=='STEERING_GATE_FAILED':
            lines.append('No evaluated candidate met all frozen DEV eligibility gates.')
        for row in sweep:
            lines.append("alpha="+str(row.get('alpha'))+" sign="+str(row.get('sign'))+" eligible="+str(row.get('eligible'))
                         +"\n false="+textnum(row.get('dev_false_call'))+" correct="+textnum(row.get('dev_correct_call'))
                         +" invalid="+textnum(row.get('dev_invalid_or_empty'))+" KL="+(format(row['completion_kl'],'.3g') if numeric(row.get('completion_kl')) else 'unavailable'))
        held=behaviour.get("steering",{}).get(condition)
        if held:
            lines.append("Heldout bounded criteria met: "+str(held.get('bounded_intervention_criteria_met'))+" (descriptive only)")
            for name,row in sorted(held.get("configs",{}).items()):
                reduction=(row.get('false_call_reduction') or {}).get('ci_b_minus_a',{})
                false=row.get('false_call_reduction') or {}
                utility=row.get('correct_call_change') or {}
                uf=utility.get('ci_b_minus_a',{})
                nf=reduction.get('n_items');nu=uf.get('n_items')
                def count_text(rate,n):
                    return str(round(rate*n))+"/"+str(n) if numeric(rate) and isinstance(n,int) else "unavailable"
                kl=(held.get('completion_kl') or {}).get(name)
                lines.append(name+": false "+count_text(false.get('rate_b'),nf)+" → "+count_text(false.get('rate_a'),nf)
                             +"; correct "+count_text(utility.get('rate_a'),nu)+" → "+count_text(utility.get('rate_b'),nu)
                             +"\n reduction="+textnum(reduction.get('mean'))+"; invalid Δ="+textnum(row.get('invalid_increase'))
                             +"; KL="+(format(kl,'.3g') if numeric(kl) else 'unavailable'))
            if all((row.get('false_call_reduction') or {}).get('ci_b_minus_a',{}).get('degenerate') for row in held.get('configs',{}).values()):
                lines.append("No paired false-call flips. Degenerate resampling is not evidence of equivalence.")
        else:lines.append("No heldout steering result in this authorized report.")
        ax.text(0,1,"\n\n".join(lines),va="top",fontsize=8,wrap=True)
    fig.suptitle(context["phase"]+" — steering actual status, including skipped/failed gates")
    fig.text(.02,-.02,CAUTION+" A failed gate is retained; no positive intervention result is imputed.",fontsize=8)
    save(fig,out,"06_steering_status",files);plt.close(fig)
    return None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--main-root',type=Path);ap.add_argument('--phase',choices=['DEV','HELDOUT'])
    ap.add_argument('--authorization',type=Path);ap.add_argument('--output-dir',type=Path);ap.add_argument('--render',action='store_true')
    args=ap.parse_args()
    if not args.render:
        print(json.dumps({'results_read':False,'figures_created':False,'requires':['explicit --render and completed-phase authorization','actual behaviour report + reporting_context_PHASE.json','optional explicitly hashed heldout_probe_results.json'],'outputs':['rate and paired raw counts/descriptive intervals','all endpoint/status CSV + endpoint PDF','all selected 2x2 probes and query/surface/control summaries when present','actual steering status/gate failures, never imputed success']},indent=2));return
    if not all((args.main_root,args.phase,args.authorization,args.output_dir)):
        ap.error('Actual mode needs main root, phase, authorization and new output directory')
    root,auth,auth_sha=authorize(args.main_root,args.phase,args.authorization)
    relative=auth.get('behaviour_report','reports/behaviour_'+args.phase.lower()+'.json')
    behaviour=approved_report(root,auth,relative)
    context_path=local_path(root,'reports/reporting_context_'+args.phase.lower()+'.json');context=read_json(context_path)
    if (context['phase']!=args.phase or context.get('design_id')!=auth['design_id']
            or context.get('split_manifest_sha256')!=auth['split_manifest_sha256']
            or context['authorization_sha256']!=auth_sha or context['input_sha256'].get(relative)!=auth['approved_reports'][relative]):
        raise RuntimeError('Context is stale or does not match this phase authorization/report')
    ensure_public_payload(context)
    probe_relative='reports/heldout_probe_results.json'
    probes=approved_report(root,auth,probe_relative) if args.phase=='HELDOUT' and probe_relative in auth['approved_reports'] else None
    out=args.output_dir.resolve()
    if out==root or out.is_relative_to(root):
        raise RuntimeError('Figures must be outside main_v1 and its frozen inputs')
    out.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    files=[];rows=export_tables(context,out);plot_behaviour(plt,behaviour,context,out,files)
    plot_endpoint_pages(plt,PdfPages,rows,context,out,files)
    if probes:plot_probes(plt,probes,context,out,files)
    missing_items=[]
    if args.phase=='DEV':
        dev_missing=plot_dev_probes(plt,context,out,files)
        if dev_missing:missing_items.append(dev_missing)
        missing_items.append('Heldout probe/control evaluations not authorized/read in DEV phase.')
    missing=plot_steering(plt,behaviour,context,out,files)
    if missing:missing_items.append(missing)
    audit={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':args.phase,
           'design_id':context['design_id'],'split_manifest_sha256':context['split_manifest_sha256'],
           'authorization_sha256':auth_sha,'context_sha256':sha(context_path),'approved_reports':auth['approved_reports'],
           'script_sha256':sha(Path(__file__)),'matplotlib_version':matplotlib.__version__,'python_version':sys.version.split()[0],
           'statistical_unit':context['statistical_unit'],'caution':CAUTION,'scoring_version':context.get('scoring_version'),
           'all_endpoints_retained':True,'missing_or_unrun':context.get('unavailable',[])+missing_items,
           'outputs':{name:sha(out/name) for name in files},'new_statistical_tests':False,'model_calls':0,'human_authorship_claim':None}
    ensure_public_payload(audit)
    (out/'FIGURE_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'phase':args.phase,'design_id':context['design_id'],'figures':len(files),'output_directory_name':out.name,'model_calls':0}))


if __name__=='__main__':
    main()
