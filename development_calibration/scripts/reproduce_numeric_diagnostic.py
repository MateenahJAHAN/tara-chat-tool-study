#!/usr/bin/env python3
"""Finite, non-modifying-hook numerical diagnostic on one declared native-train item."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
import argparse
_parser=argparse.ArgumentParser(description="Reproduce FP16/FP32 development numerical controls")
_parser.add_argument("--output-dir",required=True)
_args=_parser.parse_args()
OUT=Path(_args.output_dir).expanduser().resolve()
OUT.mkdir(parents=True,exist_ok=False)
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
import datetime,gc,hashlib,json,time,traceback
import torch
import numpy as np
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.utils import logging as hf_logging
hf_logging.disable_progress_bar()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(a,b):
 finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
 if not finite:return {"all_logits_finite":False,"max_abs":None,"rms":None,"kl":None,"top1_equal":False}
 return {"all_logits_finite":True,"max_abs":float((a-b).abs().max()),"rms":float((a-b).square().mean().sqrt()),
         "kl":float(torch.nn.functional.kl_div(torch.log_softmax(b,dim=-1),torch.softmax(a,dim=-1),reduction="batchmean")),
         "top1_equal":int(a.argmax())==int(b.argmax())}
def main():
 freeze=json.loads((ROOT/"PUBLIC_ARTIFACT_MANIFEST.json").read_text())
 if any(sha(ROOT/f)!=h for f,h in freeze["files"].items()):raise RuntimeError("D1 frozen input changed")
 with (OUT/"ATTEMPT_CONSUMED.json").open("x") as f:json.dump({"started_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"retries":0},f)
 plan=json.loads((ROOT/"protocol/D1_numeric_plan.json").read_text())
 source=json.loads(next((ROOT/"data/BFCL_v4_simple_python.json").open()))
 assert source["id"]=="simple_python_0"
 tokenizer=AutoTokenizer.from_pretrained(ROOT/"artifacts/model",local_files_only=True,trust_remote_code=False)
 functions=source["function"];catalog=json.dumps(functions,sort_keys=True)
 messages=[{"role":"system","content":"Development numerical diagnostic. Function catalogue: "+catalog}]+source["question"][0]
 rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 rows=[]
 for label,dtype in [("float16",torch.float16),("float32",torch.float32)]:
  torch.manual_seed(20261119);torch.set_num_threads(4)
  model=AutoModelForCausalLM.from_pretrained(ROOT/"artifacts/model",local_files_only=True,trust_remote_code=False,dtype=dtype,attn_implementation="eager").to("mps").eval()
  inputs=tokenizer(rendered,return_tensors="pt").to("mps");captured={};handles=[]
  with torch.inference_mode():
   a=model(**inputs,use_cache=False).logits[:,-1,:].float().cpu()
   b=model(**inputs,use_cache=False).logits[:,-1,:].float().cpu()
   def hook_for(i):
    def h(module,args,output):
     hidden=output[0] if isinstance(output,tuple) else output
     captured[i]=hidden[0,-1,:].detach().float().cpu().numpy().copy()
    return h
   try:
    for i,layer in enumerate(model.model.layers):handles.append(layer.register_forward_hook(hook_for(i)))
    c=model(**inputs,use_cache=False).logits[:,-1,:].float().cpu()
   finally:
    for h in handles:h.remove()
  finite=all(np.isfinite(x).all() for x in captured.values())
  rec={"dtype":label,"backend":"mps","input_tokens":int(inputs["input_ids"].shape[1]),"repeat_forward_metric":metric(a,b),"hook_metric":metric(a,c),
       "captured_layers":len(captured),"residual_widths":sorted({len(x) for x in captured.values()}),"all_residuals_finite":finite,
       "actual_prefill_forwards":3,"generations":0,"heldout_inference":False}
  rec["predeclared_numeric_gate_pass"]=len(captured)==28 and rec["residual_widths"]==[1536] and finite and rec["repeat_forward_metric"]["all_logits_finite"] and rec["hook_metric"]["all_logits_finite"] and rec["hook_metric"]["max_abs"]<=.001 and rec["hook_metric"]["top1_equal"]
  rows.append(rec)
  with (OUT/"RAW_DIAGNOSTICS.jsonl").open("a") as f:f.write(json.dumps(rec)+"\n")
  print(json.dumps(rec),flush=True)
  del model,inputs,a,b,c;gc.collect();torch.mps.empty_cache()
 result={"phase":"D1_NUMERIC_DEVELOPMENT_ONLY","completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"rows":rows,
         "selected_pre_main_candidate":"float32" if rows[1]["predeclared_numeric_gate_pass"] else None,
         "selection_rule":"Prefer float32 if finite28-layer1536-width hook identity maxabs<=.001 and stabletop1; do not relax guard after results.",
         "scientific_effect_claim":False,"heldout_inference_performed":False,"model_generations":0}
 (OUT/"RESULT.json").write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__":
 try:main()
 except Exception:
  (OUT/"FAILURE.json").write_text(json.dumps({"recorded_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"traceback":traceback.format_exc(),"heldout_inference":False},indent=2)+"\n")
  raise

