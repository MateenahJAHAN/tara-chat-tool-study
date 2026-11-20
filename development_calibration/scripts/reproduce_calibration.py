#!/usr/bin/env python3
"""Actual native-training-only development inference and non-modifying residual-hook checks."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
import argparse
_parser=argparse.ArgumentParser(description="Reproduce declared development-only BFCL calibration")
_parser.add_argument("--output-dir",required=True,help="A NEW output directory; never overwrite retained results")
_args=_parser.parse_args()
OUT=Path(_args.output_dir).expanduser().resolve()
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]="1"
os.environ["HF_HOME"]=str(ROOT/".cache"/"huggingface")
import datetime,hashlib,json,time
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.utils import logging as hf_logging
hf_logging.disable_progress_bar()

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()

def normalize(value):
 if isinstance(value,dict):
  return {k:({"dict":"object","float":"number"}.get(v,v) if k=="type" and isinstance(v,str) else normalize(v)) for k,v in value.items()}
 if isinstance(value,list):return [normalize(x) for x in value]
 return value

def main():
 plan=json.loads((ROOT/"protocol/C0_v2_plan.json").read_text())
 freeze_path=ROOT/"PUBLIC_ARTIFACT_MANIFEST.json"
 if not freeze_path.exists():raise RuntimeError("Public source/artifact manifest required")
 frozen=json.loads(freeze_path.read_text())
 if any(sha(ROOT/f)!=h for f,h in frozen["files"].items()):raise RuntimeError("Published artifact input changed")
 out=OUT;out.mkdir(parents=True,exist_ok=False)
 (out/"ATTEMPT_CONSUMED.json").write_text(json.dumps({"started_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"automatic_retries":0})+"\n")
 model_manifest=json.loads((ROOT/"provenance/model_artifact_manifest.json").read_text())
 if any(sha(ROOT/"artifacts/model"/f)!=x["sha256"] for f,x in model_manifest["files"].items()):raise RuntimeError("Staged model file mismatch")
 if not torch.backends.mps.is_available():raise RuntimeError("Declared MPS backend unavailable; no silent fallback")
 torch.manual_seed(plan["seed"]);torch.set_num_threads(4)
 start=time.monotonic()
 tokenizer=AutoTokenizer.from_pretrained(ROOT/"artifacts/model",local_files_only=True,trust_remote_code=False)
 model=AutoModelForCausalLM.from_pretrained(ROOT/"artifacts/model",local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation="eager")
 model.to("mps");model.eval()
 layers=model.model.layers
 selected=[]
 for filename in ["BFCL_v4_simple_python.json","BFCL_v4_irrelevance.json"]:
  for line in (ROOT/"data"/filename).read_text().splitlines():
   row=json.loads(line)
   if row["id"] in plan["selected_native_training_ids"]:
    selected.append((filename,row))
 if len(selected)!=2:raise RuntimeError("Declared C0 IDs missing or duplicated")
 records=[];vectors={}
 for source,row in selected:
  functions=normalize(row["function"])
  for condition in plan["conditions"]:
   if time.monotonic()-start>plan["maximum_wall_seconds"]:raise RuntimeError("C0 declared wall budget reached")
   user_messages=row["question"][0]
   system=plan["policy"]
   tools=[{"type":"function","function":f} for f in functions]
   if condition=="TEXT_CATALOGUE":
    system+="\nFunction catalogue (JSON):\n"+json.dumps(tools,ensure_ascii=False,sort_keys=True)
   messages=[{"role":"system","content":system}]+user_messages
   rendered=tokenizer.apply_chat_template(messages,tools=tools if condition=="NATIVE_TOOLS" else None,tokenize=False,add_generation_prompt=True)
   inputs=tokenizer(rendered,return_tensors="pt").to("mps")
   if inputs["input_ids"].shape[1]>2048:raise RuntimeError("C0 context exceeds fixed bound")
   captured={};handles=[]
   with torch.inference_mode():
    baseline=model(**inputs,use_cache=False).logits[:,-1,:].float().cpu()
    def hook_for(index):
     def hook(module,args,output):
      hidden=output[0] if isinstance(output,tuple) else output
      if not isinstance(hidden,torch.Tensor) or hidden.ndim!=3:raise RuntimeError("Unexpected decoder hook shape")
      captured[index]=hidden[0,-1,:].detach().float().cpu().numpy().copy()
     return hook
    try:
     for index,layer in enumerate(layers):handles.append(layer.register_forward_hook(hook_for(index)))
     hooked=model(**inputs,use_cache=False).logits[:,-1,:].float().cpu()
    finally:
     for handle in handles:handle.remove()
    delta=float((baseline-hooked).abs().max())
    finite=all(np.isfinite(v).all() for v in captured.values())
    if len(captured)!=len(layers) or not finite or delta>1e-3:
     raise RuntimeError("Non-modifying hook control failed")
    torch.mps.synchronize();t0=time.monotonic()
    generated=model.generate(**inputs,do_sample=False,max_new_tokens=96,pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    torch.mps.synchronize();elapsed=time.monotonic()-t0
   suffix=generated[0,inputs["input_ids"].shape[1]:].detach().cpu().tolist()
   text=tokenizer.decode(suffix,skip_special_tokens=True)
   tid=row["id"]+"__"+condition
   for index,v in captured.items():vectors[tid+"__layer"+str(index)]=v
   rec={"id":row["id"],"source_file":source,"condition":condition,"development_only":True,
        "rendered_prompt":rendered,"prompt_sha256":hashlib.sha256(rendered.encode()).hexdigest(),
        "input_tokens":int(inputs["input_ids"].shape[1]),"generated_token_ids":suffix,"generated_text":text,
        "generated_tokens":len(suffix),"hit_token_ceiling":len(suffix)>=96,
        "generation_elapsed_s":elapsed,"residual_layers_captured":len(captured),"residual_width":len(captured[0]),
        "all_residuals_finite":finite,"zero_change_hook_logit_max_abs_difference":delta,
        "expected_should_call_from_BFCL_source":source=="BFCL_v4_simple_python.json",
        "expected_function_name":functions[0]["name"] if source=="BFCL_v4_simple_python.json" else None,
        "formal_human_label":None,"external_function_executed":False}
   records.append(rec)
   with (out/"raw_generations.jsonl").open("a") as f:f.write(json.dumps(rec)+"\n")
   print(json.dumps({"calibration_generation_complete":tid,"tokens":len(suffix),"residual_layers":len(captured),"hook_logit_difference":delta}),flush=True)
 np.savez_compressed(out/"last_input_token_residuals.npz",**vectors)
 result={"phase":plan["phase"],"completed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"actual_model":plan["model"],"model_revision":plan["revision"],"actual_backend":"mps","actual_dtype":"float32","torch":torch.__version__,"scheduled_generations":4,"completed_generations":len(records),"scheduled_nonmodifying_prefill_forwards":8,"captured_vectors":len(vectors),"elapsed_s":time.monotonic()-start,"heldout_inference_performed":False,"scientific_claims":False,"next":"Review Opus prospective study, preserve C0 as development-only and freeze main before heldout."}
 (out/"CALIBRATION_RESULT.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result),flush=True)
if __name__=="__main__":main()

