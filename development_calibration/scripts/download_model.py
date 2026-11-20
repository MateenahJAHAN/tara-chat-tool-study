#!/usr/bin/env python3
"""Stage one pinned ungated Apache2 public model; no inference or remote code."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
os.environ["HF_HOME"]=str(ROOT/".cache"/"huggingface")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
os.environ["HF_HUB_DISABLE_XET"]="1"
import datetime,hashlib,json
from huggingface_hub import HfApi,snapshot_download
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
REVISION="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEST=ROOT/"artifacts"/"model"
def main():
 info=HfApi(token=False).model_info(MODEL,revision=REVISION,token=False)
 if info.sha!=REVISION or info.gated:raise RuntimeError("Unexpected revision/gated model")
 snapshot_download(repo_id=MODEL,revision=REVISION,token=False,local_dir=DEST,
                   cache_dir=ROOT/".cache"/"huggingface"/"hub",max_workers=2,
                   allow_patterns=["config.json","generation_config.json","model.safetensors","tokenizer.json","tokenizer_config.json","merges.txt","vocab.json","LICENSE","README.md"])
 files={}
 for p in sorted(DEST.iterdir()):
  if not p.is_file():continue
  h=hashlib.sha256()
  with p.open("rb") as f:
   for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
  files[p.name]={"sha256":h.hexdigest(),"bytes":p.stat().st_size}
 if "model.safetensors" not in files:raise RuntimeError("Expected safetensors absent")
 manifest={"staged_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"model":MODEL,"revision":REVISION,
           "ungated":True,"license_source":"Official pinned model card/LICENSE (Apache2)","token_argument":False,
           "trust_remote_code":False,"inference_performed":False,"files":files}
 (ROOT/"MODEL_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")
 print(json.dumps({"model":MODEL,"revision":REVISION,"files":len(files),"total_bytes":sum(v["bytes"] for v in files.values()),"inference_performed":False}))
if __name__=="__main__":main()

