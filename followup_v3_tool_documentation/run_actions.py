"""Greedy free call/abstention responses for every action condition in ACTION_PROTOCOL.md."""
import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from run import CELLS, DISTRICTS, HERE, REVISION, WEIGHTS, dump_line, file_digest, make_prompt


def classify(text, name, district):
    raw = text.strip()
    unwrapped = re.sub(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", r"\1", raw, flags=re.S).strip()
    unwrapped = re.sub(r"^\s*<tool_call>\s*(.*?)\s*</tool_call>\s*$", r"\1", unwrapped, flags=re.S).strip()
    try:
        parsed = json.loads(unwrapped)
    except (ValueError, TypeError):
        parsed = None
    exact = parsed == {"name": name, "arguments": {"district": district}}
    attempt = ("<tool_call>" in raw or re.search(r'\{[^{}]*"name"\s*:', raw, flags=re.S) is not None
               or re.search(r'["\']name["\']\s*:', raw) is not None and "{" in raw
               or re.search(r"\b" + re.escape(name) + r"\s*\(", raw) is not None)
    abstain = re.sub(r"[\s_]", "", raw).upper() == "NOCALL"
    return {"attempt": attempt, "exact": exact, "abstain": abstain, "other": not (attempt or abstain)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps", choices=["cpu", "mps"])
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    assert file_digest(args.model / "model.safetensors") == WEIGHTS
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float32,
                                                  attn_implementation="eager").to(args.device).eval()
    conf = GenerationConfig(do_sample=False, max_new_tokens=72, use_cache=True,
                            eos_token_id=[151645, 151643], pad_token_id=151643)
    families = json.loads((HERE / "bank_spec.json").read_text())
    old = ("Judge whether the listed tool covers the request. The service district limits coverage. "
           "The documentation office is where the manual was written. Respond with one letter. "
           "Answer A if the tool covers the request. Answer B otherwise.")
    new = ("Call the tool if it covers the request. Otherwise respond with NO_CALL. "
           "The service district limits coverage. The documentation office is where the manual was written. ")
    start = time.monotonic()
    stamp = lambda: datetime.now(timezone.utc).isoformat()
    meta = {"started_utc": stamp(), "weights_sha256": WEIGHTS, "revision": REVISION, "dtype": "float32",
            "device": args.device, "attention": "eager", "generation": conf.to_diff_dict(),
            "sources_sha256": {f: file_digest(HERE / f) for f in ["ACTION_PROTOCOL.md", "run_actions.py", "run.py", "bank_spec.json"]}}
    (args.out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (args.out / "raw.jsonl").open("x") as fp, torch.inference_mode():
        for (fi, family), (di, (district, alt)), mode in product(enumerate(families), enumerate(DISTRICTS), ["text", "native"]):
            prompts = []
            order = (fi + di) % 2
            ending = ('For a call, return a JSON object with fields "name" and "arguments".' if mode == "text" else
                      'For a call, return the JSON object with "name" and "arguments" in <tool_call> tags.')
            for s, o in CELLS:
                rendered, _ = make_prompt(tok, family, district, alt, order, mode, "A", s, o)
                assert rendered.count(old) == 1
                prompts.append(rendered.replace(old, new + ending))
            tokens = [tok.encode(x, add_special_tokens=False) for x in prompts]
            ids = torch.tensor(tokens, device=args.device)
            outs = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=conf)[:,ids.shape[1]:].cpu().tolist()
            for i, generated in enumerate(outs):
                s,o = CELLS[i]
                end = next((j for j,t in enumerate(generated) if t in conf.eos_token_id), None)
                used = generated[:end+1] if end is not None else generated
                text = tok.decode(used[:-1] if end is not None else used, skip_special_tokens=False)
                result = {"family": family[0], "district": district, "mode": mode, "service_match": s, "office_match": o,
                          "office_first": bool(order), "prompt": prompts[i], "input_ids": tokens[i], "generated_ids": used,
                          "finish": "stop" if end is not None else "length", "text": text, "score": classify(text, family[0], district)}
                dump_line(fp, result)
            fp.flush()
            if di == 3 and mode == "native":
                print(f"{fi+1}/8 families ({time.monotonic()-start:.1f} s)", flush=True)
    meta.update(completed_utc=stamp(), elapsed_seconds=time.monotonic()-start, raw_sha256=file_digest(args.out / "raw.jsonl"))
    (args.out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
