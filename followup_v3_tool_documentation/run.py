"""Run the full paired schema-field and block-14 intervention experiment."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import platform
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
WEIGHTS = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
DISTRICTS = [("East", "West"), ("West", "East"), ("North", "South"), ("South", "North")]
CELLS = list(product([False, True], repeat=2))


def file_digest(path):
    h = sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_line(fp, record):
    fp.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")


def make_prompt(tokenizer, family, district, alt, order, mode, positive_letter, service, office):
    name, job, target = family
    fields = [f"Service district: {district if service else alt}.", f"Documentation office district: {district if office else alt}."]
    if order:
        fields.reverse()
    tool = {"name": name, "description": f"Use this service to {job}. " + " ".join(fields),
            "parameters": {"type": "object", "properties": {"district": {"type": "string"}}, "required": ["district"]}}
    system = ("Judge whether the listed tool covers the request. The service district limits coverage. "
              "The documentation office is where the manual was written. Respond with one letter. ")
    negative_letter = "B" if positive_letter == "A" else "A"
    system += f"Answer {positive_letter} if the tool covers the request. Answer {negative_letter} otherwise."
    if mode == "text":
        system += "\n\nTool catalog:\n" + json.dumps(tool, sort_keys=True)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Please find {target} in the {district} district."}]
    rendered = tokenizer.apply_chat_template(messages, tools=[tool] if mode == "native" else None,
                                               tokenize=False, add_generation_prompt=True)
    return rendered, tokenizer.encode(rendered, add_special_tokens=False)


def diff_position(ids, donor):
    assert len(ids) == len(donor)
    changed = [j for j, (r, d) in enumerate(zip(ids, donor)) if r != d]
    assert len(changed) == 1, changed
    return changed[0]


def intervention_hook(donors, positions):
    def patch(_module, _args, value):
        rows = torch.arange(value.shape[0], device=value.device)
        result = value.clone()
        result[rows, positions] = value[donors, positions]
        return result
    return patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="mps", choices=["mps", "cpu"])
    args = p.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    start = time.monotonic()
    assert file_digest(args.model / "model.safetensors") == WEIGHTS
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                  attn_implementation="eager", local_files_only=True)
    model = model.to(args.device).eval()
    letters = {s: tok.encode(s, add_special_tokens=False) for s in "AB"}
    assert all(len(ids) == 1 for ids in letters.values())
    letter_ids = [letters[s][0] for s in "AB"]
    layers = model.model.layers
    assert len(layers) == 28
    families = json.loads((HERE / "bank_spec.json").read_text())
    def stamp():
        return datetime.now(timezone.utc).isoformat()
    run = {"started_utc": stamp(), "device": args.device, "dtype": "float32", "attention": "eager",
           "model": "Qwen/Qwen2.5-1.5B-Instruct", "revision": REVISION, "weights_sha256": WEIGHTS,
           "model_files_sha256": {f.name: file_digest(f) for f in args.model.iterdir() if f.suffix in (".txt", ".json")},
           "sources_sha256": {f: file_digest(HERE / f) for f in ["PROTOCOL.md", "bank_spec.json", "run.py"]},
           "versions": {"torch": str(torch.__version__), "transformers": transformers.__version__, "python": platform.python_version()},
           "letter_ids": letter_ids, "family_order": [f[0] for f in families]}
    path = args.out / "run.json"
    path.write_text(json.dumps(run, indent=2) + "\n")
    with (args.out / "raw.jsonl").open("x") as fp, torch.inference_mode():
        for (fi, family), (di, (district, alt)), mode, positive_letter in product(enumerate(families), enumerate(DISTRICTS), ["text", "native"], "AB"):
            group = (family[0], district, mode, positive_letter)
            order = (fi + di) % 2
            prompts = [make_prompt(tok, family, district, alt, order, mode, positive_letter, service, office) for service, office in CELLS]
            tokens = [x[1] for x in prompts]
            ids = torch.tensor(tokens, dtype=torch.long, device=args.device)
            donor_rows = {"service": [CELLS.index((not s, o)) for s, o in CELLS],
                          "office": [CELLS.index((s, not o)) for s, o in CELLS]}
            positions = {field: [diff_position(tokens[i], tokens[j]) for i, j in enumerate(donors)]
                         for field, donors in donor_rows.items()}
            for i, (s, o) in enumerate(CELLS):
                for field, match in [("service", s), ("office", o)]:
                    word = tok.decode([tokens[i][positions[field][i]]]).strip()
                    assert word == (district if match else alt), [field, word, group]
            def measure(module=None, field=None, identity=False):
                if module is not None:
                    donors = torch.tensor(list(range(4)) if identity else donor_rows[field], device=args.device)
                    pos = torch.tensor(positions[field], device=args.device)
                    handle = module.register_forward_hook(intervention_hook(donors, pos))
                try:
                    logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                    selected = logits[:, letter_ids].cpu().tolist()
                    top = logits.argmax(-1).cpu().tolist()
                    mass = logits.softmax(-1)[:, letter_ids].sum(-1).cpu().tolist()
                    return [{"logit_a": a, "logit_b": b, "margin": (a-b if positive_letter == "A" else b-a),
                             "top_id": t, "top_token": tok.decode([t]), "ab_probability": prob}
                            for (a,b), t, prob in zip(selected, top, mass)]
                finally:
                    if module is not None:
                        handle.remove()
            values = {"baseline": measure(), "patch_service": measure(layers[13], "service"),
                      "patch_office": measure(layers[13], "office")}
            if di == 0 and mode == "text" and positive_letter == "A":
                values["self"] = measure(layers[13], "service", identity=True)
                values["embed_service"] = measure(model.model.embed_tokens, "service")
                values["embed_office"] = measure(model.model.embed_tokens, "office")
            for i, (s, o) in enumerate(CELLS):
                dump_line(fp, {"family": family[0], "district": district, "mode": mode, "positive_letter": positive_letter,
                               "service_match": s, "office_match": o, "office_first": bool(order), "prompt": prompts[i][0], "tokens": tokens[i],
                               "service_token_index": positions["service"][i], "office_token_index": positions["office"][i],
                               "scores": {name: rows[i] for name, rows in values.items()}})
            fp.flush()
            if di == 3 and mode == "native" and positive_letter == "B":
                print(f"{fi+1}/8 tool families complete ({time.monotonic()-start:.1f} seconds)", flush=True)
    run.update(completed_utc=stamp(), elapsed_seconds=time.monotonic()-start, raw_sha256=file_digest(args.out / "raw.jsonl"))
    path.write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
