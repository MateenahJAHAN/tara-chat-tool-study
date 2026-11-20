"""Run the frozen benign task bank on the original pinned TARA checkpoint."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from pilot import HERE, MODES, SEED, build_cases, decode, estimate, request_body, score, validate_bank


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
WEIGHTS_SHA256 = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
DEFAULT = HERE.parent.parent / "MATEENAH_JOURNEY_AND_PLAN/03_Research_Projects/TARA_SUBSTANTIVE_STUDY_2026-11-19/artifacts/model"
NATIVE_TAG = re.compile(r"\s*<tool_call>\s*(\{.*?\})\s*</tool_call>\s*", re.S)


def native_response(text, mode):
    if mode != "native":
        return {"content": text}, False
    tagged = NATIVE_TAG.fullmatch(text)
    if tagged:
        try:
            call = decode(tagged.group(1))
            if set(call) == {"name", "arguments"} and isinstance(call["arguments"], dict):
                return {"content": "", "tool_calls": [{"type": "function", "function": call}]}, True
        except (TypeError, ValueError):
            # Keep malformed text for the original scorer.
            pass
    return {"content": text}, False


def main():
    model_path = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT
    if not torch.backends.mps.is_available():
        raise RuntimeError("Declared MPS backend unavailable")
    bank = json.loads((HERE / "bank.json").read_text())
    validate_bank(bank)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, dtype=torch.float32, attn_implementation="eager",
    ).to("mps").eval()
    output = HERE / "results_qwen25"
    output.mkdir(exist_ok=False)
    config = GenerationConfig(do_sample=False, max_new_tokens=96, use_cache=True, eos_token_id=[151645, 151643], pad_token_id=151643)
    meta = {
        "model": MODEL,
        "revision": REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "device": "mps",
        "dtype": str(next(model.parameters()).dtype),
        "attention": "eager",
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "template": tokenizer.chat_template,
        "seed": SEED,
        "generation": config.to_diff_dict(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "bank_sha256": hashlib.sha256((HERE / "bank.json").read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest(),
    }
    (output / "qwen2.5-1.5b_runtime.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    targets = [
        (case, mode)
        for i, case in enumerate(build_cases(bank, True) + build_cases(bank))
        for mode in (MODES if i % 2 == 0 else MODES[::-1])
    ]
    raw = output / "qwen2.5-1.5b_raw.jsonl"
    torch.manual_seed(SEED)
    with raw.open("x", encoding="utf-8") as handle, torch.inference_mode():
        for index, (case, mode) in enumerate(targets, start=1):
            payload = request_body(MODEL, bank["tools"], case["query"], mode)
            tools = [entry["function"] for entry in payload.get("tools", [])] or None
            tokens = tokenizer.apply_chat_template(
                payload["messages"], tools=tools, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to("mps")
            start = time.monotonic()
            generated = model.generate(**tokens, generation_config=config)[0][tokens.input_ids.shape[-1]:].tolist()
            ends = generated[-1] in config.eos_token_id
            text = tokenizer.decode(generated[:-1] if ends else generated, skip_special_tokens=False)
            message, native_call = native_response(text, mode)
            finish = ("tool_calls" if native_call else "stop") if ends else "length"
            row = {
                **case, "model": "qwen2.5-1.5b", "mode": mode, "request": payload,
                "response": {"choices": [{"message": message, "finish_reason": finish}], "raw_decoded": text,
                             "generated_token_ids": generated, "input_token_count": tokens.input_ids.shape[-1]},
                "finish_reason": finish, "score": score(message, mode, case["expected"]),
                "elapsed_s": round(time.monotonic() - start, 4),
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            if index <= 4:
                print(f"qwen2.5-1.5b smoke {mode} {case['id']}: {finish}, {row['score']}", flush=True)
            if index % 10 == 0 or index == len(targets):
                print(f"qwen2.5-1.5b: {index}/{len(targets)} requests recorded", flush=True)
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    result = estimate(rows)
    smoke = [row for row in rows if row["phase"] == "control"]
    admissible = (
        len(smoke) == 4
        and all(row["finish_reason"] != "length" for row in smoke)
        and all(row["score"]["canonical_correct_call"] for row in smoke if row["expected"] is not None)
    )
    result["runtime_smoke_admissible"] = admissible
    (output / "summary.json").write_text(json.dumps({"qwen2.5-1.5b": result}, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return int(not admissible or result["status"] != "complete")


if __name__ == "__main__":
    sys.exit(main())
