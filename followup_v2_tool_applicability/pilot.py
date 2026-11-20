#!/usr/bin/env python3
"""Run and score the frozen benign matched-pair pilot with local models."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
SEED = 20261120
MODES = ("text", "native")
SYSTEM = (
    "Select a listed tool only if it can satisfy the entire request, including "
    "every restriction on scope, format, unit, transport mode, or menu. "
    "Never substitute another value. If there is no exact match, reply with "
    "exactly NO_CALL. Make at most one call. Do not provide a workaround, "
    "estimate, explanation, or simulated tool result."
)
MODEL_CONFIG = {
    "qwen3-4b": {
        "checkpoint": "Qwen3-4B, Ollama qwen3:4b",
        "weights_sha256": "3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f",
        "default_url": "http://127.0.0.1:8779",
    },
    "qwen3.5-4b": {
        "checkpoint": "Qwen3.5-4B, Unsloth UD-Q4_K_XL",
        "weights_sha256": "b252c5610a42ca82d20fe2a12813e9d069eed89292907e26c783eeb0bc961bc7",
        "default_url": "http://127.0.0.1:8780",
    },
}


def validate_bank(bank):
    scopes = set()
    for pair in bank["pairs"]:
        assert pair["query"].count("{scope}") == 1
        assert all(len(pair[key].split()) == 1 for key in ("in_scope", "out_of_scope"))
        assert pair["in_scope"].casefold() != pair["out_of_scope"].casefold()
        scopes.add(pair["family"])
    assert len(scopes) == 5 and len(bank["pairs"]) == 20
    assert len({item["id"] for item in [*bank["pairs"], *bank["controls"]]}) == 22


def request_body(server_model, tools, query, mode):
    payload = {
        "model": server_model,
        "messages": [],
        "temperature": 0,
        "max_tokens": 96,
        "seed": SEED,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    system = SYSTEM
    if mode == "text":
        system += (
            '\nFor a call, print exactly one JSON object with keys "name" and '
            '"arguments": {"name":"listed_name","arguments":{...}}.\n'
            "Functions:\n" + json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        )
    else:
        system += "\nFor a call, use the native tool interface."
        payload["tools"] = [{"type": "function", "function": tool} for tool in tools]
        payload["tool_choice"] = "auto"
    payload["messages"] = [{"role": "system", "content": system}, {"role": "user", "content": query}]
    return payload


def unique_object(pairs):
    values = dict(pairs)
    if len(values) != len(pairs):
        raise ValueError("duplicate JSON key")
    return values


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid_number)


def invalid_number(value):
    raise ValueError(f"non-finite value: {value}")


def matching_call(call, expected):
    if not isinstance(call, dict) or expected is None or call.get("name") != expected["name"]:
        return False
    args = call.get("arguments")
    if isinstance(args, str):
        try:
            args = decode(args)
        except ValueError:
            return False
    if not isinstance(args, dict) or args.keys() != expected["arguments"].keys():
        return False
    for key, value in expected["arguments"].items():
        received = args[key]
        if isinstance(value, (int, float)):
            if isinstance(received, bool) or not isinstance(received, (int, float)) or received != value:
                return False
        elif received != value:
            return False
    return True


CALL_HINT = re.compile(r'<tool_call\b|<function\s*=|\{\s*"name"\s*:|\{\s*"arguments"\s*:', re.I)
JSON_START = re.compile(r'\{\s*"(?:name|arguments)"\s*:', re.I)


def score(message, mode, expected):
    content = (message.get("content") or "").strip()
    native = message.get("tool_calls") or []
    parsed, canonical, path = [], False, "none"
    attempt = bool(native or CALL_HINT.search(content))
    if native:
        path = "native"
        parsed = [entry.get("function") if isinstance(entry, dict) else None for entry in native]
        canonical = mode == "native" and not content and all(
            isinstance(entry, dict) and entry.get("type") == "function" for entry in native
        )
    else:
        text_decoder = json.JSONDecoder(object_pairs_hook=unique_object, parse_constant=invalid_number)
        cursor = 0
        for match in JSON_START.finditer(content):
            if match.start() < cursor:
                continue
            try:
                value, length = text_decoder.raw_decode(content[match.start():])
            except ValueError:
                continue
            cursor = match.start() + length
            if isinstance(value, dict):
                parsed.append(value)
        if parsed:
            path = "text_json"
            try:
                obj = decode(content)
                canonical = mode == "text" and isinstance(obj, dict) and set(obj) == {"name", "arguments"} and isinstance(obj["arguments"], dict)
            except ValueError:
                pass
        elif attempt:
            path = "malformed_call"
    exact_abstain = not attempt and content == "NO_CALL"
    semantic = len(parsed) == 1 and matching_call(parsed[0], expected)
    return {
        "call_attempt": attempt,
        "exact_abstain": exact_abstain,
        "semantic_correct_call": semantic,
        "canonical_correct_call": semantic and canonical,
        "parse_path": path,
    }


def api(base_url, route, body=None):
    parts = urlsplit(base_url)
    if parts.scheme != "http" or parts.hostname != "127.0.0.1" or parts.path or parts.query:
        raise ValueError("Only bare local loopback URLs are supported")
    request = Request(base_url + route, headers={"Content-Type": "application/json"})
    if body is not None:
        request.data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    with urlopen(request, timeout=90) as response:
        return json.load(response)


def build_cases(bank, controls_only=False):
    if controls_only:
        return [dict(case, phase="control", family="control", pair_id=None) for case in bank["controls"]]
    cases = []
    for pair in bank["pairs"]:
        for source, suffix in (("in_scope", "in"), ("out_of_scope", "out")):
            cases.append({
                "id": pair["id"] + "-" + suffix,
                "phase": "heldout",
                "family": pair["family"],
                "pair_id": pair["id"],
                "query": pair["query"].format(scope=pair[source]),
                "expected": pair["expected"] if suffix == "in" else None,
            })
    random.Random(SEED).shuffle(cases)
    return cases


def run_model(name, base_url, bank, output):
    model_meta = api(base_url, "/v1/models")
    model_ids = [entry["id"] for entry in model_meta["data"]]
    if len(model_ids) != 1:
        raise ValueError(f"Expected one loaded model, saw {len(model_ids)}")
    props = api(base_url, "/props")
    meta = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": name,
        "checkpoint": MODEL_CONFIG[name],
        "server_model": model_meta,
        "runtime_props": props,
        "seed": SEED,
        "bank_sha256": hashlib.sha256((HERE / "bank.json").read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((HERE / "PROTOCOL.md").read_bytes()).hexdigest(),
    }
    (output / f"{name}_runtime.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    targets = [
        (case, mode)
        for i, case in enumerate(build_cases(bank, True) + build_cases(bank))
        for mode in (MODES if i % 2 == 0 else MODES[::-1])
    ]
    path = output / f"{name}_raw.jsonl"
    with path.open("x", encoding="utf-8") as handle:
        for index, (case, mode) in enumerate(targets, start=1):
            body = request_body(model_ids[0], bank["tools"], case["query"], mode)
            row = {**case, "model": name, "mode": mode, "request": body}
            start = time.monotonic()
            try:
                reply = api(base_url, "/v1/chat/completions", body)
                first = reply["choices"][0]
                row.update({
                    "response": reply,
                    "finish_reason": first["finish_reason"],
                    "score": score(first["message"], mode, case["expected"]),
                })
            except (HTTPError, URLError, TimeoutError, ValueError, KeyError, IndexError) as error:
                row["error"] = str(error)
            row["elapsed_s"] = round(time.monotonic() - start, 4)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(targets):
                print(f"{name}: {index}/{len(targets)} requests recorded", flush=True)
            if index <= 4:
                check = row.get("score")
                print(f"{name}: smoke {mode}, {case['id']} score={check}, error={row.get('error')}", flush=True)
        print(f"{name}: saved {path.name}", flush=True)


def interval(effects_by_family):
    rng = random.Random(SEED)
    draws = []
    for _ in range(10000):
        sample = [
            value
            for family in effects_by_family.values()
            for value in rng.choices(family, k=len(family))
        ]
        draws.append(100 * sum(sample) / len(sample))
    draws.sort()
    return [round(draws[249], 3), round(draws[9749], 3)]


def binomial_sign_test(left, right):
    n = left + right
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(left, right) + 1)) / (2 ** n))


def estimate(rows):
    expected_keys = {
        (case["id"], mode)
        for case in build_cases(json.loads((HERE / "bank.json").read_text()))
        for mode in MODES
    }
    main = [row for row in rows if row["phase"] == "heldout"]
    keys = {(row["id"], row["mode"]) for row in main}
    errors = [f'{row["id"]}/{row["mode"]}' for row in main if "error" in row]
    if keys != expected_keys or len(keys) != len(main) or errors:
        return {"status": "incomplete", "records": len(main), "errors": errors, "missing": sorted(expected_keys - keys)}

    rec = {(row["pair_id"], row["mode"], row["expected"] is not None): row for row in main}
    pairs = sorted({row["pair_id"] for row in main})
    stats = {}
    for mode in MODES:
        relevant = [rec[pair, mode, True]["score"] for pair in pairs]
        irrelevant = [rec[pair, mode, False]["score"] for pair in pairs]
        stats[mode] = {
            "n_pairs": len(pairs),
            "false_calls": sum(row["call_attempt"] for row in irrelevant),
            "exact_abstentions": sum(row["exact_abstain"] for row in irrelevant),
            "semantic_correct_calls": sum(row["semantic_correct_call"] for row in relevant),
            "canonical_correct_calls": sum(row["canonical_correct_call"] for row in relevant),
            "scope_pairs_passed": sum(rel["semantic_correct_call"] and irr["exact_abstain"] for rel, irr in zip(relevant, irrelevant)),
            "generation_caps": sum(row["finish_reason"] == "length" for row in main if row["mode"] == mode),
        }
    paired = {"both": 0, "text_only": 0, "native_only": 0, "neither": 0}
    effects = {}
    for pair in pairs:
        left = rec[pair, "text", False]
        right = rec[pair, "native", False]
        before, after = left["score"]["call_attempt"], right["score"]["call_attempt"]
        status = {(True, True): "both", (True, False): "text_only", (False, True): "native_only", (False, False): "neither"}
        paired[status[before, after]] += 1
        effects.setdefault(left["family"], []).append(int(after) - int(before))
    difference = 100 * (paired["native_only"] - paired["text_only"]) / len(pairs)
    return {
        "status": "complete",
        "records": len(main),
        "smoke_control_records": [row["score"] for row in rows if row["phase"] == "control" and "score" in row],
        "conditions": stats,
        "paired_false_calls": paired,
        "difference_native_minus_text_pp": difference,
        "bootstrap_95_descriptor_pp": interval(effects),
        "exact_two_sided_discordance_sign_p": binomial_sign_test(paired["text_only"], paired["native_only"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen3", default=MODEL_CONFIG["qwen3-4b"]["default_url"])
    parser.add_argument("--qwen35", default=MODEL_CONFIG["qwen3.5-4b"]["default_url"])
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    bank = json.loads((HERE / "bank.json").read_text())
    validate_bank(bank)
    if not args.score_only:
        args.output.mkdir(parents=True, exist_ok=False)
        with ThreadPoolExecutor(max_workers=2) as pool:
            runs = [
                pool.submit(run_model, name, address, bank, args.output)
                for name, address in (("qwen3-4b", args.qwen3), ("qwen3.5-4b", args.qwen35))
            ]
            for run in runs:
                run.result()

    result = {}
    for name in MODEL_CONFIG:
        path = args.output / f"{name}_raw.jsonl"
        if path.exists():
            result[name] = estimate([json.loads(line) for line in path.read_text().splitlines()])
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(len(result) != len(MODEL_CONFIG) or any(value["status"] != "complete" for value in result.values()))


if __name__ == "__main__":
    sys.exit(main())
