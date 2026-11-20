"""P0/P1: verify staged sources, canonicalize schemas, apply input-only eligibility, group, split, render.
Never reads model outputs. Usage: python -I code/prepare.py --study-root ROOT"""
import argparse
import copy
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

LIVE_ID = re.compile(r"^(live_[a-z_]+?)_(\d+)-(\d+)-(\d+)$")


# ---------- canonical schema ----------
def map_types(value, type_map):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "type" and isinstance(v, str):
                out[k] = type_map.get(v, v)
            elif k == "type" and isinstance(v, list) and all(isinstance(x, str) for x in v):
                out[k] = [type_map.get(x, x) for x in v]
            else:
                out[k] = map_types(v, type_map)
        return out
    if isinstance(value, list):
        return [map_types(x, type_map) for x in value]
    return value


def collect_types(value, acc):
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "type" and isinstance(v, str):
                acc.add(v)
            elif k == "type" and isinstance(v, list):
                acc.update(x for x in v if isinstance(x, str))
            else:
                collect_types(v, acc)
    elif isinstance(value, list):
        for x in value:
            collect_types(x, acc)
    return acc


def canonical_tool(fn, cfg):
    """BFCL function dict -> (native tool object, flags). Same object is used by every condition."""
    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not isinstance(fn.get("description", ""), str):
        raise ValueError("E3_bad_function")
    params = fn.get("parameters", {"type": "dict", "properties": {}})
    if not isinstance(params, dict) or not isinstance(params.get("properties", {}), dict):
        raise ValueError("E3_bad_parameters")
    mapped = map_types(copy.deepcopy(params), cfg["prompt"]["type_map"])
    ordered = {"type": "object", "properties": mapped.get("properties", {}), "required": list(mapped.get("required", []))}
    for k, v in mapped.items():
        if k not in ordered:
            ordered[k] = v
    tool = {"type": "function", "function": {"name": fn["name"], "description": fn.get("description", ""), "parameters": ordered}}
    nonstd = sorted(collect_types(ordered, set()) - set(cfg["prompt"]["standard_types"]))
    return tool, {"nonstandard_types_retained": nonstd}


def tool_json(tool):
    """Matches the pinned template's `tool | tojson` (ensure_ascii=False, default separators, insertion order)."""
    return json.dumps(tool, ensure_ascii=False)


# ---------- rendering ----------
def build_messages(cfg, query, tool, condition):
    rule = cfg["prompt"]["rule"]
    if condition == "TEXT_CATALOGUE":
        system = rule + cfg["prompt"]["text_header"] + tool_json(tool)
    elif condition == "NATIVE_TOOLS":
        system = rule
    elif condition == common.QUERY_ONLY:
        system = rule + cfg["prompt"]["text_header"] + cfg["prompt"]["query_only_placeholder"]
    else:
        raise ValueError(condition)
    return [{"role": "system", "content": system}, {"role": "user", "content": query}]


def render(tokenizer, cfg, query, tool, condition):
    msgs = build_messages(cfg, query, tool, condition)
    tools = [tool] if condition == "NATIVE_TOOLS" else None
    return tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True)


# ---------- eligibility (input only) ----------
def single_user_query(row):
    q = row.get("question")
    if not (isinstance(q, list) and len(q) == 1 and isinstance(q[0], list) and len(q[0]) == 1):
        return None
    m = q[0][0]
    if not isinstance(m, dict) or m.get("role") != "user" or not isinstance(m.get("content"), str) or not m["content"].strip():
        return None
    return m["content"]


def ascii_letter_frac(text):
    letters = [c for c in text if c.isalpha()]
    return (sum(c.isascii() for c in letters) / len(letters)) if letters else 0.0


def luhn_ok(digits):
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def content_reasons(query, fn_text, cfg):
    e = cfg["eligibility"]
    reasons = []
    if ascii_letter_frac(query) < e["ascii_letter_min_frac"]:
        reasons.append("E4_not_english_dominant")
    blob = query + "\n" + fn_text
    if any(re.search(p, blob, re.IGNORECASE) for p in e["safe_domain_patterns"]):
        reasons.append("E5_safe_domain")
    if any(re.search(p, query) for p in e["pii_query_patterns"]):
        reasons.append("E6_pii_like")
    for m in re.finditer(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)", query):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            reasons.append("E6_card_like")
            break
    if any(re.search(p, blob) for p in e["secret_patterns"]):
        reasons.append("E6_secret_like")
    return reasons


# ---------- grouping ----------
def content_words(text, stopwords):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in stopwords}


def norm_query(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def near_duplicate(a, b, g):
    shared = len(a & b)
    if not shared:
        return False
    jac = shared / len(a | b)
    ovl = shared / min(len(a), len(b))
    return (jac >= g["jaccard_min"] and shared >= g["jaccard_min_shared"]) or \
           (ovl >= g["overlap_min"] and shared >= g["overlap_min_shared"])


class UnionFind:
    def __init__(self, keys):
        self.p = {k: k for k in keys}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def group_items(items, cfg):
    from grouping import group_items as independent_group_items
    return independent_group_items(items, cfg)


# ---------- splitting / selection ----------
def seeded_key(seed, text):
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def select(pool, per_class, cap, seed, forced=()):
    """pool: list of item dicts with label/group. Seeded order, per-group cap, per-class target."""
    chosen, per_group, per_label = [], defaultdict(int), defaultdict(int)
    ordered = [x for x in pool if x["uid"] in forced] + \
              sorted((x for x in pool if x["uid"] not in forced), key=lambda x: seeded_key(seed, x["uid"]))
    for x in ordered:
        if per_label[x["label"]] >= per_class or per_group[x["group"]] >= cap:
            continue
        chosen.append(x)
        per_group[x["group"]] += 1
        per_label[x["label"]] += 1
    return chosen


def split_items(eligible, cfg, calib_uids, graph_items=None):
    from group_selection import select_balanced_unique_groups
    graph_items = eligible if graph_items is None else graph_items
    s = cfg["splits"]
    calibration_groups = {x["group"] for x in graph_items if x["uid"] in calib_uids}
    native_groups = {x["group"] for x in graph_items if x["family"] == "native"}
    pools = {"native": [], "live": []}
    excluded = []
    for x in eligible:
        if x["group"] in calibration_groups:
            excluded.append((x["uid"], "CALIBRATION_ONLY_component"))
        elif x["family"] == "live" and x["group"] in native_groups:
            excluded.append((x["uid"], "live_component_touches_native"))
        else:
            pools[x["family"]].append(x)
    report = {"pool_sizes": {name: {str(label): sum(x["label"] == label for x in rows) for label in (0, 1)}
                             for name, rows in pools.items()}, "selected": {}, "stop_reasons": [],
              "calibration_component_members": sorted(x["uid"] for x in graph_items if x["group"] in calibration_groups),
              "excluded_calibration_or_overlap": len(excluded), "one_item_per_group_in_all_selected_banks": True}
    chosen = {"TRAIN": [], "DEV": [], "HELDOUT": []}
    try:
        native = select_balanced_unique_groups(pools["native"],
                    s["native_train_per_class"] + s["native_dev_per_class"], s["seed_selection"])
        for label in (0, 1):
            ordered = sorted((x for x in native if x["label"] == label),
                             key=lambda x: seeded_key(s["seed_split"], "DEV:" + x["uid"]))
            chosen["DEV"] += ordered[:s["native_dev_per_class"]]
            chosen["TRAIN"] += ordered[s["native_dev_per_class"]:]
        chosen["HELDOUT"] = select_balanced_unique_groups(pools["live"], s["heldout_per_class_target"], s["seed_selection"])
    except ValueError as exc:
        report["stop_reasons"].append(str(exc))
    targets = {"TRAIN": s["native_train_per_class"], "DEV": s["native_dev_per_class"], "HELDOUT": s["heldout_per_class_target"]}
    for split, rows in chosen.items():
        report["selected"][split] = {str(label): {"n": sum(x["label"] == label for x in rows),
                            "groups": len({x["group"] for x in rows if x["label"] == label})} for label in (0, 1)}
        for label in (0, 1):
            if report["selected"][split][str(label)]["n"] != targets[split]:
                report["stop_reasons"].append(f"{split} label{label} exact quota unmet")
        if len(rows) != len({x["group"] for x in rows}):
            raise RuntimeError("STOP: repeated selected group")
    all_chosen = [x for rows in chosen.values() for x in rows]
    if len(all_chosen) != len({x["group"] for x in all_chosen}):
        raise RuntimeError("STOP: component leakage across main splits")
    if any(x["group"] in calibration_groups for x in all_chosen):
        raise RuntimeError("STOP: calibration component in main bank")
    return {x["uid"]: split for split, rows in chosen.items() for x in rows}, report, excluded


# ---------- main ----------
def load_possible_answers(path):
    return {r["id"]: r["ground_truth"] for r in common.read_jsonl(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", required=True)
    args = ap.parse_args()
    root = Path(args.study_root).resolve()
    out = root / "manifests"
    if (out / "split_manifest.json").exists():
        raise RuntimeError("split_manifest.json exists; prepare is single-shot (move it aside with root approval)")
    cfg = common.load_config()
    common.set_offline_env(root)
    verification = common.verify_staged(root, cfg, include_model=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(root / "staged" / "model", local_files_only=True, trust_remote_code=False)
    data_dir = root / "staged" / "bfcl"
    sources = {**{k: ("native", v) for k, v in cfg["data"]["native_sources"].items()},
               **{k: ("live", v) for k, v in cfg["data"]["live_sources"].items()}}
    answers = {src: load_possible_answers(data_dir / pa) for src, pa in cfg["data"]["possible_answers"].items()}
    calib = set(cfg["data"]["calibration_ids"])
    records, exclusion_counts = {}, defaultdict(lambda: defaultdict(int))
    for src, (family, label) in sources.items():
        rows = common.read_jsonl(data_dir / src)
        if len(rows) != cfg["data"]["files"][src]["rows"]:
            raise RuntimeError(f"STOP: row count {src}")
        for row in rows:
            uid = f"{src}::{row['id']}"
            reasons = []
            fns = row.get("function") or []
            if len(fns) != 1:
                reasons.append("E1_not_one_function")
            query = single_user_query(row)
            if query is None:
                reasons.append("E2_not_single_user_message")
            tool, flags = None, {}
            if not reasons:
                try:
                    tool, flags = canonical_tool(fns[0], cfg)
                except ValueError as exc:
                    reasons.append(str(exc))
            rec = {"uid": uid, "id": row["id"], "source": src, "family": family, "label": label,
                   "query": query, "tool": tool, "flags": flags, "possible_answer": None,
                   "grouping_function": fns[0] if len(fns) == 1 else None}
            if tool is not None:
                reasons += content_reasons(query, tool_json(tool), cfg)
                if flags["nonstandard_types_retained"]:
                    reasons.append("E3_unsupported_schema_type")
                import parser as output_parser
                if output_parser._schema_issues(tool["function"]["parameters"]):
                    reasons.append("E3b_unsupported_schema_keyword_or_constraint")
                if label == 0:
                    gt = answers[src].get(row["id"])
                    if not (isinstance(gt, list) and len(gt) == 1 and list(gt[0]) == [tool["function"]["name"]]):
                        reasons.append("E8_possible_answer_mismatch")
                    rec["possible_answer"] = gt
                prompts = {}
                for cond in common.CONDITIONS + (common.QUERY_ONLY,):
                    text = render(tok, cfg, query, tool, cond)
                    ids = tok(text, add_special_tokens=False)["input_ids"]
                    prompts[cond] = {"text": text, "sha256": common.sha256_text(text), "n_tokens": len(ids), "last_token_id": ids[-1]}
                if any(prompts[c]["n_tokens"] > cfg["model"]["max_input_tokens"] for c in common.CONDITIONS):
                    reasons.append("E7_token_limit")
                if any(tool_json(tool) not in prompts[c]["text"] for c in common.CONDITIONS):
                    reasons.append("E9_catalogue_string_not_verbatim")
                if prompts["TEXT_CATALOGUE"]["last_token_id"] != prompts["NATIVE_TOOLS"]["last_token_id"]:
                    reasons.append("E9_decision_token_mismatch")
                rec["prompts"] = prompts
            rec["exclusions"] = reasons
            for r in reasons or ["eligible"]:
                exclusion_counts[src][r] += 1
            records[uid] = rec
    calib_uids = {u for u, r in records.items() if r["id"] in calib}
    if len(calib_uids) != len(calib):
        raise RuntimeError("STOP: calibration ids not found")
    groupable = {u: r for u, r in records.items() if r["query"] is not None and r["tool"] is not None}
    if any(r["tool"] is None for r in groupable.values()):
        raise RuntimeError("STOP: calibration item not canonicalizable")
    gid = group_items(groupable, cfg)
    graph_items = [{"uid": u, "family": r["family"], "label": r["label"], "group": gid[u]} for u, r in groupable.items()]
    eligible = [x for x in graph_items if not records[x["uid"]]["exclusions"]]
    assignment, split_report, excluded = split_items(eligible, cfg, calib_uids, graph_items)
    comp_sizes = defaultdict(int)
    comp_labels = defaultdict(set)
    for u, g in gid.items():
        comp_sizes[g] += 1
        comp_labels[g].add(records[u]["label"])
    subset_uids = {"DEV": set(), "HELDOUT": set()}
    for split in subset_uids:
        for label in (0, 1):
            members = [u for u, sp in assignment.items() if sp == split and records[u]["label"] == label]
            members.sort(key=lambda u: seeded_key(cfg["steering"]["subset_seed"], split + ":" + u))
            subset_uids[split].update(members[:cfg["steering"]["subset_per_class"]])
    items = []
    for uid, split in sorted(assignment.items()):
        r = records[uid]
        items.append({**{k: r[k] for k in ("uid", "id", "source", "family", "label", "query", "tool", "flags", "possible_answer", "prompts")},
                      "split": split, "group": gid[uid], "development_only_calibration": False,
                      "steering_subset": uid in subset_uids.get(split, set())})
    status = "STOP_COUNT_GATE" if split_report["stop_reasons"] else "READY_FOR_ROOT_REVIEW"
    report = {"design_id": cfg["design_id"], "created_utc": common.now_utc(), "status": status,
              "verification": verification, "exclusion_counts": {k: dict(v) for k, v in exclusion_counts.items()},
              "n_components": len(comp_sizes), "largest_component": max(comp_sizes.values()),
              "label_mixed_components": sum(1 for v in comp_labels.values() if len(v) > 1),
              "split": split_report, "model_outputs_read": False}
    common.write_json(out / "prepare_report.json", report)
    common.write_json(out / "split_manifest.json", {"design_id": cfg["design_id"], "status": status, "items": items})
    print(json.dumps({"status": status, "selected": split_report["selected"], "stop_reasons": split_report["stop_reasons"]}))
    if status != "READY_FOR_ROOT_REVIEW":
        sys.exit(2)


if __name__ == "__main__":
    main()
