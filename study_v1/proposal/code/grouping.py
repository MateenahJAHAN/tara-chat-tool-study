"""Original input-only grouping, matching the independently replayed audit."""
from collections import defaultdict
import hashlib
import json
import re
import unicodedata


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def canonical(value, key=None):
    if isinstance(value, dict):
        return {k: canonical(v, k) for k, v in sorted(value.items())}
    if isinstance(value, list):
        values = [canonical(v, key) for v in value]
        return sorted(values, key=lambda v: json.dumps(v, sort_keys=True)) if key in {"required", "enum"} else values
    if key == "type" and isinstance(value, str):
        return {"dict": "object", "float": "number"}.get(value, value)
    return value


def canon_json(value):
    return json.dumps(canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def group_items(items, cfg=None):
    uids = sorted(items)
    parent = {uid: uid for uid in uids}

    def find(uid):
        while parent[uid] != uid:
            parent[uid] = parent[parent[uid]]
            uid = parent[uid]
        return uid

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    keyed = defaultdict(list)
    queries = {}
    for uid in uids:
        item = items[uid]
        fn = item.get("grouping_function", item["tool"]["function"])
        query = normalize(item["query"])
        queries[uid] = query
        for kind, value in [("query", query), ("name", normalize(fn["name"])),
                            ("function", canon_json(fn)), ("schema", canon_json(fn.get("parameters", {})))]:
            keyed[kind + ":" + value].append(uid)
    for members in keyed.values():
        for uid in members[1:]:
            union(members[0], uid)
    shingles = {}
    for uid, query in queries.items():
        query = re.sub(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?!\w)", "<num>", query)
        tokens = re.findall(r"<num>|\w+", query)
        shingles[uid] = (len(tokens), set(zip(tokens, tokens[1:], tokens[2:])))
    for i, a in enumerate(uids):
        la, sa = shingles[a]
        if la < 8:
            continue
        for b in uids[i + 1:]:
            lb, sb = shingles[b]
            if lb < 8 or 10 * min(len(sa), len(sb)) < 9 * max(len(sa), len(sb)):
                continue
            common = len(sa & sb)
            if 10 * common >= 9 * (len(sa) + len(sb) - common):
                union(a, b)
    members = defaultdict(list)
    for uid in uids:
        members[find(uid)].append(uid)
    names = {root: "group_" + hashlib.sha256("\n".join(sorted(group)).encode()).hexdigest()[:24]
             for root, group in members.items()}
    return {uid: names[find(uid)] for uid in uids}
