"""Paired finite-bank behavioural descriptives. Both DEV and HELDOUT reports are exploratory.
Unit = independently grouped item; preserve paired interfaces; no confirmatory test or general safety claim."""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import gate  # noqa: E402
from run import dev_score_directory  # noqa: E402

RATE_KEYS = ["false_call", "broad_false_call", "operational_abstention", "strict_abstention", "call_attempt", "valid_call",
             "correct_call", "format_compliant", "parse_error", "unknown_name", "empty_output", "truncated", "errored"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def paired_cluster_ci(diff, groups, B, seed):
    diff, groups = np.asarray(diff, float), np.asarray(groups)
    ug = np.unique(groups)
    sums = np.array([diff[groups == g].sum() for g in ug])
    cnts = np.array([np.sum(groups == g) for g in ug])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ug), size=(B, len(ug)))
    vals = sums[draws].sum(1) / cnts[draws].sum(1)
    return {"mean": float(diff.mean()), "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "n_items": int(len(diff)), "n_groups": int(len(ug)), "degenerate": bool(len(np.unique(diff)) <= 1), "interpretation": "descriptive group-resampling interval for this finite bank; no population coverage guarantee"}


def load(path):
    return {r["uid"]: r for r in common.read_jsonl(path)} if Path(path).exists() else None


def paired(a, b, key, label, B, seed, require_both_attempt=False):
    uids = sorted(u for u in a if u in b and a[u]["label"] == label
                  and (not require_both_attempt or (a[u]["score"]["call_attempt"] and b[u]["score"]["call_attempt"])))
    if not uids:
        return None
    if set(a) != set(b) and not set(a).issubset(b) and not set(b).issubset(a):
        raise RuntimeError("STOP: unmatched scheduled condition banks")
    for u in uids:
        if a[u]["group"] != b[u]["group"] or a[u]["label"] != b[u]["label"]:
            raise RuntimeError("STOP: paired metadata mismatch")
        if any(not isinstance(r["score"][key], bool) for r in (a[u], b[u])):
            raise RuntimeError("STOP: missing/nonboolean applicable endpoint")
    x = np.array([int(bool(a[u]["score"][key])) for u in uids])
    y = np.array([int(bool(b[u]["score"][key])) for u in uids])
    return {"ci_b_minus_a": paired_cluster_ci(y - x, [a[u]["group"] for u in uids], B, seed), "discordant_pairs": {"a0_b1": int(np.sum((x == 0) & (y == 1))), "a1_b0": int(np.sum((x == 1) & (y == 0)))},
            "rate_a": float(x.mean()), "rate_b": float(y.mean())}


def rates(recs):
    out = {}
    for lab in (0, 1):
        rs = [r for r in recs.values() if r["label"] == lab]
        out[f"label{lab}"] = {"n_scheduled": len(rs)}
        for k in RATE_KEYS:
            vals = [bool(r["score"][k]) for r in rs if r["score"][k] is not None]
            if vals:
                out[f"label{lab}"][k] = {"k": sum(vals), "n": len(vals), "descriptive_wilson_binomial_reference95": wilson(sum(vals), len(vals)), "scope": "conditional descriptive binomial-reference interval; selected finite bank"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", required=True)
    ap.add_argument("--split", required=True, choices=["DEV", "HELDOUT"])
    args = ap.parse_args()
    root, cfg = Path(args.study_root).resolve(), common.load_config()
    A, s = cfg["analysis"], cfg["steering"]
    if args.split == "HELDOUT":
        gate.require_approval(root, "HELDOUT_BASELINE")
        d = root / "runs" / "heldout"
        if not (d / "HELDOUT_COMPLETE.json").exists():
            raise RuntimeError("STOP: heldout incomplete")
    else:
        d = dev_score_directory(root, cfg)
    keep = lambda recs: {u: r for u, r in recs.items() if r["split"] == args.split}  # noqa: E731
    text = keep(load(d / "generations_baseline_TEXT_CATALOGUE.jsonl"))
    nat = keep(load(d / "generations_baseline_NATIVE_TOOLS.jsonl"))
    manifest = common.read_json(root / "manifests/split_manifest.json")["items"]
    expected = {x["uid"] for x in manifest if x["split"] == args.split}
    if set(text) != expected or set(nat) != expected:
        raise RuntimeError("STOP: incomplete/mismatched frozen baseline bank")
    steering_expected = {x["uid"] for x in manifest if x["split"] == args.split and x["steering_subset"]}
    B, seed = A["bootstrap_B"], A["bootstrap_seed"]
    rep = {"split": args.split, "score_version": cfg["measurement_version"], "status": "HELDOUT_FINITE_BANK_DESCRIPTIVE" if args.split == "HELDOUT" else "DEVELOPMENT_DESCRIPTIVE",
           "rates": {"TEXT_CATALOGUE": rates(text), "NATIVE_TOOLS": rates(nat)}}
    prim = paired(text, nat, "false_call", 1, B, seed)
    rep["primary_false_call_native_minus_text"] = prim
    if prim:
        ci = prim["ci_b_minus_a"]
        rep["primary_verdict"] = "observed_finite_bank_paired_difference_descriptive_only"
        rep["confirmatory_claim"] = False
    sec = {"S1_operational_abstention": paired(text, nat, "operational_abstention", 1, B, seed),
           "S2_correct_call": paired(text, nat, "correct_call", 0, B, seed),
           "S3_call_rate_relevant": paired(text, nat, "call_attempt", 0, B, seed),
           "S4_format_compliance_among_attempts": paired(text, nat, "format_compliant", 0, B, seed, require_both_attempt=True)}
    for key, result in sec.items():
        if result:
            result["claim"] = False
            result["scope"] = "descriptive fixed-bank secondary endpoint; no confirmatory p-value"
    rep["secondary"] = sec
    steering = {}
    for cond, base in (("TEXT_CATALOGUE", text), ("NATIVE_TOOLS", nat)):
        files = sorted(d.glob(f"generations_steer_*_{cond}.jsonl")) if args.split == "HELDOUT" else []
        if not files:
            continue
        res = {}
        for f in files:
            name = f.name[len("generations_steer_"):-len(f"_{cond}.jsonl")]
            st = keep(load(f))
            if set(st) != steering_expected:
                raise RuntimeError("STOP: incomplete/mismatched preset steering subset")
            red = paired(st, base, "false_call", 1, B, seed)  # base - steered = reduction
            util = paired(base, st, "correct_call", 0, B, seed)
            aligned_base = {u: r for u, r in base.items() if u in st}
            if len(aligned_base) != len(st): raise RuntimeError("STOP: unmatched steering baseline")
            inv_b = np.mean([bool(r["score"]["invalid_or_empty"]) for r in aligned_base.values()])
            inv_s = np.mean([bool(r["score"]["invalid_or_empty"]) for r in st.values()])
            res[name] = {"false_call_reduction": red, "correct_call_change": util, "invalid_increase": float(inv_s - inv_b)}
        kls = {r["config"]: r["completion_kl"] for r in common.read_jsonl(d / "steering_kl.jsonl") if r["condition"] == cond}
        mp = res.get("mm_plus")
        ctrl = [v["false_call_reduction"]["ci_b_minus_a"]["mean"] for k, v in res.items()
                if (k.startswith("random") or k == "permuted") and v["false_call_reduction"]]
        claim = False
        if mp and mp["false_call_reduction"] and mp["correct_call_change"]:
            ci = mp["false_call_reduction"]["ci_b_minus_a"]
            claim = bool(len(ctrl) == 4 and ci["lo"] > 0 and all(ci["mean"] > c for c in ctrl)
                         and -mp["correct_call_change"]["ci_b_minus_a"]["mean"] <= s["max_correct_call_drop"]
                         and mp["invalid_increase"] <= s["max_invalid_increase"]
                         and kls.get("mm_plus") is not None and kls["mm_plus"] <= s["max_completion_kl"])
        steering[cond] = {"configs": res, "completion_kl": kls, "bounded_intervention_criteria_met": claim, "confirmatory_claim": False}
    rep["steering"] = steering
    rep["caveats"] = ["benchmark-defined labels", "interface-package effect, not tool availability alone",
                      "not official BFCL scoring", "contamination unknown", "all endpoints exploratory/descriptive; finite selected bank and small steering subset", "no equivalence or general safety inference"]
    common.write_json(root / "reports" / f"behaviour_{args.split.lower()}.json", rep)
    print({"primary": rep.get("primary_verdict"), "split": args.split})


if __name__ == "__main__":
    main()
