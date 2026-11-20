"""Phase runner. Usage: python -I code/run.py --study-root ROOT --phase {preflight,dev-run,dev-steer,heldout}
dev-steer requires frozen/probe_selection.json from probes.py fit-dev. heldout requires root approval."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import gate  # noqa: E402
import prepare
import parser as outparser  # noqa: E402  (local module: code/parser.py)


def manifest_items(root, splits):
    m = common.read_json(Path(root) / "manifests" / "split_manifest.json")
    if m["status"] != "READY_FOR_ROOT_REVIEW":
        raise RuntimeError("STOP: split manifest not ready")
    return [x for x in m["items"] if x["split"] in splits]


class Ledger:
    def __init__(self, root, cfg):
        self.path = Path(root) / "runs" / "GENERATION_LEDGER.json"
        self.cap = cfg["caps"]["hard_total_generations"]
        self.state = common.read_json(self.path) if self.path.exists() else {"generations": 0, "prefill_forwards": 0}

    def take(self, kind):
        if kind == "generations" and self.state["generations"] >= self.cap:
            raise RuntimeError("STOP: hard generation cap reached")
        self.state[kind] = self.state.get(kind, 0) + 1
        common.write_json(self.path, self.state)


def input_ids_for(tok, item, cond):
    p = item["prompts"][cond]
    if common.sha256_text(p["text"]) != p["sha256"]:
        raise RuntimeError("prompt hash mismatch " + item["uid"])
    ids = tok(p["text"], add_special_tokens=False)["input_ids"]
    if len(ids) != p["n_tokens"]:
        raise RuntimeError("token count mismatch " + item["uid"])
    return ids


def run_items(model, tok, items, cond, cfg, out_dir, tag, ledger, capture=False, steer=None):
    import runtime
    recs, resid = [], []
    for it in items:
        ids = input_ids_for(tok, it, cond)
        rec = {k: it[k] for k in ("uid", "id", "source", "split", "label", "group")}
        rec.update(condition=cond, config=tag, prompt_sha256=it["prompts"][cond]["sha256"], n_input_tokens=len(ids))
        ledger.take("generations")
        try:
            g = runtime.generate(model, ids, cfg, capture=capture, steer=steer, ledger=ledger)
            text = tok.decode(g["generated_ids"], skip_special_tokens=False)
            rec.update(generated_ids=g["generated_ids"], generated_text=text, stop_reason=g["stop_reason"],
                       first_token=g["first_token"], elapsed_s=g["elapsed_s"], resources=g["resources"], error=None)
            if capture:
                resid.append(g["residuals"])
        except Exception as exc:  # retained in denominators, never dropped
            text, rec = "", {**rec, "generated_ids": [], "generated_text": "", "stop_reason": "error", "error": repr(exc)}
            if capture:
                resid.append(np.full((cfg["model"]["n_layers"], cfg["model"]["width"]), np.nan, np.float32))
        rec["score"] = outparser.score(text, rec["stop_reason"], it["tool"], it["label"], it["possible_answer"])
        common.append_jsonl(out_dir / f"generations_{tag}_{cond}.jsonl", rec)
        recs.append(rec)
        if rec.get("error"):
            common.write_json(out_dir / "FAILURE_STOP.json", rec)
            raise RuntimeError("STOP: generation failure retained; no replacement or retry")
    if capture:
        np.save(out_dir / f"residuals_{tag}_{cond}.npy", np.stack(resid))
        common.write_json(out_dir / f"residuals_{tag}_{cond}.index.json", [it["uid"] for it in items])
    return recs


def run_prefill_items(model, tok, items, cond, cfg, out_dir, tag, ledger):
    import runtime
    res = []
    for it in items:
        ledger.take("prefill_forwards")
        result = runtime.prefill(model, input_ids_for(tok, it, cond), cfg, ledger=ledger)
        res.append(result["residuals"])
        common.append_jsonl(out_dir / f"prefill_{tag}_{cond}.jsonl", {
            "uid": it["uid"], "split": it["split"], "condition": cond,
            "prompt_sha256": it["prompts"][cond]["sha256"],
            "first_token": result["first_token"], "residual_shape": list(result["residuals"].shape),
            "residuals_finite": bool(np.isfinite(result["residuals"]).all()), "generation_performed": False})
    array = np.stack(res)
    np.save(out_dir / f"residuals_{tag}_{cond}.npy", array)
    common.write_json(out_dir / f"residuals_{tag}_{cond}.index.json", [it["uid"] for it in items])
    return array


def run_query_only(model, tok, items, cfg, out_dir, ledger):
    return run_prefill_items(model, tok, items, common.QUERY_ONLY, cfg, out_dir, "query_only", ledger)


def fresh_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"STOP: {path} not empty (single-shot; retain and move aside only with root approval)")
    path.mkdir(parents=True, exist_ok=True)
    return path


def rate(recs, key):
    vals = [r["score"][key] for r in recs if r["score"][key] is not None]
    return (sum(map(bool, vals)) / len(vals)) if vals else None


def steering_vectors(root, cfg, cond, layer):
    """MM direction, random and permuted-label controls, all norm-matched (alpha=1 units)."""
    d = Path(root) / "runs" / "dev"
    X = np.load(d / f"residuals_baseline_{cond}.npy")[:, layer, :]
    uids = common.read_json(d / f"residuals_baseline_{cond}.index.json")
    items = {x["uid"]: x for x in manifest_items(root, ["TRAIN"])}
    keep = [i for i, u in enumerate(uids) if u in items]
    if not np.isfinite(X[keep]).all():
        raise RuntimeError("STOP: incomplete/nonfinite training activations")
    X, y = X[keep], np.array([items[uids[i]]["label"] for i in keep])
    scale = float(np.median(np.linalg.norm(X, axis=1)))
    def unit(v):
        norm = np.linalg.norm(v)
        if not np.isfinite(v).all() or not np.isfinite(norm) or norm <= 0:
            raise RuntimeError("STOP: invalid direction")
        return v / norm
    s = cfg["steering"]
    yp = np.random.default_rng(s["permuted_seed"]).permutation(y)
    vecs = {"mm": unit(X[y == 1].mean(0) - X[y == 0].mean(0)),
            "permuted": unit(X[yp == 1].mean(0) - X[yp == 0].mean(0))}
    for k in range(s["n_random_controls"]):
        vecs[f"random{k}"] = unit(np.random.default_rng(s["random_seed_base"] + k).standard_normal(X.shape[1]))
    return {k: v.astype(np.float32) for k, v in vecs.items()}, scale


def kl_for(model, tok, base_recs, items, cond, steers, cfg, ledger):
    import runtime
    by = {r["uid"]: r for r in base_recs}
    vals = []
    for it in items:
        if it["label"] == 0 and by[it["uid"]]["stop_reason"] != "error":
            gen = [t for t in by[it["uid"]]["generated_ids"]]
            vals.append(runtime.completion_kl(model, input_ids_for(tok, it, cond), gen, steers, ledger=ledger))
    out = []
    for j in range(len(steers)):
        xs = [v[j] for v in vals if v[j] is not None]
        out.append(float(np.mean(xs)) if xs else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", required=True)
    ap.add_argument("--phase", required=True, choices=["preflight", "dev-run", "dev-steer", "heldout"])
    args = ap.parse_args()
    root = Path(args.study_root).resolve()
    cfg = common.load_config()
    common.set_offline_env(root)
    import torch
    import runtime
    torch.manual_seed(cfg["splits"]["seed_split"])
    verification = common.verify_staged(root, cfg, include_model=True)
    tok, model = runtime.load_model(root, cfg)
    import importlib.metadata
    actual_versions = {name: importlib.metadata.version(name) for name in cfg["runtime_versions"]}
    if actual_versions != cfg["runtime_versions"]:
        raise RuntimeError("STOP: runtime versions changed: " + repr(actual_versions))
    runtime_record = {"utc": common.now_utc(), "actual_model": cfg["model"]["id"], "model_revision": cfg["model"]["revision"],
                      "dtype": str(next(model.parameters()).dtype), "device": str(next(model.parameters()).device),
                      "attention": cfg["model"]["attn_implementation"], "package_versions": actual_versions,
                      "model_weights_changed": False, "external_functions_executed": False}
    ledger = Ledger(root, cfg)
    if args.phase == "preflight":
        if (root / "runs/preflight.json").exists() or (root / "runs/preflight_input.json").exists():
            raise RuntimeError("STOP: preflight attempt already consumed")
        row = next(x for x in common.read_jsonl(root / "staged/bfcl/BFCL_v4_simple_python.json") if x["id"] == "simple_python_0")
        tool, _ = prepare.canonical_tool(row["function"][0], cfg)
        prompt = prepare.render(tok, cfg, prepare.single_user_query(row), tool, "TEXT_CATALOGUE")
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        common.write_json(root / "runs/preflight_input.json", {"calibration_only_id": row["id"], "rendered_prompt": prompt,
                          "prompt_sha256": common.sha256_text(prompt), "generation_repeats": 2, "not_in_main_bank": True})
        rec = runtime.preflight(model, ids, cfg, ledger=ledger)
        common.write_json(root / "runs" / "preflight.json", {**rec, "verification": verification, "utc": common.now_utc()})
        print(json.dumps(rec))
        if not rec["pass"]:
            sys.exit(3)
        return
    if not common.read_json(root / "runs" / "preflight.json")["pass"]:
        raise RuntimeError("STOP: preflight not passed")
    if args.phase == "dev-run":
        out = fresh_dir(root / "runs" / "dev")
        common.write_json(out / "RUNTIME.json", runtime_record)
        items = manifest_items(root, ["TRAIN", "DEV"])
        common.write_json(out / "schedule.json", {"items": [x["uid"] for x in items], "conditions": list(common.CONDITIONS),
                                                  "generations": 2 * sum(x["split"] == "DEV" for x in items), "training_generations": 0, "query_only_prefills": len(items), "training_catalogue_prefills": 2 * sum(x["split"] == "TRAIN" for x in items)})
        train, dev = [x for x in items if x["split"] == "TRAIN"], [x for x in items if x["split"] == "DEV"]
        for cond in common.CONDITIONS:
            train_X = run_prefill_items(model, tok, train, cond, cfg, out, "trainingprefill", ledger)
            run_items(model, tok, dev, cond, cfg, out, "baseline", ledger, capture=True)
            dev_X = np.load(out / f"residuals_baseline_{cond}.npy")
            np.save(out / f"residuals_baseline_{cond}.npy", np.concatenate([train_X, dev_X]))
            common.write_json(out / f"residuals_baseline_{cond}.index.json", [x["uid"] for x in train + dev])
        run_query_only(model, tok, items, cfg, out, ledger)
        return
    if args.phase == "dev-steer":
        sel = common.read_json(root / "frozen" / "probe_selection.json")
        dev_dir = root / "runs" / "dev"
        out = fresh_dir(dev_dir / "steering")
        dev_all = manifest_items(root, ["DEV"])
        dev = [x for x in dev_all if x["steering_subset"]]
        steering_weights = {}
        s, steering = cfg["steering"], {}
        for cond in common.CONDITIONS:
            base_all = [r for r in common.read_jsonl(dev_dir / f"generations_baseline_{cond}.jsonl") if r["split"] == "DEV"]
            subset_uids = {x["uid"] for x in dev}
            base = [r for r in base_all if r["uid"] in subset_uids]
            mm = sel["selected"][f"MM__{cond}"]
            entry = {"dev_auroc": mm["dev_auroc"], "dev_false_call_rate": rate(base_all, "false_call")}
            ok = (mm["dev_auroc"] is not None and mm["dev_auroc"] >= s["entry_min_dev_auroc"]
                  and entry["dev_false_call_rate"] is not None and entry["dev_false_call_rate"] >= s["entry_min_dev_false_call_rate"])
            steering[cond] = {"entry": entry, "entry_pass": ok, "layer": mm["layer"], "sweep": [], "selected": None}
            if not ok:
                steering[cond]["status"] = "SKIPPED_ENTRY_GATE"
                continue
            vecs, scale = steering_vectors(root, cfg, cond, mm["layer"])
            steering[cond]["residual_norm_scale"] = scale
            for name, value in vecs.items(): steering_weights[cond + "__" + name] = value
            configs = [(a, sg) for a in s["alphas"] for sg in s["signs"]]
            tens = {c: (mm["layer"], torch.tensor(c[0] * c[1] * scale * vecs["mm"], device=runtime.DEVICE)) for c in configs}
            kls = kl_for(model, tok, base, dev, cond, [tens[c] for c in configs], cfg, ledger)
            b = {k: rate(base, k) for k in ("false_call", "correct_call", "invalid_or_empty")}
            for c, kl in zip(configs, kls):
                recs = run_items(model, tok, dev, cond, cfg, out, f"mm_a{c[0]}_s{c[1]}", ledger, steer=tens[c])
                r = {k: rate(recs, k) for k in b}
                row = {"alpha": c[0], "sign": c[1], **{f"dev_{k}": v for k, v in r.items()}, "completion_kl": kl}
                row["eligible"] = (c[1] == 1 and kl is not None
                                   and b["false_call"] - r["false_call"] >= s["min_false_call_reduction"]
                                   and b["correct_call"] - r["correct_call"] <= s["max_correct_call_drop"]
                                   and r["invalid_or_empty"] - b["invalid_or_empty"] <= s["max_invalid_increase"]
                                   and kl <= s["max_completion_kl"])
                steering[cond]["sweep"].append(row)
            el = [x for x in steering[cond]["sweep"] if x["eligible"]]
            if el:
                best = sorted(el, key=lambda x: (x["dev_false_call"], x["alpha"]))[0]
                steering[cond].update(selected={"alpha": best["alpha"], "sign": 1}, status="SELECTED")
            else:
                steering[cond]["status"] = "STEERING_GATE_FAILED"
            steering[cond]["dev_baseline"] = b
        np.savez(root / "frozen/steering_vectors.npz", **steering_weights)
        selected = {"design_id": cfg["design_id"], "created_utc": common.now_utc(),
                    "probe_selection_sha256": common.sha256_file(root / "frozen" / "probe_selection.json"),
                    "probe_weights_sha256": common.sha256_file(root / "frozen" / "probe_weights.npz"),
                    "baseline_models_sha256": common.sha256_file(root / "frozen/baseline_models.json"),
                    "control_weights_sha256": common.sha256_file(root / "frozen/control_weights.npz"),
                    "steering_vectors_sha256": common.sha256_file(root / "frozen/steering_vectors.npz"),
                    "probe_selection": sel["selected"], "steering": steering}
        common.write_json(root / "frozen" / "selected_params.json", selected)
        files = sorted(p for p in dev_dir.rglob("*") if p.is_file())
        common.write_json(dev_dir / "DEV_RECORD.json", {"created_utc": common.now_utc(), "ledger": ledger.state,
                                                        "files": {str(p.relative_to(dev_dir)): common.sha256_file(p) for p in files if p.name != "DEV_RECORD.json"},
                                                        "heldout_outputs_accessed": False})
        print(json.dumps({c: steering[c].get("status") for c in steering}))
        return
    # ---------------- heldout (gated, single shot) ----------------
    approval_sha = gate.require_approval(root, "HELDOUT_BASELINE")
    out = fresh_dir(root / "runs" / "heldout")
    common.write_json(out / "RUNTIME.json", runtime_record)
    common.write_json(out / "HELDOUT_STARTED.json", {"utc": common.now_utc(), "approval_sha256": approval_sha})
    items = manifest_items(root, ["HELDOUT"])
    selected = common.read_json(root / "frozen" / "selected_params.json")
    base = {}
    for cond in common.CONDITIONS:
        base[cond] = run_items(model, tok, items, cond, cfg, out, "baseline", ledger, capture=True)
    run_query_only(model, tok, items, cfg, out, ledger)
    steer_status = {}
    for cond in common.CONDITIONS:
        st = selected["steering"][cond]
        if st.get("status") != "SELECTED":
            steer_status[cond] = st.get("status")
            continue
        gate.require_approval(root, "HELDOUT_STEERING")
        saved_vectors = np.load(root / "frozen/steering_vectors.npz", allow_pickle=False)
        vecs = {k.split("__", 1)[1]: saved_vectors[k] for k in saved_vectors.files if k.startswith(cond + "__")}
        scale = st["residual_norm_scale"]
        steering_items = [x for x in items if x["steering_subset"]]
        steering_uids = {x["uid"] for x in steering_items}
        steering_base = [x for x in base[cond] if x["uid"] in steering_uids]
        a = st["selected"]["alpha"]
        plan = [("mm_plus", vecs["mm"], 1), ("mm_minus", vecs["mm"], -1), ("permuted", vecs["permuted"], 1)] + \
               [(k, v, 1) for k, v in vecs.items() if k.startswith("random")]
        tens = [(name, (st["layer"], torch.tensor(sg * a * scale * v, device=runtime.DEVICE))) for name, v, sg in plan]
        kls = kl_for(model, tok, steering_base, steering_items, cond, [t for _, t in tens], cfg, ledger)
        for (name, t), kl in zip(tens, kls):
            run_items(model, tok, steering_items, cond, cfg, out, "steer_" + name, ledger, steer=t)
            common.append_jsonl(out / "steering_kl.jsonl", {"condition": cond, "config": name, "alpha": a, "completion_kl": kl})
        steer_status[cond] = "RUN"
    common.write_json(out / "HELDOUT_COMPLETE.json", {"utc": common.now_utc(), "ledger": ledger.state, "steering": steer_status})


if __name__ == "__main__":
    main()
