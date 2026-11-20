"""Probes and surface baselines. fit-dev: fit on TRAIN only, select on DEV only, freeze.
eval-heldout: gated; frozen probes evaluated once on HELDOUT with controls.
Concepts (MM/LR probes, cross-dataset transfer) follow ARENA 1.3.1 and Marks & Tegmark 2023; code is original."""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import gate  # noqa: E402
import prepare  # noqa: E402

FEATURE_SETS = list(common.CONDITIONS) + [common.QUERY_ONLY]


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if len(set(y.tolist())) < 2 or not np.isfinite(s).all():
        return None
    return float(roc_auc_score(y, s))


def bacc(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return float(np.mean([np.mean(pred[y == c] == c) for c in (0, 1) if np.any(y == c)]))


def cluster_boot(y, s, groups, B, seed, fn=auroc):
    """Percentile CI resampling groups with replacement; resamples with one class are skipped and counted."""
    y, s, groups = np.asarray(y), np.asarray(s, dtype=float), np.asarray(groups)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    rng, vals, skipped = np.random.default_rng(seed), [], 0
    for _ in range(B):
        take = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        v = fn(y[take], s[take])
        if v is None:
            skipped += 1
        else:
            vals.append(v)
    if not vals:
        return {"lo": None, "hi": None, "skipped": skipped}
    return {"lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)), "skipped": skipped}


# ---------- probes ----------
def fit_mm(X, y):
    w = X[y == 1].mean(0) - X[y == 0].mean(0)
    thr = float((X[y == 1] @ w).mean() + (X[y == 0] @ w).mean()) / 2
    return {"w": w, "thr": thr}


def score_mm(p, X):
    return X @ p["w"]


def fit_lr(X, y, C, cfg):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, max_iter=cfg["probes"]["lr_max_iter"]).fit(sc.transform(X), y)
    return {"mean": sc.mean_, "scale": sc.scale_, "coef": m.coef_[0], "b": float(m.intercept_[0])}


def score_lr(p, X):
    return ((X - p["mean"]) / p["scale"]) @ p["coef"] + p["b"]


# ---------- data ----------
def load_split(root, phase_dir, featset, items_by_uid):
    tag = "query_only" if featset == common.QUERY_ONLY else "baseline"
    X = np.load(Path(phase_dir) / f"residuals_{tag}_{featset}.npy")
    uids = common.read_json(Path(phase_dir) / f"residuals_{tag}_{featset}.index.json")
    if len(uids) != len(set(uids)) or len(X) != len(uids):
        raise RuntimeError("STOP: activation UID/shape mismatch")
    return X, uids


def subset(X, uids, items, split):
    keep = [i for i, u in enumerate(uids) if items[u]["split"] == split]
    ok = [i for i in keep if np.isfinite(X[i]).all()]
    if len(ok) != len(keep):
        raise RuntimeError("STOP: missing/nonfinite activation rows; no complete-case filtering")
    sel = [uids[i] for i in ok]
    return X[ok], np.array([items[u]["label"] for u in sel]), sel


# ---------- surface baselines ----------
def surface_features(items, name, fitted=None):
    stop = set(common.load_config()["grouping"]["stopwords"])
    if name == "length":
        F = np.array([[len(x["query"]), len(x["query"].split()), len(x["tool"]["function"]["description"]),
                       len(x["tool"]["function"]["parameters"].get("properties", {})),
                       len(x["tool"]["function"]["parameters"].get("required", [])),
                       x["prompts"]["TEXT_CATALOGUE"]["n_tokens"]] for x in items], dtype=float)
        return F, None
    if name == "overlap":
        rows = []
        for x in items:
            q = prepare.content_words(x["query"], stop)
            f = x["tool"]["function"]
            fw = prepare.content_words(f["name"].replace("_", " ").replace(".", " ") + " " + f["description"] + " " +
                                       " ".join(f["parameters"].get("properties", {})), stop)
            nt = set(re.split(r"[_.]", f["name"].lower()))
            sh = len(q & fw)
            rows.append([sh / max(1, len(q | fw)), sh, len(q & nt), sh / max(1, len(q))])
        return np.array(rows, dtype=float), None
    text = {"tfidf_query": lambda x: x["query"], "tfidf_schema": lambda x: prepare.tool_json(x["tool"]),
            "tfidf_both": lambda x: x["query"] + "\n" + prepare.tool_json(x["tool"])}[name]
    docs = [text(x) for x in items]
    vec = fitted or TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True).fit(docs)
    return vec.transform(docs).toarray(), vec


BASELINES = ["length", "overlap", "tfidf_query", "tfidf_schema", "tfidf_both"]


def fit_baseline(name, train, C, cfg, return_state=False):
    F, vec = surface_features(train, name)
    y = np.array([x["label"] for x in train])
    center = name in ("length", "overlap")
    sc = StandardScaler(with_mean=center).fit(F)
    m = LogisticRegression(C=C, max_iter=cfg["probes"]["lr_max_iter"]).fit(sc.transform(F), y)
    state = {"feature": name, "C": C, "with_mean": center, "mean": sc.mean_.tolist(),
             "scale": sc.scale_.tolist(), "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
             "vocabulary": None if vec is None else vec.vocabulary_, "idf": None if vec is None else vec.idf_.tolist()}
    return state if return_state else lambda items: score_baseline(state, items)


def score_baseline(state, items):
    vec = None
    if state["vocabulary"] is not None:
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, vocabulary=state["vocabulary"])
        vec.idf_ = np.asarray(state["idf"])
    F, _ = surface_features(items, state["feature"], vec)
    if state["with_mean"]:
        F = F - np.asarray(state["mean"])
    return (F / np.asarray(state["scale"])) @ np.asarray(state["coef"]) + state["intercept"]


# ---------- phases ----------
def fit_dev(root, cfg):
    dev_dir = root / "runs" / "dev"
    man = common.read_json(root / "manifests" / "split_manifest.json")
    items = {x["uid"]: x for x in man["items"]}
    C_grid, sel, curves, weights, data = cfg["probes"]["C_grid"], {}, {}, {}, {}
    for fs in FEATURE_SETS:
        X, uids = load_split(root, dev_dir, fs, items)
        data[fs] = {sp: subset(X, uids, items, sp) for sp in ("TRAIN", "DEV")}
        (Xt, yt, _), (Xd, yd, _) = data[fs]["TRAIN"], data[fs]["DEV"]
        best = {"MM": None, "LR": None}
        curves[fs] = []
        for L in range(X.shape[1]):
            mm = fit_mm(Xt[:, L], yt)
            weights[f"LAYERMM__{fs}__{L}__w"] = mm["w"]
            row = {"layer": L, "MM_dev_auroc": auroc(yd, score_mm(mm, Xd[:, L]))}
            cand = [(row["MM_dev_auroc"], L, None, "MM", mm)]
            for C in C_grid:
                lr = fit_lr(Xt[:, L], yt, C, cfg)
                a = auroc(yd, score_lr(lr, Xd[:, L]))
                row[f"LR_C{C}_dev_auroc"] = a
                cand.append((a, L, C, "LR", lr))
            curves[fs].append(row)
            for a, L_, C, kind, p in cand:
                cur = best[kind]
                if a is not None and (cur is None or a > cur[0] or (a == cur[0] and (L_, C or 0) < (cur[1], cur[2] or 0))):
                    best[kind] = (a, L_, C, kind, p)
        for kind, (a, L, C, _, p) in best.items():
            s = score_mm(p, Xd[:, L]) if kind == "MM" else score_lr(p, Xd[:, L])
            thr = p["thr"] if kind == "MM" else 0.0
            sel[f"{kind}__{fs}"] = {"layer": L, "C": C, "dev_auroc": a, "dev_bacc": bacc(yd, (s > thr).astype(int)), "threshold": thr}
            for k, v in p.items():
                weights[f"{kind}__{fs}__{k}"] = np.asarray(v)
    matrix = {}
    for kind in ("MM", "LR"):
        for a in common.CONDITIONS:
            for b in common.CONDITIONS:
                L = sel[f"{kind}__{a}"]["layer"]
                p = {k.split("__")[-1]: weights[k] for k in weights if k.startswith(f"{kind}__{a}__")}
                Xd, yd, _ = data[b]["DEV"]
                matrix[f"{kind}:{a}->{b}"] = auroc(yd, (score_mm if kind == "MM" else score_lr)(p, Xd[:, L]))
    train = [items[u] for u in data["TEXT_CATALOGUE"]["TRAIN"][2]]
    dev = [items[u] for u in data["TEXT_CATALOGUE"]["DEV"][2]]
    yd = np.array([x["label"] for x in dev])
    base_sel = {}
    for name in BASELINES:
        res = [(auroc(yd, fit_baseline(name, train, C, cfg)(dev)), C) for C in C_grid]
        res = [r for r in res if r[0] is not None]
        if res:
            best_auc = max(r[0] for r in res)
            base_sel[name] = {"C": min(r[1] for r in res if r[0] == best_auc), "dev_auroc": best_auc}
    groups = {}
    for x in dev:
        groups.setdefault(x["group"], set()).add(x["label"])
    mixed = [x for x in dev if len(groups[x["group"]]) > 1]
    matched = {"n_per_class": [sum(1 for x in mixed if x["label"] == c) for c in (0, 1)]}
    if min(matched["n_per_class"]) >= cfg["probes"]["matched_subset_min_per_class"]:
        for key in ("MM__TEXT_CATALOGUE", "MM__NATIVE_TOOLS", "LR__TEXT_CATALOGUE", "LR__NATIVE_TOOLS"):
            kind, fs = key.split("__")
            Xd, ydd, du = data[fs]["DEV"]
            m = np.array([u in {x["uid"] for x in mixed} for u in du])
            p = {k.split("__")[-1]: weights[k] for k in weights if k.startswith(key + "__")}
            matched[key] = auroc(ydd[m], (score_mm if kind == "MM" else score_lr)(p, Xd[m][:, sel[key]["layer"]]))
    cos = []
    (Xa, ya, _), (Xb, yb, _) = data["TEXT_CATALOGUE"]["TRAIN"], data["NATIVE_TOOLS"]["TRAIN"]
    for L in range(Xa.shape[1]):
        wa, wb = fit_mm(Xa[:, L], ya)["w"], fit_mm(Xb[:, L], yb)["w"]
        cos.append(float(wa @ wb / (np.linalg.norm(wa) * np.linalg.norm(wb))))
    baseline_models = {name: fit_baseline(name, train, state["C"], cfg, return_state=True)
                       for name, state in base_sel.items()}
    common.write_json(root / "frozen/baseline_models.json", baseline_models)
    controls = {}
    for key, selected in sel.items():
        kind, fs = key.split("__")
        Xt, yt, _ = data[fs]["TRAIN"]
        L = selected["layer"]
        for k in range(cfg["probes"]["n_permutations"]):
            yp = np.random.default_rng(cfg["probes"]["permutation_seed_base"] + k).permutation(yt)
            fitted = fit_mm(Xt[:, L], yp) if kind == "MM" else fit_lr(Xt[:, L], yp, selected["C"], cfg)
            for field, value in fitted.items(): controls[f"CONTROL__{key}__perm{k}__{field}"] = np.asarray(value)
        if kind == "MM":
            for k in range(cfg["probes"]["n_random_directions"]):
                v = np.random.default_rng(cfg["probes"]["random_direction_seed_base"] + k).standard_normal(Xt.shape[2])
                controls[f"RANDOM__{key}__{k}"] = v / np.linalg.norm(v)
    np.savez(root / "frozen/control_weights.npz", **controls)
    np.savez(root / "frozen" / "probe_weights.npz", **weights)
    common.write_json(root / "frozen" / "probe_selection.json", {
        "design_id": cfg["design_id"], "created_utc": common.now_utc(), "selected": sel, "baselines": base_sel,
        "dev_transfer_matrix_optimistic": matrix, "dev_layer_curves": curves, "dev_matched_subset": matched,
        "train_mm_cosine_text_vs_native": cos, "label": "benchmark_irrelevant=1", "heldout_used": False})


def eval_heldout(root, cfg):
    gate.require_approval(root, "HELDOUT_PROBE_EVAL")
    held_dir, dev_dir = root / "runs" / "heldout", root / "runs" / "dev"
    if not (held_dir / "HELDOUT_COMPLETE.json").exists():
        raise RuntimeError("STOP: heldout generation incomplete")
    man = common.read_json(root / "manifests" / "split_manifest.json")
    items = {x["uid"]: x for x in man["items"]}
    sel = common.read_json(root / "frozen" / "probe_selection.json")
    W = np.load(root / "frozen" / "probe_weights.npz", allow_pickle=False)
    CW = np.load(root / "frozen/control_weights.npz", allow_pickle=False)
    baseline_models = common.read_json(root / "frozen/baseline_models.json")
    P, B, seed = cfg["probes"], cfg["probes"]["bootstrap_B"], cfg["probes"]["bootstrap_seed"]
    H, T = {}, {}
    for fs in FEATURE_SETS:
        X, uids = load_split(root, held_dir, fs, items)
        H[fs] = subset(X, uids, items, "HELDOUT")
        X, uids = load_split(root, dev_dir, fs, items)
        T[fs] = subset(X, uids, items, "TRAIN")
    out = {"matrix": {}, "controls": {}, "baselines": {}, "beyond_surface": {}, "layer_curves_descriptive": {}}
    scores = {}
    for key, s in sel["selected"].items():
        kind, src = key.split("__")
        p = {k.split("__")[-1]: W[k] for k in W.files if k.startswith(key + "__")}
        fn = score_mm if kind == "MM" else score_lr
        targets = common.CONDITIONS if src in common.CONDITIONS else [src]
        for tgt in targets:
            Xh, yh, uh = H[tgt]
            sc = fn(p, Xh[:, s["layer"]])
            g = [items[u]["group"] for u in uh]
            out["matrix"][f"{kind}:{src}->{tgt}"] = {"auroc": auroc(yh, sc), "ci": cluster_boot(yh, sc, g, B, seed),
                                                    "bacc": bacc(yh, (sc > s["threshold"]).astype(int)), "n": len(yh)}
            scores[(key, tgt)] = (yh, sc, g, uh)
        Xt, yt, _ = T[src]
        Xh, yh, uh = H[src]
        perm, rnd = [], []
        for k in range(P["n_permutations"]):
            prefix = f"CONTROL__{key}__perm{k}__"
            q = {field[len(prefix):]: CW[field] for field in CW.files if field.startswith(prefix)}
            perm.append(auroc(yh, fn(q, Xh[:, s["layer"]])))
        if kind == "MM":
            for k in range(P["n_random_directions"]):
                v = CW[f"RANDOM__{key}__{k}"]
                rnd.append(auroc(yh, Xh[:, s["layer"]] @ v))
        obs = out["matrix"][f"{kind}:{src}->{src}"]["auroc"]
        summ = lambda v: {"n": len(v), "median": float(np.median(v)), "max": float(np.max(v)),  # noqa: E731
                          "frac_ge_observed": float(np.mean([x >= obs for x in v])) if obs is not None else None} if v else None
        out["controls"][key] = {"permuted_label": summ([x for x in perm if x is not None]),
                                "random_direction": summ([x for x in rnd if x is not None])}
    train = [items[u] for u in T["TEXT_CATALOGUE"][2]]
    held = [items[u] for u in H["TEXT_CATALOGUE"][2]]
    yh = np.array([x["label"] for x in held])
    gh = [x["group"] for x in held]
    bscores = {}
    for name, b in sel["baselines"].items():
        sc = score_baseline(baseline_models[name], held)
        bscores[name] = sc
        out["baselines"][name] = {"auroc": auroc(yh, sc), "ci": cluster_boot(yh, sc, gh, B, seed), "dev_auroc": b["dev_auroc"]}
    if bscores:
        best = max(sel["baselines"], key=lambda n: (sel["baselines"][n]["dev_auroc"], n))
        uid_pos = {x["uid"]: i for i, x in enumerate(held)}
        for (key, tgt), (y, sc, g, uh) in scores.items():
            m = [i for i, u in enumerate(uh) if u in uid_pos]  # align items present in both feature sets
            y, sc, g, uh = y[m], sc[m], [g[i] for i in m], [uh[i] for i in m]
            bs = bscores[best][[uid_pos[u] for u in uh]]
            joint = np.stack([sc, bs], 1)
            diff = lambda yy, ss: (None if auroc(yy, ss[:, 0]) is None else auroc(yy, ss[:, 0]) - auroc(yy, ss[:, 1]))  # noqa: E731
            ug = np.unique(g)
            idx = {q: np.where(np.asarray(g) == q)[0] for q in ug}
            rng, vals = np.random.default_rng(seed), []
            for _ in range(B):
                t = np.concatenate([idx[q] for q in rng.choice(ug, len(ug), replace=True)])
                v = diff(y[t], joint[t])
                if v is not None:
                    vals.append(v)
            lo, hi = (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (None, None)
            out["beyond_surface"][f"{key}->{tgt}"] = {
                "best_dev_baseline": best, "auroc_diff": diff(y, joint), "ci": [lo, hi],
                "verdict": "positive_descriptive_bootstrap_interval" if (lo is not None and lo > 0) else "overlapping_or_nonpositive_descriptive_interval", "confirmatory_claim": False}
    for fs in common.CONDITIONS:
        Xt, yt, _ = T[fs]
        Xh, yh2, _ = H[fs]
        out["layer_curves_descriptive"][fs] = [auroc(yh2, Xh[:, L] @ W[f"LAYERMM__{fs}__{L}__w"]) for L in range(Xt.shape[1])]
    out["note"] = "All probe numbers and finite-bank group bootstrap intervals are exploratory/descriptive, including selected layers. Every fitted probe, baseline, permutation and random control was frozen before heldout; layer curves are descriptive. Benchmark source/style confounding remains."
    common.write_json(root / "reports" / "heldout_probe_results.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-root", required=True)
    ap.add_argument("--phase", required=True, choices=["fit-dev", "eval-heldout"])
    args = ap.parse_args()
    root, cfg = Path(args.study_root).resolve(), common.load_config()
    if args.phase == "fit-dev":
        if (root / "frozen" / "probe_selection.json").exists():
            raise RuntimeError("STOP: probe selection already frozen")
        (root / "frozen").mkdir(parents=True, exist_ok=True)
        fit_dev(root, cfg)
    else:
        eval_heldout(root, cfg)


if __name__ == "__main__":
    main()
