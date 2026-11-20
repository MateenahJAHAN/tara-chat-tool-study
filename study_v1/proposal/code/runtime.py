"""Offline FP32/MPS/eager model runtime: manual greedy decoding, residual capture, additive steering, KL.
Hook pattern adapted from the operator's CALIBRATION_STARTER.py (original operator code). Requires torch."""
import time
from pathlib import Path

import numpy as np
import torch

DEVICE = "mps"


def load_model(study_root, cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = Path(study_root) / "staged" / "model"
    if not torch.backends.mps.is_available():
        raise RuntimeError("STOP: declared MPS backend unavailable; no silent fallback")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                 dtype=torch.float32, attn_implementation=cfg["model"]["attn_implementation"])
    model.to(DEVICE).eval()
    if len(model.model.layers) != cfg["model"]["n_layers"] or next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("STOP: unexpected model structure/dtype")
    return tok, model


class Hooks:
    """capture=True stores block outputs at the last position of the FIRST forward only (prefill).
    steer=(layer_index, vector) adds vector to that block's output at every position of every forward."""

    def __init__(self, model, capture=False, steer=None):
        self.layers, self.capture, self.steer = model.model.layers, capture, steer
        self.active, self.store, self.handles = capture, {}, []

    def _capture(self, i):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if not isinstance(h, torch.Tensor) or h.ndim != 3:
                raise RuntimeError("Unexpected decoder hook output")
            if self.active:
                self.store[i] = h[0, -1, :].detach().float().cpu().numpy().copy()
        return hook

    def _steer(self, vec):
        def hook(module, args, output):
            if isinstance(output, tuple):
                return (output[0] + vec,) + tuple(output[1:])
            return output + vec
        return hook

    def __enter__(self):
        if self.steer is not None:
            i, vec = self.steer
            self.handles.append(self.layers[i].register_forward_hook(self._steer(vec)))
        if self.capture:  # registered after steering so captured states include the intervention
            for i, layer in enumerate(self.layers):
                self.handles.append(layer.register_forward_hook(self._capture(i)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        return False

    def residuals(self):
        if not self.capture:
            return None
        if len(self.store) != len(self.layers):
            raise RuntimeError("missing captured layers")
        return np.stack([self.store[i] for i in range(len(self.layers))]).astype(np.float32)


def first_token_stats(logits, cfg):
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("STOP: nonfinite model logits")
    lp = torch.log_softmax(logits.float(), dim=-1)
    if not bool(torch.isfinite(lp).all()):
        raise RuntimeError("STOP: nonfinite log probabilities")
    top = torch.topk(lp, 5)
    return {"lp_tool_call": float(lp[cfg["model"]["tool_call_token_id"]]), "lp_no": float(lp[cfg["model"]["no_token_id"]]),
            "top5_ids": [int(x) for x in top.indices], "top5_lp": [float(x) for x in top.values],
            "entropy": float(-(lp.exp() * lp).sum()), "finite": bool(torch.isfinite(logits).all())}


def resources():
    try:
        return {"mps_allocated": int(torch.mps.current_allocated_memory()), "mps_driver": int(torch.mps.driver_allocated_memory())}
    except Exception:  # resource probe must never alter results
        return {}


@torch.inference_mode()
def generate(model, input_ids, cfg, capture=False, steer=None, max_new_tokens=None, ledger=None):
    max_new = max_new_tokens if max_new_tokens is not None else cfg["model"]["max_new_tokens"]
    eos = set(cfg["model"]["eos_token_ids"])
    ids = torch.tensor([input_ids], device=DEVICE)
    t0 = time.monotonic()
    with Hooks(model, capture=capture, steer=steer) as hk:
        if ledger is not None: ledger.take("generation_model_forwards")
        out = model(input_ids=ids, use_cache=True)
        hk.active = False
        if capture and not np.isfinite(hk.residuals()).all():
            raise RuntimeError("STOP: nonfinite prefill residuals")
        logits = out.logits[0, -1]
        first = first_token_stats(logits, cfg)
        past, gen, stop = out.past_key_values, [], "max_new_tokens"
        nxt = int(torch.argmax(logits))
        for step in range(max_new):
            gen.append(nxt)
            if nxt in eos:
                stop = "eos"
                break
            if step == max_new - 1:
                break
            if ledger is not None: ledger.take("generation_model_forwards")
            out = model(input_ids=torch.tensor([[nxt]], device=DEVICE), past_key_values=past, use_cache=True)
            past = out.past_key_values
            if not bool(torch.isfinite(out.logits[0, -1]).all()):
                raise RuntimeError("STOP: nonfinite decode logits")
            nxt = int(torch.argmax(out.logits[0, -1]))
        torch.mps.synchronize()
        res = hk.residuals()
    if res is not None and not np.isfinite(res).all():
        raise RuntimeError("non-finite residuals")
    return {"generated_ids": gen, "stop_reason": stop, "first_token": first, "residuals": res,
            "elapsed_s": time.monotonic() - t0, "resources": resources()}


@torch.inference_mode()
def prefill(model, input_ids, cfg, steer=None, ledger=None):
    with Hooks(model, capture=True, steer=steer) as hk:
        if ledger is not None: ledger.take("prefill_model_forwards")
        out = model(input_ids=torch.tensor([input_ids], device=DEVICE), use_cache=False)
        res = hk.residuals()
    if not np.isfinite(res).all():
        raise RuntimeError("STOP: nonfinite prefill residuals")
    return {"residuals": res, "first_token": first_token_stats(out.logits[0, -1], cfg)}


@torch.inference_mode()
def completion_kl(model, prompt_ids, completion_ids, steers, ledger=None):
    """Mean per-token KL(unsteered || steered), teacher-forced on the unsteered completion,
    excluding the first generated position. Returns list aligned with steers (None if <2 tokens)."""
    if len(completion_ids) < 2:
        return [None] * len(steers)
    seq = torch.tensor([list(prompt_ids) + list(completion_ids)], device=DEVICE)
    lo, hi = len(prompt_ids), len(prompt_ids) + len(completion_ids) - 1
    if ledger is not None: ledger.take("kl_model_forwards")
    base = torch.log_softmax(model(input_ids=seq, use_cache=False).logits[0, lo:hi].float(), -1)
    if not bool(torch.isfinite(base).all()):
        raise RuntimeError("STOP: nonfinite baseline KL logits")
    out = []
    for st in steers:
        with Hooks(model, steer=st):
            if ledger is not None: ledger.take("kl_model_forwards")
            lp = torch.log_softmax(model(input_ids=seq, use_cache=False).logits[0, lo:hi].float(), -1)
        if not bool(torch.isfinite(lp).all()):
            raise RuntimeError("STOP: nonfinite steered KL logits")
        kl = float((base.exp() * (base - lp)).sum(-1).mean())
        if not np.isfinite(kl) or kl < -1e-5:
            raise RuntimeError("STOP: invalid computed KL")
        out.append(kl)
    return out


@torch.inference_mode()
def preflight(model, input_ids, cfg, ledger=None):
    """G3: finiteness, zero-change hook identity, repeat-greedy identity. Raises on failure."""
    tol = cfg["model"]["hook_identity_tol"]
    ids = torch.tensor([input_ids], device=DEVICE)
    if ledger is not None: ledger.take("preflight_model_forwards")
    base = model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu()
    with Hooks(model, capture=True) as hk:
        if ledger is not None: ledger.take("preflight_model_forwards")
        hooked = model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu()
        res = hk.residuals()
    zero = torch.zeros(cfg["model"]["width"], device=DEVICE)
    with Hooks(model, steer=(cfg["model"]["n_layers"] // 2, zero)):
        if ledger is not None: ledger.take("preflight_model_forwards")
        zsteer = model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu()
    if ledger is not None: ledger.take("generations")
    g1 = generate(model, input_ids, cfg, ledger=ledger)["generated_ids"]
    if ledger is not None: ledger.take("generations")
    g2 = generate(model, input_ids, cfg, ledger=ledger)["generated_ids"]
    rec = {"logits_finite": bool(torch.isfinite(base).all()), "residuals_finite": bool(np.isfinite(res).all()),
           "residual_shape": list(res.shape), "capture_hook_max_abs": float((base - hooked).abs().max()),
           "zero_steer_hook_max_abs": float((base - zsteer).abs().max()), "repeat_greedy_identical": g1 == g2,
           "n_generated": len(g1), "generated_token_ids_repeats": [g1, g2], "torch": torch.__version__}
    rec["pass"] = (rec["logits_finite"] and rec["residuals_finite"] and rec["capture_hook_max_abs"] <= tol
                   and rec["zero_steer_hook_max_abs"] <= tol and rec["repeat_greedy_identical"]
                   and rec["residual_shape"] == [cfg["model"]["n_layers"], cfg["model"]["width"]])
    return rec
