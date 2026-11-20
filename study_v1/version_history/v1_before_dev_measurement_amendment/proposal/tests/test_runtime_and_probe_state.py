"""Small CPU synthetic controls; no pretrained weights or BFCL/model outcomes."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import common
import probes
import runtime


class TinyForward(torch.nn.Module):
    def __init__(self, bad_call=None, bad_residual=False):
        super().__init__()
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity()]))
        self.calls, self.bad_call, self.bad_residual = 0, bad_call, bad_residual

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        hidden = torch.ones((1, input_ids.shape[1], 2))
        if self.bad_residual:
            hidden[:] = float("nan")
        hidden = self.model.layers[0](hidden)
        logits = torch.zeros((1, input_ids.shape[1], 6)) + hidden[..., :1] * 0
        logits[..., 1] = 2
        if self.calls == self.bad_call:
            logits[:] = float("nan")
        return SimpleNamespace(logits=logits, past_key_values=None)


CFG = {"model": {"max_new_tokens": 3, "eos_token_ids": [4], "tool_call_token_id": 1,
                 "no_token_id": 2, "n_layers": 1, "width": 2}}


class RuntimeGuards(unittest.TestCase):
    def setUp(self):
        self.device = runtime.DEVICE
        self.sync = torch.mps.synchronize
        runtime.DEVICE = "cpu"
        torch.mps.synchronize = lambda: None

    def tearDown(self):
        runtime.DEVICE = self.device
        torch.mps.synchronize = self.sync

    def test_nonfinite_first_logits_stop(self):
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            runtime.generate(TinyForward(bad_call=1), [1, 2], CFG)

    def test_nonfinite_later_decode_stops(self):
        model = TinyForward(bad_call=2)
        with self.assertRaisesRegex(RuntimeError, "nonfinite decode"):
            runtime.generate(model, [1, 2], CFG)
        self.assertEqual(model.calls, 2)

    def test_nonfinite_residual_stops_prefill(self):
        with self.assertRaisesRegex(RuntimeError, "nonfinite prefill residual"):
            runtime.prefill(TinyForward(bad_residual=True), [1, 2], CFG)

    def test_nonfinite_kl_stops(self):
        with self.assertRaisesRegex(RuntimeError, "nonfinite baseline KL"):
            runtime.completion_kl(TinyForward(bad_call=1), [1, 2], [1, 1], [None])


def fixture(index, label):
    return {"uid": str(index), "label": label, "query": ("triangle area " if label else "weather city ") + str(index),
            "tool": {"type": "function", "function": {"name": "area" if label else "weather",
                "description": "geometry measure" if label else "temperature forecast",
                "parameters": {"type": "object", "properties": {"value": {"type": "number"}}, "required": ["value"]}}},
            "prompts": {"TEXT_CATALOGUE": {"n_tokens": 25 + index}}}


class FittedStateChecks(unittest.TestCase):
    def test_serialized_surface_models_reconstruct_scores(self):
        train = [fixture(i, i % 2) for i in range(12)]
        test = [fixture(i, i % 2) for i in range(12, 16)]
        cfg = common.load_config()
        for name in probes.BASELINES:
            with self.subTest(name=name):
                state = probes.fit_baseline(name, train, 0.1, cfg, return_state=True)
                restored = json.loads(json.dumps(state, allow_nan=False))
                a = probes.score_baseline(state, test)
                b = probes.score_baseline(restored, test)
                np.testing.assert_array_equal(a, b)

    def test_nonfinite_activation_row_cannot_be_dropped(self):
        X = np.zeros((2, 1, 2)); X[1, 0, 0] = np.nan
        items = {"a": {"split": "TRAIN", "label": 0}, "b": {"split": "TRAIN", "label": 1}}
        with self.assertRaisesRegex(RuntimeError, "no complete-case"):
            probes.subset(X, ["a", "b"], items, "TRAIN")


if __name__ == "__main__":
    unittest.main()
