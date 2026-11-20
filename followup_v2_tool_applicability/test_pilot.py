import json
import unittest
from pathlib import Path

from pilot import MODES, build_cases, matching_call, request_body, score, validate_bank


class PilotTest(unittest.TestCase):
    def test_frozen_bank_and_exact_tool_catalogue(self):
        bank = json.loads((Path(__file__).parent / "bank.json").read_text())
        validate_bank(bank)
        cases = build_cases(bank)
        self.assertEqual(len(cases), 40)
        self.assertEqual(sum(row["expected"] is not None for row in cases), 20)
        self.assertFalse({row["query"] for row in cases} & {row["query"] for row in bank["controls"]})
        text, native = (request_body("local", bank["tools"], cases[0]["query"], mode) for mode in MODES)
        self.assertEqual(text["messages"][1], native["messages"][1])
        self.assertEqual(json.loads(text["messages"][0]["content"].split("Functions:\n")[-1]), [item["function"] for item in native["tools"]])
        for key in ("seed", "temperature", "max_tokens", "chat_template_kwargs"):
            self.assertEqual(text[key], native[key])

    def test_canonical_vs_semantic(self):
        expected = {"name": "count", "arguments": {"items": 3}}
        raw = json.dumps(expected)
        self.assertTrue(score({"content": raw}, "text", expected)["canonical_correct_call"])
        fancy = score({"content": f"<tool_call>{raw}"}, "text", expected)
        self.assertTrue(fancy["semantic_correct_call"])
        self.assertFalse(fancy["canonical_correct_call"])
        native = {"type": "function", "function": {"name": "count", "arguments": '{"items":3.0}'}}
        self.assertTrue(score({"tool_calls": [native]}, "native", expected)["canonical_correct_call"])
        self.assertFalse(score({"tool_calls": [native, native]}, "native", expected)["semantic_correct_call"])
        self.assertTrue(score({"content": "<tool_call>{\"name\":"}, "native", None)["call_attempt"])

    def test_abstention_and_argument_integrity(self):
        self.assertTrue(score({"content": " NO_CALL\n"}, "native", None)["exact_abstain"])
        self.assertFalse(score({"content": "NO_CALL because unavailable"}, "native", None)["exact_abstain"])
        expected = {"name": "count", "arguments": {"items": 1}}
        self.assertFalse(matching_call({"name": "count", "arguments": {"items": True}}, expected))
        self.assertFalse(matching_call({"name": "count", "arguments": '{"items":3,"items":1}'}, expected))
        self.assertFalse(score({"content": '{"name":"count","arguments":{"items":1}} again {"name":"count","arguments":{"items":1}}'}, "text", expected)["semantic_correct_call"])


if __name__ == "__main__":
    unittest.main()
