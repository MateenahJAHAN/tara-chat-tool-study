"""Dev-discovered format/semantic separation; synthetic public-shaped cases only."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import parser as p

TOOL = {"type": "function", "function": {"name": "calculate_triangle_area", "description": "Public C0 fixture.",
        "parameters": {"type": "object", "properties": {"base": {"type": "integer"},
            "height": {"type": "integer"}, "unit": {"type": "string"}}, "required": ["base", "height"]}}}
REFERENCE = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]


class FormatOnlyAmendmentTests(unittest.TestCase):
    def prefix(self, args=None, name="calculate_triangle_area"):
        return '<tool_call>' + json.dumps({"name": name, "arguments": {"base": 10, "height": 5} if args is None else args})

    def score(self, text, stop="eos"):
        return p.score(text, stop, TOOL, 0, REFERENCE)

    def test_complete_payload_missing_close_is_format_only(self):
        for args in ({"base":10,"height":5}, {"base":10,"height":5,"unit":"units"}):
            for suffix in ("", "\n", "<|im_end|>"):
                with self.subTest(args=args, suffix=suffix):
                    r = self.score(self.prefix(args) + suffix)
                    self.assertEqual(r["n_attempts"], 1)
                    self.assertTrue(r["valid_call"] and r["correct_call"])
                    self.assertFalse(r["parse_error"] or r["format_compliant"])
                    self.assertTrue(r["invalid_or_empty"] and r["protocol_format_error"])
                    self.assertEqual(r["format_diagnostics"], ["unclosed_tag"])
                    self.assertEqual(r["parser_diagnostics"], [])

    def test_complete_closed_control(self):
        r = self.score(self.prefix() + '</tool_call>')
        self.assertTrue(r["valid_call"] and r["correct_call"] and r["format_compliant"])
        self.assertFalse(r["invalid_or_empty"])

    def test_incomplete_json_is_semantic_failure(self):
        for text in ('<tool_call>{"name":"calculate_triangle_area","arguments":{"base":10,',
                     '<tool_call>{"name":"calculate_triangle_area","arguments":',
                     '<tool_call>{"name":"calculate_triangle_area"}'):
            r = self.score(text)
            self.assertTrue(r["call_attempt"] and r["parse_error"] and r["invalid_or_empty"])
            self.assertFalse(r["valid_call"] or r["correct_call"] or r["format_compliant"])

    def test_unclosed_python_embedded_or_alias_is_not_exempt(self):
        for text in ('<tool_call>calculate_triangle_area(base=10,height=5)',
                     '<tool_call>```json\n{"name":"calculate_triangle_area","arguments":{"base":10,"height":5}}\n```',
                     '<tool_call>{"name":"calculate_triangle_area","parameters":{"base":10,"height":5}}',
                     'prefix '+self.prefix()):
            r = self.score(text)
            # Extra surrounding text is not covered by this narrow exemption.
            self.assertTrue(r["parse_error"] and r["invalid_or_empty"])
            self.assertFalse(r["valid_call"] or r["correct_call"])

    def test_extra_call_or_text_is_not_excused_by_missing_close(self):
        for suffix in ('\nunknown_function(x=1)', '\n{"name":"g","arguments":{}}', ' extra text',
                       '<tool_call>{"name":"g","arguments":{}}</tool_call>'):
            r = self.score(self.prefix() + suffix)
            self.assertTrue(r["call_attempt"] and r["invalid_or_empty"])
            self.assertFalse(r["valid_call"] or r["correct_call"] or r["format_compliant"])

    def test_bad_payload_values_still_invalid_or_incorrect(self):
        for args in ({"base":True,"height":5}, {"base":10}, {"base":10,"height":5,"other":1}):
            r = self.score(self.prefix(args))
            self.assertFalse(r["valid_call"] or r["correct_call"])
        r = self.score(self.prefix({"base":10,"height":5,"unit":"meters"}))
        self.assertTrue(r["valid_call"])
        self.assertFalse(r["correct_call"])
        self.assertFalse(self.score(self.prefix(name="unknown_function"))["valid_call"])
        self.assertTrue(self.score(self.prefix(name="unknown_function"))["parse_error"])

    def test_non_eos_never_correct(self):
        for stop in ("error", "max_new_tokens"):
            r = self.score(self.prefix(), stop)
            self.assertTrue(r["valid_call"] and r["invalid_or_empty"])
            self.assertFalse(r["correct_call"])

    def test_primary_attempt_unchanged(self):
        for close in ("", '</tool_call>'):
            r = p.score(self.prefix() + close, 'eos', TOOL, 1)
            self.assertTrue(r["call_attempt"] and r["false_call"])


if __name__ == '__main__':
    unittest.main(verbosity=2)
