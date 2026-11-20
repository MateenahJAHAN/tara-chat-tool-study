"""Synthetic scoring regressions; no model, generated-code execution or source downloads."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import parser as p


TOOL = {"type": "function", "function": {"name": "f", "description": "Synthetic.",
        "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}}
REFERENCE = [{"f": {"x": [1]}}]


def tagged(args=None, name="f"):
    return '<tool_call>' + json.dumps({"name": name, "arguments": {"x": 1} if args is None else args}) + '</tool_call>'


class ParserRegressionTests(unittest.TestCase):
    def score(self, text, tool=TOOL, label=0, reference=REFERENCE, stop="eos"):
        return p.score(text, stop, tool, label, reference)

    def assert_invalid(self, text, tool=TOOL):
        result = self.score(text, tool)
        self.assertTrue(result["call_attempt"], result)
        self.assertFalse(result["valid_call"], result)
        self.assertFalse(result["correct_call"], result)
        self.assertTrue(result["invalid_or_empty"], result)
        return result

    def spec(self, spec):
        tool = copy.deepcopy(TOOL)
        tool["function"]["parameters"]["properties"]["x"] = spec
        return tool

    def test_canonical_control(self):
        result = self.score(tagged())
        self.assertTrue(result["valid_call"] and result["correct_call"] and result["format_compliant"])
        self.assertFalse(result["invalid_or_empty"])

    def test_exact_abstention_only(self):
        for text in ("NO_CALL", "  NO_CALL \n", "NO_CALL<|im_end|>"):
            with self.subTest(text=text):
                result = self.score(text, label=1)
                self.assertTrue(result["operational_abstention"] and result["strict_abstention"])
                self.assertFalse(result["invalid_or_empty"])
        for text in ("Perhaps tomorrow.", "NO_CALLS are permitted later.", "Reason: NO_CALL", "NO<|im_end|>_CALL", "", '{"other":1}'):
            with self.subTest(text=text):
                result = self.score(text, label=1)
                self.assertFalse(result["operational_abstention"] or result["strict_abstention"])
                self.assertTrue(result["invalid_or_empty"])

    def test_non_eos_cannot_abstain(self):
        for stop in ("error", "max_new_tokens", "other"):
            result = self.score("NO_CALL", label=1, stop=stop)
            self.assertFalse(result["strict_abstention"])
            self.assertTrue(result["invalid_or_empty"])

    def test_name_agnostic_primary(self):
        for text in ("unknown_function(x=1)", tagged(name="unknown_function"), '{"name":"unknown_function","parameters":{"x":1}}'):
            result = self.assert_invalid(text)
            self.assertTrue(result["unknown_name"])
            self.assertTrue(self.score(text, label=1)["false_call"])

    def test_malformed_unwrapped_intents(self):
        for text in ('{"name":"f","arguments":{"x":', '{"name":"f"}', '{"arguments":{"x":1}}',
                     "unknown_function(x=", "</tool_call>", "<tool_call>garbage", "{'name':'f','arguments':{'x':1}}"):
            with self.subTest(text=text):
                result = self.assert_invalid(text)
                self.assertTrue(result["parse_error"])

    def test_whole_output_multiple_encodings(self):
        texts = [tagged() + "\nunknown_function(x=2)",
                 '{"name":"f","arguments":{"x":1}}\nunknown_function(x=2)',
                 "f(x=1); unknown_function(x=2)",
                 "[f(x=1), unknown_function(x=2)]",
                 tagged() + tagged(),
                 '<tool_call>[{"name":"f","arguments":{"x":1}},{"name":"g","arguments":{}}]</tool_call>',
                 '<tool_call>{"name":"f","arguments":{"x":1}} {"name":"g","arguments":{}}</tool_call>',
                 '{"name":"f","arguments":\n' + tagged(),
                 'f(x=\n' + tagged()]
        for text in texts:
            with self.subTest(text=text):
                result = self.assert_invalid(text)
                self.assertEqual(result["n_attempts"], 2, result)
                self.assertTrue(result["multiple_calls"])
                self.assertFalse(result["format_compliant"])

    def test_chained_or_computed_expression_not_valid_prefix(self):
        for text in ('f(x=1).g()', 'f(x=1)[0](x=2)', 'f(x=1)(x=2)', 'f(x=1).attribute'):
            with self.subTest(text=text):
                result = self.assert_invalid(text)
                self.assertTrue(result["parse_error"])

    def test_only_whole_canonical_block_format(self):
        for text in ("Here is a call:\n" + tagged(), tagged() + "\nDone.", 'f(x=1)',
                     '{"name":"f","arguments":{"x":1}}', '<tool_call>f(x=1)</tool_call>',
                     '<tool_call>```json\n{"name":"f","arguments":{"x":1}}\n```</tool_call>'):
            result = self.score(text)
            self.assertEqual(result["n_attempts"], 1, result)
            self.assertFalse(result["format_compliant"], result)
            self.assertTrue(result["invalid_or_empty"], result)

    def test_no_nested_argument_double_count(self):
        tool = self.spec({"type": "object"})
        for value in ({"name": "g", "arguments": {"x": 2}}, {"text": "<tool_call>g(x=2)</tool_call>"}, {"text": "g(x=2)"}):
            text = tagged({"x": value})
            result = self.score(text, tool)
            self.assertEqual(result["n_attempts"], 1, result)
            self.assertTrue(result["valid_call"] and result["format_compliant"], result)

    def test_lossy_python_is_invalid(self):
        for text in ("f(999,x=1)", "f(x=999,x=1)", "f(*[1],x=1)", "f(**{'x':1})", "f(x=side_effect())", "f(x={[]:1})", "f(x={'a':1,'a':2})", "f(x=(y:=1))"):
            with self.subTest(text=text):
                result = self.assert_invalid(text)
                self.assertIsNone(result["scorer_error"], result)

    def test_duplicate_json_and_nonfinite_rejected(self):
        for payload in ('{"name":"f","name":"g","arguments":{"x":1}}',
                        '{"name":"f","arguments":{"x":999,"x":1}}',
                        '{"name":"f","arguments":{"x":{"a":1,"a":2}}}',
                        '{"name":"f","arguments":{"x":NaN}}',
                        '{"name":"f","arguments":{"x":Infinity}}',
                        '{"name":"f","arguments":{"x":1e999}}'):
            for wrapped in (payload, '<tool_call>' + payload + '</tool_call>'):
                with self.subTest(text=wrapped):
                    result = self.assert_invalid(wrapped)
                    self.assertFalse(result["format_compliant"])

    def test_recursive_schema(self):
        cases = [({"type": "array", "items": {"type": "integer"}}, ["bad"]),
                 ({"type": "object", "properties": {"y": {"type": "integer"}}, "required": ["y"]}, {}),
                 ({"type": "object", "properties": {"y": {"type": "string", "enum": ["red"]}}}, {"y": "blue"}),
                 ({"type": "object", "additionalProperties": False, "properties": {"y": {"type": "integer"}}}, {"z": 1}),
                 ({"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}, [[True]]),
                 ({"type": "array", "minItems": 2}, [1]),
                 ({"type": "string", "pattern": "^a$"}, "b")]
        for schema, value in cases:
            with self.subTest(schema=schema, value=value):
                self.assert_invalid(tagged({"x": value}), self.spec(schema))

    def test_bool_enum_is_not_integer_recursively(self):
        cases = [({"type": ["integer", "boolean"], "enum": [1]}, True),
                 ({"type": "object", "enum": [{"y": 1}]}, {"y": True}),
                 ({"type": "array", "enum": [[1]]}, [True])]
        for schema, value in cases:
            self.assert_invalid(tagged({"x": value}), self.spec(schema))
        self.assertTrue(p.type_ok(1, "number"))
        self.assertFalse(p.type_ok(True, "integer"))
        self.assertFalse(p.type_ok(True, "number"))

    def test_unknown_and_malformed_schemas_fail_closed(self):
        for schema in ({"type": "custom_tag"}, {"type": ["string", "custom_tag"]}, {"type": []},
                       {"type": {}}, {"type": "object", "required": "x"},
                       {"type": "object", "properties": []}, {"enum": {}}, {"enum": []},
                       {"additionalProperties": 1}, {"items": ["integer"]}, {"allOf": []},
                       {"pattern": []}, {"minimum": True}):
            with self.subTest(schema=schema):
                self.assert_invalid(tagged({"x": 1}), self.spec(schema))

    def test_invalid_aggregate_includes_known_invalid_classes(self):
        for text in (tagged(name="g"), tagged({}), tagged({"x": True}), "f(x=side_effect())", "nonsense", tagged()+tagged()):
            self.assertTrue(self.score(text)["invalid_or_empty"])
        for stop in ("error", "max_new_tokens"):
            result = self.score(tagged(), stop=stop)
            self.assertTrue(result["call_attempt"] and result["invalid_or_empty"])
            self.assertFalse(result["correct_call"])

    def test_known_C0_optional_reference(self):
        tool = {"type": "function", "function": {"name": "calculate_triangle_area", "description": "C0 reference.",
                "parameters": {"type": "object", "properties": {"base": {"type": "integer"},
                    "height": {"type": "integer"}, "unit": {"type": "string"}}, "required": ["base", "height"]}}}
        reference = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
        for args, expected in (({"base":10,"height":5}, True), ({"base":10,"height":5,"unit":"units"}, True),
                               ({"base":10,"height":5,"unit":"meters"}, False), ({"base":10}, False),
                               ({"base":True,"height":5}, False), ({"base":10,"height":5,"unit":""}, False),
                               ({"base":10,"height":5,"other":1}, False)):
            with self.subTest(args=args):
                self.assertEqual(self.score(tagged(args, "calculate_triangle_area"), tool, reference=reference)["correct_call"], expected)

    def test_BFCL_inspired_normalization_and_nested_reference(self):
        tool = self.spec({"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string"}}})
        reference = [{"f": {"x": [{"city": ["New York"], "unit": ["metric", ""]}]}}]
        for args, expected in (({"x":{"city":"new_york"}}, True), ({"x":{"city":"New York","unit":"metric"}}, True),
                               ({"x":{"city":"New York","unit":"wrong"}}, False), ({"x":{"unit":"metric"}}, False)):
            self.assertEqual(self.score(tagged(args), tool, reference=reference)["correct_call"], expected)

    def test_large_integer_reference_does_not_collapse(self):
        self.assertFalse(p.value_eq(9007199254740993, 9007199254740992))
        self.assertTrue(p.value_eq(10**400, 10**400))
        self.assertFalse(self.score(tagged({"x":9007199254740993}), reference=[{"f":{"x":[9007199254740992]}}])["correct_call"])

    def test_bounded_total_error_records(self):
        cases = [None, 3, "x"*(p.MAX_OUTPUT_CHARS+1), '<tool_call>'+'['*100+'0'+']'*100+'</tool_call>']
        for text in cases:
            result = self.score(text)
            self.assertTrue(result["invalid_or_empty"])
            self.assertFalse(result["strict_abstention"])
        self.assertTrue(p.score(tagged(), "eos", {}, 0)["invalid_or_empty"])
        huge = self.score("x"*(p.MAX_OUTPUT_CHARS+1))
        self.assertFalse(huge["call_attempt"])
        self.assertIsNotNone(huge["scorer_error"])

    def test_no_generated_execution(self):
        # This name deliberately has no definition. The parser only visits AST.
        self.assert_invalid("f(x=side_effect())")
        self.assert_invalid("f(x=__import__('module_that_does_not_exist'))")

    def test_legacy_API_shape(self):
        attempts, extra = p.extract_attempts("unknown_function(x=1)", {"f"})
        self.assertEqual(len(attempts), 1)
        self.assertEqual(extra, 0)
        self.assertTrue(p.validity_issues("f", {"x":True}, TOOL))
        self.assertTrue(p.matches_answer("f", {"x":1}, REFERENCE, TOOL))


if __name__ == "__main__":
    unittest.main(verbosity=2)
