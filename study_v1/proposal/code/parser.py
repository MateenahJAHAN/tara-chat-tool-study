"""Original bounded, execution-free project scorer; BFCL-inspired, not official BFCL.

String normalization, omission sentinel "", and nested dictionary alternative-list
matching follow the conventions in the pinned BFCL AST checker (Apache-2.0):
https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py
This is an original implementation, not copied evaluator code. Whole-output intent
accounting, strict protocol compliance, recursive schema checks and finite-value
requirements are project-specific. No generated expression is ever evaluated.
"""
import ast
import json
import math
import re

TAG_OPEN, TAG_CLOSE = "<tool_call>", "</tool_call>"
END_MARKERS = ("<|im_end|>", "<|endoftext|>")
MAX_OUTPUT_CHARS = 65536
MAX_DEPTH = 32
MAX_AST_NODES = 2048
FENCE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*$")
CALL_START = re.compile(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\(")
CALL_KEYS = re.compile(r"[\"'](?:name|arguments|parameters)[\"']\s*:")
TYPES = {"integer", "number", "string", "boolean", "array", "object", "null"}
ANNOTATIONS = {"description", "title", "default", "examples", "$comment", "deprecated", "readOnly", "writeOnly", "format"}
KEYWORDS = {"type", "properties", "required", "enum", "items", "additionalProperties", "minItems", "maxItems",
            "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "pattern", "uniqueItems"}


class ParseFailure(ValueError):
    pass


def clean(text):
    """Strip terminal decoder markers only; internal markers remain observable."""
    if not isinstance(text, str):
        raise ParseFailure("output_not_string")
    text = text.rstrip()
    while True:
        marker = next((m for m in END_MARKERS if text.endswith(m)), None)
        if marker is None:
            return text
        text = text[:-len(marker)].rstrip()


def _finite(value, depth=0):
    if depth > MAX_DEPTH:
        raise ParseFailure("value_depth_limit")
    if isinstance(value, float) and not math.isfinite(value):
        raise ParseFailure("nonfinite_number")
    if isinstance(value, (list, tuple)):
        for item in value:
            _finite(item, depth + 1)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ParseFailure("object_key_not_string")
            _finite(item, depth + 1)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ParseFailure("value_not_json")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ParseFailure("duplicate_json_key:" + key)
        result[key] = value
    return result


def _constant(value):
    raise ParseFailure("nonstandard_json_constant:" + value)


def _loads(text):
    value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    return _finite(value)


def _empty_attempt(error, kind="unparseable", name=None):
    return {"name": name, "args": None, "parse_mode": "unparseable", "strict_json": False,
            "keys": None, "kind": kind, "closed": False, "errors": [error]}


def safe_literal(node, depth=0):
    if depth > MAX_DEPTH:
        raise ParseFailure("literal_depth_limit")
    if isinstance(node, ast.Constant):
        return _finite(node.value)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [safe_literal(x, depth + 1) for x in node.elts]
    if isinstance(node, ast.Dict):
        result = {}
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                raise ParseFailure("dict_unpacking")
            key = safe_literal(key_node, depth + 1)
            if not isinstance(key, str):
                raise ParseFailure("dict_key_not_string")
            if key in result:
                raise ParseFailure("duplicate_python_dict_key:" + key)
            result[key] = safe_literal(value_node, depth + 1)
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = safe_literal(node.operand, depth + 1)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParseFailure("non_numeric_unary")
        return _finite(-value if isinstance(node.op, ast.USub) else value)
    if isinstance(node, ast.Name) and node.id in {"true", "false", "null"}:
        return {"true": True, "false": False, "null": None}[node.id]
    raise ParseFailure("nonliteral:" + type(node).__name__)


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return base + "." + node.attr if base else None
    return None


def _python_payload(text):
    tree = ast.parse(text.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ParseFailure("ast_node_limit")
    call = tree.body
    if not isinstance(call, ast.Call):
        raise ParseFailure("not_single_python_call")
    name = dotted_name(call.func)
    if name is None:
        raise ParseFailure("unsupported_call_target")
    result = {"name": name, "args": None, "parse_mode": "python_ast", "strict_json": False,
              "keys": None, "kind": "untagged_python", "closed": False, "errors": []}
    if call.args:
        result["errors"].append("positional_arguments_unsupported")
    kwargs = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            result["errors"].append("keyword_unpacking_unsupported")
            continue
        if keyword.arg in kwargs:
            result["errors"].append("duplicate_keyword:" + keyword.arg)
            continue
        try:
            kwargs[keyword.arg] = safe_literal(keyword.value)
        except Exception as exc:
            result["errors"].append("invalid_literal:" + type(exc).__name__ + ":" + str(exc))
    if not result["errors"]:
        result["args"] = kwargs
    return result


def parse_python_call(src):
    """Compatibility helper: a lossy/unsafe call returns (name,None,n_positional)."""
    try:
        if not isinstance(src, str) or len(src) > MAX_OUTPUT_CHARS:
            return None
        result = _python_payload(src)
        tree = ast.parse(src.strip(), mode="eval")
        return result["name"], result["args"], len(tree.body.args)
    except Exception:
        return None


def _object_payload(obj):
    result = {"name": obj.get("name") if isinstance(obj.get("name"), str) else None,
              "args": obj.get("arguments", obj.get("parameters")), "parse_mode": "json", "strict_json": True,
              "keys": sorted(obj), "kind": "untagged_json", "closed": False, "errors": []}
    if set(obj) not in ({"name", "arguments"}, {"name", "parameters"}):
        result["errors"].append("unexpected_call_keys")
    if result["name"] is None:
        result["errors"].append("name_not_string")
    if not isinstance(result["args"], dict):
        result["errors"].append("arguments_not_object")
    return result


def parse_payload(payload):
    try:
        if not isinstance(payload, str) or len(payload) > MAX_OUTPUT_CHARS:
            return _empty_attempt("payload_size_or_type")
        value = payload.strip()
        # Fences are understood for semantics, but never strict canonical format.
        unfenced = "\n".join(line for line in value.splitlines() if not FENCE.match(line.strip())).strip()
        try:
            obj = _loads(unfenced)
        except Exception as json_exc:
            if unfenced.startswith(("{", "[")) or CALL_KEYS.search(unfenced):
                return _empty_attempt("invalid_json:" + type(json_exc).__name__ + ":" + str(json_exc))
            result = _python_payload(unfenced)
            return result
        if not isinstance(obj, dict):
            return _empty_attempt("call_payload_not_object")
        result = _object_payload(obj)
        if unfenced != value:
            result["strict_json"] = False
        return result
    except Exception as exc:
        return _empty_attempt("invalid_payload:" + type(exc).__name__ + ":" + str(exc))


def _quoted_end(text, start):
    quote = text[start]
    triple = text.startswith(quote * 3, start)
    delimiter = quote * (3 if triple else 1)
    i = start + len(delimiter)
    while i < len(text):
        if text[i] == "\\":
            i += 2
        elif text.startswith(delimiter, i):
            return i + len(delimiter)
        else:
            i += 1
    return len(text)


def _balanced_end(text, start):
    closing = {"{": "}", "[": "]", "(": ")"}
    stack, i = [], start
    while i < len(text):
        char = text[i]
        if char in "\"'":
            i = _quoted_end(text, i)
            continue
        if char in closing:
            stack.append(closing[char])
            if len(stack) > MAX_DEPTH:
                return len(text), False
        elif char in "}])":
            if not stack or stack.pop() != char:
                return i + 1, False
            if not stack:
                return i + 1, True
        i += 1
    return len(text), False


def _tag_boundary(text, start):
    i = start
    while i < len(text):
        if text[i] in "\"'":
            i = _quoted_end(text, i)
        elif text.startswith(TAG_CLOSE, i):
            return i, i + len(TAG_CLOSE), True
        elif text.startswith(TAG_OPEN, i):
            return i, i, False
        else:
            i += 1
    return len(text), len(text), False


def _calllike(obj):
    # A missing name/argument member is malformed call syntax, not abstention.
    # Outer valid calls are consumed before their argument values are traversed.
    return isinstance(obj, dict) and bool(set(obj) & {"name", "arguments", "parameters"})


def _nested_objects(value):
    if _calllike(value):
        return [_object_payload(value)]
    result = []
    if isinstance(value, dict):
        for child in value.values():
            result.extend(_nested_objects(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_nested_objects(child))
    return result


def _recovery_end(text, start, end):
    """After an unbalanced candidate, resume at a separate line's call syntax."""
    cursor = text.find("\n", start + 1, end)
    while cursor >= 0:
        next_start = cursor + 1
        while next_start < end and text[next_start] in " \t":
            next_start += 1
        if (text.startswith((TAG_OPEN, TAG_CLOSE, "{"), next_start)
                or CALL_START.match(text, next_start)):
            return next_start
        cursor = text.find("\n", cursor + 1, end)
    return end


def _computed_expression(text, start, first_end):
    """Account for call chains; never certify a valid prefix of unsupported code."""
    cursor = first_end
    while cursor < len(text) and text[cursor] in " \t":
        cursor += 1
    if cursor >= len(text) or text[cursor] not in ".[":
        # Also reject a call result being called directly: f(...)(...).
        if cursor >= len(text) or text[cursor] != "(":
            return None
    ends = [x for x in (text.find("\n", cursor), text.find(";", cursor)) if x >= 0]
    end = min(ends) if ends else len(text)
    expression = text[start:end]
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        nodes = list(ast.walk(tree))
        if len(nodes) > MAX_AST_NODES:
            raise ParseFailure("ast_node_limit")
        calls = [node for node in nodes if isinstance(node, ast.Call)]
        attempts = []
        for node in calls:
            segment = ast.get_source_segment(expression, node) or expression
            try:
                attempt = _python_payload(segment)
            except Exception:
                attempt = _empty_attempt("computed_call_target", "untagged_python")
            attempt["errors"].append("computed_or_chained_expression_unsupported")
            attempt["args"] = None
            attempts.append(attempt)
        return attempts or [_empty_attempt("computed_expression_unsupported", "untagged_python")], end
    except Exception:
        return [_empty_attempt("computed_expression_unsupported", "untagged_python")], end


def _scan(text, allow_tags=True):
    attempts, i = [], 0
    while i < len(text):
        if allow_tags and text.startswith(TAG_OPEN, i):
            payload_start = i + len(TAG_OPEN)
            payload_end, end, closed = _tag_boundary(text, payload_start)
            payload = text[payload_start:payload_end]
            parsed = parse_payload(payload)
            # A malformed envelope can contain several explicit calls. Scan its
            # payload only when the whole payload did not already parse as one.
            inner = _scan(payload, allow_tags=False) if parsed["errors"] else []
            if inner:
                group = inner
                for attempt in group:
                    attempt["errors"] = list(attempt["errors"]) + ["noncanonical_tag_payload"]
            else:
                group = [parsed]
            for attempt in group:
                attempt.update(kind="tagged", closed=closed)
                if not closed:
                    attempt["errors"].append("unclosed_tag")
            attempts.extend(group)
            i = max(end, payload_start)
            continue
        if allow_tags and text.startswith(TAG_CLOSE, i):
            attempts.append(_empty_attempt("orphan_closing_tag", "tagged"))
            i += len(TAG_CLOSE)
            continue
        if text[i] == "{":
            end, balanced = _balanced_end(text, i)
            if not balanced:
                end = _recovery_end(text, i, end)
            chunk = text[i:end]
            try:
                value = _loads(chunk) if balanced else None
                if balanced:
                    attempts.extend(_nested_objects(value))
                elif CALL_KEYS.search(chunk):
                    attempts.append(_empty_attempt("malformed_unwrapped_json", "untagged_json"))
            except Exception as exc:
                if CALL_KEYS.search(chunk):
                    attempts.append(_empty_attempt("invalid_json:" + type(exc).__name__ + ":" + str(exc), "untagged_json"))
            i = max(end, i + 1)
            continue
        match = CALL_START.match(text, i)
        if match and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] in "_.")):
            paren = match.end() - 1
            end, balanced = _balanced_end(text, paren)
            if not balanced:
                end = _recovery_end(text, i, end)
            if balanced:
                computed = _computed_expression(text, i, end)
                if computed is not None:
                    group, end = computed
                    attempts.extend(group)
                    i = max(end, match.end())
                    continue
            chunk = text[i:end]
            if balanced:
                try:
                    attempts.append(_python_payload(chunk))
                except Exception as exc:
                    attempts.append(_empty_attempt("invalid_python:" + type(exc).__name__ + ":" + str(exc), "untagged_python"))
            else:
                attempts.append(_empty_attempt("malformed_unwrapped_python", "untagged_python"))
            i = max(end, match.end())
            continue
        if text[i] in "\"'" and (i == 0 or not text[i - 1].isalnum()):
            i = _quoted_end(text, i)
        else:
            i += 1
    return attempts


def extract_attempts(text, catalogue_names):
    """Legacy (attempts,broad_extra) API; every detected name is now primary.

    Closing-only/malformed tags, call-shaped JSON and Python syntax are intents.
    Nested argument dictionaries/strings are consumed with their outer call.
    catalogue_names is retained for compatibility; name filtering is not used.
    """
    del catalogue_names
    try:
        body = clean(text)
        if len(body) > MAX_OUTPUT_CHARS:
            return [_empty_attempt("output_size_limit")], 0
        return _scan(body), 0
    except Exception as exc:
        return [_empty_attempt("extraction_error:" + type(exc).__name__)], 0


def type_ok(value, kind):
    try:
        _finite(value)
        if isinstance(kind, list):
            return bool(kind) and all(isinstance(k, str) and k in TYPES for k in kind) and any(type_ok(value, k) for k in kind)
        if not isinstance(kind, str) or kind not in TYPES:
            return False
        return {"integer": lambda: isinstance(value, int) and not isinstance(value, bool),
                "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
                "string": lambda: isinstance(value, str), "boolean": lambda: isinstance(value, bool),
                "array": lambda: isinstance(value, list), "object": lambda: isinstance(value, dict),
                "null": lambda: value is None}[kind]()
    except Exception:
        return False


def _json_equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return type_ok(a, "number") and type_ok(b, "number") and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _schema_issues(schema, path="$", depth=0):
    if depth > MAX_DEPTH or not isinstance(schema, dict):
        return ["unsupported_schema:" + path]
    issues = ["unsupported_keyword:" + path + ":" + k for k in schema if k not in KEYWORDS | ANNOTATIONS]
    if "type" in schema:
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not kinds or not all(isinstance(k, str) and k in TYPES for k in kinds):
            issues.append("unsupported_type:" + path)
    props = schema.get("properties", {})
    if not isinstance(props, dict) or not all(isinstance(k, str) for k in props):
        return issues + ["invalid_properties:" + path]
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(k, str) for k in required):
        issues.append("invalid_required:" + path)
    if "enum" in schema:
        if not isinstance(schema["enum"], list) or not schema["enum"]:
            issues.append("invalid_enum:" + path)
        else:
            try:
                _finite(schema["enum"])
            except Exception:
                issues.append("invalid_enum:" + path)
    for key, child in props.items():
        issues.extend(_schema_issues(child, path + "." + key, depth + 1))
    if "items" in schema:
        issues.extend(_schema_issues(schema["items"], path + "[]", depth + 1))
    additional = schema.get("additionalProperties", True)
    if isinstance(additional, dict):
        issues.extend(_schema_issues(additional, path + ".*", depth + 1))
    elif not isinstance(additional, bool):
        issues.append("invalid_additionalProperties:" + path)
    for key in ("minItems", "maxItems", "minLength", "maxLength"):
        if key in schema and (not isinstance(schema[key], int) or isinstance(schema[key], bool) or schema[key] < 0):
            issues.append("invalid_" + key + ":" + path)
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema and not type_ok(schema[key], "number"):
            issues.append("invalid_" + key + ":" + path)
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        issues.append("invalid_uniqueItems:" + path)
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except Exception:
            issues.append("invalid_pattern:" + path)
    return issues


def _validate(value, schema, path="$", depth=0):
    if depth > MAX_DEPTH:
        return ["value_depth_limit:" + path]
    issues = []
    if "type" in schema and not type_ok(value, schema["type"]):
        return ["type:" + path]
    if "enum" in schema and not any(_json_equal(value, option) for option in schema["enum"]):
        issues.append("enum:" + path)
    if isinstance(value, dict):
        props = schema.get("properties", {})
        issues.extend("missing:" + path + "." + key for key in schema.get("required", []) if key not in value)
        for key, item in value.items():
            if key in props:
                issues.extend(_validate(item, props[key], path + "." + key, depth + 1))
            elif path == "$" or schema.get("additionalProperties", True) is False:
                issues.append("unknown_arg:" + path + "." + key)
            elif isinstance(schema.get("additionalProperties"), dict):
                issues.extend(_validate(item, schema["additionalProperties"], path + "." + key, depth + 1))
    elif isinstance(value, list):
        if "items" in schema:
            for index, item in enumerate(value):
                issues.extend(_validate(item, schema["items"], path + "[" + str(index) + "]", depth + 1))
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            issues.append("array_length:" + path)
        if schema.get("uniqueItems") and any(_json_equal(value[i], value[j]) for i in range(len(value)) for j in range(i)):
            issues.append("uniqueItems:" + path)
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", len(value)):
            issues.append("string_length:" + path)
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            issues.append("pattern:" + path)
    if type_ok(value, "number"):
        for key, compare in (("minimum", lambda a, b: a < b), ("maximum", lambda a, b: a > b),
                             ("exclusiveMinimum", lambda a, b: a <= b), ("exclusiveMaximum", lambda a, b: a >= b)):
            if key in schema and compare(value, schema[key]):
                issues.append(key + ":" + path)
    return issues


def validity_issues(name, args, tool):
    try:
        function = tool["function"]
        issues = [] if isinstance(name, str) and name == function["name"] else ["unknown_name"]
        schema = function["parameters"]
        issues.extend(_schema_issues(schema))
        if not isinstance(args, dict):
            return issues + ["args_not_object"]
        try:
            _finite(args)
        except Exception as exc:
            return issues + ["invalid_value:" + str(exc)]
        if issues and any(x.startswith(("unsupported_", "invalid_")) for x in issues):
            return issues
        return issues + _validate(args, schema)
    except Exception as exc:
        return ["validator_error:" + type(exc).__name__]


def standardize(text):
    return re.sub(r"[ ,./\-_*^]", "", text).lower().replace("'", '"')


def value_eq(value, answer, kind=None, depth=0):
    if depth > MAX_DEPTH:
        return False
    try:
        _finite(value)
        _finite(answer)
        if isinstance(value, bool) or isinstance(answer, bool):
            return type(value) is type(answer) and value == answer
        if isinstance(value, str) and isinstance(answer, str):
            return standardize(value) == standardize(answer)
        if type_ok(value, "number") and type_ok(answer, "number"):
            return not (kind == "integer" and isinstance(value, float)) and value == answer
        if isinstance(value, list) and isinstance(answer, list):
            return len(value) == len(answer) and all(value_eq(v, a, depth=depth + 1) for v, a in zip(value, answer))
        if isinstance(value, dict) and isinstance(answer, dict):
            if any(key not in answer for key in value):
                return False
            for key, alternatives in answer.items():
                alternatives = alternatives if isinstance(alternatives, list) else [alternatives]
                if key in value:
                    if not any(option != "" and value_eq(value[key], option, depth=depth + 1) for option in alternatives):
                        return False
                elif "" not in alternatives:
                    return False
            return True
        return type(value) is type(answer) and value == answer
    except Exception:
        return False


def matches_answer(name, args, ground_truth, tool):
    try:
        if validity_issues(name, args, tool) or not isinstance(ground_truth, list) or len(ground_truth) != 1:
            return False
        reference = ground_truth[0]
        if not isinstance(reference, dict) or len(reference) != 1:
            return False
        expected_name, parameters = next(iter(reference.items()))
        if name != expected_name or not isinstance(parameters, dict) or any(key not in parameters for key in args):
            return False
        props = tool["function"]["parameters"].get("properties", {})
        for key, alternatives in parameters.items():
            if not isinstance(alternatives, list):
                return False
            if key in args:
                kind = props.get(key, {}).get("type")
                if not any(option != "" and value_eq(args[key], option, kind) for option in alternatives):
                    return False
            elif "" not in alternatives:
                return False
        return True
    except Exception:
        return False


def _format_compliant(body, attempts):
    if len(attempts) != 1 or not body.startswith(TAG_OPEN) or not body.endswith(TAG_CLOSE):
        return False
    attempt = attempts[0]
    if attempt["kind"] != "tagged" or not attempt["closed"] or attempt["errors"]:
        return False
    payload = body[len(TAG_OPEN):-len(TAG_CLOSE)].strip()
    try:
        obj = _loads(payload)
        return (isinstance(obj, dict) and set(obj) == {"name", "arguments"}
                and isinstance(obj["name"], str) and isinstance(obj["arguments"], dict))
    except Exception:
        return False


def _score(text, stop_reason, tool, label, possible_answer):
    body = clean(text).strip()
    if len(body) > MAX_OUTPUT_CHARS:
        raise ParseFailure("output_size_limit")
    attempts, broad_extra = extract_attempts(body, {tool["function"]["name"]})
    one = attempts[0] if len(attempts) == 1 else None
    parser_errors = [error for attempt in attempts for error in attempt["errors"]]
    issues = validity_issues(one["name"], one["args"], tool) if one else None
    if one and one["errors"]:
        issues = list(issues or []) + list(one["errors"])
    exact_no_call = body == "NO_CALL"
    unknown = bool(body) and not attempts and not exact_no_call
    format_ok = _format_compliant(body, attempts)
    parse_error = bool(parser_errors) or any(attempt["name"] is None or not isinstance(attempt["args"], dict) for attempt in attempts)
    unknown_name = any(attempt["name"] is not None and attempt["name"] != tool["function"]["name"] for attempt in attempts)
    truncated, errored = stop_reason == "max_new_tokens", stop_reason == "error"
    invalid_stop = stop_reason not in {"eos", "max_new_tokens", "error"}
    valid = one is not None and not issues
    # Semantic correctness and protocol compliance remain separate diagnostics.
    # The invalid-output gate includes noncanonical call protocol, not just JSON
    # decoding failures. This definition is explicit and prospective.
    invalid = (not body or truncated or errored or invalid_stop or unknown or parse_error or unknown_name
               or len(attempts) > 1 or bool(attempts) and (not valid or not format_ok))
    abstention = exact_no_call and not attempts and stop_reason == "eos"
    return {
        "n_attempts": len(attempts), "call_attempt": bool(attempts),
        "broad_call_attempt": bool(attempts) or broad_extra > 0,
        "attempt_kinds": [attempt["kind"] for attempt in attempts],
        "attempt_names": [attempt["name"] for attempt in attempts],
        "parse_error": parse_error, "unknown_name": unknown_name, "multiple_calls": len(attempts) > 1,
        "format_compliant": format_ok, "valid_call": valid, "validity_issues": issues,
        "empty_output": not body, "truncated": truncated, "errored": errored,
        "no_call_literal": exact_no_call, "operational_abstention": abstention, "strict_abstention": abstention,
        "false_call": bool(attempts) if label == 1 else None,
        "broad_false_call": bool(attempts) if label == 1 else None,
        "correct_call": bool(valid and stop_reason == "eos" and matches_answer(one["name"], one["args"], possible_answer, tool)) if label == 0 else None,
        "invalid_or_empty": bool(invalid), "unknown_output": unknown,
        "protocol_format_error": bool(attempts) and not format_ok,
        "parser_diagnostics": parser_errors, "scorer_error": None,
        "invalid_stop_reason": invalid_stop,
    }


def score(text, stop_reason, tool, label, possible_answer=None):
    """Total scorer: failures become explicit invalid records, never dropped rows."""
    try:
        return _score(text, stop_reason, tool, label, possible_answer)
    except Exception as exc:
        sample = text[:MAX_OUTPUT_CHARS] if isinstance(text, str) else ""
        intent = bool(TAG_OPEN in sample or TAG_CLOSE in sample or CALL_START.search(sample) or CALL_KEYS.search(sample))
        return {"n_attempts": int(intent), "call_attempt": intent, "broad_call_attempt": intent,
                "attempt_kinds": ["scorer_failure"] if intent else [], "attempt_names": [None] if intent else [],
                "parse_error": True, "unknown_name": False, "multiple_calls": False,
                "format_compliant": False, "valid_call": False, "validity_issues": ["scorer_failure"],
                "empty_output": not bool(sample.strip()), "truncated": stop_reason == "max_new_tokens",
                "errored": stop_reason == "error", "no_call_literal": False,
                "operational_abstention": False, "strict_abstention": False,
                "false_call": intent if label == 1 else None, "broad_false_call": intent if label == 1 else None,
                "correct_call": False if label == 0 else None, "invalid_or_empty": True,
                "unknown_output": True, "protocol_format_error": intent,
                "parser_diagnostics": [type(exc).__name__ + ":" + str(exc)],
                "scorer_error": type(exc).__name__, "invalid_stop_reason": True}
