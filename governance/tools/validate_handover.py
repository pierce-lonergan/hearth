#!/usr/bin/env python3
"""validate_handover — check generational handover manifests.

    python governance/tools/validate_handover.py                       # all of governance/handovers/
    python governance/tools/validate_handover.py path/to/manifest.json
    python governance/tools/validate_handover.py some/dir --strict --json
    python governance/tools/validate_handover.py --new T05-governance           # print a skeleton
    python governance/tools/validate_handover.py --new T05-governance --write   # create the next file

Each manifest is checked against governance/schemas/handover.schema.json and
then against the repository:
  * every id in constraints_touched exists in governance/INVARIANTS.md;
  * task_id exists in governance/tasks.json;
  * warnings (errors with --strict): file name is not <task_id>-gen-<NNN>.json
    matching generation_id; parent_generation does not name an earlier
    manifest of the same task in the same directory (or is missing after
    generation 1); a measurement does not say whether it was measured or
    simulated (INV-HONEST).

Schema engine: `jsonschema` is used when installed (--engine auto), otherwise a
built-in validator that implements the keywords Hearth's schemas use: type
(including union types), enum, const, required, properties,
additionalProperties, items, pattern, minLength/maxLength, minItems/maxItems,
minimum/maximum, $ref to #/$defs, format. Only the "date-time" format (RFC
3339) is asserted, by both engines; other formats are annotations, as JSON
Schema 2020-12 defines them by default (jsonschema would otherwise assert
whichever formats its installed optional packages support). Any other
keyword (allOf, multipleOf, minProperties, ...) makes the schema unusable under
every engine, jsonschema included, instead of being evaluated by one engine and
refused by the other: both engines accept exactly the same schemas, so a schema
cannot validate under one engine (auto, locally) and not under the other
(builtin, in CI).

The schema is checked before any manifest, and a schema either engine cannot
use exits 2 under both. Both require: a JSON object; "$schema", if given, draft
2020-12 or 2019-09 (whose semantics the built-in engine implements); "$id" only
at the root; every "$ref" a local JSON pointer ("#/...") that resolves to a
subschema ($dynamicRef and $recursiveRef are not supported); every "pattern"
translatable (below); no "patternProperties" (jsonschema would match its keys
with Python's re, not as ECMA-262); and what the built-in engine can use: only
the keywords above and annotations, every subschema an object or boolean ("items"
is never the 2019-09 array form), keyword values of the right kind (e.g.
"required" and a "type" list hold unique strings, "title" is a string, "$id"
has no fragment, as the metaschema's pattern says) and no cycle of $refs alone. With jsonschema, the schema is then also checked against
its metaschema (formats not asserted, so the result does not depend on optional
packages).

Both engines evaluate "pattern" as an ECMA-262 regular expression with the "u"
flag (JSON Schema's dialect), translated to Python's re: "$" matches only at the
very end (Python's also matches before a final newline), "." excludes \\n \\r
U+2028 U+2029, \\d \\w \\b are ASCII, \\s is ECMA-262 white space (Python's
differs), [] never matches and [^] matches anything. Syntax that ECMA-262
rejects in "u" mode (lone braces, Python-only groups, identity escapes of
letters, a quantifier on a lookaround, \\b, \\B, ^ or $) or that has no exact
translation (backreferences, \\p{...}, variable-length lookbehind) makes the
schema unusable (exit 2) instead of being guessed at. So does a repeat count
above 65535, or nested counts whose minimums multiply to more than that: Python
performs forced repeats one by one, and (?:){2147483648} takes minutes, then
fails with MemoryError. Matching cost is not bounded otherwise (a pattern like
(a+)+$ can still backtrack for long); the schema is maintainer-owned.
Input must be strict JSON: NaN and Infinity are rejected.

A manifest that makes the validator itself fail is reported as invalid; the
other files are still checked.

Pure standard library, Python 3.9+. Exit: 0 all valid, 1 invalid manifest(s),
2 bad arguments / unusable schema.
"""
from __future__ import annotations

import argparse
import calendar
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = ROOT / "governance" / "schemas" / "handover.schema.json"
DEFAULT_DIR = ROOT / "governance" / "handovers"
INVARIANTS = ROOT / "governance" / "INVARIANTS.md"
TASKS = ROOT / "governance" / "tasks.json"


class SchemaError(Exception):
    """The schema itself uses something the built-in engine cannot evaluate."""


# ---------------------------------------------------------------- formats
_DATE_TIME = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})[Tt]([0-9]{2}):([0-9]{2}):([0-9]{2})(\.[0-9]+)?([Zz]|[+-]([0-9]{2}):([0-9]{2}))")


def is_date_time(s) -> bool:
    """RFC 3339 section 5.6 date-time (what JSON Schema's "date-time" means). Formats
    constrain strings only; a value of another type is the "type" keyword's business."""
    if not isinstance(s, str):
        return True
    m = _DATE_TIME.fullmatch(s)
    if not m:
        return False
    y, mo, d, h, mi, sec = (int(m.group(i)) for i in range(1, 7))
    if not 1 <= mo <= 12 or not 1 <= d <= calendar.monthrange(y, mo)[1]:
        return False
    if h > 23 or mi > 59 or sec > 60:  # 60 = leap second
        return False
    if m.group(9) is not None and (int(m.group(9)) > 23 or int(m.group(10)) > 59):
        return False
    return True


FORMATS = {"date-time": is_date_time}

# ------------------------------------------------- ECMA-262 regular expressions
_MAX_CP = 0x10FFFF


def _complement(ranges: list) -> list:
    out, nxt = [], 0
    for lo, hi in ranges:
        if lo > nxt:
            out.append((nxt, lo - 1))
        nxt = hi + 1
    if nxt <= _MAX_CP:
        out.append((nxt, _MAX_CP))
    return out


def _cp(c: int) -> str:
    return f"\\U{c:08x}"


def _class_body(ranges: list) -> str:
    return "".join(_cp(lo) if lo == hi else f"{_cp(lo)}-{_cp(hi)}" for lo, hi in ranges)


_DIGIT = [(0x30, 0x39)]
_WORD = [(0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A)]
_SPACE = [(0x09, 0x0D), (0x20, 0x20), (0xA0, 0xA0), (0x1680, 0x1680), (0x2000, 0x200A),   # WhiteSpace and
          (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000), (0xFEFF, 0xFEFF)]  # LineTerminator
_SETS = {"d": _DIGIT, "D": _complement(_DIGIT), "w": _WORD, "W": _complement(_WORD),
         "s": _SPACE, "S": _complement(_SPACE)}
_W = "[0-9A-Z_a-z]"
_BOUNDARY = {"b": f"(?:(?<={_W})(?!{_W})|(?<!{_W})(?={_W}))", "B": f"(?:(?<={_W})(?={_W})|(?<!{_W})(?!{_W}))"}
_CONTROL = {"t": 9, "n": 10, "v": 11, "f": 12, "r": 13}
_SYNTAX = frozenset("^$\\.*+?()[]{}|/")
_QUANT = re.compile(r"\{[0-9]+(?:,[0-9]*)?\}")
_GROUP = re.compile(r"\(\?<([A-Za-z_][A-Za-z0-9_]*)>")
_LOOKAROUND = ("(?=", "(?!", "(?<=", "(?<!")
_HEX2 = re.compile(r"[0-9A-Fa-f]{2}")
_HEX4 = re.compile(r"[0-9A-Fa-f]{4}")
_HEXN = re.compile(r"\{([0-9A-Fa-f]+)\}")
MAX_REPEAT = 65535      # largest repeat count, and largest product of nested minimum counts


def _escape(p: str, i: int, in_class: bool) -> tuple:
    """Translates the escape whose letter is p[i]: (regex, next index, code point or None)."""
    if i >= len(p):
        raise re.error("pattern ends with a backslash")
    c = p[i]
    if c in _SETS:
        body = _class_body(_SETS[c])
        return (body if in_class else f"[{body}]"), i + 1, None
    if c in _BOUNDARY and not in_class:
        return _BOUNDARY[c], i + 1, None
    v, j = None, i + 1
    if c == "b" and in_class:                                     # backspace
        v = 8
    elif c in _CONTROL:
        v = _CONTROL[c]
    elif c == "c" and p[j:j + 1].isascii() and p[j:j + 1].isalpha():
        v, j = ord(p[j]) % 32, j + 1
    elif c == "0" and not p[j:j + 1].isdigit():
        v = 0
    elif c == "x" and _HEX2.match(p, j):
        v, j = int(p[j:j + 2], 16), j + 2
    elif c == "u" and _HEXN.match(p, j) and int(_HEXN.match(p, j).group(1), 16) <= _MAX_CP:
        m = _HEXN.match(p, j)
        v, j = int(m.group(1), 16), m.end()
    elif c == "u" and _HEX4.match(p, j):
        v, j = int(p[j:j + 4], 16), j + 4
        if 0xD800 <= v <= 0xDBFF and p.startswith("\\u", j) and _HEX4.match(p, j + 2) \
                and 0xDC00 <= int(p[j + 2:j + 6], 16) <= 0xDFFF:   # a surrogate pair is one code point
            v, j = 0x10000 + ((v - 0xD800) << 10) + int(p[j + 2:j + 6], 16) - 0xDC00, j + 6
    elif c in _SYNTAX or (c == "-" and in_class):
        v = ord(c)
    elif (c in "123456789" or c == "k") and not in_class:
        # ECMA-262 matches a reference to a group that did not participate (or was reset by a
        # repeat) as empty, Python never matches it; no schema here needs them
        raise re.error(f"backreference '\\{c}' at position {i - 1} has no exact translation")
    if v is None:
        raise re.error(f"'\\{c}' at position {i - 1} is not an ECMA-262 escape")
    return _cp(v), j, v


def _class(p: str, i: int) -> tuple:
    """Translates the character class whose body starts at p[i]: (regex, next index)."""
    neg = p.startswith("^", i)
    if neg:
        i += 1
    items = []
    while True:
        if i >= len(p):
            raise re.error("unterminated character class")
        if p[i] == "]":
            i += 1
            break
        a, i, lo = _escape(p, i + 1, True) if p[i] == "\\" else (_cp(ord(p[i])), i + 1, ord(p[i]))
        if p.startswith("-", i) and p[i + 1:i + 2] not in ("", "]"):
            b, i, hi = _escape(p, i + 2, True) if p[i + 1] == "\\" else (_cp(ord(p[i + 1])), i + 2, ord(p[i + 1]))
            if lo is None or hi is None or lo > hi:
                raise re.error("bad character range")
            a = f"{a}-{b}"
        items.append(a)
    if not items:                       # [] never matches; [^] matches any character
        return ("[\\x00-\\U0010ffff]" if neg else "(?!)"), i
    return "[" + ("^" if neg else "") + "".join(items) + "]", i


def ecma_translate(p: str) -> str:
    """Python re syntax for a JSON Schema pattern, i.e. an ECMA-262 regular expression
    with the "u" flag. Raises re.error for syntax ECMA-262 rejects or that has no
    exact Python equivalent here."""
    out, i = [], 0
    prev = None         # what a quantifier here would repeat: None (nothing), atom, assertion, quantifier, lazy
    groups = []         # per open group: "atom", or "assertion" for a lookaround
    forced = [1]        # per open group (and the whole pattern): largest product of nested minimum counts
    last = None         # that product for what a quantifier here would repeat (None: nothing yet)
    while i < len(p):
        c, at, kind = p[i], i, "atom"
        last = 1 if c not in "{*+?)" else last
        if c == "\\":
            kind = "assertion" if p[i + 1:i + 2] in ("b", "B") else "atom"
            tok, i, _ = _escape(p, i + 1, False)
        elif c == "[":
            tok, i = _class(p, i + 1)
        elif c == ".":
            tok, i = "[^\\n\\r\\u2028\\u2029]", i + 1
        elif c in "^$":
            tok, i, kind = ("\\Z" if c == "$" else "^"), i + 1, "assertion"
        elif c == "(":
            prefix = next((x for x in ("(?:",) + _LOOKAROUND if p.startswith(x, i)), None)
            m = None if prefix else _GROUP.match(p, i)
            if prefix:
                tok, i = prefix, i + len(prefix)
            elif m:
                tok, i = f"(?P<{m.group(1)}>", m.end()
            elif p.startswith("(?", i):
                raise re.error(f"unsupported group syntax at position {i}")
            else:
                tok, i = "(", i + 1
            groups.append("assertion" if prefix in _LOOKAROUND else "atom")
            forced.append(1)
            kind = None
        elif c == ")":
            if not groups:
                raise re.error(f"unbalanced ')' at position {i}")
            tok, i, kind = ")", i + 1, groups.pop()
            last = forced.pop()
            forced[-1] = max(forced[-1], last)
        elif c == "|":
            tok, i, kind = "|", i + 1, None
        elif c == "{" or c in "*+?":
            m = _QUANT.match(p, i) if c == "{" else None
            if c == "{" and not m:
                raise re.error(f"'{{' at position {i} is not a quantifier")
            tok, i = (m.group(), m.end()) if m else (c, i + 1)
            if prev is None or prev == "assertion":    # u flag: lookarounds, \b, ^, $ are not quantifiable
                raise re.error(f"{tok!r} at position {at} has nothing to repeat")
            if prev == "lazy" or (prev == "quantifier" and tok != "?"):
                raise re.error(f"{tok!r} at position {at} repeats a quantifier")
            kind = "lazy" if prev == "quantifier" else "quantifier"
            if kind == "quantifier":
                counts = re.findall(r"[0-9]+", tok)
                for n in counts:
                    if len(n.lstrip("0")) > len(str(MAX_REPEAT)) or int(n) > MAX_REPEAT:
                        raise re.error(f"repeat count {n[:20]} at position {at} is above {MAX_REPEAT}")
                if counts:              # *, + and ? force at most one repetition
                    last *= int(counts[0])
                    if last > MAX_REPEAT:
                        raise re.error(f"nested repeats at position {at} force {last} repetitions, above {MAX_REPEAT}")
                    forced[-1] = max(forced[-1], last)
        elif c in "}]":
            raise re.error(f"lone {c!r} at position {i}")
        else:
            tok, i = c, i + 1
        out.append(tok)
        prev = kind
    return "".join(out)


_PATTERNS: dict = {}


def ecma_regex(pattern: str):
    """Compiled Python regex with the semantics of a JSON Schema "pattern" (ECMA-262, "u"
    flag): "$" matches only at the very end, "." excludes line terminators, \\d \\w \\b are
    ASCII, \\s is ECMA-262 white space, [] and [^] are the empty and the full class.
    Raises re.error for a pattern it cannot translate exactly."""
    rx = _PATTERNS.get(pattern)
    if rx is None:
        try:
            rx = re.compile(ecma_translate(pattern))
        except (OverflowError, RecursionError) as e:     # e.g. a repeat count above Python's limit
            raise re.error(f"no exact translation: {e}") from None
        _PATTERNS[pattern] = rx
    return rx


# where JSON Schema (2019-09, 2020-12) applies subschemas: one schema, a map of them, a list of them
_ONE = ("items", "additionalItems", "additionalProperties", "not", "if", "then", "else", "contains",
        "propertyNames", "unevaluatedItems", "unevaluatedProperties", "contentSchema")
_MAP = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_LIST = ("allOf", "anyOf", "oneOf", "prefixItems")
DIALECTS = ("https://json-schema.org/draft/2020-12/schema", "https://json-schema.org/draft/2019-09/schema")


def schema_nodes(s):
    """Every subschema object of s (s included) at a position where JSON Schema applies it;
    instance data (const, enum, default, examples) and unknown keywords are not schemas."""
    stack = [s]
    while stack:
        n = stack.pop()
        if not isinstance(n, dict):
            continue
        yield n
        for k in _ONE:
            v = n.get(k)
            stack.extend(v if isinstance(v, list) else [v])
        for k in _MAP:
            if isinstance(n.get(k), dict):
                stack.extend(n[k].values())
        for k in _LIST:
            if isinstance(n.get(k), list):
                stack.extend(n[k])


def schema_patterns(s):
    """Every "pattern" string of a schema."""
    for n in schema_nodes(s):
        if isinstance(n.get("pattern"), str):
            yield n["pattern"]


def resolve_ref(root: dict, ref):
    """The subschema a local $ref names: "#" or a JSON pointer in the URI fragment. Raises
    SchemaError for any other reference, and for one that does not resolve to a schema."""
    if not isinstance(ref, str) or not ref.startswith("#") or ref[1:2] not in ("", "/"):
        raise SchemaError(f"only local $ref to a JSON pointer ('#/...') is supported, got {json.dumps(ref)[:80]}")
    node = root
    for part in unquote(ref).split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and re.fullmatch(r"0|[1-9][0-9]*", part) and int(part) < len(node):
            node = node[int(part)]
        else:
            raise SchemaError(f"unresolvable $ref {ref!r}")
    if not isinstance(node, (dict, bool)):
        raise SchemaError(f"$ref {ref!r} does not name a schema")
    return node


def precheck_schema(schema) -> None:
    """What both engines require before anything else (module docstring); raises SchemaError."""
    if not isinstance(schema, dict):
        raise SchemaError("the schema must be a JSON object")
    if "$schema" in schema and schema["$schema"] not in DIALECTS:     # none given: 2020-12
        raise SchemaError(f"$schema {json.dumps(schema['$schema'])[:80]} is not one of {', '.join(DIALECTS)}")
    nodes = list(schema_nodes(schema))
    applied = {id(n) for n in nodes}
    for n in nodes:
        if n is not schema and "$id" in n:
            raise SchemaError("$id is supported only at the root (it would change what '#' refers to)")
        for k in ("$dynamicRef", "$recursiveRef"):
            if k in n:
                raise SchemaError(f"{k} is not supported; use a local $ref")
        if "patternProperties" in n:
            raise SchemaError("patternProperties is not supported: jsonschema would match its keys with Python's re, "
                              "not as ECMA-262")
        if "$ref" in n:
            target = resolve_ref(schema, n["$ref"])
            if isinstance(target, dict) and id(target) not in applied:
                raise SchemaError(f"$ref {n['$ref']!r} points into instance data, not at a subschema")
        if isinstance(n.get("pattern"), str):
            try:
                ecma_regex(n["pattern"])
            except re.error as e:
                raise SchemaError(f"bad pattern {n['pattern']!r}: {e}") from None
    MiniValidator(schema)       # the built-in engine's keywords and value kinds bind jsonschema too


# ---------------------------------------------------------- built-in engine
_ANNOTATIONS = {"$schema", "$id", "$comment", "title", "description", "default", "examples",
                "$defs", "definitions", "deprecated", "readOnly", "writeOnly"}
_SUPPORTED = {"type", "enum", "const", "required", "properties", "additionalProperties", "items",
              "pattern", "format", "$ref", "minLength", "maxLength", "minItems", "maxItems",
              "minimum", "maximum"}
_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


def _is_type(x, t: str) -> bool:
    if t == "object":
        return isinstance(x, dict)
    if t == "array":
        return isinstance(x, list)
    if t == "string":
        return isinstance(x, str)
    if t == "boolean":
        return isinstance(x, bool)
    if t == "null":
        return x is None
    if t == "integer":
        if isinstance(x, bool):
            return False
        return isinstance(x, int) or (isinstance(x, float) and x.is_integer())
    if t == "number":
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    raise SchemaError(f"unknown type {t!r}")


def _json_equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def _ptr(path) -> str:
    return "/" + "/".join(str(p) for p in path) if path else "(root)"


class MiniValidator:
    def __init__(self, schema: dict):
        if not isinstance(schema, dict):
            raise SchemaError("schema must be an object")
        self.root = schema
        self._check_schema(schema, "#")

    def _resolve(self, ref: str):
        return resolve_ref(self.root, ref)

    def _check_ref_chain(self, s, where: str):
        """$ref is the only supported keyword that applies a subschema to the same
        value, so a cycle made of $refs alone would recurse forever."""
        seen = {id(s)}
        node = s
        while isinstance(node, dict) and "$ref" in node:
            node = self._resolve(node["$ref"])
            if id(node) in seen:
                raise SchemaError(f"{where}: $ref cycle through {s['$ref']!r} never reaches a schema")
            seen.add(id(node))

    def _check_schema(self, s, where: str):
        if isinstance(s, bool):
            return
        if not isinstance(s, dict):
            raise SchemaError(f"{where}: subschema must be an object or boolean")
        for k in s:
            if k not in _SUPPORTED and k not in _ANNOTATIONS:
                raise SchemaError(f"{where}: keyword {k!r} is not supported (under any engine; extend "
                                  "validate_handover.py's built-in engine first)")
        self._check_values(s, where)
        if "$ref" in s:
            self._check_ref_chain(s, where)
        if "pattern" in s:
            try:
                ecma_regex(s["pattern"])
            except re.error as e:
                raise SchemaError(f"{where}: bad pattern: {e}") from None
        for k, sub in s.get("properties", {}).items():
            self._check_schema(sub, f"{where}/properties/{k}")
        for key in ("items", "additionalProperties"):
            if key in s:
                self._check_schema(s[key], f"{where}/{key}")
        for defs in ("$defs", "definitions"):
            for k, sub in s.get(defs, {}).items():
                self._check_schema(sub, f"{where}/{defs}/{k}")

    @staticmethod
    def _check_values(s: dict, where: str):
        """Keyword values of the wrong kind would otherwise surface as per-manifest failures."""
        def bad(k, what):
            raise SchemaError(f"{where}: {k!r} must be {what}, got {json.dumps(s[k])[:60]}")

        t = s.get("type", "string")
        types = t if isinstance(t, list) else [t]
        if not types or not all(isinstance(x, str) for x in types) or len(set(types)) != len(types):
            bad("type", "a type name or a non-empty list of distinct ones")
        for tt in types:
            if tt not in _TYPES:
                raise SchemaError(f"{where}: unknown type {tt!r}")
        for k in ("pattern", "format", "$ref", "$schema", "$id", "$comment", "title", "description"):
            if k in s and not isinstance(s[k], str):
                bad(k, "a string")
        if isinstance(s.get("$id"), str) and "#" in s["$id"][:-1]:     # the metaschema's ^[^#]*#?$
            bad("$id", "a URI without a fragment")
        for k in ("deprecated", "readOnly", "writeOnly"):
            if k in s and not isinstance(s[k], bool):
                bad(k, "a boolean")
        if "examples" in s and not isinstance(s["examples"], list):
            bad("examples", "an array")
        for k in ("properties", "$defs", "definitions"):
            if k in s and not isinstance(s[k], dict):
                bad(k, "an object")
        if "enum" in s and not isinstance(s["enum"], list):
            bad("enum", "an array")
        req = s.get("required", [])
        if not (isinstance(req, list) and all(isinstance(x, str) for x in req) and len(set(req)) == len(req)):
            bad("required", "an array of distinct strings")
        for k in ("minLength", "maxLength", "minItems", "maxItems"):
            if k in s and not (_is_type(s[k], "integer") and s[k] >= 0):
                bad(k, "a non-negative integer")
        for k in ("minimum", "maximum"):
            if k in s and not _is_type(s[k], "number"):
                bad(k, "a number")

    def errors(self, instance) -> list:
        out: list = []
        self._validate(instance, self.root, (), out)
        return out

    def _validate(self, x, s, path, out):
        if s is True:
            return
        if s is False:
            out.append((_ptr(path), "no value is allowed here"))
            return
        if "$ref" in s:
            self._validate(x, self._resolve(s["$ref"]), path, out)
        if "type" in s:
            types = s["type"] if isinstance(s["type"], list) else [s["type"]]
            if not any(_is_type(x, t) for t in types):
                out.append((_ptr(path), f"{json.dumps(x)[:60]} is not of type {' | '.join(types)}"))
                return
        if "enum" in s and not any(_json_equal(x, e) for e in s["enum"]):
            out.append((_ptr(path), f"{json.dumps(x)[:60]} is not one of {s['enum']}"))
        if "const" in s and not _json_equal(x, s["const"]):
            out.append((_ptr(path), f"{json.dumps(x)[:60]} is not {json.dumps(s['const'])}"))
        if isinstance(x, str):
            if "pattern" in s and not ecma_regex(s["pattern"]).search(x):
                out.append((_ptr(path), f"{x!r} does not match {s['pattern']!r}"))
            if "minLength" in s and len(x) < s["minLength"]:
                out.append((_ptr(path), f"{x!r} is shorter than {s['minLength']}"))
            if "maxLength" in s and len(x) > s["maxLength"]:
                out.append((_ptr(path), f"{x[:40]!r}... is longer than {s['maxLength']}"))
            fmt = s.get("format")
            if fmt in FORMATS and not FORMATS[fmt](x):
                out.append((_ptr(path), f"{x!r} is not a valid {fmt}"))
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            if "minimum" in s and x < s["minimum"]:
                out.append((_ptr(path), f"{x} is less than {s['minimum']}"))
            if "maximum" in s and x > s["maximum"]:
                out.append((_ptr(path), f"{x} is greater than {s['maximum']}"))
        if isinstance(x, list):
            if "minItems" in s and len(x) < s["minItems"]:
                out.append((_ptr(path), f"expected at least {s['minItems']} items"))
            if "maxItems" in s and len(x) > s["maxItems"]:
                out.append((_ptr(path), f"expected at most {s['maxItems']} items"))
            if "items" in s:
                for i, item in enumerate(x):
                    self._validate(item, s["items"], path + (i,), out)
        if isinstance(x, dict):
            for req in s.get("required", []):
                if req not in x:
                    out.append((_ptr(path), f"required property {req!r} is missing"))
            props = s.get("properties", {})
            for k, v in x.items():
                if k in props:
                    self._validate(v, props[k], path + (k,), out)
                elif "additionalProperties" in s:
                    ap = s["additionalProperties"]
                    if ap is False:
                        out.append((_ptr(path), f"additional property {k!r} is not allowed"))
                    elif isinstance(ap, dict):
                        self._validate(v, ap, path + (k,), out)


def _ecma_pattern_keyword(validator, pattern, instance, schema):
    import jsonschema

    if validator.is_type(instance, "string") and not ecma_regex(pattern).search(instance):
        yield jsonschema.ValidationError(f"{instance!r} does not match {pattern!r}")


def _jsonschema_errors(schema: dict, instance) -> list:
    import jsonschema  # optional dependency

    base = jsonschema.validators.validator_for(schema)
    cls = jsonschema.validators.extend(base, {"pattern": _ecma_pattern_keyword})
    checker = jsonschema.FormatChecker(formats=())          # only date-time is asserted, as by MiniValidator
    checker.checks("date-time")(is_date_time)
    v = cls(schema, format_checker=checker)
    errs = sorted(v.iter_errors(instance), key=lambda e: [str(p) for p in e.absolute_path])
    return [(_ptr(tuple(e.absolute_path)), e.message) for e in errs]


def jsonschema_check_schema(schema: dict) -> None:
    """Raises SchemaError if the schema is not valid against its JSON Schema metaschema, or if
    jsonschema fails on it. Formats are not asserted: "regex" would test patterns with Python's
    re, which rejects valid ECMA-262 such as (?<name>...) (precheck_schema checks them), and
    the others ("uri" for $id and $schema) depend on which optional packages are installed."""
    import jsonschema

    try:
        cls = jsonschema.validators.validator_for(schema)
        err = jsonschema.exceptions.best_match(cls(cls.META_SCHEMA).iter_errors(schema))
    except Exception as e:      # jsonschema failing on the schema makes it as unusable as an invalid one
        raise SchemaError(f"jsonschema cannot check the schema: {type(e).__name__}: {e}") from None
    if err is not None:
        raise SchemaError(f"schema is not valid JSON Schema: {err.message}")


def have_jsonschema() -> bool:
    try:
        import jsonschema  # noqa: F401
        return True
    except ImportError:
        return False


def schema_errors(schema: dict, instance, engine: str = "auto") -> list:
    if engine == "jsonschema" or (engine == "auto" and have_jsonschema()):
        return _jsonschema_errors(schema, instance)
    return MiniValidator(schema).errors(instance)


# ------------------------------------------------------- repository checks
def invariant_ids(path=INVARIANTS) -> set:
    text = Path(path).read_text(encoding="utf-8")
    return set(re.findall(r"^\|\s*\*\*(INV-[A-Z0-9-]+)\*\*\s*\|", text, flags=re.M))


def task_ids(path=TASKS) -> set:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return {t.get("id") for t in doc.get("tasks", []) if isinstance(t, dict)}


def _no_duplicate_keys(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f"duplicate key {k!r}")
        d[k] = v
    return d


def _no_constants(name):
    raise ValueError(f"{name} is not valid JSON")


def load_json(path):
    """Strict JSON: no duplicate keys, no NaN/Infinity."""
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys,
                      parse_constant=_no_constants)


_GEN = re.compile(r"gen-([0-9]{3,})-[a-z0-9-]+")
_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _lineage(path: Path, doc: dict) -> list:
    """Warnings about parent_generation, resolved within the manifest's directory."""
    tid, gid, pg = doc.get("task_id"), doc.get("generation_id"), doc.get("parent_generation")
    m = _GEN.fullmatch(gid) if isinstance(gid, str) else None
    if not m or not isinstance(tid, str):
        return []
    gen = int(m.group(1))
    if pg is None:
        return [f"/parent_generation: generation {gen} names no parent"] if gen > 1 else []
    if not isinstance(pg, str):
        return []
    pm = _GEN.fullmatch(pg)
    if not pm:
        return [f"/parent_generation: {pg!r} does not look like a generation_id"]
    if int(pm.group(1)) >= gen:
        return [f"/parent_generation: {pg!r} is not an earlier generation than {gid!r}"]
    parent = Path(path).parent / f"{tid}-gen-{pm.group(1)}.json"
    try:
        pdoc = load_json(parent)
    except (OSError, ValueError, RecursionError):
        return [f"/parent_generation: no readable manifest {parent.name} next to this one"]
    if not isinstance(pdoc, dict) or pdoc.get("generation_id") != pg:
        return [f"/parent_generation: {parent.name} does not have generation_id {pg!r}"]
    return []


def check_manifest(path: Path, schema: dict, invariants: set, tasks, engine="auto") -> dict:
    res = {"file": str(path), "errors": [], "warnings": []}
    try:
        doc = load_json(path)
    except (OSError, UnicodeDecodeError) as e:
        res["errors"].append(f"cannot read: {e}")
        return res
    except (ValueError, RecursionError) as e:
        res["errors"].append(f"invalid JSON: {e or 'nested too deeply'}")
        return res
    try:
        errs = schema_errors(schema, doc, engine)
    except RecursionError:
        errs = [("(root)", "document is nested too deeply")]
    for where, msg in errs:
        res["errors"].append(f"{where}: {msg}")
    if not isinstance(doc, dict):
        return res
    touched = doc.get("constraints_touched")
    for inv in touched if isinstance(touched, list) else []:
        if isinstance(inv, str) and inv not in invariants:
            res["errors"].append(f"/constraints_touched: {inv!r} is not an invariant id in governance/INVARIANTS.md")
    tid = doc.get("task_id")
    if tasks is not None and isinstance(tid, str) and tid not in tasks:
        res["errors"].append(f"/task_id: {tid!r} is not a task in governance/tasks.json")
    gid = doc.get("generation_id")
    m = _GEN.fullmatch(gid) if isinstance(gid, str) else None
    if m and isinstance(tid, str):
        want = f"{tid}-gen-{m.group(1)}.json"
        if Path(path).name != want:
            res["warnings"].append(f"file name should be {want} (task_id + generation number)")
    res["warnings"] += _lineage(Path(path), doc)
    meas_list = doc.get("measurements")
    for i, meas in enumerate(meas_list if isinstance(meas_list, list) else []):
        if isinstance(meas, dict) and "kind" not in meas:
            res["warnings"].append(f"/measurements/{i}: no 'kind' (measured|simulated) — INV-HONEST")
    return res


def check_one(path: Path, schema: dict, invariants: set, tasks, engine="auto") -> dict:
    """check_manifest, with an unexpected failure reported against this file only."""
    try:
        return check_manifest(path, schema, invariants, tasks, engine)
    except Exception as e:  # a crash on one manifest must not hide the verdicts on the others
        return {"file": str(path), "errors": [f"validator failed on this file: {type(e).__name__}: {e}"],
                "warnings": []}


def collect(paths) -> list:
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += sorted(q for q in p.glob("*.json") if q.is_file())
        else:
            files.append(p)
    return files


def skeleton(task: str, handovers: Path = DEFAULT_DIR) -> dict:
    gens = {}           # generation number -> its generation_id
    for f in Path(handovers).glob(f"{task}-gen-*.json"):
        m = re.fullmatch(rf"{re.escape(task)}-gen-([0-9]+)\.json", f.name)
        if not m:
            continue
        try:   # an empty or broken file (e.g. just created by a shell redirect) is no generation
            doc = load_json(f)
        except (OSError, ValueError, RecursionError):
            continue
        if isinstance(doc, dict):
            gens[int(m.group(1))] = doc.get("generation_id")
    gen = max(gens) + 1 if gens else 1
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "0000000"
        branch = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip() or "main"
    except (OSError, subprocess.SubprocessError):
        head, branch = "0000000", "main"
    slug = re.sub(r"[^a-z0-9]+", "-", task.lower().split("-", 1)[-1]).strip("-") or "work"
    parent = gens.get(gen - 1)
    if gen - 1 in gens and not (isinstance(parent, str) and _GEN.fullmatch(parent)):
        parent = f"gen-{gen - 1:03d}-{slug}"
    return {
        "generation_id": f"gen-{gen:03d}-{slug}",
        "parent_generation": parent,
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "lifecycle_status": "TASK_COMPLETE",
        "trigger": "MILESTONE_VERIFIED",
        "task_id": task,
        "workspace": {"branch": branch, "base_commit": head, "head_commit": head},
        "constraints_touched": [],
        "epistemic_ledger": {"hypotheses_validated": [], "hypotheses_falsified": [], "open_questions": []},
        "procedures": [],
        "measurements": [],
        "active_blockers": [],
        "unfulfilled_mandates": [],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="manifest files or directories (default: governance/handovers/)")
    ap.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    ap.add_argument("--invariants", default=str(INVARIANTS))
    ap.add_argument("--tasks", default=str(TASKS), help="tasks.json for task_id checks ('' to skip)")
    ap.add_argument("--engine", choices=("auto", "builtin", "jsonschema"), default="auto")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    ap.add_argument("--json", action="store_true", help="print a JSON report")
    ap.add_argument("--new", metavar="TASK_ID", help="print a skeleton manifest for TASK_ID and exit")
    ap.add_argument("--write", action="store_true",
                    help="with --new: create <handovers-dir>/<TASK_ID>-gen-<NNN>.json instead of printing")
    ap.add_argument("--handovers-dir", default=str(DEFAULT_DIR), help="manifest directory for --new")
    a = ap.parse_args(argv)

    if a.write and a.new is None:
        print("validate_handover: --write needs --new TASK_ID", file=sys.stderr)
        return 2
    if a.new is not None:
        if not _TASK_ID.fullmatch(a.new):
            print(f"validate_handover: --new {a.new!r} is not a task id", file=sys.stderr)
            return 2
        try:
            known = task_ids(a.tasks) if a.tasks else None
        except (OSError, ValueError) as e:
            print(f"validate_handover: {e}", file=sys.stderr)
            return 2
        if known is not None and a.new not in known:
            print(f"validate_handover: --new {a.new!r} is not a task in {a.tasks}", file=sys.stderr)
            return 2
        sk = skeleton(a.new, Path(a.handovers_dir))
        text = json.dumps(sk, indent=2) + "\n"
        if not a.write:
            sys.stdout.write(text)
            return 0
        num = _GEN.fullmatch(sk["generation_id"]).group(1)
        out = Path(a.handovers_dir) / f"{a.new}-gen-{num}.json"
        try:
            with open(out, "x", encoding="utf-8", newline="\n") as f:
                f.write(text)
        except OSError as e:
            print(f"validate_handover: cannot create {out}: {e}", file=sys.stderr)
            return 2
        print(out)
        return 0
    try:
        schema = load_json(a.schema)
        invariants = invariant_ids(a.invariants)
        tasks = task_ids(a.tasks) if a.tasks else None
        if a.engine == "jsonschema" and not have_jsonschema():
            raise SchemaError("--engine jsonschema requested but jsonschema is not installed")
        precheck_schema(schema)
        if a.engine != "builtin" and have_jsonschema():
            jsonschema_check_schema(schema)
    except (OSError, ValueError, SchemaError) as e:
        print(f"validate_handover: {e}", file=sys.stderr)
        return 2
    if not invariants:
        print(f"validate_handover: no invariant ids found in {a.invariants}", file=sys.stderr)
        return 2

    paths = a.paths or [str(DEFAULT_DIR)]
    for p in paths:
        if not Path(p).exists():
            print(f"validate_handover: no such file or directory: {p}", file=sys.stderr)
            return 2
    files = collect(paths)
    results = [check_one(f, schema, invariants, tasks, a.engine) for f in files]
    bad = 0
    for r in results:
        r["valid"] = not r["errors"] and not (a.strict and r["warnings"])
        bad += not r["valid"]
    if a.json:
        print(json.dumps({"engine": "jsonschema" if a.engine != "builtin" and have_jsonschema() else "builtin",
                          "files": results, "invalid": bad}, indent=2))
        return 1 if bad else 0
    if not files:
        print("no handover manifests found in " + ", ".join(paths))
        return 0
    for r in results:
        print(("OK   " if r["valid"] else "FAIL ") + r["file"])
        for e in r["errors"]:
            print(f"     error: {e}")
        for w in r["warnings"]:
            print(f"     warning: {w}")
    print(f"{len(files) - bad}/{len(files)} manifest(s) valid")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
