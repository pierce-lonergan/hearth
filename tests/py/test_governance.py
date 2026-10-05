"""Tests for the governance tools (governance/tools/*.py). Standard library only."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "governance" / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"hearth_gov_{name}", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


waves = _load("waves")
vh = _load("validate_handover")
cg = _load("check_golden")
mutate = _load("mutate")
friction = _load("friction")
adr = _load("adr")
merge_ref = _load("merge_ref")
path_aliases = _load("path_aliases")


def tool(name, *args, cwd=None, timeout=600, env=None):
    return subprocess.run([sys.executable, str(TOOLS / f"{name}.py"), *map(str, args)],
                          capture_output=True, text=True, cwd=cwd, timeout=timeout,
                          env=None if env is None else dict(os.environ, **env))


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------------ waves
def task(tid, deps=(), owns=(), status="todo"):
    return {"id": tid, "title": f"task {tid}", "deps": list(deps), "owns": list(owns), "accept": [], "status": status}


def test_waves_real_tasks_file_is_valid_and_layered():
    doc = waves.load_tasks(waves.DEFAULT_TASKS)
    errors, _, cycle = waves.validate(doc)
    assert errors == [] and cycle is None
    tasks = doc["tasks"]
    layers, leftover = waves.compute_waves(tasks)
    assert leftover == []
    wave_of = {i: n for n, w in enumerate(layers) for i in w}
    assert sorted(wave_of) == sorted(t["id"] for t in tasks)          # each task exactly once
    for t in tasks:                                                    # Kahn layering property
        want = 0 if not t["deps"] else 1 + max(wave_of[d] for d in t["deps"])
        assert wave_of[t["id"]] == want, t["id"]
    assert "T05-governance" in wave_of
    r = tool("waves", "validate")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK:" in r.stdout


def test_waves_cycle_is_detected_and_printed(tmp_path):
    doc = {"tasks": [task("A", ["C"]), task("B", ["A"]), task("C", ["B"]), task("D")]}
    errors, _, cycle = waves.validate(doc)
    assert cycle is not None and cycle[0] == cycle[-1] and set(cycle) == {"A", "B", "C"}
    assert any("dependency cycle" in e for e in errors)
    r = tool("waves", "validate", "--tasks", write_json(tmp_path / "t.json", doc))
    assert r.returncode == 1
    assert "dependency cycle:" in r.stdout and " -> " in r.stdout


def test_waves_ownership_conflicts(tmp_path):
    doc = {"tasks": [task("X", owns=["engine/src/"]), task("Y", owns=["engine/src/pool.c"])]}
    errors, _, _ = waves.validate(doc)
    assert any("ownership conflict" in e and "X" in e and "Y" in e for e in errors)
    r = tool("waves", "--tasks", write_json(tmp_path / "t.json", doc))
    assert r.returncode == 1
    done = copy.deepcopy(doc)
    done["tasks"][0]["status"] = "done"                     # finished tasks release their paths
    assert waves.validate(done)[0] == []
    assert waves.validate(doc, {"X": "done"})[0] == []
    ok = {"tasks": [task("X", owns=["engine/src/"]), task("Y", owns=["engine/srcx/a.c", "docs/A.md"])]}
    assert waves.validate(ok)[0] == []                       # prefix match respects path segments
    same = {"tasks": [task("X", owns=["./docs/A.md"]), task("Y", owns=["docs\\A.md"])]}
    assert any("ownership conflict" in e for e in waves.validate(same)[0])


@pytest.mark.parametrize("a,b", [
    ("python/hearth/sim", "python/hearth/sim/core.py"),      # a directory need not end in "/"
    ("engine/src/pool.c", "engine//src/pool.c"),
    ("engine/src/pool.c", "engine/./src/pool.c"),
    ("engine/src/.", "engine/src/pool.c"),
    ("engine/src/pool.c", "Engine/Src/Pool.c"),              # one file on Windows and macOS
    ("docs", "docs/A.md/"),
])
def test_waves_ownership_overlap_spellings(a, b, tmp_path):
    doc = {"tasks": [task("X", owns=[a]), task("Y", owns=[b])]}
    errors, _, _ = waves.validate(doc)
    assert any("ownership conflict" in e for e in errors), (a, b, errors)
    r = tool("waves", "validate", "--tasks", write_json(tmp_path / "t.json", doc))
    assert r.returncode == 1 and "ownership disjoint" not in r.stdout
    assert waves.validate(doc, {"X": "done"})[0] == []


def test_waves_path_canonicalisation():
    assert waves.norm_path("engine//src/./pool.c") == ("engine/src/pool.c", False)
    assert waves.norm_path("engine\\src\\") == ("engine/src", True)
    assert waves.norm_path("engine/src/.") == ("engine/src", True)
    for bad in (".", "./", "/abs", "C:/x", "c:x", "a/../b", "", "  "):
        assert waves.norm_path(bad) is None, bad
    assert not waves.paths_overlap("engine/src", "engine/srcx/a.c")
    assert not waves.paths_overlap("docs/A.md", "docs/A.mdx")
    errors, warnings, _ = waves.validate({"tasks": [task("X", owns=["engine//src/pool.c", "docs/"])]})
    assert errors == [] and len(warnings) == 1 and "'engine/src/pool.c'" in warnings[0]
    errors, _, _ = waves.validate({"tasks": [task("X", owns=["Docs/A.md"]), task("Y", owns=["docs/a.md"])]})
    assert any("differ only in case" in e for e in errors)
    errors, _, _ = waves.validate({"tasks": [task("X", owns=["docs/"]), task("Y", owns=["docs/a.md"])]})
    assert len(errors) == 1 and "differ only in case" not in errors[0]


@pytest.mark.parametrize("doc,needle", [
    ({"tasks": [task("A", ["Z"])]}, "unknown dependency"),
    ({"tasks": [task("A"), task("A")]}, "duplicate task id"),
    ({"tasks": [task("A", ["A"])]}, "depends on itself"),
    ({"tasks": [task("A", status="finished")]}, "status"),
    ({"tasks": [task("A", owns=["../outside"])]}, "repo-relative"),
    ({"tasks": [task("A", owns=["C:/abs/path.c"])]}, "repo-relative"),
    ({"tasks": [{"id": "A", "title": "t", "deps": "B", "owns": [], "status": "todo"}]}, "list of strings"),
    ({"tasks": [dict(task("A"), accept="ship it")]}, "accept"),
    ({"nope": []}, "tasks"),
    ([], "top level"),
])
def test_waves_invalid_documents(doc, needle, tmp_path):
    errors, _, _ = waves.validate(doc)
    assert any(needle in e for e in errors), errors
    if needle == "depends on itself":
        assert len(errors) == 1, errors              # not also reported as a cycle
    f = write_json(tmp_path / "t.json", doc)
    r = tool("waves", "validate", "--tasks", f)
    assert r.returncode == 1 and "INVALID" in r.stdout, r.stdout + r.stderr   # a report, not a crash
    r = tool("waves", "--json", "--tasks", f)
    assert r.returncode == 1 and json.loads(r.stdout)["valid"] is False


def test_waves_warnings():
    doc = {"tasks": [task("A"), task("B", ["A", "A"], owns=["docs/", "docs/A.md"], status="active")]}
    errors, warnings, _ = waves.validate(doc)
    assert errors == []
    assert any("duplicate entries in deps" in w for w in warnings)
    assert any("overlap" in w for w in warnings)
    assert any("B is active but dependencies are not done: A" in w for w in warnings)
    assert waves.validate({"tasks": [task("A", status="done"), task("B", ["A"], owns=["docs/"], status="active")]}) \
        == ([], [], None)


def test_waves_next_runnable_set(tmp_path):
    doc = {"tasks": [task("A", status="done"), task("B", ["A"]), task("C", ["B"]), task("D", status="active"),
                     task("E", status="blocked")]}
    st = waves.effective_status(doc["tasks"], {})
    p = waves.plan(doc["tasks"], st)
    assert p["runnable"] == ["B"] and p["in_progress"] == ["D"] and p["blocked"] == ["E"]
    assert p["waiting"] == [{"id": "C", "waiting_on": ["B"]}]
    assert p["waves"] == [["A", "D", "E"], ["B"], ["C"]] and p["current_wave"] == 0
    p2 = waves.plan(doc["tasks"], waves.effective_status(doc["tasks"], {"B": "done"}))
    assert p2["runnable"] == ["C"]
    f = write_json(tmp_path / "t.json", doc)
    r = tool("waves", "next", "--tasks", f, "--done", "B")
    assert r.returncode == 0 and "runnable now: C" in r.stdout
    r = tool("waves", "--tasks", f, "--json")
    rep = json.loads(r.stdout)
    assert rep["valid"] and rep["runnable"] == ["B"] and rep["waves"][1] == ["B"]
    r = tool("waves", "--tasks", f, "--mermaid")
    assert r.returncode == 0 and r.stdout.startswith("flowchart") and "n0 --> n1" in r.stdout
    assert 'n1["B<br/>task B"]:::todo' in r.stdout
    assert tool("waves", "--tasks", f, "--status", "Q=done").returncode == 1      # unknown task
    r = tool("waves", "--tasks", tmp_path / "missing.json")
    assert r.returncode == 2 and r.stderr.startswith("waves: cannot read")
    r = tool("waves", "--tasks", f)
    assert r.returncode == 0 and "wave 0: A [done], D [active], E [blocked]\nwave 1: B [todo]" in r.stdout
    assert "current wave: 0\nrunnable now: B\nin progress:  D\nblocked:      E" in r.stdout


def test_waves_large_dag_is_linear():
    # 40 layers, each task depending on both tasks of the layer below: 2^40 paths, so any
    # traversal that revisits finished tasks never ends
    tasks = [task("L0a"), task("L0b")]
    for k in range(1, 40):
        tasks += [task(f"L{k}{s}", [f"L{k - 1}a", f"L{k - 1}b"]) for s in "ab"]
    errors, _, cycle = waves.validate({"tasks": tasks})
    assert errors == [] and cycle is None
    assert waves.find_cycle([t["id"] for t in tasks], {t["id"]: t["deps"] for t in tasks}) is None


def test_waves_mermaid_ids_never_collide():
    tasks = [task("T01-a"), task("T01_a", ["T01-a"]), dict(task("T 01<a>"), title='say "hi" & #1')]
    assert waves.validate({"tasks": tasks})[0] == []
    out = waves.to_mermaid(tasks, waves.effective_status(tasks, {}))
    nodes = [ln.split("[", 1)[0].strip() for ln in out.splitlines() if '["' in ln and "subgraph" not in ln]
    assert nodes == ["n0", "n2", "n1"] and len(set(nodes)) == 3
    assert "n0 --> n1" in out and "n1 --> n1" not in out
    assert 'T 01#60;a#62;<br/>say #34;hi#34; #38; #35;1' in out
    long = [dict(task("L"), title="x" * 48), dict(task("M"), title="y" * 49)]
    out = waves.to_mermaid(long, waves.effective_status(long, {}))
    assert f'L<br/>{"x" * 48}"' in out and f'M<br/>{"y" * 45}..."' in out


# ------------------------------------------------------- validate_handover
SCHEMA = json.loads((ROOT / "governance" / "schemas" / "handover.schema.json").read_text(encoding="utf-8"))
INVARIANT_IDS = vh.invariant_ids()
TASK_IDS = vh.task_ids()
ENGINES = ["builtin"] + (["jsonschema"] if vh.have_jsonschema() else [])


def manifest():
    return {
        "generation_id": "gen-001-governance",
        "parent_generation": None,
        "timestamp": "2026-10-04T12:00:00Z",
        "lifecycle_status": "TASK_COMPLETE",
        "trigger": "MILESTONE_VERIFIED",
        "task_id": "T05-governance",
        "workspace": {"branch": "main", "base_commit": "aafa8f1", "head_commit": "aafa8f1"},
        "constraints_touched": ["INV-VERIFY", "INV-DATA"],
        "epistemic_ledger": {
            "hypotheses_validated": [{"claim": "c", "evidence": "e", "verdict": "CONFIRMED"}],
            "hypotheses_falsified": [],
            "open_questions": ["q"],
        },
        "procedures": ["python governance/tools/waves.py validate"],
        "measurements": [{"metric": "m", "value": 1.5, "unit": "s", "conditions": "x", "kind": "measured"}],
        "active_blockers": [],
        "unfulfilled_mandates": [],
    }


def check(doc, tmp_path, engine, name="T05-governance-gen-001.json"):
    return vh.check_manifest(write_json(tmp_path / name, doc), SCHEMA, INVARIANT_IDS, TASK_IDS, engine)


def test_invariant_ids_parsed():
    assert {"INV-DET-1", "INV-VERIFY", "INV-HONEST", "INV-LOSSLESS-DEFAULT"} <= INVARIANT_IDS


@pytest.mark.parametrize("engine", ENGINES)
def test_handover_valid_manifest_accepted(engine, tmp_path):
    r = check(manifest(), tmp_path, engine)
    assert r["errors"] == [] and r["warnings"] == []
    m = manifest()
    m["parent_generation"] = "gen-000-governance"      # union type: string branch
    del m["procedures"], m["measurements"], m["constraints_touched"]
    assert check(m, tmp_path, engine)["errors"] == []


def _mut(path, value=None, delete=False):
    def f(m):
        node = m
        for p in path[:-1]:
            node = node[p]
        if delete:
            del node[path[-1]]
        else:
            node[path[-1]] = value
        return m
    return f


INVALID = {
    "missing field": (_mut(["trigger"], delete=True), "trigger"),
    "missing nested field": (_mut(["workspace", "head_commit"], delete=True), "head_commit"),
    "bad enum": (_mut(["lifecycle_status"], "DONE"), "lifecycle_status"),
    "bad enum via $ref": (_mut(["epistemic_ledger", "hypotheses_validated", 0, "verdict"], "MAYBE"), "verdict"),
    "bad pattern": (_mut(["generation_id"], "gen-1-governance"), "generation_id"),
    "bad commit pattern": (_mut(["workspace", "base_commit"], "XYZ1234"), "base_commit"),
    "bad item pattern": (_mut(["constraints_touched"], ["inv-det-1"]), "constraints_touched"),
    "extra property": (_mut(["extra"], 1), "extra"),
    "extra nested property": (_mut(["workspace", "dirty"], True), "dirty"),
    "extra property via $ref": (_mut(["epistemic_ledger", "hypotheses_falsified"],
                                     [{"claim": "c", "evidence": "e", "verdict": "REJECTED", "why": 1}]), "why"),
    "wrong type": (_mut(["procedures"], "make"), "procedures"),
    "union type": (_mut(["parent_generation"], 5), "parent_generation"),
    "number type": (_mut(["measurements", 0, "value"], "fast"), "value"),
    "bool is not number": (_mut(["measurements", 0, "value"], True), "value"),
    "bad date-time": (_mut(["timestamp"], "2026-02-30T10:00:00Z"), "timestamp"),
    "not a date-time": (_mut(["timestamp"], "yesterday"), "timestamp"),
    # Python's "$" matches before a final newline; JSON Schema (ECMA-262) "$" does not
    "pattern with trailing newline": (_mut(["generation_id"], "gen-001-governance\n"), "generation_id"),
    "commit with trailing newline": (_mut(["workspace", "base_commit"], "aafa8f1\n"), "base_commit"),
    "item with trailing newline": (_mut(["constraints_touched"], ["INV-VERIFY\n"]), "constraints_touched"),
    "date-time with trailing newline": (_mut(["timestamp"], "2026-10-04T12:00:00Z\n"), "timestamp"),
    "date-time with non-ASCII digits": (_mut(["timestamp"], "٢٠٢٦-10-04T12:00:00Z"), "timestamp"),
    "commit with non-ASCII digits": (_mut(["workspace", "head_commit"], "aafa8f١"), "head_commit"),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("case", sorted(INVALID))
def test_handover_invalid_manifest_rejected(engine, case, tmp_path):
    fn, needle = INVALID[case]
    r = check(fn(manifest()), tmp_path, engine)
    assert r["errors"], case
    assert any(needle in e for e in r["errors"]), r["errors"]


@pytest.mark.parametrize("engine", ENGINES)
def test_handover_repository_checks(engine, tmp_path):
    m = manifest()
    m["constraints_touched"] = ["INV-NOT-REAL"]
    assert any("INV-NOT-REAL" in e for e in check(m, tmp_path, engine)["errors"])
    m = manifest()
    m["task_id"] = "T99-nope"
    assert any("T99-nope" in e for e in check(m, tmp_path, engine)["errors"])
    r = check(manifest(), tmp_path, engine, name="whatever.json")
    assert r["errors"] == [] and any("file name" in w for w in r["warnings"])
    m = manifest()
    del m["measurements"][0]["kind"]
    assert any("INV-HONEST" in w for w in check(m, tmp_path, engine)["warnings"])
    m = manifest()
    m["constraints_touched"] = "INV-VERIFY"                  # one type error, not one error per character
    errs = check(m, tmp_path, engine)["errors"]
    assert len(errs) == 1 and "constraints_touched" in errs[0], errs


@pytest.mark.parametrize("engine", ENGINES)
def test_handover_lineage(engine, tmp_path):
    def gen(n, parent):
        m = manifest()
        m["generation_id"], m["parent_generation"] = f"gen-{n:03d}-governance", parent
        return m

    write_json(tmp_path / "T05-governance-gen-001.json", gen(1, None))
    assert check(gen(2, "gen-001-governance"), tmp_path, engine, "T05-governance-gen-002.json")["warnings"] == []
    for parent, needle in ((None, "names no parent"), ("gen-003-governance", "not an earlier"),
                           ("gen-001-other", "does not have generation_id"),
                           ("gen-000-governance", "no readable manifest"), ("parent", "does not look like")):
        r = check(gen(2, parent), tmp_path, engine, "T05-governance-gen-002.json")
        assert r["errors"] == [] and any(needle in w for w in r["warnings"]), (parent, r["warnings"])


def test_handover_json_errors(tmp_path):
    p = tmp_path / "T05-governance-gen-001.json"
    p.write_text("{not json", encoding="utf-8")
    assert vh.check_manifest(p, SCHEMA, INVARIANT_IDS, TASK_IDS, "builtin")["errors"]
    p.write_text('{"task_id": "a", "task_id": "b"}', encoding="utf-8")
    assert any("duplicate key" in e for e in vh.check_manifest(p, SCHEMA, INVARIANT_IDS, TASK_IDS, "builtin")["errors"])
    for const in ("NaN", "Infinity", "-Infinity"):            # json.loads accepts these; JSON does not
        p.write_text(json.dumps(manifest()).replace('"value": 1.5', f'"value": {const}'), encoding="utf-8")
        for engine in ENGINES:
            errs = vh.check_manifest(p, SCHEMA, INVARIANT_IDS, TASK_IDS, engine)["errors"]
            assert any("invalid JSON" in e and const.lstrip("-") in e for e in errs), errs
    p.write_text("[" * 100000 + "]" * 100000, encoding="utf-8")
    assert vh.check_manifest(p, SCHEMA, INVARIANT_IDS, TASK_IDS, "builtin")["errors"]      # no crash


def test_ecma_patterns():
    assert vh.ecma_regex("^a$").search("a") and not vh.ecma_regex("^a$").search("a\n")
    assert vh.ecma_regex("[$]").search("$") and vh.ecma_regex(r"\$").search("$")
    assert vh.ecma_regex("[^$]x$").search("ax") and not vh.ecma_regex("[^$]x$").search("$x")
    assert not vh.ecma_regex(r"^\d+$").search("١٢") and not vh.ecma_regex(r"^\w$").search("é")
    assert vh.ecma_regex("a$|^b").search("xb\n") is None and vh.ecma_regex("a$|^b").search("b\n")
    assert vh.ecma_regex(r"^ab\.c$").search("ab.c") and not vh.ecma_regex(r"^ab\.c$").search("abxc")
    assert vh.ecma_regex(r"^x[\]$]y$").search("x$y") and vh.ecma_regex(r"^x[\]$]y$").search("x]y")
    with pytest.raises(vh.SchemaError):
        vh.MiniValidator({"type": "string", "pattern": "(unclosed"})


def bs(pattern: str) -> str:
    """Pattern text with '%' standing for a backslash, so that no escape in this file is decoded early."""
    return pattern.replace("%", chr(92))


NBSP, LS, BOM, NEL, IDEO_SPACE = chr(0xA0), chr(0x2028), chr(0xFEFF), chr(0x85), chr(0x3000)
ARABIC_ONE, E_ACUTE, GRIN, LEAD = chr(0x661), chr(0xE9), chr(0x1F600), chr(0xD83D)
ECMA_MATCHES = [(bs(p), s, want) for p, s, want in [   # (pattern, string, ECMA-262 with the "u" flag)
    ("^%s$", NBSP, True), ("^%s$", LS, True), ("^%s$", BOM, True), ("^%s$", NEL, False), ("^%s$", chr(0x0E), False),
    ("^%S+$", NBSP * 2, False), ("^%S$", NEL, True), ("^[%s]$", IDEO_SPACE, True),
    ("^a.b$", "a\rb", False), ("^a.b$", "a" + LS + "b", False), ("^a.b$", "a" + NEL + "b", True), ("^.$", GRIN, True),
    ("^[^]$", "\n", True), ("^a[]$", "a", False), ("^[^%D]$", "7", True), ("^[%W]$", "_", False),
    ("^%d$", ARABIC_ONE, False), ("^%D$", ARABIC_ONE, True), ("^%w$", E_ACUTE, False), ("%bx", E_ACUTE + "x", True),
    ("%Bx", E_ACUTE + "x", False), ("^[%b]$", "\b", True), ("^%cJ$", "\n", True), ("^%x41%u0042%u{43}$", "ABC", True),
    ("^%uD83D%uDE00$", GRIN, True), ("^[%uD83D%uDE00]$", GRIN, True), ("^%uD83D$", LEAD, True),
    ("^%uD83D%u0041$", LEAD + "A", True), ("^[%u{1F600}]$", GRIN, True), ("^[%x41-%x43]$", "B", True),
    ("^[%x41-%x43]$", "D", False), ("^(?<y>a)b$", "ab", True), ("^(a)(?:b)$", "ab", True),
    ("^[a-c-]+$", "b-a", True), ("^[%-]$", "-", True), ("^a{2,}$", "aaa", True), ("^a+?%/$", "aa/", True),
    ("^[[]$", "[", True), ("^[&&~|]+$", "&~|", True), ("^%0$", chr(0), True), ("^%$%^$", "$^", True),
    ("^a*?$", "aa", True), ("^a??b$", "b", True), ("^a{2}?$", "aa", True), ("^(?:a)*$", "aaa", True),
    ("^()*a$", "a", True), ("^(?=a)a$", "a", True), ("^(?!b)a$", "a", True), ("^(a|b)+$", "abba", True),
    ("^(?:(?=a)a)+$", "aa", True), ("^(?<=)a$", "a", True), ("(a)|b", "b", True),
]]
ECMA_INVALID = [bs(p) for p in [
    "^[]a]$", "]", "}", "a{", "a{,3}", "%a", "%e", "%-", "%01", "%c" + E_ACUTE, "%p{L}", "(?P<x>a)", "(?i)a",
    "(?#c)", "a++", "a*?+", "[%d-z]", "[a-%w]", "[z-a]", "[a", "%", "(?<=a+)b", "%k<q>", "%u{110000}", "%x4", "%uD83",
    # backreferences: ECMA-262 matches one to a group that did not take part as empty, Python never matches
    "^(?<y>a)%k<y>$", "^(a)%1$", "^(?:(z)|gen)%1-[0-9]{3,}-[a-z0-9-]+$", "%k", "(a)%2",
    # quantified assertions and empty repeats are syntax errors under the "u" flag
    "(?=a)*", "(?!a)+", "(?<=a)?", "(?<!b)+", "%b{2}", "%B*?", "^*", "$+", "a|*", "(*)", "*a", "{2}",
    "a**", "a{2}{3}", "a*??", "a?+", "a)", "(a", "(?:a))",
    # repeat counts Python's re cannot represent
    "a{3,4294967296}", "x{4294967296}", "x{99999999999999999999}",
]]


@pytest.mark.parametrize("engine", ENGINES)
def test_ecma_pattern_semantics(engine):
    for pattern, s, want in ECMA_MATCHES:
        assert bool(vh.ecma_regex(pattern).search(s)) is want, (pattern, s)
        errs = vh.schema_errors({"type": "string", "pattern": pattern}, s, engine)
        assert (errs == []) is want, (pattern, s, errs)
    for pattern in ECMA_INVALID:
        with pytest.raises(re.error):
            vh.ecma_regex(pattern)
    assert vh.ecma_regex(r"\d") is vh.ecma_regex(r"\d")                         # compiled once
    schema = {"type": "object", "properties": {"pattern": {"type": "string", "pattern": "^ok$"}},
              "enum": [{"pattern": "(("}], "default": {"pattern": "(("}}
    assert list(vh.schema_patterns(schema)) == ["^ok$"]


BAD_SCHEMAS = {   # property schema for task_id -> what must be reported
    "unknown type": {"type": "strin"},
    "type object": {"type": {"x": 1}},
    "type list": {"type": []},
    "required string": {"type": "object", "required": "generation_id"},
    "required numbers": {"type": "object", "required": [1]},
    "minLength string": {"minLength": "5"},
    "maxItems negative": {"maxItems": -1},
    "minItems float": {"minItems": 1.5},
    "minimum string": {"minimum": "0"},
    "maximum bool": {"maximum": True},
    "enum string": {"enum": "abc"},
    "properties list": {"type": "object", "properties": []},
    "pattern number": {"pattern": 5},
    "format list": {"format": ["date-time"]},
    "defs list": {"$defs": []},
    "bad items": {"type": "array", "items": {"minLength": "5"}},
    "bad additionalProperties": {"type": "object", "additionalProperties": {"required": "x"}},
    "huge repeat": {"pattern": "^gen-[0-9]{3,4294967296}$"},
    "quantified lookahead": {"pattern": "^(?=gen-)*gen-"},
    "backreference": {"pattern": "^(?:(z)|gen)" + chr(92) + "1-"},
}


@pytest.mark.parametrize("case", sorted(BAD_SCHEMAS))
def test_handover_cli_rejects_malformed_schemas(case, tmp_path):
    """Council round 1: these crashed, passed or failed per file instead of exiting 2."""
    m = write_json(tmp_path / "T05-governance-gen-001.json", manifest())
    schema = write_json(tmp_path / "s.json", {"type": "object", "properties": {"task_id": BAD_SCHEMAS[case]}})
    for engine in ["builtin"] + (["auto", "jsonschema"] if vh.have_jsonschema() else []):
        r = tool("validate_handover", m, "--schema", schema, "--engine", engine)
        assert r.returncode == 2 and r.stderr.startswith("validate_handover: "), (case, engine, r.stdout, r.stderr)
        assert "Traceback" not in r.stderr and r.stdout == "", (case, engine)
    with pytest.raises(vh.SchemaError):
        vh.MiniValidator({"type": "object", "properties": {"task_id": BAD_SCHEMAS[case]}})


def test_builtin_engine_accepts_well_formed_keyword_values(tmp_path):
    v = vh.MiniValidator({"type": ["object", "null"], "required": [], "properties": {}, "$defs": {},
                          "additionalProperties": {"type": "integer", "minimum": -1.5, "maximum": 3,
                                                   "minLength": 0, "maxItems": 2.0, "enum": [1, 2]}})
    assert v.errors({"a": 1}) == [] and v.errors(None) == [] and len(v.errors({"a": 3})) == 1
    # ECMA-262 syntax that Python's re rejects: valid patterns, whichever engine checks the schema
    m = write_json(tmp_path / "T05-governance-gen-001.json", manifest())
    for pattern in ("^(?<task>T[0-9]{2})-[a-z]+$", bs("^T%u{30}5-[^]+$")):
        schema = write_json(tmp_path / "s.json", {"type": "object", "properties": {"task_id": {"pattern": pattern}}})
        for engine in ENGINES:
            r = tool("validate_handover", m, "--schema", schema, "--engine", engine, "--tasks", "")
            assert r.returncode == 0, (pattern, engine, r.stdout, r.stderr)
    for pattern, msg in (("(a)%1", "backreference"), ("(?<y>a)%k<y>", "backreference"), ("[%1]", "not an ECMA"),
                         ("%e", "not an ECMA"), ("%q", "not an ECMA")):
        with pytest.raises(re.error, match=msg):
            vh.ecma_translate(bs(pattern))
    if vh.have_jsonschema():
        vh.jsonschema_check_schema(SCHEMA)
        with pytest.raises(vh.SchemaError, match="not valid JSON Schema"):
            vh.jsonschema_check_schema({"type": "object", "required": "x"})
        # a keyword only jsonschema implements is refused by every engine (verifier round 3)
        schema = write_json(tmp_path / "s.json", {"type": "object", "properties": {"task_id": {"anyOf": [{}]}}})
        for engine in ("auto", "jsonschema", "builtin"):
            r = tool("validate_handover", m, "--schema", schema, "--engine", engine, "--tasks", "")
            assert r.returncode == 2 and "'anyOf' is not supported (under any engine" in r.stderr, (engine, r.stderr)


UNUSABLE_SCHEMAS = {   # whole schema -> part of the message; every engine must exit 2 (verifier round 2)
    "null": (None, "must be a JSON object"), "true": (True, "must be a JSON object"),
    "false": (False, "must be a JSON object"), "list": ([], "must be a JSON object"),
    "$schema number": ({"$schema": 5, "type": "object"}, "$schema 5 is not one of"),
    "$schema draft-07": ({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"}, "is not one of"),
    "$schema unknown": ({"$schema": "http://example.com/unknown", "type": "object"}, "is not one of"),
    "ref nowhere": ({"type": "object", "properties": {"task_id": {"$ref": "#/$defs/nope"}}}, "unresolvable $ref"),
    "ref remote": ({"type": "object", "properties": {"task_id": {"$ref": "https://example.com/s.json"}}},
                   "only local $ref"),
    "ref anchor": ({"type": "object", "$defs": {"a": {"$anchor": "x"}}, "properties": {"task_id": {"$ref": "#x"}}},
                   "only local $ref"),
    "ref non-schema": ({"type": "object", "required": ["task_id"], "properties": {"task_id": {"$ref": "#/required"}}},
                       "does not name a schema"),
    "ref into data": ({"type": "object", "properties": {"task_id": {"$ref": "#/$defs/a/default"}},
                       "$defs": {"a": {"default": {"minLength": 99}}}}, "points into instance data"),
    "ref in allOf to nowhere": ({"type": "object", "allOf": [{"$ref": "#/$defs/missing"}]}, "unresolvable $ref"),
    "dynamicRef": ({"type": "object", "properties": {"task_id": {"$dynamicRef": "#meta"}}}, "$dynamicRef is not"),
    "nested $id": ({"type": "object", "properties": {"task_id": {"$id": "https://example.com/x", "type": "string"}}},
                   "$id is supported only at the root"),
    "required duplicates": ({"type": "object", "required": ["task_id", "task_id"]}, "task_id"),
    "type duplicates": ({"type": "object", "properties": {"task_id": {"type": ["string", "string"]}}}, "string"),
    "title number": ({"title": 5, "type": "object"}, "string"),
    "$id number": ({"$id": 5, "type": "object"}, "string"),
    "$id fragment": ({"$id": "https://example.com/s.json#frag", "type": "object"}, "without a fragment"),
    "description list": ({"description": [], "type": "object"}, "string"),
    "deprecated string": ({"deprecated": "yes", "type": "object"}, "boolean"),
    "examples object": ({"examples": {}, "type": "object"}, "array"),
    "repeat above bound": ({"type": "object", "properties": {"task_id": {"pattern": "(?:){2147483648}"}}},
                           "above 65535"),
    "nested repeats": ({"type": "object", "properties": {"task_id": {"pattern": "^((?:x){300}){300}$"}}},
                       "force 90000 repetitions"),
    "pattern in allOf": ({"type": "object", "allOf": [{"properties": {"task_id": {"pattern": "a{,3}"}}}]}, "pattern"),
    "patternProperties": ({"type": "object", "properties": {"workspace": {"patternProperties": {"^b": {}}}}},
                          "patternProperties is not supported"),
}


@pytest.mark.parametrize("case", sorted(UNUSABLE_SCHEMAS))
def test_handover_engines_agree_on_unusable_schemas(case, tmp_path):
    schema_doc, needle = UNUSABLE_SCHEMAS[case]
    m = write_json(tmp_path / "T05-governance-gen-001.json", manifest())
    schema = write_json(tmp_path / "s.json", schema_doc)
    for engine in ["builtin"] + (["auto", "jsonschema"] if vh.have_jsonschema() else []):
        r = tool("validate_handover", m, "--schema", schema, "--engine", engine, "--tasks", "")
        assert r.returncode == 2 and r.stderr.startswith("validate_handover: ") and needle in r.stderr, \
            (case, engine, r.stdout, r.stderr)
        assert "Traceback" not in r.stderr and r.stdout == "", (case, engine)


def test_handover_schema_precheck_and_references(tmp_path, monkeypatch):
    for dialect in vh.DIALECTS:
        vh.precheck_schema(dict(SCHEMA, **{"$schema": dialect}))
    vh.precheck_schema({"type": "object"})                                       # no $schema: 2020-12
    root = {"$defs": {"a/b": {"type": "string"}, "t~x": {"type": "integer"}, "sp ace": True},
            "allOf": [{"type": "object"}], "prefixItems": [{"const": 1}]}
    assert vh.resolve_ref(root, "#") is root and vh.resolve_ref(root, "#/$defs/a~1b") == {"type": "string"}
    assert vh.resolve_ref(root, "#/$defs/t~0x") == {"type": "integer"} and vh.resolve_ref(root, "#/$defs/sp%20ace")
    assert vh.resolve_ref(root, "#/allOf/0") == {"type": "object"}
    for bad in ("#/allOf/1", "#/allOf/01", "#/allOf/-1", "#/allOf/x", "#/$defs/a~1b/type", "x#/a", "#a", 5, None):
        with pytest.raises(vh.SchemaError):
            vh.resolve_ref(root, bad)
    nodes = list(vh.schema_nodes({"properties": {"a": {"items": [{"x": 1}], "not": {"y": 2}}},
                                  "anyOf": [True, {"z": 3}], "const": {"q": 4}, "unknown": {"w": 5}}))
    assert sorted(k for n in nodes for k in n if k in "xyzqw") == ["x", "y", "z"]
    vh.precheck_schema({"$defs": {"a": {"type": "string"}}, "properties": {"x": {"$ref": "#/$defs/a"}},
                        "items": {"$ref": "#"}})
    # the jsonschema engine: any failure of jsonschema itself on the schema is an unusable schema
    if vh.have_jsonschema():
        import jsonschema

        def boom(schema, *a, **k):
            raise AttributeError("'int' object has no attribute 'decode'")
        monkeypatch.setattr(jsonschema.validators, "validator_for", boom)
        with pytest.raises(vh.SchemaError, match="jsonschema cannot check the schema: AttributeError"):
            vh.jsonschema_check_schema({"type": "object"})
        monkeypatch.undo()
        vh.jsonschema_check_schema({"$id": "not a uri at all", "type": "object"})    # formats are not asserted
        vh.jsonschema_check_schema(SCHEMA)


def test_handover_engines_assert_the_same_formats(tmp_path):
    """jsonschema asserted every format it knows (some only with optional packages); the built-in
    engine only date-time. Now both assert date-time and treat the others as annotations."""
    m = write_json(tmp_path / "T05-governance-gen-001.json", manifest())
    for fmt, valid in (("email", True), ("ipv4", True), ("uri", True), ("hostname", True), ("regex", True),
                       ("date", True), ("date-time", False)):
        schema = write_json(tmp_path / "s.json", {"type": "object", "properties": {"task_id": {"format": fmt}}})
        for engine in ENGINES:
            r = tool("validate_handover", m, "--schema", schema, "--engine", engine, "--tasks", "")
            assert r.returncode == (0 if valid else 1), (fmt, engine, r.stdout, r.stderr)
            errs = vh.schema_errors({"type": "string", "format": fmt}, "T05-governance(", engine)
            assert (errs == []) is valid, (fmt, engine, errs)


def test_ecma_repeat_bounds():
    ok = ("x{65535}", "^(?:a{3}){3}$", "(?:x{255}){257}", "^a{0,65535}$", "(?:a*){65535}", "(?=a{9})a", "a{2}?b{3}",
          "(x{65535})", "x{65535}y{65535}", "(?:(?:a){5}|b){5}",
          "x{300}(?:(?:y)z){300}", "x{300}(?:y{2}z){300}", "(?:(?:x){0}){65535}", "(?:x+){65535}")
    for pattern in ok:
        vh.ecma_translate(pattern)
    assert vh.ecma_regex("^(?:a{2}){3}$").search("a" * 6) and not vh.ecma_regex("^(?:a{2}){3}$").search("a" * 5)
    for pattern, needle in (("x{65536}", "repeat count 65536"), ("x{0,65536}", "repeat count 65536"),
                            ("x{65536,}", "repeat count 65536"), ("x{0000000000065535}", None),
                            ("x{" + "9" * 5000 + "}", "repeat count 99999"), ("(?:x{256}){256}", "force 65536"),
                            ("(?:(?:a){2}b){32768}", "force 65536"), ("(?:x+){2}(?:y{300}){300}", "force 90000"),
                            ("(?=(?:x){300}){300}", "nothing to repeat"), ("((?:x){256}){256}", "force 65536"),
                            ("(?:(?:x{300})z){300}", "force 90000"), ("((((x{2})))){40000}", "force 80000"),
                            ("*", "nothing to repeat"), (")", "unbalanced")):
        if needle is None:
            vh.ecma_translate(pattern)
            continue
        with pytest.raises(re.error, match=re.escape(needle)):
            vh.ecma_translate(pattern)


def test_handover_cli_rejects_untranslatable_patterns(tmp_path):
    m = write_json(tmp_path / "T05-governance-gen-001.json", manifest())
    for pattern in (r"^\p{L}$", "a{,3}"):
        schema = write_json(tmp_path / "s.json", {"type": "object", "properties": {"task_id": {"pattern": pattern}}})
        for engine in ENGINES:
            r = tool("validate_handover", m, "--schema", schema, "--engine", engine)
            assert r.returncode == 2 and "pattern" in r.stderr, (pattern, engine, r.stderr)


MALFORMED = {"timestamp": 12345, "measurements": 5, "constraints_touched": True, "workspace": "main",
             "epistemic_ledger": [], "procedures": {"a": 1}}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("field", sorted(MALFORMED))
def test_handover_wrong_types_are_reported_not_raised(engine, field, tmp_path):
    for value in (MALFORMED[field], None, True, 1.5):
        r = check(dict(manifest(), **{field: value}), tmp_path, engine)
        assert any(field in e for e in r["errors"]), (field, value, r["errors"])
    assert vh.is_date_time(12345) and not vh.is_date_time("12345")


def test_handover_one_bad_file_does_not_stop_the_others(tmp_path, monkeypatch):
    d = tmp_path / "h"
    d.mkdir()
    for n, doc in ((1, manifest()), (2, dict(manifest(), timestamp=12345, measurements=5)), (3, manifest())):
        write_json(d / f"T05-governance-gen-00{n}.json", doc)
    for engine in ENGINES:
        r = tool("validate_handover", d, "--engine", engine, "--json")
        rep = json.loads(r.stdout)
        assert r.returncode == 1 and rep["invalid"] == 1 and len(rep["files"]) == 3, r.stdout + r.stderr
        assert [f["valid"] for f in rep["files"]] == [True, False, True]

    def boom(path, *a, **k):
        raise KeyError("surprise")

    monkeypatch.setattr(vh, "check_manifest", boom)
    res = vh.check_one(d / "T05-governance-gen-001.json", SCHEMA, INVARIANT_IDS, TASK_IDS, "builtin")
    assert res["errors"] == ["validator failed on this file: KeyError: 'surprise'"] and res["warnings"] == []


def test_handover_main_in_process(tmp_path, monkeypatch, capsys):
    m = write_json(tmp_path / "T05-governance-gen-001.json", manifest())
    assert vh.main(["--new", "T05-governance"]) == 0 and '"task_id": "T05-governance"' in capsys.readouterr().out
    empty = tmp_path / "INV.md"
    empty.write_text("# no table\n", encoding="utf-8")
    assert vh.main([str(m), "--invariants", str(empty)]) == 2
    assert "no invariant ids found" in capsys.readouterr().err
    monkeypatch.setitem(sys.modules, "jsonschema", None)              # not installed
    assert vh.main([str(m), "--engine", "builtin"]) == 0 and vh.main([str(m)]) == 0
    assert vh.main([str(m), "--engine", "jsonschema"]) == 2
    assert "jsonschema is not installed" in capsys.readouterr().err


def test_handover_new_needs_a_known_task_id(tmp_path):
    d = tmp_path / "h"
    d.mkdir()
    for bad, needle in (("NOPE-task", "is not a task in"), ("../escape", "is not a task id"),
                        ("T05-governance/../x", "is not a task id"), ("", "is not a task id")):
        r = tool("validate_handover", "--new", bad, "--write", "--handovers-dir", d)
        assert r.returncode == 2 and needle in r.stderr, (bad, r.stderr)
        assert tool("validate_handover", "--new", bad).returncode == 2
    assert list(tmp_path.rglob("*.json")) == []
    r = tool("validate_handover", "--new", "T99-unlisted", "--tasks", "")     # task check disabled
    assert r.returncode == 0 and json.loads(r.stdout)["task_id"] == "T99-unlisted"
    r = tool("validate_handover", "--new", "T05-governance", "--tasks", tmp_path / "missing.json")
    assert r.returncode == 2 and "missing.json" in r.stderr


def test_builtin_engine_ref_cycles():
    for schema in ({"$defs": {"a": {"$ref": "#/$defs/a"}}, "$ref": "#/$defs/a"},
                   {"$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"$ref": "#/$defs/a"}}, "type": "object",
                    "properties": {"x": {"$ref": "#/$defs/a"}}},
                   {"$ref": "#"}):
        with pytest.raises(vh.SchemaError, match="cycle"):
            vh.MiniValidator(schema)
    tree = {"$defs": {"node": {"type": "object", "additionalProperties": False,
                               "properties": {"kids": {"type": "array", "items": {"$ref": "#/$defs/node"}}}}},
            "$ref": "#/$defs/node"}                           # recursion through the instance is fine
    v = vh.MiniValidator(tree)
    assert v.errors({"kids": [{"kids": []}, {}]}) == []
    assert len(v.errors({"kids": [{"kids": [{"x": 1}]}]})) == 1


def test_date_time_format():
    for ok in ("2026-10-04T13:09:08Z", "2026-10-04t13:09:08.123+02:00", "2024-02-29T00:00:00Z",
               "2016-12-31T23:59:60Z", "2026-01-01T00:00:00-23:59"):
        assert vh.is_date_time(ok), ok
    for bad in ("2026-02-29T00:00:00Z", "2026-10-04 13:09:08Z", "2026-10-04T24:00:00Z", "2026-10-04T13:09:08",
                "2026-10-04T13:09:08+25:00", "2026-1-04T13:09:08Z", "2026-10-04T12:60:00Z",
                "2026-10-04T12:00:61Z", "2026-10-04T12:00:00+02:60", "2026-00-10T12:00:00Z", "2026-10-00T12:00:00Z",
                "2026-13-01T00:00:00Z"):
        assert not vh.is_date_time(bad), bad


def test_builtin_engine_keywords():
    with pytest.raises(vh.SchemaError):
        vh.MiniValidator({"type": "object", "properties": {"a": {"oneOf": [{"type": "string"}]}}})
    with pytest.raises(vh.SchemaError):
        vh.MiniValidator({"$defs": {"x": {"not": {}}}})
    with pytest.raises(vh.SchemaError):
        vh.MiniValidator({"$ref": "#/$defs/missing"})
    with pytest.raises(vh.SchemaError):
        vh.MiniValidator({"type": "strnig"})
    v = vh.MiniValidator({"type": "array", "minItems": 1, "maxItems": 3,
                          "items": {"type": "integer", "minimum": 0, "maximum": 9}})
    assert v.errors([1, 2.0]) == []
    assert len(v.errors([])) == 1 and len(v.errors([-1, 1.5, True])) == 3
    assert len(v.errors([10, 0, 0, 0])) == 2                          # maximum + maxItems
    s = vh.MiniValidator({"type": "string", "minLength": 2, "maxLength": 3})
    assert s.errors("ab") == [] and len(s.errors("a")) == 1 and len(s.errors("abcd")) == 1
    o = vh.MiniValidator({"type": "object", "properties": {"a": {"const": [1, {"b": None}]}},
                          "additionalProperties": {"enum": [[1, 2], "x"]}})
    assert o.errors({"a": [1, {"b": None}], "z": [1, 2], "y": "x"}) == []
    assert len(o.errors({"a": [1, {"b": 0}]})) == 1 and len(o.errors({"a": [1]})) == 1
    assert len(o.errors({"a": [True, {"b": None}]})) == 1               # true is not 1 in JSON
    assert len(o.errors({"z": [2, 1]})) == 1 and len(o.errors({"z": [1, 2, 3]})) == 1
    assert vh._json_equal(1, 1.0) and not vh._json_equal({"a": 1}, {"b": 1}) and not vh._json_equal("1", 1)
    b = vh.MiniValidator({"type": "object", "properties": {"never": False, "any": True}})
    assert b.errors({"any": [1]}) == [] and b.errors({"never": 1}) == [("/never", "no value is allowed here")]


def test_have_jsonschema_without_the_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "jsonschema", None)          # import now raises ImportError
    assert vh.have_jsonschema() is False
    assert vh.schema_errors(SCHEMA, manifest(), "auto") == []


def test_handover_cli_rejects_unusable_schema_and_non_objects(tmp_path):
    schema = tmp_path / "s.json"
    write_json(schema, {"type": "object", "properties": {"a": {"anyOf": []}}})
    m = write_json(tmp_path / "T05-governance-gen-001.json", manifest())
    r = tool("validate_handover", m, "--schema", schema, "--engine", "builtin")
    assert r.returncode == 2 and "not supported" in r.stderr
    arr = write_json(tmp_path / "T05-governance-gen-002.json", [manifest()])
    r = tool("validate_handover", arr, "--engine", "builtin")
    assert r.returncode == 1 and "FAIL" in r.stdout and "not of type object" in r.stdout


def test_handover_cli(tmp_path):
    d = tmp_path / "handovers"
    d.mkdir()
    write_json(d / "T05-governance-gen-001.json", manifest())
    r = tool("validate_handover", d, "--engine", "builtin")
    assert r.returncode == 0 and "1/1" in r.stdout, r.stdout + r.stderr
    bad = manifest()
    bad["trigger"] = "BORED"
    write_json(d / "T05-governance-gen-002.json", bad)
    r = tool("validate_handover", d, "--engine", "builtin")
    assert r.returncode == 1 and "FAIL" in r.stdout and "OK" in r.stdout
    r = tool("validate_handover", d / "T05-governance-gen-001.json", "--json")
    assert r.returncode == 0 and json.loads(r.stdout)["invalid"] == 0
    r = tool("validate_handover", d, "--json", "--engine", "builtin")
    assert r.returncode == 1 and json.loads(r.stdout)["invalid"] == 1
    write_json(tmp_path / "misnamed.json", manifest())
    assert tool("validate_handover", tmp_path / "misnamed.json").returncode == 0
    assert tool("validate_handover", tmp_path / "misnamed.json", "--strict").returncode == 1
    assert tool("validate_handover", tmp_path / "does-not-exist").returncode == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert tool("validate_handover", empty).returncode == 0


def test_handover_skeleton_validates(tmp_path):
    r = tool("validate_handover", "--new", "T05-governance")
    assert r.returncode == 0
    sk = json.loads(r.stdout)
    assert sk["task_id"] == "T05-governance" and r.stdout.startswith('{\n  "generation_id": "gen-')
    assert sk["workspace"]["branch"] and re.fullmatch(r"[0-9a-f]{7,40}", sk["workspace"]["base_commit"])
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", sk["timestamp"]), sk["timestamp"]
    assert check(sk, tmp_path, "builtin", name=f"T05-governance-gen-{sk['generation_id'][4:7]}.json")["errors"] == []


def test_handover_skeleton_numbering_and_write(tmp_path):
    d = tmp_path / "handovers"
    d.mkdir()
    first = manifest()
    write_json(d / "T05-governance-gen-001.json", first)
    (d / "T05-governance-gen-002.json").write_text("", encoding="utf-8")   # what `--new ... > file` creates first
    sk = vh.skeleton("T05-governance", d)
    assert (sk["generation_id"], sk["parent_generation"]) == ("gen-002-governance", "gen-001-governance")
    write_json(d / "T06-store-gen-001.json", dict(manifest(), generation_id="gen-001-store-io", task_id="T06-store"))
    assert vh.skeleton("T06-store", d)["parent_generation"] == "gen-001-store-io"     # the parent's real id
    write_json(d / "T06-store-gen-001.json", dict(manifest(), generation_id="bogus", task_id="T06-store"))
    assert vh.skeleton("T06-store", d)["parent_generation"] == "gen-001-store"        # unusable id: derived
    (d / "T06-store-gen-001.json").unlink()
    assert vh.skeleton("T06-store", d)["parent_generation"] is None
    write_json(d / "T06-store-gen-000.json", dict(manifest(), generation_id="gen-000-store", task_id="T06-store"))
    sk0 = vh.skeleton("T06-store", d)
    assert (sk0["generation_id"], sk0["parent_generation"]) == ("gen-001-store", "gen-000-store")
    (d / "T06-store-gen-000.json").unlink()
    (d / "T05-governance-gen-002.json").unlink()
    r = tool("validate_handover", "--new", "T05-governance", "--write", "--handovers-dir", d)
    assert r.returncode == 0, r.stderr
    made = d / "T05-governance-gen-002.json"
    assert r.stdout.strip() == str(made)
    m = json.loads(made.read_text(encoding="utf-8"))
    assert m["generation_id"] == "gen-002-governance" and m["parent_generation"] == "gen-001-governance"
    m["workspace"] = manifest()["workspace"]
    write_json(made, m)
    r = tool("validate_handover", made, "--strict", "--engine", "builtin")
    assert r.returncode == 0, r.stdout                       # file name and lineage agree
    r = tool("validate_handover", "--new", "T05-governance", "--write", "--handovers-dir", d)
    assert r.returncode == 0 and r.stdout.strip().endswith("T05-governance-gen-003.json")
    r = tool("validate_handover", "--new", "T05-governance", "--write", "--handovers-dir", tmp_path / "nowhere")
    assert r.returncode == 2 and "cannot create" in r.stderr
    assert tool("validate_handover", "--write").returncode == 2


# ------------------------------------------------------------ check_golden
def golden_copy(tmp_path) -> Path:
    root = tmp_path / "repo"
    for src in (ROOT / "tests" / "golden").rglob("*"):
        if src.is_file() and "__pycache__" not in src.parts:
            dst = root / src.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)      # not copy2: the original is read-only
    return root


def test_golden_lock_passes_on_repo():
    res = cg.check(ROOT)
    assert cg.passed(res), res
    assert "tests/golden/test_golden_invariants.py" in res["ok"]
    r = tool("check_golden")
    assert r.returncode == 0 and "golden lock OK" in r.stdout


def test_golden_lock_detects_tampering(tmp_path):
    real = ROOT / "tests" / "golden" / "test_golden_invariants.py"
    before = hashlib.sha256(real.read_bytes()).hexdigest()
    root = golden_copy(tmp_path)
    assert tool("check_golden", "--root", root).returncode == 0
    target = root / "tests" / "golden" / "test_golden_invariants.py"
    target.write_bytes(target.read_bytes().replace(b"1e-3", b"1e-1", 1))
    r = tool("check_golden", "--root", root)
    assert r.returncode == 1 and "MODIFIED" in r.stdout
    assert hashlib.sha256(real.read_bytes()).hexdigest() == before          # real file untouched


def test_golden_lock_line_endings_and_exemptions(tmp_path):
    root = golden_copy(tmp_path)
    g = root / "tests" / "golden"
    t = g / "test_golden_invariants.py"
    lf = t.read_bytes().replace(b"\r\n", b"\n")
    t.write_bytes(lf.replace(b"\n", b"\r\n"))                # CRLF checkout hashes like LF
    assert cg.passed(cg.check(root))
    t.write_bytes(lf)
    assert cg.passed(cg.check(root))
    (g / "__pycache__").mkdir()
    (g / "__pycache__" / "x.cpython-313.pyc").write_bytes(b"junk")
    (g / "sub").mkdir()
    (g / "sub" / "__init__.py").write_text("\n", encoding="utf-8")
    assert cg.passed(cg.check(root))
    (g / "sub" / "__init__.py").write_text("import os\n", encoding="utf-8")   # code runs at import
    assert cg.check(root)["unlisted"] == ["tests/golden/sub/__init__.py"]


def test_golden_lock_unlisted_missing_malformed(tmp_path):
    root = golden_copy(tmp_path)
    g = root / "tests" / "golden"
    (g / "conftest.py").write_text("def pytest_configure(): pass\n", encoding="utf-8")
    r = tool("check_golden", "--root", root)
    assert r.returncode == 1 and "UNLISTED" in r.stdout and "conftest.py" in r.stdout
    (g / "conftest.py").unlink()
    (g / "test_golden_invariants.py").unlink()
    r = tool("check_golden", "--root", root)
    assert r.returncode == 1 and "MISSING" in r.stdout
    (g / "MANIFEST.sha256").write_text("not-a-hash  tests/golden/x.py\n", encoding="utf-8")
    r = tool("check_golden", "--root", root)
    assert r.returncode == 2 and "error:" in r.stdout
    for bad in ("docs/FORMAT.md", "tests/golden/../../docs/FORMAT.md", "tests/golden//x.py", "/tests/golden/x.py",
                "x.py", "C:/tests/golden/x.py"):
        (g / "MANIFEST.sha256").write_text("0" * 64 + f"  {bad}\n", encoding="utf-8")
        assert tool("check_golden", "--root", root).returncode == 2, bad
    (g / "MANIFEST.sha256").write_text("# only a comment\n", encoding="utf-8")
    assert tool("check_golden", "--root", root).returncode == 2
    assert tool("check_golden", "--root", tmp_path / "nowhere", "--update", "--i-am-a-maintainer").returncode == 2


def test_golden_manifest_duplicates_and_junit_counts(tmp_path):
    line = "0" * 64 + "  tests/golden/a.py\n"
    with pytest.raises(cg.ManifestError, match="duplicate"):
        cg.parse_manifest(line + line.replace("0" * 64, "1" * 64))
    assert cg.parse_manifest(line + "# c\n\n" + line.replace("a.py", "b.py"))[1] == ("0" * 64, "tests/golden/b.py")
    xml = tmp_path / "r.xml"
    xml.write_text('<testsuites><testsuite errors="2" tests="1"><testcase name="t"/></testsuite>'
                   '<testsuite errors=""/></testsuites>', encoding="utf-8")
    assert cg.junit_counts(xml) == {"tests": 1, "passed": 1, "failed": 0, "errors": 2, "skipped": 0}
    xml.write_text('<testsuites><testsuite errors=""><testcase name="t"/></testsuite></testsuites>', encoding="utf-8")
    assert cg.junit_counts(xml)["errors"] == 0
    cases = "".join(f'<testcase name="{k}"><{k}/></testcase>' for k in ("failure", "error", "error", "skipped"))
    xml.write_text(f'<testsuites><testsuite errors="0">{cases}<testcase name="ok"/></testsuite></testsuites>',
                   encoding="utf-8")
    assert cg.junit_counts(xml) == {"tests": 5, "passed": 1, "failed": 1, "errors": 2, "skipped": 1}


def test_golden_update_requires_confirmation(tmp_path):
    root = golden_copy(tmp_path)
    g = root / "tests" / "golden"
    (g / "test_new.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    locked = g / "test_golden_invariants.py"
    locked.write_bytes(locked.read_bytes() + b"\n# maintainer edit\n")
    manifest_before = (g / "MANIFEST.sha256").read_bytes()
    r = tool("check_golden", "--root", root, "--update")
    assert r.returncode == 2 and "MAINTAINERS ONLY" in r.stderr and "refusing" in r.stderr
    assert (g / "MANIFEST.sha256").read_bytes() == manifest_before
    r = tool("check_golden", "--root", root, "--update", "--i-am-a-maintainer")
    assert r.returncode == 0 and "MAINTAINERS ONLY" in r.stderr
    assert "  + " in r.stderr and "tests/golden/test_new.py" in r.stderr     # shows what changed
    assert "  - " + manifest_before.decode().splitlines()[0] in r.stderr
    text = (g / "MANIFEST.sha256").read_bytes()
    assert b"\r" not in text and b"tests/golden/test_new.py" in text
    assert tool("check_golden", "--root", root).returncode == 0
    assert cg.main(["--root", str(root), "--update", "--i-am-a-maintainer"]) == 0
    assert cg.main(["--root", str(root), "-q"]) == 0


GOLDEN_DEMO = '''\
from hearth import answer


def test_answer():
    assert answer() == 42


def test_also():
    assert answer() + 1 == 43
'''

# What tampering code does: every test "passes" without its body running.
NEUTRALISE = "import _pytest.python as _p\n_p.Function.runtest = lambda self: None\n"
FAILING_GOLDEN = "import colorsys\nfrom hearth import answer\n\n\ndef test_answer():\n    assert answer() == 41\n"


def golden_demo_repo(tmp_path, test_src=GOLDEN_DEMO) -> Path:
    root = tmp_path / "repo"
    (root / "tests" / "golden").mkdir(parents=True)
    (root / "python" / "hearth").mkdir(parents=True)
    (root / "python" / "hearth" / "__init__.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    (root / "tests" / "golden" / "__init__.py").write_text("", encoding="utf-8")       # as in the repository
    (root / "tests" / "golden" / "test_gold.py").write_text(test_src, encoding="utf-8")
    r = tool("check_golden", "--root", root, "--update", "--i-am-a-maintainer")
    assert r.returncode == 0, r.stderr
    return root


def plant_pyc(tmp_path, code: str, cache_dir: Path, module: str) -> Path:
    """An unchecked-hash .pyc (PEP 552) for `module`: Python loads it without looking at the source."""
    import py_compile
    src = tmp_path / f"planted_{module}.py"
    src.write_text(code, encoding="utf-8")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{module}.{sys.implementation.cache_tag}.pyc"
    py_compile.compile(str(src), cfile=str(out), doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    return out


def make_dir_link(link: Path, target: Path) -> bool:
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":            # junctions need no privilege
        import _winapi
        try:
            _winapi.CreateJunction(str(target), str(link))
            return True
        except OSError:
            pass
    return False


@pytest.mark.parametrize("vector", ["tests/__init__.py", "python/colorsys.py", "hearth bytecode"])
def test_golden_run_ignores_code_outside_the_lock(vector, tmp_path):
    root = golden_demo_repo(tmp_path, FAILING_GOLDEN)
    if vector == "hearth bytecode":          # loaded instead of the reviewed source
        plant_pyc(tmp_path, "def answer():\n    return 41\n", root / "python" / "hearth" / "__pycache__", "__init__")
    else:
        (root / vector).write_text(NEUTRALISE, encoding="utf-8")
    plain = subprocess.run([sys.executable, "-m", "pytest", "tests/golden", "-q", "-p", "no:cacheprovider"],
                           cwd=root, capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, PYTHONPATH=str(root / "python")))
    assert "1 passed" in plain.stdout, plain.stdout             # the vector works on a plain run ...
    assert tool("check_golden", "--root", root).returncode == 0  # ... and is outside the lock
    r = subprocess.run([sys.executable, "-E", str(TOOLS / "check_golden.py"), "--root", str(root), "--run"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 1 and "golden suite FAILED (0 passed, 1 failed" in r.stdout, r.stdout


def test_golden_lock_rejects_unchecked_bytecode(tmp_path):
    import py_compile
    root = golden_demo_repo(tmp_path)
    cache = root / "tests" / "golden" / "__pycache__"
    cache.mkdir()
    src = str(root / "tests" / "golden" / "test_gold.py")
    py_compile.compile(src, cfile=str(cache / "a.pyc"), doraise=True,                # what Python itself writes
                       invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    py_compile.compile(src, cfile=str(cache / "b.pyc"), doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)
    (cache / "short.pyc").write_bytes(b"\x00\x00\x00\x00\x01\x00\x00")              # 7 bytes: no flags word
    assert tool("check_golden", "--root", root).returncode == 0
    planted = plant_pyc(tmp_path, NEUTRALISE, cache, "__init__")
    assert cg.unchecked_pyc(planted) and not cg.unchecked_pyc(cache / "b.pyc")
    r = tool("check_golden", "--root", root, "--run")
    assert r.returncode == 1 and f"BYTECODE  tests/golden/__pycache__/{planted.name}" in r.stdout, r.stdout
    assert "1 unchecked .pyc" in r.stdout and "golden suite" not in r.stdout


def test_golden_lock_rejects_links(tmp_path):
    root = golden_demo_repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "test_more.py").write_text(NEUTRALISE + "\n\ndef test_more():\n    pass\n", encoding="utf-8")
    if not make_dir_link(root / "tests" / "golden" / "more", outside):
        pytest.skip("cannot create a symbolic link or junction here")
    assert cg.is_link(root / "tests" / "golden" / "more") and not cg.is_link(outside)
    r = tool("check_golden", "--root", root, "--run")
    assert r.returncode == 1 and "LINK      tests/golden/more " in r.stdout and "1 link(s)" in r.stdout, r.stdout
    assert "test_more" not in r.stdout and "golden suite" not in r.stdout          # never followed
    os.rmdir(root / "tests" / "golden" / "more") if os.name == "nt" else os.unlink(root / "tests" / "golden" / "more")
    assert tool("check_golden", "--root", root).returncode == 0
    shutil.move(str(root / "tests" / "golden"), str(tmp_path / "real_golden"))       # the directory itself
    assert make_dir_link(root / "tests" / "golden", tmp_path / "real_golden")
    r = tool("check_golden", "--root", root)
    assert r.returncode == 1 and "LINK      tests/golden " in r.stdout, r.stdout
    r = tool("check_golden", "--root", root, "--update", "--i-am-a-maintainer")
    assert r.returncode == 2 and "refusing to lock through links: tests/golden" in r.stderr
    assert cg.check(tmp_path / "real_golden")["errors"]                            # nothing was written there
    # round 2: a link one level up, into a checkout whose lock matches (the base, in CI's workspace)
    os.rmdir(root / "tests" / "golden") if os.name == "nt" else os.unlink(root / "tests" / "golden")
    other = tmp_path / "other"
    other.mkdir()
    shutil.move(str(root / "tests"), str(other / "tests"))
    shutil.move(str(tmp_path / "real_golden"), str(other / "tests" / "golden"))
    assert cg.passed(cg.check(other))                                                # a consistent lock ...
    assert make_dir_link(root / "tests", other / "tests")
    assert (root / "tests" / "golden" / "MANIFEST.sha256").is_file()                 # ... reachable through it
    res = cg.check(root)
    assert res["links"] == ["tests"] and res["ok"] == [] and res["errors"] == [] and not cg.passed(res)
    r = tool("check_golden", "--root", root, "--run")
    assert r.returncode == 1 and "LINK      tests " in r.stdout and "golden suite" not in r.stdout, r.stdout
    r = tool("check_golden", "--root", root, "--base-manifest", other / "tests" / "golden" / "MANIFEST.sha256")
    assert r.returncode == 1 and "matches the base" not in r.stdout
    r = tool("check_golden", "--root", root, "--update", "--i-am-a-maintainer")
    assert r.returncode == 2 and "refusing to lock through links: tests" in r.stderr


def test_golden_stage_copies_only_locked_files(tmp_path):
    root = golden_demo_repo(tmp_path)
    g = root / "tests" / "golden"
    (g / "sub").mkdir()
    (g / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (g / "sub" / "test_sub.py").write_text("def test_s():\n    pass\n", encoding="utf-8")
    assert tool("check_golden", "--root", root, "--update", "--i-am-a-maintainer").returncode == 0
    res = cg.check(root)
    assert cg.passed(res) and res["inits"] == ["tests/golden/__init__.py", "tests/golden/sub/__init__.py"]
    plant_pyc(tmp_path, NEUTRALISE, g / "__pycache__", "x")         # appears after the check: not staged
    run = tmp_path / "run"
    cg.stage(root, run, res)
    staged = sorted(p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file())
    assert staged == ["tests/golden/__init__.py", "tests/golden/sub/__init__.py", "tests/golden/sub/test_sub.py",
                      "tests/golden/test_gold.py"]
    assert (run / "tests" / "golden" / "test_gold.py").read_bytes() == (g / "test_gold.py").read_bytes()
    (g / "test_gold.py").write_text("def test_answer():\n    pass\n", encoding="utf-8")      # after the check
    with pytest.raises(cg.ManifestError, match="changed after the lock was verified"):
        cg.stage(root, tmp_path / "run2", res)


def test_golden_run_uses_locked_conftest(tmp_path):
    root = golden_demo_repo(tmp_path, "def test_fixture(magic):\n    assert magic == 7\n")
    (root / "tests" / "golden" / "conftest.py").write_text(
        "import pytest\n\n\n@pytest.fixture\ndef magic():\n    return 7\n", encoding="utf-8")
    assert tool("check_golden", "--root", root, "--update", "--i-am-a-maintainer").returncode == 0
    r = tool("check_golden", "--root", root, "--run")
    assert r.returncode == 0 and "golden suite OK: 1 passed" in r.stdout, r.stdout


def test_golden_tool_ignores_modules_planted_beside_it(tmp_path):
    root = golden_demo_repo(tmp_path, "def test_a():\n    assert False\n")
    tools = tmp_path / "tools"
    tools.mkdir()
    shutil.copyfile(TOOLS / "check_golden.py", tools / "check_golden.py")
    (tools / "tempfile.py").write_text("import os\nprint('golden suite OK: planted')\nos._exit(0)\n",
                                       encoding="utf-8")
    r = subprocess.run([sys.executable, "-E", str(tools / "check_golden.py"), "--root", str(root), "--run"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 1 and "planted" not in r.stdout and "1 failed" in r.stdout, r.stdout


def test_golden_run_cannot_be_neutralised_from_outside(tmp_path):
    root = golden_demo_repo(tmp_path)
    r = tool("check_golden", "--root", root, "--run")
    assert r.returncode == 0 and "golden suite OK: 2 passed, 0 failed, 0 errors, 0 skipped" in r.stdout, r.stdout
    # the bypass: hooks and options outside tests/golden that skip or deselect golden tests
    (root / "tests" / "conftest.py").write_text(
        "import pytest\n\ndef pytest_collection_modifyitems(items):\n"
        "    for it in items:\n        it.add_marker(pytest.mark.skip(reason='neutralised'))\n", encoding="utf-8")
    (root / "conftest.py").write_text("collect_ignore_glob = ['tests/golden/*']\n", encoding="utf-8")
    (root / "pytest.ini").write_text("[pytest]\naddopts = -k nothing_matches\npython_functions = nope_*\n",
                                     encoding="utf-8")
    (root / "pytest.py").write_text("raise SystemExit('shadowed pytest')\n", encoding="utf-8")
    (root / "python" / "sitecustomize.py").write_text("import os\nos._exit(0)\n", encoding="utf-8")
    plain = subprocess.run([sys.executable, "-m", "pytest", "tests/golden", "-q", "-p", "no:cacheprovider"],
                           cwd=root, capture_output=True, text=True, timeout=300)
    assert "2 passed" not in plain.stdout                    # plain pytest is fooled ...
    env = dict(os.environ, PYTHONPATH=str(root / "python"), PYTEST_ADDOPTS="-k nothing_matches")
    hijacked = subprocess.run([sys.executable, str(TOOLS / "check_golden.py"), "--root", str(root), "--run"],
                              env=env, capture_output=True, text=True, timeout=300)
    assert hijacked.returncode == 0 and hijacked.stdout == ""      # sitecustomize ran first: why CI uses -E
    r = subprocess.run([sys.executable, "-E", str(TOOLS / "check_golden.py"), "--root", str(root), "--run"],
                       env=env, capture_output=True, text=True, timeout=300)   # exactly as in ci.yml
    assert r.returncode == 0 and "2 passed" in r.stdout, r.stdout + r.stderr   # ... the isolated run is not


@pytest.mark.parametrize("body,needle", [
    ("import pytest\n\ndef test_a():\n    pytest.skip('not today')\n", "1 skipped"),
    ("import pytest\n\n@pytest.mark.xfail(reason='x')\ndef test_a():\n    assert False\n", "1 skipped"),
    ("def test_a():\n    assert 1 == 2\n", "1 failed"),
    ("import nonexistent_module_xyz\n\ndef test_a():\n    pass\n", "1 errors"),
    ("def helper():\n    pass\n", "nothing ran"),
    ("import os\n\ndef test_a():\n    os._exit(0)\n", "produced no report"),      # exits 0 with no verdict
])
def test_golden_run_fails_on_skips_failures_and_empty_suites(body, needle, tmp_path):
    root = golden_demo_repo(tmp_path, body)
    r = tool("check_golden", "--root", root, "--run")
    assert r.returncode == 1 and "golden suite FAILED" in r.stdout and needle in r.stdout, r.stdout


def test_golden_run_times_out(tmp_path):
    root = golden_demo_repo(tmp_path, "import time\n\ndef test_slow():\n    time.sleep(120)\n")
    r = tool("check_golden", "--root", root, "--run", "--timeout", "5")
    assert r.returncode == 1 and "timed out after 5 s" in r.stdout and "produced no report" in r.stdout


def test_golden_run_refuses_a_broken_lock(tmp_path):
    root = golden_demo_repo(tmp_path)
    (root / "tests" / "golden" / "test_gold.py").write_text(GOLDEN_DEMO + "\n# edit\n", encoding="utf-8")
    r = tool("check_golden", "--root", root, "--run")
    assert r.returncode == 1 and "MODIFIED" in r.stdout and "golden suite" not in r.stdout


# ------------------------------------- golden lock against the base revision (CI: golden.yml)
GUTTED = "def test_answer():\n    pass\n"
WORKFLOWS = ROOT / ".github" / "workflows"
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def relock(root):
    """What anyone can do: the maintainer confirmation of --update is an honour system."""
    r = tool("check_golden", "--root", root, "--update", env={"HEARTH_MAINTAINER": "1"})
    assert r.returncode == 0, r.stderr


def test_golden_base_comparison(tmp_path):
    root = golden_demo_repo(tmp_path)
    g = root / "tests" / "golden"
    manifest = g / "MANIFEST.sha256"
    base_bytes = manifest.read_bytes()
    base = tmp_path / "base-MANIFEST.sha256"
    base.write_bytes(base_bytes)

    def against_base(*extra):
        return tool("check_golden", "--root", root, "--base-manifest", base, *extra)

    def reset():
        for p in g.glob("test_*.py"):
            p.unlink()
        (g / "test_gold.py").write_text(GOLDEN_DEMO, encoding="utf-8")
        manifest.write_bytes(base_bytes)

    r = against_base()
    assert r.returncode == 0 and "golden lock matches the base revision (1 file(s)" in r.stdout, r.stdout
    manifest.write_bytes(base_bytes.replace(b"\n", b"\r\n"))                  # a CRLF checkout is no change
    assert against_base().returncode == 0
    reset()
    (g / "test_gold.py").write_text(GUTTED, encoding="utf-8")
    relock(root)
    assert tool("check_golden", "--root", root).returncode == 0              # the change's own lock agrees
    r = against_base()
    assert r.returncode == 1, r.stdout
    assert "RELOCKED  tests/golden/MANIFEST.sha256 differs from the base revision's" in r.stdout
    assert "MODIFIED  tests/golden/test_gold.py  (against the base revision's lock)" in r.stdout
    assert "(1 modified, 0 removed, 0 added, manifest changed; base: " in r.stdout and "golden-reviewed" in r.stdout
    r = against_base("--acknowledged")
    assert r.returncode == 0 and "acknowledged by a maintainer" in r.stdout and "golden lock OK" in r.stdout
    r = against_base("--run")
    assert r.returncode == 1 and "golden suite" not in r.stdout             # never runs the re-locked suite
    r = against_base("--run", "--acknowledged")
    assert r.returncode == 0 and "golden suite OK: 1 passed" in r.stdout, r.stdout   # the change's own lock
    reset()
    manifest.write_bytes(base_bytes + b"# reviewed\n")                        # the manifest alone
    r = against_base()
    assert r.returncode == 1 and "RELOCKED" in r.stdout and "(0 modified, 0 removed, 0 added, manifest changed" \
        in r.stdout, r.stdout
    reset()
    (g / "test_new.py").write_text("def test_new():\n    pass\n", encoding="utf-8")
    relock(root)
    r = against_base()
    assert r.returncode == 1 and "ADDED     tests/golden/test_new.py" in r.stdout, r.stdout
    reset()
    (g / "test_gold.py").unlink()
    (g / "test_other.py").write_text(GOLDEN_DEMO, encoding="utf-8")
    relock(root)
    r = against_base()
    assert r.returncode == 1 and "REMOVED   tests/golden/test_gold.py" in r.stdout and "ADDED" in r.stdout
    reset()
    (g / "test_gold.py").write_text(GUTTED, encoding="utf-8")                # not re-locked: the own lock fails
    r = against_base()
    assert r.returncode == 1 and "golden lock BROKEN" in r.stdout and "against the base" not in r.stdout
    reset()
    assert against_base("-q").stdout == ""


def test_golden_compare_base_and_arguments(tmp_path):
    root = golden_demo_repo(tmp_path)
    g = root / "tests" / "golden"
    base_text = (g / "MANIFEST.sha256").read_text(encoding="utf-8")
    res = cg.check(root)
    same = {"manifest_changed": False, "modified": [], "removed": [], "added": []}
    assert cg.compare_base(root, res, base_text) == same and not cg.changed(same)
    for k in same:
        assert cg.changed(dict(same, **{k: True if k == "manifest_changed" else ["x"]})), k
    assert cg.compare_base(root, res, None) == dict(same, manifest_changed=True, added=["tests/golden/test_gold.py"])
    other = "0" * 64 + "  tests/golden/test_gold.py\n" + "1" * 64 + "  tests/golden/gone.py\n"
    assert cg.compare_base(root, res, other) == {"manifest_changed": True, "modified": ["tests/golden/test_gold.py"],
                                                 "removed": ["tests/golden/gone.py"], "added": []}
    (g / "MANIFEST.sha256").unlink()
    assert cg.compare_base(root, res, base_text) == dict(same, manifest_changed=True)
    bad = tmp_path / "bad.sha256"
    bad.write_text("not a manifest\n", encoding="utf-8")
    binary = tmp_path / "binary.sha256"
    binary.write_bytes(b"\xff\xfe")
    for args, needle in ((["--acknowledged"], "--acknowledged needs"),
                         (["--base-manifest", tmp_path / "none"], "base manifest"),
                         (["--base-manifest", bad], "base manifest"), (["--base-manifest", binary], "base manifest")):
        r = tool("check_golden", "--root", root, *args)
        assert r.returncode == 2 and needle in r.stderr, (args, r.stderr)
    r = tool("check_golden", "--root", root, "--update", "--i-am-a-maintainer", "--base-manifest", bad)
    assert r.returncode == 2 and "does not take a base" in r.stderr
    r = tool("check_golden", "--root", root, "--base-manifest", bad, "--base-rev", "HEAD")
    assert r.returncode == 2 and "not allowed with" in r.stderr
    nogit = tmp_path / "nogit"
    nogit.mkdir()
    with pytest.raises(cg.ManifestError):
        cg.base_manifest_at(nogit, "HEAD")
    for rev in ("-x", ""):
        with pytest.raises(cg.ManifestError, match="is not a revision"):
            cg.base_manifest_at(root, rev)


def export_rev(repo: Path, rev: str, dst: Path) -> Path:
    """The tree of a revision, as a CI checkout would have it."""
    import io
    import tarfile
    r = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", rev], capture_output=True, timeout=120)
    assert r.returncode == 0, r.stderr
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tf:
        tf.extractall(dst, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    return dst


def git_golden_repo(tmp_path) -> Path:
    """The demo golden suite, its lock and governance/tools/{check_golden,path_aliases}.py, committed
    and tagged 'base'."""
    root = golden_demo_repo(tmp_path)
    (root / "governance" / "tools").mkdir(parents=True)
    for name in ("check_golden.py", "path_aliases.py"):
        shutil.copyfile(TOOLS / name, root / "governance" / "tools" / name)
    for args in (("init", "-q"), ("add", "-A"), ("commit", "-q", "-m", "base"), ("tag", "base"),
                 ("checkout", "-q", "-b", "pr")):
        r = _git(*args, cwd=root)
        assert r.returncode == 0, (args, r.stderr)
    return root


def workflow_text(name: str, code_only: bool = False) -> str:
    text = (WORKFLOWS / name).read_text(encoding="utf-8")
    if not code_only:
        return text
    return "\n".join(line.split(" #")[0] for line in text.splitlines() if not line.lstrip().startswith("#"))


def workflow_env(name: str, step: str) -> dict:
    """The env: block of the step named `step`."""
    lines = workflow_text(name).splitlines()
    i = next(k for k, line in enumerate(lines) if line.strip() == f"- name: {step}")
    out = {}
    for line in lines[i + 1:]:
        if line.strip().startswith(("- ", "run:")):
            break
        m = re.fullmatch(r"\s+([A-Z_]+): (.+)", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def workflow_run(name: str, step: str) -> str:
    """The command of the step named `step`, joined into one line as YAML folds `run: >`."""
    lines = workflow_text(name).splitlines()
    i = next(k for k, line in enumerate(lines) if line.strip() == f"- name: {step}")
    j = next(k for k in range(i + 1, len(lines)) if lines[k].strip().startswith("run:"))
    head = lines[j].split("run:", 1)[1].strip()
    if head not in (">", "|"):
        return head
    indent = len(lines[j]) - len(lines[j].lstrip())
    body = []
    for line in lines[j + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line.strip())
    return " ".join(x for x in body if x)


def run_workflow_command(cmd: str, workspace: Path, ack: bool, head: str = "", url: str = "", pr: str = ""):
    """A workflow's `python ...` command line, run where the workflow's checkouts would be."""
    for var, value in (("$ACK", "--acknowledged" if ack else ""), ('"$HEAD_SHA"', head), ('"$REPO_URL"', url),
                       ('"$PR_NUMBER"', pr)):
        cmd = cmd.replace(var, shlex.quote(value) if value else value)
    argv = shlex.split(cmd)
    assert argv[0] == "python", cmd
    return subprocess.run([sys.executable, *argv[1:]], cwd=workspace, capture_output=True, text=True, timeout=300)


def git_rev(repo: Path, rev: str = "HEAD") -> str:
    r = _git("rev-parse", "--verify", f"{rev}^{{commit}}", cwd=repo)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def git_commit(repo: Path, msg: str, *paths) -> str:
    """Commits the given paths (everything with none) and returns the commit id."""
    r = _git("add", *(paths or ("-A",)), cwd=repo)
    assert r.returncode == 0, r.stderr
    r = _git("commit", "-q", "-m", msg, cwd=repo)
    assert r.returncode == 0, r.stderr
    return git_rev(repo)


def git_special_entry(repo: Path, path: str, target: str, mode: str = "120000") -> None:
    """Stages a symbolic link to `target` (mode 120000) or a submodule at commit `target`
    (mode 160000) at `path`, in the index only: no privilege needed for links on Windows."""
    oid = target
    if mode == "120000":
        oid = subprocess.run(["git", "-C", str(repo), "hash-object", "-w", "--stdin"], input=target,
                             capture_output=True, text=True, timeout=60).stdout.strip()
    r = _git("update-index", "--add", "--cacheinfo", f"{mode},{oid},{path}", cwd=repo)
    assert r.returncode == 0, r.stderr


def clone_at(repo: Path, rev: str, dst: Path, depth: int = 0) -> Path:
    """A checkout as actions/checkout leaves one: its own .git, HEAD detached at rev
    (depth 1: a shallow fetch, as of a ref; rev must then be a branch tip)."""
    sha = git_rev(repo, rev)
    src = repo.resolve().as_uri() if depth else str(repo)
    extra = ["--depth", str(depth), "--no-single-branch"] if depth else []
    r = _git("clone", "-q", "--no-checkout", *extra, src, str(dst), cwd=repo.parent)
    assert r.returncode == 0, r.stderr
    r = _git("checkout", "-q", "--detach", sha, cwd=dst)
    assert r.returncode == 0, r.stderr
    return dst


def rmtree_force(path: Path) -> None:
    """shutil.rmtree that also removes read-only files (git's objects on Windows)."""
    def retry(func, p, _exc):
        os.chmod(p, 0o700)
        func(p)
    shutil.rmtree(path, **({"onexc": retry} if sys.version_info >= (3, 12) else {"onerror": retry}))


def _rmlink(p: Path) -> None:
    if os.path.isdir(p) and not os.path.islink(p):
        os.rmdir(p)                       # a junction
    else:
        os.unlink(p)


def dir_link_as_checked_out(link: Path, target: Path) -> bool:
    """A committed link to a directory: git's own checkout of it where that resolves (POSIX), else
    (Windows without symlink rights checks out a plain file) a symlink or junction made here."""
    if os.path.isdir(link):
        return True
    if os.path.lexists(link):
        _rmlink(link)
    return make_dir_link(link, target)


@needs_git
def test_golden_change_that_relocks_itself_fails_against_the_base(tmp_path):
    """The council's round-1 reproduction, as a pull request against a base branch."""
    root = git_golden_repo(tmp_path)
    (root / "tests" / "golden" / "test_gold.py").write_text(GUTTED, encoding="utf-8")
    relock(root)
    weakened = "import sys\nprint('golden lock OK')\nsys.exit(0)\n"          # the change weakens its own tool too
    (root / "governance" / "tools" / "check_golden.py").write_text(weakened, encoding="utf-8")
    assert _git("commit", "-q", "-am", "gut the golden test and re-lock it", cwd=root).returncode == 0
    # the hole: the change is consistent with its own lock and the gutted suite passes
    assert tool("check_golden", "--root", root).returncode == 0
    r = tool("check_golden", "--root", root, "--run")
    assert r.returncode == 0 and "golden suite OK: 1 passed" in r.stdout, r.stdout
    # against the base revision's lock it fails, before anything runs
    r = tool("check_golden", "--root", root, "--base-rev", "base", "--run")
    assert r.returncode == 1 and "RELOCKED" in r.stdout and "MODIFIED  tests/golden/test_gold.py" in r.stdout
    assert "base: tests/golden/MANIFEST.sha256 at base" in r.stdout and "golden suite" not in r.stdout
    r = tool("check_golden", "--root", root, "--base-rev", "base", "--acknowledged")
    assert r.returncode == 0 and "acknowledged by a maintainer" in r.stdout
    assert tool("check_golden", "--root", root, "--base-rev", "pr").returncode == 0      # same revision
    # golden.yml, as GitHub would run it: tool and lock from the base, the change as git objects only
    ws = tmp_path / "ws"
    clone_at(root, "base", ws / "base")
    clone_at(root, "pr", ws / "pr")
    gate = workflow_run("golden.yml", "Golden lock against the base revision")
    r = run_workflow_command(gate, ws, ack=False)
    assert r.returncode == 1 and "golden tests CHANGED against the base" in r.stdout, r.stdout + r.stderr
    assert run_workflow_command(gate, ws, ack=True).returncode == 0
    own = subprocess.run([sys.executable, "-E", str(ws / "pr" / "governance" / "tools" / "check_golden.py")],
                         capture_output=True, text=True, timeout=60)
    assert own.returncode == 0                                # why the gate never uses the change's tool
    # a change that leaves tests/golden alone passes the gate
    _git("checkout", "-q", "-b", "harmless", "base", cwd=root)
    (root / "README.txt").write_text("docs only\n", encoding="utf-8")
    git_commit(root, "docs")
    rmtree_force(ws / "pr")
    clone_at(root, "harmless", ws / "pr")
    r = run_workflow_command(gate, ws, ack=False)
    assert r.returncode == 0 and "golden lock matches the base revision" in r.stdout, r.stdout + r.stderr
    assert "change: " in r.stdout and " at HEAD (" in r.stdout


@needs_git
def test_golden_gate_never_follows_a_linked_tests_directory(tmp_path):
    """Verifier round 2: the change replaces tests/ with an absolute link to $WS/base/tests and
    carries its gutted, re-locked suite in base/tests/. Both workflows' layouts resolve the link
    to a lock that matches: the base checkout (golden.yml) or the change's own copy (ci.yml)."""
    root = git_golden_repo(tmp_path)
    ws = tmp_path / "ws"
    _git("checkout", "-q", "-b", "linked", "base", cwd=root)
    own = root / "base" / "tests" / "golden"
    own.mkdir(parents=True)
    (own / "__init__.py").write_text("", encoding="utf-8")
    (own / "test_gold.py").write_text(GUTTED, encoding="utf-8")
    relock(root / "base")
    assert _git("rm", "-rq", "tests", cwd=root).returncode == 0
    git_special_entry(root, "tests", str(ws / "base" / "tests"))
    linked = git_commit(root, "move tests", "base")
    assert _git("ls-tree", linked, "tests", cwd=root).stdout.startswith("120000 blob")
    # golden.yml: base/ is the base branch, pr/ the change; pr/tests leads to base/tests
    clone_at(root, "base", ws / "base")
    clone_at(root, "linked", ws / "pr")
    if not dir_link_as_checked_out(ws / "pr" / "tests", ws / "base" / "tests"):
        pytest.skip("cannot create a symbolic link or junction here")
    assert (ws / "pr" / "tests" / "golden" / "MANIFEST.sha256").is_file()           # the link works ...
    r = tool("check_golden", "--root", ws / "pr", "--base-manifest", ws / "base" / "tests" / "golden" / "MANIFEST.sha256")
    assert r.returncode == 1 and "LINK      tests " in r.stdout                     # ... and is refused
    gate = workflow_run("golden.yml", "Golden lock against the base revision")
    for ack in (False, True):                                    # no label makes a link acceptable
        r = run_workflow_command(gate, ws, ack=ack)
        assert r.returncode == 1 and "LINK      tests  (symbolic links" in r.stdout, r.stdout + r.stderr
        assert "golden lock OK" not in r.stdout and "matches the base" not in r.stdout
    # ci.yml: the change is checked out at the workspace root, so the link leads to its own base/tests
    _rmlink(ws / "pr" / "tests")
    rmtree_force(ws)
    clone_at(root, "linked", ws)
    assert dir_link_as_checked_out(ws / "tests", ws / "base" / "tests")
    assert (ws / "tests" / "golden" / "test_gold.py").read_text(encoding="utf-8") == GUTTED
    shutil.copytree(TOOLS, ws / ".trusted" / "governance" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    assert cg.passed(cg.check(ws / "base"))                      # the change's own copy is consistent
    for step in ("Golden-test lock (INV-VERIFY)", "Golden tests, isolated (INV-VERIFY)"):
        r = run_workflow_command(workflow_run("ci.yml", step), ws, ack=False)
        assert r.returncode == 1 and "LINK      tests  (symbolic links" in r.stdout, (step, r.stdout, r.stderr)
        assert "golden suite" not in r.stdout
    _rmlink(ws / "tests")


@needs_git
def test_golden_rev_reads_git_objects_only(tmp_path):
    root = git_golden_repo(tmp_path)
    head = git_rev(root)
    g = "tests/golden"
    base_manifest = root / g / "MANIFEST.sha256"
    base_text = base_manifest.read_text(encoding="utf-8")
    r = tool("check_golden", "--root", root, "--rev", "HEAD", "--base-manifest", base_manifest)
    assert r.returncode == 0 and f" at HEAD ({head[:12]})" in r.stdout, r.stdout
    # the working tree is not read: gutting it changes nothing, and the commit is what is judged
    (root / g / "test_gold.py").write_text(GUTTED, encoding="utf-8")
    shutil.rmtree(root / "governance")
    assert tool("check_golden", "--root", root).returncode == 1
    assert tool("check_golden", "--root", root, "--rev", "HEAD").returncode == 0
    _git("checkout", "-q", "-f", "--", ".", cwd=root)
    res = cg.check(cg.Revision(root, "HEAD"))
    assert cg.passed(res) and res["ok"] == [f"{g}/test_gold.py"] and res["inits"] == [f"{g}/__init__.py"]

    def case():
        assert _git("checkout", "-q", "-f", "-B", "case", "base", cwd=root).returncode == 0

    def judged(rev):
        src = cg.Revision(root, rev)
        res = cg.check(src)
        return res, (cg.compare_base(src, res, base_text) if cg.passed(res) else None)

    # links and submodules in the tree, at every level (index-only entries: no privilege needed)
    for path in ("tests", g, f"{g}/more", f"{g}/MANIFEST.sha256"):
        for mode in ("120000", "160000"):
            case()
            if path != f"{g}/more":
                assert _git("rm", "-rq", "--cached", path, cwd=root).returncode == 0
            git_special_entry(root, path, head if mode == "160000" else "../elsewhere", mode)
            assert _git("commit", "-q", "-m", f"{path} {mode}", cwd=root).returncode == 0
            res, _ = judged("HEAD")
            assert res["links"] == [path] and not cg.passed(res), (path, mode, res)
            assert res["ok"] == [] or path == f"{g}/more", (path, mode, res)
    r = tool("check_golden", "--root", root, "--rev", "HEAD")       # the manifest itself is a submodule
    assert r.returncode == 2 and f"LINK      {g}/MANIFEST.sha256" in r.stdout and "not found" in r.stdout
    # a file where the directory belongs: there is no golden suite
    case()
    _git("rm", "-rq", "tests", cwd=root)
    (root / "tests").write_text("not a directory\n", encoding="utf-8")
    res, _ = judged(git_commit(root, "tests is a file"))
    assert res["links"] == [] and res["errors"] and "not found" in res["errors"][0]
    assert tool("check_golden", "--root", root, "--rev", "HEAD").returncode == 2
    # committed bytecode, __init__ files, extra and modified files
    case()
    planted = plant_pyc(tmp_path, NEUTRALISE, root / g / "__pycache__", "x")
    (root / g / "__pycache__" / "plain.txt").write_text("abcdefgh\n", encoding="utf-8")   # byte 4 looks unchecked
    (root / g / "sub").mkdir()
    (root / g / "sub" / "__init__.py").write_text("import os\n", encoding="utf-8")
    (root / g / "__init__.py").write_text("\n", encoding="utf-8")                    # whitespace only: exempt
    _git("add", "-f", "-A", cwd=root)                                               # past any global gitignore
    res, _ = judged(git_commit(root, "extras"))
    assert res["bytecode"] == [f"{g}/__pycache__/{planted.name}"] and res["unlisted"] == [f"{g}/sub/__init__.py"]
    assert res["inits"] == [f"{g}/__init__.py"] and res["links"] == [] and not cg.passed(res)
    case()
    (root / g / "test_gold.py").write_text(GUTTED, encoding="utf-8")
    relock(root)
    res, delta = judged(git_commit(root, "relocked"))
    assert cg.passed(res) and delta["manifest_changed"] and delta["modified"] == [f"{g}/test_gold.py"]
    r = tool("check_golden", "--root", root, "--rev", "HEAD", "--base-rev", "base")
    assert r.returncode == 1 and "RELOCKED" in r.stdout
    # arguments
    for args, needle in ((["--rev", "HEAD", "--run"], "--rev reads git objects"),
                         (["--rev", "HEAD", "--update", "--i-am-a-maintainer"], "--rev reads git objects"),
                         (["--rev", "no-such-rev"], "is not a commit"), (["--rev=-x"], "is not a revision")):
        r = tool("check_golden", "--root", root, *args)
        assert r.returncode == 2 and needle in r.stderr, (args, r.stderr)
    r = tool("check_golden", "--root", root / "tests", "--rev", "HEAD")          # would read the parent repo
    assert r.returncode == 2 and "is not the top level of a git checkout" in r.stderr
    nogit = tmp_path / "nogit"
    nogit.mkdir()
    with pytest.raises(cg.ManifestError, match="is not the top level"):
        cg.Revision(nogit, "HEAD")


@needs_git
def test_golden_base_rev_edge_cases(tmp_path):
    root = git_golden_repo(tmp_path)
    r = tool("check_golden", "--root", root, "--base-rev", "no-such-rev")
    assert r.returncode == 2 and "is not a commit" in r.stderr
    r = tool("check_golden", "--root", root, "--base-rev", "--version")
    assert r.returncode == 2
    blob = _git("rev-parse", "base:tests/golden/test_gold.py", cwd=root).stdout.strip()
    r = tool("check_golden", "--root", root, "--base-rev", blob)                       # a blob is not a commit
    assert r.returncode == 2 and "is not a commit" in r.stderr
    _git("checkout", "-q", "--orphan", "empty", cwd=root)                               # a base without golden tests
    _git("rm", "-rq", "--cached", "tests", cwd=root)
    assert _git("commit", "-q", "--allow-empty", "-m", "nothing locked", cwd=root).returncode == 0
    assert cg.base_manifest_at(root, "empty") is None
    assert [rel for _, rel in cg.parse_manifest(cg.base_manifest_at(root, "base"))] == ["tests/golden/test_gold.py"]
    r = tool("check_golden", "--root", root, "--base-rev", "empty")
    assert r.returncode == 1 and "NO BASE" in r.stdout and "ADDED     tests/golden/test_gold.py" in r.stdout


MERGE_STEP = "Merge commit of this event's head into this base (git objects only)"


def test_golden_and_decision_gates_never_execute_the_pull_request():
    for name, tool_name in (("golden.yml", "check_golden.py"), ("decisions.yml", "adr.py")):
        text, code = workflow_text(name), workflow_text(name, code_only=True)
        # verifier round 3: pull_request_target runs the default branch's copy with github.sha at its head,
        # whatever the base, so the gates judge pull requests into main (the default branch) only
        assert re.search(r"^on:\n  pull_request_target:\n    branches: \[main\]\n    types: \[opened, synchronize, "
                         r"reopened, labeled, unlabeled\]\n", text, re.M), name
        assert re.search(r"^permissions:\n  contents: read\n", text, re.M), name
        assert "the head of main, the default branch" in text and "base branch head" not in text, name
        # the base is the only checkout: actions/checkout refuses fork pull requests here (round 3), and the
        # gates never write the pull request to disk
        assert code.count("uses: actions/checkout@v4") == code.count("persist-credentials: false") == 1, name
        assert [r.strip() for r in re.findall(r"^\s+ref: (.*)$", code, re.M)] == ["${{ github.sha }}"], name
        assert "refs/pull" not in code and "allow-unsafe-pr-checkout" not in code, name
        assert code.count("fetch-depth: 0") == 1, name
        runs = [line.split("run:", 1)[1].strip() for line in code.splitlines() if line.strip().startswith("run:")]
        assert len(runs) == 2, name
        for banned in ("pip", "pytest", "--run", "cmake", "secrets", "npm", "make ", "cache", "PYTHONPATH"):
            assert banned not in code, (name, banned)
        assert workflow_run(name, MERGE_STEP) == ('python -I -S base/governance/tools/merge_ref.py --fetch "$REPO_URL" '
                                                  '--pr "$PR_NUMBER" --merge pr --head "$HEAD_SHA" --base base'), name
        assert workflow_env(name, MERGE_STEP) == {
            "REPO_URL": "${{ github.server_url }}/${{ github.repository }}",
            "PR_NUMBER": "${{ github.event.pull_request.number }}",
            "HEAD_SHA": "${{ github.event.pull_request.head.sha }}", "GITHUB_TOKEN": "${{ github.token }}"}, name
        assert code.count("github.token") == 1, name                           # the fetch alone gets the token
        assert code.index("merge_ref.py") < code.index(tool_name), name        # nothing is judged before it
    gate = workflow_run("golden.yml", "Golden lock against the base revision")
    assert gate == ("python -I -S base/governance/tools/check_golden.py --root pr --rev HEAD "
                    "--base-manifest base/tests/golden/MANIFEST.sha256 $ACK")
    assert "contains(github.event.pull_request.labels.*.name, 'golden-reviewed') && '--acknowledged'" \
        in workflow_text("golden.yml")
    assert workflow_run("decisions.yml", "Decision circuit breaker") == \
        "python -I -S base/governance/tools/adr.py breaker --root pr --rev HEAD --base-root base $ACK"
    assert "contains(github.event.pull_request.labels.*.name, 'decisions-reviewed') && '--acknowledged'" \
        in workflow_text("decisions.yml")



def test_ci_runs_governance_tools_from_the_base_revision():
    code = workflow_text("ci.yml", code_only=True)
    for m in re.finditer(r"\S*governance/tools/\w+\.py", code):              # never the change's own tools ...
        assert m.group().startswith(".trusted/governance/tools/"), m.group()
    calls = re.findall(r"python -I -S \.trusted/governance/tools/(\w+)\.py", code)   # ... and isolated, no site
    assert sorted(calls) == ["adr", "check_golden", "check_golden", "check_golden", "check_golden",
                             "validate_handover", "waves"], calls
    assert "GOV_TOOLS" not in code and "t=governance/tools" not in code       # no fallback to the change's copy
    assert re.findall(r"python (?:-\w+ )*-m pip", code) == re.findall(r"python -I -m pip", code) != []
    jobs = re.split(r"\n  (?=[a-z][a-z0-9-]*:\n)", code.split("\njobs:\n", 1)[1])
    trusted = [j for j in jobs if "path: .trusted" in j]
    assert len(trusted) == 4
    for job in trusted:          # whatever the change put at .trusted is gone before the base is checked out
        assert job.index("run: rm -rf .trusted") < job.index("path: .trusted") < job.index("pip install")
    gov = next(j for j in jobs if j.lstrip().startswith("governance:"))
    first_change_code = gov.index("pip install")                  # nothing from the change has run before it
    for step in ("Golden-test lock (INV-VERIFY)", "Task DAG", "Handover manifests", "Decision records"):
        assert gov.index(f"- name: {step}") < first_change_code, step
        cmd = workflow_run("ci.yml", step)
        assert cmd.startswith("python -I -S .trusted/governance/tools/"), cmd
        cmd = cmd.replace(".trusted/governance/tools", shlex.quote(TOOLS.as_posix()))
        r = run_workflow_command(cmd, ROOT, ack=False)
        assert r.returncode == 0, (step, r.stdout[-2000:], r.stderr[-2000:])
    for job in trusted[1:]:      # build jobs: the golden run comes last; the change's code ran before it
        assert job.index("pytest tests/py") < job.index("check_golden.py --root . --run")



def write_decisions(root: Path, files: dict, inv: str = None) -> None:
    inv = INV_BASE if inv is None else inv
    d = root / "governance" / "decisions"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for n, text in files.items():
        (d / n).write_text(text, encoding="utf-8")
    (root / "governance" / "INVARIANTS.md").write_text(inv, encoding="utf-8")


@needs_git
def test_decisions_gate_as_the_workflow_runs_it(tmp_path):
    base = adr_files(8)
    retired = {n: adr_text(i, status="Deprecated") for i, n in enumerate(sorted(base), 1)}
    shadowed = dict(retired, **{f"ADR-{i:04d}-zz-shadow.md": adr_text(i) for i in range(1, 9)})
    repo, ws = tmp_path / "repo", tmp_path / "ws"
    write_decisions(repo, base)
    assert _git("init", "-q", cwd=repo).returncode == 0
    git_commit(repo, "base")
    _git("tag", "base", cwd=repo)
    clone_at(repo, "base", ws / "base")                          # the trusted side: tools, decisions, invariants
    shutil.copytree(TOOLS, ws / "base" / "governance" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    cmd = workflow_run("decisions.yml", "Decision circuit breaker")

    def change(build):
        _git("checkout", "-q", "-f", "-B", "change", "base", cwd=repo)
        build()
        assert _git("commit", "-q", "-m", "change", cwd=repo).returncode == 0
        if (ws / "pr").exists():
            rmtree_force(ws / "pr")
        clone_at(repo, "change", ws / "pr")

    change(lambda: (write_decisions(repo, shadowed), _git("add", "-A", cwd=repo)))
    r = run_workflow_command(cmd, ws, ack=False)
    assert r.returncode == 1 and "duplicate ADR id(s)" in r.stdout and "8 of 8" in r.stdout, r.stdout + r.stderr
    assert run_workflow_command(cmd, ws, ack=True).returncode == 0
    write_decisions(ws / "pr", base)                             # the working tree is not what is judged
    assert run_workflow_command(cmd, ws, ack=False).returncode == 1
    # verifier round 2: governance/ becomes an absolute link to $WS/base/governance, and the change's
    # real records (every decision deprecated, the invariants gutted) live in base/governance/
    gutted = "# Invariants\n\nNone.\n"

    def linked():
        assert _git("rm", "-rq", "governance", cwd=repo).returncode == 0
        write_decisions(repo / "base", retired, gutted)
        _git("add", "base", cwd=repo)
        git_special_entry(repo, "governance", str(ws / "base" / "governance"))

    change(linked)
    if not dir_link_as_checked_out(ws / "pr" / "governance", ws / "base" / "governance"):
        pytest.skip("cannot create a symbolic link or junction here")
    assert (ws / "pr" / "governance" / "decisions" / "ADR-0001-d1.md").read_text(encoding="utf-8") == base[
        "ADR-0001-d1.md"]                                         # through the link: the base's records
    for ack in (False, True):
        r = run_workflow_command(cmd, ws, ack=ack)
        assert r.returncode == 1 and "governance is not a directory at HEAD (mode 120000)" in r.stdout, r.stdout
        assert "decision breaker OK" not in r.stdout
        r = tool("adr", "breaker", "--root", ws / "pr", "--base-root", ws / "base", *(["--acknowledged"] * ack))
        assert r.returncode == 1 and "governance is a symbolic link or junction" in r.stdout, r.stdout
    control = tmp_path / "control"
    write_decisions(control, retired, gutted)                    # the same content as plain files
    r = tool("adr", "breaker", "--root", control, "--base-root", ws / "base")
    assert r.returncode == 1 and "8 of 8" in r.stdout and "INVARIANTS.md changed" in r.stdout
    _rmlink(ws / "pr" / "governance")



def test_codeowners_cover_the_verification_paths():
    rules = {}
    for line in (ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path, *owners = line.split()
        assert owners and all(re.fullmatch(r"@[A-Za-z0-9-]+(/[A-Za-z0-9_.-]+)?", o) for o in owners), line
        assert path.startswith("/") and path not in rules, line
        rules[path] = owners
    for path in ("/tests/golden/", "/governance/INVARIANTS.md", "/governance/tools/", "/.github/",
                 "/governance/schemas/", "/governance/decisions/"):
        assert path in rules, path
    # verifier round 3: the modules tests/golden takes its expected values from are code-owned, as is
    # everything they import from the package (the engine binding is the code under test)
    for module in golden_oracle_modules():
        assert f"/python/hearth/{module}.py" in rules, module


def _hearth_imports(path: Path) -> set:
    """Modules of the hearth package that a file imports (absolute or relative, at any depth)."""
    import ast
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if (node.level == 1 and not mod) or (node.level == 0 and mod == "hearth"):   # from . / hearth import x
                out.update(a.name for a in node.names)
            elif node.level == 1:                                                       # from .x import y
                out.add(mod.split(".")[0])
            elif node.level == 0 and mod.startswith("hearth."):                         # from hearth.x import y
                out.add(mod.split(".")[1])
        elif isinstance(node, ast.Import):
            out.update(a.name.split(".")[1] for a in node.names if a.name.startswith("hearth."))
    return out


def golden_oracle_modules() -> list:
    """What the golden suite's expectations depend on: what tests/golden imports from hearth, minus the
    code under test (engine, _native), closed over the package's own imports, plus __init__."""
    under_test = {"engine", "_native"}
    todo = set().union(*(_hearth_imports(f) for f in (ROOT / "tests" / "golden").glob("*.py"))) - under_test
    seen = {"__init__"}
    while todo:
        mod = todo.pop()
        seen.add(mod)
        todo |= _hearth_imports(ROOT / "python" / "hearth" / f"{mod}.py") - under_test - seen
    return sorted(seen)


def merge_ref_repo(tmp_path) -> tuple:
    """b0 -> b1 -> b2 on 'trunk'; 'topic' (head h) and 'side' branch off b0; 'merge' = b1 + h,
    'octo' = b1 + h + side."""
    o = tmp_path / "origin"
    o.mkdir()
    assert _git("init", "-q", cwd=o).returncode == 0
    ids = {}

    def commit(name):
        (o / f"{name}.txt").write_text(name + "\n", encoding="utf-8")
        ids[name] = git_commit(o, name)

    commit("b0")
    _git("checkout", "-q", "-b", "topic", cwd=o)
    commit("h")
    _git("checkout", "-q", "-b", "side", ids["b0"], cwd=o)
    commit("s")
    _git("checkout", "-q", "-b", "trunk", ids["b0"], cwd=o)
    commit("b1")
    for branch, heads in (("merge", ["topic"]), ("octo", ["topic", "side"])):
        _git("checkout", "-q", "-b", branch, ids["b1"], cwd=o)
        assert _git("merge", "-q", "--no-ff", "-m", branch, *heads, cwd=o).returncode == 0
        ids[branch] = git_rev(o)
    _git("checkout", "-q", "trunk", cwd=o)
    commit("b2")
    return o, ids


@needs_git
def test_merge_ref_must_be_this_events_head_merged_into_the_base(tmp_path):
    """Verifier round 2: refs/pull/N/merge is updated asynchronously, so a run for a new head
    could judge the previous merge commit."""
    o, ids = merge_ref_repo(tmp_path)
    ws = tmp_path / "ws"
    pr = clone_at(o, "merge", ws / "pr", depth=1)               # as actions/checkout fetches the merge ref
    assert _git("rev-parse", "--verify", "-q", "HEAD^2", cwd=pr).returncode != 0   # shallow: no parents here
    bases = {}

    def check(base_rev, head, depth=0, merge=pr, *extra):
        key = (base_rev, depth)
        if key not in bases:
            bases[key] = clone_at(o, base_rev, ws / f"base{len(bases)}", depth=depth)
        return tool("merge_ref", "--merge", merge, "--head", head, "--base", bases[key], *extra)

    m, b1, h = ids["merge"], ids["b1"], ids["h"]
    r = check(b1, h)
    assert r.returncode == 0 and r.stdout.strip() == f"merge ref OK: {m[:12]} = base {b1[:12]} + head {h[:12]}"
    r = check(ids["b2"], h.upper())                              # the base moved on since: still this merge
    assert r.returncode == 0 and f"(an ancestor of base {ids['b2'][:12]})" in r.stdout, r.stdout
    r = check(b1, ids["s"])                                      # a newer head was pushed: stale merge ref
    assert r.returncode == 1 and f"second parent {h[:12]} is not the head commit {ids['s'][:12]}" in r.stdout
    assert "merge ref REJECTED" in r.stdout and "push again" in r.stdout
    r = check(ids["b0"], h)                                      # merged into a newer base than the lock's
    assert r.returncode == 1 and f"first parent {b1[:12]} is not the base {ids['b0'][:12]}" in r.stdout
    r = check("trunk", h, 1)                                     # a shallow base cannot vouch for ancestry
    assert r.returncode == 1 and "unknown to the base checkout" in r.stdout and "fetch-depth: 0" in r.stdout
    full = clone_at(o, "merge", ws / "full")
    for rev, n in ((ids["h"], 1), (ids["octo"], 3), (ids["b0"], 0)):
        r = check(b1, h, 0, full, "--rev", rev)
        assert r.returncode == 1 and f"has {n} parent(s), not 2" in r.stdout, (n, r.stdout)
    res = merge_ref.verify(full, h, bases[(b1, 0)], ids["octo"])
    assert res["parents"] == [b1, h, ids["s"]] and len(res["problems"]) == 1
    tree = _git("rev-parse", "HEAD^{tree}", cwd=full).stdout.strip()
    raw = f"tree {tree}\nparent --output=x\nparent {h}\nauthor a <a@b> 0 +0000\ncommitter a <a@b> 0 +0000\n\nx\n"
    bogus = subprocess.run(["git", "-C", str(full), "hash-object", "-t", "commit", "--literally", "-w", "--stdin"],
                           input=raw, capture_output=True, text=True, timeout=60).stdout.strip()
    r = check(b1, h, 0, full, "--rev", bogus)                    # git itself refuses to read it as a commit ...
    assert r.returncode == 2 and "is not a commit" in r.stderr, (r.stdout, r.stderr)
    real_git = merge_ref._git                                    # ... and parent ids are checked before use as arguments
    merge_ref._git = lambda repo, *args: subprocess.CompletedProcess(args, 0, raw.encode(), b"")
    try:
        with pytest.raises(merge_ref.GitError, match="malformed parent line '--output=x'"):
            merge_ref.parents(full, bogus)
    finally:
        merge_ref._git = real_git
    assert merge_ref.parents(full, ids["merge"]) == [b1, h]
    _git("checkout", "-q", "--detach", b1, cwd=full)             # a message line "parent X" is not a parent
    assert _git("merge", "-q", "--no-ff", "-m", f"m\n\nparent {ids['s']}\nparent {ids['b0']}", h, cwd=full).returncode == 0
    assert merge_ref.parents(full, git_rev(full)) == [b1, h]
    assert merge_ref.main(["--merge", str(full), "--head", h, "--base", str(bases[(b1, 0)])]) == 0
    # bad input: exit 2
    nogit = tmp_path / "nogit"
    nogit.mkdir()
    (pr / "sub").mkdir()
    base1 = bases[(b1, 0)]
    for over, extra, needle in (({"--head": "abc123"}, [], "is not a full commit id"),
                                ({"--merge": nogit}, [], "not the top level"),
                                ({"--merge": pr / "sub"}, [], "not the top level"),
                                ({"--base": nogit}, [], "not the top level"),
                                ({}, ["--rev=-x"], "is not a revision"),
                                ({}, ["--rev", "no-such"], "is not a commit")):
        argv = dict({"--merge": pr, "--head": h, "--base": base1}, **over)
        r = tool("merge_ref", *[x for kv in argv.items() for x in kv], *extra)
        assert r.returncode == 2 and needle in r.stderr, (over, extra, r.stdout, r.stderr)
    # the workflows' own step, in the workflows' layout: it fetches refs/pull/1/merge itself
    assert _git("update-ref", "refs/pull/1/merge", m, cwd=o).returncode == 0
    lay = tmp_path / "lay"
    clone_at(o, b1, lay / "base")
    shutil.copytree(TOOLS, lay / "base" / "governance" / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    step = workflow_run("golden.yml", MERGE_STEP)
    assert step == workflow_run("decisions.yml", MERGE_STEP)
    url = o.resolve().as_uri()
    r = run_workflow_command(step, lay, ack=False, head=h, url=url, pr="1")
    assert r.returncode == 0 and f"fetched refs/pull/1/merge = {m[:12]}" in r.stdout, r.stdout + r.stderr
    assert os.listdir(lay / "pr") == [".git"] and git_rev(lay / "pr") == m        # objects only, HEAD at the merge
    r = run_workflow_command(step, lay, ack=False, head=h, url=url, pr="1")     # a second run never reuses pr/
    assert r.returncode == 2 and "already exists" in r.stderr
    rmtree_force(lay / "pr")
    r = run_workflow_command(step, lay, ack=False, head=ids["s"], url=url, pr="1")
    assert r.returncode == 1 and "stale" in r.stdout


@needs_git
def test_merge_ref_fetches_the_merge_commit_as_objects_only(tmp_path, monkeypatch):
    """Verifier round 3: actions/checkout refuses fork pull requests under pull_request_target, so the
    gates fetch refs/pull/N/merge with git: no working tree, fsck on, the token only in the fetch's env."""
    o, ids = merge_ref_repo(tmp_path)
    m, b1, h = ids["merge"], ids["b1"], ids["h"]
    assert _git("update-ref", "refs/pull/4/merge", m, cwd=o).returncode == 0
    url = o.resolve().as_uri()
    base = clone_at(o, b1, tmp_path / "base")
    calls = []
    real_run = merge_ref._run

    def spy(args, env=None):
        calls.append((list(args), env))
        return real_run(args, env)
    monkeypatch.setattr(merge_ref, "_run", spy)
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    assert merge_ref.fetch(url, "4", tmp_path / "pr", token="tok-123") == m
    monkeypatch.undo()
    pr = tmp_path / "pr"
    assert os.listdir(pr) == [".git"] and (pr / ".git" / "shallow").read_text().strip() == m
    assert _git("symbolic-ref", "-q", "HEAD", cwd=pr).returncode == 1                  # detached at the merge
    assert merge_ref.verify(pr, h, base)["problems"] == []
    assert [c[0][0] for c in calls[:1]] == ["init"] and calls[0][1] is None and calls[2][1] is None
    fetch_args, fetch_env = calls[1]
    assert "transfer.fsckObjects=true" in fetch_args and "--depth=1" in fetch_args and "--no-tags" in fetch_args
    assert fetch_args[-1] == "+refs/pull/4/merge:refs/pull/merge" and fetch_args[-2] == url
    assert not any("tok-123" in a for c in calls for a in c[0])                         # never on a command line
    assert fetch_env["GIT_TERMINAL_PROMPT"] == "0"
    # the token travels as an HTTP header scoped to the URL, through git's environment, and git reads it
    env = merge_ref.fetch_env("https://github.com/o/r", "tok-123")
    auth = "AUTHORIZATION: basic " + __import__("base64").b64encode(b"x-access-token:tok-123").decode()
    assert (env["GIT_CONFIG_COUNT"], env["GIT_CONFIG_KEY_0"], env["GIT_CONFIG_VALUE_0"]) == \
        ("1", "http.https://github.com/o/r.extraheader", auth)
    got = subprocess.run(["git", "config", "--get-urlmatch", "http.extraheader", "https://github.com/o/r/info/refs"],
                         env=env, capture_output=True, text=True, timeout=60)
    assert got.stdout.strip() == auth, got.stderr
    other = subprocess.run(["git", "config", "--get-urlmatch", "http.extraheader", "https://example.com/o/r"],
                           env=env, capture_output=True, text=True, timeout=60)
    assert other.stdout.strip() == ""                                                   # not sent elsewhere
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    assert "GIT_CONFIG_COUNT" not in merge_ref.fetch_env(url, "") and merge_ref.fetch_env(url, "")["GIT_TERMINAL_PROMPT"] == "0"
    # main passes GITHUB_TOKEN from the environment, and only with --fetch
    seen = []

    def fake_fetch(u, n, into, token=""):
        seen.append((u, n, str(into), token))
        raise merge_ref.GitError("stop here")
    monkeypatch.setattr(merge_ref, "fetch", fake_fetch)
    monkeypatch.setenv("GITHUB_TOKEN", "from-env")
    assert merge_ref.main(["--fetch", url, "--pr", "4", "--merge", "x", "--head", h, "--base", str(base)]) == 2
    monkeypatch.delenv("GITHUB_TOKEN")
    assert merge_ref.main(["--fetch", url, "--pr", "4", "--merge", "x", "--head", h, "--base", str(base)]) == 2
    assert seen == [(url, "4", "x", "from-env"), (url, "4", "x", "")]
    monkeypatch.undo()
    # fsck is on: a malformed merge commit is refused before anything reads it
    tree = _git("rev-parse", f"{m}^{{tree}}", cwd=o).stdout.strip()
    raw = f"tree {tree}\nparent {b1}\nparent {h}\nauthor a <a@b> 0 +0000\ncommitter a <a@b> notadate +0000\n\nx\n"
    bad = subprocess.run(["git", "-C", str(o), "hash-object", "-t", "commit", "--literally", "-w", "--stdin"],
                         input=raw.encode(), capture_output=True, timeout=60).stdout.decode().strip()
    assert _git("update-ref", "refs/pull/5/merge", bad, cwd=o).returncode == 0, bad
    r = tool("merge_ref", "--fetch", url, "--pr", "5", "--merge", tmp_path / "pr5", "--head", h, "--base", base)
    assert r.returncode == 2 and "git fetch" in r.stderr, (r.stdout, r.stderr)
    # bad input: exit 2, nothing fetched
    for args, needle in ((["--fetch", "ssh://host/r", "--pr", "4"], "not an https:// or file:// URL"),
                         (["--fetch=--upload-pack=x", "--pr", "4"], "not an https:// or file:// URL"),
                         (["--fetch", "https://", "--pr", "4"], "not an https:// or file:// URL"),
                         (["--fetch", url, "--pr", "0"], "not a pull request number"),
                         (["--fetch", url, "--pr", "04"], "not a pull request number"),
                         (["--fetch", url, "--pr", "4x"], "not a pull request number"),
                         (["--fetch", url, "--pr", "12345678901"], "not a pull request number"),
                         (["--fetch", url, "--pr", "6"], "git fetch"),                   # no such pull request
                         (["--fetch", url], "go together"), (["--pr", "4"], "go together")):
        into = tmp_path / f"into{len(os.listdir(tmp_path))}"
        r = tool("merge_ref", *args, "--merge", into, "--head", h, "--base", base)
        assert r.returncode == 2 and needle in r.stderr, (args, r.stdout, r.stderr)
    assert merge_ref.main(["--fetch", url, "--pr", "1234567890", "--merge", str(tmp_path / "pr"), "--head", h,
                           "--base", str(base)]) == 2                                   # pr/ exists already
    assert sorted(os.listdir(pr)) == [".git"]


# ------------------------------------------------------- paths that alias on Windows or macOS
def commit_files(repo: Path, parents, files: dict, msg: str, base: str = None) -> str:
    """A commit of base's tree (None: empty) with files (path -> bytes; None removes it), built through
    a temporary index with core.ignorecase off: no working tree is written, so case twins and short
    names can be committed on any file system."""
    index = repo / ".git" / "commit-files-index"
    env = dict(os.environ, GIT_INDEX_FILE=str(index))

    def g(*args, data=None):
        r = subprocess.run(["git", "-C", str(repo), "-c", "core.ignorecase=false", "-c", "user.name=t", "-c",
                            "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *args],
                           input=data, capture_output=True, env=env, timeout=60)
        assert r.returncode == 0, (args, r.stderr)
        return r.stdout.decode("utf-8").strip()

    g("read-tree", *([base] if base else ["--empty"]))
    for path, data in files.items():
        if data is None:
            g("update-index", "--force-remove", "--", path)
        else:
            g("update-index", "--add", "--cacheinfo", f"100644,{g('hash-object', '-w', '--stdin', data=data)},{path}")
    tree = g("write-tree")
    index.unlink()
    return g("commit-tree", tree, "-m", msg, *[x for p in parents for x in ("-p", p)])


def test_path_aliases_fold_and_short_names():
    twins = [("governance/INVARIANTS.md", "governance/invariants.md"), ("tests/Golden", "tests/golden"),
             ("docs/caf\u00e9.md", "docs/cafe\u0301.md"),                       # NFC and NFD (macOS)
             ("\u03b1\u0345\u0301", "\u03b1\u0301\u0345"),        # canonical order of marks; U+0345 upper-cases to a letter
             ("a/\u0131nv.md", "a/INV.md"),                                     # dotless i upper-cases to I (NTFS)
             ("stra\u00dfe", "STRASSE"), ("\u017ftate", "state"), ("\u212a.md", "k.md"),   # sharp s, long s, Kelvin
             ("INVARIANTS\u200c.md", "INVARIANTS.md"), ("x\ufeff/y", "x/y"),    # code points HFS+ ignores
             ("INVARIANTS.md.", "INVARIANTS.md"), ("dir /f", "dir/f"), ("a. .", "a")]   # Win32 trailing dots/spaces
    for a, b in twins:
        assert path_aliases.fold(a) == path_aliases.fold(b), (a, b)
        assert path_aliases.find([a, b, "other"]) == {"twins": [sorted([a, b])], "short_names": []}, (a, b)
    for a, b in (("a.md", "a.mdx"), ("a/b", "a-b"), ("ADR-0001.md", "ADR-0002.md"), (".a", "a"), ("a b", "ab"),
                 ("e\u0301", "e"), ("a", "a/a"), ("x.", "x/.")):
        assert path_aliases.fold(a) != path_aliases.fold(b), (a, b)
    # exactly the code points HFS+ ignores (git's list for .git) vanish, not their neighbours
    hfs = [0x200C, 0x200D, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x206A, 0x2070), 0xFEFF]
    assert len(hfs) == 16
    for cp in range(0x2000, 0x2100):
        assert (path_aliases.fold("a" + chr(cp) + "b") == "ab") is (cp in hfs), hex(cp)
    for cp in (0xFEFE, 0xFF00):
        assert path_aliases.fold("a" + chr(cp) + "b") != "ab", hex(cp)
    # the fold is at least as coarse as macOS's comparison and as upper-casing, for every code point
    nfd = lambda t: __import__("unicodedata").normalize("NFD", t)   # noqa: E731
    first = ({}, {})
    for cp in range(0x110000):
        if not 0xD800 <= cp <= 0xDFFF:
            c = chr(cp)
            k = path_aliases.fold(c)
            for seen, key in zip(first, (nfd(nfd(c).casefold()), c.upper())):
                assert seen.setdefault(key, k) == k, (hex(cp), key)
    short = ("INVARI~1.MD", "decisi~1", "GOVERN~1", "A~1", "~1", "go1a2b~1.py", "TEST_G~1.PY", "abcdef~1",
             "abcde~12", "a~123456", "INVARI~1.MD.", "x~1.", "NOTES~1.M", "go\u200cvern~1")
    for c in short:
        assert path_aliases.is_short_name(c), c
    for c in ("INVARIANTS.md", "INVARIANTS~1.md", "a~b", "x~1.json", "abcdef~12", "GOVERN~12", "a~1234567", "a.b~1",
              "file~2023.txt", "abcdefg~1", "~", "a~", "a~1.b.c", "a~1b"):
        assert not path_aliases.is_short_name(c), c
    # a short-name directory is reported once, whether the tree lists it or only files below it
    for paths in (["governance", "governance/decisi~1", "governance/decisi~1/ADR-0004.md", "governance/decisi~1/x/y"],
                  ["governance/decisi~1/ADR-0004.md", "governance/decisi~1/x/y"]):
        assert path_aliases.find(paths) == {"twins": [], "short_names": ["governance/decisi~1"]}
    found = path_aliases.find(["b/X", "a/INV~1.MD", "b/x", "a/Y", "a/y", "a/y"])           # duplicates collapse
    assert found == {"twins": [["a/Y", "a/y"], ["b/X", "b/x"]], "short_names": ["a/INV~1.MD"]}
    assert path_aliases.describe(found) == [
        "TWINS  a/Y = a/y  (one file on Windows and macOS)", "TWINS  b/X = b/x  (one file on Windows and macOS)",
        "8.3    a/INV~1.MD  (a component shaped like an NTFS short name aliases a longer name there)"]
    assert path_aliases.find([]) == {"twins": [], "short_names": []} and path_aliases.describe(path_aliases.find([])) == []
    # reports are ASCII (a Windows console cannot print every name) and show invisible code points
    hidden = path_aliases.describe(path_aliases.find(["INVARIANTS.md", "INVARIANTS\u200c.md", "x/A\u00e9~1"]))
    assert hidden == ["TWINS  INVARIANTS.md = INVARIANTS" + chr(92) + "u200c.md  (one file on Windows and macOS)",
                      "8.3    x/A" + chr(92) + "xe9~1  (a component shaped like an NTFS short name aliases a longer name there)"]


@needs_git
def test_path_aliases_cli_reads_a_commit(tmp_path):
    repo = tmp_path / "repo"
    assert _git("init", "-q", str(repo), cwd=tmp_path).returncode == 0
    base = commit_files(repo, [], {"governance/INVARIANTS.md": b"x\n", "docs/FORMAT.md": b"f\n"}, "base")
    twin = commit_files(repo, [base], {"governance/invariants.md": b"y\n", "docs/x/NOTES~1.MD": b""}, "twin", base=base)
    r = tool("path_aliases", "--root", repo, "--rev", base)
    assert r.returncode == 0 and r.stdout.strip() == f"path aliases OK: 4 path(s) at {base}, none alias on Windows or macOS"
    r = tool("path_aliases", "--root", repo, "--rev", twin)
    assert r.returncode == 1 and r.stdout.splitlines() == [
        "TWINS  governance/INVARIANTS.md = governance/invariants.md  (one file on Windows and macOS)",
        "8.3    docs/x/NOTES~1.MD  (a component shaped like an NTFS short name aliases a longer name there)",
        f"path aliases FOUND at {twin}: 2 finding(s). A Windows or macOS checkout would hold different files than "
        "the gates judge; rename them (no label accepts this)."], r.stdout
    assert "docs/x" in path_aliases.tree_paths(repo, twin)                            # directories are entries too
    assert _git("update-ref", "refs/heads/main", twin, cwd=repo).returncode == 0
    assert _git("symbolic-ref", "HEAD", "refs/heads/main", cwd=repo).returncode == 0
    assert path_aliases.main(["--root", str(repo)]) == 1                               # default: HEAD
    assert path_aliases.main(["--root", str(repo), "--rev", base]) == 0
    assert path_aliases.ROOT == ROOT                                                   # default --root: this checkout
    (repo / "sub").mkdir()
    for args, needle in ((["--root", repo, "--rev", "no-such"], "'no-such' is not a commit"),
                         (["--root", repo, "--rev=-x"], "is not a revision"),
                         (["--root", repo, "--rev", ""], "is not a revision"),
                         (["--root", repo / "sub"], "is not the top level of a git checkout"),
                         (["--root", tmp_path / "nowhere"], "path_aliases: ")):
        r = tool("path_aliases", *args)
        assert r.returncode == 2 and needle in r.stderr and r.stdout == "", (args, r.stdout, r.stderr)
    # a path that is not UTF-8 cannot be folded: exit 2, not a guess
    oid = subprocess.run(["git", "-C", str(repo), "hash-object", "-w", "--stdin"], input=b"z", capture_output=True,
                         timeout=60).stdout.decode().strip()
    env = dict(os.environ, GIT_INDEX_FILE=str(tmp_path / "idx"))
    for args, data in ((["read-tree", base], None), (["update-index", "-z", "--index-info"],
                                                     b"100644 blob " + oid.encode() + b"\tdocs/\xff.md\0")):
        assert subprocess.run(["git", "-C", str(repo), *args], input=data, env=env, capture_output=True,
                              timeout=60).returncode == 0
    tree = subprocess.run(["git", "-C", str(repo), "write-tree"], env=env, capture_output=True, text=True,
                          timeout=60).stdout.strip()
    latin = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@e", "commit-tree", tree,
                            "-m", "x"], capture_output=True, text=True, timeout=60).stdout.strip()
    r = tool("path_aliases", "--root", repo, "--rev", latin)
    assert r.returncode == 2 and "is not UTF-8" in r.stderr, (r.stdout, r.stderr)


ADR4 = (ROOT / "governance" / "decisions" / "ADR-0004-lfu-default-eviction.md").read_bytes()
ALIAS_CASES = {   # verifier round 3, and the NTFS short-name variant: one file in Windows and macOS checkouts
    "invariants twin": {"governance/invariants.md": b"# Invariants\n\nNone. Anything goes.\n"},
    "ADR twin": {"governance/decisions/adr-0004-lfu-default-eviction.md": ADR4.replace(b"LFU", b"LRU")},
    "golden directory twin": {"tests/Golden/conftest.py": b"import pytest\n"},
    "golden manifest twin": {"tests/golden/manifest.sha256": b""},
    "contract twin": {"docs/format.md": b"# nothing\n"},
    "NFC and NFD": {"docs/caf\u00e9.md": b"a", "docs/cafe\u0301.md": b"b"},
    "short name": {"governance/INVARI~1.MD": b"# Invariants\n\nNone.\n"},
    "short directory": {"governance/decisi~1/ADR-0004-lfu-default-eviction.md": ADR4.replace(b"LFU", b"LRU")},
    "short golden test": {"tests/golden/TEST_G~1.PY": b"def test_x():\n    pass\n"},
}


def gate_origin(tmp_path) -> tuple:
    """An origin repository whose main holds this repository's decisions, invariants, golden suite,
    contracts and governance tools, as GitHub would hold the project."""
    files = {}
    for rel in ("governance/decisions", "tests/golden", "governance/tools"):
        for f in sorted((ROOT / rel).glob("*")):
            if f.is_file() and f.suffix in (".md", ".py", ".sha256"):
                files[f.relative_to(ROOT).as_posix()] = f.read_bytes()
    for rel in ("governance/INVARIANTS.md", "docs/FORMAT.md", "docs/NUMERICS.md"):
        files[rel] = (ROOT / rel).read_bytes()
    o = tmp_path / "origin"
    assert _git("init", "-q", str(o), cwd=tmp_path).returncode == 0
    base = commit_files(o, [], files, "base")
    assert _git("update-ref", "refs/heads/main", base, cwd=o).returncode == 0
    return o, base


def open_pull_request(o: Path, base: str, number: int, files: dict) -> str:
    """A head commit with files on top of base, and GitHub's merge commit for it at
    refs/pull/<number>/merge. Returns the head commit."""
    head = commit_files(o, [base], files, f"pull request {number}", base=base)
    merge = commit_files(o, [base, head], {}, f"Merge {head} into {base}", base=head)
    assert _git("update-ref", f"refs/pull/{number}/merge", merge, cwd=o).returncode == 0
    return head


@needs_git
def test_gates_refuse_paths_that_alias_on_windows_or_macos(tmp_path):
    """Verifier round 3: a twin of governance/INVARIANTS.md, of a decided ADR or of tests/golden passed both
    gates without a label, while git leaves the twin's text in every Windows and macOS checkout (an NTFS 8.3
    short name too: reproduced with git for Windows). Each gate runs here as its workflow runs it: the base
    checked out, the pull request fetched by the workflow's own merge step, then the workflow's judging step."""
    o, base = gate_origin(tmp_path)
    url = o.resolve().as_uri()
    ws = tmp_path / "ws"
    clone_at(o, base, ws / "base")
    merge_step = workflow_run("golden.yml", MERGE_STEP)
    gates = {"golden.yml": (workflow_run("golden.yml", "Golden lock against the base revision"), "ALIAS     ",
                            "golden lock OK"),
             "decisions.yml": (workflow_run("decisions.yml", "Decision circuit breaker"),
                               "that are one file on Windows or macOS", "decision breaker OK")}

    def judge(number, files):
        head = open_pull_request(o, base, number, files)
        if (ws / "pr").exists():
            rmtree_force(ws / "pr")
        r = run_workflow_command(merge_step, ws, ack=False, head=head, url=url, pr=str(number))
        assert r.returncode == 0 and "merge ref OK" in r.stdout, r.stdout + r.stderr
        return {name: [run_workflow_command(cmd, ws, ack=ack) for ack in (False, True)]
                for name, (cmd, _, _) in gates.items()}

    for number, (case, files) in enumerate(sorted(ALIAS_CASES.items()), 1):
        for name, runs in judge(number, files).items():
            _, needle, ok = gates[name]
            for ack, r in zip((False, True), runs):        # no label makes an alias acceptable
                assert r.returncode == 1 and needle in r.stdout and ok not in r.stdout, (case, name, ack, r.stdout)
        assert tool("path_aliases", "--root", ws / "pr").returncode == 1, case
    # control: a pull request without aliases passes both gates with no label
    for name, runs in judge(90, {"docs/notes.md": b"notes\n"}).items():
        assert runs[0].returncode == 0 and gates[name][2] in runs[0].stdout, (name, runs[0].stdout, runs[0].stderr)
    # a case-only rename is no twin: the breaker sees INVARIANTS.md deleted and wants a decision
    rename = {"governance/INVARIANTS.md": None, "governance/invariants.md": b"# Invariants\n\nNone.\n"}
    runs = judge(91, rename)["decisions.yml"]
    assert runs[0].returncode == 1 and "governance/INVARIANTS.md deleted" in runs[0].stdout, runs[0].stdout


@needs_git
def test_golden_rev_refuses_paths_that_alias(tmp_path):
    root = git_golden_repo(tmp_path)
    base = git_rev(root)
    manifest = root / "tests" / "golden" / "MANIFEST.sha256"
    for files, needle in (({"tests/Golden/conftest.py": b"import pytest\n"}, "TWINS  tests/Golden = tests/golden"),
                          ({"docs/a.md": b"", "docs/A.md": b""}, "TWINS  docs/A.md = docs/a.md"),
                          ({"tests/golden/TEST_G~1.PY": b"x = 1\n"}, "8.3    tests/golden/TEST_G~1.PY")):
        c = commit_files(root, [base], files, "alias", base=base)
        res = cg.check(cg.Revision(root, c))
        assert len(res["aliases"]) == 1 and res["aliases"][0].startswith(needle) and not cg.passed(res), res
        for ack in ([], ["--acknowledged"]):
            r = tool("check_golden", "--root", root, "--rev", c, "--base-manifest", manifest, *ack)
            assert r.returncode == 1 and f"ALIAS     {needle}" in r.stdout and "1 aliasing path(s)" in r.stdout, r.stdout
            assert "golden lock OK" not in r.stdout and "acknowledged" not in r.stdout
    res = cg.check(cg.Revision(root, base))
    assert res["aliases"] == [] and cg.passed(res) and cg.check(root)["aliases"] == []
    # without its sibling the tool cannot vouch for a commit: exit 2, never a pass
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copyfile(TOOLS / "check_golden.py", alone / "check_golden.py")
    r = subprocess.run([sys.executable, str(alone / "check_golden.py"), "--root", str(root), "--rev", "HEAD"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "cannot load" in r.stderr and "path_aliases.py" in r.stderr, (r.stdout, r.stderr)
    r = subprocess.run([sys.executable, str(alone / "check_golden.py"), "--root", str(root)], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0                                       # the working-tree check does not need it


@needs_git
def test_adr_rev_refuses_paths_that_alias(tmp_path):
    repo = tmp_path / "repo"
    write_decisions(repo, adr_files(3))
    assert _git("init", "-q", cwd=repo).returncode == 0
    base = git_commit(repo, "base")
    d2 = adr_files(3)["ADR-0002-d2.md"].encode()
    for files, needle in (({"governance/invariants.md": b"None.\n"},
                           "TWINS  governance/INVARIANTS.md = governance/invariants.md"),
                          ({"governance/decisions/adr-0002-d2.md": d2.replace(b"Decision", b"Reversal")},
                           "TWINS  governance/decisions/ADR-0002-d2.md = governance/decisions/adr-0002-d2.md"),
                          ({"governance/INVARI~1.MD": b"None.\n"}, "8.3    governance/INVARI~1.MD"),
                          ({"governance/decisi~1/ADR-0002-d2.md": d2}, "8.3    governance/decisi~1  ("),
                          ({"docs/X.md": b"", "docs/x.md": b""}, "TWINS  docs/X.md = docs/x.md")):
        c = commit_files(repo, [base], files, "alias", base=base)
        with pytest.raises(adr.AliasError, match=re.escape(needle)):
            adr.read_rev(repo, c)
        for args in (["lint", "--rev", c], ["index", "--rev", c],
                     ["breaker", "--rev", c, "--base-rev", base, "--acknowledged"],
                     ["breaker", "--rev", base, "--base-rev", c, "--acknowledged"]):   # the base side too
            r = tool("adr", *args, "--root", repo)
            assert r.returncode == 1 and needle in r.stdout and "one file on Windows or macOS" in r.stdout, \
                (args, r.stdout, r.stderr)
            assert r.stdout.rstrip().endswith("(not even --acknowledged accepts this)") and "plain files" not in r.stdout
    assert issubclass(adr.AliasError, adr.RecordError)
    # a case-only rename is no twin: the breaker sees the decided ADR deleted, and lint the misspelt record
    c = commit_files(repo, [base], {"governance/decisions/ADR-0002-d2.md": None,
                                    "governance/decisions/adr-0002-d2.md": d2}, "rename", base=base)
    assert sorted(adr.read_rev(repo, c)[0]) == ["ADR-0001-d1.md", "ADR-0003-d3.md", "adr-0002-d2.md"]
    r = tool("adr", "breaker", "--root", repo, "--rev", c, "--base-rev", base)
    assert r.returncode == 1 and "decided ADR(s) ADR-0002 rewritten in place, moved back or deleted" in r.stdout
    r = tool("adr", "lint", "--root", repo, "--rev", c)
    assert r.returncode == 1 and "adr-0002-d2.md: file name must be ADR-NNNN-lowercase-slug.md" in r.stdout
    for name, record in (("ADR-0001-a.md", True), ("adr-0001-a.md", True), ("Adr-0001-a.MD", True),
                         ("ADR-0001-a.mdx", False), ("ADR-0001-a.m", False), ("TEMPLATE.md", False),
                         ("ADR0001-a.md", False), ("xADR-0001-a.md", False), ("ADR-.md", True)):
        assert adr._is_adr_name(name) is record, name
    # without its sibling the tool cannot vouch for a commit: exit 2
    alone = tmp_path / "alone"
    alone.mkdir()
    shutil.copyfile(TOOLS / "adr.py", alone / "adr.py")
    r = subprocess.run([sys.executable, str(alone / "adr.py"), "lint", "--root", str(repo), "--rev", "HEAD"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "cannot load" in r.stderr and "path_aliases.py" in r.stderr, (r.stdout, r.stderr)


EXTRA_KEYWORDS = {   # verifier round 3: builtin refused these (exit 2) while jsonschema evaluated them (0 or 1)
    "multipleOf": ("task_id", {"multipleOf": 2}, "'multipleOf' is not supported"),
    "allOf": (None, {"allOf": [{"required": ["no_such_key"]}]}, "'allOf' is not supported"),
    "minProperties": (None, {"minProperties": 999}, "'minProperties' is not supported"),
    "not": (None, {"not": {"required": ["task_id"]}}, "'not' is not supported"),
    "if/then": ("task_id", {"if": {"const": "x"}, "then": False}, "'if' is not supported"),
    "uniqueItems": ("procedures", {"uniqueItems": True}, "'uniqueItems' is not supported"),
    "tuple items (2019-09)": ("procedures", {"items": [{"type": "string"}]}, "subschema must be an object or boolean"),
}


@pytest.mark.parametrize("case", sorted(EXTRA_KEYWORDS))
def test_handover_engines_refuse_the_same_keywords(case, tmp_path):
    where, extra, needle = EXTRA_KEYWORDS[case]
    schema_doc = copy.deepcopy(SCHEMA)
    target = schema_doc["properties"][where] if where else schema_doc
    target.update(copy.deepcopy(extra))
    if case.startswith("tuple"):
        schema_doc["$schema"] = vh.DIALECTS[1]
    m = write_json(tmp_path / "T05-governance-gen-001.json", manifest())
    schema = write_json(tmp_path / "s.json", schema_doc)
    for engine in ["builtin"] + (["auto", "jsonschema"] if vh.have_jsonschema() else []):
        r = tool("validate_handover", m, "--schema", schema, "--engine", engine, "--tasks", "")
        assert r.returncode == 2 and needle in r.stderr and r.stdout == "", (case, engine, r.stdout, r.stderr)
    with pytest.raises(vh.SchemaError, match=re.escape(needle)):
        vh.precheck_schema(schema_doc)


def test_handover_metaschema_check_runs_for_jsonschema_only(tmp_path, monkeypatch):
    """The metaschema check (jsonschema) runs only with that engine: builtin, as CI runs it, never depends
    on an optional package. With every keyword and value rule shared, it should never fire on its own."""
    m = write_json(tmp_path / "T05-governance-gen-001.json", manifest())
    schema = write_json(tmp_path / "s.json", SCHEMA)
    vh.precheck_schema(dict(SCHEMA, **{"$id": "https://example.com/s.json#"}))     # an empty fragment is allowed
    with pytest.raises(vh.SchemaError, match="without a fragment"):
        vh.precheck_schema(dict(SCHEMA, **{"$id": "https://example.com/s.json#a"}))
    calls = []

    def metaschema(s):
        calls.append(s)
        raise vh.SchemaError("metaschema says no")
    monkeypatch.setattr(vh, "jsonschema_check_schema", metaschema)
    monkeypatch.setattr(vh, "have_jsonschema", lambda: True)
    args = [str(m), "--schema", str(schema), "--tasks", ""]
    assert vh.main(args + ["--engine", "builtin"]) == 0 and calls == []
    for engine in ("auto", "jsonschema"):
        assert vh.main(args + ["--engine", engine]) == 2 and len(calls) == 1, engine
        calls.clear()
    monkeypatch.setattr(vh, "have_jsonschema", lambda: False)
    assert vh.main(args + ["--engine", "auto"]) == 0 and calls == []


def test_handover_annotations_stay_usable_under_every_engine(tmp_path):
    schema_doc = copy.deepcopy(SCHEMA)
    schema_doc.update({"$comment": "c", "examples": [{}], "deprecated": False})
    schema_doc["properties"]["task_id"].update({"description": "d", "readOnly": True, "default": "x"})
    vh.precheck_schema(schema_doc)
    m = write_json(tmp_path / "T05-governance-gen-001.json", manifest())
    schema = write_json(tmp_path / "s.json", schema_doc)
    for engine in ENGINES:
        r = tool("validate_handover", m, "--schema", schema, "--engine", engine, "--tasks", "")
        assert r.returncode == 0, (engine, r.stdout, r.stderr)


# ------------------------------------------------------------------ mutate
CALC = '''\
def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def mean(xs):
    if not xs:
        return 0.0
    total = 0
    for v in xs:
        total += v
    return total / len(xs)


def is_even(n):
    return n % 2 == 0
'''

STRONG_TEST = '''\
from calc import clamp, mean, is_even
assert clamp(5, 0, 10) == 5
assert clamp(-1, 0, 10) == 0, "error C2065: looks like a compiler message but is a test failure"
assert clamp(11, 0, 10) == 10
assert mean([]) == 0.0
assert mean([1, 2, 3, 6]) == 3.0
assert is_even(4) and not is_even(7) and is_even(0)
'''


def calc_repo(tmp_path, test_src=STRONG_TEST) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(CALC, encoding="utf-8")
    (root / "test_calc.py").write_text(test_src, encoding="utf-8")
    return root


def run_mutate(root, *extra):
    r = tool("mutate", "--root", root, "--file", "calc.py", "--test", "{python} test_calc.py",
             "--jobs", "2", "--json", "-q", "--out-dir", root.parent / "report", *extra)
    return r


def test_mutate_kills_obvious_mutants_without_touching_the_tree(tmp_path):
    root = calc_repo(tmp_path)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    r = run_mutate(root, "--max-mutants", "0", "--seed", "1")
    assert r.returncode == 0, r.stderr
    rep = json.loads(r.stdout)
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before      # working tree untouched
    assert rep["selected"] == rep["candidates"] > 15
    c = rep["counts"]
    assert c["killed"] >= 15 and c["build-error"] == 0
    assert rep["score"] >= 0.75
    status = {(m["op"], m["line"], m["before"], m["after"]): m["status"] for m in rep["mutants"]}
    for key in [("relational", 2, "<", ">"), ("relational", 4, ">", "<"), ("arithmetic", 15, "/", "*"),
                ("arithmetic", 14, "+=", "-="), ("constant", 19, "2", "3"), ("negation", 10, "not xs", "xs"),
                ("return", 15, "return total / len(xs)", "return None")]:
        assert status.get(key) == "killed", (key, status.get(key))
    # x <= lo returns lo == x: an equivalent mutant that no test can kill
    assert status[("relational", 2, "<", "<=")] == "survived"
    surv = [m for m in rep["mutants"] if m["status"] == "survived"]
    assert all(m["diff"].startswith("--- a/calc.py") for m in surv)
    md = (tmp_path / "report" / "mutation.md").read_text(encoding="utf-8")
    assert "Mutation score" in md and "Surviving mutants" in md
    assert json.loads((tmp_path / "report" / "mutation.json").read_text(encoding="utf-8"))["score"] == rep["score"]


def test_mutate_weak_tests_survive_and_min_score_gates(tmp_path):
    root = calc_repo(tmp_path, "from calc import clamp\nassert clamp(5, 0, 10) == 5\n")
    r = run_mutate(root, "--max-mutants", "0", "--min-score", "0.9")
    assert r.returncode == 1
    rep = json.loads(r.stdout)
    assert rep["counts"]["survived"] > rep["counts"]["killed"]


def test_mutate_failing_baseline_and_bad_args(tmp_path):
    root = calc_repo(tmp_path, "print('failing-baseline-marker')\nraise SystemExit(3)\n")
    r = run_mutate(root, "--max-mutants", "3")
    assert r.returncode == 2 and "does not pass on the unmutated code" in r.stderr
    assert "failing-baseline-marker" in r.stderr
    assert tool("mutate", "--root", root, "--file", "nope.py", "--list").returncode == 2
    assert tool("mutate", "--root", root, "--file", "calc.py", "--lines", "9-x", "--list").returncode == 2


def test_mutate_timeout_is_detected(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "loop.py").write_text("def stop(n):\n    while n > 0:\n        n -= 1\n    return n\n", encoding="utf-8")
    (root / "t.py").write_text("from loop import stop\nassert stop(5) == 0\n", encoding="utf-8")
    r = tool("mutate", "--root", root, "--file", "loop.py", "--test", "{python} t.py", "--json", "-q",
             "--timeout", "3", "--lines", "3", "--max-mutants", "0", "--out-dir", tmp_path / "rep")
    assert r.returncode == 0, r.stderr
    rep = json.loads(r.stdout)
    status = {(m["before"], m["after"]): m["status"] for m in rep["mutants"]}
    assert status[("-=", "+=")] == "timeout"     # n grows forever
    assert rep["counts"]["killed"] and rep["warnings"] == []


FAKE_CC = '''\
import sys
src = open("demo.c").read()
if "<=" in src:
    print("demo.c(3): error C2143: syntax error")
    print("hxcc: compile failed: demo.c")
    sys.exit(2)
sys.exit(0 if src == "int small(int x) {\\n    if (x < 3)\\n        return 1;\\n    return 0;\\n}\\n" else 1)
'''


def test_mutate_c_outcomes_classified(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "demo.c").write_text("int small(int x) {\n    if (x < 3)\n        return 1;\n    return 0;\n}\n",
                                 encoding="utf-8")
    (root / "fake_cc.py").write_text(FAKE_CC, encoding="utf-8")
    r = tool("mutate", "--root", root, "--file", "demo.c", "--test", "{python} fake_cc.py", "--max-mutants", "0",
             "--jobs", "2", "--json", "-q", "--out-dir", tmp_path / "rep")
    assert r.returncode == 0, r.stderr
    rep = json.loads(r.stdout)
    status = {(m["op"], m["before"], m["after"]): m["status"] for m in rep["mutants"]}
    assert status[("relational", "<", "<=")] == "build-error"
    assert status[("relational", "<", ">")] == "killed"
    assert status[("constant", "3", "4")] == "killed"
    assert rep["counts"]["build-error"] == 1 and rep["score"] == 1.0
    md = (tmp_path / "rep" / "mutation.md").read_text(encoding="utf-8")
    assert "1 build error(s) excluded" in md and "## Surviving mutants (0)" in md and "None." in md
    built = [m for m in rep["mutants"] if m["status"] == "build-error"][0]
    assert "hxcc: compile failed" in built["output_tail"]


SMALL_C = "int small(int x) {\n    if (x < 3)\n        return 1;\n    return 0;\n}\n"

# Behaves like scripts/hxcc.py --run: a relative -o lands in the shared <data dir>/build/hxcc.
FAKE_HXCC = '''\
import os, sys, time
from pathlib import Path
out = Path(sys.argv[sys.argv.index("-o") + 1])
if not out.is_absolute():
    out = Path(os.environ["HEARTH_DATA"]) / "build" / "hxcc" / out
out.parent.mkdir(parents=True, exist_ok=True)
src = open("demo.c").read()
out.write_text(src)
time.sleep(0.3)
if out.read_text() != src:
    print("LINK : fatal error LNK1104: cannot open file (written by another build)")
    print("hxcc: link failed")
    sys.exit(2)
sys.exit(0 if src == ORIGINAL else 1)
'''


def fake_hxcc_repo(tmp_path) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "demo.c").write_text(SMALL_C, encoding="utf-8")
    (root / "fake_hxcc.py").write_text(FAKE_HXCC.replace("ORIGINAL", repr(SMALL_C)), encoding="utf-8")
    return root


def test_mutate_placeholders_are_unique_per_run():
    a, b = mutate.new_run_id(), mutate.new_run_id()
    assert a != b and re.fullmatch(r"mr[0-9a-f]{8}", a)
    cmd = "x -o out{job}.exe {run} {tmp}/t {root} {python} {jobs}"
    e = mutate.expand(cmd, 2, Path("C:/copy"), Path("C:/t mp"), a)
    assert f"-o out{a}-2.exe {a} " in e and "{jobs}" in e
    assert mutate.quote(str(Path("C:/t mp"))) + "/t" in e and mutate.quote(sys.executable) in e
    assert mutate.expand("{tmp}", 0, Path("r"), Path("{job}"), a) == mutate.quote("{job}")   # no re-expansion


def test_mutate_concurrent_runs_do_not_collide(tmp_path):
    root = fake_hxcc_repo(tmp_path)
    data = tmp_path / "data"
    env = dict(os.environ, HEARTH_DATA=str(data))
    procs = []
    for n in range(2):     # two independent mutate.py runs at once, as agents do
        procs.append(subprocess.Popen(
            [sys.executable, str(TOOLS / "mutate.py"), "--root", str(root), "--file", "demo.c", "--max-mutants", "0",
             "--jobs", "3", "--json", "-q", "--out-dir", str(tmp_path / f"rep{n}"),
             "--test", "{python} fake_hxcc.py -o mut_demo{job}.exe"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env))
    for p in procs:
        out, err = p.communicate(timeout=600)
        assert p.returncode == 0, err
        rep = json.loads(out)
        assert rep["counts"]["build-error"] == 0 and rep["counts"]["survived"] == 0, rep["counts"]
        assert rep["counts"]["killed"] == rep["selected"] >= 8 and rep["score"] == 1.0
        assert re.fullmatch(r"mr[0-9a-f]{8}", rep["run"])
    left = [p.name for p in (data / "build" / "hxcc").rglob("*")]
    assert left == [], left                                   # products named with the run id are removed
    r = tool("mutate", "--root", root, "--file", "demo.c", "--max-mutants", "2", "--jobs", "2", "--json", "-q",
             "--out-dir", tmp_path / "rep2", "--test", "{python} fake_hxcc.py -o {tmp}/mut_demo.exe", env=env)
    assert r.returncode == 0 and json.loads(r.stdout)["counts"]["killed"] == 2, r.stderr
    assert "warning" not in r.stderr and list((data / "build" / "hxcc").rglob("*")) == []
    r = tool("mutate", "--root", root, "--file", "demo.c", "--max-mutants", "1", "--jobs", "1", "--json", "-q",
             "--out-dir", tmp_path / "rep3", "--test", "{python} fake_hxcc.py -o fixed.exe", env=env)
    assert r.returncode == 0 and "warning: the -o output is shared" in r.stderr


FLAKY_TEST = '''\
import hashlib, os, sys, time
from pathlib import Path
src = Path("calc.py").read_text()
if src != ORIGINAL:
    mark = Path(os.environ["MARK_DIR"]) / hashlib.sha1(src.encode()).hexdigest()
    if not mark.exists():          # first run of this mutant: far too slow
        mark.write_text("x")
        time.sleep(120)
sys.exit(0)
'''


@pytest.mark.parametrize("retry", [True, False])
def test_mutate_timeouts_are_retried_alone(retry, tmp_path):
    root = calc_repo(tmp_path, FLAKY_TEST.replace("ORIGINAL", repr(CALC)))
    (tmp_path / "marks").mkdir()
    r = tool("mutate", "--root", root, "--file", "calc.py", "--test", "{python} test_calc.py", "--max-mutants", "2",
             "--jobs", "2", "--timeout", "3", "--json", "--out-dir", tmp_path / "rep",
             *([] if retry else ["--no-timeout-retry"]), env={"MARK_DIR": str(tmp_path / "marks")})
    assert r.returncode == 0, r.stderr
    rep = json.loads(r.stdout)
    progress = [ln.split()[0] for ln in r.stderr.splitlines() if ln.startswith("[")]
    if retry:     # slow only the first time: survived on the retry, not scored as a kill
        assert rep["counts"]["survived"] == 2 and rep["timeouts_retried"] == 2 and rep["timeouts_cleared"] == 2
        assert all(m["retried_after"] == "timeout" for m in rep["mutants"])
        assert "re-run alone" in (tmp_path / "rep" / "mutation.md").read_text(encoding="utf-8")
        assert sorted(progress) == ["[", "[", "[retry]", "[retry]"] and "  2/2]" in r.stderr
    else:
        assert rep["counts"]["timeout"] == 2 and rep["timeouts_retried"] == 0


def test_mutate_remove_build_products(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path))
    base = tmp_path / "build" / "hxcc"
    rid = "mr" + "a1" * 4
    for name in (f"t{rid}-0.exe", f"t{rid}-0.pdb", "keep.exe", f"sub/x{rid}-1.exe"):
        (base / name).parent.mkdir(parents=True, exist_ok=True)
        (base / name).write_text("x", encoding="utf-8")
    (base / f".obj-t{rid}-0-0123456789").mkdir()
    (base / f".obj-t{rid}-0-0123456789" / "a.obj").write_text("x", encoding="utf-8")
    (base / ".obj-keep-0123456789").mkdir()
    for bad in ("", "mr", "mra1a1a1", "a1a1a1a1-0"):                  # matching names, but not a run id
        assert mutate.remove_build_products(bad) == 0
    assert mutate.remove_build_products(rid) == 4
    left = sorted(p.relative_to(base).as_posix() for p in base.rglob("*"))
    assert left == [".obj-keep-0123456789", "keep.exe", "sub"]
    long_id = "mr" + "b" * 9                                            # longer ids are fine too
    (base / f"y{long_id}.exe").write_text("x", encoding="utf-8")
    assert mutate.remove_build_products(long_id) == 1
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "nothing-here"))
    assert mutate.remove_build_products(rid) == 0


def test_mutate_run_shell_isolates_process_groups(monkeypatch):
    seen = {}

    class FakePopen:
        def __init__(self, cmd, **kw):
            seen.update(kw)
            self.returncode, self.pid = 0, 1

        def communicate(self, timeout=None):
            return b"ok", None

    monkeypatch.setattr(mutate.subprocess, "Popen", FakePopen)
    rc, out, _, timed_out = mutate.run_shell("x", Path("."), {}, 5)
    assert (rc, out, timed_out) == (0, "ok", False)
    if os.name == "nt":       # Ctrl-C reaches mutate.py, which then kills the children itself
        assert seen["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP and "start_new_session" not in seen
    else:
        assert seen["start_new_session"] is True and "creationflags" not in seen


BUILD_SCRIPT = '''\
import sys, time
time.sleep(0.2)
sys.exit(1 if "<=" in open("demo.c").read() else 0)
'''


def test_mutate_separate_build_command(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "demo.c").write_text(SMALL_C, encoding="utf-8")
    (root / "build.py").write_text(BUILD_SCRIPT, encoding="utf-8")
    (root / "check.py").write_text(f"import sys\nsys.exit(0 if open('demo.c').read() == {SMALL_C!r} else 1)\n",
                                   encoding="utf-8")
    r = tool("mutate", "--root", root, "--file", "demo.c", "--build", "{python} build.py", "--test", "{python} check.py",
             "--max-mutants", "0", "--jobs", "2", "--json", "-q", "--out-dir", tmp_path / "rep")
    assert r.returncode == 0, r.stderr
    rep = json.loads(r.stdout)
    status = {(m["before"], m["after"]): m for m in rep["mutants"]}
    assert status[("<", "<=")]["status"] == "build-error" and status[("<", ">")]["status"] == "killed"
    assert all(m["seconds"] >= 0.2 for m in rep["mutants"])          # build time is counted
    (root / "build" ).mkdir()
    (root / "build" / "gen.c").write_text(SMALL_C, encoding="utf-8")   # build/ is never copied
    r = tool("mutate", "--root", root, "--file", "build/gen.c", "--test", "{python} check.py", "-q")
    assert r.returncode == 2 and "was not copied" in r.stderr


INTERRUPT_HELPER = '''\
import _thread, importlib.util, os, sys, threading, time
spec = importlib.util.spec_from_file_location("hearth_mutate", sys.argv[1])
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
marks = sys.argv[2]

def interrupt_when_running():      # Ctrl-C as soon as a mutant's test is running
    for _ in range(1200):
        if os.listdir(marks):
            break
        time.sleep(0.05)
    _thread.interrupt_main()

threading.Thread(target=interrupt_when_running, daemon=True).start()
t0 = time.monotonic()
rc = m.main(sys.argv[3:])
print("RC", rc, "ELAPSED", round(time.monotonic() - t0, 1), flush=True)
'''


def test_mutate_ctrl_c_cancels_kills_and_cleans_up(tmp_path):
    root = calc_repo(tmp_path, FLAKY_TEST.replace("ORIGINAL", repr(CALC)))
    marks = tmp_path / "marks"
    marks.mkdir()
    helper = tmp_path / "helper.py"
    helper.write_text(INTERRUPT_HELPER, encoding="utf-8")
    r = subprocess.run([sys.executable, str(helper), str(TOOLS / "mutate.py"), str(marks), "--root", str(root),
                        "--file", "calc.py", "--test", "{python} test_calc.py", "--max-mutants", "0", "--jobs", "2",
                        "--timeout", "300", "--out-dir", str(tmp_path / "rep")],
                       capture_output=True, text=True, timeout=600, env=dict(os.environ, MARK_DIR=str(marks)))
    m = re.search(r"RC (\d+) ELAPSED ([\d.]+)", r.stdout)
    assert m and m.group(1) == "130", r.stdout + r.stderr
    assert "mutate: interrupted" in r.stderr
    assert float(m.group(2)) < 60, r.stdout          # running tests were killed, queued mutants never started
    assert len(os.listdir(marks)) <= 2               # at most one started mutant per job
    run_id = re.search(r"run (mr[0-9a-f]{8})", r.stderr).group(1)
    assert not list(Path(tempfile.gettempdir()).glob(f"hearth-mutate-{run_id}-*"))
    assert not (tmp_path / "rep").exists()


def test_mutate_c_preprocessor_regions(tmp_path):
    src = ("#if 0\nint dead(int x) { return x + 1; }\n#endif\n"           # 1-3
           "#ifdef FOO\nint foo(int x) { return x - 1; }\n"                # 4-5
           "#elif BAR > 1\nint bar(int x) { return x * 2; }\n"             # 6-7
           "#else\nint other(int x) { return x % 2; }\n#endif\n"           # 8-10
           "#if 1\nint one(int x) { return x + 7; }\n#else\nint never(int x) { return x - 7; }\n#endif\n")  # 11-15
    lines = lambda muts: sorted({m.line for m in muts})   # noqa: E731
    st = {}
    unknown = mutate.gen_c(src, stats=st)
    assert lines(unknown) == [5, 7, 9, 12] and st["inactive"] > 0     # 2 and 14 are never compiled
    cond = {m.line: m.cond for m in unknown}
    assert cond[5] == "#ifdef FOO" and cond[7] == "#elif BAR > 1" and cond[9].startswith("#else of")
    assert cond[12] == ""
    assert lines(mutate.gen_c(src, defs={"FOO": mutate.UNDEF, "BAR": 2})) == [7, 12]
    assert lines(mutate.gen_c(src, defs={"FOO": mutate.UNDEF, "BAR": 1})) == [9, 12]
    assert lines(mutate.gen_c(src, defs={"FOO": 1})) == [5, 12]
    assert all(m.cond == "" for m in mutate.gen_c(src, defs={"FOO": 1}))
    c23 = ("#ifdef A\nint a(int x) { return x + 1; }\n#elifdef B\nint b(int x) { return x + 2; }\n"
           "#elifndef C\nint c(int x) { return x + 3; }\n#endif\n")
    no = mutate.UNDEF
    assert lines(mutate.gen_c(c23, defs={"A": no, "B": 1})) == [4]
    assert lines(mutate.gen_c(c23, defs={"A": no, "B": no, "C": no})) == [6]
    assert lines(mutate.gen_c(c23, defs={"A": no, "B": no, "C": 1})) == []
    assert lines(mutate.gen_c(c23, defs={"A": 1})) == [2]
    # defines from quoted headers (with include guards) decide later conditions
    (tmp_path / "cfg.h").write_text("#ifndef CFG_H\n#define CFG_H\n#define MODE (1 + 1)\n#undef GONE\n#endif\n",
                                    encoding="utf-8")
    main = tmp_path / "main.c"
    main.write_text('#include "cfg.h"\n#if MODE == 2 && !defined(GONE) && !defined(EXTRA)\n'
                    'int a(int x) { return x + 1; }\n#else\nint b(int x) { return x - 1; }\n#endif\n',
                    encoding="utf-8")
    muts, _ = mutate.generate(main.read_text(encoding="utf-8"), main, tmp_path, {"EXTRA": mutate.UNDEF})
    assert lines(muts) == [3] and all(m.cond == "" for m in muts)
    muts, _ = mutate.generate(main.read_text(encoding="utf-8"), main, tmp_path, {})
    assert lines(muts) == [3, 5] and all("EXTRA" in m.cond for m in muts)   # undecidable: kept, marked
    for flag, want in (("-U", {3}), ("-D", {5})):
        r = tool("mutate", "--root", tmp_path, "--file", main, "--list", "--json", "--max-mutants", "0", flag, "EXTRA")
        assert {m["line"] for m in json.loads(r.stdout)} == want, (flag, r.stdout + r.stderr)
    r = tool("mutate", "--root", tmp_path, "--file", main, "--list", "--json", "--max-mutants", "0", "-D", "GONE",
             "-U", "EXTRA")
    assert {m["line"] for m in json.loads(r.stdout)} == {3}      # the header's #undef GONE wins, as in C
    r = tool("mutate", "--root", tmp_path, "--file", main, "--list", "--max-mutants", "0", "--pp-model", "none")
    assert r.returncode == 0 and "in inactive #if branches" in r.stdout
    # headers that include each other without guards are each read once
    (tmp_path / "a.h").write_text('#include "b.h"\n#define FROM_A 1\n', encoding="utf-8")
    (tmp_path / "b.h").write_text('#include "a.h"\n#define FROM_B 1\n', encoding="utf-8")
    loop = tmp_path / "loop.c"
    loop.write_text('#include "a.h"\n#if FROM_A && FROM_B\nint z(int x) { return x + 1; }\n#endif\n',
                    encoding="utf-8")
    muts, _ = mutate.generate(loop.read_text(encoding="utf-8"), loop, tmp_path, {})
    assert lines(muts) == [3] and all(m.cond == "" for m in muts)


FRAG_INC = ("#if HX_FAST\n"                                                     # 1
            "static int frag_fast(int x) { return x + 1; }\n"                    # 2
            "#else\n"                                                            # 3
            "static int frag_slow(int x) { return x - 1; }\n"                    # 4
            "#endif\n"                                                           # 5
            "static int frag_cast(const int *p, int k) { return (hx_word)*p + k; }\n")   # 6


def inc_repo(tmp_path, includer_text='#include "hx_cfg.h"\n#include "frag.inc"\n') -> Path:
    src = tmp_path / "proj" / "engine" / "src"
    src.mkdir(parents=True)
    (src / "hx_cfg.h").write_text("#ifndef HX_CFG_H\n#define HX_CFG_H\n#define HX_FAST 1\ntypedef int hx_word;\n"
                                  "#endif\n", encoding="utf-8")
    (src / "frag.inc").write_text(FRAG_INC, encoding="utf-8")
    (src / "user.c").write_text(includer_text + "int use(int x) { return frag_fast(x); }\n", encoding="utf-8")
    return tmp_path / "proj"


def test_mutate_inc_files_are_c_in_their_includers_context(tmp_path):
    for name in ("a.inc", "A.INC", "a.c", "a.h"):
        assert mutate.language_of(Path(name)) == "c"
    assert mutate.language_of(Path("a.py")) == "python"
    with pytest.raises(mutate.MutateError, match=r"unsupported file type '\.txt' \(C: \.c/\.h/\.inc"):
        mutate.language_of(Path("a.txt"))
    root = inc_repo(tmp_path)
    frag = root / "engine" / "src" / "frag.inc"
    (root / "engine" / "src" / "NOTES.md").write_text('#include "frag.inc"\n', encoding="utf-8")   # not C
    (root / "engine" / "src" / "dir.c").mkdir()
    st = {}
    muts, _ = mutate.generate(FRAG_INC, frag, root, {}, st)
    assert sorted({m.line for m in muts}) == [2, 6] and all(m.cond == "" for m in muts)   # #else never compiled
    assert st["includers"] == ["engine/src/user.c"] and st["inactive"] > 0
    ops6 = {(m.op, m.before, m.after) for m in muts if m.line == 6}
    assert ("arithmetic", "+", "-") in ops6 and not any(m.before == "*" for m in muts)    # (hx_word)*p is a cast
    alone = mutate.gen_c(FRAG_INC, set(), {}, mutate.header_resolver(root), frag)        # without the context
    assert any(m.line == 4 and m.cond == "#else of #if HX_FAST" for m in alone)
    assert any(m.before == "*" for m in alone)
    defs, types, files = mutate.fragment_context(frag, root, {})
    assert defs["HX_FAST"] == 1 and "hx_word" in types and files == [root / "engine" / "src" / "user.c"]
    r = tool("mutate", "--root", root, "--file", "engine/src/frag.inc", "--list", "--max-mutants", "0")
    assert r.returncode == 0 and "fragment context: included by engine/src/user.c" in r.stdout, r.stdout + r.stderr
    (root / "check.py").write_text(f"import sys\nsys.exit(open('engine/src/frag.inc').read() != {FRAG_INC!r})\n",
                                   encoding="utf-8")
    out = tmp_path / "rep"
    r = tool("mutate", "--root", root, "--file", "engine/src/frag.inc", "--test", "{python} check.py",
             "--max-mutants", "3", "--jobs", "1", "--out-dir", out, "--quiet")
    rep = json.loads((out / "mutation.json").read_text(encoding="utf-8"))
    assert r.returncode == 0 and rep["language"] == "c" and rep["included_by"] == ["engine/src/user.c"], r.stderr
    assert rep["counts"]["killed"] == 3 and rep["score"] == 1.0


@pytest.mark.parametrize("includers,defs,decided", [
    ({"user.c": '#define HX_FAST 1\n#include "frag.inc"\n',
      "other.c": '#define HX_FAST 0\n#include "frag.inc"\n'}, {}, False),                 # disagree: unknown
    ({"user.c": '#define HX_FAST 1\n#include "frag.inc"\n',
      "other.c": '#define HX_FAST 0\n#if 0\n#include "frag.inc"\n#endif\n'}, {}, True),     # never included there
    ({"user.c": '#define HX_FAST 1\n#include "frag.inc"\n',
      "other.c": '#define HX_FAST 1\n#include "elsewhere/frag.inc"\n'}, {}, True),        # another file
    ({}, {}, False),                                                                      # no includer: plain C
    ({}, {"HX_FAST": 1}, True),                                                           # ... with -D HX_FAST
    ({"user.c": '#include "frag.inc"\n'}, {"HX_FAST": 1}, True),                          # includer keeps -D
])
def test_mutate_inc_context_from_several_includers(includers, defs, decided, tmp_path):
    root = inc_repo(tmp_path)
    src = root / "engine" / "src"
    (src / "user.c").unlink()
    for name, text in includers.items():
        (src / name).write_text(text, encoding="utf-8")
    muts, _ = mutate.generate(FRAG_INC, src / "frag.inc", root, defs)
    slow = [m for m in muts if m.line == 4]
    assert (slow == []) is decided and all(m.cond for m in slow)


def test_mutate_lists_the_real_platform_fragment():
    r = tool("mutate", "--file", "engine/src/platform_common.inc", "--list", "--max-mutants", "3")
    assert r.returncode == 0 and "3 selected" in r.stdout and "platform_win.c" in r.stdout, r.stdout + r.stderr


def test_mutate_pp_eval_and_macro_args():
    defs = {"ONE": 1, "NO": mutate.UNDEF, "OPAQUE": mutate.DEFINED}
    cases = {"0": 0, "1": 1, "defined(ONE) && !defined NO": 1, "UNKNOWN || 1": 1, "UNKNOWN && 0": 0,
             "UNKNOWN": None, "defined(UNKNOWN)": None, "OPAQUE": None, "NO": 0, "(3 + 4) * 2 == 14": 1,
             "-7 / 2 == -3": 1, "-7 % 2 == -1": 1, "-1 < 0": 1, "~0 == -1": 1, "+2 == 2": 1, "!ONE": 0, "010 == 8": 1, "0x1F == 31": 1, "1 << 70": None, "1 / 0": None,
             "ONE ? 1 : 0": None, "__has_include(<x.h>)": None, "(1": None, "1 2": None, "": None}
    for expr, want in cases.items():
        assert mutate.pp_eval(expr, defs) == want, expr
    assert mutate.parse_macro_args(["A=3", "B", "C=x+"], ["D"]) == \
        {"A": 3, "B": 1, "C": mutate.DEFINED, "D": mutate.UNDEF}
    with pytest.raises(mutate.MutateError):
        mutate.parse_macro_args(["1BAD"], [])
    assert mutate.host_macros("none") == {}
    assert mutate.resolve_pp_model("gnu") == "gnu" and mutate.resolve_pp_model("auto") in ("msvc", "os", "gnu")
    assert repr(mutate.UNDEF) == "UNDEF" and repr(mutate.DEFINED) == "DEFINED"
    import platform
    x64 = platform.machine().lower() in ("amd64", "x86_64", "x64")
    if sys.platform == "win32":
        m = mutate.host_macros("msvc")
        assert m["_WIN32"] == 1 and m["__linux__"] is mutate.UNDEF and m["__GNUC__"] is mutate.UNDEF
        if x64:
            assert m["_WIN64"] == 1 and m["_M_X64"] == 1 and m["__x86_64__"] is mutate.UNDEF
        g = mutate.host_macros("gnu")
        assert "_WIN64" not in g and g["_MSC_VER"] is mutate.UNDEF
        assert "_M_X64" not in mutate.host_macros("os") and mutate.host_macros("os")["_WIN32"] == 1
    elif sys.platform.startswith("linux"):
        g = mutate.host_macros("gnu")
        assert g["__linux__"] == 1 and g["_WIN32"] is mutate.UNDEF and g["_WIN64"] is mutate.UNDEF
        if x64:
            assert g["__x86_64__"] == 1 and g["_M_X64"] is mutate.UNDEF


def test_mutate_c_file_scope_products_and_atomic_names():
    src = ("static const int K = 4 * 1024;\n"
           "int *gp = 0, **gq;\n"
           "static int arr[2] = {3 * 4, 5};\n"
           "static const size_t S = sizeof(int *) * 2;\n"
           "int f(int atomic_n, atomic_int *p, const int * const q) { return atomic_n * 3 + *p * q[0]; }\n")
    stars = sorted((m.line, m.col) for m in mutate.gen_c(src) if m.before == "*")
    assert [ln for ln, _ in stars] == [1, 3, 4, 5, 5], stars    # line 2 declares; sizeof(int *) is a type
    line5 = src.splitlines()[4]
    assert [c for ln, c in stars if ln == 5] == [line5.index("* 3") + 1, line5.index("* q[0]") + 1]
    more = ("int g(void) { return 1; }\n"
            "Widget *make(void);\n"                       # file scope again after a body: a declarator
            "static const int M = MAX(1, 2) * 3, *mp;\n")  # a comma inside parentheses does not end the initializer
    stars = [(m.line, m.col) for m in mutate.gen_c(more) if m.before == "*"]
    assert stars == [(3, more.splitlines()[2].index("* 3") + 1)], stars


@pytest.mark.skipif(not hasattr(tokenize, "TSTRING_START"), reason="t-strings need Python 3.14+")
def test_mutate_python_skips_template_strings():
    src = 'def f(a):\n    return t"{a + 1}" if a > 0 else None\n'
    muts, _ = mutate.generate(src, Path("t.py"), ROOT)
    assert not any(m.before in ("+", "1") for m in muts) and any(m.before == ">" for m in muts)


def test_mutate_listing_is_deterministic_and_stratified(tmp_path):
    root = calc_repo(tmp_path)
    args = ("--root", root, "--file", "calc.py", "--list", "--json", "--max-mutants", "6")
    a = tool("mutate", *args, "--seed", "5")
    b = tool("mutate", *args, "--seed", "5")
    assert a.returncode == 0 and a.stdout == b.stdout
    picked = json.loads(a.stdout)
    assert len(picked) == 6 and len({m["op"] for m in picked}) == 6     # one per operator class first
    lines = json.loads(tool("mutate", "--root", root, "--file", "calc.py", "--list", "--json",
                            "--max-mutants", "0", "--lines", "9-12").stdout)
    assert lines and all(9 <= m["line"] <= 12 for m in lines)
    assert mutate.parse_line_spec("1-3,7") == {1, 2, 3, 7}


def test_mutate_python_generator_skips_strings_and_keeps_validity():
    src = ('"""1 < 2"""\n'
           'def f(x, *a, **k):\n'
           '    s = f"{x + 1}" + "a < b"  # 3 > 4\n'
           '    if x not in k and x is not None:\n'
           '        print(x)\n'
           '    return -x\n')
    muts, invalid = mutate.generate(src, Path("t.py"), ROOT)
    assert invalid == 0
    seen = {(m.op, m.line, m.before, m.after) for m in muts}
    assert all(m.line >= 3 for m in muts)                       # docstring / signature untouched
    assert not any(m.line == 3 and m.op != "arithmetic" for m in muts)
    assert [m for m in muts if m.line == 3] and all(m.before == "+" for m in muts if m.line == 3)
    assert ("call-delete", 5, "print(x)", "pass") in seen
    assert ("condition", 4, "if x not in k and x is not None:", "if not (x not in k and x is not None):") in seen
    assert ("negation", 6, "-x", "x") in seen
    assert not any(m.op == "negation" and m.before.startswith("not") for m in muts)   # 'not in' / 'is not' kept
    for m in muts:
        compile(m.apply(src), "t.py", "exec")


def test_mutate_c_generator_respects_lexical_structure():
    src = ('#include <stdio.h>\n'
           '#define LIM(a) ((a) < 3 ? 1 : 0)\n'
           'typedef struct pool pool;\n'
           '/* x < 1 */\n'
           'static int f(pool *p, const char *s, int n) {\n'
           '    int buf[8];\n'
           '    const char *t = "a < 2 && b"; // y > 2\n'
           '    for (int i = 0; i < n; i++) p->count += i * 2;\n'
           '    if (!s || n >= 4) return -1;\n'
           '    log_line(p, t);\n'
           '    return (int)*s - n;\n'
           '}\n')
    muts = mutate.gen_c(src)
    by_line = {}
    for m in muts:
        by_line.setdefault(m.line, []).append(m)
    assert not set(by_line) & {1, 2, 3, 4, 5, 6, 7}             # pp, typedef, comment, decls, strings
    ops8 = {(m.op, m.before, m.after) for m in by_line[8]}
    assert ("relational", "<", "<=") in ops8 and ("arithmetic", "*", "/") in ops8
    assert ("arithmetic", "+=", "-=") in ops8 and ("constant", "0", "1") in ops8
    assert any(m.op == "condition" and "!(i < n)" in m.after for m in by_line[8])
    ops9 = {(m.op, m.before, m.after) for m in by_line[9]}
    assert {("logical", "||", "&&"), ("negation", "!", ""), ("relational", ">=", ">"),
            ("return", "return -1", "return 0")} <= ops9
    assert ("call-delete", "log_line(p, t);", "(void)0;") in {(m.op, m.before, m.after) for m in by_line[10]}
    ops11 = {(m.op, m.before, m.after) for m in by_line[11]}
    assert ("arithmetic", "-", "+") in ops11 and not any(m.before == "*" for m in by_line[11])   # cast deref
    for m in muts:
        assert "->" not in m.before.replace("p->count", "")


def test_mutate_c_lexer_edge_cases():
    src = ('#define A(x) \\\n'
           '    ((x) < 1) /* multi\n'
           '    line < 2 */ + 3\n'
           '#include "a/*b.h"\n'
           'typedef void (*cb_fn)(int a, int b);\n'
           '/**/int f(Widget *q, cb_fn cb) {\n'
           '    const char *s = "esc \\" < 4";\n'
           "    char c = '\\'';\n"
           '    const char *names[3];\n'
           '    Widget *w = q;\n'
           '    int y = g() * 2;  // note \\\n'
           '    continued < 6\n'
           '    return y < 0x1F;\n'
           '}\n')
    muts = mutate.gen_c(src)
    lines = {m.line for m in muts}
    assert lines <= {11, 13}, sorted(lines)               # directive, comments, strings, decls skipped
    ops11 = {(m.op, m.before, m.after) for m in muts if m.line == 11}
    assert ("arithmetic", "*", "/") in ops11              # g() * 2 after empty parens is a product
    ops13 = {(m.op, m.before, m.after) for m in muts if m.line == 13}
    assert ("constant", "0x1F", "0x20") in ops13 and ("constant", "0x1F", "0x1E") in ops13
    assert "cb_fn" in mutate.c_typedef_names(mutate.lex_c(src))
    toks = mutate.lex_c('x = L"a < b" + u8"c";')
    assert [t.kind for t in toks] == ["id", "op", "str", "op", "str", "op"]


def test_mutate_python_statement_forms():
    src = "import os\nx = 1; os.path.join('a', 'b')\nobj.method(2)\nfoo(x)(y)\n"
    muts, _ = mutate.generate(src, Path("t.py"), ROOT)
    dels = {(m.line, m.before) for m in muts if m.op == "call-delete"}
    assert (2, "os.path.join('a', 'b')") in dels and (3, "obj.method(2)") in dels
    assert not any(line == 4 for line, _ in dels)          # foo(x)(y) is not a single call


def test_mutate_hunk_lines():
    diff = ("@@ -5 +5 @@\n-a\n+b\n@@ -10,2 +12,3 @@\n@@ -20,4 +22,0 @@\n@@ -30,0 +31 @@\n")
    assert mutate.hunk_lines(diff) == {5, 12, 13, 14, 31}


def test_mutate_cli_messages(tmp_path):
    root = calc_repo(tmp_path)
    r = tool("mutate", "--root", root, "--file", "calc.py")
    assert r.returncode == 2 and "--test is required" in r.stderr
    r = tool("mutate", "--root", root, "--file", "calc.py", "--lines", "200-300", "--test", "x")
    assert r.returncode == 2 and "no mutants" in r.stderr
    (root / "notes.txt").write_text("x < 1\n", encoding="utf-8")
    r = tool("mutate", "--root", root, "--file", "notes.txt", "--list")
    assert r.returncode == 2 and "unsupported file type" in r.stderr


def test_mutate_timeout_floor_and_parsing():
    assert mutate.mutant_timeout(None, 1.0) == 10.0 and mutate.mutant_timeout(None, 4.0) == 17.0
    assert mutate.mutant_timeout(3.0, 0.5) == 5.5            # baseline + 5
    assert mutate.mutant_timeout(15.0, 10.0) == 20.0         # 2 x baseline
    assert mutate.mutant_timeout(60.0, 10.0) == 60.0 and mutate.mutant_timeout(20.0, 10.0) == 20.0
    assert mutate.seconds_arg("0.25") == 0.25 and mutate.seconds_arg("1e3") == 1000.0
    for bad in ("0", "-1", "-0.0", "nan", "inf", "x", ""):
        with pytest.raises(mutate.argparse.ArgumentTypeError):
            mutate.seconds_arg(bad)


def test_mutate_short_timeout_cannot_inflate_the_score(tmp_path):
    root = calc_repo(tmp_path, "from calc import clamp\nassert clamp(5, 0, 10) == 5\n")    # weak: most survive
    for bad in ("0", "-1", "nan"):
        r = run_mutate(root, "--timeout", bad)
        assert r.returncode == 2 and "positive number of seconds" in r.stderr, r.stderr
    r = run_mutate(root, "--max-mutants", "0", "--timeout", "3", "--min-score", "0.9")
    rep = json.loads(r.stdout)
    assert r.returncode == 1 and "leaves no margin" in r.stderr, r.stderr
    assert rep["timeout_requested_s"] == 3.0 and rep["timeout_s"] >= 5.0 + rep["baseline_s"] - 0.001
    assert rep["counts"]["timeout"] == 0 and rep["counts"]["survived"] > rep["counts"]["killed"]
    r = run_mutate(root, "--max-mutants", "1", "--timeout", "60")
    assert r.returncode == 0 and "warning" not in r.stderr and json.loads(r.stdout)["timeout_s"] == 60.0
    (root / "test_calc.py").write_text("print('slow-baseline-marker', flush=True)\nimport time\ntime.sleep(30)\n",
                                       encoding="utf-8")
    r = run_mutate(root, "--max-mutants", "2", "--timeout", "1")
    assert r.returncode == 2 and "did not finish within 1 s on the unmutated code; raise --timeout" in r.stderr
    assert "slow-baseline-marker" in r.stderr


def test_mutate_all_killed_report(tmp_path):
    root = calc_repo(tmp_path)
    r = run_mutate(root, "--lines", "19", "--max-mutants", "0", "--min-score", "1.0")
    rep = json.loads(r.stdout)
    assert r.returncode == 0 and rep["counts"]["killed"] == rep["selected"] >= 4, rep["counts"]
    assert rep["warnings"] == [] and rep["inactive_skipped"] == 0 and 0 <= rep["elapsed_s"] < 600
    assert rep["timeout_requested_s"] is None and rep["timeout_s"] >= 10.0


def test_mutate_score_made_only_of_timeouts_is_flagged(tmp_path):
    hang = f"import sys, time\nif open('calc.py').read() != {CALC!r}:\n    time.sleep(120)\n"
    root = calc_repo(tmp_path, hang)
    r = run_mutate(root, "--max-mutants", "2", "--no-timeout-retry", "--min-score", "0.5")
    rep = json.loads(r.stdout)
    assert rep["counts"]["timeout"] == 2 and rep["score"] == 1.0
    assert r.returncode == 1 and "every scored mutant timed out" in r.stderr and rep["warnings"]
    assert "**Warning:** every scored mutant timed out" in (tmp_path / "report" / "mutation.md").read_text("utf-8")
    (tmp_path / "strong").mkdir()
    good = calc_repo(tmp_path / "strong")
    r = run_mutate(good, "--max-mutants", "3", "--min-score", "0.5")
    assert r.returncode == 0 and json.loads(r.stdout)["warnings"] == [] and "warning" not in r.stderr


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_mutate_since_needs_a_real_revision(tmp_path):
    root = calc_repo(tmp_path)

    def since(rev):
        return tool("mutate", "--root", root, "--file", "calc.py", "--list", "--max-mutants", "0", "--since", rev)

    r = since("HEAD")
    assert r.returncode == 2 and "needs a git checkout" in r.stderr, r.stderr
    assert _git("init", "-q", cwd=root).returncode == 0
    _git("add", "calc.py", cwd=root)
    assert _git("commit", "-q", "-m", "base", cwd=root).returncode == 0
    for bad in ("no-such-rev-xyz", "HEAD~5"):
        r = since(bad)
        assert r.returncode == 2 and "no such commit" in r.stderr, (bad, r.stderr)
    r = tool("mutate", "--root", root, "--file", "calc.py", "--list", "--since=--output=x")
    assert r.returncode == 2 and "not a revision" in r.stderr and not (root / "x").exists()
    assert since("HEAD").stdout.startswith("0 selected")                    # nothing changed yet
    (root / "calc.py").write_text(CALC.replace("x > hi", "x >= hi"), encoding="utf-8")
    listed = since("HEAD").stdout.splitlines()
    assert len(listed) > 2 and all(" 4:" in ln for ln in listed[:-1]), listed
    (root / "extra.py").write_text("def f(a):\n    return a + 1\n", encoding="utf-8")   # untracked: every line
    r = tool("mutate", "--root", root, "--file", "extra.py", "--list", "--max-mutants", "0", "--since", "HEAD")
    assert r.returncode == 0 and not r.stdout.startswith("0 selected"), r.stdout


# ---------------------------------------------------------------- friction
def recs(*rows):
    return list(enumerate(rows, 1))


A = {"cmd": "build", "exit_code": 2, "error_signature": "error C2065 at 0x1f"}


def test_friction_detects_thrashing():
    res = friction.analyze(recs(A, dict(A, error_signature="error  C2065 at 0x2e"), A), threshold=3)
    assert len(res["thrashing"]) == 1 and res["thrashing"][0]["lines"] == [1, 2, 3]
    assert friction.analyze(recs(A, A), threshold=3)["thrashing"] == []
    assert friction.analyze(recs(A, A, A), threshold=3, exact=True)["thrashing"]   # identical rows
    assert friction.analyze(recs(A, dict(A, error_signature="error C2065 at 0x2e"), A), exact=True)["thrashing"] == []


@pytest.mark.parametrize("reset", [
    {"type": "change", "note": "edited"},
    {"type": "root_cause", "error_signature": "error C2065 at 0x99", "root_cause": "scope"},
    {"cmd": "build", "exit_code": 0},
])
def test_friction_resets(reset):
    assert friction.analyze(recs(A, A, reset, A), threshold=3)["thrashing"] == []


def test_friction_tree_hash_counts_as_change_and_escalation():
    rows = [dict(A, tree="t1"), dict(A, tree="t1"), dict(A, tree="t2"), dict(A, tree="t2")]
    assert friction.analyze(recs(*rows), threshold=3)["thrashing"] == []
    assert friction.analyze(recs(*[dict(A, tree="t1")] * 3), threshold=3)["thrashing"]   # same tree: no change
    assert friction.analyze(recs(A, dict(A, change=True), A), threshold=3)["thrashing"] == []   # change before #2
    assert friction.analyze(recs(A, dict(A, change=False), A), threshold=3)["thrashing"]
    no_cmd = {"exit_code": 2, "error_signature": "E"}
    assert friction.analyze(recs(no_cmd, no_cmd), threshold=2)["thrashing"]
    note = {"type": "root_cause", "error_signature": A["error_signature"], "root_cause": "x"}
    once = friction.analyze(recs(A, A, A, note, A, A, A), threshold=3)
    assert once["thrashing"][0]["escalate"] is False and once["thrashing"][0]["notes"] == 1
    res = friction.analyze(recs(A, A, A, note, A, A, A, note, A, A, A), threshold=3)
    assert res["thrashing"][0]["escalate"] is True
    with pytest.raises(ValueError):
        friction.analyze(recs(dict(A, exit_code="2")))
    with pytest.raises(ValueError):
        friction.analyze(recs({"type": "mystery"}))


def test_friction_cli(tmp_path):
    log = tmp_path / "a.jsonl"
    for _ in range(2):
        assert tool("friction", "record", log, "--cmd", "pytest", "--exit-code", "1", "--signature", "E1").returncode == 0
    assert tool("friction", "check", log).returncode == 0
    tool("friction", "record", log, "--cmd", "pytest", "--exit-code", "1", "--signature", "E1", "--tree", "abc")
    assert json.loads(log.read_text(encoding="utf-8").splitlines()[-1])["tree"] == "abc"
    assert [n for n, _ in friction.load(log)] == [1, 2, 3]
    r = tool("friction", "check", log)
    assert r.returncode == 1 and "Root-cause note" in r.stdout and "Mechanism" in r.stdout
    assert "signature: E1" in r.stdout and "lines 1, 2, 3" in r.stdout
    r = tool("friction", "check", log, "--json")
    assert r.returncode == 1 and json.loads(r.stdout)["thrashing"][0]["count"] == 3
    assert tool("friction", "check", log, "--threshold", "1").returncode == 2
    r = tool("friction", "note", log, "--signature", "E1", "--root-cause", "short")
    assert r.returncode == 2 and "substance" in r.stderr
    assert tool("friction", "note", log, "--signature", "E1",
                "--root-cause", "fixture leaks state between tests; isolate tmp dirs").returncode == 0
    assert tool("friction", "check", log).returncode == 0
    r = tool("friction", "check", log, "--json")
    assert r.returncode == 0 and json.loads(r.stdout)["thrashing"] == []
    for _ in range(3):
        tool("friction", "record", log, "--cmd", "pytest", "--exit-code", "1", "--signature", "E1")
    r = tool("friction", "check", log)                       # one note so far: a second one is still allowed
    assert r.returncode == 1 and "Root-cause note 2 of 2" in r.stdout and "ESCALATE" not in r.stdout
    tool("friction", "note", log, "--signature", "E1", "--root-cause", "second theory: the fixture is fine")
    for _ in range(3):
        tool("friction", "record", log, "--cmd", "pytest", "--exit-code", "1", "--signature", "E1")
    r = tool("friction", "check", log)
    assert r.returncode == 1 and "ESCALATE" in r.stdout and "FRICTION_ABORT" in r.stdout
    assert "Root-cause note" not in r.stdout                 # no third note: stop instead
    log.write_text("{bad\n", encoding="utf-8")
    r = tool("friction", "check", log)
    assert r.returncode == 2 and "invalid JSON" in r.stderr
    log.write_text("", encoding="utf-8")
    assert tool("friction", "check", log, "--threshold", "2").returncode == 0      # 2 is the minimum
    assert friction.main(["record", str(log), "--cmd", "x", "--exit-code", "1", "--signature", "E9"]) == 0
    assert friction.main(["change", str(log)]) == 0
    assert friction.main(["note", str(log), "--signature", "E9", "--root-cause", "y" * 25]) == 0


def test_friction_unsigned_failures_are_keyed_by_command(tmp_path):
    hx, py, ct = ({"cmd": c, "exit_code": 1} for c in ("python scripts/hxcc.py x", "pytest tests/py", "ctest"))
    assert friction.analyze(recs(hx, py, ct), threshold=3)["thrashing"] == []
    res = friction.analyze(recs(hx, py, hx, ct, hx), threshold=3)
    assert [t["signature"] for t in res["thrashing"]] == ["exit code 1 from python scripts/hxcc.py x"]
    assert friction.analyze(recs(hx, dict(hx, exit_code=2), hx), threshold=3)["thrashing"] == []
    blank = [dict(hx, error_signature="  "), dict(hx, error_signature=None), dict(hx, error_signature="")]
    assert friction.analyze(recs(*blank), threshold=3)["thrashing"]                 # blank = no signature
    assert friction.signature(42, "x", 1) == "42" and friction.signature(None, "a  b", 3) == "exit code 3 from a b"
    log = tmp_path / "a.jsonl"
    for _ in range(3):
        assert tool("friction", "record", log, "--cmd", "ctest  -j 4", "--exit-code", "8").returncode == 0
    r = tool("friction", "check", log)
    assert r.returncode == 1 and "signature: exit code 8 from ctest -j 4" in r.stdout, r.stdout
    assert "EPISTEMIC FRICTION: the same failure 3x without an intervening change" in r.stdout
    assert tool("friction", "note", log, "--signature", "exit code 8 from ctest -j 4",
                "--root-cause", "z" * 30).returncode == 0                       # the template's key resets it
    assert tool("friction", "check", log).returncode == 0
    for c in ("a", "b", "c"):
        assert tool("friction", "record", log, "--cmd", c, "--exit-code", "1").returncode == 0
    assert tool("friction", "check", log).returncode == 0
    assert friction.main(["check", str(log)]) == 0


MSVC_PTRS = ("000001F2A3B4C5D6", "00000213B4C5D6E7", "0000024C1D2E3F40")   # printf("%p") with MSVC on x64


def test_friction_masks_msvc_pointers_and_run_specific_names(tmp_path):
    """Council round 1: a repeated crash on the primary platform was never detected as thrashing."""
    rows = [{"cmd": "c", "exit_code": 1, "error_signature": f"FAIL test_store.c:211: slot 0x7 ptr {p} not aligned"}
            for p in MSVC_PTRS]
    res = friction.analyze(recs(*rows), threshold=3)
    assert len(res["thrashing"]) == 1 and res["thrashing"][0]["signature"] == \
        "FAIL test_store.c:211: slot 0x? ptr <addr> not aligned"
    n = friction.normalize
    same = [("p=0095F031 bad", "p=00A1B2C3 bad"),                                  # x86: 8 digits
            (r"C:\Users\x\AppData\Local\Temp\tmpab1_xyz9\a.c(3): error", r"C:\Users\x\AppData\Local\Temp\tmp0q2w3e4r\a.c(3): error"),
            ("/tmp/pytest-of-ci/pytest-17/test_x0/out.txt missing", "/tmp/pytest-of-ci/pytest-18/test_x0/out.txt missing"),
            ("hearth-golden-k2j4h5g6/run failed", "hearth-golden-a1b2c3d4/run failed"),
            ("hearth-mutate-mr0011aabb-x1y2z3w4/job0 locked", "hearth-mutate-mr99ff0011-q9w8e7r6/job0 locked"),
            ("worker pid 4242 died", "worker pid 77 died"), ("PID: 1 exited", "PID: 31337 exited"),
            ("took 1.5 s", "took 20 ms")]
    for a, b in same:
        assert n(a) == n(b) and n(a) != a, (a, b, n(a))
    different = [("sha 4b825dc642cb6eb9a060e54bf8d69288fbee4904 bad", "sha 5b825dc642cb6eb9a060e54bf8d69288fbee4904 bad"),
                 ("code 0095F03 x", "code 0095F04 x"),                       # 7 digits: not a pointer
                 ("v 0095F031FCC01 x", "v 0095F031FCC02 x"),                  # 13 digits: neither width
                 ("tag DEADBEEF", "tag FEEDFACE"),                            # no digit: a word, not an address
                 ("error C2065", "error C2066"), ("test_x7", "test_x8"), ("id_00A1B2C3", "id_00A1B2C4"),
                 ("open tmpfile_a.c", "open tmpfile_b.c")]                  # not tempfile's 8 random characters
    for a, b in different:
        assert n(a) != n(b), (a, b)
    assert n("pid 12 and 0x1f and 00000095F031FCC0", exact=True) == "pid 12 and 0x1f and 00000095F031FCC0"
    log = tmp_path / "a.jsonl"
    for p in MSVC_PTRS:
        assert tool("friction", "record", log, "--cmd", "test_store.exe", "--exit-code", "3",
                    "--signature", f"ptr {p} not aligned").returncode == 0
    r = tool("friction", "check", log)
    assert r.returncode == 1 and "signature: ptr <addr> not aligned" in r.stdout, r.stdout


def test_friction_masks_sanitizer_pid_prefixes(tmp_path):
    """Verifier round 2: AddressSanitizer starts every report with ==<pid>==, so the same crash never repeated."""
    asan = ["=={0}==ERROR: AddressSanitizer: heap-use-after-free on address 0x6020000000{1} at pc 0x55d4 "
            "bp 0x7ffc sp 0x7ff8 READ of size 8 at 0x6020000000{1} thread T0 =={0}==ABORTING".format(pid, pid % 97)
            for pid in (11111, 22222, 33333)]
    res = friction.analyze(recs(*[{"cmd": "ctest", "exit_code": 1, "error_signature": s} for s in asan]), threshold=3)
    assert len(res["thrashing"]) == 1 and res["thrashing"][0]["signature"].startswith(
        "==?==ERROR: AddressSanitizer: heap-use-after-free on address 0x?") and "==?==ABORTING" in \
        res["thrashing"][0]["signature"]
    n = friction.normalize
    assert n("==12== x") == n("==345== x") and n("==1==x") != n("==1==y") and n("a == 12 == b") == "a == 12 == b"
    assert n("==12==ERROR", exact=True) == "==12==ERROR"
    log = tmp_path / "asan.jsonl"
    for s in asan:
        assert tool("friction", "record", log, "--cmd", "ctest", "--exit-code", "1", "--signature", s).returncode == 0
    r = tool("friction", "check", log)
    assert r.returncode == 1 and "EPISTEMIC FRICTION" in r.stdout, r.stdout


def test_friction_check_without_a_log_is_ok(tmp_path):
    env = {"HEARTH_DATA": str(tmp_path / "data")}
    r = tool("friction", "check", "--task", "T99-none", env=env)
    assert r.returncode == 0 and "no attempts recorded" in r.stdout, r.stdout + r.stderr
    r = tool("friction", "check", "--task", "T99-none", "--json", env=env)
    assert r.returncode == 0 and json.loads(r.stdout)["thrashing"] == []
    assert not (tmp_path / "data").exists()                   # checking creates nothing
    assert friction.main(["check", str(tmp_path / "none.jsonl")]) == 0
    (tmp_path / "dir.jsonl").mkdir()
    assert tool("friction", "check", tmp_path / "dir.jsonl").returncode == 2


def test_friction_default_data_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("HEARTH_DATA", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    native = tmp_path / ("local" if os.name == "nt" else "xdg") / "hearth"
    assert friction.task_log("T06-store") == native / "attempts" / "T06-store.jsonl"
    assert friction.data_dir(windows=True) == tmp_path / "local" / "hearth"
    assert friction.data_dir(windows=False) == tmp_path / "xdg" / "hearth"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert friction.data_dir(windows=False) == Path.home() / ".cache" / "hearth"
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "explicit"))
    assert friction.data_dir() == tmp_path / "explicit"


def test_friction_log_location_and_write_errors(tmp_path):
    env = {"HEARTH_DATA": str(tmp_path / "data")}
    r = tool("friction", "record", "--task", "T06-store", "--cmd", "b", "--exit-code", "1", "--signature", "E",
             env=env)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "data" / "attempts" / "T06-store.jsonl").is_file()
    assert tool("friction", "check", "--task", "T06-store", env=env).returncode == 0
    for args in (["check"], ["check", tmp_path / "a.jsonl", "--task", "T06-store"], ["check", "--task", "../x"]):
        r = tool("friction", *args, env=env)
        assert r.returncode == 2 and "friction:" in r.stderr, args
    blocker = tmp_path / "afile.txt"
    blocker.write_text("x", encoding="utf-8")
    for args in (["record", blocker / "a.jsonl", "--cmd", "b", "--exit-code", "1"],
                 ["change", blocker / "a.jsonl"],
                 ["note", blocker / "a.jsonl", "--signature", "E", "--root-cause", "x" * 30]):
        r = tool("friction", *args)
        assert r.returncode == 2 and "cannot append" in r.stderr and "Traceback" not in r.stderr, args


# --------------------------------------------------------------------- adr
def test_adr_repository_records_lint_clean():
    files, inv = adr.read_checkout(ROOT)
    assert "INV-VERIFY" in inv
    assert len(files) >= 7
    assert adr.lint(files) == []
    r = tool("adr", "lint")
    assert r.returncode == 0 and f"ADR lint OK: {len(files)} decision record(s)" in r.stdout


def adr_text(n, status="Accepted", supersedes="none", superseded_by="none", body=True, decision="d",
             day="2026-10-04"):
    sections = (f"\n## Context\n\nc\n\n## Decision\n\n{decision}\n\n## Consequences\n\nq\n" if body
                else "\n## Context\n\nc\n")
    return (f"# ADR-{n:04d}: Decision {n}\n\n- Status: {status}\n- Date: {day}\n"
            f"- Supersedes: {supersedes}\n- Superseded-by: {superseded_by}\n{sections}")


def adr_files(n=8, **over):
    files = {f"ADR-{i:04d}-d{i}.md": adr_text(i) for i in range(1, n + 1)}
    files.update(over)
    return files


def test_adr_lint_catches_problems():
    assert adr.lint(adr_files()) == []
    assert any("Consequences" in e for e in adr.lint(adr_files(**{"ADR-0002-d2.md": adr_text(2, body=False)})))
    assert any("Status" in e for e in adr.lint(adr_files(**{"ADR-0002-d2.md": adr_text(2, status="Maybe")})))
    one_sided = adr_files(**{"ADR-0009-d9.md": adr_text(9, supersedes="ADR-0002")})
    errs = adr.lint(one_sided)
    assert any("ADR-0002" in e and "Superseded" in e for e in errs)
    good = adr_files(**{"ADR-0009-d9.md": adr_text(9, supersedes="ADR-0002"),
                        "ADR-0002-d2.md": adr_text(2, status="Superseded", superseded_by="ADR-0009")})
    assert adr.lint(good) == []
    assert any("unknown ADR" in e for e in adr.lint(adr_files(**{"ADR-0003-d3.md": adr_text(3, supersedes="ADR-0042")})))
    assert any("file name" in e for e in adr.lint({"ADR-1-x.md": adr_text(1)}))
    dup = adr.lint(adr_files(**{"ADR-0002-other.md": adr_text(2)}))
    assert any("duplicate id ADR-0002" in e for e in dup)
    stale = adr_files(**{"ADR-0009-d9.md": adr_text(9, supersedes="ADR-0002"),
                         "ADR-0002-d2.md": adr_text(2, superseded_by="ADR-0009")})       # still Accepted
    assert [e for e in adr.lint(stale) if "status must be Superseded" in e] == \
        ["ADR-0002-d2.md: status must be Superseded (superseded by accepted ADR-0009)"]
    # date.fromisoformat accepts these on Python 3.11+ but not on 3.9/3.10; the format is YYYY-MM-DD everywhere
    for day in ("20261004", "2026-W40-1", "2026-02-30", "2026-10-4", "٢026-10-04"):
        errs = adr.lint(adr_files(**{"ADR-0002-d2.md": adr_text(2, day=day)}))
        assert any("Date" in e for e in errs), day


def test_adr_circuit_breaker(tmp_path):
    base = adr_files(8)
    one = adr_files(8, **{"ADR-0009-d9.md": adr_text(9, supersedes="ADR-0001"),
                          "ADR-0001-d1.md": adr_text(1, status="Superseded", superseded_by="ADR-0009")})
    assert adr.breaker(base, one, "inv", "inv", 0.15)["tripped"] == []          # 1/8 = 12.5%
    two = dict(one, **{"ADR-0002-d2.md": adr_text(2, status="Deprecated")})
    res = adr.breaker(base, two, "inv", "inv", 0.15)
    assert res["deprecated"] == ["ADR-0001", "ADR-0002"] and res["tripped"]      # 2/8 = 25%
    assert adr.breaker(base, base, "inv", "inv changed", 0.15)["tripped"]       # invariants without ADR
    assert adr.breaker(base, one, "inv", "inv changed", 0.15)["tripped"]        # an ADR, but not about invariants
    cites = adr_files(8, **{"ADR-0009-d9.md": adr_text(9, decision="Relax governance/INVARIANTS.md row X.")})
    assert adr.breaker(base, cites, "inv", "inv changed", 0.15)["tripped"] == []
    # CLI with --base-root checkouts
    for name, files in (("base", base), ("cur", two)):
        d = tmp_path / name / "governance" / "decisions"
        d.mkdir(parents=True)
        for fn, text in files.items():
            (d / fn).write_text(text, encoding="utf-8")
        (tmp_path / name / "governance" / "INVARIANTS.md").write_text("inv\n", encoding="utf-8")
    r = tool("adr", "breaker", "--root", tmp_path / "cur", "--base-root", tmp_path / "base")
    assert r.returncode == 1 and "CIRCUIT BREAKER" in r.stdout
    r = tool("adr", "breaker", "--root", tmp_path / "cur", "--base-root", tmp_path / "base", "--acknowledged")
    assert r.returncode == 0 and "acknowledged" in r.stdout
    r = tool("adr", "breaker", "--root", tmp_path / "cur")
    assert r.returncode == 2 and "exactly one" in r.stderr
    r = tool("adr", "lint", "--root", tmp_path / "cur")
    assert r.returncode == 0
    (tmp_path / "cur" / "governance" / "decisions" / "ADR-0003-d3.md").write_text(adr_text(3, body=False),
                                                                                    encoding="utf-8")
    r = tool("adr", "lint", "--root", tmp_path / "cur")
    assert r.returncode == 1 and "error:" in r.stdout and "FAILED" in r.stdout
    r = tool("adr", "index", "--root", tmp_path / "base")
    assert r.returncode == 0 and r.stdout.count("| [ADR-") == 8
    assert "| [ADR-0003](ADR-0003-d3.md) | Decision 3 | Accepted | - | - |" in r.stdout


def _git(*args, cwd):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "core.autocrlf=false",
                           "-c", "commit.gpgsign=false", *args], cwd=cwd, capture_output=True, text=True, timeout=60)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_adr_reads_git_revisions(tmp_path):
    repo = tmp_path / "g"
    d = repo / "governance" / "decisions"
    d.mkdir(parents=True)
    for name, text in adr_files(3).items():
        (d / name).write_text(text, encoding="utf-8")
    (d / "README.txt").write_text("not an ADR\n", encoding="utf-8")
    (repo / "governance" / "INVARIANTS.md").write_text(INV_BASE, encoding="utf-8")
    assert _git("init", "-q", cwd=repo).returncode == 0
    _git("add", "-A", cwd=repo)
    assert _git("commit", "-q", "-m", "base", cwd=repo).returncode == 0
    files, inv = adr.read_rev(repo, "HEAD")
    assert files == adr_files(3) and inv.replace("\r\n", "\n") == INV_BASE
    (repo / "governance" / "INVARIANTS.md").write_text(INV_BASE.replace("bit-identical", "close"), encoding="utf-8")
    r = tool("adr", "breaker", "--root", repo, "--base-rev", "HEAD")
    assert r.returncode == 1 and "INV-A" in r.stdout
    r = tool("adr", "breaker", "--root", repo, "--base-rev", "no-such-rev")
    assert r.returncode == 2 and "adr:" in r.stderr


INV_BASE = ("| id | invariant | enforced by |\n|----|---|---|\n"
            "| **INV-A** | logits are bit-identical | golden |\n| **INV-B** | no OOB access | asan |\n")


def test_adr_invariants_breaker_needs_a_relevant_decision():
    base = adr_files(8)
    weakened = INV_BASE.replace("bit-identical", "close")
    assert adr.changed_invariants(INV_BASE, weakened) == ["INV-A"]
    assert adr.changed_invariants(INV_BASE, INV_BASE.replace("| asan |", "|   asan |")) == []
    assert adr.changed_invariants(INV_BASE, INV_BASE + "| **INV-C** | new | review |\n") == ["INV-C"]

    def tripped(cur):
        return adr.breaker(base, cur, INV_BASE, weakened, 0.15)["tripped"]

    trivial = dict(base, **{"ADR-0001-d1.md": base["ADR-0001-d1.md"] + "\n\n"})      # whitespace only
    assert tripped(trivial) and "INV-A" in tripped(trivial)[0]
    assert tripped(dict(base, **{"ADR-0009-d9.md": adr_text(9, decision="Unrelated: about INV-B.")}))
    assert tripped(dict(base, **{"ADR-0009-d9.md": adr_text(9, decision="About INV-AB, not INV-A-ish.")}))
    assert not tripped(dict(base, **{"ADR-0009-d9.md": adr_text(9, decision="Relax INV-A to a tolerance.")}))
    amended = adr_text(3, decision="d. Amends INVARIANTS.md.")
    proposed = dict(base, **{"ADR-0003-d3.md": adr_text(3, status="Proposed")})
    assert adr.breaker(proposed, dict(proposed, **{"ADR-0003-d3.md": amended}), INV_BASE, weakened,
                       0.15)["tripped"] == []                      # a changed (still open) ADR covers it
    assert [r for r in tripped(dict(base, **{"ADR-0003-d3.md": amended})) if "rewritten" in r]   # not a decided one
    assert adr.breaker(base, trivial, INV_BASE, INV_BASE.replace("\n", "\r\n"), 0.15)["tripped"] == []   # CRLF only
    preamble = adr.breaker(base, base, INV_BASE, "intro\n" + INV_BASE, 0.15)
    assert preamble["invariant_ids_changed"] == [] and preamble["tripped"]
    assert not adr.breaker(base, dict(base, **{"ADR-0009-d9.md": adr_text(9, decision="Covers INV-B.")}),
                           INV_BASE, "intro\n" + INV_BASE, 0.15)["tripped"]   # any INV id when no row changed


def test_adr_breaker_counts_a_deleted_invariants_file(tmp_path):
    base = adr_files(8)
    res = adr.breaker(base, base, INV_BASE, None, 0.15)
    assert res["invariants_changed"] and res["invariant_ids_changed"] == ["INV-A", "INV-B"]
    assert "INVARIANTS.md deleted (INV-A, INV-B)" in res["tripped"][0]
    assert "INVARIANTS.md added" in adr.breaker(base, base, None, INV_BASE, 0.15)["tripped"][0]
    assert adr.breaker(base, base, None, None, 0.15)["tripped"] == []
    retire = dict(base, **{"ADR-0009-d9.md": adr_text(9, decision="Retire INVARIANTS.md.")})
    assert adr.breaker(base, retire, INV_BASE, None, 0.15)["tripped"] == []
    for side in ("base", "cur"):
        d = tmp_path / side / "governance" / "decisions"
        d.mkdir(parents=True)
        for name, text in base.items():
            (d / name).write_text(text, encoding="utf-8")
    (tmp_path / "base" / "governance" / "INVARIANTS.md").write_text(INV_BASE, encoding="utf-8")
    r = tool("adr", "breaker", "--root", tmp_path / "cur", "--base-root", tmp_path / "base")
    assert r.returncode == 1 and "CIRCUIT BREAKER: governance/INVARIANTS.md deleted" in r.stdout, r.stdout


def test_adr_breaker_limit_is_exclusive():
    base = adr_files(20)
    cur = dict(base, **{f"ADR-{i:04d}-d{i}.md": adr_text(i, status="Deprecated") for i in (1, 2, 3)})
    res = adr.breaker(base, cur, "inv", "inv", 0.15)
    assert res["fraction"] == 0.15 and res["tripped"] == []                    # exactly 15% is allowed
    cur["ADR-0004-d4.md"] = adr_text(4, status="Deprecated")
    assert adr.breaker(base, cur, "inv", "inv", 0.15)["tripped"]


def test_adr_breaker_duplicate_ids_cannot_hide_retired_decisions(tmp_path):
    """Council round 1: Accepted shadow copies under the same ids hid 8 of 8 retired decisions."""
    base = adr_files(8)
    cur = {n: adr_text(i, status="Deprecated") for i, n in enumerate(sorted(base), 1)}
    cur.update({f"ADR-{i:04d}-zz-shadow.md": adr_text(i) for i in range(1, 9)})
    res = adr.breaker(base, cur, "inv", "inv", 0.15)
    assert res["duplicates"] == [f"ADR-{i:04d}" for i in range(1, 9)] and len(res["deprecated"]) == 8
    assert res["tripped"][0].startswith("duplicate ADR id(s) ADR-0001, ADR-0002") and "8 of 8" in res["tripped"][1]
    one = dict(base, **{"ADR-0002-copy.md": adr_text(2)})               # one identical copy, nothing retired
    res = adr.breaker(base, one, "inv", "inv", 0.15)
    assert res["duplicates"] == ["ADR-0002"] and res["deprecated"] == ["ADR-0002"] and len(res["tripped"]) == 1
    assert res["rewritten"] == []
    for side, files in (("base", base), ("cur", cur)):
        d = tmp_path / side / "governance" / "decisions"
        d.mkdir(parents=True)
        for n, t in files.items():
            (d / n).write_text(t, encoding="utf-8")
    r = tool("adr", "breaker", "--root", tmp_path / "cur", "--base-root", tmp_path / "base")
    assert r.returncode == 1 and "CIRCUIT BREAKER: duplicate ADR id(s)" in r.stdout, r.stdout


def test_adr_breaker_detects_decided_adrs_rewritten_in_place():
    base = adr_files(8, **{"ADR-0006-d6.md": adr_text(6, status="Deprecated"),
                           "ADR-0007-d7.md": adr_text(7, status="Proposed")})

    def rewritten(**changes):
        return adr.breaker(base, dict(base, **changes), "inv", "inv", 0.15)["rewritten"]

    lru = adr_text(4, decision="The default policy is `HEARTH_POLICY_LRU`").replace("Decision 4", "LRU by default")
    assert rewritten(**{"ADR-0004-d4.md": lru}) == ["ADR-0004"]                  # council probe 2
    res = adr.breaker(base, dict(base, **{"ADR-0004-d4.md": lru}), "inv", "inv", 0.15)
    assert res["deprecated"] == [] and any("ADR-0004 rewritten in place" in r for r in res["tripped"])
    assert rewritten(**{"ADR-0004-d4.md": adr_text(4).replace("Decision 4", "Decision four")}) == ["ADR-0004"]
    assert rewritten(**{"ADR-0004-d4.md": adr_text(4).replace("\nq\n", "\nq, and more\n")}) == ["ADR-0004"]
    assert rewritten(**{"ADR-0004-d4.md": adr_text(4).replace("\nc\n", "\nc  \n\n")}) == []      # whitespace
    assert rewritten(**{"ADR-0004-d4.md": adr_text(4) + "\n## Notes\n\nlater\n"}) == ["ADR-0004"]
    assert rewritten(**{"ADR-0004-d4.md": adr_text(4, supersedes="ADR-0001")}) == ["ADR-0004"]
    assert rewritten(**{"ADR-0004-d4.md": adr_text(4, status="Superseded", superseded_by="ADR-0009",
                                                    day="2026-12-01")}) == []   # how an ADR is retired
    assert rewritten(**{"ADR-0006-d6.md": adr_text(6, status="Deprecated", decision="other")}) == ["ADR-0006"]
    assert rewritten(**{"ADR-0007-d7.md": adr_text(7, status="Accepted", decision="settled")}) == []   # was open
    gone = dict(base)
    del gone["ADR-0006-d6.md"]
    assert adr.breaker(base, gone, "inv", "inv", 0.15)["rewritten"] == ["ADR-0006"]          # history deleted
    renamed = dict(base)
    renamed["ADR-0005-new-slug.md"] = renamed.pop("ADR-0005-d5.md")
    assert adr.breaker(base, renamed, "inv", "inv", 0.15)["tripped"] == []


def test_adr_breaker_freezes_decided_records_except_retirement():
    """Verifier round 2: a decided ADR demoted to Proposed could be rewritten in a second change, and
    free text on the field lines or before the first section was not compared."""
    base = adr_files(9, **{"ADR-0006-d6.md": adr_text(6, status="Deprecated"),
                           "ADR-0007-d7.md": adr_text(7, status="Proposed")})

    def rewritten(n, text):
        return adr.breaker(base, dict(base, **{f"ADR-{n:04d}-d{n}.md": text}), "inv", "inv", 1.0)["rewritten"]

    four = adr_text(4)
    assert rewritten(4, adr_text(4, status="Proposed")) == ["ADR-0004"]           # step 1 of the two-step rewrite
    res = adr.breaker(base, dict(base, **{"ADR-0004-d4.md": adr_text(4, status="Proposed")}), "inv", "inv", 0.15)
    assert res["deprecated"] == ["ADR-0004"] and any("moved back" in r for r in res["tripped"])
    for edit in (lambda x: x.replace("- Status: Accepted", "- Status: Accepted (void: LRU is the default now)"),
                 lambda x: x.replace("- Date: 2026-10-04", "- Date: 2026-10-04, void since PR 12"),
                 lambda x: x.replace("- Date: 2026-10-04", "- Date: soon"),
                 lambda x: x.replace("- Superseded-by: none", "- Superseded-by: [ADR-0009](void.md)"),
                 lambda x: x.replace("- Superseded-by: none", "- Superseded-by: ADR-0009 (void)"),
                 lambda x: x.replace("- Supersedes: none", "- Supersedes: ADR-0001"),
                 lambda x: x.replace("\n## Context", "\nNOTE: void, LRU is the default.\n\n## Context"),
                 lambda x: "Preface: void.\n\n" + x,
                 lambda x: x.replace("- Supersedes: none", "- Supersedes: none\n- Status: Deprecated"),
                 lambda x: x.replace("- Status: Accepted", "Status: Accepted")):
        assert rewritten(4, edit(four)) == ["ADR-0004"], edit(four)
    for same in (four + "\n\n", four.replace("- Status: Accepted", "* **Status**: Accepted"),
                 four.replace("- Date: 2026-10-04", "- Date: 2026-12-01"),
                 adr_text(4, status="Superseded", superseded_by="[ADR-0009](ADR-0009-d9.md)"),
                 adr_text(4, status="Deprecated", superseded_by="ADR-0009, ADR-0008")):
        assert rewritten(4, same) == [], same
    assert rewritten(7, adr_text(7, status="Accepted", decision="settled")) == []    # was open: may change

    def rewritten_from(b, text):
        return adr.breaker(b, dict(b, **{"ADR-0004-d4.md": text}), "inv", "inv", 1.0)["rewritten"]

    noted = four.replace("- Status: Accepted", "- Status: Accepted (as of PR 3)")    # a base that has a note
    assert rewritten_from(dict(base, **{"ADR-0004-d4.md": noted}), noted.replace("PR 3", "PR 12")) == ["ADR-0004"]
    assert rewritten_from(dict(base, **{"ADR-0004-d4.md": noted}), noted.replace("Accepted (as", "Deprecated (as")) \
        == []                                                                       # retired, note kept
    body = adr_text(4, decision="- Status: Accepted\nis how ADR-0002 put it")      # field-like text in a section
    assert rewritten_from(dict(base, **{"ADR-0004-d4.md": body}),
                          body.replace("- Status: Accepted\nis how", "- Status: Rejected\nis how")) == ["ADR-0004"]
    assert rewritten(9, adr_text(9).replace("\nc\n", "\nNOTE: x\n")) == ["ADR-0009"]
    transitions = [(a, b) for a in adr.STATUSES[1:] for b in adr.STATUSES + ("Acceptd",)]
    allowed = {(s, s) for s in adr.STATUSES} | adr.RETIREMENTS
    assert adr.RETIREMENTS == {("Accepted", "Deprecated"), ("Accepted", "Superseded"), ("Deprecated", "Superseded")}
    for old, new in transitions:
        by = "ADR-0002" if "Superseded" in (old, new) else "none"
        b = adr_files(2, **{"ADR-0001-d1.md": adr_text(1, status=old, superseded_by=by)})
        c = dict(b, **{"ADR-0001-d1.md": adr_text(1, status=new, superseded_by=by)})
        assert (adr.breaker(b, c, "inv", "inv", 1.0)["rewritten"] == []) is ((old, new) in allowed), (old, new)


def test_adr_lint_takes_exact_status_and_successor_forms():
    errs = adr.lint(adr_files(**{"ADR-0002-d2.md": adr_text(2, status="Accepted (void)")}))
    assert errs == ["ADR-0002-d2.md: Status 'Accepted (void)' must be exactly one of "
                    "Proposed|Accepted|Deprecated|Superseded"]
    for form in ("ADR-0009", "[ADR-0009](ADR-0009-d9.md)", "[ADR-0009](./ADR-0009-d9.md)"):
        good = adr_files(**{"ADR-0009-d9.md": adr_text(9, supersedes=form.replace("9", "2")),
                            "ADR-0002-d2.md": adr_text(2, status="Superseded", superseded_by=form)})
        assert adr.lint(good) == [], form
    for form in ("ADR-0009 (void)", "[ADR-0009](void.md)", "[ADR-0009](ADR-0008-d8.md)", "[ADR-0009]", "ADR-9",
                 "ADR-0009(x)"):
        errs = adr.lint(adr_files(**{"ADR-0009-d9.md": adr_text(9, supersedes="ADR-0002"),
                                     "ADR-0002-d2.md": adr_text(2, status="Superseded", superseded_by=form)}))
        assert any("is not an ADR id (ADR-NNNN), a link" in e for e in errs), (form, errs)


def test_adr_records_behind_links_or_not_files_are_refused(tmp_path):
    root, other = tmp_path / "root", tmp_path / "other"
    write_decisions(root, adr_files(3))
    write_decisions(other, adr_files(3))
    assert adr.read_checkout(root) == (adr_files(3), INV_BASE)
    assert adr.read_checkout(tmp_path / "empty") == ({}, None)
    d = root / "governance" / "decisions"
    (d / "ADR-0004-x.md").mkdir()
    with pytest.raises(adr.RecordError, match="ADR-0004-x.md is not a plain file"):
        adr.read_checkout(root)
    r = tool("adr", "lint", "--root", root)
    assert r.returncode == 1 and "decision records must be plain files" in r.stdout
    (d / "ADR-0004-x.md").rmdir()
    (d / "ADR-0004-x.md").write_bytes(b"# ADR-0004: \xff\n")
    with pytest.raises(adr.RecordError, match="not UTF-8"):
        adr.read_checkout(root)
    (d / "ADR-0004-x.md").unlink()
    (d / "README.txt").write_bytes(b"\xff")                       # not a record: not read
    (d / "adr-0005-lower.md").write_text("x", encoding="utf-8")  # a misspelt record is read, and lint reports it
    assert adr.read_checkout(root)[0] == dict(adr_files(3), **{"adr-0005-lower.md": "x"})
    assert "adr-0005-lower.md: file name must be ADR-NNNN-lowercase-slug.md" in adr.lint(adr.read_checkout(root)[0])
    (d / "adr-0005-lower.md").unlink()
    inv = root / "governance" / "INVARIANTS.md"
    inv.unlink()
    inv.mkdir()
    with pytest.raises(adr.RecordError, match="INVARIANTS.md is not a plain file"):
        adr.read_checkout(root)
    inv.rmdir()
    shutil.rmtree(d)
    if not make_dir_link(d, other / "governance" / "decisions"):
        pytest.skip("cannot create a symbolic link or junction here")
    assert adr.read_checkout(other)[0] == adr_files(3) and (d / "ADR-0001-d1.md").is_file()
    with pytest.raises(adr.RecordError, match="governance/decisions is a symbolic link or junction"):
        adr.read_checkout(root)
    for args in (["lint", "--root", root], ["index", "--root", root],
                 ["breaker", "--root", root, "--base-root", other, "--acknowledged"],
                 ["breaker", "--root", other, "--base-root", root, "--acknowledged"]):
        r = tool("adr", *args)
        assert r.returncode == 1 and "is a symbolic link or junction" in r.stdout, (args, r.stdout, r.stderr)
    _rmlink(d)
    shutil.rmtree(root / "governance")
    assert make_dir_link(root / "governance", other / "governance")
    with pytest.raises(adr.RecordError, match="^governance is a symbolic link"):
        adr.read_checkout(root)
    _rmlink(root / "governance")
    try:
        os.symlink(other / "governance" / "INVARIANTS.md", tmp_path / "inv-link")
    except (OSError, NotImplementedError):
        return                                                  # file links need a privilege on Windows
    write_decisions(root, adr_files(3))
    (root / "governance" / "INVARIANTS.md").unlink()
    os.replace(tmp_path / "inv-link", root / "governance" / "INVARIANTS.md")
    with pytest.raises(adr.RecordError, match="INVARIANTS.md is a symbolic link"):
        adr.read_checkout(root)


@needs_git
def test_adr_read_rev_takes_only_plain_files(tmp_path):
    repo = tmp_path / "repo"
    write_decisions(repo, adr_files(3))
    (repo / "governance" / "decisions" / "README.txt").write_text("not a record\n", encoding="utf-8")
    (repo / "governance" / "decisions" / "old").mkdir()
    (repo / "governance" / "decisions" / "old" / "ADR-0009-nested.md").write_text("ignored\n", encoding="utf-8")
    assert _git("init", "-q", cwd=repo).returncode == 0
    git_commit(repo, "base")
    _git("tag", "base", cwd=repo)
    write_decisions(repo, adr_files(1), "changed\n")             # the working tree is not read
    assert adr.read_rev(repo, "HEAD") == (adr_files(3), INV_BASE)
    _git("checkout", "-q", "-f", "--", ".", cwd=repo)
    head = git_rev(repo)
    cases = [("governance", "120000", "governance is not a directory"),
             ("governance", "160000", "governance is not a directory"),
             ("governance/decisions", "120000", "governance/decisions is not a directory"),
             ("governance/decisions", "160000", "governance/decisions is not a directory"),
             ("governance/decisions/ADR-0002-d2.md", "120000", "ADR-0002-d2.md is not a plain file"),
             ("governance/decisions/ADR-0002-d2.md", "160000", "ADR-0002-d2.md is not a plain file"),
             ("governance/INVARIANTS.md", "120000", "INVARIANTS.md is not a plain file"),
             ("governance/INVARIANTS.md", "160000", "INVARIANTS.md is not a plain file")]
    for path, mode, needle in cases:
        _git("checkout", "-q", "-f", "-B", "case", "base", cwd=repo)
        assert _git("rm", "-rq", "--cached", path, cwd=repo).returncode == 0
        git_special_entry(repo, path, head if mode == "160000" else "../../elsewhere", mode)
        assert _git("commit", "-q", "-m", f"{path} {mode}", cwd=repo).returncode == 0
        with pytest.raises(adr.RecordError, match=needle):
            adr.read_rev(repo, "HEAD")
        r = tool("adr", "breaker", "--root", repo, "--rev", "HEAD", "--base-rev", "base", "--acknowledged")
        assert r.returncode == 1 and needle in r.stdout and "plain files" in r.stdout, (path, mode, r.stdout)
    _git("checkout", "-q", "-f", "-B", "case", "base", cwd=repo)
    (repo / "governance" / "decisions" / "ADR-0002-d2.md").write_bytes(b"# ADR-0002: \xff\n")
    git_commit(repo, "not utf-8")
    with pytest.raises(adr.RecordError, match="ADR-0002-d2.md at HEAD is not UTF-8"):
        adr.read_rev(repo, "HEAD")
    _git("checkout", "-q", "-f", "-B", "case", "base", cwd=repo)
    _git("rm", "-rq", "governance", cwd=repo)
    (repo / "keep.txt").write_text("x\n", encoding="utf-8")
    git_commit(repo, "no governance")
    assert adr.read_rev(repo, "HEAD") == ({}, None)
    for rev, needle in (("-x", "is not a revision"), ("", "is not a revision"), ("no-such", "is not a commit")):
        with pytest.raises(RuntimeError, match=needle):
            adr.read_rev(repo, rev)
    (repo / "sub").mkdir()
    with pytest.raises(RuntimeError, match="is not the top level"):
        adr.read_rev(repo / "sub", "HEAD")
    r = tool("adr", "lint", "--root", repo, "--rev", "base")
    assert r.returncode == 0 and "3 decision record(s)" in r.stdout


def _git_ok() -> bool:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _git_ok(), reason="needs a git checkout")
def test_adr_breaker_against_git_revision():
    files, inv = adr.read_rev(ROOT, "HEAD")
    assert "INV-VERIFY" in inv and len(files) >= 8
    r = tool("adr", "breaker", "--base-rev", "no-such-revision-xyz")
    assert r.returncode == 2 and "adr:" in r.stderr
