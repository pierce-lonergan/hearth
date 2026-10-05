"""Tests for the governance tools (governance/tools/*.py). Standard library only."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
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
    ("^[%x41-%x43]$", "D", False), ("^(?<y>a)%k<y>$", "aa", True), ("^(a)%1$", "aa", True),
    ("^[a-c-]+$", "b-a", True), ("^[%-]$", "-", True), ("^a{2,}$", "aaa", True), ("^a+?%/$", "aa/", True),
    ("^[[]$", "[", True), ("^[&&~|]+$", "&~|", True), ("^%0$", chr(0), True), ("^%$%^$", "$^", True),
]]
ECMA_INVALID = [bs(p) for p in [
    "^[]a]$", "]", "}", "a{", "a{,3}", "%a", "%e", "%-", "%01", "%c" + E_ACUTE, "%p{L}", "(?P<x>a)", "(?i)a",
    "(?#c)", "a++", "a*?+", "[%d-z]", "[a-%w]", "[z-a]", "[a", "%", "(?<=a+)b", "%k<q>", "%u{110000}", "%x4", "%uD83",
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
    (cache / "short.pyc").write_bytes(b"\x00\x00\x00\x00\x01")
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
    files = adr.load_dir(ROOT / "governance" / "decisions")
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
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args], cwd=cwd,
                          capture_output=True, text=True, timeout=60)


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
    files = adr.load_rev(repo, "HEAD", "governance/decisions")
    assert files == adr_files(3)
    assert adr.show_rev(repo, "HEAD", "governance/INVARIANTS.md").replace("\r\n", "\n") == INV_BASE
    assert adr.show_rev(repo, "HEAD", "governance/missing.md") is None
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
    assert not tripped(dict(base, **{"ADR-0003-d3.md": adr_text(3, decision="d. Amends INVARIANTS.md.")}))
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


def _git_ok() -> bool:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              timeout=30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _git_ok(), reason="needs a git checkout")
def test_adr_breaker_against_git_revision():
    assert "INV-VERIFY" in adr.show_rev(ROOT, "HEAD", "governance/INVARIANTS.md")
    assert adr.show_rev(ROOT, "HEAD", "governance/no-such-file.md") is None
    assert isinstance(adr.load_rev(ROOT, "HEAD", "governance/decisions"), dict)
    r = tool("adr", "breaker", "--base-rev", "no-such-revision-xyz")
    assert r.returncode == 2 and "adr:" in r.stderr
