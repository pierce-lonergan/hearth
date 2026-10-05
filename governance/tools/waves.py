#!/usr/bin/env python3
"""waves — validate Hearth's task DAG and schedule it in topological waves.

    python governance/tools/waves.py                  # validate, print waves + next runnable set
    python governance/tools/waves.py validate         # validation only (CI)
    python governance/tools/waves.py next --status T01-platform=done --status T06-store=active
    python governance/tools/waves.py --json           # machine-readable
    python governance/tools/waves.py --mermaid        # flowchart for docs / PRs

Validation (exit 1 if any check fails):
  * every task has id, title, deps, owns, status (todo|active|review|done|blocked);
  * ids are unique; every dep exists; no self-dependencies;
  * the dependency graph is acyclic (the offending cycle is printed);
  * owned paths are repo-relative and owned by at most one task that is not done.
    A path owns everything below it, so "engine/src/" (or "engine/src")
    conflicts with "engine/src/pool.c". Paths are compared after removing "."
    and empty segments and ignoring case (Windows and macOS file systems are
    case-insensitive); a non-canonical spelling is a warning.

Waves are Kahn layers: wave 0 holds the tasks without dependencies, wave k the
tasks whose dependencies all sit in waves < k. Tasks in one wave can run in
parallel because ownership is disjoint. The *runnable* set is every `todo` task
whose dependencies are all `done` (statuses from tasks.json, overridable with
--status/--done for what-if planning).

Pure standard library, Python 3.9+. Exit codes: 0 valid, 1 invalid DAG,
2 unreadable input / bad arguments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = ROOT / "governance" / "tasks.json"
STATUSES = ("todo", "active", "review", "done", "blocked")


class TaskFileError(Exception):
    pass


def load_tasks(path) -> dict:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise TaskFileError(f"cannot read {path}: {e}") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise TaskFileError(f"{path}: invalid JSON: {e}") from None


def norm_path(p: str):
    """Returns (canonical_path, is_dir) or None if p is not a repo-relative path.

    The canonical form uses "/" and has no "." or empty segments; is_dir records
    a trailing "/" (or "/."), which only matters for the spelling warning.
    """
    q = p.strip().replace("\\", "/")
    if not q or q.startswith("/") or ":" in q.split("/")[0]:
        return None
    is_dir = q.endswith("/") or q.endswith("/.")
    parts = [s for s in q.split("/") if s not in ("", ".")]
    if not parts or ".." in parts:
        return None
    return "/".join(parts), is_dir


def canonical(p: str):
    n = norm_path(p)
    return None if n is None else n[0] + ("/" if n[1] else "")


def paths_overlap(a: str, b: str) -> bool:
    """True if one path is (case-insensitively) a segment prefix of the other, or equal."""
    na, nb = norm_path(a), norm_path(b)
    if na is None or nb is None:
        return False
    sa, sb = na[0].casefold().split("/"), nb[0].casefold().split("/")
    k = min(len(sa), len(sb))
    return sa[:k] == sb[:k]


def _tasks(doc) -> list:
    if not isinstance(doc, dict) or not isinstance(doc.get("tasks"), list):
        return []
    return [t for t in doc["tasks"] if isinstance(t, dict)]


def find_cycle(ids: list, deps: dict):
    """Returns one dependency cycle as [a, b, ..., a] (a depends on b ...), or None."""
    color = {i: 0 for i in ids}  # 0 new, 1 on stack, 2 finished
    for start in ids:
        if color[start]:
            continue
        stack = [(start, iter(deps.get(start, ())))]
        path = [start]
        color[start] = 1
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                path.pop()
                color[node] = 2
                continue
            if nxt not in color:
                continue
            if color[nxt] == 1:
                return path[path.index(nxt):] + [nxt]
            if color[nxt] == 0:
                color[nxt] = 1
                path.append(nxt)
                stack.append((nxt, iter(deps.get(nxt, ()))))
    return None


def compute_waves(tasks: list):
    """Kahn layering in file order. Returns (waves, leftover_ids)."""
    ids = [t["id"] for t in tasks]
    known = set(ids)
    deps = {t["id"]: [d for d in t.get("deps", []) if d in known] for t in tasks}
    placed: dict = {}
    waves = []
    remaining = list(ids)
    while remaining:
        layer = [i for i in remaining if all(d in placed for d in deps[i])]
        if not layer:
            break
        for i in layer:
            placed[i] = len(waves)
        waves.append(layer)
        remaining = [i for i in remaining if i not in placed]
    return waves, remaining


def effective_status(tasks: list, overrides: dict) -> dict:
    st = {t["id"]: t.get("status", "todo") for t in tasks}
    st.update(overrides)
    return st


def ownership_conflicts(tasks: list, status: dict) -> list:
    live = [t for t in tasks if status.get(t["id"]) != "done"]
    out = []
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            for pa in a.get("owns", []):
                for pb in b.get("owns", []):
                    if isinstance(pa, str) and isinstance(pb, str) and paths_overlap(pa, pb):
                        out.append({"a": a["id"], "path_a": pa, "b": b["id"], "path_b": pb})
    return out


def validate(doc, overrides=None):
    """Returns (errors, warnings, cycle)."""
    errors, warnings = [], []
    if not isinstance(doc, dict):
        return ["top level must be a JSON object"], warnings, None
    if not isinstance(doc.get("tasks"), list):
        return ['missing "tasks" array'], warnings, None
    seen = set()
    for n, t in enumerate(doc["tasks"]):
        where = f"tasks[{n}]"
        if not isinstance(t, dict):
            errors.append(f"{where}: not an object")
            continue
        tid = t.get("id")
        if not isinstance(tid, str) or not tid.strip():
            errors.append(f"{where}: missing or empty id")
            continue
        where = tid
        if tid in seen:
            errors.append(f"duplicate task id {tid!r}")
        seen.add(tid)
        if not isinstance(t.get("title"), str) or not t.get("title"):
            errors.append(f"{where}: missing title")
        for key in ("deps", "owns"):
            v = t.get(key)
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                errors.append(f"{where}: {key!r} must be a list of strings")
        if "accept" in t and (not isinstance(t["accept"], list) or not all(isinstance(x, str) for x in t["accept"])):
            errors.append(f"{where}: 'accept' must be a list of strings")
        if t.get("status") not in STATUSES:
            errors.append(f"{where}: status {t.get('status')!r} not in {'|'.join(STATUSES)}")
        for p in t.get("owns", []) if isinstance(t.get("owns"), list) else []:
            if not isinstance(p, str):
                continue
            if norm_path(p) is None:
                errors.append(f"{where}: owned path {p!r} is not a repo-relative path")
            elif canonical(p) != p:
                warnings.append(f"{where}: owned path {p!r} is not canonical; write {canonical(p)!r}")
    if errors:
        return errors, warnings, None

    tasks = _tasks(doc)
    ids = [t["id"] for t in tasks]
    known = set(ids)
    overrides = overrides or {}
    for k, v in overrides.items():
        if k not in known:
            errors.append(f"--status: unknown task {k!r}")
        if v not in STATUSES:
            errors.append(f"--status: {k}={v!r} not in {'|'.join(STATUSES)}")
    for t in tasks:
        for d in t["deps"]:
            if d == t["id"]:
                errors.append(f"{t['id']}: depends on itself")
            elif d not in known:
                errors.append(f"{t['id']}: unknown dependency {d!r}")
        if len(set(t["deps"])) != len(t["deps"]):
            warnings.append(f"{t['id']}: duplicate entries in deps")

    deps = {t["id"]: [d for d in t["deps"] if d in known and d != t["id"]] for t in tasks}
    cycle = find_cycle(ids, deps)
    if cycle:
        errors.append("dependency cycle: " + " -> ".join(cycle))

    status = effective_status(tasks, overrides)
    for c in ownership_conflicts(tasks, status):
        ca, cb = norm_path(c["path_a"])[0], norm_path(c["path_b"])[0]
        how = " (paths differ only in case)" if ca != cb and ca.casefold() == cb.casefold() else ""
        errors.append(f"ownership conflict: {c['a']} owns {c['path_a']!r}, {c['b']} owns {c['path_b']!r}{how}")
    for t in tasks:
        own = [p for p in t["owns"]]
        for i, p in enumerate(own):
            for q in own[i + 1:]:
                if paths_overlap(p, q):
                    warnings.append(f"{t['id']}: owned paths {p!r} and {q!r} overlap")
        st = status[t["id"]]
        if st in ("active", "review", "done"):
            pending = [d for d in t["deps"] if d in known and status.get(d) != "done"]
            if pending:
                warnings.append(f"{t['id']} is {st} but dependencies are not done: {', '.join(pending)}")
    return errors, warnings, cycle


def plan(tasks: list, status: dict) -> dict:
    waves, _ = compute_waves(tasks)
    wave_of = {i: n for n, w in enumerate(waves) for i in w}
    done = {i for i, s in status.items() if s == "done"}
    runnable, in_progress, blocked, waiting = [], [], [], []
    for t in tasks:
        s = status[t["id"]]
        if s == "done":
            continue
        missing = [d for d in t["deps"] if d not in done]
        if s in ("active", "review"):
            in_progress.append(t["id"])
        elif s == "blocked":
            blocked.append(t["id"])
        elif missing:
            waiting.append({"id": t["id"], "waiting_on": missing})
        else:
            runnable.append(t["id"])
    unfinished = [wave_of[i] for i, s in status.items() if s != "done" and i in wave_of]
    return {
        "waves": waves,
        "current_wave": min(unfinished) if unfinished else None,
        "runnable": runnable,
        "in_progress": in_progress,
        "blocked": blocked,
        "waiting": waiting,
        "done": [t["id"] for t in tasks if status[t["id"]] == "done"],
    }


def to_mermaid(tasks: list, status: dict) -> str:
    waves, _ = compute_waves(tasks)
    by_id = {t["id"]: t for t in tasks}
    node = {t["id"]: f"n{k}" for k, t in enumerate(tasks)}   # task ids may contain any character
    lines = ["flowchart LR"]
    for n, w in enumerate(waves):
        lines.append(f'  subgraph W{n}["wave {n}"]')
        for i in w:
            title = by_id[i]["title"]
            if len(title) > 48:
                title = title[:45] + "..."
            label = f"{_esc(i)}<br/>{_esc(title)}"
            lines.append(f'    {node[i]}["{label}"]:::{status[i]}')
        lines.append("  end")
    for t in tasks:
        for d in t["deps"]:
            if d in by_id:
                lines.append(f"  {node[d]} --> {node[t['id']]}")
    lines += [
        "  classDef todo fill:#f5f5f5,stroke:#888",
        "  classDef active fill:#fff3c4,stroke:#c90",
        "  classDef review fill:#dbeafe,stroke:#36c",
        "  classDef done fill:#d1fadf,stroke:#393",
        "  classDef blocked fill:#fde2e2,stroke:#c33",
    ]
    return "\n".join(lines)


def _esc(text: str) -> str:
    return "".join(f"#{ord(c)};" if c in '"<>#&' else c for c in text)


def parse_overrides(status_args, done_args) -> dict:
    out = {}
    for s in status_args or []:
        if "=" not in s:
            raise ValueError(f"--status expects ID=STATUS, got {s!r}")
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    for d in done_args or []:
        for k in d.split(","):
            if k.strip():
                out[k.strip()] = "done"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="show", choices=("show", "validate", "next"))
    ap.add_argument("--tasks", default=str(DEFAULT_TASKS), help="task DAG (default: governance/tasks.json)")
    ap.add_argument("--status", action="append", metavar="ID=STATUS", help="override a task status (repeatable)")
    ap.add_argument("--done", action="append", metavar="ID[,ID...]", help="treat these tasks as done")
    ap.add_argument("--json", action="store_true", help="print a JSON report")
    ap.add_argument("--mermaid", action="store_true", help="print a Mermaid flowchart")
    a = ap.parse_args(argv)

    try:
        doc = load_tasks(a.tasks)
        overrides = parse_overrides(a.status, a.done)
    except (TaskFileError, ValueError) as e:
        print(f"waves: {e}", file=sys.stderr)
        return 2

    errors, warnings, cycle = validate(doc, overrides)
    tasks = _tasks(doc)
    report = {"tasks_file": str(a.tasks), "valid": not errors, "errors": errors, "warnings": warnings,
              "cycle": cycle}
    if not errors:
        status = effective_status(tasks, overrides)
        report.update(plan(tasks, status))
        report["status"] = status

    if a.json:
        print(json.dumps(report, indent=2))
        return 0 if not errors else 1
    if a.mermaid:
        if errors:
            for e in errors:
                print(f"error: {e}", file=sys.stderr)
            return 1
        print(to_mermaid(tasks, report["status"]))
        return 0

    for w in warnings:
        print(f"warning: {w}")
    if errors:
        for e in errors:
            print(f"error: {e}")
        print(f"INVALID: {len(errors)} error(s) in {a.tasks}")
        return 1
    n = len(tasks)
    if a.command == "validate":
        print(f"OK: {n} tasks, {len(report['waves'])} waves, acyclic, ownership disjoint")
        return 0
    status = report["status"]
    if a.command == "show":
        for k, w in enumerate(report["waves"]):
            print(f"wave {k}: " + ", ".join(f"{i} [{status[i]}]" for i in w))
        print()
    cw = report["current_wave"]
    print("current wave: " + (str(cw) if cw is not None else "none (all done)"))
    print("runnable now: " + (", ".join(report["runnable"]) or "-"))
    if report["in_progress"]:
        print("in progress:  " + ", ".join(report["in_progress"]))
    if report["blocked"]:
        print("blocked:      " + ", ".join(report["blocked"]))
    if a.command == "next":
        for w in report["waiting"]:
            print(f"waiting:      {w['id']} <- {', '.join(w['waiting_on'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
