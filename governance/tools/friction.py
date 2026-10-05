#!/usr/bin/env python3
"""friction — stop retry loops before they burn a session (epistemic friction).

An agent (or human) appends one JSON object per line to an attempt log:

    {"cmd": "python scripts/hxcc.py ...", "exit_code": 2, "error_signature": "error C2065: 'n': undeclared"}
    {"type": "change", "note": "declared n in pool.c"}          # the code changed since the last attempt
    {"type": "root_cause", "error_signature": "...", "root_cause": "..."}

Attempts may carry "tree" (e.g. a hash of `git diff`); a different tree than the
previous attempt counts as a change. A successful attempt (exit_code 0) clears
the failures recorded for the same command. An attempt without an
error_signature is keyed by its command and exit code ("exit code 1 from
<cmd>"), so unrelated commands that fail the same way are not one signature. A
log that does not exist yet holds no attempts.

Thrashing = the same error signature failing --threshold times (default 3)
with no change and no root-cause note in between. `check` then prints the
mandatory root-cause template and exits 1; writing the note (`note`) resets that
signature. A signature that thrashes again after two notes escalates: `check`
prints no template but tells you to end the session with a FRICTION_ABORT
handover instead of a third try.

    python governance/tools/friction.py check attempts.jsonl [--threshold 3] [--json]
    python governance/tools/friction.py record attempts.jsonl --cmd "..." --exit-code 1 --signature "..."
    python governance/tools/friction.py change attempts.jsonl [--note "..."]
    python governance/tools/friction.py note attempts.jsonl --signature "..." --root-cause "..."

Instead of a log path, --task ID uses <data dir>/attempts/<ID>.jsonl (data dir:
$HEARTH_DATA, default %LOCALAPPDATA%\\hearth or ~/.cache/hearth), which keeps
scratch logs out of the repository:

    python governance/tools/friction.py check --task T06-store

Signatures are compared after collapsing whitespace and masking what differs
between runs of the same failure: hex addresses (0x... and MSVC's bare 8 or 16
digit %p), temporary directory names, process ids (also the ==PID== prefix of
sanitizer reports) and durations (--exact disables this). Pure standard library, Python 3.9+.
Exit: 0 ok, 1 thrashing detected, 2 bad input.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ESCALATE_AFTER_NOTES = 2

TEMPLATE = """\
Root-cause note {nth} of {limit} for this signature (required before the next attempt; every field):
  1. Observation  - the failing check and its first error line, verbatim.
  2. Mechanism    - why it fails: the cause, not the symptom. Cite the code or doc line.
  3. Evidence     - what you ran or read that supports this, and what would refute it.
  4. Falsified    - approaches already tried for this signature and why each failed.
  5. Next attempt - the single change you will make and the result you expect.
If the signature thrashes again after note {limit}, `check` escalates: no further
attempts; hand over with lifecycle_status FRICTION_ABORT, trigger EPISTEMIC_FRICTION.
Record the note with:
  python governance/tools/friction.py note {log} --signature "{sig}" --root-cause "<1..5>"
"""

ESCALATION = """\
  ESCALATE: {notes} root-cause notes already written for this signature. Do not retry.
  End the session with a handover: lifecycle_status FRICTION_ABORT, trigger
  EPISTEMIC_FRICTION; put both notes and what they falsified in the epistemic_ledger."""


def data_dir(windows: bool = None) -> Path:
    if os.environ.get("HEARTH_DATA"):
        return Path(os.environ["HEARTH_DATA"])
    if os.name == "nt" if windows is None else windows:
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "hearth"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "hearth"


def task_log(task: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", task):
        raise ValueError(f"--task {task!r}: use a task id such as T06-store")
    return data_dir() / "attempts" / f"{task}.jsonl"


_VOLATILE = [   # (pattern, replacement): parts of an error line that differ between identical failures
    (re.compile(r"0x[0-9a-fA-F]+"), "0x?"),
    # MSVC prints %p as 16 (x64) or 8 (x86) bare hex digits: 00000095F031FCC0
    (re.compile(r"(?<![0-9A-Za-z_])(?=[0-9A-Fa-f]*[0-9])(?:[0-9A-Fa-f]{16}|[0-9A-Fa-f]{8})(?![0-9A-Za-z_])"),
     "<addr>"),
    (re.compile(r"\btmp[a-z0-9_]{8}\b"), "tmp?"),                       # tempfile's default names
    (re.compile(r"\bpytest-\d+\b"), "pytest-?"),                          # pytest's numbered base temp
    (re.compile(r"\bhearth-(golden|mutate)-[^\s/\\]+"), r"hearth-\1-?"),   # this directory's tools' temp dirs
    (re.compile(r"==\d+=="), "==?=="),                                  # sanitizers: ==<pid>==ERROR: ...
    (re.compile(r"\b(pid|PID)([ =:#]+)\d+\b"), r"\1\2?"),
    (re.compile(r"\b\d+(\.\d+)?\s?(ns|us|ms|s)\b"), "<t>"),
]


def normalize(sig, exact: bool = False) -> str:
    s = ("" if sig is None else str(sig)).strip()
    if exact:
        return s
    s = re.sub(r"\s+", " ", s)
    for pat, rep in _VOLATILE:
        s = pat.sub(rep, s)
    return s


def signature(error_signature, cmd: str, code: int, exact: bool = False) -> str:
    """The key failures are counted under. Without an error signature, different commands
    failing with the same exit code are different failures."""
    return normalize(error_signature, exact) or normalize(f"exit code {code} from {cmd}", exact)


def load(path: Path) -> list:
    recs = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{n}: invalid JSON: {e}") from None
        if not isinstance(rec, dict):
            raise ValueError(f"{path}:{n}: expected a JSON object")
        recs.append((n, rec))
    return recs


def analyze(records: list, threshold: int = 3, exact: bool = False) -> dict:
    open_fail: dict = {}     # signature -> [(line, cmd)] since the last reset
    notes: dict = {}         # signature -> number of root-cause notes
    last_tree = None
    for n, rec in records:
        kind = rec.get("type") or ("attempt" if "cmd" in rec or "exit_code" in rec else "unknown")
        if kind == "change" or rec.get("change") is True:
            open_fail.clear()
            if kind == "change":
                continue
        if kind == "root_cause":
            sig = normalize(rec.get("error_signature", ""), exact)
            notes[sig] = notes.get(sig, 0) + 1
            open_fail.pop(sig, None)
            continue
        if kind != "attempt":
            raise ValueError(f"line {n}: unknown record type {kind!r}")
        tree = rec.get("tree")
        if tree is not None:
            if last_tree is not None and tree != last_tree:
                open_fail.clear()
            last_tree = tree
        code = rec.get("exit_code")
        if not isinstance(code, int) or isinstance(code, bool):
            raise ValueError(f"line {n}: exit_code must be an integer")
        cmd = rec.get("cmd", "")
        if code == 0:
            for sig in [s for s, hits in open_fail.items() if any(c == cmd for _, c in hits)]:
                del open_fail[sig]
            continue
        sig = signature(rec.get("error_signature"), cmd, code, exact)
        open_fail.setdefault(sig, []).append((n, cmd))
    thrash = []
    for sig, hits in open_fail.items():
        if len(hits) >= threshold:
            thrash.append({"signature": sig, "count": len(hits), "lines": [h[0] for h in hits],
                           "cmds": sorted({h[1] for h in hits}), "notes": notes.get(sig, 0),
                           "escalate": notes.get(sig, 0) >= ESCALATE_AFTER_NOTES})
    return {"threshold": threshold, "thrashing": thrash,
            "open": {s: len(h) for s, h in open_fail.items()}, "notes": notes}


def _append(path: Path, rec: dict) -> None:
    rec = {"ts": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"), **rec}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    c = sub.add_parser("check", help="detect thrashing")
    c.add_argument("--threshold", type=int, default=3)
    c.add_argument("--exact", action="store_true", help="compare signatures verbatim")
    c.add_argument("--json", action="store_true")
    r = sub.add_parser("record", help="append an attempt")
    r.add_argument("--cmd", required=True)
    r.add_argument("--exit-code", type=int, required=True)
    r.add_argument("--signature", default="")
    r.add_argument("--tree")
    ch = sub.add_parser("change", help="record that the code changed")
    ch.add_argument("--note", default="")
    nt = sub.add_parser("note", help="record a root-cause note for a signature")
    nt.add_argument("--signature", required=True)
    nt.add_argument("--root-cause", required=True)
    for p in (c, r, ch, nt):
        p.add_argument("log", nargs="?", help="attempt log (JSONL)")
        p.add_argument("--task", help="use <data dir>/attempts/<TASK>.jsonl as the log")
    a = ap.parse_args(argv)
    try:
        if bool(a.log) == bool(a.task):
            raise ValueError("give exactly one of LOG or --task")
        log = Path(a.log) if a.log else task_log(a.task)
    except ValueError as e:
        print(f"friction: {e}", file=sys.stderr)
        return 2

    if a.action in ("record", "change", "note"):
        if a.action == "record":
            rec = {"cmd": a.cmd, "exit_code": a.exit_code, "error_signature": a.signature}
            if a.tree:
                rec["tree"] = a.tree
        elif a.action == "change":
            rec = {"type": "change", "note": a.note}
        else:
            if len(a.root_cause.strip()) < 20:
                print("friction: a root-cause note needs substance (>= 20 characters); see the template",
                      file=sys.stderr)
                return 2
            rec = {"type": "root_cause", "error_signature": a.signature, "root_cause": a.root_cause}
        try:
            _append(log, rec)
        except OSError as e:
            print(f"friction: cannot append to {log}: {e}", file=sys.stderr)
            return 2
        return 0

    if a.threshold < 2:
        print("friction: --threshold must be >= 2", file=sys.stderr)
        return 2
    try:
        records = load(log) if log.exists() else []        # no log yet: no attempts
        res = analyze(records, a.threshold, a.exact)
    except (OSError, ValueError) as e:
        print(f"friction: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(res, indent=2))
        return 1 if res["thrashing"] else 0
    if not records:
        print(f"friction: ok (no attempts recorded in {log})")
        return 0
    if not res["thrashing"]:
        worst = max(res["open"].values(), default=0)
        print(f"friction: ok (worst open signature: {worst}/{a.threshold} failures without a change)")
        return 0
    for t in res["thrashing"]:
        print(f"EPISTEMIC FRICTION: the same failure {t['count']}x without an intervening change")
        print(f"  signature: {t['signature']}")
        print(f"  attempts:  {log} lines {', '.join(map(str, t['lines']))}")
        for cmd in t["cmds"]:
            print(f"  command:   {cmd}")
        if t["escalate"]:
            print(ESCALATION.format(notes=t["notes"]))
            print()
            continue
        print()
        print(TEMPLATE.format(nth=t["notes"] + 1, limit=ESCALATE_AFTER_NOTES, log=log,
                              sig=t["signature"].replace('"', '\\"')))
    return 1


if __name__ == "__main__":
    sys.exit(main())
