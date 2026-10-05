#!/usr/bin/env python3
"""adr — lint architecture decision records and enforce the decision circuit breaker.

    python governance/tools/adr.py lint                     # format + Supersedes links (CI)
    python governance/tools/adr.py index                    # Markdown table of all ADRs
    python governance/tools/adr.py breaker --base-rev origin/main [--acknowledged]
    python governance/tools/adr.py breaker --base-root path/to/base/checkout
    python -I -S base/governance/tools/adr.py breaker --root pr --rev HEAD --base-root base   # CI

ADRs live in governance/decisions/ADR-NNNN-short-slug.md:

    # ADR-0007: Title
    - Status: Proposed | Accepted | Deprecated | Superseded
    - Date: YYYY-MM-DD
    - Supersedes: none | ADR-0003[, ADR-0004]
    - Superseded-by: none | ADR-0009
    ## Context / ## Decision / ## Consequences   (all required, non-empty)

The Status line holds exactly one of those words; Supersedes and Superseded-by
list ADR ids, plain (ADR-0009) or as a link to the file ([ADR-0009](ADR-0009-slug.md)).
lint also checks that supersession is recorded on both sides: if A supersedes B,
B lists A under Superseded-by and (once A is Accepted) B's status is Superseded.

Records are read from the working tree, or with --rev REV from git objects of
that revision of --root (CI's breaker does this: links and other working-tree
properties then play no part). A symbolic link, junction or submodule at
governance/, governance/decisions/, an ADR file or INVARIANTS.md, or an ADR-*.md
that is not a plain UTF-8 file, fails every command (exit 1, even with
--acknowledged): the tool would otherwise judge whatever the link points at.
With --rev, so does a commit with paths that are one file on Windows or macOS
anywhere in its tree (path_aliases.py, loaded from this tool's directory): a
governance/invariants.md or decisions/adr-0004-....md twin, or an NTFS short
name such as governance/INVARI~1.MD, would replace the real record in those
checkouts while this tool judged the real one. Any governance/decisions/*.md
whose name starts with ADR- in any case is a record, so lint reports a
misspelt one instead of skipping it.

breaker compares against a base revision (git, read-only) or a base checkout and
fails when
  * two files carry the same ADR id (a second copy must not stand in for a
    retired decision), or
  * more than --max-fraction (default 15%) of the decisions Accepted at the base
    stop being Accepted in one change; an id counts as still Accepted only if
    exactly one file carries it, or
  * an ADR that was decided at the base (any status but Proposed) is deleted,
    rewritten in place or moved back: any text changed other than the status
    word, a YYYY-MM-DD Date and the ids of Superseded-by (free text on those
    lines counts as text), or a status change other than Accepted -> Deprecated,
    Accepted -> Superseded or Deprecated -> Superseded. That is how an ADR is
    retired; demoting it to Proposed (to rewrite it in a later change) trips, or
  * governance/INVARIANTS.md changed (deleting it counts) and no ADR that was
    added, or whose title, status or sections changed, mentions INVARIANTS.md
    or one of the changed invariant ids (any INV id if only text outside the
    table changed).
Whitespace-only edits to an ADR do not count. Each case requires human review;
CI passes --acknowledged once a maintainer has labelled the pull request.

Pure standard library, Python 3.9+. Exit: 0 ok, 1 lint failure, breaker
tripped or unreadable record, 2 bad arguments or git failure.
"""
from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECISIONS = "governance/decisions"
INVARIANTS = "governance/INVARIANTS.md"
STATUSES = ("Proposed", "Accepted", "Deprecated", "Superseded")
SECTIONS = ("Context", "Decision", "Consequences")
_FILE = re.compile(r"^(ADR-\d{4})-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
_ID = re.compile(r"^ADR-\d{4}$")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_INV_ROW = re.compile(r"^\|\s*\*\*(INV-[A-Z0-9-]+)\*\*\s*\|(.*)$", re.M)
_FIELD = re.compile(r"^[-*]\s+\**(Status|Date|Supersedes|Superseded-by)\**:\s*(.*)$")
_SECTION = re.compile(r"^##\s+(.+?)\s*$")
_REF = re.compile(r"(ADR-\d{4})|\[(ADR-\d{4})\]\((?:\./)?(ADR-\d{4})-[a-z0-9]+(?:-[a-z0-9]+)*\.md\)")
# the only status changes that leave a decided ADR's meaning alone: retiring it
RETIREMENTS = {("Accepted", "Deprecated"), ("Accepted", "Superseded"), ("Deprecated", "Superseded")}
_NAME_SURROGATE = 0x20000000    # Windows reparse-tag bit shared by symbolic links and junctions


class RecordError(Exception):
    """A decision record that cannot be judged: reached through a link, or not a plain UTF-8 file."""


class AliasError(RecordError):
    """A tree whose paths are one file on Windows or macOS (path_aliases.py)."""


@dataclass
class Adr:
    id: str
    file: str
    title: str = ""
    status: str = ""
    date: str = ""
    supersedes: list = field(default_factory=list)
    superseded_by: list = field(default_factory=list)
    sections: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    frozen: str = ""        # what a decided ADR may not change (see _mask)


def _refs(value: str, where: str, errors: list) -> list:
    v = value.strip()
    if v.lower() in ("none", "-", ""):
        return []
    out = []
    for part in v.split(","):
        p = part.strip()
        m = _REF.fullmatch(p)
        if not m or (m.group(2) and m.group(2) != m.group(3)):
            errors.append(f"{where}: {p!r} is not an ADR id (ADR-NNNN), a link [ADR-NNNN](ADR-NNNN-slug.md) "
                          "or 'none'")
            continue
        out.append(m.group(1) or m.group(2))
    return out


def _mask(key: str, value: str) -> str:
    """A field line as frozen text, without what a decided ADR may change (the status word, a
    YYYY-MM-DD date, well-formed Superseded-by ids); anything else on the line stays."""
    if key == "Status":
        value = value.split(None, 1)[1] if len(value.split(None, 1)) == 2 else ""
    elif key == "Date" and _DATE.fullmatch(value):
        value = ""
    elif key == "Superseded-by":
        errors = []
        _refs(value, "", errors)
        value = value if errors else ""
    return f"- {key}: {value}"


def parse(name: str, text: str) -> Adr:
    m = _FILE.match(name)
    adr = Adr(id=m.group(1) if m else name, file=name)
    if not m:
        adr.errors.append("file name must be ADR-NNNN-lowercase-slug.md")
    lines = text.replace("\r\n", "\n").split("\n")
    head = next((l for l in lines if l.strip()), "")
    hm = re.match(r"^# (ADR-\d{4}): (.+)$", head.strip())
    if not hm:
        adr.errors.append("first line must be '# ADR-NNNN: Title'")
    else:
        adr.title = hm.group(2).strip()
        if m and hm.group(1) != adr.id:
            adr.errors.append(f"title id {hm.group(1)} does not match file name {adr.id}")
    fields = {}
    section, body, kept = None, [], []
    for line in lines:
        sm = _SECTION.match(line)
        fm = None if section or sm else _FIELD.match(line.strip())
        kept.append(_mask(fm.group(1), fm.group(2).strip()) if fm else line)
        if sm:
            if section:
                adr.sections[section] = "\n".join(body).strip()
            section, body = sm.group(1), []
            continue
        if section:
            body.append(line)
            continue
        if fm:
            fields[fm.group(1)] = fm.group(2).strip()
    if section:
        adr.sections[section] = "\n".join(body).strip()
    adr.frozen = _words("\n".join(kept))
    for k in ("Status", "Date", "Supersedes", "Superseded-by"):
        if k not in fields:
            adr.errors.append(f"missing '- {k}:' line")
    words = fields.get("Status", "").split()
    adr.status = words[0] if words else ""
    if "Status" in fields and (adr.status not in STATUSES or len(words) != 1):
        adr.errors.append(f"Status {fields['Status']!r} must be exactly one of {'|'.join(STATUSES)}")
    adr.date = fields.get("Date", "")
    if "Date" in fields:
        try:   # fromisoformat alone accepts 20261004 and 2026-W40-1 on Python 3.11+
            if not _DATE.fullmatch(adr.date):
                raise ValueError
            date.fromisoformat(adr.date)
        except ValueError:
            adr.errors.append(f"Date {adr.date!r} is not a calendar date YYYY-MM-DD")
    adr.supersedes = _refs(fields.get("Supersedes", ""), "Supersedes", adr.errors)
    adr.superseded_by = _refs(fields.get("Superseded-by", ""), "Superseded-by", adr.errors)
    for s in SECTIONS:
        if not adr.sections.get(s):
            adr.errors.append(f"missing or empty '## {s}' section")
    return adr


def _is_adr_name(name: str) -> bool:
    return name[:4].upper() == "ADR-" and name[-3:].lower() == ".md"


def _utf8(data: bytes, where: str) -> str:
    """The text of a record, CRLF read as LF (a Windows checkout says the same as a Linux one)."""
    try:
        return data.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError as e:
        raise RecordError(f"{where} is not UTF-8: {e}") from None


def is_link(path) -> bool:
    """A symbolic link, or on Windows a junction or other name-surrogate reparse point."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_reparse_tag", 0) & _NAME_SURROGATE)


def read_checkout(root) -> tuple:
    """(ADR files {name: text}, INVARIANTS.md text or None) from the working tree under root."""
    root = Path(root)
    for rel in ("governance", DECISIONS, INVARIANTS):
        if is_link(root / rel):
            raise RecordError(f"{rel} is a symbolic link or junction")
    files = {}
    d = root / DECISIONS
    for name in sorted(os.listdir(d)) if d.is_dir() else []:
        if not _is_adr_name(name):
            continue
        p = d / name
        if is_link(p) or not p.is_file():
            raise RecordError(f"{DECISIONS}/{name} is not a plain file")
        files[name] = _utf8(p.read_bytes(), f"{DECISIONS}/{name}")
    inv = root / INVARIANTS
    if inv.exists() and not inv.is_file():
        raise RecordError(f"{INVARIANTS} is not a plain file")
    return files, (_utf8(inv.read_bytes(), INVARIANTS) if inv.is_file() else None)


def _sibling(name: str):
    """A module from this tool's own directory (the same revision of the tools), loaded by path."""
    import importlib.util
    path = Path(__file__).resolve().parent / f"{name}.py"
    try:
        spec = importlib.util.spec_from_file_location(f"hearth_gov_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except (OSError, ImportError) as e:
        raise RuntimeError(f"cannot load {path}: {e}") from None
    return mod


def _git(root: Path, *args) -> bytes:
    try:
        r = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"cannot run git: {e}") from None
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout


def read_rev(root, rev: str) -> tuple:
    """(ADR files, INVARIANTS.md text or None) as committed at git revision rev of the checkout
    at root, from git objects only. RuntimeError for git failures, RecordError for links."""
    root = Path(root)
    if not rev or rev.startswith("-"):
        raise RuntimeError(f"{rev!r} is not a revision")
    if _git(root, "rev-parse", "--show-cdup").strip():
        raise RuntimeError(f"{root} is not the top level of a git checkout")
    try:
        commit = _git(root, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").decode("ascii").strip()
    except RuntimeError:
        raise RuntimeError(f"{rev!r} is not a commit in {root}") from None
    entries, paths = {}, []
    for rec in _git(root, "ls-tree", "-r", "-t", "-z", "--full-tree", commit).split(b"\0"):
        meta, _, raw = rec.partition(b"\t")
        if raw:
            paths.append(_utf8(raw, f"a path at {rev}"))
        if raw == b"governance" or raw.startswith(b"governance/"):
            mode, _, oid = meta.decode("ascii").split(" ")
            entries[paths[-1]] = (mode, oid)
    aliases = _sibling("path_aliases")
    found = aliases.describe(aliases.find(paths))
    if found:
        raise AliasError(f"paths at {rev} that are one file on Windows or macOS: {'; '.join(found)}")

    def blob(rel):
        mode, oid = entries[rel]
        if mode not in ("100644", "100755"):
            raise RecordError(f"{rel} is not a plain file at {rev} (mode {mode})")
        return _utf8(_git(root, "cat-file", "blob", oid), f"{rel} at {rev}")

    for rel in ("governance", DECISIONS):
        if entries.get(rel, ("040000",))[0] != "040000":
            raise RecordError(f"{rel} is not a directory at {rev} (mode {entries[rel][0]})")
    files = {rel.rsplit("/", 1)[1]: blob(rel) for rel in sorted(entries)
             if rel.rsplit("/", 1)[0] == DECISIONS and _is_adr_name(rel.rsplit("/", 1)[1])}
    return files, (blob(INVARIANTS) if INVARIANTS in entries else None)


def lint(files: dict) -> list:
    adrs = [parse(n, t) for n, t in sorted(files.items())]
    errors = []
    by_id = {}
    for a in adrs:
        if a.id in by_id and _ID.match(a.id):
            errors.append(f"{a.file}: duplicate id {a.id} (also {by_id[a.id].file})")
        by_id[a.id] = a
    for a in adrs:
        for e in a.errors:
            errors.append(f"{a.file}: {e}")
        for ref in a.supersedes + a.superseded_by:
            if ref == a.id:
                errors.append(f"{a.file}: refers to itself")
            elif ref not in by_id:
                errors.append(f"{a.file}: unknown ADR {ref}")
        for b_id in a.supersedes:
            b = by_id.get(b_id)
            if b is None:
                continue
            if a.id not in b.superseded_by:
                errors.append(f"{b.file}: superseded by {a.id} but 'Superseded-by' does not say so")
            if a.status == "Accepted" and b.status != "Superseded":
                errors.append(f"{b.file}: status must be Superseded (superseded by accepted {a.id})")
        for b_id in a.superseded_by:
            b = by_id.get(b_id)
            if b is not None and a.id not in b.supersedes:
                errors.append(f"{b.file}: {a.id} names it as successor but its 'Supersedes' does not list {a.id}")
        if a.status == "Superseded" and not a.superseded_by:
            errors.append(f"{a.file}: status Superseded needs a 'Superseded-by' ADR")
    return errors


def _words(text: str) -> str:
    return " ".join(text.split())


def _content(a: Adr) -> tuple:
    """What an ADR says, ignoring whitespace-only edits."""
    return (_words(a.title), a.status, tuple(sorted((k, _words(v)) for k, v in a.sections.items())))


def _inv_rows(text: str) -> dict:
    return {m.group(1): _words(m.group(2)) for m in _INV_ROW.finditer(text)}


def changed_invariants(base_inv: str, cur_inv: str) -> list:
    b, c = _inv_rows(base_inv), _inv_rows(cur_inv)
    return sorted(i for i in set(b) | set(c) if b.get(i) != c.get(i))


def _mentions(a: Adr, ids: list) -> bool:
    text = "\n".join([a.title, *a.sections.values()])
    if "INVARIANTS.md" in text:
        return True
    pat = "|".join(re.escape(i) for i in ids) if ids else r"INV-[A-Z0-9]+(?:-[A-Z0-9]+)*"
    return re.search(rf"(?<![A-Z0-9-])(?:{pat})(?![A-Z0-9-])", text) is not None


def _by_id(files: dict) -> dict:
    out = {}
    for n, t in sorted(files.items()):
        a = parse(n, t)
        out.setdefault(a.id, []).append(a)
    return out


def breaker(base_files: dict, cur_files: dict, base_inv, cur_inv, max_fraction: float) -> dict:
    base, cur = _by_id(base_files), _by_id(cur_files)
    dups = sorted(i for i, l in cur.items() if len(l) > 1)
    active = sorted(i for i, l in base.items() if any(a.status == "Accepted" for a in l))
    # retired unless exactly one file still carries the id and it is Accepted: a second, Accepted
    # copy must not hide a deprecated original
    gone = [i for i in active if len(cur.get(i, [])) != 1 or cur[i][0].status != "Accepted"]
    fraction = len(gone) / len(active) if active else 0.0
    # once decided (no longer Proposed), an ADR's text is frozen; it may only be retired
    rewritten = []
    for i, l in sorted(base.items()):
        decided = [a for a in l if a.status != "Proposed"]
        was = decided[0] if decided else None
        if was and (i not in cur or any(a.frozen != was.frozen or not (
                a.status == was.status or (was.status, a.status) in RETIREMENTS) for a in cur[i])):
            rewritten.append(i)
    changed = sorted(i for i, l in cur.items()
                     if any(all(_content(a) != _content(b) for b in base.get(i, [])) for a in l))

    def lf(text):
        return None if text is None else text.replace("\r\n", "\n")

    inv_changed = lf(base_inv) != lf(cur_inv)          # deleting or adding the file is a change too
    inv_ids = changed_invariants(base_inv or "", cur_inv or "") if inv_changed else []
    covering = [i for i in changed if any(_mentions(a, inv_ids) for a in cur[i])] if inv_changed else []
    reasons = []
    if dups:
        reasons.append(f"duplicate ADR id(s) {', '.join(dups)}: each decision must live in exactly one file")
    if fraction > max_fraction:
        reasons.append(f"{len(gone)} of {len(active)} accepted decisions ({100 * fraction:.0f}%) deprecated or "
                       f"superseded in one change (limit {100 * max_fraction:.0f}%): {', '.join(gone)}")
    if rewritten:
        reasons.append(f"decided ADR(s) {', '.join(rewritten)} rewritten in place, moved back or deleted (only the "
                       "status word may move to Deprecated or Superseded, and the Date and Superseded-by ids "
                       "may change): supersede them with a new ADR instead")
    if inv_changed and not covering:
        how = "deleted" if cur_inv is None else "added" if base_inv is None else "changed"
        what = f" ({', '.join(inv_ids)})" if inv_ids else ""
        reasons.append(f"governance/INVARIANTS.md {how}{what} but no new or changed ADR in "
                       "governance/decisions/ references INVARIANTS.md or the changed invariant ids")
    return {"active_at_base": active, "deprecated": gone, "fraction": fraction, "duplicates": dups,
            "rewritten": rewritten, "invariants_changed": inv_changed, "invariant_ids_changed": inv_ids,
            "adrs_changed": changed, "adrs_covering_invariants": covering, "tripped": reasons}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("lint", "index", "breaker"))
    ap.add_argument("--root", default=str(ROOT), help="repository root (default: this checkout)")
    ap.add_argument("--base-rev", help="breaker: git revision to compare against")
    ap.add_argument("--base-root", help="breaker: a checkout of the base to compare against")
    ap.add_argument("--max-fraction", type=float, default=0.15)
    ap.add_argument("--acknowledged", action="store_true", help="breaker: a maintainer approved this change")
    ap.add_argument("--rev", help="read the records from git revision REV of --root instead of its working tree")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if a.command == "breaker" and bool(a.base_rev) == bool(a.base_root):
        print("adr: breaker needs exactly one of --base-rev / --base-root", file=sys.stderr)
        return 2
    try:
        files, cur_inv = read_rev(root, a.rev) if a.rev is not None else read_checkout(root)
        if a.command == "breaker":
            base_files, base_inv = read_rev(root, a.base_rev) if a.base_rev else read_checkout(a.base_root)
    except RecordError as e:
        why = "" if isinstance(e, AliasError) else ": decision records must be plain files"
        print(f"adr: {e}{why} (not even --acknowledged accepts this)")
        return 1
    except (OSError, RuntimeError) as e:
        print(f"adr: {e}", file=sys.stderr)
        return 2

    if a.command == "lint":
        errors = lint(files)
        for e in errors:
            print(f"error: {e}")
        if errors:
            print(f"ADR lint FAILED: {len(errors)} problem(s)")
            return 1
        print(f"ADR lint OK: {len(files)} decision record(s)")
        return 0

    if a.command == "index":
        print("| ADR | Title | Status | Supersedes | Superseded by |")
        print("|-----|-------|--------|------------|---------------|")
        for n, t in sorted(files.items()):
            x = parse(n, t)
            print(f"| [{x.id}]({x.file}) | {x.title} | {x.status} | {', '.join(x.supersedes) or '-'} | "
                  f"{', '.join(x.superseded_by) or '-'} |")
        return 0

    res = breaker(base_files, files, base_inv, cur_inv, a.max_fraction)
    if not res["tripped"]:
        print(f"decision breaker OK: {len(res['deprecated'])} of {len(res['active_at_base'])} accepted "
              f"decisions retired; invariants {'changed with an ADR' if res['invariants_changed'] else 'unchanged'}")
        return 0
    for r in res["tripped"]:
        print(f"CIRCUIT BREAKER: {r}")
    if a.acknowledged:
        print("acknowledged by a maintainer; continuing")
        return 0
    print("This change needs human review before it can merge (a maintainer adds the "
          "'decisions-reviewed' label).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
