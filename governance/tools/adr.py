#!/usr/bin/env python3
"""adr — lint architecture decision records and enforce the decision circuit breaker.

    python governance/tools/adr.py lint                     # format + Supersedes links (CI)
    python governance/tools/adr.py index                    # Markdown table of all ADRs
    python governance/tools/adr.py breaker --base-rev origin/main [--acknowledged]
    python governance/tools/adr.py breaker --base-root path/to/base/checkout

ADRs live in governance/decisions/ADR-NNNN-short-slug.md:

    # ADR-0007: Title
    - Status: Proposed | Accepted | Deprecated | Superseded
    - Date: YYYY-MM-DD
    - Supersedes: none | ADR-0003[, ADR-0004]
    - Superseded-by: none | ADR-0009
    ## Context / ## Decision / ## Consequences   (all required, non-empty)

lint also checks that supersession is recorded on both sides: if A supersedes B,
B lists A under Superseded-by and (once A is Accepted) B's status is Superseded.

breaker compares against a base revision (git, read-only) or a base checkout and
fails when
  * more than --max-fraction (default 15%) of the decisions Accepted at the base
    stop being Accepted in one change, or
  * governance/INVARIANTS.md changed (deleting it counts) and no ADR that was
    added, or whose title, status or sections changed, mentions INVARIANTS.md
    or one of the changed invariant ids (any INV id if only text outside the
    table changed).
    Whitespace-only edits to an ADR do not count.
Both require human review; CI passes --acknowledged once a maintainer has
labelled the pull request.

Pure standard library, Python 3.9+. Exit: 0 ok, 1 lint failure / breaker
tripped, 2 bad arguments.
"""
from __future__ import annotations

import argparse
import re
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


def _refs(value: str, where: str, errors: list) -> list:
    v = value.strip()
    if v.lower() in ("none", "-", ""):
        return []
    out = []
    for part in v.split(","):
        p = part.strip()
        m = re.match(r"^\[?(ADR-\d{4})\]?(\(.*\))?$", p)
        if not m:
            errors.append(f"{where}: {p!r} is not an ADR id (ADR-NNNN) or 'none'")
            continue
        out.append(m.group(1))
    return out


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
    section, body = None, []
    for line in lines:
        sm = re.match(r"^##\s+(.+?)\s*$", line)
        if sm:
            if section:
                adr.sections[section] = "\n".join(body).strip()
            section, body = sm.group(1), []
            continue
        if section:
            body.append(line)
            continue
        fm = re.match(r"^[-*]\s+\**(Status|Date|Supersedes|Superseded-by)\**:\s*(.*)$", line.strip())
        if fm:
            fields[fm.group(1)] = fm.group(2).strip()
    if section:
        adr.sections[section] = "\n".join(body).strip()
    for k in ("Status", "Date", "Supersedes", "Superseded-by"):
        if k not in fields:
            adr.errors.append(f"missing '- {k}:' line")
    adr.status = fields.get("Status", "").split()[0] if fields.get("Status") else ""
    if "Status" in fields and adr.status not in STATUSES:
        adr.errors.append(f"Status {fields['Status']!r} not in {'|'.join(STATUSES)}")
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


def load_dir(d: Path) -> dict:
    out = {}
    if d.is_dir():
        for p in sorted(d.glob("ADR-*.md")):
            out[p.name] = p.read_text(encoding="utf-8")
    return out


def load_rev(root: Path, rev: str, rel: str) -> dict:
    def git(*args):
        r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
        return r.stdout
    out = {}
    for path in git("ls-tree", "-r", "--name-only", rev, "--", rel).splitlines():
        name = path.rsplit("/", 1)[-1]
        if name.startswith("ADR-") and name.endswith(".md"):
            out[name] = git("show", f"{rev}:{path}")
    return out


def show_rev(root: Path, rev: str, rel: str):
    r = subprocess.run(["git", "-C", str(root), "show", f"{rev}:{rel}"], capture_output=True, text=True,
                       encoding="utf-8")
    return r.stdout if r.returncode == 0 else None


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


def breaker(base_files: dict, cur_files: dict, base_inv, cur_inv, max_fraction: float) -> dict:
    base = {a.id: a for a in (parse(n, t) for n, t in base_files.items())}
    cur = {a.id: a for a in (parse(n, t) for n, t in cur_files.items())}
    active = sorted(i for i, a in base.items() if a.status == "Accepted")
    gone = [i for i in active if i not in cur or cur[i].status != "Accepted"]
    fraction = len(gone) / len(active) if active else 0.0
    changed = sorted(i for i, a in cur.items() if i not in base or _content(base[i]) != _content(a))
    def lf(text):
        return None if text is None else text.replace("\r\n", "\n")

    inv_changed = lf(base_inv) != lf(cur_inv)          # deleting or adding the file is a change too
    inv_ids = changed_invariants(base_inv or "", cur_inv or "") if inv_changed else []
    covering = [i for i in changed if _mentions(cur[i], inv_ids)] if inv_changed else []
    reasons = []
    if fraction > max_fraction:
        reasons.append(f"{len(gone)} of {len(active)} accepted decisions ({100 * fraction:.0f}%) deprecated or "
                       f"superseded in one change (limit {100 * max_fraction:.0f}%): {', '.join(gone)}")
    if inv_changed and not covering:
        how = "deleted" if cur_inv is None else "added" if base_inv is None else "changed"
        what = f" ({', '.join(inv_ids)})" if inv_ids else ""
        reasons.append(f"governance/INVARIANTS.md {how}{what} but no new or changed ADR in "
                       "governance/decisions/ references INVARIANTS.md or the changed invariant ids")
    return {"active_at_base": active, "deprecated": gone, "fraction": fraction,
            "invariants_changed": inv_changed, "invariant_ids_changed": inv_ids, "adrs_changed": changed,
            "adrs_covering_invariants": covering, "tripped": reasons}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("lint", "index", "breaker"))
    ap.add_argument("--root", default=str(ROOT), help="repository root (default: this checkout)")
    ap.add_argument("--base-rev", help="breaker: git revision to compare against")
    ap.add_argument("--base-root", help="breaker: a checkout of the base to compare against")
    ap.add_argument("--max-fraction", type=float, default=0.15)
    ap.add_argument("--acknowledged", action="store_true", help="breaker: a maintainer approved this change")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    files = load_dir(root / DECISIONS)

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

    if bool(a.base_rev) == bool(a.base_root):
        print("adr: breaker needs exactly one of --base-rev / --base-root", file=sys.stderr)
        return 2
    try:
        if a.base_rev:
            base_files = load_rev(root, a.base_rev, DECISIONS)
            base_inv = show_rev(root, a.base_rev, INVARIANTS)
        else:
            br = Path(a.base_root)
            base_files = load_dir(br / DECISIONS)
            base_inv = (br / INVARIANTS).read_text(encoding="utf-8") if (br / INVARIANTS).is_file() else None
    except (OSError, RuntimeError) as e:
        print(f"adr: {e}", file=sys.stderr)
        return 2
    cur_inv = (root / INVARIANTS).read_text(encoding="utf-8") if (root / INVARIANTS).is_file() else None
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
