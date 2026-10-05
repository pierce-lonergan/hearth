#!/usr/bin/env python3
"""check_golden — verify the hash lock on maintainer-owned golden tests (INV-VERIFY).

    python governance/tools/check_golden.py            # CI: exit 1 on any mismatch
    python -E governance/tools/check_golden.py --run   # CI: verify the lock, then run the suite
    python governance/tools/check_golden.py --update --i-am-a-maintainer   # rewrite the lock

tests/golden/MANIFEST.sha256 holds one line per locked file:

    <sha256 hex>  <repo-relative path>

The hash is SHA-256 of the file bytes with CRLF normalised to LF, so a Windows
checkout with autocrlf hashes the same as Linux. The check fails if a listed
file is missing or differs, if the manifest is malformed or points outside
tests/golden/, if tests/golden/ contains a file the manifest does not list, or
if it contains a symbolic link or junction (it is never followed, but pytest
would collect through it). Exempt from the "unlisted" rule: MANIFEST.sha256
itself, an *empty* __init__.py (a non-empty one runs code at import, so it must
be locked like any test) and __pycache__/, except that an unchecked-hash .pyc
fails the check: Python loads one without comparing it to its source.

--run verifies the lock and then runs the golden suite from a new temporary
directory holding only the locked files (hashed again as they are copied) and
an empty __init__.py wherever tests/golden/ has one. Nothing else in the
repository reaches the run: no conftest.py, ini file or __init__.py outside the
locked set, no PYTEST_* variables, no auto-loaded plugins, no stale bytecode
(-X pycache_prefix points at an empty directory) and, with `python -E`, no
PYTHONPATH. Of the repository's python/ directory only the hearth package can
be imported, so a python/numpy.py cannot shadow numpy. The run fails unless at
least one test ran and none failed, errored, was skipped or xfailed. The engine
library is found as usual (HEARTH_LIB).

What --run cannot isolate: the code under test. The hearth package and the
engine library it loads run inside the pytest process and could tamper with
pytest or its report; changes there are caught by review, not by this tool.
Neither can the tool check itself, so CI runs the copy from the pull request's
base revision.

--update is for maintainers changing golden tests in a reviewed commit (with an
ADR when an invariant changes). Contributors and agents never run it.

Standard library only (--run needs pytest), Python 3.9+. Exit: 0 ok,
1 mismatch or golden-suite failure, 2 malformed manifest or bad arguments.
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":      # a module planted next to this script must not shadow the standard library
    _HERE = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:] = [p for p in sys.path if os.path.realpath(p or os.curdir) != _HERE]

import argparse
import hashlib
import re
import stat
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = "tests/golden"
MANIFEST_NAME = "MANIFEST.sha256"
PACKAGES = ("hearth",)          # what the golden suite may import from <root>/python
_LINE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")
_NAME_SURROGATE = 0x20000000    # Windows reparse-tag bit shared by symbolic links and junctions

BANNER = """\
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!  check_golden --update rewrites the golden-test lock (INV-VERIFY).      !!
!!  MAINTAINERS ONLY. Contributors and AI agents must never run this.      !!
!!  Golden tests define correctness; re-locking them must land in its own  !!
!!  reviewed commit, with an ADR in governance/decisions/ if an invariant  !!
!!  changes.                                                               !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"""


class ManifestError(Exception):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def file_hash(path: Path) -> str:
    return digest(path.read_bytes())


def _safe_rel(rel: str) -> bool:
    parts = rel.split("/")
    return bool(rel) and not rel.startswith("/") and ":" not in parts[0] and ".." not in parts and "" not in parts


def parse_manifest(text: str) -> list:
    """Returns [(sha256, rel_path)]; raises ManifestError on malformed lines."""
    entries, seen = [], set()
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            raise ManifestError(f"line {n}: expected '<sha256>  <path>', got {line!r}")
        sha, rel = m.group(1).lower(), m.group(2).strip().replace("\\", "/")
        if not _safe_rel(rel) or not rel.startswith(GOLDEN + "/"):
            raise ManifestError(f"line {n}: {rel!r} is not a path under {GOLDEN}/")
        if rel in seen:
            raise ManifestError(f"line {n}: duplicate entry for {rel}")
        seen.add(rel)
        entries.append((sha, rel))
    return entries


def is_link(path) -> bool:
    """A symbolic link, or on Windows a junction or other name-surrogate reparse point."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_reparse_tag", 0) & _NAME_SURROGATE)


def unchecked_pyc(path: Path) -> bool:
    """A hash-based .pyc with check_source off (PEP 552): imported without reading the source."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return False
    return len(head) == 8 and int.from_bytes(head[4:8], "little") & 0b11 == 0b01


def scan(root: Path) -> dict:
    """Walks tests/golden/ without following links. Repo-relative paths of: files that need
    a lock entry, exempt empty __init__.py files, links, unchecked-hash .pyc files."""
    out = {"files": [], "inits": [], "links": [], "bytecode": []}
    base = Path(root) / GOLDEN
    if is_link(base):
        out["links"].append(GOLDEN)
        return out
    stack = [(base, False)] if base.is_dir() else []
    while stack:
        d, in_cache = stack.pop()
        with os.scandir(d) as it:
            entries = list(it)
        for e in entries:
            p = Path(e.path)
            rel = p.relative_to(root).as_posix()
            if is_link(p):
                out["links"].append(rel)
            elif e.is_dir(follow_symlinks=False):
                stack.append((p, in_cache or e.name == "__pycache__"))
            elif in_cache:
                if e.name.endswith(".pyc") and unchecked_pyc(p):
                    out["bytecode"].append(rel)
            elif rel == f"{GOLDEN}/{MANIFEST_NAME}":
                continue
            elif e.name == "__init__.py" and not p.read_bytes().strip():
                out["inits"].append(rel)
            else:
                out["files"].append(rel)
    return {k: sorted(v) for k, v in out.items()}


def golden_files(root: Path) -> list:
    """Repo-relative paths of every file under tests/golden/ that must be locked."""
    return scan(root)["files"]


def check(root: Path) -> dict:
    root = Path(root)
    manifest = root / GOLDEN / MANIFEST_NAME
    res = {"root": str(root), "ok": [], "mismatch": [], "missing": [], "unlisted": [], "links": [],
           "bytecode": [], "inits": [], "locked": [], "errors": []}
    if not manifest.is_file():
        res["errors"].append(f"{manifest} not found")
        return res
    try:
        entries = parse_manifest(manifest.read_text(encoding="utf-8"))
    except (ManifestError, UnicodeDecodeError) as e:
        res["errors"].append(f"{GOLDEN}/{MANIFEST_NAME}: {e}")
        return res
    if not entries:
        res["errors"].append(f"{GOLDEN}/{MANIFEST_NAME} locks no files")
    res["locked"] = entries
    for sha, rel in entries:
        p = root / rel
        if not p.is_file():
            res["missing"].append(rel)
            continue
        actual = file_hash(p)
        if actual != sha:
            res["mismatch"].append({"path": rel, "expected": sha, "actual": actual})
        else:
            res["ok"].append(rel)
    found = scan(root)
    listed = {rel for _, rel in entries}
    res["unlisted"] = [rel for rel in found["files"] if rel not in listed]
    for k in ("links", "bytecode", "inits"):
        res[k] = found[k]
    return res


def passed(res: dict) -> bool:
    return not any(res[k] for k in ("errors", "mismatch", "missing", "unlisted", "links", "bytecode"))


def render_manifest(root: Path) -> str:
    return "".join(f"{file_hash(root / rel)}  {rel}\n" for rel in golden_files(root))


def stage(root: Path, dst: Path, res: dict) -> None:
    """Copies the locked files of a passed check() into dst, hashing the bytes actually
    copied, and creates an empty __init__.py wherever tests/golden/ has one."""
    for sha, rel in res["locked"]:
        data = (root / rel).read_bytes()
        if digest(data) != sha:
            raise ManifestError(f"{rel} changed after the lock was verified")
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    for rel in res["inits"]:
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            out.write_bytes(b"")


# Runs inside `python -E -c`. sys.argv: python dir, package names, pytest arguments.
_RUNNER = """\
import importlib.machinery, os, sys
cwd = os.path.abspath(os.getcwd())
sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != cwd]
import pytest

class RepositoryPackages:
    where, names = [sys.argv[1]], set(sys.argv[2].split(","))

    @classmethod
    def find_spec(cls, name, path=None, target=None):
        if path is None and name in cls.names:
            return importlib.machinery.PathFinder.find_spec(name, cls.where)
        return None

sys.meta_path.insert(0, RepositoryPackages)
sys.exit(pytest.main(sys.argv[3:]))
"""


def junit_counts(xml_path: Path) -> dict:
    """Counts testcases by outcome in a pytest JUnit XML report."""
    counts = {"tests": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    tree = ET.parse(str(xml_path))
    for case in tree.iter("testcase"):
        counts["tests"] += 1
        tags = {child.tag for child in case}
        if "failure" in tags:
            counts["failed"] += 1
        elif "error" in tags:
            counts["errors"] += 1
        elif "skipped" in tags:
            counts["skipped"] += 1
        else:
            counts["passed"] += 1
    for suite in tree.iter("testsuite"):    # collection errors produce no testcase
        counts["errors"] = max(counts["errors"], int(suite.get("errors", "0") or 0))
    return counts


def _tempdir():
    extra = {"ignore_cleanup_errors": True} if sys.version_info >= (3, 10) else {}
    return tempfile.TemporaryDirectory(prefix="hearth-golden-", **extra)


def run_suite(root: Path, res: dict, timeout: float = 3600.0) -> tuple:
    """Runs the locked golden suite in isolation. Returns (passed, counts, pytest_output);
    raises ManifestError if a locked file changed after check()."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    with _tempdir() as tmp:
        tmp = Path(tmp).resolve()
        run = tmp / "run"
        stage(root, run, res)
        ini = run / "pytest.ini"
        ini.write_text("[pytest]\n", encoding="utf-8")
        xml = tmp / "golden.xml"
        cmd = [sys.executable, "-E", "-X", "utf8", "-X", f"pycache_prefix={tmp / 'pyc'}", "-c", _RUNNER,
               str(root / "python"), ",".join(PACKAGES), str(run / GOLDEN),
               "-c", str(ini), "--rootdir", str(run), "--confcutdir", str(run), "-p", "no:cacheprovider",
               "-o", "addopts=", "-rfEsxX", "-q", f"--junitxml={xml}"]
        try:
            r = subprocess.run(cmd, cwd=str(run), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=timeout)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode("utf-8", "replace")
            return False, None, out + f"\ngolden suite timed out after {timeout:.0f} s"
        out = r.stdout.decode("utf-8", "replace")
        if not xml.is_file():
            return False, None, out
        counts = junit_counts(xml)
    ok = (r.returncode == 0 and counts["tests"] > 0 and counts["passed"] == counts["tests"]
          and counts["errors"] == 0)
    return ok, counts, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT), help="repository root (default: this checkout)")
    ap.add_argument("--update", action="store_true", help="MAINTAINERS ONLY: rewrite MANIFEST.sha256")
    ap.add_argument("--run", action="store_true", help="after verifying the lock, run the golden suite in isolation")
    ap.add_argument("--timeout", type=float, default=3600.0, help="--run: seconds before the suite is stopped")
    ap.add_argument("--i-am-a-maintainer", action="store_true", dest="maintainer",
                    help="required with --update (or HEARTH_MAINTAINER=1)")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()

    if a.update:
        print(BANNER, file=sys.stderr)
        if not (a.maintainer or os.environ.get("HEARTH_MAINTAINER") == "1"):
            print("refusing: pass --i-am-a-maintainer (or set HEARTH_MAINTAINER=1) to confirm.", file=sys.stderr)
            return 2
        if not (root / GOLDEN).is_dir():
            print(f"check_golden: {root / GOLDEN} does not exist", file=sys.stderr)
            return 2
        manifest = root / GOLDEN / MANIFEST_NAME
        old = manifest.read_text(encoding="utf-8") if manifest.is_file() else ""
        new = render_manifest(root)
        with open(manifest, "w", encoding="utf-8", newline="\n") as f:
            f.write(new)
        before = set(old.splitlines())
        after = set(new.splitlines())
        for line in sorted(before - after):
            print(f"  - {line}", file=sys.stderr)
        for line in sorted(after - before):
            print(f"  + {line}", file=sys.stderr)
        print(f"rewrote {manifest} ({len(after)} file(s)); commit it separately for review.", file=sys.stderr)
        return 0

    res = check(root)
    for e in res["errors"]:
        print(f"error: {e}")
    for m in res["mismatch"]:
        print(f"MODIFIED  {m['path']}\n          expected {m['expected']}\n          actual   {m['actual']}")
    for rel in res["missing"]:
        print(f"MISSING   {rel}")
    for rel in res["unlisted"]:
        print(f"UNLISTED  {rel}  (not in {GOLDEN}/{MANIFEST_NAME})")
    for rel in res["links"]:
        print(f"LINK      {rel}  (symbolic links and junctions are not allowed in {GOLDEN}/)")
    for rel in res["bytecode"]:
        print(f"BYTECODE  {rel}  (unchecked-hash .pyc: Python would run it instead of its source)")
    if res["errors"]:
        return 2
    if not passed(res):
        print(f"golden lock BROKEN: {len(res['mismatch'])} modified, {len(res['missing'])} missing, "
              f"{len(res['unlisted'])} unlisted, {len(res['links'])} link(s), {len(res['bytecode'])} unchecked "
              ".pyc. Golden tests are maintainer-owned (INV-VERIFY).")
        return 1
    if not a.quiet:
        print(f"golden lock OK: {len(res['ok'])} file(s) verified")
    if not a.run:
        return 0
    try:
        ok, counts, out = run_suite(root, res, a.timeout)
    except (ManifestError, OSError) as e:
        print(f"golden suite FAILED: {e}")
        return 1
    print(out.rstrip())
    if counts is None:
        print("golden suite FAILED: pytest produced no report (timeout, interpreter exit, or pytest missing)")
        return 1
    summary = ", ".join(f"{counts[k]} {k}" for k in ("passed", "failed", "errors", "skipped"))
    if not ok:
        why = "nothing ran" if counts["tests"] == 0 else "every golden test must run and pass; skips count as failures"
        print(f"golden suite FAILED ({summary}): {why}")
        return 1
    print(f"golden suite OK: {summary} (isolated: locked files only, no outside conftest, ini, plugins or bytecode)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
