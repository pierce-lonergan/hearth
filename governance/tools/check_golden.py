#!/usr/bin/env python3
"""check_golden — verify the hash lock on maintainer-owned golden tests (INV-VERIFY).

    python governance/tools/check_golden.py            # exit 1 on any mismatch with the lock
    python -E governance/tools/check_golden.py --run   # verify the lock, then run the suite
    python governance/tools/check_golden.py --base-rev origin/main     # also compare with the base's lock
    python -I -S base/governance/tools/check_golden.py --root pr --rev HEAD \\
        --base-manifest base/tests/golden/MANIFEST.sha256 [--acknowledged]   # CI (golden.yml)
    python governance/tools/check_golden.py --update --i-am-a-maintainer   # rewrite the lock

tests/golden/MANIFEST.sha256 holds one line per locked file:

    <sha256 hex>  <repo-relative path>

The hash is SHA-256 of the file bytes with CRLF normalised to LF, so a Windows
checkout with autocrlf hashes the same as Linux. The check fails if a listed
file is missing or differs, if the manifest is malformed or points outside
tests/golden/, if tests/golden/ contains a file the manifest does not list, or
if tests/, tests/golden/ or anything below it is a symbolic link or junction
(never followed: a linked tests/ could point the check at another checkout,
and pytest would collect through a link). Exempt from the "unlisted" rule:
MANIFEST.sha256 itself, an *empty* __init__.py (a non-empty one runs code at
import, so it must be locked like any test) and __pycache__/, except that an
unchecked-hash .pyc fails the check: Python loads one without comparing it to
its source.

--rev REV reads tests/golden/ from git revision REV of the --root checkout
instead of its working tree: only git objects are read, so links, attributes
and anything else in the working tree play no part, and a link or submodule
entry (mode 120000 or 160000) at tests/, tests/golden/ or below fails. So does
a commit with paths that are one file on Windows or macOS anywhere in its tree
(case or Unicode twins, NTFS 8.3 short names; see path_aliases.py, loaded from
this tool's directory): there a twin of a golden file or of tests/golden/ itself
would replace what was judged. Neither is accepted with --acknowledged. CI's
golden gate judges the pull request this way.

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

The lock in the checkout only proves that the files match the manifest next to
them, and a change can rewrite both. --base-manifest FILE (a manifest taken from
the base revision) or --base-rev REV (read with git from the --root checkout)
adds the check that matters for review: every file the base locked is present
with the base's hash, no golden file was added, and the manifest itself is
unchanged (compared after CRLF normalisation). Any difference fails unless
--acknowledged says that a maintainer reviewed it; the run then continues
against the change's own lock. A base revision without a manifest locked
nothing, so every golden file counts as added. The tool cannot check itself
either: CI runs the copy from the base revision (see governance/README.md for
what that relies on).

--update is for maintainers changing golden tests in a reviewed commit (with an
ADR when an invariant changes). Contributors and agents never run it; the
confirmation flag is an honour system, which is why CI compares with the base.

Standard library only (--run needs pytest; --rev and --base-rev need git), Python 3.9+.
Exit: 0 ok, 1 mismatch, change against the base without --acknowledged, or
golden-suite failure, 2 malformed manifest or bad arguments.
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


def _sibling(name: str):
    """A module from this tool's own directory (the same revision of the tools), loaded by
    path: that directory is deliberately not on sys.path."""
    import importlib.util
    path = Path(__file__).resolve().parent / f"{name}.py"
    try:
        spec = importlib.util.spec_from_file_location(f"hearth_gov_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except (OSError, ImportError) as e:
        raise ManifestError(f"cannot load {path}: {e}") from None
    return mod


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


def _unchecked_head(head: bytes) -> bool:
    return len(head) >= 8 and head[4] & 0b11 == 0b01           # flags word, little-endian: hash-based, unchecked


def unchecked_pyc(path: Path) -> bool:
    """A hash-based .pyc with check_source off (PEP 552): imported without reading the source."""
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return False
    return _unchecked_head(head)


# tests, tests/golden: every directory on the way to the golden suite
_COMPONENTS = (GOLDEN.split("/")[0], GOLDEN)


def scan(root: Path) -> dict:
    """Walks tests/golden/ without following links. Repo-relative paths of: files that need
    a lock entry, exempt empty __init__.py files, links, unchecked-hash .pyc files. A link on
    the way to tests/golden/ is reported as the only link, and nothing behind it is read."""
    out = {"files": [], "inits": [], "links": [], "bytecode": [], "aliases": []}
    for rel in _COMPONENTS:
        if is_link(Path(root) / rel):
            out["links"].append(rel)
            return out
    base = Path(root) / GOLDEN
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


def _git(root: Path, *args) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise ManifestError(f"cannot run git: {e}") from None


def resolve_commit(root: Path, rev: str, option: str) -> str:
    """The commit `rev` names in the git checkout whose top level is root."""
    if not rev or rev.startswith("-"):
        raise ManifestError(f"{option} {rev!r} is not a revision")
    r = _git(root, "rev-parse", "--show-cdup")      # "" at the top level; a subdirectory would read its parent repo
    if r.returncode or r.stdout.strip():
        raise ManifestError(f"{root} is not the top level of a git checkout")
    r = _git(root, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
    if r.returncode:
        raise ManifestError(f"{option} {rev!r} is not a commit in {root}")
    return r.stdout.decode("ascii", "replace").strip()


class Checkout:
    """tests/golden/ as files in the working tree under root."""

    def __init__(self, root):
        self.root = Path(root)

    def __str__(self):
        return str(self.root)

    def scan(self) -> dict:
        return scan(self.root)

    def read(self, rel: str):
        p = self.root / rel
        return p.read_bytes() if p.is_file() else None


class Revision:
    """tests/golden/ as committed at a git revision of the checkout at root. Only git objects
    are read: links, attributes and every other property of the working tree play no part."""

    def __init__(self, root, rev: str):
        self.root, self.rev = Path(root), rev
        self.commit = resolve_commit(self.root, rev, "--rev")
        r = _git(self.root, "ls-tree", "-r", "-t", "-z", "--full-tree", self.commit)
        if r.returncode:
            raise ManifestError(f"git ls-tree {rev}: {r.stderr.decode('utf-8', 'replace').strip()}")
        self.entries = {}       # path -> (mode, object id) for tests, tests/golden and everything below
        paths = []
        for rec in r.stdout.split(b"\0"):
            meta, _, raw = rec.partition(b"\t")
            if not raw:
                continue
            try:
                path = raw.decode("utf-8")
            except UnicodeDecodeError:
                raise ManifestError(f"{rev}: a path is not UTF-8: {raw!r}") from None
            paths.append(path)
            if path in _COMPONENTS or path.startswith(GOLDEN + "/"):
                mode, _, oid = meta.decode("ascii").split(" ")
                self.entries[path] = (mode, oid)
        aliases = _sibling("path_aliases")
        self.aliases = aliases.describe(aliases.find(paths))      # anywhere in the tree
        self._blobs = {}

    def __str__(self):
        return f"{self.root} at {self.rev} ({self.commit[:12]})"

    def _blob(self, oid: str) -> bytes:
        if oid not in self._blobs:
            r = _git(self.root, "cat-file", "blob", oid)       # raw bytes: no filters or textconv
            if r.returncode:
                raise ManifestError(f"git cat-file blob {oid}: {r.stderr.decode('utf-8', 'replace').strip()}")
            self._blobs[oid] = r.stdout
        return self._blobs[oid]

    def scan(self) -> dict:
        out = {"files": [], "inits": [], "links": [], "bytecode": [], "aliases": self.aliases}
        for rel in _COMPONENTS:
            mode = self.entries.get(rel, ("040000",))[0]
            if mode in ("120000", "160000"):              # symbolic link, submodule
                out["links"].append(rel)
                return out
            if mode != "040000":                          # a file: there is no golden directory
                return out
        for rel, (mode, oid) in sorted(self.entries.items()):
            if rel in _COMPONENTS or mode == "040000":
                continue
            parts = rel.split("/")
            if mode in ("120000", "160000"):
                out["links"].append(rel)
            elif "__pycache__" in parts[:-1]:
                if parts[-1].endswith(".pyc") and _unchecked_head(self._blob(oid)):
                    out["bytecode"].append(rel)
            elif rel == f"{GOLDEN}/{MANIFEST_NAME}":
                continue
            elif parts[-1] == "__init__.py" and not self._blob(oid).strip():
                out["inits"].append(rel)
            else:
                out["files"].append(rel)
        return out

    def read(self, rel: str):
        mode, oid = self.entries.get(rel, ("", ""))
        return self._blob(oid) if mode in ("100644", "100755") else None


def _source(root):
    return root if isinstance(root, (Checkout, Revision)) else Checkout(root)


def check(root) -> dict:
    """The lock of root: a path (the working tree), a Checkout or a Revision."""
    src = _source(root)
    res = {"root": str(src), "ok": [], "mismatch": [], "missing": [], "unlisted": [], "links": [],
           "bytecode": [], "aliases": [], "inits": [], "files": [], "locked": [], "errors": []}
    found = src.scan()
    for k in ("links", "bytecode", "aliases", "inits", "files"):
        res[k] = found[k]
    if set(found["links"]) & set(_COMPONENTS):           # never read the lock through a link
        return res
    data = src.read(f"{GOLDEN}/{MANIFEST_NAME}")
    if data is None:
        res["errors"].append(f"{GOLDEN}/{MANIFEST_NAME} not found in {src}")
        return res
    try:
        entries = parse_manifest(data.decode("utf-8"))
    except (ManifestError, UnicodeDecodeError) as e:
        res["errors"].append(f"{GOLDEN}/{MANIFEST_NAME}: {e}")
        return res
    if not entries:
        res["errors"].append(f"{GOLDEN}/{MANIFEST_NAME} locks no files")
    res["locked"] = entries
    for sha, rel in entries:
        data = src.read(rel)
        if data is None:
            res["missing"].append(rel)
            continue
        actual = digest(data)
        if actual != sha:
            res["mismatch"].append({"path": rel, "expected": sha, "actual": actual})
        else:
            res["ok"].append(rel)
    listed = {rel for _, rel in entries}
    res["unlisted"] = [rel for rel in found["files"] if rel not in listed]
    return res


def passed(res: dict) -> bool:
    return not any(res[k] for k in ("errors", "mismatch", "missing", "unlisted", "links", "bytecode", "aliases"))


def _decode(data: bytes, where: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ManifestError(f"{where}: {e}") from None


def base_manifest_at(root: Path, rev: str):
    """The manifest text at git revision `rev` of the checkout at root, or None if that
    revision has no manifest. Raises ManifestError if rev is not a commit there."""
    sha = resolve_commit(Path(root), rev, "--base-rev")
    r = _git(Path(root), "cat-file", "blob", f"{sha}:{GOLDEN}/{MANIFEST_NAME}")   # raw bytes: no filters or textconv
    return _decode(r.stdout, f"{GOLDEN}/{MANIFEST_NAME} at {rev}") if r.returncode == 0 else None


def compare_base(root, res: dict, base_text) -> dict:
    """What the change (a path, Checkout or Revision) changes relative to the base revision's
    lock (base_text None: the base had no manifest). res is check()'s result for the same
    source; links were rejected there."""
    src = _source(root)
    base = parse_manifest(base_text) if base_text is not None else []
    head = src.read(f"{GOLDEN}/{MANIFEST_NAME}")
    delta = {"manifest_changed": head is None or base_text is None or digest(head) != digest(base_text.encode()),
             "modified": [], "removed": [], "added": []}
    for sha, rel in base:
        data = src.read(rel)
        if data is None:
            delta["removed"].append(rel)
        elif digest(data) != sha:
            delta["modified"].append(rel)
    locked = {rel for _, rel in base}
    delta["added"] = [rel for rel in res["files"] if rel not in locked]
    return delta


def changed(delta: dict) -> bool:
    return any(delta[k] for k in ("manifest_changed", "modified", "removed", "added"))


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
    base_src = ap.add_mutually_exclusive_group()
    base_src.add_argument("--base-manifest", metavar="FILE", help="the manifest of the base revision to compare with")
    base_src.add_argument("--base-rev", metavar="REV", help="read the base revision's manifest with git from --root")
    ap.add_argument("--acknowledged", action="store_true",
                    help="a maintainer reviewed the golden change against the base (CI: 'golden-reviewed' label)")
    ap.add_argument("--rev", metavar="REV",
                    help="judge tests/golden/ as committed at REV of --root (git objects only, not the working tree)")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    with_base = a.base_manifest is not None or a.base_rev is not None
    if a.acknowledged and not with_base:
        print("check_golden: --acknowledged needs --base-manifest or --base-rev", file=sys.stderr)
        return 2
    if a.update and with_base:
        print("check_golden: --update does not take a base", file=sys.stderr)
        return 2
    if a.rev is not None and (a.run or a.update):
        print("check_golden: --rev reads git objects; --run and --update work on the working tree", file=sys.stderr)
        return 2

    if a.update:
        print(BANNER, file=sys.stderr)
        if not (a.maintainer or os.environ.get("HEARTH_MAINTAINER") == "1"):
            print("refusing: pass --i-am-a-maintainer (or set HEARTH_MAINTAINER=1) to confirm.", file=sys.stderr)
            return 2
        if not (root / GOLDEN).is_dir():
            print(f"check_golden: {root / GOLDEN} does not exist", file=sys.stderr)
            return 2
        links = scan(root)["links"]
        if links:
            print(f"check_golden: refusing to lock through links: {', '.join(links)}", file=sys.stderr)
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

    base_text, base_name = None, None
    try:
        if a.base_manifest is not None:
            base_name = a.base_manifest
            base_text = _decode(Path(a.base_manifest).read_bytes(), a.base_manifest)
        elif a.base_rev is not None:
            base_name = f"{GOLDEN}/{MANIFEST_NAME} at {a.base_rev}"
            base_text = base_manifest_at(root, a.base_rev)
        if base_text is not None:
            parse_manifest(base_text)
    except (OSError, ManifestError) as e:
        print(f"check_golden: base manifest: {e}", file=sys.stderr)
        return 2
    try:
        src = Revision(root, a.rev) if a.rev is not None else Checkout(root)
        res = check(src)
        delta = compare_base(src, res, base_text) if with_base and passed(res) else None
    except ManifestError as e:
        print(f"check_golden: {e}", file=sys.stderr)
        return 2
    for e in res["errors"]:
        print(f"error: {e}")
    for m in res["mismatch"]:
        print(f"MODIFIED  {m['path']}\n          expected {m['expected']}\n          actual   {m['actual']}")
    for rel in res["missing"]:
        print(f"MISSING   {rel}")
    for rel in res["unlisted"]:
        print(f"UNLISTED  {rel}  (not in {GOLDEN}/{MANIFEST_NAME})")
    for rel in res["links"]:
        print(f"LINK      {rel}  (symbolic links, junctions and submodules are not allowed in or on the way "
              f"to {GOLDEN}/)")
    for rel in res["bytecode"]:
        print(f"BYTECODE  {rel}  (unchecked-hash .pyc: Python would run it instead of its source)")
    for line in res["aliases"]:
        print(f"ALIAS     {line}")
    if res["errors"]:
        return 2
    if not passed(res):
        print(f"golden lock BROKEN: {len(res['mismatch'])} modified, {len(res['missing'])} missing, "
              f"{len(res['unlisted'])} unlisted, {len(res['links'])} link(s), {len(res['bytecode'])} unchecked "
              f".pyc, {len(res['aliases'])} aliasing path(s). Golden tests are maintainer-owned (INV-VERIFY).")
        return 1
    if delta is not None:
        if base_text is None:
            print(f"NO BASE   the base revision has no {GOLDEN}/{MANIFEST_NAME}: every golden file is new")
        elif delta["manifest_changed"]:
            print(f"RELOCKED  {GOLDEN}/{MANIFEST_NAME} differs from the base revision's")
        for k, label in (("modified", "MODIFIED"), ("removed", "REMOVED "), ("added", "ADDED   ")):
            for rel in delta[k]:
                print(f"{label}  {rel}  (against the base revision's lock)")
        if changed(delta):
            summary = (f"{len(delta['modified'])} modified, {len(delta['removed'])} removed, "
                       f"{len(delta['added'])} added, manifest {'changed' if delta['manifest_changed'] else 'same'}")
            if not a.acknowledged:
                print(f"golden tests CHANGED against the base ({summary}; base: {base_name}). Golden tests are "
                      "maintainer-owned (INV-VERIFY): a maintainer (CODEOWNERS) must review the change and add "
                      "the 'golden-reviewed' label.")
                return 1
            print(f"golden tests changed against the base ({summary}); acknowledged by a maintainer, "
                  "continuing with the change's own lock")
        elif not a.quiet:
            print(f"golden lock matches the base revision ({len(res['ok'])} file(s); base: {base_name}; "
                  f"change: {src})")
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
