#!/usr/bin/env python3
"""path_aliases — refuse a commit whose paths name one file on Windows or macOS.

    python governance/tools/path_aliases.py                                  # HEAD of this checkout
    python -I -S base/governance/tools/path_aliases.py --root pr --rev HEAD   # one commit, as the gates read it

Git compares path names byte by byte, and so do GitHub's CODEOWNERS patterns and the
review gates. The file systems of Windows (NTFS) and macOS (APFS, HFS+) do not: there
governance/INVARIANTS.md and governance/invariants.md are one file. git checks out
both and the one written last wins (in git's byte order the lowercase twin). A pull
request could add a twin of a protected file (the invariants, a decided ADR, a golden
test, a contract) and pass every gate, which read the real file and see the twin as
an unrelated new one. Every Windows or macOS checkout, where agents read the
invariants, would then hold the twin's text. NTFS also resolves 8.3 short names:
where they are generated (the default on system volumes),
governance/INVARI~1.MD and governance/decisi~1/ADR-0004-... are written into
governance/INVARIANTS.md and governance/decisions/ by git for Windows.

find() reports two kinds of paths in a tree (files, links, submodules and
directories alike):
  * twins: paths that are equal once every component has the code points HFS+
    ignores (U+200C-U+200F, U+202A-U+202E, U+206A-U+206F, U+FEFF) removed, its
    trailing dots and spaces dropped (Win32 drops them), and is decomposed (NFD)
    and folded as casefold(upper(...)). That key is at least as coarse as both
    macOS's comparison (NFD + case folding) and upper-casing (NTFS), checked for
    every code point;
  * short names: a component shaped like an NTFS 8.3 alias (up to 6 characters,
    '~', digits, at most 8 before the optional extension of up to 3), in any case.
    Which long name it aliases depends on creation order, so the shape alone is
    refused.
This is broader than any one file system (it also joins sharp s, U+00DF, and 'ss'). A false
alarm costs a rename; a miss lets a change overwrite what it was judged without.
No label overrides it.

check_golden.py --rev and adr.py --rev load this module from their own directory
and refuse a tree with any finding, so the golden gate and the decision breaker
never judge a tree that checks out differently on Windows or macOS. Only git
objects are read (`git ls-tree -r -t -z` of the commit). Standard library, Python 3.9+.
Exit: 0 nothing found, 1 aliasing paths, 2 bad arguments or git failure.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_IGNORED = dict.fromkeys([0x200C, 0x200D, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x206A, 0x2070), 0xFEFF])
_SHORT = re.compile(r"([^.~]{0,6})~([0-9]+)(?:\.[^.]{0,3})?")


class GitError(Exception):
    pass


def _nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def fold(path: str) -> str:
    """The key under which Windows and macOS file systems may see two paths as one."""
    return "/".join(_nfd(_nfd(_nfd(c.translate(_IGNORED).rstrip(". ")).upper()).casefold())
                    for c in path.split("/"))


def is_short_name(component: str) -> bool:
    m = _SHORT.fullmatch(component.translate(_IGNORED).rstrip(". "))
    return m is not None and len(m.group(1)) + len(m.group(2)) < 8


def find(paths) -> dict:
    """{"twins": [[path, ...], ...], "short_names": [path, ...]} for the given repo-relative paths."""
    groups, short = {}, set()
    for p in sorted(set(paths)):
        groups.setdefault(fold(p), []).append(p)
        parts = p.split("/")
        for i, c in enumerate(parts):
            if is_short_name(c):            # reported once, at the first such component
                short.add("/".join(parts[:i + 1]))
                break
    return {"twins": [g for _, g in sorted(groups.items()) if len(g) > 1], "short_names": sorted(short)}


def _show(path: str) -> str:
    """A path in ASCII, other code points as backslash escapes: an invisible one is the point of some twins."""
    return path.encode("ascii", "backslashreplace").decode("ascii")


def describe(found: dict) -> list:
    """One line per finding."""
    return ([f"TWINS  {' = '.join(map(_show, g))}  (one file on Windows and macOS)" for g in found["twins"]] +
            [f"8.3    {_show(p)}  (a component shaped like an NTFS short name aliases a longer name there)"
             for p in found["short_names"]])


def tree_paths(root, rev: str) -> list:
    """Every path in the tree of commit rev of the git checkout whose top level is root."""
    root = Path(root)
    if not rev or rev.startswith("-"):
        raise GitError(f"{rev!r} is not a revision")

    def git(*args):
        try:
            r = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=120)
        except (OSError, subprocess.SubprocessError) as e:
            raise GitError(f"cannot run git: {e}") from None
        if r.returncode:
            raise GitError(f"git {args[0]} in {root}: {r.stderr.decode('utf-8', 'replace').strip()}")
        return r.stdout

    if git("rev-parse", "--show-cdup").strip():
        raise GitError(f"{root} is not the top level of a git checkout")
    try:
        commit = git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").decode("ascii").strip()
    except GitError:
        raise GitError(f"{rev!r} is not a commit in {root}") from None
    out = []
    for raw in git("ls-tree", "-r", "-t", "-z", "--full-tree", "--name-only", commit).split(b"\0"):
        if raw:
            try:
                out.append(raw.decode("utf-8"))
            except UnicodeDecodeError:
                raise GitError(f"a path at {rev} is not UTF-8: {raw!r}") from None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT), help="top level of a git checkout (default: this one)")
    ap.add_argument("--rev", default="HEAD", help="the commit whose tree is checked (default: HEAD)")
    a = ap.parse_args(argv)
    try:
        paths = tree_paths(a.root, a.rev)
    except GitError as e:
        print(f"path_aliases: {e}", file=sys.stderr)
        return 2
    lines = describe(find(paths))
    for line in lines:
        print(line)
    if lines:
        print(f"path aliases FOUND at {a.rev}: {len(lines)} finding(s). A Windows or macOS checkout would hold "
              "different files than the gates judge; rename them (no label accepts this).")
        return 1
    print(f"path aliases OK: {len(paths)} path(s) at {a.rev}, none alias on Windows or macOS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
