#!/usr/bin/env python3
"""merge_ref — fetch a pull request's merge commit and check it belongs to the CI event being judged.

    python -I -S base/governance/tools/merge_ref.py --fetch "$REPO_URL" --pr "$PR_NUMBER" \\
        --merge pr --head "$HEAD_SHA" --base base                                   # CI
    python governance/tools/merge_ref.py --merge some/checkout --head <sha> --base other/checkout

The review gates (golden.yml, decisions.yml) judge refs/pull/<N>/merge, the commit GitHub
creates by merging the pull request into its base branch. They never check it out:
--fetch creates the repository --merge (which must not exist yet) and fetches only that
commit's objects into it (depth 1, verified by git's fsck), with no working tree, so
nothing from the pull request reaches the disk as a file, link or case twin. Its HEAD is
detached at the merge commit, which the gates then read as git objects (--rev HEAD).
actions/checkout is not used for it: since 2026-07-20 it refuses to fetch fork pull
request code under pull_request_target unless allow-unsafe-pr-checkout is set. If the
environment holds GITHUB_TOKEN, that fetch alone sends it, as an HTTP header for the
--fetch URL passed through git's environment (not the command line).

GitHub updates the merge ref asynchronously and pull_request_target runs do not wait for
it, so a run started by a new push can see the previous merge commit and report its
verdict on the new head. The check therefore fails unless the commit in --merge (HEAD,
or --rev) has exactly two parents, the second being --head (the event's head commit) and
the first being the commit checked out in --base or an ancestor of it. --base is the
gate's own checkout of github.sha, which under pull_request_target is the head of the
default branch (main) since 2025-12-08, whatever the pull request's base branch; the
gates only run for pull requests into main. A merge computed against a newer base than
--base fails too: the base's lock and decisions would not describe it. A failed check
means "start a new run": push, or remove and re-add a label (re-running a job replays
the old event).

--base needs the base branch's history (actions/checkout fetch-depth: 0) for the ancestor
test; --merge may be shallow, since parents are read from the commit object itself. Only git
objects are read, nothing from either side is executed. Standard library, Python 3.9+.
Exit: 0 the merge commit matches, 1 it does not, 2 bad arguments, fetch or git failure.
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import subprocess
import sys
from pathlib import Path

_OID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")      # SHA-1 or SHA-256 object names
_URL = re.compile(r"(?:https|file)://\S+")
_PR = re.compile(r"[1-9][0-9]{0,9}")
_LOCAL = "refs/pull/merge"                          # where --fetch stores the merge commit in --merge


class GitError(Exception):
    pass


def _run(args, env=None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        raise GitError(f"cannot run git: {e}") from None


def _git(repo: Path, *args) -> subprocess.CompletedProcess:
    return _run(["-C", str(repo), *args])


def fetch_env(url: str, token: str) -> dict:
    """The fetch's environment: no prompts, and a token as an HTTP header for url only."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    if token:
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode("ascii")
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0=f"http.{url}.extraheader",
                   GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {auth}")
    return env


def fetch(url: str, pr: str, into: Path, token: str = "") -> str:
    """Fetches refs/pull/<pr>/merge of url into a new repository at into, as objects only
    (no working tree), detaches HEAD there at it and returns its commit id."""
    if not _URL.fullmatch(url):
        raise GitError(f"--fetch {url!r} is not an https:// or file:// URL")
    if not _PR.fullmatch(pr):
        raise GitError(f"--pr {pr!r} is not a pull request number")
    if os.path.lexists(into):
        raise GitError(f"{into} already exists: --fetch makes a new repository")
    steps = (("init", ["init", "-q", str(into)], None),
             ("fetch", ["-C", str(into), "-c", "transfer.fsckObjects=true", "fetch", "-q", "--no-tags",
                        "--no-recurse-submodules", "--depth=1", url, f"+refs/pull/{pr}/merge:{_LOCAL}"],
              fetch_env(url, token)),
             ("update-ref", ["-C", str(into), "update-ref", "--no-deref", "HEAD", _LOCAL], None))
    for name, args, env in steps:
        r = _run(args, env)
        if r.returncode:
            raise GitError(f"git {name}: {r.stderr.decode('utf-8', 'replace').strip()}")
    return commit_of(into, "HEAD")


def commit_of(repo: Path, rev: str) -> str:
    """The commit `rev` names in the checkout whose top level is repo."""
    if not rev or rev.startswith("-"):
        raise GitError(f"{rev!r} is not a revision")
    r = _git(repo, "rev-parse", "--show-cdup")      # "" at the top level; a subdirectory would read its parent repo
    if r.returncode or r.stdout.strip():
        raise GitError(f"{repo} is not the top level of a git checkout")
    r = _git(repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
    if r.returncode:
        raise GitError(f"{rev!r} is not a commit in {repo}")
    return r.stdout.decode("ascii", "replace").strip()


def parents(repo: Path, commit: str) -> list:
    """The parents recorded in the commit object (a shallow clone hides them from rev-parse)."""
    r = _git(repo, "cat-file", "commit", commit)
    if r.returncode:
        raise GitError(f"cannot read commit {commit}: {r.stderr.decode('utf-8', 'replace').strip()}")
    head = r.stdout.split(b"\n\n", 1)[0].decode("utf-8", "replace")
    ps = [line[len("parent "):] for line in head.split("\n") if line.startswith("parent ")]
    for p in ps:                # they become git arguments below
        if not _OID.fullmatch(p):
            raise GitError(f"commit {commit} has a malformed parent line {p[:80]!r}")
    return ps


def verify(merge: Path, head: str, base: Path, rev: str = "HEAD") -> dict:
    """{"merge", "parents", "base", "problems"}; no problems means the merge commit is
    head merged into base (or into an ancestor of base)."""
    head = head.strip().lower()
    if not _OID.fullmatch(head):
        raise GitError(f"--head {head!r} is not a full commit id")
    m = commit_of(merge, rev)
    b = commit_of(base, "HEAD")
    ps = parents(merge, m)
    problems = []
    if len(ps) != 2:
        problems.append(f"{m[:12]} has {len(ps)} parent(s), not 2: not a pull request merge commit")
    else:
        if ps[1] != head:
            problems.append(f"its second parent {ps[1][:12]} is not the head commit {head[:12]} of this event: "
                            "the merge ref is stale (GitHub has not updated it yet)")
        r = _git(base, "merge-base", "--is-ancestor", ps[0], b)         # 0 also when they are equal
        if r.returncode == 1:
            problems.append(f"its first parent {ps[0][:12]} is not the base {b[:12]} or an ancestor of it: "
                            "merged into a different or newer base")
        elif r.returncode:
            problems.append(f"its first parent {ps[0][:12]} is unknown to the base checkout (needs the base "
                            "branch's history, fetch-depth: 0)")
    return {"merge": m, "parents": ps, "base": b, "problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", metavar="URL", help="first fetch refs/pull/<--pr>/merge from URL into --merge, "
                                                   "a new repository without a working tree")
    ap.add_argument("--pr", help="with --fetch: the pull request number")
    ap.add_argument("--merge", required=True, help="repository holding the pull request's merge commit")
    ap.add_argument("--rev", default="HEAD", help="the merge commit in --merge (default: HEAD)")
    ap.add_argument("--head", required=True, help="the event's head commit (github.event.pull_request.head.sha)")
    ap.add_argument("--base", required=True, help="checkout of the base revision, with its history")
    a = ap.parse_args(argv)
    if (a.fetch is None) != (a.pr is None):
        print("merge_ref: --fetch and --pr go together", file=sys.stderr)
        return 2
    try:
        if a.fetch is not None:
            got = fetch(a.fetch, a.pr, Path(a.merge), os.environ.get("GITHUB_TOKEN", ""))
            print(f"fetched refs/pull/{a.pr}/merge = {got[:12]} into {a.merge} (objects only, no working tree)")
        res = verify(Path(a.merge), a.head, Path(a.base), a.rev)
    except GitError as e:
        print(f"merge_ref: {e}", file=sys.stderr)
        return 2
    if res["problems"]:
        for p in res["problems"]:
            print(f"MERGE REF {p}")
        print(f"merge ref REJECTED: {res['merge'][:12]} is not this event's head merged into base {res['base'][:12]}. "
              "Nothing was judged; push again or re-add a label to start a new run.")
        return 1
    first, second = res["parents"]
    older = "" if first == res["base"] else f" (an ancestor of base {res['base'][:12]})"
    print(f"merge ref OK: {res['merge'][:12]} = base {first[:12]}{older} + head {second[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
