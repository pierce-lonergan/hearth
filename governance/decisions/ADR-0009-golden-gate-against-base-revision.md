# ADR-0009: Judge golden-test changes against the base revision's lock

- Status: Proposed
- Date: 2026-10-05
- Supersedes: none
- Superseded-by: none

## Context

INV-VERIFY makes golden tests maintainer-owned, and `check_golden.py` hash-locks
them with `tests/golden/MANIFEST.sha256`. Until now CI compared the files with
the manifest *of the change under review*. A change could therefore gut a
golden test, re-lock it with `check_golden.py --update` (whose maintainer flag
is an honour system) and pass both the lock check and the golden run. The
verification council reproduced this in round 1 (T05-governance, major).

Two further gaps: CI ran the decision lint and the handover and DAG validators
from the change's own `governance/tools/`, and a `pull_request` workflow is the
pull request's own copy of the workflow file, so a change could edit the gate
that judges it.

Options considered for the maintainer acknowledgement: a label (as the decision
breaker already uses), a GitHub environment with required reviewers, or an API
check that the label was applied by a maintainer. The label is the simplest to
reason about, matches `decisions-reviewed`, and needs no token permissions; the
other two cannot be exercised outside GitHub either.

The round-2 verifier found three more ways around the first version of these
gates. Both checked the pull request's working tree, and a change could
replace `tests/` (or `governance/`) with an absolute link to `$WS/base/...`,
the base checkout next to it in the runner's workspace, so the gates judged the
base's own files. They checked out `refs/pull/<N>/merge` by name, which GitHub
updates asynchronously, so a run for a new head could judge the previous merge
commit. And in `ci.yml` the change's own code (`python -m pip` importing a
`pip/` directory, the build, conftest) ran before the "base revision" tools.

The round-3 verifier found that a case twin (`governance/invariants.md` beside
`INVARIANTS.md`, `decisions/adr-0004-...md` beside `ADR-0004-...md`,
`tests/Golden/`) passed both gates without a label, while git, writing both,
leaves the twin's text in every Windows and macOS checkout; CODEOWNERS
patterns are case-sensitive too. NTFS 8.3 short names alias the same way
(`governance/INVARI~1.MD`, `governance/decisi~1/`; reproduced with git for
Windows 2.45). It also found that `actions/checkout` refuses fork pull
requests under `pull_request_target` since 2026-07-20, and that since
2025-12-08 that event always runs the default branch's workflow and sets
`github.sha` to its head, whatever the pull request's base.

## Decision

* `check_golden.py --base-manifest FILE` / `--base-rev REV` compares the
  change's `tests/golden/` with the base revision's lock: every base-locked file
  present with the base hash, no added file, and a byte-identical manifest
  (after CRLF normalisation). Any difference fails unless `--acknowledged`.
* `.github/workflows/golden.yml` runs that check on `pull_request_target`, with
  the tool and the lock checked out at the base branch and the pull request's
  merge ref checked out as data only. A maintainer acknowledges a reviewed
  golden change with the `golden-reviewed` label. The workflow must never
  execute anything from the pull request.
* `decisions.yml` (the ADR breaker) moves to `pull_request_target` the same
  way, comparing with a base checkout instead of git history.
* `ci.yml` runs every governance tool (`check_golden`, `waves`,
  `validate_handover`, `adr lint`) from the base revision.
* `.github/CODEOWNERS` names maintainers for `tests/golden/`,
  `governance/INVARIANTS.md`, `governance/tools/`, `governance/schemas/`,
  `.github/` and the contract files.
* The ADR breaker also trips on duplicate ADR ids and on decided ADRs that are
  rewritten in place, moved back to an earlier status or deleted. Everything
  in a decided ADR is frozen except the status word (Accepted to Deprecated or
  Superseded, Deprecated to Superseded), a well-formed Date and the ids of
  Superseded-by.
* The gates read the pull request as git objects of its merge commit
  (`check_golden.py --rev HEAD`, `adr.py breaker --rev HEAD`), never its
  working tree. A symbolic link or submodule at `tests/`, `tests/golden/`,
  `governance/`, `governance/decisions/`, a record or `INVARIANTS.md` fails,
  label or not. The working-tree checks (`ci.yml`, local use) refuse links on
  the same paths.
* Before judging, both gates run `merge_ref.py --fetch`: it fetches
  `refs/pull/<N>/merge` as git objects (depth 1, fsck) into an empty
  repository, never a working tree, instead of checking it out with
  `actions/checkout` and `allow-unsafe-pr-checkout`; nothing from the pull
  request is written to disk as a file. The merge commit's second parent
  must be the event's head commit, its first the base checkout's commit or an
  ancestor of it (the base is fetched with history for this).
* The gates run for pull requests into `main` (the default branch) only.
* `check_golden.py --rev` and `adr.py --rev` refuse, label or not, a commit
  with paths that are one file on Windows or macOS anywhere in its tree
  (`path_aliases.py`: case and Unicode twins under a fold broader than NTFS,
  APFS and HFS+, and 8.3 short-name shapes).
* CODEOWNERS also covers `governance/decisions/` and the golden suite's
  oracle in `python/hearth/` (reference, quant, synth and their package
  imports).
* In `ci.yml` the tools run with `python -I -S`, in the governance job before
  anything from the change is installed or run, and pip runs as
  `python -I -m pip`. The build jobs' golden run necessarily follows the
  change's build and tests, so it is documented as only as trustworthy as
  review of that code.

## Consequences

* A golden change now needs two maintainer acts: a code-owner approval and the
  label. Re-locking alone no longer passes CI.
* The guarantee still rests on settings outside the repository: branch
  protection on main with the `golden lock vs base` and `decision circuit
  breaker` checks required, required code-owner review, stale approvals
  dismissed, and no bypass for administrators. governance/README.md lists these
  as residual trust assumptions; nothing in the repository can verify them.
* The label is not removed when new commits arrive; dismissing stale approvals
  is what forces a fresh review. Anyone with triage access can add a label, so
  the label is a switch, not the approval itself.
* Changes to `golden.yml` or `decisions.yml` take effect only after they are
  merged, because `pull_request_target` always runs the base branch's copy.
* A stale merge ref fails the gate instead of being judged; a new event (push,
  label) starts a fresh run, because re-running a job replays its old event.
* A file name such as `notes~1.md`, or two names that differ only in case,
  can no longer be merged anywhere in the repository; the check is
  deliberately broader than any one file system.
* None of this has run on GitHub yet (the `--fetch` step was exercised only
  against a public repository's merge ref, anonymously). Revisit if the first real runs show that
  `pull_request_target` check runs cannot be made required, or that stale
  merge refs or base drift cause false failures often.
