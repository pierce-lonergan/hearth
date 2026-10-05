# Hearth governance: how work gets done here

Hearth is maintained by humans and AI agents working in short, parallel
sessions. Nobody, human or agent, is expected to remember anything between
sessions. Everything a successor needs is in this directory, in the repository
and in CI. This file is the operating manual. The tools it mentions live in
`governance/tools/`; they need only Python 3.9+ and its standard library, and
each one prints its full usage with `--help`.

## The session loop

1. **Rehydrate.** Read `governance/INVARIANTS.md` (verbatim, every time), your
   task's entry in `governance/tasks.json`, and the latest manifest for it in
   `governance/handovers/`. Do not reconstruct state from chat transcripts.
2. **Pick work.** `python governance/tools/waves.py next` lists the runnable
   tasks: `todo` tasks whose dependencies are all `done`. Edit only the paths
   your task `owns`.
3. **Build and test** with the commands in `CONTRIBUTING.md`. New behaviour
   gets contributor tests next to the code: `engine/tests/test_*.c`,
   `tests/py/test_*.py`.
4. **Prove the tests bite.** Run `mutate.py` on what you changed. Fix or
   explain every surviving mutant.
5. **Hand over.** Before the session ends, write and validate a manifest.

## Knowledge lanes

Knowledge is filed by how much distortion it can tolerate:

| lane | examples | lives in | rule |
|------|----------|----------|------|
| **Constraints** | invariants, normative formats, numerics | `governance/INVARIANTS.md`, `docs/FORMAT.md`, `docs/NUMERICS.md`, `tests/golden/` | Zero loss: copy ids and text verbatim, never paraphrase. Changes need a maintainer, an ADR and a golden-test update. |
| **Procedures** | how to build, test, benchmark | manifest `procedures`, `CONTRIBUTING.md`, tool docstrings | Must stay behaviour-equivalent: exact commands that still run. A command that no longer works is a bug. |
| **Beliefs** | design decisions, hypotheses | ADRs in `governance/decisions/`, manifest `epistemic_ledger` | Never rewritten in place. A new belief supersedes the old one with a link. Hypotheses carry evidence and a verdict. |
| **Episodic** | what happened, discussion | commit messages, PR threads, logs, mutation reports | May be summarised or dropped. Nothing in this lane is binding. |

If something matters for correctness and exists only in the episodic lane,
promote it: to a test, an invariant proposal, an ADR or a procedure.

## Tasks, waves and ownership

`governance/tasks.json` is the roadmap as a DAG. Each task has `id`, `title`,
`deps`, `owns` (paths), `accept` (acceptance criteria) and `status`
(`todo | active | review | done | blocked`).

```
python governance/tools/waves.py              # waves + runnable set
python governance/tools/waves.py validate     # what CI runs; exit 1 on an invalid DAG
python governance/tools/waves.py next --done T01-platform --status T06-store=active   # what-if
python governance/tools/waves.py --mermaid    # diagram for PRs and docs
```

* The DAG must be acyclic, and every dependency must exist; `waves.py` prints
  the offending cycle.
* **Disjoint ownership.** No two unfinished tasks may own overlapping paths. A
  path owns everything below it, with or without a trailing `/`. Paths are
  compared after dropping `.` and empty segments and ignoring case, because
  Windows and macOS file systems are case-insensitive; write them canonically
  (`engine/src/pool.c`, `python/hearth/sim/`), or `waves.py` warns. Disjoint
  ownership is what lets one wave run in parallel without merge conflicts.
* Outside your `owns`, everything is read-only, including contract headers
  (`engine/include/hearth.h`, `engine/src/hx_*.h`) and `docs/`. If a contract
  is wrong or insufficient, do not edit it. Work around it if that is safe,
  and report a precise proposed change (file, current text, new text, reason)
  in your handover.

## Verification authority

The people and agents who implement a change do not get to define what
"correct" means for it.

**Golden tests (`tests/golden/`)** encode the invariants as executable checks.
Maintainers own them (INV-VERIFY). They are hash-locked by
`tests/golden/MANIFEST.sha256`:

```
python governance/tools/check_golden.py               # exit 1 if any locked file changed, vanished or appeared
python -E governance/tools/check_golden.py --run      # CI: lock check, then the golden suite in isolation
```

Never edit, add, rename or delete anything there. If a golden test is wrong,
report it. A maintainer changes it in a dedicated commit, re-locks it with
`check_golden.py --update --i-am-a-maintainer`, and records an ADR if an
invariant moved.

The lock alone is not enough. Code outside `tests/golden/` can skip or
neutralise golden tests: a `conftest.py` hook, a `pytest.ini` option,
`PYTEST_ADDOPTS`, a `tests/__init__.py` that pytest imports as the package
parent, a module in `python/` that shadows one the tests import, or a planted
`.pyc` that Python loads instead of its source. pytest reports a skipped or
emptied test as success, so CI never runs golden tests with a plain
`pytest tests/golden`. `--run` copies only the locked files (hashed again as
they are copied) into a temporary directory and runs them there, without
outside conftest or ini files, pytest environment variables, auto-loaded
plugins or old bytecode; from `python/` only the `hearth` package is
importable. It fails unless every golden test ran and passed; a skip or an
xfail is a failure. Start it with `python -E` and without `PYTHONPATH`, so
that no `sitecustomize.py` from the repository can run before it.
`HEARTH_LIB` must point at the built library. The lock check itself fails on
symbolic links or junctions in `tests/golden/` and on unchecked-hash `.pyc`
files there.

Two things stay outside the isolation. The code under test (the `hearth`
package and the engine library) runs in the same process as pytest and could
tamper with it; only review catches that. And the tool cannot vouch for
itself, so CI runs `check_golden.py` (and the decision breaker) as checked out
at the pull request's base revision: a change cannot weaken the check it is
judged by.

**Contributor tests** are yours, and they must be able to fail. `mutate.py`
injects small faults (flipped comparisons, swapped operators, off-by-one
constants, negated conditions, replaced return values, deleted calls) into a
copy of the repository and runs your test command against each one. The
working tree is never touched.

```
# C: build into the job's private directory with {tmp}
python governance/tools/mutate.py --file engine/src/pool.c --jobs 2 --timeout 120 --max-mutants 40 --seed 1 \
  --test "python scripts/hxcc.py --platform --run -o {tmp}/mut_pool.exe engine/src/pool.c engine/tests/test_pool.c -- --quick"
# Python
python governance/tools/mutate.py --file python/hearth/format.py --jobs 4 \
  --test "{python} -m pytest -q -x tests/py/test_format.py"
# only what you changed
python governance/tools/mutate.py --file engine/src/store.c --since origin/main --test "..."
```

* **Output names.** Several agents run `mutate.py` on one machine at the same
  time, and hxcc puts a relative `-o name.exe` (and its object files) in the
  shared `<data dir>/build/hxcc`. A fixed name there makes concurrent runs
  link each other's mutants and report wrong results. Use `-o {tmp}/name.exe`
  (the job's own directory, deleted afterwards) or put `{job}` in the name: it
  expands to `<run id>-<job index>`, unique across runs, and those files are
  deleted at the end. `mutate.py` warns about a `-o` without either.
* Every surviving mutant is either a missing test (add it) or an equivalent
  mutant: the change cannot alter behaviour, as with `k > n` versus `k >= n`
  before `k = n`. Say which, per survivor, in your handover.
* Report the score together with the seed, the sample size and the test
  command; reports land in `<data dir>/mutation/`.
* Timeouts count as detected, but a mutant that times out is first re-run on
  its own after the parallel phase, so load alone does not turn a slow pass
  into a kill (the report lists retried mutants). Build errors are excluded
  from the score; with a separate `--build` command, a compile that times out
  is a build error rather than a kill.
* `--timeout` also bounds the unmutated baseline run, which must finish in
  time (exit 2 otherwise), and a mutant's timeout is never below
  max(2 x baseline, baseline + 5 s); a smaller value is raised with a warning.
  A score made only of timeouts is flagged in the report and fails
  `--min-score`.
* For tests that spawn many threads, use fewer jobs and an explicit
  `--timeout`; CPU oversubscription slows everything down.
* C code in `#if` branches that are not compiled on this machine is not
  mutated (`#if 0`, the other OS's branch). Conditions are evaluated from the
  platform's predefined macros and the `#define`s of the file and its quoted
  headers; a branch that cannot be decided is mutated, and its survivors are
  marked with the condition. `-D NAME` / `-U NAME` settle a macro by hand.
* Ctrl-C stops a run cleanly: queued mutants are cancelled, running commands
  are killed and the job directories are removed.
* Out-of-bounds mutants often survive a normal build and are only caught under
  AddressSanitizer. That is why the ASan CI job exists.
* Keep test commands fast; prefer a `--quick` mode for benchmarks embedded in
  tests.
* Job copies contain no `.git`, so a test that reads the checkout's history
  skips inside them and the code it covers shows up as surviving mutants.
  Test git-dependent code against a throwaway repository created in the
  test's temporary directory instead (as `test_adr_reads_git_revisions` does).
* Mark a line that must not be mutated (logging, timing printouts) with a
  `nomutate` comment.

## Handover manifests

A session ends ("generational death") at a verified milestone, before its
context fills up, when its budget runs out, or on a friction abort. It always
writes `governance/handovers/<TASK_ID>-gen-<NNN>.json`, conforming to
`governance/schemas/handover.schema.json`.

```
python governance/tools/validate_handover.py --new T06-store --write   # creates the next T06-store-gen-NNN.json
python governance/tools/validate_handover.py                           # validates every manifest (CI)
python governance/tools/validate_handover.py --strict governance/handovers/T06-store-gen-002.json
```

`--new` accepts only task ids from `governance/tasks.json`. `--write` picks the
generation number and `parent_generation` from the manifests already there and
never overwrites a file. The validator warns
(fails with `--strict`) when the file name does not match `task_id` and
`generation_id`, or when `parent_generation` does not name an earlier
manifest of the same task.

* `lifecycle_status` and `trigger` say why the session ended. Use
  `TASK_COMPLETE`/`MILESTONE_VERIFIED` only if the acceptance criteria were
  verified by running them.
* `constraints_touched` lists invariant ids exactly as in INVARIANTS.md. The
  validator rejects unknown ids.
* `epistemic_ledger` holds hypotheses with evidence (the command you ran and
  what you saw) and a verdict: CONFIRMED, REJECTED or INCONCLUSIVE. Keep what
  you *ran* separate from what you only *wrote*.
* `procedures` are exact commands that build and test the work, copy-pasteable
  by the successor.
* `measurements` state their conditions (hardware, load, settings) and `kind`:
  `measured` or `simulated` (INV-HONEST). Numbers taken while other jobs share
  the machine are noisy; say so.
* `active_blockers` and `unfulfilled_mandates` are the successor's to-do list.
  An empty list is a claim, so make sure it is true.

## Decisions (ADRs)

Design decisions are recorded in `governance/decisions/ADR-NNNN-slug.md`
(copy `TEMPLATE.md`): Status, Date, Supersedes, Superseded-by, then
Context, Decision and Consequences.

* Accepted decisions are not edited to mean something else. Write a new ADR
  with `Supersedes: ADR-XXXX`, and in the same change set the old one to
  `Status: Superseded` and `Superseded-by: ADR-NNNN`.
* `python governance/tools/adr.py lint` checks format and two-sided links;
  `adr.py index` prints the table of decisions.

## Epistemic friction

Retrying the same failing thing is the most common way a session burns its
budget. Log attempts and check before retrying:

```
# --task keeps the log in <data dir>/attempts/T06-store.jsonl, outside the repository
python governance/tools/friction.py record --task T06-store --cmd "<cmd>" --exit-code 1 --signature "<first error line>"
python governance/tools/friction.py change --task T06-store --note "what you changed"
python governance/tools/friction.py check  --task T06-store      # exit 1 = stop
```

The same error signature three times with no change in between trips the
check. Without `--signature`, an attempt counts under its command and exit
code. A task with no log yet has no attempts (`check` exits 0). It prints the mandatory root-cause template: observation, mechanism,
evidence, approaches already falsified, and the next attempt with its expected
result. Record the note with `friction.py note`; only then retry. Each
signature gets two notes. If it trips again after the second, `check` prints
ESCALATE instead of a template: stop and hand over with
`lifecycle_status: FRICTION_ABORT`, `trigger: EPISTEMIC_FRICTION`.

## Elastic autonomy and circuit breakers

How much autonomy you have depends on how strongly the change is verified:

| change | who may make it | required evidence |
|--------|-----------------|-------------------|
| Code and tests inside your `owns` | any contributor or agent | CI green, mutation report, manifest |
| New ADR, or superseding up to 15% of accepted decisions | contributor, maintainer merges | `adr.py lint` |
| Contracts (`hearth.h`, `hx_*.h`, `docs/FORMAT.md`, `docs/NUMERICS.md`, `tasks.json`) | maintainer | proposal from the handover's contract issues |
| Invariants, golden tests | maintainer | ADR + golden update + `check_golden.py --update` |

Circuit breakers stop the line automatically:

* **Decision churn.** A change that retires (supersedes or deprecates) more
  than 15% of the decisions that were Accepted at the base revision needs
  human review: `adr.py breaker --base-rev <base>`. In CI
  (`.github/workflows/decisions.yml`), a maintainer acknowledges it with the
  `decisions-reviewed` label; adding the label starts a new run.
* **Invariants without a decision.** Changing (or deleting)
  `governance/INVARIANTS.md` fails the same check unless the change also adds
  an ADR, or changes one in substance (not just whitespace), that cites
  `INVARIANTS.md` or the invariant ids whose rows changed.
* **Golden lock.** Any change under `tests/golden/` that the manifest does not
  match fails CI (`check_golden.py`), and so does a golden test that does not
  run and pass (`check_golden.py --run`).
* **Friction.** `friction.py check` exiting 1 means stop and write the
  root-cause note.
* **Invalid DAG or manifest.** `waves.py validate` and `validate_handover.py`
  fail CI.

## CI

`.github/workflows/ci.yml` runs:
* Linux GCC and Clang (CMake Release, ctest, pytest on `tests/py` against the
  built library, then the golden suite through `check_golden.py --run`);
* Linux AddressSanitizer + UBSan (ctest);
* Windows MSVC (Visual Studio generator), the same tests;
* macOS arm64, the same tests;
* a `governance` job on Python 3.9 and 3.13: `check_golden`, `waves validate`,
  `validate_handover`, `adr lint` and `tests/py/test_governance.py`.

`.github/workflows/decisions.yml` runs the decision breaker on pull requests,
including when labels change. The golden steps and the breaker use
`governance/tools/` as checked out at the pull request's base revision
(`.trusted/`), falling back to the change's own copy only while the base has
no tools yet.

torch and transformers are not installed by default. Tests that need them
must `pytest.importorskip` them, and a manually triggered job runs them.

## Tool reference

| tool | purpose | exit codes |
|------|---------|------------|
| `waves.py [show\|validate\|next] [--json\|--mermaid]` | DAG validation, waves, runnable set | 0 ok, 1 invalid DAG, 2 bad input |
| `validate_handover.py [paths] [--strict] [--json] [--new TASK [--write]]` | manifest schema + repository checks | 0 ok, 1 invalid, 2 bad input |
| `check_golden.py [--run] [--update --i-am-a-maintainer]` | golden-test hash lock, isolated golden run | 0 ok, 1 broken or suite failed, 2 malformed |
| `mutate.py --file F --test CMD [...]` | mutation testing, C and Python | 0 done, 1 below `--min-score`, 2 bad input or failing baseline, 130 interrupted |
| `friction.py check\|record\|change\|note LOG\|--task ID` | retry-loop detection | 0 ok, 1 thrashing, 2 bad input |
| `adr.py lint\|index\|breaker` | ADR format, supersession, circuit breaker | 0 ok, 1 failed/tripped, 2 bad input |

`governance/tools/demo/` holds a small C module and its tests. It is the
standing demonstration that `mutate.py` kills mutants:

```
python governance/tools/mutate.py --file governance/tools/demo/hxdemo.c --max-mutants 0 --jobs 4 \
  --test "python scripts/hxcc.py --run -o {tmp}/mut_demo.exe governance/tools/demo/hxdemo.c governance/tools/demo/test_hxdemo.c"
```
