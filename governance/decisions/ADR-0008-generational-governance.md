# ADR-0008: Generational governance with separated verification authority

- Status: Accepted
- Date: 2026-10-04
- Supersedes: none
- Superseded-by: none

## Context

Hearth is meant to be maintained indefinitely by humans and AI agents working
in parallel. Agent sessions are short-lived and lose context. Transcripts are
too large and too lossy to hand over. An implementer that also writes the
tests that judge its work can make them pass by weakening them. Agents also
tend to retry a failing approach instead of understanding it, and a single
large change can quietly overturn many earlier decisions.

## Decision

The operating model is described in governance/README.md and enforced by
stdlib-only tools in governance/tools/ that run in CI:

* **Generations.** A session ends with a handover manifest in
  governance/handovers/, validated against
  governance/schemas/handover.schema.json (`validate_handover.py`). A successor
  starts from INVARIANTS.md + the manifest + tasks.json, never from a
  transcript.
* **Typed knowledge lanes.**
  * Constraints are carried verbatim (INVARIANTS.md).
  * Procedures are kept as exact runnable commands (manifest `procedures`).
  * Beliefs are recorded as ADRs with supersession links (governance/decisions/).
  * Episodic history may be summarised (PRs, logs).
* **DAG scheduling.** Tasks run in topological waves with disjoint path
  ownership (`waves.py`).
* **Separated verification.**
  * Golden tests are maintainer-owned and hash-locked (`check_golden.py`,
    INV-VERIFY).
  * Contributor tests must kill mutants (`mutate.py`); surviving mutants are
    explained or fixed.
* **Epistemic friction.** The same failure three times without a change
  requires a written root-cause note before the next try (`friction.py`).
* **Circuit breakers.**
  * Retiring more than 15% of accepted decisions in one change needs human
    review.
  * So does changing INVARIANTS.md without an ADR (`adr.py breaker`).

## Consequences

* Each session carries more ceremony: a manifest, mutation runs, ADRs. That is
  the price of letting many short-lived contributors work safely in parallel.
* The tools must stay dependency-free (Python 3.9+ stdlib) so they run on
  every CI image and every contributor machine.
* The thresholds (15%, three repeats, the 40-mutant default sample) are
  initial values. Revisit them with evidence from real handovers and mutation
  reports, through a superseding ADR.
