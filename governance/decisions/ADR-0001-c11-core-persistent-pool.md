# ADR-0001: C11 engine core with a persistent thread pool, no OpenMP

- Status: Accepted
- Date: 2026-10-04
- Supersedes: none
- Superseded-by: none

## Context

The engine must build with nothing but a C compiler on Windows, Linux and
macOS, and it must be easy to audit (INV-DEP). A decode step of a large MoE
model runs hundreds of small parallel regions per token (attention heads,
per-expert matmuls, norms). OpenMP was the obvious way to parallelise them,
but MSVC only ships OpenMP 2.0, and per-region fork/join costs on the order of
5-10 µs (engine/src/hx_pool.h). At hundreds of regions per token that is a
noticeable share of the step time. A dependency-free core also keeps the
shared library loadable from Python via ctypes without a runtime to ship.

## Decision

* The engine core is C11 plus OS APIs. No third-party libraries, no C++
  runtime, no OpenMP (INV-DEP). Atomics come from `<stdatomic.h>`; no VLAs.
* All OS calls live in `platform_win.c` / `platform_posix.c` behind
  `engine/src/hx_platform.h`; compiler-specific code lives only in that header.
* Parallelism goes through one persistent pool (`hx_pool.h`): workers spin for
  a bounded time and then sleep. Work is dispatched by bumping a single
  generation counter. `hx_pool_for` hands out chunks dynamically, but each
  output element is produced by exactly one thread, so results never depend on
  scheduling (INV-DET-1).
* A whole layer's experts may share one parallel region (cross-expert
  parallelism) rather than one region per matmul.

## Consequences

* We own the threading code, including its correctness under every memory
  model we target. `engine/tests/test_pool.c` and the ASan CI job carry that load.
* The design target in hx_pool.h is about 1 µs per dispatch. Actual numbers
  are T01's to measure and record with the hardware (INV-HONEST); this ADR
  makes no performance claim.
* MSVC needs `/experimental:c11atomics` (set in CMakeLists.txt and
  scripts/hxcc.py).
* Revisit if MSVC gains a modern OpenMP with cheap regions. Even then the
  dependency and determinism arguments still stand.
