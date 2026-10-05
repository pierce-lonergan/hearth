# ADR-0003: Canonical floating-point order for bit-identical output

- Status: Accepted
- Date: 2026-10-04
- Supersedes: none
- Superseded-by: none

## Context

Hearth moves weights between NVMe, a DRAM cache, pinned slots and prefetch
buffers. It runs on any number of threads, dispatches scalar, AVX2 or AVX-512
kernels, and evaluates tokens one at a time or in batches (prefill, speculative
verification). Float addition is not associative, and FMA contraction changes
rounding. Unless the order of operations is fixed, any of these choices can
change logits. Then a cache-policy bug and an ordinary rounding difference look
the same, and lossless speculative decoding becomes impossible to verify.

## Decision

* docs/NUMERICS.md is normative. Every kernel on every ISA reproduces its
  scalar definition exactly:
  * `sum16` / `dot16`: 16 lanes, then a fixed pairwise tree.
  * Q8/Q4 dot products use exact int32 block sums with a fixed float
    accumulation order.
  * Routed expert outputs are summed in rank order, whatever order they
    finished computing in.
  * A batched result equals the single-token result for every (token, row).
* No FMA contraction in canonical float math: MSVC `/fp:precise`, GCC/Clang
  `-ffp-contract=off`, never `-ffast-math`.
* The invariants INV-DET-1 (scheduling and placement independence), INV-DET-2
  (batch equals sequential) and INV-DET-3 (ISA identity) are enforced by golden
  tests.

## Consequences

* SIMD kernels cannot use some throughput tricks, such as FMA in canonical
  accumulation or reassociated reductions. The 16-lane layout maps directly
  onto AVX-512 and onto two AVX2 registers.
* Any output difference between configurations is a bug, which makes cache,
  prefetch and threading bugs visible in tests.
* The engine matches the numpy reference within tolerance (INV-NUM-1), not
  bit for bit. MLA uses the absorbed form, which is algebraically equal to the
  reference but not identical in rounding.
* A GPU backend must reproduce these semantics or document its tolerance
  (task R01).
