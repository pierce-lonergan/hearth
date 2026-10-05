# ADR-0004: Decayed-frequency (LFU) expert eviction by default; LRU as baseline

- Status: Accepted
- Date: 2026-10-04
- Supersedes: none
- Superseded-by: none

## Context

Layers are visited cyclically: token t touches the MoE layers 0..L-1, then
token t+1 touches them again. Suppose the cache holds fewer slabs than one
token's working set (`L_moe * top_k`), which is the normal case when a model
far exceeds RAM. Then global LRU always evicts the slab that will be needed
soonest, and its hit rate collapses toward zero, even though expert activation
frequencies are strongly skewed (docs/ARCHITECTURE.md, "Why LFU beats LRU
here"). A policy that tracks frequency keeps every layer's hot experts
resident whatever the cycle length.

## Decision

* The default policy is `HEARTH_POLICY_LFU`: a per-token decayed "heat" count
  (default decay 0.995) with sampled-candidate eviction.
* Slots that are pinned, in use (refcount > 0) or loading are never evicted.
  Experts prefetched for the current token are protected until they are used
  or the token ends (engine/src/hx_store.h).
* `HEARTH_POLICY_LRU` stays available as the named baseline for measurements.
* A saved heat profile (`.usage`, FORMAT.md §8) can pin the globally hottest
  experts and warm the cache at open.
* The policy can affect speed, never output (INV-DET-1, INV-LOSSLESS-DEFAULT).

## Consequences

* Hit-rate and speed-up claims must name LRU (or another measured policy) as
  the baseline, on the same machine, labelled measured or simulated
  (INV-HONEST). The trace-driven simulator (`hearth.sim`, task T04) compares
  LRU, LFU, pinned and Belady policies on real routing traces.
* LFU adapts more slowly to a change of topic. The decay constant is the knob,
  and it should be tuned on traces, not by intuition.
* Revisit if traces show a policy (for example frequency plus prediction) that
  beats decayed LFU on representative workloads.
