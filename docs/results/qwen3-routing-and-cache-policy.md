# Real routing on Qwen3-30B-A3B and what it means for the expert cache

**Status:** measured traces, **simulated** cache behaviour (hearth.sim; docs/SIMULATOR.md).
Date 2026-10-05. Model: Qwen/Qwen3-30B-A3B converted to Q4 experts / Q8 dense
(17.0 GB). Traces: `scripts/record_traces.py`, greedy decoding, chat template,
5 prompts (code, math, story, facts, translation), 1,280 tokens total, recorded by
the engine (docs/FORMAT.md §9). Trace file: `docs/results/traces/qwen3-30b-a3b-5tasks.hrtr`.

## Routing statistics

| scope | entropy (uniform = 7.00 bits) | mass on top 10% of experts | next-token expert reuse |
|---|---|---|---|
| single task (5 tasks) | 5.22 – 5.89 bits | 42 – 58% | 40 – 50% |
| all tasks mixed | 6.55 bits | 26% | 45% |

Skew is strong *within* a task and shifts *between* tasks; temporal locality is high.

## Cache policies on the real trace (simulated, this PC profile)

Hit rate per (token, layer, expert) activation, no prefetch:

| cache | LRU | LRU per layer | LFU decay 0.995 (old default) | LFU decay 0.9 | static pinned | LFU+pinned | Belady (optimal) |
|---|---|---|---|---|---|---|---|
| 1 GiB | 45.0% | **46.6%** | 31.4% | 44.4% | 7.1% | 23.1% | 64.7% |
| 2 GiB | 60.5% | 62.8% | 47.3% | **62.6%** (0.8: 63.1%) | 15.6% | 37.6% | 79.9% |
| 4 GiB | 82.4% | 82.0% | 69.2% | **82.8%** | 31.2% | 59.9% | 92.0% |
| 8 GiB | **95.9%** | 95.3% | 91.8% | — | 58.5% | 89.0% | 98.5% |

## Findings (epistemic ledger)

* **FALSIFIED** — "LFU with slow decay beats LRU because of cyclic layer access"
  (docs/ARCHITECTURE.md, ADR on the default policy). On real traces the cyclic
  pathology does not bite once the cache exceeds one token's working set
  (48 × 8 slabs ≈ 0.9 GiB here), and decay 0.995 adapts too slowly to task shifts:
  it loses 8–13 points to LRU.
* **CONFIRMED** — a fast-decaying LFU (decay 0.8–0.9 per token, half-life 3–7 tokens)
  matches or slightly beats LRU (+0.4 to +2.6 points at 2–4 GiB).
* **CONFIRMED** — a static heat profile goes stale under task shift: pinning half the
  cache from a warm-up profile is the worst policy measured.
* **NEW** — one-layer-ahead prefetch (75% recall, 2 GiB, LRU) turns demand misses from
  39.5% into 12.4% (+32.0% prefetched) but 72.8% of prefetches arrive late on an
  NVMe-bound configuration, so decode only improves 10.9 → 11.5 tok/s (simulated).
  Multi-layer lookahead is the lever, not more recall.
* **NEW** — the gap to Belady (15–20 points at 1–2 GiB) is the head-room for
  prediction-aware eviction.

## Consequences

1. Default eviction: fast-decay LFU (0.9) or LRU — decided by the real-engine
   ablation in docs/BENCHMARKS.md, not by this simulation alone.
2. Roadmap: multi-layer lookahead prediction; prediction-aware eviction; pinning only
   as an explicit per-workload option.
