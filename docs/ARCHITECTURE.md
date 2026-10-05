# Hearth architecture

## The physics

A decoding step of an MoE model is **memory-bound**: every weight touched is
used for one multiply-add per token. For a model with `L_moe` MoE layers, top-`k`
routing and slab size `S` bytes per expert, the routed traffic per token is

```
bytes/token = L_moe · k · S · (1 − hit_rate)          (from storage)
            + L_moe · k · S · hit_rate                (from DRAM)
            + dense_bytes                             (from DRAM, every token)
```

and the best possible decode rate is roughly
`1 / ( storage_bytes/storage_BW + dram_bytes/dram_BW + fixed_compute )`,
with storage reads overlapped against compute where possible. Every Hearth
feature attacks one term:

| lever | term it shrinks | mechanism |
|-------|-----------------|-----------|
| expert slabs + direct I/O | storage latency/overhead | one aligned read per expert, no page-cache copies or thrash; on Windows, overlapped handles so concurrent reads reach queue depth > 1 |
| LFU heat cache, pinning | `1 − hit_rate` | activation frequencies are power-law; global LRU degenerates under cyclic layer access |
| next-layer prediction | exposed storage latency | reads for layer L+1 start while layer L computes |
| rank-order accumulation + completion-order compute | exposed latency | experts are computed as they arrive, summed in canonical order (bit-identical) |
| batched eval (INV-DET-2) | reads per token | prefill / speculative verification load each needed expert once for many tokens |
| prompt-lookup speculative decoding | reads per *accepted* token | lossless; cost amortised over accepted drafts |
| mirrors | storage BW | reads spread over byte-identical copies on separate drives |
| Q4 (4.25 bpw), later Q3 | `S` | fewer bytes per expert |
| persistent pool, cross-expert regions | fixed compute | no per-region fork/join |

## Components

```
            ┌──────────────────────── hearth_eval(tokens) ──────────────────────────┐
            │ model.c                                                                │
            │  embed → [attn (resident) → router → MoE (streamed) ] × L → head       │
            │           │                    │  ▲                                    │
            │           │       predict L+1  │  │ completion-order compute,          │
            │           │      ─────────────►│  │ rank-order sum                     │
            └───────────┼────────────────────┼──┼────────────────────────────────────┘
                        │                    ▼  │
   pool.c  ◄────────────┘           store.c (expert store)
   persistent workers                ├─ slots: DRAM arena, slab-sized, 4096-aligned
   hx_pool_for                       ├─ slot_of[layer·E + e] direct index
                                     ├─ LFU heat / LRU, pins, prefetch protection
                                     └─ reader threads ── demand queue (high)
                                                       └─ prefetch queue (low)
                                                             │ pread, O_DIRECT
                                                ┌────────────┴────────────┐
                                           model.hearth            mirror copies
                                           (NVMe 0)                (NVMe 1..n)
```

* **modelfile.c** — parses and validates the container (docs/FORMAT.md), loads
  the dense region into one arena.
* **store.c** — the tiered expert store (engine/src/hx_store.h).
* **quant.c / quant_avx2.c / quant_avx512.c** — formats and kernels
  (docs/NUMERICS.md), runtime ISA dispatch.
* **model.c** — forward pass for GQA and MLA attention, softmax/sigmoid/group
  routers, shared experts, dense layers; batched evaluation; prediction and
  prefetch; routing trace.
* **api.c** — `hearth.h`.
* **python/hearth** — converter, reference, runtime, simulator (docs/PYTHON_API.md).

## MoE layer schedule (decode and batched)

1. Router for all tokens in the batch → top-k ids and weights per token.
2. Union of needed experts; for each, `hx_store_try_acquire`. Misses become
   demand reads immediately.
3. Prediction for layer L+1 (two-step: post-attention state + shared-expert
   output, routed through layer L+1's router) → `hx_store_prefetch`. Shared
   expert compute happens here too, overlapping outstanding reads.
4. Loop: compute every resident expert for all tokens routed to it (one pass over
   the weights, T activations — `hx_matmul_fn` with T>1), writing per-(token,
   rank) output buffers; release; `hx_store_wait_any` for the rest.
5. Per token: sum expert outputs in rank order (+ shared) — NUMERICS §5.3.

Because each expert's output lands in its own buffer and the final sum follows
rank order, completion order, cache state and thread count cannot change the
result (INV-DET-1).

## Why LFU beats LRU here

Layers are visited cyclically: token t touches layer 0..L-1, then token t+1 does
it again. If the cache holds fewer slabs than one token's working set
(`L_moe·k`), global LRU always evicts exactly the slab that will be needed
soonest — hit rate collapses to ~0 even though activation frequencies are
highly skewed. A frequency (heat) policy keeps the hot experts of every layer
resident regardless of cycle length. `hearth.sim` quantifies this on real
routing traces.
