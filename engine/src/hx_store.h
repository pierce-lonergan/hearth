/*
 * hx_store.h — the tiered expert store: NVMe -> DRAM cache, asynchronous.
 *
 * The store owns a fixed pool of slab-sized, 4096-aligned DRAM slots and a set
 * of reader threads. Compute asks for experts; hits return immediately, misses
 * are queued as *demand* reads (highest priority). The forward pass hints the
 * next layer's likely experts as *prefetch* reads (low priority, droppable).
 *
 * Key design points (docs/ARCHITECTURE.md):
 *   - slot_of[layer*E + e] direct index, no hashing.
 *   - Eviction LFU: decayed per-token "heat", sampled-candidate eviction, never
 *     evicts pinned / in-use (refcount>0) / loading slots; experts prefetched for
 *     the current token are protected until used or the token ends.
 *     LRU is provided as a baseline. Global LRU thrashes when the cache is smaller
 *     than one token's working set because layers are accessed cyclically.
 *   - Pinning: with a heat profile, the hottest experts (globally ranked) are
 *     loaded at open and never evicted.
 *   - Mirrors: reads are spread over byte-identical copies (least outstanding I/O).
 *   - Weights are bytes; where they came from cannot change numerics (INV-DET-1).
 *
 * Threading: the store is driven by one compute thread (the engine's caller)
 * plus its own reader threads. All metadata is guarded by one mutex; slab reads
 * happen outside the lock.
 */
#ifndef HX_STORE_H
#define HX_STORE_H

#include "hx_platform.h"
#include "hx_modelfile.h"

typedef struct hx_store hx_store;

typedef struct hx_store_opts {
    uint64_t cache_bytes;        /* DRAM budget for slots; raised to the minimum (see below) if smaller */
    int n_io_threads;            /* >= 1 */
    int direct_io;               /* 1 = unbuffered reads */
    int policy;                  /* HEARTH_POLICY_LRU / HEARTH_POLICY_LFU */
    float heat_decay;            /* per-token multiplicative decay for LFU heat (default 0.995) */
    const char *usage_in;        /* optional heat profile (FORMAT.md §8) */
    const char *usage_out;       /* optional: written on close */
    float pin_fraction;          /* 0..0.9: fraction of slots pinned with hottest experts (needs usage_in) */
    int warm_start;              /* fill remaining slots with next-hottest experts at open */
    const char *const *mirrors;  /* extra copies of the model file */
    int n_mirrors;
} hx_store_opts;

/* Minimum slot count: 2*top_k + n_io_threads + 2 (guarantees forward progress). */
hx_store *hx_store_open(const hx_modelfile *mf, const hx_store_opts *o, char *err, size_t errlen);
void      hx_store_close(hx_store *s);   /* joins readers; writes usage_out if set */

/* Non-blocking. Hit: pins the slot (refcount++), counts a use, returns the slab.
 * Miss: makes sure a demand read is queued (promoting a queued/in-flight prefetch),
 * returns NULL. Call again later (e.g. after hx_store_wait_any). */
const void *hx_store_try_acquire(hx_store *s, int layer, int expert);
/* Blocking variant: returns the slab once resident (pinned). Never NULL for valid ids. */
const void *hx_store_acquire(hx_store *s, int layer, int expert);
/* Unpin (refcount--). */
void hx_store_release(hx_store *s, int layer, int expert);
/* Block until some read completes or timeout_us elapses. */
void hx_store_wait_any(hx_store *s, uint32_t timeout_us);
/* Low-priority hints. Ignored for resident/in-flight experts; may be dropped when
 * no slot can be evicted without hurting demand traffic. */
void hx_store_prefetch(hx_store *s, int layer, const int *experts, int n);
/* Token boundary: advances the heat clock and expires prefetch protection. */
void hx_store_tick(hx_store *s);
/* 1 if (layer, expert) is resident right now (for diagnostics/tests). */
int  hx_store_is_resident(hx_store *s, int layer, int expert);

typedef struct hx_store_stats {
    uint64_t hits, misses, evictions;
    uint64_t prefetch_issued, prefetch_used, prefetch_wasted;
    uint64_t bytes_read, reads;
    uint64_t read_ns;            /* summed reader busy time */
    uint64_t stall_ns;           /* time callers spent blocked in acquire/wait_any */
    int n_slots, resident, pinned;
    uint64_t slot_bytes;
} hx_store_stats;
void hx_store_get_stats(hx_store *s, hx_store_stats *out);
void hx_store_reset_stats(hx_store *s);
/* Activation counts per (layer, expert) since open (incl. usage_in seed). Length n_layers*n_experts. */
const float *hx_store_heat(hx_store *s);

#endif /* HX_STORE_H */
