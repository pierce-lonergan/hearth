/*
 * hx_pool.h — persistent compute thread pool.
 *
 * Why not OpenMP: MSVC only ships OpenMP 2.0, and per-region fork/join costs
 * 5–10 µs with hundreds of regions per token. Hearth keeps one pool of
 * spinning-then-sleeping workers and dispatches with a single generation
 * counter, so a dispatch costs ~1 µs and a whole layer's experts can share one
 * region (cross-expert parallelism).
 *
 * Determinism: work items are assigned dynamically (atomic counter), but every
 * output element is produced entirely by one thread, so results never depend on
 * scheduling (docs/NUMERICS.md).
 */
#ifndef HX_POOL_H
#define HX_POOL_H

#include "hx_platform.h"

typedef struct hx_pool hx_pool;

/* fn is called once on every thread (tid 0 = caller) with nthreads = pool size. */
typedef void (*hx_task_fn)(void *ctx, int tid, int nthreads);
/* fn processes items [begin, end). */
typedef void (*hx_range_fn)(void *ctx, int64_t begin, int64_t end, int tid);

/* n_threads includes the calling thread (spawns n_threads-1 workers). n_threads >= 1.
 * Workers spin for ~spin_us microseconds after finishing a task before sleeping; while spinning
 * they yield the CPU every few microseconds so descheduled siblings and the caller can run. */
hx_pool *hx_pool_create(int n_threads, int spin_us);
void     hx_pool_destroy(hx_pool *p);
int      hx_pool_size(const hx_pool *p);

/* Run fn on all threads; returns when every thread has returned. Not reentrant:
 * must not be called from inside a task. */
void hx_pool_run(hx_pool *p, hx_task_fn fn, void *ctx);

/* Dynamic parallel-for over [0, n) in chunks of `chunk` items (last may be short).
 * Each chunk is processed by exactly one thread. chunk >= 1. */
void hx_pool_for(hx_pool *p, int64_t n, int64_t chunk, hx_range_fn fn, void *ctx);
/* Same, but wakes and waits for at most max_threads threads (incl. the caller), so small
 * regions don't pay for the whole pool. max_threads <= 1 runs on the caller only. */
void hx_pool_for_n(hx_pool *p, int max_threads, int64_t n, int64_t chunk, hx_range_fn fn, void *ctx);

/* Deterministic static split helper: contiguous balanced partition of n items. */
HX_INLINE void hx_split(int64_t n, int tid, int nt, int64_t *b, int64_t *e) {
    int64_t q = n / nt, r = n % nt;
    *b = tid * q + (tid < r ? tid : r);
    *e = *b + q + (tid < r ? 1 : 0);
}

#endif /* HX_POOL_H */
