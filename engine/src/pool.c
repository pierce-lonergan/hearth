/*
 * pool.c — persistent compute thread pool (hx_pool.h).
 *
 * Dispatch protocol:
 *   caller: write fn/ctx, pending = n-1, gen++ (release); run tid 0; wait pending == 0.
 *   worker: wait until gen != seen (spin spin_us, then sleep on cv); run; pending--.
 * Sleeping is coordinated without a lock on the fast path: a sleeper bumps
 * `sleepers` before re-checking gen, the caller bumps gen before reading
 * `sleepers` (both seq_cst), so at least one side sees the other and no
 * wake-up is lost. The caller's own completion wait uses the same pattern
 * with `caller_sleeping`.
 */
#include "hx_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HX__LINE 64
#define HX__SPIN_CHECK 64   /* relax iterations between clock reads */

typedef struct hx__worker {
    hx_pool *pool;
    hx_thread *th;
    int tid;
} hx__worker;

struct hx_pool {
    /* written once per dispatch by the caller, read by every worker */
    atomic_uint gen;
    atomic_int stop;
    hx_task_fn fn;
    void *ctx;
    char pad0[HX__LINE];

    /* written by every worker once per dispatch */
    atomic_int pending;
    char pad1[HX__LINE];

    /* sleep coordination (cold while the pool is busy) */
    atomic_int sleepers;
    atomic_int caller_sleeping;
    hx_mutex mu;
    hx_cond cv;
    hx_mutex done_mu;
    hx_cond done_cv;
    char pad2[HX__LINE];

    /* caller-private / read-only after create */
    atomic_int busy;
    int n;
    uint64_t spin_ns;
    hx__worker *workers;     /* n-1 entries; tid = index + 1 */
};

static int hx__spin_until(atomic_uint *gen, unsigned seen, uint64_t spin_ns, unsigned *out) {
    if (spin_ns == 0) return 0;
    uint64_t deadline = hx_now_ns() + spin_ns;
    for (;;) {
        for (int i = 0; i < HX__SPIN_CHECK; i++) {
            unsigned g = atomic_load_explicit(gen, memory_order_acquire);
            if (g != seen) { *out = g; return 1; }
            hx_cpu_relax();
        }
        if (hx_now_ns() >= deadline) return 0;
    }
}

static unsigned hx__wait_gen(hx_pool *p, unsigned seen, uint64_t spin_ns) {
    unsigned g;
    if (hx__spin_until(&p->gen, seen, spin_ns, &g)) return g;
    hx_mutex_lock(&p->mu);
    atomic_fetch_add(&p->sleepers, 1);
    while ((g = atomic_load(&p->gen)) == seen) hx_cond_wait(&p->cv, &p->mu);
    atomic_fetch_sub(&p->sleepers, 1);
    hx_mutex_unlock(&p->mu);
    return g;
}

static void *hx__worker_main(void *arg) {
    hx__worker *w = (hx__worker *)arg;
    hx_pool *p = w->pool;
    const int tid = w->tid, n = p->n;
    const uint64_t spin_ns = p->spin_ns;          /* local: p's last line is written per dispatch */
    unsigned seen = 0;
    for (;;) {
        seen = hx__wait_gen(p, seen, spin_ns);
        if (atomic_load_explicit(&p->stop, memory_order_acquire)) break;
        p->fn(p->ctx, tid, n);
        if (atomic_fetch_sub(&p->pending, 1) == 1 && atomic_load(&p->caller_sleeping)) {
            hx_mutex_lock(&p->done_mu);
            hx_cond_signal(&p->done_cv);
            hx_mutex_unlock(&p->done_mu);
        }
    }
    return NULL;
}

static void hx__wait_done(hx_pool *p) {
    if (p->spin_ns) {
        uint64_t deadline = 0;
        for (;;) {
            for (int i = 0; i < HX__SPIN_CHECK; i++) {
                if (atomic_load_explicit(&p->pending, memory_order_acquire) == 0) return;
                hx_cpu_relax();
            }
            uint64_t now = hx_now_ns();
            if (!deadline) deadline = now + p->spin_ns;
            else if (now >= deadline) break;
        }
    }
    hx_mutex_lock(&p->done_mu);
    atomic_store(&p->caller_sleeping, 1);
    while (atomic_load(&p->pending) != 0) hx_cond_wait(&p->done_cv, &p->done_mu);
    atomic_store(&p->caller_sleeping, 0);
    hx_mutex_unlock(&p->done_mu);
}

static void hx__release(hx_pool *p) {
    if (p->workers) {
        for (int i = 0; i < p->n - 1; i++) hx_thread_join(p->workers[i].th);
        free(p->workers);
    }
    hx_cond_destroy(&p->done_cv);
    hx_mutex_destroy(&p->done_mu);
    hx_cond_destroy(&p->cv);
    hx_mutex_destroy(&p->mu);
    hx_aligned_free(p);
}

static void hx__stop_workers(hx_pool *p) {
    atomic_store(&p->stop, 1);
    hx_mutex_lock(&p->mu);
    atomic_fetch_add(&p->gen, 1);
    hx_cond_broadcast(&p->cv);
    hx_mutex_unlock(&p->mu);
}

hx_pool *hx_pool_create(int n_threads, int spin_us) {
    if (n_threads < 1) return NULL;
    hx_pool *p = (hx_pool *)hx_aligned_alloc(HX__LINE, sizeof *p);
    if (!p) return NULL;
    memset(p, 0, sizeof *p);
    atomic_init(&p->gen, 0u);
    atomic_init(&p->stop, 0);
    atomic_init(&p->pending, 0);
    atomic_init(&p->sleepers, 0);
    atomic_init(&p->caller_sleeping, 0);
    atomic_init(&p->busy, 0);
    hx_mutex_init(&p->mu);
    hx_cond_init(&p->cv);
    hx_mutex_init(&p->done_mu);
    hx_cond_init(&p->done_cv);
    p->n = n_threads;
    p->spin_ns = spin_us > 0 ? (uint64_t)spin_us * 1000u : 0;
    if (n_threads == 1) return p;

    p->workers = (hx__worker *)calloc((size_t)(n_threads - 1), sizeof *p->workers);
    if (!p->workers) { hx__release(p); return NULL; }
    for (int i = 0; i < n_threads - 1; i++) {
        p->workers[i].pool = p;
        p->workers[i].tid = i + 1;
        if (hx_thread_create(&p->workers[i].th, hx__worker_main, &p->workers[i]) != 0) {
            hx_log(HX_LOG_ERROR, "hx_pool_create: could not start worker %d of %d", i + 1, n_threads - 1);
            hx__stop_workers(p);
            for (int j = 0; j < i; j++) hx_thread_join(p->workers[j].th);
            free(p->workers);
            p->workers = NULL;
            hx__release(p);
            return NULL;
        }
    }
    return p;
}

void hx_pool_destroy(hx_pool *p) {
    if (!p) return;
    if (p->n > 1) hx__stop_workers(p);
    hx__release(p);
}

int hx_pool_size(const hx_pool *p) { return p ? p->n : 0; }

void hx_pool_run(hx_pool *p, hx_task_fn fn, void *ctx) {
    if (p->n == 1) { fn(ctx, 0, 1); return; }
    if (atomic_exchange_explicit(&p->busy, 1, memory_order_acquire)) {
        hx_log(HX_LOG_ERROR, "hx_pool_run: called from inside a task or from two threads at once");
        abort();
    }
    p->fn = fn;
    p->ctx = ctx;
    atomic_store_explicit(&p->pending, p->n - 1, memory_order_relaxed);
    atomic_fetch_add(&p->gen, 1);
    if (atomic_load(&p->sleepers) > 0) {
        hx_mutex_lock(&p->mu);
        hx_cond_broadcast(&p->cv);
        hx_mutex_unlock(&p->mu);
    }
    fn(ctx, 0, p->n);
    hx__wait_done(p);
    atomic_store_explicit(&p->busy, 0, memory_order_release);
}

typedef struct hx__for {
    atomic_llong next;       /* next chunk index */
    int64_t n, chunk, nchunks;
    hx_range_fn fn;
    void *ctx;
} hx__for;

static void hx__for_task(void *c, int tid, int nthreads) {
    hx__for *f = (hx__for *)c;
    const int64_t n = f->n, chunk = f->chunk, nchunks = f->nchunks;
    const hx_range_fn fn = f->fn;
    void *ctx = f->ctx;
    (void)nthreads;
    for (;;) {
        int64_t i = atomic_fetch_add_explicit(&f->next, 1, memory_order_relaxed);
        if (i >= nchunks) break;
        int64_t b = i * chunk;
        int64_t e = n - b > chunk ? b + chunk : n;
        fn(ctx, b, e, tid);
    }
}

void hx_pool_for(hx_pool *p, int64_t n, int64_t chunk, hx_range_fn fn, void *ctx) {
    if (n <= 0) return;
    if (chunk < 1) chunk = 1;
    int64_t nchunks = n / chunk + (n % chunk != 0);
    if (p->n == 1 || nchunks == 1) {
        for (int64_t i = 0; i < nchunks; i++) {
            int64_t b = i * chunk;
            fn(ctx, b, n - b > chunk ? b + chunk : n, 0);
        }
        return;
    }
    hx__for f;
    atomic_init(&f.next, 0);
    f.n = n;
    f.chunk = chunk;
    f.nchunks = nchunks;
    f.fn = fn;
    f.ctx = ctx;
    hx_pool_run(p, hx__for_task, &f);
}
