/*
 * pool.c — persistent compute thread pool (hx_pool.h).
 *
 * Dispatch protocol:
 *   caller: write fn/ctx, pending = k-1, publish word = (seq << 16) | k;
 *           wake the sleeping workers among tids 1..k-1; run tid 0; wait pending == 0.
 *   worker: wait for a word with a new seq and k > tid (spin, then sleep on its own
 *           condvar); run fn(ctx, tid, k); pending--.
 * One 64-bit word carries both the sequence number and k, so a worker outside a dispatch
 * (tid >= k) never reads fn/ctx: the caller does not wait for it and may already be
 * publishing the next dispatch. 48 sequence bits do not wrap in practice.
 *
 * Sleeping is coordinated without a lock on the fast path: a sleeper sets its `sleeping` flag
 * and bumps `sleepers` before re-reading the word, the caller publishes the word before
 * reading `sleepers` and the flags (all seq_cst), so at least one side sees the other and no
 * wake-up is lost. The caller's completion wait uses the same pattern with `caller_sleeping`.
 * Per-worker condvars let a dispatch wake only its own threads, and woken workers do not
 * queue on one mutex (sleeping-pool dispatch, 32 threads: p50 106 -> 57 us).
 *
 * Spinning threads (workers waiting for work, the caller waiting for workers) yield the CPU
 * every HX__YIELD_NS. A pool as large as the machine needs all of its threads on a CPU to
 * finish a dispatch; when anything else takes a CPU, the displaced pool thread is ready but
 * every other CPU is held by a spinner. Without yields it got one back only when a spinner
 * reached its spin_us deadline and slept, so every dispatch took ~spin_us (32 threads on 32
 * logical CPUs: p50 237 / 1031 / 3034 us at spin_us 200 / 1000 / 3000).
 *
 * A yield only reaches threads queued on the yielding CPU. A participant queued behind a
 * thread of another process (which does not yield) still waited for a spinner's deadline or
 * that thread's time slice: rare, but such dispatches took ~spin_us and dominated the mean.
 * Freshly woken workers were caught the same way (32 threads beside 16 busy threads: 3 ms
 * dispatches with every worker but one started). So a dispatch with a participant that has
 * not picked it up HX__STALL_NS after the caller started waiting is declared stalled: the
 * caller and every worker that has seen it stop spinning and sleep, an idle CPU takes the
 * stranded thread (Windows does so at once), and the next dispatch wakes the sleepers. Only
 * a participant that has not started counts, so slow tasks are not stalls; a dispatch that
 * had to wake workers allows them HX__WAKE_STALL_NS, so slow wake-ups (VMs) do not chain
 * stall -> sleep -> wake -> stall. The caller's first yield comes at its first stall check: a
 * yield can hand its CPU to another process for a whole time slice. It does yield after that,
 * or with the process pinned to one CPU it would hold the CPU its workers wait for.
 *
 * When no CPU is idle (other processes keep every CPU busy) a stall cannot be rescued, and a
 * spinner that yields may get its CPU back only after another process's time slice: a spinning
 * pool then loses to a sleeping one, whose woken threads the kernel favours (32 threads beside
 * 32 busy threads: 10-31 ms vs 0.1-0.2 ms per dispatch). So a stall that took longer than
 * HX__STARVED_NS to clear, or a waiting caller kept off its CPU that long, makes the whole pool
 * sleep-only (nobody spins) for a back-off period that doubles while it keeps happening, from
 * HX__CALM_MIN_NS to HX__CALM_MAX_NS. (Workers kept off their CPUs while idle are not counted:
 * on a busy machine that happens all the time and costs nothing until the next stall.)
 *
 * A worker's spin window starts when it finishes its own task; dispatches without it do not
 * extend it (hx_pool.h), so workers left out of small hx_pool_for_n regions go to sleep.
 */
#include "hx_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HX__LINE 64
#define HX__SPIN_CHECK 64      /* polls between clock reads (~1 us) */
#define HX__YIELD_NS 2000      /* spinning threads yield the CPU this often */
#define HX__STALL_NS 20000     /* a participant that has not started by then is descheduled */
#define HX__WAKE_STALL_NS 200000   /* the same for a dispatch that had to wake workers */
#define HX__STARVED_NS 500000  /* kept from a CPU this long: no CPU is idle */
#define HX__CALM_MIN_NS 10000000ull
#define HX__CALM_MAX_NS 1000000000ull
#define HX__K_BITS 16
#define HX__K_MASK ((1u << HX__K_BITS) - 1u)
#define HX__NO_SEQ (~0ull)     /* never a sequence number (48 bits) */

typedef struct hx__worker {
    hx_pool *pool;
    hx_thread *th;
    int tid;
    atomic_int sleeping;     /* polled by the caller when sleepers > 0 */
    atomic_ullong started;   /* seq of the last dispatch it picked up; read by the stall check */
    hx_mutex mu;
    hx_cond cv;
} hx__worker;
/* one worker per cache line */
typedef union hx__wslot {
    hx__worker w;
    char line[(sizeof(hx__worker) + HX__LINE - 1) / HX__LINE * HX__LINE];
} hx__wslot;

struct hx_pool {
    /* written once per dispatch by the caller, read by every worker */
    atomic_ullong word;      /* (seq << HX__K_BITS) | threads in the dispatch */
    atomic_ullong stalled;   /* seq of a stalled dispatch, else HX__NO_SEQ */
    atomic_ullong calm_until; /* hx_now_ns() before which nobody spins; 0: none */
    atomic_int stop;
    hx_task_fn fn;
    void *ctx;
    char pad0[HX__LINE];

    /* written by every participating worker once per dispatch */
    atomic_int pending;
    char pad1[HX__LINE];

    /* sleep coordination (cold while the pool is busy) */
    atomic_int sleepers;
    atomic_int caller_sleeping;
    hx_mutex done_mu;
    hx_cond done_cv;
    char pad2[HX__LINE];

    /* caller-private / read-only after create */
    atomic_int busy;
    int n;
    int started;             /* worker threads that exist */
    uint64_t spin_ns;
    uint64_t seq;
    uint64_t calm_ns;        /* length of the last sleep-only period */
    uint64_t calm_end;       /* end of the last sleep-only period */
    hx__wslot *slots;        /* n-1 entries; tid = index + 1 */
};

/* Inside a sleep-only period (now: hx_now_ns()). */
static int hx__calm(hx_pool *p, uint64_t now) {
    uint64_t until = atomic_load_explicit(&p->calm_until, memory_order_relaxed);
    return until && now < until;
}

HX_INLINE int hx__k(uint64_t w) { return (int)(w & HX__K_MASK); }
HX_INLINE uint64_t hx__seq(uint64_t w) { return w >> HX__K_BITS; }

/* The next dispatch this worker belongs to (new seq, k > tid). Spins for spin_ns after its
 * last task, or until the newest dispatch it has seen is declared stalled; then sleeps. */
static uint64_t hx__wait_word(hx_pool *p, hx__worker *wk, uint64_t last_seq, uint64_t spin_ns) {
    const int tid = wk->tid;
    uint64_t w, seen = last_seq;
    uint64_t now = spin_ns ? hx_now_ns() : 0;
    if (spin_ns && !hx__calm(p, now)) {
        uint64_t deadline = now + spin_ns, next_yield = now + HX__YIELD_NS;
        for (;;) {
            for (int i = 0; i < HX__SPIN_CHECK; i++) {
                w = atomic_load(&p->word);
                if (hx__seq(w) != seen) {
                    if (hx__k(w) > tid) return w;
                    seen = hx__seq(w);            /* a dispatch without this worker */
                }
                hx_cpu_relax();
            }
            if (atomic_load_explicit(&p->stalled, memory_order_relaxed) == seen) break;
            now = hx_now_ns();
            if (now >= deadline || hx__calm(p, now)) break;
            if (now >= next_yield) {
                hx_yield();
                next_yield = hx_now_ns() + HX__YIELD_NS;
            }
        }
    }
    hx_mutex_lock(&wk->mu);
    atomic_store(&wk->sleeping, 1);
    atomic_fetch_add(&p->sleepers, 1);
    for (;;) {
        w = atomic_load(&p->word);
        if (hx__seq(w) != last_seq && hx__k(w) > tid) break;
        hx_cond_wait(&wk->cv, &wk->mu);
    }
    atomic_fetch_sub(&p->sleepers, 1);
    atomic_store(&wk->sleeping, 0);
    hx_mutex_unlock(&wk->mu);
    return w;
}

static void *hx__worker_main(void *arg) {
    hx__worker *wk = (hx__worker *)arg;
    hx_pool *p = wk->pool;
    const int tid = wk->tid;
    const uint64_t spin_ns = p->spin_ns;          /* local: p's last line is written per dispatch */
    uint64_t last_seq = 0;
    for (;;) {
        uint64_t w = hx__wait_word(p, wk, last_seq, spin_ns);
        if (atomic_load_explicit(&p->stop, memory_order_acquire)) break;
        last_seq = hx__seq(w);
        atomic_store_explicit(&wk->started, last_seq, memory_order_relaxed);
        p->fn(p->ctx, tid, hx__k(w));
        if (atomic_fetch_sub(&p->pending, 1) == 1 && atomic_load(&p->caller_sleeping)) {
            hx_mutex_lock(&p->done_mu);
            hx_cond_signal(&p->done_cv);
            hx_mutex_unlock(&p->done_mu);
        }
    }
    return NULL;
}

/* Some participant has not picked up the current dispatch. */
static int hx__lagging(hx_pool *p, int k) {
    for (int t = 1; t < k; t++)
        if (atomic_load_explicit(&p->slots[t - 1].w.started, memory_order_relaxed) != p->seq) return 1;
    return 0;
}

/* The caller found no idle CPU (a slow stall, or itself kept from its CPU): start a sleep-only
 * period, twice as long as the last one if that ended recently. */
static void hx__backoff(hx_pool *p) {
    uint64_t now = hx_now_ns();
    if (now >= p->calm_end + 4 * p->calm_ns) p->calm_ns = 0;         /* the last period is long past */
    p->calm_ns = !p->calm_ns ? HX__CALM_MIN_NS : p->calm_ns < HX__CALM_MAX_NS / 2 ? 2 * p->calm_ns : HX__CALM_MAX_NS;
    p->calm_end = now + p->calm_ns;
    atomic_store_explicit(&p->calm_until, p->calm_end, memory_order_relaxed);
}

/* Spin for the workers (no yield in the first HX__STALL_NS); on a stall, sleep. */
static void hx__wait_done(hx_pool *p, int k, int spin, uint64_t first_check_ns) {
    uint64_t stalled_at = 0;
    int starved = 0;                              /* this thread lost its CPU for HX__STARVED_NS */
    if (spin) {
        uint64_t now = 0, deadline = 0, next_yield = 0, next_check = 0;
        for (;;) {
            for (int i = 0; i < HX__SPIN_CHECK; i++) {
                if (atomic_load_explicit(&p->pending, memory_order_acquire) == 0) {
                    if (starved) hx__backoff(p);
                    return;
                }
                hx_cpu_relax();
            }
            uint64_t t = hx_now_ns();
            if (!deadline) {                      /* no clock read on the fast path */
                deadline = t + p->spin_ns;
                next_check = t + first_check_ns;
                next_yield = t + HX__STALL_NS;
                now = t;
                continue;
            }
            starved |= t - now > HX__STARVED_NS;
            now = t;
            if (now >= deadline) break;
            if (now >= next_check) {
                if (hx__lagging(p, k)) {
                    atomic_store(&p->stalled, p->seq);
                    stalled_at = now;
                    break;
                }
                next_check = now + HX__STALL_NS;
            }
            if (now >= next_yield) {
                hx_yield();
                t = hx_now_ns();
                starved |= t - now > HX__STARVED_NS;
                now = t;
                next_yield = now + HX__YIELD_NS;
            }
        }
    }
    hx_mutex_lock(&p->done_mu);
    atomic_store(&p->caller_sleeping, 1);
    while (atomic_load(&p->pending) != 0) hx_cond_wait(&p->done_cv, &p->done_mu);
    atomic_store(&p->caller_sleeping, 0);
    hx_mutex_unlock(&p->done_mu);
    if (starved || (stalled_at && hx_now_ns() - stalled_at > HX__STARVED_NS)) hx__backoff(p);
}

static void hx__signal(hx__worker *wk) {
    hx_mutex_lock(&wk->mu);
    hx_cond_signal(&wk->cv);
    hx_mutex_unlock(&wk->mu);
}

/* Publish a dispatch of k threads; wake the sleepers among tids 1..k-1. Returns how many. */
static int hx__publish(hx_pool *p, int k) {
    int woke = 0;
    p->seq++;
    atomic_store(&p->word, (p->seq << HX__K_BITS) | (uint64_t)k);
    if (atomic_load(&p->sleepers) == 0) return 0;
    for (int t = 1; t < k; t++)
        if (atomic_load(&p->slots[t - 1].w.sleeping)) {
            hx__signal(&p->slots[t - 1].w);
            woke++;
        }
    return woke;
}

static void hx__release(hx_pool *p) {
    if (p->slots) {
        for (int i = 0; i < p->started; i++) hx_thread_join(p->slots[i].w.th);
        for (int i = 0; i < p->n - 1; i++) {
            hx_cond_destroy(&p->slots[i].w.cv);
            hx_mutex_destroy(&p->slots[i].w.mu);
        }
        hx_aligned_free(p->slots);
    }
    hx_cond_destroy(&p->done_cv);
    hx_mutex_destroy(&p->done_mu);
    hx_aligned_free(p);
}

static void hx__stop_workers(hx_pool *p) {
    atomic_store(&p->stop, 1);
    p->seq++;
    atomic_store(&p->word, (p->seq << HX__K_BITS) | (uint64_t)p->n);
    /* every worker, flag or not: one may be between its flag store and its wait */
    for (int t = 1; t <= p->started; t++) hx__signal(&p->slots[t - 1].w);
}

hx_pool *hx_pool_create(int n_threads, int spin_us) {
    if (n_threads < 1 || n_threads > (int)HX__K_MASK) return NULL;
    hx_pool *p = (hx_pool *)hx_aligned_alloc(HX__LINE, sizeof *p);
    if (!p) return NULL;
    memset(p, 0, sizeof *p);
    atomic_init(&p->word, 0u);
    atomic_init(&p->stalled, HX__NO_SEQ);
    atomic_init(&p->calm_until, 0u);
    atomic_init(&p->stop, 0);
    atomic_init(&p->pending, 0);
    atomic_init(&p->sleepers, 0);
    atomic_init(&p->caller_sleeping, 0);
    atomic_init(&p->busy, 0);
    hx_mutex_init(&p->done_mu);
    hx_cond_init(&p->done_cv);
    p->n = n_threads;
    p->spin_ns = spin_us > 0 ? (uint64_t)spin_us * 1000u : 0;
    if (n_threads == 1) return p;

    size_t bytes = (size_t)(n_threads - 1) * sizeof *p->slots;
    p->slots = (hx__wslot *)hx_aligned_alloc(HX__LINE, bytes);
    if (!p->slots) { hx__release(p); return NULL; }
    memset(p->slots, 0, bytes);
    for (int i = 0; i < n_threads - 1; i++) {
        hx__worker *wk = &p->slots[i].w;
        wk->pool = p;
        wk->tid = i + 1;
        atomic_init(&wk->sleeping, 0);
        atomic_init(&wk->started, 0u);
        hx_mutex_init(&wk->mu);
        hx_cond_init(&wk->cv);
    }
    for (int i = 0; i < n_threads - 1; i++) {
        if (hx_thread_create(&p->slots[i].w.th, hx__worker_main, &p->slots[i].w) != 0) {
            hx_log(HX_LOG_ERROR, "hx_pool_create: could not start worker %d of %d", i + 1, n_threads - 1);
            hx__stop_workers(p);
            hx__release(p);
            return NULL;
        }
        p->started = i + 1;
    }
    return p;
}

void hx_pool_destroy(hx_pool *p) {
    if (!p) return;
    if (p->n > 1) hx__stop_workers(p);
    hx__release(p);
}

int hx_pool_size(const hx_pool *p) { return p ? p->n : 0; }

/* fn on tids 0..k-1 (2 <= k <= n); tid 0 is the caller. */
static void hx__dispatch(hx_pool *p, int k, hx_task_fn fn, void *ctx) {
    if (atomic_exchange_explicit(&p->busy, 1, memory_order_acquire)) {
        hx_log(HX_LOG_ERROR, "hx_pool: dispatch from inside a task or from two threads at once");
        abort();
    }
    p->fn = fn;
    p->ctx = ctx;
    atomic_store_explicit(&p->pending, k - 1, memory_order_relaxed);
    int spin = p->spin_ns != 0;
    if (spin && atomic_load_explicit(&p->calm_until, memory_order_relaxed)) {   /* no clock read otherwise */
        if (hx__calm(p, hx_now_ns())) spin = 0;
        else atomic_store_explicit(&p->calm_until, 0u, memory_order_relaxed);
    }
    int woke = hx__publish(p, k);
    fn(ctx, 0, k);
    hx__wait_done(p, k, spin, woke ? HX__WAKE_STALL_NS : HX__STALL_NS);
    atomic_store_explicit(&p->busy, 0, memory_order_release);
}

void hx_pool_run(hx_pool *p, hx_task_fn fn, void *ctx) {
    if (p->n == 1) { fn(ctx, 0, 1); return; }
    hx__dispatch(p, p->n, fn, ctx);
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

void hx_pool_for_n(hx_pool *p, int max_threads, int64_t n, int64_t chunk, hx_range_fn fn, void *ctx) {
    if (n <= 0) return;
    if (chunk < 1) chunk = 1;
    int64_t nchunks = n / chunk + (n % chunk != 0);
    int k = max_threads < p->n ? max_threads : p->n;
    if ((int64_t)k > nchunks) k = (int)nchunks;   /* idle participants would only add waiting */
    if (k <= 1) {
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
    hx__dispatch(p, k, hx__for_task, &f);
}

void hx_pool_for(hx_pool *p, int64_t n, int64_t chunk, hx_range_fn fn, void *ctx) {
    hx_pool_for_n(p, p->n, n, chunk, fn, ctx);
}
