/*
 * test_pool.c — self-checking tests and dispatch-latency measurement for hx_pool.h.
 *
 *   test_pool [--quick] [--no-timing]
 *
 * --quick: fewer repetitions, no latency section (for mutation testing).
 * --no-timing: skip the assertions that depend on timing or CPU time (they are self-calibrated,
 *   but a sanitizer build or a heavily loaded machine can still upset them); on by default in
 *   AddressSanitizer / ThreadSanitizer builds. Latency numbers are printed, not asserted.
 * Some checks re-run this program pinned to one CPU (start /affinity, taskset), with
 * --pinned-child FILE. Some start one busy thread per CPU for a second. Concurrent runs (unless
 * timing checks are off) take turns through a lock file in the temp directory.
 * Exit code 0 = all checks passed.
 */
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>              /* GetProcessTimes (the CRT's clock() is wall time), OpenProcess */
#else
#  if !defined(_POSIX_C_SOURCE)
#    define _POSIX_C_SOURCE 200809L /* kill, getpid under -std=c11 */
#  endif
#  include <signal.h>
#  include <unistd.h>
#endif
#include "hx_pool.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_checks, g_fail, g_no_timing;

#define CHECK(cond, ...)                                                         \
    do {                                                                         \
        g_checks++;                                                              \
        if (!(cond)) {                                                           \
            if (++g_fail <= 40) {                                                \
                printf("  FAIL %s:%d: ", __FILE__, __LINE__);                    \
                printf(__VA_ARGS__);                                             \
                printf("\n");                                                    \
            }                                                                    \
        }                                                                        \
    } while (0)

static uint64_t splitmix(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

/* CPU time of the whole process (all threads), in seconds. */
static double process_cpu_s(void) {
#if defined(_WIN32)
    FILETIME c, e, k, u;
    if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return 0.0;
    uint64_t kt = ((uint64_t)k.dwHighDateTime << 32) | k.dwLowDateTime;
    uint64_t ut = ((uint64_t)u.dwHighDateTime << 32) | u.dwLowDateTime;
    return (double)(kt + ut) * 1e-7;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}

/* ------------------------------------------------------------ hx_pool_for */

typedef struct {
    atomic_int *hits;          /* per item; NULL when n is too large to track */
    int64_t n, chunk;
    int tid_limit;             /* every tid must be below this */
    atomic_int bad_range, bad_tid, calls;
    atomic_llong covered;
} for_ctx;

static void for_body(void *c, int64_t b, int64_t e, int tid) {
    for_ctx *f = (for_ctx *)c;
    atomic_fetch_add(&f->calls, 1);
    int64_t want_e = f->n - b > f->chunk ? b + f->chunk : f->n;
    if (b < 0 || b >= e || e > f->n || b % f->chunk != 0 || e != want_e) atomic_fetch_add(&f->bad_range, 1);
    if (tid < 0 || tid >= f->tid_limit) atomic_fetch_add(&f->bad_tid, 1);
    atomic_fetch_add(&f->covered, e - b);
    if (f->hits)
        for (int64_t i = b; i < e && i < f->n; i++) atomic_fetch_add_explicit(&f->hits[i], 1, memory_order_relaxed);
}

/* max_threads == INT_MAX: hx_pool_for; else hx_pool_for_n. */
static int run_for_case(hx_pool *p, int max_threads, int64_t n, int64_t chunk) {
    for_ctx f;
    int track = n <= (1 << 20);
    int nt = hx_pool_size(p);
    f.hits = track && n > 0 ? (atomic_int *)calloc((size_t)n, sizeof(atomic_int)) : NULL;
    f.n = n;
    f.chunk = chunk;
    f.tid_limit = max_threads < 1 ? 1 : max_threads < nt ? max_threads : nt;
    atomic_init(&f.bad_range, 0);
    atomic_init(&f.bad_tid, 0);
    atomic_init(&f.calls, 0);
    atomic_init(&f.covered, 0);
    if (max_threads == INT_MAX) hx_pool_for(p, n, chunk, for_body, &f);
    else hx_pool_for_n(p, max_threads, n, chunk, for_body, &f);
    int ok = 1;
    if (n <= 0) {
        ok = atomic_load(&f.calls) == 0;
    } else {
        int64_t nchunks = n / chunk + (n % chunk != 0);
        ok = atomic_load(&f.bad_range) == 0 && atomic_load(&f.bad_tid) == 0 &&
             atomic_load(&f.calls) == nchunks && atomic_load(&f.covered) == n;
        if (f.hits)
            for (int64_t i = 0; i < n && ok; i++) ok = atomic_load(&f.hits[i]) == 1;
    }
    free((void *)f.hits);
    return ok;
}

static void test_for(void) {
    printf("hx_pool_for coverage\n");
    const int threads[] = {1, 2, 3, 4, 7, 16, 32};
    const int64_t ns[] = {0, 1, 2, 3, 7, 64, 100, 1000, 4097, 100003};
    int cases = 0;
    for (size_t ti = 0; ti < sizeof threads / sizeof threads[0]; ti++) {
        hx_pool *p = hx_pool_create(threads[ti], 50);
        CHECK(p != NULL && hx_pool_size(p) == threads[ti], "create %d", threads[ti]);
        if (!p) continue;
        for (size_t ni = 0; ni < sizeof ns / sizeof ns[0]; ni++) {
            int64_t n = ns[ni];
            int64_t chunks[] = {1, 2, 3, 7, 64, 1000, n > 0 ? n : 1, n + 1, n * 3 + 5, INT64_MAX};
            for (size_t ci = 0; ci < sizeof chunks / sizeof chunks[0]; ci++) {
                CHECK(run_for_case(p, INT_MAX, n, chunks[ci]), "for threads=%d n=%lld chunk=%lld", threads[ti],
                      (long long)n, (long long)chunks[ci]);
                cases++;
            }
        }
        /* ranges near INT64_MAX must not overflow (only the ranges are checked) */
        CHECK(run_for_case(p, INT_MAX, INT64_MAX, INT64_MAX / 2 + 1), "for n=INT64_MAX two chunks");
        CHECK(run_for_case(p, INT_MAX, INT64_MAX, INT64_MAX / 7), "for n=INT64_MAX seven chunks");
        CHECK(run_for_case(p, INT_MAX, -5, 3), "for negative n");
        cases += 3;
        hx_pool_destroy(p);
    }
    printf("  %d (threads, n, chunk) cases\n", cases);
}

static void test_for_n(void) {
    printf("hx_pool_for_n coverage\n");
    const int threads[] = {1, 2, 5, 16, 33};
    const int64_t ns[] = {0, 1, 3, 7, 100, 4097};
    const int64_t chunks[] = {1, 3, 64};
    int cases = 0;
    for (size_t ti = 0; ti < sizeof threads / sizeof threads[0]; ti++) {
        int nt = threads[ti];
        hx_pool *p = hx_pool_create(nt, 50);
        CHECK(p != NULL, "create %d", nt);
        if (!p) continue;
        const int maxes[] = {INT_MIN, -1, 0, 1, 2, 3, nt - 1, nt, nt + 1, INT_MAX - 1};
        for (size_t mi = 0; mi < sizeof maxes / sizeof maxes[0]; mi++)
            for (size_t ni = 0; ni < sizeof ns / sizeof ns[0]; ni++)
                for (size_t ci = 0; ci < sizeof chunks / sizeof chunks[0]; ci++) {
                    CHECK(run_for_case(p, maxes[mi], ns[ni], chunks[ci]), "for_n threads=%d max=%d n=%lld chunk=%lld",
                          nt, maxes[mi], (long long)ns[ni], (long long)chunks[ci]);
                    cases++;
                }
        CHECK(run_for_case(p, 2, INT64_MAX, INT64_MAX / 3), "for_n n=INT64_MAX");
        hx_pool_destroy(p);
    }
    printf("  %d (threads, max_threads, n, chunk) cases\n", cases);
}

/* ------------------------------------------------------------ hx_pool_run */

#define MAXT 64

typedef struct {
    atomic_int count[MAXT];
    atomic_int bad_nt;
    int expect_nt;
} run_ctx;

static void run_body(void *c, int tid, int nt) {
    run_ctx *r = (run_ctx *)c;
    if (nt != r->expect_nt || tid < 0 || tid >= MAXT) { atomic_fetch_add(&r->bad_nt, 1); return; }
    atomic_fetch_add_explicit(&r->count[tid], 1, memory_order_relaxed);
}

static void run_init(run_ctx *r, int nt) {
    for (int i = 0; i < MAXT; i++) atomic_init(&r->count[i], 0);
    atomic_init(&r->bad_nt, 0);
    r->expect_nt = nt;
}

static int run_check(run_ctx *r, int nt, int reps) {
    if (atomic_load(&r->bad_nt)) return 0;
    for (int i = 0; i < MAXT; i++)
        if (atomic_load(&r->count[i]) != (i < nt ? reps : 0)) return 0;
    return 1;
}

/* All `want` participants must be live at once: a spin barrier with a timeout. */
typedef struct { atomic_int arrived; atomic_int timeouts; int want; } barrier_ctx;

static uint64_t barrier_timeout_ns(void) { return g_no_timing ? 120000000000ull : 10000000000ull; }

static void barrier_wait(barrier_ctx *b) {
    atomic_fetch_add(&b->arrived, 1);
    uint64_t t0 = hx_now_ns(), limit = barrier_timeout_ns();
    while (atomic_load(&b->arrived) < b->want) {
        if (hx_now_ns() - t0 > limit) { atomic_fetch_add(&b->timeouts, 1); return; }
        hx_yield();
    }
}

static void barrier_body(void *c, int tid, int nt) { (void)tid; (void)nt; barrier_wait((barrier_ctx *)c); }
static void barrier_range(void *c, int64_t b, int64_t e, int tid) { (void)b; (void)e; (void)tid; barrier_wait((barrier_ctx *)c); }

static void barrier_init(barrier_ctx *b, int want) {
    atomic_init(&b->arrived, 0);
    atomic_init(&b->timeouts, 0);
    b->want = want;
}

static void test_run(int quick) {
    printf("hx_pool_run\n");
    const int threads[] = {1, 2, 4, 5, 16, 32, 33};
    const int spins[] = {0, 3, 200};
    for (size_t ti = 0; ti < sizeof threads / sizeof threads[0]; ti++) {
        for (size_t si = 0; si < sizeof spins / sizeof spins[0]; si++) {
            int nt = threads[ti];
            hx_pool *p = hx_pool_create(nt, spins[si]);
            CHECK(p != NULL, "create %d", nt);
            if (!p) continue;
            run_ctx r;
            run_init(&r, nt);
            int reps = quick ? 200 : 2000;
            for (int k = 0; k < reps; k++) hx_pool_run(p, run_body, &r);
            CHECK(run_check(&r, nt, reps), "every tid exactly once per run (threads=%d spin=%d)", nt, spins[si]);

            barrier_ctx b;
            barrier_init(&b, nt);
            hx_pool_run(p, barrier_body, &b);
            CHECK(atomic_load(&b.timeouts) == 0 && atomic_load(&b.arrived) == nt,
                  "all %d threads run concurrently (spin=%d)", nt, spins[si]);
            hx_pool_destroy(p);
        }
    }

    /* intermittent dispatch: workers fall asleep between runs, exercising the wake path */
    {
        hx_pool *p = hx_pool_create(8, 20);
        run_ctx r;
        run_init(&r, 8);
        uint64_t s = 99;
        int reps = quick ? 100 : 600;
        for (int k = 0; k < reps; k++) {
            hx_pool_run(p, run_body, &r);
            uint64_t z = splitmix(&s) % 4;
            if (z == 0) hx_sleep_us(50 + (uint32_t)(splitmix(&s) % 200));
            else if (z == 1) hx_yield();
        }
        CHECK(run_check(&r, 8, reps), "intermittent dispatch");
        hx_pool_destroy(p);
    }

    /* back-to-back dispatches */
    const int bb_threads[] = {4, 16, 32};
    for (size_t ti = 0; ti < sizeof bb_threads / sizeof bb_threads[0]; ti++) {
        int nt = bb_threads[ti];
        for (int spin = 0; spin < 2; spin++) {
            int reps = spin ? 200000 : 20000;
            /* more threads than cores: each run may wait out a preempted thread's time slice */
            if (nt > hx_num_physical_cores()) reps /= 10;
            if (quick) reps /= 10;
            hx_pool *p = hx_pool_create(nt, spin ? 500 : 0);
            run_ctx r;
            run_init(&r, nt);
            uint64_t t0 = hx_now_ns();
            for (int k = 0; k < reps; k++) hx_pool_run(p, run_body, &r);
            double us = (double)(hx_now_ns() - t0) / 1e3 / reps;
            CHECK(run_check(&r, nt, reps), "%d back-to-back dispatches, %d threads, spin=%d", reps, nt, spin ? 500 : 0);
            printf("  %6d back-to-back runs, %2d threads, spin_us=%3d: ok (%.2f us each)\n", reps, nt, spin ? 500 : 0, us);
            hx_pool_destroy(p);
        }
    }
}

/* for_n(k) must put exactly min(k, n) threads to work at once: k chunks that each wait until k
 * chunks are in progress only finish if k distinct threads take part. */
static void test_for_n_concurrency(void) {
    printf("hx_pool_for_n concurrency\n");
    const int threads[] = {4, 16, 33};
    int ok = 1;
    for (size_t ti = 0; ti < sizeof threads / sizeof threads[0]; ti++) {
        int nt = threads[ti];
        for (int spin = 0; spin <= 200; spin += 200) {
            hx_pool *p = hx_pool_create(nt, spin);
            if (!p) { ok = 0; continue; }
            const int ks[] = {2, 3, nt - 1, nt};
            for (size_t ki = 0; ki < sizeof ks / sizeof ks[0]; ki++) {
                barrier_ctx b;
                barrier_init(&b, ks[ki]);
                hx_pool_for_n(p, ks[ki], ks[ki], 1, barrier_range, &b);
                int good = atomic_load(&b.timeouts) == 0 && atomic_load(&b.arrived) == ks[ki];
                CHECK(good, "for_n(%d) on %d threads (spin %d): %d chunks in flight together", ks[ki], nt, spin,
                      atomic_load(&b.arrived));
                ok &= good;
            }
            hx_pool_destroy(p);
        }
    }
    printf("  %s\n", ok ? "ok" : "FAILED");
}

/* Random interleaving of full runs and for_n regions of every width, with idle gaps so
 * workers fall asleep, are skipped while asleep, and are woken by a later full run. */
static void test_mixed(int quick) {
    printf("mixed hx_pool_run / hx_pool_for_n sequences\n");
    const int threads[] = {3, 8, 17};
    const int spins[] = {0, 3, 200};
    uint64_t s = 2024;
    int ops = 0;
    for (size_t ti = 0; ti < sizeof threads / sizeof threads[0]; ti++)
        for (size_t si = 0; si < sizeof spins / sizeof spins[0]; si++) {
            int nt = threads[ti];
            hx_pool *p = hx_pool_create(nt, spins[si]);
            if (!p) { CHECK(0, "create %d", nt); continue; }
            int reps = quick ? 300 : 3000, ok = 1;
            for (int k = 0; k < reps; k++) {
                uint64_t r = splitmix(&s);
                if (r % 3 == 0) {
                    run_ctx rc;
                    run_init(&rc, nt);
                    hx_pool_run(p, run_body, &rc);
                    ok &= run_check(&rc, nt, 1);
                } else {
                    int maxt = (int)((r >> 8) % (uint64_t)(nt + 3)) - 1;     /* -1 .. nt+1 */
                    int64_t n = (int64_t)((r >> 16) % 200), chunk = 1 + (int64_t)((r >> 32) % 8);
                    ok &= run_for_case(p, maxt, n, chunk);
                }
                if ((r >> 40) % 16 == 0) hx_sleep_us(20 + (uint32_t)((r >> 44) % 300));
                ops++;
            }
            CHECK(ok, "mixed sequence, %d threads, spin %d", nt, spins[si]);
            hx_pool_destroy(p);
        }
    printf("  %d operations\n", ops);
}

static void test_create_destroy(int quick) {
    printf("create/destroy\n");
    uint64_t s = 4242;
    int loops = quick ? 50 : 300;
    int ok = 1;
    for (int i = 0; i < loops; i++) {
        int nt = 1 + (int)(splitmix(&s) % 33);
        int spin = (int)(splitmix(&s) % 3) * 100;
        hx_pool *p = hx_pool_create(nt, spin);
        if (!p) { ok = 0; continue; }
        int runs = (int)(splitmix(&s) % 4);
        run_ctx r;
        run_init(&r, nt);
        for (int k = 0; k < runs; k++) hx_pool_run(p, run_body, &r);
        ok &= run_check(&r, nt, runs);
        hx_pool_destroy(p);
    }
    CHECK(ok, "%d create/run/destroy cycles", loops);
    CHECK(hx_pool_create(0, 0) == NULL && hx_pool_create(-3, 10) == NULL, "n_threads < 1 rejected");
    CHECK(hx_pool_create(1 << 16, 0) == NULL && hx_pool_create(INT_MAX, 0) == NULL, "n_threads >= 65536 rejected");
    hx_pool_destroy(NULL);
    printf("  %d cycles\n", loops);
}

static void test_split(void) {
    int ok = 1;
    for (int64_t n = 0; n < 200; n++)
        for (int nt = 1; nt <= 33; nt++) {
            int64_t prev = 0;
            for (int t = 0; t < nt; t++) {
                int64_t b, e;
                hx_split(n, t, nt, &b, &e);
                ok &= b == prev && e >= b && e - b <= n / nt + 1 && e - b >= n / nt;
                prev = e;
            }
            ok &= prev == n;
        }
    CHECK(ok, "hx_split partitions");
}

/* -------------------------------------------------------------- latency */

static void empty_task(void *c, int tid, int nt) { (void)c; (void)tid; (void)nt; }
static void empty_range(void *c, int64_t b, int64_t e, int tid) { (void)c; (void)b; (void)e; (void)tid; }

static void latency(int nt, int spin_us, int reps) {
    if (nt > hx_num_physical_cores() && reps > 20000) reps = 20000;
    hx_pool *p = hx_pool_create(nt, spin_us);
    if (!p) { CHECK(0, "create %d", nt); return; }
    for (int i = 0; i < 2000; i++) hx_pool_run(p, empty_task, NULL);
    uint64_t t0 = hx_now_ns();
    for (int i = 0; i < reps; i++) hx_pool_run(p, empty_task, NULL);
    double mean = (double)(hx_now_ns() - t0) / reps / 1e3;
    int ns = reps < 20000 ? reps : 20000;
    uint64_t *lat = (uint64_t *)malloc((size_t)ns * sizeof *lat);
    for (int i = 0; i < ns; i++) {
        uint64_t a = hx_now_ns();
        hx_pool_run(p, empty_task, NULL);
        lat[i] = hx_now_ns() - a;
    }
    qsort(lat, (size_t)ns, sizeof *lat, cmp_u64);
    t0 = hx_now_ns();
    for (int i = 0; i < reps / 4; i++) hx_pool_for(p, 4 * (int64_t)nt, 1, empty_range, NULL);
    double for_mean = (double)(hx_now_ns() - t0) / (reps / 4) / 1e3;
    printf("  threads=%2d spin_us=%4d: run mean %6.2f us  p50 %6.2f  p99 %7.2f  max %8.1f | for(4*nt, chunk 1) %6.2f us\n",
           nt, spin_us, mean, (double)lat[ns / 2] / 1e3, (double)lat[ns * 99 / 100] / 1e3, (double)lat[ns - 1] / 1e3,
           for_mean);
    free(lat);
    hx_pool_destroy(p);
}

/* Quantile q (0..1) of the round trip of `samples` dispatches: hx_pool_run (k == 0) or
 * hx_pool_for_n(k, k items), each after idle_us of idling. */
static double quantile_us(hx_pool *p, int k, int samples, uint32_t idle_us, double q) {
    uint64_t *lat = (uint64_t *)malloc((size_t)samples * sizeof *lat);
    for (int i = 0; i < samples; i++) {
        if (idle_us) hx_sleep_us(idle_us);
        uint64_t a = hx_now_ns();
        if (k) hx_pool_for_n(p, k, k, 1, empty_range, NULL);
        else hx_pool_run(p, empty_task, NULL);
        lat[i] = hx_now_ns() - a;
    }
    qsort(lat, (size_t)samples, sizeof *lat, cmp_u64);
    double r = (double)lat[(int)(q * (samples - 1))] / 1e3;
    free(lat);
    return r;
}

static double p50_us(hx_pool *p, int k, int samples, uint32_t idle_us) { return quantile_us(p, k, samples, idle_us, 0.5); }

static void busy_us(uint32_t us) {
    uint64_t t0 = hx_now_ns();
    while (hx_now_ns() - t0 < (uint64_t)us * 1000u) {}
}

/* Dispatch until one is over in < 20 us: no stall was declared, so every worker has just
 * finished a task and is in its spin window. */
static void spin_up(hx_pool *p) {
    for (int i = 0; i < 1000; i++) {
        uint64_t a = hx_now_ns();
        hx_pool_run(p, empty_task, NULL);
        if (hx_now_ns() - a < 20000u) return;
    }
}

/* CPUs the process keeps busy while the caller sleeps idle_ms right after spin_up. */
static double idle_cpus(hx_pool *p, int idle_ms) {
    spin_up(p);
    double c0 = process_cpu_s();
    uint64_t t0 = hx_now_ns();
    hx_sleep_us((uint32_t)idle_ms * 1000u);
    return (process_cpu_s() - c0) / ((double)(hx_now_ns() - t0) * 1e-9);
}

/*
 * The spin-then-sleep contract. Spinning workers must answer >= 3x faster than sleeping ones,
 * checked only where waking a sleeping pool is measurably slow (> 10 us), on the spinning
 * pool's p25 (load from other processes only adds slow samples), with three attempts. Workers
 * must sleep once spin_us has elapsed: an idle pool (spin_us = 2 ms) keeps fewer than nt/4 CPUs
 * busy over 300 ms (process CPU time; spinning on would keep nt busy).
 */
static void test_spin_behaviour(void) {
    printf("spin/sleep behaviour\n");
    int nt = hx_num_physical_cores();
    if (nt > 16) nt = 16;
    if (nt < 4) { printf("  skipped (fewer than 4 cores)\n"); return; }
    hx_pool *sleepy = hx_pool_create(nt, 0);
    hx_pool *spinny = hx_pool_create(nt, 2000);
    for (int i = 0; i < 200; i++) { hx_pool_run(sleepy, empty_task, NULL); hx_pool_run(spinny, empty_task, NULL); }
    double sleep_p50 = 0, spin_p25 = 0;
    int fast = 0;
    for (int attempt = 0; attempt < 3 && !fast; attempt++) {
        sleep_p50 = p50_us(sleepy, 0, 301, 0);
        spin_p25 = quantile_us(spinny, 0, 2001, 0, 0.25);
        fast = spin_p25 * 3 < sleep_p50;
        if (g_no_timing) break;
    }
    double idle = idle_cpus(spinny, 300);
    printf("  %d threads: p50 sleeping pool %.2f us, p25 spinning pool %.2f us; idle for 300 ms (spin_us=2000): "
           "%.2f CPUs busy\n", nt, sleep_p50, spin_p25, idle);
    if (g_no_timing) {
        printf("  (--no-timing: timing checks skipped)\n");
    } else {
        if (sleep_p50 > 10.0) CHECK(fast, "spinning workers should answer much faster than sleeping ones");
        else printf("  (wake-up from sleep is fast here: latency check skipped)\n");
        CHECK(idle < nt / 4.0, "workers should sleep once spin_us has elapsed (%.2f CPUs busy)", idle);
    }
    hx_pool_destroy(sleepy);
    hx_pool_destroy(spinny);
}

static void workers_busy(void *c, int tid, int nt) {
    (void)c; (void)nt;
    if (tid) busy_us(200);
}

/* Process CPU seconds of `rounds` x (one dispatch of task, then the caller idles 1 ms). */
static double gap_cpu(hx_pool *p, hx_task_fn task, int rounds) {
    double c0 = process_cpu_s();
    for (int r = 0; r < rounds; r++) {
        hx_pool_run(p, task, NULL);
        hx_sleep_us(1000);
    }
    return process_cpu_s() - c0;
}

/*
 * Slow tasks are not stalls: after a dispatch whose workers ran for 200 us (the caller waiting
 * meanwhile) the workers spin through the caller's 1 ms pauses just as after empty dispatches.
 * Treating them as stalled would put them to sleep and leave the pauses without spinners.
 * Compared as process CPU time spent outside the tasks.
 */
static void test_slow_tasks(void) {
    printf("slow tasks\n");
    int nt = hx_num_physical_cores();
    if (nt > 16) nt = 16;
    if (nt < 4) { printf("  skipped (fewer than 4 cores)\n"); return; }
    hx_pool *p = hx_pool_create(nt, 5000);
    spin_up(p);
    double after_empty = gap_cpu(p, empty_task, 100);
    double after_slow = gap_cpu(p, workers_busy, 100) - (nt - 1) * 100 * 200e-6;
    hx_pool_destroy(p);
    printf("  %d threads, spin_us=5000: CPU s spent beside the tasks over 100 pauses: %.3f after empty tasks, "
           "%.3f after 200 us tasks\n", nt, after_empty, after_slow);
    if (g_no_timing) printf("  (--no-timing: check skipped)\n");
    else CHECK(after_slow > 0.5 * after_empty, "workers stopped spinning after slow tasks (%.3f vs %.3f CPU s)",
               after_slow, after_empty);
}

static void busy_range(void *c, int64_t b, int64_t e, int tid) {
    (void)c; (void)b; (void)e; (void)tid;
    busy_us(5);
}

/*
 * hx_pool.h: workers spin ~spin_us after finishing *a task*. Workers left out of a stream of
 * hx_pool_for_n(2) regions therefore sleep after spin_us, and the process keeps about two CPUs
 * busy, not the whole pool (they used to restart their spin window on every region they saw).
 * Measured as process CPU time over wall time.
 */
static void test_for_n_idle_cpu(void) {
    printf("workers left out of small regions\n");
    int nt = hx_num_physical_cores();
    if (nt > 16) nt = 16;
    if (nt < 8) { printf("  skipped (fewer than 8 cores)\n"); return; }
    hx_pool *p = hx_pool_create(nt, 2000);
    spin_up(p);
    double c0 = process_cpu_s();
    uint64_t t0 = hx_now_ns();
    long regions = 0;
    while (hx_now_ns() - t0 < 300000000u) {
        hx_pool_for_n(p, 2, 2, 1, busy_range, NULL);
        regions++;
    }
    double cores = (process_cpu_s() - c0) / ((double)(hx_now_ns() - t0) * 1e-9);
    hx_pool_destroy(p);
    printf("  %d threads, spin_us=2000, %ld for_n(2) regions in 300 ms: %.1f CPUs busy\n", nt, regions, cores);
    if (g_no_timing) printf("  (--no-timing: check skipped)\n");
    else CHECK(cores < nt / 2.0, "%.1f CPUs busy running 2-thread regions on a %d-thread pool", cores, nt);
}

/* --------------------------------- a pool as large as the machine, under competing load */

static atomic_int g_hog_stop;

static void *hog_main(void *arg) {
    (void)arg;
    volatile uint64_t x = 0;
    while (!atomic_load_explicit(&g_hog_stop, memory_order_relaxed)) x++;
    return NULL;
}

/* Starts up to n threads that never yield; returns how many. */
static int hogs_start(hx_thread **hogs, int n) {
    atomic_store(&g_hog_stop, 0);
    for (int i = 0; i < n; i++)
        if (hx_thread_create(&hogs[i], hog_main, NULL) != 0) return i;
    return n;
}

static void hogs_stop(hx_thread **hogs, int n) {
    atomic_store(&g_hog_stop, 1);
    for (int i = 0; i < n; i++) hx_thread_join(hogs[i]);
}

/* hx_pool_run back to back for ms milliseconds: mean round trip (us) and the share of the
 * time spent in dispatches slower than 2 ms. */
static void timed_runs_on(hx_pool *p, int ms, double *mean_us, double *slow_share) {
    uint64_t t0 = hx_now_ns(), end = t0 + (uint64_t)ms * 1000000u, slow = 0;
    long n = 0;
    while (hx_now_ns() < end) {
        uint64_t a = hx_now_ns();
        hx_pool_run(p, empty_task, NULL);
        uint64_t d = hx_now_ns() - a;
        if (d > 2000000u) slow += d;
        n++;
    }
    double total = (double)(hx_now_ns() - t0);
    *mean_us = total / (double)n / 1e3;
    *slow_share = (double)slow / total;
}

/* The same on a fresh pool. */
static void timed_runs(int nt, int spin_us, int ms, double *mean_us, double *slow_share) {
    hx_pool *p = hx_pool_create(nt, spin_us);
    if (!p) { CHECK(0, "create %d", nt); *mean_us = 0; *slow_share = 1; return; }
    hx_pool_run(p, empty_task, NULL);
    timed_runs_on(p, ms, mean_us, slow_share);
    hx_pool_destroy(p);
}

/*
 * A pool with one thread per logical CPU while nt/2 threads that never yield compete for the
 * CPUs: participants keep being stranded behind them. Before the stall rule a stranded thread
 * waited for a time slice or for the spinners' deadline (32 threads, Windows: 3.4-15x the
 * sleeping pool's mean, 70-83% of the time in dispatches > 2 ms; WSL2: 7.7x); now the pool
 * sleeps and an idle CPU takes it (Windows 0.8-1.2x; WSL2 1.6-2.1x, where the rescue works
 * less well). Judged against a sleeping pool (spin_us = 0) measured in alternation under the
 * same load, which never holds CPUs: spinning may not cost more than 2.5x its mean, nor spend
 * 20% more of its time in dispatches > 2 ms (without the stall rule the back-off below keeps
 * the mean down, but stranded threads still cost 26-68% of the time; with it 0-12%). Up to
 * three attempts (other processes' load comes and goes); an attempt whose control spent more
 * than 5% of its time in > 2 ms dispatches only shows that the machine itself is overloaded
 * and is not judged.
 */
static void test_stranded(int quick) {
    printf("pool as large as the machine under competing load\n");
    int nt = hx_num_cpus();
    if (nt > 64) nt = 64;
    if (nt < 4) { printf("  skipped (fewer than 4 CPUs)\n"); return; }
    int ms = quick || g_no_timing ? 100 : 200;
    hx_thread *hogs[32];
    int nhog = hogs_start(hogs, nt / 2);
    int passed = 0, judged = 0;
    for (int attempt = 0; attempt < 3 && !passed; attempt++) {
        double ctl_mean = 0, ctl_slow = 0, spin_mean = 0, spin_slow = 0;
        for (int r = 0; r < 3; r++) {
            double m, s;
            timed_runs(nt, 0, ms, &m, &s);
            ctl_mean += m / 3;
            ctl_slow += s / 3;
            timed_runs(nt, 5000, ms, &m, &s);
            spin_mean += m / 3;
            spin_slow += s / 3;
        }
        printf("  %d threads + %d busy threads: mean dispatch %.1f us spinning (spin_us=5000), %.1f us sleeping; "
               "time in dispatches > 2 ms: %.1f%% / %.1f%%\n",
               nt, nhog, spin_mean, ctl_mean, 100 * spin_slow, 100 * ctl_slow);
        if (g_no_timing) break;
        if (ctl_slow > 0.05) continue;
        judged++;
        passed = spin_mean < 2.5 * ctl_mean + 20 && spin_slow < ctl_slow + 0.20;
    }
    hogs_stop(hogs, nhog);
    if (g_no_timing) printf("  (--no-timing: check skipped)\n");
    else if (!judged) printf("  (machine overloaded: even the sleeping pool stalls; check skipped)\n");
    else CHECK(passed, "stranded pool threads: the spinning pool was more than 2.5x slower than a sleeping one, or "
                       "spent 20%% more of its time in dispatches > 2 ms");
}

/*
 * One busy thread per logical CPU: no CPU goes idle, a stall cannot be rescued, and the
 * stranded thread waits for a time slice. After such slow stalls the pool has to stop spinning
 * (back-off growing to 1 s) and cost about what a sleeping pool costs: measured after a 0.5 s
 * warm-up, in alternation with a sleeping pool. Without the back-off a 32-thread pool took
 * 10-31 ms per dispatch here, a sleeping pool 0.1-0.2 ms; with it 1-2x the sleeping pool.
 */
static void test_overload(void) {
    printf("pool as large as the machine, every CPU busy\n");
    int nt = hx_num_cpus();
    if (nt > 64) nt = 64;
    if (nt < 4) { printf("  skipped (fewer than 4 CPUs)\n"); return; }
    hx_thread *hogs[64];
    int nhog = hogs_start(hogs, nt);
    hx_pool *p = hx_pool_create(nt, 5000), *ctl = hx_pool_create(nt, 0);
    int passed = 0;
    if (p && ctl) {
        double m, s;
        timed_runs_on(p, g_no_timing ? 100 : 500, &m, &s);
        for (int attempt = 0; attempt < 2 && !passed; attempt++) {
            double spin_mean = 0, ctl_mean = 0;
            for (int r = 0; r < 3; r++) {
                timed_runs_on(p, 100, &m, &s);
                spin_mean += m / 3;
                timed_runs_on(ctl, 100, &m, &s);
                ctl_mean += m / 3;
            }
            printf("  %d threads + %d busy threads: mean dispatch %.1f us spinning (spin_us=5000), %.1f us sleeping\n",
                   nt, nhog, spin_mean, ctl_mean);
            passed = spin_mean < 10 * ctl_mean + 100;
            if (g_no_timing) break;
        }
    }
    hogs_stop(hogs, nhog);
    /* Once the load is gone and the back-off (at most HX__CALM_MAX_NS = 1 s) has run out, the
     * pool spins again: its fastest dispatch beats the fastest wake-up of a sleeping pool 3x
     * (minima: load only adds time). The machine may still be busy enough to start a new
     * back-off, so this gets three attempts. */
    double spin_min = 0, sleep_min = 0;
    int recovered = 0;
    if (p && ctl) {
        hx_sleep_us(1100000);
        for (int attempt = 0; attempt < 3 && !recovered; attempt++) {
            sleep_min = quantile_us(ctl, 0, 301, 0, 0.0);
            spin_min = quantile_us(p, 0, 2001, 0, 0.0);
            recovered = spin_min * 3 < sleep_min;
        }
        printf("  1.1 s after the load: fastest dispatch %.2f us (sleeping pool %.2f us)\n", spin_min, sleep_min);
    }
    hx_pool_destroy(p);
    hx_pool_destroy(ctl);
    CHECK(p && ctl, "create %d", nt);
    if (g_no_timing) {
        printf("  (--no-timing: checks skipped)\n");
    } else {
        CHECK(passed, "overloaded machine: the spinning pool was more than 10x slower than a sleeping one");
        if (sleep_min > 10.0) CHECK(recovered, "the pool should spin again once the machine is no longer overloaded");
        else printf("  (wake-up from sleep is fast here: recovery check skipped)\n");
    }
}

/* for_n(2) and a 2-chunk hx_pool_for on a sleeping pool wake one worker, not all of them. */
static void test_for_n_cost(void) {
    printf("small regions on a sleeping pool\n");
    int nt = hx_num_cpus() < 32 ? hx_num_cpus() : 32;
    if (nt < 8) { printf("  skipped (fewer than 8 CPUs)\n"); return; }
    hx_pool *p = hx_pool_create(nt, 0);
    for (int i = 0; i < 50; i++) hx_pool_run(p, empty_task, NULL);
    double all = p50_us(p, 0, 301, 0);
    double two = p50_us(p, 2, 301, 0);
    uint64_t *lat = (uint64_t *)malloc(301 * sizeof *lat);
    for (int i = 0; i < 301; i++) {
        uint64_t a = hx_now_ns();
        hx_pool_for(p, 2, 1, empty_range, NULL);
        lat[i] = hx_now_ns() - a;
    }
    qsort(lat, 301, sizeof *lat, cmp_u64);
    double for2 = (double)lat[150] / 1e3;
    free(lat);
    printf("  %d threads, spin_us=0, p50: hx_pool_run %.2f us, for_n(2) %.2f us, hx_pool_for(2 chunks) %.2f us\n", nt,
           all, two, for2);
    if (g_no_timing) {
        printf("  (--no-timing: timing checks skipped)\n");
    } else if (all > 20.0) {
        CHECK(two * 2 < all, "for_n(2) should not wake or wait for the whole pool");
        CHECK(for2 * 2 < all, "hx_pool_for with 2 chunks should not wake the whole pool");
    } else {
        printf("  (wake-up from sleep is fast here; timing checks skipped)\n");
    }
    hx_pool_destroy(p);
}

/* -------------------------------------------- pinned to one CPU (child process) */

#define PINNED_THREADS 4
#define PINNED_SPIN_US 200000

/* Child: report the p50 dispatch time of a pool with more threads than CPUs and a long spin
 * window. Without yields each dispatch waits for a time slice or the spin deadline. */
static int pinned_child(const char *out) {
    hx_pool *p = hx_pool_create(PINNED_THREADS, PINNED_SPIN_US);
    if (!p) return 3;
    for (int i = 0; i < 3; i++) hx_pool_run(p, empty_task, NULL);
    double us = p50_us(p, 0, 41, 0);
    hx_pool_destroy(p);
    FILE *fp = fopen(out, "w");
    if (!fp) return 4;
    fprintf(fp, "%.3f\n", us);
    fclose(fp);
    return 0;
}

static char g_self[1100], g_tmp[1100];

static int locate_self(const char *argv0) {
    snprintf(g_self, sizeof g_self, "%s", argv0 ? argv0 : "");
    if (!hx_path_exists(g_self)) snprintf(g_self, sizeof g_self, "%s.exe", argv0 ? argv0 : "");
    return hx_path_exists(g_self);
}

/* Runs this program pinned to the CPUs in mask; returns the child's p50 in us, or -1. */
static double run_pinned(unsigned long long mask) {
    char out[1300], cmd[3000];
    snprintf(out, sizeof out, "%s/hx_test_pool_%llx.txt", g_tmp, (unsigned long long)hx_now_ns());
#if defined(HX_OS_WINDOWS)
    snprintf(cmd, sizeof cmd, "start \"\" /b /wait /affinity %llx \"%s\" --pinned-child \"%s\"", mask, g_self, out);
#else
    snprintf(cmd, sizeof cmd, "taskset 0x%llx '%s' --pinned-child '%s'", mask, g_self, out);
#endif
    fflush(stdout);
    int rc = system(cmd);
    double us = -1.0;
    FILE *fp = fopen(out, "r");
    if (fp) {
        if (fscanf(fp, "%lf", &us) != 1) us = -1.0;
        fclose(fp);
        remove(out);
    }
    if (rc != 0 && us >= 0) printf("  (child exit status %d)\n", rc);
    return us;
}

static void test_pinned(void) {
    printf("pool on one CPU (more threads than CPUs)\n");
#if !defined(HX_OS_WINDOWS)
    if (system("command -v taskset >/dev/null 2>&1") != 0) { printf("  skipped: no taskset\n"); return; }
#endif
    if (!g_self[0]) { printf("  skipped: cannot locate this executable from argv[0]\n"); return; }
    int cpu = hx_num_cpus() - 1;                  /* keep off CPU 0, which takes most interrupts */
    if (cpu > 63) cpu = 63;
    unsigned long long mask = 1ull << cpu;
    double us = run_pinned(mask);
    if (us > 250.0 && !g_no_timing) us = run_pinned(mask);   /* one retry: the CPU may have been busy */
    printf("  %d threads, spin_us=%d, pinned to CPU %d: p50 dispatch %.2f us\n", PINNED_THREADS, PINNED_SPIN_US, cpu, us);
    CHECK(us >= 0, "pinned child ran");
    if (g_no_timing) printf("  (--no-timing: timing check skipped)\n");
    else CHECK(us >= 0 && us < 250.0, "spinning threads must yield the CPU to siblings sharing it (p50 %.1f us)", us);
}

/* ------------------------------------------------------ one run per machine at a time */

/*
 * Concurrent runs of this program (ctest -j, mutation testing with several jobs) would time
 * each other's busy threads and spinning pools, so a run takes a lock file in the temp
 * directory (C11 fopen "wx"). It holds the owner's process id; a lock whose owner is gone, or
 * older than 10 minutes, is a leftover of a killed run and is taken over.
 */
static char g_lock[1300];

static unsigned long my_pid(void) {
#if defined(_WIN32)
    return (unsigned long)GetCurrentProcessId();
#else
    return (unsigned long)getpid();
#endif
}

static int pid_alive(unsigned long pid) {
#if defined(_WIN32)
    HANDLE h = OpenProcess(SYNCHRONIZE, FALSE, (DWORD)pid);
    if (!h) return GetLastError() == ERROR_ACCESS_DENIED;
    int alive = WaitForSingleObject(h, 0) == WAIT_TIMEOUT;
    CloseHandle(h);
    return alive;
#else
    return kill((pid_t)pid, 0) == 0 || errno == EPERM;
#endif
}

static void timing_lock(void) {
    snprintf(g_lock, sizeof g_lock, "%s/hx_test_pool_timing.lock", g_tmp);
    uint64_t t0 = hx_now_ns();
    int said = 0;
    for (;;) {
        FILE *fp = fopen(g_lock, "wx");
        if (fp) {
            fprintf(fp, "%lu %llu\n", my_pid(), (unsigned long long)hx_now_ns());
            fclose(fp);
            return;
        }
        unsigned long pid = 0;
        unsigned long long when = 0;
        fp = fopen(g_lock, "r");
        if (fp) {
            if (fscanf(fp, "%lu %llu", &pid, &when) != 2) pid = 0;
            fclose(fp);
        }
        uint64_t now = hx_now_ns();
        if (pid && (!pid_alive(pid) || (now > when && now - when > 600000000000ull))) {
            remove(g_lock);                       /* a killed run's leftover */
            continue;
        }
        if (now - t0 > 600000000000ull) {
            printf("  (lock %s still taken after 10 min: going ahead)\n", g_lock);
            g_lock[0] = 0;
            return;
        }
        if (!said) printf("  (waiting for another run of test_pool)\n");
        said = 1;
        hx_sleep_us(20000);
    }
}

static void timing_unlock(void) {
    if (g_lock[0]) remove(g_lock);
    g_lock[0] = 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "--pinned-child") == 0) return pinned_child(argv[2]);
    int quick = 0;
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    g_no_timing = 1;
#endif
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) quick = 1;
        else if (strcmp(argv[i], "--no-timing") == 0) g_no_timing = 1;
        else { printf("usage: test_pool [--quick] [--no-timing]\n"); return 2; }
    }
    if (!locate_self(argv[0])) g_self[0] = 0;
    {
        const char *cands[] = {"HEARTH_TEST_DIR", "TMPDIR", "TEMP", "TMP"};
        const char *dir = NULL;
        for (size_t i = 0; i < sizeof cands / sizeof cands[0] && !dir; i++) {
            const char *v = hx_env_str(cands[i]);
            if (v && *v && hx_path_exists(v)) dir = v;
        }
        snprintf(g_tmp, sizeof g_tmp, "%s", dir ? dir : hx_path_exists("/tmp") ? "/tmp" : ".");
    }
    if (!g_no_timing) timing_lock();
    uint64_t t0 = hx_now_ns();
    printf("cpus: %d logical, %d physical%s\n", hx_num_cpus(), hx_num_physical_cores(),
           g_no_timing ? " (timing checks off)" : "");
    test_split();
    test_for();
    test_for_n();
    printf("  [%.1f s]\n", (double)(hx_now_ns() - t0) * 1e-9);
    test_run(quick);
    test_for_n_concurrency();
    test_mixed(quick);
    printf("  [%.1f s]\n", (double)(hx_now_ns() - t0) * 1e-9);
    test_create_destroy(quick);
    printf("  [%.1f s]\n", (double)(hx_now_ns() - t0) * 1e-9);
    test_spin_behaviour();
    test_slow_tasks();
    test_for_n_idle_cpu();
    test_for_n_cost();
    test_pinned();
    test_stranded(quick);
    test_overload();
    if (!quick) {
        printf("dispatch latency (empty task, round trip incl. caller's share)\n");
        const int lt[] = {1, 4, 16, 32};
        for (size_t i = 0; i < sizeof lt / sizeof lt[0]; i++) latency(lt[i], 1000, 200000);
        latency(hx_num_cpus(), 10000, 50000);    /* the tail must not follow spin_us */
        latency(16, 0, 20000);
    }
    timing_unlock();
    printf("test_pool: %d checks, %d failed (%.1f s)\n", g_checks, g_fail, (double)(hx_now_ns() - t0) * 1e-9);
    return g_fail ? 1 : 0;
}
