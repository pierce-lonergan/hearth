/*
 * test_pool.c — self-checking tests and dispatch-latency measurement for hx_pool.h.
 *
 *   test_pool [--quick]
 *
 * --quick: correctness only, fewer repetitions, no latency section (for
 * mutation testing). Latency numbers are printed, not asserted: they depend
 * on machine load.
 * Exit code 0 = all checks passed.
 */
#include "hx_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_fail;

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

/* ------------------------------------------------------------ hx_pool_for */

typedef struct {
    atomic_int *hits;          /* per item; NULL when n is too large to track */
    int64_t n, chunk;
    int nthreads;
    atomic_int bad_range, bad_tid, calls;
    atomic_llong covered;
} for_ctx;

static void for_body(void *c, int64_t b, int64_t e, int tid) {
    for_ctx *f = (for_ctx *)c;
    atomic_fetch_add(&f->calls, 1);
    int64_t want_e = f->n - b > f->chunk ? b + f->chunk : f->n;
    if (b < 0 || b >= e || e > f->n || b % f->chunk != 0 || e != want_e) atomic_fetch_add(&f->bad_range, 1);
    if (tid < 0 || tid >= f->nthreads) atomic_fetch_add(&f->bad_tid, 1);
    atomic_fetch_add(&f->covered, e - b);
    if (f->hits)
        for (int64_t i = b; i < e && i < f->n; i++) atomic_fetch_add_explicit(&f->hits[i], 1, memory_order_relaxed);
}

static int run_for_case(hx_pool *p, int64_t n, int64_t chunk) {
    for_ctx f;
    int track = n <= (1 << 20);
    f.hits = track && n > 0 ? (atomic_int *)calloc((size_t)n, sizeof(atomic_int)) : NULL;
    f.n = n;
    f.chunk = chunk;
    f.nthreads = hx_pool_size(p);
    atomic_init(&f.bad_range, 0);
    atomic_init(&f.bad_tid, 0);
    atomic_init(&f.calls, 0);
    atomic_init(&f.covered, 0);
    hx_pool_for(p, n, chunk, for_body, &f);
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
                CHECK(run_for_case(p, n, chunks[ci]), "for threads=%d n=%lld chunk=%lld", threads[ti],
                      (long long)n, (long long)chunks[ci]);
                cases++;
            }
        }
        /* ranges near INT64_MAX must not overflow (only the ranges are checked) */
        CHECK(run_for_case(p, INT64_MAX, INT64_MAX / 2 + 1), "for n=INT64_MAX two chunks");
        CHECK(run_for_case(p, INT64_MAX, INT64_MAX / 7), "for n=INT64_MAX seven chunks");
        CHECK(run_for_case(p, -5, 3), "for negative n");
        cases += 3;
        hx_pool_destroy(p);
    }
    printf("  %d (threads, n, chunk) cases\n", cases);
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

/* All threads must be live at once: a spin barrier with a timeout. */
typedef struct { atomic_int arrived; atomic_int timeouts; int nt; } barrier_ctx;

static void barrier_body(void *c, int tid, int nt) {
    barrier_ctx *b = (barrier_ctx *)c;
    (void)tid;
    atomic_fetch_add(&b->arrived, 1);
    uint64_t t0 = hx_now_ns();
    while (atomic_load(&b->arrived) < nt) {
        if (hx_now_ns() - t0 > 10000000000ull) { atomic_fetch_add(&b->timeouts, 1); return; }
        hx_yield();
    }
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
            atomic_init(&b.arrived, 0);
            atomic_init(&b.timeouts, 0);
            b.nt = nt;
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

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

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

static double p50_us(hx_pool *p, int samples, uint32_t idle_us) {
    uint64_t *lat = (uint64_t *)malloc((size_t)samples * sizeof *lat);
    for (int i = 0; i < samples; i++) {
        if (idle_us) hx_sleep_us(idle_us);
        uint64_t a = hx_now_ns();
        hx_pool_run(p, empty_task, NULL);
        lat[i] = hx_now_ns() - a;
    }
    qsort(lat, (size_t)samples, sizeof *lat, cmp_u64);
    double r = (double)lat[samples / 2] / 1e3;
    free(lat);
    return r;
}

/*
 * The spin-then-sleep contract, checked by timing but self-calibrated: only
 * where waking a sleeping pool is measurably slow (> 10 us) do we require
 * that spinning workers answer >= 3x faster, and that an idle pool goes back
 * to sleep once spin_us has elapsed.
 */
static void test_spin_behaviour(void) {
    printf("spin/sleep behaviour\n");
    int nt = hx_num_physical_cores();
    if (nt > 16) nt = 16;
    if (nt < 2) { printf("  skipped (single core)\n"); return; }
    hx_pool *sleepy = hx_pool_create(nt, 0);
    hx_pool *spinny = hx_pool_create(nt, 2000);
    hx_pool *brief = hx_pool_create(nt, 100);
    for (int i = 0; i < 200; i++) { hx_pool_run(sleepy, empty_task, NULL); hx_pool_run(spinny, empty_task, NULL); }
    double sleep_p50 = p50_us(sleepy, 301, 0);
    double spin_p50 = p50_us(spinny, 2001, 0);
    double idle_p50 = p50_us(brief, 61, 2000);
    printf("  %d threads, p50 dispatch: sleeping pool %.2f us, spinning pool %.2f us, after 2 ms idle (spin_us=100) %.2f us\n",
           nt, sleep_p50, spin_p50, idle_p50);
    if (sleep_p50 > 10.0) {
        CHECK(spin_p50 * 3 < sleep_p50, "spinning workers should answer much faster than sleeping ones");
        CHECK(idle_p50 > spin_p50 * 3, "workers should sleep once spin_us has elapsed");
    } else {
        printf("  (wake-up from sleep is fast here; timing checks skipped)\n");
    }
    hx_pool_destroy(sleepy);
    hx_pool_destroy(spinny);
    hx_pool_destroy(brief);
}

int main(int argc, char **argv) {
    int quick = argc > 1 && strcmp(argv[1], "--quick") == 0;
    uint64_t t0 = hx_now_ns();
    printf("cpus: %d logical, %d physical\n", hx_num_cpus(), hx_num_physical_cores());
    test_split();
    test_for();
    printf("  [%.1f s]\n", (double)(hx_now_ns() - t0) * 1e-9);
    test_run(quick);
    printf("  [%.1f s]\n", (double)(hx_now_ns() - t0) * 1e-9);
    test_create_destroy(quick);
    printf("  [%.1f s]\n", (double)(hx_now_ns() - t0) * 1e-9);
    test_spin_behaviour();
    if (!quick) {
        printf("dispatch latency (empty task, round trip incl. caller's share)\n");
        const int lt[] = {1, 4, 16, 32};
        for (size_t i = 0; i < sizeof lt / sizeof lt[0]; i++) latency(lt[i], 1000, 200000);
        latency(16, 0, 20000);
    }
    printf("test_pool: %d checks, %d failed (%.1f s)\n", g_checks, g_fail, (double)(hx_now_ns() - t0) * 1e-9);
    return g_fail ? 1 : 0;
}
