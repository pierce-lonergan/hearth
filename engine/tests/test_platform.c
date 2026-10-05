/*
 * test_platform.c — self-checking tests for hx_platform.h.
 *
 *   test_platform [--dir DIR] [--replace-dir DIR] [--exhaustive] [--large] [--expect-cpu LIST]
 *
 * --dir         where scratch files go (default: $HEARTH_TEST_DIR, $HEARTH_DATA,
 *               $TMPDIR, $TEMP, $TMP, /tmp, .); files are small and removed. Its volume
 *               must rename with POSIX semantics (NTFS, any POSIX file system).
 * --replace-dir also run the hx_file_replace checks in DIR, with whatever rename semantics its
 *               volume has (\\wsl.localhost\<distro>\tmp exercises the MoveFileExW fallback).
 * --exhaustive  also check f32->f16 and f32->bf16 for all 2^32 inputs (threaded).
 * --large       also do >1 GiB single requests at offsets above 2^31 (writes and
 *               removes a 2.25 GiB file in DIR; keep DIR off cloud-synced folders).
 * --expect-cpu  comma list of hx_cpu features this machine must report, '-name'
 *               for ones it must not, plus optional cores=N / cpus=N
 *               (e.g. avx2,avx512f,-neon,cores=16,cpus=32 on a Ryzen 9 9950X).
 * The program re-runs itself (--log-child FILE, --replace-log-child FILE DIR) to check what it
 * logs, and pinned to a few CPUs (--cpu-child FILE, via start /affinity or taskset) to check
 * the CPU counts.
 * Exit code 0 = all checks passed.
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L   /* setenv/unsetenv, mkdir under -std=c11 */
#endif
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>              /* SetEnvironmentVariableA: the CRT cannot create an empty variable */
#  include <direct.h>
#  include <io.h>
#  include <sys/stat.h>
#else
#  include <sys/stat.h>
#  include <unistd.h>
#endif
#include "hx_platform.h"

#include <float.h>
#include <locale.h>
#include <math.h>
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

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* ------------------------------------------------------------------------
 * Independent references: values from ldexp on the field definitions, and
 * rounding by nearest-neighbour search in that value table (ties -> even
 * pattern). Positive patterns of both formats are monotonic in value.
 */
#define F16_LAST 0x7c00   /* table entry standing for 2^16 (first value that is inf) */
#define BF16_LAST 0x7f80  /* table entry standing for 2^128 */
static double tv16[F16_LAST + 1], tvb16[BF16_LAST + 1];

static double ref_f16_value(uint16_t h) {
    int e = (h >> 10) & 0x1f, m = h & 0x3ff;
    double v = e == 0 ? ldexp((double)m, -24) : ldexp((double)(1024 + m), e - 25);
    return (h & 0x8000) ? -v : v;
}

static double ref_bf16_value(uint16_t h) {
    int e = (h >> 7) & 0xff, m = h & 0x7f;
    double v = e == 0 ? ldexp((double)m, -133) : ldexp((double)(128 + m), e - 134);
    return (h & 0x8000) ? -v : v;
}

static void build_tables(void) {
    for (int p = 0; p < F16_LAST; p++) tv16[p] = ref_f16_value((uint16_t)p);
    tv16[F16_LAST] = 65536.0;
    for (int p = 0; p < BF16_LAST; p++) tvb16[p] = ref_bf16_value((uint16_t)p);
    tvb16[BF16_LAST] = ldexp(1.0, 128);
}

static uint16_t ref_round(const double *tv, int last, float f) {
    uint16_t sign = signbit(f) ? 0x8000 : 0;
    double a = fabs((double)f);
    if (a >= tv[last]) return (uint16_t)(sign | last);
    int lo = 0, hi = last;                         /* tv[lo] <= a < tv[hi] */
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (tv[mid] <= a) lo = mid; else hi = mid;
    }
    double dl = a - tv[lo], dh = tv[lo + 1] - a;  /* exact in double */
    int r = dl < dh ? lo : dh < dl ? lo + 1 : ((lo & 1) ? lo + 1 : lo);
    return (uint16_t)(sign | r);
}

/* Expected NaN encodings: quiet bit set, sign and top payload bits kept (F16C / AVX512-BF16 behaviour). */
static uint16_t nan_f16(uint32_t x) { return (uint16_t)(((x >> 16) & 0x8000u) | 0x7e00u | ((x >> 13) & 0x3ffu)); }
static uint16_t nan_bf16(uint32_t x) { return (uint16_t)((x >> 16) | 0x40u); }

static int check_f16_one(float f) {
    uint32_t x = fbits(f);
    uint16_t got = hx_f32_to_f16(f);
    uint16_t want = isnan(f) ? nan_f16(x) : ref_round(tv16, F16_LAST, f);
    if (got != want) {
        CHECK(0, "f32->f16 of %08x (%.9g): got %04x want %04x", (unsigned)x, (double)f, got, want);
        return 0;
    }
    return 1;
}

static int check_bf16_one(float f) {
    uint32_t x = fbits(f);
    uint16_t got = hx_f32_to_bf16(f);
    uint16_t want = isnan(f) ? nan_bf16(x) : ref_round(tvb16, BF16_LAST, f);
    if (got != want) {
        CHECK(0, "f32->bf16 of %08x (%.9g): got %04x want %04x", (unsigned)x, (double)f, got, want);
        return 0;
    }
    return 1;
}

static void test_f16(void) {
    printf("f16 conversions\n");
    int bad = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        float f = hx_f16_to_f32((uint16_t)h);
        int e = (h >> 10) & 0x1f, m = h & 0x3ff;
        if (e == 31 && m) {
            /* as VCVTPH2PS (the F16C kernels): quiet bit set, sign and payload kept */
            bad += fbits(f) != (((uint32_t)(h & 0x8000) << 16) | 0x7fc00000u | ((uint32_t)m << 13));
            uint16_t back = hx_f32_to_f16(f);
            bad += !(((back >> 10) & 0x1f) == 31 && (back & 0x3ff) && (back & 0x200) && (back & 0x8000) == (h & 0x8000));
            bad += back != (uint16_t)(h | 0x200);              /* payload kept, quieted */
        } else if (e == 31) {
            bad += !(isinf(f) && (signbit(f) != 0) == ((h & 0x8000) != 0));
            bad += hx_f32_to_f16(f) != h;
        } else {
            double want = ref_f16_value((uint16_t)h);
            bad += (double)f != want || (signbit(f) != 0) != ((h & 0x8000) != 0);
            bad += hx_f32_to_f16(f) != h;
        }
    }
    CHECK(bad == 0, "f16 exhaustive decode / round trip: %d mismatches", bad);

    /* every rounding boundary: representable values, midpoints, midpoints +- 1 ulp */
    int okb = 1;
    for (int p = 0; p < F16_LAST; p++) {
        float v = (float)tv16[p];
        float mid = (float)((tv16[p] + tv16[p + 1]) * 0.5);
        float cases[4] = {v, mid, nextafterf(mid, 0.0f), nextafterf(mid, INFINITY)};
        for (int i = 0; i < 4; i++) {
            okb &= check_f16_one(cases[i]);
            okb &= check_f16_one(-cases[i]);
        }
    }
    CHECK(okb, "f16 boundary cases");

    static const uint32_t special[] = {
        0x00000000u, 0x80000000u, 0x00000001u, 0x007fffffu, 0x00800000u,      /* zeros, f32 subnormals */
        0x33000000u, 0x33000001u, 0x32ffffffu, 0x33800000u, 0x337fffffu,      /* 2^-25 tie, 2^-24 */
        0x38800000u, 0x387fffffu, 0x387fe000u, 0x387ff000u,                   /* smallest normal f16 */
        0x477fe000u, 0x477fefffu, 0x477ff000u, 0x477ff001u, 0x47800000u,      /* 65504 .. 65536 */
        0x7f7fffffu, 0x7f800000u, 0xff800000u,                                /* FLT_MAX, inf */
        0x7fc00000u, 0xffc00000u, 0x7f800001u, 0x7fbfffffu, 0x7f802000u, 0xffffffffu  /* NaNs */
    };
    int oks = 1;
    for (size_t i = 0; i < sizeof special / sizeof special[0]; i++) {
        oks &= check_f16_one(bitsf(special[i]));
        oks &= check_f16_one(bitsf(special[i] ^ 0x80000000u));
    }
    CHECK(oks, "f16 special cases");
    CHECK(hx_f32_to_f16(65520.0f) == 0x7c00 && hx_f32_to_f16(65519.996f) == 0x7bff, "f16 overflow threshold");
    CHECK(hx_f32_to_f16(1.0f + 1.0f / 2048) == 0x3c00 && hx_f32_to_f16(1.0f + 3.0f / 2048) == 0x3c02, "f16 ties to even");
    CHECK((hx_f32_to_f16(bitsf(0x7f800001u)) & 0x7fff) > 0x7c00, "f16 sNaN with low payload stays NaN");

    uint64_t s = 12345;
    int okr = 1;
    const int N = 4000000;
    for (int i = 0; i < N; i++) {
        uint64_t r = splitmix(&s);
        uint32_t x = (uint32_t)r;
        if (i & 1) x = (x & 0x807fffffu) | ((uint32_t)(98 + (r >> 32) % 48) << 23);   /* f16-relevant exponents */
        okr &= check_f16_one(bitsf(x));
    }
    CHECK(okr, "f16 random");
    printf("  %d random + %d boundary inputs checked\n", N, F16_LAST * 8);
}

static void test_bf16(void) {
    printf("bf16 conversions\n");
    int bad = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        float f = hx_bf16_to_f32((uint16_t)h);
        int e = (h >> 7) & 0xff, m = h & 0x7f;
        if (e == 255 && m) {
            bad += !isnan(f);
            bad += hx_f32_to_bf16(f) != (uint16_t)(h | 0x40);
        } else if (e == 255) {
            bad += !isinf(f) || hx_f32_to_bf16(f) != h;
        } else {
            bad += (double)f != ref_bf16_value((uint16_t)h) || (signbit(f) != 0) != ((h & 0x8000) != 0);
            bad += hx_f32_to_bf16(f) != h;
        }
    }
    CHECK(bad == 0, "bf16 exhaustive decode / round trip: %d mismatches", bad);

    int okb = 1;
    for (int p = 0; p < BF16_LAST; p++) {
        float v = (float)tvb16[p];
        float mid = (float)((tvb16[p] + tvb16[p + 1]) * 0.5);
        float cases[4] = {v, mid, nextafterf(mid, 0.0f), nextafterf(mid, INFINITY)};
        for (int i = 0; i < 4; i++) {
            okb &= check_bf16_one(cases[i]);
            okb &= check_bf16_one(-cases[i]);
        }
    }
    CHECK(okb, "bf16 boundary cases");
    static const uint32_t special[] = {0x00000000u, 0x00000001u, 0x00008000u, 0x00018000u, 0x007fffffu,
                                       0x7f7fffffu, 0x7f7f7fffu, 0x7f7f8000u, 0x7f800000u, 0x7f800001u,
                                       0x7fc00000u, 0x7f80ffffu, 0x3f808000u, 0x3f818000u, 0xffffffffu};
    int oks = 1;
    for (size_t i = 0; i < sizeof special / sizeof special[0]; i++) {
        oks &= check_bf16_one(bitsf(special[i]));
        oks &= check_bf16_one(bitsf(special[i] ^ 0x80000000u));
    }
    CHECK(oks, "bf16 special cases");
    CHECK(hx_f32_to_bf16(bitsf(0x7f800001u)) != 0x7f80, "bf16 NaN with low payload stays NaN");

    uint64_t s = 777;
    int okr = 1;
    const int N = 4000000;
    for (int i = 0; i < N; i++) okr &= check_bf16_one(bitsf((uint32_t)splitmix(&s)));
    CHECK(okr, "bf16 random");
    printf("  %d random + %d boundary inputs checked\n", N, BF16_LAST * 8);
}

/* ---- optional: every f32 input, split over threads ---- */
typedef struct { uint64_t lo, hi; long long bad16, badb; } exh_job;

static void *exh_main(void *arg) {
    exh_job *j = (exh_job *)arg;
    for (uint64_t x = j->lo; x < j->hi; x++) {
        float f = bitsf((uint32_t)x);
        int nan = isnan(f);
        uint16_t w16 = nan ? nan_f16((uint32_t)x) : ref_round(tv16, F16_LAST, f);
        uint16_t wb = nan ? nan_bf16((uint32_t)x) : ref_round(tvb16, BF16_LAST, f);
        j->bad16 += hx_f32_to_f16(f) != w16;
        j->badb += hx_f32_to_bf16(f) != wb;
    }
    return NULL;
}

static void test_exhaustive(void) {
    enum { MAXT = 64 };
    int nt = hx_num_cpus();
    if (nt > MAXT) nt = MAXT;
    exh_job jobs[MAXT];
    hx_thread *th[MAXT];
    uint64_t t0 = hx_now_ns();
    for (int i = 0; i < nt; i++) {
        jobs[i].lo = (1ull << 32) * (uint64_t)i / (uint64_t)nt;
        jobs[i].hi = (1ull << 32) * (uint64_t)(i + 1) / (uint64_t)nt;
        jobs[i].bad16 = jobs[i].badb = 0;
        CHECK(hx_thread_create(&th[i], exh_main, &jobs[i]) == 0, "thread create");
    }
    long long b16 = 0, bb = 0;
    for (int i = 0; i < nt; i++) {
        hx_thread_join(th[i]);
        b16 += jobs[i].bad16;
        bb += jobs[i].badb;
    }
    CHECK(b16 == 0, "exhaustive f32->f16: %lld mismatches", b16);
    CHECK(bb == 0, "exhaustive f32->bf16: %lld mismatches", bb);
    printf("exhaustive: all 2^32 f32 inputs, f16 mismatches %lld, bf16 mismatches %lld (%.1f s, %d threads)\n",
           b16, bb, (double)(hx_now_ns() - t0) * 1e-9, nt);
}

/* ------------------------------------------------------------------ memory */

static void test_memory(void) {
    printf("memory\n");
    size_t aligns[] = {1, 8, 64, 4096, 1u << 21};
    for (size_t i = 0; i < sizeof aligns / sizeof aligns[0]; i++) {
        uint8_t *p = (uint8_t *)hx_aligned_alloc(aligns[i], 1000);
        CHECK(p && ((uintptr_t)p % aligns[i]) == 0, "hx_aligned_alloc align %zu", aligns[i]);
        if (p) { memset(p, 0xab, 1000); hx_aligned_free(p); }
    }
    CHECK(hx_aligned_alloc(48, 64) == NULL, "non power-of-two alignment rejected");
    hx_aligned_free(NULL);

    for (int huge = 0; huge < 2; huge++) {
        size_t sz = (64u << 20) + 12345;
        uint8_t *p = (uint8_t *)hx_alloc_large(sz, huge);
        CHECK(p != NULL && ((uintptr_t)p % HX_PAGE) == 0, "hx_alloc_large(%zu, %d)", sz, huge);
        if (!p) continue;
        size_t nz = 0;
        for (size_t i = 0; i < sz; i++) nz += p[i] != 0;
        CHECK(nz == 0, "hx_alloc_large zero-filled (%zu nonzero)", nz);
        memset(p, 0x5a, sz);
        hx_free_large(p, sz);
    }
    CHECK(hx_alloc_large(0, 0) == NULL, "hx_alloc_large(0) is NULL");
    hx_free_large(NULL, 0);

    uint64_t tot = hx_ram_total(), av = hx_ram_available();
    printf("  RAM total %.2f GiB, available %.2f GiB\n", (double)tot / (1u << 30), (double)av / (1u << 30));
    CHECK(tot >= (256ull << 20), "RAM total sane");
    CHECK(av > 0 && av <= tot, "RAM available sane");
}

/* -------------------------------------------------------------------- time */

static void test_time(void) {
    printf("time\n");
    uint64_t prev = hx_now_ns(), first = prev;
    int back = 0;
    for (int i = 0; i < 1000000; i++) {
        uint64_t t = hx_now_ns();
        back += t < prev;
        prev = t;
    }
    CHECK(back == 0, "hx_now_ns went backwards %d times", back);
    printf("  hx_now_ns: %.1f ns per call\n", (double)(prev - first) / 1e6);
    uint32_t us_list[] = {0, 50, 200, 1000, 5000};
    for (size_t i = 0; i < sizeof us_list / sizeof us_list[0]; i++) {
        uint32_t us = us_list[i];
        uint64_t best = UINT64_MAX;
        int early = 0;
        for (int k = 0; k < 5; k++) {
            uint64_t t0 = hx_now_ns();
            hx_sleep_us(us);
            uint64_t d = hx_now_ns() - t0;
            if (d < best) best = d;
            early += d < (uint64_t)us * 1000u;
        }
        printf("  hx_sleep_us(%u): min %.1f us\n", us, (double)best / 1e3);
        CHECK(early == 0, "hx_sleep_us(%u) returned early %d of 5 times (min %.1f us)", us, early, (double)best / 1e3);
        CHECK(best < (uint64_t)us * 1000 + 100000000ull, "hx_sleep_us(%u) far too long", us);
    }
}

/* ----------------------------------------------------------------- threads */

typedef struct {
    hx_mutex mu;
    hx_cond cv;
    long long counter;
    int turn;            /* ping-pong */
    int go, arrived;     /* broadcast */
} sync_state;

static void *mutex_worker(void *arg) {
    sync_state *s = (sync_state *)arg;
    for (int i = 0; i < 100000; i++) {
        hx_mutex_lock(&s->mu);
        s->counter++;
        hx_mutex_unlock(&s->mu);
    }
    return arg;
}

static void *pong(void *arg) {
    sync_state *s = (sync_state *)arg;
    for (int i = 0; i < 10000; i++) {
        hx_mutex_lock(&s->mu);
        while (s->turn != 1) hx_cond_wait(&s->cv, &s->mu);
        s->turn = 0;
        hx_cond_signal(&s->cv);
        hx_mutex_unlock(&s->mu);
    }
    return NULL;
}

static void *bcast_waiter(void *arg) {
    sync_state *s = (sync_state *)arg;
    hx_mutex_lock(&s->mu);
    while (!s->go) hx_cond_wait(&s->cv, &s->mu);
    s->arrived++;
    hx_mutex_unlock(&s->mu);
    return NULL;
}

static void *late_signal(void *arg) {
    sync_state *s = (sync_state *)arg;
    hx_sleep_us(5000);
    hx_mutex_lock(&s->mu);
    s->go = 1;
    hx_cond_signal(&s->cv);
    hx_mutex_unlock(&s->mu);
    return NULL;
}

static void *cpu_probe(void *arg) {
    *(const hx_cpu **)arg = hx_cpu_features();
    return NULL;
}

static void test_threads(void) {
    printf("threads\n");
    printf("  logical cpus %d, physical cores %d\n", hx_num_cpus(), hx_num_physical_cores());
    CHECK(hx_num_cpus() >= 1, "hx_num_cpus");
    CHECK(hx_num_physical_cores() >= 1 && hx_num_physical_cores() <= hx_num_cpus(), "hx_num_physical_cores");

    sync_state s;
    memset(&s, 0, sizeof s);
    hx_mutex_init(&s.mu);
    hx_cond_init(&s.cv);

    hx_thread *th[8];
    for (int i = 0; i < 8; i++) CHECK(hx_thread_create(&th[i], mutex_worker, &s) == 0, "thread create");
    for (int i = 0; i < 8; i++) hx_thread_join(th[i]);
    CHECK(s.counter == 800000, "mutex counter %lld", s.counter);

    hx_thread *pt;
    uint64_t t0 = hx_now_ns();
    CHECK(hx_thread_create(&pt, pong, &s) == 0, "thread create");
    for (int i = 0; i < 10000; i++) {
        hx_mutex_lock(&s.mu);
        s.turn = 1;
        hx_cond_signal(&s.cv);
        while (s.turn != 0) hx_cond_wait(&s.cv, &s.mu);
        hx_mutex_unlock(&s.mu);
    }
    hx_thread_join(pt);
    printf("  condvar ping-pong: %.2f us per round trip\n", (double)(hx_now_ns() - t0) / 1e4 / 1e3);

    for (int i = 0; i < 8; i++) CHECK(hx_thread_create(&th[i], bcast_waiter, &s) == 0, "thread create");
    hx_sleep_us(2000);
    hx_mutex_lock(&s.mu);
    s.go = 1;
    hx_cond_broadcast(&s.cv);
    hx_mutex_unlock(&s.mu);
    for (int i = 0; i < 8; i++) hx_thread_join(th[i]);
    CHECK(s.arrived == 8, "broadcast woke %d of 8", s.arrived);

    /* timed wait that must time out */
    hx_mutex_lock(&s.mu);
    t0 = hx_now_ns();
    int r = hx_cond_timedwait(&s.cv, &s.mu, 20000);
    uint64_t dt = hx_now_ns() - t0;
    hx_mutex_unlock(&s.mu);
    printf("  timedwait(20 ms) -> %d after %.2f ms\n", r, (double)dt / 1e6);
    CHECK(r == 1, "timedwait should time out");
    CHECK(dt >= 20000000ull && dt < 2000000000ull, "timedwait duration %.2f ms", (double)dt / 1e6);
    hx_mutex_lock(&s.mu);
    CHECK(hx_cond_timedwait(&s.cv, &s.mu, 0) == 1, "timedwait(0) times out");
    hx_mutex_unlock(&s.mu);
    /* a timeout is never reported before the interval has passed (Windows' millisecond timer
     * used to expire up to ~75 us early, about one wait in four) */
    {
        int early = 0, timeouts = 0;
        uint64_t worst = UINT64_MAX;
        for (int k = 0; k < 24; k++) {
            uint32_t us = 1000u + 500u * (uint32_t)(k % 3);
            hx_mutex_lock(&s.mu);
            uint64_t a = hx_now_ns();
            int rr = hx_cond_timedwait(&s.cv, &s.mu, us);
            uint64_t d = hx_now_ns() - a;
            hx_mutex_unlock(&s.mu);
            if (rr == 1) {
                timeouts++;
                early += d < (uint64_t)us * 1000u;
                if (d < worst) worst = d;
            }
        }
        CHECK(timeouts > 0 && early == 0, "timedwait timed out early %d of %d times (shortest %.1f us)", early, timeouts,
              (double)worst / 1e3);
    }

    /* timed wait that is signalled */
    s.go = 0;
    CHECK(hx_thread_create(&pt, late_signal, &s) == 0, "thread create");
    hx_mutex_lock(&s.mu);
    t0 = hx_now_ns();
    int timeouts = 0;
    while (!s.go && hx_now_ns() - t0 < 5000000000ull) timeouts += hx_cond_timedwait(&s.cv, &s.mu, 5000000);
    dt = hx_now_ns() - t0;
    hx_mutex_unlock(&s.mu);
    hx_thread_join(pt);
    CHECK(s.go && timeouts == 0 && dt < 2000000000ull, "signalled timedwait (go=%d, timeouts=%d, %.1f ms)",
          s.go, timeouts, (double)dt / 1e6);
    hx_cond_destroy(&s.cv);
    hx_mutex_destroy(&s.mu);
    hx_yield();

    /* cpu features: one cached object, same from every thread */
    const hx_cpu *c = hx_cpu_features();
    const hx_cpu *seen[8];
    for (int i = 0; i < 8; i++) CHECK(hx_thread_create(&th[i], cpu_probe, &seen[i]) == 0, "thread create");
    for (int i = 0; i < 8; i++) { hx_thread_join(th[i]); CHECK(seen[i] == c, "hx_cpu_features not cached"); }
    printf("  cpu: \"%s\"\n", c->brand);
    printf("  sse42=%d avx=%d avx2=%d fma=%d f16c=%d avx512f=%d avx512bw=%d avx512vl=%d avx512vnni=%d "
           "avx512bf16=%d avxvnni=%d neon=%d dotprod=%d\n",
           c->sse42, c->avx, c->avx2, c->fma, c->f16c, c->avx512f, c->avx512bw, c->avx512vl, c->avx512vnni,
           c->avx512bf16, c->avxvnni, c->neon, c->dotprod);
    CHECK(!c->avx2 || c->avx, "avx2 implies avx");
    CHECK(!c->fma || c->avx, "fma implies avx");
    CHECK(!(c->avx512bw || c->avx512vl || c->avx512vnni || c->avx512bf16) || c->avx512f, "avx512 subsets imply avx512f");
    CHECK(!c->avx512f || c->avx2, "avx512f implies avx2 (OS state checks consistent)");
    size_t bl = strlen(c->brand);
#if defined(HX_ARCH_X86_64)
    int printable = 1;
    for (size_t i = 0; i < bl; i++) printable &= c->brand[i] >= 0x20 && c->brand[i] < 0x7f;
    CHECK(bl >= 4 && printable, "brand string present and printable");
#endif
    CHECK(bl < sizeof c->brand && (bl == 0 || (c->brand[0] != ' ' && c->brand[bl - 1] != ' ')), "brand trimmed");
}

/* --expect-cpu a,b,-c: features this machine is known to have (or, with '-', lack). */
static void check_expected_cpu(const char *list) {
    const hx_cpu *c = hx_cpu_features();
    struct { const char *name; int v; } f[] = {
        {"sse42", c->sse42}, {"avx", c->avx}, {"avx2", c->avx2}, {"fma", c->fma}, {"f16c", c->f16c},
        {"avx512f", c->avx512f}, {"avx512bw", c->avx512bw}, {"avx512vl", c->avx512vl},
        {"avx512vnni", c->avx512vnni}, {"avx512bf16", c->avx512bf16}, {"avxvnni", c->avxvnni},
        {"neon", c->neon}, {"dotprod", c->dotprod}};
    char buf[512];
    snprintf(buf, sizeof buf, "%s", list);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        if (strncmp(tok, "cores=", 6) == 0) {
            CHECK(hx_num_physical_cores() == atoi(tok + 6), "physical cores %d, expected %s", hx_num_physical_cores(), tok + 6);
            continue;
        }
        if (strncmp(tok, "cpus=", 5) == 0) {
            CHECK(hx_num_cpus() == atoi(tok + 5), "logical cpus %d, expected %s", hx_num_cpus(), tok + 5);
            continue;
        }
        int want = 1;
        if (*tok == '-') { want = 0; tok++; }
        int found = 0;
        for (size_t i = 0; i < sizeof f / sizeof f[0]; i++)
            if (strcmp(f[i].name, tok) == 0) {
                found = 1;
                CHECK(f[i].v == want, "cpu feature %s: detected %d, expected %d", tok, f[i].v, want);
            }
        CHECK(found, "unknown cpu feature name '%s'", tok);
    }
}

/* ------------------------------------------------------------------- files */

static uint8_t pat(uint64_t o) { return (uint8_t)((o * 2654435761u) >> 13) ^ (uint8_t)(o >> 12); }

static char g_dir[1024];
static uint64_t g_tag;

static void scratch_path(char *out, size_t len, const char *name) {
    snprintf(out, len, "%s/hx_test_platform_%llx_%s", g_dir, (unsigned long long)g_tag, name);
}

static void remove_utf8(const char *path) {
    /* On Windows the narrow CRT reads paths in the ANSI code page unless LC_CTYPE is UTF-8. */
    char *old = setlocale(LC_CTYPE, NULL);
    char saved[128];
    snprintf(saved, sizeof saved, "%s", old ? old : "C");
    setlocale(LC_CTYPE, ".UTF-8");
    remove(path);
    setlocale(LC_CTYPE, saved);
}

static int verify(const uint8_t *buf, uint64_t off, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (buf[i] != pat(off + i)) return 0;
    return 1;
}

typedef struct { hx_file *f; uint64_t size; uint64_t seed; int bad, reads; } rd_job;

static void *conc_reader(void *arg) {
    rd_job *j = (rd_job *)arg;
    size_t cap = 256u << 10;
    uint8_t *buf = (uint8_t *)hx_alloc_large(cap, 0);
    if (!buf) { j->bad++; return NULL; }
    uint64_t pages = (j->size + HX_PAGE - 1) / HX_PAGE;
    for (int i = 0; i < 1500; i++) {
        uint64_t r = splitmix(&j->seed);
        uint64_t off = (r % pages) * HX_PAGE;
        size_t n = (size_t)(1 + (r >> 40) % (cap / HX_PAGE)) * HX_PAGE;
        int64_t got = hx_file_pread(j->f, buf, n, off);
        uint64_t want = off + n <= j->size ? n : j->size - off;
        if (got != (int64_t)want || !verify(buf, off, (size_t)want)) j->bad++;
        j->reads++;
    }
    hx_free_large(buf, cap);
    return NULL;
}

static void test_files(void) {
    printf("files (dir %s)\n", g_dir);
    char path[1200], path2[1200], upath[1200], err[256];
    scratch_path(path, sizeof path, "a.bin");
    scratch_path(path2, sizeof path2, "b.bin");
    scratch_path(upath, sizeof upath, "\xc3\xa9t\xc3\xa9_\xe6\x97\xa5\xe6\x9c\xac.bin");   /* "été_日本" */

    const uint64_t size = (8u << 20) + 1234;      /* not a page multiple: tail tests */
    uint8_t *src = (uint8_t *)malloc((size_t)size);
    for (uint64_t i = 0; i < size; i++) src[i] = pat(i);

    /* error paths */
    err[0] = 0;
    char missing[1200];
    scratch_path(missing, sizeof missing, "does_not_exist.bin");
    CHECK(hx_file_open(missing, HX_FILE_READ, err, sizeof err) == NULL && err[0], "open missing file fails with message");
    printf("  missing-file error: %s\n", err);
    {
        const char *code = strstr(err, " (error ");
        CHECK(!strchr(err, '\r') && !strchr(err, '\n'), "error message has no line breaks");
#if defined(HX_OS_WINDOWS)
        CHECK(code && code > err && code[-1] != '.' && code[-1] != ' ', "system message trimmed before the error code");
#else
        (void)code;
#endif
    }
    {
        uint8_t tmp[16];
        CHECK(hx_file_pread(NULL, tmp, sizeof tmp, 0) == -1 && hx_file_pwrite(NULL, tmp, sizeof tmp, 0) == -1,
              "NULL file rejected");
        CHECK(hx_file_size(NULL) == -1 && hx_file_is_direct(NULL) == 0, "NULL file size / is_direct");
        hx_file_close(NULL);
        CHECK(!hx_path_exists(NULL) && !hx_path_exists(""), "hx_path_exists(NULL/empty)");
        CHECK(hx_file_open("", HX_FILE_READ, err, sizeof err) == NULL, "empty path rejected");
    }
    CHECK(hx_file_open(path, HX_FILE_CREATE | HX_FILE_READ, err, sizeof err) == NULL, "CREATE without WRITE rejected");
    CHECK(hx_file_open(path, 0, NULL, 0) == NULL, "no access mode rejected");
    CHECK(!hx_path_exists(missing), "hx_path_exists(missing)");
    CHECK(hx_path_exists(g_dir), "hx_path_exists(dir)");

    /* buffered write in irregular pieces, out of order */
    hx_file *w = hx_file_open(path, HX_FILE_WRITE | HX_FILE_CREATE, err, sizeof err);
    CHECK(w != NULL, "create: %s", err);
    if (!w) { free(src); return; }
    uint64_t cuts[] = {0, 777, 100000, 3u << 20, 5000001, size};
    for (int i = 4; i >= 0; i--) {
        size_t n = (size_t)(cuts[i + 1] - cuts[i]);
        CHECK(hx_file_pwrite(w, src + cuts[i], n, cuts[i]) == (int64_t)n, "pwrite piece %d", i);
    }
    CHECK(hx_file_size(w) == (int64_t)size, "size after write %lld", (long long)hx_file_size(w));
    CHECK(!hx_file_is_direct(w), "buffered handle not direct");
    {
        uint8_t tmp[16];
        CHECK(hx_file_pread(w, tmp, sizeof tmp, 0) == -1, "pread on write-only handle fails");
    }
    hx_file_close(w);
    CHECK(hx_path_exists(path), "hx_path_exists(created)");

    /* buffered reads: whole file, unaligned pieces, short read at EOF, past EOF */
    hx_file *b = hx_file_open(path, HX_FILE_READ, err, sizeof err);
    CHECK(b != NULL, "open buffered: %s", err);
    if (b) {
        uint8_t *buf = (uint8_t *)malloc((size_t)size + 8192);
        CHECK(hx_file_pread(b, buf, (size_t)size, 0) == (int64_t)size && memcmp(buf, src, (size_t)size) == 0,
              "buffered full read");
        CHECK(hx_file_pread(b, buf, 1001, 12345) == 1001 && verify(buf, 12345, 1001), "buffered unaligned read");
        CHECK(hx_file_pread(b, buf, 8192, size - 100) == 100 && verify(buf, size - 100, 100), "buffered short read at EOF");
        CHECK(hx_file_pread(b, buf, 10, size) == 0, "buffered read at EOF");
        CHECK(hx_file_pread(b, buf, 10, size + 99999) == 0, "buffered read past EOF");
        CHECK(hx_file_pread(b, buf, 0, 0) == 0, "zero-length read");
        CHECK(hx_file_pread(b, NULL, 10, 0) == -1, "NULL buffer rejected");
        CHECK(hx_file_pread(b, buf, 10, UINT64_MAX - 5) == -1, "offset + size overflow rejected");
        CHECK(hx_file_pread(b, buf, 10, (uint64_t)INT64_MAX) == -1, "offset beyond INT64_MAX rejected");
        CHECK(hx_file_pread(b, buf, 10, (uint64_t)INT64_MAX - 9) == -1, "read ending 1 byte past INT64_MAX rejected");
        CHECK(hx_file_pread(b, buf, 10, (uint64_t)INT64_MAX - 10) == 0, "read ending at INT64_MAX: past EOF, 0 bytes");
        free(buf);
        /* A request larger than the 1 GiB internal chunk whose first chunk still fits below
         * INT64_MAX: only the up-front range check can reject it (the OS would return EOF). The
         * buffer is committed but never touched. */
        size_t bign = ((size_t)1 << 30) + HX_PAGE;
        uint8_t *bigbuf = sizeof(size_t) > 4 ? (uint8_t *)hx_alloc_large(bign, 0) : NULL;
        if (bigbuf) {
            CHECK(hx_file_pread(b, bigbuf, bign, (uint64_t)INT64_MAX - ((uint64_t)1 << 30)) == -1,
                  "multi-chunk read crossing INT64_MAX rejected up front");
            hx_free_large(bigbuf, bign);
        }
        hx_file_close(b);
    }

    /* direct reads at many aligned offsets */
    hx_file *d = hx_file_open(path, HX_FILE_READ | HX_FILE_DIRECT, err, sizeof err);
    CHECK(d != NULL, "open direct: %s", err);
    if (d) {
        printf("  HX_FILE_DIRECT read handle: is_direct=%d\n", hx_file_is_direct(d));
#if defined(HX_OS_WINDOWS)
        CHECK(hx_file_is_direct(d), "unbuffered open expected to succeed on a local NTFS volume");
#endif
        size_t cap = 1u << 20;
        uint8_t *buf = (uint8_t *)hx_alloc_large(cap, 0);
        int bad = 0, n_reads = 0;
        uint64_t strides[] = {1, 3, 7, 61, 255};
        for (size_t si = 0; si < sizeof strides / sizeof strides[0]; si++) {
            for (uint64_t off = 0; off < size + HX_PAGE; off += strides[si] * HX_PAGE) {
                size_t n = (size_t)((1 + (off / HX_PAGE) % 64) * HX_PAGE);
                int64_t got = hx_file_pread(d, buf, n, off);
                uint64_t want = off >= size ? 0 : (off + n <= size ? n : size - off);
                if (got != (int64_t)want || !verify(buf, off, (size_t)want)) bad++;
                n_reads++;
            }
        }
        CHECK(bad == 0, "direct aligned reads: %d of %d wrong", bad, n_reads);
        uint64_t tail = size / HX_PAGE * HX_PAGE;
        CHECK(hx_file_pread(d, buf, cap, tail) == (int64_t)(size - tail), "direct short read at EOF");
        CHECK(hx_file_pread(d, buf, cap, 0) == (int64_t)cap && verify(buf, 0, cap), "direct 1 MiB read");
        hx_free_large(buf, cap);

        rd_job jobs[8];
        hx_thread *th[8];
        for (int i = 0; i < 8; i++) {
            jobs[i].f = d; jobs[i].size = size; jobs[i].seed = 1000 + (uint64_t)i; jobs[i].bad = 0; jobs[i].reads = 0;
            CHECK(hx_thread_create(&th[i], conc_reader, &jobs[i]) == 0, "thread create");
        }
        int cbad = 0, creads = 0;
        for (int i = 0; i < 8; i++) { hx_thread_join(th[i]); cbad += jobs[i].bad; creads += jobs[i].reads; }
        CHECK(cbad == 0, "concurrent direct preads: %d of %d wrong", cbad, creads);
        printf("  %d aligned direct reads + %d concurrent (8 threads) verified\n", n_reads, creads);
        hx_file_close(d);
    }

    /* direct write, read back buffered */
    hx_file *dw = hx_file_open(path2, HX_FILE_READ | HX_FILE_WRITE | HX_FILE_CREATE | HX_FILE_DIRECT, err, sizeof err);
    CHECK(dw != NULL, "create direct: %s", err);
    if (dw) {
        printf("  HX_FILE_DIRECT write handle: is_direct=%d\n", hx_file_is_direct(dw));
        size_t blk = 512u << 10, total = 4u << 20;
        uint8_t *buf = (uint8_t *)hx_alloc_large(blk, 0);
        for (int i = (int)(total / blk) - 1; i >= 0; i--) {
            memcpy(buf, src + (size_t)i * blk, blk);
            CHECK(hx_file_pwrite(dw, buf, blk, (uint64_t)i * blk) == (int64_t)blk, "direct pwrite block %d", i);
        }
        CHECK(hx_file_size(dw) == (int64_t)total, "direct-written size");
        memset(buf, 0, blk);
        CHECK(hx_file_pread(dw, buf, blk, blk) == (int64_t)blk && verify(buf, blk, blk), "direct read-back");
        hx_free_large(buf, blk);
        hx_file_close(dw);
        hx_file *rb = hx_file_open(path2, HX_FILE_READ, err, sizeof err);
        if (rb) {
            uint8_t *all = (uint8_t *)malloc(total);
            CHECK(hx_file_pread(rb, all, total, 0) == (int64_t)total && memcmp(all, src, total) == 0,
                  "direct-written content");
            free(all);
            hx_file_close(rb);
        }
    }

    /* CREATE truncates */
    hx_file *t = hx_file_open(path2, HX_FILE_WRITE | HX_FILE_CREATE, err, sizeof err);
    CHECK(t && hx_file_size(t) == 0, "CREATE truncates");
    if (t) hx_file_close(t);

    /* UTF-8 file name */
    hx_file *u = hx_file_open(upath, HX_FILE_READ | HX_FILE_WRITE | HX_FILE_CREATE, err, sizeof err);
    CHECK(u != NULL, "UTF-8 path create: %s", err);
    if (u) {
        uint8_t tmp[100];
        CHECK(hx_file_pwrite(u, src, 100, 0) == 100, "UTF-8 path write");
        CHECK(hx_file_pread(u, tmp, 100, 0) == 100 && memcmp(tmp, src, 100) == 0, "UTF-8 path read");
        hx_file_close(u);
        CHECK(hx_path_exists(upath), "UTF-8 path exists");
    }
    CHECK(hx_file_open("\xff\xfe-not-utf8", HX_FILE_READ, err, sizeof err) == NULL, "invalid UTF-8 rejected");

    /* path longer than the classic 260-character limit (component stays under 255) */
    char lname[240], lpath[1500];
    memset(lname, 'L', 200);
    snprintf(lname + 200, sizeof lname - 200, ".bin");
    scratch_path(lpath, sizeof lpath, lname);
    hx_file *lf = hx_file_open(lpath, HX_FILE_READ | HX_FILE_WRITE | HX_FILE_CREATE, err, sizeof err);
    CHECK(lf != NULL, "long path (%zu chars) create: %s", strlen(lpath), err);
    if (lf) {
        CHECK(hx_file_pwrite(lf, src, 4096, 0) == 4096, "long path write");
        hx_file_close(lf);
        CHECK(hx_path_exists(lpath), "long path exists");
#if defined(HX_OS_WINDOWS)
        /* the same file through an explicit \\?\ path (no normalisation: backslashes only) */
        char xpath[1600];
        snprintf(xpath, sizeof xpath, "\\\\?\\%s", lpath);
        for (char *q = xpath + 4; *q; q++) if (*q == '/') *q = '\\';
        hx_file *xf = hx_file_open(xpath, HX_FILE_READ, err, sizeof err);
        CHECK(xf && hx_file_size(xf) == 4096, "explicit \\\\?\\ long path open: %s", err);
        if (xf) hx_file_close(xf);
        remove(xpath);                             /* the CRT's remove() is MAX_PATH-limited */
#else
        remove(lpath);
#endif
        CHECK(!hx_path_exists(lpath), "long path removed");
    }

    remove(path);
    remove(path2);
    remove_utf8(upath);
    CHECK(!hx_path_exists(path) && !hx_path_exists(path2), "scratch files removed");
    CHECK(!hx_path_exists(upath), "UTF-8 scratch file removed");
    free(src);
}

/* ---------------------------------------------------------- hx_file_replace */

static int put_text(const char *path, const char *text) {
    hx_file *f = hx_file_open(path, HX_FILE_WRITE | HX_FILE_CREATE, NULL, 0);
    if (!f) return 0;
    size_t n = strlen(text);
    int ok = hx_file_pwrite(f, text, n, 0) == (int64_t)n;
    hx_file_close(f);
    return ok;
}

/* Whole file as a string ("" if missing or unreadable). */
static const char *get_text(const char *path, char *buf, size_t cap) {
    buf[0] = 0;
    hx_file *f = hx_file_open(path, HX_FILE_READ, NULL, 0);
    if (!f) return buf;
    int64_t n = hx_file_pread(f, buf, cap - 1, 0);
    buf[n > 0 ? (size_t)n : 0] = 0;
    hx_file_close(f);
    return buf;
}

static int make_dir(const char *path) {
#if defined(HX_OS_WINDOWS)
    return _mkdir(path) == 0;
#else
    return mkdir(path, 0700) == 0;
#endif
}

static void remove_dir(const char *path) {
#if defined(HX_OS_WINDOWS)
    _rmdir(path);
#else
    rmdir(path);
#endif
}

#if defined(HX_OS_WINDOWS)
typedef struct { FILE *fp; uint32_t after_us; } closer_job;
static void *close_later(void *arg) {
    closer_job *j = (closer_job *)arg;
    hx_sleep_us(j->after_us);
    fclose(j->fp);
    return NULL;
}
#endif

/* Readers racing a stream of replaces see a complete old or new version, never a torn one. */
#define REP_LEN 65536
typedef struct { const char *path; atomic_int stop; int torn, opened, missing; } rep_reader;

static void *replace_reader(void *arg) {
    rep_reader *r = (rep_reader *)arg;
    uint8_t *buf = (uint8_t *)malloc(REP_LEN + 1);
    while (buf && !atomic_load(&r->stop)) {
        hx_file *f = hx_file_open(r->path, HX_FILE_READ, NULL, 0);
        if (!f) { r->missing++; continue; }
        int64_t n = hx_file_pread(f, buf, REP_LEN + 1, 0);
        hx_file_close(f);
        r->opened++;
        int ok = n == REP_LEN;
        for (int64_t i = 1; ok && i < n; i++) ok = buf[i] == buf[0];
        r->torn += !ok;
    }
    free(buf);
    return NULL;
}

static int make_read_only(const char *path, int ro) {
#if defined(HX_OS_WINDOWS)
    return _chmod(path, ro ? _S_IREAD : _S_IREAD | _S_IWRITE) == 0;
#else
    return chmod(path, ro ? 0444 : 0644) == 0;
#endif
}

/* 1 made, 0 not permitted here (Windows without the symlink privilege or developer mode). */
static int make_symlink(const char *target, const char *link) {
#if defined(HX_OS_WINDOWS)
    wchar_t wt[1200], wl[1200];
    if (!MultiByteToWideChar(CP_UTF8, 0, target, -1, wt, 1200) || !MultiByteToWideChar(CP_UTF8, 0, link, -1, wl, 1200))
        return 0;
    for (wchar_t *q = wt; *q; q++) if (*q == L'/') *q = L'\\';   /* link targets are not normalized */
    return CreateSymbolicLinkW(wl, wt, 0x2 /* SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE */) != 0;
#else
    return symlink(target, link) == 0;
#endif
}

static int is_symlink(const char *path) {
#if defined(HX_OS_WINDOWS)
    wchar_t w[1200];
    if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, w, 1200)) return 0;
    DWORD a = GetFileAttributesW(w);
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_REPARSE_POINT);
#else
    struct stat st;
    return lstat(path, &st) == 0 && S_ISLNK(st.st_mode);
#endif
}

/*
 * require_posix: the volume must rename with POSIX semantics (NTFS on Windows 10+, any POSIX
 * file system), as the default scratch directory does. Otherwise (--replace-dir on FAT, SMB
 * or 9P) the semantics are probed with a replace over an open dst, and the checks follow them:
 * there hx_file_replace uses MoveFileExW, which open handles and a read-only dst refuse.
 */
static void test_replace(int require_posix) {
    printf("hx_file_replace (dir %s)\n", g_dir);
    char tmp[1200], dst[1200], dir[1200], ut[1200], ud[1200], lt[1500], ld[1500], txt[64];
    scratch_path(tmp, sizeof tmp, "rep.tmp");
    scratch_path(dst, sizeof dst, "rep.dst");
    scratch_path(dir, sizeof dir, "rep_dir");
    scratch_path(ut, sizeof ut, "rep_\xc3\xa9t\xc3\xa9.tmp");                  /* "Ã©tÃ©" */
    scratch_path(ud, sizeof ud, "rep_\xe6\x97\xa5\xe6\x9c\xac.dst");          /* "æ—¥æœ¬" */

    CHECK(hx_file_replace(NULL, dst) == -1 && hx_file_replace(tmp, NULL) == -1 && hx_file_replace("", dst) == -1 &&
          hx_file_replace(tmp, "") == -1, "NULL / empty paths rejected");
    CHECK(put_text(dst, "old"), "write dst");
    CHECK(hx_file_replace(tmp, dst) == -1 && strcmp(get_text(dst, txt, sizeof txt), "old") == 0,
          "missing tmp: fails, dst untouched");

    CHECK(put_text(tmp, "new"), "write tmp");
    CHECK(hx_file_replace(tmp, dst) == 0, "replace existing dst");
    CHECK(strcmp(get_text(dst, txt, sizeof txt), "new") == 0 && !hx_path_exists(tmp), "dst has tmp's content, tmp gone");
    remove(dst);
    CHECK(put_text(tmp, "fresh") && hx_file_replace(tmp, dst) == 0 && strcmp(get_text(dst, txt, sizeof txt), "fresh") == 0,
          "replace when dst does not exist");

    /* dst open through hx_file (shares delete access): replaced, and the open handle keeps the old
     * data -- or, without POSIX renames, refused with both files untouched */
    hx_file *h = hx_file_open(dst, HX_FILE_READ, NULL, 0);
    CHECK(h != NULL, "open dst");
    CHECK(put_text(tmp, "newer"), "write tmp");
    int posix = hx_file_replace(tmp, dst) == 0;
    if (require_posix) CHECK(posix, "replace while dst is open");
    else printf("  %s\n", posix ? "POSIX renames" : "no POSIX renames here: open handles block a replace");
    if (h) {
        char old[16] = {0};
        CHECK(hx_file_pread(h, old, sizeof old - 1, 0) == 5 && strcmp(old, "fresh") == 0, "open handle still reads old data");
        hx_file_close(h);
    }
    if (!posix) {
        CHECK(strcmp(get_text(dst, txt, sizeof txt), "fresh") == 0 && hx_path_exists(tmp), "refused replace: both untouched");
        CHECK(hx_file_replace(tmp, dst) == 0, "replace once dst is closed");
    }
    CHECK(strcmp(get_text(dst, txt, sizeof txt), "newer") == 0, "dst replaced");

    /* tmp still open by its writer */
    hx_file *w = hx_file_open(tmp, HX_FILE_WRITE | HX_FILE_CREATE, NULL, 0);
    CHECK(w && hx_file_pwrite(w, "open", 4, 0) == 4, "write tmp");
    CHECK(hx_file_replace(tmp, dst) == 0 && strcmp(get_text(dst, txt, sizeof txt), "open") == 0, "replace from an open tmp");
    if (w) hx_file_close(w);

    /* directories are never replaced, nor moved over a file */
    CHECK(make_dir(dir), "mkdir %s", dir);
    CHECK(put_text(tmp, "x"), "write tmp");
    uint64_t t0 = hx_now_ns();
    CHECK(hx_file_replace(tmp, dir) == -1, "replace onto a directory fails");
    CHECK(hx_now_ns() - t0 < 200000000u, "directory refused without retrying (%.0f ms)", (double)(hx_now_ns() - t0) / 1e6);
    CHECK(hx_file_replace(dir, dst) == -1 && hx_path_exists(dir), "a directory is not moved over a file");
    CHECK(hx_path_exists(tmp) && strcmp(get_text(dst, txt, sizeof txt), "open") == 0, "both files untouched");
    {
        char dir2[1300];                          /* rename(2) itself would move a directory over an empty one */
        snprintf(dir2, sizeof dir2, "%s2", dir);
        CHECK(make_dir(dir2), "mkdir %s", dir2);
        CHECK(hx_file_replace(dir, dir2) == -1 && hx_path_exists(dir) && hx_path_exists(dir2),
              "a directory is not moved over an empty directory");
        remove_dir(dir2);
        CHECK(hx_file_replace(dir, dir2) == -1 && hx_path_exists(dir) && !hx_path_exists(dir2),
              "a directory is not moved to a new name");
        remove_dir(dir2);
    }
    remove_dir(dir);
    CHECK(!hx_path_exists(dir), "scratch directory removed");

    /* a destination in a missing directory fails at once: not a sharing error to wait out */
    {
        char nodir[1300];
        snprintf(nodir, sizeof nodir, "%s_missing/x.dst", dir);
        t0 = hx_now_ns();
        CHECK(hx_file_replace(tmp, nodir) == -1 && hx_path_exists(tmp), "replace into a missing directory fails");
        CHECK(hx_now_ns() - t0 < 200000000u, "missing directory refused without retrying (%.0f ms)",
              (double)(hx_now_ns() - t0) / 1e6);
    }

    /* UTF-8 names, and paths beyond the classic 260 characters */
    CHECK(put_text(ut, "utf8") && hx_file_replace(ut, ud) == 0 && strcmp(get_text(ud, txt, sizeof txt), "utf8") == 0 &&
          !hx_path_exists(ut), "UTF-8 paths");
    {
        char lname[240];
        memset(lname, 'R', 200);
        snprintf(lname + 200, sizeof lname - 200, ".tmp");
        scratch_path(lt, sizeof lt, lname);
        snprintf(lname + 200, sizeof lname - 200, ".dst");
        scratch_path(ld, sizeof ld, lname);
        CHECK(put_text(lt, "long") && hx_file_replace(lt, ld) == 0 && strcmp(get_text(ld, txt, sizeof txt), "long") == 0 &&
              !hx_path_exists(lt), "long paths (%zu chars)", strlen(ld));
        CHECK(put_text(lt, "long2") && hx_file_replace(lt, ld) == 0 && strcmp(get_text(ld, txt, sizeof txt), "long2") == 0,
              "long path onto an existing file");
    }

#if defined(HX_OS_WINDOWS)
    /* A reader without FILE_SHARE_DELETE (the CRT's fopen) blocks the rename: retried for a while, then given up. */
    CHECK(put_text(tmp, "retry"), "write tmp");
    FILE *fp = fopen(dst, "rb");
    CHECK(fp != NULL, "fopen dst");
    if (fp) {
        closer_job cj = {fp, 30000};
        hx_thread *ct;
        CHECK(hx_thread_create(&ct, close_later, &cj) == 0, "thread create");
        CHECK(hx_file_replace(tmp, dst) == 0, "replace succeeds once a non-sharing reader closes dst");
        hx_thread_join(ct);
        CHECK(strcmp(get_text(dst, txt, sizeof txt), "retry") == 0, "dst replaced after the retry");
    }
    CHECK(put_text(tmp, "blocked"), "write tmp");
    fp = fopen(dst, "rb");
    if (fp) {
        t0 = hx_now_ns();
        int rc = hx_file_replace(tmp, dst);
        double ms = (double)(hx_now_ns() - t0) / 1e6;
        fclose(fp);
        CHECK(rc == -1 && ms < 5000.0, "replace gives up while dst stays locked (%d after %.0f ms)", rc, ms);
        CHECK(strcmp(get_text(dst, txt, sizeof txt), "retry") == 0 && hx_path_exists(tmp), "locked dst untouched");
    }
    /* tmp held without delete sharing: nothing is renamed, and that is reported */
    fp = fopen(tmp, "rb");
    if (fp) {
        int rc = hx_file_replace(tmp, dst);
        fclose(fp);
        CHECK(rc == -1 && strcmp(get_text(dst, txt, sizeof txt), "retry") == 0 && hx_path_exists(tmp),
              "replace from a tmp held without delete sharing fails, dst untouched (%d)", rc);
    }
#else
    CHECK(put_text(dst, "retry"), "write dst");
#endif

    /* a read-only dst is replaced, as rename(2) does (permission is the directory's); without POSIX
     * renames it is up to the file system (FAT refuses, a 9P share replaces): a refusal comes at
     * once, since waiting would not help */
    CHECK(put_text(tmp, "ro") && make_read_only(dst, 1), "make dst read-only");
    t0 = hx_now_ns();
    {
        int rc = hx_file_replace(tmp, dst);
        double ms = (double)(hx_now_ns() - t0) / 1e6;
        make_read_only(dst, 0);
        if (posix || rc == 0)
            CHECK(rc == 0 && strcmp(get_text(dst, txt, sizeof txt), "ro") == 0, "read-only dst replaced (%d)", rc);
        else
            CHECK(ms < 200.0 && strcmp(get_text(dst, txt, sizeof txt), "retry") == 0 && hx_path_exists(tmp),
                  "read-only dst refused without retrying (%d after %.0f ms)", rc, ms);
        remove(tmp);
    }

    /* a symbolic link tmp is moved as a link; its target stays where it is */
    {
        char target[1200];
        scratch_path(target, sizeof target, "rep.target");
        CHECK(put_text(target, "target"), "write link target");
        if (make_symlink(target, tmp)) {
            CHECK(hx_file_replace(tmp, dst) == 0, "replace from a symbolic link");
            CHECK(is_symlink(dst) && !hx_path_exists(tmp), "dst is now the link");
            CHECK(strcmp(get_text(target, txt, sizeof txt), "target") == 0 &&
                      strcmp(get_text(dst, txt, sizeof txt), "target") == 0,
                  "the link's target stays in place");
            remove(dst);
        } else {
            printf("  (symbolic links not permitted here: link check skipped)\n");
        }
        remove(target);
    }

    /* concurrent readers never see a missing or torn dst */
    {
        uint8_t *blk = (uint8_t *)malloc(REP_LEN);
        rep_reader rr;
        rr.path = dst;
        atomic_init(&rr.stop, 0);
        rr.torn = rr.opened = rr.missing = 0;
        memset(blk, 'A', REP_LEN);
        hx_file *f = hx_file_open(dst, HX_FILE_WRITE | HX_FILE_CREATE, NULL, 0);
        if (f) { hx_file_pwrite(f, blk, REP_LEN, 0); hx_file_close(f); }
        hx_thread *rt;
        CHECK(hx_thread_create(&rt, replace_reader, &rr) == 0, "thread create");
        int fails = 0, rounds = 300;
        for (int i = 0; i < rounds; i++) {
            memset(blk, 'B' + i % 20, REP_LEN);
            f = hx_file_open(tmp, HX_FILE_WRITE | HX_FILE_CREATE, NULL, 0);
            if (!f) { fails++; continue; }
            hx_file_pwrite(f, blk, REP_LEN, 0);
            hx_file_close(f);
            fails += hx_file_replace(tmp, dst) != 0;
        }
        atomic_store(&rr.stop, 1);
        hx_thread_join(rt);
        free(blk);
        printf("  %d replaces under a concurrent reader: %d failed; reader: %d reads, %d torn, %d open failures\n",
               rounds, fails, rr.opened, rr.torn, rr.missing);
        if (posix) CHECK(fails == 0, "%d of %d replaces failed under a concurrent reader", fails, rounds);
        CHECK(rr.torn == 0 && rr.missing == 0 && rr.opened > 0, "reader saw %d torn and %d missing files", rr.torn,
              rr.missing);
    }

    remove(tmp);
    remove(dst);
    remove_utf8(ud);
#if defined(HX_OS_WINDOWS)
    {
        char x[1600];                             /* the CRT needs the \\?\ form for a long path */
        int unc = (ld[0] == '\\' || ld[0] == '/') && (ld[1] == '\\' || ld[1] == '/');
        snprintf(x, sizeof x, unc ? "\\\\?\\UNC\\%s" : "\\\\?\\%s", unc ? ld + 2 : ld);
        for (char *q = x + 4; *q; q++) if (*q == '/') *q = '\\';
        remove(x);
    }
#else
    remove(ld);
#endif
    CHECK(!hx_path_exists(tmp) && !hx_path_exists(dst) && !hx_path_exists(ud) && !hx_path_exists(ld),
          "replace scratch files removed");
}

/* Child: log (at debug level) a replace of a missing tmp into a missing directory. */
static int replace_log_child(const char *out, const char *dir) {
    char tmp[1200], dst[1200];
    unsigned long long tag = (unsigned long long)hx_now_ns();
    snprintf(tmp, sizeof tmp, "%s/hx_missing_%llx.tmp", dir, tag);
    snprintf(dst, sizeof dst, "%s/hx_missing_dir_%llx/x.dst", dir, tag);
    if (!freopen(out, "w", stderr)) return 3;
    hx_set_log_level(HX_LOG_DEBUG);
    int rc = hx_file_replace(tmp, dst);
    fflush(stderr);
    return rc == -1 ? 0 : 4;
}

/* The failure is reported with tmp's error (file not found), not dst's (path not found). */
static void test_replace_log(const char *self) {
    printf("hx_file_replace error report\n");
    if (!self[0]) { printf("  skipped: cannot locate this executable from argv[0]\n"); return; }
    char out[1200], cmd[4096];
    scratch_path(out, sizeof out, "replace_log.txt");
#if defined(HX_OS_WINDOWS)
    snprintf(cmd, sizeof cmd, "\"\"%s\" --replace-log-child \"%s\" \"%s\"\"", self, out, g_dir);
    const char *want = "(error 2)";             /* ERROR_FILE_NOT_FOUND; dst's would be 3 */
#else
    snprintf(cmd, sizeof cmd, "'%s' --replace-log-child '%s' '%s'", self, out, g_dir);
    const char *want = "No such file or directory";
#endif
    fflush(stdout);
    int rc = system(cmd);
    char got[2048];
    size_t n = 0;
    FILE *fp = fopen(out, "r");
    if (fp) { n = fread(got, 1, sizeof got - 1, fp); fclose(fp); }
    got[n] = 0;
    remove(out);
    CHECK(rc == 0 && strstr(got, "hx_file_replace") && strstr(got, want), "replace log (exit %d): %s", rc, got);
}

/* ------------------------------------------------- CPU counts under an affinity mask */

static int cpu_child(const char *out) {
    FILE *fp = fopen(out, "w");
    if (!fp) return 3;
    fprintf(fp, "%d %d\n", hx_num_cpus(), hx_num_physical_cores());
    fclose(fp);
    return 0;
}

static char g_self[1100];

/* Runs this program pinned to `mask`; 1 and the child's counts on success. */
static int counts_pinned(unsigned long long mask, int *cpus, int *cores) {
    char out[1300], cmd[3000];
    scratch_path(out, sizeof out, "cpus.txt");
#if defined(HX_OS_WINDOWS)
    snprintf(cmd, sizeof cmd, "start \"\" /b /wait /affinity %llx \"%s\" --cpu-child \"%s\"", mask, g_self, out);
#else
    snprintf(cmd, sizeof cmd, "taskset 0x%llx '%s' --cpu-child '%s'", mask, g_self, out);
#endif
    fflush(stdout);
    int rc = system(cmd);
    FILE *fp = fopen(out, "r");
    int ok = fp && fscanf(fp, "%d %d", cpus, cores) == 2;
    if (fp) fclose(fp);
    remove(out);
    if (!ok) printf("  pinned child failed (status %d)\n", rc);
    return ok;
}

/* Whether CPUs 0 and 1 are hardware threads of one core: -1 if unknown. */
static int cpu01_siblings(void) {
#if defined(HX_OS_WINDOWS)
    /* Windows numbers the SMT threads of a core next to each other. */
    return hx_num_cpus() == 2 * hx_num_physical_cores();
#else
    FILE *fp = fopen("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list", "r");
    if (!fp) return -1;
    char line[256];
    int res = -1;
    if (fgets(line, sizeof line, fp)) {
        res = 0;
        for (char *tok = strtok(line, ",\n"); tok; tok = strtok(NULL, ",\n")) {
            int a = -1, b = -1;
            if (sscanf(tok, "%d-%d", &a, &b) == 2) res |= a <= 1 && 1 <= b;
            else if (sscanf(tok, "%d", &a) == 1) res |= a == 1;
        }
    }
    fclose(fp);
    return res;
#endif
}

static void test_cpu_affinity(void) {
    printf("cpu counts under an affinity mask\n");
    if (hx_num_cpus() < 3) { printf("  skipped (fewer than 3 CPUs)\n"); return; }
    if (!g_self[0]) { printf("  skipped: cannot locate this executable from argv[0]\n"); return; }
#if !defined(HX_OS_WINDOWS)
    if (system("command -v taskset >/dev/null 2>&1") != 0) { printf("  skipped: no taskset\n"); return; }
#endif
    int sib = cpu01_siblings();
    struct { unsigned long long mask; int cpus, cores; } cases[3] = {{0x1, 1, 1}, {0x5, 2, 2}, {0x3, 2, -1}};
    cases[2].cores = sib < 0 ? -1 : sib ? 1 : 2;
    for (int i = 0; i < 3; i++) {
        int cpus = -1, cores = -1;
        if (!counts_pinned(cases[i].mask, &cpus, &cores)) { CHECK(0, "pinned child (mask %llx)", cases[i].mask); continue; }
        printf("  mask %llx: %d cpus, %d cores\n", cases[i].mask, cpus, cores);
        CHECK(cpus == cases[i].cpus, "mask %llx: hx_num_cpus %d, want %d", cases[i].mask, cpus, cases[i].cpus);
        if (cases[i].cores >= 0)
            CHECK(cores == cases[i].cores, "mask %llx: hx_num_physical_cores %d, want %d", cases[i].mask, cores,
                  cases[i].cores);
        else
            CHECK(cores >= 1 && cores <= cpus, "mask %llx: hx_num_physical_cores %d", cases[i].mask, cores);
    }
}

/* Opt-in (--large): one request larger than the 1 GiB internal chunk, at an
 * offset above 2^31, on direct and buffered handles. Writes a 2.25 GiB file. */
static uint64_t big_word(uint64_t off) { return off * 0x9e3779b97f4a7c15ull ^ 0x5bd1e995u; }

static int verify_words(const uint8_t *buf, uint64_t off, size_t n) {
    for (size_t i = 0; i < n; i += 8) {
        uint64_t v;
        memcpy(&v, buf + i, 8);
        if (v != big_word(off + i)) return 0;
    }
    return 1;
}

static void test_large(void) {
    printf("large I/O\n");
    char path[1200], err[256];
    scratch_path(path, sizeof path, "large.bin");
    const uint64_t total = (9ull << 30) / 4;      /* 2.25 GiB */
    const size_t blk = 64u << 20;
    uint8_t *buf = (uint8_t *)hx_alloc_large(blk, 0);
    hx_file *w = hx_file_open(path, HX_FILE_WRITE | HX_FILE_CREATE | HX_FILE_DIRECT, err, sizeof err);
    CHECK(w && buf, "large create: %s", err);
    if (!w || !buf) { if (w) hx_file_close(w); hx_free_large(buf, blk); return; }
    uint64_t t0 = hx_now_ns();
    int okw = 1;
    for (uint64_t off = 0; off < total; off += blk) {
        for (size_t i = 0; i < blk; i += 8) { uint64_t v = big_word(off + i); memcpy(buf + i, &v, 8); }
        okw &= hx_file_pwrite(w, buf, blk, off) == (int64_t)blk;
    }
    hx_file_close(w);
    hx_free_large(buf, blk);
    CHECK(okw, "large write");
    printf("  wrote %.2f GiB in %.2f s\n", (double)total / (1u << 30), (double)(hx_now_ns() - t0) * 1e-9);

    const uint64_t off = 3ull << 28;              /* 0.75 GiB: the request crosses 2^31 */
    const size_t n = (size_t)(3ull << 29);        /* 1.5 GiB: more than one internal chunk */
    uint8_t *big = (uint8_t *)hx_alloc_large(n, 0);
    CHECK(big != NULL, "large buffer");
    for (int direct = 1; direct >= 0 && big; direct--) {
        hx_file *f = hx_file_open(path, HX_FILE_READ | (direct ? HX_FILE_DIRECT : 0), err, sizeof err);
        CHECK(f != NULL, "large open: %s", err);
        if (!f) continue;
        t0 = hx_now_ns();
        int64_t got = hx_file_pread(f, big, n, off);
        double s = (double)(hx_now_ns() - t0) * 1e-9;
        CHECK(got == (int64_t)n && verify_words(big, off, n), "large pread (direct=%d) got %lld", hx_file_is_direct(f),
              (long long)got);
        int64_t tail = hx_file_pread(f, big, n, total - (64u << 20));
        CHECK(tail == (int64_t)(64u << 20) && verify_words(big, total - (64u << 20), (size_t)tail),
              "large short read at EOF (direct=%d)", hx_file_is_direct(f));
        printf("  1.5 GiB pread at 0.75 GiB, is_direct=%d: ok, %.2f GB/s (single request)\n", hx_file_is_direct(f),
               (double)n / s / 1e9);
        hx_file_close(f);
    }
    hx_free_large(big, n);
    remove(path);
    CHECK(!hx_path_exists(path), "large file removed");
}

/* --------------------------------------------------------------- env & log */

static void set_env(const char *k, const char *v) {
#if defined(HX_OS_WINDOWS)
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

/* A variable set to "" reads as "" (not NULL), whatever lookups came before it. */
static void test_env_empty(void) {
#if defined(HX_OS_WINDOWS)
    SetEnvironmentVariableA("HX_TEST_EMPTY", "");
#else
    setenv("HX_TEST_EMPTY", "", 1);
#endif
    const char *a = hx_env_str("HX_TEST_EMPTY");
    CHECK(hx_env_str("HX_TEST_SURELY_UNSET_VAR") == NULL, "unset var is NULL");   /* leaves "not found" behind */
    const char *b = hx_env_str("HX_TEST_EMPTY");
    CHECK(a && *a == 0, "empty variable reads as \"\" (got %s)", a ? "a non-empty string" : "NULL");
    CHECK(b && *b == 0, "empty variable still reads as \"\" after an unset lookup (got %s)", b ? "a non-empty string" : "NULL");
    CHECK(hx_env_int("HX_TEST_EMPTY", 7) == 7 && hx_env_double("HX_TEST_EMPTY", 2.5) == 2.5, "empty variable -> defaults");
#if defined(HX_OS_WINDOWS)
    SetEnvironmentVariableA("HX_TEST_EMPTY", NULL);   /* not in the CRT's copy, so _putenv_s would not remove it */
#else
    set_env("HX_TEST_EMPTY", NULL);
#endif
    CHECK(hx_env_str("HX_TEST_EMPTY") == NULL, "removed empty variable is NULL");
}

/* A variable whose value keeps changing must not slow down later lookups of others (the
 * Windows cache once kept every value ever seen, in one list searched linearly). */
static void test_env_churn(void) {
    set_env("HX_TEST_A", "a");
    hx_env_str("HX_TEST_A");
    uint64_t t0 = hx_now_ns();
    for (int i = 0; i < 2000; i++) hx_env_str("HX_TEST_A");
    uint64_t before = hx_now_ns() - t0;
    char v[32];
    t0 = hx_now_ns();
    for (int i = 0; i < 30000; i++) {
        snprintf(v, sizeof v, "%d", i);
        set_env("HX_TEST_CHURN", v);
        const char *got = hx_env_str("HX_TEST_CHURN");
        if (!got || strcmp(got, v) != 0) { CHECK(0, "churn value %d read back as %s", i, got ? got : "NULL"); break; }
    }
    uint64_t churn = hx_now_ns() - t0;
    t0 = hx_now_ns();
    for (int i = 0; i < 2000; i++) hx_env_str("HX_TEST_A");
    uint64_t after = hx_now_ns() - t0;
    printf("  env: 2000 lookups %.2f ms before / %.2f ms after 30000 values of another variable (%.2f s)\n",
           (double)before / 1e6, (double)after / 1e6, (double)churn * 1e-9);
    CHECK(after < before * 10 + 20000000u, "lookups slowed down by another variable's history");
#if defined(HX_OS_WINDOWS)
    /* returned pointers stay valid for the next 8 changes of the variable (ASan checks this) */
    set_env("HX_TEST_CHURN", "first");
    const char *first = hx_env_str("HX_TEST_CHURN");
    for (int i = 0; i < 8; i++) {
        snprintf(v, sizeof v, "next%d", i);
        set_env("HX_TEST_CHURN", v);
        hx_env_str("HX_TEST_CHURN");
    }
    CHECK(first && strcmp(first, "first") == 0, "value pointer valid after 8 changes");
    /* each variable keeps its own cached copy across lookups of others */
    set_env("HX_TEST_B", "b");
    const char *pa = hx_env_str("HX_TEST_A"), *pb = hx_env_str("HX_TEST_B");
    CHECK(pa && pb && hx_env_str("HX_TEST_A") == pa && hx_env_str("HX_TEST_B") == pb && strcmp(pa, "a") == 0,
          "interleaved lookups keep each variable's cached copy");
    set_env("HX_TEST_B", NULL);
#endif
    set_env("HX_TEST_CHURN", NULL);
    set_env("HX_TEST_A", NULL);
}

/* hx_env_double parses C syntax whatever LC_NUMERIC the host process uses. */
/* A locale with a decimal comma, found before any library call could have disturbed the
 * thread's locale (NULL if none is installed). */
static const char *g_comma_locale;

static void find_comma_locale(void) {
    static const char *const comma[] = {"de_DE.UTF-8", "de_DE.utf8", "de_DE", "de-DE", "German_Germany.1252",
                                        "fr_FR.UTF-8", "fr_FR.utf8", "fr-FR", "nl_NL.UTF-8", "ru_RU.UTF-8"};
    for (size_t i = 0; i < sizeof comma / sizeof comma[0] && !g_comma_locale; i++)
        if (setlocale(LC_NUMERIC, comma[i]) && localeconv()->decimal_point[0] == ',') g_comma_locale = comma[i];
    setlocale(LC_NUMERIC, "C");
}

static void test_env_locale(void) {
    const char *used = g_comma_locale;
    if (!used) {
        printf("  (no locale with a decimal comma is installed: LC_NUMERIC check skipped)\n");
        return;
    }
    setlocale(LC_NUMERIC, used);
    CHECK(localeconv()->decimal_point[0] == ',', "LC_NUMERIC=%s no longer takes effect on this thread", used);
    int saved = hx_get_log_level();
    hx_set_log_level(HX_LOG_ERROR);
    set_env("HX_TEST_V", "0.25");
    double a = hx_env_double("HX_TEST_V", -1.0);
    set_env("HX_TEST_V", "0,25");
    double b = hx_env_double("HX_TEST_V", -1.0);
    set_env("HX_TEST_V", "-1.5e3");
    double c = hx_env_double("HX_TEST_V", -1.0);
    int kept = localeconv()->decimal_point[0] == ',';   /* the caller's locale is left as it was */
    setlocale(LC_NUMERIC, "C");
    CHECK(kept, "hx_env_double restored the calling thread's locale");
    hx_set_log_level(saved);
    set_env("HX_TEST_V", NULL);
    printf("  LC_NUMERIC=%s: \"0.25\" -> %g, \"0,25\" -> %g\n", used, a, b);
    CHECK(a == 0.25 && c == -1500.0, "hx_env_double under LC_NUMERIC=%s: %g, %g", used, a, c);
    CHECK(b == -1.0, "decimal comma rejected under LC_NUMERIC=%s (got %g)", used, b);
}

static void test_env_log(int initial_level, const char *initial_env) {
    printf("env & logging\n");
    long long want = HX_LOG_WARN;
    if (initial_env && *initial_env) {
        char *end;
        long long v = strtoll(initial_env, &end, 10);
        int ok = end != initial_env;
        while (ok && (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')) end++;   /* as the library */
        if (ok && *end == 0) want = v < 0 ? 0 : v > 3 ? 3 : v;
    }
    CHECK(initial_level == (int)want, "initial log level %d (HEARTH_LOG=%s), want %lld", initial_level,
          initial_env ? initial_env : "<unset>", want);
    int saved = hx_get_log_level();
    hx_set_log_level(7);
    CHECK(hx_get_log_level() == 3, "level clamps high");
    hx_set_log_level(-4);
    CHECK(hx_get_log_level() == 0, "level clamps low");

    CHECK(hx_env_str("HX_TEST_SURELY_UNSET_VAR") == NULL, "unset var is NULL");
    CHECK(hx_env_str(NULL) == NULL && hx_env_str("") == NULL, "NULL / empty name is NULL");
    CHECK(hx_env_int("HX_TEST_SURELY_UNSET_VAR", 17) == 17, "unset int default");
    CHECK(hx_env_double("HX_TEST_SURELY_UNSET_VAR", 2.5) == 2.5, "unset double default");
    set_env("HX_TEST_V", "42");
    CHECK(hx_env_int("HX_TEST_V", 0) == 42, "int 42");
    CHECK(hx_env_str("HX_TEST_V") && strcmp(hx_env_str("HX_TEST_V"), "42") == 0, "str 42");
    const char *keep = hx_env_str("HX_TEST_V");
    int same = 1;
    for (int i = 0; i < 100; i++) same &= hx_env_str("HX_TEST_V") == keep;
#if defined(HX_OS_WINDOWS)
    CHECK(same, "unchanged value reuses its cached UTF-8 copy");
#else
    (void)same;
#endif
    set_env("HX_TEST_V", "  -7 ");
    CHECK(hx_env_int("HX_TEST_V", 0) == -7, "int -7 with spaces");
    CHECK(keep && strcmp(keep, "42") == 0, "earlier returned pointer still valid");
    set_env("HX_TEST_V", "8\t\r\n");
    CHECK(hx_env_int("HX_TEST_V", 0) == 8, "int with trailing tab/CR/LF");
    set_env("HX_TEST_V", "12abc");
    CHECK(hx_env_int("HX_TEST_V", 5) == 5, "trailing junk -> default");
    set_env("HX_TEST_V", "99999999999");
    CHECK(hx_env_int("HX_TEST_V", 5) == 5, "out of range -> default");
    set_env("HX_TEST_V", "1.5e-3");
    CHECK(hx_env_double("HX_TEST_V", 0) == 1.5e-3, "double 1.5e-3");
    set_env("HX_TEST_V", "x1");
    CHECK(hx_env_double("HX_TEST_V", 9.0) == 9.0, "bad double -> default");
    set_env("HX_TEST_V", "1e999");
    CHECK(hx_env_double("HX_TEST_V", 9.0) == 9.0, "out-of-range double -> default");
    set_env("HX_TEST_V", " 0.25 ");
    CHECK(hx_env_double("HX_TEST_V", 9.0) == 0.25, "double with spaces");
#if defined(HX_OS_WINDOWS)
    _wputenv_s(L"HX_TEST_V", L"été");   /* the narrow CRT setter would use the ANSI code page */
#else
    set_env("HX_TEST_V", "\xc3\xa9t\xc3\xa9");
#endif
    CHECK(hx_env_str("HX_TEST_V") && strcmp(hx_env_str("HX_TEST_V"), "\xc3\xa9t\xc3\xa9") == 0, "env value is UTF-8");
    set_env("HX_TEST_V", NULL);
    CHECK(hx_env_int("HX_TEST_V", 3) == 3, "removed var -> default");
    test_env_empty();
    test_env_churn();
    test_env_locale();

    char e[16];
    CHECK(hx_fail(e, sizeof e, "code %d: %s", 12, "a long message here") == NULL, "hx_fail returns NULL");
    CHECK(strcmp(e, "code 12: a long") == 0, "hx_fail truncates: \"%s\"", e);
    CHECK(hx_fail(NULL, 0, "x") == NULL, "hx_fail without buffer");

    hx_set_log_level(HX_LOG_DEBUG);
    char big[1100];                                /* longer than the logger's stack buffer */
    memset(big, 'x', sizeof big - 1);
    big[sizeof big - 1] = 0;
    hx_log(HX_LOG_DEBUG, "long line test (%zu chars): %.40s...", strlen(big), big);
    hx_log(HX_LOG_INFO, "%s", big);                /* exercises the heap path */
    hx_set_log_level(HX_LOG_ERROR);
    hx_log(HX_LOG_WARN, "this warning must not print");
    hx_set_log_level(saved);
}

/* ---- log output: a child run of this program logs into a file we inspect ---- */
#define BIGLOG 1100   /* longer than the logger's stack buffer */

static int log_child(const char *path) {
    int initial = hx_get_log_level();              /* parent sets HEARTH_LOG=3 */
    if (!freopen(path, "w", stderr)) return 3;
    hx_log(HX_LOG_ERROR, "initial %d", initial);
    hx_set_log_level(HX_LOG_WARN);
    hx_log(HX_LOG_WARN, "w %d", 1);
    hx_log(HX_LOG_INFO, "hidden");
    hx_log(HX_LOG_ERROR, "e\n");                   /* no doubled newline */
    hx_set_log_level(HX_LOG_DEBUG);
    hx_log(HX_LOG_INFO, "%s", "");
    char big[BIGLOG + 1];
    memset(big, 'y', BIGLOG);
    big[BIGLOG] = 0;
    hx_log(HX_LOG_DEBUG, "%s", big);
    char e[8];
    hx_fail(e, sizeof e, "fail %d", 7);            /* echoed at debug level */
    set_env("HX_TEST_V", "zz");
    hx_env_int("HX_TEST_V", 1);
    hx_env_double("HX_TEST_V", 1.0);
    hx_set_log_level(HX_LOG_ERROR);
    hx_env_int("HX_TEST_V", 1);
    hx_log(HX_LOG_WARN, "suppressed");
    hx_log(-5, "neg %d", -5);
    hx_log(9, "too verbose");
    fflush(stderr);
    return 0;
}

static void test_log_output(const char *self) {
    printf("log output\n");
    char exe[1100];
    snprintf(exe, sizeof exe, "%s", self ? self : "");
    if (!hx_path_exists(exe)) snprintf(exe, sizeof exe, "%s.exe", self ? self : "");
    if (!hx_path_exists(exe)) { printf("  skipped: cannot locate this executable from argv[0]\n"); return; }
    char out[1200], cmd[2600];
    scratch_path(out, sizeof out, "log.txt");
#if defined(HX_OS_WINDOWS)
    snprintf(cmd, sizeof cmd, "\"\"%s\" --log-child \"%s\"\"", exe, out);   /* cmd /c strips the outer pair */
#else
    snprintf(cmd, sizeof cmd, "'%s' --log-child '%s'", exe, out);
#endif
    char saved[256];
    const char *old = getenv("HEARTH_LOG");
    int had = old != NULL;
    snprintf(saved, sizeof saved, "%s", old ? old : "");
    set_env("HEARTH_LOG", "3");
    fflush(stdout);
    int rc = system(cmd);
    set_env("HEARTH_LOG", had ? saved : NULL);
    CHECK(rc == 0, "log child exit %d", rc);

    char want[BIGLOG + 1024];
    int k = snprintf(want, sizeof want,
                     "[hearth] error: initial 3\n[hearth] warning: w 1\n[hearth] error: e\n[hearth] \n[hearth] ");
    memset(want + k, 'y', BIGLOG);
    snprintf(want + k + BIGLOG, sizeof want - (size_t)k - BIGLOG,
             "\n[hearth] fail 7\n"
             "[hearth] warning: ignoring HX_TEST_V=\"zz\": not an integer\n"
             "[hearth] warning: ignoring HX_TEST_V=\"zz\": not a number\n"
             "[hearth] error: neg -5\n");
    char got[sizeof want + 256];
    size_t n = 0;
    FILE *fp = fopen(out, "r");
    if (fp) { n = fread(got, 1, sizeof got - 1, fp); fclose(fp); }
    got[n] = 0;
    CHECK(fp && strcmp(got, want) == 0, "log output mismatch:\n----- got\n%s----- want\n%s-----", got, want);
    remove(out);
}

int main(int argc, char **argv) {
    if (argc == 3 && strcmp(argv[1], "--log-child") == 0) return log_child(argv[2]);
    if (argc == 3 && strcmp(argv[1], "--cpu-child") == 0) return cpu_child(argv[2]);
    if (argc == 4 && strcmp(argv[1], "--replace-log-child") == 0) return replace_log_child(argv[2], argv[3]);
    find_comma_locale();
    int initial_level = hx_get_log_level();
    const char *initial_env = getenv("HEARTH_LOG");
    int exhaustive = 0, large = 0;
    const char *dir = NULL, *expect_cpu = NULL, *replace_dir = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--exhaustive") == 0) exhaustive = 1;
        else if (strcmp(argv[i], "--large") == 0) large = 1;
        else if (strcmp(argv[i], "--expect-cpu") == 0 && i + 1 < argc) expect_cpu = argv[++i];
        else if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc) dir = argv[++i];
        else if (strcmp(argv[i], "--replace-dir") == 0 && i + 1 < argc) replace_dir = argv[++i];
        else {
            printf("usage: test_platform [--dir DIR] [--replace-dir DIR] [--exhaustive] [--large] [--expect-cpu LIST]\n");
            return 2;
        }
    }
    if (!dir) {
        const char *cands[] = {"HEARTH_TEST_DIR", "HEARTH_DATA", "TMPDIR", "TEMP", "TMP"};
        for (size_t i = 0; i < sizeof cands / sizeof cands[0] && !dir; i++) {
            const char *v = hx_env_str(cands[i]);
            if (v && *v && hx_path_exists(v)) dir = v;
        }
        if (!dir) dir = hx_path_exists("/tmp") ? "/tmp" : ".";
    }
    snprintf(g_dir, sizeof g_dir, "%s", dir);
    g_tag = hx_now_ns();

    build_tables();
    uint64_t t0 = hx_now_ns();
    test_env_log(initial_level, initial_env);
    test_f16();
    test_bf16();
    if (exhaustive) test_exhaustive();
    test_memory();
    test_time();
    test_threads();
    snprintf(g_self, sizeof g_self, "%s", argv[0]);
    if (!hx_path_exists(g_self)) snprintf(g_self, sizeof g_self, "%s.exe", argv[0]);
    if (!hx_path_exists(g_self)) g_self[0] = 0;
    test_files();
    test_replace(1);
    test_replace_log(g_self);
    if (replace_dir) {
        snprintf(g_dir, sizeof g_dir, "%s", replace_dir);
        test_replace(0);
        snprintf(g_dir, sizeof g_dir, "%s", dir);
    }
    test_cpu_affinity();
    test_log_output(argv[0]);
    if (expect_cpu) check_expected_cpu(expect_cpu);
    if (large) test_large();
    printf("test_platform: %d checks, %d failed (%.1f s)\n", g_checks, g_fail, (double)(hx_now_ns() - t0) * 1e-9);
    return g_fail ? 1 : 0;
}
