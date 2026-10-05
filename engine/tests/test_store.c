/*
 * test_store.c — self-checking tests and an NVMe benchmark for store.c (hx_store.h).
 *
 *   test_store [--quick] [--seed N] [--dir D]
 *   test_store --bench [--gib N] [--dir D]
 *
 * Tests: every slab handed out is compared (64-bit content hash) with the bytes
 * the test wrote for that (layer, expert), both when acquired and again just
 * before release, so an eviction of an in-use slot or a torn read is caught.
 * Random acquire / try_acquire / release / prefetch / tick / wait_any sequences
 * run against 1..16 readers, LRU and LFU, minimum-to-full caches, direct and
 * buffered I/O, with one or several caller threads. A watchdog fails the run
 * if no operation completes for 60 s (deadlock). Further checks: the minimum
 * slot count with every top-k demanded at once while the previous layer is
 * still held, pinning (count as hearth.sim computes it, capped so the minimum
 * stays unpinned) and the usage file round trip (also for all-dense containers,
 * exact u64 token counts), seeded heat, ignored corrupt profiles, warm start,
 * mirrors (validation, reads spread over the copies, failover), direct vs
 * buffered (bytes and handle mode), read failures, closing with queued work, a
 * decode loop that detects lost wait_any wakeups, and the policy check (cyclic
 * layer access, cache smaller than one token's working set, Zipf-skewed routing:
 * LRU vs LFU hit rate).
 * Scheduling-order properties (prefetch promotion, protection expiry, the
 * prefetch queue bound, the prefetch hot rule, starvation wakeups and holds,
 * wait_any's wakeups) run with readers whose reads are gated through store.c's
 * read hook, which makes them deterministic.
 *
 * --bench writes a ~N GiB (default 6) synthetic container into the data dir
 * ($HEARTH_DATA, else %LOCALAPPDATA%/hearth or ~/.cache/hearth; never the
 * current directory) with unbuffered writes, measures cold demand-read
 * throughput for 1/2/4/8/16 readers with
 * Qwen3-30B-A3B-sized (2.39 MiB) and DeepSeek-V3-sized (22.3 MiB) Q4 slabs, direct
 * and buffered, prints a table and deletes the file. Synthetic weights; the
 * numbers depend on whatever else the machine is doing.
 */
#include "hx_store.h"
#include "hx_modelfile.h"
#include "hx_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static atomic_int g_checks, g_fail;

#define CHECK(cond, ...)                                                         \
    do {                                                                         \
        atomic_fetch_add(&g_checks, 1);                                          \
        if (!(cond)) {                                                           \
            if (atomic_fetch_add(&g_fail, 1) < 40) {                             \
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
static uint32_t rn(uint64_t *s, uint32_t n) { return n ? (uint32_t)(splitmix(s) % n) : 0; }

/* n is a multiple of 8 (slabs are whole pages) */
static uint64_t hash_bytes(const void *p, uint64_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 0x243F6A8885A308D3ull ^ n;
    for (uint64_t i = 0; i < n; i += 8) {
        uint64_t w;
        memcpy(&w, b + i, 8);
        h = (h ^ w) * 0x9E3779B97F4A7C15ull;
        h ^= h >> 29;
    }
    return h;
}

static void fill_random(uint8_t *p, uint64_t n, uint64_t seed) {
    uint64_t x = seed * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull;
    for (uint64_t i = 0; i + 8 <= n; i += 8) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        memcpy(p + i, &x, 8);
    }
}

/* ------------------------------------------------------------ watchdog */

static atomic_ullong g_progress;
static atomic_int g_watch_stop;
static const char *volatile g_phase = "start";

static void *watchdog(void *arg) {
    (void)arg;
    unsigned long long last = 0;
    int idle = 0;
    while (!atomic_load(&g_watch_stop)) {
        hx_sleep_us(250000);
        unsigned long long p = atomic_load(&g_progress);
        idle = p == last ? idle + 1 : 0;
        last = p;
        if (idle >= 240) {
            printf("  FAIL: no progress for 60 s in phase '%s' (deadlock?)\n", g_phase);
            fflush(stdout);
            exit(3);
        }
    }
    return NULL;
}
static void tickle(void) { atomic_fetch_add(&g_progress, 1); }

/* ---------------------------------------------------- container writer */

typedef struct {
    int L, E, D, F, K;
    uint8_t lk[64];
    int *dtype, *alias;          /* per entry; alias = target entry or -1 */
    uint64_t *off, *nbytes, *used, *hash;
    uint64_t size, slab_max, seed;
    int no_edir;                 /* write n_expert_entries = 0 (allowed when no layer is MoE) */
    char path[700];
} spec;

static void spec_free(spec *s) {
    free(s->dtype); free(s->alias); free(s->off); free(s->nbytes); free(s->used); free(s->hash);
    memset(s, 0, sizeof *s);
}

static void put16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void put64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }

static size_t meta_u32(uint8_t *m, const char *k, uint32_t v) {
    size_t n = strlen(k);
    if (m) { put16(m, (uint32_t)n); memcpy(m + 2, k, n); m[2 + n] = 1; put32(m + 3 + n, v); }
    return 7 + n;
}

/* Lays out a container with an optional dense layer mask, per-entry dtype and aliases,
 * computes every slab's expected hash and writes the file (unbuffered if direct). */
static int write_container(spec *s, int direct) {
    int ne = s->L * s->E, ne_dir = s->no_edir ? 0 : ne;
    s->off = (uint64_t *)calloc((size_t)ne, 8);
    s->nbytes = (uint64_t *)calloc((size_t)ne, 8);
    s->used = (uint64_t *)calloc((size_t)ne, 8);
    s->hash = (uint64_t *)calloc((size_t)ne, 8);
    int any_dense = 0;
    for (int l = 0; l < s->L; l++) any_dense |= !s->lk[l];

    /* metadata */
    uint8_t meta[512];
    size_t mn = 0;
    mn += meta_u32(meta + mn, "n_layers", (uint32_t)s->L);
    mn += meta_u32(meta + mn, "d_model", (uint32_t)s->D);
    mn += meta_u32(meta + mn, "vocab_size", 64);
    mn += meta_u32(meta + mn, "n_heads", 1);
    mn += meta_u32(meta + mn, "n_experts", (uint32_t)s->E);
    mn += meta_u32(meta + mn, "top_k", (uint32_t)s->K);
    mn += meta_u32(meta + mn, "expert_ffn_dim", (uint32_t)s->F);
    if (any_dense) mn += meta_u32(meta + mn, "dense_ffn_dim", 64);
    put16(meta + mn, 10); memcpy(meta + mn + 2, "layer_kind", 10); meta[mn + 12] = 7; put32(meta + mn + 13, (uint32_t)s->L);
    memcpy(meta + mn + 17, s->lk, (size_t)s->L);
    mn += 17 + (size_t)s->L;

    uint64_t meta_off = 64, tdir_off = meta_off + mn, edir_off = tdir_off + 128;
    uint64_t rope_off = (edir_off + 32 * (uint64_t)ne_dir + 63) / 64 * 64, rope_n = (uint64_t)s->D / 2 * 4;
    uint64_t hdr = (rope_off + rope_n + 4095) / 4096 * 4096, pos = hdr;
    s->slab_max = 0;
    for (int i = 0; i < ne; i++) {
        if (!s->lk[i / s->E]) continue;
        if (s->alias && s->alias[i] >= 0) continue;
        uint64_t og, ou, od;
        s->nbytes[i] = hx_slab_layout(s->dtype[i], s->D, s->F, &og, &ou, &od);
        s->used[i] = od + (uint64_t)s->D * hx_row_bytes(s->dtype[i], s->F);
        s->off[i] = pos;
        pos += s->nbytes[i];
        if (s->nbytes[i] > s->slab_max) s->slab_max = s->nbytes[i];
    }
    for (int i = 0; i < ne; i++)
        if (s->alias && s->alias[i] >= 0) {
            int t = s->alias[i];
            s->off[i] = s->off[t]; s->nbytes[i] = s->nbytes[t]; s->dtype[i] = s->dtype[t]; s->used[i] = s->used[t];
        }
    s->size = pos;

    uint8_t *h = (uint8_t *)hx_aligned_alloc(4096, (size_t)hdr);
    if (!h) return 0;
    memset(h, 0, (size_t)hdr);
    put32(h, HX_MAGIC); put32(h + 4, 1); put64(h + 8, meta_off); put64(h + 16, mn); put64(h + 24, tdir_off);
    put64(h + 32, 1); put64(h + 40, edir_off); put64(h + 48, (uint64_t)ne_dir); put32(h + 56, 4096);
    memcpy(h + meta_off, meta, mn);
    memcpy(h + tdir_off, "rope_inv_freq", 13);
    put32(h + tdir_off + 84, 1); put32(h + tdir_off + 88, (uint32_t)s->D / 2);
    for (int k = 1; k < 4; k++) put32(h + tdir_off + 88 + 4 * k, 1);
    put64(h + tdir_off + 104, rope_off); put64(h + tdir_off + 112, rope_n);
    for (int i = 0; i < ne_dir; i++) {
        uint8_t *e = h + edir_off + 32 * (uint64_t)i;
        put64(e, s->off[i]); put64(e + 8, s->nbytes[i]);
        put32(e + 16, (uint32_t)(s->nbytes[i] ? s->dtype[i] : 0));
        put32(e + 20, (s->alias && s->alias[i] >= 0) ? 1u : 0u);
    }
    for (uint64_t i = 0; i < rope_n / 4; i++) {
        float f = 1.0f / (float)(i + 1);
        memcpy(h + rope_off + 4 * i, &f, 4);
    }

    char err[300];
    hx_file *f = hx_file_open(s->path, HX_FILE_WRITE | HX_FILE_CREATE | (direct ? HX_FILE_DIRECT : 0), err, sizeof err);
    if (!f) { printf("  cannot create %s: %s\n", s->path, err); hx_aligned_free(h); return 0; }
    int ok = hx_file_pwrite(f, h, (size_t)hdr, 0) == (int64_t)hdr;
    hx_aligned_free(h);
    /* slabs are consecutive: batch them into large writes */
    const uint64_t chunk_cap = (uint64_t)64 << 20;
    uint64_t cap = chunk_cap > s->slab_max ? chunk_cap : s->slab_max;
    uint8_t *buf = (uint8_t *)hx_alloc_large((size_t)cap, 0);
    uint64_t cstart = hdr, clen = 0;
    for (int i = 0; ok && buf && i < ne; i++) {
        if (!s->nbytes[i] || (s->alias && s->alias[i] >= 0)) continue;
        if (clen + s->nbytes[i] > cap) {
            ok = hx_file_pwrite(f, buf, (size_t)clen, cstart) == (int64_t)clen;
            cstart += clen;
            clen = 0;
        }
        uint8_t *p = buf + clen;
        memset(p + s->used[i] / 8 * 8, 0, (size_t)(s->nbytes[i] - s->used[i] / 8 * 8));
        fill_random(p, s->used[i] / 8 * 8, s->seed * 1000003ull + (uint64_t)i);
        s->hash[i] = hash_bytes(p, s->nbytes[i]);
        clen += s->nbytes[i];
    }
    if (ok && buf && clen) ok = hx_file_pwrite(f, buf, (size_t)clen, cstart) == (int64_t)clen;
    if (buf) hx_free_large(buf, (size_t)cap);
    else ok = 0;
    for (int i = 0; i < ne; i++)
        if (s->alias && s->alias[i] >= 0) s->hash[i] = s->hash[s->alias[i]];
    hx_file_close(f);
    return ok;
}

static char g_dir[600];
static char g_tag[40];   /* per-process: concurrent runs (mutation jobs) must not share scratch files */

static void make_spec(spec *s, const char *name, int L, int E, int D, int F, int K, uint64_t seed, int dense_layer,
                      int alias_pct, int mixed) {
    memset(s, 0, sizeof *s);
    s->L = L; s->E = E; s->D = D; s->F = F; s->K = K; s->seed = seed;
    for (int l = 0; l < L; l++) s->lk[l] = (uint8_t)(l != dense_layer);
    int ne = L * E;
    s->dtype = (int *)malloc(sizeof(int) * (size_t)ne);
    s->alias = (int *)malloc(sizeof(int) * (size_t)ne);
    uint64_t r = seed;
    for (int i = 0; i < ne; i++) {
        static const int dts[] = {HEARTH_Q4, HEARTH_Q8, HEARTH_F16, HEARTH_BF16, HEARTH_F32};
        s->dtype[i] = mixed ? dts[rn(&r, 5)] : HEARTH_Q4;
        s->alias[i] = -1;
        if (alias_pct && s->lk[i / E] && i % E > 0 && (int)rn(&r, 100) < alias_pct) {
            int t = i - 1 - (int)rn(&r, (uint32_t)(i % E));
            while (s->alias[t] >= 0) t = s->alias[t];
            s->alias[i] = t;
        }
    }
    if (snprintf(s->path, sizeof s->path, "%s/%s%s", g_dir, g_tag, name) >= (int)sizeof s->path) {
        fprintf(stderr, "test_store: scratch path too long: %s\n", g_dir);
        exit(2);
    }
}

/* --------------------------------------------------------- store helpers */

typedef struct {
    hx_modelfile *mf;
    hx_store *st;
    const spec *sp;
} handle;

static hx_store *open_store(const hx_modelfile *mf, uint64_t cache, int n_io, int direct, int policy, const char *uin,
                            const char *uout, float pin, int warm, const char *const *mirrors, int n_mirrors, char *err,
                            size_t errlen) {
    hx_store_opts o;
    memset(&o, 0, sizeof o);
    o.cache_bytes = cache;
    o.n_io_threads = n_io;
    o.direct_io = direct;
    o.policy = policy;
    o.heat_decay = 0.995f;
    o.usage_in = uin;
    o.usage_out = uout;
    o.pin_fraction = pin;
    o.warm_start = warm;
    o.mirrors = mirrors;
    o.n_mirrors = n_mirrors;
    return hx_store_open(mf, &o, err, errlen);
}

static int moe_key(const spec *sp, uint64_t *r) {
    for (;;) {
        int k = (int)rn(r, (uint32_t)(sp->L * sp->E));
        if (sp->nbytes[k]) return k;
    }
}

static int verify_slab(const spec *sp, int key, const void *p, const char *what) {
    int ok = p && hash_bytes(p, sp->nbytes[key]) == sp->hash[key];
    CHECK(ok, "%s: slab (%d, %d) %s", what, key / sp->E, key % sp->E, p ? "has wrong bytes" : "is NULL");
    return ok;
}

/* non-empty keys in (layer, expert) order */
static int list_keys(const spec *sp, int *keys) {
    int n = 0;
    for (int k = 0; k < sp->L * sp->E; k++)
        if (sp->nbytes[k]) keys[n++] = k;
    return n;
}

/* Polls without calling wait_any (which may drop holds when the store is starved). */
static int wait_resident(hx_store *st, const spec *sp, const int *keys, int n, uint32_t timeout_ms) {
    uint64_t end = hx_now_ns() + (uint64_t)timeout_ms * 1000000;
    for (;;) {
        int all = 1;
        for (int i = 0; i < n && all; i++) all = hx_store_is_resident(st, keys[i] / sp->E, keys[i] % sp->E);
        if (all) return 1;
        if (hx_now_ns() > end) return 0;
        hx_sleep_us(200);
    }
}

static void acq_rel(hx_store *st, const spec *sp, int key, const char *what) {
    const void *p = hx_store_acquire(st, key / sp->E, key % sp->E);
    verify_slab(sp, key, p, what);
    if (p) hx_store_release(st, key / sp->E, key % sp->E);
    tickle();
}

/* ------------------------------------------- read hook (store.c test seam) */

typedef int (*read_hook_fn)(void *ctx, int layer, int expert, int file, int direct);
void hx_store_set_read_hook(hx_store *s, read_hook_fn fn, void *ctx);

#define GATE_LOG 1024
typedef struct {
    hx_mutex mu;
    hx_cond cv;
    int E;
    int closed;                          /* readers block in the hook while set */
    int n, key[GATE_LOG], file[GATE_LOG];/* read attempts in order */
    int direct[2];                       /* attempts on buffered / direct handles */
    int fail_key, fail_file, fail_left;  /* fail attempts on fail_key (file fail_file, or any if -1), fail_left more times (-1: always) */
} gate;

static void gate_init(gate *g, int E) {
    memset(g, 0, sizeof *g);
    hx_mutex_init(&g->mu);
    hx_cond_init(&g->cv);
    g->E = E;
    g->fail_key = g->fail_file = -1;
}

static void gate_destroy(gate *g) {
    hx_cond_destroy(&g->cv);
    hx_mutex_destroy(&g->mu);
}

static int gate_hook(void *ctx, int layer, int expert, int file, int direct) {
    gate *g = (gate *)ctx;
    int key = layer * g->E + expert, fail = 0;
    hx_mutex_lock(&g->mu);
    if (g->n < GATE_LOG) { g->key[g->n] = key; g->file[g->n] = file; }
    g->n++;
    g->direct[direct != 0]++;
    hx_cond_broadcast(&g->cv);
    while (g->closed) hx_cond_wait(&g->cv, &g->mu);
    if (key == g->fail_key && (g->fail_file < 0 || file == g->fail_file) && g->fail_left) {
        fail = 1;
        if (g->fail_left > 0) g->fail_left--;
    }
    hx_mutex_unlock(&g->mu);
    return fail;
}

static void gate_set(gate *g, int closed) {
    hx_mutex_lock(&g->mu);
    g->closed = closed;
    hx_cond_broadcast(&g->cv);
    hx_mutex_unlock(&g->mu);
}

static void gate_fail(gate *g, int key, int file, int times) {
    hx_mutex_lock(&g->mu);
    g->fail_key = key;
    g->fail_file = file;
    g->fail_left = times;
    hx_mutex_unlock(&g->mu);
}

static int gate_count(gate *g) {
    hx_mutex_lock(&g->mu);
    int n = g->n;
    hx_mutex_unlock(&g->mu);
    return n;
}

/* 1 once at least n read attempts have entered the hook, 0 on timeout */
static int gate_wait(gate *g, int n, uint32_t timeout_ms) {
    uint64_t end = hx_now_ns() + (uint64_t)timeout_ms * 1000000;
    hx_mutex_lock(&g->mu);
    while (g->n < n) {
        uint64_t now = hx_now_ns();
        if (now >= end) break;
        hx_cond_timedwait(&g->cv, &g->mu, (uint32_t)((end - now) / 1000 + 1));
    }
    int ok = g->n >= n;
    hx_mutex_unlock(&g->mu);
    return ok;
}

static int gate_reads_of(gate *g, int key) {
    int c = 0;
    hx_mutex_lock(&g->mu);
    for (int i = 0; i < g->n && i < GATE_LOG; i++) c += g->key[i] == key;
    hx_mutex_unlock(&g->mu);
    return c;
}

/* Opens the gate from another thread after a delay, while the test thread is blocked
 * inside the store. */
typedef struct { gate *g; uint32_t delay_us; hx_thread *th; } gate_opener;

static void *gate_opener_main(void *arg) {
    gate_opener *o = (gate_opener *)arg;
    hx_sleep_us(o->delay_us);
    gate_set(o->g, 0);
    return NULL;
}

static int gate_open_later(gate_opener *o, gate *g, uint32_t delay_us) {
    o->g = g;
    o->delay_us = delay_us;
    if (hx_thread_create(&o->th, gate_opener_main, o) == 0) return 1;
    o->th = NULL;
    gate_set(g, 0);
    return 0;
}

static void gate_opener_join(gate_opener *o) {
    if (o->th) hx_thread_join(o->th);
    o->th = NULL;
}

static double timed_wait_any(hx_store *st, uint32_t timeout_us) {
    uint64_t t0 = hx_now_ns();
    hx_store_wait_any(st, timeout_us);
    return (double)(hx_now_ns() - t0) / 1e6;
}

/* --------------------------------------------------- usage file helpers */

static int write_usage(const char *path, int L, int E, uint64_t tokens, const float *heat) {
    size_t n = 24 + 4 * (size_t)L * (size_t)E;
    uint8_t *b = (uint8_t *)calloc(1, n);
    if (!b) return 0;
    put32(b, 0x53555248u); put32(b + 4, 1); put32(b + 8, (uint32_t)L); put32(b + 12, (uint32_t)E); put64(b + 16, tokens);
    for (size_t k = 0; k < (size_t)L * (size_t)E; k++) {
        uint32_t u;
        memcpy(&u, &heat[k], 4);
        put32(b + 24 + 4 * k, u);
    }
    FILE *f = fopen(path, "wb");
    int ok = f && fwrite(b, 1, n, f) == n;
    if (f) ok &= fclose(f) == 0;
    free(b);
    return ok;
}

/* ------------------------------------------------------- random stress */

typedef struct {
    const spec *sp;
    hx_store *st;
    uint64_t seed;
    int ops, budget;            /* budget: max distinct keys held + pending */
    int zipf;
    atomic_int *errors;
} stress_arg;

static int contains(const int *a, int n, int k) {
    for (int i = 0; i < n; i++)
        if (a[i] == k) return 1;
    return 0;
}

static int distinct(const int *a, int n) {
    int d = 0;
    for (int i = 0; i < n; i++) d += !contains(a, i, a[i]);
    return d;
}

static void *stress_main(void *argp) {
    stress_arg *a = (stress_arg *)argp;
    const spec *sp = a->sp;
    hx_store *st = a->st;
    uint64_t r = a->seed;
    int held[64], nh = 0, pend[64], np = 0;
    char what[64];
    for (int op = 0; op < a->ops; op++) {
        int x = (int)rn(&r, 100);
        int k = moe_key(sp, &r);
        if (a->zipf && rn(&r, 2)) k = (k / sp->E) * sp->E + (int)(rn(&r, (uint32_t)sp->E) * rn(&r, 1000) / 1000) % sp->E;
        if (!sp->nbytes[k]) k = moe_key(sp, &r);
        int room = distinct(held, nh) + np < a->budget || contains(held, nh, k) || contains(pend, np, k);
        snprintf(what, sizeof what, "op %d", op);
        if (x < 30) {
            if (room && nh < 60) {
                const void *p = hx_store_try_acquire(st, k / sp->E, k % sp->E);
                if (p) {
                    if (!verify_slab(sp, k, p, what)) atomic_fetch_add(a->errors, 1);
                    held[nh++] = k;
                } else if (!contains(pend, np, k) && !contains(held, nh, k)) {
                    pend[np++] = k;
                }
            }
        } else if (x < 50) {
            int i = np ? (int)rn(&r, (uint32_t)np) : -1;
            if (i < 0 && room && nh < 60) { pend[np++] = k; i = np - 1; }
            if (i >= 0 && nh < 60) {
                k = pend[i];
                const void *p = hx_store_acquire(st, k / sp->E, k % sp->E);
                if (!verify_slab(sp, k, p, what)) atomic_fetch_add(a->errors, 1);
                if (p) held[nh++] = k;
                pend[i] = pend[--np];
            }
        } else if (x < 70) {
            if (nh) {
                int i = (int)rn(&r, (uint32_t)nh);
                k = held[i];
                CHECK(hx_store_is_resident(st, k / sp->E, k % sp->E), "held slab not resident");
                /* the slot must still hold the same bytes: acquire again (hit) and compare */
                const void *p = hx_store_try_acquire(st, k / sp->E, k % sp->E);
                CHECK(p != NULL, "re-acquire of a held slab missed");
                if (p) {
                    if (!verify_slab(sp, k, p, "before release")) atomic_fetch_add(a->errors, 1);
                    hx_store_release(st, k / sp->E, k % sp->E);
                }
                hx_store_release(st, k / sp->E, k % sp->E);
                held[i] = held[--nh];
            }
        } else if (x < 80) {
            int ids[16], n = 1 + (int)rn(&r, 16), layer = (int)rn(&r, (uint32_t)sp->L);
            for (int i = 0; i < n; i++) ids[i] = (int)rn(&r, (uint32_t)sp->E + 1) - (rn(&r, 50) == 0);   /* a few invalid ids */
            hx_store_prefetch(st, layer, ids, n);
        } else if (x < 83) {
            hx_store_tick(st);
        } else if (x < 88) {
            hx_store_wait_any(st, 200);
        } else if (x < 93) {
            for (int i = 0; i < nh; i++)
                CHECK(hx_store_is_resident(st, held[i] / sp->E, held[i] % sp->E), "held slab evicted");
        } else {
            hx_store_stats s;
            hx_store_get_stats(st, &s);
            CHECK(s.resident <= s.n_slots && s.pinned <= s.n_slots, "stats: resident/pinned exceed slots");
            CHECK(s.prefetch_used + s.prefetch_wasted <= s.prefetch_issued + (uint64_t)s.n_slots,
                  "stats: prefetch used+wasted > issued");
        }
        tickle();
    }
    for (int i = 0; i < np; i++) {
        const void *p = hx_store_acquire(st, pend[i] / sp->E, pend[i] % sp->E);
        if (!verify_slab(sp, pend[i], p, "final")) atomic_fetch_add(a->errors, 1);
        if (p) hx_store_release(st, pend[i] / sp->E, pend[i] % sp->E);
    }
    for (int i = 0; i < nh; i++) hx_store_release(st, held[i] / sp->E, held[i] % sp->E);
    return NULL;
}

static void stress(const spec *sp, hx_modelfile *mf, int quick, uint64_t seed) {
    static const int ios[] = {1, 2, 3, 4, 8, 16};
    int configs = 0, n_ops = 0;
    uint64_t r = seed;
    atomic_int errors;
    atomic_init(&errors, 0);
    for (int pol = 0; pol < 2; pol++)
        for (size_t ii = 0; ii < sizeof ios / sizeof ios[0]; ii++)
            for (int cache = 0; cache < 4; cache++)
                for (int direct = 0; direct < 2; direct++) {
                    if (quick && rn(&r, 4)) continue;
                    int n_io = ios[ii];
                    int min_slots = 2 * sp->K + n_io + 2;
                    uint64_t slots = cache == 0 ? 0 : cache == 1 ? (uint64_t)min_slots + 3 : cache == 2 ? 40 : 1000;
                    char err[400];
                    g_phase = "stress open";
                    hx_store *st = open_store(mf, slots * mf->slab_bytes_max, n_io, direct, pol, NULL, NULL, 0, 0, NULL, 0,
                                              err, sizeof err);
                    CHECK(st != NULL, "open: %s", err);
                    if (!st) continue;
                    hx_store_stats s0;
                    hx_store_get_stats(st, &s0);
                    if (cache == 0) CHECK(s0.n_slots == min_slots, "minimum slots %d, want %d", s0.n_slots, min_slots);
                    if (cache == 3) {   /* everything fits: one slot per non-empty expert, never more */
                        int n_ne = 0;
                        for (int k = 0; k < sp->L * sp->E; k++) n_ne += sp->nbytes[k] != 0;
                        CHECK(s0.n_slots == n_ne, "cache for all experts: %d slots, want %d", s0.n_slots, n_ne);
                    }
                    CHECK(s0.slot_bytes == mf->slab_bytes_max, "slot_bytes");
                    int callers = (int)rn(&r, 3) == 0 ? 1 + (int)rn(&r, 4) : 1;
                    int budget = 2 * sp->K / callers;
                    stress_arg args[4];
                    hx_thread *th[4];
                    int nt = 0;
                    g_phase = "stress run";
                    for (int c = 0; c < callers; c++) {
                        args[c].sp = sp; args[c].st = st; args[c].seed = splitmix(&r); args[c].errors = &errors;
                        args[c].ops = quick ? 800 : 2500; args[c].budget = budget < 1 ? 1 : budget; args[c].zipf = c & 1;
                        if (c > 0 && hx_thread_create(&th[nt], stress_main, &args[c]) == 0) nt++;
                    }
                    stress_main(&args[0]);
                    for (int i = 0; i < nt; i++) hx_thread_join(th[i]);
                    hx_store_stats s;
                    hx_store_get_stats(st, &s);
                    CHECK(s.reads >= s.misses / 2 && s.bytes_read >= s.reads * 4096, "stats: reads %llu misses %llu",
                          (unsigned long long)s.reads, (unsigned long long)s.misses);
                    g_phase = "stress close";
                    hx_store_close(st);
                    configs++;
                    n_ops += args[0].ops * callers;
                    tickle();
                }
    CHECK(atomic_load(&errors) == 0, "stress: %d slabs with wrong bytes", atomic_load(&errors));
    printf("  stress: %d store configurations (LRU/LFU x 1..16 readers x 4 cache sizes x direct/buffered, 1-4 callers), "
           "%d operations\n", configs, n_ops);
}

/* ----------------------------------------------- minimum slots, top-k burst */

/* Per token and MoE layer: hint the next layer, demand the whole top-k at once,
 * then acquire it while the previous layer's top-k is still held. */
static void topk_bursts(hx_store *st, const spec *sp, int tokens, uint64_t r, const char *what) {
    int prev[64], np = 0, prev_layer = -1;
    for (int t = 0; t < tokens; t++) {
        for (int l = 0; l < sp->L; l++) {
            if (!sp->lk[l]) continue;
            int cur[64], nc = 0;
            while (nc < sp->K) {
                int e = (int)rn(&r, (uint32_t)sp->E);
                if (!contains(cur, nc, e)) cur[nc++] = e;
            }
            int nxt = (l + 1) % sp->L;
            hx_store_prefetch(st, nxt, cur, nc);       /* noise: hints for the next layer */
            for (int i = 0; i < nc; i++) {             /* all top-k demanded at once */
                const void *p = hx_store_try_acquire(st, l, cur[i]);
                if (p) hx_store_release(st, l, cur[i]);
            }
            for (int i = 0; i < nc; i++) {             /* acquire all while the previous layer is held */
                const void *p = hx_store_acquire(st, l, cur[i]);
                verify_slab(sp, l * sp->E + cur[i], p, what);
                tickle();
            }
            for (int i = 0; i < np; i++) hx_store_release(st, prev_layer, prev[i]);
            memcpy(prev, cur, sizeof(int) * (size_t)nc);
            np = nc;
            prev_layer = l;
        }
        hx_store_tick(st);
    }
    for (int i = 0; i < np; i++) hx_store_release(st, prev_layer, prev[i]);
}

static void min_slots_burst(const spec *sp, hx_modelfile *mf, int quick) {
    static const int ios[] = {1, 2, 4, 16};
    int tokens = quick ? 15 : 60;
    for (size_t ii = 0; ii < sizeof ios / sizeof ios[0]; ii++)
        for (int pol = 0; pol < 2; pol++) {
            char err[400];
            g_phase = "min-slots burst";
            hx_store *st = open_store(mf, 0, ios[ii], (int)(ii & 1), pol, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
            CHECK(st != NULL, "open: %s", err);
            if (!st) continue;
            topk_bursts(st, sp, tokens, 99 + ii, "min-slots");
            hx_store_stats s;
            hx_store_get_stats(st, &s);
            CHECK(s.n_slots == 2 * sp->K + ios[ii] + 2, "min-slots: %d slots", s.n_slots);
            hx_store_close(st);
        }
    printf("  minimum slot count: top-%d bursts with the previous layer held, 1/2/4/16 readers, LRU/LFU: no deadlock\n",
           sp->K);
}

/* More demands than slots (a batch union), then blocking acquires in the
 * opposite order: the slabs loaded first sit held for a caller that is blocked
 * on a later one. The store must give up those holds rather than deadlock. */
static void overcommit(const spec *sp, hx_modelfile *mf) {
    static const int ios[] = {1, 3, 8};
    char err[400];
    g_phase = "overcommitted demands";
    int ne = sp->L * sp->E, n = 0;
    int *keys = (int *)malloc(sizeof(int) * (size_t)ne);
    for (int k = 0; k < ne; k++)
        if (sp->nbytes[k]) keys[n++] = k;
    for (size_t ii = 0; ii < sizeof ios / sizeof ios[0]; ii++)
        for (int pol = 0; pol < 2; pol++) {
            hx_store *st = open_store(mf, 0, ios[ii], 0, pol, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
            CHECK(st != NULL, "open: %s", err);
            if (!st) continue;
            for (int i = 0; i < n; i++) {
                const void *p = hx_store_try_acquire(st, keys[i] / sp->E, keys[i] % sp->E);
                if (p) hx_store_release(st, keys[i] / sp->E, keys[i] % sp->E);
            }
            for (int i = n - 1; i >= 0; i--) {
                const void *p = hx_store_acquire(st, keys[i] / sp->E, keys[i] % sp->E);
                verify_slab(sp, keys[i], p, "overcommit");
                if (p) hx_store_release(st, keys[i] / sp->E, keys[i] % sp->E);
                tickle();
            }
            hx_store_close(st);

            /* fresh store, same burst; the token ends before anything is collected and the caller
             * then busy-polls the first expert still queued: the tick must restart the readers */
            st = open_store(mf, 0, ios[ii], 0, pol, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
            if (!st) continue;
            for (int i = 0; i < n; i++) CHECK(!hx_store_try_acquire(st, keys[i] / sp->E, keys[i] % sp->E), "cold hit");
            int head = -1;
            for (int settled = 0, last = -1; settled < 20; hx_sleep_us(2000)) {   /* until the readers are starved */
                int r = 0;
                while (r < n && hx_store_is_resident(st, keys[r] / sp->E, keys[r] % sp->E)) r++;
                settled = r == last ? settled + 1 : 0;
                last = r;
                head = r;
            }
            CHECK(head > 0 && head < n, "overcommit: %d of %d served before starving", head, n);
            hx_store_tick(st);
            if (head > 0 && head < n) {
                const void *p;
                while (!(p = hx_store_try_acquire(st, keys[head] / sp->E, keys[head] % sp->E))) hx_sleep_us(500);
                verify_slab(sp, keys[head], p, "overcommit after tick");
                hx_store_release(st, keys[head] / sp->E, keys[head] % sp->E);
            }
            hx_store_close(st);
            tickle();

            /* fresh store, same burst, no tick: polling only the last expert with wait_any must
             * still get it (a starved wait_any gives up holds nobody is collecting) */
            st = open_store(mf, 0, ios[ii], 0, pol, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
            if (!st) continue;
            for (int i = 0; i < n; i++) hx_store_try_acquire(st, keys[i] / sp->E, keys[i] % sp->E);
            {
                const void *p;
                while (!(p = hx_store_try_acquire(st, keys[n - 1] / sp->E, keys[n - 1] % sp->E)))
                    hx_store_wait_any(st, 5000);
                verify_slab(sp, keys[n - 1], p, "overcommit via wait_any");
                hx_store_release(st, keys[n - 1] / sp->E, keys[n - 1] % sp->E);
            }
            hx_store_close(st);
        }
    free(keys);
    printf("  overcommit: %d experts demanded at once into the minimum cache, then collected in reverse order / polled "
           "after a tick / polled with wait_any only: no deadlock\n", n);
}

/* Cache policy details at the minimum slot count with one reader (deterministic order):
 * demanded slabs are not evicted by the rest of their burst; prefetch can evict cold
 * slots in a full cache; demand evicts unprotected slabs before prefetched ones. */
static void policy_details(const spec *sp, hx_modelfile *mf) {
    char err[400];
    g_phase = "policy details";
    for (int pol = 0; pol < 2; pol++) {
        hx_store *st = open_store(mf, 0, 1, 0, pol, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
        CHECK(st != NULL, "open: %s", err);
        if (!st) continue;
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int slots = s.n_slots, nu = 0;
        /* fill every slot with an expert used once (MoE layers 0-3) */
        for (int k = 0; k < sp->L * sp->E && nu < slots; k++) {
            if (!sp->nbytes[k] || k / sp->E > 3) continue;
            const void *p = hx_store_acquire(st, k / sp->E, k % sp->E);
            if (p) { hx_store_release(st, k / sp->E, k % sp->E); nu++; }
        }
        int fresh[16], nf = 0;   /* experts never touched, from the last layer */
        for (int e = 0; e < sp->E && nf < 16; e++) {
            int k = (sp->L - 1) * sp->E + e;
            if (sp->nbytes[k] && sp->alias[k] < 0) fresh[nf++] = e;
        }
        CHECK(nu == slots && nf >= 2 * sp->K, "policy details setup (%d/%d, %d)", nu, slots, nf);
        int L1 = sp->L - 1;

        /* burst: K demands at once; each must be read exactly once */
        hx_store_reset_stats(st);
        for (int i = 0; i < sp->K; i++) CHECK(!hx_store_try_acquire(st, L1, fresh[i]), "burst: unexpected hit");
        for (int w = 0; w < 400; w++) {
            int all = 1;
            for (int i = 0; i < sp->K; i++) all &= hx_store_is_resident(st, L1, fresh[i]);
            if (all) break;
            hx_store_wait_any(st, 5000);
        }
        for (int i = 0; i < sp->K; i++) {
            const void *p = hx_store_acquire(st, L1, fresh[i]);
            verify_slab(sp, L1 * sp->E + fresh[i], p, "burst");
            if (p) hx_store_release(st, L1, fresh[i]);
        }
        hx_store_get_stats(st, &s);
        CHECK(s.misses == (uint64_t)sp->K && s.reads == (uint64_t)sp->K,
              "policy %d: a burst of %d demands caused %llu reads (demanded slabs evicted before collection?)", pol, sp->K,
              (unsigned long long)s.reads);
        hx_store_tick(st);

        /* prefetch into the full cache, then demand others: prefetched slabs survive */
        hx_store_reset_stats(st);
        int pf[3] = {fresh[sp->K], fresh[sp->K + 1], fresh[sp->K + 2]};
        hx_store_prefetch(st, L1, pf, 3);
        for (int w = 0; w < 400; w++) {
            if (hx_store_is_resident(st, L1, pf[0]) && hx_store_is_resident(st, L1, pf[1]) && hx_store_is_resident(st, L1, pf[2]))
                break;
            hx_store_wait_any(st, 5000);
        }
        hx_store_get_stats(st, &s);
        CHECK(s.prefetch_issued == 3 && s.reads == 3, "policy %d: prefetch into a full cache issued %llu", pol,
              (unsigned long long)s.prefetch_issued);
        for (int i = sp->K + 3; i < sp->K + 6 && i < nf; i++) {
            const void *p = hx_store_acquire(st, L1, fresh[i]);
            if (p) hx_store_release(st, L1, fresh[i]);
        }
        int kept = 1;
        for (int i = 0; i < 3; i++) kept &= hx_store_is_resident(st, L1, pf[i]);
        hx_store_get_stats(st, &s);
        CHECK(kept && s.prefetch_wasted == 0, "policy %d: demand evicted a protected prefetch (wasted %llu)", pol,
              (unsigned long long)s.prefetch_wasted);
        hx_store_close(st);
        tickle();
    }
}

/* --------------------------------------------------- usage file & pinning */

static int read_usage(const char *path, int L, int E, uint64_t *tokens, float *heat) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint8_t hdr[24];
    int ok = fread(hdr, 1, 24, f) == 24;
    uint32_t v[4];
    for (int i = 0; i < 4; i++) memcpy(&v[i], hdr + 4 * i, 4);
    ok = ok && v[0] == 0x53555248u && v[1] == 1 && v[2] == (uint32_t)L && v[3] == (uint32_t)E;
    memcpy(tokens, hdr + 16, 8);
    ok = ok && fread(heat, 4, (size_t)(L * E), f) == (size_t)(L * E);
    uint8_t extra;
    ok = ok && fread(&extra, 1, 1, f) == 0;
    fclose(f);
    return ok;
}

static void usage_and_pins(const spec *sp, hx_modelfile *mf) {
    char upath[800], upath2[800], err[400];
    int ne = sp->L * sp->E;
    snprintf(upath, sizeof upath, "%s.usage", sp->path);
    snprintf(upath2, sizeof upath2, "%s.usage2", sp->path);
    remove(upath);
    double *tally = (double *)calloc((size_t)ne, sizeof(double));
    float *heat = (float *)calloc((size_t)ne, sizeof(float)), *heat2 = (float *)calloc((size_t)ne, sizeof(float));
    uint64_t r = 7;
    g_phase = "usage write";

    /* missing usage_in is fine (first run) */
    hx_store *st = open_store(mf, 0, 2, 0, HEARTH_POLICY_LFU, upath, upath, 0.5f, 1, NULL, 0, err, sizeof err);
    CHECK(st != NULL, "open with a missing usage_in: %s", err);
    int ticks = 120;
    for (int t = 0; st && t < ticks; t++) {
        for (int l = 0; l < sp->L; l++) {
            if (!sp->lk[l]) continue;
            for (int j = 0; j < sp->K; j++) {
                int e = (int)(rn(&r, (uint32_t)sp->E) * rn(&r, 1000) / 1000);   /* skewed */
                const void *p = hx_store_acquire(st, l, e);
                verify_slab(sp, l * sp->E + e, p, "usage run");
                if (p) { tally[l * sp->E + e] += 1; hx_store_release(st, l, e); }
            }
        }
        hx_store_tick(st);
        tickle();
    }
    const float *h = st ? hx_store_heat(st) : NULL;
    int same = h != NULL;
    for (int k = 0; h && k < ne; k++) same &= h[k] == (float)tally[k];
    CHECK(same, "hx_store_heat differs from the test's tally");
    hx_store_close(st);
    uint64_t tok = 0;
    CHECK(read_usage(upath, sp->L, sp->E, &tok, heat), "usage file unreadable or wrong header/size");
    CHECK(tok == (uint64_t)ticks, "usage tokens_observed %llu want %d", (unsigned long long)tok, ticks);
    same = 1;
    for (int k = 0; k < ne; k++) same &= heat[k] == (float)tally[k];
    CHECK(same, "usage heat differs from the activation tally");

    /* reopen with pins: the top floor(0.5*slots) (capped to keep the minimum unpinned) by count */
    g_phase = "pins";
    uint64_t slots = (uint64_t)(2 * sp->K + 2 + 2) + 12;
    st = open_store(mf, slots * mf->slab_bytes_max, 2, 1, HEARTH_POLICY_LFU, upath, upath2, 0.5f, 0, NULL, 0, err, sizeof err);
    CHECK(st != NULL, "open with usage_in: %s", err);
    if (st) {
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int n_nz = 0;
        for (int k = 0; k < ne; k++) n_nz += sp->nbytes[k] && tally[k] > 0;
        int want_pin = (int)floor(0.5 * s.n_slots);
        if (want_pin > s.n_slots - (2 * sp->K + 2 + 2)) want_pin = s.n_slots - (2 * sp->K + 2 + 2);
        if (want_pin > n_nz) want_pin = n_nz;
        CHECK(s.pinned == want_pin && s.resident == want_pin, "pinned %d resident %d, want %d", s.pinned, s.resident, want_pin);
        /* the pinned set is the top want_pin keys by (count desc, key asc) */
        int *rank = (int *)malloc(sizeof(int) * (size_t)ne), nr = 0;
        for (int k = 0; k < ne; k++)
            if (sp->nbytes[k] && tally[k] > 0) rank[nr++] = k;
        for (int i = 0; i < nr; i++)
            for (int j = i + 1; j < nr; j++)
                if (tally[rank[j]] > tally[rank[i]] || (tally[rank[j]] == tally[rank[i]] && rank[j] < rank[i])) {
                    int t = rank[i]; rank[i] = rank[j]; rank[j] = t;
                }
        for (int i = 0; i < want_pin; i++)
            CHECK(hx_store_is_resident(st, rank[i] / sp->E, rank[i] % sp->E), "hot expert rank %d not pinned", i);
        const float *h0 = hx_store_heat(st);
        same = 1;
        for (int k = 0; k < ne; k++) same &= h0[k] == heat[k];
        CHECK(same, "hx_store_heat after open differs from the usage profile");
        /* heavy churn with a cache barely above the minimum: pins survive */
        for (int it = 0; it < 4000; it++) {
            int k = moe_key(sp, &r);
            const void *p = hx_store_acquire(st, k / sp->E, k % sp->E);
            verify_slab(sp, k, p, "pin churn");
            if (p) hx_store_release(st, k / sp->E, k % sp->E);
            if (it % 50 == 0) {
                int ids[8];
                for (int i = 0; i < 8; i++) ids[i] = (int)rn(&r, (uint32_t)sp->E);
                hx_store_prefetch(st, (int)rn(&r, (uint32_t)sp->L), ids, 8);
                hx_store_tick(st);
            }
            tickle();
        }
        int kept = 1;
        for (int i = 0; i < want_pin; i++) kept &= hx_store_is_resident(st, rank[i] / sp->E, rank[i] % sp->E);
        CHECK(kept, "a pinned expert was evicted");
        hx_store_get_stats(st, &s);
        CHECK(s.pinned == want_pin && s.evictions > 0, "pinned count changed or no evictions happened");
        hx_store_close(st);
        uint64_t tok2 = 0;
        CHECK(read_usage(upath2, sp->L, sp->E, &tok2, heat2), "second usage file");
        CHECK(tok2 == (uint64_t)ticks + 80, "tokens_observed accumulates: %llu", (unsigned long long)tok2);
        double sum1 = 0, sum2 = 0;
        for (int k = 0; k < ne; k++) { sum1 += heat[k]; sum2 += heat2[k]; }
        CHECK(fabs(sum2 - sum1 - 4000.0) < 0.5, "counts accumulate on top of the seed (%g -> %g)", sum1, sum2);
        free(rank);
    }

    /* warm start fills every unpinned slot with the next-hottest experts */
    g_phase = "warm start";
    st = open_store(mf, 20 * mf->slab_bytes_max, 3, 0, HEARTH_POLICY_LFU, upath, NULL, 0.25f, 1, NULL, 0, err, sizeof err);
    CHECK(st != NULL, "open with warm_start: %s", err);
    if (st) {
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int n_nz = 0;
        for (int k = 0; k < ne; k++) n_nz += sp->nbytes[k] && tally[k] > 0;
        int want = n_nz < s.n_slots ? n_nz : s.n_slots;
        CHECK(s.resident == want && s.reads == (uint64_t)want, "warm start: resident %d reads %llu want %d", s.resident,
              (unsigned long long)s.reads, want);
        hx_store_close(st);
    }

    g_phase = "bad usage";
    uint8_t *raw = (uint8_t *)malloc(24 + 4 * (size_t)ne + 8);
    FILE *f = fopen(upath, "rb");
    size_t n = f ? fread(raw, 1, 24 + 4 * (size_t)ne, f) : 0;
    if (f) fclose(f);
    CHECK(n == 24 + 4 * (size_t)ne, "re-read usage");

    /* equal heat everywhere: ties are broken by (layer, expert) order, deterministically */
    {
        uint8_t *eq = (uint8_t *)malloc(n);
        memcpy(eq, raw, n);
        float one = 1.0f;
        for (int k = 0; k < ne; k++) memcpy(eq + 24 + 4 * (size_t)k, &one, 4);
        f = fopen(upath2, "wb");
        fwrite(eq, 1, n, f);
        fclose(f);
        free(eq);
        st = open_store(mf, 24 * mf->slab_bytes_max, 1, 0, HEARTH_POLICY_LFU, upath2, NULL, 0.5f, 0, NULL, 0, err, sizeof err);
        CHECK(st != NULL, "open with a flat profile: %s", err);
        if (st) {
            hx_store_stats s;
            hx_store_get_stats(st, &s);
            int seen = 0, ok = s.pinned > 0;
            for (int k = 0; k < ne; k++) {
                if (!sp->nbytes[k]) continue;
                ok &= hx_store_is_resident(st, k / sp->E, k % sp->E) == (seen < s.pinned);
                seen++;
            }
            CHECK(ok, "flat profile: the %d pinned experts are not the first ones in (layer, expert) order", s.pinned);
            hx_store_close(st);
        }
    }

    /* broken or mismatched profiles are ignored (open succeeds, nothing seeded or pinned) and,
     * when usage_out names the same file, replaced by a valid profile on close */
    struct { const char *what; size_t at; uint32_t v; int len_delta; } bad[] = {
        {"bad magic", 0, 0x12345678u, 0}, {"bad version", 4, 2, 0}, {"wrong n_layers", 8, (uint32_t)sp->L + 1, 0},
        {"wrong n_experts", 12, (uint32_t)sp->E - 1, 0}, {"NaN heat", 24, 0x7FC00000u, 0}, {"negative heat", 28, 0xBF800000u, 0},
        {"infinite heat", 24 + 4 * (size_t)(ne - 1), 0x7F800000u, 0},
        {"truncated", 0, 0x53555248u, -4}, {"trailing bytes", 0, 0x53555248u, 4},
    };
    for (size_t b = 0; b < sizeof bad / sizeof bad[0]; b++) {
        uint8_t *d = (uint8_t *)malloc(n + 8);
        memcpy(d, raw, n);
        memset(d + n, 0, 8);
        memcpy(d + bad[b].at, &bad[b].v, 4);
        f = fopen(upath2, "wb");
        fwrite(d, 1, (size_t)((int64_t)n + bad[b].len_delta), f);
        fclose(f);
        free(d);
        err[0] = 0;
        st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LFU, upath2, upath2, 0.5f, 1, NULL, 0, err, sizeof err);
        CHECK(st != NULL, "usage '%s' blocked open: %s", bad[b].what, err);
        if (!st) continue;
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        const float *h0 = hx_store_heat(st);
        int zero = 1;
        for (int k = 0; k < ne; k++) zero &= h0[k] == 0.0f;
        CHECK(s.pinned == 0 && s.resident == 0 && zero, "usage '%s': partially applied (pinned %d, resident %d)",
              bad[b].what, s.pinned, s.resident);
        hx_store_close(st);
        uint64_t tok3 = 1;
        CHECK(read_usage(upath2, sp->L, sp->E, &tok3, heat2) && tok3 == 0, "usage '%s' not replaced on close", bad[b].what);
    }
    /* an unreadable profile path (a directory) is ignored too */
    st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LFU, g_dir, NULL, 0.5f, 0, NULL, 0, err, sizeof err);
    CHECK(st != NULL, "a directory as usage_in blocked open: %s", err);
    hx_store_close(st);
    free(raw);
    remove(upath);
    remove(upath2);
    free(tally);
    free(heat);
    free(heat2);
    printf("  usage profile: round trip exact, pins = top int(pin_fraction*slots) by heat and never evicted, "
           "warm start fills the cache, corrupt or mismatched profiles ignored and replaced\n");
}

/* ----------------------------------------------------------------- mirrors */

static int copy_file(const char *from, const char *to, int64_t patch_at, uint8_t patch_xor) {
    FILE *a = fopen(from, "rb"), *b = fopen(to, "wb");
    int ok = a && b;
    static uint8_t buf[1 << 16];
    int64_t pos = 0;
    size_t n;
    while (ok && (n = fread(buf, 1, sizeof buf, a)) > 0) {
        if (patch_at >= pos && patch_at < pos + (int64_t)n) buf[patch_at - pos] ^= patch_xor;
        ok = fwrite(buf, 1, n, b) == n;
        pos += (int64_t)n;
    }
    if (a) fclose(a);
    if (b) fclose(b);
    return ok;
}

static void mirrors(const spec *sp, hx_modelfile *mf) {
    char m1[800], m2[800], err[400];
    snprintf(m1, sizeof m1, "%s.mirror1", sp->path);
    snprintf(m2, sizeof m2, "%s.mirror2", sp->path);
    g_phase = "mirrors";
    int ne = sp->L * sp->E, first = -1, victim = -1;
    for (int k = 0; k < ne; k++)
        if (sp->nbytes[k] && sp->alias[k] < 0) { if (first < 0) first = k; else if (sp->nbytes[k] > 4096) victim = k; }

    /* identical mirrors: correct bytes from every combination */
    CHECK(copy_file(sp->path, m1, -1, 0) && copy_file(sp->path, m2, -1, 0), "copy mirrors");
    const char *ms[2] = {m1, m2};
    for (int nm = 1; nm <= 2; nm++) {
        hx_store *st = open_store(mf, 0, 3, nm == 2, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, ms, nm, err, sizeof err);
        CHECK(st != NULL, "open with %d mirrors: %s", nm, err);
        uint64_t r = 5;
        for (int it = 0; st && it < 1500; it++) {
            int k = moe_key(sp, &r);
            const void *p = hx_store_acquire(st, k / sp->E, k % sp->E);
            verify_slab(sp, k, p, "mirror read");
            if (p) hx_store_release(st, k / sp->E, k % sp->E);
            tickle();
        }
        hx_store_close(st);
    }

    /* reads are really spread: a mirror whose copy of one slab differs past the first
     * page (invisible to the open-time spot check) must sometimes serve that slab */
    CHECK(victim >= 0, "need a multi-page slab");
    if (victim >= 0) {
        CHECK(copy_file(sp->path, m1, (int64_t)(sp->off[victim] + sp->nbytes[victim] - 8), 0x5A), "patched mirror");
        hx_store *st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LRU, NULL, NULL, 0, 0, ms, 1, err, sizeof err);
        CHECK(st != NULL, "open with patched mirror: %s", err);
        int from_primary = 0, from_mirror = 0;
        uint64_t r = 11;
        for (int it = 0; st && it < 60; it++) {
            const void *p = hx_store_acquire(st, victim / sp->E, victim % sp->E);
            if (p && hash_bytes(p, sp->nbytes[victim]) == sp->hash[victim]) from_primary++;
            else if (p) from_mirror++;
            if (p) hx_store_release(st, victim / sp->E, victim % sp->E);
            /* LRU with the minimum 2K+3 slots: that many other distinct experts evict it;
             * a varying count makes successive reads of it land on either file */
            int seen[64], others = 0, n_other = 2 * sp->K + 3 + (int)rn(&r, 4);
            while (others < n_other) {
                int k = moe_key(sp, &r);
                if (k == victim || sp->off[k] == sp->off[victim] || contains(seen, others, k)) continue;
                const void *q = hx_store_acquire(st, k / sp->E, k % sp->E);
                if (q) hx_store_release(st, k / sp->E, k % sp->E);
                seen[others++] = k;
            }
            CHECK(!hx_store_is_resident(st, victim / sp->E, victim % sp->E), "victim should have been evicted");
            tickle();
        }
        CHECK(from_primary > 5 && from_mirror > 5, "reads not spread over mirrors (primary %d, mirror %d)", from_primary,
              from_mirror);
        hx_store_close(st);
        printf("  mirrors: 1 and 2 identical copies verified; one slab read %d times from the primary, %d from the mirror\n",
               from_primary, from_mirror);
    }

    /* mirrors that are not byte-identical are refused at open: the first slab, and one
     * in the middle of the file (the open-time check samples slabs across the directory) */
    CHECK(copy_file(sp->path, m1, (int64_t)sp->off[first] + 100, 0x01), "patched mirror (first page)");
    hx_store *st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, ms, 1, err, sizeof err);
    CHECK(st == NULL && strstr(err, "differs"), "mirror with different slab data accepted (%s)", err);
    hx_store_close(st);
    int mid = ne / 2;
    while (mid < ne && !sp->nbytes[mid]) mid++;
    CHECK(copy_file(sp->path, m1, (int64_t)sp->off[mid] + 8, 0x01), "patched mirror (middle slab)");
    st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, ms, 1, err, sizeof err);
    CHECK(st == NULL && strstr(err, "differs"), "mirror with a different middle slab accepted (%s)", err);
    hx_store_close(st);
    /* ... and copies that differ only inside the metadata or either directory */
    {
        uint8_t pre[64];
        FILE *pf = fopen(sp->path, "rb");
        CHECK(pf && fread(pre, 1, 64, pf) == 64, "read preamble");
        if (pf) fclose(pf);
        uint64_t meta_off, tdir_off, edir_off;
        memcpy(&meta_off, pre + 8, 8);
        memcpy(&tdir_off, pre + 24, 8);
        memcpy(&edir_off, pre + 40, 8);
        const struct { const char *what; uint64_t at; } where[] = {
            {"metadata", meta_off + 3}, {"tensor directory", tdir_off + 100}, {"expert directory", edir_off + 20}};
        for (int w = 0; w < 3; w++) {
            CHECK(copy_file(sp->path, m1, (int64_t)where[w].at, 0x04), "patched mirror (%s)", where[w].what);
            st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, ms, 1, err, sizeof err);
            CHECK(st == NULL && strstr(err, "differ"), "mirror with a different %s accepted", where[w].what);
            hx_store_close(st);
        }
    }
    const char *empty[1] = {""};
    err[0] = 0;
    st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, empty, 1, err, sizeof err);
    CHECK(st == NULL && err[0], "empty mirror path accepted");
    hx_store_close(st);
    CHECK(copy_file(sp->path, m1, 20, 0x01), "patched mirror (preamble)");
    st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, ms, 1, err, sizeof err);
    CHECK(st == NULL && err[0], "mirror with a different preamble accepted");
    hx_store_close(st);
    FILE *f = fopen(m1, "ab");
    if (f) { fputc(0, f); fclose(f); }
    CHECK(copy_file(sp->path, m2, -1, 0), "copy");
    f = fopen(m2, "ab");
    if (f) { fputc(0, f); fclose(f); }
    st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, ms + 1, 1, err, sizeof err);
    CHECK(st == NULL && strstr(err, "size"), "mirror of a different size accepted (%s)", err);
    hx_store_close(st);
    char missing[820];
    snprintf(missing, sizeof missing, "%s.no_such_mirror", sp->path);
    const char *mm[1] = {missing};
    st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, mm, 1, err, sizeof err);
    CHECK(st == NULL && err[0], "missing mirror accepted");
    hx_store_close(st);
    remove(m1);
    remove(m2);
}

/* ------------------------------------------------------- direct vs buffered */

/* Same bytes either way, and direct_io really selects unbuffered handles whenever the
 * platform grants them for this file (the read hook sees each handle's mode). */
static void direct_vs_buffered(const spec *sp, hx_modelfile *mf) {
    int ne = sp->L * sp->E;
    uint64_t *h[2];
    int modes[2][2];
    char err[400];
    g_phase = "direct vs buffered";
    hx_file *probe = hx_file_open(sp->path, HX_FILE_READ | HX_FILE_DIRECT, err, sizeof err);
    int can_direct = probe && hx_file_is_direct(probe);
    if (probe) hx_file_close(probe);
    for (int d = 0; d < 2; d++) {
        gate g;
        gate_init(&g, sp->E);
        h[d] = (uint64_t *)calloc((size_t)ne, 8);
        hx_store *st = open_store(mf, 0, 4, d, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
        CHECK(st != NULL, "open: %s", err);
        if (st) hx_store_set_read_hook(st, gate_hook, &g);
        for (int k = 0; st && k < ne; k++) {
            if (!sp->nbytes[k]) continue;
            const void *p = hx_store_acquire(st, k / sp->E, k % sp->E);
            if (p) { h[d][k] = hash_bytes(p, sp->nbytes[k]); hx_store_release(st, k / sp->E, k % sp->E); }
            tickle();
        }
        hx_store_close(st);
        modes[d][0] = g.direct[0];
        modes[d][1] = g.direct[1];
        gate_destroy(&g);
    }
    int same = 1, right = 1;
    for (int k = 0; k < ne; k++) {
        same &= h[0][k] == h[1][k];
        right &= !sp->nbytes[k] || h[0][k] == sp->hash[k];
    }
    CHECK(same && right, "direct and buffered reads differ (%d) or are wrong (%d)", !same, !right);
    CHECK(modes[0][0] > 0 && modes[0][1] == 0, "direct_io 0: %d unbuffered reads", modes[0][1]);
    CHECK(modes[1][can_direct] > 0 && modes[1][!can_direct] == 0,
          "direct_io 1 (platform grants unbuffered I/O here: %d): %d buffered and %d unbuffered reads", can_direct,
          modes[1][0], modes[1][1]);
    printf("  direct vs buffered: identical bytes; unbuffered handles %s for this file\n",
           can_direct ? "granted and used" : "not granted by the platform (buffered fallback)");
    free(h[0]);
    free(h[1]);
}

/* ------------------------------------------------------------- misc API */

static void api_edges(const spec *sp, hx_modelfile *mf) {
    char err[400];
    g_phase = "api edges";
    hx_store_opts o;
    memset(&o, 0, sizeof o);
    o.n_io_threads = 1;
    o.policy = 7;
    CHECK(hx_store_open(mf, &o, err, sizeof err) == NULL && err[0], "unknown policy accepted");
    o.policy = HEARTH_POLICY_LFU;
    CHECK(hx_store_open(NULL, &o, err, sizeof err) == NULL && err[0], "NULL model accepted");
    CHECK(hx_store_open(mf, NULL, err, sizeof err) == NULL && err[0], "NULL options accepted");
    o.n_mirrors = 1;
    err[0] = 0;
    CHECK(hx_store_open(mf, &o, err, sizeof err) == NULL && err[0], "n_mirrors without a list accepted");
    const char *null_mirror[1] = {NULL};
    o.mirrors = null_mirror;
    err[0] = 0;
    CHECK(hx_store_open(mf, &o, err, sizeof err) == NULL && err[0], "a NULL mirror path accepted");
    o.mirrors = NULL;
    o.n_mirrors = 0;
    o.n_io_threads = 0;   /* clamped to 1 */
    hx_store *st = hx_store_open(mf, &o, err, sizeof err);
    CHECK(st != NULL, "n_io_threads 0: %s", err);
    if (st) {
        int dense = -1;
        for (int l = 0; l < sp->L; l++) if (!sp->lk[l]) dense = l;
        CHECK(hx_store_acquire(st, -1, 0) == NULL && hx_store_acquire(st, 0, sp->E) == NULL &&
              hx_store_try_acquire(st, sp->L, 0) == NULL, "out-of-range ids");
        if (dense >= 0) CHECK(hx_store_acquire(st, dense, 0) == NULL, "dense-layer expert");
        hx_store_release(st, 0, 0);                 /* unmatched release: warning only */
        hx_store_release(st, -5, 99);
        hx_store_prefetch(st, 0, NULL, 3);
        hx_store_wait_any(st, 1000);                /* nothing pending: waits out the 1 ms */
        /* counters: one miss then one use; a second acquire is a hit */
        uint64_t r3 = 3;
        int k = moe_key(sp, &r3);
        uint64_t t_reset = hx_now_ns();
        hx_store_reset_stats(st);
        CHECK(hx_store_try_acquire(st, k / sp->E, k % sp->E) == NULL, "cold try_acquire hit");
        uint64_t t0 = hx_now_ns();
        hx_store_wait_any(st, 1000000);   /* returns at the read's completion */
        const void *p = hx_store_acquire(st, k / sp->E, k % sp->E);
        const void *p2 = hx_store_acquire(st, k / sp->E, k % sp->E);
        uint64_t wall = hx_now_ns() - t0, since_reset = hx_now_ns() - t_reset;
        CHECK(p && p == p2, "second acquire returns the same slot");
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        CHECK(s.stall_ns > 0 && s.stall_ns <= wall && s.read_ns > 0 && s.read_ns <= since_reset,
              "stall_ns %llu (wall %llu), read_ns %llu (one read, %llu ns since the reset)", (unsigned long long)s.stall_ns,
              (unsigned long long)wall, (unsigned long long)s.read_ns, (unsigned long long)since_reset);
        CHECK(s.misses == 1 && s.hits == 1 && s.reads == 1 && s.bytes_read == sp->nbytes[k], "counters: m %llu h %llu r %llu",
              (unsigned long long)s.misses, (unsigned long long)s.hits, (unsigned long long)s.reads);
        hx_store_release(st, k / sp->E, k % sp->E);
        hx_store_release(st, k / sp->E, k % sp->E);
        /* a blocking acquire of a cold expert counts its wait as stall */
        int k2 = moe_key(sp, &r3);
        while (hx_store_is_resident(st, k2 / sp->E, k2 % sp->E)) k2 = moe_key(sp, &r3);
        hx_store_reset_stats(st);
        t0 = hx_now_ns();
        p = hx_store_acquire(st, k2 / sp->E, k2 % sp->E);
        wall = hx_now_ns() - t0;
        hx_store_get_stats(st, &s);
        CHECK(p && s.misses == 1 && s.stall_ns > 0 && s.stall_ns <= wall, "blocking acquire: stall_ns %llu (wall %llu)",
              (unsigned long long)s.stall_ns, (unsigned long long)wall);
        if (p) hx_store_release(st, k2 / sp->E, k2 % sp->E);
        /* prefetch, then use: counted as used; a second prefetch of a resident expert reads nothing */
        int ids[4], n = 0;
        for (int e = 0; e < sp->E && n < 4; e++)
            if (sp->nbytes[(k / sp->E) * sp->E + e] && !hx_store_is_resident(st, k / sp->E, e)) ids[n++] = e;
        hx_store_reset_stats(st);
        hx_store_prefetch(st, k / sp->E, ids, n);
        for (int w = 0; w < 200; w++) {
            int all = 1;
            for (int i = 0; i < n; i++) all &= hx_store_is_resident(st, k / sp->E, ids[i]);
            if (all) break;
            hx_store_wait_any(st, 10000);
        }
        hx_store_prefetch(st, k / sp->E, ids, n);
        hx_store_get_stats(st, &s);
        CHECK(s.prefetch_issued == (uint64_t)n && s.reads == (uint64_t)n, "prefetch issued %llu reads %llu want %d",
              (unsigned long long)s.prefetch_issued, (unsigned long long)s.reads, n);
        for (int i = 0; i < n; i++) {
            const void *q = hx_store_try_acquire(st, k / sp->E, ids[i]);
            verify_slab(sp, (k / sp->E) * sp->E + ids[i], q, "prefetched");
            if (q) hx_store_release(st, k / sp->E, ids[i]);
        }
        hx_store_get_stats(st, &s);
        CHECK(s.prefetch_used == (uint64_t)n && s.hits == (uint64_t)n && s.misses == 0, "prefetch_used %llu hits %llu",
              (unsigned long long)s.prefetch_used, (unsigned long long)s.hits);
        hx_store_close(st);
    }
    /* store.c starts at most 64 readers (with a warning); the minimum slot count follows */
    o.n_io_threads = 100;
    st = hx_store_open(mf, &o, err, sizeof err);
    CHECK(st != NULL, "n_io_threads 100: %s", err);
    if (st) {
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        CHECK(s.n_slots == 2 * sp->K + 64 + 2, "n_io_threads 100: %d slots, want 2*top_k + 64 + 2", s.n_slots);
        uint64_t r4 = 4;
        acq_rel(st, sp, moe_key(sp, &r4), "64 readers");
        hx_store_close(st);
    }
    hx_store_close(NULL);
    /* a NULL store is inert everywhere */
    int one = 0;
    hx_store_stats z;
    hx_store_prefetch(NULL, 0, &one, 1);
    hx_store_tick(NULL);
    hx_store_wait_any(NULL, 10);
    hx_store_release(NULL, 0, 0);
    hx_store_reset_stats(NULL);
    hx_store_set_read_hook(NULL, gate_hook, NULL);
    hx_store_get_stats(NULL, &z);
    CHECK(hx_store_try_acquire(NULL, 0, 0) == NULL && hx_store_acquire(NULL, 0, 0) == NULL &&
          !hx_store_is_resident(NULL, 0, 0) && hx_store_heat(NULL) == NULL && z.n_slots == 0, "NULL store");
}

/* A prefetch never evicts a hot expert. LFU: with heat_decay 0.5 the hot threshold
 * (heat of an expert used in half of all tokens) is 1: a slab used this token is hot,
 * one last used three tokens ago (heat 0.125) is not. LRU: a slab used during this
 * token is hot, after one tick it is not. The cache is full of slabs used this token. */
static void hot_rule(const spec *sp, hx_modelfile *mf) {
    static const char *names[2] = {"LRU", "LFU"};
    char err[400];
    g_phase = "hot rule";
    for (int pol = 0; pol < 2; pol++) {
        hx_store_opts o;
        memset(&o, 0, sizeof o);
        o.n_io_threads = 1;
        o.policy = pol;
        o.heat_decay = 0.5f;
        hx_store *st = hx_store_open(mf, &o, err, sizeof err);
        CHECK(st != NULL, "open: %s", err);
        if (!st) continue;
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int nu = 0, L1 = sp->L - 1, cold = -1;
        for (int k = 0; k < sp->L * sp->E && nu < s.n_slots; k++) {
            if (!sp->nbytes[k] || k / sp->E >= L1) continue;
            const void *p = hx_store_acquire(st, k / sp->E, k % sp->E);
            if (p) { hx_store_release(st, k / sp->E, k % sp->E); nu++; }
        }
        for (int e = 0; e < sp->E && cold < 0; e++)
            if (sp->nbytes[L1 * sp->E + e]) cold = e;
        CHECK(nu == s.n_slots, "%s hot rule: %d of %d slots filled", names[pol], nu, s.n_slots);
        hx_store_reset_stats(st);
        hx_store_prefetch(st, L1, &cold, 1);
        for (int w = 0; w < 20; w++) hx_store_wait_any(st, 2000);
        hx_store_get_stats(st, &s);
        CHECK(s.prefetch_issued == 0 && !hx_store_is_resident(st, L1, cold),
              "%s: a prefetch evicted an expert used this token (issued %llu)", names[pol],
              (unsigned long long)s.prefetch_issued);
        for (int t = 0; t < (pol == HEARTH_POLICY_LFU ? 3 : 1); t++) hx_store_tick(st);
        hx_store_prefetch(st, L1, &cold, 1);
        for (int w = 0; w < 200 && !hx_store_is_resident(st, L1, cold); w++) hx_store_wait_any(st, 5000);
        hx_store_get_stats(st, &s);
        CHECK(s.prefetch_issued == 1 && hx_store_is_resident(st, L1, cold), "%s: a prefetch could not evict a cooled expert",
              names[pol]);
        hx_store_close(st);
    }
}

/* Slabs past the end of a truncated file: acquire returns NULL, others still work. */
static void unreadable(const spec *sp) {
    char path[800], err[400];
    snprintf(path, sizeof path, "%s.trunc", sp->path);
    g_phase = "unreadable slabs";
    CHECK(copy_file(sp->path, path, -1, 0), "copy");
    hx_modelfile *mf = hx_modelfile_open(path, 1, err, sizeof err);
    CHECK(mf != NULL, "open copy: %s", err);
    if (!mf) return;
    int ne = sp->L * sp->E, last = -1;
    for (int k = 0; k < ne; k++)
        if (sp->nbytes[k] && (last < 0 || sp->off[k] > sp->off[last])) last = k;
    /* rewrite the copy without its last slab (the reader validated the full file already) */
    FILE *in = fopen(sp->path, "rb"), *out = fopen(path, "wb");
    uint64_t keep = sp->off[last];
    uint8_t *buf = (uint8_t *)malloc((size_t)keep);
    int ok = in && out && fread(buf, 1, (size_t)keep, in) == keep && fwrite(buf, 1, (size_t)keep, out) == keep;
    if (in) fclose(in);
    if (out) fclose(out);
    free(buf);
    CHECK(ok, "truncate copy");
    printf("  unreadable slab (truncated copy): the read-error messages below are expected\n");
    fflush(stdout);
    hx_store *st = open_store(mf, 0, 2, 0, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
    CHECK(st != NULL, "open: %s", err);
    if (st) {
        CHECK(hx_store_acquire(st, last / sp->E, last % sp->E) == NULL, "unreadable slab returned");
        CHECK(hx_store_try_acquire(st, last / sp->E, last % sp->E) == NULL, "unreadable slab try_acquire");
        hx_store_tick(st);   /* retried after a tick, and still unreadable */
        CHECK(hx_store_acquire(st, last / sp->E, last % sp->E) == NULL, "unreadable slab returned after a tick");
        for (int k = 0; k < ne; k++) {
            if (!sp->nbytes[k] || sp->off[k] == sp->off[last]) continue;
            const void *p = hx_store_acquire(st, k / sp->E, k % sp->E);
            verify_slab(sp, k, p, "readable slab next to an unreadable one");
            if (p) hx_store_release(st, k / sp->E, k % sp->E);
            tickle();
        }
        hx_store_close(st);
    }
    /* pinning it at open: the open fails (after waiting for the pin reads) */
    {
        char up[820];
        float *heat = (float *)calloc((size_t)ne, sizeof(float));
        heat[last] = 100.0f;
        snprintf(up, sizeof up, "%s.usage", path);
        CHECK(write_usage(up, sp->L, sp->E, 10, heat), "write usage profile");
        err[0] = 0;
        st = open_store(mf, 30 * mf->slab_bytes_max, 2, 0, HEARTH_POLICY_LFU, up, NULL, 0.5f, 0, NULL, 0, err, sizeof err);
        CHECK(st == NULL && strstr(err, "at open"), "a store whose pinned expert cannot be read opened (%s)", err);
        hx_store_close(st);
        remove(up);
        free(heat);
    }
    hx_modelfile_close(mf);
    remove(path);
}

/* close() with demands and prefetches still queued */
static void close_with_queue(const spec *sp, hx_modelfile *mf) {
    char err[400];
    g_phase = "close with queued work";
    for (int rep = 0; rep < 20; rep++) {
        hx_store *st = open_store(mf, 0, 1 + rep % 3, rep & 1, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
        CHECK(st != NULL, "open: %s", err);
        if (!st) continue;
        uint64_t r = (uint64_t)rep;
        for (int i = 0; i < 2 * sp->K; i++) {
            int k = moe_key(sp, &r);
            hx_store_try_acquire(st, k / sp->E, k % sp->E);
        }
        for (int l = 0; l < sp->L; l++) {
            int ids[32];
            for (int e = 0; e < sp->E && e < 32; e++) ids[e] = e;
            hx_store_prefetch(st, l, ids, sp->E < 32 ? sp->E : 32);
        }
        hx_store_close(st);
        tickle();
    }
}

/* ---------------------------------------- deterministic scheduling (1 reader) */

/* (a) A queued prefetch that gets demanded jumps the prefetch queue and is read
 * once. (b) A prefetch demanded while its read is in flight is then held like a
 * demand: with every other slot acquired or held, a further demand waits rather
 * than evicting it. The read hook blocks the reader to fix the interleaving. */
static void promotion(const spec *sp, hx_modelfile *mf) {
    char err[400];
    int *keys = (int *)malloc(sizeof(int) * (size_t)(sp->L * sp->E));
    int nk = list_keys(sp, keys);
    gate g;
    g_phase = "promotion of a queued prefetch";
    gate_init(&g, sp->E);
    hx_store *st = open_store(mf, 40 * mf->slab_bytes_max, 1, 0, HEARTH_POLICY_LRU, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
    CHECK(st != NULL && keys[3] / sp->E == keys[0] / sp->E, "open: %s", err);
    if (st) {
        int l = keys[0] / sp->E, pf[4], k[4];
        for (int i = 0; i < 4; i++) { k[i] = keys[i]; pf[i] = keys[i] % sp->E; }
        hx_store_set_read_hook(st, gate_hook, &g);
        gate_set(&g, 1);
        hx_store_prefetch(st, l, pf, 4);
        CHECK(gate_wait(&g, 1, 10000), "the first prefetch never reached the reader");
        CHECK(!hx_store_try_acquire(st, l, pf[3]), "a queued prefetch was reported as a hit");
        gate_set(&g, 0);
        acq_rel(st, sp, k[3], "promoted prefetch");
        CHECK(wait_resident(st, sp, k, 4, 10000), "prefetches not completed");
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        hx_mutex_lock(&g.mu);
        int n = g.n, order_ok = n == 4 && g.key[0] == k[0] && g.key[1] == k[3] && g.key[2] == k[1] && g.key[3] == k[2];
        CHECK(order_ok, "reads %d: %d %d %d %d, want %d %d %d %d (the demanded prefetch must be read next, once)", n,
              g.key[0], g.key[1], g.key[2], g.key[3], k[0], k[3], k[1], k[2]);
        hx_mutex_unlock(&g.mu);
        CHECK(s.reads == 4 && s.misses == 1 && s.hits == 0 && s.prefetch_issued == 3,
              "promotion: reads %llu misses %llu hits %llu prefetch_issued %llu", (unsigned long long)s.reads,
              (unsigned long long)s.misses, (unsigned long long)s.hits, (unsigned long long)s.prefetch_issued);
        hx_store_close(st);
    }
    gate_destroy(&g);

    g_phase = "promotion of an in-flight prefetch";
    gate_init(&g, sp->E);
    st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LRU, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
    CHECK(st != NULL, "open: %s", err);
    if (st) {
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int ns = s.n_slots;
        CHECK(nk > ns, "need more experts than slots");
        int x = keys[ns - 2], y1 = keys[ns - 1], y2 = keys[ns], xe = x % sp->E;
        for (int i = 0; i < ns - 2; i++)
            verify_slab(sp, keys[i], hx_store_acquire(st, keys[i] / sp->E, keys[i] % sp->E), "held by refs");
        hx_store_set_read_hook(st, gate_hook, &g);
        gate_set(&g, 1);
        hx_store_prefetch(st, x / sp->E, &xe, 1);
        CHECK(gate_wait(&g, 1, 10000), "the prefetch never reached the reader");
        CHECK(!hx_store_try_acquire(st, x / sp->E, xe), "an in-flight prefetch was reported as a hit");
        gate_set(&g, 0);
        CHECK(wait_resident(st, sp, &x, 1, 10000), "in-flight prefetch not completed");
        CHECK(!hx_store_try_acquire(st, y1 / sp->E, y1 % sp->E), "cold hit");
        CHECK(wait_resident(st, sp, &y1, 1, 10000), "demand not completed");
        /* every slot is now acquired or demanded-but-uncollected: a further demand must wait */
        int n0 = gate_count(&g);
        gate_set(&g, 1);
        CHECK(!hx_store_try_acquire(st, y2 / sp->E, y2 % sp->E), "cold hit");
        int started = gate_wait(&g, n0 + 1, 300);
        CHECK(!started && hx_store_is_resident(st, x / sp->E, xe),
              "a prefetch demanded while in flight was evicted before it was collected");
        gate_set(&g, 0);
        acq_rel(st, sp, x, "in-flight promotion");
        acq_rel(st, sp, y1, "held demand");
        acq_rel(st, sp, y2, "waiting demand");
        for (int i = 0; i < ns - 2; i++) hx_store_release(st, keys[i] / sp->E, keys[i] % sp->E);
        CHECK(gate_reads_of(&g, x) == 1, "in-flight promotion: read %d times", gate_reads_of(&g, x));
        hx_store_close(st);
    }
    gate_destroy(&g);
    free(keys);
    printf("  promotion: a demanded queued prefetch is read next and once; one demanded in flight is held until collected\n");
}

/* Prefetch protection (LRU, minimum slots, one reader): an unused prefetched slab
 * is evicted only after every unprotected one; using it, or the end of the token,
 * removes the protection and it becomes an ordinary victim (here the oldest one).
 * A hint for an expert that is already resident protects it the same way. */
static void protection(const spec *sp, hx_modelfile *mf) {
    static const char *what[4] = {"unused, same token", "used", "unused, after a tick", "hinted while resident"};
    char err[400];
    int *keys = (int *)malloc(sizeof(int) * (size_t)(sp->L * sp->E));
    list_keys(sp, keys);
    g_phase = "prefetch protection";
    for (int v = 0; v < 4; v++) {
        hx_store *st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LRU, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
        CHECK(st != NULL, "open: %s", err);
        if (!st) continue;
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int ns = s.n_slots, P = keys[0], pe = P % sp->E, A0 = keys[1];
        if (v == 3) acq_rel(st, sp, P, "protection");   /* resident and the oldest before the hint */
        hx_store_prefetch(st, P / sp->E, &pe, 1);
        CHECK(wait_resident(st, sp, &P, 1, 10000), "prefetch not completed");
        if (v == 1) acq_rel(st, sp, P, "protection");
        for (int i = 1; i < ns; i++) acq_rel(st, sp, keys[i], "protection fill");
        if (v == 2) hx_store_tick(st);
        hx_store_reset_stats(st);
        acq_rel(st, sp, keys[ns], "protection evict");
        hx_store_get_stats(st, &s);
        int p_res = hx_store_is_resident(st, P / sp->E, pe), a_res = hx_store_is_resident(st, A0 / sp->E, A0 % sp->E);
        if (v == 0 || v == 3)
            CHECK(p_res && !a_res && s.evictions == 1 && s.prefetch_wasted == 0,
                  "prefetch %s: protected slab evicted before the oldest unprotected one", what[v]);
        else
            CHECK(!p_res && a_res && s.evictions == 1 && s.prefetch_wasted == (uint64_t)(v == 2),
                  "prefetch %s: still protected (resident %d, wasted %llu)", what[v], p_res,
                  (unsigned long long)s.prefetch_wasted);
        hx_store_close(st);
    }

    /* Last resort: with every other slot acquired, a demand evicts an unused prefetch
     * (protected, not held) instead of waiting. Polled with is_resident only (acquire and
     * wait_any may also drop holds). */
    for (int pol = 0; pol < 2; pol++) {
        hx_store *st = open_store(mf, 0, 1, 0, pol, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
        CHECK(st != NULL, "open: %s", err);
        if (!st) continue;
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int ns = s.n_slots, P = keys[0], pe = P % sp->E, Y = keys[ns];
        hx_store_prefetch(st, P / sp->E, &pe, 1);
        CHECK(wait_resident(st, sp, &P, 1, 10000), "prefetch not completed");
        for (int i = 1; i < ns; i++)
            verify_slab(sp, keys[i], hx_store_acquire(st, keys[i] / sp->E, keys[i] % sp->E), "every other slot acquired");
        hx_store_reset_stats(st);
        CHECK(!hx_store_try_acquire(st, Y / sp->E, Y % sp->E), "cold hit");
        int got = wait_resident(st, sp, &Y, 1, 3000);
        hx_store_get_stats(st, &s);
        CHECK(got && !hx_store_is_resident(st, P / sp->E, pe) && s.prefetch_wasted == 1,
              "%s: a demand waited instead of evicting the only evictable slab, an unused prefetch",
              pol == HEARTH_POLICY_LFU ? "LFU" : "LRU");
        acq_rel(st, sp, Y, "last resort");
        for (int i = 1; i < ns; i++) hx_store_release(st, keys[i] / sp->E, keys[i] % sp->E);
        hx_store_close(st);
    }

    /* LFU samples a few slots; when the sample holds only protected slabs, the full scan
     * must still find the single unprotected one instead of evicting a prefetch */
    hx_store *st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
    CHECK(st != NULL, "open: %s", err);
    if (st) {
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int ns = s.n_slots, rounds = 24, kept = 1;
        acq_rel(st, sp, keys[0], "LFU protection");
        for (int i = 1; i < ns;) {   /* the prefetch queue is bounded: small batches */
            int ids[4], n = 0, layer = keys[i] / sp->E;
            while (i < ns && n < 4 && keys[i] / sp->E == layer) ids[n++] = keys[i++] % sp->E;
            hx_store_prefetch(st, layer, ids, n);
            CHECK(wait_resident(st, sp, keys + i - n, n, 10000), "prefetch not completed");
        }
        hx_store_reset_stats(st);
        for (int r = 0; r < rounds; r++) {
            acq_rel(st, sp, keys[ns + r], "LFU protection churn");
            for (int i = 1; i < ns; i++) kept &= hx_store_is_resident(st, keys[i] / sp->E, keys[i] % sp->E);
        }
        hx_store_get_stats(st, &s);
        CHECK(kept && s.evictions == (uint64_t)rounds && s.prefetch_wasted == 0,
              "LFU: a demand evicted a protected prefetch although an unprotected slab was evictable (wasted %llu)",
              (unsigned long long)s.prefetch_wasted);
        hx_store_close(st);
    }
    free(keys);
    printf("  prefetch protection: unused prefetches and hinted resident slabs outlive unprotected slabs (LRU, LFU); "
           "use or a tick ends it; a demand evicts one as the last resort\n");
}

/* The usage profile seeds LFU heat: its hottest expert survives churn that evicts
 * it (as the oldest of equally used slabs) without the profile. */
static void seeded_heat(const spec *sp, hx_modelfile *mf) {
    char up[800], err[400];
    int ne = sp->L * sp->E;
    int *keys = (int *)malloc(sizeof(int) * (size_t)ne);
    float *heat = (float *)calloc((size_t)ne, sizeof(float));
    int nk = list_keys(sp, keys), H = keys[0];
    heat[H] = 1000.0f;
    snprintf(up, sizeof up, "%s.seed.usage", sp->path);
    CHECK(write_usage(up, sp->L, sp->E, 10, heat), "write usage profile");
    g_phase = "seeded heat";
    for (int with = 0; with < 2; with++) {
        hx_store *st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LFU, with ? up : NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
        CHECK(st != NULL, "open: %s", err);
        if (!st) continue;
        acq_rel(st, sp, H, "seeded");
        for (int i = 1; i <= 40 && i < nk; i++) acq_rel(st, sp, keys[i], "seeded churn");
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int res = hx_store_is_resident(st, H / sp->E, H % sp->E);
        CHECK(s.evictions >= 20, "seeded heat: only %llu evictions", (unsigned long long)s.evictions);
        if (with) CHECK(res, "the profile's hottest expert was evicted: seeded heat not used by LFU");
        else CHECK(!res, "control run: the oldest of equally used experts should have been evicted");
        hx_store_close(st);
    }

    /* Seeded heat is a per-token rate on the scale of the decayed counter: rate r gives
     * r/(1-decay). The prefetch hot rule (heat >= 0.5/(1-decay) = 100, decay 0.995) makes
     * the scale visible: with every other slot acquired, a prefetch may evict an expert
     * seeded at 0.4 uses/token but not one seeded at 0.6. A profile with tokens_observed 0
     * gives no rate: its values are used as heat directly (80 is not hot, 120 is). */
    static const struct { uint64_t tokens; float a, b; } hv[2] = {{1000, 400.0f, 600.0f}, {0, 80.0f, 120.0f}};
    int A = keys[1], B = keys[2];
    for (int v = 0; v < 2; v++) {
        memset(heat, 0, sizeof(float) * (size_t)ne);
        heat[A] = hv[v].a;
        heat[B] = hv[v].b;
        CHECK(write_usage(up, sp->L, sp->E, hv[v].tokens, heat), "write usage profile");
        hx_store *st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LFU, up, NULL, 0, 0, NULL, 0, err, sizeof err);
        CHECK(st != NULL, "open: %s", err);
        if (!st) continue;
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int ns = s.n_slots, held = 0, x = -1, y = -1;
        for (int i = 3; i < nk && held < ns - 1; i++, held++)
            verify_slab(sp, keys[i], hx_store_acquire(st, keys[i] / sp->E, keys[i] % sp->E), "held by refs");
        for (int i = 3 + held; i < nk && y < 0; i++) {   /* two more experts of one layer, never loaded */
            if (x < 0) x = keys[i];
            else if (keys[i] / sp->E == x / sp->E) y = keys[i];
        }
        acq_rel(st, sp, A, "seeded cool");
        int xe = x % sp->E, ye = y % sp->E;
        hx_store_prefetch(st, x / sp->E, &xe, 1);
        wait_resident(st, sp, &x, 1, 2000);
        CHECK(hx_store_is_resident(st, x / sp->E, xe) && !hx_store_is_resident(st, A / sp->E, A % sp->E),
              "a prefetch could not evict an expert seeded at %g over %llu tokens (seeded heat on the wrong scale)",
              (double)hv[v].a, (unsigned long long)hv[v].tokens);
        acq_rel(st, sp, B, "seeded hot");   /* evicts the prefetched slab: the only evictable one */
        hx_store_prefetch(st, y / sp->E, &ye, 1);
        hx_store_wait_any(st, 50000);
        CHECK(!hx_store_is_resident(st, y / sp->E, ye) && hx_store_is_resident(st, B / sp->E, B % sp->E),
              "a prefetch evicted an expert seeded at %g over %llu tokens (seeded heat on the wrong scale)",
              (double)hv[v].b, (unsigned long long)hv[v].tokens);
        for (int i = 3; i < 3 + held; i++) hx_store_release(st, keys[i] / sp->E, keys[i] % sp->E);
        hx_store_close(st);
    }
    remove(up);
    free(heat);
    free(keys);
    printf("  seeded heat: the profile steers LFU eviction, on the decayed-counter scale (prefetch hot rule), raw "
           "values when tokens_observed is 0\n");
}

/* Prefetch hints beyond the queue bound (store.c: max(2*top_k, 2*n_io, 4) queued,
 * plus max(n_io-1, 1) in flight) are dropped rather than queued, and can be hinted
 * again later. Readers are blocked in the read hook while the hints arrive. */
static void pf_queue_limit(const spec *sp, hx_modelfile *mf) {
    static const int ios[] = {1, 5};
    char err[400];
    int ne = sp->L * sp->E, *keys = (int *)malloc(sizeof(int) * (size_t)ne), nk = list_keys(sp, keys);
    g_phase = "prefetch queue bound";
    for (size_t ii = 0; ii < sizeof ios / sizeof ios[0]; ii++) {
        int n_io = ios[ii], qmax = 2 * sp->K, inflight = n_io > 1 ? n_io - 1 : 1;
        if (2 * n_io > qmax) qmax = 2 * n_io;
        if (qmax < 4) qmax = 4;
        int n = inflight + qmax + 4;
        CHECK(nk >= n, "prefetch queue test needs %d experts", n);
        if (nk < n) break;
        gate g;
        gate_init(&g, sp->E);
        hx_store *st = open_store(mf, 40 * mf->slab_bytes_max, n_io, 0, HEARTH_POLICY_LRU, NULL, NULL, 0, 0, NULL, 0, err,
                                  sizeof err);
        CHECK(st != NULL, "open: %s", err);
        if (st) {
            hx_store_set_read_hook(st, gate_hook, &g);
            gate_set(&g, 1);
            for (int i = 0; i < n; i++) {   /* one hint per call, in key order */
                int e = keys[i] % sp->E;
                hx_store_prefetch(st, keys[i] / sp->E, &e, 1);
                if (i == inflight - 1)
                    CHECK(gate_wait(&g, inflight, 10000), "%d readers: the first prefetches never reached the readers", n_io);
            }
            gate_set(&g, 0);
            int kept = inflight + qmax;
            CHECK(wait_resident(st, sp, keys, kept, 10000), "%d readers: queued prefetches not completed", n_io);
            hx_sleep_us(20000);
            hx_store_stats s;
            hx_store_get_stats(st, &s);
            int extra = 0;
            for (int i = kept; i < n; i++) extra += hx_store_is_resident(st, keys[i] / sp->E, keys[i] % sp->E);
            CHECK(s.prefetch_issued == (uint64_t)kept && extra == 0,
                  "%d readers: %llu prefetches issued, want %d (%d in flight + queue bound %d); %d hints beyond were read",
                  n_io, (unsigned long long)s.prefetch_issued, kept, inflight, qmax, extra);
            for (int i = kept; i < n; i++) {   /* hinted again later, they are read */
                int e = keys[i] % sp->E;
                hx_store_prefetch(st, keys[i] / sp->E, &e, 1);
            }
            CHECK(wait_resident(st, sp, keys + kept, n - kept, 10000), "dropped hints could not be hinted again");
            gate_set(&g, 0);
            hx_store_close(st);
        }
        gate_destroy(&g);
    }
    free(keys);
    printf("  prefetch queue bound: hints beyond in-flight + queue bound are dropped (1 and 5 readers) and can be re-hinted\n");
}

/* pin_fraction 0.9 with a cache just above the minimum: pinning stops where fewer
 * than the minimum slots would stay unpinned, so top-k bursts still complete. */
static void pin_cap(const spec *sp, hx_modelfile *mf, int quick) {
    char up[800], err[400];
    int ne = sp->L * sp->E, n_io = 2, min = 2 * sp->K + n_io + 2;
    float *heat = (float *)calloc((size_t)ne, sizeof(float));
    for (int k = 0; k < ne; k++) heat[k] = sp->nbytes[k] ? (float)(1 + k % 7) : 0.0f;
    snprintf(up, sizeof up, "%s.pins.usage", sp->path);
    CHECK(write_usage(up, sp->L, sp->E, 100, heat), "write usage profile");
    g_phase = "pin cap";
    for (int pol = 0; pol < 2; pol++) {
        hx_store *st = open_store(mf, (uint64_t)(min + 6) * mf->slab_bytes_max, n_io, 0, pol, up, NULL, 0.9f, 0, NULL, 0,
                                  err, sizeof err);
        CHECK(st != NULL, "open: %s", err);
        if (!st) continue;
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int ok = s.n_slots == min + 6 && s.pinned > 0 && s.n_slots - s.pinned == min;
        CHECK(ok, "pin_fraction 0.9: %d of %d slots pinned, %d must stay unpinned", s.pinned, s.n_slots, min);
        if (ok) topk_bursts(st, sp, quick ? 10 : 30, 7 + (uint64_t)pol, "bursts beside pins");   /* would deadlock otherwise */
        hx_store_close(st);
    }
    /* warm start with a few more profiled experts than slots fills exactly the unpinned slots */
    for (int k = 0, n_hot = 0; k < ne; k++)
        if (heat[k] > 0 && ++n_hot > min + 9) heat[k] = 0.0f;
    CHECK(write_usage(up, sp->L, sp->E, 100, heat), "write usage profile");
    hx_store *st = open_store(mf, (uint64_t)(min + 6) * mf->slab_bytes_max, n_io, 0, HEARTH_POLICY_LFU, up, NULL, 0.9f, 1,
                              NULL, 0, err, sizeof err);
    CHECK(st != NULL, "open: %s", err);
    if (st) {
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        CHECK(s.resident == s.n_slots && s.reads == (uint64_t)s.n_slots && s.n_slots - s.pinned == min,
              "warm start: %d resident, %llu reads, %d pinned of %d slots", s.resident, (unsigned long long)s.reads,
              s.pinned, s.n_slots);
        if (s.n_slots - s.pinned == min) topk_bursts(st, sp, 3, 5, "bursts after a warm start");
        hx_store_close(st);
    }
    /* LRU: warm-loaded slabs are ordinary victims. After a churn through twice as many
     * unprofiled experts as there are slots, only the pinned ones (the top of the
     * ranking by heat, then key) are still resident. */
    st = open_store(mf, (uint64_t)(min + 6) * mf->slab_bytes_max, n_io, 0, HEARTH_POLICY_LRU, up, NULL, 0.9f, 1, NULL, 0,
                    err, sizeof err);
    CHECK(st != NULL, "open: %s", err);
    if (st) {
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int *rank = (int *)malloc(sizeof(int) * (size_t)ne), nr = 0, churn = 0, pins_kept = 1, warm_gone = 1;
        for (int k = 0; k < ne; k++)
            if (sp->nbytes[k] && heat[k] > 0) rank[nr++] = k;
        for (int i = 1; i < nr; i++)   /* heat desc, key asc */
            for (int j = i; j > 0 && heat[rank[j]] > heat[rank[j - 1]]; j--) {
                int t = rank[j]; rank[j] = rank[j - 1]; rank[j - 1] = t;
            }
        for (int k = 0; k < ne && churn < 2 * s.n_slots; k++)
            if (sp->nbytes[k] && heat[k] == 0.0f) { acq_rel(st, sp, k, "churn after a warm start"); churn++; }
        for (int i = 0; i < nr; i++) {
            int res = hx_store_is_resident(st, rank[i] / sp->E, rank[i] % sp->E);
            if (i < s.pinned) pins_kept &= res;
            else warm_gone &= !res;
        }
        CHECK(s.pinned > 0 && churn == 2 * s.n_slots && pins_kept && warm_gone,
              "LRU after a warm start: pinned slabs kept %d, warm-loaded slabs evicted %d (%d pinned, churn %d)", pins_kept,
              warm_gone, s.pinned, churn);
        free(rank);
        hx_store_close(st);
    }
    remove(up);
    free(heat);
    printf("  pin cap: pin_fraction 0.9 near the minimum leaves the minimum unpinned (bursts complete); warm start fills the "
           "rest with ordinary (evictable) slabs\n");
}

/* A demand read that fails on every attempt: acquire returns NULL for the rest of
 * the token without queuing more reads, wait_any waits out its timeout (a polling
 * loop must not spin), other experts are unaffected, and after the next tick the
 * expert is read again. With a mirror, a read that fails on one copy is retried on
 * the other. */
static void read_failures(const spec *sp, hx_modelfile *mf) {
    char err[400], m1[800];
    int *keys = (int *)malloc(sizeof(int) * (size_t)(sp->L * sp->E));
    int nk = list_keys(sp, keys);
    gate g;
    g_phase = "read failures";
    printf("  read failures (injected): the read-error messages below are expected\n");
    fflush(stdout);
    gate_init(&g, sp->E);
    hx_store *st = open_store(mf, 0, 2, 0, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
    CHECK(st != NULL, "open: %s", err);
    if (st) {
        int X = keys[5], Y = keys[6];
        hx_store_set_read_hook(st, gate_hook, &g);
        gate_fail(&g, X, -1, -1);
        CHECK(hx_store_acquire(st, X / sp->E, X % sp->E) == NULL, "acquire of an unreadable expert returned a slab");
        int n1 = gate_count(&g);
        CHECK(n1 >= 2, "a failed read was not retried (%d attempts)", n1);
        CHECK(!hx_store_try_acquire(st, X / sp->E, X % sp->E), "unreadable expert: try_acquire hit");
        uint64_t t0 = hx_now_ns();
        hx_store_wait_any(st, 30000);
        double wait_ms = (double)(hx_now_ns() - t0) / 1e6;
        CHECK(wait_ms >= 30.0, "wait_any returned after %.3f ms with only a failed expert missing (polling would spin)",
              wait_ms);
        acq_rel(st, sp, Y, "next to an unreadable expert");
        gate_fail(&g, -1, -1, 0);   /* the device recovers */
        CHECK(!hx_store_try_acquire(st, X / sp->E, X % sp->E), "failed expert: hit within the same token");
        hx_store_wait_any(st, 20000);
        CHECK(gate_count(&g) == n1 + 1, "a failed expert was re-read within the same token (%d extra attempts)",
              gate_count(&g) - n1 - 1);
        hx_store_tick(st);
        acq_rel(st, sp, X, "failed expert after a tick");
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        CHECK(s.reads == 2, "reads %llu, want 2", (unsigned long long)s.reads);
        /* a failed prefetch is dropped without marking the expert: a demand right after reads it */
        int Z = keys[8], ze = Z % sp->E, nz = gate_count(&g);
        gate_fail(&g, Z, -1, -1);
        hx_store_prefetch(st, Z / sp->E, &ze, 1);
        CHECK(gate_wait(&g, nz + 2, 10000), "failing prefetch not retried");
        hx_store_wait_any(st, 50000);
        gate_fail(&g, -1, -1, 0);
        acq_rel(st, sp, Z, "demand after a failed prefetch");
        hx_store_close(st);
    }
    gate_destroy(&g);

    g_phase = "mirror failover";
    snprintf(m1, sizeof m1, "%s.failover", sp->path);
    CHECK(copy_file(sp->path, m1, -1, 0), "copy mirror");
    const char *ms[1] = {m1};
    int churn = 2 * sp->K + 4;
    CHECK(nk >= 20 + churn, "failover test needs %d experts", 20 + churn);
    for (int bad_file = 0; bad_file < 2 && nk >= 20 + churn; bad_file++) {
        gate_init(&g, sp->E);
        st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LRU, NULL, NULL, 0, 0, ms, 1, err, sizeof err);
        CHECK(st != NULL, "open with a mirror: %s", err);
        if (st) {
            int X = keys[7];
            hx_store_set_read_hook(st, gate_hook, &g);
            gate_fail(&g, X, bad_file, -1);   /* this copy of X is unreadable */
            for (int rep = 0; rep < 4; rep++) {
                acq_rel(st, sp, X, "failover");
                for (int i = 0; i < churn; i++) acq_rel(st, sp, keys[20 + i], "failover churn");   /* LRU: evicts X */
            }
            hx_mutex_lock(&g.mu);
            int over = 0;
            for (int i = 0; i + 1 < g.n && i + 1 < GATE_LOG; i++)
                over += g.key[i] == X && g.file[i] == bad_file && g.key[i + 1] == X && g.file[i + 1] == !bad_file;
            hx_mutex_unlock(&g.mu);
            CHECK(over > 0, "a read that failed on copy %d was not retried on the other one", bad_file);
            hx_store_close(st);
        }
        gate_destroy(&g);
    }
    remove(m1);
    free(keys);
    printf("  read failures: NULL for the rest of the token, wait_any does not spin, re-read after a tick; "
           "mirror failover both ways\n");
}

/* Starvation at the minimum slot count, one deterministic scenario each:
 * (a) a failed read gives its slot back: after several failures every slot can still
 *     be filled with a demanded slab at once;
 * (b) a release alone restarts a reader starved behind acquired slabs (no acquire or
 *     wait_any call, which would also wake it);
 * (c) a caller blocked in acquire or wait_any while a read is in flight keeps the holds
 *     on slabs it demanded but has not collected yet: that read is progress, so nothing
 *     is evicted and read again. */
static void starvation(const spec *sp, hx_modelfile *mf) {
    char err[400];
    int *keys = (int *)malloc(sizeof(int) * (size_t)(sp->L * sp->E));
    int nk = list_keys(sp, keys);
    gate g;

    g_phase = "failed reads keep their slots";
    printf("  failed reads at the minimum slot count (injected): the read-error messages below are expected\n");
    fflush(stdout);
    gate_init(&g, sp->E);
    hx_store *st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LRU, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
    CHECK(st != NULL, "open: %s", err);
    if (st) {
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int ns = s.n_slots, nfail = 5;
        CHECK(nk >= nfail + ns, "failed-reads test needs %d experts", nfail + ns);
        hx_store_set_read_hook(st, gate_hook, &g);
        for (int i = 0; i < nfail; i++) {
            gate_fail(&g, keys[i], -1, -1);
            CHECK(hx_store_acquire(st, keys[i] / sp->E, keys[i] % sp->E) == NULL, "an injected failure returned a slab");
        }
        gate_fail(&g, -1, -1, 0);
        hx_store_get_stats(st, &s);
        CHECK(s.resident == 0 && s.reads == 0, "after %d failed reads: %d resident, %llu reads", nfail, s.resident,
              (unsigned long long)s.reads);
        const int *fill = keys + nfail;
        for (int i = 0; i < ns; i++) CHECK(!hx_store_try_acquire(st, fill[i] / sp->E, fill[i] % sp->E), "cold hit");
        int ok = wait_resident(st, sp, fill, ns, 5000);
        hx_store_get_stats(st, &s);
        CHECK(ok && s.resident == ns, "after %d failed reads only %d of %d slots could be filled (slots lost)", nfail,
              s.resident, ns);
        for (int i = 0; i < ns; i++) acq_rel(st, sp, fill[i], "fill after failed reads");
        hx_store_tick(st);
        if (ok) topk_bursts(st, sp, 5, 3, "bursts after failed reads");
        hx_store_close(st);
    }
    gate_destroy(&g);

    g_phase = "a release wakes a starved reader";
    gate_init(&g, sp->E);
    st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LRU, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
    CHECK(st != NULL, "open: %s", err);
    if (st) {
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int ns = s.n_slots, Y = keys[ns];
        for (int i = 0; i < ns; i++)
            verify_slab(sp, keys[i], hx_store_acquire(st, keys[i] / sp->E, keys[i] % sp->E), "every slot acquired");
        hx_store_set_read_hook(st, gate_hook, &g);
        CHECK(!hx_store_try_acquire(st, Y / sp->E, Y % sp->E), "cold hit");
        hx_sleep_us(30000);   /* the reader finds no slot and waits */
        CHECK(gate_count(&g) == 0, "a read started with every slot acquired");
        hx_store_release(st, keys[0] / sp->E, keys[0] % sp->E);
        CHECK(gate_wait(&g, 1, 2000), "a release did not restart the reader starved behind acquired slabs");
        acq_rel(st, sp, Y, "after a release");
        for (int i = 1; i < ns; i++) hx_store_release(st, keys[i] / sp->E, keys[i] % sp->E);
        hx_store_close(st);
    }
    gate_destroy(&g);

    for (int v = 0; v < 2; v++) {
        g_phase = v ? "wait_any with a read in flight" : "acquire with a read in flight";
        gate_init(&g, sp->E);
        st = open_store(mf, 0, 2, 0, HEARTH_POLICY_LRU, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
        CHECK(st != NULL, "open: %s", err);
        if (!st) { gate_destroy(&g); continue; }
        hx_store_stats s;
        hx_store_get_stats(st, &s);
        int ns = s.n_slots, nref = ns - 3;
        int H[2] = {keys[nref], keys[nref + 1]}, X = keys[nref + 2], Y = keys[nref + 3];
        hx_store_set_read_hook(st, gate_hook, &g);
        for (int i = 0; i < nref; i++)
            verify_slab(sp, keys[i], hx_store_acquire(st, keys[i] / sp->E, keys[i] % sp->E), "held by refs");
        for (int i = 0; i < 2; i++) CHECK(!hx_store_try_acquire(st, H[i] / sp->E, H[i] % sp->E), "cold hit");
        CHECK(wait_resident(st, sp, H, 2, 10000), "demands not completed");
        hx_store_wait_any(st, 1000);   /* reports those completions */
        gate_set(&g, 1);
        int n0 = gate_count(&g);
        CHECK(!hx_store_try_acquire(st, X / sp->E, X % sp->E), "cold hit");
        CHECK(gate_wait(&g, n0 + 1, 10000), "the read never started");
        CHECK(!hx_store_try_acquire(st, Y / sp->E, Y % sp->E), "cold hit");
        hx_sleep_us(20000);   /* the other reader finds no slot for Y */
        CHECK(gate_count(&g) == n0 + 1, "a read started although every slot was acquired, held or loading");
        gate_opener op;
        gate_open_later(&op, &g, 150000);
        const void *p;
        if (v == 0) {
            p = hx_store_acquire(st, X / sp->E, X % sp->E);
        } else {
            hx_store_wait_any(st, 5000000);
            p = hx_store_try_acquire(st, X / sp->E, X % sp->E);
        }
        gate_opener_join(&op);
        verify_slab(sp, X, p, g_phase);
        int rh = 0;
        for (int i = 0; i < 2; i++) {
            verify_slab(sp, H[i], hx_store_acquire(st, H[i] / sp->E, H[i] % sp->E), "held while a read was in flight");
            rh += gate_reads_of(&g, H[i]);
        }
        CHECK(rh == 2, "%s: holds were broken although a read was in flight (%d reads of the 2 held slabs)", g_phase, rh);
        for (int i = 0; i < 2; i++) hx_store_release(st, H[i] / sp->E, H[i] % sp->E);
        if (p) hx_store_release(st, X / sp->E, X % sp->E);
        for (int i = 0; i < nref; i++) hx_store_release(st, keys[i] / sp->E, keys[i] % sp->E);
        acq_rel(st, sp, Y, "starved demand");
        hx_store_close(st);
        gate_destroy(&g);
        tickle();
    }
    free(keys);
    printf("  starvation: failed reads keep their slots; a release restarts a starved reader; a caller blocked while a "
           "read is in flight keeps its holds (acquire, wait_any)\n");
}

/* hx_store_wait_any returns at once for a demand read that completed after the
 * caller's last look at that expert, also when the caller made other calls in between
 * (the rest of a pass over the missing experts); it reports each completion once, so
 * with nothing new it waits for the next completion or out its timeout. */
static void wait_any_wakeups(const spec *sp, hx_modelfile *mf) {
    char err[400];
    int *keys = (int *)malloc(sizeof(int) * (size_t)(sp->L * sp->E));
    list_keys(sp, keys);
    gate g;
    g_phase = "wait_any wakeups";
    gate_init(&g, sp->E);
    hx_store *st = open_store(mf, 40 * mf->slab_bytes_max, 2, 0, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
    CHECK(st != NULL, "open: %s", err);
    if (st) {
        int A = keys[0], B = keys[1], C = keys[2];
        double ms;
        hx_store_set_read_hook(st, gate_hook, &g);
        CHECK(!hx_store_try_acquire(st, A / sp->E, A % sp->E), "cold hit");
        CHECK(wait_resident(st, sp, &A, 1, 10000), "demand not completed");
        ms = timed_wait_any(st, 1000000);
        CHECK(ms < 10.0, "wait_any blocked %.1f ms although the expert it was waiting for was already resident", ms);

        CHECK(!hx_store_try_acquire(st, B / sp->E, B % sp->E), "cold hit");
        CHECK(wait_resident(st, sp, &B, 1, 10000), "demand not completed");
        const void *p = hx_store_try_acquire(st, A / sp->E, A % sp->E);   /* a hit later in the same pass */
        verify_slab(sp, A, p, "hit between a miss and wait_any");
        if (p) hx_store_release(st, A / sp->E, A % sp->E);
        ms = timed_wait_any(st, 1000000);
        CHECK(ms < 10.0, "wait_any blocked %.1f ms although an expert missed earlier in the pass was resident", ms);

        ms = timed_wait_any(st, 30000);   /* B still uncollected, but already reported */
        CHECK(ms >= 30.0, "wait_any returned after %.3f ms for a completion it had reported (polling would spin)", ms);
        acq_rel(st, sp, B, "collected");
        ms = timed_wait_any(st, 30000);
        CHECK(ms >= 30.0, "wait_any returned after %.3f ms with nothing pending", ms);

        /* a completion the caller has seen, by collecting it with try_acquire or acquire, is no news */
        int D = keys[3], E2 = keys[4], F = keys[5];
        CHECK(!hx_store_try_acquire(st, D / sp->E, D % sp->E), "cold hit");
        CHECK(wait_resident(st, sp, &D, 1, 10000), "demand not completed");
        p = hx_store_try_acquire(st, D / sp->E, D % sp->E);
        verify_slab(sp, D, p, "collected by try_acquire");
        if (p) hx_store_release(st, D / sp->E, D % sp->E);
        ms = timed_wait_any(st, 30000);
        CHECK(ms >= 30.0, "wait_any returned after %.3f ms for a slab already collected by try_acquire", ms);
        acq_rel(st, sp, E2, "cold blocking acquire");
        ms = timed_wait_any(st, 30000);
        CHECK(ms >= 30.0, "wait_any returned after %.3f ms for a slab already collected by acquire", ms);

        /* a demand that failed before the call is news once; then polling it cannot spin */
        printf("  wait_any after a failed demand (injected): the read-error message below is expected\n");
        fflush(stdout);
        int n1 = gate_count(&g);
        gate_fail(&g, F, -1, -1);
        CHECK(!hx_store_try_acquire(st, F / sp->E, F % sp->E), "cold hit");
        CHECK(gate_wait(&g, n1 + 3, 10000), "the failing read was not attempted 3 times");
        hx_sleep_us(20000);
        ms = timed_wait_any(st, 1000000);
        CHECK(ms < 500.0, "wait_any blocked %.1f ms although a demand had failed before the call", ms);
        CHECK(!hx_store_try_acquire(st, F / sp->E, F % sp->E), "a failed expert was returned");
        ms = timed_wait_any(st, 30000);
        CHECK(ms >= 30.0, "wait_any returned after %.3f ms with only a failed expert missing", ms);
        gate_fail(&g, -1, -1, 0);

        gate_set(&g, 1);
        int n0 = gate_count(&g);
        CHECK(!hx_store_try_acquire(st, C / sp->E, C % sp->E), "cold hit");
        CHECK(gate_wait(&g, n0 + 1, 10000), "the read never started");
        gate_opener op;
        gate_open_later(&op, &g, 50000);
        ms = timed_wait_any(st, 2000000);
        gate_opener_join(&op);
        CHECK(ms >= 40.0 && ms < 1000.0, "wait_any took %.1f ms for a read that completed after about 50 ms", ms);
        acq_rel(st, sp, C, "completed during wait_any");
        hx_store_close(st);
    }
    gate_destroy(&g);
    free(keys);
    printf("  wait_any: returns at once for a completion the caller has not seen (also after other calls), reports it "
           "once, otherwise waits for the next completion or the timeout\n");
}

/* A decode loop as in docs/ARCHITECTURE.md (MoE layer schedule, steps 2-4): try_acquire
 * the layer's top-k (misses become demands), hint the next layer, then repeatedly
 * compute (verify) and release what is resident and wait_any for the rest. Whenever
 * an expert still missing is already resident when wait_any is called, the call must
 * return at once: sleeping out the timeout there is a lost wakeup. */
static void decode_loop(const spec *sp, hx_modelfile *mf, int quick) {
    static const int ios[] = {1, 4};
    const uint32_t timeout_us = 200000;
    char err[400];
    g_phase = "decode loop";
    for (size_t ii = 0; ii < sizeof ios / sizeof ios[0]; ii++) {
        hx_store *st = open_store(mf, 24 * mf->slab_bytes_max, ios[ii], 0, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, NULL, 0, err,
                                  sizeof err);
        CHECK(st != NULL, "open: %s", err);
        if (!st) continue;
        uint64_t r = 31 + ii, t0 = hx_now_ns();
        int tokens = quick ? 60 : 200, waits = 0, raced = 0, lost = 0;
        double worst = 0;
        for (int t = 0; t < tokens; t++) {
            for (int l = 0; l < sp->L; l++) {
                if (!sp->lk[l]) continue;
                int pend[64], np = 0, hint[4];
                while (np < sp->K) {
                    int e = (int)(rn(&r, (uint32_t)sp->E) * rn(&r, 1000) / 1000);   /* skewed */
                    if (!contains(pend, np, e)) pend[np++] = e;
                }
                for (int i = 0; i < 4; i++) hint[i] = (int)(rn(&r, (uint32_t)sp->E) * rn(&r, 1000) / 1000);
                for (int pass = 0; np; pass++) {
                    if (pass) {
                        int ready = 0;
                        for (int i = 0; i < np; i++) ready |= hx_store_is_resident(st, l, pend[i]);
                        double ms = timed_wait_any(st, timeout_us);
                        waits++;
                        raced += ready;
                        if (ready && ms > worst) worst = ms;
                        lost += ready && ms >= 0.75 * timeout_us / 1000.0;
                    }
                    for (int i = 0; i < np; i++) {
                        const void *p = hx_store_try_acquire(st, l, pend[i]);
                        if (!p) continue;
                        verify_slab(sp, l * sp->E + pend[i], p, "decode loop");
                        hx_store_release(st, l, pend[i]);
                        pend[i--] = pend[--np];
                    }
                    if (!pass) hx_store_prefetch(st, (l + 1) % sp->L, hint, 4);
                }
                tickle();
            }
            hx_store_tick(st);
        }
        double sec = (double)(hx_now_ns() - t0) / 1e9;
        CHECK(lost == 0, "%d readers: %d wait_any calls slept out the %u ms timeout although an expert they waited for "
              "was resident", ios[ii], lost, timeout_us / 1000);
        printf("  decode loop, %d reader(s): %d tokens in %.2f s, %d waits, %d with a missing expert already resident "
               "(longest %.2f ms), %d lost wakeups\n", ios[ii], tokens, sec, waits, raced, worst, lost);
        hx_store_close(st);
    }
}

/* tokens_observed is an exact u64 count: seed + ticks (2^60 + 1 + 3 = 2^60 + 4, beyond
 * double precision), saturating at 2^64 - 1. */
static void usage_tokens(const spec *sp, hx_modelfile *mf) {
    static const struct { uint64_t seed, want; } v[] = {
        {0, 3}, {(1ull << 60) + 1, (1ull << 60) + 4}, {UINT64_MAX - 1, UINT64_MAX}, {UINT64_MAX, UINT64_MAX}};
    char up[800], up2[800], err[400];
    int ne = sp->L * sp->E;
    float *heat = (float *)calloc((size_t)ne, sizeof(float)), *back = (float *)calloc((size_t)ne, sizeof(float));
    for (int k = 0; k < ne; k++) heat[k] = sp->nbytes[k] ? (float)(k % 5) : 0.0f;
    snprintf(up, sizeof up, "%s.tok.usage", sp->path);
    snprintf(up2, sizeof up2, "%s.tok2.usage", sp->path);
    g_phase = "usage tokens";
    size_t n2 = strlen(up2) + 1;
    char *out_exact = (char *)malloc(n2);   /* exactly sized: ASan sees any read past its NUL */
    memcpy(out_exact, up2, n2);
    for (size_t i = 0; i < sizeof v / sizeof v[0]; i++) {
        CHECK(write_usage(up, sp->L, sp->E, v[i].seed, heat), "write usage profile");
        hx_store *st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LFU, up, out_exact, 0, 0, NULL, 0, err, sizeof err);
        CHECK(st != NULL, "open: %s", err);
        if (!st) continue;
        for (int t = 0; t < 3; t++) hx_store_tick(st);
        hx_store_close(st);
        uint64_t tok = 0;
        int ok = read_usage(up2, sp->L, sp->E, &tok, back), same = ok;
        for (int k = 0; same && k < ne; k++) same = back[k] == heat[k];
        CHECK(ok && same && tok == v[i].want, "usage seeded with %llu tokens, 3 ticks: wrote %llu tokens, want %llu",
              (unsigned long long)v[i].seed, (unsigned long long)tok, (unsigned long long)v[i].want);
    }
    remove(up);
    remove(up2);
    free(out_exact);
    free(heat);
    free(back);
    printf("  usage tokens_observed: exact u64 (2^60 + 1 + 3 ticks), saturating at 2^64 - 1\n");
}

/* The pinned count is int(pin_fraction * slots) as hearth.sim computes it (double
 * precision) for the decimal value the caller wrote, although the option is a float:
 * of 100 slots 0.7 pins 70 (0.7f * 100 = 69.99999881) and 0.35 pins 35; 0.29 pins 28
 * like the simulator (0.29 * 100 = 28.999999999999996). 0.9 stops where fewer than the
 * minimum 11 slots would stay unpinned. 4 x 64 experts, all profiled. */
static void pin_count(void) {
    static const struct { float pf; int want; } v[] = {
        {0.7f, 70}, {0.35f, 35}, {0.29f, 28}, {0.57f, 56}, {0.55f, 55}, {0.3f, 30}, {0.9f, 89}};
    char err[400], up[800];
    spec sp;
    g_phase = "pin count";
    make_spec(&sp, "hx_test_store_pins.hearth", 4, 64, 64, 64, 4, 61, -1, 0, 0);
    CHECK(write_container(&sp, 0), "write container");
    hx_modelfile *mf = hx_modelfile_open(sp.path, 1, err, sizeof err);
    CHECK(mf != NULL, "open: %s", err);
    if (mf) {
        int ne = sp.L * sp.E;
        float *heat = (float *)calloc((size_t)ne, sizeof(float));
        for (int k = 0; k < ne; k++) heat[k] = (float)(1 + (k * 7) % 13);
        snprintf(up, sizeof up, "%s.usage", sp.path);
        CHECK(write_usage(up, sp.L, sp.E, 100, heat), "write usage profile");
        for (size_t i = 0; i < sizeof v / sizeof v[0]; i++) {
            hx_store *st = open_store(mf, 100 * mf->slab_bytes_max, 1, 0, HEARTH_POLICY_LFU, up, NULL, v[i].pf, 0, NULL, 0,
                                      err, sizeof err);
            CHECK(st != NULL, "open: %s", err);
            if (!st) continue;
            hx_store_stats s;
            hx_store_get_stats(st, &s);
            CHECK(s.n_slots == 100 && s.pinned == v[i].want, "pin_fraction %.2f of %d slots: %d pinned, want %d",
                  (double)v[i].pf, s.n_slots, s.pinned, v[i].want);
            hx_store_close(st);
            tickle();
        }
        remove(up);
        free(heat);
        hx_modelfile_close(mf);
    }
    remove(sp.path);
    spec_free(&sp);
    printf("  pin count: int(pin_fraction * slots) as hearth.sim computes it (0.7 of 100 slots pins 70)\n");
}

/* Mirror validation compares header sections in 1 MiB chunks: an expert directory of
 * 1.25 MiB (5 x 8192 entries, all but one slab per layer aliased) whose copy differs
 * only after its first MiB must be refused; an identical copy is accepted. */
static void mirror_big_header(void) {
    char err[400], m1[800];
    spec sp;
    g_phase = "mirror with a large expert directory";
    make_spec(&sp, "hx_test_store_bigdir.hearth", 5, 8192, 64, 64, 2, 31, -1, 100, 0);
    CHECK(write_container(&sp, 0), "write container");
    hx_modelfile *mf = hx_modelfile_open(sp.path, 0, err, sizeof err);
    CHECK(mf != NULL, "open: %s", err);
    snprintf(m1, sizeof m1, "%s.mirror", sp.path);
    if (mf) {
        uint8_t pre[64];
        uint64_t edir_off = 0;
        FILE *f = fopen(sp.path, "rb");
        CHECK(f && fread(pre, 1, 64, f) == 64, "read preamble");
        if (f) fclose(f);
        memcpy(&edir_off, pre + 40, 8);
        for (int patched = 0; patched < 2; patched++) {
            CHECK(copy_file(sp.path, m1, patched ? (int64_t)(edir_off + (1u << 20) + 4 * 32 + 20) : -1, 0x02), "copy mirror");
            const char *ms[1] = {m1};
            hx_store *st = open_store(mf, 0, 1, 0, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, ms, 1, err, sizeof err);
            if (patched) CHECK(st == NULL && strstr(err, "differ"), "a mirror differing after the first MiB of its expert directory was accepted");
            else CHECK(st != NULL, "identical mirror with a large directory refused: %s", err);
            if (st) acq_rel(st, &sp, 8192 + 5, "big directory");
            hx_store_close(st);
        }
        hx_modelfile_close(mf);
    }
    remove(m1);
    remove(sp.path);
    spec_free(&sp);
    printf("  mirror with a 1.25 MiB expert directory: a copy differing after its first MiB is refused\n");
}

/* Containers whose layers are all dense still have an n_layers x n_experts heat
 * profile (FORMAT.md §8): n_experts 8 without an expert directory, n_experts 8 with
 * an all-empty one, and n_experts 0. The profile must load, be reported by
 * hx_store_heat and be written back whole. */
static void dense_usage(void) {
    static const struct { int E, K, no_edir; } v[3] = {{8, 2, 1}, {8, 2, 0}, {0, 0, 1}};
    char err[400], up[800], name[64];
    g_phase = "dense-only usage profile";
    for (int i = 0; i < 3; i++) {
        spec sp;
        snprintf(name, sizeof name, "hx_test_store_dense%d.hearth", i);
        make_spec(&sp, name, 3, v[i].E, 64, 64, v[i].K, 50 + (uint64_t)i, -1, 0, 0);
        for (int l = 0; l < sp.L; l++) sp.lk[l] = 0;
        sp.no_edir = v[i].no_edir;
        CHECK(write_container(&sp, 0), "write dense container %d", i);
        hx_modelfile *mf = hx_modelfile_open(sp.path, 1, err, sizeof err);
        CHECK(mf != NULL, "open dense container %d: %s", i, err);
        if (!mf) { remove(sp.path); spec_free(&sp); continue; }
        CHECK(mf->experts == NULL && mf->cfg.n_moe_layers == 0 && mf->cfg.n_experts == v[i].E, "dense container %d", i);
        int nh = sp.L * v[i].E;
        float *heat = (float *)calloc((size_t)(nh ? nh : 1), sizeof(float)), *back = (float *)calloc((size_t)(nh ? nh : 1), sizeof(float));
        for (int k = 0; k < nh; k++) heat[k] = (float)k + 0.5f;
        snprintf(up, sizeof up, "%s.usage", sp.path);
        CHECK(write_usage(up, sp.L, v[i].E, 7, heat), "write usage profile");
        for (int round = 0; round < 2; round++) {
            hx_store *st = open_store(mf, 1u << 20, 2, 0, HEARTH_POLICY_LFU, up, up, 0.5f, 1, NULL, 0, err, sizeof err);
            CHECK(st != NULL, "dense container %d, round %d: %s", i, round, err);
            if (!st) break;
            const float *h = hx_store_heat(st);
            int same = h != NULL;
            for (int k = 0; same && k < nh; k++) same = h[k] == heat[k];
            CHECK(same, "dense container %d, round %d: hx_store_heat differs from the profile", i, round);
            hx_store_stats s;
            hx_store_get_stats(st, &s);
            CHECK(s.n_slots == 0 && s.pinned == 0 && s.resident == 0, "dense container %d: %d slots", i, s.n_slots);
            CHECK(!hx_store_try_acquire(st, 0, 0) && !hx_store_acquire(st, 0, 0), "dense container %d: expert returned", i);
            int e0 = 0;
            hx_store_prefetch(st, 0, &e0, 1);
            hx_store_wait_any(st, 100);
            for (int t = 0; t < 3; t++) hx_store_tick(st);
            hx_store_close(st);
            uint64_t tok = 0;
            int ok = read_usage(up, sp.L, v[i].E, &tok, back);
            CHECK(ok, "dense container %d, round %d: usage_out is not a %d x %d profile", i, round, sp.L, v[i].E);
            same = ok && tok == 7 + 3 * (uint64_t)(round + 1);
            for (int k = 0; same && k < nh; k++) same = back[k] == heat[k];
            CHECK(same, "dense container %d, round %d: usage round trip (tokens %llu)", i, round, (unsigned long long)tok);
        }
        free(heat);
        free(back);
        remove(up);
        hx_modelfile_close(mf);
        remove(sp.path);
        spec_free(&sp);
        tickle();
    }
    printf("  dense-only containers (n_experts 8 without / with an empty expert directory, n_experts 0): "
           "usage profile round trip\n");
}

/* ------------------------------------------------------------ policy check */

static double policy_run(const spec *sp, hx_modelfile *mf, int policy, int tokens, const uint16_t *trace, int slots,
                         int *hot_key, double *hot_heat) {
    char err[400];
    hx_store *st = open_store(mf, (uint64_t)slots * mf->slab_bytes_max, 1, 0, policy, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
    CHECK(st != NULL, "open: %s", err);
    if (!st) return -1;
    int ne = sp->L * sp->E, warm = tokens / 5;
    double *h = (double *)calloc((size_t)ne, sizeof(double));
    const uint16_t *t = trace;
    for (int tok = 0; tok < tokens; tok++) {
        if (tok == warm) hx_store_reset_stats(st);
        for (int l = 0; l < sp->L; l++)
            for (int j = 0; j < sp->K; j++, t++) {
                const void *p = hx_store_acquire(st, l, *t);
                if (p) hx_store_release(st, l, *t);
            }
        for (int k = 0; k < ne; k++) h[k] *= 0.995;
        for (int l = 0; l < sp->L; l++)
            for (int j = 0; j < sp->K; j++) h[l * sp->E + t[-(sp->L - l) * sp->K + j]] += 1.0;
        hx_store_tick(st);
        tickle();
    }
    hx_store_stats s;
    hx_store_get_stats(st, &s);
    double rate = (double)s.hits / (double)(s.hits + s.misses);
    if (hot_key) {
        /* prefetching everything cold must not push out an expert that is used in most tokens */
        int best = 0;
        for (int k = 1; k < ne; k++) if (h[k] > h[best]) best = k;
        *hot_key = best;
        *hot_heat = h[best];
        CHECK(hx_store_is_resident(st, best / sp->E, best % sp->E), "hottest expert not resident under LFU");
        for (int l = 0; l < sp->L; l++) {
            int ids[64], n = 0;
            for (int e = 0; e < sp->E; e++) ids[n++] = e;
            for (int b = 0; b < n; b += 4) {
                hx_store_prefetch(st, l, ids + b, n - b < 4 ? n - b : 4);
                hx_store_wait_any(st, 20000);
            }
        }
        for (int w = 0; w < 50; w++) hx_store_wait_any(st, 2000);
        CHECK(h[best] < 120 || hx_store_is_resident(st, best / sp->E, best % sp->E),
              "a prefetch evicted the hottest expert (heat %.0f)", h[best]);
    }
    free(h);
    hx_store_close(st);
    return rate;
}

static void policy_check(int quick) {
    spec sp;
    char err[400];
    g_phase = "policy check";
    make_spec(&sp, "hx_test_store_policy.hearth", 8, 32, 64, 64, 4, 77, -1, 0, 0);
    CHECK(write_container(&sp, 0), "write policy container");
    hx_modelfile *mf = hx_modelfile_open(sp.path, 1, err, sizeof err);
    CHECK(mf != NULL, "open: %s", err);
    if (!mf) { spec_free(&sp); return; }
    int tokens = quick ? 300 : 800, slots = 16;
    /* Zipf(1.2) per layer over a per-layer permutation, top-k distinct */
    uint16_t *trace = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)tokens * sp.L * sp.K);
    double cdf[64], z = 0;
    for (int i = 0; i < sp.E; i++) { z += 1.0 / pow(i + 1, 1.2); cdf[i] = z; }
    uint64_t r = 2024;
    int perm[8][64];
    for (int l = 0; l < sp.L; l++) {
        for (int e = 0; e < sp.E; e++) perm[l][e] = e;
        for (int e = sp.E - 1; e > 0; e--) { int j = (int)rn(&r, (uint32_t)e + 1), t = perm[l][e]; perm[l][e] = perm[l][j]; perm[l][j] = t; }
    }
    uint16_t *t = trace;
    for (int tok = 0; tok < tokens; tok++)
        for (int l = 0; l < sp.L; l++) {
            int got[8], n = 0;
            while (n < sp.K) {
                double u = (double)(splitmix(&r) >> 11) / 9007199254740992.0 * z;
                int i = 0;
                while (cdf[i] < u) i++;
                int e = perm[l][i];
                if (!contains(got, n, e)) got[n++] = e;
            }
            for (int j = 0; j < n; j++) *t++ = (uint16_t)got[j];
        }
    int hot = -1;
    double hh = 0;
    double lru = policy_run(&sp, mf, HEARTH_POLICY_LRU, tokens, trace, slots, NULL, NULL);
    double lfu = policy_run(&sp, mf, HEARTH_POLICY_LFU, tokens, trace, slots, &hot, &hh);
    printf("  policy check: %d MoE layers x top-%d = %d slabs per token, cache %d slots, Zipf(1.2) routing, %d tokens:\n"
           "    LRU hit rate %.3f, LFU hit rate %.3f\n", sp.L, sp.K, sp.L * sp.K, slots, tokens, lru, lfu);
    CHECK(lru >= 0 && lru < 0.05, "LRU hit rate %.3f should collapse under cyclic access", lru);
    CHECK(lfu > 0.20 && lfu > 4 * lru, "LFU hit rate %.3f should be well above LRU %.3f", lfu, lru);
    free(trace);
    hx_modelfile_close(mf);
    remove(sp.path);
    spec_free(&sp);
}

/* --------------------------------------------------------------- benchmark */

static int writable_dir(const char *d) {
    char probe[700];
    snprintf(probe, sizeof probe, "%s/%sprobe", d, g_tag);
    FILE *f = fopen(probe, "wb");
    if (!f) return 0;
    fclose(f);
    remove(probe);
    return 1;
}

/* INV-DATA: the benchmark container (GiB) goes only into the data dir, chosen as
 * hxcc.py / hearth._native do: $HEARTH_DATA, else %LOCALAPPDATA%/hearth (Windows)
 * or $XDG_CACHE_HOME/hearth, ~/.cache/hearth. No fallback to temp (may be RAM-backed)
 * or the current directory (may be the repository). */
static int data_dir(char *out, size_t n, const char *arg) {
    const char *d = arg && *arg ? arg : hx_env_str("HEARTH_DATA");
    if (d && *d) snprintf(out, n, "%s", d);
#if defined(HX_OS_WINDOWS)
    else if ((d = hx_env_str("LOCALAPPDATA")) != NULL && *d) snprintf(out, n, "%s/hearth", d);
    else if ((d = hx_env_str("USERPROFILE")) != NULL && *d) snprintf(out, n, "%s/hearth", d);
#else
    else if ((d = hx_env_str("XDG_CACHE_HOME")) != NULL && *d) snprintf(out, n, "%s/hearth", d);
    else if ((d = hx_env_str("HOME")) != NULL && *d) snprintf(out, n, "%s/.cache/hearth", d);
#endif
    else {
        printf("  no data dir: set HEARTH_DATA or pass --dir\n");
        return 0;
    }
    if (writable_dir(out)) return 1;
    printf("  data dir %s is not writable (create it, set HEARTH_DATA or pass --dir)\n", out);
    return 0;
}

/* Small scratch files: an explicit or temp directory, never the current one. */
static int scratch_dir(const char *arg) {
    const char *c[] = {arg, hx_env_str("HEARTH_TEST_DIR"), hx_env_str("HEARTH_MUTATION_TMP"), hx_env_str("TEMP"),
                       hx_env_str("TMP"), hx_env_str("TMPDIR"),
#if defined(HX_OS_POSIX)
                       "/tmp",
#endif
                       NULL};
    for (size_t i = 0; i < sizeof c / sizeof c[0]; i++)
        if (c[i] && *c[i] && writable_dir(c[i])) {
            snprintf(g_dir, sizeof g_dir, "%s", c[i]);
            return 1;
        }
    return 0;
}

typedef struct { double gbps, lat_ms; uint64_t bytes; } bench_pt;

static bench_pt bench_once(hx_modelfile *mf, const spec *sp, const int *keys, int nkeys, int n_io, int direct) {
    bench_pt res = {0, 0, 0};
    char err[400];
    int window = 2 * n_io;
    uint64_t cache = (uint64_t)(window + 2 * sp->K + n_io + 4) * mf->slab_bytes_max;
    hx_store *st = open_store(mf, cache, n_io, direct, HEARTH_POLICY_LFU, NULL, NULL, 0, 0, NULL, 0, err, sizeof err);
    if (!st) { printf("  bench open failed: %s\n", err); return res; }
    uint64_t t0 = hx_now_ns();
    int issued = 0;
    for (int done = 0; done < nkeys; done++) {
        while (issued < nkeys && issued < done + window) {
            const void *p = hx_store_try_acquire(st, keys[issued] / sp->E, keys[issued] % sp->E);
            if (p) hx_store_release(st, keys[issued] / sp->E, keys[issued] % sp->E);
            issued++;
        }
        const void *p = hx_store_acquire(st, keys[done] / sp->E, keys[done] % sp->E);
        if (!p) { printf("  bench: read failed\n"); break; }
        hx_store_release(st, keys[done] / sp->E, keys[done] % sp->E);
        tickle();
    }
    uint64_t t1 = hx_now_ns();
    hx_store_stats s;
    hx_store_get_stats(st, &s);
    hx_store_close(st);
    res.bytes = s.bytes_read;
    res.gbps = (double)s.bytes_read / (double)(t1 - t0);
    res.lat_ms = s.reads ? (double)s.read_ns / (double)s.reads / 1e6 : 0;
    return res;
}

static int bench(double gib, const char *dir_arg) {
    char dir[600];
    if (!data_dir(dir, sizeof dir, dir_arg)) return 1;
    static const struct { const char *name; int D, F, E; } shapes[] = {
        {"Qwen3-30B-A3B Q4 expert (2048x768)", 2048, 768, 128},
        {"DeepSeek-V3/Kimi-K2 Q4 expert (7168x2048)", 7168, 2048, 64},
    };
    static const int ios[] = {1, 2, 4, 8, 16};
    printf("test_store --bench: %s, %.1f GiB synthetic container per slab size, data dir %s\n",
           hx_cpu_features()->brand, gib, dir);
    printf("  cold reads (direct: the whole file per column, never cached; buffered: each column reads a disjoint,\n"
           "  never-read fifth of the file, which was written unbuffered),\n"
           "  random expert order, demand window 2 x readers. GB = 1e9 bytes. Measured on a shared machine: noisy.\n");
    for (size_t si = 0; si < sizeof shapes / sizeof shapes[0]; si++) {
        uint64_t og, ou, od;
        uint64_t slab = hx_slab_layout(HEARTH_Q4, shapes[si].D, shapes[si].F, &og, &ou, &od);
        int E = shapes[si].E;
        int L = (int)((gib * (double)(1ull << 30)) / (double)slab / E + 0.5);
        if (L < 1) L = 1;
        if (L > 60) L = 60;
        spec sp;
        char name[64];
        snprintf(name, sizeof name, "hx_bench_store_%d.hearth", (int)si);
        snprintf(g_dir, sizeof g_dir, "%s", dir);
        make_spec(&sp, name, L, E, shapes[si].D, shapes[si].F, 8, 1234 + si, -1, 0, 0);
        g_phase = "bench write";
        uint64_t t0 = hx_now_ns();
        int ok = write_container(&sp, 1);
        double wsec = (double)(hx_now_ns() - t0) / 1e9;
        if (!ok) { printf("  cannot write %s\n", sp.path); remove(sp.path); spec_free(&sp); return 1; }
        char err[400];
        hx_modelfile *mf = hx_modelfile_open(sp.path, 1, err, sizeof err);
        if (!mf) { printf("  %s\n", err); remove(sp.path); spec_free(&sp); return 1; }
        int ne = L * E;
        printf("\n  %s: slab %.2f MiB, %d experts, %.2f GiB written in %.1f s (%.2f GB/s, unbuffered)\n",
               shapes[si].name, (double)slab / (1 << 20), ne, (double)sp.size / (1ull << 30), wsec,
               (double)sp.size / wsec / 1e9);
        int *keys = (int *)malloc(sizeof(int) * (size_t)ne);
        uint64_t r = 42 + si;
        for (int i = 0; i < ne; i++) keys[i] = i;
        for (int i = ne - 1; i > 0; i--) { int j = (int)rn(&r, (uint32_t)i + 1), t = keys[i]; keys[i] = keys[j]; keys[j] = t; }
        printf("    %-9s", "readers");
        for (size_t ii = 0; ii < sizeof ios / sizeof ios[0]; ii++) printf("  %8d", ios[ii]);
        printf("\n");
        for (int direct = 1; direct >= 0; direct--) {
            bench_pt pt[5];
            g_phase = "bench read";
            for (size_t ii = 0; ii < sizeof ios / sizeof ios[0]; ii++) {
                int part = ne / 5, start = (int)ii * part;
                pt[ii] = direct ? bench_once(mf, &sp, keys, ne, ios[ii], 1)
                                : bench_once(mf, &sp, keys + start, part, ios[ii], 0);
            }
            printf("    %-9s", direct ? "direct" : "buffered");
            for (int i = 0; i < 5; i++) printf("  %6.2f  ", pt[i].gbps);
            printf(" GB/s\n    %-9s", "");
            for (int i = 0; i < 5; i++) printf("  %6.1f  ", pt[i].lat_ms);
            printf(" ms per read (reader busy time)\n");
            printf("    %-9s", "");
            for (int i = 0; i < 5; i++) printf("  %6.2f  ", (double)pt[i].bytes / (1ull << 30));
            printf(" GiB read\n");
        }
        free(keys);
        hx_modelfile_close(mf);
        CHECK(remove(sp.path) == 0, "delete %s", sp.path);
        spec_free(&sp);
    }
    return 0;
}

/* -------------------------------------------------------------------- main */

int main(int argc, char **argv) {
    int quick = 0, do_bench = 0;
    double gib = 6.0;
    uint64_t seed = 1;
    const char *dir = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--quick")) quick = 1;
        else if (!strcmp(argv[i], "--bench")) do_bench = 1;
        else if (!strcmp(argv[i], "--gib") && i + 1 < argc) gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--dir") && i + 1 < argc) dir = argv[++i];
        else { printf("usage: test_store [--quick] [--seed N] [--dir D] | --bench [--gib N] [--dir D]\n"); return 2; }
    }
    hx_set_log_level(HX_LOG_ERROR);
    {
        uint64_t x = hx_now_ns() ^ (uint64_t)(uintptr_t)&quick;
        snprintf(g_tag, sizeof g_tag, "%08llx_", (unsigned long long)(splitmix(&x) & 0xFFFFFFFFull));
    }
    hx_thread *wd;
    atomic_init(&g_watch_stop, 0);
    hx_thread_create(&wd, watchdog, NULL);
    int rc;
    if (do_bench) {
        rc = bench(gib, dir);
    } else {
        uint64_t t0 = hx_now_ns();
        if (!scratch_dir(dir)) {
            printf("test_store: no writable scratch directory (set HEARTH_TEST_DIR or pass --dir)\n");
            return 2;
        }
        printf("test_store: seed %llu, scratch dir %s\n", (unsigned long long)seed, g_dir);
        fflush(stdout);
        spec sp;
        char err[400];
        /* 6 layers (layer 2 dense), 16 experts, top-4, mixed dtypes (slabs 8-48 KiB), ~15%% aliased */
        make_spec(&sp, "hx_test_store_a.hearth", 6, 16, 64, 64, 4, seed, 2, 15, 1);
        CHECK(write_container(&sp, 0), "write test container");
        hx_modelfile *mf = hx_modelfile_open(sp.path, 1, err, sizeof err);
        CHECK(mf != NULL, "open test container: %s", err);
        if (mf) {
            api_edges(&sp, mf);
            direct_vs_buffered(&sp, mf);
            promotion(&sp, mf);
            protection(&sp, mf);
            seeded_heat(&sp, mf);
            pf_queue_limit(&sp, mf);
            read_failures(&sp, mf);
            starvation(&sp, mf);
            wait_any_wakeups(&sp, mf);
            decode_loop(&sp, mf, quick);
            usage_tokens(&sp, mf);
            min_slots_burst(&sp, mf, quick);
            pin_cap(&sp, mf, quick);
            overcommit(&sp, mf);
            policy_details(&sp, mf);
            hot_rule(&sp, mf);
            stress(&sp, mf, quick, seed);
            usage_and_pins(&sp, mf);
            mirrors(&sp, mf);
            close_with_queue(&sp, mf);
            unreadable(&sp);
            hx_modelfile_close(mf);
        }
        remove(sp.path);
        spec_free(&sp);
        dense_usage();
        pin_count();
        mirror_big_header();
        policy_check(quick);
        printf("test_store: %d checks, %d failed (%.1f s)\n", atomic_load(&g_checks), atomic_load(&g_fail),
               (double)(hx_now_ns() - t0) / 1e9);
        rc = atomic_load(&g_fail) ? 1 : 0;
    }
    atomic_store(&g_watch_stop, 1);
    hx_thread_join(wd);
    return rc;
}
