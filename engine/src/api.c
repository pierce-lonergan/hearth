/*
 * api.c — the engine half of the public C API (engine/include/hearth.h); the
 * quantization utilities live in quant.c.
 *
 * Every entry point checks its arguments and never crashes on bad input: NULL
 * handles or pointers, n < 0, token ids outside [0, vocab), pos + n beyond the KV
 * capacity and rewinds past the current position return negative codes
 * (hx_model.h) and leave the engine unchanged.
 *
 * hearth_open: caller options, then environment overrides (HEARTH_ISA,
 * HEARTH_THREADS, HEARTH_IO_THREADS, HEARTH_CACHE_GB; HEARTH_LOG sets the log
 * level, otherwise verbose 1/2 raises it to info/debug while the engine is open),
 * then validation and resolution of the 0 = default values (n_threads: physical
 * cores, capped at the logical CPUs; n_io_threads 8; max_batch 512, at most 4096).
 * Anything that cannot be honoured, such as an ISA this CPU cannot run, fails
 * with a message.
 *
 * The log level is process-wide. Engines opened with verbose > 0 raise it to the
 * most verbose level any of them asked for; when the last of them closes, the level
 * the host had before is restored. If the host set a level in between, that level
 * is left alone.
 */
#include "hx_model.h"
#include "hx_quant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_THREADS    256
#define MAX_IO_THREADS 64
#define MAX_BATCH      4096

struct hearth_engine {
    hx_model *m;
    int verbose;                 /* 1/2: counted in g_log_n */
};

/* Engines may be opened and closed from different threads: a tiny spin lock. */
static atomic_flag g_log_lock = ATOMIC_FLAG_INIT;
static int g_log_n[2];           /* open engines that asked for info / debug */
static int g_log_base;           /* the host's level before they raised it */
static int g_log_set = -1;       /* the level they set last */

static void log_lock(void) {
    while (atomic_flag_test_and_set(&g_log_lock)) hx_yield();
}
static void log_unlock(void) { atomic_flag_clear(&g_log_lock); }

/* Under the lock. */
static void log_apply(void) {
    int lvl = g_log_base;
    if (g_log_n[1] && lvl < HX_LOG_DEBUG) lvl = HX_LOG_DEBUG;
    else if (g_log_n[0] && lvl < HX_LOG_INFO) lvl = HX_LOG_INFO;
    hx_set_log_level(lvl);
    g_log_set = lvl;
}

static void log_engine_open(int verbose) {
    log_lock();
    if ((g_log_n[0] == 0 && g_log_n[1] == 0) || hx_get_log_level() != g_log_set) g_log_base = hx_get_log_level();
    g_log_n[verbose - 1]++;
    log_apply();
    log_unlock();
}

static void log_engine_close(int verbose) {
    log_lock();
    g_log_n[verbose - 1]--;
    if (hx_get_log_level() == g_log_set) log_apply();
    log_unlock();
}

static int lower_eq(const char *a, const char *b) {
    for (; *a && *b; a++, b++) {
        char c = (*a >= 'A' && *a <= 'Z') ? (char)(*a - 'A' + 'a') : *a;
        if (c != *b) return 0;
    }
    return *a == 0 && *b == 0;
}

int hx_parse_isa(const char *s, int *out) {
    static const char *names[] = {"auto", "scalar", "avx2", "avx512"};
    while (*s == ' ') s++;
    for (int i = 0; i < 4; i++)
        if (lower_eq(s, names[i])) { *out = i; return 1; }
    if (s[0] >= '0' && s[0] <= '3' && s[1] == 0) { *out = s[0] - '0'; return 1; }
    return 0;
}

static const char *isa_name(int isa) {
    static const char *names[] = {"auto", "scalar", "avx2", "avx512"};
    return isa >= 0 && isa <= 3 ? names[isa] : "?";
}

hx_model *hx_engine_model(hearth_engine *e) { return e ? e->m : NULL; }

HEARTH_API void hearth_default_options(hearth_options *opt) {
    if (!opt) return;
    memset(opt, 0, sizeof *opt);
    opt->cache_gb = 8.0;
    opt->direct_io = 1;
    opt->policy = HEARTH_POLICY_LFU;
    opt->prefetch = HEARTH_PREFETCH_SHARED;
    opt->isa = HEARTH_ISA_AUTO;
}

HEARTH_API hearth_engine *hearth_open(const hearth_options *in, char *err, size_t errlen) {
    hearth_options o;
    const char *s;
    hx_model *m;
    hearth_engine *e;
    int verbose = 0;
    if (err && errlen) err[0] = 0;
    if (!in) return (hearth_engine *)hx_fail(err, errlen, "hearth_open: options are NULL");
    o = *in;

    /* HEARTH_LOG always wins; otherwise verbose raises the level while the engine is open */
    s = hx_env_str("HEARTH_LOG");
    if (s && *s) hx_set_log_level(hx_env_int("HEARTH_LOG", HX_LOG_WARN));
    else if (o.verbose > 0) verbose = o.verbose == 1 ? 1 : 2;

    s = hx_env_str("HEARTH_ISA");
    if (s && *s && !hx_parse_isa(s, &o.isa))
        return (hearth_engine *)hx_fail(err, errlen, "HEARTH_ISA=%s is not one of auto, scalar, avx2, avx512", s);
    o.n_threads = hx_env_int("HEARTH_THREADS", o.n_threads);
    o.n_io_threads = hx_env_int("HEARTH_IO_THREADS", o.n_io_threads);
    o.cache_gb = hx_env_double("HEARTH_CACHE_GB", o.cache_gb);

    if (!o.model_path || !*o.model_path) return (hearth_engine *)hx_fail(err, errlen, "hearth_open: no model path");
    if (o.n_mirrors < 0 || o.n_mirrors > 8)
        return (hearth_engine *)hx_fail(err, errlen, "hearth_open: n_mirrors %d outside 0..8", o.n_mirrors);
    for (int i = 0; i < o.n_mirrors; i++)
        if (!o.mirror_paths[i] || !*o.mirror_paths[i])
            return (hearth_engine *)hx_fail(err, errlen, "hearth_open: mirror %d has no path", i);
    if (!(o.cache_gb >= 0.0) || isinf(o.cache_gb))
        return (hearth_engine *)hx_fail(err, errlen, "hearth_open: cache_gb must be a finite number >= 0");
    if (!(o.pin_fraction >= 0.0f) || o.pin_fraction > 1.0f)
        return (hearth_engine *)hx_fail(err, errlen, "hearth_open: pin_fraction must be in [0, 1]");
    if (o.n_threads < 0 || o.n_io_threads < 0 || o.max_seq < 0 || o.max_batch < 0 || o.prefetch_extra < 0)
        return (hearth_engine *)hx_fail(err, errlen,
                                        "hearth_open: n_threads, n_io_threads, max_seq, max_batch and prefetch_extra must be >= 0");
    if (o.policy != HEARTH_POLICY_LRU && o.policy != HEARTH_POLICY_LFU)
        return (hearth_engine *)hx_fail(err, errlen, "hearth_open: unknown cache policy %d", o.policy);
    if (o.prefetch < HEARTH_PREFETCH_OFF || o.prefetch > HEARTH_PREFETCH_SHARED)
        return (hearth_engine *)hx_fail(err, errlen, "hearth_open: unknown prefetch mode %d", o.prefetch);
    if (o.isa < HEARTH_ISA_AUTO || o.isa > HEARTH_ISA_AVX512)
        return (hearth_engine *)hx_fail(err, errlen, "hearth_open: unknown ISA %d", o.isa);

    if (o.isa == HEARTH_ISA_AUTO) o.isa = hearth_cpu_isa();
    else if (!hx_kernels_for(o.isa))
        return (hearth_engine *)hx_fail(err, errlen, "hearth_open: ISA %s is not supported by this CPU or build",
                                        isa_name(o.isa));
    if (verbose) log_engine_open(verbose);   /* undone below if the open fails */
    {
        const int cpus = hx_num_cpus() > 0 ? hx_num_cpus() : 1;
        const int cores = hx_num_physical_cores() > 0 ? hx_num_physical_cores() : cpus;
        if (o.n_threads == 0) o.n_threads = cores;
        /* every parallel region waits for all compute threads, so a thread that has
         * no CPU stalls the others: never more threads than logical CPUs */
        if (o.n_threads > cpus) {
            hx_log(HX_LOG_WARN, "n_threads %d capped at the %d logical CPUs", o.n_threads, cpus);
            o.n_threads = cpus;
        }
        if (o.n_threads > MAX_THREADS) {
            hx_log(HX_LOG_WARN, "n_threads %d capped at %d", o.n_threads, MAX_THREADS);
            o.n_threads = MAX_THREADS;
        }
        if (o.n_threads > cores)
            hx_log(HX_LOG_INFO, "%d threads on %d physical cores: SMT threads are used for large batches only",
                   o.n_threads, cores);
    }
    if (o.n_io_threads == 0) o.n_io_threads = 8;
    if (o.n_io_threads > MAX_IO_THREADS) {
        hx_log(HX_LOG_WARN, "n_io_threads %d capped at %d", o.n_io_threads, MAX_IO_THREADS);
        o.n_io_threads = MAX_IO_THREADS;
    }
    if (o.max_batch == 0) o.max_batch = 512;
    if (o.max_batch > MAX_BATCH) o.max_batch = MAX_BATCH;
    o.direct_io = o.direct_io ? 1 : 0;
    o.warm_start = o.warm_start ? 1 : 0;

    m = hx_model_open(&o, err, errlen);
    e = m ? (hearth_engine *)calloc(1, sizeof *e) : NULL;
    if (!e) {
        if (m) {
            hx_model_close(m);
            hx_fail(err, errlen, "out of memory");
        }
        if (verbose) log_engine_close(verbose);
        return NULL;
    }
    e->m = m;
    e->verbose = verbose;
    return e;
}

HEARTH_API void hearth_close(hearth_engine *e) {
    if (!e) return;
    hx_model_close(e->m);
    if (e->verbose) log_engine_close(e->verbose);
    free(e);
}

HEARTH_API int hearth_info(hearth_engine *e, hearth_model_info *out) {
    if (!e || !out) return HX_E_ARG;
    hx_model_info(e->m, out);
    return HX_OK;
}

HEARTH_API int hearth_eval(hearth_engine *e, const int32_t *tokens, int n, float *logits, int all_logits) {
    int V;
    if (!e || n < 0) return HX_E_ARG;
    if (n == 0) return HX_OK;
    if (!tokens) return HX_E_ARG;
    V = hx_model_vocab(e->m);
    for (int i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= V) return HX_E_TOKEN;
    if ((int64_t)hx_model_pos(e->m) + n > (int64_t)hx_model_capacity(e->m)) return HX_E_CAPACITY;
    return hx_model_eval(e->m, tokens, n, logits, all_logits ? 1 : 0);
}

HEARTH_API int hearth_pos(hearth_engine *e) { return e ? hx_model_pos(e->m) : HX_E_ARG; }

HEARTH_API int hearth_reset(hearth_engine *e) {
    if (!e) return HX_E_ARG;
    hx_model_set_pos(e->m, 0);
    return HX_OK;
}

HEARTH_API int hearth_rewind(hearth_engine *e, int pos) {
    if (!e || pos < 0) return HX_E_ARG;
    if (pos > hx_model_pos(e->m)) return HX_E_CAPACITY;
    hx_model_set_pos(e->m, pos);
    return HX_OK;
}

HEARTH_API int hearth_get_stats(hearth_engine *e, hearth_stats *out) {
    if (!e || !out) return HX_E_ARG;
    hx_model_get_stats(e->m, out);
    return HX_OK;
}

HEARTH_API void hearth_reset_stats(hearth_engine *e) {
    if (e) hx_model_reset_stats(e->m);
}

HEARTH_API int hearth_trace_start(hearth_engine *e, const char *path) {
    if (!e || !path || !*path) return HX_E_ARG;
    return hx_model_trace_start(e->m, path);
}

HEARTH_API int hearth_trace_stop(hearth_engine *e) {
    if (!e) return HX_E_ARG;
    return hx_model_trace_stop(e->m);
}

HEARTH_API int hearth_route_replay(hearth_engine *e, const char *trace_path) {
    if (!e) return HX_E_ARG;
    return hx_model_route_replay(e->m, trace_path);
}
