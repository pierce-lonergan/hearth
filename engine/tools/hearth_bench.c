/*
 * hearth_bench.c — prefill + greedy-decode throughput of one container.
 *
 *   hearth-bench <model.hearth> [--tokens N] [--prompt N] [--warmup N]
 *                [--cache-gb X] [--threads N] [--io-threads N] [--policy lru|lfu]
 *                [--prefetch off|next|shared] [--prefetch-extra N] [--direct 0|1]
 *                [--usage-in f] [--usage-out f] [--pin f] [--warm 0|1] [--mirror f]...
 *                [--replay trace] [--trace out] [--isa auto|scalar|avx2|avx512]
 *                [--max-batch N] [--max-seq N] [--seed N] [--verbose N]
 *
 * The prompt is N pseudo-random token ids (seeded) evaluated in one hearth_eval
 * call; decoding then feeds back the argmax token. --warmup decodes that many
 * tokens before the timed decode. With --replay the routing comes from a trace
 * (hearth_route_replay), so the output is not the model's: use it to measure
 * throughput under real routing skew on synthetic-weight containers, and say so
 * when reporting (INV-HONEST).
 *
 * engine/tests/test_model.c compiles this file with HEARTH_BENCH_NO_MAIN and runs
 * bench_main() on a tiny container as a smoke test.
 */
#include "hearth.h"
#include "hx_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void bench_usage(void) {
    fprintf(stderr,
            "usage: hearth-bench <model.hearth> [--tokens N] [--prompt N] [--warmup N] [--cache-gb X]\n"
            "         [--threads N] [--io-threads N] [--policy lru|lfu] [--prefetch off|next|shared]\n"
            "         [--prefetch-extra N] [--direct 0|1] [--usage-in f] [--usage-out f] [--pin f]\n"
            "         [--warm 0|1] [--mirror f]... [--replay trace] [--trace out]\n"
            "         [--isa auto|scalar|avx2|avx512] [--max-batch N] [--seed N] [--verbose N]\n");
}

static int bench_pick(const char *v, const char *const *names, int n) {
    for (int i = 0; i < n; i++)
        if (!strcmp(v, names[i])) return i;
    return -1;
}

static int bench_argmax(const float *x, int n) {
    int b = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[b]) b = i;
    return b;
}

/* Hits per (token, layer, rank) activation, and per unique (call, layer, expert) load:
 * a batch acquires each expert once and counts at most one miss for it, so for
 * prefill the second says how often a needed slab was already in DRAM. Percent. */
static double bench_hit_rate(const hearth_stats *s) {
    const uint64_t acc = s->cache_hits + s->cache_misses;
    return acc ? 100.0 * (double)s->cache_hits / (double)acc : 0.0;
}
static double bench_unique_hit_rate(const hearth_stats *s) {
    if (!s->expert_loads_unique || s->cache_misses >= s->expert_loads_unique) return 0.0;
    return 100.0 * (1.0 - (double)s->cache_misses / (double)s->expert_loads_unique);
}

static void bench_report(const char *what, int n, double secs, const hearth_stats *s) {
    const double tps = secs > 0 ? n / secs : 0.0;
    const double gb = (double)s->bytes_read / 1e9;
    const double wall = s->wall_s > 0 ? s->wall_s : secs;
    printf("%-8s %6d tok %9.3f s %9.2f tok/s | hit %5.1f%% (unique loads %5.1f%%) | read %7.3f GB %6.2f GB/s | stall %5.1f%%\n",
           what, n, secs, tps, bench_hit_rate(s), bench_unique_hit_rate(s), gb, wall > 0 ? gb / wall : 0.0,
           wall > 0 ? 100.0 * s->stall_s / wall : 0.0);
    printf("         phases: attention %.3f s, moe %.3f s (stall %.3f s), dense+head %.3f s; per token %.2f ms\n",
           s->attn_s, s->moe_s, s->stall_s, s->dense_s, n ? 1e3 * wall / n : 0.0);
    printf("         experts: %llu uses, %llu unique loads, %llu hits, %llu misses, %llu evictions; "
           "prefetch %llu issued / %llu used / %llu wasted; reader busy %.3f s; %llu read errors\n",
           (unsigned long long)s->expert_uses, (unsigned long long)s->expert_loads_unique,
           (unsigned long long)s->cache_hits, (unsigned long long)s->cache_misses, (unsigned long long)s->evictions,
           (unsigned long long)s->prefetch_issued, (unsigned long long)s->prefetch_used,
           (unsigned long long)s->prefetch_wasted, s->read_s, (unsigned long long)s->read_errors);
}

static int bench_main(int argc, char **argv) {
    static const char *const policies[] = {"lru", "lfu"};
    static const char *const prefetches[] = {"off", "next", "shared"};
    static const char *const isas[] = {"auto", "scalar", "avx2", "avx512"};
    hearth_options o;
    hearth_model_info info;
    hearth_stats sp, sd;
    const char *replay = NULL, *trace = NULL;
    int n_tok = 64, n_prompt = 32, n_warm = 0, max_seq_set = 0;
    unsigned long long seed = 1;
    char err[1024];
    hearth_engine *e;
    float *logits;
    int32_t *prompt;
    int rc = 0, next;
    uint64_t t0;
    double prefill_s = 0.0, decode_s = 0.0;

    hearth_default_options(&o);
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        int k;
        if (a[0] != '-') {
            if (o.model_path) { bench_usage(); return 2; }
            o.model_path = a;
            continue;
        }
        if (!v) { bench_usage(); return 2; }
        i++;
        if (!strcmp(a, "--tokens")) n_tok = atoi(v);
        else if (!strcmp(a, "--prompt")) n_prompt = atoi(v);
        else if (!strcmp(a, "--warmup")) n_warm = atoi(v);
        else if (!strcmp(a, "--cache-gb")) o.cache_gb = atof(v);
        else if (!strcmp(a, "--threads")) o.n_threads = atoi(v);
        else if (!strcmp(a, "--io-threads")) o.n_io_threads = atoi(v);
        else if (!strcmp(a, "--policy") && (k = bench_pick(v, policies, 2)) >= 0) o.policy = k;
        else if (!strcmp(a, "--prefetch") && (k = bench_pick(v, prefetches, 3)) >= 0) o.prefetch = k;
        else if (!strcmp(a, "--prefetch-extra")) o.prefetch_extra = atoi(v);
        else if (!strcmp(a, "--direct")) o.direct_io = atoi(v) != 0;
        else if (!strcmp(a, "--usage-in")) o.usage_in = v;
        else if (!strcmp(a, "--usage-out")) o.usage_out = v;
        else if (!strcmp(a, "--pin")) o.pin_fraction = (float)atof(v);
        else if (!strcmp(a, "--warm")) o.warm_start = atoi(v) != 0;
        else if (!strcmp(a, "--mirror") && o.n_mirrors < 8) o.mirror_paths[o.n_mirrors++] = v;
        else if (!strcmp(a, "--replay")) replay = v;
        else if (!strcmp(a, "--trace")) trace = v;
        else if (!strcmp(a, "--isa") && (k = bench_pick(v, isas, 4)) >= 0) o.isa = k;
        else if (!strcmp(a, "--max-batch")) o.max_batch = atoi(v);
        else if (!strcmp(a, "--max-seq")) { o.max_seq = atoi(v); max_seq_set = 1; }
        else if (!strcmp(a, "--seed")) seed = strtoull(v, NULL, 10);
        else if (!strcmp(a, "--verbose")) o.verbose = atoi(v);
        else { fprintf(stderr, "hearth-bench: bad option %s %s\n", a, v); bench_usage(); return 2; }
    }
    if (!o.model_path || n_tok < 0 || n_prompt < 0 || n_warm < 0) { bench_usage(); return 2; }
    if (!max_seq_set) o.max_seq = n_prompt + n_warm + n_tok + 1;

    e = hearth_open(&o, err, sizeof err);
    if (!e) { fprintf(stderr, "hearth-bench: %s\n", err); return 1; }
    hearth_info(e, &info);
    printf("hearth-bench %s: %s (%s), %d layers (%d MoE, %d experts, top-%d), %.2fB params, %.2fB active\n",
           hearth_version(), o.model_path, info.arch, info.n_layers, info.n_moe_layers, info.n_experts, info.top_k,
           info.params_total / 1e9, info.params_active / 1e9);
    printf("  isa %s, %d threads, %d I/O threads, cache %d slots x %.2f MiB = %.2f GiB (experts %.2f GiB), %s, prefetch %s+%d, %s I/O%s\n",
           isas[info.isa >= 0 && info.isa <= 3 ? info.isa : 0], info.n_threads, info.n_io_threads, info.cache_slots,
           (double)info.slab_bytes_max / (1 << 20), (double)info.cache_slots * (double)info.slab_bytes_max / (1u << 30),
           (double)info.expert_bytes / (1u << 30), policies[o.policy], prefetches[o.prefetch], o.prefetch_extra,
           o.direct_io ? "direct" : "buffered", replay ? ", routing REPLAYED from a trace (synthetic workload)" : "");
    fflush(stdout);
    if (replay && hearth_route_replay(e, replay) != 0) {
        fprintf(stderr, "hearth-bench: cannot replay %s\n", replay);
        hearth_close(e);
        return 1;
    }
    if (trace && hearth_trace_start(e, trace) != 0) {
        fprintf(stderr, "hearth-bench: cannot write trace %s\n", trace);
        hearth_close(e);
        return 1;
    }

    logits = (float *)malloc(sizeof(float) * (size_t)info.vocab_size);
    prompt = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n_prompt ? n_prompt : 1));
    if (!logits || !prompt) {
        fprintf(stderr, "hearth-bench: out of memory\n");
        rc = -1;
        goto done;
    }
    for (int i = 0; i < n_prompt; i++) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        prompt[i] = (int32_t)((seed >> 33) % (unsigned long long)info.vocab_size);
    }

    hearth_reset_stats(e);
    next = info.bos_id >= 0 ? info.bos_id : 1 % info.vocab_size;
    if (n_prompt) {
        t0 = hx_now_ns();
        rc = hearth_eval(e, prompt, n_prompt, logits, 0);
        prefill_s = (double)(hx_now_ns() - t0) * 1e-9;
        if (rc) { fprintf(stderr, "hearth-bench: prefill failed (%d)\n", rc); goto done; }
        next = bench_argmax(logits, info.vocab_size);
    }
    hearth_get_stats(e, &sp);
    for (int i = 0; i < n_warm && rc == 0; i++) {
        rc = hearth_eval(e, &next, 1, logits, 0);
        next = bench_argmax(logits, info.vocab_size);
    }
    if (rc) { fprintf(stderr, "hearth-bench: warmup failed (%d)\n", rc); goto done; }
    hearth_reset_stats(e);
    t0 = hx_now_ns();
    for (int i = 0; i < n_tok && rc == 0; i++) {
        rc = hearth_eval(e, &next, 1, logits, 0);
        next = bench_argmax(logits, info.vocab_size);
    }
    decode_s = (double)(hx_now_ns() - t0) * 1e-9;
    if (rc) { fprintf(stderr, "hearth-bench: decode failed (%d)\n", rc); goto done; }
    hearth_get_stats(e, &sd);
    if (n_prompt) bench_report("prefill", n_prompt, prefill_s, &sp);
    if (n_tok) bench_report("decode", n_tok, decode_s, &sd);
    printf("  cache: %d slots, %d resident, %d pinned\n", sd.cache_slots, sd.cache_resident, sd.cache_pinned);
done:
    if (trace) hearth_trace_stop(e);
    hearth_close(e);
    free(logits);
    free(prompt);
    return rc ? 1 : 0;
}

#ifndef HEARTH_BENCH_NO_MAIN
int main(int argc, char **argv) { return bench_main(argc, argv); }
#endif
