/*
 * hearth.h — public C API of the Hearth inference engine.
 *
 * Hearth runs very large Mixture-of-Experts language models on ordinary
 * hardware: the resident "backbone" (attention, norms, routers, shared experts)
 * lives in RAM, and the routed experts are streamed on demand from NVMe into a
 * DRAM expert cache. Output never depends on where a weight came from
 * (see docs/NUMERICS.md).
 *
 * This header is the stable ABI used by the Python package (ctypes) and by
 * third-party bindings. All functions are thread-compatible but an engine
 * handle must not be used from two threads at the same time.
 */
#ifndef HEARTH_H
#define HEARTH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  if defined(HEARTH_BUILD_DLL)
#    define HEARTH_API __declspec(dllexport)
#  elif defined(HEARTH_USE_DLL)
#    define HEARTH_API __declspec(dllimport)
#  else
#    define HEARTH_API
#  endif
#else
#  define HEARTH_API __attribute__((visibility("default")))
#endif

#define HEARTH_VERSION_MAJOR 0
#define HEARTH_VERSION_MINOR 1
#define HEARTH_VERSION_PATCH 0

/* ---- dtypes (docs/FORMAT.md §6) -------------------------------------- */
enum {
    HEARTH_F32 = 0, HEARTH_F16 = 1, HEARTH_BF16 = 2,
    HEARTH_Q8 = 3, HEARTH_Q4 = 4, HEARTH_I32 = 5, HEARTH_U8 = 6
};

/* ---- expert cache eviction policies ----------------------------------- */
enum {
    HEARTH_POLICY_LRU = 0, /* classic global LRU (baseline; pathological under cyclic layer access) */
    HEARTH_POLICY_LFU = 1  /* decayed-frequency ("heat") with sampled eviction — default */
};

/* ---- prefetch modes ---------------------------------------------------- */
enum {
    HEARTH_PREFETCH_OFF = 0,
    HEARTH_PREFETCH_NEXT = 1,   /* route layer L+1 with layer L's post-attention state */
    HEARTH_PREFETCH_SHARED = 2  /* two-step: add shared-expert output first (falls back to NEXT) */
};

/* ---- ISA selection ------------------------------------------------------ */
enum { HEARTH_ISA_AUTO = 0, HEARTH_ISA_SCALAR = 1, HEARTH_ISA_AVX2 = 2, HEARTH_ISA_AVX512 = 3 };

typedef struct hearth_engine hearth_engine;

typedef struct hearth_options {
    const char *model_path;        /* path to the .hearth file (UTF-8) */
    const char *mirror_paths[8];   /* optional byte-identical copies on other drives */
    int n_mirrors;
    double cache_gb;               /* DRAM budget for routed experts in GiB (default 8.0); clamped up to a safe minimum */
    int n_threads;                 /* compute threads incl. caller (0 = number of physical cores) */
    int n_io_threads;              /* expert reader threads (0 = 8) */
    int direct_io;                 /* 1 = bypass OS page cache (default), 0 = buffered */
    int policy;                    /* HEARTH_POLICY_* (default LFU) */
    int prefetch;                  /* HEARTH_PREFETCH_* (default SHARED) */
    int prefetch_extra;            /* predict top_k + extra experts for the next layer (default 0) */
    const char *usage_in;          /* heat profile to seed the cache / pin hot experts (optional) */
    const char *usage_out;         /* write updated heat profile here on close (optional) */
    float pin_fraction;            /* fraction of cache slots pinned with the hottest experts at open (default 0.0) */
    int warm_start;                /* 1 = at open, fill the unpinned cache with the next-hottest experts (default 0) */
    int max_seq;                   /* KV capacity in tokens (0 = min(model max_seq, 4096)) */
    int max_batch;                 /* max tokens per hearth_eval call (0 = 512) */
    int isa;                       /* HEARTH_ISA_* (default AUTO; env HEARTH_ISA overrides) */
    int verbose;                   /* 0 quiet, 1 info, 2 debug (to stderr) */
} hearth_options;

typedef struct hearth_model_info {
    char arch[32];
    int n_layers, d_model, vocab_size, max_seq;
    int n_heads, n_kv_heads, head_dim, attn_kind;
    int n_experts, top_k, expert_ffn_dim, n_moe_layers;
    int bos_id;                     /* -1 if none */
    int n_eos; int eos_ids[8];
    uint64_t dense_bytes;           /* resident backbone bytes */
    uint64_t expert_bytes;          /* routed expert region bytes (sum of slab sizes) */
    uint64_t slab_bytes_max;        /* largest expert slab */
    double params_total;            /* parameter count (all experts) */
    double params_active;           /* parameters touched per token */
    int cache_slots;                /* expert cache capacity in slabs */
    int isa;                        /* HEARTH_ISA_* actually in use */
    int n_threads, n_io_threads;
} hearth_model_info;

typedef struct hearth_stats {
    uint64_t tokens;                /* tokens evaluated since reset */
    uint64_t forward_calls;
    double   wall_s;                /* time inside hearth_eval */
    double   attn_s, moe_s, dense_s, stall_s; /* stall = compute waiting on expert I/O */
    uint64_t expert_uses;           /* (token, layer, expert) activations */
    uint64_t expert_loads_unique;   /* distinct (layer, expert) fetches needed per forward call, summed */
    uint64_t cache_hits, cache_misses;
    uint64_t prefetch_issued, prefetch_used, prefetch_wasted;
    uint64_t bytes_read;            /* bytes read from storage for experts */
    double   read_s;                /* summed reader-thread busy time */
    uint64_t evictions;
    int      cache_slots, cache_resident, cache_pinned;
} hearth_stats;

/* ---- lifecycle --------------------------------------------------------- */
HEARTH_API const char *hearth_version(void);
HEARTH_API void hearth_default_options(hearth_options *opt);
/* Returns NULL on failure and writes a message into err (if err != NULL). */
HEARTH_API hearth_engine *hearth_open(const hearth_options *opt, char *err, size_t errlen);
HEARTH_API void hearth_close(hearth_engine *e);
HEARTH_API int hearth_info(hearth_engine *e, hearth_model_info *out);

/* ---- evaluation ---------------------------------------------------------
 * Evaluate tokens[0..n) at positions [pos, pos+n) where pos = hearth_pos(e),
 * appending them to the KV cache. n may exceed max_batch (processed in chunks).
 * logits: NULL, or room for vocab floats (all_logits = 0: last token only) or
 * n*vocab floats (all_logits = 1, row-major per token).
 * Returns 0 on success, negative on error (e.g. KV capacity exceeded). */
HEARTH_API int hearth_eval(hearth_engine *e, const int32_t *tokens, int n, float *logits, int all_logits);
HEARTH_API int hearth_pos(hearth_engine *e);
HEARTH_API int hearth_reset(hearth_engine *e);            /* pos = 0 */
HEARTH_API int hearth_rewind(hearth_engine *e, int pos);  /* truncate KV to pos (<= current) */

/* ---- telemetry --------------------------------------------------------- */
HEARTH_API int hearth_get_stats(hearth_engine *e, hearth_stats *out);
HEARTH_API void hearth_reset_stats(hearth_engine *e);
/* Record routing decisions to a trace file (docs/FORMAT.md §9) for the simulator. */
HEARTH_API int hearth_trace_start(hearth_engine *e, const char *path);
HEARTH_API int hearth_trace_stop(hearth_engine *e);
/* Benchmark hook: replace computed routing with ids replayed from a trace
 * (cycling). Output is then NOT the model's output; used only to measure
 * throughput under realistic routing skew on synthetic-weight containers. */
HEARTH_API int hearth_route_replay(hearth_engine *e, const char *trace_path);

/* ---- quantization utilities (used by the converter) -------------------- */
HEARTH_API size_t hearth_row_bytes(int dtype, int64_t n_cols);
/* Quantize/convert n_rows x n_cols f32 rows into dtype. n_threads 0 = auto. Returns 0 on success. */
HEARTH_API int hearth_quantize(int dtype, const float *src, int64_t n_rows, int64_t n_cols, void *dst, int n_threads);
HEARTH_API int hearth_dequantize(int dtype, const void *src, int64_t n_rows, int64_t n_cols, float *dst);
/* Reference matvec/matmul through the dispatched kernels (for tests/tools).
 * X is T x n_cols f32 (row-major), Y is T x n_rows. Follows docs/NUMERICS.md §3. */
HEARTH_API int hearth_matmul(int dtype, const void *W, int64_t n_rows, int64_t n_cols,
                             const float *X, int T, float *Y, int isa);
HEARTH_API int hearth_cpu_isa(void); /* best ISA supported by this CPU */

#ifdef __cplusplus
}
#endif
#endif /* HEARTH_H */
