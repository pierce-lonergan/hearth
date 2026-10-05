/*
 * hx_model.h — the forward pass behind hearth.h (model.c).
 *
 * api.c resolves defaults and environment overrides into a hearth_options and
 * validates every public argument; model.c trusts what it is given except for
 * the container, which it checks tensor by tensor (FORMAT.md §4.1) at open.
 *
 * Numerics: docs/NUMERICS.md §5. Evaluating T tokens in one call is bit-identical
 * to evaluating them one at a time (INV-DET-2), and output never depends on
 * thread count, I/O scheduling or cache state (INV-DET-1): every matmul output
 * element is produced by one kernel call whose result does not depend on T or on
 * the row range, attention for each token follows the single-token arithmetic,
 * and routed expert results land in per-(token, rank) buffers summed in rank order.
 *
 * Threads: batches of fewer than 256 tokens (decode, speculative verification)
 * run on min(n_threads, physical cores) threads, larger ones on n_threads; small
 * regions of work stay on the calling thread. Neither choice changes a result.
 *
 * Routing replay (benchmarks): the token at absolute position p uses trace row
 * p mod n_rows instead of the router's top-k; its gate weights are this model's
 * own scores of the replayed experts, then norm_topk_prob and routed_scale apply
 * as usual (the same rule as hearth.reference.Reference.replay_routes). The trace
 * header must match n_experts, top_k and n_moe_layers; its n_layers only has to
 * be >= n_moe_layers.
 *
 * Usage counting: the store counts one use per (token, rank) activation, also in
 * a batch, so LFU heat, usage_out and cache hits + misses (= expert_uses) do not
 * depend on how tokens were batched.
 */
#ifndef HX_MODEL_H
#define HX_MODEL_H

#include "hx_platform.h"
#include "../include/hearth.h"

/* Error codes returned (negated) through hearth.h. */
enum {
    HX_OK = 0,
    HX_E_ARG = -1,        /* NULL handle/pointer, n < 0, bad option value */
    HX_E_TOKEN = -2,      /* token id outside [0, vocab) */
    HX_E_CAPACITY = -3,   /* pos + n > KV capacity, or rewind beyond pos */
    HX_E_IO = -4,         /* expert slab unreadable, trace/replay file error */
    HX_E_NOMEM = -5,
    HX_E_FORMAT = -6      /* routing trace does not fit this model */
};

typedef struct hx_model hx_model;

/* opt is resolved by the caller (n_threads, n_io_threads, max_batch >= 1, isa !=
 * AUTO); max_seq 0 means min(model max_seq, 4096), larger requests are capped at
 * the model's max_seq. Returns NULL with a message on failure. */
hx_model *hx_model_open(const hearth_options *opt, char *err, size_t errlen);
void      hx_model_close(hx_model *m);

void hx_model_info(hx_model *m, hearth_model_info *out);
int  hx_model_vocab(const hx_model *m);
int  hx_model_capacity(const hx_model *m);   /* KV positions */

/* tokens already validated (ids in range, pos + n <= capacity, n >= 1).
 * A failed call (also one that fails in a later max_batch chunk) leaves the
 * position, the routing trace and the token / forward-call / expert counters as
 * they were; the time counters and the store's statistics include the attempt. */
int  hx_model_eval(hx_model *m, const int32_t *tokens, int n, float *logits, int all_logits);
int  hx_model_pos(const hx_model *m);
void hx_model_set_pos(hx_model *m, int pos);

void hx_model_get_stats(hx_model *m, hearth_stats *out);
void hx_model_reset_stats(hx_model *m);

int  hx_model_trace_start(hx_model *m, const char *path);
int  hx_model_trace_stop(hx_model *m);
/* path NULL or "" turns replay off. */
int  hx_model_route_replay(hx_model *m, const char *path);

/* Test seams (engine/tests/test_model.c): the model behind a public handle (api.c),
 * its expert store, a smaller attention sub-batch (1..the allocated size) so tests
 * reach the sub-batch boundaries that only long prompts hit otherwise, the
 * threshold (estimated ns of one thread's work) below which a region stays on the
 * caller (0 = every region of more than one item on the pool, < 0 = default), the
 * resolved max_batch, the thread count used for small batches and the number of
 * regions run on a pool so far. */
hx_model *hx_engine_model(hearth_engine *e);
/* HEARTH_ISA syntax: auto|scalar|avx2|avx512 (any case, leading spaces) or 0..3. Returns 1 if valid. */
int hx_parse_isa(const char *s, int *out);
struct hx_store *hx_model_store(hx_model *m);
void hx_model_set_attention_batch(hx_model *m, int tb);
void hx_model_set_parallel_min(hx_model *m, double ns);
int  hx_model_max_batch(const hx_model *m);
int  hx_model_core_threads(const hx_model *m);
uint64_t hx_model_regions(const hx_model *m);

#endif /* HX_MODEL_H */
