/*
 * model.c — the forward pass (docs/NUMERICS.md §5) and the MoE expert schedule
 * (docs/ARCHITECTURE.md), behind hx_model.h.
 *
 * One forward call evaluates T <= max_batch tokens at positions pos0..pos0+T-1:
 *   embed -> per layer [rmsnorm, attention (GQA | MLA), residual,
 *                       rmsnorm, dense FFN | MoE, residual] -> out_norm -> lm_head
 *
 * Batched == sequential (INV-DET-2):
 *   - every matmul is one kernel call per row chunk over all T tokens (row-outer,
 *     token-inner); hx_quant.h guarantees each (token, row) equals the T=1 value
 *     whatever T and row range;
 *   - all T tokens' keys/values are written to the cache first; then every
 *     attention value of a token is computed over its causal prefix with the
 *     single-token arithmetic (see the attention section);
 *   - MoE routes all T tokens, takes the union of their experts and runs each
 *     expert once for all tokens routed to it, each (token, rank) into its own
 *     buffer, summed per token in rank order at the end.
 * Scheduling independence (INV-DET-1): pool workers take dynamic chunks but every
 * output element is written by exactly one kernel call; experts are computed in
 * whatever order they become resident, and the rank-order sum makes that order
 * irrelevant. Transcendentals go through quant.c's helpers or the non-inlined
 * wrappers below, so no compiler can swap in a vector libm for some elements.
 *
 * MoE layer L: router -> top-k per token -> union -> hx_store_try_acquire each
 * (misses become demand reads) -> predict layer L+1 (PREFETCH_NEXT: L+1's router
 * on rmsnorm(h, ffn_norm[L+1]); PREFETCH_SHARED: on rmsnorm(h + rs*shared, ...),
 * the shared expert being computed at this point, overlapping the reads) ->
 * hx_store_prefetch(top_k + prefetch_extra per token, while recent predictions were
 * precise) -> compute resident experts in waves, release, hx_store_wait_any for
 * the rest -> rank-order sum + shared. An expert used by n (token, rank) pairs of
 * a batch is acquired once and counted n times (hx_store_count_uses).
 */
#include "hx_model.h"
#include "hx_modelfile.h"
#include "hx_pool.h"
#include "hx_quant.h"
#include "hx_store.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MLA_TS        32                  /* tokens per q_lat work item */
#define ATT_PC        256                 /* positions per score work item */
#define ATT_DC        32                  /* output dims per value work item */
#define MLA_HC        16                  /* MLA heads per score work item */
#define ATT_BUDGET    ((size_t)64 << 20)  /* bytes of score / latent scratch: bounds the attention sub-batch */
/* Pool workers spin this long after their last task before sleeping. Before the
 * pool yielded while spinning, 1000 us collapsed decode at 32 threads (2.1 vs 14.2
 * tok/s with 50 us, every region on the pool). With hx_pool_for_n regions and the
 * yielding pool, 50 / 300 / 1000 us and one pool vs a per-core + an all-thread pool
 * measured the same within noise (synthetic Qwen3-30B-A3B Q4, Ryzen 9 9950X, shared
 * machine; fully cached decode and 512-token prefill at 16 and 32 threads, 4 GiB
 * direct-I/O decode at 16, a tiny model at 1-32): one pool, 50 us. */
#define POOL_SPIN_US  50
/* A batch of fewer tokens runs on at most one thread per physical core: it is
 * memory-bound, so SMT siblings add no bandwidth but every region then waits for
 * oversubscribed threads (same container and machine, prefill tok/s with 16 vs 32
 * threads: T=64 172 vs 157, T=128 241 vs 240, T=256 323 vs 336, T=512 391 vs 437). */
#define SMT_MIN_T     256
/* A parallel region costs a barrier over its threads (a few us; far more when a
 * thread is descheduled). Work estimated below PAR_MIN_NS of one thread's time
 * stays on the caller; above it a region gets one thread per PAR_THREAD_NS of work
 * (hx_pool_for_n), so small regions wake few workers. The per-site estimates are
 * calibrated on a Zen 5 core and only steer scheduling, never a result. */
#define PAR_MIN_NS    6000.0
#define PAR_THREAD_NS 3000.0
#define WAIT_US       20000u              /* one hx_store_wait_any */
#define STUCK_NS      1000000000ull       /* nothing arrived this long: blocking acquire (detects unreadable slabs) */
#define PF_MIN_PREC   0.5                 /* see prefetch_next */
#define PF_EWMA       0.05
#define TRACE_MAGIC   0x52545248u         /* "HRTR" */
#define TRACE_HDR     24u
#define MAX_ALLOCS    96

enum { EX_PENDING = 0, EX_HELD = 1, EX_DONE = 2 };
/* Prefetch hints a failed call must wait for (see drain_hints): hinted by the previous
 * call / by this one and not acquired since, or listed but acquired (resolved). */
enum { HINT_NONE = 0, HINT_PREV = 1, HINT_NOW = 2, HINT_USED = 3 };

typedef struct mat {
    const uint8_t *w;
    int dtype;
    int64_t rows, cols;
    size_t rb;
} mat;

typedef struct layer_w {
    int moe;
    const float *attn_norm, *ffn_norm;
    mat q, k, v, o;                      /* GQA; q also MLA without q_lora */
    const float *bq, *bk, *bv, *qn, *kn;
    mat q_a, q_b, kv_a, kv_b;            /* MLA */
    const float *q_a_norm, *kv_a_norm;
    mat gate, up, down;                  /* dense FFN */
    mat router;                          /* F32 [E, D] */
    const float *router_bias;
    mat sh_gate, sh_up, sh_down;
    const float *sh_gate_inp;
} layer_w;

typedef struct mm_task {
    hx_matmul_fn fn;
    const void *W;
    int64_t cols, rows;
    const void *act;
    int T, dtype;
    float *Y;
    int64_t ldy;
} mm_task;

typedef struct tscratch {
    float *wrow;                         /* MLA: one dequantized W_UK row [C] */
    uint8_t *act;                        /* MLA: att_tb activations of C */
} tscratch;

typedef struct mstats {
    uint64_t tokens, forward_calls, expert_uses, expert_loads_unique;
    uint64_t wall_ns, attn_ns, moe_ns, dense_ns, stall_ns;
} mstats;

struct hx_model {
    hx_modelfile *mf;
    const hx_config *c;
    hx_store *store;
    hx_pool *pool;                       /* n_threads threads */
    int core_threads;                    /* min(n_threads, physical cores): batches under SMT_MIN_T */
    int fwd_threads;                     /* threads available to the running forward call */
    double par_min_ns;
    uint64_t regions, region_threads;    /* regions run on a pool and their summed thread counts (test seams) */
    int region_max;                      /* most threads of one region since the seam last read it */
    const hx_kernels *k;
    int isa, n_threads, n_io;

    int L, D, V, H, Hkv, hd, rope_dim, rope_half, mla;
    int nope, rdim, vd, C, ql;
    int q_dim, kv_dim, o_in, kva_dim, max_in;
    int E, K, F, Fd, Fs, n_moe, ffn_max;
    int cap, model_max_seq, max_batch, prefetch, prefetch_extra;
    int quant_experts;                   /* some slab is Q8/Q4 */
    float eps;

    layer_w *lw;
    int *moe_index;                      /* layer -> MoE ordinal, -1 for dense */
    mat tok_embd, lm_head;
    const float *out_norm, *inv_freq;
    int pos;

    float *kv;
    size_t kv_bytes, kv_layer;           /* bytes of the cache, floats per layer */

    /* scratch, B = max_batch */
    float *h, *x, *xs, *ao, *q, *kk, *vv, *qa, *kva, *att, *f1, *f2, *sh, *rope;
    uint8_t *act, *xact;                 /* B activations of up to max_in / of D */
    float *logit_e, *plog, *score, *sel, *gsc;
    uint8_t *taken, *gkeep, *pf_mark;
    int *ids, *pids, *pf_list;
    int pred_layer, pred_n;              /* the last prediction: layer, size (members flagged in pf_mark) */
    double pf_prec;                      /* recent precision of the predictions */
    float *wts;
    int *cnt, *ulist, *u_p0, *u_n, *ustate;
    const void **uslab;
    int *pair_tok, *pair_of, *wave, *wave_pairs;
    uint8_t **pair_mid;
    uint8_t *act_in, *act_mid;
    size_t stride_in, stride_mid;
    float *G, *U, *Y;
    mm_task *tasks;
    int64_t *task_start;
    int max_tasks;
    tscratch *ts;
    float *S, *QL, *OL;                  /* attention: [att_tb][H][cap], MLA [att_tb][H][C], [H][att_tb][C] */
    size_t ldS;
    int att_tb, att_tb_max;
    uint16_t *routes;                    /* [B][n_moe*K] */

    hx_file *trace;
    uint64_t trace_off;
    uint8_t *tpend;                      /* this call's trace rows, written when the whole call succeeds */
    size_t tpend_len, tpend_cap;
    uint16_t *replay;
    int64_t replay_rows;

    mstats st;
    hx_store_stats hid;                  /* store counters of failed calls, left out of hearth_stats */
    uint64_t seen_errors;                /* store read_errors the MoE wait loop has reacted to */
    uint8_t *hint;                       /* [L*E] HINT_* of (layer, expert) keys handed to hx_store_prefetch */
    int *hint_keys, n_hints;             /* the keys whose state is not HINT_NONE */
    uint64_t hints_drained;              /* test seam: hints failed calls have waited for */
    int *pred_log;                       /* test seam: [layer, n, ids...] per prediction */
    int pred_log_cap, pred_log_len;
    void *allocs[MAX_ALLOCS];
    int n_allocs;
};

/* ------------------------------------------------------------ small helpers */

static HX_NOINLINE float m_cosf(float x) { return cosf(x); }
static HX_NOINLINE float m_sinf(float x) { return sinf(x); }

static size_t szmul(size_t a, size_t b) { return (a && b > SIZE_MAX / a) ? SIZE_MAX : a * b; }

static void *m_alloc(hx_model *m, size_t n) {
    void *p;
    if (n == SIZE_MAX || m->n_allocs >= MAX_ALLOCS) return NULL;
    if (n == 0) n = 64;
    p = hx_aligned_alloc(64, n);
    if (!p) return NULL;
    memset(p, 0, n);
    m->allocs[m->n_allocs++] = p;
    return p;
}

static int imin(int a, int b) { return a < b ? a : b; }
static int imax(int a, int b) { return a > b ? a : b; }

/* a ranks before b: larger value first, NaN last, ties to the lower index. */
static int ranks_before(float va, int ia, float vb, int ib) {
    int na = va != va, nb = vb != vb;
    if (na || nb) return na == nb ? ia < ib : nb;
    if (va != vb) return va > vb;
    return ia < ib;
}

/* GQA cache [layer][K, V][kv head][pos][head_dim]: a head's rows are a [cap, hd] matrix.
 * MLA cache [layer][c: pos][C], [kpe: pos][rdim]. */
static float *kcache(const hx_model *m, int l, int g) {
    return m->kv + (size_t)l * m->kv_layer + (size_t)g * (size_t)m->cap * (size_t)m->hd;
}
static float *vcache(const hx_model *m, int l, int g) {
    return m->kv + (size_t)l * m->kv_layer + ((size_t)m->Hkv + (size_t)g) * (size_t)m->cap * (size_t)m->hd;
}
static float *ccache(const hx_model *m, int l, int p) {
    return m->kv + (size_t)l * m->kv_layer + (size_t)p * (size_t)m->C;
}
static float *pcache(const hx_model *m, int l, int p) {
    return m->kv + (size_t)l * m->kv_layer + (size_t)m->cap * (size_t)m->C + (size_t)p * (size_t)m->rdim;
}

/* ------------------------------------------------------------ parallel regions */

/* Threads for a region of n items and ~ns of one thread's work: the caller alone
 * below par_min_ns, else one per PAR_THREAD_NS of work (at least 2), at most the
 * forward call's threads. par_min_ns 0 (test seam) gives every region all of them. */
static int region_k(const hx_model *m, int64_t n, double ns) {
    int k = m->fwd_threads;
    if (n <= 1 || k <= 1 || ns < m->par_min_ns) return 1;
    if (m->par_min_ns > 0.0 && ns < PAR_THREAD_NS * (double)k) k = ns < 2.0 * PAR_THREAD_NS ? 2 : (int)(ns / PAR_THREAD_NS);
    return n < k ? (int)n : k;
}

/* fn over items [0, n) on k threads (the caller as tid 0 when k <= 1). */
static void par_run(hx_model *m, int64_t n, int k, hx_range_fn fn, void *ctx) {
    if (n <= 0) return;
    if (n < k) k = (int)n;
    if (k <= 1) {
        fn(ctx, 0, n, 0);
        return;
    }
    m->regions++;
    m->region_threads += (uint64_t)k;
    if (k > m->region_max) m->region_max = k;
    hx_pool_for_n(m->pool, k, n, 1, fn, ctx);
}

static void par_for(hx_model *m, int64_t n, double ns, hx_range_fn fn, void *ctx) {
    par_run(m, n, region_k(m, n, ns), fn, ctx);
}

/* Sum over the tokens of a sub-batch of their causal prefix lengths. */
static double prefix_sum(int pos0, int t0, int tn) {
    return (double)tn * (double)(pos0 + t0 + 1) + 0.5 * (double)tn * (double)(tn - 1);
}

/* ------------------------------------------------------------ parallel matmul */

typedef struct mm_job {
    const mm_task *t;
    const int64_t *start;
    int n;
    int64_t chunk;
} mm_job;

static void mm_range(void *ctx, int64_t b, int64_t e, int tid) {
    const mm_job *j = (const mm_job *)ctx;
    int lo = 0;
    (void)tid;
    for (int64_t it = b; it < e; it++) {
        int hi = j->n - 1;
        while (lo < hi) {
            int mid = lo + (hi - lo + 1) / 2;
            if (j->start[mid] <= it) lo = mid;
            else hi = mid - 1;
        }
        const mm_task *k = &j->t[lo];
        int64_t r0 = (it - j->start[lo]) * j->chunk;
        int64_t r1 = r0 + j->chunk < k->rows ? r0 + j->chunk : k->rows;
        k->fn(k->W, k->cols, k->act, k->T, k->Y, k->ldy, r0, r1);
    }
}

/* Runs every task's rows in dynamic chunks (multiples of 16 rows, the kernels' tile).
 * Estimate: weights stream at ~30 B/ns, ~100 multiply-adds per ns once T is large. */
static void mm_run(hx_model *m, const mm_task *t, int n) {
    int64_t total = 0, chunk, maxr = 1, tiles = 0;
    int nt;
    double ns = 0.0;
    if (n <= 0) return;
    for (int i = 0; i < n; i++) {
        total += t[i].rows;
        tiles += (t[i].rows + 15) / 16;
        if (t[i].rows > maxr) maxr = t[i].rows;
        ns += (double)t[i].rows * ((double)hx_row_bytes(t[i].dtype, t[i].cols) / 30.0 + (double)t[i].T * (double)t[i].cols / 100.0);
    }
    nt = region_k(m, tiles, ns);
    if (nt <= 1) {
        chunk = maxr;
    } else {
        chunk = (total + (int64_t)nt * 4 - 1) / ((int64_t)nt * 4);
        chunk = (chunk + 15) & ~(int64_t)15;
        if (chunk < 16) chunk = 16;
        if (chunk > 1024) chunk = 1024;
    }
    m->task_start[0] = 0;
    for (int i = 0; i < n; i++) {
        int64_t items = (t[i].rows > 0 && t[i].T > 0) ? (t[i].rows + chunk - 1) / chunk : 0;
        m->task_start[i + 1] = m->task_start[i] + items;
    }
    if (m->task_start[n] == 0) return;
    mm_job j;
    j.t = t;
    j.start = m->task_start;
    j.n = n;
    j.chunk = chunk;
    par_run(m, m->task_start[n], nt, mm_range, &j);
}

static void mm_set(mm_task *t, const hx_model *m, const mat *w, const void *act, int T, float *Y, int64_t ldy) {
    t->fn = m->k->matmul[w->dtype];
    t->W = w->w;
    t->cols = w->cols;
    t->rows = w->rows;
    t->act = act;
    t->T = T;
    t->dtype = w->dtype;
    t->Y = Y;
    t->ldy = ldy;
}

typedef struct actq_job {
    const hx_kernels *k;
    const float *x;
    int64_t n;
    uint8_t *dst;
    size_t ab;
} actq_job;

static void actq_range(void *ctx, int64_t b, int64_t e, int tid) {
    const actq_job *j = (const actq_job *)ctx;
    (void)tid;
    for (int64_t t = b; t < e; t++)
        j->k->act_quantize_q8(j->x + (size_t)t * (size_t)j->n, (hx_act_q8 *)(j->dst + (size_t)t * j->ab), j->n);
}

/* T activations of x[T][n] for weights of dtype: x itself for float weights,
 * NUMERICS §2 blocks in buf for Q8/Q4 (prepared once while *ready). */
static const void *act_of(hx_model *m, int dtype, const float *x, int64_t n, int T, uint8_t *buf, int *ready) {
    if (!hx_dtype_is_quant(dtype)) return x;
    if (ready && *ready) return buf;
    actq_job j;
    j.k = m->k;
    j.x = x;
    j.n = n;
    j.dst = buf;
    j.ab = hx_act_bytes(HEARTH_Q8, n);
    par_for(m, T, (double)T * (double)n * 0.5, actq_range, &j);
    if (ready) *ready = 1;
    return buf;
}

/* ------------------------------------------------------------ elementwise */

static void rope_tables(hx_model *m, int T, int pos0) {
    const float f = m->c->rope_attn_factor;
    for (int t = 0; t < T; t++) {
        float *cs = m->rope + (size_t)t * (size_t)m->rope_dim;
        const float p = (float)(pos0 + t);
        for (int j = 0; j < m->rope_half; j++) {
            float th = p * m->inv_freq[j];
            cs[2 * j] = m_cosf(th) * f;
            cs[2 * j + 1] = m_sinf(th) * f;
        }
    }
}

/* NUMERICS §4 RoPE on the first rope_dim dims of x, with precomputed (cos, sin). */
static void rope_apply(float *x, const float *cs, int half, int style) {
    for (int j = 0; j < half; j++) {
        int ia = style == HX_ROPE_GPTJ ? 2 * j : j;
        int ib = style == HX_ROPE_GPTJ ? 2 * j + 1 : j + half;
        float a = x[ia], b = x[ib], c = cs[2 * j], s = cs[2 * j + 1];
        x[ia] = a * c - b * s;
        x[ib] = b * c + a * s;
    }
}

typedef struct swi_job {
    const hx_kernels *k;
    float *g;
    const float *u;
    int64_t n;
    const int *idx;          /* NULL: rows 0..; else row = idx[i] */
    uint8_t **mid;           /* NULL or per row: quantize the result there */
} swi_job;

static void swi_range(void *ctx, int64_t b, int64_t e, int tid) {
    const swi_job *j = (const swi_job *)ctx;
    (void)tid;
    for (int64_t i = b; i < e; i++) {
        int64_t r = j->idx ? j->idx[i] : i;
        float *g = j->g + (size_t)r * (size_t)j->n;
        hx_swiglu(g, g, j->u + (size_t)r * (size_t)j->n, j->n);
        if (j->mid && j->mid[r]) j->k->act_quantize_q8(g, (hx_act_q8 *)j->mid[r], j->n);
    }
}

static void swiglu_rows(hx_model *m, float *g, const float *u, int64_t n, int rows, const int *idx, uint8_t **mid) {
    swi_job j;
    j.k = m->k;
    j.g = g;
    j.u = u;
    j.n = n;
    j.idx = idx;
    j.mid = mid;
    par_for(m, rows, (double)rows * (double)n * 1.6, swi_range, &j);
}

/* ------------------------------------------------------------ dense FFN */

/* out[T][D] = down(swiglu(gate x, up x)); xq: x's Q8 activations (prepared on demand). */
static void ffn(hx_model *m, const mat *g, const mat *u, const mat *d, int T, const float *x, int *xq_ready, float *out) {
    const int64_t Fx = g->rows;
    mm_task t[2];
    int r = 0;
    mm_set(&t[0], m, g, act_of(m, g->dtype, x, m->D, T, m->xact, xq_ready), T, m->f1, Fx);
    mm_set(&t[1], m, u, act_of(m, u->dtype, x, m->D, T, m->xact, xq_ready), T, m->f2, Fx);
    mm_run(m, t, 2);
    swiglu_rows(m, m->f1, m->f2, Fx, T, NULL, NULL);
    mm_set(&t[0], m, d, act_of(m, d->dtype, m->f1, Fx, T, m->act, &r), T, out, m->D);
    mm_run(m, t, 1);
}

/* ------------------------------------------------------------ attention
 *
 * The cache rows a token attends to are shared by several heads: the H/Hkv query
 * heads of a GQA group, and all H heads in MLA's absorbed form (one latent cache).
 * The core therefore runs in passes over a sub-batch of at most att_tb tokens,
 * each pass reading a cache row once for all the heads that share it:
 *   scores   S[i][h][r] = dot16(q_h, row_r) through the F32 kernel (NUMERICS §1;
 *            one call per token, group and chunk of ATT_PC positions, the group's
 *            heads as the T activations);
 *   softmax  per (token, head): scale (MLA: + dot16(q_pe, kpe_r)), softmax;
 *   values   per (token, group, ATT_DC dims): o[d] = sum_r p_r * v_r[d] in
 *            position order from 0.0f (NUMERICS §5.1/§5.2).
 * Every value is computed with the single-token arithmetic, so the chunking, the
 * sub-batch and the thread count cannot change it.
 */

typedef struct post_job {
    hx_model *m;
    const layer_w *w;
    int l, pos0;
} post_job;

typedef struct core_job {
    hx_model *m;
    int l, t0, tn, pos0, npc, ndc, nsub, nhc;
} core_job;

/* GQA per token: biases, qk-norm, RoPE, append (k, v) to the cache. */
static void gqa_post_range(void *ctx, int64_t b, int64_t e, int tid) {
    const post_job *j = (const post_job *)ctx;
    hx_model *m = j->m;
    const layer_w *w = j->w;
    const int H = m->H, Hkv = m->Hkv, hd = m->hd;
    (void)tid;
    for (int64_t t = b; t < e; t++) {
        const int pos = j->pos0 + (int)t;
        float *q = m->q + (size_t)t * (size_t)m->q_dim;
        float *k = m->kk + (size_t)t * (size_t)m->kv_dim;
        float *v = m->vv + (size_t)t * (size_t)m->kv_dim;
        const float *cs = m->rope + (size_t)t * (size_t)m->rope_dim;
        if (w->bq) {
            for (int i = 0; i < m->q_dim; i++) q[i] = q[i] + w->bq[i];
            for (int i = 0; i < m->kv_dim; i++) k[i] = k[i] + w->bk[i];
            for (int i = 0; i < m->kv_dim; i++) v[i] = v[i] + w->bv[i];
        }
        if (m->c->qk_norm == HX_QKNORM_HEAD) {
            for (int h = 0; h < H; h++) hx_rmsnorm(q + (size_t)h * hd, q + (size_t)h * hd, w->qn, hd, m->eps);
            for (int h = 0; h < Hkv; h++) hx_rmsnorm(k + (size_t)h * hd, k + (size_t)h * hd, w->kn, hd, m->eps);
        } else if (m->c->qk_norm == HX_QKNORM_FULL) {
            hx_rmsnorm(q, q, w->qn, m->q_dim, m->eps);
            hx_rmsnorm(k, k, w->kn, m->kv_dim, m->eps);
        }
        if (m->rope_half) {
            for (int h = 0; h < H; h++) rope_apply(q + (size_t)h * hd, cs, m->rope_half, m->c->rope_style);
            for (int h = 0; h < Hkv; h++) rope_apply(k + (size_t)h * hd, cs, m->rope_half, m->c->rope_style);
        }
        for (int g = 0; g < Hkv; g++) {
            memcpy(kcache(m, j->l, g) + (size_t)pos * hd, k + (size_t)g * hd, sizeof(float) * (size_t)hd);
            memcpy(vcache(m, j->l, g) + (size_t)pos * hd, v + (size_t)g * hd, sizeof(float) * (size_t)hd);
        }
    }
}

static void gqa_scores(void *ctx, int64_t b, int64_t e, int tid) {
    const core_job *j = (const core_job *)ctx;
    hx_model *m = j->m;
    const int Hkv = m->Hkv, hd = m->hd, grp = m->H / m->Hkv;
    const hx_matmul_fn fn = m->k->matmul[HEARTH_F32];
    (void)tid;
    for (int64_t it = b; it < e; it++) {
        const int pc = (int)(it % j->npc), g = (int)(it / j->npc) % Hkv, i = (int)(it / j->npc) / Hkv;
        const int pos = j->pos0 + j->t0 + i, r0 = pc * ATT_PC;
        if (r0 > pos) continue;
        fn(kcache(m, j->l, g), hd, m->q + (size_t)(j->t0 + i) * (size_t)m->q_dim + (size_t)g * grp * hd, grp,
           m->S + ((size_t)i * m->H + (size_t)g * grp) * m->ldS, (int64_t)m->ldS, r0, imin(r0 + ATT_PC, pos + 1));
    }
}

static void gqa_softmax(void *ctx, int64_t b, int64_t e, int tid) {
    const core_job *j = (const core_job *)ctx;
    hx_model *m = j->m;
    const float scale = m->c->attn_scale;
    (void)tid;
    for (int64_t it = b; it < e; it++) {
        const int i = (int)(it / m->H), pos = j->pos0 + j->t0 + i;
        float *s = m->S + (size_t)it * m->ldS;
        for (int r = 0; r <= pos; r++) s[r] = scale * s[r];
        hx_softmax(s, pos + 1);
    }
}

/* o_h[d] = sum_r p_h[r] * v_r[d] for r in [0, n), sequentially from 0.0f (NUMERICS
 * §5.1/§5.2), for nh heads (o_h = o + h*ldo, p_h = p + h*ldp) and dn <= ATT_DC
 * dims of the rows v_r = v + r*ldv; each row is read once for all nh heads. The
 * full-width case has a constant trip count and restrict pointers, so compilers
 * vectorise it across d (MSVC/SSE2: ~15% faster); every element keeps its order. */
static void value_sum(float *HX_RESTRICT o, size_t ldo, const float *HX_RESTRICT p, size_t ldp, int nh,
                      const float *HX_RESTRICT v, size_t ldv, int n, int dn) {
    for (int h = 0; h < nh; h++)
        for (int d = 0; d < dn; d++) o[(size_t)h * ldo + d] = 0.0f;
    if (dn == ATT_DC) {
        for (int r = 0; r < n; r++) {
            const float *HX_RESTRICT vr = v + (size_t)r * ldv;
            for (int h = 0; h < nh; h++) {
                const float ph = p[(size_t)h * ldp + (size_t)r];
                float *HX_RESTRICT oh = o + (size_t)h * ldo;
                for (int d = 0; d < ATT_DC; d++) oh[d] = oh[d] + ph * vr[d];
            }
        }
    } else {
        for (int r = 0; r < n; r++) {
            const float *HX_RESTRICT vr = v + (size_t)r * ldv;
            for (int h = 0; h < nh; h++) {
                const float ph = p[(size_t)h * ldp + (size_t)r];
                float *HX_RESTRICT oh = o + (size_t)h * ldo;
                for (int d = 0; d < dn; d++) oh[d] = oh[d] + ph * vr[d];
            }
        }
    }
}

static void gqa_values(void *ctx, int64_t b, int64_t e, int tid) {
    const core_job *j = (const core_job *)ctx;
    hx_model *m = j->m;
    const int Hkv = m->Hkv, hd = m->hd, grp = m->H / m->Hkv;
    (void)tid;
    for (int64_t it = b; it < e; it++) {
        const int dc = (int)(it % j->ndc), g = (int)(it / j->ndc) % Hkv, i = (int)(it / j->ndc) / Hkv;
        const int pos = j->pos0 + j->t0 + i, d0 = dc * ATT_DC;
        value_sum(m->att + (size_t)(j->t0 + i) * (size_t)m->o_in + (size_t)g * grp * hd + d0, (size_t)hd,
                  m->S + ((size_t)i * m->H + (size_t)g * grp) * m->ldS, m->ldS, grp, vcache(m, j->l, g) + d0, (size_t)hd,
                  pos + 1, imin(ATT_DC, hd - d0));
    }
}

/* per_token: ~ns of one token's work */
static void run_tokens(hx_model *m, int T, double per_token, hx_range_fn fn, void *ctx) {
    par_for(m, T, (double)T * per_token, fn, ctx);
}

static void attn_gqa(hx_model *m, const layer_w *w, int l, int T, int pos0) {
    mm_task t[3];
    int xr = 0, ar = 0;
    mm_set(&t[0], m, &w->q, act_of(m, w->q.dtype, m->x, m->D, T, m->xact, &xr), T, m->q, m->q_dim);
    mm_set(&t[1], m, &w->k, act_of(m, w->k.dtype, m->x, m->D, T, m->xact, &xr), T, m->kk, m->kv_dim);
    mm_set(&t[2], m, &w->v, act_of(m, w->v.dtype, m->x, m->D, T, m->xact, &xr), T, m->vv, m->kv_dim);
    mm_run(m, t, 3);

    post_job pj;
    pj.m = m;
    pj.w = w;
    pj.l = l;
    pj.pos0 = pos0;
    run_tokens(m, T, (double)(m->q_dim + m->kv_dim) * 2.0, gqa_post_range, &pj);

    for (int t0 = 0; t0 < T; t0 += m->att_tb) {
        core_job j;
        j.m = m;
        j.l = l;
        j.t0 = t0;
        j.tn = imin(m->att_tb, T - t0);
        j.pos0 = pos0;
        j.npc = (pos0 + t0 + j.tn + ATT_PC - 1) / ATT_PC;
        j.ndc = (m->hd + ATT_DC - 1) / ATT_DC;
        j.nsub = j.nhc = 0;
        const double ps = prefix_sum(pos0, t0, j.tn) * (double)m->H;
        par_for(m, (int64_t)j.tn * m->Hkv * j.npc, ps * (double)m->hd / 40.0, gqa_scores, &j);
        par_for(m, (int64_t)j.tn * m->H, ps * 1.5, gqa_softmax, &j);
        par_for(m, (int64_t)j.tn * m->Hkv * j.ndc, ps * (double)m->hd / 7.0, gqa_values, &j);
    }

    mm_set(&t[0], m, &w->o, act_of(m, w->o.dtype, m->att, m->o_in, T, m->act, &ar), T, m->ao, m->D);
    mm_run(m, t, 1);
}

/* MLA per token: c = rmsnorm(kva[0:C]) and RoPE'd kpe into the cache, RoPE on q_pe. */
static void mla_post_range(void *ctx, int64_t b, int64_t e, int tid) {
    const post_job *j = (const post_job *)ctx;
    hx_model *m = j->m;
    const int qh = m->nope + m->rdim;
    (void)tid;
    for (int64_t t = b; t < e; t++) {
        const float *kva = m->kva + (size_t)t * (size_t)m->kva_dim;
        const float *cs = m->rope + (size_t)t * (size_t)m->rope_dim;
        float *q = m->q + (size_t)t * (size_t)m->q_dim;
        float *kp = pcache(m, j->l, j->pos0 + (int)t);
        hx_rmsnorm(ccache(m, j->l, j->pos0 + (int)t), kva, j->w->kv_a_norm, m->C, m->eps);
        if (m->rdim) memcpy(kp, kva + m->C, sizeof(float) * (size_t)m->rdim);
        if (m->rope_half) {
            rope_apply(kp, cs, m->rope_half, m->c->rope_style);
            for (int h = 0; h < m->H; h++) rope_apply(q + (size_t)h * qh + m->nope, cs, m->rope_half, m->c->rope_style);
        }
    }
}

/* q_lat[i][h][c] = sum_n q_nope[n] * W_UK[n][c], n ascending from 0.0f (NUMERICS §5.2);
 * one item = head h for MLA_TS tokens, each W_UK row dequantized once. */
static void mla_qlat(void *ctx, int64_t b, int64_t e, int tid) {
    const core_job *j = (const core_job *)ctx;
    hx_model *m = j->m;
    const mat *kvb = &m->lw[j->l].kv_b;
    const int C = m->C, H = m->H, nope = m->nope, qh = m->nope + m->rdim;
    float *wrow = m->ts[tid].wrow;
    for (int64_t it = b; it < e; it++) {
        const int h = (int)(it / j->nsub), s0 = (int)(it % j->nsub) * MLA_TS, sn = imin(MLA_TS, j->tn - s0);
        const size_t wr0 = (size_t)h * (size_t)(nope + m->vd);
        for (int i = 0; i < sn; i++) memset(m->QL + ((size_t)(s0 + i) * H + h) * C, 0, sizeof(float) * (size_t)C);
        for (int n = 0; n < nope; n++) {
            hx_dequantize_row(kvb->dtype, kvb->w + (wr0 + (size_t)n) * kvb->rb, wrow, C);
            for (int i = 0; i < sn; i++) {
                const float a = m->q[(size_t)(j->t0 + s0 + i) * (size_t)m->q_dim + (size_t)h * qh + (size_t)n];
                float *ql = m->QL + ((size_t)(s0 + i) * H + h) * C;
                for (int c = 0; c < C; c++) ql[c] = ql[c] + a * wrow[c];
            }
        }
    }
}

static void mla_scores(void *ctx, int64_t b, int64_t e, int tid) {
    const core_job *j = (const core_job *)ctx;
    hx_model *m = j->m;
    const hx_matmul_fn fn = m->k->matmul[HEARTH_F32];
    (void)tid;
    for (int64_t it = b; it < e; it++) {
        const int hc = (int)(it % j->nhc), pc = (int)(it / j->nhc % j->npc), i = (int)(it / j->nhc / j->npc);
        const int pos = j->pos0 + j->t0 + i, r0 = pc * ATT_PC, h0 = hc * MLA_HC, hn = imin(MLA_HC, m->H - h0);
        if (r0 > pos) continue;
        fn(ccache(m, j->l, 0), m->C, m->QL + ((size_t)i * m->H + h0) * m->C, hn, m->S + ((size_t)i * m->H + h0) * m->ldS,
           (int64_t)m->ldS, r0, imin(r0 + ATT_PC, pos + 1));
    }
}

static void mla_softmax(void *ctx, int64_t b, int64_t e, int tid) {
    const core_job *j = (const core_job *)ctx;
    hx_model *m = j->m;
    const float scale = m->c->attn_scale;
    const int rdim = m->rdim, qh = m->nope + m->rdim;
    const float *kp = pcache(m, j->l, 0);
    (void)tid;
    for (int64_t it = b; it < e; it++) {
        const int i = (int)(it / m->H), h = (int)(it % m->H), pos = j->pos0 + j->t0 + i;
        const float *qpe = m->q + (size_t)(j->t0 + i) * (size_t)m->q_dim + (size_t)h * qh + (size_t)m->nope;
        float *s = m->S + (size_t)it * m->ldS;
        for (int r = 0; r <= pos; r++) s[r] = scale * (s[r] + hx_dot16(qpe, kp + (size_t)r * rdim, rdim));
        hx_softmax(s, pos + 1);
    }
}

/* o_lat[h][i][c] = sum_r p_r * c_r[c] in position order; OL is [H][att_tb][C]. */
static void mla_values(void *ctx, int64_t b, int64_t e, int tid) {
    const core_job *j = (const core_job *)ctx;
    hx_model *m = j->m;
    const int C = m->C, H = m->H;
    (void)tid;
    for (int64_t it = b; it < e; it++) {
        const int dc = (int)(it % j->ndc), i = (int)(it / j->ndc);
        const int pos = j->pos0 + j->t0 + i, c0 = dc * ATT_DC;
        value_sum(m->OL + (size_t)i * C + c0, (size_t)m->att_tb * C, m->S + (size_t)i * H * m->ldS, m->ldS, H,
                  ccache(m, j->l, 0) + c0, (size_t)C, pos + 1, imin(ATT_DC, C - c0));
    }
}

/* o_h = W_UV · o_lat (canonical matvec) for the sub-batch, one item per head. */
static void mla_out(void *ctx, int64_t b, int64_t e, int tid) {
    const core_job *j = (const core_job *)ctx;
    hx_model *m = j->m;
    const mat *kvb = &m->lw[j->l].kv_b;
    const int C = m->C, nope = m->nope, vd = m->vd;
    for (int64_t h = b; h < e; h++) {
        const float *ol = m->OL + (size_t)h * m->att_tb * C;
        const void *act = ol;
        if (hx_dtype_is_quant(kvb->dtype)) {
            const size_t ab = hx_act_bytes(HEARTH_Q8, C);
            uint8_t *dst = m->ts[tid].act;
            for (int i = 0; i < j->tn; i++)
                m->k->act_quantize_q8(ol + (size_t)i * C, (hx_act_q8 *)(dst + (size_t)i * ab), C);
            act = dst;
        }
        m->k->matmul[kvb->dtype](kvb->w + ((size_t)h * (size_t)(nope + vd) + (size_t)nope) * kvb->rb, C, act, j->tn,
                                 m->att + (size_t)j->t0 * (size_t)m->o_in + (size_t)h * vd, m->o_in, 0, vd);
    }
}

static void attn_mla(hx_model *m, const layer_w *w, int l, int T, int pos0) {
    mm_task t[2];
    int xr = 0, ar = 0;
    if (m->ql) {
        int qr = 0;
        mm_set(&t[0], m, &w->q_a, act_of(m, w->q_a.dtype, m->x, m->D, T, m->xact, &xr), T, m->qa, m->ql);
        mm_set(&t[1], m, &w->kv_a, act_of(m, w->kv_a.dtype, m->x, m->D, T, m->xact, &xr), T, m->kva, m->kva_dim);
        mm_run(m, t, 2);
        for (int i = 0; i < T; i++) {
            float *qa = m->qa + (size_t)i * (size_t)m->ql;
            hx_rmsnorm(qa, qa, w->q_a_norm, m->ql, m->eps);
        }
        mm_set(&t[0], m, &w->q_b, act_of(m, w->q_b.dtype, m->qa, m->ql, T, m->act, &qr), T, m->q, m->q_dim);
        mm_run(m, t, 1);
    } else {
        mm_set(&t[0], m, &w->q, act_of(m, w->q.dtype, m->x, m->D, T, m->xact, &xr), T, m->q, m->q_dim);
        mm_set(&t[1], m, &w->kv_a, act_of(m, w->kv_a.dtype, m->x, m->D, T, m->xact, &xr), T, m->kva, m->kva_dim);
        mm_run(m, t, 2);
    }

    post_job pj;
    pj.m = m;
    pj.w = w;
    pj.l = l;
    pj.pos0 = pos0;
    run_tokens(m, T, (double)(m->q_dim + m->kva_dim) * 1.0, mla_post_range, &pj);

    for (int t0 = 0; t0 < T; t0 += m->att_tb) {
        core_job j;
        j.m = m;
        j.l = l;
        j.t0 = t0;
        j.tn = imin(m->att_tb, T - t0);
        j.pos0 = pos0;
        j.npc = (pos0 + t0 + j.tn + ATT_PC - 1) / ATT_PC;
        j.ndc = (m->C + ATT_DC - 1) / ATT_DC;
        j.nsub = (j.tn + MLA_TS - 1) / MLA_TS;
        j.nhc = (m->H + MLA_HC - 1) / MLA_HC;
        const double ps = prefix_sum(pos0, t0, j.tn) * (double)m->H, H = (double)m->H, C = (double)m->C;
        par_for(m, (int64_t)m->H * j.nsub, H * m->nope * C * (j.tn + 1) / 7.0, mla_qlat, &j);
        par_for(m, (int64_t)j.tn * j.npc * j.nhc, ps * C / 40.0, mla_scores, &j);
        par_for(m, (int64_t)j.tn * m->H, ps * (1.5 + m->rdim / 7.0), mla_softmax, &j);
        par_for(m, (int64_t)j.tn * j.ndc, ps * C / 7.0, mla_values, &j);
        par_for(m, m->H, H * (m->vd * ((double)w->kv_b.rb / 30.0 + j.tn * C / 100.0) + j.tn * C * 0.5), mla_out, &j);
    }

    mm_set(&t[0], m, &w->o, act_of(m, w->o.dtype, m->att, m->o_in, T, m->act, &ar), T, m->ao, m->D);
    mm_run(m, t, 1);
}

/* ------------------------------------------------------------ routing */

/* NUMERICS §5.3: score from logits (into m->score), selection value with bias and
 * group limit, the kk best experts (descending, ties to the lower index). */
static void route_select(hx_model *m, const layer_w *w, const float *logits, int kk, int *ids) {
    const hx_config *c = m->c;
    const int E = m->E;
    float *score = m->score, *sel = m->sel;
    memcpy(score, logits, sizeof(float) * (size_t)E);
    if (c->score_fn == HX_SCORE_SOFTMAX) hx_softmax(score, E);
    else for (int e = 0; e < E; e++) score[e] = hx_sigmoid(score[e]);
    if (c->score_bias) for (int e = 0; e < E; e++) sel[e] = score[e] + w->router_bias[e];
    else memcpy(sel, score, sizeof(float) * (size_t)E);
    if (c->n_group > 1) {
        const int ng = c->n_group, gs = E / ng;
        for (int g = 0; g < ng; g++) {
            const float *v = sel + (size_t)g * gs;
            int i1 = 0, i2 = -1;
            for (int i = 1; i < gs; i++) {
                if (ranks_before(v[i], i, v[i1], i1)) { i2 = i1; i1 = i; }
                else if (i2 < 0 || ranks_before(v[i], i, v[i2], i2)) i2 = i;
            }
            m->gsc[g] = i2 >= 0 ? v[i1] + v[i2] : v[i1];
            m->gkeep[g] = 0;
        }
        for (int r = 0; r < c->topk_group; r++) {
            int best = -1;
            for (int g = 0; g < ng; g++)
                if (!m->gkeep[g] && (best < 0 || ranks_before(m->gsc[g], g, m->gsc[best], best))) best = g;
            m->gkeep[best] = 1;
        }
        for (int g = 0; g < ng; g++)
            if (!m->gkeep[g]) for (int i = 0; i < gs; i++) sel[(size_t)g * gs + i] = 0.0f;
    }
    for (int j = 0; j < kk; j++) {
        int best = -1;
        for (int e = 0; e < E; e++)
            if (!m->taken[e] && (best < 0 || ranks_before(sel[e], e, sel[best], best))) best = e;
        ids[j] = best;
        m->taken[best] = 1;
    }
    for (int j = 0; j < kk; j++) m->taken[ids[j]] = 0;
}

static void route_weights(const hx_model *m, const int *ids, float *w) {
    const hx_config *c = m->c;
    for (int j = 0; j < m->K; j++) w[j] = m->score[ids[j]];
    if (c->norm_topk_prob) {
        float s = 0.0f;
        for (int j = 0; j < m->K; j++) s = s + w[j];
        const float den = s + 1e-20f;
        for (int j = 0; j < m->K; j++) w[j] = w[j] / den;
    }
    for (int j = 0; j < m->K; j++) w[j] = w[j] * c->routed_scale;
}

/* ------------------------------------------------------------ MoE */

static void compute_wave(hx_model *m, int l, int nw, int T) {
    const int D = m->D, F = m->F;
    mm_task *tg = m->tasks, *td = m->tasks + 2 * nw;
    int np = 0;
    (void)T;
    for (int w = 0; w < nw; w++) {
        const int u = m->wave[w], e = m->ulist[u];
        const int dt = (int)hx_mf_expert(m->mf, l, e)->dtype;
        const int p0 = m->u_p0[u], n = m->u_n[u];
        const int quant = hx_dtype_is_quant(dt);
        const size_t ab = hx_act_bytes(dt, D), abm = hx_act_bytes(dt, F);
        const uint8_t *src = quant ? m->xact : (const uint8_t *)m->x;
        const void *act;
        hx_slab_view v;
        int run = 1;
        hx_slab_view_make(m->mf, m->uslab[u], dt, &v);
        for (int i = 1; i < n; i++) run &= m->pair_tok[p0 + i] == m->pair_tok[p0] + i;
        if (run) {
            act = src + (size_t)m->pair_tok[p0] * ab;
        } else {
            uint8_t *dst = m->act_in + (size_t)p0 * m->stride_in;
            for (int i = 0; i < n; i++) memcpy(dst + (size_t)i * ab, src + (size_t)m->pair_tok[p0 + i] * ab, ab);
            act = dst;
        }
        tg[2 * w].fn = tg[2 * w + 1].fn = td[w].fn = m->k->matmul[dt];
        tg[2 * w].dtype = tg[2 * w + 1].dtype = td[w].dtype = dt;
        tg[2 * w].W = v.gate;
        tg[2 * w + 1].W = v.up;
        tg[2 * w].cols = tg[2 * w + 1].cols = D;
        tg[2 * w].rows = tg[2 * w + 1].rows = F;
        tg[2 * w].act = tg[2 * w + 1].act = act;
        tg[2 * w].T = tg[2 * w + 1].T = n;
        tg[2 * w].Y = m->G + (size_t)p0 * (size_t)F;
        tg[2 * w + 1].Y = m->U + (size_t)p0 * (size_t)F;
        tg[2 * w].ldy = tg[2 * w + 1].ldy = F;
        td[w].W = v.down;
        td[w].cols = F;
        td[w].rows = D;
        td[w].act = quant ? (const void *)(m->act_mid + (size_t)p0 * m->stride_mid) : (const void *)(m->G + (size_t)p0 * (size_t)F);
        td[w].T = n;
        td[w].Y = m->Y + (size_t)p0 * (size_t)D;
        td[w].ldy = D;
        for (int i = 0; i < n; i++) {
            m->wave_pairs[np++] = p0 + i;
            m->pair_mid[p0 + i] = quant ? m->act_mid + (size_t)p0 * m->stride_mid + (size_t)i * abm : NULL;
        }
    }
    mm_run(m, tg, 2 * nw);
    swiglu_rows(m, m->G, m->U, F, np, m->wave_pairs, m->pair_mid);
    mm_run(m, td, nw);
}

typedef struct sum_job {
    hx_model *m;
} sum_job;

/* Per token: out = sum_j w_j * expert_j(x) in rank order (+ shared); h += rs * out. */
static void moe_sum_range(void *ctx, int64_t b, int64_t e, int tid) {
    hx_model *m = ((const sum_job *)ctx)->m;
    const int D = m->D, K = m->K;
    const float rs = m->c->residual_scale;
    (void)tid;
    for (int64_t t = b; t < e; t++) {
        float *out = m->ao + (size_t)t * (size_t)D, *h = m->h + (size_t)t * (size_t)D;
        for (int i = 0; i < D; i++) out[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            const float wj = m->wts[(size_t)t * K + j];
            const float *y = m->Y + (size_t)m->pair_of[(size_t)t * K + j] * (size_t)D;
            for (int i = 0; i < D; i++) out[i] = out[i] + wj * y[i];
        }
        if (m->Fs) {
            const float *s = m->sh + (size_t)t * (size_t)D;
            for (int i = 0; i < D; i++) out[i] = out[i] + s[i];
        }
        for (int i = 0; i < D; i++) h[i] = h[i] + rs * out[i];
    }
}

static void shared_expert(hx_model *m, const layer_w *w, int T, int *xq_ready) {
    ffn(m, &w->sh_gate, &w->sh_up, &w->sh_down, T, m->x, xq_ready, m->sh);
    if (w->sh_gate_inp) {
        for (int t = 0; t < T; t++) {
            float *s = m->sh + (size_t)t * (size_t)m->D;
            const float g = hx_sigmoid(hx_dot16(w->sh_gate_inp, m->x + (size_t)t * (size_t)m->D, m->D));
            for (int i = 0; i < m->D; i++) s[i] = s[i] * g;
        }
    }
}

/* Next-layer prediction from m->plog (router of layer nl on the predicted input):
 * top_k + prefetch_extra per token, union over the batch. The prediction is always
 * made, and scored when layer nl routes (precision = predicted experts actually
 * used / predicted); it is handed to the store only while the recent precision is
 * at least PF_MIN_PREC * top_k / (top_k + extra). Wrong prefetches cost a slab read
 * and an eviction each, which on a bandwidth-bound NVMe halved decode speed in a
 * measurement where the routing came from a replayed trace the router cannot
 * predict. Reads only: the output never depends on this. */
static void prefetch_next(hx_model *m, int nl, int T) {
    const int kk = imin(m->K + m->prefetch_extra, m->E);
    int n = 0;
    for (int i = 0; i < m->pred_n; i++) m->pf_mark[m->pf_list[i]] = 0;
    for (int t = 0; t < T; t++) {
        route_select(m, &m->lw[nl], m->plog + (size_t)t * (size_t)m->E, kk, m->pids);
        for (int j = 0; j < kk; j++)
            if (!m->pf_mark[m->pids[j]]) {
                m->pf_mark[m->pids[j]] = 1;
                m->pf_list[n++] = m->pids[j];
            }
    }
    m->pred_layer = nl;
    m->pred_n = n;
    if (m->pred_log && m->pred_log_len + 2 + n <= m->pred_log_cap) {
        int *d = m->pred_log + m->pred_log_len;
        d[0] = nl;
        d[1] = n;
        memcpy(d + 2, m->pf_list, sizeof(int) * (size_t)n);
        m->pred_log_len += 2 + n;
    }
    if (m->pf_prec >= PF_MIN_PREC * (double)m->K / (double)kk) {
        hx_store_prefetch(m->store, nl, m->pf_list, n);
        for (int i = 0; i < n; i++) {
            const int key = nl * m->E + m->pf_list[i];
            if (m->hint[key] == HINT_NONE) m->hint_keys[m->n_hints++] = key;
            m->hint[key] = HINT_NOW;
        }
    }
}

/* Scores the prediction made for layer l against its actual experts (ulist). */
static void score_prediction(hx_model *m, int l, int nu) {
    int hit = 0;
    if (m->pred_layer != l || m->pred_n <= 0) return;
    for (int u = 0; u < nu; u++) hit += m->pf_mark[m->ulist[u]];
    m->pf_prec += PF_EWMA * ((double)hit / (double)m->pred_n - m->pf_prec);
    for (int i = 0; i < m->pred_n; i++) m->pf_mark[m->pf_list[i]] = 0;
    m->pred_layer = -1;
    m->pred_n = 0;
}

static int moe_layer(hx_model *m, const layer_w *w, int l, int T, int pos0) {
    const int E = m->E, K = m->K, D = m->D;
    const int mi = m->moe_index[l];
    const int nl = (l + 1 < m->L && m->lw[l + 1].moe) ? l + 1 : -1;
    const int pf = (m->prefetch == HEARTH_PREFETCH_OFF || nl < 0) ? 0
                 : (m->prefetch == HEARTH_PREFETCH_SHARED && m->Fs > 0) ? 2 : 1;
    int xr = 0, nu = 0, remaining, rc = HX_OK;
    mm_task t[2];
    int nt = 0;

    mm_set(&t[nt++], m, &w->router, m->x, T, m->logit_e, E);
    if (pf == 1) {
        for (int i = 0; i < T; i++)
            hx_rmsnorm(m->xs + (size_t)i * D, m->h + (size_t)i * D, m->lw[nl].ffn_norm, D, m->eps);
        mm_set(&t[nt++], m, &m->lw[nl].router, m->xs, T, m->plog, E);
    }
    mm_run(m, t, nt);

    for (int i = 0; i < T; i++) {
        const float *lg = m->logit_e + (size_t)i * E;
        int *ids = m->ids + (size_t)i * K;
        if (m->replay) {
            const uint16_t *row = m->replay + (size_t)((pos0 + i) % m->replay_rows) * (size_t)(m->n_moe * K) + (size_t)mi * K;
            memcpy(m->score, lg, sizeof(float) * (size_t)E);
            if (m->c->score_fn == HX_SCORE_SOFTMAX) hx_softmax(m->score, E);
            else for (int e = 0; e < E; e++) m->score[e] = hx_sigmoid(m->score[e]);
            for (int j = 0; j < K; j++) ids[j] = row[j];
        } else {
            route_select(m, w, lg, K, ids);
        }
        route_weights(m, ids, m->wts + (size_t)i * K);
        for (int j = 0; j < K; j++) m->routes[(size_t)i * (m->n_moe * K) + (size_t)mi * K + j] = (uint16_t)ids[j];
    }

    /* union; pairs grouped by expert (ascending id), tokens ascending within */
    memset(m->cnt, 0, sizeof(int) * (size_t)E);
    for (int i = 0; i < T * K; i++) m->cnt[m->ids[i]]++;
    for (int e = 0, p = 0; e < E; e++) {
        if (!m->cnt[e]) continue;
        m->ulist[nu] = e;
        m->u_p0[nu] = p;
        m->u_n[nu] = m->cnt[e];
        p += m->cnt[e];
        m->cnt[e] = m->u_p0[nu];     /* now a fill cursor */
        nu++;
    }
    for (int i = 0; i < T; i++)
        for (int j = 0; j < K; j++) {
            const int p = m->cnt[m->ids[(size_t)i * K + j]]++;
            m->pair_tok[p] = i;
            m->pair_of[(size_t)i * K + j] = p;
        }
    m->st.expert_uses += (uint64_t)T * (uint64_t)K;
    m->st.expert_loads_unique += (uint64_t)nu;
    score_prediction(m, l, nu);
    /* acquiring these settles their prefetches (also on failure: the drain below) */
    for (int u = 0; u < nu; u++)
        if (m->hint[l * E + m->ulist[u]]) m->hint[l * E + m->ulist[u]] = HINT_USED;

    for (int u = 0; u < nu; u++) {
        m->uslab[u] = hx_store_try_acquire(m->store, l, m->ulist[u]);
        m->ustate[u] = m->uslab[u] ? EX_HELD : EX_PENDING;
    }
    if (pf == 1) prefetch_next(m, nl, T);
    if (m->quant_experts) act_of(m, HEARTH_Q8, m->x, D, T, m->xact, &xr);
    if (m->Fs) shared_expert(m, w, T, &xr);
    if (pf == 2) {
        const float rs = m->c->residual_scale;
        for (int i = 0; i < T; i++) {
            float *ps = m->xs + (size_t)i * D;
            const float *h = m->h + (size_t)i * D, *s = m->sh + (size_t)i * D;
            for (int d = 0; d < D; d++) ps[d] = h[d] + rs * s[d];
            hx_rmsnorm(ps, ps, m->lw[nl].ffn_norm, D, m->eps);
        }
        mm_set(&t[0], m, &m->lw[nl].router, m->xs, T, m->plog, E);
        mm_run(m, t, 1);
        prefetch_next(m, nl, T);
    }

    remaining = nu;
    uint64_t idle_ns = 0;
    int blocking = 0;
    for (;;) {
        int nw = 0, got = 0;
        for (int u = 0; u < nu; u++)
            if (m->ustate[u] == EX_HELD) m->wave[nw++] = u;
        if (nw) {
            compute_wave(m, l, nw, T);
            for (int i = 0; i < nw; i++) {
                const int u = m->wave[i];
                /* one use per (token, rank) routed here, as sequential evaluation counts */
                if (m->u_n[u] > 1) hx_store_count_uses(m->store, l, m->ulist[u], (uint32_t)(m->u_n[u] - 1));
                hx_store_release(m->store, l, m->ulist[u]);
                m->ustate[u] = EX_DONE;
                m->uslab[u] = NULL;
            }
            remaining -= nw;
            idle_ns = 0;
        }
        if (!remaining) break;
        for (int u = 0; u < nu; u++)
            if (m->ustate[u] == EX_PENDING && (m->uslab[u] = hx_store_try_acquire(m->store, l, m->ulist[u]))) {
                m->ustate[u] = EX_HELD;
                got++;
            }
        if (got) continue;
        uint64_t t0 = hx_now_ns();
        if (!blocking) {   /* a failed read, or nothing for a long time: wait on the experts one by one */
            hx_store_stats ss;
            hx_store_get_stats(m->store, &ss);
            blocking = ss.read_errors != m->seen_errors || idle_ns >= STUCK_NS;
            m->seen_errors = ss.read_errors;
        }
        if (!blocking) {
            hx_store_wait_any(m->store, WAIT_US);
        } else {
            /* the blocking acquire returns NULL at once for an expert that could not be read */
            for (int u = 0; u < nu; u++)
                if (m->ustate[u] == EX_PENDING) {
                    m->uslab[u] = hx_store_acquire(m->store, l, m->ulist[u]);
                    if (!m->uslab[u]) {
                        hx_log(HX_LOG_ERROR, "layer %d expert %d could not be read", l, m->ulist[u]);
                        rc = HX_E_IO;
                    } else {
                        m->ustate[u] = EX_HELD;
                    }
                    break;
                }
        }
        uint64_t dt = hx_now_ns() - t0;
        idle_ns += dt;
        m->st.stall_ns += dt;
        if (rc) {
            /* let this layer's other demand reads finish, so none completes after the call */
            for (int u = 0; u < nu; u++)
                if (m->ustate[u] == EX_PENDING && hx_store_acquire(m->store, l, m->ulist[u]))
                    hx_store_release(m->store, l, m->ulist[u]);
            return rc;
        }
    }

    sum_job sj;
    sj.m = m;
    run_tokens(m, T, (double)(K + 2) * D * 0.5, moe_sum_range, &sj);
    return HX_OK;
}

static void dense_layer(hx_model *m, const layer_w *w, int T) {
    const int D = m->D;
    const float rs = m->c->residual_scale;
    int xr = 0;
    ffn(m, &w->gate, &w->up, &w->down, T, m->x, &xr, m->ao);
    for (int t = 0; t < T; t++) {
        float *h = m->h + (size_t)t * D;
        const float *a = m->ao + (size_t)t * D;
        for (int i = 0; i < D; i++) h[i] = h[i] + rs * a[i];
    }
}

/* ------------------------------------------------------------ forward */

/* A chunk's routing rows (FORMAT §9) into the call's pending buffer (sized by hx_model_eval). */
static void trace_rows(hx_model *m, int T) {
    const size_t n = (size_t)T * (size_t)m->n_moe * (size_t)m->K;
    uint8_t *d = m->tpend + m->tpend_len;
    for (size_t i = 0; i < n; i++) {
        d[2 * i] = (uint8_t)(m->routes[i] & 0xff);
        d[2 * i + 1] = (uint8_t)(m->routes[i] >> 8);
    }
    m->tpend_len += 2 * n;
}

static void trace_flush(hx_model *m) {
    if (!m->trace || !m->tpend_len) return;
    if (hx_file_pwrite(m->trace, m->tpend, m->tpend_len, m->trace_off) != (int64_t)m->tpend_len) {
        hx_log(HX_LOG_ERROR, "routing trace: write failed; tracing stopped");
        hx_file_close(m->trace);
        m->trace = NULL;
    } else {
        m->trace_off += m->tpend_len;
    }
    m->tpend_len = 0;
}

/* mode 0: no logits, 1: last token into logits[V], 2: all T tokens into logits[T][V]. */
static int forward(hx_model *m, const int32_t *tok, int T, int pos0, float *logits, int mode) {
    const int D = m->D;
    const float rs = m->c->residual_scale, es = m->c->emb_scale;
    uint64_t t0 = hx_now_ns(), t1;
    int rc = HX_OK;

    m->fwd_threads = T >= SMT_MIN_T ? m->n_threads : m->core_threads;
    for (int t = 0; t < T; t++) {
        float *h = m->h + (size_t)t * D;
        hx_dequantize_row(m->tok_embd.dtype, m->tok_embd.w + (size_t)tok[t] * m->tok_embd.rb, h, D);
        for (int i = 0; i < D; i++) h[i] = es * h[i];
    }
    if (m->rope_half) rope_tables(m, T, pos0);
    t1 = hx_now_ns();
    m->st.dense_ns += t1 - t0;

    for (int l = 0; l < m->L && rc == HX_OK; l++) {
        const layer_w *w = &m->lw[l];
        t0 = hx_now_ns();
        for (int t = 0; t < T; t++)
            hx_rmsnorm(m->x + (size_t)t * D, m->h + (size_t)t * D, w->attn_norm, D, m->eps);
        if (m->mla) attn_mla(m, w, l, T, pos0);
        else attn_gqa(m, w, l, T, pos0);
        for (int t = 0; t < T; t++) {
            float *h = m->h + (size_t)t * D;
            const float *a = m->ao + (size_t)t * D;
            for (int i = 0; i < D; i++) h[i] = h[i] + rs * a[i];
        }
        for (int t = 0; t < T; t++)
            hx_rmsnorm(m->x + (size_t)t * D, m->h + (size_t)t * D, w->ffn_norm, D, m->eps);
        t1 = hx_now_ns();
        m->st.attn_ns += t1 - t0;
        if (w->moe) {
            rc = moe_layer(m, w, l, T, pos0);
            t0 = hx_now_ns();
            m->st.moe_ns += t0 - t1;
        } else {
            dense_layer(m, w, T);
            t0 = hx_now_ns();
            m->st.dense_ns += t0 - t1;
        }
    }
    if (rc != HX_OK) return rc;

    t0 = hx_now_ns();
    if (mode) {
        const int first = mode == 1 ? T - 1 : 0, n = mode == 1 ? 1 : T;
        const size_t V = (size_t)m->V;
        const float ls = m->c->logit_scale;
        int r = 0;
        mm_task t;
        for (int i = 0; i < n; i++)
            hx_rmsnorm(m->x + (size_t)i * D, m->h + (size_t)(first + i) * D, m->out_norm, D, m->eps);
        mm_set(&t, m, &m->lm_head, act_of(m, m->lm_head.dtype, m->x, D, n, m->act, &r), n, logits, m->V);
        mm_run(m, &t, 1);
        for (size_t i = 0; i < (size_t)n * V; i++) logits[i] = ls * logits[i];
    }
    m->st.dense_ns += hx_now_ns() - t0;
    m->st.forward_calls++;
    m->st.tokens += (uint64_t)T;
    if (m->trace && m->n_moe) trace_rows(m, T);
    for (int t = 0; t < T; t++) hx_store_tick(m->store);
    return HX_OK;
}

/* At the start of a call: this call's hints become the previous call's; older and
 * resolved ones are forgotten. */
static void age_hints(hx_model *m) {
    int n = 0;
    for (int i = 0; i < m->n_hints; i++) {
        const int key = m->hint_keys[i];
        if (m->hint[key] == HINT_NOW) {
            m->hint[key] = HINT_PREV;
            m->hint_keys[n++] = key;
        } else {
            m->hint[key] = HINT_NONE;
        }
    }
    m->n_hints = n;
}

/* A failed call waits for every prefetch that it or the previous call asked for and
 * did not acquire: a blocking acquire turns a queued prefetch into a demand read and
 * waits for one in flight (an unreadable expert fails at once). So no read it caused
 * lands after it returns, and what those reads count falls inside the hidden delta.
 * Costs at most one read per such hint (also for hints the store dropped). */
static void drain_hints(hx_model *m) {
    for (int i = 0; i < m->n_hints; i++) {
        const int key = m->hint_keys[i], l = key / m->E, e = key % m->E;
        if (m->hint[key] == HINT_USED) continue;
        if (hx_store_acquire(m->store, l, e)) hx_store_release(m->store, l, e);
        m->hint[key] = HINT_USED;
        m->hints_drained++;
    }
}

/* Store counters a failed call added (from s0 to s1) join the hidden ones; read_errors stays visible. */
static void hide_store_delta(hx_model *m, const hx_store_stats *s0, const hx_store_stats *s1) {
    hx_store_stats *h = &m->hid;
    h->hits += s1->hits - s0->hits;
    h->misses += s1->misses - s0->misses;
    h->evictions += s1->evictions - s0->evictions;
    h->prefetch_issued += s1->prefetch_issued - s0->prefetch_issued;
    h->prefetch_used += s1->prefetch_used - s0->prefetch_used;
    h->prefetch_wasted += s1->prefetch_wasted - s0->prefetch_wasted;
    h->bytes_read += s1->bytes_read - s0->bytes_read;
    h->read_ns += s1->read_ns - s0->read_ns;
}

int hx_model_eval(hx_model *m, const int32_t *tokens, int n, float *logits, int all_logits) {
    const uint64_t t0 = hx_now_ns();
    const int start = m->pos;
    const mstats before = m->st;
    hx_store_stats s0;
    int rc = HX_OK;
    hx_store_get_stats(m->store, &s0);
    m->tpend_len = 0;
    if (m->trace) {   /* the call's rows are written only if every chunk succeeds */
        const size_t need = (size_t)n * (size_t)m->n_moe * (size_t)m->K * 2;
        if (need > m->tpend_cap) {
            uint8_t *p = (uint8_t *)realloc(m->tpend, need);
            if (!p) {
                hx_log(HX_LOG_ERROR, "routing trace: no memory for %d tokens of routing", n);
                return HX_E_NOMEM;
            }
            m->tpend = p;
            m->tpend_cap = need;
        }
    }
    age_hints(m);
    for (int off = 0; off < n && rc == HX_OK; off += m->max_batch) {
        const int T = imin(m->max_batch, n - off);
        float *lg = NULL;
        int mode = 0;
        if (logits && all_logits) {
            lg = logits + (size_t)off * (size_t)m->V;
            mode = 2;
        } else if (logits && off + T == n) {
            lg = logits;
            mode = 1;
        }
        rc = forward(m, tokens + off, T, m->pos, lg, mode);
        if (rc == HX_OK) m->pos += T;
    }
    if (rc != HX_OK) {
        /* the failed call leaves no trace: position, routing rows and every counter
         * but read_errors are as before */
        hx_store_stats s1;
        m->pos = start;
        m->tpend_len = 0;
        m->st = before;
        drain_hints(m);            /* before s1: what the reads it waits for count is hidden too */
        hx_store_tick(m->store);   /* ends this attempt: unreadable experts may be retried, holds expire */
        hx_store_get_stats(m->store, &s1);
        hide_store_delta(m, &s0, &s1);
        return rc;
    }
    trace_flush(m);
    m->st.wall_ns += hx_now_ns() - t0;
    return rc;
}

/* ------------------------------------------------------------ open: binding */

static void tname(char *buf, size_t n, int layer, const char *name) {
    if (layer < 0) snprintf(buf, n, "%s", name);
    else snprintf(buf, n, "blk.%d.%s", layer, name);
}

static const hx_tensor *find_t(const hx_modelfile *mf, int layer, const char *name) {
    return layer < 0 ? hx_mf_tensor(mf, name) : hx_mf_layer_tensor(mf, layer, name);
}

static int bind_mat(hx_model *m, mat *o, int layer, const char *name, int64_t rows, int64_t cols, int f32_only,
                    char *msg, size_t ml) {
    const hx_tensor *t = find_t(m->mf, layer, name);
    char full[HX_NAME_LEN + 16];
    tname(full, sizeof full, layer, name);
    if (!t) { snprintf(msg, ml, "tensor '%s' is missing", full); return 0; }
    if (t->ndim != 2 || t->shape[0] != rows || t->shape[1] != cols) {
        snprintf(msg, ml, "tensor '%s' has shape [%lld, %lld, %lld, %lld] (ndim %d), expected [%lld, %lld]", full,
                 (long long)t->shape[0], (long long)t->shape[1], (long long)t->shape[2], (long long)t->shape[3],
                 t->ndim, (long long)rows, (long long)cols);
        return 0;
    }
    if (f32_only ? t->dtype != HEARTH_F32 : (t->dtype < HEARTH_F32 || t->dtype > HEARTH_Q4 || !m->k->matmul[t->dtype])) {
        snprintf(msg, ml, "tensor '%s' has dtype %s, which is not supported here", full, hx_dtype_name(t->dtype));
        return 0;
    }
    if (!t->data) { snprintf(msg, ml, "tensor '%s' was not loaded", full); return 0; }
    o->w = (const uint8_t *)t->data;
    o->dtype = t->dtype;
    o->rows = rows;
    o->cols = cols;
    o->rb = hx_row_bytes(t->dtype, cols);
    return 1;
}

static int bind_vec(hx_model *m, const float **o, int layer, const char *name, int64_t n, char *msg, size_t ml) {
    const hx_tensor *t = find_t(m->mf, layer, name);
    char full[HX_NAME_LEN + 16];
    tname(full, sizeof full, layer, name);
    if (!t) { snprintf(msg, ml, "tensor '%s' is missing", full); return 0; }
    if (!((t->ndim == 1 && t->shape[0] == n) || (t->ndim == 2 && t->shape[0] == 1 && t->shape[1] == n))) {
        snprintf(msg, ml, "tensor '%s' must have %lld elements (shape [%lld, %lld], ndim %d)", full, (long long)n,
                 (long long)t->shape[0], (long long)t->shape[1], t->ndim);
        return 0;
    }
    if (t->dtype != HEARTH_F32) { snprintf(msg, ml, "tensor '%s' must be F32", full); return 0; }
    if (!t->data) { snprintf(msg, ml, "tensor '%s' was not loaded", full); return 0; }
    *o = (const float *)t->data;
    return 1;
}

#define BM(o, l, n, r, c) do { if (!bind_mat(m, o, l, n, r, c, 0, msg, ml)) return 0; } while (0)
#define BF(o, l, n, r, c) do { if (!bind_mat(m, o, l, n, r, c, 1, msg, ml)) return 0; } while (0)
#define BV(o, l, n, k) do { if (!bind_vec(m, o, l, n, k, msg, ml)) return 0; } while (0)

static int dims_ok(int64_t v, const char *what, char *msg, size_t ml) {
    if (v < 1 || v > (INT_MAX / 8)) { snprintf(msg, ml, "%s = %lld is out of range", what, (long long)v); return 0; }
    return 1;
}

static int setup_dims(hx_model *m, char *msg, size_t ml) {
    const hx_config *c = m->c;
    m->L = c->n_layers;
    m->D = c->d_model;
    m->V = c->vocab_size;
    m->H = c->n_heads;
    m->eps = c->norm_eps;
    m->mla = c->attn_kind == HX_ATTN_MLA;
    m->rope_dim = c->rope_dim;
    m->rope_half = c->rope_dim / 2;
    if (m->mla) {
        m->nope = c->qk_nope_dim;
        m->rdim = c->qk_rope_dim;
        m->vd = c->v_head_dim;
        m->C = c->kv_lora_rank;
        m->ql = c->q_lora_rank;
        m->Hkv = m->H;
        m->hd = m->nope + m->rdim;
        if (!dims_ok((int64_t)m->H * (m->nope + m->rdim), "n_heads*(qk_nope_dim+qk_rope_dim)", msg, ml)) return 0;
        if (!dims_ok((int64_t)m->H * m->vd, "n_heads*v_head_dim", msg, ml)) return 0;
        if (!dims_ok((int64_t)m->H * (m->nope + m->vd), "n_heads*(qk_nope_dim+v_head_dim)", msg, ml)) return 0;
        m->q_dim = m->H * (m->nope + m->rdim);
        m->o_in = m->H * m->vd;
        m->kva_dim = m->C + m->rdim;
        m->kv_dim = 0;
    } else {
        m->Hkv = c->n_kv_heads;
        m->hd = c->head_dim;
        if (!dims_ok((int64_t)m->H * m->hd, "n_heads*head_dim", msg, ml)) return 0;
        m->q_dim = m->H * m->hd;
        m->kv_dim = m->Hkv * m->hd;
        m->o_in = m->q_dim;
    }
    m->E = c->n_experts;
    m->K = c->top_k;
    m->F = c->expert_ffn_dim;
    m->Fd = c->dense_ffn_dim;
    m->Fs = c->shared_ffn_dim;
    m->n_moe = c->n_moe_layers;
    if (m->n_moe == 0) { m->E = 0; m->K = 0; m->Fs = 0; }
    m->ffn_max = imax(m->Fd, m->Fs);
    m->max_in = imax(imax(m->D, m->ql), imax(m->C, m->o_in));
    m->max_in = imax(m->max_in, imax(m->ffn_max, m->F));
    m->model_max_seq = c->max_seq;
    return 1;
}

static int bind_all(hx_model *m, char *msg, size_t ml) {
    const hx_config *c = m->c;
    const int D = m->D;
    m->lw = (layer_w *)m_alloc(m, sizeof(layer_w) * (size_t)m->L);
    m->moe_index = (int *)m_alloc(m, sizeof(int) * (size_t)m->L);
    if (!m->lw || !m->moe_index) { snprintf(msg, ml, "out of memory"); return 0; }
    BM(&m->tok_embd, -1, "tok_embd", m->V, D);
    if (c->tie_embeddings) m->lm_head = m->tok_embd;
    else BM(&m->lm_head, -1, "lm_head", m->V, D);
    BV(&m->out_norm, -1, "out_norm", D);
    m->inv_freq = m->mf->rope_inv_freq;
    if (m->rope_half && !m->inv_freq) { snprintf(msg, ml, "tensor 'rope_inv_freq' was not loaded"); return 0; }
    for (int l = 0, mi = 0; l < m->L; l++) {
        layer_w *w = &m->lw[l];
        BV(&w->attn_norm, l, "attn_norm", D);
        BV(&w->ffn_norm, l, "ffn_norm", D);
        if (m->mla) {
            if (m->ql) {
                BM(&w->q_a, l, "attn_q_a", m->ql, D);
                BV(&w->q_a_norm, l, "attn_q_a_norm", m->ql);
                BM(&w->q_b, l, "attn_q_b", m->q_dim, m->ql);
            } else {
                BM(&w->q, l, "attn_q", m->q_dim, D);
            }
            BM(&w->kv_a, l, "attn_kv_a", m->kva_dim, D);
            BV(&w->kv_a_norm, l, "attn_kv_a_norm", m->C);
            BM(&w->kv_b, l, "attn_kv_b", (int64_t)m->H * (m->nope + m->vd), m->C);
            BM(&w->o, l, "attn_o", D, m->o_in);
        } else {
            BM(&w->q, l, "attn_q", m->q_dim, D);
            BM(&w->k, l, "attn_k", m->kv_dim, D);
            BM(&w->v, l, "attn_v", m->kv_dim, D);
            BM(&w->o, l, "attn_o", D, m->o_in);
            if (c->qkv_bias) {
                BV(&w->bq, l, "attn_q_bias", m->q_dim);
                BV(&w->bk, l, "attn_k_bias", m->kv_dim);
                BV(&w->bv, l, "attn_v_bias", m->kv_dim);
            }
            if (c->qk_norm == HX_QKNORM_HEAD) {
                BV(&w->qn, l, "attn_q_norm", m->hd);
                BV(&w->kn, l, "attn_k_norm", m->hd);
            } else if (c->qk_norm == HX_QKNORM_FULL) {
                BV(&w->qn, l, "attn_q_norm", m->q_dim);
                BV(&w->kn, l, "attn_k_norm", m->kv_dim);
            }
        }
        w->moe = c->layer_kind[l] == HX_LAYER_MOE;
        m->moe_index[l] = w->moe ? mi++ : -1;
        if (!w->moe) {
            BM(&w->gate, l, "ffn_gate", m->Fd, D);
            BM(&w->up, l, "ffn_up", m->Fd, D);
            BM(&w->down, l, "ffn_down", D, m->Fd);
            continue;
        }
        BF(&w->router, l, "moe_router", m->E, D);
        if (c->score_bias) BV(&w->router_bias, l, "moe_router_bias", m->E);
        if (m->Fs) {
            BM(&w->sh_gate, l, "shexp_gate", m->Fs, D);
            BM(&w->sh_up, l, "shexp_up", m->Fs, D);
            BM(&w->sh_down, l, "shexp_down", D, m->Fs);
            if (c->shared_gate) BV(&w->sh_gate_inp, l, "shexp_gate_inp", D);
        }
    }
    if (m->n_moe) {
        for (int l = 0; l < m->L; l++) {
            if (!m->lw[l].moe) continue;
            for (int e = 0; e < m->E; e++) {
                const hx_expert_entry *ent = hx_mf_expert(m->mf, l, e);
                if (!ent || !ent->nbytes || ent->dtype > HEARTH_Q4 || !m->k->matmul[ent->dtype]) {
                    snprintf(msg, ml, "expert (%d, %d) has no usable slab", l, e);
                    return 0;
                }
                if (hx_dtype_is_quant((int)ent->dtype)) {
                    m->quant_experts = 1;
                    m->stride_in = m->stride_in > hx_act_bytes(HEARTH_Q8, D) ? m->stride_in : hx_act_bytes(HEARTH_Q8, D);
                    m->stride_mid = m->stride_mid > hx_act_bytes(HEARTH_Q8, m->F) ? m->stride_mid : hx_act_bytes(HEARTH_Q8, m->F);
                } else {
                    m->stride_in = m->stride_in > (size_t)D * 4 ? m->stride_in : (size_t)D * 4;
                    m->stride_mid = m->stride_mid > (size_t)m->F * 4 ? m->stride_mid : (size_t)m->F * 4;
                }
            }
        }
    }
    return 1;
}

#undef BM
#undef BF
#undef BV

static int alloc_scratch(hx_model *m, char *msg, size_t ml) {
    const size_t B = (size_t)m->max_batch, D = (size_t)m->D, E = (size_t)m->E, K = (size_t)m->K;
    const size_t P = B * K, F = (size_t)m->F;
    const size_t actn = ((size_t)m->max_in / 64 + 1) * sizeof(hx_act_q8);
    const size_t actd = (D / 64 + 1) * sizeof(hx_act_q8);
    size_t kvf;
    int ok = 1;

    if (m->mla) kvf = szmul((size_t)m->cap, (size_t)m->C + (size_t)m->rdim);
    else kvf = szmul(szmul((size_t)m->cap, (size_t)m->kv_dim), 2);
    m->kv_layer = kvf;
    m->kv_bytes = szmul(szmul(kvf, (size_t)m->L), sizeof(float));
    if (m->kv_bytes == SIZE_MAX) { snprintf(msg, ml, "KV cache size overflows"); return 0; }
    m->kv = (float *)hx_alloc_large(m->kv_bytes ? m->kv_bytes : 64, 0);
    if (!m->kv) {
        snprintf(msg, ml, "cannot allocate %.1f MiB for the KV cache (%d positions)", (double)m->kv_bytes / (1 << 20), m->cap);
        return 0;
    }

#define A(p, n) do { if (ok && !((p) = m_alloc(m, (n)))) ok = 0; } while (0)
    A(m->h, szmul(B * D, 4));
    A(m->x, szmul(B * D, 4));
    A(m->xs, szmul(B * D, 4));
    A(m->ao, szmul(B * D, 4));
    A(m->q, szmul(B, (size_t)m->q_dim * 4));
    A(m->att, szmul(B, (size_t)m->o_in * 4));
    if (m->mla) {
        A(m->kva, szmul(B, (size_t)m->kva_dim * 4));
        if (m->ql) A(m->qa, szmul(B, (size_t)m->ql * 4));
    } else {
        A(m->kk, szmul(B, (size_t)m->kv_dim * 4));
        A(m->vv, szmul(B, (size_t)m->kv_dim * 4));
    }
    A(m->rope, szmul(B, (size_t)(m->rope_dim + 2) * 4));
    A(m->act, szmul(B, actn));
    A(m->xact, szmul(B, actd));
    if (m->ffn_max) {
        A(m->f1, szmul(B, (size_t)m->ffn_max * 4));
        A(m->f2, szmul(B, (size_t)m->ffn_max * 4));
    }
    if (m->Fs) A(m->sh, szmul(B * D, 4));
    m->max_tasks = 3 * (m->n_moe ? m->E : 1) + 4;
    A(m->tasks, sizeof(mm_task) * (size_t)m->max_tasks);
    A(m->task_start, sizeof(int64_t) * ((size_t)m->max_tasks + 1));
    if (m->n_moe) {
        A(m->logit_e, szmul(B * E, 4));
        A(m->plog, szmul(B * E, 4));
        A(m->score, E * 4);
        A(m->sel, E * 4);
        A(m->gsc, E * 4);
        A(m->taken, E);
        A(m->gkeep, E);
        A(m->pf_mark, E);
        A(m->pids, E * sizeof(int));
        A(m->pf_list, E * sizeof(int));
        A(m->ids, szmul(P, sizeof(int)));
        A(m->wts, szmul(P, 4));
        A(m->cnt, E * sizeof(int));
        A(m->ulist, E * sizeof(int));
        A(m->u_p0, E * sizeof(int));
        A(m->u_n, E * sizeof(int));
        A(m->ustate, E * sizeof(int));
        A(m->uslab, E * sizeof(void *));
        A(m->wave, E * sizeof(int));
        A(m->pair_tok, szmul(P, sizeof(int)));
        A(m->pair_of, szmul(P, sizeof(int)));
        A(m->wave_pairs, szmul(P, sizeof(int)));
        A(m->pair_mid, szmul(P, sizeof(uint8_t *)));
        A(m->act_in, szmul(P, m->stride_in));
        A(m->act_mid, szmul(P, m->stride_mid));
        A(m->G, szmul(P * F, 4));
        A(m->U, szmul(P * F, 4));
        A(m->Y, szmul(P * D, 4));
        A(m->routes, szmul(B, (size_t)m->n_moe * K * 2));
        A(m->hint, szmul((size_t)m->L, E));
        A(m->hint_keys, szmul(szmul((size_t)m->L, E), sizeof(int)));
    }
    {   /* attention sub-batch: score (and MLA latent) scratch within ATT_BUDGET */
        const size_t per = szmul(szmul((size_t)m->H, (size_t)m->cap), 4) +
                           (m->mla ? szmul(szmul((size_t)m->H, (size_t)m->C), 8) : 0);
        const size_t tb = per ? ATT_BUDGET / per : B;
        m->att_tb = m->att_tb_max = (int)(tb < 1 ? 1 : tb > B ? B : tb);
        m->ldS = (size_t)m->cap;
        A(m->S, szmul(szmul((size_t)m->att_tb, (size_t)m->H), szmul(m->ldS, 4)));
        if (m->mla) {
            A(m->QL, szmul(szmul((size_t)m->att_tb, (size_t)m->H), (size_t)m->C * 4));
            A(m->OL, szmul(szmul((size_t)m->att_tb, (size_t)m->H), (size_t)m->C * 4));
        }
    }
    A(m->ts, sizeof(tscratch) * (size_t)m->n_threads);
    if (ok && m->mla) {   /* per-thread scratch, one block */
        const size_t wr = HX_ALIGN_UP((size_t)m->C * 4, 64);
        const size_t ac = HX_ALIGN_UP((size_t)m->att_tb * (((size_t)m->C / 64 + 1) * sizeof(hx_act_q8)), 64);
        uint8_t *blk = NULL;
        A(blk, szmul(wr + ac, (size_t)m->n_threads));
        for (int i = 0; ok && i < m->n_threads; i++) {
            m->ts[i].wrow = (float *)(blk + (size_t)i * (wr + ac));
            m->ts[i].act = blk + (size_t)i * (wr + ac) + wr;
        }
    }
#undef A
    if (!ok) { snprintf(msg, ml, "out of memory for scratch buffers (max_batch %d)", m->max_batch); return 0; }
    return 1;
}

/* ------------------------------------------------------------ open / close */

void hx_model_close(hx_model *m) {
    if (!m) return;
    if (m->trace) hx_file_close(m->trace);
    if (m->store) hx_store_close(m->store);
    if (m->pool) hx_pool_destroy(m->pool);
    if (m->kv) hx_free_large(m->kv, m->kv_bytes ? m->kv_bytes : 64);
    for (int i = 0; i < m->n_allocs; i++) hx_aligned_free(m->allocs[i]);
    free(m->replay);
    free(m->tpend);
    if (m->mf) hx_modelfile_close(m->mf);
    free(m);
}

hx_model *hx_model_open(const hearth_options *o, char *err, size_t errlen) {
    char msg[512] = "";
    hx_model *m;
    if (err && errlen) err[0] = 0;
    if (!o || !o->model_path) return (hx_model *)hx_fail(err, errlen, "hx_model_open: no model path");
    m = (hx_model *)calloc(1, sizeof *m);
    if (!m) return (hx_model *)hx_fail(err, errlen, "out of memory");
    m->k = hx_kernels_for(o->isa);
    if (!m->k) {
        free(m);
        return (hx_model *)hx_fail(err, errlen, "ISA %d is not supported by this CPU or build", o->isa);
    }
    m->isa = m->k->isa;
    m->n_threads = o->n_threads < 1 ? 1 : o->n_threads;
    m->n_io = o->n_io_threads < 1 ? 1 : o->n_io_threads;
    m->max_batch = o->max_batch < 1 ? 1 : o->max_batch;
    m->prefetch = o->prefetch;
    m->pred_layer = -1;
    m->pf_prec = 1.0;
    m->par_min_ns = PAR_MIN_NS;

    m->mf = hx_modelfile_open(o->model_path, 1, err, errlen);
    if (!m->mf) { free(m); return NULL; }
    m->c = &m->mf->cfg;
    if (!setup_dims(m, msg, sizeof msg) || !bind_all(m, msg, sizeof msg)) goto fail;
    /* capped at n_experts (<= 65536), so top_k + extra cannot overflow in prefetch_next */
    m->prefetch_extra = o->prefetch_extra < 0 ? 0 : imin(o->prefetch_extra, m->E);
    m->cap = o->max_seq > 0 ? imin(o->max_seq, m->model_max_seq) : imin(m->model_max_seq, 4096);
    if (m->cap < 1) m->cap = 1;
    if (!alloc_scratch(m, msg, sizeof msg)) goto fail;

    /* threads beyond the physical cores only for large batches (SMT_MIN_T) */
    m->core_threads = imax(1, imin(m->n_threads, hx_num_physical_cores()));
    m->fwd_threads = m->core_threads;
    m->pool = hx_pool_create(m->n_threads, POOL_SPIN_US);
    if (!m->pool) {
        snprintf(msg, sizeof msg, "cannot start %d compute threads", m->n_threads);
        goto fail;
    }

    {
        hx_store_opts so;
        memset(&so, 0, sizeof so);
        double gb = o->cache_gb > 0.0 ? o->cache_gb : 0.0;
        double bytes = gb * 1073741824.0;
        so.cache_bytes = bytes >= 1.8e19 ? UINT64_MAX / 2 : (uint64_t)bytes;
        so.n_io_threads = m->n_io;
        so.direct_io = o->direct_io ? 1 : 0;
        so.policy = o->policy;
        so.heat_decay = 0.0f;
        so.usage_in = o->usage_in;
        so.usage_out = o->usage_out;
        so.pin_fraction = o->pin_fraction;
        so.warm_start = o->warm_start ? 1 : 0;
        so.mirrors = o->mirror_paths;
        so.n_mirrors = o->n_mirrors;
        m->store = hx_store_open(m->mf, &so, msg, sizeof msg);
        if (!m->store) goto fail;
    }
    hx_log(HX_LOG_INFO, "%s: %s, %d layers (%d MoE), d_model %d, vocab %d, KV %d positions (%.1f MiB), isa %d, "
           "%d threads (%d for batches under %d tokens)", o->model_path, m->c->arch, m->L, m->n_moe, m->D, m->V, m->cap,
           (double)m->kv_bytes / (1 << 20), m->isa, m->n_threads, m->core_threads, SMT_MIN_T);
    return m;
fail:
    hx_fail(err, errlen, "%s: %s", o->model_path, msg[0] ? msg : "cannot open");
    hx_model_close(m);
    return NULL;
}

/* ------------------------------------------------------------ queries */

void hx_model_info(hx_model *m, hearth_model_info *out) {
    const hx_config *c = m->c;
    hx_store_stats ss;
    memset(out, 0, sizeof *out);
    memcpy(out->arch, c->arch, sizeof out->arch);
    out->arch[sizeof out->arch - 1] = 0;
    out->n_layers = c->n_layers;
    out->d_model = c->d_model;
    out->vocab_size = c->vocab_size;
    out->max_seq = c->max_seq;
    out->n_heads = c->n_heads;
    out->n_kv_heads = m->Hkv;
    out->head_dim = m->hd;
    out->attn_kind = c->attn_kind;
    out->n_experts = m->E;
    out->top_k = m->K;
    out->expert_ffn_dim = c->expert_ffn_dim;
    out->n_moe_layers = c->n_moe_layers;
    out->bos_id = c->bos_id;
    out->n_eos = c->n_eos < 8 ? c->n_eos : 8;
    for (int i = 0; i < out->n_eos; i++) out->eos_ids[i] = c->eos_ids[i];
    out->dense_bytes = m->mf->dense_bytes;
    out->expert_bytes = m->mf->expert_region_bytes;
    out->slab_bytes_max = m->mf->slab_bytes_max;
    out->params_total = m->mf->params_total;
    out->params_active = m->mf->params_active;
    hx_store_get_stats(m->store, &ss);
    out->cache_slots = ss.n_slots;
    out->isa = m->isa;
    out->n_threads = m->n_threads;
    out->n_io_threads = m->n_io;
}

int hx_model_vocab(const hx_model *m) { return m->V; }
int hx_model_capacity(const hx_model *m) { return m->cap; }
int hx_model_pos(const hx_model *m) { return m->pos; }
void hx_model_set_pos(hx_model *m, int pos) { m->pos = pos; }
struct hx_store *hx_model_store(hx_model *m) { return m ? m->store : NULL; }
void hx_model_set_attention_batch(hx_model *m, int tb) {
    if (m && tb >= 1 && tb <= m->att_tb_max) m->att_tb = tb;
}
void hx_model_set_parallel_min(hx_model *m, double ns) {
    if (m) m->par_min_ns = ns >= 0.0 ? ns : PAR_MIN_NS;
}
int hx_model_max_batch(const hx_model *m) { return m ? m->max_batch : 0; }
uint64_t hx_model_regions(const hx_model *m) { return m ? m->regions : 0; }
uint64_t hx_model_region_threads(const hx_model *m) { return m ? m->region_threads : 0; }
int hx_model_region_max(hx_model *m) {
    int k = m ? m->region_max : 0;
    if (m) m->region_max = 0;
    return k;
}
int hx_model_core_threads(const hx_model *m) { return m ? m->core_threads : 0; }
void hx_model_set_prediction_log(hx_model *m, int *buf, int cap) {
    if (!m) return;
    m->pred_log = cap > 0 ? buf : NULL;
    m->pred_log_cap = cap > 0 ? cap : 0;
    m->pred_log_len = 0;
}
int hx_model_prediction_log_len(const hx_model *m) { return m ? m->pred_log_len : 0; }
uint64_t hx_model_hints_drained(const hx_model *m) { return m ? m->hints_drained : 0; }

static uint64_t minus(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }

void hx_model_get_stats(hx_model *m, hearth_stats *out) {
    hx_store_stats ss;
    memset(out, 0, sizeof *out);
    hx_store_get_stats(m->store, &ss);
    ss.hits = minus(ss.hits, m->hid.hits);
    ss.misses = minus(ss.misses, m->hid.misses);
    ss.evictions = minus(ss.evictions, m->hid.evictions);
    ss.prefetch_issued = minus(ss.prefetch_issued, m->hid.prefetch_issued);
    ss.prefetch_used = minus(ss.prefetch_used, m->hid.prefetch_used);
    ss.prefetch_wasted = minus(ss.prefetch_wasted, m->hid.prefetch_wasted);
    ss.bytes_read = minus(ss.bytes_read, m->hid.bytes_read);
    ss.read_ns = minus(ss.read_ns, m->hid.read_ns);
    out->tokens = m->st.tokens;
    out->forward_calls = m->st.forward_calls;
    out->wall_s = (double)m->st.wall_ns * 1e-9;
    out->attn_s = (double)m->st.attn_ns * 1e-9;
    out->moe_s = (double)m->st.moe_ns * 1e-9;
    out->dense_s = (double)m->st.dense_ns * 1e-9;
    out->stall_s = (double)m->st.stall_ns * 1e-9;
    out->expert_uses = m->st.expert_uses;
    out->expert_loads_unique = m->st.expert_loads_unique;
    out->cache_hits = ss.hits;
    out->cache_misses = ss.misses;
    out->prefetch_issued = ss.prefetch_issued;
    out->prefetch_used = ss.prefetch_used;
    out->prefetch_wasted = ss.prefetch_wasted;
    out->bytes_read = ss.bytes_read;
    out->read_s = (double)ss.read_ns * 1e-9;
    out->evictions = ss.evictions;
    out->cache_slots = ss.n_slots;
    out->cache_resident = ss.resident;
    out->cache_pinned = ss.pinned;
    out->read_errors = ss.read_errors;
}

void hx_model_reset_stats(hx_model *m) {
    memset(&m->st, 0, sizeof m->st);
    memset(&m->hid, 0, sizeof m->hid);
    m->seen_errors = 0;
    hx_store_reset_stats(m->store);
}

/* ------------------------------------------------------------ trace / replay */

static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static uint32_t get32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* The new file is opened (and its header written) before an active trace is
 * closed, so a path that cannot be written leaves the current trace running.
 * An existing non-empty file is overwritten only if it is a routing trace: a slip
 * must not truncate the model, a mirror or a heat profile. One that exists but
 * cannot be read (write-only permissions, a handle denying read sharing) cannot be
 * checked, so it is refused too. */
int hx_model_trace_start(hx_model *m, const char *path) {
    char err[512];
    uint8_t hdr[TRACE_HDR];
    hx_file *f = hx_file_open(path, HX_FILE_READ, err, sizeof err);
    if (f) {
        const int64_t size = hx_file_size(f);
        const int trace = size == 0 || (size >= 4 && hx_file_pread(f, hdr, 4, 0) == 4 && get32(hdr) == TRACE_MAGIC);
        hx_file_close(f);
        if (!trace) {
            hx_log(HX_LOG_ERROR, "routing trace: %s exists and is not a routing trace; not overwritten", path);
            return HX_E_ARG;
        }
    } else if (hx_path_exists(path)) {
        hx_log(HX_LOG_ERROR, "routing trace: %s exists but cannot be read to check it is a routing trace (%s); "
               "not overwritten", path, err);
        return HX_E_ARG;
    }
    f = hx_file_open(path, HX_FILE_WRITE | HX_FILE_CREATE, err, sizeof err);
    if (!f) { hx_log(HX_LOG_ERROR, "routing trace: %s", err); return HX_E_IO; }
    put32(hdr, TRACE_MAGIC);
    put32(hdr + 4, 1);
    put32(hdr + 8, (uint32_t)m->L);
    put32(hdr + 12, (uint32_t)m->E);
    put32(hdr + 16, (uint32_t)m->K);
    put32(hdr + 20, (uint32_t)m->n_moe);
    if (hx_file_pwrite(f, hdr, TRACE_HDR, 0) != (int64_t)TRACE_HDR) {
        hx_log(HX_LOG_ERROR, "routing trace: cannot write %s", path);
        hx_file_close(f);
        return HX_E_IO;
    }
    hx_model_trace_stop(m);
    m->trace = f;
    m->trace_off = TRACE_HDR;
    return HX_OK;
}

int hx_model_trace_stop(hx_model *m) {
    if (m->trace) {
        hx_file_close(m->trace);
        m->trace = NULL;
    }
    return HX_OK;
}

int hx_model_route_replay(hx_model *m, const char *path) {
    char err[512];
    uint8_t hdr[TRACE_HDR];
    hx_file *f;
    int64_t size;
    uint64_t row, body, rows;
    uint8_t *raw = NULL;
    uint16_t *ids = NULL;
    int rc = HX_OK;

    if (!path || !*path) {
        free(m->replay);
        m->replay = NULL;
        m->replay_rows = 0;
        return HX_OK;
    }
    if (!m->n_moe) { hx_log(HX_LOG_ERROR, "route replay: the model has no MoE layers"); return HX_E_FORMAT; }
    f = hx_file_open(path, HX_FILE_READ, err, sizeof err);
    if (!f) { hx_log(HX_LOG_ERROR, "route replay: %s", err); return HX_E_IO; }
    size = hx_file_size(f);
    row = (uint64_t)m->n_moe * (uint64_t)m->K;
    if (size < (int64_t)TRACE_HDR || hx_file_pread(f, hdr, TRACE_HDR, 0) != (int64_t)TRACE_HDR) {
        hx_log(HX_LOG_ERROR, "route replay: %s is not a routing trace", path);
        rc = HX_E_FORMAT;
        goto done;
    }
    /* n_layers only has to be consistent (>= n_moe_layers): replay needs the MoE
     * geometry alone, and hearth.sim.trace.synthetic() without n_layers writes
     * n_layers = n_moe_layers, also for models with leading dense layers */
    if (get32(hdr) != TRACE_MAGIC || get32(hdr + 4) != 1 || get32(hdr + 8) < (uint32_t)m->n_moe ||
        get32(hdr + 12) != (uint32_t)m->E || get32(hdr + 16) != (uint32_t)m->K || get32(hdr + 20) != (uint32_t)m->n_moe) {
        hx_log(HX_LOG_ERROR, "route replay: %s does not match this model (needs %d experts, top-%d, %d MoE layers)",
               path, m->E, m->K, m->n_moe);
        rc = HX_E_FORMAT;
        goto done;
    }
    body = (uint64_t)size - TRACE_HDR;
    if (body == 0 || body % (2 * row) != 0 || body > ((uint64_t)1 << 32)) {
        hx_log(HX_LOG_ERROR, "route replay: %s has a body of %llu bytes (rows of %llu bytes)", path,
               (unsigned long long)body, (unsigned long long)(2 * row));
        rc = HX_E_FORMAT;
        goto done;
    }
    rows = body / (2 * row);
    raw = (uint8_t *)malloc((size_t)body);
    ids = (uint16_t *)malloc((size_t)body);
    if (!raw || !ids) { rc = HX_E_NOMEM; goto done; }
    if (hx_file_pread(f, raw, (size_t)body, TRACE_HDR) != (int64_t)body) {
        hx_log(HX_LOG_ERROR, "route replay: cannot read %s", path);
        rc = HX_E_IO;
        goto done;
    }
    for (uint64_t i = 0; i < body / 2; i++) ids[i] = (uint16_t)(raw[2 * i] | (raw[2 * i + 1] << 8));
    for (uint64_t r = 0; r < rows * (uint64_t)m->n_moe && rc == HX_OK; r++) {
        const uint16_t *v = ids + r * (uint64_t)m->K;
        for (int j = 0; j < m->K && rc == HX_OK; j++) {
            if (v[j] >= m->E) rc = HX_E_FORMAT;
            for (int i = 0; i < j; i++)
                if (v[i] == v[j]) rc = HX_E_FORMAT;
        }
        if (rc) hx_log(HX_LOG_ERROR, "route replay: %s token %llu has an invalid or repeated expert id", path,
                       (unsigned long long)(r / (uint64_t)m->n_moe));
    }
    if (rc == HX_OK) {
        free(m->replay);
        m->replay = ids;
        m->replay_rows = (int64_t)rows;
        ids = NULL;
    }
done:
    free(raw);
    free(ids);
    hx_file_close(f);
    return rc;
}
