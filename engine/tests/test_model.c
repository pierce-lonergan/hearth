/*
 * test_model.c — self-checking tests for model.c / api.c through the public API.
 *
 *   test_model [--dir D] [--seed N]
 *
 * Writes small random containers (FORMAT.md) into a scratch directory and checks:
 *   - error paths: NULL handles and pointers, bad options, missing/garbage files,
 *     token ids outside the vocabulary, n < 0, KV capacity overflow (and that the
 *     engine is unchanged afterwards), rewind beyond pos, trace/replay files that
 *     do not fit, an expert slab that cannot be read (eval fails, pos unchanged);
 *   - INV-DET-2: one batched eval == token-by-token == chunked (max_batch 3, 5);
 *   - INV-DET-1: identical logits across thread counts, minimum/full cache, LRU/LFU,
 *     prefetch off/next/shared (+extra), buffered/direct I/O, a mirror, pinning and
 *     warm start from a heat profile written by usage_out;
 *   - INV-DET-3: scalar / AVX2 / AVX-512 engines agree bit for bit (where runnable);
 *   - trace format (FORMAT.md §9), and replaying an engine's own trace reproduces
 *     its logits exactly; rewind; stats bookkeeping;
 *   - bit-exact agreement with a naive single-token forward pass written from
 *     NUMERICS §1-5 in this file (independent of model.c), which pins the
 *     canonical operation order that a tolerance check cannot see.
 * Two model flavours cover the forward pass: GQA (qkv bias, per-head qk-norm,
 * partial NEOX RoPE, a dense layer, softmax routing with norm_topk_prob, gated
 * shared expert, tied embeddings, non-unit emb/residual/logit scales, mixed
 * per-expert dtypes) and MLA (q_lora, GPT-J RoPE with attn factor, sigmoid +
 * bias + group-limited routing, routed_scale, ungated shared expert), plus GQA
 * with full-width qk-norm and MLA without q_lora.
 * Two more flavours duplicate router rows so that routing ties exactly (ties go to
 * the lower expert / group index), and the naive reference's routing is compared
 * with the engine's trace. Further: option defaults, caps and environment
 * overrides, log level rules, LFU heat / usage counts independent of batching, a
 * multi-chunk eval that fails in its second chunk (no trace rows, counters or
 * position kept), a trace_start that fails while a trace is active, regions of
 * tiny work staying on the caller (thread-count overhead smoke check), and a
 * hearth-bench smoke run (engine/tools/hearth_bench.c is compiled in).
 * Agreement with the Python ground truth (hearth.reference) is the golden suite's
 * job (tests/golden).
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L   /* setenv/unsetenv under -std=c11 */
#endif
#include "hearth.h"
#include "hx_model.h"
#include "hx_modelfile.h"
#include "hx_quant.h"
#include "hx_store.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEARTH_BENCH_NO_MAIN
#include "../tools/hearth_bench.c"

static int g_checks, g_fail;
static char g_dir[600];
static char g_tag[32];

#define CHECK(cond, ...)                                          \
    do {                                                          \
        g_checks++;                                               \
        if (!(cond)) {                                            \
            if (g_fail++ < 50) {                                  \
                printf("  FAIL %s:%d: ", __FILE__, __LINE__);     \
                printf(__VA_ARGS__);                              \
                printf("\n");                                     \
                fflush(stdout);                                   \
            }                                                     \
        }                                                         \
    } while (0)

/* ------------------------------------------------------------ random */

static uint64_t g_rng = 1;
static uint64_t splitmix(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static float urand(void) { return (float)((splitmix(&g_rng) >> 40) * (1.0 / 16777216.0)); }
static float nrand(void) {   /* Irwin-Hall approximation of N(0, 1) */
    float s = 0.0f;
    for (int i = 0; i < 12; i++) s += urand();
    return s - 6.0f;
}

/* ------------------------------------------------------------ container writer */

typedef struct wtensor {
    char name[80];
    int dtype, ndim;
    int64_t shape[2];
    uint8_t *data;
    uint64_t nbytes, off;
} wtensor;

typedef struct writer {
    uint8_t meta[4096];
    size_t meta_len;
    wtensor t[256];
    int nt;
    int L, E, D, F;
    uint8_t lk[16];
    int *edtype;          /* [L*E] */
    uint8_t **slab;       /* [L*E] */
    uint64_t *slab_nb;
} writer;

static void put16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void put64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }

static void meta_key(writer *w, const char *k, uint8_t type) {
    size_t n = strlen(k);
    put16(w->meta + w->meta_len, (uint32_t)n);
    memcpy(w->meta + w->meta_len + 2, k, n);
    w->meta[w->meta_len + 2 + n] = type;
    w->meta_len += 3 + n;
}
static void meta_u32(writer *w, const char *k, uint32_t v) {
    meta_key(w, k, 1);
    put32(w->meta + w->meta_len, v);
    w->meta_len += 4;
}
static void meta_f32(writer *w, const char *k, float v) {
    uint32_t b;
    memcpy(&b, &v, 4);
    meta_key(w, k, 2);
    put32(w->meta + w->meta_len, b);
    w->meta_len += 4;
}
static void meta_str(writer *w, const char *k, const char *v) {
    size_t n = strlen(v);
    meta_key(w, k, 4);
    put32(w->meta + w->meta_len, (uint32_t)n);
    memcpy(w->meta + w->meta_len + 4, v, n);
    w->meta_len += 4 + n;
}
static void meta_u8a(writer *w, const char *k, const uint8_t *v, uint32_t n) {
    meta_key(w, k, 7);
    put32(w->meta + w->meta_len, n);
    memcpy(w->meta + w->meta_len + 4, v, n);
    w->meta_len += 4 + n;
}

/* Random [rows, cols] (or [n] when rows == 0) tensor with std ~ gain/sqrt(cols). */
static void add_tensor(writer *w, const char *name, int dtype, int64_t rows, int64_t cols, float gain, float mean) {
    wtensor *t = &w->t[w->nt++];
    const int64_t r = rows ? rows : 1;
    const size_t rb = hx_row_bytes(dtype, cols);
    float *row = (float *)malloc(sizeof(float) * (size_t)cols);
    snprintf(t->name, sizeof t->name, "%s", name);
    t->dtype = dtype;
    t->ndim = rows ? 2 : 1;
    t->shape[0] = rows ? rows : cols;
    t->shape[1] = rows ? cols : 1;
    t->nbytes = (uint64_t)r * rb;
    t->data = (uint8_t *)calloc(1, (size_t)t->nbytes);
    for (int64_t i = 0; i < r; i++) {
        for (int64_t j = 0; j < cols; j++) row[j] = mean + gain * nrand() / sqrtf((float)cols);
        hx_quantize_row(dtype, row, t->data + (size_t)i * rb, cols);
    }
    free(row);
}

static void add_layer_tensor(writer *w, int l, const char *suffix, int dtype, int64_t rows, int64_t cols, float gain, float mean) {
    char name[80];
    snprintf(name, sizeof name, "blk.%d.%s", l, suffix);
    add_tensor(w, name, dtype, rows, cols, gain, mean);
}

static void add_norm(writer *w, int l, const char *suffix, int64_t n) {
    char name[80];
    wtensor *t;
    if (l < 0) snprintf(name, sizeof name, "%s", suffix);
    else snprintf(name, sizeof name, "blk.%d.%s", l, suffix);
    add_tensor(w, name, HEARTH_F32, 0, n, 0.0f, 0.0f);
    t = &w->t[w->nt - 1];
    for (int64_t i = 0; i < n; i++) {
        float v = 0.8f + 0.4f * urand();
        memcpy(t->data + 4 * i, &v, 4);
    }
}

static void add_experts(writer *w, int L, int E, int D, int F, const uint8_t *lk, int (*dtype_of)(int, int)) {
    w->L = L; w->E = E; w->D = D; w->F = F;
    memcpy(w->lk, lk, (size_t)L);
    w->edtype = (int *)calloc((size_t)(L * E), sizeof(int));
    w->slab = (uint8_t **)calloc((size_t)(L * E), sizeof(uint8_t *));
    w->slab_nb = (uint64_t *)calloc((size_t)(L * E), sizeof(uint64_t));
    float *row = (float *)malloc(sizeof(float) * (size_t)(D > F ? D : F));
    for (int l = 0; l < L; l++) {
        if (!lk[l]) continue;
        for (int e = 0; e < E; e++) {
            const int i = l * E + e, dt = dtype_of(l, e);
            uint64_t og, ou, od, nb = hx_slab_layout(dt, D, F, &og, &ou, &od);
            uint8_t *s = (uint8_t *)calloc(1, (size_t)nb);
            const size_t rd = hx_row_bytes(dt, D), rf = hx_row_bytes(dt, F);
            for (int r = 0; r < F; r++) {
                for (int j = 0; j < D; j++) row[j] = nrand() / sqrtf((float)D);
                hx_quantize_row(dt, row, s + og + (size_t)r * rd, D);
                for (int j = 0; j < D; j++) row[j] = nrand() / sqrtf((float)D);
                hx_quantize_row(dt, row, s + ou + (size_t)r * rd, D);
            }
            for (int r = 0; r < D; r++) {
                for (int j = 0; j < F; j++) row[j] = nrand() / sqrtf((float)F);
                hx_quantize_row(dt, row, s + od + (size_t)r * rf, F);
            }
            w->edtype[i] = dt;
            w->slab[i] = s;
            w->slab_nb[i] = nb;
        }
    }
    free(row);
}

static uint64_t al(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

static int write_container(writer *w, const char *path) {
    const uint64_t ne = w->E ? (uint64_t)w->L * (uint64_t)w->E : 0;
    const uint64_t meta_off = 64, tdir_off = al(meta_off + w->meta_len, 64);
    const uint64_t edir_off = tdir_off + 128ull * (uint64_t)w->nt;
    uint64_t off = al(edir_off + 32 * ne, 64), size;
    uint8_t *buf;
    FILE *f;
    int ok;
    for (int i = 0; i < w->nt; i++) {
        w->t[i].off = off;
        off = al(off + w->t[i].nbytes, 64);
    }
    off = al(off, 4096);
    uint64_t *eoff = (uint64_t *)calloc(ne ? (size_t)ne : 1, sizeof(uint64_t));
    for (uint64_t i = 0; i < ne; i++) {
        if (!w->slab_nb[i]) continue;
        eoff[i] = off;
        off += w->slab_nb[i];
    }
    size = off;
    buf = (uint8_t *)calloc(1, (size_t)size);
    put32(buf, HX_MAGIC);
    put32(buf + 4, 1);
    put64(buf + 8, meta_off);
    put64(buf + 16, w->meta_len);
    put64(buf + 24, tdir_off);
    put64(buf + 32, (uint64_t)w->nt);
    put64(buf + 40, edir_off);
    put64(buf + 48, ne);
    put32(buf + 56, 4096);
    memcpy(buf + meta_off, w->meta, w->meta_len);
    for (int i = 0; i < w->nt; i++) {
        uint8_t *e = buf + tdir_off + 128ull * (uint64_t)i;
        const wtensor *t = &w->t[i];
        memcpy(e, t->name, strlen(t->name));
        put32(e + 80, (uint32_t)t->dtype);
        put32(e + 84, (uint32_t)t->ndim);
        put32(e + 88, (uint32_t)t->shape[0]);
        put32(e + 92, (uint32_t)t->shape[1]);
        put32(e + 96, 1);
        put32(e + 100, 1);
        put64(e + 104, t->off);
        put64(e + 112, t->nbytes);
        memcpy(buf + t->off, t->data, (size_t)t->nbytes);
    }
    for (uint64_t i = 0; i < ne; i++) {
        uint8_t *e = buf + edir_off + 32 * i;
        put64(e, eoff[i]);
        put64(e + 8, w->slab_nb[i]);
        put32(e + 16, (uint32_t)(w->slab_nb[i] ? w->edtype[i] : 0));
        if (w->slab_nb[i]) memcpy(buf + eoff[i], w->slab[i], (size_t)w->slab_nb[i]);
    }
    f = fopen(path, "wb");
    ok = f && fwrite(buf, 1, (size_t)size, f) == (size_t)size;
    if (f) ok = fclose(f) == 0 && ok;
    free(buf);
    free(eoff);
    return ok;
}

static void writer_free(writer *w) {
    for (int i = 0; i < w->nt; i++) free(w->t[i].data);
    if (w->slab)
        for (int i = 0; i < w->L * w->E; i++) free(w->slab[i]);
    free(w->slab);
    free(w->slab_nb);
    free(w->edtype);
}

/* ------------------------------------------------------------ test models */

enum { V_ = 160, D_ = 128, L_ = 3, E_ = 8, F_ = 64, MAXSEQ_ = 48, LONG_ = 300 };

static int g_max_seq = MAXSEQ_;      /* max_seq written by make_gqa / make_mla */
static int g_bad_dtype = 0;          /* make_gqa: 1 = attn_o as I32 (no kernel), 2 = attn_norm too short,
                                        3 = attn_q with too few rows */
static int g_no_shared = 0;          /* make_gqa: no shared expert (prefetch shared falls back to next) */
static int g_mla_heads = 4;          /* make_mla: > 16 reaches the second MLA score head chunk */
static int g_tie = 0;                /* duplicate router rows: GQA triples, MLA groups 0, 1, 2 (of 4) */
static double g_par_min = 0.0;       /* hx_model_set_parallel_min for engines opened by open_with */

/* Router row e (and its bias) becomes a copy of row src(e): exact routing ties.
 * GQA (top-2): triples, so two of three equal experts are chosen. MLA (4 groups,
 * topk_group 2, top-3): the first three groups are equal, so the kept groups and
 * the experts within them are decided by ties. */
static void tie_rows(wtensor *t, int E, int mla) {
    const size_t rb = t->nbytes / (uint64_t)E;
    for (int e = 0; e < E; e++) {
        const int src = mla ? (e < E - E / 4 ? e % (E / 4) : e) : e - e % 3;
        if (src != e) memcpy(t->data + (size_t)e * rb, t->data + (size_t)src * rb, rb);
    }
}

static int dt_mixed(int l, int e) {
    static const int dts[] = {HEARTH_Q4, HEARTH_Q8, HEARTH_Q4, HEARTH_F16, HEARTH_Q4, HEARTH_F32, HEARTH_BF16, HEARTH_Q4};
    return dts[(l * 3 + e) % 8];
}
static int dt_q4(int l, int e) { (void)l; (void)e; return HEARTH_Q4; }

/* GQA flavour; qk_norm 1 or 2. */
static int make_gqa(const char *path, int qk_norm) {
    writer *w = (writer *)calloc(1, sizeof *w);
    const int H = 4, Hkv = 2, hd = 64, rope = 48, Fd = 128, Fs = g_no_shared ? 0 : 64;
    const uint8_t lk[L_] = {0, 1, 1};
    meta_str(w, "arch", "qwen2_moe");
    meta_u32(w, "n_layers", L_);
    meta_u32(w, "d_model", D_);
    meta_u32(w, "vocab_size", V_);
    meta_u32(w, "max_seq", (uint32_t)g_max_seq);
    meta_u32(w, "n_heads", H);
    meta_u32(w, "n_kv_heads", Hkv);
    meta_u32(w, "head_dim", hd);
    meta_u32(w, "qk_norm", (uint32_t)qk_norm);
    meta_u32(w, "qkv_bias", 1);
    meta_u32(w, "rope_dim", rope);
    meta_u32(w, "dense_ffn_dim", Fd);
    meta_u32(w, "n_experts", E_);
    meta_u32(w, "top_k", 2);
    meta_u32(w, "expert_ffn_dim", F_);
    meta_u32(w, "shared_ffn_dim", Fs);
    meta_u32(w, "shared_gate", Fs ? 1 : 0);
    meta_u32(w, "norm_topk_prob", 1);
    meta_f32(w, "emb_scale", 1.5f);
    meta_f32(w, "residual_scale", 0.75f);
    meta_f32(w, "logit_scale", 0.5f);
    meta_u32(w, "tie_embeddings", 1);
    meta_u8a(w, "layer_kind", lk, L_);
    meta_u32(w, "bos_id", 1);
    add_tensor(w, "tok_embd", HEARTH_Q8, V_, D_, 8.0f, 0.0f);
    add_norm(w, -1, "out_norm", D_);
    add_tensor(w, "rope_inv_freq", HEARTH_F32, 0, rope / 2, 0.0f, 0.0f);
    for (int j = 0; j < rope / 2; j++) {
        float v = powf(10000.0f, -(float)(2 * j) / (float)rope);
        memcpy(w->t[w->nt - 1].data + 4 * j, &v, 4);
    }
    for (int l = 0; l < L_; l++) {
        add_norm(w, l, "attn_norm", g_bad_dtype == 2 && l == 1 ? D_ - 64 : D_);
        add_norm(w, l, "ffn_norm", D_);
        add_layer_tensor(w, l, "attn_q", HEARTH_Q8, H * hd - (g_bad_dtype == 3 ? 64 : 0), D_, 1.0f, 0.0f);
        add_layer_tensor(w, l, "attn_k", HEARTH_F16, Hkv * hd, D_, 1.0f, 0.0f);
        add_layer_tensor(w, l, "attn_v", HEARTH_Q4, Hkv * hd, D_, 1.0f, 0.0f);
        add_layer_tensor(w, l, "attn_o", g_bad_dtype == 1 ? HEARTH_I32 : HEARTH_BF16, D_, H * hd, 1.0f, 0.0f);
        add_layer_tensor(w, l, "attn_q_bias", HEARTH_F32, 0, H * hd, 1.0f, 0.0f);
        add_layer_tensor(w, l, "attn_k_bias", HEARTH_F32, 0, Hkv * hd, 1.0f, 0.0f);
        add_layer_tensor(w, l, "attn_v_bias", HEARTH_F32, 0, Hkv * hd, 1.0f, 0.0f);
        add_norm(w, l, "attn_q_norm", qk_norm == 1 ? hd : H * hd);
        add_norm(w, l, "attn_k_norm", qk_norm == 1 ? hd : Hkv * hd);
        if (!lk[l]) {
            add_layer_tensor(w, l, "ffn_gate", HEARTH_Q8, Fd, D_, 1.0f, 0.0f);
            add_layer_tensor(w, l, "ffn_up", HEARTH_Q8, Fd, D_, 1.0f, 0.0f);
            add_layer_tensor(w, l, "ffn_down", HEARTH_Q4, D_, Fd, 1.0f, 0.0f);
        } else {
            add_layer_tensor(w, l, "moe_router", HEARTH_F32, E_, D_, 3.0f, 0.0f);
            if (g_tie) tie_rows(&w->t[w->nt - 1], E_, 0);
            if (Fs) {
                add_layer_tensor(w, l, "shexp_gate", HEARTH_Q8, Fs, D_, 1.0f, 0.0f);
                add_layer_tensor(w, l, "shexp_up", HEARTH_F32, Fs, D_, 1.0f, 0.0f);
                add_layer_tensor(w, l, "shexp_down", HEARTH_Q8, D_, Fs, 1.0f, 0.0f);
                add_layer_tensor(w, l, "shexp_gate_inp", HEARTH_F32, 1, D_, 2.0f, 0.0f);
            }
        }
    }
    add_experts(w, L_, E_, D_, F_, lk, dt_mixed);
    int ok = write_container(w, path);
    writer_free(w);
    free(w);
    return ok;
}

/* MLA flavour; q_lora 0 or 64. */
static int make_mla(const char *path, int q_lora) {
    writer *w = (writer *)calloc(1, sizeof *w);
    const int H = g_mla_heads, nope = 32, rope = 16, vd = 32, C = 64, Fs = 64, E = 16, K = 3, L = 4;
    const uint8_t lk[4] = {1, 1, 0, 1};
    meta_str(w, "arch", "deepseek_v3");
    meta_u32(w, "n_layers", L);
    meta_u32(w, "d_model", D_);
    meta_u32(w, "vocab_size", V_);
    meta_u32(w, "max_seq", (uint32_t)g_max_seq);
    meta_u32(w, "n_heads", H);
    meta_u32(w, "attn_kind", 1);
    meta_u32(w, "q_lora_rank", (uint32_t)q_lora);
    meta_u32(w, "kv_lora_rank", C);
    meta_u32(w, "qk_nope_dim", nope);
    meta_u32(w, "qk_rope_dim", rope);
    meta_u32(w, "v_head_dim", vd);
    meta_u32(w, "rope_dim", q_lora ? rope : rope / 2);
    meta_u32(w, "rope_style", 1);
    meta_f32(w, "rope_attn_factor", 1.1f);
    meta_f32(w, "attn_scale", 0.17f);
    meta_u32(w, "dense_ffn_dim", 192);
    meta_u32(w, "n_experts", E);
    meta_u32(w, "top_k", K);
    meta_u32(w, "expert_ffn_dim", F_);
    meta_u32(w, "shared_ffn_dim", Fs);
    meta_u32(w, "score_fn", 1);
    meta_u32(w, "score_bias", 1);
    meta_u32(w, "n_group", 4);
    meta_u32(w, "topk_group", 2);
    meta_u32(w, "norm_topk_prob", 1);
    meta_f32(w, "routed_scale", 2.5f);
    meta_u8a(w, "layer_kind", lk, (uint32_t)L);
    add_tensor(w, "tok_embd", HEARTH_F16, V_, D_, 8.0f, 0.0f);
    add_tensor(w, "lm_head", HEARTH_Q8, V_, D_, 2.0f, 0.0f);
    add_norm(w, -1, "out_norm", D_);
    {
        const int rd = q_lora ? rope : rope / 2;
        add_tensor(w, "rope_inv_freq", HEARTH_F32, 0, rd / 2, 0.0f, 0.0f);
        for (int j = 0; j < rd / 2; j++) {
            float v = powf(10000.0f, -(float)(2 * j) / (float)rd);
            memcpy(w->t[w->nt - 1].data + 4 * j, &v, 4);
        }
    }
    for (int l = 0; l < L; l++) {
        add_norm(w, l, "attn_norm", D_);
        add_norm(w, l, "ffn_norm", D_);
        if (q_lora) {
            add_layer_tensor(w, l, "attn_q_a", HEARTH_Q8, q_lora, D_, 1.0f, 0.0f);
            add_norm(w, l, "attn_q_a_norm", q_lora);
            add_layer_tensor(w, l, "attn_q_b", HEARTH_Q8, H * (nope + rope), q_lora, 1.0f, 0.0f);
        } else {
            add_layer_tensor(w, l, "attn_q", HEARTH_F16, H * (nope + rope), D_, 1.0f, 0.0f);
        }
        add_layer_tensor(w, l, "attn_kv_a", HEARTH_Q8, C + rope, D_, 1.0f, 0.0f);
        add_norm(w, l, "attn_kv_a_norm", C);
        add_layer_tensor(w, l, "attn_kv_b", q_lora ? HEARTH_Q8 : HEARTH_BF16, H * (nope + vd), C, 1.0f, 0.0f);
        add_layer_tensor(w, l, "attn_o", HEARTH_Q4, D_, H * vd, 1.0f, 0.0f);
        if (!lk[l]) {
            add_layer_tensor(w, l, "ffn_gate", HEARTH_Q8, 192, D_, 1.0f, 0.0f);
            add_layer_tensor(w, l, "ffn_up", HEARTH_Q8, 192, D_, 1.0f, 0.0f);
            add_layer_tensor(w, l, "ffn_down", HEARTH_Q8, D_, 192, 1.0f, 0.0f);
        } else {
            add_layer_tensor(w, l, "moe_router", HEARTH_F32, E, D_, 3.0f, 0.0f);
            if (g_tie) tie_rows(&w->t[w->nt - 1], E, 1);
            add_layer_tensor(w, l, "moe_router_bias", HEARTH_F32, 0, E, 1.0f, 0.0f);
            if (g_tie) tie_rows(&w->t[w->nt - 1], E, 1);
            add_layer_tensor(w, l, "shexp_gate", HEARTH_Q8, Fs, D_, 1.0f, 0.0f);
            add_layer_tensor(w, l, "shexp_up", HEARTH_Q8, Fs, D_, 1.0f, 0.0f);
            add_layer_tensor(w, l, "shexp_down", HEARTH_Q8, D_, Fs, 1.0f, 0.0f);
        }
    }
    add_experts(w, L, E, D_, F_, lk, dt_q4);
    int ok = write_container(w, path);
    writer_free(w);
    free(w);
    return ok;
}

/* ------------------------------------------------------------ helpers */

static void path_in(char *out, size_t n, const char *name) { snprintf(out, n, "%s/%s%s", g_dir, g_tag, name); }

static hearth_engine *open_with(const char *path, hearth_options *o, char *err, size_t errlen) {
    hearth_engine *e;
    o->model_path = path;
    e = hearth_open(o, err, errlen);
    if (e) hx_model_set_parallel_min(hx_engine_model(e), g_par_min);
    return e;
}

static int imin_t(int a, int b) { return a < b ? a : b; }

static void set_env(const char *k, const char *v) {
#if defined(HX_OS_WINDOWS)
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

static hearth_options defaults(void) {
    hearth_options o;
    hearth_default_options(&o);
    o.n_threads = 4;
    o.n_io_threads = 2;
    o.cache_gb = 1.0;
    return o;
}

/* Logits of tokens[0..n): mode 0 one at a time, 1 one batch, 2 all_logits=0 per token
 * prefix checks; out: n*V. Returns 0 on any failure. */
static int run_tokens(hearth_engine *e, const int32_t *tok, int n, int batched, float *out) {
    if (batched) return hearth_eval(e, tok, n, out, 1) == 0;
    for (int i = 0; i < n; i++)
        if (hearth_eval(e, tok + i, 1, out + (size_t)i * V_, 0) != 0) return 0;
    return 1;
}

static int run_cfg(const char *path, hearth_options o, const int32_t *tok, int n, int batched, float *out, hearth_stats *st) {
    char err[512];
    hearth_engine *e = open_with(path, &o, err, sizeof err);
    int ok;
    if (!e) { printf("  open failed: %s\n", err); return 0; }
    ok = run_tokens(e, tok, n, batched, out);
    if (st) hearth_get_stats(e, st);
    hearth_close(e);
    return ok;
}

static int same(const float *a, const float *b, size_t n) { return memcmp(a, b, n * sizeof(float)) == 0; }

static size_t file_size(const char *path) {
    FILE *f = fopen(path, "rb");
    long n;
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fclose(f);
    return n < 0 ? 0 : (size_t)n;
}

static int write_bytes(const char *path, const void *p, size_t n) {
    FILE *f = fopen(path, "wb");
    int ok = f && fwrite(p, 1, n, f) == n;
    if (f) ok = fclose(f) == 0 && ok;
    return ok;
}

static uint8_t *read_all(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    uint8_t *b;
    *n = file_size(path);
    if (!f) return NULL;
    b = (uint8_t *)malloc(*n ? *n : 1);
    if (b && fread(b, 1, *n, f) != *n) { free(b); b = NULL; }
    fclose(f);
    return b;
}

/* ------------------------------------------------------------ tests */

static void test_defaults_and_open_errors(const char *good) {
    hearth_options o;
    char err[512], small[6];
    char garbage[700], missing[700];
    hearth_default_options(&o);
    CHECK(o.cache_gb == 8.0 && o.n_threads == 0 && o.n_io_threads == 0 && o.direct_io == 1, "default options");
    CHECK(o.policy == HEARTH_POLICY_LFU && o.prefetch == HEARTH_PREFETCH_SHARED && o.isa == HEARTH_ISA_AUTO, "default enums");
    CHECK(o.model_path == NULL && o.n_mirrors == 0 && o.max_seq == 0 && o.max_batch == 0 && o.pin_fraction == 0.0f,
          "default zero fields");
    hearth_default_options(NULL);

    CHECK(hearth_open(NULL, err, sizeof err) == NULL && err[0], "open(NULL options) fails with a message");
    CHECK(hearth_open(NULL, NULL, 0) == NULL, "open(NULL, NULL, 0)");
    o = defaults();
    CHECK(hearth_open(&o, err, sizeof err) == NULL && strstr(err, "model path"), "no model path: %s", err);
    o.model_path = "";
    CHECK(hearth_open(&o, err, sizeof err) == NULL && err[0], "empty model path");

    path_in(missing, sizeof missing, "does_not_exist.hearth");
    o = defaults();
    CHECK(open_with(missing, &o, err, sizeof err) == NULL && strstr(err, "does_not_exist"), "missing file: %s", err);
    path_in(garbage, sizeof garbage, "garbage.hearth");
    {
        uint8_t junk[300];
        for (int i = 0; i < 300; i++) junk[i] = (uint8_t)(splitmix(&g_rng) & 0xff);
        memcpy(junk, "HRTH", 4);
        CHECK(write_bytes(garbage, junk, sizeof junk), "write garbage");
        o = defaults();
        CHECK(open_with(garbage, &o, err, sizeof err) == NULL && err[0], "garbage container fails");
        CHECK(open_with(garbage, &o, small, sizeof small) == NULL && strlen(small) < sizeof small,
              "error message truncated to errlen");
        CHECK(open_with(garbage, &o, NULL, 0) == NULL, "garbage, no error buffer");
        remove(garbage);
    }

    const double nan_ = strtod("nan", NULL), inf_ = strtod("inf", NULL);
    struct { const char *what; int field; double v; } bad[] = {
        {"n_threads -1", 0, -1}, {"n_io_threads -2", 1, -2}, {"policy 7", 2, 7}, {"prefetch 9", 3, 9},
        {"isa 9", 4, 9}, {"isa -1", 4, -1}, {"cache_gb NaN", 5, 0}, {"cache_gb -1", 5, -1}, {"cache_gb inf", 5, 0},
        {"n_mirrors 9", 6, 9}, {"n_mirrors -1", 6, -1}, {"max_seq -5", 7, -5}, {"max_batch -1", 8, -1},
        {"prefetch_extra -1", 9, -1}, {"pin_fraction 2", 10, 2}, {"pin_fraction NaN", 10, 0},
        {"mirror NULL", 11, 0},
    };
    bad[6].v = nan_;
    bad[8].v = inf_;
    bad[15].v = nan_;
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
        hearth_engine *e;
        o = defaults();
        switch (bad[i].field) {
        case 0: o.n_threads = (int)bad[i].v; break;
        case 1: o.n_io_threads = (int)bad[i].v; break;
        case 2: o.policy = (int)bad[i].v; break;
        case 3: o.prefetch = (int)bad[i].v; break;
        case 4: o.isa = (int)bad[i].v; break;
        case 5: o.cache_gb = bad[i].v; break;
        case 6: o.n_mirrors = (int)bad[i].v; break;
        case 7: o.max_seq = (int)bad[i].v; break;
        case 8: o.max_batch = (int)bad[i].v; break;
        case 9: o.prefetch_extra = (int)bad[i].v; break;
        case 10: o.pin_fraction = (float)bad[i].v; break;
        case 11: o.n_mirrors = 1; o.mirror_paths[0] = NULL; break;
        }
        err[0] = 0;
        e = open_with(good, &o, err, sizeof err);
        CHECK(e == NULL && err[0], "bad option %s accepted", bad[i].what);
        if (bad[i].field == 11) CHECK(strstr(err, "mirror 0 has no path") != NULL, "NULL mirror: %s", err);
        hearth_close(e);
    }
    {   /* a mirror that is not a copy of the model */
        char other[700];
        hearth_engine *e;
        path_in(other, sizeof other, "not_a_mirror.hearth");
        CHECK(make_gqa(other, 2), "write second container");
        o = defaults();
        o.n_mirrors = 1;
        o.mirror_paths[0] = other;
        e = open_with(good, &o, err, sizeof err);
        CHECK(e == NULL && err[0], "a different file is rejected as a mirror");
        hearth_close(e);
        remove(other);
    }
    hearth_close(NULL);
    {   /* the HEARTH_ISA parser */
        static const struct { const char *s; int ok, isa; } cases[] = {
            {"auto", 1, 0}, {"scalar", 1, 1}, {"AVX2", 1, 2}, {"  avx512", 1, 3}, {"Avx512", 1, 3}, {"0", 1, 0},
            {"3", 1, 3}, {"4", 0, 0}, {"", 0, 0}, {"avx", 0, 0}, {"avx5120", 0, 0}, {"sse", 0, 0}, {"33", 0, 0},
            {"scalar ", 0, 0}, {"@uto", 0, 0}, {"[calar", 0, 0},
        };
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
            int isa = -7, ok = hx_parse_isa(cases[i].s, &isa);
            CHECK(ok == cases[i].ok && (!ok || isa == cases[i].isa), "hx_parse_isa(\"%s\") = %d (isa %d)", cases[i].s, ok, isa);
        }
    }
}

static void test_call_errors(const char *path) {
    hearth_options o = defaults();
    char err[512], tpath[700];
    hearth_model_info info;
    hearth_stats st;
    float lg[V_], lg2[V_];
    int32_t tok[4] = {3, 7, 11, 5};
    hearth_engine *e;
    o.max_seq = 8;
    e = open_with(path, &o, err, sizeof err);
    CHECK(e != NULL, "open: %s", err);
    if (!e) return;
    CHECK(hearth_info(e, &info) == 0 && info.vocab_size == V_ && info.n_layers == L_ && info.max_seq == MAXSEQ_,
          "info: vocab %d layers %d max_seq %d", info.vocab_size, info.n_layers, info.max_seq);
    CHECK(info.n_threads == imin_t(4, hx_num_cpus()) && info.n_io_threads == 2 && info.isa == hearth_cpu_isa() && info.cache_slots > 0,
          "info: threads %d io %d isa %d slots %d", info.n_threads, info.n_io_threads, info.isa, info.cache_slots);
    CHECK(info.n_experts == E_ && info.top_k == 2 && info.n_moe_layers == 2 && info.bos_id == 1 && info.n_eos == 0,
          "info: experts %d top_k %d moe %d bos %d", info.n_experts, info.top_k, info.n_moe_layers, info.bos_id);
    CHECK(info.dense_bytes > 0 && info.expert_bytes > 0 && info.slab_bytes_max > 0 && info.params_total > info.params_active,
          "info: sizes");
    CHECK(strcmp(info.arch, "qwen2_moe") == 0, "info.arch '%s'", info.arch);
    CHECK(info.cache_slots == 2 * E_, "1 GiB holds every expert: %d slots", info.cache_slots);
    {   /* cache_gb 0: the store's minimum, 2*top_k + n_io_threads + 2 */
        hearth_options o0 = defaults();
        hearth_model_info i0;
        hearth_engine *e0;
        o0.cache_gb = 0.0;
        e0 = open_with(path, &o0, err, sizeof err);
        CHECK(e0 && hearth_info(e0, &i0) == 0 && i0.cache_slots == 2 * 2 + 2 + 2, "minimum cache: %d slots",
              e0 ? i0.cache_slots : -1);
        hearth_close(e0);
    }

    CHECK(hearth_info(NULL, &info) < 0 && hearth_info(e, NULL) < 0, "info NULL");
    CHECK(hearth_eval(NULL, tok, 1, lg, 0) < 0, "eval NULL engine");
    CHECK(hearth_eval(e, NULL, 1, lg, 0) < 0, "eval NULL tokens");
    CHECK(hearth_eval(e, tok, -1, lg, 0) < 0, "eval n < 0");
    CHECK(hearth_eval(e, tok, 0, lg, 0) == 0 && hearth_pos(e) == 0, "eval n == 0 is a no-op");
    CHECK(hearth_eval(e, NULL, 0, NULL, 0) == 0, "eval n == 0 with NULL tokens");
    {
        int32_t badtok[3] = {1, V_, 2}, negtok[2] = {-1, 0};
        CHECK(hearth_eval(e, badtok, 3, lg, 1) < 0 && hearth_pos(e) == 0, "token id == vocab rejected, pos unchanged");
        CHECK(hearth_eval(e, negtok, 2, lg, 0) < 0 && hearth_pos(e) == 0, "negative token id rejected");
    }
    CHECK(hearth_pos(NULL) < 0, "pos NULL");
    CHECK(hearth_reset(NULL) < 0 && hearth_rewind(NULL, 0) < 0, "reset/rewind NULL");
    CHECK(hearth_get_stats(NULL, &st) < 0 && hearth_get_stats(e, NULL) < 0, "stats NULL");
    hearth_reset_stats(NULL);
    CHECK(hearth_trace_start(NULL, "x") < 0 && hearth_trace_start(e, NULL) < 0 && hearth_trace_start(e, "") < 0,
          "trace_start bad args");
    path_in(tpath, sizeof tpath, "no_such_dir/sub/t.hrtr");
    CHECK(hearth_trace_start(e, tpath) < 0, "trace into a missing directory fails");
    CHECK(hearth_trace_stop(NULL) < 0 && hearth_trace_stop(e) == 0, "trace_stop");
    {   /* a second trace_start closes the first trace */
        char ta[700], tb[700];
        int32_t t3[3] = {4, 5, 6};
        path_in(ta, sizeof ta, "first.hrtr");
        path_in(tb, sizeof tb, "second.hrtr");
        CHECK(hearth_trace_start(e, ta) == 0 && hearth_eval(e, t3, 2, NULL, 0) == 0, "first trace");
        CHECK(hearth_trace_start(e, tb) == 0 && hearth_eval(e, t3, 3, NULL, 0) == 0 && hearth_trace_stop(e) == 0,
              "second trace");
        CHECK(file_size(ta) == 24 + 2 * 2 * 2 * 2 && file_size(tb) == 24 + 3 * 2 * 2 * 2, "trace sizes %zu %zu",
              file_size(ta), file_size(tb));
        CHECK(remove(ta) == 0, "the first trace file is closed");
        remove(tb);
        hearth_reset(e);
    }
    CHECK(hearth_route_replay(NULL, "x") < 0 && hearth_route_replay(e, NULL) == 0 && hearth_route_replay(e, "") == 0,
          "route_replay NULL handle / off");
    path_in(tpath, sizeof tpath, "missing.hrtr");
    CHECK(hearth_route_replay(e, tpath) < 0, "replay of a missing file fails");

    /* capacity: max_seq 8 */
    {
        int32_t t8[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        CHECK(hearth_eval(e, t8, 9, lg, 0) < 0 && hearth_pos(e) == 0, "9 tokens into capacity 8 rejected");
        CHECK(hearth_eval(e, t8, 6, lg, 0) == 0 && hearth_pos(e) == 6, "6 tokens fit");
        CHECK(hearth_eval(e, t8, 3, lg2, 0) < 0 && hearth_pos(e) == 6, "pos 6 + 3 > 8 rejected, pos unchanged");
        CHECK(hearth_eval(e, t8 + 6, 2, lg2, 0) == 0 && hearth_pos(e) == 8, "exactly full");
        CHECK(hearth_eval(e, t8, 1, lg2, 0) < 0 && hearth_pos(e) == 8, "full cache rejects one more");
        CHECK(hearth_rewind(e, 9) < 0 && hearth_pos(e) == 8, "rewind beyond pos rejected");
        CHECK(hearth_rewind(e, -1) < 0 && hearth_pos(e) == 8, "negative rewind rejected");
        CHECK(hearth_rewind(e, 5) == 0 && hearth_pos(e) == 5, "rewind to 5");
        CHECK(hearth_eval(e, t8 + 5, 1, lg2, 0) == 0, "eval after rewind");
        {   /* the logits at pos 5 after the failures equal a clean run's */
            hearth_engine *e2 = open_with(path, &o, err, sizeof err);
            float ref[V_];
            CHECK(e2 && hearth_eval(e2, t8, 6, ref, 0) == 0 && same(ref, lg2, V_), "failed calls left no trace");
            hearth_close(e2);
        }
        CHECK(hearth_reset(e) == 0 && hearth_pos(e) == 0, "reset");
        CHECK(hearth_eval(e, t8, 2, NULL, 1) == 0 && hearth_pos(e) == 2, "eval with NULL logits");
    }
    {   /* stats bookkeeping */
        int32_t t4[4] = {9, 8, 7, 6};
        hearth_reset(e);
        hearth_reset_stats(e);
        CHECK(hearth_get_stats(e, &st) == 0 && st.tokens == 0 && st.expert_uses == 0 && st.cache_hits == 0, "reset stats");
        CHECK(hearth_eval(e, t4, 4, NULL, 0) == 0 && hearth_eval(e, t4, 1, lg, 0) == 0, "evals for stats");
        hearth_get_stats(e, &st);
        CHECK(st.tokens == 5 && st.forward_calls == 2, "tokens %llu forward_calls %llu", (unsigned long long)st.tokens,
              (unsigned long long)st.forward_calls);
        CHECK(st.expert_uses == 5ull * 2 * 2, "expert_uses %llu", (unsigned long long)st.expert_uses);
        CHECK(st.expert_loads_unique >= 2 * 2 && st.expert_loads_unique <= st.expert_uses, "unique loads %llu",
              (unsigned long long)st.expert_loads_unique);
        CHECK(st.cache_hits + st.cache_misses == st.expert_uses, "hits %llu + misses %llu != uses %llu",
              (unsigned long long)st.cache_hits, (unsigned long long)st.cache_misses, (unsigned long long)st.expert_uses);
        CHECK(st.wall_s > 0 && st.attn_s > 0 && st.moe_s > 0 && st.dense_s > 0 &&
              st.attn_s + st.moe_s + st.dense_s <= st.wall_s * 1.01 + 1e-4, "phase times");
        CHECK(st.stall_s <= st.moe_s + 1e-6, "stall %.6f within moe %.6f", st.stall_s, st.moe_s);
        CHECK(st.cache_slots == info.cache_slots && st.cache_resident <= st.cache_slots, "slots");
    }
    hearth_close(e);
}

/* ------------------------------------------------------------ naive reference
 *
 * An independent single-token forward pass written straight from NUMERICS §1-5,
 * one plain loop per formula, sharing no code with model.c (only the format
 * decoding: hx_dequantize_row, hx_f16_to_f32, the container reader). The engine
 * must reproduce it bit for bit, which pins the canonical operation order the
 * golden tests' tolerance cannot see: rank-order expert sums, sequential
 * attention sums, dot16 lanes, norm_topk's sequential sum, group routing.
 */

typedef struct nref {
    hx_modelfile *mf;
    const hx_config *c;
    FILE *f;
    float *cache;            /* per layer: GQA K [cap][kv], V [cap][kv]; MLA c [cap][C], kpe [cap][r] */
    size_t per_layer;
    int cap, pos;
    int *routes;             /* [token][MoE layer][rank] selected ids, as in a trace */
    size_t n_routes;
} nref;

static float n_dot16(const float *a, const float *b, int64_t n) {
    float L[16] = {0};
    for (int64_t i = 0; i < n; i++) L[i & 15] = L[i & 15] + a[i] * b[i];
    for (int s = 8; s >= 1; s >>= 1)
        for (int j = 0; j < s; j++) L[j] = L[j] + L[j + s];
    return L[0];
}

static void n_rmsnorm(float *y, const float *x, const float *w, int64_t n, float eps) {
    const float ms = n_dot16(x, x, n) / (float)n;
    const float r = 1.0f / sqrtf(ms + eps);
    for (int64_t i = 0; i < n; i++) y[i] = (x[i] * r) * w[i];
}

static void n_softmax(float *x, int64_t n) {
    float m = x[0], L[16] = {0};
    for (int64_t i = 1; i < n; i++) m = x[i] > m ? x[i] : m;
    for (int64_t i = 0; i < n; i++) x[i] = expf(x[i] - m);
    for (int64_t i = 0; i < n; i++) L[i & 15] = L[i & 15] + x[i];
    for (int s = 8; s >= 1; s >>= 1)
        for (int j = 0; j < s; j++) L[j] = L[j] + L[j + s];
    for (int64_t i = 0; i < n; i++) x[i] = x[i] / L[0];
}

static float n_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/* y[r - r0] = row r . x (NUMERICS §3), rows [r0, r1) of a [*, cols] matrix. */
static void n_matvec(int dt, const uint8_t *W, int64_t cols, const float *x, float *y, int64_t r0, int64_t r1) {
    const size_t rb = hx_row_bytes(dt, cols);
    if (dt == HEARTH_Q8 || dt == HEARTH_Q4) {
        const int64_t nb = cols / 64;
        int8_t *xq = (int8_t *)malloc((size_t)cols);
        float *xd = (float *)malloc(sizeof(float) * (size_t)nb);
        for (int64_t g = 0; g < nb; g++) {
            float amax = 0.0f, d, id;
            for (int i = 0; i < 64; i++) amax = fabsf(x[g * 64 + i]) > amax ? fabsf(x[g * 64 + i]) : amax;
            d = amax / 127.0f;
            id = d != 0.0f ? 1.0f / d : 0.0f;
            for (int i = 0; i < 64; i++) {
                float v = nearbyintf(x[g * 64 + i] * id);
                xq[g * 64 + i] = (int8_t)(v > 127.0f ? 127.0f : v < -127.0f ? -127.0f : v);
            }
            xd[g] = d;
        }
        for (int64_t r = r0; r < r1; r++) {
            const uint8_t *row = W + (size_t)r * rb;
            float acc = 0.0f;
            for (int64_t g = 0; g < nb; g++) {
                int32_t isum = 0;
                uint16_t dh;
                if (dt == HEARTH_Q8) {
                    const uint8_t *blk = row + (size_t)g * 66;
                    memcpy(&dh, blk, 2);
                    for (int i = 0; i < 64; i++) isum += (int32_t)(int8_t)blk[2 + i] * xq[g * 64 + i];
                } else {
                    const uint8_t *blk = row + (size_t)g * 34;
                    memcpy(&dh, blk, 2);
                    for (int i = 0; i < 32; i++) {
                        isum += ((int32_t)(blk[2 + i] & 15) - 8) * xq[g * 64 + i];
                        isum += ((int32_t)(blk[2 + i] >> 4) - 8) * xq[g * 64 + 32 + i];
                    }
                }
                acc = acc + (float)isum * (hx_f16_to_f32(dh) * xd[g]);
            }
            y[r - r0] = acc;
        }
        free(xq);
        free(xd);
    } else {
        float *row = (float *)malloc(sizeof(float) * (size_t)cols);
        for (int64_t r = r0; r < r1; r++) {
            hx_dequantize_row(dt, W + (size_t)r * rb, row, cols);
            y[r - r0] = n_dot16(row, x, cols);
        }
        free(row);
    }
}

static const hx_tensor *n_t(const nref *r, int l, const char *name) {
    return l < 0 ? hx_mf_tensor(r->mf, name) : hx_mf_layer_tensor(r->mf, l, name);
}

static void n_lin(const nref *r, int l, const char *name, const float *x, float *y) {
    const hx_tensor *t = n_t(r, l, name);
    n_matvec(t->dtype, (const uint8_t *)t->data, t->shape[1], x, y, 0, t->shape[0]);
}

static const float *n_vec(const nref *r, int l, const char *name) {
    const hx_tensor *t = n_t(r, l, name);
    return t ? (const float *)t->data : NULL;
}

static void n_rope(float *x, int rope_dim, int style, int pos, const float *inv, float f) {
    const int half = rope_dim / 2;
    for (int j = 0; j < half; j++) {
        const float th = (float)pos * inv[j], c = cosf(th) * f, s = sinf(th) * f;
        const int ia = style ? 2 * j : j, ib = style ? 2 * j + 1 : j + half;
        const float a = x[ia], b = x[ib];
        x[ia] = a * c - b * s;
        x[ib] = b * c + a * s;
    }
}

static int nref_open(nref *r, const char *path, int cap) {
    char err[512];
    memset(r, 0, sizeof *r);
    r->mf = hx_modelfile_open(path, 1, err, sizeof err);
    if (!r->mf) { printf("  naive reference: %s\n", err); return 0; }
    r->c = &r->mf->cfg;
    r->f = fopen(path, "rb");
    r->cap = cap;
    if (r->c->attn_kind == HX_ATTN_MLA) r->per_layer = (size_t)cap * (size_t)(r->c->kv_lora_rank + r->c->qk_rope_dim);
    else r->per_layer = (size_t)cap * (size_t)r->c->n_kv_heads * (size_t)r->c->head_dim * 2;
    r->cache = (float *)calloc(r->per_layer * (size_t)r->c->n_layers, sizeof(float));
    r->routes = (int *)calloc((size_t)cap * (size_t)(r->c->n_moe_layers ? r->c->n_moe_layers : 1) *
                                  (size_t)(r->c->top_k ? r->c->top_k : 1), sizeof(int));
    return r->f && r->cache && r->routes;
}

static void nref_close(nref *r) {
    if (r->f) fclose(r->f);
    free(r->cache);
    free(r->routes);
    hx_modelfile_close(r->mf);
}

static void n_attn_gqa(nref *r, int l, const float *x, float *out) {
    const hx_config *c = r->c;
    const int H = c->n_heads, Hkv = c->n_kv_heads, hd = c->head_dim, qd = H * hd, kd = Hkv * hd;
    float *q = (float *)malloc(sizeof(float) * (size_t)qd), *k = (float *)malloc(sizeof(float) * (size_t)kd);
    float *v = (float *)malloc(sizeof(float) * (size_t)kd), *o = (float *)malloc(sizeof(float) * (size_t)qd);
    float *sc = (float *)malloc(sizeof(float) * (size_t)(r->pos + 1));
    float *K = r->cache + (size_t)l * r->per_layer, *Vc = K + (size_t)r->cap * kd;
    n_lin(r, l, "attn_q", x, q);
    n_lin(r, l, "attn_k", x, k);
    n_lin(r, l, "attn_v", x, v);
    if (c->qkv_bias) {
        const float *bq = n_vec(r, l, "attn_q_bias"), *bk = n_vec(r, l, "attn_k_bias"), *bv = n_vec(r, l, "attn_v_bias");
        for (int i = 0; i < qd; i++) q[i] = q[i] + bq[i];
        for (int i = 0; i < kd; i++) k[i] = k[i] + bk[i];
        for (int i = 0; i < kd; i++) v[i] = v[i] + bv[i];
    }
    if (c->qk_norm == 1) {
        for (int h = 0; h < H; h++) n_rmsnorm(q + h * hd, q + h * hd, n_vec(r, l, "attn_q_norm"), hd, c->norm_eps);
        for (int h = 0; h < Hkv; h++) n_rmsnorm(k + h * hd, k + h * hd, n_vec(r, l, "attn_k_norm"), hd, c->norm_eps);
    } else if (c->qk_norm == 2) {
        n_rmsnorm(q, q, n_vec(r, l, "attn_q_norm"), qd, c->norm_eps);
        n_rmsnorm(k, k, n_vec(r, l, "attn_k_norm"), kd, c->norm_eps);
    }
    for (int h = 0; h < H; h++) n_rope(q + h * hd, c->rope_dim, c->rope_style, r->pos, r->mf->rope_inv_freq, c->rope_attn_factor);
    for (int h = 0; h < Hkv; h++) n_rope(k + h * hd, c->rope_dim, c->rope_style, r->pos, r->mf->rope_inv_freq, c->rope_attn_factor);
    memcpy(K + (size_t)r->pos * kd, k, sizeof(float) * (size_t)kd);
    memcpy(Vc + (size_t)r->pos * kd, v, sizeof(float) * (size_t)kd);
    for (int h = 0; h < H; h++) {
        const int g = h / (H / Hkv);
        for (int t = 0; t <= r->pos; t++) sc[t] = c->attn_scale * n_dot16(q + h * hd, K + (size_t)t * kd + g * hd, hd);
        n_softmax(sc, r->pos + 1);
        for (int d = 0; d < hd; d++) {
            float acc = 0.0f;
            for (int t = 0; t <= r->pos; t++) acc = acc + sc[t] * Vc[(size_t)t * kd + g * hd + d];
            o[h * hd + d] = acc;
        }
    }
    n_lin(r, l, "attn_o", o, out);
    free(q); free(k); free(v); free(o); free(sc);
}

static void n_attn_mla(nref *r, int l, const float *x, float *out) {
    const hx_config *c = r->c;
    const int H = c->n_heads, nope = c->qk_nope_dim, rd = c->qk_rope_dim, vd = c->v_head_dim, C = c->kv_lora_rank;
    const int qh = nope + rd;
    const hx_tensor *kvb = n_t(r, l, "attn_kv_b");
    float *q = (float *)malloc(sizeof(float) * (size_t)H * qh), *kva = (float *)malloc(sizeof(float) * (size_t)(C + rd));
    float *ql = (float *)malloc(sizeof(float) * (size_t)C), *ol = (float *)malloc(sizeof(float) * (size_t)C);
    float *wrow = (float *)malloc(sizeof(float) * (size_t)C), *o = (float *)malloc(sizeof(float) * (size_t)H * vd);
    float *sc = (float *)malloc(sizeof(float) * (size_t)(r->pos + 1));
    float *Cc = r->cache + (size_t)l * r->per_layer, *Kp = Cc + (size_t)r->cap * C;
    if (c->q_lora_rank) {
        float *qa = (float *)malloc(sizeof(float) * (size_t)c->q_lora_rank);
        n_lin(r, l, "attn_q_a", x, qa);
        n_rmsnorm(qa, qa, n_vec(r, l, "attn_q_a_norm"), c->q_lora_rank, c->norm_eps);
        n_lin(r, l, "attn_q_b", qa, q);
        free(qa);
    } else {
        n_lin(r, l, "attn_q", x, q);
    }
    n_lin(r, l, "attn_kv_a", x, kva);
    n_rmsnorm(Cc + (size_t)r->pos * C, kva, n_vec(r, l, "attn_kv_a_norm"), C, c->norm_eps);
    memcpy(Kp + (size_t)r->pos * rd, kva + C, sizeof(float) * (size_t)rd);
    n_rope(Kp + (size_t)r->pos * rd, c->rope_dim, c->rope_style, r->pos, r->mf->rope_inv_freq, c->rope_attn_factor);
    for (int h = 0; h < H; h++) {
        float *qn = q + h * qh, *qpe = qn + nope;
        n_rope(qpe, c->rope_dim, c->rope_style, r->pos, r->mf->rope_inv_freq, c->rope_attn_factor);
        for (int i = 0; i < C; i++) ql[i] = 0.0f;
        for (int n = 0; n < nope; n++) {
            hx_dequantize_row(kvb->dtype, (const uint8_t *)kvb->data + (size_t)(h * (nope + vd) + n) * hx_row_bytes(kvb->dtype, C), wrow, C);
            for (int i = 0; i < C; i++) ql[i] = ql[i] + qn[n] * wrow[i];
        }
        for (int t = 0; t <= r->pos; t++)
            sc[t] = c->attn_scale * (n_dot16(ql, Cc + (size_t)t * C, C) + n_dot16(qpe, Kp + (size_t)t * rd, rd));
        n_softmax(sc, r->pos + 1);
        for (int i = 0; i < C; i++) {
            float acc = 0.0f;
            for (int t = 0; t <= r->pos; t++) acc = acc + sc[t] * Cc[(size_t)t * C + i];
            ol[i] = acc;
        }
        n_matvec(kvb->dtype, (const uint8_t *)kvb->data, C, ol, o + h * vd, h * (nope + vd) + nope, h * (nope + vd) + nope + vd);
    }
    n_lin(r, l, "attn_o", o, out);
    free(q); free(kva); free(ql); free(ol); free(wrow); free(o); free(sc);
}

static void n_ffn(nref *r, int l, const char *g, const char *u, const char *d, const float *x, float *out) {
    const hx_tensor *tg = n_t(r, l, g);
    const int64_t F = tg->shape[0];
    float *a = (float *)malloc(sizeof(float) * (size_t)F), *b = (float *)malloc(sizeof(float) * (size_t)F);
    n_lin(r, l, g, x, a);
    n_lin(r, l, u, x, b);
    for (int64_t i = 0; i < F; i++) a[i] = (a[i] / (1.0f + expf(-a[i]))) * b[i];
    n_lin(r, l, d, a, out);
    free(a);
    free(b);
}

static void n_expert(nref *r, int l, int e, const float *x, float *out) {
    const hx_expert_entry *ent = hx_mf_expert(r->mf, l, e);
    const int D = r->c->d_model, F = r->c->expert_ffn_dim, dt = (int)ent->dtype;
    uint64_t og, ou, od;
    uint8_t *slab = (uint8_t *)malloc((size_t)ent->nbytes);
    float *a = (float *)malloc(sizeof(float) * (size_t)F), *b = (float *)malloc(sizeof(float) * (size_t)F);
    hx_slab_layout(dt, D, F, &og, &ou, &od);
    fseek(r->f, (long)ent->offset, SEEK_SET);
    if (fread(slab, 1, (size_t)ent->nbytes, r->f) != (size_t)ent->nbytes) memset(slab, 0, (size_t)ent->nbytes);
    n_matvec(dt, slab + og, D, x, a, 0, F);
    n_matvec(dt, slab + ou, D, x, b, 0, F);
    for (int i = 0; i < F; i++) a[i] = (a[i] / (1.0f + expf(-a[i]))) * b[i];
    n_matvec(dt, slab + od, F, a, out, 0, D);
    free(slab);
    free(a);
    free(b);
}

static int n_better(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

static void n_moe(nref *r, int l, const float *x, float *out) {
    const hx_config *c = r->c;
    const int E = c->n_experts, K = c->top_k, D = c->d_model;
    float *lg = (float *)malloc(sizeof(float) * (size_t)E), *sel = (float *)malloc(sizeof(float) * (size_t)E);
    float *y = (float *)malloc(sizeof(float) * (size_t)D), w[64], gs[64];
    int ids[64], used[1024] = {0}, keep[64] = {0};
    n_lin(r, l, "moe_router", x, lg);
    if (c->score_fn == 0) n_softmax(lg, E);
    else for (int e = 0; e < E; e++) lg[e] = n_sigmoid(lg[e]);
    for (int e = 0; e < E; e++) sel[e] = c->score_bias ? lg[e] + n_vec(r, l, "moe_router_bias")[e] : lg[e];
    if (c->n_group > 1) {
        const int ng = c->n_group, sz = E / ng;
        for (int g = 0; g < ng; g++) {   /* sum of the two largest */
            int a = -1, b = -1;
            for (int i = g * sz; i < (g + 1) * sz; i++) {
                if (a < 0 || n_better(sel[i], i, sel[a], a)) { b = a; a = i; }
                else if (b < 0 || n_better(sel[i], i, sel[b], b)) b = i;
            }
            gs[g] = b >= 0 ? sel[a] + sel[b] : sel[a];
        }
        for (int k = 0; k < c->topk_group; k++) {
            int best = -1;
            for (int g = 0; g < ng; g++)
                if (!keep[g] && (best < 0 || n_better(gs[g], g, gs[best], best))) best = g;
            keep[best] = 1;
        }
        for (int i = 0; i < E; i++)
            if (!keep[i / sz]) sel[i] = 0.0f;
    }
    for (int j = 0; j < K; j++) {
        int best = -1;
        for (int e = 0; e < E; e++)
            if (!used[e] && (best < 0 || n_better(sel[e], e, sel[best], best))) best = e;
        used[best] = 1;
        ids[j] = best;
        w[j] = lg[best];
        r->routes[r->n_routes++] = best;
    }
    if (c->norm_topk_prob) {
        float s = 0.0f;
        for (int j = 0; j < K; j++) s = s + w[j];
        for (int j = 0; j < K; j++) w[j] = w[j] / (s + 1e-20f);
    }
    for (int j = 0; j < K; j++) w[j] = w[j] * c->routed_scale;
    for (int i = 0; i < D; i++) out[i] = 0.0f;
    for (int j = 0; j < K; j++) {
        n_expert(r, l, ids[j], x, y);
        for (int i = 0; i < D; i++) out[i] = out[i] + w[j] * y[i];
    }
    if (c->shared_ffn_dim) {
        n_ffn(r, l, "shexp_gate", "shexp_up", "shexp_down", x, y);
        if (c->shared_gate) {
            const float g = n_sigmoid(n_dot16(n_vec(r, l, "shexp_gate_inp"), x, D));
            for (int i = 0; i < D; i++) y[i] = y[i] * g;
        }
        for (int i = 0; i < D; i++) out[i] = out[i] + y[i];
    }
    free(lg);
    free(sel);
    free(y);
}

static void nref_forward(nref *r, int tok, float *logits) {
    const hx_config *c = r->c;
    const int D = c->d_model;
    const hx_tensor *emb = n_t(r, -1, "tok_embd"), *head = c->tie_embeddings ? emb : n_t(r, -1, "lm_head");
    float *h = (float *)malloc(sizeof(float) * (size_t)D), *x = (float *)malloc(sizeof(float) * (size_t)D);
    float *a = (float *)malloc(sizeof(float) * (size_t)D);
    hx_dequantize_row(emb->dtype, (const uint8_t *)emb->data + (size_t)tok * hx_row_bytes(emb->dtype, D), h, D);
    for (int i = 0; i < D; i++) h[i] = c->emb_scale * h[i];
    for (int l = 0; l < c->n_layers; l++) {
        n_rmsnorm(x, h, n_vec(r, l, "attn_norm"), D, c->norm_eps);
        if (c->attn_kind == HX_ATTN_MLA) n_attn_mla(r, l, x, a);
        else n_attn_gqa(r, l, x, a);
        for (int i = 0; i < D; i++) h[i] = h[i] + c->residual_scale * a[i];
        n_rmsnorm(x, h, n_vec(r, l, "ffn_norm"), D, c->norm_eps);
        if (c->layer_kind[l]) n_moe(r, l, x, a);
        else n_ffn(r, l, "ffn_gate", "ffn_up", "ffn_down", x, a);
        for (int i = 0; i < D; i++) h[i] = h[i] + c->residual_scale * a[i];
    }
    n_rmsnorm(x, h, n_vec(r, -1, "out_norm"), D, c->norm_eps);
    n_matvec(head->dtype, (const uint8_t *)head->data, D, x, logits, 0, c->vocab_size);
    for (int i = 0; i < c->vocab_size; i++) logits[i] = c->logit_scale * logits[i];
    r->pos++;
    free(h);
    free(x);
    free(a);
}

/* Logits of the reference configuration (single-threaded, full cache, no prefetch). */
static float *baseline(const char *path, const int32_t *tok, int n) {
    hearth_options o = defaults();
    float *out = (float *)malloc(sizeof(float) * (size_t)n * V_);
    o.n_threads = 1;
    o.n_io_threads = 1;
    o.cache_gb = 64.0;
    o.prefetch = HEARTH_PREFETCH_OFF;
    o.direct_io = 0;
    CHECK(run_cfg(path, o, tok, n, 0, out, NULL), "baseline run");
    return out;
}

static void test_invariants(const char *path, const char *label, int n) {
    int32_t tok[MAXSEQ_];
    float *base, *got = (float *)malloc(sizeof(float) * (size_t)n * V_);
    char usage[700], err[512];
    hearth_stats st;
    int *nroutes = NULL;     /* the naive reference's routing of tok, for the trace check */
    size_t n_nroutes = 0;
    for (int i = 0; i < n; i++) tok[i] = (int32_t)(splitmix(&g_rng) % V_);
    base = baseline(path, tok, n);
    {
        int finite = 1, varies = 0;
        for (int i = 0; i < n * V_; i++) finite &= isfinite(base[i]) != 0;
        for (int i = 1; i < n; i++) varies |= !same(base, base + (size_t)i * V_, V_) || tok[i] == tok[0];
        CHECK(finite && varies, "%s: logits finite and token dependent", label);
    }
    {   /* bit for bit against the naive NUMERICS reference */
        nref r;
        float *want = (float *)malloc(sizeof(float) * (size_t)n * V_);
        if (nref_open(&r, path, MAXSEQ_)) {
            int worst = -1;
            float md = 0.0f;
            for (int i = 0; i < n; i++) nref_forward(&r, tok[i], want + (size_t)i * V_);
            for (int i = 0; i < n * V_; i++)
                if (memcmp(&want[i], &base[i], 4) && (worst < 0 || fabsf(want[i] - base[i]) > md)) {
                    worst = i;
                    md = fabsf(want[i] - base[i]);
                }
            CHECK(worst < 0, "%s: engine differs from the naive NUMERICS reference (max |diff| %g at token %d)", label,
                  (double)md, worst / V_);
            n_nroutes = r.n_routes;
            nroutes = (int *)malloc(sizeof(int) * (n_nroutes ? n_nroutes : 1));
            if (nroutes) memcpy(nroutes, r.routes, sizeof(int) * n_nroutes);
        } else {
            CHECK(0, "%s: naive reference could not open the container", label);
        }
        nref_close(&r);
        free(want);
    }
    /* par 1: the default scheduling (small regions on the caller); 0: every region of
     * more than one item on the pool, so the parallel paths are exercised */
    struct { const char *what; int batched, threads, io, max_batch, policy, prefetch, extra, direct, isa, mirror; double gb; int par; } cfg[] = {
        {"batched", 1, 4, 2, 0, 1, 2, 0, 1, 0, 0, 1.0, 0},
        {"batched, default scheduling, 16 threads", 1, 16, 2, 0, 1, 2, 0, 1, 0, 0, 1.0, 1},
        {"sequential, default scheduling, 7 threads", 0, 7, 2, 0, 1, 2, 0, 1, 0, 0, 0.0, 1},
        {"batched, chunks of 3", 1, 3, 2, 3, 1, 2, 0, 1, 0, 0, 1.0, 0},
        {"batched, chunks of 5, min cache", 1, 6, 1, 5, 1, 2, 2, 1, 0, 0, 0.0, 0},
        {"batched, min cache, LRU, 16 threads", 1, 16, 4, 0, 0, 1, 0, 1, 0, 0, 0.0, 0},
        {"sequential, min cache, prefetch next+3", 0, 2, 3, 0, 1, 1, 3, 0, 0, 0, 0.0, 0},
        {"sequential, min cache, prefetch shared+1, mirror", 0, 5, 3, 0, 1, 2, 1, 1, 0, 1, 0.0, 0},
        {"sequential, LRU, prefetch off", 0, 1, 8, 0, 0, 0, 0, 1, 0, 0, 0.0, 0},
        {"batched, scalar ISA", 1, 3, 2, 7, 1, 2, 0, 1, HEARTH_ISA_SCALAR, 0, 0.0, 0},
        {"batched, AVX2", 1, 3, 2, 0, 1, 2, 0, 1, HEARTH_ISA_AVX2, 0, 1.0, 0},
        {"sequential, AVX-512", 0, 3, 2, 0, 1, 2, 0, 1, HEARTH_ISA_AVX512, 0, 0.0, 0},
    };
    for (size_t c = 0; c < sizeof cfg / sizeof cfg[0]; c++) {
        hearth_options o = defaults();
        if (cfg[c].isa > hearth_cpu_isa()) {
            hearth_engine *e;
            o.isa = cfg[c].isa;
            e = open_with(path, &o, err, sizeof err);
            CHECK(e == NULL && strstr(err, "ISA"), "%s: an ISA the CPU lacks must fail to open", label);
            hearth_close(e);
            continue;
        }
        o.n_threads = cfg[c].threads;
        o.n_io_threads = cfg[c].io;
        o.max_batch = cfg[c].max_batch;
        o.policy = cfg[c].policy;
        o.prefetch = cfg[c].prefetch;
        o.prefetch_extra = cfg[c].extra;
        o.direct_io = cfg[c].direct;
        o.isa = cfg[c].isa;
        o.cache_gb = cfg[c].gb;
        if (cfg[c].mirror) {
            o.n_mirrors = 1;
            o.mirror_paths[0] = path;
        }
        memset(got, 0, sizeof(float) * (size_t)n * V_);
        g_par_min = cfg[c].par ? -1.0 : 0.0;
        CHECK(run_cfg(path, o, tok, n, cfg[c].batched, got, &st), "%s: %s run", label, cfg[c].what);
        g_par_min = 0.0;
        CHECK(same(got, base, (size_t)n * V_), "%s: %s differs from the baseline", label, cfg[c].what);
    }

    {   /* batched again on a warm cache: whole layers in one compute wave */
        hearth_options o = defaults();
        hearth_engine *e = open_with(path, &o, err, sizeof err);
        CHECK(e && hearth_eval(e, tok, n, got, 1) == 0 && hearth_reset(e) == 0, "%s: cold batched run", label);
        memset(got, 0, sizeof(float) * (size_t)n * V_);
        CHECK(e && hearth_eval(e, tok, n, got, 1) == 0 && same(got, base, (size_t)n * V_), "%s: warm batched run differs", label);
        hearth_close(e);
    }
    /* last-token logits of a batch == the batch's last row; NULL logits then continue */
    {
        hearth_options o = defaults();
        hearth_engine *e = open_with(path, &o, err, sizeof err);
        float last[V_];
        CHECK(e && hearth_eval(e, tok, n, last, 0) == 0 && same(last, base + (size_t)(n - 1) * V_, V_),
              "%s: all_logits=0 returns the last token", label);
        CHECK(e && hearth_rewind(e, n / 2) == 0 && hearth_eval(e, tok + n / 2, n - n / 2 - 1, NULL, 0) == 0 &&
                  hearth_eval(e, tok + n - 1, 1, last, 0) == 0 && same(last, base + (size_t)(n - 1) * V_, V_),
              "%s: rewind, NULL-logits eval, then the last token again", label);
        hearth_close(e);
    }

    /* heat profile round trip: usage_out, then pin + warm start from it */
    path_in(usage, sizeof usage, "model.usage");
    {
        hearth_options o = defaults();
        o.usage_out = usage;
        CHECK(run_cfg(path, o, tok, n, 1, got, NULL), "%s: usage_out run", label);
        CHECK(file_size(usage) > 24, "%s: usage file written (%zu bytes)", label, file_size(usage));
        {   /* a cache a few slots above the minimum: the store never pins into the minimum */
            hearth_model_info mi;
            hearth_engine *e;
            o = defaults();
            o.cache_gb = 0.0;
            e = open_with(path, &o, err, sizeof err);
            CHECK(e && hearth_info(e, &mi) == 0, "%s: info for cache sizing", label);
            hearth_close(e);
            o = defaults();
            o.cache_gb = e ? (double)(mi.cache_slots + 6) * (double)mi.slab_bytes_max / 1073741824.0 : 0.0;
        }
        o.usage_in = usage;
        o.pin_fraction = 0.5f;
        o.warm_start = 1;
        CHECK(run_cfg(path, o, tok, n, 0, got, &st), "%s: pinned run", label);
        CHECK(st.cache_pinned > 0, "%s: some experts pinned (%d)", label, st.cache_pinned);
        CHECK(same(got, base, (size_t)n * V_), "%s: pinning / warm start changed the output", label);
        remove(usage);
    }

    /* trace + replay */
    {
        char tr[700];
        hearth_options o = defaults();
        hearth_engine *e = open_with(path, &o, err, sizeof err);
        hearth_model_info info;
        size_t sz;
        uint8_t *raw;
        path_in(tr, sizeof tr, "model.hrtr");
        CHECK(e && hearth_info(e, &info) == 0, "%s: open for trace", label);
        if (!e) goto out;
        CHECK(hearth_trace_start(e, tr) == 0, "%s: trace_start", label);
        CHECK(hearth_eval(e, tok, 3, got, 1) == 0 && hearth_eval(e, tok + 3, n - 3, got + 3 * V_, 1) == 0, "%s: traced eval", label);
        CHECK(hearth_trace_stop(e) == 0, "%s: trace_stop", label);
        hearth_close(e);
        CHECK(same(got, base, (size_t)n * V_), "%s: traced run differs", label);
        raw = read_all(tr, &sz);
        {
            const size_t row = (size_t)info.n_moe_layers * (size_t)info.top_k;
            uint32_t h[6];
            CHECK(raw && sz == 24 + 2 * row * (size_t)n, "%s: trace size %zu", label, sz);
            for (int i = 0; raw && i < 6; i++) h[i] = raw[4 * i] | (raw[4 * i + 1] << 8) | (raw[4 * i + 2] << 16) | ((uint32_t)raw[4 * i + 3] << 24);
            CHECK(raw && h[0] == 0x52545248u && h[1] == 1 && h[2] == (uint32_t)info.n_layers && h[3] == (uint32_t)info.n_experts &&
                      h[4] == (uint32_t)info.top_k && h[5] == (uint32_t)info.n_moe_layers, "%s: trace header", label);
            int valid = 1;
            for (size_t t = 0; raw && t < (size_t)n * (size_t)info.n_moe_layers; t++) {
                const uint8_t *r = raw + 24 + 2 * t * (size_t)info.top_k;
                for (int j = 0; j < info.top_k; j++) {
                    int id = r[2 * j] | (r[2 * j + 1] << 8);
                    valid &= id < info.n_experts;
                    for (int i = 0; i < j; i++) valid &= id != (r[2 * i] | (r[2 * i + 1] << 8));
                }
            }
            CHECK(valid, "%s: trace ids valid and distinct per top-k", label);
            if (nroutes) {   /* rank order and tie-breaks, as the naive reference routed */
                int agree = raw && sz == 24 + 2 * row * (size_t)n && n_nroutes == row * (size_t)n;
                for (size_t i = 0; agree && i < n_nroutes; i++) agree = (raw[24 + 2 * i] | (raw[25 + 2 * i] << 8)) == nroutes[i];
                CHECK(agree, "%s: traced routing differs from the naive reference's", label);
            }
        }
        /* replaying the model's own routing reproduces its logits, batched and not */
        o = defaults();
        o.cache_gb = 0.0;
        e = open_with(path, &o, err, sizeof err);
        CHECK(e && hearth_route_replay(e, tr) == 0, "%s: route_replay", label);
        if (e) {
            CHECK(run_tokens(e, tok, n, 1, got) && same(got, base, (size_t)n * V_), "%s: replay (batched) differs", label);
            hearth_reset(e);
            CHECK(run_tokens(e, tok, n, 0, got) && same(got, base, (size_t)n * V_), "%s: replay (sequential) differs", label);
            /* a different token sequence follows the replayed ids, not its own routing */
            {
                int32_t other[MAXSEQ_];
                float *own = (float *)malloc(sizeof(float) * (size_t)n * V_);
                for (int i = 0; i < n; i++) other[i] = (tok[i] + 1) % V_;
                hearth_reset(e);
                CHECK(run_tokens(e, other, n, 1, got), "%s: replay other tokens", label);
                hearth_route_replay(e, NULL);
                hearth_reset(e);
                CHECK(run_tokens(e, other, n, 1, own), "%s: own routing other tokens", label);
                CHECK(!same(got, own, (size_t)n * V_), "%s: replay had no effect on other tokens", label);
                free(own);
            }
            /* malformed traces are rejected and leave replay as it was */
            if (raw && sz > 24) {
                char bad[700];
                uint8_t *cp = (uint8_t *)malloc(sz);
                path_in(bad, sizeof bad, "bad.hrtr");
                for (int k = 0; k < 9; k++) {
                    size_t len = sz;
                    memcpy(cp, raw, sz);
                    switch (k) {
                    case 7: put32(cp + 8, (uint32_t)info.n_moe_layers - 1); break;   /* n_layers < n_moe_layers */
                    case 8: put32(cp + 4, 2); break;                                 /* version */
                    case 0: cp[0] ^= 1; break;                          /* magic */
                    case 1: cp[12] ^= 1; break;                         /* n_experts */
                    case 2: cp[16] ^= 3; break;                         /* top_k */
                    case 3: cp[20] ^= 1; break;                         /* n_moe_layers */
                    case 4: len = sz - 1; break;                        /* partial row */
                    case 5: len = 24; break;                            /* no rows */
                    case 6: cp[24] = (uint8_t)info.n_experts; cp[25] = 0; break;   /* id out of range */
                    }
                    CHECK(write_bytes(bad, cp, len), "write bad trace");
                    CHECK(hearth_route_replay(e, bad) < 0, "%s: malformed trace %d accepted", label, k);
                }
                if (info.top_k > 1) {   /* duplicate id within one top-k */
                    memcpy(cp, raw, sz);
                    cp[26] = cp[24];
                    cp[27] = cp[25];
                    CHECK(write_bytes(bad, cp, sz) && hearth_route_replay(e, bad) < 0, "%s: duplicate id accepted", label);
                }
                CHECK(write_bytes(bad, raw, 23) && hearth_route_replay(e, bad) < 0, "%s: truncated header accepted", label);
                /* n_layers only needs to be >= n_moe_layers; a rejected file leaves the replay running */
                {
                    float *want = (float *)malloc(sizeof(float) * (size_t)n * V_);
                    int32_t other[MAXSEQ_];
                    for (int i = 0; i < n; i++) other[i] = (tok[i] + 1) % V_;
                    memcpy(cp, raw, sz);
                    put32(cp + 8, (uint32_t)info.n_moe_layers);
                    CHECK(write_bytes(bad, cp, sz) && hearth_route_replay(e, bad) == 0, "%s: n_layers = n_moe_layers rejected", label);
                    put32(cp + 8, 1000);
                    CHECK(write_bytes(bad, cp, sz) && hearth_route_replay(e, bad) == 0, "%s: n_layers = 1000 rejected", label);
                    hearth_reset(e);
                    CHECK(want && run_tokens(e, other, n, 1, want), "%s: replay run", label);
                    cp[0] ^= 1;
                    CHECK(write_bytes(bad, cp, sz) && hearth_route_replay(e, bad) < 0, "%s: bad magic accepted", label);
                    hearth_reset(e);
                    CHECK(want && run_tokens(e, other, n, 1, got) && same(got, want, (size_t)n * V_),
                          "%s: a rejected trace changed the active replay", label);
                    free(want);
                }
                remove(bad);
                free(cp);
            }
            hearth_close(e);
        }
        free(raw);
        remove(tr);
    }
out:
    free(base);
    free(got);
    free(nroutes);
}

/* Positions past the score chunk (256) and batches past the q_lat chunk (32), the
 * attention sub-batch forced small: batched == sequential == naive reference. */
static void test_long(const char *path, const char *label) {
    const int n = LONG_;
    int32_t *tok = (int32_t *)malloc(sizeof(int32_t) * n);
    float *seq = (float *)malloc(sizeof(float) * (size_t)n * V_), *got = (float *)malloc(sizeof(float) * (size_t)n * V_);
    char err[512];
    nref r;
    for (int i = 0; i < n; i++) tok[i] = (int32_t)(splitmix(&g_rng) % V_);
    {
        hearth_options o = defaults();
        o.cache_gb = 0.0;
        CHECK(run_cfg(path, o, tok, n, 0, seq, NULL), "%s: sequential", label);
    }
    if (nref_open(&r, path, n)) {
        int bad = -1;
        for (int i = 0; i < n && bad < 0; i++) {
            nref_forward(&r, tok[i], got);
            if (!same(got, seq + (size_t)i * V_, V_)) bad = i;
        }
        CHECK(bad < 0, "%s: sequential differs from the naive reference at position %d", label, bad);
    }
    nref_close(&r);
    /* the last two use the default scheduling; the last one also more threads than
     * physical cores where the CPU has SMT, so the 300-token batch runs on the
     * all-thread pool and decode-sized ones on the per-core pool */
    for (int c = 0; c < 5; c++) {
        static const int tb[5] = {0, 7, 1, 0, 0}, mb[5] = {0, 0, 64, 0, 0};
        const int smt = imin_t(hx_num_cpus(), hx_num_physical_cores() + 2);
        hearth_options o = defaults();
        hearth_engine *e;
        o.max_batch = mb[c];
        o.n_threads = c < 3 ? 3 + c : c == 3 ? 16 : smt;
        g_par_min = c < 3 ? 0.0 : -1.0;
        e = open_with(path, &o, err, sizeof err);
        g_par_min = 0.0;
        CHECK(e != NULL, "%s: open", label);
        if (!e) continue;
        if (tb[c]) hx_model_set_attention_batch(hx_engine_model(e), tb[c]);
        memset(got, 0, sizeof(float) * (size_t)n * V_);
        CHECK(hearth_eval(e, tok, n, got, 1) == 0, "%s: batched eval", label);
        CHECK(same(got, seq, (size_t)n * V_), "%s: batched (attention sub-batch %d, max_batch %d, %d threads) differs", label,
              tb[c], mb[c], o.n_threads);
        if (c == 4) {   /* then decode-sized calls on the same engine */
            float lg[V_];
            CHECK(hearth_rewind(e, n - 4) == 0, "%s: rewind", label);
            for (int i = n - 4; i < n; i++) {
                CHECK(hearth_eval(e, tok + i, 1, lg, 0) == 0 && same(lg, seq + (size_t)i * V_, V_),
                      "%s: decode after a large batch differs at %d (%d threads, %d per core)", label, i, o.n_threads,
                      hx_model_core_threads(hx_engine_model(e)));
            }
        }
        hearth_close(e);
    }
    free(tok);
    free(seq);
    free(got);
}

/* ------------------------------------------------------------ I/O failure */

typedef int (*read_hook_fn)(void *ctx, int layer, int expert, int file, int direct);
void hx_store_set_read_hook(hx_store *s, read_hook_fn fn, void *ctx);

static int fail_all_reads(void *ctx, int layer, int expert, int file, int direct) {
    (void)layer; (void)expert; (void)file; (void)direct;
    return atomic_load((atomic_int *)ctx);
}

static void test_io_failure(const char *path) {
    hearth_options o = defaults();
    char err[512];
    int32_t tok[4] = {5, 6, 7, 8};
    float a[V_ * 4], b[V_ * 4];
    atomic_int failing;
    hearth_engine *e;
    atomic_init(&failing, 1);
    o.cache_gb = 0.0;
    e = open_with(path, &o, err, sizeof err);
    CHECK(e != NULL, "open: %s", err);
    if (!e) return;
    hx_store_set_read_hook(hx_model_store(hx_engine_model(e)), fail_all_reads, (void *)&failing);
    CHECK(hearth_eval(e, tok, 4, a, 1) < 0, "eval with unreadable experts must fail");
    CHECK(hearth_pos(e) == 0, "failed eval leaves pos at 0 (got %d)", hearth_pos(e));
    atomic_store(&failing, 0);
    CHECK(hearth_eval(e, tok, 4, a, 1) == 0 && hearth_pos(e) == 4, "eval works once reads succeed again");
    hearth_close(e);
    o = defaults();
    CHECK(run_cfg(path, o, tok, 4, 1, b, NULL) && same(a, b, 4 * V_), "output after a failed eval is unaffected");
}

/* ------------------------------------------------------------ option resolution */

/* Defaults, caps and environment overrides as hearth_open resolves them, and the
 * log level rules (HEARTH_LOG wins; verbose 1/2 only ever raise the level). */
static void test_resolution(const char *path) {
    const int cpus = hx_num_cpus(), cores = imin_t(hx_num_physical_cores(), cpus);
    hearth_options o;
    hearth_model_info info;
    char err[512];
    hearth_engine *e;

    hearth_default_options(&o);
    e = open_with(path, &o, err, sizeof err);
    CHECK(e && hearth_info(e, &info) == 0, "open with default options: %s", err);
    if (e) {
        hx_model *m = hx_engine_model(e);
        CHECK(info.n_threads == cores && info.n_io_threads == 8 && info.isa == hearth_cpu_isa(),
              "defaults: %d threads (want %d), %d I/O threads, isa %d", info.n_threads, cores, info.n_io_threads, info.isa);
        CHECK(hx_model_max_batch(m) == 512 && hx_model_core_threads(m) == cores, "defaults: max_batch %d, %d per-core threads",
              hx_model_max_batch(m), hx_model_core_threads(m));
        CHECK(info.cache_slots == 2 * E_, "8 GiB holds every expert: %d slots", info.cache_slots);
    }
    hearth_close(e);

    o = defaults();
    o.n_threads = cpus + 3;
    o.n_io_threads = 65;
    o.max_batch = 4097;
    e = open_with(path, &o, err, sizeof err);
    CHECK(e && hearth_info(e, &info) == 0, "open with large counts: %s", err);
    if (e) {
        hx_model *m = hx_engine_model(e);
        CHECK(info.n_threads == imin_t(cpus, 256) && info.n_io_threads == 64, "caps: %d threads (%d CPUs), %d I/O threads",
              info.n_threads, cpus, info.n_io_threads);
        CHECK(hx_model_max_batch(m) == 4096 && hx_model_core_threads(m) == cores, "caps: max_batch %d, %d per-core threads",
              hx_model_max_batch(m), hx_model_core_threads(m));
    }
    hearth_close(e);
    o = defaults();
    o.n_threads = 1;
    o.n_io_threads = 1;
    o.max_batch = 7;
    e = open_with(path, &o, err, sizeof err);
    CHECK(e && hearth_info(e, &info) == 0 && info.n_threads == 1 && info.n_io_threads == 1 &&
              hx_model_max_batch(hx_engine_model(e)) == 7 && hx_model_core_threads(hx_engine_model(e)) == 1,
          "explicit small counts are kept");
    hearth_close(e);

    /* environment overrides beat the options */
    set_env("HEARTH_THREADS", "3");
    set_env("HEARTH_IO_THREADS", "5");
    set_env("HEARTH_CACHE_GB", "0");
    set_env("HEARTH_ISA", "Scalar");
    o = defaults();
    o.n_threads = 2;
    o.n_io_threads = 1;
    o.cache_gb = 64.0;
    e = open_with(path, &o, err, sizeof err);
    CHECK(e && hearth_info(e, &info) == 0, "open with environment overrides: %s", err);
    if (e)
        CHECK(info.n_threads == imin_t(3, cpus) && info.n_io_threads == 5 && info.isa == HEARTH_ISA_SCALAR &&
                  info.cache_slots == 2 * 2 + 5 + 2,
              "env: %d threads, %d I/O threads, isa %d, %d slots", info.n_threads, info.n_io_threads, info.isa, info.cache_slots);
    hearth_close(e);
    set_env("HEARTH_THREADS", "lots");   /* not a number: ignored */
    set_env("HEARTH_ISA", NULL);
    e = open_with(path, &o, err, sizeof err);
    CHECK(e && hearth_info(e, &info) == 0 && info.n_threads == imin_t(2, cpus) && info.isa == hearth_cpu_isa(),
          "an unparsable HEARTH_THREADS is ignored");
    hearth_close(e);
    set_env("HEARTH_ISA", "avx9");
    err[0] = 0;
    e = open_with(path, &o, err, sizeof err);
    CHECK(e == NULL && strstr(err, "HEARTH_ISA"), "HEARTH_ISA=avx9 must fail to open: %s", err);
    hearth_close(e);
    set_env("HEARTH_ISA", NULL);
    set_env("HEARTH_THREADS", NULL);
    set_env("HEARTH_IO_THREADS", NULL);
    set_env("HEARTH_CACHE_GB", NULL);

    {
        static const struct { int before; const char *env; int verbose, after; } c[] = {
            {HX_LOG_ERROR, NULL, 1, HX_LOG_INFO}, {HX_LOG_ERROR, NULL, 2, HX_LOG_DEBUG}, {HX_LOG_DEBUG, NULL, 1, HX_LOG_DEBUG},
            {HX_LOG_INFO, NULL, 0, HX_LOG_INFO},  {HX_LOG_WARN, NULL, 0, HX_LOG_WARN},   {HX_LOG_ERROR, "1", 2, HX_LOG_WARN},
            {HX_LOG_DEBUG, "0", 0, HX_LOG_ERROR}, {HX_LOG_ERROR, "3", 0, HX_LOG_DEBUG},
        };
        for (size_t i = 0; i < sizeof c / sizeof c[0]; i++) {
            int lvl;
            hx_set_log_level(c[i].before);
            set_env("HEARTH_LOG", c[i].env);
            o = defaults();
            o.verbose = c[i].verbose;
            e = open_with(path, &o, err, sizeof err);
            lvl = hx_get_log_level();
            hx_set_log_level(HX_LOG_ERROR);
            CHECK(e != NULL && lvl == c[i].after, "log case %d: level %d, want %d", (int)i, lvl, c[i].after);
            hearth_close(e);
        }
        set_env("HEARTH_LOG", NULL);
        hx_set_log_level(HX_LOG_ERROR);
    }
}

/* ------------------------------------------------------------ prefetch gate */

/* Predictions are handed to the store only while they have been precise: under a
 * replayed random routing the router's predictions hit ~top_k/E = 25% of the time,
 * below the gate (50%), so after a few dozen tokens nothing more is prefetched.
 * Reads only: the output is unaffected either way. */
static void test_prefetch_gate(const char *path) {
    const int K = 2, n_moe = 2, rows = 64;
    char tr[700], err[512];
    uint8_t *buf = (uint8_t *)malloc(24 + 2 * (size_t)rows * n_moe * K);
    hearth_options o = defaults();
    hearth_stats st;
    hearth_engine *e;
    int32_t tok[24];
    path_in(tr, sizeof tr, "random.hrtr");
    put32(buf, 0x52545248u);
    put32(buf + 4, 1);
    put32(buf + 8, L_);
    put32(buf + 12, E_);
    put32(buf + 16, (uint32_t)K);
    put32(buf + 20, (uint32_t)n_moe);
    for (int r = 0; r < rows * n_moe; r++) {
        const int a = (int)(splitmix(&g_rng) % E_);
        const int b = (a + 1 + (int)(splitmix(&g_rng) % (E_ - 1))) % E_;
        put16(buf + 24 + 4 * r, (uint32_t)a);
        put16(buf + 26 + 4 * r, (uint32_t)b);
    }
    CHECK(write_bytes(tr, buf, 24 + 2 * (size_t)rows * n_moe * K), "write random trace");
    free(buf);
    for (int i = 0; i < 24; i++) tok[i] = (int32_t)(splitmix(&g_rng) % V_);
    o.cache_gb = 0.0;
    o.prefetch = HEARTH_PREFETCH_NEXT;
    e = open_with(path, &o, err, sizeof err);
    CHECK(e && hearth_route_replay(e, tr) == 0, "prefetch gate: open and replay: %s", err);
    if (e) {
        for (int pass = 0; pass < 3; pass++) {
            if (pass == 2) hearth_reset_stats(e);
            hearth_reset(e);
            for (int i = 0; i < 24; i++) CHECK(hearth_eval(e, tok + i, 1, NULL, 0) == 0, "prefetch gate: eval");
            hearth_get_stats(e, &st);
            if (pass == 0) CHECK(st.prefetch_issued > 0, "prefetch gate: predictions are issued at first");
        }
        CHECK(st.prefetch_issued == 0, "prefetch gate: %llu imprecise predictions still issued",
              (unsigned long long)st.prefetch_issued);
        hearth_close(e);
    }
    remove(tr);
}

/* ------------------------------------------------------------ usage counting */

/* The store counts one use per (token, rank) however the tokens were batched, so
 * the per-expert counts (LFU heat, usage_out) and hits + misses agree for one
 * batch, chunks of 5 and token-by-token evaluation. */
static void test_usage_counts(const char *path) {
    int32_t tok[12];
    float *heat[3] = {NULL, NULL, NULL};
    hearth_stats st;
    hearth_model_info info;
    char err[512];
    int n_heat = 0;
    for (int i = 0; i < 12; i++) tok[i] = (int32_t)(splitmix(&g_rng) % V_);
    for (int c = 0; c < 3; c++) {
        hearth_options o = defaults();
        hearth_engine *e;
        double sum = 0.0;
        o.max_batch = c == 1 ? 5 : 0;
        e = open_with(path, &o, err, sizeof err);
        CHECK(e && hearth_info(e, &info) == 0, "usage counts: open: %s", err);
        if (!e) continue;
        n_heat = info.n_layers * info.n_experts;
        if (c < 2) {
            CHECK(hearth_eval(e, tok, 12, NULL, 0) == 0, "usage counts: batched eval");
        } else {
            for (int i = 0; i < 12; i++) CHECK(hearth_eval(e, tok + i, 1, NULL, 0) == 0, "usage counts: eval");
        }
        hearth_get_stats(e, &st);
        heat[c] = (float *)malloc(sizeof(float) * (size_t)n_heat);
        if (heat[c]) memcpy(heat[c], hx_store_heat(hx_model_store(hx_engine_model(e))), sizeof(float) * (size_t)n_heat);
        hearth_close(e);
        for (int k = 0; heat[c] && k < n_heat; k++) sum += heat[c][k];
        CHECK(sum == (double)st.expert_uses && st.expert_uses == 12ull * 2 * 2, "usage counts %d: heat %.1f, uses %llu", c, sum,
              (unsigned long long)st.expert_uses);
        CHECK(st.cache_hits + st.cache_misses == st.expert_uses, "usage counts %d: hits %llu + misses %llu, uses %llu", c,
              (unsigned long long)st.cache_hits, (unsigned long long)st.cache_misses, (unsigned long long)st.expert_uses);
    }
    CHECK(heat[0] && heat[1] && heat[2] && !memcmp(heat[0], heat[2], sizeof(float) * (size_t)n_heat) &&
              !memcmp(heat[1], heat[2], sizeof(float) * (size_t)n_heat),
          "per-expert use counts depend on batching");
    for (int c = 0; c < 3; c++) free(heat[c]);
}

/* ------------------------------------------------------------ failures and the trace */

typedef struct fail_one {
    atomic_int on;
    int layer, expert;
} fail_one;

static int fail_one_expert(void *ctx, int layer, int expert, int file, int direct) {
    fail_one *f = (fail_one *)ctx;
    (void)file; (void)direct;
    return atomic_load(&f->on) && layer == f->layer && expert == f->expert;
}

/* Logits and routing (ids[t][MoE layer][rank], per = n_moe*top_k) of tok[0..n), traced. */
static int traced_routes(const char *path, const char *tr, const int32_t *tok, int n, float *logits, int *ids, int per) {
    hearth_options o = defaults();
    char err[512];
    size_t sz;
    uint8_t *raw;
    int ok;
    hearth_engine *e = open_with(path, &o, err, sizeof err);
    if (!e) return 0;
    ok = hearth_trace_start(e, tr) == 0 && hearth_eval(e, tok, n, logits, 1) == 0 && hearth_trace_stop(e) == 0;
    hearth_close(e);
    raw = read_all(tr, &sz);
    ok = ok && raw && sz == 24 + 2 * (size_t)per * (size_t)n;
    for (int i = 0; ok && i < n * per; i++) ids[i] = raw[24 + 2 * i] | (raw[25 + 2 * i] << 8);
    free(raw);
    remove(tr);
    return ok;
}

static void test_trace_failures(const char *path) {
    hearth_options o = defaults();
    char err[512], ta[700], tb[700], bad[700];
    hearth_model_info info;
    hearth_stats st;
    int32_t tok[8];
    float want[8 * V_], got[8 * V_];
    int ids[8 * 2 * 2];
    hearth_engine *e;
    size_t row;
    fail_one f;

    path_in(ta, sizeof ta, "keep.hrtr");
    path_in(tb, sizeof tb, "chunks.hrtr");
    path_in(bad, sizeof bad, "no_such_dir/x.hrtr");
    e = open_with(path, &o, err, sizeof err);
    CHECK(e && hearth_info(e, &info) == 0, "trace failures: open: %s", err);
    if (!e) return;
    row = 2 * (size_t)info.n_moe_layers * (size_t)info.top_k;
    for (int i = 0; i < 8; i++) tok[i] = (int32_t)(splitmix(&g_rng) % V_);
    CHECK(hearth_trace_start(e, ta) == 0 && hearth_trace_start(e, bad) < 0, "trace_start into a missing directory fails");
    CHECK(hearth_eval(e, tok, 2, NULL, 0) == 0 && file_size(ta) == 24 + 2 * row,
          "the active trace keeps recording after a failed trace_start (%zu bytes)", file_size(ta));
    CHECK(hearth_trace_start(e, ta) == 0 && file_size(ta) == 24, "restarting the same path starts it afresh (%zu bytes)",
          file_size(ta));
    CHECK(hearth_eval(e, tok, 1, NULL, 0) == 0 && hearth_trace_stop(e) == 0 && file_size(ta) == 24 + row,
          "the restarted trace records (%zu bytes)", file_size(ta));
    CHECK(hearth_eval(e, tok + 1, 2, NULL, 0) == 0 && file_size(ta) == 24 + row, "a stopped trace records (%zu bytes)",
          file_size(ta));
    hearth_close(e);
    remove(ta);

    /* an 8-token eval in chunks of 4 that fails in its second chunk: an expert that
     * only tokens 4..7 use cannot be read (make_gqa: the MoE layers are 1 and 2) */
    atomic_init(&f.on, 0);
    f.layer = -1;
    f.expert = -1;
    for (int attempt = 0; attempt < 64 && f.layer < 0; attempt++) {
        for (int i = 0; i < 8; i++) tok[i] = (int32_t)(splitmix(&g_rng) % V_);
        if (!traced_routes(path, tb, tok, 8, want, ids, 2 * 2)) continue;
        for (int mi = 0; mi < 2 && f.layer < 0; mi++)
            for (int x = 0; x < E_ && f.layer < 0; x++) {
                int early = 0, late = 0;
                for (int t = 0; t < 8; t++)
                    for (int j = 0; j < 2; j++)
                        if (ids[(t * 2 + mi) * 2 + j] == x) {
                            if (t < 4) early++;
                            else late++;
                        }
                if (late && !early) {
                    f.layer = 1 + mi;
                    f.expert = x;
                }
            }
    }
    CHECK(f.layer >= 0, "found an expert that only the second chunk uses");
    if (f.layer < 0) return;
    o = defaults();
    o.max_batch = 4;
    o.prefetch = HEARTH_PREFETCH_OFF;
    e = open_with(path, &o, err, sizeof err);
    CHECK(e != NULL, "open: %s", err);
    if (!e) return;
    hx_store_set_read_hook(hx_model_store(hx_engine_model(e)), fail_one_expert, &f);
    CHECK(hearth_trace_start(e, tb) == 0, "trace_start");
    atomic_store(&f.on, 1);
    CHECK(hearth_eval(e, tok, 8, got, 1) < 0 && hearth_pos(e) == 0, "eval that fails in its second chunk: pos %d", hearth_pos(e));
    hearth_get_stats(e, &st);
    CHECK(file_size(tb) == 24, "the failed call left trace rows (%zu bytes)", file_size(tb));
    CHECK(st.tokens == 0 && st.forward_calls == 0 && st.expert_uses == 0 && st.expert_loads_unique == 0,
          "the failed call left counters: tokens %llu, calls %llu, uses %llu", (unsigned long long)st.tokens,
          (unsigned long long)st.forward_calls, (unsigned long long)st.expert_uses);
    CHECK(st.wall_s > 0.0, "the failed attempt's time is counted");
    atomic_store(&f.on, 0);
    CHECK(hearth_eval(e, tok, 8, got, 1) == 0 && hearth_pos(e) == 8, "the retry succeeds");
    hearth_get_stats(e, &st);
    CHECK(file_size(tb) == 24 + 8 * row && st.tokens == 8 && st.forward_calls == 2 && st.expert_uses == 8 * 2 * 2,
          "retry: trace %zu bytes, tokens %llu, calls %llu", file_size(tb), (unsigned long long)st.tokens,
          (unsigned long long)st.forward_calls);
    CHECK(same(got, want, 8 * V_), "the retry's logits differ from a clean run");
    CHECK(hearth_trace_stop(e) == 0, "trace_stop");
    hearth_close(e);
    {
        size_t sz;
        uint8_t *raw = read_all(tb, &sz);
        int agree = raw && sz == 24 + 8 * row;
        for (int i = 0; agree && i < 8 * 2 * 2; i++) agree = (raw[24 + 2 * i] | (raw[25 + 2 * i] << 8)) == ids[i];
        CHECK(agree, "the trace after the retry differs from a clean trace");
        free(raw);
    }
    remove(tb);
}

/* ------------------------------------------------------------ hearth-bench, overhead */

static void test_bench(const char *path) {
    char tr[700], missing[700], nodir[700], usage[700];
    char *p = (char *)path;
    path_in(tr, sizeof tr, "bench.hrtr");
    path_in(usage, sizeof usage, "bench.usage");
    path_in(missing, sizeof missing, "missing.file");
    path_in(nodir, sizeof nodir, "no_such_dir/x.hrtr");
    {
        char *a[] = {"hearth-bench", p, "--prompt", "5", "--warmup", "2", "--tokens", "4", "--threads", "2", "--io-threads", "1",
                     "--cache-gb", "0", "--policy", "lru", "--prefetch", "next", "--prefetch-extra", "1", "--trace", tr,
                     "--seed", "7", "--isa", "auto", "--max-batch", "3", "--usage-out", usage, "--verbose", "0"};
        CHECK(bench_main((int)(sizeof a / sizeof a[0]), a) == 0, "hearth-bench run");
        CHECK(file_size(tr) == 24 + 2 * 2 * 2 * (5 + 2 + 4), "hearth-bench trace: %zu bytes", file_size(tr));
        CHECK(file_size(usage) > 24, "hearth-bench usage_out: %zu bytes", file_size(usage));
    }
    {
        char *a[] = {"hearth-bench", p, "--prompt", "3", "--tokens", "3", "--replay", tr, "--direct", "0", "--usage-in", usage,
                     "--pin", "0.5", "--warm", "1", "--cache-gb", "1", "--mirror", p, "--max-seq", "20", "--prefetch", "shared"};
        CHECK(bench_main((int)(sizeof a / sizeof a[0]), a) == 0, "hearth-bench replay run");
    }
    {
        char *a[] = {"hearth-bench", p, "--prompt", "0", "--tokens", "2", "--policy", "lfu", "--prefetch", "off", "--isa", "scalar"};
        CHECK(bench_main((int)(sizeof a / sizeof a[0]), a) == 0, "hearth-bench without a prompt");
    }
    {
        char *a0[] = {"hearth-bench"};
        char *a1[] = {"hearth-bench", p, "--tokens"};
        char *a2[] = {"hearth-bench", p, "--bogus", "1"};
        char *a3[] = {"hearth-bench", p, "--tokens", "-1"};
        char *a4[] = {"hearth-bench", p, p};
        char *a5[] = {"hearth-bench", p, "--policy", "fifo"};
        char *a6[] = {"hearth-bench", missing};
        char *a7[] = {"hearth-bench", p, "--replay", missing};
        char *a8[] = {"hearth-bench", p, "--trace", nodir, "--tokens", "1"};
        char *a9[] = {"hearth-bench", p, "--threads", "-2"};
        CHECK(bench_main(1, a0) == 2 && bench_main(3, a1) == 2 && bench_main(4, a2) == 2 && bench_main(4, a3) == 2 &&
                  bench_main(3, a4) == 2 && bench_main(4, a5) == 2,
              "hearth-bench usage errors exit 2");
        CHECK(bench_main(2, a6) == 1 && bench_main(4, a7) == 1 && bench_main(6, a8) == 1 && bench_main(4, a9) == 1,
              "hearth-bench open / replay / trace failures exit 1");
    }
    remove(tr);
    remove(usage);
}

/* Regions of tiny work stay on the calling thread: a region on the pool waits for
 * every worker, and once threads plus other load exceed the logical CPUs each one
 * costs about a worker spin period (decode of a model this size was ~100x slower
 * at 32 threads with every region dispatched and 1 ms spins). Checked by count (a
 * tiny model's decode runs no region on the pool; with the threshold at 0 it does)
 * and by time (best of three, 4x + 20 ms: only a loaded machine shows the cliff). */
static void test_overhead(const char *path) {
    const int many = hx_num_cpus();
    double best[2] = {1e30, 1e30};
    float out[2][V_];
    char err[512];
    for (int rep = 0; rep < 3; rep++)
        for (int c = 0; c < 2; c++) {
            hearth_options o = defaults();
            hearth_engine *e;
            uint64_t t0;
            double s;
            int32_t t = 3;
            o.n_threads = c ? many : 1;
            g_par_min = rep == 2 && c ? 0.0 : -1.0;
            e = open_with(path, &o, err, sizeof err);
            g_par_min = 0.0;
            CHECK(e != NULL, "overhead: open: %s", err);
            if (!e) return;
            for (int i = 0; i < 40; i++) {   /* loads the experts */
                t = (t * 7 + 1) % V_;
                hearth_eval(e, &t, 1, out[c], 0);
            }
            hearth_reset(e);
            t = 3;
            t0 = hx_now_ns();
            for (int i = 0; i < 40; i++) {
                t = (t * 7 + 1) % V_;
                CHECK(hearth_eval(e, &t, 1, out[c], 0) == 0, "overhead: eval");
            }
            s = (double)(hx_now_ns() - t0) * 1e-9;
            if (rep == 2 && c) {
                CHECK(many < 2 || hx_model_regions(hx_engine_model(e)) > 0, "with threshold 0 decode runs regions on the pool");
            } else {
                CHECK(hx_model_regions(hx_engine_model(e)) == 0, "a tiny model's decode ran %llu regions on the pool",
                      (unsigned long long)hx_model_regions(hx_engine_model(e)));
                if (s < best[c]) best[c] = s;
            }
            hearth_close(e);
        }
    CHECK(same(out[0], out[1], V_), "overhead: the thread count changed the output");
    CHECK(best[1] <= 4.0 * best[0] + 0.02, "40 tiny decode steps take %.4f s on %d threads vs %.4f s on 1", best[1], many, best[0]);
    printf("test_model: 40 tiny decode steps: %.2f ms on 1 thread, %.2f ms on %d\n", best[0] * 1e3, best[1] * 1e3, many);
}

/* ------------------------------------------------------------ main */

static int writable_dir(const char *d) {
    char probe[700];
    FILE *f;
    snprintf(probe, sizeof probe, "%s/hx_test_model_probe_%s", d, g_tag);
    f = fopen(probe, "wb");
    if (!f) return 0;
    fclose(f);
    remove(probe);
    return 1;
}

int main(int argc, char **argv) {
    const char *dir = NULL;
    uint64_t seed = 1, t0 = hx_now_ns();
    char gqa[700], gqa2[700], mla[700], mla0[700];
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dir") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else { printf("usage: test_model [--dir D] [--seed N]\n"); return 2; }
    }
    g_rng = seed;
    {
        uint64_t x = hx_now_ns() ^ (uint64_t)(uintptr_t)&seed;
        snprintf(g_tag, sizeof g_tag, "hxtm_%08llx_", (unsigned long long)(splitmix(&x) & 0xFFFFFFFFull));
    }
    {
        const char *c[] = {dir, hx_env_str("HEARTH_TEST_DIR"), hx_env_str("TEMP"), hx_env_str("TMP"), hx_env_str("TMPDIR"),
#if defined(HX_OS_POSIX)
                           "/tmp",
#endif
                           NULL};
        g_dir[0] = 0;
        for (size_t i = 0; i < sizeof c / sizeof c[0]; i++)
            if (c[i] && *c[i] && writable_dir(c[i])) { snprintf(g_dir, sizeof g_dir, "%s", c[i]); break; }
        if (!g_dir[0]) { printf("test_model: no writable scratch directory (set HEARTH_TEST_DIR or pass --dir)\n"); return 2; }
    }
    {   /* the options under test, not the caller's environment */
        static const char *const vars[] = {"HEARTH_ISA", "HEARTH_THREADS", "HEARTH_IO_THREADS", "HEARTH_CACHE_GB", "HEARTH_LOG"};
        for (size_t i = 0; i < sizeof vars / sizeof vars[0]; i++) set_env(vars[i], NULL);
    }
    hx_set_log_level(HX_LOG_ERROR);
    printf("test_model: seed %llu, scratch dir %s\n", (unsigned long long)seed, g_dir);
    fflush(stdout);

    path_in(gqa, sizeof gqa, "gqa.hearth");
    path_in(gqa2, sizeof gqa2, "gqa_fullnorm.hearth");
    path_in(mla, sizeof mla, "mla.hearth");
    path_in(mla0, sizeof mla0, "mla_noqlora.hearth");
    CHECK(make_gqa(gqa, 1) && make_gqa(gqa2, 2) && make_mla(mla, 64) && make_mla(mla0, 0), "write test containers");

    test_defaults_and_open_errors(gqa);
    test_call_errors(gqa);
    test_invariants(gqa, "gqa", 20);
    test_invariants(gqa2, "gqa qk_norm=2", 9);
    test_invariants(mla, "mla", 20);
    test_invariants(mla0, "mla q_lora=0", 9);
    {   /* no shared expert: prefetch shared falls back to next; Q4/Q8/F16... experts need x quantized */
        char gn[700];
        path_in(gn, sizeof gn, "gqa_noshared.hearth");
        g_no_shared = 1;
        CHECK(make_gqa(gn, 1), "write container without shared experts");
        g_no_shared = 0;
        test_invariants(gn, "gqa no shared expert", 9);
        test_prefetch_gate(gn);
        remove(gn);
    }
    {   /* exact routing ties (tie_rows) */
        char gt[700], mt[700];
        path_in(gt, sizeof gt, "gqa_ties.hearth");
        path_in(mt, sizeof mt, "mla_ties.hearth");
        g_tie = 1;
        CHECK(make_gqa(gt, 1) && make_mla(mt, 64), "write containers with tied routers");
        g_tie = 0;
        test_invariants(gt, "gqa ties", 9);
        test_invariants(mt, "mla ties", 9);
        remove(gt);
        remove(mt);
    }
    test_io_failure(mla);
    test_resolution(gqa);
    test_usage_counts(gqa);
    test_trace_failures(gqa);
    test_bench(gqa);
    test_overhead(gqa);
    {
        char lg[700], lm[700], bad[700], err[512];
        hearth_options o = defaults();
        hearth_engine *e;
        g_max_seq = LONG_ + 8;
        path_in(lg, sizeof lg, "gqa_long.hearth");
        path_in(lm, sizeof lm, "mla_long.hearth");
        g_mla_heads = 20;
        CHECK(make_gqa(lg, 1) && make_mla(lm, 64), "write long containers");
        g_mla_heads = 4;
        test_long(lg, "gqa long");
        test_long(lm, "mla long");
        remove(lg);
        remove(lm);
        g_max_seq = MAXSEQ_;
        g_bad_dtype = 1;
        path_in(bad, sizeof bad, "i32_matrix.hearth");
        CHECK(make_gqa(bad, 1), "write container with an I32 matrix");
        e = open_with(bad, &o, err, sizeof err);
        CHECK(e == NULL && strstr(err, "attn_o"), "a matrix without a kernel must be rejected: %s", err);
        hearth_close(e);
        g_bad_dtype = 2;
        CHECK(make_gqa(bad, 1), "write container with a short norm vector");
        e = open_with(bad, &o, err, sizeof err);
        CHECK(e == NULL && strstr(err, "blk.1.attn_norm"), "a norm vector of the wrong length must be rejected: %s", err);
        hearth_close(e);
        g_bad_dtype = 3;
        CHECK(make_gqa(bad, 1), "write container with a short attn_q");
        e = open_with(bad, &o, err, sizeof err);
        CHECK(e == NULL && strstr(err, "blk.0.attn_q"), "a matrix with the wrong row count must be rejected: %s", err);
        hearth_close(e);
        g_bad_dtype = 0;
        /* KV capacity: 0 = min(model max_seq, 4096); requests above the model's max_seq are capped */
        g_max_seq = 5000;
        CHECK(make_gqa(bad, 1), "write container with max_seq 5000");
        g_max_seq = MAXSEQ_;
        {
            static const int req[3] = {0, 6000, 100}, want[3] = {4096, 5000, 100};
            for (int i = 0; i < 3; i++) {
                o = defaults();
                o.max_seq = req[i];
                e = open_with(bad, &o, err, sizeof err);
                CHECK(e && hx_model_capacity(hx_engine_model(e)) == want[i], "max_seq %d: capacity %d, want %d", req[i],
                      e ? hx_model_capacity(hx_engine_model(e)) : -1, want[i]);
                hearth_close(e);
            }
        }
        remove(bad);
    }
    remove(gqa);
    remove(gqa2);
    remove(mla);
    remove(mla0);
    printf("test_model: %d checks, %d failed (%.1f s)\n", g_checks, g_fail, (double)(hx_now_ns() - t0) / 1e9);
    return g_fail ? 1 : 0;
}
