/*
 * test_modelfile.c — self-checking tests for modelfile.c (hx_modelfile.h, docs/FORMAT.md).
 *
 *   test_modelfile [--quick] [--seed N] [--iters N] [--dir D]
 *   test_modelfile --dump <file.hearth>
 *
 * The test has its own container writer, written from FORMAT.md independently
 * of the reader: random metadata (every §3.1 key either present or defaulted,
 * unknown keys of every type, duplicated keys), tensors of every dtype and rank,
 * mixed-precision and aliased expert slabs, shuffled section order, tensors
 * placed before or after the expert region. Every container is opened and each
 * parsed field and byte is compared with what was written.
 *
 * The fuzzer then opens thousands of variants. Variants that are invalid by
 * construction (truncations, huge counts, misaligned / overlapping /
 * out-of-range entries, inconsistent metadata) must fail with a message.
 * Random corruption must never crash; when such a file is accepted anyway its
 * contents are walked so AddressSanitizer (--asan build) sees every access.
 *
 * --dump prints what the reader parsed as JSON for cross-checking against
 * python/hearth/format.py. Exit code 0 = all checks passed.
 */
#include "hx_modelfile.h"
#include "hx_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_fail;

#define CHECK(cond, ...)                                                         \
    do {                                                                         \
        g_checks++;                                                              \
        if (!(cond)) {                                                           \
            if (++g_fail <= 60) {                                                \
                printf("  FAIL %s:%d: ", __FILE__, __LINE__);                    \
                printf(__VA_ARGS__);                                             \
                printf("\n");                                                    \
            }                                                                    \
        }                                                                        \
    } while (0)

static uint64_t g_rng = 1;
static uint64_t rnd(void) {
    uint64_t z = (g_rng += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static uint32_t rndn(uint32_t n) { return n ? (uint32_t)(rnd() % n) : 0; }
static int chance(int pct) { return (int)rndn(100) < pct; }

static uint64_t al(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

static void w16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void w32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void w64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static uint32_t r32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static uint64_t fnv(const void *p, uint64_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

/* ------------------------------------------------------------ byte buffer */

typedef struct { uint8_t *p; size_t n, cap; } bbuf;

static void bb_put(bbuf *b, const void *d, size_t n) {
    if (b->n + n > b->cap) {
        size_t c = b->cap ? b->cap * 2 : 256;
        while (c < b->n + n) c *= 2;
        b->p = (uint8_t *)realloc(b->p, c);
        if (!b->p) { printf("out of memory\n"); exit(2); }
        b->cap = c;
    }
    if (n) memcpy(b->p + b->n, d, n);
    b->n += n;
}
static void bb_u8(bbuf *b, uint32_t v) { uint8_t x = (uint8_t)v; bb_put(b, &x, 1); }
static void bb_u16(bbuf *b, uint32_t v) { uint8_t x[2]; w16(x, v); bb_put(b, x, 2); }
static void bb_u32(bbuf *b, uint32_t v) { uint8_t x[4]; w32(x, v); bb_put(b, x, 4); }
static void bb_u64(bbuf *b, uint64_t v) { uint8_t x[8]; w64(x, v); bb_put(b, x, 8); }

/* ------------------------------------------------------- FORMAT §3.1 keys */

enum { T_U32 = 1, T_F32 = 2, T_U64 = 3, T_STR = 4, T_U32A = 5, T_F32A = 6, T_U8A = 7 };

enum {
    G_ARCH, G_N_LAYERS, G_D_MODEL, G_VOCAB, G_MAX_SEQ, G_NORM_EPS, G_ATTN_KIND, G_N_HEADS, G_N_KV_HEADS,
    G_HEAD_DIM, G_QK_NORM, G_QKV_BIAS, G_Q_LORA, G_KV_LORA, G_QK_NOPE, G_QK_ROPE, G_V_HEAD, G_ROPE_DIM,
    G_ROPE_STYLE, G_ROPE_ATTN_FACTOR, G_ATTN_SCALE, G_DENSE_FFN, G_N_EXPERTS, G_TOP_K, G_EXPERT_FFN,
    G_SHARED_FFN, G_SHARED_GATE, G_SCORE_FN, G_SCORE_BIAS, G_N_GROUP, G_TOPK_GROUP, G_NORM_TOPK,
    G_ROUTED_SCALE, G_EMB_SCALE, G_RESID_SCALE, G_LOGIT_SCALE, G_TIE, G_LAYER_KIND, G_BOS, G_EOS,
    G_TOKENIZER, G_CHAT_TEMPLATE, G_SOURCE, G_EXPERT_DTYPE, G_COUNT
};

static const struct { const char *name; int type; } GK[G_COUNT] = {
    {"arch", T_STR}, {"n_layers", T_U32}, {"d_model", T_U32}, {"vocab_size", T_U32}, {"max_seq", T_U32},
    {"norm_eps", T_F32}, {"attn_kind", T_U32}, {"n_heads", T_U32}, {"n_kv_heads", T_U32}, {"head_dim", T_U32},
    {"qk_norm", T_U32}, {"qkv_bias", T_U32}, {"q_lora_rank", T_U32}, {"kv_lora_rank", T_U32},
    {"qk_nope_dim", T_U32}, {"qk_rope_dim", T_U32}, {"v_head_dim", T_U32}, {"rope_dim", T_U32},
    {"rope_style", T_U32}, {"rope_attn_factor", T_F32}, {"attn_scale", T_F32}, {"dense_ffn_dim", T_U32},
    {"n_experts", T_U32}, {"top_k", T_U32}, {"expert_ffn_dim", T_U32}, {"shared_ffn_dim", T_U32},
    {"shared_gate", T_U32}, {"score_fn", T_U32}, {"score_bias", T_U32}, {"n_group", T_U32},
    {"topk_group", T_U32}, {"norm_topk_prob", T_U32}, {"routed_scale", T_F32}, {"emb_scale", T_F32},
    {"residual_scale", T_F32}, {"logit_scale", T_F32}, {"tie_embeddings", T_U32}, {"layer_kind", T_U8A},
    {"bos_id", T_U32}, {"eos_ids", T_U32A}, {"tokenizer", T_STR}, {"chat_template", T_STR}, {"source", T_STR},
    {"expert_dtype", T_U32},
};

typedef struct {
    int has[G_COUNT];
    uint32_t u[G_COUNT];
    float f[G_COUNT];
    char s[G_COUNT][320];
    uint32_t slen[G_COUNT];
    uint32_t eos[12];
    int n_eos;
    uint8_t lk[HX_MAX_LAYERS];
    int n_lk;
    int bad_type_key;          /* >= 0: emit this key with a mismatching type (mutation) */
    int arch_then_long_key;    /* emit arch first, then a key of 130 chars (length byte 0x82) */
} gencfg;

static void set_u(gencfg *g, int k, uint32_t v) { g->has[k] = 1; g->u[k] = v; }
static void set_f(gencfg *g, int k, float v) { g->has[k] = 1; g->f[k] = v; }
static void set_s(gencfg *g, int k, const char *s) {
    g->has[k] = 1;
    g->slen[k] = (uint32_t)strlen(s);
    memcpy(g->s[k], s, g->slen[k] + 1);
}

static const char *ARCHS[] = {"qwen3_moe", "deepseek_v3", "", "olmoe",
                              "abcdefghijklmnopqrstuvwxyz0123\xC3\xA9tail" /* 2-byte char straddles byte 31 */};
static const char *TEMPLATES[] = {"{% for m in messages %}{{ m.content }}{% endfor %}",
                                  "\xE2\x82\xAC\xF0\x9F\x98\x80 caf\xC3\xA9 \xE4\xB8\xAD", ""};

/* A random but valid configuration. small: shapes for the fuzzer. */
static void gen_cfg(gencfg *g, int small) {
    memset(g, 0, sizeof *g);
    g->bad_type_key = -1;
    int L = 1 + (int)rndn(small ? 3 : 5);
    int D = 64 * (1 + (int)rndn(small ? 1 : 2));
    int V = small ? 16 + (int)rndn(40) : 32 + (int)rndn(200);
    int mla = chance(30);
    int E = chance(15) ? 0 : 2 + (int)rndn(small ? 4 : 10);
    set_u(g, G_N_LAYERS, (uint32_t)L);
    set_u(g, G_D_MODEL, (uint32_t)D);
    set_u(g, G_VOCAB, (uint32_t)V);
    if (chance(50)) set_u(g, G_MAX_SEQ, 1 + rndn(100000));
    if (chance(50)) set_f(g, G_NORM_EPS, chance(50) ? 1e-5f : 0.0f);
    if (mla || chance(30)) set_u(g, G_ATTN_KIND, (uint32_t)mla);
    int H = 1 << rndn(4);
    set_u(g, G_N_HEADS, (uint32_t)H);
    if (!mla) {
        if (chance(60)) {
            int lg = 0;
            while ((1 << lg) < H) lg++;
            set_u(g, G_N_KV_HEADS, (uint32_t)H >> rndn((uint32_t)lg + 1));   /* a power-of-two divisor of H */
        }
        if (chance(40)) set_u(g, G_HEAD_DIM, 16u << rndn(3));
        if (chance(30)) set_u(g, G_QK_NORM, rndn(3));
        if (chance(30)) set_u(g, G_QKV_BIAS, rndn(2));
    } else {
        if (chance(40)) set_u(g, G_N_KV_HEADS, 1 + rndn(7));   /* ignored for MLA, need not divide */
        if (chance(50)) set_u(g, G_Q_LORA, chance(50) ? 0 : 24);
        set_u(g, G_KV_LORA, 16u << rndn(2));
        uint32_t nope = 2 * rndn(9), rope = 2 * rndn(9);
        if (nope + rope == 0) rope = 8;
        if (nope || chance(50)) set_u(g, G_QK_NOPE, nope);
        if (rope || chance(50)) set_u(g, G_QK_ROPE, rope);
        set_u(g, G_V_HEAD, 8 + rndn(9));
    }
    int hd = g->has[G_HEAD_DIM] ? (int)g->u[G_HEAD_DIM] : D / H;
    int cap = mla ? (int)g->u[G_QK_ROPE] : hd;
    if (chance(40)) set_u(g, G_ROPE_DIM, 2 * rndn((uint32_t)cap / 2 + 1));
    if (chance(40)) set_u(g, G_ROPE_STYLE, rndn(2));
    if (chance(40)) set_f(g, G_ROPE_ATTN_FACTOR, 0.5f + (float)rndn(100) / 64.0f);
    if (chance(30)) set_f(g, G_ATTN_SCALE, (float)rndn(1000) / 997.0f);
    for (int k = G_ROUTED_SCALE; k <= G_LOGIT_SCALE; k++)
        if (chance(40)) set_f(g, k, -2.0f + (float)rndn(4000) / 999.0f);

    int lk_written = chance(50);
    int n_moe = 0;
    for (int i = 0; i < L; i++) {
        g->lk[i] = (uint8_t)(E > 0 ? (lk_written ? rndn(2) : 1) : 0);
        n_moe += g->lk[i];
    }
    if (lk_written) { g->has[G_LAYER_KIND] = 1; g->n_lk = L; }
    if (E > 0 || chance(20)) set_u(g, G_N_EXPERTS, (uint32_t)E);
    if (E > 0) {
        uint32_t ng = 1, tg = 1;
        if (chance(30)) {
            int divs[16], nd = 0;
            for (int d = 1; d <= E; d++)
                if (E % d == 0) divs[nd++] = d;
            ng = (uint32_t)divs[rndn((uint32_t)nd)];
            tg = 1 + rndn(ng);
            set_u(g, G_N_GROUP, ng);
            set_u(g, G_TOPK_GROUP, tg);
        }
        uint32_t kmax = tg * ((uint32_t)E / ng);
        if (n_moe > 0 || chance(50)) set_u(g, G_TOP_K, 1 + rndn(kmax));
        if (n_moe > 0 || chance(50)) set_u(g, G_EXPERT_FFN, 64u << rndn(small ? 1 : 2));
    }
    if (n_moe < L || chance(30)) set_u(g, G_DENSE_FFN, 64u << rndn(2));
    if (chance(30)) set_u(g, G_SHARED_FFN, 64u * rndn(3));
    if (chance(30)) set_u(g, G_SHARED_GATE, rndn(2));
    if (chance(30)) set_u(g, G_SCORE_FN, rndn(2));
    if (chance(30)) set_u(g, G_SCORE_BIAS, rndn(2));
    if (chance(30)) set_u(g, G_NORM_TOPK, rndn(2));
    if (chance(40)) set_u(g, G_TIE, rndn(2));
    switch (rndn(3)) {
    case 0: break;
    case 1: set_u(g, G_BOS, 0xFFFFFFFFu); break;
    default: set_u(g, G_BOS, rndn((uint32_t)V)); break;
    }
    if (chance(60)) {
        g->has[G_EOS] = 1;
        g->n_eos = (int)rndn(12);
        for (int i = 0; i < g->n_eos; i++) g->eos[i] = rndn((uint32_t)V);
    }
    if (chance(70)) set_s(g, G_ARCH, ARCHS[rndn(sizeof ARCHS / sizeof ARCHS[0])]);
    if (chance(50)) set_s(g, G_SOURCE, chance(50) ? "synthetic" : "org/model-with-a-rather-long-repository-name");
    if (chance(50)) set_s(g, G_TOKENIZER, chance(50) ? "model.tokenizer.json" : "tok/\xC3\xBC.json");
    if (chance(60)) set_s(g, G_CHAT_TEMPLATE, TEMPLATES[rndn(3)]);
    if (chance(40)) set_u(g, G_EXPERT_DTYPE, rndn(5));
}

/* Longest prefix of at most cap-1 bytes that does not split a UTF-8 sequence. */
static void expect_str(char *dst, size_t cap, const char *s, size_t n) {
    size_t k = 0;
    while (k < n) {
        unsigned char c = (unsigned char)s[k];
        size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
        if (k + len > cap - 1) break;
        k += len;
    }
    memcpy(dst, s, k);
    dst[k] = 0;
}

/* FORMAT.md §3.1 defaults, written out independently of modelfile.c. */
static void expect_cfg(const gencfg *g, hx_config *c) {
#define GU(k, d) (g->has[k] ? g->u[k] : (uint32_t)(d))
#define GF(k, d) (g->has[k] ? g->f[k] : (d))
    memset(c, 0, sizeof *c);
    if (g->has[G_ARCH]) expect_str(c->arch, sizeof c->arch, g->s[G_ARCH], g->slen[G_ARCH]);
    c->n_layers = (int)g->u[G_N_LAYERS];
    c->d_model = (int)g->u[G_D_MODEL];
    c->vocab_size = (int)g->u[G_VOCAB];
    c->max_seq = (int)GU(G_MAX_SEQ, 4096);
    c->norm_eps = GF(G_NORM_EPS, 1e-6f);
    c->attn_kind = (int)GU(G_ATTN_KIND, 0);
    c->n_heads = (int)g->u[G_N_HEADS];
    c->n_kv_heads = (int)GU(G_N_KV_HEADS, c->n_heads);
    c->head_dim = (int)GU(G_HEAD_DIM, c->d_model / c->n_heads);
    c->qk_norm = (int)GU(G_QK_NORM, 0);
    c->qkv_bias = (int)GU(G_QKV_BIAS, 0);
    c->q_lora_rank = (int)GU(G_Q_LORA, 0);
    c->kv_lora_rank = (int)GU(G_KV_LORA, 0);
    c->qk_nope_dim = (int)GU(G_QK_NOPE, 0);
    c->qk_rope_dim = (int)GU(G_QK_ROPE, 0);
    c->v_head_dim = (int)GU(G_V_HEAD, 0);
    int mla = c->attn_kind == 1;
    c->rope_dim = (int)GU(G_ROPE_DIM, mla ? c->qk_rope_dim : c->head_dim);
    c->rope_style = (int)GU(G_ROPE_STYLE, 0);
    c->rope_attn_factor = GF(G_ROPE_ATTN_FACTOR, 1.0f);
    int qk = mla ? c->qk_nope_dim + c->qk_rope_dim : c->head_dim;
    c->attn_scale = GF(G_ATTN_SCALE, (float)(1.0f / sqrtf((float)qk)));
    c->dense_ffn_dim = (int)GU(G_DENSE_FFN, 0);
    c->n_experts = (int)GU(G_N_EXPERTS, 0);
    c->top_k = (int)GU(G_TOP_K, 0);
    c->expert_ffn_dim = (int)GU(G_EXPERT_FFN, 0);
    c->shared_ffn_dim = (int)GU(G_SHARED_FFN, 0);
    c->shared_gate = (int)GU(G_SHARED_GATE, 0);
    c->score_fn = (int)GU(G_SCORE_FN, 0);
    c->score_bias = (int)GU(G_SCORE_BIAS, 0);
    c->n_group = (int)GU(G_N_GROUP, 1);
    c->topk_group = (int)GU(G_TOPK_GROUP, 1);
    c->norm_topk_prob = (int)GU(G_NORM_TOPK, 0);
    c->routed_scale = GF(G_ROUTED_SCALE, 1.0f);
    c->emb_scale = GF(G_EMB_SCALE, 1.0f);
    c->residual_scale = GF(G_RESID_SCALE, 1.0f);
    c->logit_scale = GF(G_LOGIT_SCALE, 1.0f);
    c->tie_embeddings = (int)GU(G_TIE, 0);
    for (int i = 0; i < c->n_layers; i++) {
        c->layer_kind[i] = g->has[G_LAYER_KIND] ? g->lk[i] : (uint8_t)(c->n_experts > 0);
        c->n_moe_layers += c->layer_kind[i];
    }
    uint32_t bos = GU(G_BOS, 0xFFFFFFFFu);
    c->bos_id = bos == 0xFFFFFFFFu ? -1 : (int)bos;
    for (int i = 0; i < g->n_eos && i < HX_MAX_EOS; i++) c->eos_ids[c->n_eos++] = (int)g->eos[i];
    if (g->has[G_TOKENIZER]) memcpy(c->tokenizer, g->s[G_TOKENIZER], g->slen[G_TOKENIZER] + 1);
    if (g->has[G_SOURCE]) expect_str(c->source, sizeof c->source, g->s[G_SOURCE], g->slen[G_SOURCE]);
#undef GU
#undef GF
}

/* ---------------------------------------------------------- the container */

static uint64_t t_rowbytes(int dt, uint64_t n) {
    switch (dt) {
    case 0: case 5: return 4 * n;
    case 1: case 2: return 2 * n;
    case 3: return n % 64 ? 0 : 66 * n / 64;
    case 4: return n % 64 ? 0 : 34 * n / 64;
    case 6: return n;
    default: return 0;
    }
}

static uint64_t t_layout(int dt, uint64_t D, uint64_t F, uint64_t *og, uint64_t *ou, uint64_t *od) {
    uint64_t rd = t_rowbytes(dt, D), rf = t_rowbytes(dt, F);
    *og = 0;
    *ou = al(F * rd, 64);
    *od = al(*ou + F * rd, 64);
    return al(*od + D * rf, 4096);
}

typedef struct {
    char name[HX_NAME_LEN];
    int dtype, ndim;
    uint32_t shape[4];
    uint64_t nbytes, off;
    uint8_t *data;
} ttensor;

typedef struct {
    uint64_t off, nbytes, need, used;   /* used: bytes before zero padding */
    uint32_t dtype, flags;
    int alias_of;                        /* entry index, or -1 */
    uint8_t *data;                       /* nbytes bytes (non-alias only) */
} texpert;

typedef struct {
    gencfg g;
    hx_config want;
    ttensor *t;
    int nt;
    texpert *e;
    int ne;
    int L, E, D, F;
    /* layout options */
    int shuffle_sections, tensors_after_slabs, big_gap;
    /* assembled file */
    bbuf meta;
    uint64_t meta_off, tdir_off, edir_off;
    uint8_t *file;
    size_t size;
} model;

static void free_model(model *m) {
    for (int i = 0; i < m->nt; i++) free(m->t[i].data);
    for (int i = 0; i < m->ne; i++) free(m->e[i].data);
    free(m->t);
    free(m->e);
    free(m->meta.p);
    free(m->file);
    memset(m, 0, sizeof *m);
}

static void emit_value(bbuf *b, const gencfg *g, int k, int decoy) {
    switch (GK[k].type) {
    case T_U32: bb_u32(b, decoy ? g->u[k] + 7 : g->u[k]); break;
    case T_F32: { float f = decoy ? g->f[k] + 1.0f : g->f[k]; bb_u32(b, fbits(f)); break; }
    case T_STR:
        if (decoy) { bb_u32(b, 5); bb_put(b, "decoy", 5); }
        else { bb_u32(b, g->slen[k]); bb_put(b, g->s[k], g->slen[k]); }
        break;
    case T_U32A:
        bb_u32(b, (uint32_t)g->n_eos);
        for (int i = 0; i < g->n_eos; i++) bb_u32(b, decoy ? 0xFFFFFFF0u : g->eos[i]);
        break;
    case T_U8A:
        bb_u32(b, (uint32_t)g->n_lk);
        for (int i = 0; i < g->n_lk; i++) bb_u8(b, decoy ? 7 : g->lk[i]);
        break;
    }
}

static void emit_unknown(bbuf *b) {
    char key[32];
    int n = snprintf(key, sizeof key, chance(10) ? "%.0d" : "zz_unknown_%d", (int)rndn(1000));
    int t = 1 + (int)rndn(7);
    bb_u16(b, (uint32_t)n);
    bb_put(b, key, (size_t)n);
    bb_u8(b, (uint32_t)t);
    uint32_t cnt = rndn(6);
    switch (t) {
    case T_U32: case T_F32: bb_u32(b, (uint32_t)rnd()); break;
    case T_U64: bb_u64(b, rnd()); break;
    case T_STR: bb_u32(b, 3); bb_put(b, "\xC3\xA9x", 3); break;
    case T_U32A: case T_F32A: bb_u32(b, cnt); for (uint32_t i = 0; i < cnt; i++) bb_u32(b, (uint32_t)rnd()); break;
    case T_U8A: bb_u32(b, cnt); for (uint32_t i = 0; i < cnt; i++) bb_u8(b, (uint32_t)rnd()); break;
    }
}

static void emit_meta(model *m) {
    gencfg *g = &m->g;
    int order[G_COUNT], n = 0;
    m->meta.n = 0;
    for (int k = 0; k < G_COUNT; k++)
        if (g->has[k]) order[n++] = k;
    for (int i = n - 1; i > 0; i--) {
        int j = (int)rndn((uint32_t)i + 1), t = order[i];
        order[i] = order[j];
        order[j] = t;
    }
    if (g->arch_then_long_key && g->has[G_ARCH]) {
        for (int i = 0; i < n; i++)
            if (order[i] == G_ARCH) { order[i] = order[0]; order[0] = G_ARCH; }
    }
    for (int i = 0; i < n; i++) {
        int k = order[i];
        if (i == 1 && g->arch_then_long_key) {
            char longkey[130];
            memset(longkey, 'k', sizeof longkey);
            bb_u16(&m->meta, sizeof longkey);
            bb_put(&m->meta, longkey, sizeof longkey);
            bb_u8(&m->meta, T_U32);
            bb_u32(&m->meta, 1);
        }
        if (chance(10) && !g->arch_then_long_key) emit_unknown(&m->meta);
        for (int decoy = g->arch_then_long_key ? 0 : chance(5); decoy >= 0; decoy--) {
            size_t kl = strlen(GK[k].name);
            bb_u16(&m->meta, (uint32_t)kl);
            bb_put(&m->meta, GK[k].name, kl);
            if (k == g->bad_type_key && decoy == 0) {
                /* same payload size, wrong type id */
                bb_u8(&m->meta, GK[k].type == T_U32 ? T_F32 : T_U32);
                bb_u32(&m->meta, GK[k].type == T_U32 ? g->u[k] : fbits(g->f[k]));
                continue;
            }
            bb_u8(&m->meta, (uint32_t)GK[k].type);
            emit_value(&m->meta, g, k, decoy);
        }
    }
    if (chance(30)) emit_unknown(&m->meta);
}

static void add_tensor(model *m, const char *name, int dtype, int ndim, const uint32_t *shape) {
    ttensor *t = &m->t[m->nt++];
    memset(t, 0, sizeof *t);
    snprintf(t->name, sizeof t->name, "%s", name);
    t->dtype = dtype;
    t->ndim = ndim;
    uint64_t rows = 1;
    for (int k = 0; k < 4; k++) {
        t->shape[k] = k < ndim ? shape[k] : 1;
        if (k < ndim - 1) rows *= shape[k];
    }
    t->nbytes = rows * t_rowbytes(dtype, shape[ndim - 1]);
    t->data = (uint8_t *)malloc((size_t)t->nbytes);
    for (uint64_t i = 0; i < t->nbytes; i++) t->data[i] = (uint8_t)rnd();
}

static int rnd_wdtype(int small) {
    static const int wsmall[] = {4, 4, 4, 3, 1, 2, 0};
    return small ? wsmall[rndn(7)] : (int)rndn(5);
}

static void gen_tensors(model *m, int small) {
    const hx_config *c = &m->want;
    char name[HX_NAME_LEN];
    uint32_t sh[4];
    m->t = (ttensor *)calloc(16 + 4 * (size_t)c->n_layers, sizeof *m->t);
    if (c->rope_dim > 0) { sh[0] = (uint32_t)c->rope_dim / 2; add_tensor(m, "rope_inv_freq", 0, 1, sh); }
    sh[0] = (uint32_t)c->vocab_size; sh[1] = (uint32_t)c->d_model;
    add_tensor(m, "tok_embd", rnd_wdtype(small), 2, sh);
    if (!c->tie_embeddings) add_tensor(m, "lm_head", rnd_wdtype(small), 2, sh);
    sh[0] = (uint32_t)c->d_model;
    add_tensor(m, "out_norm", 0, 1, sh);
    for (int i = 0; i < c->n_layers; i++) {
        snprintf(name, sizeof name, "blk.%d.attn_norm", i);
        sh[0] = (uint32_t)c->d_model;
        add_tensor(m, name, 0, 1, sh);
        if (c->layer_kind[i]) {
            snprintf(name, sizeof name, "blk.%d.moe_router", i);
            sh[0] = (uint32_t)c->n_experts; sh[1] = (uint32_t)c->d_model;
            add_tensor(m, name, 0, 2, sh);
        } else if (!small || chance(50)) {
            snprintf(name, sizeof name, "blk.%d.ffn_down", i);
            sh[0] = (uint32_t)c->d_model; sh[1] = (uint32_t)c->dense_ffn_dim;
            add_tensor(m, name, rnd_wdtype(small), 2, sh);
        }
    }
    int extra = (int)rndn(small ? 3 : 6);
    for (int x = 0; x < extra; x++) {
        int dt = (int)rndn(7), nd = 1 + (int)rndn(4);
        for (int k = 0; k < nd; k++) sh[k] = 1 + rndn(4);
        sh[nd - 1] = (dt == 3 || dt == 4) ? 64u << rndn(2) : 1 + rndn(9);
        snprintf(name, sizeof name, chance(20) ? "x%d.%s" : "zz.extra.%d.%s", x,
                 chance(50) ? "w" : "a_long_tensor_name_padded_out_to_exactly_seventy_nine_characters_xxxxxx");
        name[HX_NAME_LEN - 1] = 0;
        add_tensor(m, name, dt, nd, sh);
    }
}

static void gen_experts(model *m, int small) {
    const hx_config *c = &m->want;
    m->L = c->n_layers; m->E = c->n_experts; m->D = c->d_model; m->F = c->expert_ffn_dim;
    int has_moe = c->n_experts > 0 && c->n_moe_layers > 0;
    if (!has_moe && (c->n_experts == 0 || chance(50))) { m->ne = 0; return; }
    m->ne = m->L * m->E;
    m->e = (texpert *)calloc((size_t)m->ne, sizeof *m->e);
    for (int i = 0; i < m->ne; i++) {
        texpert *e = &m->e[i];
        e->alias_of = -1;
        if (!c->layer_kind[i / m->E]) {
            if (chance(20)) { e->dtype = rndn(9); e->flags = rndn(4); }   /* ignored for empty entries */
            continue;
        }
        int tgt = -1;
        if (i > 0 && chance(20)) {
            int j = (int)rndn((uint32_t)i);
            if (m->e[j].nbytes && m->e[j].alias_of < 0) tgt = j;
        }
        if (tgt >= 0) {
            e->alias_of = tgt;
            e->flags = 1 | (chance(20) ? 2u : 0u);
            e->dtype = m->e[tgt].dtype;
            e->nbytes = m->e[tgt].nbytes;
            e->need = m->e[tgt].need;
            continue;
        }
        uint64_t og, ou, od;
        e->dtype = (uint32_t)rnd_wdtype(small);
        e->need = t_layout((int)e->dtype, (uint64_t)m->D, (uint64_t)m->F, &og, &ou, &od);
        e->used = od + (uint64_t)m->D * t_rowbytes((int)e->dtype, (uint64_t)m->F);
        e->nbytes = e->need + (chance(small ? 20 : 5) ? 4096 : 0);   /* slack: legal, readers accept it */
        e->flags = chance(10) ? 2u : 0u;   /* unknown flag bits are ignored */
        e->data = (uint8_t *)calloc(1, (size_t)e->nbytes);
        for (uint64_t b = 0; b < e->used; b++) e->data[b] = (uint8_t)rnd();
    }
}

static void place_tensors(model *m, uint64_t *pos) {
    int *perm = (int *)malloc(sizeof *perm * (size_t)m->nt);
    for (int i = 0; i < m->nt; i++) perm[i] = i;
    for (int i = m->nt - 1; i > 0; i--) {
        int j = (int)rndn((uint32_t)i + 1), t = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
    }
    *pos = al(*pos, 64);
    for (int i = 0; i < m->nt; i++) {
        ttensor *t = &m->t[perm[i]];
        t->off = *pos;
        *pos = al(*pos + t->nbytes, 64);
        if (i + 1 < m->nt) {
            if (chance(10)) *pos += 64 * rndn(5);
            if (m->big_gap && i == m->nt / 2) *pos += (uint64_t)3 << 20;
        }
    }
    free(perm);
}

static void place_slabs(model *m, uint64_t *pos) {
    *pos = al(*pos, 4096);
    int last = -1;
    for (int i = 0; i < m->ne; i++)
        if (m->e[i].nbytes && m->e[i].alias_of < 0) last = i;
    for (int i = 0; i < m->ne; i++) {
        texpert *e = &m->e[i];
        if (!e->nbytes || e->alias_of >= 0) continue;
        e->off = *pos;
        *pos += e->nbytes;
        if (i != last && chance(10)) *pos += 4096;
    }
    for (int i = 0; i < m->ne; i++)
        if (m->e[i].alias_of >= 0) m->e[i].off = m->e[m->e[i].alias_of].off;
}

static uint64_t region_end(const model *m) {
    uint64_t end = 64;
    uint64_t ends[3] = {m->meta_off + m->meta.n, m->tdir_off + 128 * (uint64_t)m->nt, m->edir_off + 32 * (uint64_t)m->ne};
    for (int i = 0; i < 3; i++) if (ends[i] > end) end = ends[i];
    for (int i = 0; i < m->nt; i++) if (m->t[i].off + m->t[i].nbytes > end) end = m->t[i].off + m->t[i].nbytes;
    for (int i = 0; i < m->ne; i++) if (m->e[i].off + m->e[i].nbytes > end) end = m->e[i].off + m->e[i].nbytes;
    return end;
}

/* Lays out sections, tensors and slabs and serializes the file (preamble last). */
static void assemble(model *m) {
    uint64_t pos = 64;
    int order[3] = {0, 1, 2};
    if (m->shuffle_sections)
        for (int i = 2; i > 0; i--) {
            int j = (int)rndn((uint32_t)i + 1), t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
    for (int s = 0; s < 3; s++) {
        if (order[s] == 0) { m->meta_off = pos; pos += m->meta.n; }
        else if (order[s] == 1) { m->tdir_off = pos; pos += 128 * (uint64_t)m->nt; }
        else { m->edir_off = pos; pos += 32 * (uint64_t)m->ne; }
        if (m->shuffle_sections && chance(30)) pos += rndn(100);
    }
    if (m->tensors_after_slabs) { place_slabs(m, &pos); place_tensors(m, &pos); }
    else { place_tensors(m, &pos); place_slabs(m, &pos); }
    m->size = (size_t)region_end(m);
    free(m->file);
    m->file = (uint8_t *)calloc(1, m->size);
    uint8_t *f = m->file;
    memcpy(f + m->meta_off, m->meta.p, m->meta.n);
    for (int i = 0; i < m->nt; i++) {
        uint8_t *d = f + m->tdir_off + 128 * (uint64_t)i;
        const ttensor *t = &m->t[i];
        memcpy(d, t->name, strlen(t->name));
        w32(d + 80, (uint32_t)t->dtype);
        w32(d + 84, (uint32_t)t->ndim);
        for (int k = 0; k < 4; k++) w32(d + 88 + 4 * k, t->shape[k]);
        w64(d + 104, t->off);
        w64(d + 112, t->nbytes);
        memcpy(f + t->off, t->data, (size_t)t->nbytes);
    }
    for (int i = 0; i < m->ne; i++) {
        uint8_t *d = f + m->edir_off + 32 * (uint64_t)i;
        const texpert *e = &m->e[i];
        w64(d, e->off);
        w64(d + 8, e->nbytes);
        w32(d + 16, e->dtype);
        w32(d + 20, e->flags);
        if (e->data) memcpy(f + e->off, e->data, (size_t)e->nbytes);
    }
    w32(f, HX_MAGIC);
    w32(f + 4, 1);
    w64(f + 8, m->meta_off);
    w64(f + 16, m->meta.n);
    w64(f + 24, m->tdir_off);
    w64(f + 32, (uint64_t)m->nt);
    w64(f + 40, m->edir_off);
    w64(f + 48, (uint64_t)m->ne);
    w32(f + 56, 4096);
    w32(f + 60, 0);
}

static void make_model(model *m, int small) {
    memset(m, 0, sizeof *m);
    gen_cfg(&m->g, small);
    expect_cfg(&m->g, &m->want);
    gen_tensors(m, small);
    gen_experts(m, small);
    m->shuffle_sections = chance(40);
    m->tensors_after_slabs = chance(25);
    m->big_gap = !small && chance(15);
    emit_meta(m);
    assemble(m);
}

/* ------------------------------------------------------------- file I/O */

static char g_dir[600];
static char g_path[700];

static int write_file(const char *path, const uint8_t *d, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    int ok = fwrite(d, 1, n, f) == n;
    return fclose(f) == 0 && ok;
}

/* Small scratch files: an explicit or temp directory, never the current one (it may be the repository). */
static void pick_dir(const char *arg) {
    const char *c[] = {arg, hx_env_str("HEARTH_TEST_DIR"), hx_env_str("HEARTH_MUTATION_TMP"), hx_env_str("TEMP"),
                       hx_env_str("TMP"), hx_env_str("TMPDIR"),
#if defined(HX_OS_POSIX)
                       "/tmp",
#endif
                       NULL};
    uint64_t tag = hx_now_ns() ^ (uint64_t)(uintptr_t)&tag;   /* concurrent runs must not share the file */
    tag = (tag ^ (tag >> 29)) * 0xbf58476d1ce4e5b9ull;
    for (size_t i = 0; i < sizeof c / sizeof c[0]; i++) {
        if (!c[i] || !*c[i]) continue;
        snprintf(g_path, sizeof g_path, "%s/hx_test_modelfile_%08llx.tmp", c[i], (unsigned long long)(tag >> 32));
        uint8_t x = 0;
        if (write_file(g_path, &x, 1)) {
            snprintf(g_dir, sizeof g_dir, "%s", c[i]);
            return;
        }
    }
    printf("no writable scratch directory (set HEARTH_TEST_DIR or pass --dir)\n");
    exit(2);
}

/* ---------------------------------------------------------- verification */

#define EQI(field) CHECK(got->field == w->field, "%s: cfg." #field " = %d, want %d", tag, (int)got->field, (int)w->field)
#define EQF(field) CHECK(fbits(got->field) == fbits(w->field), "%s: cfg." #field " = %.9g, want %.9g", tag, (double)got->field, (double)w->field)

static void check_cfg(const char *tag, const hx_config *got, const hx_config *w) {
    CHECK(strcmp(got->arch, w->arch) == 0, "%s: arch '%s' want '%s'", tag, got->arch, w->arch);
    EQI(n_layers); EQI(d_model); EQI(vocab_size); EQI(max_seq); EQF(norm_eps);
    EQI(attn_kind); EQI(n_heads); EQI(n_kv_heads); EQI(head_dim); EQI(qk_norm); EQI(qkv_bias);
    EQI(q_lora_rank); EQI(kv_lora_rank); EQI(qk_nope_dim); EQI(qk_rope_dim); EQI(v_head_dim);
    EQI(rope_dim); EQI(rope_style); EQF(rope_attn_factor); EQF(attn_scale);
    EQI(dense_ffn_dim); EQI(n_experts); EQI(top_k); EQI(expert_ffn_dim); EQI(shared_ffn_dim); EQI(shared_gate);
    EQI(score_fn); EQI(score_bias); EQI(n_group); EQI(topk_group); EQI(norm_topk_prob);
    EQF(routed_scale); EQF(emb_scale); EQF(residual_scale); EQF(logit_scale); EQI(tie_embeddings);
    EQI(n_moe_layers); EQI(bos_id); EQI(n_eos);
    for (int i = 0; i < w->n_layers; i++) CHECK(got->layer_kind[i] == w->layer_kind[i], "%s: layer_kind[%d]", tag, i);
    for (int i = 0; i < w->n_eos; i++) CHECK(got->eos_ids[i] == w->eos_ids[i], "%s: eos_ids[%d]", tag, i);
    CHECK(strcmp(got->tokenizer, w->tokenizer) == 0, "%s: tokenizer '%s' want '%s'", tag, got->tokenizer, w->tokenizer);
    CHECK(strcmp(got->source, w->source) == 0, "%s: source '%s' want '%s'", tag, got->source, w->source);
}

static void verify_model(const model *m, const char *tag, int load_dense) {
    char err[600];
    hx_modelfile *mf = hx_modelfile_open(g_path, load_dense, err, sizeof err);
    CHECK(mf != NULL, "%s: open failed: %s", tag, err);
    if (!mf) return;
    const hx_config *c = &mf->cfg;
    check_cfg(tag, c, &m->want);
    CHECK(mf->file_size == m->size, "%s: file_size", tag);
    CHECK(strcmp(mf->path, g_path) == 0, "%s: path", tag);
    const char *want_ct = m->g.has[G_CHAT_TEMPLATE] && m->g.slen[G_CHAT_TEMPLATE] ? m->g.s[G_CHAT_TEMPLATE] : NULL;
    CHECK(want_ct ? (mf->chat_template && strcmp(mf->chat_template, want_ct) == 0) : mf->chat_template == NULL,
          "%s: chat_template", tag);

    /* tensors */
    CHECK(mf->n_tensors == m->nt, "%s: n_tensors %d want %d", tag, mf->n_tensors, m->nt);
    uint64_t dense_bytes = 0;
    double dense_params = 0, emb = 0;
    for (int i = 0; i < m->nt; i++) {
        const ttensor *t = &m->t[i];
        const hx_tensor *h = hx_mf_tensor(mf, t->name);
        CHECK(h != NULL, "%s: tensor %s not found", tag, t->name);
        dense_bytes += t->nbytes;
        double el = 1;
        for (int k = 0; k < t->ndim; k++) el *= t->shape[k];
        if (strcmp(t->name, "rope_inv_freq")) dense_params += el;
        if (!strcmp(t->name, "tok_embd")) emb = el;
        if (!h) continue;
        CHECK(h->dtype == t->dtype && h->ndim == t->ndim, "%s: %s dtype/ndim", tag, t->name);
        for (int k = 0; k < 4; k++) CHECK(h->shape[k] == (int64_t)t->shape[k], "%s: %s shape[%d]", tag, t->name, k);
        CHECK(h->offset == t->off && h->nbytes == t->nbytes, "%s: %s offset/nbytes", tag, t->name);
        if (load_dense) {
            CHECK(h->data && ((uintptr_t)h->data % 64) == 0, "%s: %s data pointer not 64-aligned", tag, t->name);
            if (h->data) CHECK(memcmp(h->data, t->data, (size_t)t->nbytes) == 0, "%s: %s bytes differ", tag, t->name);
        } else {
            CHECK(h->data == NULL, "%s: %s data should be NULL without load_dense", tag, t->name);
        }
        if (!strncmp(t->name, "blk.", 4)) {
            int layer = atoi(t->name + 4);
            const char *suf = strchr(t->name + 4, '.') + 1;
            CHECK(hx_mf_layer_tensor(mf, layer, suf) == h, "%s: hx_mf_layer_tensor(%d, %s)", tag, layer, suf);
        }
    }
    for (int i = 1; i < mf->n_tensors; i++)
        CHECK(strcmp(mf->tensors[i - 1].name, mf->tensors[i].name) < 0, "%s: tensors not sorted", tag);
    CHECK(hx_mf_tensor(mf, "no.such.tensor") == NULL, "%s: lookup of a missing name", tag);
    CHECK(hx_mf_tensor(mf, "") == NULL && hx_mf_layer_tensor(mf, -1, "attn_norm") == NULL &&
          hx_mf_tensor(mf, NULL) == NULL && hx_mf_layer_tensor(mf, 0, NULL) == NULL, "%s: bad lookups", tag);
    CHECK(mf->dense_bytes == dense_bytes, "%s: dense_bytes %llu want %llu", tag, (unsigned long long)mf->dense_bytes,
          (unsigned long long)dense_bytes);
    const hx_tensor *rf = hx_mf_tensor(mf, "rope_inv_freq");
    CHECK(mf->rope_inv_freq == (load_dense && rf ? rf->data : NULL), "%s: rope_inv_freq pointer", tag);

    /* experts */
    int has_moe = c->n_experts > 0 && c->n_moe_layers > 0;
    CHECK(has_moe ? mf->experts != NULL : mf->experts == NULL, "%s: experts array presence", tag);
    uint64_t region = 0, smax = 0;
    for (int i = 0; i < m->ne && has_moe; i++) {
        const texpert *e = &m->e[i];
        const hx_expert_entry *h = hx_mf_expert(mf, i / m->E, i % m->E);
        CHECK(h == &mf->experts[i], "%s: hx_mf_expert(%d)", tag, i);
        if (!h) continue;
        CHECK(h->offset == e->off && h->nbytes == e->nbytes && h->dtype == e->dtype && h->flags == e->flags,
              "%s: expert entry %d", tag, i);
        if (e->nbytes && e->alias_of < 0) {
            region += e->nbytes;
            if (e->nbytes > smax) smax = e->nbytes;
            uint64_t og, ou, od;
            uint64_t need = t_layout((int)e->dtype, (uint64_t)m->D, (uint64_t)m->F, &og, &ou, &od);
            uint64_t g2, u2, d2;
            CHECK(hx_slab_layout((int)e->dtype, m->D, m->F, &g2, &u2, &d2) == need && g2 == og && u2 == ou && d2 == od,
                  "%s: hx_slab_layout dtype %u", tag, e->dtype);
            hx_slab_view v;
            const uint8_t *slab = m->file + e->off;
            hx_slab_view_make(mf, slab, (int)e->dtype, &v);
            CHECK(v.gate == slab + og && v.up == slab + ou && v.down == slab + od && v.dtype == (int)e->dtype,
                  "%s: hx_slab_view_make", tag);
        }
    }
    if (has_moe) {
        CHECK(hx_mf_expert(mf, -1, 0) == NULL && hx_mf_expert(mf, 0, m->E) == NULL && hx_mf_expert(mf, m->L, 0) == NULL,
              "%s: out-of-range expert lookups", tag);
        CHECK(mf->expert_region_bytes == region, "%s: expert_region_bytes %llu want %llu", tag,
              (unsigned long long)mf->expert_region_bytes, (unsigned long long)region);
        CHECK(mf->slab_bytes_max == smax, "%s: slab_bytes_max", tag);
    } else {
        CHECK(mf->expert_region_bytes == 0 && mf->slab_bytes_max == 0 && hx_mf_expert(mf, 0, 0) == NULL,
              "%s: dense model expert fields", tag);
    }
    double pe = 3.0 * c->d_model * c->expert_ffn_dim;
    double tot = dense_params + (has_moe ? c->n_moe_layers * (double)c->n_experts * pe : 0);
    double act = dense_params - (c->tie_embeddings ? 0 : emb) + (has_moe ? c->n_moe_layers * (double)c->top_k * pe : 0);
    CHECK(mf->params_total == tot && mf->params_active == act, "%s: params %g/%g want %g/%g", tag, mf->params_total,
          mf->params_active, tot, act);
    hx_modelfile_close(mf);
}

/* ------------------------------------------------------------------ fuzz */

static int g_opens, g_accepted_corrupt;

/* Touch everything an accepted container exposes (ASan sees out-of-bounds pointers). */
static void walk(const hx_modelfile *mf) {
    volatile uint8_t sink = 0;
    const hx_config *c = &mf->cfg;
    CHECK(c->n_layers >= 1 && c->n_layers <= HX_MAX_LAYERS && c->n_eos <= HX_MAX_EOS, "accepted: config out of range");
    for (int i = 0; i < mf->n_tensors; i++) {
        const hx_tensor *t = &mf->tensors[i];
        CHECK(t->offset + t->nbytes <= mf->file_size && t->offset + t->nbytes >= t->offset, "accepted: tensor past EOF");
        CHECK(memchr(t->name, 0, HX_NAME_LEN) != NULL, "accepted: name not terminated");
        if (t->data) {
            const uint8_t *p = (const uint8_t *)t->data;
            for (uint64_t k = 0; k < t->nbytes; k++) sink ^= p[k];
        }
    }
    if (mf->experts)
        for (int i = 0; i < c->n_layers * c->n_experts; i++) {
            const hx_expert_entry *e = &mf->experts[i];
            CHECK(e->offset + e->nbytes <= mf->file_size && e->offset % 4096 == 0, "accepted: expert entry %d invalid", i);
            CHECK(e->nbytes <= mf->slab_bytes_max, "accepted: slab larger than slab_bytes_max");
        }
    if (mf->chat_template) sink ^= (uint8_t)strlen(mf->chat_template);
    if (mf->rope_inv_freq) {   /* the bytes: a fuzzed value need not fit an integer type */
        uint32_t u;
        memcpy(&u, &mf->rope_inv_freq[c->rope_dim / 2 - 1], sizeof u);
        sink ^= (uint8_t)u;
    }
    (void)sink;
}

/* mode 0: must fail; 1: must succeed; 2: anything but a crash. */
static void try_variant(const char *what, const uint8_t *d, size_t n, int mode) {
    char err[600];
    if (!write_file(g_path, d, n)) { CHECK(0, "%s: cannot write scratch file", what); return; }
    int load = (int)(rnd() & 1) || mode == 1;
    memset(err, 'X', sizeof err);
    g_opens++;
    hx_modelfile *mf = hx_modelfile_open(g_path, load, err, sizeof err);
    if (mf) {
        CHECK(mode != 0, "%s: invalid container was accepted", what);
        if (mode == 2) g_accepted_corrupt++;
        walk(mf);
        hx_modelfile_close(mf);
    } else {
        CHECK(mode != 1, "%s: valid container rejected: %s", what, err);
        CHECK(memchr(err, 0, sizeof err) != NULL && err[0] != 0 && err[0] != 'X', "%s: no error message", what);
    }
}

static uint8_t *dup_file(const model *m) {
    uint8_t *d = (uint8_t *)malloc(m->size);
    memcpy(d, m->file, m->size);
    return d;
}

static void fuzz_truncations(const model *m, int n_random) {
    uint64_t pts[4096];
    int np = 0;
#define PT(x) do { uint64_t v_ = (x); for (int d_ = -1; d_ <= 1 && np < 4093; d_++) pts[np++] = v_ + (uint64_t)d_; } while (0)
    PT(0); PT(64); PT(m->meta_off); PT(m->meta_off + m->meta.n); PT(m->tdir_off); PT(m->edir_off);
    for (int i = 0; i <= m->nt; i++) PT(m->tdir_off + 128 * (uint64_t)i);
    for (int i = 0; i <= m->ne; i++) PT(m->edir_off + 32 * (uint64_t)i);
    for (int i = 0; i < m->nt; i++) { PT(m->t[i].off); PT(m->t[i].off + m->t[i].nbytes); }
    for (int i = 0; i < m->ne; i++) if (m->e[i].nbytes) { PT(m->e[i].off); PT(m->e[i].off + m->e[i].nbytes); }
#undef PT
    for (int i = 0; i < n_random && np < 4096; i++) pts[np++] = rnd() % m->size;
    for (int i = 0; i < np; i++) {
        if (pts[i] >= m->size) continue;   /* also drops the -1 wraparound */
        char what[64];
        snprintf(what, sizeof what, "truncate@%llu/%llu", (unsigned long long)pts[i], (unsigned long long)m->size);
        try_variant(what, m->file, (size_t)pts[i], 0);
    }
}

static void patch64(const model *m, const char *what, size_t at, uint64_t v, int mode) {
    uint8_t *d = dup_file(m);
    w64(d + at, v);
    try_variant(what, d, m->size, mode);
    free(d);
}
static void patch32(const model *m, const char *what, size_t at, uint32_t v, int mode) {
    uint8_t *d = dup_file(m);
    w32(d + at, v);
    try_variant(what, d, m->size, mode);
    free(d);
}

static void fuzz_preamble(const model *m) {
    uint64_t sz = m->size, ne = (uint64_t)m->ne;
    patch32(m, "magic", 0, 0x12345678u, 0);
    patch32(m, "version 2", 4, 2, 0);
    patch32(m, "version 0", 4, 0, 0);
    patch32(m, "align 512", 56, 512, 0);
    patch32(m, "reserved ignored", 60, 0xDEADBEEFu, 1);
    patch64(m, "meta_off=size", 8, sz, 0);
    patch64(m, "meta_off=max", 8, UINT64_MAX, 0);
    patch64(m, "meta_bytes=size", 16, sz, 0);
    patch64(m, "meta_bytes wraps", 16, UINT64_MAX - m->meta_off + 1, 0);
    patch64(m, "meta_bytes-1", 16, m->meta.n - 1, 0);
    patch64(m, "n_tensors=size/128+1", 32, sz / 128 + 1, 0);
    patch64(m, "n_tensors=2^40", 32, (uint64_t)1 << 40, 0);
    patch64(m, "n_tensors=max", 32, UINT64_MAX, 0);
    patch64(m, "n_tensors*128 wraps", 32, UINT64_MAX / 128 + 2, 0);
    patch64(m, "n_tensors+1", 32, (uint64_t)m->nt + 1, 0);
    patch64(m, "tdir_off=size", 24, sz, 0);
    patch64(m, "tdir_off=max", 24, UINT64_MAX - 100, 0);
    patch64(m, "n_e+1", 48, ne + 1, 0);
    patch64(m, "n_e=max", 48, UINT64_MAX, 0);
    patch64(m, "n_e*32 wraps", 48, UINT64_MAX / 32 + 2, 0);
    patch64(m, "n_e=2^40", 48, (uint64_t)1 << 40, 0);
    if (ne) {
        patch64(m, "n_e-1", 48, ne - 1, 0);
        patch64(m, "edir_off=max", 40, UINT64_MAX - 7, 0);
        patch64(m, "edir_off=size", 40, sz, 0);
    }
}

static void fuzz_tensor_entries(const model *m) {
    char w[128];
    for (int rep = 0; rep < 3; rep++) {
        int i = (int)rndn((uint32_t)m->nt);
        const ttensor *t = &m->t[i];
        size_t e = (size_t)(m->tdir_off + 128 * (uint64_t)i);
#define TV(fmt, ...) (snprintf(w, sizeof w, "tensor %s: " fmt, t->name, __VA_ARGS__), w)
        patch64(m, TV("offset+%d", 1 + (int)rndn(63)), e + 104, t->off + 1 + rndn(63), 0);
        patch64(m, TV("offset=%s", "size"), e + 104, m->size, 0);
        patch64(m, TV("offset=%s", "max"), e + 104, UINT64_MAX & ~(uint64_t)63, 0);
        patch64(m, TV("offset=%s", "0 (preamble)"), e + 104, 0, 0);
        if ((m->meta_off & ~(uint64_t)63) + t->nbytes > m->meta_off)
            patch64(m, TV("offset=%s", "metadata"), e + 104, m->meta_off & ~(uint64_t)63, 0);
        if (m->nt > 1) {
            int j = (i + 1 + (int)rndn((uint32_t)m->nt - 1)) % m->nt;
            patch64(m, TV("offset=%s", "other tensor"), e + 104, m->t[j].off, 0);
            uint8_t *d = dup_file(m);
            memcpy(d + e, d + m->tdir_off + 128 * (uint64_t)j, 80);
            try_variant(TV("%s", "duplicate name"), d, m->size, 0);
            free(d);
        }
        for (int k = 0; k < m->ne; k++)
            if (m->e[k].nbytes) { patch64(m, TV("offset=%s", "slab"), e + 104, m->e[k].off, 0); break; }
        patch64(m, TV("nbytes+%d", 1), e + 112, t->nbytes + 1, 0);
        patch64(m, TV("nbytes=%d", 0), e + 112, 0, 0);
        patch64(m, TV("nbytes=%s", "max"), e + 112, UINT64_MAX, 0);
        patch32(m, TV("dtype=%d", 7), e + 80, 7, 0);
        patch32(m, TV("dtype=%s", "max"), e + 80, 0xFFFFFFFFu, 0);
        patch32(m, TV("ndim=%d", 0), e + 84, 0, 0);
        patch32(m, TV("ndim=%d", 5), e + 84, 5, 0);
        patch32(m, TV("shape[0]=%d", 0), e + 88, 0, 0);
        patch32(m, TV("shape[0]+%d", 1), e + 88, t->shape[0] + 1, 0);
        if (t->ndim < 4) patch32(m, TV("unused dim %d = 2", t->ndim), e + 88 + 4 * (size_t)t->ndim, 2, 0);
        if (t->dtype == 3 || t->dtype == 4) patch32(m, TV("quant cols %d", 63), e + 88 + 4 * (size_t)(t->ndim - 1), 63, 0);
        {
            uint8_t *d = dup_file(m);
            for (int k = 0; k < 4; k++) w32(d + e + 88 + 4 * k, 0xFFFFFFFFu);
            w32(d + e + 84, 4);
            try_variant(TV("%s", "huge shape"), d, m->size, 0);
            memset(d + e, 'a', 80);
            try_variant(TV("%s", "name without NUL"), d, m->size, 0);
            free(d);
        }
        {
            uint8_t *d = dup_file(m);
            d[e] = 0;
            try_variant(TV("%s", "empty name"), d, m->size, 0);
            d[e] = 0x80;
            try_variant(TV("%s", "non-ASCII name"), d, m->size, 0);
            free(d);
        }
#undef TV
    }
}

static void fuzz_expert_entries(const model *m) {
    char w[128];
    int moe[4096], nm = 0, dense_e = -1;
    for (int i = 0; i < m->ne && nm < 4096; i++) {
        if (m->e[i].nbytes) moe[nm++] = i;
        else dense_e = i;
    }
    if (dense_e >= 0) patch64(m, "dense-layer entry with a slab", (size_t)(m->edir_off + 32 * (uint64_t)dense_e + 8), 4096, 0);
    if (!nm) return;
    /* A slab with a spare page and no aliases can move inside its own region: only
     * alignment decides validity (nothing else overlaps or runs out of the file). */
    for (int k = 0; k < nm; k++) {
        int i = moe[k], aliased = 0;
        const texpert *x = &m->e[i];
        for (int j = 0; j < m->ne; j++) aliased |= m->e[j].alias_of == i;
        if (x->alias_of >= 0 || aliased || x->nbytes < x->need + 4096) continue;
        uint8_t *d = dup_file(m);
        size_t e = (size_t)(m->edir_off + 32 * (uint64_t)i);
        w64(d + e, x->off + 2048); w64(d + e + 8, x->nbytes - 4096);
        try_variant("slack slab: offset off by 2048", d, m->size, 0);
        w64(d + e, x->off); w64(d + e + 8, x->nbytes - 2048);
        try_variant("slack slab: size off by 2048", d, m->size, 0);
        w64(d + e, x->off + 4096); w64(d + e + 8, x->nbytes - 4096);
        try_variant("slack slab: shifted by one page (valid)", d, m->size, 1);
        free(d);
        break;
    }
    for (int rep = 0; rep < 3; rep++) {
        int i = moe[rndn((uint32_t)nm)];
        const texpert *x = &m->e[i];
        size_t e = (size_t)(m->edir_off + 32 * (uint64_t)i);
#define EV(fmt, ...) (snprintf(w, sizeof w, "expert %d: " fmt, i, __VA_ARGS__), w)
        patch64(m, EV("offset+%d", 2048), e, x->off + 2048, 0);
        patch64(m, EV("offset=%s", "size"), e, al(m->size, 4096), 0);
        patch64(m, EV("offset=%s", "max"), e, UINT64_MAX & ~(uint64_t)4095, 0);
        patch64(m, EV("offset=%s", "0"), e, 0, 0);
        patch64(m, EV("nbytes+%d", 1), e + 8, x->nbytes + 1, 0);
        patch64(m, EV("nbytes=%d", 0), e + 8, 0, 0);
        patch64(m, EV("nbytes=%s", "need-4096"), e + 8, x->need - 4096, 0);
        patch64(m, EV("nbytes=%s", "max"), e + 8, UINT64_MAX & ~(uint64_t)4095, 0);
        patch32(m, EV("dtype=%d", 5), e + 16, 5, 0);
        patch32(m, EV("dtype=%d", 99), e + 16, 99, 0);
        uint64_t og, ou, od;
        if (x->dtype != 0 && x->nbytes < t_layout(0, (uint64_t)m->D, (uint64_t)m->F, &og, &ou, &od))
            patch32(m, EV("dtype %u->F32 (slab too small)", x->dtype), e + 16, 0, 0);
        /* overlap with other physical slabs */
        for (int k = 0; k < nm; k++) {
            const texpert *y = &m->e[moe[k]];
            int yi = moe[k];
            if (y->alias_of >= 0 || yi == i || y->off == x->off) continue;
            if (x->alias_of < 0) patch64(m, EV("offset=slab of %d", yi), e, y->off, 0);
            if (y->nbytes >= 8192) patch64(m, EV("offset inside slab of %d", yi), e, y->off + 4096, 0);
            break;
        }
        if (x->alias_of >= 0) {
            patch32(m, EV("%s", "alias without flag"), e + 20, x->flags & ~1u, 0);
            patch32(m, EV("%s", "alias with other dtype"), e + 16, (x->dtype + 1) % 5, 0);
        }
#undef EV
    }
}

/* Metadata corruption, both raw (bytes) and semantic (re-emitted from a modified config). */
static void fuzz_metadata(model *m) {
    /* raw: walk the entries to find a known u32 key and a string */
    const uint8_t *b = m->meta.p;
    size_t p = 0, first_u32 = 0, first_str = 0;
    int have_u32 = 0, have_str = 0;
    while (p < m->meta.n) {
        uint32_t kl = (uint32_t)(b[p] | b[p + 1] << 8);
        size_t tpos = p + 2 + kl;
        uint8_t t = b[tpos];
        size_t vpos = tpos + 1, len;
        if (t == T_U32 || t == T_F32) len = 4;
        else if (t == T_U64) len = 8;
        else { uint32_t cnt = r32(b + vpos); len = 4 + (size_t)cnt * ((t == T_STR || t == T_U8A) ? 1 : 4); }
        if (t == T_U32 && !have_u32 && kl == 8 && !memcmp(b + p + 2, "n_layers", 8)) { first_u32 = tpos; have_u32 = 1; }
        if (t == T_STR && !have_str && r32(b + vpos) >= 4) { first_str = vpos; have_str = 1; }
        p = vpos + len;
    }
    uint8_t *d = dup_file(m);
    size_t mo = (size_t)m->meta_off;
    if (have_u32) {
        d[mo + first_u32] = T_F32;
        try_variant("meta: n_layers typed f32", d, m->size, 0);
        d[mo + first_u32] = 0;
        try_variant("meta: type 0", d, m->size, 0);
        d[mo + first_u32] = 8;
        try_variant("meta: type 8", d, m->size, 0);
        d[mo + first_u32] = T_U32;
    }
    if (have_str) {
        uint8_t keep = d[mo + first_str + 4];
        d[mo + first_str + 4] = 0xFF;
        try_variant("meta: invalid UTF-8", d, m->size, 0);
        d[mo + first_str + 4] = 0xC0;   /* overlong lead byte */
        try_variant("meta: invalid UTF-8 (C0)", d, m->size, 0);
        d[mo + first_str + 4] = keep;
        if (r32(b + first_str) >= 4) {
            static const struct { const char *what, *bytes; size_t n; } bad[] = {
                {"surrogate U+D800", "\xED\xA0\x80", 3}, {"overlong NUL", "\xC0\x80", 2},
                {"overlong 3-byte", "\xE0\x80\xAF", 3}, {"above U+10FFFF", "\xF4\x90\x80\x80", 4}};
            uint8_t k4[4];
            memcpy(k4, d + mo + first_str + 4, 4);
            for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) {
                char w[64];
                snprintf(w, sizeof w, "meta: UTF-8 %s", bad[i].what);
                memcpy(d + mo + first_str + 4, bad[i].bytes, bad[i].n);
                try_variant(w, d, m->size, 0);
                memcpy(d + mo + first_str + 4, k4, 4);
            }
        }
        w32(d + mo + first_str, 0xFFFFFFFFu);
        try_variant("meta: string length overflow", d, m->size, 0);
        w32(d + mo + first_str, r32(m->file + mo + first_str));
    }
    w16(d + mo, 0xFFFF);
    try_variant("meta: key_len 65535", d, m->size, 0);
    d[mo] = m->file[mo]; d[mo + 1] = m->file[mo + 1];
    d[mo + 2] = 0x80;
    if (m->file[mo] | m->file[mo + 1]) try_variant("meta: non-ASCII key", d, m->size, 0);
    free(d);

    /* semantic: rebuild with a changed config, same tensors/experts */
    gencfg keep = m->g;
    const hx_config want = m->want;
    struct { const char *what; int key; uint32_t val; } bad_u[] = {
        {"n_layers 0", G_N_LAYERS, 0}, {"n_layers 513", G_N_LAYERS, 513}, {"d_model 0", G_D_MODEL, 0},
        {"vocab 0", G_VOCAB, 0}, {"n_heads 0", G_N_HEADS, 0}, {"attn_kind 2", G_ATTN_KIND, 2},
        {"qk_norm 3", G_QK_NORM, 3}, {"max_seq 0", G_MAX_SEQ, 0}, {"bos = vocab", G_BOS, (uint32_t)want.vocab_size},
        {"tie 2", G_TIE, 2}, {"n_group 0", G_N_GROUP, 0}, {"rope_dim odd", G_ROPE_DIM, 1},
        {"rope_dim > cap", G_ROPE_DIM, 2 * 65536},
    };
    for (size_t k = 0; k < sizeof bad_u / sizeof bad_u[0]; k++) {
        m->g = keep;
        set_u(&m->g, bad_u[k].key, bad_u[k].val);
        emit_meta(m); assemble(m);
        try_variant(bad_u[k].what, m->file, m->size, 0);
    }
    static const char *req[] = {"n_layers", "d_model", "vocab_size", "n_heads"};
    static const int reqk[] = {G_N_LAYERS, G_D_MODEL, G_VOCAB, G_N_HEADS};
    for (int k = 0; k < 4; k++) {
        m->g = keep;
        m->g.has[reqk[k]] = 0;
        emit_meta(m); assemble(m);
        char w[64];
        snprintf(w, sizeof w, "missing %s", req[k]);
        try_variant(w, m->file, m->size, 0);
    }
    m->g = keep; m->g.bad_type_key = G_N_HEADS; emit_meta(m); assemble(m);
    try_variant("n_heads typed f32", m->file, m->size, 0);
    if (want.attn_kind == 0 && want.n_heads >= 2) {
        m->g = keep; set_u(&m->g, G_N_HEADS, 3); set_u(&m->g, G_N_KV_HEADS, 2); emit_meta(m); assemble(m);
        try_variant("n_heads % n_kv_heads != 0", m->file, m->size, 0);
    }
    m->g = keep; set_f(&m->g, G_NORM_EPS, NAN); emit_meta(m); assemble(m);
    try_variant("norm_eps NaN", m->file, m->size, 0);
    m->g = keep; set_f(&m->g, G_ATTN_SCALE, INFINITY); emit_meta(m); assemble(m);
    try_variant("attn_scale inf", m->file, m->size, 0);
    m->g = keep; set_f(&m->g, G_NORM_EPS, -1.0f); emit_meta(m); assemble(m);
    try_variant("norm_eps negative", m->file, m->size, 0);
    m->g = keep; m->g.has[G_LAYER_KIND] = 1; m->g.n_lk = want.n_layers + 1; emit_meta(m); assemble(m);
    try_variant("layer_kind count", m->file, m->size, 0);
    m->g = keep; m->g.has[G_LAYER_KIND] = 1; m->g.n_lk = want.n_layers;
    for (int i = 0; i < want.n_layers; i++) m->g.lk[i] = want.layer_kind[i];
    m->g.lk[0] = 2; emit_meta(m); assemble(m);
    try_variant("layer_kind value 2", m->file, m->size, 0);
    {
        char longp[300];
        memset(longp, 'p', sizeof longp - 1);
        longp[sizeof longp - 1] = 0;
        m->g = keep; set_s(&m->g, G_TOKENIZER, longp); emit_meta(m); assemble(m);
        try_variant("tokenizer path too long", m->file, m->size, 0);
    }
    if (want.n_eos > 0 || keep.has[G_EOS]) {
        m->g = keep; m->g.has[G_EOS] = 1; m->g.n_eos = 1; m->g.eos[0] = (uint32_t)want.vocab_size; emit_meta(m); assemble(m);
        try_variant("eos = vocab", m->file, m->size, 0);
    }
    if (want.n_moe_layers > 0) {
        m->g = keep; set_u(&m->g, G_TOP_K, (uint32_t)want.n_experts + 1); emit_meta(m); assemble(m);
        try_variant("top_k > n_experts", m->file, m->size, 0);
        m->g = keep; set_u(&m->g, G_TOP_K, 0); emit_meta(m); assemble(m);
        try_variant("top_k 0", m->file, m->size, 0);
        m->g = keep; m->g.has[G_EXPERT_FFN] = 0; emit_meta(m); assemble(m);
        try_variant("expert_ffn_dim missing", m->file, m->size, 0);
        m->g = keep; set_u(&m->g, G_EXPERT_FFN, (uint32_t)want.expert_ffn_dim * 2); emit_meta(m); assemble(m);
        try_variant("expert_ffn_dim doubled (slabs too small)", m->file, m->size, 0);
        m->g = keep; m->g.has[G_N_EXPERTS] = 0; m->g.has[G_LAYER_KIND] = 1; m->g.n_lk = want.n_layers;
        for (int i = 0; i < want.n_layers; i++) m->g.lk[i] = want.layer_kind[i];
        emit_meta(m); assemble(m);
        try_variant("MoE layers without n_experts", m->file, m->size, 0);
        /* turning a MoE layer dense leaves non-empty slabs in a dense layer */
        m->g = keep; m->g.has[G_LAYER_KIND] = 1; m->g.n_lk = want.n_layers;
        for (int i = 0; i < want.n_layers; i++) m->g.lk[i] = want.layer_kind[i];
        for (int i = 0; i < want.n_layers; i++) if (m->g.lk[i]) { m->g.lk[i] = 0; break; }
        if (!m->g.has[G_DENSE_FFN]) set_u(&m->g, G_DENSE_FFN, 64);
        emit_meta(m); assemble(m);
        try_variant("MoE layer flipped to dense", m->file, m->size, 0);
        if (want.n_moe_layers < want.n_layers) {
            m->g = keep; m->g.has[G_DENSE_FFN] = 0; emit_meta(m); assemble(m);
            try_variant("dense layers without dense_ffn_dim", m->file, m->size, 0);
        }
    }
    if (want.rope_dim >= 4) {   /* right dtype, wrong length for the configured rope_dim */
        m->g = keep; set_u(&m->g, G_ROPE_DIM, (uint32_t)want.rope_dim - 2); emit_meta(m); assemble(m);
        try_variant("rope_inv_freq shorter than rope_dim/2", m->file, m->size, 0);
    }
    m->g = keep;
    emit_meta(m);
    assemble(m);

    /* rope_inv_freq: missing / wrong dtype / wrong shape */
    for (int i = 0; i < m->nt; i++) {
        if (strcmp(m->t[i].name, "rope_inv_freq")) continue;
        size_t e = (size_t)(m->tdir_off + 128 * (uint64_t)i);
        uint8_t *dd = dup_file(m);
        memcpy(dd + e, "rope_inv_freX", 13);
        try_variant("rope_inv_freq missing", dd, m->size, 0);
        free(dd);
        patch32(m, "rope_inv_freq as I32", e + 80, 5, 0);
        break;
    }
}

static void fuzz_random(const model *m, int iters) {
    static const uint64_t interesting[] = {0, 1, 63, 64, 4095, 4096, 0x7FFFFFFFFFFFFFFFull, 0x8000000000000000ull,
                                           0xFFFFFFFFull, 0x100000000ull, UINT64_MAX};
    for (int it = 0; it < iters; it++) {
        uint8_t *d = dup_file(m);
        int r = (int)rndn(100);
        size_t lo, hi;
        if (r < 15) { lo = 0; hi = 64; }
        else if (r < 45) { lo = (size_t)m->meta_off; hi = lo + m->meta.n; }
        else if (r < 75) { lo = (size_t)m->tdir_off; hi = lo + 128 * (size_t)m->nt; }
        else if (r < 95 && m->ne) { lo = (size_t)m->edir_off; hi = lo + 32 * (size_t)m->ne; }
        else { lo = 0; hi = m->size; }
        if (hi <= lo) { free(d); continue; }
        int nflip = 1 + (int)rndn(4);
        for (int k = 0; k < nflip; k++) {
            size_t at = lo + rndn((uint32_t)(hi - lo));
            switch (rndn(4)) {
            case 0: d[at] ^= (uint8_t)(1u << rndn(8)); break;
            case 1: d[at] = (uint8_t)rnd(); break;
            case 2: if (at + 8 <= m->size) w64(d + at, interesting[rndn(sizeof interesting / sizeof interesting[0])]); break;
            default: if (at + 4 <= m->size) w32(d + at, (uint32_t)interesting[rndn(sizeof interesting / sizeof interesting[0])]); break;
            }
        }
        size_t n = m->size;
        if (chance(5)) n = (size_t)(rnd() % m->size);
        try_variant("random corruption", d, n, 2);
        free(d);
    }
}

/* ---------------------------------------------------------------- misc */

static void test_slab_layout(void) {
    uint64_t g, u, d;
    /* Qwen3-30B-A3B Q4 expert: D=2048, F=768 -> 3 * 835584 bytes, already page aligned */
    CHECK(hx_slab_layout(HEARTH_Q4, 2048, 768, &g, &u, &d) == 2506752 && g == 0 && u == 835584 && d == 1671168,
          "Qwen3-30B-A3B Q4 slab layout");
    /* DeepSeek-V3 Q4 expert: D=7168, F=2048 */
    CHECK(hx_slab_layout(HEARTH_Q4, 7168, 2048, &g, &u, &d) == 23396352, "DeepSeek-V3 Q4 slab layout");
    CHECK(hx_slab_layout(HEARTH_F32, 3, 5, &g, &u, &d) == 4096 && u == 64 && d == 128, "F32 3x5 layout (64-byte gaps)");
    CHECK(hx_slab_layout(HEARTH_Q4, 100, 64, &g, &u, &d) == 0 && u == 0 && d == 0, "Q4 with D %% 64 != 0 rejected");
    CHECK(hx_slab_layout(HEARTH_F16, 0, 64, NULL, NULL, NULL) == 0, "D = 0 rejected");
    CHECK(hx_slab_layout(HEARTH_F16, (int64_t)1 << 39, (int64_t)1 << 39, NULL, NULL, NULL) == 0, "overflowing dims rejected");
    CHECK(hx_slab_layout(99, 64, 64, NULL, NULL, NULL) == 0, "invalid dtype rejected");
    for (int dt = 0; dt <= 4; dt++)
        for (int k = 0; k < 50; k++) {
            uint64_t D = 64 * (1 + rndn(64)), F = 64 * (1 + rndn(64)), a, b, c;
            uint64_t want = t_layout(dt, D, F, &a, &b, &c);
            CHECK(hx_slab_layout(dt, (int64_t)D, (int64_t)F, &g, &u, &d) == want && u == b && d == c,
                  "layout dtype %d D %llu F %llu", dt, (unsigned long long)D, (unsigned long long)F);
        }
    hx_slab_view v;
    hx_slab_view_make(NULL, &v, 0, &v);
    CHECK(v.gate == NULL, "hx_slab_view_make(NULL mf)");
}

static void test_open_errors(void) {
    char err[300];
    char p[800];
    snprintf(p, sizeof p, "%s/definitely_missing_%llu.hearth", g_dir, (unsigned long long)hx_now_ns());
    CHECK(hx_modelfile_open(p, 1, err, sizeof err) == NULL && err[0], "missing file");
    CHECK(hx_modelfile_open("", 1, err, sizeof err) == NULL && err[0], "empty path");
    CHECK(hx_modelfile_open(NULL, 1, NULL, 0) == NULL, "NULL path, NULL err");
    char longp[1100];
    memset(longp, 'p', sizeof longp - 1);
    longp[sizeof longp - 1] = 0;
    err[0] = 0;
    CHECK(hx_modelfile_open(longp, 1, err, sizeof err) == NULL && err[0], "over-long path");
    hx_modelfile_close(NULL);
    CHECK(hx_mf_tensor(NULL, "x") == NULL && hx_mf_layer_tensor(NULL, 0, "x") == NULL && hx_mf_expert(NULL, 0, 0) == NULL,
          "lookups on a NULL model");
    uint64_t og, ou, od;
    CHECK(hx_slab_layout(HEARTH_Q4, 64, 64, &og, &ou, &od) == 8192, "slab layout sanity");
    try_variant("empty file", (const uint8_t *)"", 0, 0);
}

/* Python's attn_scale default is float32(1)/sqrt(float32(qk_dim)); a few exact values. */
static void test_attn_scale_values(void) {
    static const struct { uint32_t hd; float s; } cases[] = {{64, 0.125f}, {16, 0.25f}, {128, 0.0883883461356163f}};
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        model m;
        for (;;) {
            make_model(&m, 1);
            if (m.want.attn_kind == 0) break;
            free_model(&m);
        }
        m.g.has[G_ATTN_SCALE] = 0;
        set_u(&m.g, G_HEAD_DIM, cases[i].hd);
        set_u(&m.g, G_ROPE_DIM, 0);
        for (int k = 0; k < m.nt; k++)
            if (!strcmp(m.t[k].name, "rope_inv_freq")) strcpy(m.t[k].name, "zz.not_rope");
        emit_meta(&m);
        assemble(&m);
        char err[600];
        write_file(g_path, m.file, m.size);
        hx_modelfile *mf = hx_modelfile_open(g_path, 0, err, sizeof err);
        CHECK(mf && fbits(mf->cfg.attn_scale) == fbits(cases[i].s), "attn_scale default for head_dim %u: %s", cases[i].hd,
              mf ? "" : err);
        hx_modelfile_close(mf);
        free_model(&m);
    }
}

/* A string that fits is copied whole even when the byte after it looks like a UTF-8
 * continuation byte (here the low byte, 0x82, of the next key's length). */
static void test_string_boundary(void) {
    static const char *archs[] = {"qwen3_moe", "abcdefghijklmnopqrstuvwxyz01234", "x"};
    for (size_t i = 0; i < sizeof archs / sizeof archs[0]; i++) {
        model m;
        make_model(&m, 1);
        set_s(&m.g, G_ARCH, archs[i]);
        m.g.arch_then_long_key = 1;
        expect_cfg(&m.g, &m.want);
        emit_meta(&m);
        assemble(&m);
        char err[600];
        write_file(g_path, m.file, m.size);
        hx_modelfile *mf = hx_modelfile_open(g_path, 0, err, sizeof err);
        CHECK(mf && strcmp(mf->cfg.arch, archs[i]) == 0, "arch '%s' read back as '%s'", archs[i], mf ? mf->cfg.arch : err);
        hx_modelfile_close(mf);
        free_model(&m);
    }
}

/* --------------------------------------------------------------- --dump */

static void json_str(const char *s) {
    putchar('"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') printf("\\%c", c);
        else if (c < 0x20) printf("\\u%04x", c);
        else putchar(c);
    }
    putchar('"');
}

static int dump(const char *path) {
    char err[600];
    hx_modelfile *mf = hx_modelfile_open(path, 1, err, sizeof err);
    if (!mf) { printf("{\"error\": "); json_str(err); printf("}\n"); return 1; }
    const hx_config *c = &mf->cfg;
    printf("{\"cfg\": {\"arch\": "); json_str(c->arch);
#define DI(f) printf(", \"" #f "\": %d", c->f)
#define DF(f) printf(", \"" #f "\": %.9g", (double)c->f)
    DI(n_layers); DI(d_model); DI(vocab_size); DI(max_seq); DF(norm_eps); DI(attn_kind); DI(n_heads); DI(n_kv_heads);
    DI(head_dim); DI(qk_norm); DI(qkv_bias); DI(q_lora_rank); DI(kv_lora_rank); DI(qk_nope_dim); DI(qk_rope_dim);
    DI(v_head_dim); DI(rope_dim); DI(rope_style); DF(rope_attn_factor); DF(attn_scale); DI(dense_ffn_dim);
    DI(n_experts); DI(top_k); DI(expert_ffn_dim); DI(shared_ffn_dim); DI(shared_gate); DI(score_fn); DI(score_bias);
    DI(n_group); DI(topk_group); DI(norm_topk_prob); DF(routed_scale); DF(emb_scale); DF(residual_scale);
    DF(logit_scale); DI(tie_embeddings); DI(n_moe_layers); DI(bos_id);
#undef DI
#undef DF
    printf(", \"layer_kind\": [");
    for (int i = 0; i < c->n_layers; i++) printf("%s%d", i ? ", " : "", c->layer_kind[i]);
    printf("], \"eos_ids\": [");
    for (int i = 0; i < c->n_eos; i++) printf("%s%d", i ? ", " : "", c->eos_ids[i]);
    printf("], \"tokenizer\": "); json_str(c->tokenizer);
    printf(", \"source\": "); json_str(c->source);
    printf("}, \"chat_template\": ");
    if (mf->chat_template) json_str(mf->chat_template); else printf("null");
    printf(", \"file_size\": %llu, \"dense_bytes\": %llu, \"expert_region_bytes\": %llu, \"slab_bytes_max\": %llu",
           (unsigned long long)mf->file_size, (unsigned long long)mf->dense_bytes,
           (unsigned long long)mf->expert_region_bytes, (unsigned long long)mf->slab_bytes_max);
    printf(", \"params_total\": %.17g, \"params_active\": %.17g", mf->params_total, mf->params_active);
    printf(", \"tensors\": [");
    for (int i = 0; i < mf->n_tensors; i++) {
        const hx_tensor *t = &mf->tensors[i];
        printf("%s{\"name\": ", i ? ", " : "");
        json_str(t->name);
        printf(", \"dtype\": %d, \"shape\": [", t->dtype);
        for (int k = 0; k < t->ndim; k++) printf("%s%lld", k ? ", " : "", (long long)t->shape[k]);
        printf("], \"offset\": %llu, \"nbytes\": %llu, \"fnv\": \"%016llx\"}", (unsigned long long)t->offset,
               (unsigned long long)t->nbytes, (unsigned long long)fnv(t->data, t->nbytes));
    }
    printf("], \"experts\": ");
    if (!mf->experts) printf("null");
    else {
        printf("[");
        for (int i = 0; i < c->n_layers * c->n_experts; i++) {
            const hx_expert_entry *e = &mf->experts[i];
            printf("%s[%llu, %llu, %u, %u]", i ? ", " : "", (unsigned long long)e->offset, (unsigned long long)e->nbytes,
                   e->dtype, e->flags);
        }
        printf("]");
    }
    printf("}\n");
    hx_modelfile_close(mf);
    return 0;
}

/* ---------------------------------------------------------------- main */

int main(int argc, char **argv) {
    int quick = 0, iters = -1;
    uint64_t seed = 12345;
    const char *dir = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dump") && i + 1 < argc) return dump(argv[i + 1]);
        else if (!strcmp(argv[i], "--quick")) quick = 1;
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dir") && i + 1 < argc) dir = argv[++i];
        else { printf("usage: test_modelfile [--quick] [--seed N] [--iters N] [--dir D] | --dump file\n"); return 2; }
    }
    if (iters < 0) iters = quick ? 300 : 2500;
    g_rng = seed;
    hx_set_log_level(HX_LOG_ERROR);
    pick_dir(dir);
    uint64_t t0 = hx_now_ns();
    printf("test_modelfile: seed %llu, scratch %s\n", (unsigned long long)seed, g_path);

    test_slab_layout();
    test_open_errors();
    test_attn_scale_values();
    test_string_boundary();

    /* round trips */
    int n_models = quick ? 40 : 200, n_mla = 0, n_dense = 0, n_alias = 0, n_shuffled = 0, n_after = 0, n_gap = 0;
    for (int k = 0; k < n_models; k++) {
        model m;
        make_model(&m, k % 3 == 0);
        char tag[64];
        snprintf(tag, sizeof tag, "model %d", k);
        if (!write_file(g_path, m.file, m.size)) { CHECK(0, "cannot write %s", g_path); free_model(&m); break; }
        verify_model(&m, tag, 1);
        verify_model(&m, tag, 0);
        n_mla += m.want.attn_kind;
        n_dense += m.want.n_moe_layers == 0;
        for (int i = 0; i < m.ne; i++) if (m.e[i].alias_of >= 0) { n_alias++; break; }
        n_shuffled += m.shuffle_sections;
        n_after += m.tensors_after_slabs;
        n_gap += m.big_gap;
        free_model(&m);
    }
    printf("  round trip: %d containers (%d MLA, %d without MoE layers, %d with aliased slabs, %d shuffled sections, "
           "%d tensors after slabs, %d with a >1 MiB gap) x load_dense 0/1\n",
           n_models, n_mla, n_dense, n_alias, n_shuffled, n_after, n_gap);

    /* fuzz */
    int bases = quick ? 3 : 8;
    int before = g_opens;
    for (int b = 0; b < bases; b++) {
        model m;
        make_model(&m, 1);
        fuzz_truncations(&m, quick ? 20 : 60);
        fuzz_preamble(&m);
        fuzz_tensor_entries(&m);
        fuzz_expert_entries(&m);
        fuzz_metadata(&m);
        fuzz_random(&m, iters / bases);
        free_model(&m);
    }
    printf("  fuzz: %d variants opened over %d base containers; %d randomly corrupted files were still valid and were "
           "walked\n", g_opens - before, bases, g_accepted_corrupt);

    remove(g_path);
    printf("test_modelfile: %d checks, %d failed (%.1f s)\n", g_checks, g_fail, (double)(hx_now_ns() - t0) / 1e9);
    return g_fail ? 1 : 0;
}
