/*
 * modelfile.c — .hearth container reader (docs/FORMAT.md, hx_modelfile.h).
 *
 * Order of trust: preamble -> section bounds -> metadata/config -> tensor
 * directory -> expert directory -> region overlap -> dense load. Nothing taken
 * from the file is used as a size, offset, count or index before it has been
 * range-checked with overflow-safe arithmetic (INV-FMT, INV-SAFE). Every
 * failure returns NULL with a message; nothing aborts.
 *
 * Acceptance follows python/hearth/format.py's ContainerReader, plus checks the
 * engine relies on: config consistency, non-MoE expert entries empty, and no
 * two regions (sections, tensors, distinct slabs) overlapping. Entries that
 * share one physical slab must agree on size and dtype, and all but one of
 * them must carry the aliased flag. Every canonical tensor of the config
 * (FORMAT.md §4.1) must be present with its exact shape and an allowed dtype,
 * so the forward pass can bind them without further checks.
 */
#include "hx_modelfile.h"
#include "hx_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PREAMBLE_BYTES 64u
#define TDIR_ENTRY     128u
#define EDIR_ENTRY     32u
#define TENSOR_ALIGN   64u
#define FLAG_ALIASED   1u
#define META_MAX       ((uint64_t)1 << 30)
#define TENSORS_MAX    ((uint64_t)1 << 20)
#define EXPERTS_MAX    65536u
#define HEADS_MAX      65536u
#define DIM_MAX        (1u << 20)
#define FFN_MAX        (1u << 22)
#define VOCAB_MAX      (1u << 27)
#define SEQ_MAX        (1u << 28)
#define RUN_GAP        ((uint64_t)1 << 20)    /* dense reads merge across gaps up to this */
#define LOAD_CHUNK     ((uint64_t)64 << 20)
#define LOAD_THREADS   4

typedef struct mf_impl {
    hx_modelfile pub;              /* first member: hx_modelfile * <-> mf_impl * */
    size_t arena_alloc;
} mf_impl;

/* ------------------------------------------------------------- helpers */

static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }
static float rdf32(const uint8_t *p) {
    uint32_t u = rd32(p);
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

/* [off, off + n) lies inside [0, size). */
static int in_file(uint64_t off, uint64_t n, uint64_t size) { return off <= size && n <= size - off; }

static int mul_ok(uint64_t a, uint64_t b, uint64_t *r) {
    if (a && b > UINT64_MAX / a) return 0;
    *r = a * b;
    return 1;
}

static uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) & ~(a - 1); }

static int read_at(hx_file *f, void *buf, uint64_t n, uint64_t off) {
    if (n == 0) return 1;
    if (n > (uint64_t)SIZE_MAX) return 0;
    return hx_file_pread(f, buf, (size_t)n, off) == (int64_t)n;
}

/* Strict UTF-8 (no overlongs, surrogates or code points above U+10FFFF), as Python decodes. */
static int utf8_valid(const uint8_t *s, uint32_t n) {
    uint32_t i = 0;
    while (i < n) {
        uint8_t c = s[i];
        if (c < 0x80) { i++; continue; }
        uint32_t len, cp, min;
        if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; min = 0x80; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; min = 0x800; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; min = 0x10000; }
        else return 0;
        if (n - i < len) return 0;
        for (uint32_t k = 1; k < len; k++) {
            if ((s[i + k] & 0xC0) != 0x80) return 0;
            cp = (cp << 6) | (s[i + k] & 0x3F);
        }
        if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return 0;
        i += len;
    }
    return 1;
}

/* Copy at most cap-1 bytes, never splitting a UTF-8 sequence. */
static void copy_str(char *dst, size_t cap, const uint8_t *s, uint32_t n) {
    size_t k = n < cap - 1 ? n : cap - 1;
    if (k < n)
        while (k > 0 && (s[k] & 0xC0) == 0x80) k--;
    memcpy(dst, s, k);
    dst[k] = 0;
}

/* ------------------------------------------------------------ metadata */

enum { MT_U32 = 1, MT_F32 = 2, MT_U64 = 3, MT_STR = 4, MT_U32A = 5, MT_F32A = 6, MT_U8A = 7 };

enum {
    K_ARCH, K_N_LAYERS, K_D_MODEL, K_VOCAB, K_MAX_SEQ, K_NORM_EPS, K_ATTN_KIND, K_N_HEADS, K_N_KV_HEADS,
    K_HEAD_DIM, K_QK_NORM, K_QKV_BIAS, K_Q_LORA, K_KV_LORA, K_QK_NOPE, K_QK_ROPE, K_V_HEAD, K_ROPE_DIM,
    K_ROPE_STYLE, K_ROPE_ATTN_FACTOR, K_ATTN_SCALE, K_DENSE_FFN, K_N_EXPERTS, K_TOP_K, K_EXPERT_FFN,
    K_SHARED_FFN, K_SHARED_GATE, K_SCORE_FN, K_SCORE_BIAS, K_N_GROUP, K_TOPK_GROUP, K_NORM_TOPK,
    K_ROUTED_SCALE, K_EMB_SCALE, K_RESID_SCALE, K_LOGIT_SCALE, K_TIE, K_LAYER_KIND, K_BOS, K_EOS,
    K_TOKENIZER, K_CHAT_TEMPLATE, K_SOURCE, K_EXPERT_DTYPE, K_COUNT
};

static const struct { const char *name; uint8_t type; } KEYS[K_COUNT] = {
    {"arch", MT_STR}, {"n_layers", MT_U32}, {"d_model", MT_U32}, {"vocab_size", MT_U32},
    {"max_seq", MT_U32}, {"norm_eps", MT_F32}, {"attn_kind", MT_U32}, {"n_heads", MT_U32},
    {"n_kv_heads", MT_U32}, {"head_dim", MT_U32}, {"qk_norm", MT_U32}, {"qkv_bias", MT_U32},
    {"q_lora_rank", MT_U32}, {"kv_lora_rank", MT_U32}, {"qk_nope_dim", MT_U32}, {"qk_rope_dim", MT_U32},
    {"v_head_dim", MT_U32}, {"rope_dim", MT_U32}, {"rope_style", MT_U32}, {"rope_attn_factor", MT_F32},
    {"attn_scale", MT_F32}, {"dense_ffn_dim", MT_U32}, {"n_experts", MT_U32}, {"top_k", MT_U32},
    {"expert_ffn_dim", MT_U32}, {"shared_ffn_dim", MT_U32}, {"shared_gate", MT_U32}, {"score_fn", MT_U32},
    {"score_bias", MT_U32}, {"n_group", MT_U32}, {"topk_group", MT_U32}, {"norm_topk_prob", MT_U32},
    {"routed_scale", MT_F32}, {"emb_scale", MT_F32}, {"residual_scale", MT_F32}, {"logit_scale", MT_F32},
    {"tie_embeddings", MT_U32}, {"layer_kind", MT_U8A}, {"bos_id", MT_U32}, {"eos_ids", MT_U32A},
    {"tokenizer", MT_STR}, {"chat_template", MT_STR}, {"source", MT_STR}, {"expert_dtype", MT_U32},
};

typedef struct { int present; const uint8_t *p; uint32_t n; } mval;

/* Later duplicates win (as in the Python reader); known keys must have their schema type. */
static int parse_meta(const uint8_t *b, uint64_t n, mval *kv, char *msg, size_t ml) {
    uint64_t p = 0;
    while (p < n) {
        if (n - p < 2) { snprintf(msg, ml, "metadata entry at +%llu is truncated", (unsigned long long)p); return 0; }
        uint32_t klen = rd16(b + p);
        uint64_t at = p;
        p += 2;
        if (n - p < (uint64_t)klen + 1) {
            snprintf(msg, ml, "metadata entry at +%llu runs past the end of the section", (unsigned long long)at);
            return 0;
        }
        const uint8_t *key = b + p;
        p += klen;
        for (uint32_t i = 0; i < klen; i++)
            if (key[i] >= 0x80) { snprintf(msg, ml, "metadata key at +%llu is not ASCII", (unsigned long long)at); return 0; }
        uint8_t type = b[p++];
        uint64_t plen;
        uint32_t count = 0;
        switch (type) {
        case MT_U32: case MT_F32: plen = 4; break;
        case MT_U64: plen = 8; break;
        case MT_STR: case MT_U32A: case MT_F32A: case MT_U8A:
            if (n - p < 4) { snprintf(msg, ml, "metadata entry at +%llu is truncated", (unsigned long long)at); return 0; }
            count = rd32(b + p);
            p += 4;
            plen = (uint64_t)count * ((type == MT_STR || type == MT_U8A) ? 1u : 4u);
            break;
        default:
            snprintf(msg, ml, "metadata '%.*s': unknown type %u", (int)(klen > 64 ? 64 : klen), (const char *)key, type);
            return 0;
        }
        if (n - p < plen) {
            snprintf(msg, ml, "metadata '%.*s': value runs past the end of the section", (int)(klen > 64 ? 64 : klen),
                     (const char *)key);
            return 0;
        }
        const uint8_t *val = b + p;
        p += plen;
        if (type == MT_STR && !utf8_valid(val, count)) {
            snprintf(msg, ml, "metadata '%.*s': invalid UTF-8", (int)(klen > 64 ? 64 : klen), (const char *)key);
            return 0;
        }
        for (int k = 0; k < K_COUNT; k++) {
            if (strlen(KEYS[k].name) != klen || memcmp(KEYS[k].name, key, klen) != 0) continue;
            if (KEYS[k].type != type) {
                snprintf(msg, ml, "metadata '%.40s' has type %u, expected %u", KEYS[k].name, type, KEYS[k].type);
                return 0;
            }
            kv[k].present = 1;
            kv[k].p = val;
            kv[k].n = count;
            break;
        }
    }
    return 1;
}

static int get_u32(const mval *kv, int k, uint32_t def, uint32_t lo, uint32_t hi, int *out, char *msg, size_t ml) {
    uint32_t v = kv[k].present ? rd32(kv[k].p) : def;
    if (v < lo || v > hi) {
        snprintf(msg, ml, "metadata %s = %u out of range [%u, %u]%s", KEYS[k].name, v, lo, hi,
                 kv[k].present ? "" : " (default)");
        return 0;
    }
    *out = (int)v;
    return 1;
}

static int get_f32(const mval *kv, int k, float def, float *out, char *msg, size_t ml) {
    float v = kv[k].present ? rdf32(kv[k].p) : def;
    if (!isfinite(v)) { snprintf(msg, ml, "metadata %s is not finite", KEYS[k].name); return 0; }
    *out = v;
    return 1;
}

#define U32(k, def, lo, hi, dst) do { if (!get_u32(kv, k, def, lo, hi, dst, msg, ml)) return 0; } while (0)
#define F32(k, def, dst) do { if (!get_f32(kv, k, def, dst, msg, ml)) return 0; } while (0)
#define BAD(...) do { snprintf(msg, ml, __VA_ARGS__); return 0; } while (0)

/* FORMAT.md §3.1 defaults plus the consistency rules the forward pass depends on. */
static int build_config(const mval *kv, hx_config *c, char **chat_template, char *msg, size_t ml) {
    static const int required[] = {K_N_LAYERS, K_D_MODEL, K_VOCAB, K_N_HEADS};
    memset(c, 0, sizeof *c);
    for (size_t i = 0; i < sizeof required / sizeof required[0]; i++)
        if (!kv[required[i]].present) BAD("metadata is missing required key '%s'", KEYS[required[i]].name);

    U32(K_N_LAYERS, 0, 1, HX_MAX_LAYERS, &c->n_layers);
    U32(K_D_MODEL, 0, 1, DIM_MAX, &c->d_model);
    U32(K_VOCAB, 0, 1, VOCAB_MAX, &c->vocab_size);
    U32(K_MAX_SEQ, 4096, 1, SEQ_MAX, &c->max_seq);
    F32(K_NORM_EPS, 1e-6f, &c->norm_eps);
    if (c->norm_eps < 0.0f) BAD("metadata norm_eps is negative");
    U32(K_ATTN_KIND, 0, 0, 1, &c->attn_kind);
    U32(K_N_HEADS, 0, 1, HEADS_MAX, &c->n_heads);
    U32(K_N_KV_HEADS, (uint32_t)c->n_heads, 1, HEADS_MAX, &c->n_kv_heads);
    U32(K_HEAD_DIM, (uint32_t)(c->d_model / c->n_heads), 0, HEADS_MAX, &c->head_dim);
    U32(K_QK_NORM, 0, 0, 2, &c->qk_norm);
    U32(K_QKV_BIAS, 0, 0, 1, &c->qkv_bias);
    U32(K_Q_LORA, 0, 0, DIM_MAX, &c->q_lora_rank);
    U32(K_KV_LORA, 0, 0, DIM_MAX, &c->kv_lora_rank);
    U32(K_QK_NOPE, 0, 0, HEADS_MAX, &c->qk_nope_dim);
    U32(K_QK_ROPE, 0, 0, HEADS_MAX, &c->qk_rope_dim);
    U32(K_V_HEAD, 0, 0, HEADS_MAX, &c->v_head_dim);

    const int mla = c->attn_kind == HX_ATTN_MLA;
    int qk_dim;
    if (mla) {
        if (c->kv_lora_rank < 1) BAD("MLA needs kv_lora_rank >= 1");
        if (c->v_head_dim < 1) BAD("MLA needs v_head_dim >= 1");
        qk_dim = c->qk_nope_dim + c->qk_rope_dim;
        if (qk_dim < 1) BAD("MLA needs qk_nope_dim + qk_rope_dim >= 1");
    } else {
        if (c->head_dim < 1) BAD("head_dim is 0 (d_model %d < n_heads %d and no head_dim key)", c->d_model, c->n_heads);
        if (c->n_heads % c->n_kv_heads != 0)
            BAD("n_heads %d is not a multiple of n_kv_heads %d", c->n_heads, c->n_kv_heads);
        qk_dim = c->head_dim;
    }
    const int rope_cap = mla ? c->qk_rope_dim : c->head_dim;
    U32(K_ROPE_DIM, (uint32_t)rope_cap, 0, (uint32_t)rope_cap, &c->rope_dim);
    if (c->rope_dim % 2) BAD("rope_dim %d is odd", c->rope_dim);
    U32(K_ROPE_STYLE, 0, 0, 1, &c->rope_style);
    F32(K_ROPE_ATTN_FACTOR, 1.0f, &c->rope_attn_factor);
    /* float arithmetic, matching hearth.format.apply_defaults (np.float32) */
    F32(K_ATTN_SCALE, 1.0f / sqrtf((float)qk_dim), &c->attn_scale);

    U32(K_DENSE_FFN, 0, 0, FFN_MAX, &c->dense_ffn_dim);
    U32(K_N_EXPERTS, 0, 0, EXPERTS_MAX, &c->n_experts);
    U32(K_TOP_K, 0, 0, EXPERTS_MAX, &c->top_k);
    U32(K_EXPERT_FFN, 0, 0, FFN_MAX, &c->expert_ffn_dim);
    U32(K_SHARED_FFN, 0, 0, FFN_MAX, &c->shared_ffn_dim);
    U32(K_SHARED_GATE, 0, 0, 1, &c->shared_gate);
    U32(K_SCORE_FN, 0, 0, 1, &c->score_fn);
    U32(K_SCORE_BIAS, 0, 0, 1, &c->score_bias);
    U32(K_N_GROUP, 1, 1, EXPERTS_MAX, &c->n_group);
    U32(K_TOPK_GROUP, 1, 1, EXPERTS_MAX, &c->topk_group);
    U32(K_NORM_TOPK, 0, 0, 1, &c->norm_topk_prob);
    F32(K_ROUTED_SCALE, 1.0f, &c->routed_scale);
    F32(K_EMB_SCALE, 1.0f, &c->emb_scale);
    F32(K_RESID_SCALE, 1.0f, &c->residual_scale);
    F32(K_LOGIT_SCALE, 1.0f, &c->logit_scale);
    U32(K_TIE, 0, 0, 1, &c->tie_embeddings);

    if (kv[K_LAYER_KIND].present) {
        if (kv[K_LAYER_KIND].n != (uint32_t)c->n_layers)
            BAD("layer_kind has %u entries for %d layers", kv[K_LAYER_KIND].n, c->n_layers);
        for (int i = 0; i < c->n_layers; i++) {
            uint8_t k = kv[K_LAYER_KIND].p[i];
            if (k > 1) BAD("layer_kind[%d] = %u (must be 0 or 1)", i, k);
            c->layer_kind[i] = k;
        }
    } else {
        for (int i = 0; i < c->n_layers; i++) c->layer_kind[i] = c->n_experts > 0 ? HX_LAYER_MOE : HX_LAYER_DENSE;
    }
    c->n_moe_layers = 0;
    for (int i = 0; i < c->n_layers; i++) c->n_moe_layers += c->layer_kind[i];

    if (c->n_moe_layers > 0) {
        if (c->n_experts < 1) BAD("layer_kind marks %d MoE layers but n_experts is 0", c->n_moe_layers);
        if (c->top_k < 1 || c->top_k > c->n_experts) BAD("top_k %d out of range [1, n_experts %d]", c->top_k, c->n_experts);
        if (c->expert_ffn_dim < 1) BAD("MoE layers need expert_ffn_dim >= 1");
        if (c->n_experts % c->n_group) BAD("n_experts %d is not a multiple of n_group %d", c->n_experts, c->n_group);
        if (c->topk_group > c->n_group) BAD("topk_group %d > n_group %d", c->topk_group, c->n_group);
        if ((int64_t)c->top_k > (int64_t)c->topk_group * (c->n_experts / c->n_group))
            BAD("top_k %d exceeds the experts in topk_group %d groups", c->top_k, c->topk_group);
    }
    if (c->n_moe_layers < c->n_layers && c->dense_ffn_dim < 1) BAD("dense layers need dense_ffn_dim >= 1");

    uint32_t bos = kv[K_BOS].present ? rd32(kv[K_BOS].p) : 0xFFFFFFFFu;
    if (bos == 0xFFFFFFFFu) c->bos_id = -1;
    else if (bos >= (uint32_t)c->vocab_size) BAD("bos_id %u >= vocab_size %d", bos, c->vocab_size);
    else c->bos_id = (int)bos;
    if (kv[K_EOS].present) {
        uint32_t n = kv[K_EOS].n;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t id = rd32(kv[K_EOS].p + 4 * (size_t)i);
            if (id >= (uint32_t)c->vocab_size) BAD("eos_ids[%u] = %u >= vocab_size %d", i, id, c->vocab_size);
            if (c->n_eos < HX_MAX_EOS) c->eos_ids[c->n_eos++] = (int)id;
        }
        if (n > HX_MAX_EOS) hx_log(HX_LOG_WARN, "eos_ids has %u entries; keeping the first %d", n, HX_MAX_EOS);
    }

    if (kv[K_ARCH].present) copy_str(c->arch, sizeof c->arch, kv[K_ARCH].p, kv[K_ARCH].n);
    if (kv[K_SOURCE].present) copy_str(c->source, sizeof c->source, kv[K_SOURCE].p, kv[K_SOURCE].n);
    if (kv[K_TOKENIZER].present) {
        uint32_t n = kv[K_TOKENIZER].n;
        if (n >= sizeof c->tokenizer) BAD("tokenizer path is %u bytes (max %u)", n, (unsigned)sizeof c->tokenizer - 1);
        if (memchr(kv[K_TOKENIZER].p, 0, n)) BAD("tokenizer path contains a NUL byte");
        memcpy(c->tokenizer, kv[K_TOKENIZER].p, n);
        c->tokenizer[n] = 0;
    }
    if (kv[K_CHAT_TEMPLATE].present && kv[K_CHAT_TEMPLATE].n > 0) {
        uint32_t n = kv[K_CHAT_TEMPLATE].n;
        char *t = (char *)malloc((size_t)n + 1);
        if (!t) BAD("out of memory (chat_template, %u bytes)", n);
        memcpy(t, kv[K_CHAT_TEMPLATE].p, n);
        t[n] = 0;
        *chat_template = t;
    }
    return 1;
}

/* ------------------------------------------------------- slab layout */

uint64_t hx_slab_layout(int dtype, int64_t D, int64_t F, uint64_t *off_gate, uint64_t *off_up, uint64_t *off_down) {
    uint64_t og = 0, ou = 0, od = 0, total = 0, a, b, c;
    size_t rd = (D > 0 && F > 0) ? hx_row_bytes(dtype, D) : 0;
    size_t rf = (D > 0 && F > 0) ? hx_row_bytes(dtype, F) : 0;
    if (rd && rf && mul_ok((uint64_t)F, rd, &a) && mul_ok((uint64_t)D, rf, &c) && a <= (UINT64_MAX >> 4) &&
        c <= (UINT64_MAX >> 4)) {
        ou = align_up(og + a, 64);
        od = align_up(ou + a, 64);
        b = od + c;
        total = align_up(b, HX_SLAB_ALIGN);
    }
    if (!total) ou = od = 0;
    if (off_gate) *off_gate = og;
    if (off_up) *off_up = ou;
    if (off_down) *off_down = od;
    return total;
}

void hx_slab_view_make(const hx_modelfile *mf, const void *slab, int dtype, hx_slab_view *out) {
    uint64_t og, ou, od;
    memset(out, 0, sizeof *out);
    out->dtype = dtype;
    if (!mf || !slab) return;
    if (!hx_slab_layout(dtype, mf->cfg.d_model, mf->cfg.expert_ffn_dim, &og, &ou, &od)) return;
    const uint8_t *p = (const uint8_t *)slab;
    out->gate = p + og;
    out->up = p + ou;
    out->down = p + od;
}

/* ------------------------------------------------------------ regions */

enum { R_PRE, R_META, R_TDIR, R_EDIR, R_TENSOR, R_SLAB };
typedef struct { uint64_t lo, hi; int what, idx; } region;

static int cmp_region(const void *a, const void *b) {
    const region *x = (const region *)a, *y = (const region *)b;
    if (x->lo != y->lo) return x->lo < y->lo ? -1 : 1;
    if (x->hi != y->hi) return x->hi < y->hi ? -1 : 1;
    return 0;
}

static void describe(const region *r, const hx_modelfile *mf, char *buf, size_t n) {
    static const char *names[] = {"preamble", "metadata", "tensor directory", "expert directory"};
    if (r->what == R_TENSOR) snprintf(buf, n, "tensor '%s'", mf->tensors[r->idx].name);
    else if (r->what == R_SLAB) snprintf(buf, n, "expert slab of entry %d", r->idx);
    else snprintf(buf, n, "%s", names[r->what]);
}

typedef struct { uint64_t off, nbytes; uint32_t dtype, flags; int idx; } slabref;

static int cmp_slabref(const void *a, const void *b) {
    const slabref *x = (const slabref *)a, *y = (const slabref *)b;
    if (x->off != y->off) return x->off < y->off ? -1 : 1;
    return (x->idx > y->idx) - (x->idx < y->idx);
}

/* Groups entries by physical slab, checks aliases, computes the region totals, and
 * verifies that no two regions overlap. */
static int check_regions(hx_modelfile *mf, const uint64_t sec[4][2], char *msg, size_t ml) {
    int ne = mf->experts ? mf->cfg.n_layers * mf->cfg.n_experts : 0;
    slabref *sr = NULL;
    region *rg = NULL;
    int ok = 0, ns = 0, nr = 0;
    if (ne) {
        sr = (slabref *)malloc(sizeof *sr * (size_t)ne);
        if (!sr) { snprintf(msg, ml, "out of memory"); goto done; }
        for (int i = 0; i < ne; i++) {
            const hx_expert_entry *e = &mf->experts[i];
            if (!e->nbytes) continue;
            sr[ns].off = e->offset; sr[ns].nbytes = e->nbytes; sr[ns].dtype = e->dtype;
            sr[ns].flags = e->flags; sr[ns].idx = i;
            ns++;
        }
        qsort(sr, (size_t)ns, sizeof *sr, cmp_slabref);
    }
    rg = (region *)malloc(sizeof *rg * ((size_t)mf->n_tensors + (size_t)ns + 4));
    if (!rg) { snprintf(msg, ml, "out of memory"); goto done; }
    for (int s = 0; s < 4; s++)
        if (sec[s][1]) { rg[nr].lo = sec[s][0]; rg[nr].hi = sec[s][0] + sec[s][1]; rg[nr].what = s; rg[nr].idx = 0; nr++; }
    for (int i = 0; i < mf->n_tensors; i++) {
        rg[nr].lo = mf->tensors[i].offset; rg[nr].hi = mf->tensors[i].offset + mf->tensors[i].nbytes;
        rg[nr].what = R_TENSOR; rg[nr].idx = i;
        nr++;
    }
    mf->expert_region_bytes = 0;
    mf->slab_bytes_max = 0;
    for (int i = 0; i < ns;) {
        int j = i + 1, plain = !(sr[i].flags & FLAG_ALIASED);
        for (; j < ns && sr[j].off == sr[i].off; j++) {
            if (sr[j].nbytes != sr[i].nbytes || sr[j].dtype != sr[i].dtype) {
                snprintf(msg, ml, "expert entries %d and %d share offset %llu but differ in size or dtype", sr[i].idx,
                         sr[j].idx, (unsigned long long)sr[i].off);
                goto done;
            }
            plain += !(sr[j].flags & FLAG_ALIASED);
        }
        if (plain > 1) {
            snprintf(msg, ml, "expert entries %d and %d point at the same slab without the aliased flag", sr[i].idx,
                     sr[i + 1].idx);
            goto done;
        }
        rg[nr].lo = sr[i].off; rg[nr].hi = sr[i].off + sr[i].nbytes; rg[nr].what = R_SLAB; rg[nr].idx = sr[i].idx;
        nr++;
        mf->expert_region_bytes += sr[i].nbytes;
        if (sr[i].nbytes > mf->slab_bytes_max) mf->slab_bytes_max = sr[i].nbytes;
        i = j;
    }
    qsort(rg, (size_t)nr, sizeof *rg, cmp_region);
    for (int i = 1, m = 0; i < nr; i++) {
        if (rg[i].lo < rg[m].hi) {
            char a[128], b[128];
            describe(&rg[m], mf, a, sizeof a);
            describe(&rg[i], mf, b, sizeof b);
            snprintf(msg, ml, "%s overlaps %s", a, b);
            goto done;
        }
        if (rg[i].hi > rg[m].hi) m = i;
    }
    ok = 1;
done:
    free(sr);
    free(rg);
    return ok;
}

/* -------------------------------------------------------- dense loading */

typedef struct { uint64_t foff, len, need; uint8_t *dst; } ld_chunk;
typedef struct { hx_file *f; ld_chunk *c; int n; atomic_int next, bad; } ld_ctx;

static void *ld_worker(void *arg) {
    ld_ctx *x = (ld_ctx *)arg;
    for (;;) {
        int i = atomic_fetch_add(&x->next, 1);
        if (i >= x->n || atomic_load(&x->bad)) break;
        int64_t got = hx_file_pread(x->f, x->c[i].dst, (size_t)x->c[i].len, x->c[i].foff);
        if (got < 0 || (uint64_t)got < x->c[i].need) atomic_store(&x->bad, 1);
    }
    return NULL;
}

typedef struct { uint64_t off; int idx; } offidx;

static int cmp_offidx(const void *a, const void *b) {
    const offidx *x = (const offidx *)a, *y = (const offidx *)b;
    if (x->off != y->off) return x->off < y->off ? -1 : 1;
    return (x->idx > y->idx) - (x->idx < y->idx);
}

/*
 * Tensors are grouped into runs of nearby file data; each run is copied with
 * page-aligned reads into an arena region that mirrors the file layout modulo
 * 4096, so tensor.data keeps the file's 64-byte alignment and the reads can be
 * unbuffered (no second copy of the backbone in the OS page cache).
 */
static int load_dense(mf_impl *m, char *msg, size_t ml) {
    hx_modelfile *mf = &m->pub;
    int nt = mf->n_tensors, nchunks = 0, ok = 0;
    offidx *order = NULL;
    uint64_t *aoff = NULL;
    ld_chunk *ch = NULL;
    hx_file *f = NULL;
    if (nt == 0) return 1;
    order = (offidx *)malloc(sizeof *order * (size_t)nt);
    aoff = (uint64_t *)malloc(sizeof *aoff * (size_t)nt);
    if (!order || !aoff) { snprintf(msg, ml, "out of memory"); goto done; }
    for (int i = 0; i < nt; i++) { order[i].off = mf->tensors[i].offset; order[i].idx = i; }
    qsort(order, (size_t)nt, sizeof *order, cmp_offidx);

    /* pass 1: arena layout and chunk count */
    uint64_t arena = 0;
    for (int i = 0; i < nt;) {
        const hx_tensor *t0 = &mf->tensors[order[i].idx];
        uint64_t lo = t0->offset & ~(uint64_t)(HX_PAGE - 1), end = t0->offset + t0->nbytes;
        int j = i + 1;
        for (; j < nt; j++) {
            const hx_tensor *t = &mf->tensors[order[j].idx];
            if (t->offset > end + RUN_GAP) break;
            if (t->offset + t->nbytes > end) end = t->offset + t->nbytes;
        }
        uint64_t hi = align_up(end, HX_PAGE);
        for (int k = i; k < j; k++) aoff[order[k].idx] = arena + (mf->tensors[order[k].idx].offset - lo);
        nchunks += (int)((hi - lo + LOAD_CHUNK - 1) / LOAD_CHUNK);
        arena += hi - lo;
        i = j;
    }
    if (arena > (uint64_t)SIZE_MAX) { snprintf(msg, ml, "dense region too large"); goto done; }
    m->arena_alloc = (size_t)arena;
    mf->dense_arena = hx_alloc_large((size_t)arena, 1);
    if (!mf->dense_arena) {
        snprintf(msg, ml, "cannot allocate %.1f MiB for the dense tensors", (double)arena / (1 << 20));
        goto done;
    }
    ch = (ld_chunk *)malloc(sizeof *ch * (size_t)nchunks);
    if (!ch) { snprintf(msg, ml, "out of memory"); goto done; }

    /* pass 2: chunks (same walk) */
    uint8_t *base = (uint8_t *)mf->dense_arena;
    int nc = 0;
    arena = 0;
    for (int i = 0; i < nt;) {
        const hx_tensor *t0 = &mf->tensors[order[i].idx];
        uint64_t lo = t0->offset & ~(uint64_t)(HX_PAGE - 1), end = t0->offset + t0->nbytes;
        int j = i + 1;
        for (; j < nt; j++) {
            const hx_tensor *t = &mf->tensors[order[j].idx];
            if (t->offset > end + RUN_GAP) break;
            if (t->offset + t->nbytes > end) end = t->offset + t->nbytes;
        }
        uint64_t hi = align_up(end, HX_PAGE);
        for (uint64_t o = lo; o < hi; o += LOAD_CHUNK) {
            ld_chunk *c = &ch[nc++];
            c->foff = o;
            c->len = hi - o < LOAD_CHUNK ? hi - o : LOAD_CHUNK;
            c->need = end > o ? (end - o < c->len ? end - o : c->len) : 0;
            c->dst = base + arena + (o - lo);
        }
        arena += hi - lo;
        i = j;
    }

    char ferr[256];
    f = hx_file_open(mf->path, HX_FILE_READ | HX_FILE_DIRECT, ferr, sizeof ferr);
    if (!f) { snprintf(msg, ml, "%s", ferr); goto done; }
    ld_ctx x;
    x.f = f; x.c = ch; x.n = nc;
    atomic_init(&x.next, 0);
    atomic_init(&x.bad, 0);
    int nthr = nc < LOAD_THREADS ? nc : LOAD_THREADS;
    hx_thread *th[LOAD_THREADS];
    int started = 0;
    for (int i = 1; i < nthr; i++)
        if (hx_thread_create(&th[started], ld_worker, &x) == 0) started++;
    ld_worker(&x);
    for (int i = 0; i < started; i++) hx_thread_join(th[i]);
    if (atomic_load(&x.bad)) { snprintf(msg, ml, "read error while loading dense tensors"); goto done; }

    for (int i = 0; i < nt; i++) mf->tensors[i].data = base + aoff[i];
    ok = 1;
done:
    if (f) hx_file_close(f);
    free(order);
    free(aoff);
    free(ch);
    return ok;
}

/* ---------------------------------------------- canonical tensors (§4.1) */

/* Tensor "blk.<layer>.<suffix>" (layer >= 0) or "<suffix>" exists with exactly the shape
 * [d0] (ndim 1) or [d0, d1] (ndim 2), and is F32 (f32_only) or a matrix dtype. */
static int canon(const hx_modelfile *mf, int layer, const char *suffix, int ndim, int64_t d0, int64_t d1, int f32_only,
                 char *msg, size_t ml) {
    char name[HX_NAME_LEN + 16];
    if (layer < 0) snprintf(name, sizeof name, "%s", suffix);
    else snprintf(name, sizeof name, "blk.%d.%s", layer, suffix);
    const hx_tensor *t = hx_mf_tensor(mf, name);
    if (!t) BAD("tensor '%s' is missing", name);
    if (t->ndim != ndim || t->shape[0] != d0 || (ndim == 2 && t->shape[1] != d1)) {
        char got[96];
        int n = 0;
        for (int k = 0; k < t->ndim && n >= 0 && n < (int)sizeof got; k++)
            n += snprintf(got + n, sizeof got - (size_t)n, "%s%lld", k ? ", " : "", (long long)t->shape[k]);
        if (ndim == 1) BAD("tensor '%s' has shape [%s], expected [%lld]", name, got, (long long)d0);
        BAD("tensor '%s' has shape [%s], expected [%lld, %lld]", name, got, (long long)d0, (long long)d1);
    }
    if (f32_only && t->dtype != HEARTH_F32) BAD("tensor '%s' is %s, must be F32", name, hx_dtype_name(t->dtype));
    if (t->dtype > HEARTH_Q4)
        BAD("tensor '%s' has dtype %s; matrices must be F32, F16, BF16, Q8 or Q4", name, hx_dtype_name(t->dtype));
    return 1;
}

/* FORMAT.md §4.1 for this config, as hearth.format.canonical_tensors: every tensor the forward
 * pass reads exists with its exact shape and an allowed dtype (norms, biases, rope_inv_freq and
 * the router F32). Shared experts and router biases belong to MoE layers. Other tensors are
 * ignored, except rope_inv_freq, which is present iff rope_dim > 0. Called with the tensors
 * sorted (lookups), before the dense data is loaded. */
static int check_canonical(const hx_modelfile *mf, char *msg, size_t ml) {
#define VEC(l, s, n) do { if (!canon(mf, l, s, 1, n, 1, 1, msg, ml)) return 0; } while (0)
#define MAT(l, s, r, k) do { if (!canon(mf, l, s, 2, r, k, 0, msg, ml)) return 0; } while (0)
#define F32MAT(l, s, r, k) do { if (!canon(mf, l, s, 2, r, k, 1, msg, ml)) return 0; } while (0)
    const hx_config *c = &mf->cfg;
    const int64_t D = c->d_model, H = c->n_heads;
    MAT(-1, "tok_embd", c->vocab_size, D);
    if (!c->tie_embeddings) MAT(-1, "lm_head", c->vocab_size, D);
    VEC(-1, "out_norm", D);
    if (c->rope_dim > 0) VEC(-1, "rope_inv_freq", c->rope_dim / 2);
    else if (hx_mf_tensor(mf, "rope_inv_freq")) BAD("tensor 'rope_inv_freq' is present although rope_dim is 0");
    for (int l = 0; l < c->n_layers; l++) {
        VEC(l, "attn_norm", D);
        VEC(l, "ffn_norm", D);
        if (c->attn_kind == HX_ATTN_MLA) {
            const int64_t nope = c->qk_nope_dim, rope = c->qk_rope_dim, C = c->kv_lora_rank, ql = c->q_lora_rank;
            if (ql > 0) {
                MAT(l, "attn_q_a", ql, D);
                VEC(l, "attn_q_a_norm", ql);
                MAT(l, "attn_q_b", H * (nope + rope), ql);
            } else {
                MAT(l, "attn_q", H * (nope + rope), D);
            }
            MAT(l, "attn_kv_a", C + rope, D);
            VEC(l, "attn_kv_a_norm", C);
            MAT(l, "attn_kv_b", H * (nope + c->v_head_dim), C);
            MAT(l, "attn_o", D, H * c->v_head_dim);
        } else {
            const int64_t hd = c->head_dim, q = H * hd, kv = (int64_t)c->n_kv_heads * hd;
            MAT(l, "attn_q", q, D);
            MAT(l, "attn_k", kv, D);
            MAT(l, "attn_v", kv, D);
            MAT(l, "attn_o", D, q);
            if (c->qkv_bias) {
                VEC(l, "attn_q_bias", q);
                VEC(l, "attn_k_bias", kv);
                VEC(l, "attn_v_bias", kv);
            }
            if (c->qk_norm != HX_QKNORM_NONE) {
                VEC(l, "attn_q_norm", c->qk_norm == HX_QKNORM_HEAD ? hd : q);
                VEC(l, "attn_k_norm", c->qk_norm == HX_QKNORM_HEAD ? hd : kv);
            }
        }
        if (c->layer_kind[l] == HX_LAYER_DENSE) {
            MAT(l, "ffn_gate", c->dense_ffn_dim, D);
            MAT(l, "ffn_up", c->dense_ffn_dim, D);
            MAT(l, "ffn_down", D, c->dense_ffn_dim);
            continue;
        }
        F32MAT(l, "moe_router", c->n_experts, D);
        if (c->score_bias) VEC(l, "moe_router_bias", c->n_experts);
        if (c->shared_ffn_dim > 0) {
            MAT(l, "shexp_gate", c->shared_ffn_dim, D);
            MAT(l, "shexp_up", c->shared_ffn_dim, D);
            MAT(l, "shexp_down", D, c->shared_ffn_dim);
            if (c->shared_gate) MAT(l, "shexp_gate_inp", 1, D);
        }
    }
    return 1;
#undef VEC
#undef MAT
#undef F32MAT
}

/* --------------------------------------------------------------- open */

static int cmp_tensor_name(const void *a, const void *b) {
    return strcmp(((const hx_tensor *)a)->name, ((const hx_tensor *)b)->name);
}

static int parse(mf_impl *m, hx_file *f, int want_dense, char *msg, size_t ml) {
    hx_modelfile *mf = &m->pub;
    uint8_t pre[PREAMBLE_BYTES];
    uint8_t *meta = NULL, *tdir = NULL, *edir = NULL;
    int ok = 0;
    const uint64_t size = mf->file_size;

    if (size < PREAMBLE_BYTES) { snprintf(msg, ml, "file too small (%llu bytes) for a .hearth container", (unsigned long long)size); goto done; }
    if (!read_at(f, pre, PREAMBLE_BYTES, 0)) { snprintf(msg, ml, "cannot read the preamble"); goto done; }
    uint32_t magic = rd32(pre), version = rd32(pre + 4), align = rd32(pre + 56);
    uint64_t meta_off = rd64(pre + 8), meta_bytes = rd64(pre + 16), tdir_off = rd64(pre + 24);
    uint64_t n_t = rd64(pre + 32), edir_off = rd64(pre + 40), n_e = rd64(pre + 48);
    if (magic != HX_MAGIC) { snprintf(msg, ml, "not a .hearth container (magic 0x%08x)", magic); goto done; }
    if (version != HX_FORMAT_VER) { snprintf(msg, ml, "unsupported format version %u", version); goto done; }
    if (align != HX_SLAB_ALIGN) { snprintf(msg, ml, "slab alignment %u != %u", align, HX_SLAB_ALIGN); goto done; }
    if (!in_file(meta_off, meta_bytes, size)) { snprintf(msg, ml, "metadata section lies outside the file"); goto done; }
    if (meta_bytes > META_MAX) { snprintf(msg, ml, "metadata section of %llu bytes is implausibly large", (unsigned long long)meta_bytes); goto done; }
    if (n_t > size / TDIR_ENTRY || n_t > TENSORS_MAX) { snprintf(msg, ml, "tensor count %llu impossible for this file", (unsigned long long)n_t); goto done; }
    if (!in_file(tdir_off, n_t * TDIR_ENTRY, size)) { snprintf(msg, ml, "tensor directory lies outside the file"); goto done; }
    if (n_e > size / EDIR_ENTRY) { snprintf(msg, ml, "expert entry count %llu impossible for this file", (unsigned long long)n_e); goto done; }
    if (!in_file(edir_off, n_e * EDIR_ENTRY, size)) { snprintf(msg, ml, "expert directory lies outside the file"); goto done; }

    /* metadata */
    meta = (uint8_t *)malloc(meta_bytes ? (size_t)meta_bytes : 1);
    if (!meta) { snprintf(msg, ml, "out of memory (metadata)"); goto done; }
    if (!read_at(f, meta, meta_bytes, meta_off)) { snprintf(msg, ml, "cannot read the metadata section"); goto done; }
    mval kv[K_COUNT];
    memset(kv, 0, sizeof kv);
    if (!parse_meta(meta, meta_bytes, kv, msg, ml)) goto done;
    if (!build_config(kv, &mf->cfg, &mf->chat_template, msg, ml)) goto done;
    hx_config *c = &mf->cfg;

    const int L = c->n_layers, E = c->n_experts;
    const int has_moe = E > 0 && c->n_moe_layers > 0;
    const uint64_t want_e = (uint64_t)L * (uint64_t)E;
    if (has_moe ? n_e != want_e : (n_e != 0 && n_e != want_e)) {
        snprintf(msg, ml, "%llu expert entries, expected n_layers*n_experts = %llu", (unsigned long long)n_e,
                 (unsigned long long)want_e);
        goto done;
    }

    /* tensor directory */
    mf->n_tensors = (int)n_t;
    if (n_t) {
        tdir = (uint8_t *)malloc((size_t)(n_t * TDIR_ENTRY));
        mf->tensors = (hx_tensor *)calloc((size_t)n_t, sizeof *mf->tensors);
        if (!tdir || !mf->tensors) { snprintf(msg, ml, "out of memory (tensor directory)"); goto done; }
        if (!read_at(f, tdir, n_t * TDIR_ENTRY, tdir_off)) { snprintf(msg, ml, "cannot read the tensor directory"); goto done; }
    }
    for (uint64_t i = 0; i < n_t; i++) {
        const uint8_t *e = tdir + i * TDIR_ENTRY;
        hx_tensor *t = &mf->tensors[i];
        const uint8_t *nul = (const uint8_t *)memchr(e, 0, HX_NAME_LEN);
        if (!nul) { snprintf(msg, ml, "tensor entry %llu: name not NUL-terminated", (unsigned long long)i); goto done; }
        size_t nlen = (size_t)(nul - e);
        if (nlen == 0) { snprintf(msg, ml, "tensor entry %llu: empty name", (unsigned long long)i); goto done; }
        for (size_t k = 0; k < nlen; k++)
            if (e[k] >= 0x80) { snprintf(msg, ml, "tensor entry %llu: name is not ASCII", (unsigned long long)i); goto done; }
        memcpy(t->name, e, nlen);
        t->name[nlen] = 0;
        uint32_t dtype = rd32(e + 80), ndim = rd32(e + 84);
        if (!hx_dtype_valid((int)dtype)) { snprintf(msg, ml, "tensor '%s': unknown dtype %u", t->name, dtype); goto done; }
        if (ndim < 1 || ndim > 4) { snprintf(msg, ml, "tensor '%s': ndim %u", t->name, ndim); goto done; }
        uint64_t rows = 1;
        for (uint32_t k = 0; k < 4; k++) {
            uint32_t d = rd32(e + 88 + 4 * k);
            if (k < ndim ? d < 1 : d != 1) { snprintf(msg, ml, "tensor '%s': bad shape entry %u = %u", t->name, k, d); goto done; }
            t->shape[k] = d;
            if (k + 1 < ndim && !mul_ok(rows, d, &rows)) { snprintf(msg, ml, "tensor '%s': shape overflows", t->name); goto done; }
        }
        t->dtype = (int)dtype;
        t->ndim = (int)ndim;
        t->offset = rd64(e + 104);
        t->nbytes = rd64(e + 112);
        size_t rb = hx_row_bytes(t->dtype, t->shape[ndim - 1]);
        uint64_t want;
        if (!rb) { snprintf(msg, ml, "tensor '%s': %lld columns invalid for dtype %s", t->name, (long long)t->shape[ndim - 1], hx_dtype_name(t->dtype)); goto done; }
        if (!mul_ok(rows, rb, &want) || t->nbytes != want) {
            snprintf(msg, ml, "tensor '%s': nbytes %llu does not match its shape and dtype", t->name, (unsigned long long)t->nbytes);
            goto done;
        }
        if (t->offset % TENSOR_ALIGN) { snprintf(msg, ml, "tensor '%s': offset %llu not 64-byte aligned", t->name, (unsigned long long)t->offset); goto done; }
        if (!in_file(t->offset, t->nbytes, size)) { snprintf(msg, ml, "tensor '%s' lies outside the file", t->name); goto done; }
    }

    /* expert directory */
    if (n_e) {
        edir = (uint8_t *)malloc((size_t)(n_e * EDIR_ENTRY));
        if (!edir) { snprintf(msg, ml, "out of memory (expert directory)"); goto done; }
        if (!read_at(f, edir, n_e * EDIR_ENTRY, edir_off)) { snprintf(msg, ml, "cannot read the expert directory"); goto done; }
        if (has_moe) {
            mf->experts = (hx_expert_entry *)calloc((size_t)n_e, sizeof *mf->experts);
            if (!mf->experts) { snprintf(msg, ml, "out of memory (expert directory)"); goto done; }
        }
    }
    for (uint64_t i = 0; i < n_e; i++) {
        const uint8_t *e = edir + i * EDIR_ENTRY;
        uint64_t off = rd64(e), nb = rd64(e + 8);
        uint32_t dt = rd32(e + 16), fl = rd32(e + 20);
        int layer = (int)(i / (uint64_t)E);
        if (off % HX_SLAB_ALIGN || nb % HX_SLAB_ALIGN) { snprintf(msg, ml, "expert entry %llu: offset/size not 4096-aligned", (unsigned long long)i); goto done; }
        if (!in_file(off, nb, size)) { snprintf(msg, ml, "expert entry %llu: slab lies outside the file", (unsigned long long)i); goto done; }
        if (c->layer_kind[layer] == HX_LAYER_MOE) {
            if (!nb) { snprintf(msg, ml, "expert entry %llu: empty slab in MoE layer %d", (unsigned long long)i, layer); goto done; }
            if (dt > HEARTH_Q4) { snprintf(msg, ml, "expert entry %llu: bad slab dtype %u", (unsigned long long)i, dt); goto done; }
            uint64_t need = hx_slab_layout((int)dt, c->d_model, c->expert_ffn_dim, NULL, NULL, NULL);
            if (!need) { snprintf(msg, ml, "expert entry %llu: dims %dx%d invalid for dtype %s", (unsigned long long)i, c->d_model, c->expert_ffn_dim, hx_dtype_name((int)dt)); goto done; }
            if (nb < need) { snprintf(msg, ml, "expert entry %llu: slab of %llu bytes < %llu needed", (unsigned long long)i, (unsigned long long)nb, (unsigned long long)need); goto done; }
        } else if (nb) {
            snprintf(msg, ml, "expert entry %llu: non-empty slab in dense layer %d", (unsigned long long)i, layer);
            goto done;
        }
        if (mf->experts) {
            mf->experts[i].offset = off;
            mf->experts[i].nbytes = nb;
            mf->experts[i].dtype = dt;
            mf->experts[i].flags = fl;
        }
    }

    {
        const uint64_t sec[4][2] = {{0, PREAMBLE_BYTES}, {meta_off, meta_bytes}, {tdir_off, n_t * TDIR_ENTRY},
                                    {edir_off, n_e * EDIR_ENTRY}};
        if (!check_regions(mf, sec, msg, ml)) goto done;
    }

    if (mf->n_tensors > 1) qsort(mf->tensors, (size_t)mf->n_tensors, sizeof *mf->tensors, cmp_tensor_name);
    for (int i = 1; i < mf->n_tensors; i++)
        if (strcmp(mf->tensors[i - 1].name, mf->tensors[i].name) == 0) {
            snprintf(msg, ml, "duplicate tensor '%s'", mf->tensors[i].name);
            goto done;
        }

    if (!check_canonical(mf, msg, ml)) goto done;
    const hx_tensor *rf = hx_mf_tensor(mf, "rope_inv_freq");   /* present iff rope_dim > 0 */

    /* Parameter counts mirror hearth.presets.Shape: active excludes the input
     * embedding table unless it doubles as the LM head. */
    double dense = 0.0, emb = 0.0;
    mf->dense_bytes = 0;
    for (int i = 0; i < mf->n_tensors; i++) {
        const hx_tensor *t = &mf->tensors[i];
        double el = 1.0;
        for (int k = 0; k < t->ndim; k++) el *= (double)t->shape[k];
        mf->dense_bytes += t->nbytes;
        if (strcmp(t->name, "rope_inv_freq") == 0) continue;
        dense += el;
        if (strcmp(t->name, "tok_embd") == 0) emb = el;
    }
    double per_expert = 3.0 * (double)c->d_model * (double)c->expert_ffn_dim;
    mf->params_total = dense + (has_moe ? (double)c->n_moe_layers * E * per_expert : 0.0);
    mf->params_active = dense - (c->tie_embeddings ? 0.0 : emb) + (has_moe ? (double)c->n_moe_layers * c->top_k * per_expert : 0.0);

    if (want_dense) {
        if (!load_dense(m, msg, ml)) goto done;
        if (rf) mf->rope_inv_freq = (const float *)rf->data;
    }
    ok = 1;
done:
    free(meta);
    free(tdir);
    free(edir);
    return ok;
}

hx_modelfile *hx_modelfile_open(const char *path, int load_dense, char *err, size_t errlen) {
    if (err && errlen) err[0] = 0;
    if (!path || !*path) return (hx_modelfile *)hx_fail(err, errlen, "hx_modelfile_open: empty path");
    mf_impl *m = (mf_impl *)calloc(1, sizeof *m);
    if (!m) return (hx_modelfile *)hx_fail(err, errlen, "%s: out of memory", path);
    hx_modelfile *mf = &m->pub;
    if (strlen(path) >= sizeof mf->path) {
        free(m);
        return (hx_modelfile *)hx_fail(err, errlen, "%.200s...: path too long", path);
    }
    strcpy(mf->path, path);
    hx_file *f = hx_file_open(path, HX_FILE_READ, err, errlen);
    if (!f) { free(m); return NULL; }
    char msg[512] = "";
    int64_t sz = hx_file_size(f);
    int ok = 0;
    if (sz < 0) snprintf(msg, sizeof msg, "cannot determine the file size");
    else {
        mf->file_size = (uint64_t)sz;
        ok = parse(m, f, load_dense, msg, sizeof msg);
    }
    hx_file_close(f);
    if (!ok) {
        hx_modelfile_close(mf);
        return (hx_modelfile *)hx_fail(err, errlen, "%s: %s", path, msg[0] ? msg : "invalid container");
    }
    return mf;
}

void hx_modelfile_close(hx_modelfile *mf) {
    if (!mf) return;
    mf_impl *m = (mf_impl *)mf;
    if (mf->dense_arena) hx_free_large(mf->dense_arena, m->arena_alloc);
    free(mf->tensors);
    free(mf->experts);
    free(mf->chat_template);
    free(m);
}

/* ------------------------------------------------------------- lookup */

const hx_tensor *hx_mf_tensor(const hx_modelfile *mf, const char *name) {
    if (!mf || !name) return NULL;
    int lo = 0, hi = mf->n_tensors - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int c = strcmp(mf->tensors[mid].name, name);
        if (c == 0) return &mf->tensors[mid];
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return NULL;
}

const hx_tensor *hx_mf_layer_tensor(const hx_modelfile *mf, int layer, const char *suffix) {
    char name[HX_NAME_LEN + 16];
    if (!mf || !suffix || layer < 0) return NULL;
    int n = snprintf(name, sizeof name, "blk.%d.%s", layer, suffix);
    if (n < 0 || n >= HX_NAME_LEN) return NULL;
    return hx_mf_tensor(mf, name);
}

const hx_expert_entry *hx_mf_expert(const hx_modelfile *mf, int layer, int expert) {
    if (!mf || !mf->experts || layer < 0 || layer >= mf->cfg.n_layers || expert < 0 || expert >= mf->cfg.n_experts)
        return NULL;
    return &mf->experts[(size_t)layer * (size_t)mf->cfg.n_experts + (size_t)expert];
}
