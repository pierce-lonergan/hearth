/*
 * hx_modelfile.h — reading and validating a .hearth container (docs/FORMAT.md).
 *
 * hx_modelfile_open parses the preamble, metadata, tensor and expert directories,
 * validates every offset/size against the file size and alignment rules
 * (INV-SAFE: a malformed file must produce an error, never a crash or an
 * out-of-bounds read), and — if load_dense — reads all dense tensors into one
 * aligned RAM arena so that hx_tensor.data is valid.
 */
#ifndef HX_MODELFILE_H
#define HX_MODELFILE_H

#include "hx_platform.h"

#define HX_MAGIC        0x48545248u   /* "HRTH" */
#define HX_FORMAT_VER   1u
#define HX_SLAB_ALIGN   4096u
#define HX_NAME_LEN     80
#define HX_MAX_LAYERS   512
#define HX_MAX_EOS      8

enum { HX_ATTN_GQA = 0, HX_ATTN_MLA = 1 };
enum { HX_QKNORM_NONE = 0, HX_QKNORM_HEAD = 1, HX_QKNORM_FULL = 2 };
enum { HX_ROPE_NEOX = 0, HX_ROPE_GPTJ = 1 };
enum { HX_SCORE_SOFTMAX = 0, HX_SCORE_SIGMOID = 1 };
enum { HX_LAYER_DENSE = 0, HX_LAYER_MOE = 1 };

/* Hyper-parameters with FORMAT.md §3.1 defaults applied. */
typedef struct hx_config {
    char arch[32];
    int n_layers, d_model, vocab_size, max_seq;
    float norm_eps;
    int attn_kind, n_heads, n_kv_heads, head_dim, qk_norm, qkv_bias;
    int q_lora_rank, kv_lora_rank, qk_nope_dim, qk_rope_dim, v_head_dim;
    int rope_dim, rope_style;
    float rope_attn_factor, attn_scale;
    int dense_ffn_dim;
    int n_experts, top_k, expert_ffn_dim, shared_ffn_dim, shared_gate;
    int score_fn, score_bias, n_group, topk_group, norm_topk_prob;
    float routed_scale, emb_scale, residual_scale, logit_scale;
    int tie_embeddings;
    uint8_t layer_kind[HX_MAX_LAYERS];
    int n_moe_layers;
    int bos_id;                      /* -1 if none */
    int n_eos; int eos_ids[HX_MAX_EOS];
    char tokenizer[260];             /* relative path, may be empty */
    char source[128];
} hx_config;

typedef struct hx_tensor {
    char name[HX_NAME_LEN];
    int dtype, ndim;
    int64_t shape[4];
    uint64_t offset, nbytes;         /* in file */
    void *data;                      /* resident pointer (64-byte aligned) or NULL */
} hx_tensor;

typedef struct hx_expert_entry {
    uint64_t offset, nbytes;
    uint32_t dtype, flags;
} hx_expert_entry;

typedef struct hx_modelfile {
    char path[1024];
    uint64_t file_size;
    hx_config cfg;
    int n_tensors;
    hx_tensor *tensors;              /* sorted by name for binary search */
    hx_expert_entry *experts;        /* [n_layers * n_experts], NULL if none */
    uint64_t expert_region_bytes;    /* sum of distinct slab bytes (non-aliased) */
    uint64_t slab_bytes_max;
    void *dense_arena; uint64_t dense_bytes;
    const float *rope_inv_freq;      /* points into dense arena */
    double params_total, params_active;
    char *chat_template;             /* heap, may be NULL */
} hx_modelfile;

hx_modelfile *hx_modelfile_open(const char *path, int load_dense, char *err, size_t errlen);
void          hx_modelfile_close(hx_modelfile *mf);

/* Lookup by exact name; NULL if absent. */
const hx_tensor *hx_mf_tensor(const hx_modelfile *mf, const char *name);
/* Lookup "blk.%d.<suffix>". */
const hx_tensor *hx_mf_layer_tensor(const hx_modelfile *mf, int layer, const char *suffix);
const hx_expert_entry *hx_mf_expert(const hx_modelfile *mf, int layer, int expert);

/* Slab layout (FORMAT.md §5). Returns padded slab size; writes the three offsets. */
uint64_t hx_slab_layout(int dtype, int64_t D, int64_t F, uint64_t *off_gate, uint64_t *off_up, uint64_t *off_down);

typedef struct hx_slab_view { const void *gate, *up, *down; int dtype; } hx_slab_view;
void hx_slab_view_make(const hx_modelfile *mf, const void *slab, int dtype, hx_slab_view *out);

#endif /* HX_MODELFILE_H */
