/*
 * hx_quant.h — weight formats, quantizers and the hot kernels.
 *
 * Normative semantics: docs/NUMERICS.md. Every ISA variant (scalar, AVX2,
 * AVX-512) of every kernel must be bit-identical to the scalar definition.
 *
 * Files:
 *   quant.c        formats, quantizers, scalar kernels, ISA dispatch, elementwise ops
 *   quant_avx2.c   AVX2+FMA+F16C kernels   (compiled with -mavx2 -mfma -mf16c / MSVC /arch:AVX2)
 *   quant_avx512.c AVX-512 F/BW/VL/VNNI    (compiled with -mavx512f -mavx512bw -mavx512vl -mavx512vnni / MSVC /arch:AVX512)
 * (FMA instructions may be used only where they cannot change results, e.g. never
 *  for canonical float accumulation.)
 */
#ifndef HX_QUANT_H
#define HX_QUANT_H

#include "hx_platform.h"
#include "../include/hearth.h"

#define HX_QK 64  /* block size for Q8/Q4 weights and Q8 activations */

typedef struct { uint16_t d; int8_t  q[64]; } hx_block_q8;  /* 66 bytes */
typedef struct { uint16_t d; uint8_t qs[32]; } hx_block_q4; /* 34 bytes; byte j: lo nibble = elem j, hi = elem j+32; stored q+8 */
typedef struct { float d; int32_t sum; int8_t q[64]; } hx_act_q8; /* 72 bytes, activation block */

_Static_assert(sizeof(hx_block_q8) == 66, "hx_block_q8 must be 66 bytes");
_Static_assert(sizeof(hx_block_q4) == 34, "hx_block_q4 must be 34 bytes");
_Static_assert(sizeof(hx_act_q8) == 72, "hx_act_q8 must be 72 bytes");

/* ---- format helpers ---------------------------------------------------- */
int         hx_dtype_valid(int dtype);
int         hx_dtype_is_quant(int dtype);           /* Q8 or Q4 */
size_t      hx_row_bytes(int dtype, int64_t n);      /* bytes per row of n elements; 0 if invalid (e.g. n%64 for quant) */
const char *hx_dtype_name(int dtype);

/* ---- quantize / dequantize (authoritative, scalar, deterministic) ------- */
void hx_quantize_row(int dtype, const float *src, void *dst, int64_t n);
void hx_dequantize_row(int dtype, const void *src, float *dst, int64_t n);

/* ---- activations ---------------------------------------------------------
 * An "activation" is an input vector prepared for a given weight dtype:
 * for Q8/Q4 weights it is n/64 hx_act_q8 blocks (NUMERICS §2), otherwise the f32
 * vector itself. hx_act_bytes gives the buffer size needed for one vector. */
size_t hx_act_bytes(int wdtype, int64_t n);
/* Prepares x into act (act may alias nothing). Returns pointer to use as activation
 * (act for quant dtypes, or x itself for float dtypes — no copy). */
const void *hx_act_prepare(int wdtype, const float *x, int64_t n, void *act);

/* ---- kernels --------------------------------------------------------------
 * Row-range kernels: compute rows [r0, r1) of Y = W · X for T activations.
 *   W: n_rows x n_cols in wdtype (row-major, hx_row_bytes(wdtype, n_cols) per row)
 *   act: T prepared activations, consecutive, each hx_act_bytes(wdtype, n_cols) bytes
 *        (for float dtypes: T f32 vectors of n_cols, consecutive)
 *   Y: Y[t*ldy + r]
 * Must equal, for every (t, r), the T=1 result (NUMERICS §3). */
typedef void (*hx_matmul_fn)(const void *W, int64_t n_cols, const void *act, int T,
                             float *Y, int64_t ldy, int64_t r0, int64_t r1);

typedef struct hx_kernels {
    int isa;                       /* HEARTH_ISA_* */
    hx_matmul_fn matmul[7];        /* indexed by dtype; NULL if unsupported (I32/U8) */
    void (*act_quantize_q8)(const float *x, hx_act_q8 *out, int64_t n);   /* NUMERICS §2 */
} hx_kernels;

/* Best kernels for this CPU (respects env HEARTH_ISA=scalar|avx2|avx512). */
const hx_kernels *hx_kernels_get(void);
/* Kernels for a specific ISA; NULL if the CPU cannot run it. */
const hx_kernels *hx_kernels_for(int isa);

/* Per-ISA tables, defined in their own translation units. */
extern const hx_kernels hx_kernels_scalar;
const hx_kernels *hx_kernels_avx2_table(void);    /* NULL if not compiled in */
const hx_kernels *hx_kernels_avx512_table(void);  /* NULL if not compiled in */

/* ---- canonical float helpers (NUMERICS §1, §4) — scalar, used everywhere --- */
float hx_sum16(const float *a, int64_t n);
float hx_dot16(const float *a, const float *b, int64_t n);
void  hx_rmsnorm(float *y, const float *x, const float *w, int64_t n, float eps);  /* y may alias x */
void  hx_softmax(float *x, int64_t n);                                           /* in place */
void  hx_swiglu(float *out, const float *g, const float *u, int64_t n);           /* out may alias g */
float hx_sigmoid(float x);
/* Rotate the first rope_dim dims of a head vector in place (NUMERICS §4). */
void  hx_rope(float *x, int rope_dim, int style, int pos, const float *inv_freq, float attn_factor);
/* y += a * x (elementwise, y_i = y_i + a*x_i). */
void  hx_axpy(float *y, float a, const float *x, int64_t n);

#endif /* HX_QUANT_H */
