/*
 * quant.c — weight formats, the authoritative quantizers, the scalar reference
 * kernels, runtime ISA dispatch and the canonical float helpers.
 *
 * The scalar code here IS the numeric definition (docs/NUMERICS.md); the AVX2 and
 * AVX-512 kernels in quant_avx2.c / quant_avx512.c must reproduce it bit for bit.
 * This file is compiled without ISA flags, so the compiler cannot emit FMA.
 */
#include "hx_quant.h"

#include <math.h>
#include <string.h>

/* ------------------------------------------------------------ small helpers */

/* Exact f16 -> f32, identical to F16C vcvtph2ps (signalling NaNs come back quiet). */
HX_INLINE float q_f16(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (uint32_t)(h >> 10) & 0x1fu, m = (uint32_t)h & 0x3ffu, bits;
    float r;
    if (e == 0x1fu) {
        bits = sign | 0x7f800000u | (m << 13) | (m ? 0x00400000u : 0u);
    } else if (e != 0) {
        bits = sign | ((e + 112u) << 23) | (m << 13);
    } else if (m == 0) {
        bits = sign;
    } else {
        float f = (float)m * 5.9604644775390625e-8f;   /* m * 2^-24, exact */
        uint32_t fb;
        memcpy(&fb, &f, 4);
        bits = sign | fb;
    }
    memcpy(&r, &bits, 4);
    return r;
}

HX_INLINE float q_bf16(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

/* Round half to even for |v| < 2^22 (callers clamp first); equals nearbyintf()
 * in the default rounding mode. */
HX_INLINE float q_rint(float v) {
    const float magic = 12582912.0f;   /* 1.5 * 2^23 */
    return (v + magic) - magic;
}

HX_INLINE float q_clean(float v) { return v == v ? v : 0.0f; }   /* NaN -> 0 */

/* Kernel outputs that are NaN are stored as this one quiet NaN. Which NaN an IEEE
 * add/mul propagates depends on operand order, and compilers commute the operands
 * of SIMD adds, so raw NaN bits would differ between ISAs, T and row ranges.
 * quant_avx2.c and quant_avx512.c use the same constant. */
#define HX_CANON_NAN 0x7fc00000u

HX_INLINE float q_canon(float v) {
    if (v != v) {
        uint32_t b = HX_CANON_NAN;
        memcpy(&v, &b, 4);
    }
    return v;
}

/* Block scales are stored as f16; overflow saturates to the largest finite f16. */
static uint16_t q_scale_f16(float d) {
    uint16_t h = hx_f32_to_f16(d);
    if ((h & 0x7c00u) == 0x7c00u) h = (uint16_t)((h & 0x8000u) | 0x7bffu);
    return h;
}

/* ------------------------------------------------------------ format helpers */

int hx_dtype_valid(int dtype) { return dtype >= HEARTH_F32 && dtype <= HEARTH_U8; }

int hx_dtype_is_quant(int dtype) { return dtype == HEARTH_Q8 || dtype == HEARTH_Q4; }

size_t hx_row_bytes(int dtype, int64_t n) {
    if (n <= 0 || n > ((int64_t)1 << 40)) return 0;   /* the limit keeps byte counts far from overflow */
    switch (dtype) {
    case HEARTH_F32: case HEARTH_I32: return (size_t)n * 4;
    case HEARTH_F16: case HEARTH_BF16: return (size_t)n * 2;
    case HEARTH_U8: return (size_t)n;
    case HEARTH_Q8: return (n % HX_QK) ? 0 : (size_t)(n / HX_QK) * sizeof(hx_block_q8);
    case HEARTH_Q4: return (n % HX_QK) ? 0 : (size_t)(n / HX_QK) * sizeof(hx_block_q4);
    default: return 0;
    }
}

const char *hx_dtype_name(int dtype) {
    static const char *names[] = {"f32", "f16", "bf16", "q8", "q4", "i32", "u8"};
    return hx_dtype_valid(dtype) ? names[dtype] : "invalid";
}

/* ------------------------------------------------------------ quantizers */

/* Q8 (NUMERICS §6): d = amax/127 stored as f16, q = rint(x / f16(d)) in [-127, 127].
 * NaN inputs count as 0; a scale that underflows to 0 gives q = 0. */
static void quantize_q8(const float *x, hx_block_q8 *out, int64_t nb) {
    for (int64_t b = 0; b < nb; b++, x += HX_QK) {
        float amax = 0.0f, d;
        for (int i = 0; i < HX_QK; i++) {
            float a = fabsf(q_clean(x[i]));
            if (a > amax) amax = a;
        }
        out[b].d = q_scale_f16(amax / 127.0f);
        d = q_f16(out[b].d);
        for (int i = 0; i < HX_QK; i++) {
            float v = d != 0.0f ? q_clean(x[i]) / d : 0.0f;
            v = v < -127.0f ? -127.0f : (v > 127.0f ? 127.0f : v);
            out[b].q[i] = (int8_t)q_rint(v);
        }
    }
}

/*
 * Q4 scale search (authoritative, deterministic, scalar only).
 *
 * Per 64-block (NaN inputs count as 0; an all-zero block stores d = 0, q = 0):
 *  1. xm = the element of largest magnitude (the first one on ties).
 *  2. For each divisor m of q4_div[], in order, the candidate scale d = f16(xm / m)
 *     (f16 rounding saturates at +-65504) is scored exactly as it would be stored:
 *     q_i = clamp(rint(x_i * (1/d)), -8, 7) and
 *     err = sum(x^2) - 2*d*sum(x*q) + d^2*sum(q^2), in double (x*q is exact,
 *     summed in 4 fixed lanes). A scale already scored is skipped (it would score
 *     the same). The lowest error wins; ties keep the earlier candidate.
 *  3. The winner is refined up to 3 times by its least-squares scale
 *     d' = f16(sum(x*q) / sum(q^2)), scored the same way, while that strictly
 *     lowers the error.
 *  4. Stored: f16(d) and q_i + 8 as nibbles (FORMAT.md §6).
 * The divisors map xm to about -8 (the wider negative half of [-8, 7]) with
 * clipping variants down to -12, or to about +7. m = +-7 gives the symmetric
 * absmax scale amax/7, so the result is never worse than that baseline.
 * Measured on N(0,1), Student-t(4) and Laplace data, refitting every candidate
 * instead of only the winner changed RMSE by < 0.01% at ~1.7x the cost.
 */
static const float q4_div[] = {
    -8.0f, -7.75f, -8.25f, -7.5f, -8.5f, -7.25f, -8.75f, -7.0f, -9.0f, -9.25f,
    -9.5f, -9.75f, -10.0f, -10.5f, -11.0f, -12.0f, 7.0f, 7.5f, 7.75f, 8.0f,
};

/* q_i = clamp(rint(x_i * id), -8, 7) as a float (an exact small integer). */
HX_INLINE float q4_q(float x, float id) {
    float v = x * id;
    v = v < -8.0f ? -8.0f : v;
    v = v > 7.0f ? 7.0f : v;
    return q_rint(v);
}

/* Scores scale bits h on block x (xd = x as double): err = sum(x^2) - 2*d*sum(x*q)
 * + d^2*sum(q^2) in double, with x*q exact and summed in 4 fixed lanes. */
static double q4_score(const float *x, const double *xd, double sx2, uint16_t h, double *sxq, double *sqq) {
    const float d = q_f16(h);
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const double dd = (double)d;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s;
    int32_t qq = 0;
    for (int i = 0; i < HX_QK; i += 4) {
        int q0 = (int)q4_q(x[i], id), q1 = (int)q4_q(x[i + 1], id);
        int q2 = (int)q4_q(x[i + 2], id), q3 = (int)q4_q(x[i + 3], id);
        s0 = s0 + xd[i] * q0;
        s1 = s1 + xd[i + 1] * q1;
        s2 = s2 + xd[i + 2] * q2;
        s3 = s3 + xd[i + 3] * q3;
        qq += q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
    }
    s = (s0 + s1) + (s2 + s3);
    *sxq = s;
    *sqq = (double)qq;
    return (sx2 - (2.0 * dd) * s) + (dd * dd) * (double)qq;
}

static void quantize_q4(const float *src, hx_block_q4 *out, int64_t nb) {
    enum { NDIV = sizeof(q4_div) / sizeof(q4_div[0]), REFITS = 3 };
    float x[HX_QK];
    double xd[HX_QK];
    for (int64_t b = 0; b < nb; b++, src += HX_QK) {
        float xm = 0.0f, amax = 0.0f, d, id;
        uint16_t best = 0, seen[NDIV];
        int nseen = 0;
        double best_err = 0.0, best_sxq = 0.0, best_sqq = 0.0, sx2 = 0.0;
        int8_t q[HX_QK];
        for (int i = 0; i < HX_QK; i++) {
            float a;
            x[i] = q_clean(src[i]);
            a = fabsf(x[i]);
            if (a > amax) { amax = a; xm = x[i]; }
            xd[i] = (double)x[i];
            sx2 = sx2 + xd[i] * xd[i];
        }
        if (amax > 0.0f) {
            for (int k = 0; k < NDIV; k++) {
                double sxq, sqq, err;
                uint16_t h = q_scale_f16(xm / q4_div[k]);
                int dup = 0;
                for (int j = 0; j < nseen; j++) dup |= seen[j] == h;
                if (dup) continue;
                seen[nseen++] = h;
                err = q4_score(x, xd, sx2, h, &sxq, &sqq);
                if (k == 0 || err < best_err) {
                    best = h;
                    best_err = err;
                    best_sxq = sxq;
                    best_sqq = sqq;
                }
            }
            for (int it = 0; it < REFITS && best_sqq > 0.0; it++) {
                double sxq, sqq, err;
                uint16_t h = q_scale_f16((float)(best_sxq / best_sqq));
                if (h == best) break;
                err = q4_score(x, xd, sx2, h, &sxq, &sqq);
                if (!(err < best_err)) break;
                best = h;
                best_err = err;
                best_sxq = sxq;
                best_sqq = sqq;
            }
        }
        out[b].d = best;
        d = q_f16(best);
        id = d != 0.0f ? 1.0f / d : 0.0f;
        for (int i = 0; i < HX_QK; i++) q[i] = (int8_t)q4_q(x[i], id);
        for (int j = 0; j < 32; j++)
            out[b].qs[j] = (uint8_t)((q[j] + 8) | ((q[j + 32] + 8) << 4));
    }
}

void hx_quantize_row(int dtype, const float *src, void *dst, int64_t n) {
    if (n <= 0) return;
    switch (dtype) {
    case HEARTH_F32:
        memcpy(dst, src, (size_t)n * sizeof(float));
        break;
    case HEARTH_F16: {
        uint16_t *o = (uint16_t *)dst;
        for (int64_t i = 0; i < n; i++) o[i] = hx_f32_to_f16(src[i]);
        break;
    }
    case HEARTH_BF16: {
        uint16_t *o = (uint16_t *)dst;
        for (int64_t i = 0; i < n; i++) o[i] = hx_f32_to_bf16(src[i]);
        break;
    }
    case HEARTH_Q8:
        if (n % HX_QK == 0) quantize_q8(src, (hx_block_q8 *)dst, n / HX_QK);
        break;
    case HEARTH_Q4:
        if (n % HX_QK == 0) quantize_q4(src, (hx_block_q4 *)dst, n / HX_QK);
        break;
    default:
        break;   /* I32 / U8 are not produced from floats */
    }
}

void hx_dequantize_row(int dtype, const void *src, float *dst, int64_t n) {
    if (n <= 0) return;
    switch (dtype) {
    case HEARTH_F32:
        memcpy(dst, src, (size_t)n * sizeof(float));
        break;
    case HEARTH_F16: {
        const uint16_t *s = (const uint16_t *)src;
        for (int64_t i = 0; i < n; i++) dst[i] = q_f16(s[i]);
        break;
    }
    case HEARTH_BF16: {
        const uint16_t *s = (const uint16_t *)src;
        for (int64_t i = 0; i < n; i++) dst[i] = q_bf16(s[i]);
        break;
    }
    case HEARTH_Q8: {
        const hx_block_q8 *b = (const hx_block_q8 *)src;
        if (n % HX_QK) return;
        for (int64_t k = 0; k < n / HX_QK; k++) {
            float d = q_f16(b[k].d);
            for (int i = 0; i < HX_QK; i++) dst[k * HX_QK + i] = d * (float)b[k].q[i];
        }
        break;
    }
    case HEARTH_Q4: {
        const hx_block_q4 *b = (const hx_block_q4 *)src;
        if (n % HX_QK) return;
        for (int64_t k = 0; k < n / HX_QK; k++) {
            float d = q_f16(b[k].d);
            for (int j = 0; j < 32; j++) {
                dst[k * HX_QK + j] = d * (float)((int)(b[k].qs[j] & 0x0f) - 8);
                dst[k * HX_QK + j + 32] = d * (float)((int)(b[k].qs[j] >> 4) - 8);
            }
        }
        break;
    }
    case HEARTH_I32: {
        const int32_t *s = (const int32_t *)src;
        for (int64_t i = 0; i < n; i++) dst[i] = (float)s[i];
        break;
    }
    case HEARTH_U8: {
        const uint8_t *s = (const uint8_t *)src;
        for (int64_t i = 0; i < n; i++) dst[i] = (float)s[i];
        break;
    }
    default:
        break;
    }
}

/* ------------------------------------------------------------ activations */

size_t hx_act_bytes(int wdtype, int64_t n) {
    if (n <= 0) return 0;
    if (hx_dtype_is_quant(wdtype)) return (n % HX_QK) ? 0 : (size_t)(n / HX_QK) * sizeof(hx_act_q8);
    return (size_t)n * sizeof(float);
}

const void *hx_act_prepare(int wdtype, const float *x, int64_t n, void *act) {
    if (!hx_dtype_is_quant(wdtype)) return x;
    hx_kernels_get()->act_quantize_q8(x, (hx_act_q8 *)act, n);
    return act;
}

/* NUMERICS §2. Degenerate inputs, identically on every ISA: NaN products
 * (NaN input, 0*inf) become 0 and +-inf saturates to +-127, so |q| <= 127 always. */
static void act_q8_scalar(const float *x, hx_act_q8 *out, int64_t n) {
    for (int64_t b = 0; b < n / HX_QK; b++, x += HX_QK) {
        float amax = 0.0f, d, id;
        int32_t sum = 0;
        for (int i = 0; i < HX_QK; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        d = amax / 127.0f;
        id = d != 0.0f ? 1.0f / d : 0.0f;
        for (int i = 0; i < HX_QK; i++) {
            float v = x[i] * id;
            int q;
            if (v != v) v = 0.0f;
            v = v > 127.0f ? 127.0f : v;
            v = v < -127.0f ? -127.0f : v;
            q = (int)q_rint(v);
            out[b].q[i] = (int8_t)q;
            sum += q;
        }
        out[b].d = d;
        out[b].sum = sum;
    }
}

/* ------------------------------------------------------------ scalar kernels */

static float dot16_conv(int dt, const void *w, const float *x, int64_t n) {
    float L[16] = {0};
    int64_t i;
    for (i = 0; i < n; i++) {
        float wi;
        if (dt == HEARTH_F32) wi = ((const float *)w)[i];
        else if (dt == HEARTH_F16) wi = q_f16(((const uint16_t *)w)[i]);
        else wi = q_bf16(((const uint16_t *)w)[i]);
        L[i & 15] = L[i & 15] + wi * x[i];
    }
    for (int s = 8; s >= 1; s >>= 1)
        for (int j = 0; j < s; j++) L[j] = L[j] + L[j + s];
    return L[0];
}

static void mm_float_scalar(int dt, const void *W, int64_t n, const void *act, int T,
                            float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    const size_t rb = hx_row_bytes(dt, n);
    const float *X = (const float *)act;
    for (int64_t r = r0; r < r1; r++) {
        const void *w = (const uint8_t *)W + (size_t)r * rb;
        for (int t = 0; t < T; t++)
            Y[t * ldy + r] = q_canon(dot16_conv(dt, w, X + (size_t)t * n, n));
    }
}

static void mm_f32_scalar(const void *W, int64_t n, const void *act, int T, float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    mm_float_scalar(HEARTH_F32, W, n, act, T, Y, ldy, r0, r1);
}
static void mm_f16_scalar(const void *W, int64_t n, const void *act, int T, float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    mm_float_scalar(HEARTH_F16, W, n, act, T, Y, ldy, r0, r1);
}
static void mm_bf16_scalar(const void *W, int64_t n, const void *act, int T, float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    mm_float_scalar(HEARTH_BF16, W, n, act, T, Y, ldy, r0, r1);
}

static void mm_q8_scalar(const void *W, int64_t n, const void *act, int T,
                         float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    const int64_t nb = n / HX_QK;
    const hx_act_q8 *A = (const hx_act_q8 *)act;
    for (int64_t r = r0; r < r1; r++) {
        const hx_block_q8 *w = (const hx_block_q8 *)W + (size_t)r * nb;
        for (int t = 0; t < T; t++) {
            const hx_act_q8 *a = A + (size_t)t * nb;
            float acc = 0.0f;
            for (int64_t g = 0; g < nb; g++) {
                int32_t isum = 0;
                for (int i = 0; i < HX_QK; i++) isum += (int32_t)w[g].q[i] * (int32_t)a[g].q[i];
                acc = acc + (float)isum * (q_f16(w[g].d) * a[g].d);
            }
            Y[t * ldy + r] = q_canon(acc);
        }
    }
}

static void mm_q4_scalar(const void *W, int64_t n, const void *act, int T,
                         float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    const int64_t nb = n / HX_QK;
    const hx_act_q8 *A = (const hx_act_q8 *)act;
    for (int64_t r = r0; r < r1; r++) {
        const hx_block_q4 *w = (const hx_block_q4 *)W + (size_t)r * nb;
        for (int t = 0; t < T; t++) {
            const hx_act_q8 *a = A + (size_t)t * nb;
            float acc = 0.0f;
            for (int64_t g = 0; g < nb; g++) {
                int32_t isum = 0;
                for (int j = 0; j < 32; j++) {
                    isum += ((int32_t)(w[g].qs[j] & 0x0f) - 8) * (int32_t)a[g].q[j];
                    isum += ((int32_t)(w[g].qs[j] >> 4) - 8) * (int32_t)a[g].q[j + 32];
                }
                acc = acc + (float)isum * (q_f16(w[g].d) * a[g].d);
            }
            Y[t * ldy + r] = q_canon(acc);
        }
    }
}

const hx_kernels hx_kernels_scalar = {
    HEARTH_ISA_SCALAR,
    {mm_f32_scalar, mm_f16_scalar, mm_bf16_scalar, mm_q8_scalar, mm_q4_scalar, NULL, NULL},
    act_q8_scalar,
};

/* ------------------------------------------------------------ dispatch */

/* The ISA translation units are only built for x86-64. */
#if !defined(HX_ARCH_X86_64)
const hx_kernels *hx_kernels_avx2_table(void) { return NULL; }
const hx_kernels *hx_kernels_avx512_table(void) { return NULL; }
#endif

static int isa_runnable(int isa) {
    const hx_cpu *c;
    if (isa == HEARTH_ISA_SCALAR) return 1;
    c = hx_cpu_features();
    if (isa == HEARTH_ISA_AVX2)
        return hx_kernels_avx2_table() != NULL && c->avx2 && c->fma && c->f16c;
    if (isa == HEARTH_ISA_AVX512)
        return hx_kernels_avx512_table() != NULL && c->avx2 && c->fma && c->f16c &&
               c->avx512f && c->avx512bw && c->avx512vl && c->avx512vnni;
    return 0;
}

const hx_kernels *hx_kernels_for(int isa) {
    if (!isa_runnable(isa)) return NULL;
    switch (isa) {
    case HEARTH_ISA_SCALAR: return &hx_kernels_scalar;
    case HEARTH_ISA_AVX2: return hx_kernels_avx2_table();
    case HEARTH_ISA_AVX512: return hx_kernels_avx512_table();
    default: return NULL;
    }
}

int hearth_cpu_isa(void) {
    if (isa_runnable(HEARTH_ISA_AVX512)) return HEARTH_ISA_AVX512;
    if (isa_runnable(HEARTH_ISA_AVX2)) return HEARTH_ISA_AVX2;
    return HEARTH_ISA_SCALAR;
}

static int q_streq_ci(const char *a, const char *b) {
    for (; *a && *b; a++, b++) {
        char ca = (*a >= 'A' && *a <= 'Z') ? (char)(*a - 'A' + 'a') : *a;
        if (ca != *b) return 0;
    }
    return *a == 0 && *b == 0;
}

const hx_kernels *hx_kernels_get(void) {
    static _Atomic(const hx_kernels *) cached = NULL;
    const hx_kernels *k = atomic_load_explicit(&cached, memory_order_acquire);
    if (k) return k;
    {
        const char *env = hx_env_str("HEARTH_ISA");
        int want = HEARTH_ISA_AUTO, best = hearth_cpu_isa();
        if (env && *env) {
            if (q_streq_ci(env, "scalar")) want = HEARTH_ISA_SCALAR;
            else if (q_streq_ci(env, "avx2")) want = HEARTH_ISA_AVX2;
            else if (q_streq_ci(env, "avx512")) want = HEARTH_ISA_AVX512;
            else if (!q_streq_ci(env, "auto"))
                hx_log(HX_LOG_WARN, "HEARTH_ISA=%s not recognised (scalar|avx2|avx512); using auto", env);
        }
        if (want != HEARTH_ISA_AUTO && !isa_runnable(want)) {
            hx_log(HX_LOG_WARN, "HEARTH_ISA=%s is not supported on this CPU/build; falling back", env);
            want = HEARTH_ISA_AUTO;
        }
        k = hx_kernels_for(want == HEARTH_ISA_AUTO ? best : want);
        atomic_store_explicit(&cached, k, memory_order_release);
    }
    return k;
}

/* ------------------------------------------------------------ public API */

HEARTH_API const char *hearth_version(void) { return "0.1.0"; }

HEARTH_API size_t hearth_row_bytes(int dtype, int64_t n_cols) { return hx_row_bytes(dtype, n_cols); }

typedef struct {
    int dtype;
    const float *src;
    uint8_t *dst;
    int64_t r0, r1, n_cols;
    size_t rb;
} q_job;

static void *q_job_run(void *p) {
    const q_job *j = (const q_job *)p;
    for (int64_t r = j->r0; r < j->r1; r++)
        hx_quantize_row(j->dtype, j->src + (size_t)r * (size_t)j->n_cols, j->dst + (size_t)r * j->rb, j->n_cols);
    return NULL;
}

#define Q_MAX_THREADS 256

HEARTH_API int hearth_quantize(int dtype, const float *src, int64_t n_rows, int64_t n_cols, void *dst, int n_threads) {
    q_job jobs[Q_MAX_THREADS];
    hx_thread *th[Q_MAX_THREADS];
    size_t rb = hx_row_bytes(dtype, n_cols);
    int64_t nt;
    if (dtype < HEARTH_F32 || dtype > HEARTH_Q4 || rb == 0 || n_rows < 0) return -1;
    if (n_rows == 0) return 0;
    if (!src || !dst) return -1;
    nt = n_threads > 0 ? n_threads : hx_num_physical_cores();
    if (nt < 1) nt = 1;
    if (nt > Q_MAX_THREADS) nt = Q_MAX_THREADS;
    if (nt > n_rows) nt = n_rows;
    {   /* at least ~64K elements per thread */
        int64_t by_work = (n_rows * n_cols + 65535) / 65536;
        if (nt > by_work) nt = by_work < 1 ? 1 : by_work;
    }
    for (int64_t t = 0; t < nt; t++) {
        int64_t q = n_rows / nt, rem = n_rows % nt;
        jobs[t].dtype = dtype;
        jobs[t].src = src;
        jobs[t].dst = (uint8_t *)dst;
        jobs[t].n_cols = n_cols;
        jobs[t].rb = rb;
        jobs[t].r0 = t * q + (t < rem ? t : rem);
        jobs[t].r1 = jobs[t].r0 + q + (t < rem ? 1 : 0);
        th[t] = NULL;
    }
    for (int64_t t = 1; t < nt; t++)
        if (hx_thread_create(&th[t], q_job_run, &jobs[t]) != 0) th[t] = NULL;
    q_job_run(&jobs[0]);
    for (int64_t t = 1; t < nt; t++) {
        if (th[t]) hx_thread_join(th[t]);
        else q_job_run(&jobs[t]);   /* thread creation failed: do it here */
    }
    return 0;
}

HEARTH_API int hearth_dequantize(int dtype, const void *src, int64_t n_rows, int64_t n_cols, float *dst) {
    size_t rb = hx_row_bytes(dtype, n_cols);
    if (!hx_dtype_valid(dtype) || rb == 0 || n_rows < 0) return -1;
    if (n_rows == 0) return 0;
    if (!src || !dst) return -1;
    for (int64_t r = 0; r < n_rows; r++)
        hx_dequantize_row(dtype, (const uint8_t *)src + (size_t)r * rb, dst + (size_t)r * (size_t)n_cols, n_cols);
    return 0;
}

/* Returns 0 ok, -1 bad arguments, -2 ISA not runnable on this CPU, -3 out of memory. */
HEARTH_API int hearth_matmul(int dtype, const void *W, int64_t n_rows, int64_t n_cols,
                             const float *X, int T, float *Y, int isa) {
    const hx_kernels *k;
    hx_matmul_fn fn;
    size_t ab;
    void *buf = NULL;
    const void *act = X;
    if (isa < HEARTH_ISA_AUTO || isa > HEARTH_ISA_AVX512) return -1;
    if (!hx_dtype_valid(dtype) || n_rows < 0 || T < 0 || hx_row_bytes(dtype, n_cols) == 0) return -1;
    k = isa == HEARTH_ISA_AUTO ? hx_kernels_get() : hx_kernels_for(isa);
    if (!k) return -2;
    fn = k->matmul[dtype];
    if (!fn) return -1;
    if (T == 0 || n_rows == 0) return 0;
    if (!W || !X || !Y) return -1;
    ab = hx_act_bytes(dtype, n_cols);
    if (hx_dtype_is_quant(dtype)) {
        buf = (size_t)T <= SIZE_MAX / ab ? hx_aligned_alloc(64, ab * (size_t)T) : NULL;
        if (!buf) return -3;   /* T activations do not fit in size_t, or out of memory */
        for (int t = 0; t < T; t++)
            k->act_quantize_q8(X + (size_t)t * (size_t)n_cols, (hx_act_q8 *)((uint8_t *)buf + (size_t)t * ab), n_cols);
        act = buf;
    }
    fn(W, n_cols, act, T, Y, n_rows, 0, n_rows);
    hx_aligned_free(buf);
    return 0;
}

/* ------------------------------------------------------------ canonical helpers */

float hx_sum16(const float *a, int64_t n) {
    float L[16] = {0};
    int64_t i = 0;
    for (; i + 16 <= n; i += 16)
        for (int j = 0; j < 16; j++) L[j] = L[j] + a[i + j];
    for (; i < n; i++) L[i & 15] = L[i & 15] + a[i];
    for (int s = 8; s >= 1; s >>= 1)
        for (int j = 0; j < s; j++) L[j] = L[j] + L[j + s];
    return L[0];
}

float hx_dot16(const float *a, const float *b, int64_t n) {
    float L[16] = {0};
    int64_t i = 0;
    for (; i + 16 <= n; i += 16)
        for (int j = 0; j < 16; j++) L[j] = L[j] + a[i + j] * b[i + j];
    for (; i < n; i++) L[i & 15] = L[i & 15] + a[i] * b[i];
    for (int s = 8; s >= 1; s >>= 1)
        for (int j = 0; j < s; j++) L[j] = L[j] + L[j + s];
    return L[0];
}

void hx_rmsnorm(float *y, const float *x, const float *w, int64_t n, float eps) {
    float ms, r;
    if (n <= 0) return;
    ms = hx_dot16(x, x, n) / (float)n;
    r = 1.0f / sqrtf(ms + eps);
    for (int64_t i = 0; i < n; i++) y[i] = (x[i] * r) * w[i];
}

void hx_softmax(float *x, int64_t n) {
    float m, s;
    if (n <= 0) return;
    m = x[0];
    for (int64_t i = 1; i < n; i++) m = x[i] > m ? x[i] : m;
    for (int64_t i = 0; i < n; i++) x[i] = expf(x[i] - m);
    s = hx_sum16(x, n);
    for (int64_t i = 0; i < n; i++) x[i] = x[i] / s;
}

void hx_swiglu(float *out, const float *g, const float *u, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float gi = g[i];
        out[i] = (gi / (1.0f + expf(-gi))) * u[i];
    }
}

float hx_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

void hx_rope(float *x, int rope_dim, int style, int pos, const float *inv_freq, float attn_factor) {
    const int half = rope_dim / 2;
    for (int j = 0; j < half; j++) {
        float th = (float)pos * inv_freq[j];
        float c = cosf(th) * attn_factor, s = sinf(th) * attn_factor;
        int ia = style == 1 ? 2 * j : j, ib = style == 1 ? 2 * j + 1 : j + half;
        float a = x[ia], b = x[ib];
        x[ia] = a * c - b * s;
        x[ib] = b * c + a * s;
    }
}

void hx_axpy(float *y, float a, const float *x, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] = y[i] + a * x[i];
}
