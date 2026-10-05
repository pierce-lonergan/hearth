/*
 * quant_avx512.c — AVX-512 F/BW/VL/VNNI kernels.
 *
 * Bit-identical to the scalar definitions in quant.c (docs/NUMERICS.md):
 *  - Q8/Q4: group sums are exact integers (VNNI vpdpbusd on unsigned weights,
 *    corrected by bias*sum(x)); the float part is the same IEEE mul/mul/add per
 *    (row, token) in group order, with SIMD lanes = independent rows.
 *  - F32/F16/BF16: one __m512 is exactly the 16 dot16 lanes, then the fixed tree.
 *  - NaN outputs are stored as the canonical quiet NaN, as in quant.c.
 * Separate mul and add intrinsics everywhere; never FMA.
 */
#include "hx_quant.h"

#if defined(HX_ARCH_X86_64)

#include <immintrin.h>
#include <string.h>

#define Q4_BS 34
#define Q8_BS 66
#define TCH 8              /* tokens per pass over a row tile */
#define HX_CANON_NAN 0x7fc00000   /* NaN outputs are stored as this quiet NaN (see quant.c) */

/* ------------------------------------------------------------ reductions */

HX_INLINE __m512 canon16(__m512 v) {
    return _mm512_mask_mov_ps(v, _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q), _mm512_castsi512_ps(_mm512_set1_epi32(HX_CANON_NAN)));
}

HX_INLINE __m128 canon4(__m128 v) {
    return _mm_mask_mov_ps(v, _mm_cmp_ps_mask(v, v, _CMP_UNORD_Q), _mm_castsi128_ps(_mm_set1_epi32(HX_CANON_NAN)));
}

/* Canonical dot16 tree on the 16 lanes: s = 8, 4, 2, 1. */
HX_INLINE float tree16(__m512 v) {
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(v), 1));
    __m256 s8 = _mm256_add_ps(lo, hi);
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    return _mm_cvtss_f32(canon4(s1));
}

HX_INLINE float hmax16(__m512 m) {   /* m holds no NaN, so the order is irrelevant */
    __m256 a = _mm256_max_ps(_mm512_castps512_ps256(m),
                             _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(m), 1)));
    __m128 b = _mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    b = _mm_max_ps(b, _mm_movehl_ps(b, b));
    b = _mm_max_ss(b, _mm_shuffle_ps(b, b, 1));
    return _mm_cvtss_f32(b);
}

HX_INLINE int32_t hsum_i32(__m512i v) {
    __m256i a = _mm256_add_epi32(_mm512_castsi512_si256(v), _mm512_extracti64x4_epi64(v, 1));
    __m128i b = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    b = _mm_add_epi32(b, _mm_shuffle_epi32(b, 0x4e));
    b = _mm_add_epi32(b, _mm_shuffle_epi32(b, 0xb1));
    return _mm_cvtsi128_si32(b);
}

/* Four "quad" vectors, each holding 4 rows (one per 128-bit lane) x 4 partial
 * sums; returns lane 4k+j = total of quad j's lane k. */
HX_INLINE __m512i red_quads(__m512i v0, __m512i v1, __m512i v2, __m512i v3) {
    __m512i s01 = _mm512_add_epi32(_mm512_unpacklo_epi32(v0, v1), _mm512_unpackhi_epi32(v0, v1));
    __m512i s23 = _mm512_add_epi32(_mm512_unpacklo_epi32(v2, v3), _mm512_unpackhi_epi32(v2, v3));
    return _mm512_add_epi32(_mm512_unpacklo_epi64(s01, s23), _mm512_unpackhi_epi64(s01, s23));
}

/* One quad: lane k (of 4) = total of row k. */
HX_INLINE __m128i red_quad1(__m512i v) {
    const __m512i idx = _mm512_setr_epi32(0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12);
    v = _mm512_add_epi32(v, _mm512_shuffle_epi32(v, (_MM_PERM_ENUM)0x4e));
    v = _mm512_add_epi32(v, _mm512_shuffle_epi32(v, (_MM_PERM_ENUM)0xb1));
    return _mm512_castsi512_si128(_mm512_permutexvar_epi32(idx, v));
}

/* ------------------------------------------------------------ activation quantization */

static void act_q8_avx512(const float *x, hx_act_q8 *out, int64_t n) {
    const __m512 c127 = _mm512_set1_ps(127.0f), cm127 = _mm512_set1_ps(-127.0f);
    for (int64_t b = 0; b < n / HX_QK; b++, x += HX_QK) {
        __m512 v[4], m = _mm512_setzero_ps(), vid;
        __m512i s = _mm512_setzero_si512();
        float amax, d, id;
        for (int k = 0; k < 4; k++) {
            v[k] = _mm512_loadu_ps(x + 16 * k);
            m = _mm512_max_ps(_mm512_abs_ps(v[k]), m);   /* NaN in the first operand is skipped */
        }
        amax = hmax16(m);
        d = amax / 127.0f;
        id = d != 0.0f ? 1.0f / d : 0.0f;
        vid = _mm512_set1_ps(id);
        for (int k = 0; k < 4; k++) {
            __m512 p = _mm512_mul_ps(v[k], vid);
            __m512i q;
            p = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(p, p, _CMP_ORD_Q), p);
            p = _mm512_max_ps(_mm512_min_ps(p, c127), cm127);
            q = _mm512_cvtps_epi32(p);
            s = _mm512_add_epi32(s, q);
            _mm_storeu_si128((__m128i *)(out[b].q + 16 * k), _mm512_cvtepi32_epi8(q));
        }
        out[b].d = d;
        out[b].sum = hsum_i32(s);
    }
}

/* ------------------------------------------------------------ float weights */

HX_INLINE __m512 ld16(int dt, const void *w, int64_t i) {
    if (dt == HEARTH_F32) return _mm512_loadu_ps((const float *)w + i);
    {
        __m256i h = _mm256_loadu_si256((const __m256i *)((const uint16_t *)w + i));
        if (dt == HEARTH_F16) return _mm512_cvtph_ps(h);
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
    }
}

HX_INLINE __m512 ld16_tail(int dt, const void *w, int64_t i, __mmask16 m) {
    if (dt == HEARTH_F32) return _mm512_maskz_loadu_ps(m, (const float *)w + i);
    {
        __m256i h = _mm256_maskz_loadu_epi16(m, (const uint16_t *)w + i);
        if (dt == HEARTH_F16) return _mm512_cvtph_ps(h);
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
    }
}

HX_INLINE void mm_float(int dt, const void *W, int64_t n, const void *act, int T,
                        float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    const size_t rb = (size_t)n * (dt == HEARTH_F32 ? 4u : 2u);
    const int64_t n16 = n & ~(int64_t)15;
    const __mmask16 tm = (__mmask16)((1u << (unsigned)(n - n16)) - 1u);
    const float *X = (const float *)act;
    int64_t r = r0;
    for (; r + 4 <= r1; r += 4) {
        const uint8_t *w0 = (const uint8_t *)W + (size_t)r * rb;
        const uint8_t *w1 = w0 + rb, *w2 = w1 + rb, *w3 = w2 + rb;
        for (int t = 0; t < T; t++) {
            const float *x = X + (size_t)t * (size_t)n;
            float *y = Y + (size_t)t * (size_t)ldy + r;
            __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
            for (int64_t i = 0; i < n16; i += 16) {
                __m512 xv = _mm512_loadu_ps(x + i);
                a0 = _mm512_add_ps(a0, _mm512_mul_ps(ld16(dt, w0, i), xv));
                a1 = _mm512_add_ps(a1, _mm512_mul_ps(ld16(dt, w1, i), xv));
                a2 = _mm512_add_ps(a2, _mm512_mul_ps(ld16(dt, w2, i), xv));
                a3 = _mm512_add_ps(a3, _mm512_mul_ps(ld16(dt, w3, i), xv));
            }
            if (tm) {
                __m512 xv = _mm512_maskz_loadu_ps(tm, x + n16);
                a0 = _mm512_mask_add_ps(a0, tm, a0, _mm512_mul_ps(ld16_tail(dt, w0, n16, tm), xv));
                a1 = _mm512_mask_add_ps(a1, tm, a1, _mm512_mul_ps(ld16_tail(dt, w1, n16, tm), xv));
                a2 = _mm512_mask_add_ps(a2, tm, a2, _mm512_mul_ps(ld16_tail(dt, w2, n16, tm), xv));
                a3 = _mm512_mask_add_ps(a3, tm, a3, _mm512_mul_ps(ld16_tail(dt, w3, n16, tm), xv));
            }
            y[0] = tree16(a0);
            y[1] = tree16(a1);
            y[2] = tree16(a2);
            y[3] = tree16(a3);
        }
    }
    for (; r < r1; r++) {
        const uint8_t *w0 = (const uint8_t *)W + (size_t)r * rb;
        for (int t = 0; t < T; t++) {
            const float *x = X + (size_t)t * (size_t)n;
            __m512 a0 = _mm512_setzero_ps();
            for (int64_t i = 0; i < n16; i += 16)
                a0 = _mm512_add_ps(a0, _mm512_mul_ps(ld16(dt, w0, i), _mm512_loadu_ps(x + i)));
            if (tm)
                a0 = _mm512_mask_add_ps(a0, tm, a0, _mm512_mul_ps(ld16_tail(dt, w0, n16, tm),
                                                                  _mm512_maskz_loadu_ps(tm, x + n16)));
            Y[(size_t)t * (size_t)ldy + r] = tree16(a0);
        }
    }
}

static void mm_f32_avx512(const void *W, int64_t n, const void *act, int T, float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    mm_float(HEARTH_F32, W, n, act, T, Y, ldy, r0, r1);
}
static void mm_f16_avx512(const void *W, int64_t n, const void *act, int T, float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    mm_float(HEARTH_F16, W, n, act, T, Y, ldy, r0, r1);
}
static void mm_bf16_avx512(const void *W, int64_t n, const void *act, int T, float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    mm_float(HEARTH_BF16, W, n, act, T, Y, ldy, r0, r1);
}

/* ------------------------------------------------------------ Q8 / Q4 weights */

/* Block g of four rows a..d, one row per 128-bit lane, as unsigned bytes:
 * u[0] = elements 0-15, u[1] = 16-31, u[2] = 32-47, u[3] = 48-63.
 * Q4: nibbles (q+8). Q8: q xor 0x80 = q+128. a..d point at the blocks' quants. */
HX_INLINE void unpack_quad(int q4, const uint8_t *a, const uint8_t *b, const uint8_t *c,
                           const uint8_t *d, __m512i *u) {
    if (q4) {
        const __m512i m4 = _mm512_set1_epi8(0x0f);
        __m512i ab = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)a)),
                                        _mm256_loadu_si256((const __m256i *)b), 1);
        __m512i cd = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)c)),
                                        _mm256_loadu_si256((const __m256i *)d), 1);
        __m512i z0 = _mm512_shuffle_i64x2(ab, cd, 0x88);   /* bytes 0-15 of a,b,c,d */
        __m512i z1 = _mm512_shuffle_i64x2(ab, cd, 0xdd);   /* bytes 16-31 */
        u[0] = _mm512_and_si512(z0, m4);
        u[1] = _mm512_and_si512(z1, m4);
        u[2] = _mm512_and_si512(_mm512_srli_epi16(z0, 4), m4);
        u[3] = _mm512_and_si512(_mm512_srli_epi16(z1, 4), m4);
    } else {
        const __m512i sb = _mm512_set1_epi8((char)0x80);
        __m512i t0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)a)),
                                        _mm256_loadu_si256((const __m256i *)b), 1);
        __m512i t1 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(a + 32))),
                                        _mm256_loadu_si256((const __m256i *)(b + 32)), 1);
        __m512i t2 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)c)),
                                        _mm256_loadu_si256((const __m256i *)d), 1);
        __m512i t3 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(c + 32))),
                                        _mm256_loadu_si256((const __m256i *)(d + 32)), 1);
        u[0] = _mm512_xor_si512(_mm512_shuffle_i64x2(t0, t2, 0x88), sb);
        u[1] = _mm512_xor_si512(_mm512_shuffle_i64x2(t0, t2, 0xdd), sb);
        u[2] = _mm512_xor_si512(_mm512_shuffle_i64x2(t1, t3, 0x88), sb);
        u[3] = _mm512_xor_si512(_mm512_shuffle_i64x2(t1, t3, 0xdd), sb);
    }
}

HX_INLINE int ld_u16(const uint8_t *p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

HX_INLINE void act_bcast(const hx_act_q8 *a, __m512i *x) {
    x[0] = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(a->q + 0)));
    x[1] = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(a->q + 16)));
    x[2] = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(a->q + 32)));
    x[3] = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(a->q + 48)));
}

HX_INLINE __m512i dot_quad(const __m512i *u, const __m512i *x) {
    __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), u[0], x[0]);
    acc = _mm512_dpbusd_epi32(acc, u[1], x[1]);
    acc = _mm512_dpbusd_epi32(acc, u[2], x[2]);
    return _mm512_dpbusd_epi32(acc, u[3], x[3]);
}

/* Four f16 scales at p, p+rb, p+2rb, p+3rb packed into 64 bits (lane 0 lowest).
 * Packing in GPRs then one move measured faster than per-scale inserts or vpgatherdd. */
HX_INLINE uint64_t ld4s(const uint8_t *p, size_t rb) {
    return (uint64_t)ld_u16(p) | ((uint64_t)ld_u16(p + rb) << 16) |
           ((uint64_t)ld_u16(p + 2 * rb) << 32) | ((uint64_t)ld_u16(p + 3 * rb) << 48);
}

/* 16-row tiles: lane i of every vector is row r+i. Row i of block g starts at
 * p + i*rb with p = row r + g*bs; quad j holds rows j, j+4, j+8, j+12 so that
 * red_quads() returns the rows in lane order. */
HX_INLINE __m512 q16_scales(const uint8_t *p, size_t rb) {
    __m128i h0 = _mm_set_epi64x((long long)ld4s(p + 4 * rb, rb), (long long)ld4s(p, rb));
    __m128i h1 = _mm_set_epi64x((long long)ld4s(p + 12 * rb, rb), (long long)ld4s(p + 8 * rb, rb));
    return _mm512_cvtph_ps(_mm256_inserti128_si256(_mm256_castsi128_si256(h0), h1, 1));
}

HX_INLINE void q16_unpack(int q4, const uint8_t *p, size_t rb, __m512i *u) {
    const uint8_t *q = p + 2;
    for (int j = 0; j < 4; j++)
        unpack_quad(q4, q + j * rb, q + (j + 4) * rb, q + (j + 8) * rb, q + (j + 12) * rb, u + 4 * j);
}

/* acc + (float)isum * (dw * dx) for the 16 rows of one block and one token. */
HX_INLINE __m512 q16_step(const __m512i *u, __m512 dw, const hx_act_q8 *a, int32_t bias, __m512 acc) {
    __m512i x[4], is;
    act_bcast(a, x);
    is = red_quads(dot_quad(u, x), dot_quad(u + 4, x), dot_quad(u + 8, x), dot_quad(u + 12, x));
    is = _mm512_sub_epi32(is, _mm512_set1_epi32(a->sum * bias));
    return _mm512_add_ps(acc, _mm512_mul_ps(_mm512_cvtepi32_ps(is), _mm512_mul_ps(dw, _mm512_set1_ps(a->d))));
}

/* Rows [r, r+16). */
HX_INLINE void qtile16(int q4, const uint8_t *W, size_t rb, int64_t nb, const hx_act_q8 *A, int T,
                       float *Y, int64_t ldy, int64_t r) {
    const size_t bs = q4 ? Q4_BS : Q8_BS;
    const int32_t bias = q4 ? 8 : 128;
    const uint8_t *p0 = W + (size_t)r * rb;
    if (T == 1) {   /* decode: keep the accumulator in a register */
        __m512 acc = _mm512_setzero_ps();
        for (int64_t g = 0; g < nb; g++) {
            const uint8_t *p = p0 + (size_t)g * bs;
            __m512i u[16];
            q16_unpack(q4, p, rb, u);
            acc = q16_step(u, q16_scales(p, rb), A + g, bias, acc);
        }
        _mm512_storeu_ps(Y + r, canon16(acc));
        return;
    }
    for (int t0 = 0; t0 < T; t0 += TCH) {
        const int tn = T - t0 < TCH ? T - t0 : TCH;
        __m512 acc[TCH];
        for (int t = 0; t < tn; t++) acc[t] = _mm512_setzero_ps();
        for (int64_t g = 0; g < nb; g++) {
            const uint8_t *p = p0 + (size_t)g * bs;
            const __m512 dw = q16_scales(p, rb);
            __m512i u[16];
            q16_unpack(q4, p, rb, u);
            for (int t = 0; t < tn; t++)
                acc[t] = q16_step(u, dw, A + (size_t)(t0 + t) * (size_t)nb + g, bias, acc[t]);
        }
        for (int t = 0; t < tn; t++)
            _mm512_storeu_ps(Y + (size_t)(t0 + t) * (size_t)ldy + r, canon16(acc[t]));
    }
}

/* Rows [r, r+nr), nr <= 4, one row per 128-bit lane. */
HX_INLINE void qtile4(int q4, const uint8_t *W, size_t rb, int64_t nb, const hx_act_q8 *A, int T,
                      float *Y, int64_t ldy, int64_t r, int nr) {
    const size_t bs = q4 ? Q4_BS : Q8_BS;
    const int32_t bias = q4 ? 8 : 128;
    const __mmask8 sm = (__mmask8)((1u << nr) - 1u);
    const uint8_t *rp[4];
    for (int i = 0; i < 4; i++) rp[i] = W + (size_t)(r + (i < nr ? i : nr - 1)) * rb;
    for (int t0 = 0; t0 < T; t0 += TCH) {
        const int tn = T - t0 < TCH ? T - t0 : TCH;
        __m128 acc[TCH];
        for (int t = 0; t < tn; t++) acc[t] = _mm_setzero_ps();
        for (int64_t g = 0; g < nb; g++) {
            const size_t go = (size_t)g * bs;
            __m512i u[4];
            const uint64_t h = (uint64_t)ld_u16(rp[0] + go) | ((uint64_t)ld_u16(rp[1] + go) << 16) |
                               ((uint64_t)ld_u16(rp[2] + go) << 32) | ((uint64_t)ld_u16(rp[3] + go) << 48);
            const __m128 dw = _mm_cvtph_ps(_mm_cvtsi64_si128((long long)h));
            unpack_quad(q4, rp[0] + go + 2, rp[1] + go + 2, rp[2] + go + 2, rp[3] + go + 2, u);
            for (int t = 0; t < tn; t++) {
                const hx_act_q8 *a = A + (size_t)(t0 + t) * (size_t)nb + g;
                __m512i x[4];
                __m128i is;
                act_bcast(a, x);
                is = red_quad1(dot_quad(u, x));
                is = _mm_sub_epi32(is, _mm_set1_epi32(a->sum * bias));
                acc[t] = _mm_add_ps(acc[t], _mm_mul_ps(_mm_cvtepi32_ps(is), _mm_mul_ps(dw, _mm_set1_ps(a->d))));
            }
        }
        for (int t = 0; t < tn; t++)
            _mm_mask_storeu_ps(Y + (size_t)(t0 + t) * (size_t)ldy + r, sm, canon4(acc[t]));
    }
}

HX_INLINE void mm_quant(int q4, const void *W, int64_t n, const void *act, int T,
                        float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    const int64_t nb = n / HX_QK;
    const size_t rb = (size_t)nb * (q4 ? Q4_BS : Q8_BS);
    const uint8_t *w = (const uint8_t *)W;
    const hx_act_q8 *A = (const hx_act_q8 *)act;
    int64_t r = r0;
    for (; r + 16 <= r1; r += 16) {
        qtile16(q4, w, rb, nb, A, T, Y, ldy, r);
    }
    for (; r < r1; r += 4) {
        int nr = r1 - r < 4 ? (int)(r1 - r) : 4;
        qtile4(q4, w, rb, nb, A, T, Y, ldy, r, nr);
    }
}

static void mm_q8_avx512(const void *W, int64_t n, const void *act, int T, float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    mm_quant(0, W, n, act, T, Y, ldy, r0, r1);
}
static void mm_q4_avx512(const void *W, int64_t n, const void *act, int T, float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    mm_quant(1, W, n, act, T, Y, ldy, r0, r1);
}

static const hx_kernels k_avx512 = {
    HEARTH_ISA_AVX512,
    {mm_f32_avx512, mm_f16_avx512, mm_bf16_avx512, mm_q8_avx512, mm_q4_avx512, NULL, NULL},
    act_q8_avx512,
};

const hx_kernels *hx_kernels_avx512_table(void) { return &k_avx512; }

#else
typedef int hx_quant_avx512_unused;   /* non-x86 builds: quant.c reports "not compiled in" */
#endif
