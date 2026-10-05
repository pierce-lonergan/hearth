/*
 * quant_avx2.c — AVX2 + F16C kernels.
 *
 * Bit-identical to the scalar definitions in quant.c (docs/NUMERICS.md):
 *  - Q8/Q4: exact integer group sums via maddubs/madd (int16 bounds below),
 *    then the same IEEE mul/mul/add per (row, token) in group order, with the
 *    8 lanes of a __m256 = 8 independent rows.
 *  - F32/F16/BF16: the 16 dot16 lanes are two __m256 (lanes 0-7, 8-15), then
 *    the fixed tree. Separate mul and add intrinsics everywhere; never FMA.
 *  - NaN outputs are stored as the canonical quiet NaN, as in quant.c.
 * Activation quants must lie in [-127, 127] (guaranteed by act_quantize_q8).
 */
#include "hx_quant.h"

#if defined(HX_ARCH_X86_64)

#include <immintrin.h>
#include <string.h>

#define Q4_BS 34
#define Q8_BS 66
#define TCH 8
#define HX_CANON_NAN 0x7fc00000   /* NaN outputs are stored as this quiet NaN (see quant.c) */

/* ------------------------------------------------------------ reductions */

HX_INLINE __m256 canon8(__m256 v) {
    return _mm256_blendv_ps(v, _mm256_castsi256_ps(_mm256_set1_epi32(HX_CANON_NAN)), _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
}

HX_INLINE __m128 canon4(__m128 v) {
    return _mm_blendv_ps(v, _mm_castsi128_ps(_mm_set1_epi32(HX_CANON_NAN)), _mm_cmpunord_ps(v, v));
}

HX_INLINE float tree16(__m256 lo, __m256 hi) {   /* lanes 0-7, 8-15 */
    __m256 s8 = _mm256_add_ps(lo, hi);
    __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps(s8, 1));
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    return _mm_cvtss_f32(canon4(s1));
}

HX_INLINE float hmax8(__m256 m) {   /* no NaN in m */
    __m128 b = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    b = _mm_max_ps(b, _mm_movehl_ps(b, b));
    b = _mm_max_ss(b, _mm_shuffle_ps(b, b, 1));
    return _mm_cvtss_f32(b);
}

HX_INLINE int32_t hsum8_i32(__m256i v) {
    __m128i b = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    b = _mm_add_epi32(b, _mm_shuffle_epi32(b, 0x4e));
    b = _mm_add_epi32(b, _mm_shuffle_epi32(b, 0xb1));
    return _mm_cvtsi128_si32(b);
}

HX_INLINE int ld_u16(const uint8_t *p) {
    uint16_t v;
    memcpy(&v, p, 2);
    return v;
}

/* The f16 block scales of rows rp[0..3] at offset go, packed into 64 bits (lane 0
 * lowest). Packing in GPRs measured faster than vpgatherdd or per-scale inserts. */
HX_INLINE uint64_t ld4s(const uint8_t *const *rp, size_t go) {
    return (uint64_t)ld_u16(rp[0] + go) | ((uint64_t)ld_u16(rp[1] + go) << 16) |
           ((uint64_t)ld_u16(rp[2] + go) << 32) | ((uint64_t)ld_u16(rp[3] + go) << 48);
}

/* s[i] = 8 partial sums of row i -> lane i = total of row i. */
HX_INLINE __m256i red8(const __m256i *s) {
    __m256i h01 = _mm256_hadd_epi32(s[0], s[1]), h23 = _mm256_hadd_epi32(s[2], s[3]);
    __m256i h45 = _mm256_hadd_epi32(s[4], s[5]), h67 = _mm256_hadd_epi32(s[6], s[7]);
    __m256i a = _mm256_hadd_epi32(h01, h23), b = _mm256_hadd_epi32(h45, h67);
    return _mm256_add_epi32(_mm256_permute2x128_si256(a, b, 0x20), _mm256_permute2x128_si256(a, b, 0x31));
}

/* ------------------------------------------------------------ activation quantization */

static void act_q8_avx2(const float *x, hx_act_q8 *out, int64_t n) {
    const __m256 c127 = _mm256_set1_ps(127.0f), cm127 = _mm256_set1_ps(-127.0f);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int64_t b = 0; b < n / HX_QK; b++, x += HX_QK) {
        __m256 v[8], m = _mm256_setzero_ps(), vid;
        __m256i q[8], s = _mm256_setzero_si256();
        float amax, d, id;
        for (int k = 0; k < 8; k++) {
            v[k] = _mm256_loadu_ps(x + 8 * k);
            m = _mm256_max_ps(_mm256_andnot_ps(sign, v[k]), m);   /* NaN in the first operand is skipped */
        }
        amax = hmax8(m);
        d = amax / 127.0f;
        id = d != 0.0f ? 1.0f / d : 0.0f;
        vid = _mm256_set1_ps(id);
        for (int k = 0; k < 8; k++) {
            __m256 p = _mm256_mul_ps(v[k], vid);
            p = _mm256_and_ps(p, _mm256_cmp_ps(p, p, _CMP_ORD_Q));
            p = _mm256_max_ps(_mm256_min_ps(p, c127), cm127);
            q[k] = _mm256_cvtps_epi32(p);
            s = _mm256_add_epi32(s, q[k]);
        }
        for (int h = 0; h < 2; h++) {
            __m256i w = _mm256_packs_epi16(_mm256_packs_epi32(q[4 * h], q[4 * h + 1]),
                                           _mm256_packs_epi32(q[4 * h + 2], q[4 * h + 3]));
            _mm256_storeu_si256((__m256i *)(out[b].q + 32 * h), _mm256_permutevar8x32_epi32(w, perm));
        }
        out[b].d = d;
        out[b].sum = hsum8_i32(s);
    }
}

/* ------------------------------------------------------------ float weights */

HX_INLINE void ld16(int dt, const void *w, int64_t i, __m256 *lo, __m256 *hi) {
    if (dt == HEARTH_F32) {
        *lo = _mm256_loadu_ps((const float *)w + i);
        *hi = _mm256_loadu_ps((const float *)w + i + 8);
    } else {
        __m128i h0 = _mm_loadu_si128((const __m128i *)((const uint16_t *)w + i));
        __m128i h1 = _mm_loadu_si128((const __m128i *)((const uint16_t *)w + i + 8));
        if (dt == HEARTH_F16) {
            *lo = _mm256_cvtph_ps(h0);
            *hi = _mm256_cvtph_ps(h1);
        } else {
            *lo = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h0), 16));
            *hi = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h1), 16));
        }
    }
}

HX_INLINE __m128 w_one(int dt, const void *w, int64_t i) {   /* element i as f32 in lane 0, exact */
    if (dt == HEARTH_F32) return _mm_load_ss((const float *)w + i);
    {
        int h = ((const uint16_t *)w)[i];
        if (dt == HEARTH_F16) return _mm_cvtph_ps(_mm_cvtsi32_si128(h));
        return _mm_castsi128_ps(_mm_cvtsi32_si128(h << 16));
    }
}

/* Adds the last rem (< 16) products into lanes 0..rem-1, then the canonical tree.
 * Done on a stored copy so the accumulators never have their address taken. */
HX_INLINE float finish16(int dt, __m256 lo, __m256 hi, const void *w, const float *x, int64_t i0, int rem) {
    if (rem) {
        HX_ALIGNED(32) float L[16];
        _mm256_store_ps(L, lo);
        _mm256_store_ps(L + 8, hi);
        for (int j = 0; j < rem; j++)
            _mm_store_ss(L + j, _mm_add_ss(_mm_load_ss(L + j), _mm_mul_ss(w_one(dt, w, i0 + j), _mm_load_ss(x + i0 + j))));
        lo = _mm256_load_ps(L);
        hi = _mm256_load_ps(L + 8);
    }
    return tree16(lo, hi);
}

HX_INLINE void mm_float(int dt, const void *W, int64_t n, const void *act, int T,
                        float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    const size_t rb = (size_t)n * (dt == HEARTH_F32 ? 4u : 2u);
    const int64_t n16 = n & ~(int64_t)15;
    const int rem = (int)(n - n16);
    const float *X = (const float *)act;
    int64_t r = r0;
    for (; r + 4 <= r1; r += 4) {
        const uint8_t *w0 = (const uint8_t *)W + (size_t)r * rb;
        const uint8_t *w1 = w0 + rb, *w2 = w1 + rb, *w3 = w2 + rb;
        for (int t = 0; t < T; t++) {
            const float *x = X + (size_t)t * (size_t)n;
            float *y = Y + (size_t)t * (size_t)ldy + r;
            __m256 a0 = _mm256_setzero_ps(), b0 = a0, a1 = a0, b1 = a0, a2 = a0, b2 = a0, a3 = a0, b3 = a0;
            for (int64_t i = 0; i < n16; i += 16) {
                __m256 xl = _mm256_loadu_ps(x + i), xh = _mm256_loadu_ps(x + i + 8), wl, wh;
                ld16(dt, w0, i, &wl, &wh);
                a0 = _mm256_add_ps(a0, _mm256_mul_ps(wl, xl));
                b0 = _mm256_add_ps(b0, _mm256_mul_ps(wh, xh));
                ld16(dt, w1, i, &wl, &wh);
                a1 = _mm256_add_ps(a1, _mm256_mul_ps(wl, xl));
                b1 = _mm256_add_ps(b1, _mm256_mul_ps(wh, xh));
                ld16(dt, w2, i, &wl, &wh);
                a2 = _mm256_add_ps(a2, _mm256_mul_ps(wl, xl));
                b2 = _mm256_add_ps(b2, _mm256_mul_ps(wh, xh));
                ld16(dt, w3, i, &wl, &wh);
                a3 = _mm256_add_ps(a3, _mm256_mul_ps(wl, xl));
                b3 = _mm256_add_ps(b3, _mm256_mul_ps(wh, xh));
            }
            y[0] = finish16(dt, a0, b0, w0, x, n16, rem);
            y[1] = finish16(dt, a1, b1, w1, x, n16, rem);
            y[2] = finish16(dt, a2, b2, w2, x, n16, rem);
            y[3] = finish16(dt, a3, b3, w3, x, n16, rem);
        }
    }
    for (; r < r1; r++) {
        const uint8_t *w0 = (const uint8_t *)W + (size_t)r * rb;
        for (int t = 0; t < T; t++) {
            const float *x = X + (size_t)t * (size_t)n;
            __m256 a0 = _mm256_setzero_ps(), b0 = a0;
            for (int64_t i = 0; i < n16; i += 16) {
                __m256 wl, wh;
                ld16(dt, w0, i, &wl, &wh);
                a0 = _mm256_add_ps(a0, _mm256_mul_ps(wl, _mm256_loadu_ps(x + i)));
                b0 = _mm256_add_ps(b0, _mm256_mul_ps(wh, _mm256_loadu_ps(x + i + 8)));
            }
            Y[(size_t)t * (size_t)ldy + r] = finish16(dt, a0, b0, w0, x, n16, rem);
        }
    }
}

static void mm_f32_avx2(const void *W, int64_t n, const void *act, int T, float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    mm_float(HEARTH_F32, W, n, act, T, Y, ldy, r0, r1);
}
static void mm_f16_avx2(const void *W, int64_t n, const void *act, int T, float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    mm_float(HEARTH_F16, W, n, act, T, Y, ldy, r0, r1);
}
static void mm_bf16_avx2(const void *W, int64_t n, const void *act, int T, float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    mm_float(HEARTH_BF16, W, n, act, T, Y, ldy, r0, r1);
}

/* ------------------------------------------------------------ Q8 / Q4 weights */

/*
 * Rows [r, r+nr), nr <= 8; lane i = row r+i (rows past nr repeat the last one
 * and are not stored).
 * Q4: maddubs(nibble 0..15, x) pairs are <= 2*15*127 = 3810 in magnitude, so the
 *     low+high int16 sum (<= 7620) cannot saturate; isum = sum(nib*x) - 8*sum(x).
 * Q8: maddubs(|w| <= 128, sign(x, w)) pairs are <= 2*128*127 = 32512 < 32767.
 */
HX_INLINE void qtile8(int q4, const uint8_t *W, size_t rb, int64_t nb, const hx_act_q8 *A, int T,
                      float *Y, int64_t ldy, int64_t r, int nr) {
    const size_t bs = q4 ? Q4_BS : Q8_BS;
    const __m256i ones = _mm256_set1_epi16(1), m4 = _mm256_set1_epi8(0x0f);
    const __m256i smask = _mm256_cmpgt_epi32(_mm256_set1_epi32(nr), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    const uint8_t *rp[8];
    for (int i = 0; i < 8; i++) rp[i] = W + (size_t)(r + (i < nr ? i : nr - 1)) * rb;
    if (T == 1) {   /* decode: row by row, accumulator in a register */
        __m256 acc = _mm256_setzero_ps();
        for (int64_t g = 0; g < nb; g++) {
            const size_t go = (size_t)g * bs;
            const hx_act_q8 *a = A + g;
            const __m256i x0 = _mm256_loadu_si256((const __m256i *)a->q);
            const __m256i x1 = _mm256_loadu_si256((const __m256i *)(a->q + 32));
            const __m256 dw = _mm256_cvtph_ps(_mm_set_epi64x((long long)ld4s(rp + 4, go), (long long)ld4s(rp, go)));
            __m256i s[8], is;
            for (int i = 0; i < 8; i++) {
                const uint8_t *p = rp[i] + go + 2;
                if (q4) {
                    __m256i v = _mm256_loadu_si256((const __m256i *)p);
                    __m256i lo = _mm256_and_si256(v, m4), hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
                    s[i] = _mm256_madd_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(lo, x0), _mm256_maddubs_epi16(hi, x1)), ones);
                } else {
                    __m256i w0 = _mm256_loadu_si256((const __m256i *)p);
                    __m256i w1 = _mm256_loadu_si256((const __m256i *)(p + 32));
                    __m256i p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(w0), _mm256_sign_epi8(x0, w0));
                    __m256i p1 = _mm256_maddubs_epi16(_mm256_abs_epi8(w1), _mm256_sign_epi8(x1, w1));
                    s[i] = _mm256_add_epi32(_mm256_madd_epi16(p0, ones), _mm256_madd_epi16(p1, ones));
                }
            }
            is = red8(s);
            if (q4) is = _mm256_sub_epi32(is, _mm256_set1_epi32(a->sum * 8));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(is), _mm256_mul_ps(dw, _mm256_set1_ps(a->d))));
        }
        acc = canon8(acc);
        if (nr == 8) _mm256_storeu_ps(Y + r, acc);
        else _mm256_maskstore_ps(Y + r, smask, acc);
        return;
    }
    for (int t0 = 0; t0 < T; t0 += TCH) {
        const int tn = T - t0 < TCH ? T - t0 : TCH;
        __m256 acc[TCH];
        for (int t = 0; t < tn; t++) acc[t] = _mm256_setzero_ps();
        for (int64_t g = 0; g < nb; g++) {
            const size_t go = (size_t)g * bs;
            __m256i wa[8], wb[8];   /* Q4: low / high nibbles. Q8: |w| halves */
            __m256i sa[8], sb[8];   /* Q8 only: signs */
            const __m256 dw = _mm256_cvtph_ps(_mm_set_epi64x((long long)ld4s(rp + 4, go), (long long)ld4s(rp, go)));
            for (int i = 0; i < 8; i++) {
                const uint8_t *p = rp[i] + go + 2;
                if (q4) {
                    __m256i v = _mm256_loadu_si256((const __m256i *)p);
                    wa[i] = _mm256_and_si256(v, m4);
                    wb[i] = _mm256_and_si256(_mm256_srli_epi16(v, 4), m4);
                } else {
                    sa[i] = _mm256_loadu_si256((const __m256i *)p);
                    sb[i] = _mm256_loadu_si256((const __m256i *)(p + 32));
                    wa[i] = _mm256_abs_epi8(sa[i]);
                    wb[i] = _mm256_abs_epi8(sb[i]);
                }
            }
            for (int t = 0; t < tn; t++) {
                const hx_act_q8 *a = A + (size_t)(t0 + t) * (size_t)nb + g;
                const __m256i x0 = _mm256_loadu_si256((const __m256i *)a->q);
                const __m256i x1 = _mm256_loadu_si256((const __m256i *)(a->q + 32));
                __m256i s[8], is;
                for (int i = 0; i < 8; i++) {
                    if (q4) {
                        __m256i p = _mm256_add_epi16(_mm256_maddubs_epi16(wa[i], x0), _mm256_maddubs_epi16(wb[i], x1));
                        s[i] = _mm256_madd_epi16(p, ones);
                    } else {
                        __m256i p0 = _mm256_maddubs_epi16(wa[i], _mm256_sign_epi8(x0, sa[i]));
                        __m256i p1 = _mm256_maddubs_epi16(wb[i], _mm256_sign_epi8(x1, sb[i]));
                        s[i] = _mm256_add_epi32(_mm256_madd_epi16(p0, ones), _mm256_madd_epi16(p1, ones));
                    }
                }
                is = red8(s);
                if (q4) is = _mm256_sub_epi32(is, _mm256_set1_epi32(a->sum * 8));
                acc[t] = _mm256_add_ps(acc[t], _mm256_mul_ps(_mm256_cvtepi32_ps(is),
                                                             _mm256_mul_ps(dw, _mm256_set1_ps(a->d))));
            }
        }
        for (int t = 0; t < tn; t++) {
            float *y = Y + (size_t)(t0 + t) * (size_t)ldy + r;
            const __m256 v = canon8(acc[t]);
            if (nr == 8) _mm256_storeu_ps(y, v);
            else _mm256_maskstore_ps(y, smask, v);
        }
    }
}

HX_INLINE void mm_quant(int q4, const void *W, int64_t n, const void *act, int T,
                        float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    const int64_t nb = n / HX_QK;
    const size_t rb = (size_t)nb * (q4 ? Q4_BS : Q8_BS);
    const uint8_t *w = (const uint8_t *)W;
    const hx_act_q8 *A = (const hx_act_q8 *)act;
    for (int64_t r = r0; r < r1; r += 8) {
        int nr = r1 - r < 8 ? (int)(r1 - r) : 8;
        qtile8(q4, w, rb, nb, A, T, Y, ldy, r, nr);
    }
}

static void mm_q8_avx2(const void *W, int64_t n, const void *act, int T, float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    mm_quant(0, W, n, act, T, Y, ldy, r0, r1);
}
static void mm_q4_avx2(const void *W, int64_t n, const void *act, int T, float *Y, int64_t ldy, int64_t r0, int64_t r1) {
    mm_quant(1, W, n, act, T, Y, ldy, r0, r1);
}

static const hx_kernels k_avx2 = {
    HEARTH_ISA_AVX2,
    {mm_f32_avx2, mm_f16_avx2, mm_bf16_avx2, mm_q8_avx2, mm_q4_avx2, NULL, NULL},
    act_q8_avx2,
};

const hx_kernels *hx_kernels_avx2_table(void) { return &k_avx2; }

#else
typedef int hx_quant_avx2_unused;   /* non-x86 builds: quant.c reports "not compiled in" */
#endif
