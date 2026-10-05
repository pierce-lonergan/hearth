/*
 * test_quant.c — formats, quantizers, kernels and canonical helpers.
 *
 *   test_quant [--small] [--no-bench] [--bench-only]
 *
 * Checks (self-verifying, exit code = number of failures, capped at 255):
 *  - row/act byte sizes, dtype helpers, every public API argument error alone,
 *    I32/U8 dequantization, hx_act_prepare
 *  - ISA tables: SIMD kernels present whenever the CPU reports the features;
 *    clearing any one required feature bit disables its ISA (hearth_matmul -2);
 *    HEARTH_ISA dispatch, checked in child processes (the program re-runs itself
 *    with --dispatch-child, so run it by its full path)
 *  - quantize -> dequantize error bounds, Q8 spec conformance incl. exact .5 ties,
 *    NaN inputs quantized exactly like 0, Q4 never worse than the absmax baseline,
 *    determinism across thread counts; RMSE report
 *  - activation quantization: every ISA memcmp-identical to scalar, scalar == spec
 *  - matmul: every dtype, many shapes / T / input kinds: scalar == an independent
 *    implementation of NUMERICS §3, every ISA memcmp-identical to scalar, batched
 *    == T single calls, arbitrary row splits == whole, no writes outside Y rows,
 *    and the result close to a double-precision product
 *  - matmul on raw-bit weights (NaN/inf/subnormal values and block scales) and
 *    non-finite activations: the same identities, every NaN output canonical
 *  - canonical helpers: bitwise vs an independent canonical-order version, and
 *    close to double-precision references
 *  - throughput table (GB/s of weight bytes): L2-resident 1 thread, and a DRAM
 *    resident matrix (1 GiB; 64 MiB with --small) with 16 threads
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#  define _POSIX_C_SOURCE 200809L   /* setenv/unsetenv under -std=c11 */
#endif
#include "hx_quant.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail, g_checks;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        g_checks++;                                                        \
        if (!(cond)) {                                                     \
            if (++g_fail <= 40) {                                          \
                printf("FAIL %s:%d: ", __FILE__, __LINE__);                \
                printf(__VA_ARGS__);                                       \
                printf("\n");                                              \
            }                                                              \
        }                                                                  \
    } while (0)

static int g_small;

/* see test_pinned() */
#define PIN_Q8 0xda285a267ed1fb09ull
#define PIN_Q4 0x5ed717f72d36fb0cull
#define PIN_Y_F32 0x78ee7e5c72e238a0ull
#define PIN_Y_F16 0xacbfba9fb13b1207ull
#define PIN_Y_BF16 0xaa0b6460cbd8ad23ull
#define PIN_Y_Q8 0x532a2d32570c08dcull
#define PIN_Y_Q4 0xa8a0369ab6f0ed22ull
#define PIN_Q4_TAILS 0xb020847e5d2b0807ull

/* ------------------------------------------------------------ rng */

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint64_t rnd64(void) {
    uint64_t z = (g_rng += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static double rnd_u(void) { return (double)(rnd64() >> 11) * (1.0 / 9007199254740992.0); }
static float rnd_gauss(void) {
    double u1 = rnd_u(), u2 = rnd_u();
    if (u1 < 1e-300) u1 = 1e-300;
    return (float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2));
}

static float bits_f32(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f32_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

#define CANON_NAN 0x7fc00000u   /* the one NaN the kernels store */

static void set_env(const char *k, const char *v) {
#if defined(HX_OS_WINDOWS)
    _putenv_s(k, v ? v : "");
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}

/* ------------------------------------------------------------ independent references */

/* f16 -> float written from the IEEE definition (finite values and inf only). */
static float ref_f16(uint16_t h) {
    int e = (h >> 10) & 31, m = h & 1023;
    double v;
    if (e == 31) v = m ? NAN : INFINITY;
    else if (e == 0) v = ldexp((double)m, -24);
    else v = ldexp((double)(1024 + m), e - 25);
    return (float)((h & 0x8000) ? -v : v);
}

static float ref_w(int dt, const void *row, int64_t i) {
    if (dt == HEARTH_F32) return ((const float *)row)[i];
    if (dt == HEARTH_F16) return ref_f16(((const uint16_t *)row)[i]);
    return bits_f32((uint32_t)((const uint16_t *)row)[i] << 16);
}

/* NUMERICS §1 dot16, written independently of quant.c. */
static float ref_dot16_w(int dt, const void *row, const float *x, int64_t n) {
    float lane[16];
    for (int j = 0; j < 16; j++) lane[j] = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float p = ref_w(dt, row, i) * x[i];
        lane[i % 16] = lane[i % 16] + p;
    }
    for (int s = 8; s > 0; s /= 2)
        for (int j = 0; j < s; j++) lane[j] = lane[j] + lane[j + s];
    return lane[0];
}

/* NUMERICS §2 for finite input. */
static void ref_act(const float *x, int64_t n, hx_act_q8 *out) {
    for (int64_t b = 0; b < n / 64; b++) {
        float amax = 0.0f, d, id;
        int32_t sum = 0;
        for (int i = 0; i < 64; i++) amax = fmaxf(amax, fabsf(x[b * 64 + i]));
        d = amax / 127.0f;
        id = d != 0.0f ? 1.0f / d : 0.0f;
        memset(&out[b], 0, sizeof out[b]);
        for (int i = 0; i < 64; i++) {
            float q = nearbyintf(x[b * 64 + i] * id);
            out[b].q[i] = (int8_t)q;
            sum += (int32_t)q;
        }
        out[b].d = d;
        out[b].sum = sum;
    }
}

/* NUMERICS §3 for Q8/Q4 rows, decoding FORMAT.md §6 byte layout directly. */
static float ref_qrow(int dt, const uint8_t *row, const hx_act_q8 *a, int64_t n) {
    float acc = 0.0f;
    const size_t bs = dt == HEARTH_Q8 ? 66 : 34;
    for (int64_t g = 0; g < n / 64; g++) {
        const uint8_t *blk = row + (size_t)g * bs;
        uint16_t dh = (uint16_t)(blk[0] | (blk[1] << 8));
        int32_t isum = 0;
        for (int i = 0; i < 64; i++) {
            int wq;
            if (dt == HEARTH_Q8) wq = (int8_t)blk[2 + i];
            else wq = (i < 32 ? (blk[2 + i] & 15) : (blk[2 + i - 32] >> 4)) - 8;
            isum += wq * a[g].q[i];
        }
        acc = acc + (float)isum * (ref_f16(dh) * a[g].d);
    }
    return acc;
}

/* ------------------------------------------------------------ helpers */

static const char *isa_name(int isa) {
    return isa == 1 ? "scalar" : isa == 2 ? "avx2" : isa == 3 ? "avx512" : "?";
}

static int n_isas;
static const hx_kernels *isas[3];

static void *xalloc(size_t n) {
    void *p = hx_aligned_alloc(64, n ? n : 64);
    if (!p) { printf("out of memory (%zu bytes)\n", n); exit(2); }
    return p;
}

/* Input kinds shared by weights and activations. */
enum { K_GAUSS, K_ZERO_BLOCKS, K_OUTLIER, K_MAX, K_TIES, K_TINY, K_CANCEL, K_COUNT };
static const char *kind_name[] = {"gauss", "zero-blocks", "outlier", "max", "ties", "tiny", "cancel"};

static void fill(float *x, int64_t n, int kind, int is_quant_target) {
    for (int64_t i = 0; i < n; i++) x[i] = rnd_gauss();
    switch (kind) {
    case K_ZERO_BLOCKS:
        for (int64_t i = 0; i < n; i++) if ((i / 64) % 2 == 0) x[i] = 0.0f;
        break;
    case K_OUTLIER:
        x[rnd64() % (uint64_t)n] = (rnd64() & 1) ? 3.0e4f : -3.0e4f;
        break;
    case K_MAX: {
        /* extreme but finite: quant targets stay inside the f16 scale range */
        float m = is_quant_target ? 60000.0f : 1.0e18f;
        for (int64_t i = 0; i < n; i++) {
            uint64_t r = rnd64() % 4;
            x[i] = r == 0 ? m : r == 1 ? -m : x[i] * m * 0.5f;
        }
        break;
    }
    case K_TIES:
        /* blocks whose amax is 127, so d = 1 and x*id lands exactly on .5 */
        for (int64_t i = 0; i < n; i++) {
            int k = (int)(rnd64() % 253) - 126;
            x[i] = (float)k + ((rnd64() & 1) ? 0.5f : -0.5f);
            if (x[i] > 127.0f) x[i] = 127.0f;
            if (x[i] < -127.0f) x[i] = -127.0f;
            if (i % 64 == 0) x[i] = (rnd64() & 1) ? 127.0f : -127.0f;
        }
        break;
    case K_TINY:
        for (int64_t i = 0; i < n; i++) {
            uint64_t r = rnd64() % 3;
            x[i] = r == 0 ? x[i] * 1e-30f : r == 1 ? x[i] * 1e-41f : 0.0f;
        }
        break;
    case K_CANCEL:
        for (int64_t i = 0; i + 1 < n; i += 2) x[i + 1] = -x[i];
        break;
    default:
        break;
    }
}

/* ------------------------------------------------------------ 1. sizes and API */

static void test_sizes(void) {
    CHECK(hx_row_bytes(HEARTH_F32, 10) == 40, "f32 row bytes");
    CHECK(hx_row_bytes(HEARTH_F16, 7) == 14, "f16 row bytes");
    CHECK(hx_row_bytes(HEARTH_BF16, 7) == 14, "bf16 row bytes");
    CHECK(hx_row_bytes(HEARTH_Q8, 64) == 66 && hx_row_bytes(HEARTH_Q8, 4096) == 66 * 64, "q8 row bytes");
    CHECK(hx_row_bytes(HEARTH_Q4, 64) == 34 && hx_row_bytes(HEARTH_Q4, 7168) == 34 * 112, "q4 row bytes");
    CHECK(hx_row_bytes(HEARTH_Q8, 65) == 0 && hx_row_bytes(HEARTH_Q4, 100) == 0, "quant needs n%%64 == 0");
    CHECK(hx_row_bytes(HEARTH_I32, 3) == 12 && hx_row_bytes(HEARTH_U8, 3) == 3, "i32/u8 row bytes");
    CHECK(hx_row_bytes(HEARTH_F32, 0) == 0 && hx_row_bytes(HEARTH_F32, -5) == 0, "n <= 0");
    CHECK(hx_row_bytes(7, 64) == 0 && hx_row_bytes(-1, 64) == 0, "invalid dtype");
    CHECK(hearth_row_bytes(HEARTH_Q4, 128) == 68, "public row bytes");
    CHECK(hx_act_bytes(HEARTH_Q4, 128) == 144 && hx_act_bytes(HEARTH_F16, 10) == 40, "act bytes");
    CHECK(hx_act_bytes(HEARTH_Q8, 70) == 0 && hx_act_bytes(HEARTH_F32, 0) == 0, "act bytes invalid");
    CHECK(hx_act_bytes(HEARTH_F32, 1) == 4 && hx_act_bytes(HEARTH_BF16, 3) == 12, "act bytes of short float vectors");
    {   /* rows are limited to 2^40 elements so byte counts cannot overflow */
        const int64_t big = (int64_t)1 << 40;
        CHECK(hx_row_bytes(HEARTH_F32, big) == ((size_t)4 << 40) && hx_row_bytes(HEARTH_F32, big + 1) == 0, "f32 row limit");
        CHECK(hx_row_bytes(HEARTH_Q4, big) == (size_t)(big / 64) * 34 && hx_row_bytes(HEARTH_Q4, big + 64) == 0, "q4 row limit");
        CHECK(hx_row_bytes(HEARTH_U8, big - 1) == (size_t)(big - 1), "u8 row near limit");
    }
    CHECK(hx_dtype_valid(HEARTH_U8) && !hx_dtype_valid(7) && !hx_dtype_valid(-1), "dtype valid");
    CHECK(hx_dtype_is_quant(HEARTH_Q4) && hx_dtype_is_quant(HEARTH_Q8) && !hx_dtype_is_quant(HEARTH_BF16), "is quant");
    CHECK(strcmp(hx_dtype_name(HEARTH_Q4), "q4") == 0 && strcmp(hx_dtype_name(HEARTH_BF16), "bf16") == 0, "names");
    CHECK(strcmp(hearth_version(), "0.1.0") == 0, "version");
    {   /* argument errors: each bad argument alone is rejected and Y is not written */
        float x[256] = {0}, y[4], ys[4];
        uint8_t w[1024] = {0};
        for (int i = 0; i < 4; i++) y[i] = ys[i] = -3.0f;
        CHECK(hearth_matmul(HEARTH_I32, w, 1, 64, x, 1, y, 0) == -1, "matmul i32 rejected");
        CHECK(hearth_matmul(HEARTH_U8, w, 1, 64, x, 1, y, 1) == -1, "matmul u8 rejected");
        CHECK(hearth_matmul(9, w, 1, 64, x, 1, y, 0) == -1, "matmul bad dtype");
        CHECK(hearth_matmul(-1, w, 1, 64, x, 1, y, 0) == -1, "matmul negative dtype");
        CHECK(hearth_matmul(HEARTH_Q4, w, 1, 63, x, 1, y, 0) == -1, "matmul bad cols");
        CHECK(hearth_matmul(HEARTH_F32, w, 1, 0, x, 1, y, 0) == -1, "matmul cols 0");
        CHECK(hearth_matmul(HEARTH_Q4, w, 1, 64, x, 1, y, 7) == -1, "matmul bad isa");
        CHECK(hearth_matmul(HEARTH_Q4, w, 1, 64, x, 1, y, 4) == -1, "matmul isa 4");
        CHECK(hearth_matmul(HEARTH_Q4, w, 1, 64, x, 1, y, -1) == -1, "matmul isa -1");
        CHECK(hearth_matmul(HEARTH_Q4, w, -1, 64, x, 1, y, 0) == -1, "matmul n_rows < 0");
        CHECK(hearth_matmul(HEARTH_Q4, w, 1, 64, x, -1, y, 0) == -1, "matmul T < 0");
        CHECK(hearth_matmul(HEARTH_Q4, NULL, 1, 64, x, 1, y, 0) == -1, "matmul W NULL");
        CHECK(hearth_matmul(HEARTH_Q4, w, 1, 64, NULL, 1, y, 0) == -1, "matmul X NULL");
        CHECK(hearth_matmul(HEARTH_Q4, w, 1, 64, x, 1, NULL, 0) == -1, "matmul Y NULL");
        CHECK(hearth_matmul(HEARTH_F32, NULL, 2, 16, x, 1, y, HEARTH_ISA_SCALAR) == -1, "matmul f32 W NULL");
        CHECK(hearth_matmul(HEARTH_F32, w, 2, 16, NULL, 1, y, HEARTH_ISA_SCALAR) == -1, "matmul f32 X NULL");
        CHECK(memcmp(y, ys, sizeof y) == 0, "rejected matmul wrote Y");
        CHECK(hearth_matmul(HEARTH_Q4, w, 1, 64, x, 0, y, 0) == 0, "matmul T=0");
        CHECK(hearth_matmul(HEARTH_Q4, NULL, 0, 64, NULL, 1, NULL, 0) == 0, "matmul 0 rows: nothing to read");
        CHECK(hearth_matmul(HEARTH_Q4, NULL, 1, 64, NULL, 0, NULL, 0) == 0, "matmul T=0: nothing to read");
        CHECK(memcmp(y, ys, sizeof y) == 0, "empty matmul wrote Y");
        CHECK(hearth_matmul(HEARTH_Q4, w, 4, 64, x, 1, y, 0) == 0 && y[0] == 0.0f && y[3] == 0.0f, "matmul zero");
        /* T activation buffers of a 2^40-column row exceed size_t: -3, nothing allocated or read.
         * With T = 2^30 the byte count wraps to exactly 0, so only the guard prevents an overflow. */
        CHECK(hearth_matmul(HEARTH_Q8, w, 1, (int64_t)1 << 40, x, INT32_MAX, y, HEARTH_ISA_SCALAR) == -3, "matmul activation size overflow");
        CHECK(hearth_matmul(HEARTH_Q4, w, 1, (int64_t)1 << 40, x, 1 << 30, y, HEARTH_ISA_SCALAR) == -3, "matmul activation size wraps to 0");
        for (int isa = 1; isa <= 3; isa++)
            CHECK((hearth_matmul(HEARTH_Q4, w, 1, 64, x, 1, y, isa) == -2) == (hx_kernels_for(isa) == NULL),
                  "matmul -2 iff ISA unavailable (%s)", isa_name(isa));

        CHECK(hearth_quantize(HEARTH_I32, x, 1, 64, w, 1) == -1, "quantize i32 rejected");
        CHECK(hearth_quantize(HEARTH_U8, x, 1, 64, w, 1) == -1, "quantize u8 rejected");
        CHECK(hearth_quantize(-1, x, 1, 64, w, 1) == -1, "quantize negative dtype");
        CHECK(hearth_quantize(HEARTH_Q8, x, 1, 60, w, 1) == -1, "quantize bad cols");
        CHECK(hearth_quantize(HEARTH_F32, x, 1, 0, w, 1) == -1, "quantize cols 0");
        CHECK(hearth_quantize(HEARTH_Q8, x, -1, 64, w, 1) == -1, "quantize n_rows < 0");
        CHECK(hearth_quantize(HEARTH_Q8, NULL, 1, 64, w, 1) == -1, "quantize src NULL");
        CHECK(hearth_quantize(HEARTH_Q8, x, 1, 64, NULL, 1) == -1, "quantize dst NULL");
        CHECK(hearth_quantize(HEARTH_Q8, NULL, 0, 64, NULL, 1) == 0, "quantize 0 rows");
        for (int i = 0; i < 128; i++) x[i] = (float)(i - 50) * 0.25f;
        memset(w, 0xee, sizeof w);
        CHECK(hearth_quantize(HEARTH_F32, x, 2, 64, w, -3) == 0 && memcmp(w, x, 512) == 0 && w[512] == 0xee,
              "quantize negative n_threads = auto");
        CHECK(hearth_quantize(HEARTH_F32, x, 2, 64, w, 99) == 0 && memcmp(w, x, 512) == 0, "quantize more threads than rows");

        CHECK(hearth_dequantize(HEARTH_Q8, w, 1, 60, x) == -1, "dequantize bad cols");
        CHECK(hearth_dequantize(7, w, 1, 64, x) == -1 && hearth_dequantize(-1, w, 1, 64, x) == -1, "dequantize bad dtype");
        CHECK(hearth_dequantize(HEARTH_Q8, w, -1, 64, x) == -1, "dequantize n_rows < 0");
        CHECK(hearth_dequantize(HEARTH_Q8, NULL, 1, 64, x) == -1, "dequantize src NULL");
        CHECK(hearth_dequantize(HEARTH_Q8, w, 1, 64, NULL) == -1, "dequantize dst NULL");
        CHECK(hearth_dequantize(HEARTH_Q8, NULL, 0, 64, NULL) == 0, "dequantize 0 rows");
    }
    {   /* I32 / U8 dequantize to (float)value, row by row */
        const int32_t iv[6] = {-5, 0, 7, 123456789, INT32_MIN, INT32_MAX};
        const uint32_t iw[6] = {0xc0a00000u, 0x00000000u, 0x40e00000u, 0x4ceb79a3u, 0xcf000000u, 0x4f000000u};
        const uint8_t uv[6] = {0, 1, 2, 127, 200, 255};
        const float uw[6] = {0.0f, 1.0f, 2.0f, 127.0f, 200.0f, 255.0f};
        float o[7];
        int bad = 0;
        o[6] = -9.0f;
        CHECK(hearth_dequantize(HEARTH_I32, iv, 2, 3, o) == 0, "dequantize i32 rc");
        for (int i = 0; i < 6; i++) bad += f32_bits(o[i]) != iw[i];
        CHECK(hearth_dequantize(HEARTH_U8, uv, 3, 2, o) == 0, "dequantize u8 rc");
        for (int i = 0; i < 6; i++) bad += f32_bits(o[i]) != f32_bits(uw[i]);
        CHECK(bad == 0 && o[6] == -9.0f, "i32/u8 dequantized values (%d wrong)", bad);
    }
    {   /* hx_act_prepare: float dtypes use x itself; quant dtypes get NUMERICS §2 blocks in act */
        float x[256];
        HX_ALIGNED(64) uint8_t act[4 * sizeof(hx_act_q8) + 64];
        hx_act_q8 ref[4];
        for (int i = 0; i < 256; i++) x[i] = rnd_gauss();
        hx_kernels_scalar.act_quantize_q8(x, ref, 256);
        for (int dt = HEARTH_F32; dt <= HEARTH_U8; dt++) {
            const void *p;
            int untouched = 1;
            memset(act, 0x5a, sizeof act);
            p = hx_act_prepare(dt, x, 256, act);
            if (hx_dtype_is_quant(dt)) {
                CHECK(p == (const void *)act, "act_prepare %s returns act", hx_dtype_name(dt));
                CHECK(memcmp(act, ref, sizeof ref) == 0, "act_prepare %s == act_quantize_q8", hx_dtype_name(dt));
                for (size_t i = sizeof ref; i < sizeof act; i++) untouched &= act[i] == 0x5a;
                CHECK(untouched, "act_prepare %s wrote past n/64 blocks", hx_dtype_name(dt));
            } else {
                CHECK(p == (const void *)x, "act_prepare %s returns x", hx_dtype_name(dt));
                for (size_t i = 0; i < sizeof act; i++) untouched &= act[i] == 0x5a;
                CHECK(untouched, "act_prepare %s must not write act", hx_dtype_name(dt));
            }
        }
    }
    {   /* in-process dispatch: auto = best; an explicit runnable HEARTH_ISA is honoured */
        const char *env = getenv("HEARTH_ISA");
        const hx_kernels *k = hx_kernels_get();
        int want = hearth_cpu_isa();
        CHECK(k != NULL && hx_kernels_get() == k, "kernels_get stable");
        if (env && *env) {
            int req = 0;
            char low[16] = {0};
            for (int i = 0; env[i] && i < 15; i++) low[i] = (char)(env[i] >= 'A' && env[i] <= 'Z' ? env[i] + 32 : env[i]);
            if (!strcmp(low, "scalar")) req = HEARTH_ISA_SCALAR;
            else if (!strcmp(low, "avx2")) req = HEARTH_ISA_AVX2;
            else if (!strcmp(low, "avx512")) req = HEARTH_ISA_AVX512;
            if (req && hx_kernels_for(req)) want = req;
        }
        CHECK(k && k->isa == want, "dispatch with HEARTH_ISA=%s: got %s, want %s", env ? env : "(unset)",
              k ? isa_name(k->isa) : "NULL", isa_name(want));
        CHECK(hx_kernels_for(HEARTH_ISA_SCALAR) == &hx_kernels_scalar, "scalar table");
        CHECK(hx_kernels_for(HEARTH_ISA_AUTO) == NULL && hx_kernels_for(4) == NULL && hx_kernels_for(-1) == NULL,
              "kernels_for invalid isa");
        CHECK(hearth_matmul(HEARTH_Q4, NULL, 0, 64, NULL, 1, NULL, HEARTH_ISA_AUTO) == 0, "auto isa accepted");
    }
}

/* ------------------------------------------------------------ 1b. ISA tables and gating */

#if defined(HX_ARCH_X86_64)
static int cpu_has_avx2(const hx_cpu *c) { return c->avx2 && c->fma && c->f16c; }
static int cpu_has_avx512(const hx_cpu *c) {
    return cpu_has_avx2(c) && c->avx512f && c->avx512bw && c->avx512vl && c->avx512vnni;
}
#endif

/* Tables are complete, and SIMD kernels are in use whenever this CPU can run them
 * (a build or dispatch regression must not silently fall back to scalar). */
static void test_isa_tables(void) {
    const hx_kernels *t[3];
    t[0] = &hx_kernels_scalar;
    t[1] = hx_kernels_avx2_table();
    t[2] = hx_kernels_avx512_table();
#if defined(HX_ARCH_X86_64)
    {
        const hx_cpu *c = hx_cpu_features();
        const int has2 = cpu_has_avx2(c), has512 = cpu_has_avx512(c);
        CHECK(t[1] != NULL && t[2] != NULL, "x86-64 build must compile in the AVX2 and AVX-512 kernels");
        CHECK((hx_kernels_for(HEARTH_ISA_AVX2) != NULL) == has2, "AVX2 kernels available iff CPU has avx2+fma+f16c");
        CHECK((hx_kernels_for(HEARTH_ISA_AVX512) != NULL) == has512,
              "AVX-512 kernels available iff CPU has avx512f/bw/vl/vnni (+avx2/fma/f16c)");
        CHECK(hearth_cpu_isa() == (has512 ? HEARTH_ISA_AVX512 : has2 ? HEARTH_ISA_AVX2 : HEARTH_ISA_SCALAR),
              "hearth_cpu_isa = best ISA (%d)", hearth_cpu_isa());
        CHECK(n_isas == 1 + has2 + has512, "ISAs under test (%d)", n_isas);
    }
#else
    CHECK(t[1] == NULL && t[2] == NULL && hearth_cpu_isa() == HEARTH_ISA_SCALAR, "no x86 kernels on this architecture");
#endif
    for (int k = 0; k < 3; k++) {
        if (!t[k]) continue;
        CHECK(t[k]->isa == k + 1, "table %d isa field %d", k, t[k]->isa);
        CHECK(t[k]->act_quantize_q8 != NULL, "table %s act_quantize_q8", isa_name(k + 1));
        for (int dt = HEARTH_F32; dt <= HEARTH_U8; dt++)
            CHECK((t[k]->matmul[dt] != NULL) == (dt <= HEARTH_Q4), "table %s matmul[%s]", isa_name(k + 1), hx_dtype_name(dt));
    }
}

/* Every required CPU feature gates its ISA. hx_cpu_features() returns the platform
 * layer's cached, non-const detection result (a static hx_cpu in platform_win.c /
 * platform_posix.c), so clearing one bit here simulates a CPU without that
 * feature; everything is restored before returning. Runs single-threaded. */
static void test_isa_gating(void) {
#if defined(HX_ARCH_X86_64)
    hx_cpu *c = (hx_cpu *)hx_cpu_features();
    const hx_cpu saved = *c;
    int *bit[7];
    static const char *name[7] = {"avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512vl", "avx512vnni"};
    float x[64] = {0}, y[1];
    uint8_t w[34] = {0};
    int tested = 0;
    bit[0] = &c->avx2; bit[1] = &c->fma; bit[2] = &c->f16c; bit[3] = &c->avx512f;
    bit[4] = &c->avx512bw; bit[5] = &c->avx512vl; bit[6] = &c->avx512vnni;
    for (int b = 0; b < 7; b++) {
        const int avx2_left = b >= 3 && cpu_has_avx2(&saved);
        if (!*bit[b]) continue;
        *bit[b] = 0;
        tested++;
        CHECK(hx_kernels_for(HEARTH_ISA_AVX512) == NULL, "AVX-512 kernels offered without %s", name[b]);
        CHECK(hearth_matmul(HEARTH_Q4, w, 1, 64, x, 1, y, HEARTH_ISA_AVX512) == -2, "matmul avx512 without %s: not -2", name[b]);
        CHECK((hx_kernels_for(HEARTH_ISA_AVX2) != NULL) == avx2_left, "AVX2 availability without %s", name[b]);
        CHECK((hearth_matmul(HEARTH_Q4, w, 1, 64, x, 1, y, HEARTH_ISA_AVX2) == -2) == !avx2_left,
              "matmul avx2 without %s", name[b]);
        CHECK(hearth_cpu_isa() == (avx2_left ? HEARTH_ISA_AVX2 : HEARTH_ISA_SCALAR), "best ISA without %s", name[b]);
        CHECK(hearth_matmul(HEARTH_Q4, w, 1, 64, x, 1, y, HEARTH_ISA_SCALAR) == 0, "scalar always runs");
        *c = saved;
    }
    CHECK(hearth_cpu_isa() == (cpu_has_avx512(c) ? HEARTH_ISA_AVX512 : cpu_has_avx2(c) ? HEARTH_ISA_AVX2 : HEARTH_ISA_SCALAR),
          "features restored");
    printf("  ISA gating: %d feature bits checked\n", tested);
#endif
}

/* Child mode (--dispatch-child WANT MASK NWARN): the parent sets HEARTH_ISA; MASK
 * clears a feature first ("no-avx512", "no-avx2" or "-"). Exit 0 iff the dispatched
 * ISA is WANT and exactly NWARN HEARTH_ISA warnings were logged over two
 * hx_kernels_get() calls (the choice is made, and warned about, once). */
static int dispatch_child(const char *self, int want, const char *mask, int nwarn) {
    const hx_kernels *k, *k2;
    char logp[1200], line[512];
    int warns = 0;
    FILE *f;
#if defined(HX_ARCH_X86_64)
    hx_cpu *c = (hx_cpu *)hx_cpu_features();
    if (!strcmp(mask, "no-avx512")) c->avx512vnni = 0;
    if (!strcmp(mask, "no-avx2")) c->avx2 = 0;
#endif
    snprintf(logp, sizeof logp, "%s.stderr.txt", self);
    if (!freopen(logp, "w", stderr)) { printf("  dispatch child: cannot redirect stderr to %s\n", logp); return 3; }
    hx_set_log_level(HX_LOG_WARN);
    k = hx_kernels_get();
    k2 = hx_kernels_get();
    fflush(stderr);
    f = fopen(logp, "r");
    while (f && fgets(line, sizeof line, f)) warns += strstr(line, "HEARTH_ISA") != NULL;
    if (f) fclose(f);
    fclose(stderr);
    remove(logp);
    if (!k || k != k2 || k->isa != want || warns != nwarn) {
        const char *env = getenv("HEARTH_ISA");
        printf("  dispatch child: HEARTH_ISA=%s mask %s gave %s with %d warnings, want %s with %d\n", env ? env : "(unset)",
               mask, k ? isa_name(k->isa) : "NULL", warns, isa_name(want), nwarn);
        return 1;
    }
    return 0;
}

static void test_dispatch_env(const char *self) {
    char exe[1100], cmd[2600], saved[256];
    const char *old = getenv("HEARTH_ISA");
    const int had = old != NULL;
    const int best = hearth_cpu_isa(), has2 = hx_kernels_for(HEARTH_ISA_AVX2) != NULL;
    const int has512 = hx_kernels_for(HEARTH_ISA_AVX512) != NULL;
    struct { const char *env, *mask; int want, nwarn; } cases[] = {   /* nwarn: unknown or unsupported value */
        {"scalar", "-", HEARTH_ISA_SCALAR, 0},
        {"avx2", "-", has2 ? HEARTH_ISA_AVX2 : best, !has2},
        {"AVX512", "-", has512 ? HEARTH_ISA_AVX512 : best, !has512},
        {"Avx2", "-", has2 ? HEARTH_ISA_AVX2 : best, !has2},
        {"auto", "-", best, 0},
        {"", "-", best, 0},
        {"bogus", "-", best, 1},
        {"avx", "-", best, 1},
        {"avx2x", "-", best, 1},
        {"scalar2", "-", best, 1},
        {"avx512", "no-avx512", has2 ? HEARTH_ISA_AVX2 : HEARTH_ISA_SCALAR, 1},   /* unsupported: falls back */
        {"avx2", "no-avx2", HEARTH_ISA_SCALAR, 1},
    };
    int fails = 0;
    snprintf(exe, sizeof exe, "%s", self ? self : "");
    if (!hx_path_exists(exe)) snprintf(exe, sizeof exe, "%s.exe", self ? self : "");
    CHECK(hx_path_exists(exe), "cannot locate this executable from argv[0] (%s); run test_quant by its full path", self);
    if (!hx_path_exists(exe)) return;
    snprintf(saved, sizeof saved, "%s", old ? old : "");
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        int rc;
#if defined(HX_OS_WINDOWS)
        snprintf(cmd, sizeof cmd, "\"\"%s\" --dispatch-child %d %s %d\"", exe, cases[i].want, cases[i].mask, cases[i].nwarn);
#else
        snprintf(cmd, sizeof cmd, "'%s' --dispatch-child %d %s %d", exe, cases[i].want, cases[i].mask, cases[i].nwarn);
#endif
        set_env("HEARTH_ISA", cases[i].env);
        fflush(stdout);
        rc = system(cmd);
        CHECK(rc == 0, "HEARTH_ISA=\"%s\" (mask %s): child exit %d, want dispatch %s", cases[i].env, cases[i].mask, rc,
              isa_name(cases[i].want));
        fails += rc != 0;
    }
    set_env("HEARTH_ISA", had ? saved : NULL);
    printf("  HEARTH_ISA dispatch: %d child runs, %d failed\n", (int)(sizeof cases / sizeof cases[0]), fails);
}

/* ------------------------------------------------------------ 2. quantizers */

static double q4_block_err_baseline(const float *x) {
    float amax = 0.0f, d, id;
    double e = 0.0;
    uint16_t h;
    for (int i = 0; i < 64; i++) amax = fmaxf(amax, fabsf(x[i]));
    h = hx_f32_to_f16(amax / 7.0f);
    d = ref_f16(h);
    id = d != 0.0f ? 1.0f / d : 0.0f;   /* same rounding rule as the quantizer: rint(x * (1/d)) */
    for (int i = 0; i < 64; i++) {
        float v = nearbyintf(x[i] * id);
        double r;
        v = v < -8.0f ? -8.0f : v > 7.0f ? 7.0f : v;
        r = (double)x[i] - (double)d * v;
        e += r * r;
    }
    return e;
}

static void test_quantizers(void) {
    const int64_t rows = g_small ? 64 : 256, cols = 4096, n = rows * cols;
    float *x = (float *)xalloc((size_t)n * 4), *y = (float *)xalloc((size_t)n * 4);
    uint8_t *q = (uint8_t *)xalloc(hx_row_bytes(HEARTH_F32, cols) * (size_t)rows);
    uint8_t *q2 = (uint8_t *)xalloc(hx_row_bytes(HEARTH_F32, cols) * (size_t)rows);
    double se4 = 0, se4b = 0, se8 = 0, sx = 0;
    int worse = 0;

    for (int64_t i = 0; i < n; i++) x[i] = rnd_gauss();
    for (int64_t i = 0; i < n; i++) sx += (double)x[i] * x[i];

    /* F32 exact; F16 / BF16 within half an ulp of their format */
    CHECK(hearth_quantize(HEARTH_F32, x, rows, cols, q, 0) == 0 && hearth_dequantize(HEARTH_F32, q, rows, cols, y) == 0, "f32 rt");
    CHECK(memcmp(x, y, (size_t)n * 4) == 0, "f32 round trip exact");
    hearth_quantize(HEARTH_F16, x, rows, cols, q, 0);
    hearth_dequantize(HEARTH_F16, q, rows, cols, y);
    {
        int bad = 0;
        for (int64_t i = 0; i < n; i++) {
            double ax = fabs((double)x[i]), tol = ax >= 6.103515625e-05 ? ax * 0x1p-11 : 0x1p-25;
            if (fabs((double)x[i] - y[i]) > tol) bad++;
            if (ref_f16(((uint16_t *)q)[i]) != y[i]) bad++;
        }
        CHECK(bad == 0, "f16 round trip error (%d bad)", bad);
    }
    {   /* signalling NaN comes back quiet with its payload, as F16C does */
        uint16_t h[2] = {0x7c01, 0xfe55};
        float o[2];
        hx_dequantize_row(HEARTH_F16, h, o, 2);
        CHECK(f32_bits(o[0]) == 0x7fc02000u && f32_bits(o[1]) == 0xffcaa000u, "f16 NaN conversion %08x %08x",
              (unsigned)f32_bits(o[0]), (unsigned)f32_bits(o[1]));
    }
    hearth_quantize(HEARTH_BF16, x, rows, cols, q, 0);
    hearth_dequantize(HEARTH_BF16, q, rows, cols, y);
    {
        int bad = 0;
        for (int64_t i = 0; i < n; i++)
            if (fabs((double)x[i] - y[i]) > fabs((double)x[i]) * 0x1p-8) bad++;
        CHECK(bad == 0, "bf16 round trip error (%d bad)", bad);
    }

    /* Q8: conforms to NUMERICS §6 exactly; |x - w| <= d/2 */
    hearth_quantize(HEARTH_Q8, x, rows, cols, q, 0);
    hearth_dequantize(HEARTH_Q8, q, rows, cols, y);
    {
        int bad_spec = 0, bad_err = 0;
        const hx_block_q8 *b = (const hx_block_q8 *)q;
        for (int64_t k = 0; k < n / 64; k++) {
            float amax = 0.0f, d;
            for (int i = 0; i < 64; i++) amax = fmaxf(amax, fabsf(x[k * 64 + i]));
            if (b[k].d != hx_f32_to_f16(amax / 127.0f)) bad_spec++;
            d = ref_f16(b[k].d);
            for (int i = 0; i < 64; i++) {
                float v = nearbyintf(x[k * 64 + i] / d);
                v = v > 127.0f ? 127.0f : v < -127.0f ? -127.0f : v;
                if ((int)v != b[k].q[i]) bad_spec++;
                if (fabs((double)x[k * 64 + i] - y[k * 64 + i]) > 0.5 * d * (1.0 + 1e-6)) bad_err++;
                if (y[k * 64 + i] != d * (float)b[k].q[i]) bad_spec++;
                se8 += ((double)x[k * 64 + i] - y[k * 64 + i]) * ((double)x[k * 64 + i] - y[k * 64 + i]);
            }
        }
        CHECK(bad_spec == 0, "q8 differs from NUMERICS §6 (%d)", bad_spec);
        CHECK(bad_err == 0, "q8 error above d/2 (%d)", bad_err);
    }

    /* Q4: never worse than the absmax baseline, block by block */
    hearth_quantize(HEARTH_Q4, x, rows, cols, q, 0);
    hearth_dequantize(HEARTH_Q4, q, rows, cols, y);
    for (int64_t k = 0; k < n / 64; k++) {
        double e = 0.0, eb = q4_block_err_baseline(x + k * 64);
        for (int i = 0; i < 64; i++) {
            double r = (double)x[k * 64 + i] - y[k * 64 + i];
            e += r * r;
        }
        se4 += e;
        se4b += eb;
        if (e > eb * (1.0 + 1e-9) + 1e-30) worse++;
    }
    CHECK(worse == 0, "q4 worse than absmax baseline in %d blocks", worse);
    {
        double rms = sqrt(sx / (double)n);
        double r4 = sqrt(se4 / (double)n), r4b = sqrt(se4b / (double)n), r8 = sqrt(se8 / (double)n);
        printf("  RMSE on N(0,1), %lld values (signal rms %.4f):\n", (long long)n, rms);
        printf("    Q4 scale search  %.6f   (absmax/7 baseline %.6f, %.1f%% lower; SNR %.2f dB vs %.2f dB)\n",
               r4, r4b, 100.0 * (1.0 - r4 / r4b), 20.0 * log10(rms / r4), 20.0 * log10(rms / r4b));
        printf("    Q8 absmax/127    %.6f   (SNR %.2f dB)\n", r8, 20.0 * log10(rms / r8));
        CHECK(r4 < r4b, "q4 rmse should beat baseline");
    }

    /* deterministic and thread-count independent, incl. splits with remainder rows */
    for (int dt = HEARTH_F32; dt <= HEARTH_Q4; dt++) {
        size_t bytes = hx_row_bytes(dt, cols) * (size_t)rows;
        const int64_t rows2 = 67, cols2 = 2048;
        const size_t bytes2 = hx_row_bytes(dt, cols2) * (size_t)rows2;
        hearth_quantize(dt, x, rows, cols, q, 1);
        hearth_quantize(dt, x, rows, cols, q2, 0);
        CHECK(memcmp(q, q2, bytes) == 0, "quantize %s: 1 thread != auto", hx_dtype_name(dt));
        hearth_quantize(dt, x, rows, cols, q2, 7);
        CHECK(memcmp(q, q2, bytes) == 0, "quantize %s: 1 thread != 7", hx_dtype_name(dt));
        hearth_quantize(dt, x, rows2, cols2, q, 1);
        for (int nt = 2; nt <= 3; nt++) {
            memset(q2, 0xcd, bytes2 + 64);
            hearth_quantize(dt, x, rows2, cols2, q2, nt);
            CHECK(memcmp(q, q2, bytes2) == 0 && q2[bytes2] == 0xcd, "quantize %s: 67 rows, 1 thread != %d", hx_dtype_name(dt), nt);
        }
    }

    /* adversarial blocks */
    {
        float b[64 * 8], o[64 * 8];
        hx_block_q4 *b4 = (hx_block_q4 *)q;
        hx_block_q8 *b8 = (hx_block_q8 *)q2;
        for (int i = 0; i < 64 * 8; i++) b[i] = rnd_gauss();
        for (int i = 0; i < 64; i++) b[i] = 0.0f;                       /* block 0: zeros */
        b[64 + 5] = 1.0e4f;                                             /* block 1: outlier */
        for (int i = 128; i < 192; i++) b[i] = (i & 1) ? 65504.0f * 7.0f : -65504.0f * 8.0f; /* block 2: max */
        for (int i = 192; i < 256; i++) b[i] *= 1e-30f;                 /* block 3: tiny */
        b[256 + 3] = NAN;                                               /* block 4: NaN */
        b[320 + 9] = INFINITY;                                          /* block 5: inf */
        for (int i = 384; i < 448; i++) b[i] = 1.0e9f * ((i & 1) ? 1.0f : -1.0f);  /* block 6: beyond f16 */
        hx_quantize_row(HEARTH_Q4, b, b4, 64 * 8);
        hx_quantize_row(HEARTH_Q8, b, b8, 64 * 8);
        CHECK(b4[0].d == 0 && b8[0].d == 0, "zero block scale");
        for (int j = 0; j < 32; j++) CHECK(b4[0].qs[j] == 0x88, "zero block q4 nibbles");
        for (int i = 0; i < 64; i++) CHECK(b8[0].q[i] == 0, "zero block q8");
        hx_dequantize_row(HEARTH_Q4, b4, o, 64 * 8);
        CHECK(fabsf(o[64 + 5] - 1.0e4f) < 1.0e4f / 14.0f, "q4 outlier kept (%g)", o[64 + 5]);
        for (int i = 128; i < 192; i++) CHECK(fabsf(o[i] - b[i]) <= fabsf(b[i]) * 0.07f, "q4 max block %d: %g vs %g", i, o[i], b[i]);
        for (int i = 192; i < 256; i++) CHECK(fabsf(o[i] - b[i]) <= 1e-29f, "q4 tiny");
        for (int i = 0; i < 64 * 8; i++) CHECK(o[i] == o[i] && fabsf(o[i]) < FLT_MAX, "q4 finite output");
        hx_dequantize_row(HEARTH_Q8, b8, o, 64 * 8);
        CHECK(fabsf(o[64 + 5] - 1.0e4f) <= 1.0e4f / 254.0f + 8.0f, "q8 outlier kept (%g)", o[64 + 5]);
        for (int i = 0; i < 64 * 8; i++) CHECK(o[i] == o[i] && fabsf(o[i]) < FLT_MAX, "q8 finite output");
        CHECK(b8[6].d == 0x7bff && b4[6].d != 0x7c00 && b4[6].d != 0xfc00, "scale saturates at max f16");
        for (int i = 192; i < 256; i++) CHECK(fabsf(o[i] - b[i]) <= 1e-29f, "q8 tiny");
    }
    hx_aligned_free(x);
    hx_aligned_free(y);
    hx_aligned_free(q);
    hx_aligned_free(q2);
}

/* Q8 rounding of exact .5 ties (NUMERICS §6: q = nearbyint(x / d), half to even).
 * amax = 127*d with d an f16-exact non-power-of-two, so the stored scale is d
 * itself, and x = (k + 0.5)*d makes x / d exactly k + 0.5. (Multiplying by a
 * rounded 1/d instead would push such quotients off the tie.) */
static void test_q8_ties(void) {
    static const float ds[] = {3.0f, 5.0f, 6.0f, 7.0f, 0.75f, 1.25f, 0.875f, 3.5f, 0.0029296875f, 96.0f, 0.375f, 11.0f};
    int bad = 0;
    for (size_t di = 0; di < sizeof ds / sizeof ds[0]; di++) {
        const float d = ds[di];
        float b[64];
        int want[64];
        hx_block_q8 q;
        b[0] = (di & 1) ? -127.0f * d : 127.0f * d;
        want[0] = (di & 1) ? -127 : 127;
        for (int i = 1; i < 64; i++) {
            const int k = -127 + (int)((i * 4 + di * 7) % 254);   /* k + 0.5 in [-126.5, 126.5] */
            b[i] = ((float)k + 0.5f) * d;
            want[i] = (k % 2 == 0) ? k : k + 1;
        }
        hx_quantize_row(HEARTH_Q8, b, &q, 64);
        if (q.d != hx_f32_to_f16(d) || ref_f16(q.d) != d) bad++;
        for (int i = 0; i < 64; i++) bad += q.q[i] != want[i];
    }
    CHECK(bad == 0, "q8 exact .5 ties not rounded half to even of x / d (%d wrong)", bad);
}

/* NaN inputs are quantized exactly as 0.0f would be: the whole block (scale and
 * quants) is memcmp-equal to the same block with the NaN replaced by 0. */
static void test_quant_nan(void) {
    for (int dt = HEARTH_Q8; dt <= HEARTH_Q4; dt++) {
        const size_t bs = hx_row_bytes(dt, 64);
        uint8_t qa[66], qb[66];
        int bad = 0;
        for (int blk = 0; blk < 24; blk++) {
            float a[64], c[64];
            const int pos = (blk * 23 + 5) % 64;
            uint32_t nb = 0x7fc00000u | ((uint32_t)(blk * 0x1235) % 0x400000u);
            if (blk % 3 == 1) nb = 0x7f800001u + (uint32_t)blk;   /* signalling */
            if (blk & 1) nb |= 0x80000000u;
            for (int i = 0; i < 64; i++) a[i] = rnd_gauss() * (blk % 4 == 3 ? 1e-3f : 1.0f);
            if (blk % 5 == 2) a[(pos + 9) % 64] = 9.0f;   /* an outlier elsewhere in the block */
            memcpy(c, a, sizeof a);
            a[pos] = bits_f32(nb);
            c[pos] = 0.0f;
            if (blk == 23) {   /* all NaN == all zero */
                for (int i = 0; i < 64; i++) a[i] = bits_f32(0xffc00000u | (uint32_t)i), c[i] = 0.0f;
            }
            hx_quantize_row(dt, a, qa, 64);
            hx_quantize_row(dt, c, qb, 64);
            bad += memcmp(qa, qb, bs) != 0;
        }
        CHECK(bad == 0, "%s: a NaN input is not quantized like 0 (%d of 24 blocks differ)", hx_dtype_name(dt), bad);
    }
}

/* Short rows convert exactly, and dequantization writes exactly n_rows*n_cols floats. */
static void test_row_io(void) {
    float src[3 * 64], out[3 * 64 + 1];
    uint8_t q[3 * 66 + 8];
    for (int i = 0; i < 3 * 64; i++) src[i] = (float)(i % 13) * 0.25f - 1.5f;   /* exact in f16 and bf16 */
    for (int dt = HEARTH_F32; dt <= HEARTH_BF16; dt++) {
        float one = 1.75f, back = 0.0f;
        uint8_t b[4] = {0};
        hx_quantize_row(dt, &one, b, 1);
        hx_dequantize_row(dt, b, &back, 1);
        CHECK(back == 1.75f, "%s: one-element row round trip (%g)", hx_dtype_name(dt), back);
        memset(out, 0, sizeof out);
        CHECK(hearth_quantize(dt, src, 3, 1, q, 1) == 0 && hearth_dequantize(dt, q, 3, 1, out) == 0 &&
              out[0] == src[0] && out[1] == src[1] && out[2] == src[2], "%s: 3x1 round trip", hx_dtype_name(dt));
    }
    for (int dt = HEARTH_F32; dt <= HEARTH_U8; dt++) {
        const int64_t rows = 3, cols = hx_dtype_is_quant(dt) ? 64 : 5;
        if (dt <= HEARTH_Q4) hearth_quantize(dt, src, rows, cols, q, 1);
        else for (size_t i = 0; i < sizeof q; i++) q[i] = (uint8_t)(i * 37);
        out[rows * cols] = -7.0f;
        CHECK(hearth_dequantize(dt, q, rows, cols, out) == 0 && out[rows * cols] == -7.0f,
              "%s: dequantize wrote past n_rows*n_cols", hx_dtype_name(dt));
    }
}

/* Q4 scale search as documented in quant.c: xm = the first element of largest
 * magnitude; candidates f16(xm / m) for the listed divisors, each scored with
 * q = clamp(rint(x * (1/d)), -8, 7). The stored block must be at least as good as
 * every candidate (error summed directly here, in double), so dropping or
 * corrupting a candidate or the scoring shows up on blocks it would have won. */
static const float q4_divisors[] = {
    -8.0f, -7.75f, -8.25f, -7.5f, -8.5f, -7.25f, -8.75f, -7.0f, -9.0f, -9.25f,
    -9.5f, -9.75f, -10.0f, -10.5f, -11.0f, -12.0f, 7.0f, 7.5f, 7.75f, 8.0f,
};

static double q4_err_scale(const float *x, uint16_t h) {
    const float d = ref_f16(h), id = d != 0.0f ? 1.0f / d : 0.0f;
    double e = 0.0;
    for (int i = 0; i < 64; i++) {
        float v = nearbyintf(x[i] * id);
        double r;
        v = v < -8.0f ? -8.0f : v > 7.0f ? 7.0f : v;
        r = (double)x[i] - (double)d * (double)v;
        e += r * r;
    }
    return e;
}

static void test_q4_search(void) {
    const int nblk = g_small ? 1500 : 6000;
    int worse = 0, beats = 0;
    for (int b = 0; b < nblk; b++) {
        float x[64], o[64], xm = 0.0f, amax = 0.0f;
        hx_block_q4 q;
        double e = 0.0, best = 1e300;
        for (int i = 0; i < 64; i++) {
            const double u = rnd_u() + 1e-12;
            switch (b % 5) {
            case 0: x[i] = rnd_gauss() * ((rnd64() % 24 == 0) ? 4.0f : 1.0f); break;      /* outliers */
            case 1: x[i] = (float)((rnd64() & 1) ? -log(u) : log(u)); break;              /* Laplace */
            case 2: {                                                                      /* Student-t, 3 dof */
                double g1 = rnd_gauss(), g2 = rnd_gauss(), g3 = rnd_gauss();
                x[i] = (float)(rnd_gauss() / sqrt((g1 * g1 + g2 * g2 + g3 * g3) / 3.0 + 1e-12));
                break;
            }
            case 3: x[i] = (rnd64() % 2) ? 0.0f : rnd_gauss(); break;                     /* half zeros */
            default: x[i] = 0.5f * (float)((int)(rnd64() % 11) - 5); break;                /* coarse grid, ties */
            }
        }
        for (int i = 0; i < 64; i++)
            if (fabsf(x[i]) > amax) amax = fabsf(x[i]), xm = x[i];
        hx_quantize_row(HEARTH_Q4, x, &q, 64);
        hx_dequantize_row(HEARTH_Q4, &q, o, 64);
        for (int i = 0; i < 64; i++) e += ((double)x[i] - o[i]) * ((double)x[i] - o[i]);
        for (size_t k = 0; k < sizeof q4_divisors / sizeof q4_divisors[0]; k++) {
            uint16_t h = hx_f32_to_f16(xm / q4_divisors[k]);
            double ek;
            if ((h & 0x7c00u) == 0x7c00u) h = (uint16_t)((h & 0x8000u) | 0x7bffu);
            ek = q4_err_scale(x, h);
            if (ek < best) best = ek;
        }
        if (e > best * (1.0 + 1e-9) + 1e-30) worse++;
        if (e < best * (1.0 - 1e-6)) beats++;   /* improved further by the least-squares refit */
    }
    CHECK(worse == 0, "q4: %d of %d blocks worse than one of the documented candidate scales", worse, nblk);
    printf("  q4 search: %d blocks, none worse than any candidate scale; %d improved by the refit\n", nblk, beats);

    {   /* magnitude ties: xm is the first of them. x = (7, -7, 0, ...) is exact with d = 7/-7 = -1
         * (divisor -7 precedes +7); x = (-7, 7, 0, ...) gives d = +1. Same nibbles: q = (-7, 7) -> 1, 15. */
        float a[64] = {0}, c[64] = {0};
        hx_block_q4 qa, qc;
        int ok = 1;
        a[0] = 7.0f, a[1] = -7.0f, c[0] = -7.0f, c[1] = 7.0f;
        hx_quantize_row(HEARTH_Q4, a, &qa, 64);
        hx_quantize_row(HEARTH_Q4, c, &qc, 64);
        ok &= qa.d == 0xbc00 && qc.d == 0x3c00;
        ok &= qa.qs[0] == 0x81 && qa.qs[1] == 0x8f && qc.qs[0] == 0x81 && qc.qs[1] == 0x8f;
        for (int j = 2; j < 32; j++) ok &= qa.qs[j] == 0x88 && qc.qs[j] == 0x88;
        CHECK(ok, "q4 magnitude tie: d %04x / %04x, qs %02x %02x", qa.d, qc.d, qa.qs[0], qa.qs[1]);
    }
}

/* hearth_quantize on a matrix big enough for its thread caps to matter (> 256 rows,
 * more than 256 x 64K elements): any thread count gives the 1-thread bytes. */
static void test_quantize_threads_large(void) {
    const int64_t rows = 300, cols = 57344;   /* 263 x 64K elements: the work-based cap exceeds 256 */
    const size_t n = (size_t)(rows * cols), qb = hx_row_bytes(HEARTH_Q8, cols) * (size_t)rows;
    float *x = (float *)xalloc(n * 4);
    uint8_t *q1 = (uint8_t *)xalloc(qb + 64), *q2 = (uint8_t *)xalloc(qb + 64);
    static const int nts[] = {1000, 0, 5};
    for (size_t i = 0; i < n; i++) x[i] = (float)((int32_t)((uint32_t)(i * 2654435761u) >> 12) - (1 << 19)) * 0x1p-17f;
    CHECK(hearth_quantize(HEARTH_Q8, x, rows, cols, q1, 1) == 0, "large quantize, 1 thread");
    for (size_t k = 0; k < sizeof nts / sizeof nts[0]; k++) {
        memset(q2, 0xcd, qb + 64);
        CHECK(hearth_quantize(HEARTH_Q8, x, rows, cols, q2, nts[k]) == 0 && memcmp(q1, q2, qb) == 0 && q2[qb] == 0xcd,
              "large quantize: %d threads requested != 1 thread", nts[k]);
    }
    hx_aligned_free(x);
    hx_aligned_free(q1);
    hx_aligned_free(q2);
}

/* Pinned outputs. Inputs come from an integer LCG and exact conversions (no libm),
 * and every quantizer / kernel step is a basic IEEE operation, so these hashes must
 * be identical on every platform and compiler. A change here means the quantizer
 * or the canonical kernels changed numerically: that changes model files and
 * logits, so it needs a deliberate decision (update the constants with it). */
static uint64_t fnv1a(const void *p, size_t n, uint64_t h) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}

static void det_fill(float *x, int64_t n, uint64_t seed) {
    uint64_t s = seed;
    for (int64_t i = 0; i < n; i++) {
        double v = 0.0;
        int64_t blk = i / 64;
        for (int k = 0; k < 4; k++) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            v += (double)((int64_t)(s >> 40) - (1 << 23)) * 0x1p-23;   /* exact */
        }
        v *= ldexp(1.0, (int)(blk % 7) - 3);
        if (blk % 13 == 5 && i % 64 == 17) v *= 40.0;
        if (blk % 11 == 3) v = 0.0;
        x[i] = (float)v;
    }
}

/* Heavy-tailed, libm-free: a 4-uniform sum over (0.125 + |2-uniform sum|), up to ~32x. */
static void det_fill_tails(float *x, int64_t n, uint64_t seed) {
    uint64_t s = seed;
    for (int64_t i = 0; i < n; i++) {
        double num = 0.0, den = 0.0;
        for (int k = 0; k < 4; k++) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            num += (double)((int64_t)(s >> 40) - (1 << 23)) * 0x1p-23;
        }
        for (int k = 0; k < 2; k++) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            den += (double)((int64_t)(s >> 40) - (1 << 23)) * 0x1p-23;
        }
        x[i] = (float)(num / (0.125 + fabs(den)));
    }
}

static void test_pinned(void) {
    enum { R = 24, C = 512, TT = 3 };
    {   /* Q4 on heavy tails, where clipping candidates and refit iterations decide more blocks.
         * The hash was reproduced by an independent Python implementation of the documented
         * algorithm (which also reproduces PIN_Q4). */
        const int64_t rt = 64, ct = 1024;
        float *xt = (float *)xalloc((size_t)(rt * ct) * 4);
        uint8_t *qt = (uint8_t *)xalloc(hx_row_bytes(HEARTH_Q4, ct) * (size_t)rt);
        uint64_t h;
        det_fill_tails(xt, rt * ct, 3);
        hearth_quantize(HEARTH_Q4, xt, rt, ct, qt, 4);
        h = fnv1a(qt, hx_row_bytes(HEARTH_Q4, ct) * (size_t)rt, 0xcbf29ce484222325ull);
        CHECK(h == PIN_Q4_TAILS, "pinned q4 (heavy tails) quantizer output changed: 0x%016llx", (unsigned long long)h);
        hx_aligned_free(xt);
        hx_aligned_free(qt);
    }
    static const uint64_t want_q[2] = {PIN_Q8, PIN_Q4};
    static const uint64_t want_y[5] = {PIN_Y_F32, PIN_Y_F16, PIN_Y_BF16, PIN_Y_Q8, PIN_Y_Q4};
    float *W = (float *)xalloc(R * C * 4), *X = (float *)xalloc(TT * C * 4), *Y = (float *)xalloc(TT * R * 4);
    uint8_t *q = (uint8_t *)xalloc((size_t)R * C * 4);
    det_fill(W, R * C, 1);
    det_fill(X, TT * C, 2);
    for (int dt = HEARTH_F32; dt <= HEARTH_Q4; dt++) {
        uint64_t hq, hy;
        hearth_quantize(dt, W, R, C, q, 3);
        hq = fnv1a(q, hx_row_bytes(dt, C) * R, 0xcbf29ce484222325ull);
        hearth_matmul(dt, q, R, C, X, TT, Y, HEARTH_ISA_SCALAR);
        hy = fnv1a(Y, sizeof(float) * TT * R, 0xcbf29ce484222325ull);
        if (dt >= HEARTH_Q8)
            CHECK(hq == want_q[dt - HEARTH_Q8], "pinned %s quantizer output changed: 0x%016llx", hx_dtype_name(dt), (unsigned long long)hq);
        CHECK(hy == want_y[dt], "pinned %s matmul output changed: 0x%016llx", hx_dtype_name(dt), (unsigned long long)hy);
    }
    hx_aligned_free(W);
    hx_aligned_free(X);
    hx_aligned_free(Y);
    hx_aligned_free(q);
}

/* ------------------------------------------------------------ 3. activation quantization */

static void test_act_quant(void) {
    const int64_t ns[] = {64, 128, 640, 4096, 7168};
    for (int kind = 0; kind < K_COUNT + 2; kind++) {
        for (size_t ni = 0; ni < sizeof ns / sizeof ns[0]; ni++) {
            int64_t n = ns[ni], nb = n / 64;
            float *x = (float *)xalloc((size_t)n * 4);
            hx_act_q8 *ref = (hx_act_q8 *)xalloc((size_t)nb * sizeof(hx_act_q8));
            hx_act_q8 *a0 = (hx_act_q8 *)xalloc((size_t)nb * sizeof(hx_act_q8));
            hx_act_q8 *a1 = (hx_act_q8 *)xalloc((size_t)nb * sizeof(hx_act_q8));
            int finite = 1;
            if (kind < K_COUNT) {
                fill(x, n, kind, 0);
            } else if (kind == K_COUNT) {     /* non-finite and subnormal-scale blocks */
                fill(x, n, K_GAUSS, 0);
                x[1] = NAN;
                if (n > 64) x[70] = INFINITY;
                if (n > 128) for (int i = 128; i < 192; i++) x[i] = (i & 1) ? 1e-39f : -3e-40f;
                if (n > 192) x[200] = -INFINITY, x[201] = NAN;
                finite = 0;
            } else {                           /* FLT_MAX magnitudes */
                fill(x, n, K_GAUSS, 0);
                for (int64_t i = 0; i < n; i += 7) x[i] = (i & 8) ? FLT_MAX : -FLT_MAX;
            }
            memset(a0, 0x5a, (size_t)nb * sizeof(hx_act_q8));
            hx_kernels_scalar.act_quantize_q8(x, a0, n);
            if (finite) {
                int bad = 0;
                ref_act(x, n, ref);
                for (int64_t b = 0; b < nb; b++) {
                    if (f32_bits(ref[b].d) != f32_bits(a0[b].d) || ref[b].sum != a0[b].sum ||
                        memcmp(ref[b].q, a0[b].q, 64) != 0) bad++;
                }
                CHECK(bad == 0, "act quant scalar != NUMERICS §2 (kind %d n %lld, %d blocks)", kind, (long long)n, bad);
            }
            for (int64_t b = 0; b < nb; b++) {
                int32_t s = 0;
                for (int i = 0; i < 64; i++) {
                    s += a0[b].q[i];
                    CHECK(a0[b].q[i] >= -127, "act q >= -127");
                }
                CHECK(s == a0[b].sum, "act sum field");
            }
            for (int k = 1; k < n_isas; k++) {
                memset(a1, 0xa5, (size_t)nb * sizeof(hx_act_q8));
                isas[k]->act_quantize_q8(x, a1, n);
                CHECK(memcmp(a0, a1, (size_t)nb * sizeof(hx_act_q8)) == 0,
                      "act quant %s != scalar (kind %d, n %lld)", isa_name(isas[k]->isa), kind, (long long)n);
            }
            hx_aligned_free(x);
            hx_aligned_free(ref);
            hx_aligned_free(a0);
            hx_aligned_free(a1);
        }
    }
}

/* Every f16 / bf16 bit pattern (incl. subnormals, inf, quiet and signalling NaN)
 * converts identically in the scalar kernel and the F16C / AVX-512 paths:
 * 65536 one-column rows times x = 1 give back the converted weight. */
static void test_half_all_values(void) {
    uint16_t *w = (uint16_t *)xalloc(65536 * 2);
    float *y[3], x = 1.0f;
    for (int i = 0; i < 65536; i++) w[i] = (uint16_t)i;
    for (int dt = HEARTH_F16; dt <= HEARTH_BF16; dt++) {
        for (int k = 0; k < n_isas; k++) {
            y[k] = (float *)xalloc(65536 * 4);
            isas[k]->matmul[dt](w, 1, &x, 1, y[k], 65536, 0, 65536);
            if (k > 0)
                CHECK(memcmp(y[k], y[0], 65536 * 4) == 0, "%s: all-values conversion differs from scalar (%s)",
                      isa_name(isas[k]->isa), hx_dtype_name(dt));
        }
        {
            int bad = 0;
            for (int i = 0; i < 65536; i++) {
                float want = dt == HEARTH_F16 ? ref_f16((uint16_t)i) : bits_f32((uint32_t)i << 16);
                if (want == 0.0f) want = 0.0f;   /* lanes start at +0, so -0 weights give +0 */
                if (want == want ? f32_bits(want) != f32_bits(y[0][i]) : y[0][i] == y[0][i]) bad++;
            }
            CHECK(bad == 0, "%s conversion wrong for %d values", hx_dtype_name(dt), bad);
        }
        for (int k = 0; k < n_isas; k++) hx_aligned_free(y[k]);
    }
    hx_aligned_free(w);
}

/* ------------------------------------------------------------ 4. matmul identity */

typedef struct { int64_t rows, cols; int T; int wkind, xkind; } mm_case;

static int g_cases_run;

static void run_case(int dt, const mm_case *c) {
    const int64_t rows = c->rows, cols = c->cols, ldy = rows + 3;
    const int T = c->T;
    const size_t rb = hx_row_bytes(dt, cols), ab = hx_act_bytes(dt, cols);
    const int quant = hx_dtype_is_quant(dt);
    float *W = (float *)xalloc((size_t)(rows * cols) * 4);
    float *Wd = (float *)xalloc((size_t)(rows * cols) * 4);
    float *X = (float *)xalloc((size_t)(T * cols) * 4);
    uint8_t *Wq = (uint8_t *)xalloc(rb * (size_t)rows);
    uint8_t *act[3];
    float *Y[3], *Y1 = (float *)xalloc((size_t)(T * ldy) * 4), *Ys = (float *)xalloc((size_t)(T * ldy) * 4);
    const size_t ybytes = (size_t)(T * ldy) * 4;
    const uint32_t sentinel = 0x7fc0dead;

    fill(W, rows * cols, c->wkind, quant);
    fill(X, T * cols, c->xkind, 0);
    CHECK(hearth_quantize(dt, W, rows, cols, Wq, 2) == 0, "quantize");
    hearth_dequantize(dt, Wq, rows, cols, Wd);

    for (int k = 0; k < n_isas; k++) {
        act[k] = (uint8_t *)xalloc(ab * (size_t)T);
        Y[k] = (float *)xalloc(ybytes);
        for (int64_t i = 0; i < T * ldy; i++) memcpy(&Y[k][i], &sentinel, 4);
        if (quant)
            for (int t = 0; t < T; t++) isas[k]->act_quantize_q8(X + (size_t)t * cols, (hx_act_q8 *)(act[k] + (size_t)t * ab), cols);
        else
            memcpy(act[k], X, (size_t)(T * cols) * 4);
        isas[k]->matmul[dt](Wq, cols, act[k], T, Y[k], ldy, 0, rows);
    }

    for (int k = 0; k < n_isas; k++) {
        const char *in = isa_name(isas[k]->isa);
        int bad_pad = 0;
        for (int t = 0; t < T; t++)
            for (int64_t r = rows; r < ldy; r++)
                if (memcmp(&Y[k][t * ldy + r], &sentinel, 4) != 0) bad_pad++;
        CHECK(bad_pad == 0, "%s %s wrote outside rows", in, hx_dtype_name(dt));
        if (k > 0)
            CHECK(memcmp(Y[k], Y[0], ybytes) == 0, "%s != scalar: %s rows %lld cols %lld T %d w=%s x=%s", in,
                  hx_dtype_name(dt), (long long)rows, (long long)cols, T, kind_name[c->wkind], kind_name[c->xkind]);
        /* batched == T single-token calls */
        memcpy(Y1, Y[k], ybytes);
        for (int t = 0; t < T; t++)
            isas[k]->matmul[dt](Wq, cols, act[k] + (size_t)t * ab, 1, Y1 + (size_t)t * ldy, ldy, 0, rows);
        CHECK(memcmp(Y1, Y[k], ybytes) == 0, "%s %s batched != single (rows %lld cols %lld T %d)", in,
              hx_dtype_name(dt), (long long)rows, (long long)cols, T);
        /* arbitrary row splits == whole */
        memcpy(Ys, Y[k], ybytes);
        for (int t = 0; t < T; t++)
            for (int64_t r = 0; r < rows; r++) Ys[t * ldy + r] = -1.0f;
        for (int64_t r0 = 0; r0 < rows;) {
            int64_t r1 = r0 + 1 + (int64_t)(rnd64() % 23);
            if (r1 > rows) r1 = rows;
            isas[k]->matmul[dt](Wq, cols, act[k], T, Ys, ldy, r0, r1);
            r0 = r1;
        }
        CHECK(memcmp(Ys, Y[k], ybytes) == 0, "%s %s row split != whole", in, hx_dtype_name(dt));
    }

    /* public entry point agrees with the table */
    {
        float *Yp = (float *)xalloc((size_t)(T * rows) * 4);
        int bad = 0;
        for (int k = 0; k < n_isas; k++) {
            CHECK(hearth_matmul(dt, Wq, rows, cols, X, T, Yp, isas[k]->isa) == 0, "hearth_matmul rc");
            for (int t = 0; t < T; t++)
                if (memcmp(Yp + (size_t)t * rows, Y[k] + (size_t)t * ldy, (size_t)rows * 4) != 0) bad++;
        }
        CHECK(bad == 0, "hearth_matmul != kernel table (%s)", hx_dtype_name(dt));
        hx_aligned_free(Yp);
    }

    /* scalar kernel == independent NUMERICS implementation; result close to double */
    {
        int bad_spec = 0, bad_acc = 0;
        int check_acc = c->wkind != K_MAX && c->xkind != K_MAX;
        /* worst-case float rounding: one rounding per accumulated term (per lane for dot16) plus a few */
        const double tol_terms = quant ? (double)(cols / 64 + 4) : (double)((cols + 15) / 16 + 6);
        for (int t = 0; t < T; t++) {
            const float *x = X + (size_t)t * cols;
            const hx_act_q8 *a = (const hx_act_q8 *)(act[0] + (size_t)t * ab);
            for (int64_t r = 0; r < rows; r++) {
                const uint8_t *wr = Wq + (size_t)r * rb;
                float want = quant ? ref_qrow(dt, wr, a, cols) : ref_dot16_w(dt, wr, x, cols);
                float got = Y[0][t * ldy + r];
                double ref = 0.0, mag = 0.0;
                if (f32_bits(want) != f32_bits(got) && !(want != want && got != got)) bad_spec++;
                if (!check_acc) continue;
                for (int64_t i = 0; i < cols; i++) {
                    double xv = quant ? (double)a[i / 64].q[i % 64] * a[i / 64].d : (double)x[i];
                    double p = (double)Wd[r * cols + i] * xv;
                    ref += p;
                    mag += fabs(p);
                }
                if (fabs((double)got - ref) > tol_terms * 0x1p-24 * mag + 1e-30) bad_acc++;
            }
        }
        CHECK(bad_spec == 0, "scalar %s differs from NUMERICS reference in %d outputs (rows %lld cols %lld w=%s x=%s)",
              hx_dtype_name(dt), bad_spec, (long long)rows, (long long)cols, kind_name[c->wkind], kind_name[c->xkind]);
        CHECK(bad_acc == 0, "%s inaccurate vs double in %d outputs (w=%s x=%s)", hx_dtype_name(dt), bad_acc,
              kind_name[c->wkind], kind_name[c->xkind]);
    }

    for (int k = 0; k < n_isas; k++) {
        hx_aligned_free(act[k]);
        hx_aligned_free(Y[k]);
    }
    hx_aligned_free(W);
    hx_aligned_free(Wd);
    hx_aligned_free(X);
    hx_aligned_free(Wq);
    hx_aligned_free(Y1);
    hx_aligned_free(Ys);
    g_cases_run++;
}

static void test_matmul(void) {
    static const int64_t rows_l[] = {1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 20, 24, 31, 32, 33, 40, 47, 48, 49, 63, 64, 65, 66, 67};
    static const int64_t qcols[] = {64, 128, 192, 320, 512, 1024, 1088, 2048, 4096};
    static const int64_t fcols[] = {1, 5, 15, 16, 17, 31, 33, 64, 100, 255, 320, 1000, 2048, 4096};
    static const int Ts[] = {1, 2, 3, 4, 5, 7, 8, 9};
    const int nr = (int)(sizeof rows_l / sizeof rows_l[0]);
    int idx = 0;
    for (int dt = HEARTH_F32; dt <= HEARTH_Q4; dt++) {
        const int quant = hx_dtype_is_quant(dt);
        const int64_t *cl = quant ? qcols : fcols;
        const int nc = quant ? (int)(sizeof qcols / sizeof qcols[0]) : (int)(sizeof fcols / sizeof fcols[0]);
        int before = g_fail;
        /* every rows value x two cols values x rotating T and input kinds */
        for (int ri = 0; ri < nr; ri++) {
            for (int rep = 0; rep < 2; rep++, idx++) {
                mm_case c;
                c.rows = rows_l[ri];
                c.cols = cl[(ri * 2 + rep * 5) % nc];
                c.T = Ts[idx % (int)(sizeof Ts / sizeof Ts[0])];
                c.wkind = idx % K_COUNT;
                c.xkind = (idx / K_COUNT + rep) % K_COUNT;
                if (g_small && c.cols > 1024) c.cols = 1024;
                run_case(dt, &c);
            }
        }
        /* every input kind with a large shape */
        for (int wk = 0; wk < K_COUNT; wk++) {
            mm_case c;
            c.rows = g_small ? 33 : 67;
            c.cols = g_small ? 512 : 4096;
            c.T = 9;
            c.wkind = wk;
            c.xkind = (wk + 3) % K_COUNT;
            run_case(dt, &c);
        }
        printf("  matmul %-4s: %s\n", hx_dtype_name(dt), g_fail == before ? "identical across ISAs, T, row splits; == spec" : "FAILURES");
    }
}

/* ------------------------------------------------------------ 4b. non-finite inputs */

static float wild_f32(void) {
    switch (rnd64() % 10) {
    case 0: return bits_f32(0x7f800001u | (uint32_t)(rnd64() & 0x807fffffu));    /* NaN, any payload / sign */
    case 1: return (rnd64() & 1) ? INFINITY : -INFINITY;
    case 2: return bits_f32((uint32_t)(rnd64() & 0x807fffffu));                   /* subnormal or 0 */
    case 3: return rnd_gauss() * 1e30f;
    default: return rnd_gauss();
    }
}

static uint16_t wild_half(void) {
    switch (rnd64() % 6) {
    case 0: return (uint16_t)rnd64();                                              /* any pattern */
    case 1: return (uint16_t)(0x7c01u | (rnd64() & 0x83ffu));                      /* f16 NaN (bf16: large) */
    case 2: return (uint16_t)(0x7f81u | (rnd64() & 0x807eu));                      /* bf16 NaN */
    default: return hx_f32_to_f16(rnd_gauss());
    }
}

/* Raw-bit weights (NaN with arbitrary payloads, inf, subnormals; Q8/Q4 blocks with
 * NaN / inf / subnormal scales) and activations with NaN, inf and huge values. The
 * identities of run_case must hold bit for bit, and every NaN output must be the
 * canonical quiet NaN: the IEEE choice of which NaN operand propagates depends on
 * operand order, which differs between kernels, tiles and compilers. */
static void test_nonfinite(void) {
    const int ncases = g_small ? 150 : 400;
    int bad_isa = 0, bad_batch = 0, bad_split = 0, bad_canon = 0, bad_spec = 0, bad_pad = 0, nans = 0;
    for (int ci = 0; ci < ncases; ci++) {
        const int dt = ci % 5;
        const int quant = hx_dtype_is_quant(dt);
        const int64_t rows = 1 + (int64_t)(rnd64() % 70);
        const int64_t cols = quant ? 64 * (1 + (int64_t)(rnd64() % 5)) : 1 + (int64_t)(rnd64() % 300);
        const int T = 1 + (int)(rnd64() % 10);
        const int64_t ldy = rows + 2;
        const size_t rb = hx_row_bytes(dt, cols), ab = hx_act_bytes(dt, cols), ybytes = (size_t)(T * ldy) * 4;
        uint8_t *W = (uint8_t *)xalloc(rb * (size_t)rows);
        float *X = (float *)xalloc((size_t)(T * cols) * 4);
        uint8_t *act[3];
        float *Y[3], *Y1 = (float *)xalloc(ybytes);
        for (int64_t i = 0; i < T * cols; i++) X[i] = (rnd64() % 4 == 0) ? wild_f32() : rnd_gauss();
        if (dt == HEARTH_F32) {
            for (int64_t i = 0; i < rows * cols; i++) ((float *)W)[i] = (rnd64() % 4 == 0) ? wild_f32() : rnd_gauss();
        } else if (dt == HEARTH_F16 || dt == HEARTH_BF16) {
            for (int64_t i = 0; i < rows * cols; i++) {
                const float g = rnd_gauss();
                ((uint16_t *)W)[i] = (rnd64() % 4 == 0) ? wild_half() : dt == HEARTH_F16 ? hx_f32_to_f16(g) : hx_f32_to_bf16(g);
            }
        } else {
            const size_t bs = dt == HEARTH_Q8 ? 66 : 34;
            for (size_t i = 0; i < rb * (size_t)rows; i++) W[i] = (uint8_t)rnd64();
            for (size_t g = 0; g < rb * (size_t)rows; g += bs) {
                uint16_t h = (rnd64() % 3 == 0) ? wild_half() : hx_f32_to_f16(0.01f * (float)(1 + rnd64() % 100));
                W[g] = (uint8_t)h;
                W[g + 1] = (uint8_t)(h >> 8);
            }
        }
        for (int k = 0; k < n_isas; k++) {
            act[k] = (uint8_t *)xalloc(ab * (size_t)T);
            Y[k] = (float *)xalloc(ybytes);
            for (int64_t i = 0; i < T * ldy; i++) Y[k][i] = -7.0f;
            if (quant)
                for (int t = 0; t < T; t++) isas[k]->act_quantize_q8(X + (size_t)t * cols, (hx_act_q8 *)(act[k] + (size_t)t * ab), cols);
            else
                memcpy(act[k], X, (size_t)(T * cols) * 4);
            isas[k]->matmul[dt](W, cols, act[k], T, Y[k], ldy, 0, rows);
        }
        for (int k = 0; k < n_isas; k++) {
            if (k > 0) bad_isa += memcmp(Y[k], Y[0], ybytes) != 0;
            for (int t = 0; t < T; t++) {
                for (int64_t r = 0; r < ldy; r++) {
                    const float v = Y[k][t * ldy + r];
                    if (r >= rows) { bad_pad += v != -7.0f; continue; }
                    if (v != v) { nans += k == 0; bad_canon += f32_bits(v) != CANON_NAN; }
                }
            }
            for (int t = 0; t < T; t++)
                isas[k]->matmul[dt](W, cols, act[k] + (size_t)t * ab, 1, Y1 + (size_t)t * ldy, ldy, 0, rows);
            for (int t = 0; t < T; t++)
                bad_batch += memcmp(Y1 + (size_t)t * ldy, Y[k] + (size_t)t * ldy, (size_t)rows * 4) != 0;
            for (int64_t r0 = 0; r0 < rows;) {
                int64_t r1 = r0 + 1 + (int64_t)(rnd64() % 19);
                if (r1 > rows) r1 = rows;
                isas[k]->matmul[dt](W, cols, act[k], T, Y1, ldy, r0, r1);
                r0 = r1;
            }
            for (int t = 0; t < T; t++)
                bad_split += memcmp(Y1 + (size_t)t * ldy, Y[k] + (size_t)t * ldy, (size_t)rows * 4) != 0;
        }
        for (int t = 0; t < T; t++) {   /* scalar == NUMERICS reference (NaN compares as NaN) */
            const hx_act_q8 *a = (const hx_act_q8 *)(act[0] + (size_t)t * ab);
            for (int64_t r = 0; r < rows; r++) {
                const float want = quant ? ref_qrow(dt, W + (size_t)r * rb, a, cols) : ref_dot16_w(dt, W + (size_t)r * rb, X + (size_t)t * cols, cols);
                const float got = Y[0][t * ldy + r];
                if (want != want ? got == got : f32_bits(want) != f32_bits(got)) bad_spec++;
            }
        }
        for (int k = 0; k < n_isas; k++) {
            hx_aligned_free(act[k]);
            hx_aligned_free(Y[k]);
        }
        hx_aligned_free(W);
        hx_aligned_free(X);
        hx_aligned_free(Y1);
    }
    CHECK(bad_isa == 0, "non-finite inputs: %d cases differ between ISAs", bad_isa);
    CHECK(bad_batch == 0, "non-finite inputs: %d batched != single", bad_batch);
    CHECK(bad_split == 0, "non-finite inputs: %d row split != whole", bad_split);
    CHECK(bad_canon == 0, "non-finite inputs: %d NaN outputs are not the canonical NaN", bad_canon);
    CHECK(bad_spec == 0, "non-finite inputs: %d scalar outputs differ from the NUMERICS reference", bad_spec);
    CHECK(bad_pad == 0, "non-finite inputs: %d writes outside rows", bad_pad);
    CHECK(nans > ncases, "non-finite inputs produced too few NaN outputs to test (%d)", nans);
    printf("  non-finite matmul: %d cases, %d NaN outputs (all canonical, identical across ISAs, T, row splits)\n", ncases, nans);
}

/* ------------------------------------------------------------ 5. canonical helpers */

static void test_helpers(void) {
    const int64_t ns[] = {1, 3, 15, 16, 17, 64, 100, 1000, 4099};
    /* empty inputs touch no memory */
    CHECK(hx_sum16(NULL, 0) == 0.0f && hx_dot16(NULL, NULL, 0) == 0.0f, "empty sum16/dot16");
    hx_softmax(NULL, 0);
    hx_rmsnorm(NULL, NULL, NULL, 0, 1e-6f);
    hx_swiglu(NULL, NULL, NULL, 0);
    hx_axpy(NULL, 2.0f, NULL, 0);
    {   /* softmax subtracts the true max wherever it is: expf(200 - m) would overflow otherwise */
        float s[4] = {-3.0f, 200.0f, 10.0f, 1.0f};
        hx_softmax(s, 4);
        CHECK(s[0] == 0.0f && s[1] == 1.0f && s[2] == 0.0f && s[3] == 0.0f, "softmax max at index 1: %g %g %g %g",
              s[0], s[1], s[2], s[3]);
    }
    for (size_t k = 0; k < sizeof ns / sizeof ns[0]; k++) {
        const int64_t n = ns[k];
        /* one sentinel element past n in every buffer: helpers must not read or write it */
        float *a = (float *)xalloc((size_t)(n + 1) * 4), *b = (float *)xalloc((size_t)(n + 1) * 4);
        float *y = (float *)xalloc((size_t)(n + 1) * 4), *w = (float *)xalloc((size_t)(n + 1) * 4);
        double ds = 0, dd = 0, mag = 0, magd = 0;
        float lane[16];
        for (int64_t i = 0; i < n; i++) a[i] = rnd_gauss(), b[i] = rnd_gauss(), w[i] = 0.5f + (float)rnd_u();
        a[n] = b[n] = y[n] = w[n] = 1.0e30f;
        for (int64_t i = 0; i < n; i++) ds += a[i], dd += (double)a[i] * b[i], mag += fabs(a[i]), magd += fabs((double)a[i] * b[i]);

        /* bitwise vs canonical order written here */
        for (int j = 0; j < 16; j++) lane[j] = 0.0f;
        for (int64_t i = 0; i < n; i++) lane[i % 16] = lane[i % 16] + a[i];
        for (int s = 8; s > 0; s /= 2) for (int j = 0; j < s; j++) lane[j] = lane[j] + lane[j + s];
        CHECK(f32_bits(hx_sum16(a, n)) == f32_bits(lane[0]), "sum16 order (n %lld)", (long long)n);
        CHECK(f32_bits(hx_dot16(a, b, n)) == f32_bits(ref_dot16_w(HEARTH_F32, a, b, n)), "dot16 order (n %lld)", (long long)n);
        CHECK(fabs(hx_sum16(a, n) - ds) <= 1e-6 * mag + 1e-30, "sum16 accuracy");
        CHECK(fabs(hx_dot16(a, b, n) - dd) <= 1e-6 * magd + 1e-30, "dot16 accuracy");

        /* rmsnorm (and in place): accuracy, and bitwise vs NUMERICS §4 written here */
        for (int e = 0; e < 2; e++) {
            const float eps = e ? 0.25f : 1e-6f;
            double ms = 0;
            int bad = 0, badbits = 0;
            float r;
            for (int64_t i = 0; i < n; i++) ms += (double)a[i] * a[i];
            ms /= (double)n;
            hx_rmsnorm(y, a, w, n, eps);
            r = 1.0f / sqrtf(ref_dot16_w(HEARTH_F32, a, a, n) / (float)n + eps);
            for (int64_t i = 0; i < n; i++) {
                double want = a[i] / sqrt(ms + eps) * w[i];
                if (fabs(y[i] - want) > 1e-5 * (fabs(want) + 1e-3)) bad++;
                if (f32_bits(y[i]) != f32_bits((a[i] * r) * w[i])) badbits++;
            }
            CHECK(bad == 0 && badbits == 0, "rmsnorm (n %lld, eps %g): %d inaccurate, %d not canonical", (long long)n, eps, bad, badbits);
            memcpy(b, a, (size_t)n * 4);
            hx_rmsnorm(b, b, w, n, eps);
            CHECK(memcmp(b, y, (size_t)n * 4) == 0, "rmsnorm in place");
        }
        /* softmax */
        {
            double m = -1e300, s = 0, tot = 0;
            float fm = -INFINITY, fs;
            int bad = 0, badbits = 0;
            for (int64_t i = 0; i < n; i++) b[i] = 4.0f * a[i], m = b[i] > m ? b[i] : m, fm = b[i] > fm ? b[i] : fm;
            for (int64_t i = 0; i < n; i++) s += exp((double)b[i] - m), y[i] = expf(b[i] - fm);
            for (int j = 0; j < 16; j++) lane[j] = 0.0f;
            for (int64_t i = 0; i < n; i++) lane[i % 16] = lane[i % 16] + y[i];
            for (int st = 8; st > 0; st /= 2) for (int j = 0; j < st; j++) lane[j] = lane[j] + lane[j + st];
            fs = lane[0];
            hx_softmax(b, n);
            for (int64_t i = 0; i < n; i++) {
                double want = exp(4.0 * a[i] - m) / s;
                if (fabs(b[i] - want) > 1e-5 * want + 1e-9) bad++;
                if (f32_bits(b[i]) != f32_bits(y[i] / fs)) badbits++;
                tot += b[i];
            }
            CHECK(bad == 0 && badbits == 0 && fabs(tot - 1.0) < 1e-4, "softmax (n %lld): %d inaccurate, %d not canonical",
                  (long long)n, bad, badbits);
        }
        /* swiglu (and in place), sigmoid */
        {
            int bad = 0, badbits = 0;
            hx_swiglu(y, a, w, n);
            for (int64_t i = 0; i < n; i++) {
                double want = a[i] / (1.0 + exp(-(double)a[i])) * w[i];
                if (fabs(y[i] - want) > 1e-6 * (fabs(want) + 1e-6)) bad++;
                if (fabs(hx_sigmoid(a[i]) - 1.0 / (1.0 + exp(-(double)a[i]))) > 1e-6) bad++;
                if (f32_bits(y[i]) != f32_bits((a[i] / (1.0f + expf(-a[i]))) * w[i])) badbits++;
                if (f32_bits(hx_sigmoid(a[i])) != f32_bits(1.0f / (1.0f + expf(-a[i])))) badbits++;
            }
            memcpy(b, a, (size_t)n * 4);
            hx_swiglu(b, b, w, n);
            CHECK(bad == 0 && badbits == 0 && memcmp(b, y, (size_t)n * 4) == 0, "swiglu/sigmoid: %d inaccurate, %d not canonical", bad, badbits);
        }
        /* axpy */
        {
            int bad = 0;
            memcpy(y, b, (size_t)n * 4);
            hx_axpy(y, 0.37f, a, n);
            for (int64_t i = 0; i < n; i++) if (f32_bits(y[i]) != f32_bits(b[i] + 0.37f * a[i])) bad++;
            CHECK(bad == 0, "axpy");
        }
        CHECK(a[n] == 1.0e30f && b[n] == 1.0e30f && y[n] == 1.0e30f && w[n] == 1.0e30f,
              "helpers wrote past n (n %lld)", (long long)n);
        hx_aligned_free(a);
        hx_aligned_free(b);
        hx_aligned_free(y);
        hx_aligned_free(w);
    }
    /* RoPE both styles, partial rotation */
    for (int style = 0; style < 2; style++) {
        float x[128], x0[128], inv[32];
        const int rd = 64, pos = 1234;
        const float af = 1.25f;
        int bad = 0;
        for (int i = 0; i < 128; i++) x[i] = x0[i] = rnd_gauss();
        for (int j = 0; j < rd / 2; j++) inv[j] = (float)pow(10000.0, -2.0 * j / rd);
        hx_rope(x, rd, style, pos, inv, af);
        for (int j = 0; j < rd / 2; j++) {
            int ia = style ? 2 * j : j, ib = style ? 2 * j + 1 : j + rd / 2;
            double th = (double)((float)pos * inv[j]);
            double c = cos(th) * af, s = sin(th) * af;
            double wa = x0[ia] * c - x0[ib] * s, wb = x0[ib] * c + x0[ia] * s;
            float th32 = (float)pos * inv[j], c32 = cosf(th32) * af, s32 = sinf(th32) * af;
            if (fabs(x[ia] - wa) > 1e-5 * (fabs(x0[ia]) + fabs(x0[ib]) + 1) ||
                fabs(x[ib] - wb) > 1e-5 * (fabs(x0[ia]) + fabs(x0[ib]) + 1)) bad++;
            if (f32_bits(x[ia]) != f32_bits(x0[ia] * c32 - x0[ib] * s32) ||
                f32_bits(x[ib]) != f32_bits(x0[ib] * c32 + x0[ia] * s32)) bad++;
        }
        for (int i = rd; i < 128; i++) if (x[i] != x0[i]) bad++;
        CHECK(bad == 0, "rope style %d", style);
    }
}

/* ------------------------------------------------------------ 6. throughput */

static void fill_weights(int dt, uint8_t *w, size_t bytes, size_t rb) {
    uint64_t s = 0x1234567ull;
    for (size_t i = 0; i + 8 <= bytes; i += 8) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        memcpy(w + i, &s, 8);
    }
    if (dt == HEARTH_F32) {
        uint32_t *p = (uint32_t *)w;
        for (size_t i = 0; i < bytes / 4; i++) p[i] = 0x3f000000u | (p[i] & 0x807fffffu);
    } else if (dt == HEARTH_F16) {
        uint16_t *p = (uint16_t *)w;
        for (size_t i = 0; i < bytes / 2; i++) p[i] = (uint16_t)(0x3800u | (p[i] & 0x83ffu));
    } else if (dt == HEARTH_BF16) {
        uint16_t *p = (uint16_t *)w;
        for (size_t i = 0; i < bytes / 2; i++) p[i] = (uint16_t)(0x3f00u | (p[i] & 0x807fu));
    } else {
        const size_t bs = dt == HEARTH_Q8 ? 66 : 34;
        for (size_t r = 0; r + rb <= bytes; r += rb)
            for (size_t g = 0; g < rb; g += bs) { w[r + g] = 0x00; w[r + g + 1] = 0x2c; }
    }
}

typedef struct {
    const hx_kernels *k;
    int dt, T, reps;
    const uint8_t *W;
    const void *act;
    int64_t cols, r0, r1, ldy;
    float *Y;
    atomic_int *ready, *go;
} bench_job;

static void *bench_thread(void *p) {
    bench_job *j = (bench_job *)p;
    atomic_fetch_add(j->ready, 1);
    while (!atomic_load(j->go)) hx_cpu_relax();
    for (int i = 0; i < j->reps; i++)
        j->k->matmul[j->dt](j->W, j->cols, j->act, j->T, j->Y, j->ldy, j->r0, j->r1);
    return NULL;
}

static void *make_act(const hx_kernels *k, int dt, int64_t cols, int T) {
    size_t ab = hx_act_bytes(dt, cols);
    void *act = xalloc(ab * (size_t)T);
    float *x = (float *)xalloc((size_t)cols * 4);
    for (int t = 0; t < T; t++) {
        for (int64_t i = 0; i < cols; i++) x[i] = rnd_gauss();
        if (hx_dtype_is_quant(dt)) k->act_quantize_q8(x, (hx_act_q8 *)((uint8_t *)act + (size_t)t * ab), cols);
        else memcpy((uint8_t *)act + (size_t)t * ab, x, (size_t)cols * 4);
    }
    hx_aligned_free(x);
    return act;
}

/* Runs `reps` passes over rows [0, rows) split statically across nt threads; GB/s of weight bytes. */
static double bench_run(const hx_kernels *k, int dt, const uint8_t *W, int64_t rows, int64_t cols, int T,
                        int nt, int reps) {
    bench_job jobs[64];
    hx_thread *th[64];
    atomic_int ready = 0, go = 0;
    const size_t rb = hx_row_bytes(dt, cols);
    void *act = make_act(k, dt, cols, T);
    float *Y = (float *)xalloc((size_t)rows * (size_t)T * 4);
    uint64_t t0, t1;
    const int64_t unit = 16;
    const int64_t units = (rows + unit - 1) / unit;
    for (int i = 0; i < nt; i++) {
        int64_t b = units * i / nt * unit, e = units * (i + 1) / nt * unit;
        jobs[i].k = k; jobs[i].dt = dt; jobs[i].T = T; jobs[i].reps = reps;
        jobs[i].W = W; jobs[i].act = act; jobs[i].cols = cols;
        jobs[i].r0 = b < rows ? b : rows; jobs[i].r1 = e < rows ? e : rows;
        jobs[i].ldy = rows; jobs[i].Y = Y;
        jobs[i].ready = &ready; jobs[i].go = &go;
    }
    for (int i = 1; i < nt; i++)
        if (hx_thread_create(&th[i], bench_thread, &jobs[i]) != 0) { printf("thread create failed\n"); exit(2); }
    while (atomic_load(&ready) < nt - 1) hx_yield();
    t0 = hx_now_ns();
    atomic_store(&go, 1);
    atomic_fetch_add(&ready, 1);
    bench_thread(&jobs[0]);
    for (int i = 1; i < nt; i++) hx_thread_join(th[i]);
    t1 = hx_now_ns();
    hx_aligned_free(act);
    hx_aligned_free(Y);
    return (double)rb * (double)rows * reps / (double)(t1 - t0);
}

static void bench(void) {
    const int64_t cols = 4096;
    const size_t l2_target = 512 * 1024;
    const size_t big = g_small ? ((size_t)64 << 20) : ((size_t)1 << 30);
    const int nt = g_small ? 4 : 16;
    uint8_t *mem = (uint8_t *)hx_alloc_large(big, 0);
    double res[5][3][3];   /* dtype, isa, {L2 T=1, L2 T=8 per token-pass, DRAM} */
    if (!mem) { printf("  bench: cannot allocate %zu bytes\n", big); return; }
    memset(res, 0, sizeof res);
    printf("\n  throughput (GB/s of weight bytes; cols %lld; measured on a shared machine, noisy)\n", (long long)cols);
    for (int dt = HEARTH_F32; dt <= HEARTH_Q4; dt++) {
        const size_t rb = hx_row_bytes(dt, cols);
        const int64_t l2rows = (int64_t)(l2_target / rb) / 16 * 16;
        const int64_t drows = (int64_t)(big / rb);
        fill_weights(dt, mem, drows * rb, rb);
        for (int k = 0; k < n_isas; k++) {
            const hx_kernels *kk = isas[k];
            /* L2: warm then time ~0.15 s */
            int reps = 4;
            double gbs;
            for (;;) {
                uint64_t t0 = hx_now_ns();
                gbs = bench_run(kk, dt, mem, l2rows, cols, 1, 1, reps);
                if (hx_now_ns() - t0 > 150000000ull || reps > (1 << 22)) break;
                reps *= 4;
            }
            res[dt][k][0] = gbs;
            res[dt][k][1] = bench_run(kk, dt, mem, l2rows, cols, 8, 1, reps / 8 + 1) * 8.0;
            bench_run(kk, dt, mem, drows, cols, 1, nt, 1);   /* warm-up pass */
            res[dt][k][2] = bench_run(kk, dt, mem, drows, cols, 1, nt, g_small ? 2 : 3);
        }
    }
    printf("  %-5s", "dtype");
    for (int k = 0; k < n_isas; k++) printf(" | %-6s L2 T=1  L2 T=8* DRAM x%-2d", isa_name(isas[k]->isa), nt);
    printf("\n");
    for (int dt = HEARTH_F32; dt <= HEARTH_Q4; dt++) {
        printf("  %-5s", hx_dtype_name(dt));
        for (int k = 0; k < n_isas; k++) printf(" | %13.1f %8.1f %8.1f", res[dt][k][0], res[dt][k][1], res[dt][k][2]);
        printf("\n");
    }
    printf("  (* T=8: weight bytes x 8 tokens per second, i.e. effective GB/s if each token re-read the weights.\n"
           "   L2 = %zu KiB matrix, 1 thread; DRAM = %zu MiB matrix, %d threads via hx_thread_create.)\n",
           l2_target / 1024, big >> 20, nt);
    hx_free_large(mem, big);
}

/* ------------------------------------------------------------ main */

int main(int argc, char **argv) {
    int do_bench = 1, do_tests = 1;
    if (argc == 5 && !strcmp(argv[1], "--dispatch-child")) return dispatch_child(argv[0], atoi(argv[2]), argv[3], atoi(argv[4]));
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--small")) g_small = 1;
        else if (!strcmp(argv[i], "--no-bench")) do_bench = 0;
        else if (!strcmp(argv[i], "--bench-only")) do_tests = 0;
    }
    for (int isa = HEARTH_ISA_SCALAR; isa <= HEARTH_ISA_AVX512; isa++)
        if (hx_kernels_for(isa)) isas[n_isas++] = hx_kernels_for(isa);
    printf("test_quant: ISAs available:");
    for (int k = 0; k < n_isas; k++) printf(" %s", isa_name(isas[k]->isa));
    printf(" (dispatch: %s)%s\n", isa_name(hx_kernels_get()->isa), g_small ? " [small]" : "");
    if (n_isas < 2) printf("  note: only scalar available; cross-ISA identity is not exercised\n");

    if (do_tests) {
        test_sizes();
        printf("  sizes/API: done\n");
        test_isa_tables();
        test_isa_gating();
        test_dispatch_env(argv[0]);
        test_quantizers();
        test_q8_ties();
        test_quant_nan();
        test_row_io();
        test_q4_search();
        test_quantize_threads_large();
        test_pinned();
        printf("  quantizers: done (incl. .5 ties, NaN inputs, pinned cross-platform hashes)\n");
        test_act_quant();
        printf("  activation quantization: done\n");
        test_half_all_values();
        test_matmul();
        printf("  matmul cases: %d\n", g_cases_run);
        test_nonfinite();
        test_helpers();
        printf("  canonical helpers: done\n");
    }
    if (do_bench) bench();
    printf("test_quant: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail > 255 ? 255 : g_fail;
}
