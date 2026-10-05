/*
 * platform_common.inc — OS-independent half of the platform layer.
 * Included (not compiled on its own) by platform_win.c and platform_posix.c so
 * both produce bit-identical conversions and identical parsing/logging.
 * The including file must provide hx_env_str() and hx__strtod_c() (strtod in the "C"
 * locale) and include hx_platform.h.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>

/* ------------------------------------------------------- half conversions */

static uint32_t hx__f32_bits(float f) { uint32_t u; memcpy(&u, &f, sizeof u); return u; }
static float hx__bits_f32(uint32_t u) { float f; memcpy(&f, &u, sizeof f); return f; }

float hx_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x3ffu;
    uint32_t bits;
    if (exp == 0x1fu) {                           /* inf; NaN quieted, payload kept (as F16C) */
        bits = sign | 0x7f800000u | (mant ? 0x400000u : 0u) | (mant << 13);
    } else if (exp != 0) {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    } else if (mant == 0) {
        bits = sign;
    } else {
        /* subnormal mant * 2^-24: shift until the implicit bit (bit 10) is set */
        uint32_t e = 113u;
        while (!(mant & 0x400u)) { mant <<= 1; e--; }
        bits = sign | (e << 23) | ((mant & 0x3ffu) << 13);
    }
    return hx__bits_f32(bits);
}

uint16_t hx_f32_to_f16(float f) {
    uint32_t x = hx__f32_bits(f);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t ax = x & 0x7fffffffu;
    if (ax > 0x7f800000u)                         /* NaN: quieted, top payload bits kept (as F16C) */
        return (uint16_t)(sign | 0x7e00u | ((ax >> 13) & 0x3ffu));
    if (ax >= 0x477ff000u)                        /* >= 65520 (incl. inf) rounds to inf */
        return (uint16_t)(sign | 0x7c00u);
    if (ax >= 0x38800000u) {                      /* normal result: rebias, RNE on 13 dropped bits */
        uint32_t r = ax - 0x38000000u;
        r += 0x0fffu + ((r >> 13) & 1u);
        return (uint16_t)(sign | (r >> 13));
    }
    if (ax <= 0x33000000u)                        /* <= 2^-25 (tie goes to even = 0) */
        return (uint16_t)sign;
    {
        /* subnormal result: q = RNE(value / 2^-24) */
        uint32_t e = ax >> 23;
        uint32_t m = (ax & 0x7fffffu) | 0x800000u;
        uint32_t sh = 126u - e;                   /* 14..24 */
        uint32_t q = m >> sh;
        uint32_t rem = m & ((1u << sh) - 1u);
        uint32_t half = 1u << (sh - 1u);
        if (rem > half || (rem == half && (q & 1u))) q++;
        return (uint16_t)(sign | q);
    }
}

float hx_bf16_to_f32(uint16_t h) { return hx__bits_f32((uint32_t)h << 16); }

uint16_t hx_f32_to_bf16(float f) {
    uint32_t x = hx__f32_bits(f);
    if ((x & 0x7fffffffu) > 0x7f800000u)          /* NaN: quiet bit forced so it cannot become inf */
        return (uint16_t)((x >> 16) | 0x40u);
    x += 0x7fffu + ((x >> 16) & 1u);
    return (uint16_t)(x >> 16);
}

/* ---------------------------------------------------------- x86 cpu probe */

#if defined(HX_ARCH_X86_64)
#  if defined(__GNUC__) || defined(__clang__)
#    include <cpuid.h>
static void hx__cpuid(uint32_t leaf, uint32_t sub, uint32_t r[4]) {
    unsigned a, b, c, d;
    __cpuid_count(leaf, sub, a, b, c, d);
    r[0] = a; r[1] = b; r[2] = c; r[3] = d;
}
static uint64_t hx__xgetbv0(void) {
    uint32_t lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
}
#  elif defined(_MSC_VER)
#    include <intrin.h>
static void hx__cpuid(uint32_t leaf, uint32_t sub, uint32_t r[4]) {
    int v[4];
    __cpuidex(v, (int)leaf, (int)sub);
    r[0] = (uint32_t)v[0]; r[1] = (uint32_t)v[1]; r[2] = (uint32_t)v[2]; r[3] = (uint32_t)v[3];
}
static uint64_t hx__xgetbv0(void) { return (uint64_t)_xgetbv(0); }
#  else
#    error "no cpuid implementation for this compiler"
#  endif

#define HX__BIT(r, n) (((r) >> (n)) & 1u)

/* ISA bits are reported only if the OS saves the matching register state. zmm_lazy: the OS
 * enables AVX-512 state on a thread's first use (Darwin), so XCR0 cannot show it up front and
 * the caller has confirmed support another way. */
static void hx__detect_x86(hx_cpu *c, int zmm_lazy) {
    uint32_t r[4];
    hx__cpuid(0, 0, r);
    uint32_t max_leaf = r[0];
    if (max_leaf < 1) return;

    hx__cpuid(1, 0, r);
    uint32_t ecx1 = r[2];
    c->sse42 = (int)HX__BIT(ecx1, 20);
    int ymm_ok = 0, zmm_ok = 0;
    if (HX__BIT(ecx1, 27)) {                      /* OSXSAVE */
        uint64_t xcr0 = hx__xgetbv0();
        ymm_ok = (xcr0 & 0x6u) == 0x6u;           /* SSE + AVX state */
        zmm_ok = ymm_ok && ((xcr0 & 0xe0u) == 0xe0u || zmm_lazy);   /* opmask, ZMM_Hi256, Hi16_ZMM */
    }
    c->avx  = ymm_ok && HX__BIT(ecx1, 28);
    c->fma  = ymm_ok && HX__BIT(ecx1, 12);
    c->f16c = ymm_ok && HX__BIT(ecx1, 29);

    if (max_leaf >= 7) {
        hx__cpuid(7, 0, r);
        uint32_t max_sub = r[0], ebx7 = r[1], ecx7 = r[2];
        c->avx2       = ymm_ok && HX__BIT(ebx7, 5);
        c->avx512f    = zmm_ok && HX__BIT(ebx7, 16);
        c->avx512bw   = c->avx512f && HX__BIT(ebx7, 30);
        c->avx512vl   = c->avx512f && HX__BIT(ebx7, 31);
        c->avx512vnni = c->avx512f && HX__BIT(ecx7, 11);
        if (max_sub >= 1) {
            hx__cpuid(7, 1, r);
            c->avxvnni    = ymm_ok && HX__BIT(r[0], 4);
            c->avx512bf16 = c->avx512f && HX__BIT(r[0], 5);
        }
    }

    hx__cpuid(0x80000000u, 0, r);
    if (r[0] >= 0x80000004u) {
        char b[49];
        for (uint32_t i = 0; i < 3; i++) {
            hx__cpuid(0x80000002u + i, 0, r);
            memcpy(b + 16 * i, r, 16);
        }
        b[48] = 0;
        const char *s = b;
        while (*s == ' ') s++;
        size_t n = strlen(s);
        while (n && s[n - 1] == ' ') n--;
        _Static_assert(sizeof c->brand > 48, "brand must hold the 48-byte CPUID string");
        memcpy(c->brand, s, n);
        c->brand[n] = 0;
    }
}
#endif /* HX_ARCH_X86_64 */

/* ------------------------------------------------------------------- env */

static int hx__is_space(char ch) { return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n'; }

/* Whole-string base-10 integer; surrounding whitespace allowed. */
static int hx__parse_ll(const char *s, long long *out) {
    char *end;
    errno = 0;
    long long v = strtoll(s, &end, 10);
    if (end == s || errno == ERANGE) return 0;
    while (hx__is_space(*end)) end++;
    if (*end) return 0;
    *out = v;
    return 1;
}

int hx_env_int(const char *name, int def) {
    const char *s = hx_env_str(name);
    long long v;
    if (!s || !*s) return def;
    if (!hx__parse_ll(s, &v) || v < INT_MIN || v > INT_MAX) {
        hx_log(HX_LOG_WARN, "ignoring %s=\"%s\": not an integer", name, s);
        return def;
    }
    return (int)v;
}

static double hx__strtod_c(const char *s, char **end);

/* Parsed in the "C" locale: a host process may have set LC_NUMERIC to one with a decimal comma. */
double hx_env_double(const char *name, double def) {
    const char *s = hx_env_str(name);
    char *end;
    if (!s || !*s) return def;
    errno = 0;
    double v = hx__strtod_c(s, &end);
    if (end != s && errno != ERANGE) {
        while (hx__is_space(*end)) end++;
        if (!*end) return v;
    }
    hx_log(HX_LOG_WARN, "ignoring %s=\"%s\": not a number", name, s);
    return def;
}

/* --------------------------------------------------------------- logging */

static atomic_int hx__log_level = -1;            /* -1: not yet read from HEARTH_LOG */

static int hx__clamp_level(long long v) { return v < 0 ? 0 : v > 3 ? 3 : (int)v; }

void hx_set_log_level(int level) {
    atomic_store_explicit(&hx__log_level, hx__clamp_level(level), memory_order_relaxed);
}

int hx_get_log_level(void) {
    int l = atomic_load_explicit(&hx__log_level, memory_order_relaxed);
    if (l >= 0) return l;
    /* Parsed without hx_env_int: a bad value must not recurse into hx_log. */
    int v = HX_LOG_WARN;
    long long p;
    const char *s = hx_env_str("HEARTH_LOG");
    if (s && hx__parse_ll(s, &p)) v = hx__clamp_level(p);
    int expect = -1;
    atomic_compare_exchange_strong(&hx__log_level, &expect, v);
    return atomic_load_explicit(&hx__log_level, memory_order_relaxed);
}

static void hx__vlog(int level, const char *fmt, va_list ap) {
    static const char *const tag[4] = {"error: ", "warning: ", "", ""};
    char stackbuf[1024];
    char *buf = stackbuf;
    size_t cap = sizeof stackbuf;
    const char *t = tag[level < 0 ? 0 : level > 3 ? 3 : level];
    va_list ap2;
    va_copy(ap2, ap);
    int pre = snprintf(buf, cap, "[hearth] %s", t);
    int n = vsnprintf(buf + pre, cap - (size_t)pre, fmt, ap);
    if (n < 0) { va_end(ap2); return; }
    size_t need = (size_t)pre + (size_t)n + 2;    /* + newline + NUL */
    if (need > cap) {
        char *big = (char *)malloc(need);
        if (big) {
            memcpy(big, buf, (size_t)pre);
            vsnprintf(big + pre, need - (size_t)pre, fmt, ap2);
            buf = big;
            cap = need;
        } else {
            n = (int)(cap - (size_t)pre - 2);     /* emit the truncated message */
        }
    }
    va_end(ap2);
    size_t len = (size_t)pre + (size_t)n;         /* <= cap - 2 on every path above */
    if (buf[len - 1] != '\n') buf[len++] = '\n';
    buf[len] = 0;
    fwrite(buf, 1, len, stderr);                  /* one write per line keeps threads' lines whole */
    fflush(stderr);
    if (buf != stackbuf) free(buf);
}

void hx_log(int level, const char *fmt, ...) {
    if (level > hx_get_log_level()) return;
    va_list ap;
    va_start(ap, fmt);
    hx__vlog(level, fmt, ap);
    va_end(ap);
}

void *hx_fail(char *err, size_t errlen, const char *fmt, ...) {
    va_list ap;
    if (err && errlen) {
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    if (hx_get_log_level() >= HX_LOG_DEBUG) {
        va_start(ap, fmt);
        hx__vlog(HX_LOG_DEBUG, fmt, ap);
        va_end(ap);
    }
    return NULL;
}
