/*
 * hx_platform.h — the only place that knows about the operating system.
 * Implemented in platform_win.c (Win32) and platform_posix.c (Linux/macOS/BSD).
 * Everything else in the engine is portable C11 written against this header.
 *
 * Rules for engine code (see governance/INVARIANTS.md):
 *   - C11, no VLAs (MSVC), no compiler-specific code outside this header.
 *   - Atomics: <stdatomic.h> (MSVC needs /experimental:c11atomics; CMake adds it).
 */
#ifndef HX_PLATFORM_H
#define HX_PLATFORM_H

#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdatomic.h>

#if defined(_MSC_VER)
#  define HX_INLINE static __forceinline
#  define HX_NOINLINE __declspec(noinline)
#  define HX_RESTRICT __restrict
#  define HX_ALIGNED(n) __declspec(align(n))
#  define HX_LIKELY(x) (x)
#  define HX_UNLIKELY(x) (x)
#else
#  define HX_INLINE static inline __attribute__((always_inline))
#  define HX_NOINLINE __attribute__((noinline))
#  define HX_RESTRICT __restrict
#  define HX_ALIGNED(n) __attribute__((aligned(n)))
#  define HX_LIKELY(x) __builtin_expect(!!(x), 1)
#  define HX_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

#if defined(_WIN32)
#  define HX_OS_WINDOWS 1
#elif defined(__APPLE__)
#  define HX_OS_MACOS 1
#  define HX_OS_POSIX 1
#else
#  define HX_OS_LINUX 1
#  define HX_OS_POSIX 1
#endif

#if defined(__x86_64__) || defined(_M_X64)
#  define HX_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define HX_ARCH_ARM64 1
#endif

#define HX_PAGE 4096u
#define HX_ALIGN_UP(x, a) (((x) + ((a) - 1)) & ~((uint64_t)(a) - 1))

/* ---------------------------------------------------------------- memory */
/* Aligned heap allocation; align must be a power of two (else NULL). size 0 returns a valid
 * minimal block. Returns NULL on failure. */
void *hx_aligned_alloc(size_t align, size_t size);
void  hx_aligned_free(void *p);
/* Large page-aligned allocation straight from the OS (VirtualAlloc / mmap),
 * zero-filled. try_huge: attempt large/huge pages, silently fall back. size 0 -> NULL. */
void *hx_alloc_large(size_t size, int try_huge);
void  hx_free_large(void *p, size_t size);
/* Physical RAM in bytes, and currently available RAM in bytes (best effort). */
uint64_t hx_ram_total(void);
uint64_t hx_ram_available(void);

/* ----------------------------------------------------------------- time */
uint64_t hx_now_ns(void);          /* monotonic */
/* Never returns early; may overshoot by the OS timer granularity (~0.5 ms on Windows). */
void     hx_sleep_us(uint32_t us);

/* --------------------------------------------------------------- threads */
typedef struct hx_thread hx_thread;
typedef void *(*hx_thread_fn)(void *arg);
/* Returns 0 on success. */
int  hx_thread_create(hx_thread **out, hx_thread_fn fn, void *arg);
void hx_thread_join(hx_thread *t);           /* also frees t */
void hx_yield(void);
int  hx_num_cpus(void);                       /* logical processors */
int  hx_num_physical_cores(void);             /* best effort; falls back to logical */

/* Mutex / condition variable. Concrete types so they can be embedded. */
#if defined(HX_OS_WINDOWS)
typedef struct { void *opaque; } hx_mutex;   /* SRWLOCK */
typedef struct { void *opaque; } hx_cond;    /* CONDITION_VARIABLE */
#else
#  include <pthread.h>
typedef struct { pthread_mutex_t m; } hx_mutex;
typedef struct { pthread_cond_t c; } hx_cond;
#endif
void hx_mutex_init(hx_mutex *m);
void hx_mutex_destroy(hx_mutex *m);
void hx_mutex_lock(hx_mutex *m);
void hx_mutex_unlock(hx_mutex *m);
void hx_cond_init(hx_cond *c);
void hx_cond_destroy(hx_cond *c);
void hx_cond_wait(hx_cond *c, hx_mutex *m);
/* Returns 0 if signalled, 1 on timeout. Windows rounds the timeout up to whole milliseconds. */
int  hx_cond_timedwait(hx_cond *c, hx_mutex *m, uint32_t timeout_us);
void hx_cond_signal(hx_cond *c);
void hx_cond_broadcast(hx_cond *c);

#if defined(_MSC_VER) && defined(HX_ARCH_X86_64)
#  include <immintrin.h>
#endif
HX_INLINE void hx_cpu_relax(void) {
#if defined(HX_ARCH_X86_64)
#  if defined(_MSC_VER)
    _mm_pause();
#  else
    __builtin_ia32_pause();
#  endif
#elif defined(HX_ARCH_ARM64) && !defined(_MSC_VER)
    __asm__ __volatile__("yield");
#endif
}

/* ------------------------------------------------------------------ files */
enum {
    HX_FILE_READ   = 1,
    HX_FILE_WRITE  = 2,   /* open for writing */
    HX_FILE_CREATE = 4,   /* create/truncate (with WRITE) */
    HX_FILE_DIRECT = 8    /* unbuffered: O_DIRECT / FILE_FLAG_NO_BUFFERING / F_NOCACHE.
                             Caller guarantees buffer, offset and size are multiples of HX_PAGE.
                             If the OS/filesystem refuses, hx_file_open falls back to buffered
                             and hx_file_is_direct() returns 0. */
};
typedef struct hx_file hx_file;
/* path is UTF-8. Returns NULL on failure with a message in err. */
hx_file *hx_file_open(const char *path, int flags, char *err, size_t errlen);
void     hx_file_close(hx_file *f);
int64_t  hx_file_size(hx_file *f);
int      hx_file_is_direct(const hx_file *f);
/* Positional read/write; thread-safe on the same hx_file (no shared file pointer).
 * Windows opens handles with FILE_FLAG_OVERLAPPED and waits per request: a synchronous
 * handle serialises concurrent reads (measured 0.62 vs 5.73 GB/s at 16 threads).
 * Loops until n bytes or EOF. Returns bytes transferred, or -1 on error. */
int64_t  hx_file_pread(hx_file *f, void *buf, size_t n, uint64_t off);
int64_t  hx_file_pwrite(hx_file *f, const void *buf, size_t n, uint64_t off);
int      hx_path_exists(const char *path);
/* Atomically replace dst with tmp (MoveFileExW REPLACE_EXISTING / rename). 0 on success. */
int      hx_file_replace(const char *tmp_path, const char *dst_path);

/* ------------------------------------------------------------ cpu features */
typedef struct hx_cpu {
    int sse42, avx, avx2, fma, f16c;
    int avx512f, avx512bw, avx512vl, avx512vnni, avx512bf16, avxvnni;
    int neon, dotprod;
    char brand[64];
} hx_cpu;
/* Detected once and cached; includes OS XSAVE support checks. Returns the process-wide cache
 * (a non-const object): tests may override fields while single-threaded to simulate older CPUs. */
const hx_cpu *hx_cpu_features(void);

/* ------------------------------------------------------- half conversions */
float    hx_f16_to_f32(uint16_t h);
uint16_t hx_f32_to_f16(float f);       /* round-to-nearest-even, handles inf/nan/subnormals */
float    hx_bf16_to_f32(uint16_t h);
uint16_t hx_f32_to_bf16(float f);      /* round-to-nearest-even; NaN stays NaN */

/* ---------------------------------------------------------- env & logging */
int         hx_env_int(const char *name, int def);
double      hx_env_double(const char *name, double def);
const char *hx_env_str(const char *name);          /* NULL if unset */
enum { HX_LOG_ERROR = 0, HX_LOG_WARN = 1, HX_LOG_INFO = 2, HX_LOG_DEBUG = 3 };
void hx_set_log_level(int level);
int  hx_get_log_level(void);
void hx_log(int level, const char *fmt, ...);       /* to stderr, prefixed "[hearth] " */
/* snprintf into err buffer if non-NULL; returns NULL for convenient `return hx_fail(...)`. */
void *hx_fail(char *err, size_t errlen, const char *fmt, ...);

#endif /* HX_PLATFORM_H */
