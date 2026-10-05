/*
 * platform_posix.c — POSIX implementation of hx_platform.h (Linux, macOS, BSD).
 * Semantics mirror platform_win.c; shared logic lives in platform_common.inc.
 */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#  define _GNU_SOURCE            /* O_DIRECT, sched_getaffinity, MADV_HUGEPAGE under -std=c11 */
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#  define _DARWIN_C_SOURCE
#endif
#ifndef _FILE_OFFSET_BITS
#  define _FILE_OFFSET_BITS 64   /* 64-bit off_t on 32-bit systems */
#endif

#include "hx_platform.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__DragonFly__)
#  include <sys/sysctl.h>
#  define HX__HAVE_SYSCTL 1
#endif
#if defined(__APPLE__)
#  include <mach/mach.h>
#endif
#if defined(__linux__) && defined(HX_ARCH_ARM64)
#  include <sys/auxv.h>
#endif

#include "platform_common.inc"

/* ---------------------------------------------------------------- memory */

void *hx_aligned_alloc(size_t align, size_t size) {
    if (align == 0 || (align & (align - 1))) return NULL;
    if (align < sizeof(void *)) align = sizeof(void *);      /* posix_memalign minimum */
    void *p = NULL;
    if (posix_memalign(&p, align, size ? size : 1) != 0) return NULL;
    return p;
}

void hx_aligned_free(void *p) { free(p); }

#define HX__HUGE (2u << 20)

static size_t hx__map_len(size_t size) {
    if (size > SIZE_MAX - (HX_PAGE - 1)) return 0;
    return (size + (HX_PAGE - 1)) & ~(size_t)(HX_PAGE - 1);
}

void *hx_alloc_large(size_t size, int try_huge) {
    size_t len = size ? hx__map_len(size) : 0;
    if (len == 0) return NULL;
#if defined(MADV_HUGEPAGE)
    if (try_huge && len >= HX__HUGE && len <= SIZE_MAX - HX__HUGE) {
        /* Transparent huge pages need 2 MiB-aligned ranges: over-map, then trim both ends. */
        uint8_t *raw = (uint8_t *)mmap(NULL, len + HX__HUGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw != (uint8_t *)MAP_FAILED) {
            uintptr_t a = ((uintptr_t)raw + (HX__HUGE - 1)) & ~(uintptr_t)(HX__HUGE - 1);
            uint8_t *p = (uint8_t *)a;
            size_t head = (size_t)(p - raw), tail = HX__HUGE - head;
            if (head) munmap(raw, head);
            if (tail) munmap(p + len, tail);
            madvise(p, len, MADV_HUGEPAGE);                    /* advisory; failure is harmless */
            return p;
        }
    }
#else
    (void)try_huge;
#endif
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

void hx_free_large(void *p, size_t size) {
    if (p) munmap(p, hx__map_len(size));
}

uint64_t hx_ram_total(void) {
#if defined(__APPLE__)
    uint64_t v = 0;
    size_t len = sizeof v;
    if (sysctlbyname("hw.memsize", &v, &len, NULL, 0) == 0) return v;
#endif
    long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGESIZE);
    return pages > 0 && psz > 0 ? (uint64_t)pages * (uint64_t)psz : 0;
}

uint64_t hx_ram_available(void) {
#if defined(__linux__)
    FILE *fp = fopen("/proc/meminfo", "r");
    if (fp) {
        char line[256];
        unsigned long long kb = 0;
        int found = 0;
        while (fgets(line, sizeof line, fp))
            if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { found = 1; break; }
        fclose(fp);
        if (found) return (uint64_t)kb * 1024u;
    }
#elif defined(__APPLE__)
    vm_statistics64_data_t vs;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    mach_port_t host = mach_host_self();
    kern_return_t kr = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vs, &cnt);
    mach_port_deallocate(mach_task_self(), host);
    if (kr == KERN_SUCCESS)
        return ((uint64_t)vs.free_count + (uint64_t)vs.inactive_count) * (uint64_t)vm_kernel_page_size;
#endif
#if defined(_SC_AVPHYS_PAGES)
    {
        long pages = sysconf(_SC_AVPHYS_PAGES), psz = sysconf(_SC_PAGESIZE);
        if (pages > 0 && psz > 0) return (uint64_t)pages * (uint64_t)psz;
    }
#endif
    return hx_ram_total();
}

/* ------------------------------------------------------------------ time */

uint64_t hx_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void hx_sleep_us(uint32_t us) {
    if (us == 0) { sched_yield(); return; }
    struct timespec req, rem;
    req.tv_sec = (time_t)(us / 1000000u);
    req.tv_nsec = (long)(us % 1000000u) * 1000L;
    while (nanosleep(&req, &rem) != 0 && errno == EINTR) req = rem;
}

/* --------------------------------------------------------------- threads */

struct hx_thread { pthread_t t; };

int hx_thread_create(hx_thread **out, hx_thread_fn fn, void *arg) {
    *out = NULL;
    hx_thread *t = (hx_thread *)calloc(1, sizeof *t);
    if (!t) return -1;
    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0) { free(t); return -1; }
    size_t ss = 0;
    /* macOS gives secondary threads 512 KiB; match the 1 MiB Windows default at least. */
    if (pthread_attr_getstacksize(&attr, &ss) == 0 && ss < (1u << 20)) pthread_attr_setstacksize(&attr, 1u << 20);
    int rc = pthread_create(&t->t, &attr, fn, arg);
    pthread_attr_destroy(&attr);
    if (rc != 0) { free(t); return -1; }
    *out = t;
    return 0;
}

void hx_thread_join(hx_thread *t) {
    if (!t) return;
    pthread_join(t->t, NULL);
    free(t);
}

void hx_yield(void) { sched_yield(); }

int hx_num_cpus(void) {
#if defined(__linux__) && defined(CPU_COUNT)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        int n = CPU_COUNT(&set);
        if (n > 0) return n;
    }
#endif
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

#if defined(__linux__)
/* A CPU is the first thread of its core if it is the lowest id in its sibling list. */
static int hx__linux_cores(void) {
    long ncfg = sysconf(_SC_NPROCESSORS_CONF);
    int cores = 0;
    for (long cpu = 0; cpu < ncfg; cpu++) {
        static const char *const names[2] = {"core_cpus_list", "thread_siblings_list"};
        int first = -1;
        for (int k = 0; k < 2 && first < 0; k++) {
            char path[128];
            snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%ld/topology/%s", cpu, names[k]);
            FILE *fp = fopen(path, "r");
            if (!fp) continue;
            if (fscanf(fp, "%d", &first) != 1) first = -1;
            fclose(fp);
        }
        if (first == (int)cpu) cores++;
    }
    return cores;
}
#endif

int hx_num_physical_cores(void) {
    int cores = 0;
#if defined(__linux__)
    cores = hx__linux_cores();
#elif defined(HX__HAVE_SYSCTL)
    static const char *const keys[2] = {"hw.physicalcpu", "kern.smp.cores"};   /* macOS, FreeBSD */
    for (int k = 0; k < 2 && cores <= 0; k++) {
        int v = 0;
        size_t len = sizeof v;
        if (sysctlbyname(keys[k], &v, &len, NULL, 0) == 0) cores = v;
    }
#endif
    int logical = hx_num_cpus();
    if (cores <= 0) return logical;
    return cores < logical ? cores : logical;   /* affinity may hide some cores */
}

void hx_mutex_init(hx_mutex *m) { pthread_mutex_init(&m->m, NULL); }
void hx_mutex_destroy(hx_mutex *m) { pthread_mutex_destroy(&m->m); }
void hx_mutex_lock(hx_mutex *m) { pthread_mutex_lock(&m->m); }
void hx_mutex_unlock(hx_mutex *m) { pthread_mutex_unlock(&m->m); }

void hx_cond_init(hx_cond *c) {
#if defined(__APPLE__)
    pthread_cond_init(&c->c, NULL);              /* timed waits use the relative variant */
#else
    pthread_condattr_t a;
    pthread_condattr_init(&a);
    pthread_condattr_setclock(&a, CLOCK_MONOTONIC);
    pthread_cond_init(&c->c, &a);
    pthread_condattr_destroy(&a);
#endif
}

void hx_cond_destroy(hx_cond *c) { pthread_cond_destroy(&c->c); }
void hx_cond_wait(hx_cond *c, hx_mutex *m) { pthread_cond_wait(&c->c, &m->m); }

int hx_cond_timedwait(hx_cond *c, hx_mutex *m, uint32_t timeout_us) {
    int rc;
#if defined(__APPLE__)
    struct timespec rel;
    rel.tv_sec = (time_t)(timeout_us / 1000000u);
    rel.tv_nsec = (long)(timeout_us % 1000000u) * 1000L;
    rc = pthread_cond_timedwait_relative_np(&c->c, &m->m, &rel);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    ts.tv_sec += (time_t)(timeout_us / 1000000u);
    ts.tv_nsec += (long)(timeout_us % 1000000u) * 1000L;
    if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    rc = pthread_cond_timedwait(&c->c, &m->m, &ts);
#endif
    return rc == ETIMEDOUT ? 1 : 0;
}

void hx_cond_signal(hx_cond *c) { pthread_cond_signal(&c->c); }
void hx_cond_broadcast(hx_cond *c) { pthread_cond_broadcast(&c->c); }

/* ------------------------------------------------------------------ files */

struct hx_file {
    int fd;
    int direct;
};

static int hx__open(const char *path, int oflags) {
    int fd;
    do fd = open(path, oflags, 0644);
    while (fd < 0 && errno == EINTR);
    return fd;
}

hx_file *hx_file_open(const char *path, int flags, char *err, size_t errlen) {
    if (!path || !*path) return (hx_file *)hx_fail(err, errlen, "hx_file_open: empty path");
    if (!(flags & (HX_FILE_READ | HX_FILE_WRITE)))
        return (hx_file *)hx_fail(err, errlen, "%s: open needs HX_FILE_READ and/or HX_FILE_WRITE", path);
    if ((flags & HX_FILE_CREATE) && !(flags & HX_FILE_WRITE))
        return (hx_file *)hx_fail(err, errlen, "%s: HX_FILE_CREATE requires HX_FILE_WRITE", path);

    int oflags = (flags & HX_FILE_READ) && (flags & HX_FILE_WRITE) ? O_RDWR : (flags & HX_FILE_WRITE) ? O_WRONLY : O_RDONLY;
#if defined(O_CLOEXEC)
    oflags |= O_CLOEXEC;
#endif
    if (flags & HX_FILE_CREATE) oflags |= O_CREAT | O_TRUNC;

    int fd = -1, direct = 0;
#if defined(O_DIRECT)
    if (flags & HX_FILE_DIRECT) {
        /* EINVAL: the filesystem (tmpfs, some FUSE/overlay) refuses O_DIRECT. Any
         * other error is reported by the buffered attempt below. */
        fd = hx__open(path, oflags | O_DIRECT);
        if (fd >= 0) direct = 1;
        else hx_log(HX_LOG_DEBUG, "%s: O_DIRECT open failed (%s), using buffered I/O", path, strerror(errno));
    }
#endif
    if (fd < 0) fd = hx__open(path, oflags);
    if (fd < 0) return (hx_file *)hx_fail(err, errlen, "%s: %s", path, strerror(errno));

    struct stat st;
    if (fstat(fd, &st) == 0 && S_ISDIR(st.st_mode)) {
        close(fd);
        return (hx_file *)hx_fail(err, errlen, "%s: is a directory", path);
    }
#if defined(__APPLE__) && defined(F_NOCACHE)
    if ((flags & HX_FILE_DIRECT) && fcntl(fd, F_NOCACHE, 1) == 0) direct = 1;
#endif
    hx_file *f = (hx_file *)calloc(1, sizeof *f);
    if (!f) { close(fd); return (hx_file *)hx_fail(err, errlen, "%s: out of memory", path); }
    f->fd = fd;
    f->direct = direct;
    return f;
}

void hx_file_close(hx_file *f) {
    if (!f) return;
    close(f->fd);
    free(f);
}

int64_t hx_file_size(hx_file *f) {
    struct stat st;
    if (!f || fstat(f->fd, &st) != 0) return -1;
    return (int64_t)st.st_size;
}

int hx_file_is_direct(const hx_file *f) { return f ? f->direct : 0; }

/* Largest single request: page multiple below Linux's 0x7ffff000 per-call cap. */
#define HX__IO_CHUNK ((size_t)1 << 30)

static int64_t hx__rw(hx_file *f, int wr, void *buf, size_t n, uint64_t off) {
    if (!f || (!buf && n)) return -1;
    if (n == 0) return 0;
    if (n > (size_t)INT64_MAX || off > (uint64_t)INT64_MAX - n) return -1;   /* off_t is signed */
    uint8_t *p = (uint8_t *)buf;
    size_t done = 0;
    while (done < n) {
        size_t want = n - done > HX__IO_CHUNK ? HX__IO_CHUNK : n - done;
        ssize_t r = wr ? pwrite(f->fd, p + done, want, (off_t)(off + done))
                       : pread(f->fd, p + done, want, (off_t)(off + done));
        if (r < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (r == 0) {
            if (wr) return -1;                    /* no progress on a write is an error */
            break;
        }
        done += (size_t)r;
        /* A short read means EOF; continuing unaligned would fail on a direct fd. */
        if (!wr && (size_t)r < want && (f->direct || (size_t)r % HX_PAGE)) break;
    }
    return (int64_t)done;
}

int64_t hx_file_pread(hx_file *f, void *buf, size_t n, uint64_t off) { return hx__rw(f, 0, buf, n, off); }
int64_t hx_file_pwrite(hx_file *f, const void *buf, size_t n, uint64_t off) { return hx__rw(f, 1, (void *)buf, n, off); }

int hx_path_exists(const char *path) {
    struct stat st;
    return path && *path && stat(path, &st) == 0;
}

/* ------------------------------------------------------------ cpu features */

static hx_cpu hx__cpu;
static pthread_once_t hx__cpu_once = PTHREAD_ONCE_INIT;

static void hx__cpu_init(void) {
    memset(&hx__cpu, 0, sizeof hx__cpu);
#if defined(HX_ARCH_X86_64)
    hx__detect_x86(&hx__cpu);
#elif defined(HX_ARCH_ARM64)
    hx__cpu.neon = 1;                             /* mandatory in AArch64 */
#  if defined(__linux__)
#    ifndef HWCAP_ASIMDDP
#      define HWCAP_ASIMDDP (1ul << 20)
#    endif
    hx__cpu.dotprod = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) ? 1 : 0;
    snprintf(hx__cpu.brand, sizeof hx__cpu.brand, "aarch64");
#  elif defined(__APPLE__)
    int v = 0;
    size_t len = sizeof v;
    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &len, NULL, 0) == 0) hx__cpu.dotprod = v != 0;
#  endif
#endif
#if defined(__APPLE__)
    if (!hx__cpu.brand[0]) {
        size_t len = sizeof hx__cpu.brand;
        if (sysctlbyname("machdep.cpu.brand_string", hx__cpu.brand, &len, NULL, 0) != 0) hx__cpu.brand[0] = 0;
        hx__cpu.brand[sizeof hx__cpu.brand - 1] = 0;
    }
#endif
}

const hx_cpu *hx_cpu_features(void) {
    pthread_once(&hx__cpu_once, hx__cpu_init);
    return &hx__cpu;
}

/* ------------------------------------------------------------------- env */

const char *hx_env_str(const char *name) {
    if (!name || !*name) return NULL;
    return getenv(name);
}
