/*
 * platform_win.c — Win32 implementation of hx_platform.h.
 */
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>

#include "hx_platform.h"
#include "platform_common.inc"

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#  define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#ifndef PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE
#  define PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE 43
#endif

_Static_assert(sizeof(SRWLOCK) == sizeof(hx_mutex), "hx_mutex must hold an SRWLOCK");
_Static_assert(sizeof(CONDITION_VARIABLE) == sizeof(hx_cond), "hx_cond must hold a CONDITION_VARIABLE");

/* ---------------------------------------------------------------- strings */

/*
 * UTF-8 -> heap UTF-16. A path whose absolute form would exceed the classic
 * MAX_PATH limit is rewritten as \\?\C:\... or \\?\UNC\srv\... so long paths
 * work without the process being long-path aware.
 */
static wchar_t *hx__widen(const char *s, int is_path) {
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
    if (n <= 0) return NULL;
    wchar_t *w = (wchar_t *)malloc((size_t)n * sizeof(wchar_t));
    if (!w) return NULL;
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, w, n) != n) { free(w); return NULL; }
    if (!is_path || wcsncmp(w, L"\\\\?\\", 4) == 0 || wcsncmp(w, L"\\\\.\\", 4) == 0) return w;

    DWORD full = GetFullPathNameW(w, 0, NULL, NULL);       /* includes the NUL */
    if (full == 0 || full < 248) return w;                 /* 248: directory-name limit */
    wchar_t *abs = (wchar_t *)malloc((size_t)full * sizeof(wchar_t));
    if (!abs) return w;
    DWORD got = GetFullPathNameW(w, full, abs, NULL);
    if (got == 0 || got >= full) { free(abs); return w; }
    int unc = abs[0] == L'\\' && abs[1] == L'\\';
    size_t len = wcslen(abs);
    wchar_t *res = (wchar_t *)malloc((len + 8) * sizeof(wchar_t));
    if (!res) { free(abs); return w; }
    if (unc) { wcscpy(res, L"\\\\?\\UNC"); wcscat(res, abs + 1); }
    else     { wcscpy(res, L"\\\\?\\");    wcscat(res, abs); }
    free(abs);
    free(w);
    return res;
}

static char *hx__narrow(const wchar_t *w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    if (n <= 0) return NULL;
    char *s = (char *)malloc((size_t)n);
    if (s && WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL) != n) { free(s); s = NULL; }
    return s;
}

static void hx__winerr(char *buf, size_t len, DWORD code) {
    wchar_t wmsg[256];
    DWORD n = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, code,
                             0, wmsg, (DWORD)(sizeof wmsg / sizeof wmsg[0]), NULL);
    while (n && (wmsg[n - 1] == L'\r' || wmsg[n - 1] == L'\n' || wmsg[n - 1] == L'.')) wmsg[--n] = 0;
    char *m = n ? hx__narrow(wmsg) : NULL;
    snprintf(buf, len, "%s (error %lu)", m ? m : "system error", (unsigned long)code);
    free(m);
}

/* ---------------------------------------------------------------- memory */

void *hx_aligned_alloc(size_t align, size_t size) {
    if (align == 0 || (align & (align - 1))) return NULL;
    return _aligned_malloc(size ? size : 1, align);
}

void hx_aligned_free(void *p) { if (p) _aligned_free(p); }

static INIT_ONCE hx__lockmem_once = INIT_ONCE_STATIC_INIT;
static int hx__lockmem_ok;

/* advapi32 is resolved at runtime so callers need no extra import library. */
typedef BOOL (WINAPI *hx__OpenProcessToken_t)(HANDLE, DWORD, PHANDLE);
typedef BOOL (WINAPI *hx__LookupPrivilegeValueW_t)(LPCWSTR, LPCWSTR, PLUID);
typedef BOOL (WINAPI *hx__AdjustTokenPrivileges_t)(HANDLE, BOOL, PTOKEN_PRIVILEGES, DWORD, PTOKEN_PRIVILEGES, PDWORD);

static BOOL CALLBACK hx__lockmem_init(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    (void)once; (void)param; (void)ctx;
    hx__lockmem_ok = 0;
    if (GetLargePageMinimum() == 0) return TRUE;
    HMODULE adv = LoadLibraryW(L"advapi32.dll");
    if (!adv) return TRUE;
    hx__OpenProcessToken_t open_tok = (hx__OpenProcessToken_t)GetProcAddress(adv, "OpenProcessToken");
    hx__LookupPrivilegeValueW_t lookup = (hx__LookupPrivilegeValueW_t)GetProcAddress(adv, "LookupPrivilegeValueW");
    hx__AdjustTokenPrivileges_t adjust = (hx__AdjustTokenPrivileges_t)GetProcAddress(adv, "AdjustTokenPrivileges");
    HANDLE tok;
    if (open_tok && lookup && adjust && open_tok(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
        TOKEN_PRIVILEGES tp;
        tp.PrivilegeCount = 1;
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        if (lookup(NULL, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
            /* Succeeds even when not held; ERROR_NOT_ALL_ASSIGNED tells. */
            if (adjust(tok, FALSE, &tp, 0, NULL, NULL) && GetLastError() == ERROR_SUCCESS) hx__lockmem_ok = 1;
        }
        CloseHandle(tok);
    }
    /* advapi32 stays loaded: it is already mapped in practically every process. */
    if (!hx__lockmem_ok) hx_log(HX_LOG_DEBUG, "large pages unavailable (SeLockMemoryPrivilege not held)");
    return TRUE;
}

void *hx_alloc_large(size_t size, int try_huge) {
    if (size == 0) return NULL;
    if (try_huge) {
        SIZE_T lp = GetLargePageMinimum();
        if (lp && size >= lp) {
            InitOnceExecuteOnce(&hx__lockmem_once, hx__lockmem_init, NULL, NULL);
            if (hx__lockmem_ok) {
                size_t rounded = (size + lp - 1) / lp * lp;
                if (rounded >= size) {
                    void *p = VirtualAlloc(NULL, rounded, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
                    if (p) return p;
                }
            }
        }
    }
    return VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
}

void hx_free_large(void *p, size_t size) {
    (void)size;
    if (p) VirtualFree(p, 0, MEM_RELEASE);
}

uint64_t hx_ram_total(void) {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t)ms.ullTotalPhys : 0;
}

uint64_t hx_ram_available(void) {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t)ms.ullAvailPhys : 0;
}

/* ------------------------------------------------------------------ time */

static atomic_ullong hx__qpc_freq;

uint64_t hx_now_ns(void) {
    uint64_t f = atomic_load_explicit(&hx__qpc_freq, memory_order_relaxed);
    if (!f) {
        LARGE_INTEGER q;
        QueryPerformanceFrequency(&q);
        f = (uint64_t)q.QuadPart;
        atomic_store_explicit(&hx__qpc_freq, f, memory_order_relaxed);
    }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    uint64_t t = (uint64_t)c.QuadPart;
    /* split so t * 1e9 cannot overflow: (t % f) < f, and f is far below 1.8e10 */
    return (t / f) * 1000000000ull + (t % f) * 1000000000ull / f;
}

/* Overshoot is set by the kernel timer: measured ~0.5 ms granularity on Windows 11
 * even with the high-resolution timer (plain Sleep(1) took ~2 ms). */
void hx_sleep_us(uint32_t us) {
    if (us == 0) { SwitchToThread(); return; }
    HANDLE t = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (t) {
        LARGE_INTEGER due;
        due.QuadPart = -(LONGLONG)us * 10;        /* relative, 100 ns units */
        if (SetWaitableTimerEx(t, &due, 0, NULL, NULL, NULL, 0)) {
            WaitForSingleObject(t, INFINITE);
            CloseHandle(t);
            return;
        }
        CloseHandle(t);
    }
    Sleep((us + 999) / 1000);                     /* pre-1803 Windows: millisecond sleep */
}

/* --------------------------------------------------------------- threads */

struct hx_thread {
    HANDLE h;
    hx_thread_fn fn;
    void *arg;
};

static DWORD WINAPI hx__thread_main(LPVOID p) {
    hx_thread *t = (hx_thread *)p;
    t->fn(t->arg);
    return 0;
}

int hx_thread_create(hx_thread **out, hx_thread_fn fn, void *arg) {
    *out = NULL;
    hx_thread *t = (hx_thread *)calloc(1, sizeof *t);
    if (!t) return -1;
    t->fn = fn;
    t->arg = arg;
    t->h = CreateThread(NULL, 0, hx__thread_main, t, 0, NULL);
    if (!t->h) { free(t); return -1; }
    *out = t;
    return 0;
}

void hx_thread_join(hx_thread *t) {
    if (!t) return;
    WaitForSingleObject(t->h, INFINITE);
    CloseHandle(t->h);
    free(t);
}

void hx_yield(void) { SwitchToThread(); }

int hx_num_cpus(void) {
    DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (n == 0) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        n = si.dwNumberOfProcessors;
    }
    return n ? (int)n : 1;
}

int hx_num_physical_cores(void) {
    DWORD len = 0;
    int cores = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && len) {
        char *buf = (char *)malloc(len);
        if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore,
                                                    (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
            for (DWORD off = 0; off < len;) {
                PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off);
                if (e->Size == 0) break;
                if (e->Relationship == RelationProcessorCore) cores++;
                off += e->Size;
            }
        }
        free(buf);
    }
    return cores > 0 ? cores : hx_num_cpus();
}

void hx_mutex_init(hx_mutex *m) { InitializeSRWLock((PSRWLOCK)&m->opaque); }
void hx_mutex_destroy(hx_mutex *m) { (void)m; }
void hx_mutex_lock(hx_mutex *m) { AcquireSRWLockExclusive((PSRWLOCK)&m->opaque); }
void hx_mutex_unlock(hx_mutex *m) { ReleaseSRWLockExclusive((PSRWLOCK)&m->opaque); }

void hx_cond_init(hx_cond *c) { InitializeConditionVariable((PCONDITION_VARIABLE)&c->opaque); }
void hx_cond_destroy(hx_cond *c) { (void)c; }

void hx_cond_wait(hx_cond *c, hx_mutex *m) {
    SleepConditionVariableSRW((PCONDITION_VARIABLE)&c->opaque, (PSRWLOCK)&m->opaque, INFINITE, 0);
}

int hx_cond_timedwait(hx_cond *c, hx_mutex *m, uint32_t timeout_us) {
    DWORD ms = (DWORD)(((uint64_t)timeout_us + 999u) / 1000u);   /* never INFINITE: max ~4.3e6 */
    if (SleepConditionVariableSRW((PCONDITION_VARIABLE)&c->opaque, (PSRWLOCK)&m->opaque, ms, 0)) return 0;
    return GetLastError() == ERROR_TIMEOUT ? 1 : 0;
}

void hx_cond_signal(hx_cond *c) { WakeConditionVariable((PCONDITION_VARIABLE)&c->opaque); }
void hx_cond_broadcast(hx_cond *c) { WakeAllConditionVariable((PCONDITION_VARIABLE)&c->opaque); }

/* ------------------------------------------------------------------ files */

/*
 * Handles are opened with FILE_FLAG_OVERLAPPED and every call waits for its
 * own request. A synchronous handle would also be positional, but the I/O
 * manager serialises all requests on a synchronous file object, which caps
 * concurrent reader threads at queue depth 1 (measured: 64 KiB unbuffered
 * reads stay at ~0.6 GB/s with 16 threads, vs ~5.7 GB/s overlapped).
 */
struct hx_file {
    HANDLE h;
    int direct;
};

/*
 * Manual-reset events for the per-request waits, recycled through a small
 * global stack (no thread-exit hooks, which a DLL could not unregister on
 * unload). ReadFile/WriteFile reset the event when a request starts.
 */
#define HX__EV_CACHE 64
static SRWLOCK hx__ev_lock = SRWLOCK_INIT;
static HANDLE hx__ev_cache[HX__EV_CACHE];
static int hx__ev_n;

static HANDLE hx__ev_get(void) {
    HANDLE ev = NULL;
    AcquireSRWLockExclusive(&hx__ev_lock);
    if (hx__ev_n > 0) ev = hx__ev_cache[--hx__ev_n];
    ReleaseSRWLockExclusive(&hx__ev_lock);
    return ev ? ev : CreateEventW(NULL, TRUE, FALSE, NULL);
}

static void hx__ev_put(HANDLE ev) {
    AcquireSRWLockExclusive(&hx__ev_lock);
    if (hx__ev_n < HX__EV_CACHE) { hx__ev_cache[hx__ev_n++] = ev; ev = NULL; }
    ReleaseSRWLockExclusive(&hx__ev_lock);
    if (ev) CloseHandle(ev);
}

static HANDLE hx__open(const wchar_t *w, int flags, DWORD extra, DWORD *errout) {
    DWORD access = ((flags & HX_FILE_READ) ? GENERIC_READ : 0) | ((flags & HX_FILE_WRITE) ? GENERIC_WRITE : 0);
    DWORD disp = (flags & HX_FILE_CREATE) ? CREATE_ALWAYS : OPEN_EXISTING;
    HANDLE h = CreateFileW(w, access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, disp,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | extra, NULL);
    *errout = h == INVALID_HANDLE_VALUE ? GetLastError() : 0;
    return h;
}

/* Unbuffered I/O needs HX_PAGE to be a multiple of the volume sector size. */
static int hx__sector_ok(HANDLE h) {
    FILE_STORAGE_INFO si;
    if (!GetFileInformationByHandleEx(h, FileStorageInfo, &si, sizeof si)) return 1;   /* unknown: trust the open */
    ULONG s = si.LogicalBytesPerSector;
    return s == 0 || (s <= HX_PAGE && HX_PAGE % s == 0);
}

hx_file *hx_file_open(const char *path, int flags, char *err, size_t errlen) {
    if (!path || !*path) return (hx_file *)hx_fail(err, errlen, "hx_file_open: empty path");
    if (!(flags & (HX_FILE_READ | HX_FILE_WRITE)))
        return (hx_file *)hx_fail(err, errlen, "%s: open needs HX_FILE_READ and/or HX_FILE_WRITE", path);
    if ((flags & HX_FILE_CREATE) && !(flags & HX_FILE_WRITE))
        return (hx_file *)hx_fail(err, errlen, "%s: HX_FILE_CREATE requires HX_FILE_WRITE", path);
    wchar_t *w = hx__widen(path, 1);
    if (!w) return (hx_file *)hx_fail(err, errlen, "%s: path is not valid UTF-8", path);

    DWORD e = 0;
    int direct = 0;
    HANDLE h = INVALID_HANDLE_VALUE;
    if (flags & HX_FILE_DIRECT) {
        h = hx__open(w, flags, FILE_FLAG_NO_BUFFERING, &e);
        if (h != INVALID_HANDLE_VALUE) {
            if (hx__sector_ok(h)) direct = 1;
            else { CloseHandle(h); h = INVALID_HANDLE_VALUE; }
        }
        if (!direct) hx_log(HX_LOG_DEBUG, "%s: unbuffered open refused, using buffered I/O", path);
    }
    if (h == INVALID_HANDLE_VALUE) h = hx__open(w, flags, 0, &e);
    free(w);
    if (h == INVALID_HANDLE_VALUE) {
        char msg[300];
        hx__winerr(msg, sizeof msg, e);
        return (hx_file *)hx_fail(err, errlen, "%s: %s", path, msg);
    }
    hx_file *f = (hx_file *)calloc(1, sizeof *f);
    if (!f) { CloseHandle(h); return (hx_file *)hx_fail(err, errlen, "%s: out of memory", path); }
    f->h = h;
    f->direct = direct;
    return f;
}

void hx_file_close(hx_file *f) {
    if (!f) return;
    CloseHandle(f->h);
    free(f);
}

int64_t hx_file_size(hx_file *f) {
    LARGE_INTEGER sz;
    if (!f || !GetFileSizeEx(f->h, &sz)) return -1;
    return (int64_t)sz.QuadPart;
}

int hx_file_is_direct(const hx_file *f) { return f ? f->direct : 0; }

/* Largest single request: page multiple below 4 GiB, so unbuffered chunks stay aligned. */
#define HX__IO_CHUNK (1u << 30)

/* One request at off; returns bytes (0 = EOF) or -1. */
static int64_t hx__io(hx_file *f, int wr, void *buf, DWORD n, uint64_t off, HANDLE ev) {
    OVERLAPPED ov;
    DWORD got = 0;
    memset(&ov, 0, sizeof ov);
    ov.Offset = (DWORD)off;
    ov.OffsetHigh = (DWORD)(off >> 32);
    ov.hEvent = ev;
    BOOL ok = wr ? WriteFile(f->h, buf, n, NULL, &ov) : ReadFile(f->h, buf, n, NULL, &ov);
    if (!ok) {
        DWORD e = GetLastError();
        if (e == ERROR_HANDLE_EOF && !wr) return 0;
        if (e != ERROR_IO_PENDING) { SetLastError(e); return -1; }
    }
    if (!GetOverlappedResult(f->h, &ov, &got, TRUE)) {
        if (GetLastError() == ERROR_HANDLE_EOF && !wr) return 0;
        return -1;
    }
    return (int64_t)got;
}

static int64_t hx__rw(hx_file *f, int wr, void *buf, size_t n, uint64_t off) {
    if (!f || (!buf && n)) return -1;
    if (n == 0) return 0;
    if (n > (size_t)INT64_MAX || off > (uint64_t)INT64_MAX - n) return -1;   /* NT offsets are signed */
    HANDLE ev = hx__ev_get();
    if (!ev) return -1;
    uint8_t *p = (uint8_t *)buf;
    size_t done = 0;
    int64_t rc = 0;
    while (done < n) {
        size_t left = n - done;
        DWORD want = left > HX__IO_CHUNK ? HX__IO_CHUNK : (DWORD)left;
        int64_t got = hx__io(f, wr, p + done, want, off + done, ev);
        if (got < 0) { rc = -1; break; }
        if (got == 0) {
            if (wr) rc = -1;                      /* no progress on a write is an error */
            break;
        }
        done += (size_t)got;
        /* A short read means EOF; continuing unaligned would fail on a direct handle. */
        if (!wr && (DWORD)got < want && (f->direct || got % HX_PAGE)) break;
    }
    hx__ev_put(ev);
    return rc < 0 ? -1 : (int64_t)done;
}

int64_t hx_file_pread(hx_file *f, void *buf, size_t n, uint64_t off) { return hx__rw(f, 0, buf, n, off); }
int64_t hx_file_pwrite(hx_file *f, const void *buf, size_t n, uint64_t off) { return hx__rw(f, 1, (void *)buf, n, off); }

int hx_path_exists(const char *path) {
    if (!path || !*path) return 0;
    wchar_t *w = hx__widen(path, 1);
    if (!w) return 0;
    DWORD a = GetFileAttributesW(w);
    free(w);
    return a != INVALID_FILE_ATTRIBUTES;
}

/* ------------------------------------------------------------ cpu features */

static hx_cpu hx__cpu;
static INIT_ONCE hx__cpu_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK hx__cpu_init(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    (void)once; (void)param; (void)ctx;
    memset(&hx__cpu, 0, sizeof hx__cpu);
#if defined(HX_ARCH_X86_64)
    hx__detect_x86(&hx__cpu);
#elif defined(HX_ARCH_ARM64)
    hx__cpu.neon = 1;
    hx__cpu.dotprod = IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE) ? 1 : 0;
    snprintf(hx__cpu.brand, sizeof hx__cpu.brand, "ARM64");
#endif
    return TRUE;
}

const hx_cpu *hx_cpu_features(void) {
    InitOnceExecuteOnce(&hx__cpu_once, hx__cpu_init, NULL, NULL);
    return &hx__cpu;
}

/* ------------------------------------------------------------------- env */

/*
 * Values are read from the process environment as UTF-16 and returned as
 * UTF-8. Each distinct (name, value) is kept for the life of the process, so
 * returned pointers stay valid even if the variable later changes.
 */
typedef struct hx__env_ent { struct hx__env_ent *next; char *name; char *value; } hx__env_ent;
static SRWLOCK hx__env_lock = SRWLOCK_INIT;
static hx__env_ent *hx__env_list;

const char *hx_env_str(const char *name) {
    if (!name || !*name) return NULL;
    wchar_t *wn = hx__widen(name, 0);
    if (!wn) return NULL;
    DWORD need = GetEnvironmentVariableW(wn, NULL, 0);
    if (need == 0) { free(wn); return NULL; }     /* unset (or empty and unreadable) */
    wchar_t *wv = NULL;
    for (;;) {
        wchar_t *nw = (wchar_t *)realloc(wv, (size_t)need * sizeof(wchar_t));
        if (!nw) { free(wv); free(wn); return NULL; }
        wv = nw;
        DWORD got = GetEnvironmentVariableW(wn, wv, need);
        if (got == 0 && GetLastError() == ERROR_ENVVAR_NOT_FOUND) { free(wv); free(wn); return NULL; }
        if (got < need) break;
        need = got;                               /* grew between calls */
    }
    free(wn);
    char *v = hx__narrow(wv);
    free(wv);
    if (!v) return NULL;

    const char *res = NULL;
    AcquireSRWLockExclusive(&hx__env_lock);
    for (hx__env_ent *e = hx__env_list; e; e = e->next)
        if (strcmp(e->name, name) == 0 && strcmp(e->value, v) == 0) { res = e->value; break; }
    if (!res) {
        hx__env_ent *e = (hx__env_ent *)malloc(sizeof *e);
        size_t nl = strlen(name) + 1;
        char *nm = (char *)malloc(nl);
        if (e && nm) {
            memcpy(nm, name, nl);
            e->name = nm;
            e->value = v;
            e->next = hx__env_list;
            hx__env_list = e;
            res = v;
            v = NULL;
        } else {
            free(e);
            free(nm);
        }
    }
    ReleaseSRWLockExclusive(&hx__env_lock);
    free(v);
    return res;
}
