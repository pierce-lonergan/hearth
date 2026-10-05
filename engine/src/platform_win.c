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
#include <locale.h>
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
 * even with the high-resolution timer (plain Sleep(1) took ~2 ms). Should a wait end before
 * hx_now_ns (QPC) has advanced by us, the rest is slept again: the header promises no early return. */
void hx_sleep_us(uint32_t us) {
    if (us == 0) { SwitchToThread(); return; }
    const uint64_t deadline = hx_now_ns() + (uint64_t)us * 1000u;
    HANDLE t = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    for (;;) {
        uint64_t now = hx_now_ns();
        if (now >= deadline) break;
        uint64_t left_us = (deadline - now + 999u) / 1000u;
        LARGE_INTEGER due;
        due.QuadPart = -(LONGLONG)left_us * 10;   /* relative, 100 ns units */
        if (t && SetWaitableTimerEx(t, &due, 0, NULL, NULL, NULL, 0)) WaitForSingleObject(t, INFINITE);
        else Sleep((DWORD)((left_us + 999u) / 1000u));   /* pre-1803 Windows: millisecond sleep */
    }
    if (t) CloseHandle(t);
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

static int hx__popcount(uint64_t m) {
    int n = 0;
    for (; m; m &= m - 1) n++;
    return n;
}

/*
 * The process affinity mask (start /affinity, job objects), or 0 when it does not describe
 * the process: with several processor groups it covers only one group, so there the counts
 * below fall back to every active processor.
 */
static KAFFINITY hx__affinity(void) {
    DWORD_PTR pm = 0, sm = 0;
    if (GetActiveProcessorGroupCount() != 1) return 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm)) return 0;
    return (KAFFINITY)pm;
}

int hx_num_cpus(void) {
    KAFFINITY aff = hx__affinity();
    if (aff) return hx__popcount((uint64_t)aff);
    DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (n == 0) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        n = si.dwNumberOfProcessors;
    }
    return n ? (int)n : 1;
}

/* Cores with at least one logical processor in the affinity mask. */
int hx_num_physical_cores(void) {
    KAFFINITY aff = hx__affinity();
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
                if (e->Relationship == RelationProcessorCore && (!aff || (e->Processor.GroupMask[0].Mask & aff)))
                    cores++;
                off += e->Size;
            }
        }
        free(buf);
    }
    int logical = hx_num_cpus();
    return cores > 0 && cores <= logical ? cores : logical;
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

/* The kernel's millisecond timeout can expire a little before the QPC interval has passed
 * (council measurement: 976 us for a 1000 us wait), so a timeout is reported only after the
 * deadline; an early expiry waits again for the rest. */
int hx_cond_timedwait(hx_cond *c, hx_mutex *m, uint32_t timeout_us) {
    const uint64_t deadline = hx_now_ns() + (uint64_t)timeout_us * 1000u;
    for (;;) {
        uint64_t now = hx_now_ns();
        uint64_t left = deadline > now ? deadline - now : 0;
        DWORD ms = (DWORD)((left + 999999u) / 1000000u);           /* never INFINITE: max ~4.3e6 */
        if (SleepConditionVariableSRW((PCONDITION_VARIABLE)&c->opaque, (PSRWLOCK)&m->opaque, ms, 0)) return 0;
        if (GetLastError() != ERROR_TIMEOUT) return 0;
        if (hx_now_ns() >= deadline) return 1;
    }
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

/*
 * hx_file_replace: a rename that supersedes dst in one step, with no copy fallback (across
 * volumes it fails rather than be non-atomic).
 *
 * MoveFileExW(MOVEFILE_REPLACE_EXISTING) refuses to replace a file that anyone has open, even
 * handles opened with FILE_SHARE_DELETE such as hx_file's (measured: ERROR_ACCESS_DENIED), so
 * the rename uses POSIX semantics (FileRenameInfoEx, Windows 10 1607+, NTFS): open handles
 * keep the old file and the name moves to the new one, as rename(2) does. As rename(2), it
 * also replaces a read-only dst (IGNORE_READONLY_ATTRIBUTE, 1809+; without that flag it is
 * retried without it) and moves a symbolic link tmp itself, not its target. Volumes or systems
 * without POSIX renames (FAT, SMB and 9P shares, older Windows) use MoveFileExW with
 * MOVEFILE_WRITE_THROUGH: there any open handle on dst, and a read-only dst, make the replace
 * fail. A holder that did not share delete access (the CRT's fopen, some virus scanners)
 * blocks both ways; that error is retried for up to HX__REPLACE_RETRY_MS.
 * Directories are refused up front: replacing one is never intended, and under app-container
 * file virtualization MoveFileExW was seen to report success doing it.
 */
#define HX__REPLACE_RETRY_MS 250
#define HX__FILE_RENAME_INFO_EX ((FILE_INFO_BY_HANDLE_CLASS)22)   /* FileRenameInfoEx */
#ifndef FILE_RENAME_FLAG_REPLACE_IF_EXISTS
#  define FILE_RENAME_FLAG_REPLACE_IF_EXISTS 0x1
#endif
#ifndef FILE_RENAME_FLAG_POSIX_SEMANTICS
#  define FILE_RENAME_FLAG_POSIX_SEMANTICS 0x2
#endif
#ifndef FILE_RENAME_FLAG_IGNORE_READONLY_ATTRIBUTE
#  define FILE_RENAME_FLAG_IGNORE_READONLY_ATTRIBUTE 0x40
#endif
#ifndef ERROR_DIRECTORY_NOT_SUPPORTED
#  define ERROR_DIRECTORY_NOT_SUPPORTED 336L
#endif

/* FILE_RENAME_INFO with the Flags member of FileRenameInfoEx (older SDKs lack the union). */
typedef struct hx__rename_info { DWORD flags; HANDLE root; DWORD name_bytes; WCHAR name[1]; } hx__rename_info;
_Static_assert(offsetof(hx__rename_info, name) == offsetof(FILE_RENAME_INFO, FileName), "FILE_RENAME_INFO layout");

static wchar_t *hx__full_path(const wchar_t *w) {
    if (wcsncmp(w, L"\\\\?\\", 4) == 0) return _wcsdup(w);   /* hx__widen made it absolute */
    DWORD n = GetFullPathNameW(w, 0, NULL, NULL);
    wchar_t *abs = n ? (wchar_t *)malloc((size_t)n * sizeof(wchar_t)) : NULL;
    if (abs && GetFullPathNameW(w, n, abs, NULL) >= n) { free(abs); abs = NULL; }
    return abs;
}

/* 0 on success, else a Win32 error code. The handle is on tmp's own name (a link is renamed
 * as a link). */
static DWORD hx__rename_posix(const wchar_t *from, const wchar_t *to_abs, DWORD flags) {
    HANDLE h = CreateFileW(from, DELETE | SYNCHRONIZE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
    if (h == INVALID_HANDLE_VALUE) return GetLastError();
    size_t len = wcslen(to_abs);
    size_t bytes = offsetof(hx__rename_info, name) + (len + 1) * sizeof(WCHAR);
    hx__rename_info *ri = (hx__rename_info *)calloc(1, bytes);
    DWORD e = ERROR_NOT_ENOUGH_MEMORY;
    if (ri && bytes <= MAXDWORD) {
        ri->flags = flags;
        ri->name_bytes = (DWORD)(len * sizeof(WCHAR));
        memcpy(ri->name, to_abs, (len + 1) * sizeof(WCHAR));
        e = SetFileInformationByHandle(h, HX__FILE_RENAME_INFO_EX, ri, (DWORD)bytes) ? 0 : GetLastError();
    }
    free(ri);
    CloseHandle(h);
    return e;
}

/* What a file system or Windows version without a rename flavour answers. */
static int hx__unsupported(DWORD e) {
    return e == ERROR_INVALID_PARAMETER || e == ERROR_NOT_SUPPORTED || e == ERROR_INVALID_FUNCTION;
}

int hx_file_replace(const char *tmp_path, const char *dst_path) {
    if (!tmp_path || !*tmp_path || !dst_path || !*dst_path) return -1;
    wchar_t *wt = hx__widen(tmp_path, 1), *wd = hx__widen(dst_path, 1);
    wchar_t *wabs = wd ? hx__full_path(wd) : NULL;
    int rc = -1;
    DWORD e = ERROR_INVALID_NAME;
    if (wt && wd && wabs) {
        DWORD at = GetFileAttributesW(wt);
        e = at == INVALID_FILE_ATTRIBUTES ? GetLastError() : 0;
        DWORD ad = GetFileAttributesW(wd);
        if (ad == INVALID_FILE_ATTRIBUTES) ad = 0;             /* dst need not exist */
        if (e == 0 && ((at | ad) & FILE_ATTRIBUTE_DIRECTORY)) e = ERROR_DIRECTORY_NOT_SUPPORTED;
        const uint64_t deadline = hx_now_ns() + (uint64_t)HX__REPLACE_RETRY_MS * 1000000u;
        const DWORD posix = FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS;
        while (e == 0) {
            e = hx__rename_posix(wt, wabs, posix | FILE_RENAME_FLAG_IGNORE_READONLY_ATTRIBUTE);
            if (hx__unsupported(e)) e = hx__rename_posix(wt, wabs, posix);
            if (hx__unsupported(e))
                e = MoveFileExW(wt, wd, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : GetLastError();
            if (e == 0) { rc = 0; break; }
            int busy = e == ERROR_SHARING_VIOLATION || e == ERROR_LOCK_VIOLATION;
            if (e == ERROR_ACCESS_DENIED) {       /* also what a read-only dst gives, which will not clear */
                DWORD a = GetFileAttributesW(wd);
                busy = a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_READONLY);
            }
            if (!busy || hx_now_ns() >= deadline) break;
            hx_sleep_us(1000);
            e = 0;
        }
    }
    if (rc != 0) {
        char msg[300];
        hx__winerr(msg, sizeof msg, e);
        hx_log(HX_LOG_DEBUG, "hx_file_replace %s -> %s: %s", tmp_path, dst_path, msg);
    }
    free(wt);
    free(wd);
    free(wabs);
    return rc;
}

/* ------------------------------------------------------------ cpu features */

static hx_cpu hx__cpu;
static INIT_ONCE hx__cpu_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK hx__cpu_init(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    (void)once; (void)param; (void)ctx;
    memset(&hx__cpu, 0, sizeof hx__cpu);
#if defined(HX_ARCH_X86_64)
    hx__detect_x86(&hx__cpu, 0);
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
 * Values are read from the process environment as UTF-16 and returned as UTF-8 copies owned
 * here. Per name, the current value and the HX__ENV_KEEP values before it are kept: a returned
 * pointer stays valid until the variable has changed HX__ENV_KEEP more times, and memory stays
 * bounded when a polled variable keeps changing. An unchanged value returns the same pointer.
 */
#define HX__ENV_KEEP 8
typedef struct hx__env_ent {
    struct hx__env_ent *next;
    char *name;
    char *value;
    char *old[HX__ENV_KEEP];  /* ring of earlier values */
    int old_next;
} hx__env_ent;
static SRWLOCK hx__env_lock = SRWLOCK_INIT;
static hx__env_ent *hx__env_list;

/* Reading into a buffer, GetEnvironmentVariableW returns 0 both for "not found" and for an
 * empty value, and leaves the last error untouched in the second case: clear it first. (The
 * size query cannot be confused: it counts the terminator, so an empty value needs 1.) */
static wchar_t *hx__getenv_w(const wchar_t *wn) {
    DWORD need = GetEnvironmentVariableW(wn, NULL, 0);
    if (need == 0) return NULL;
    wchar_t *wv = NULL;
    for (;;) {
        wchar_t *nw = (wchar_t *)realloc(wv, (size_t)need * sizeof(wchar_t));
        if (!nw) { free(wv); return NULL; }
        wv = nw;
        SetLastError(ERROR_SUCCESS);
        DWORD got = GetEnvironmentVariableW(wn, wv, need);
        if (got == 0 && GetLastError() == ERROR_ENVVAR_NOT_FOUND) { free(wv); return NULL; }
        if (got < need) { wv[got] = 0; return wv; }
        need = got;                               /* grew between calls */
    }
}

const char *hx_env_str(const char *name) {
    if (!name || !*name) return NULL;
    wchar_t *wn = hx__widen(name, 0);
    if (!wn) return NULL;
    wchar_t *wv = hx__getenv_w(wn);
    free(wn);
    if (!wv) return NULL;
    char *v = hx__narrow(wv);
    free(wv);
    if (!v) return NULL;

    const char *res = NULL;
    AcquireSRWLockExclusive(&hx__env_lock);
    hx__env_ent *e = hx__env_list;
    while (e && strcmp(e->name, name) != 0) e = e->next;
    if (!e) {
        size_t nl = strlen(name) + 1;
        e = (hx__env_ent *)calloc(1, sizeof *e);
        char *nm = (char *)malloc(nl);
        if (e && nm) {
            memcpy(nm, name, nl);
            e->name = nm;
            e->next = hx__env_list;
            hx__env_list = e;
        } else {
            free(e);
            free(nm);
            e = NULL;
        }
    }
    if (e) {
        if (!e->value || strcmp(e->value, v) != 0) {
            free(e->old[e->old_next]);
            e->old[e->old_next] = e->value;       /* still valid for whoever holds it */
            e->old_next = (e->old_next + 1) % HX__ENV_KEEP;
            e->value = v;
            v = NULL;
        }
        res = e->value;
    }
    ReleaseSRWLockExclusive(&hx__env_lock);
    free(v);
    return res;
}

/* strtod in the "C" locale, whatever LC_NUMERIC the host process has set. */
static _locale_t hx__c_locale;
static INIT_ONCE hx__c_locale_once = INIT_ONCE_STATIC_INIT;

static BOOL CALLBACK hx__c_locale_init(PINIT_ONCE once, PVOID param, PVOID *ctx) {
    (void)once; (void)param; (void)ctx;
    hx__c_locale = _create_locale(LC_NUMERIC, "C");   /* kept for the life of the process */
    return TRUE;
}

static double hx__strtod_c(const char *s, char **end) {
    InitOnceExecuteOnce(&hx__c_locale_once, hx__c_locale_init, NULL, NULL);
    return hx__c_locale ? _strtod_l(s, end, hx__c_locale) : strtod(s, end);
}
