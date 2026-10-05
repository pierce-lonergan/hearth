/*
 * store.c — the tiered expert store (hx_store.h): NVMe -> DRAM slots, async readers.
 *
 * Request lifecycle for one (layer, expert) key:
 *   prefetch queue  --reader-->  LOADING slot  --read done-->  READY slot
 *   demand queue    --reader-->  LOADING slot  --read done-->  READY slot
 * A slot is chosen (free or evicted) only when a reader starts the read, so
 * queued requests hold no memory. Demands are served before prefetches; a
 * demand that finds no evictable slot waits ("starved") until a release or a
 * completion makes one evictable. It is never dropped.
 *
 * Slot protection, from strongest to weakest:
 *   pinned       loaded at open from the heat profile, never evicted
 *   refs > 0     acquired by the caller
 *   LOADING      a read is in flight into it
 *   held         a demanded slab not yet acquired (until acquire or token end):
 *                otherwise a burst of demands could evict each other's results.
 *                A caller blocked in acquire/wait_any while the store is starved
 *                drops all holds (see break_holds)
 *   protected    prefetched (or hinted) this token and not yet used: demand may
 *                evict it as a last resort, prefetch never does
 * With at most 2*top_k slots acquired or held by the caller, at most
 * n_io_threads LOADING, and n_slots >= 2*top_k + n_io_threads + 2 unpinned,
 * a demand can always find a slot.
 *
 * A demand read that fails on every attempt (each retry moves to the next
 * mirror) makes acquire return NULL for that expert until the next tick, when
 * it becomes readable again; a prefetch that fails is simply dropped.
 *
 * hx_store_wait_any must not sleep through a completion the caller has not
 * seen. A demand read that finishes (or fails) before the caller looks at that
 * expert again marks it "fresh"; wait_any returns at once while anything is
 * fresh and then counts every completion so far as reported, so a caller that
 * leaves a slab uncollected (or polls an expert that failed) gets one early
 * return, not a spin. Fresh flags are per generation (fresh[k] == wgen + 1),
 * so reporting them all is wgen++.
 *
 * The heat profile (usage_in/usage_out, FORMAT.md §8) always has
 * n_layers*n_experts entries, even when no layer is MoE and there are no keys.
 * It only steers caching, so a missing, mismatched or corrupt one is ignored
 * with a warning.
 *
 * One mutex guards all metadata; reads (hx_file_pread) run outside it.
 *
 * hx_store_set_read_hook (not in hx_store.h) is a test seam: the hook runs on
 * the reader thread before every read attempt, outside the lock, and may block
 * (to order reads deterministically) or fail the attempt.
 */
#include "hx_store.h"
#include "../include/hearth.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SLOT_FREE = 0, SLOT_LOADING = 1, SLOT_READY = 2 };
enum { Q_NONE = 0, Q_PF = 1, Q_DEM = 2 };
enum { JOB_PRE = 0, JOB_DEMAND = 1, JOB_PF = 2 };

#define USAGE_MAGIC   0x53555248u   /* "HRUS" */
#define USAGE_VERSION 1u
#define USAGE_HDR     24u
#define LFU_SAMPLES   8
#define READ_ATTEMPTS 3
#define DECAY_TAB     1024
#define MAX_IO        64
#define MIRROR_SPOT   16             /* slabs whose first page is compared per mirror */
#define FAIL_NOW      1              /* failed[]: the last demand read failed (cleared at tick) */
#define FAIL_EVER     2              /* failed[]: has failed before (quieter logging) */

typedef int (*hx_store_read_hook_fn)(void *ctx, int layer, int expert, int file, int direct);

typedef struct slot {
    int key;                /* layer*E + e, -1 when free */
    int state;
    int refs;
    uint8_t pinned, pf_unused;
    uint64_t hold_tick;     /* held while == now + 1 */
    uint64_t prot_tick;     /* protected while == now + 1 */
    uint64_t last_seq;      /* LRU order (use sequence number) */
    uint64_t last_tick;
} slot;

struct hx_store {
    const hx_modelfile *mf;
    int L, E, top_k, policy, n_io;
    int nkeys;              /* L*E if the file has expert entries, else 0 */
    int n_heat;             /* L*E: length of the heat profile, independent of nkeys */
    uint64_t slot_bytes;
    int n_slots, n_pinned, min_slots;
    uint8_t *arena;
    size_t arena_bytes;
    slot *slots;
    int *slot_of;           /* [nkeys], -1 = no slot */
    uint8_t *kq, *miss_pending, *failed;
    int n_failed_now;       /* keys with FAIL_NOW set */
    double *heat;           /* [nkeys] decayed activation count (LFU), lazily decayed */
    uint64_t *heat_tick;    /* [nkeys] */
    double *count;          /* [nkeys] activations since open, incl. usage seed */
    float *count_f;         /* [n_heat] float mirror of count for hx_store_heat and usage_out */
    uint64_t seed_tokens;   /* tokens_observed of usage_in */
    uint64_t *fresh;        /* [nkeys] == wgen + 1: demand completed, not yet seen by the caller */
    uint64_t wgen;          /* wait_any generation */
    int n_fresh;
    int *dq, dq_head, dq_len;           /* demand ring [nkeys] */
    int *pfq, pfq_len, pfq_max;         /* prefetch FIFO */
    int *pre, pre_head, pre_len;        /* open-time loads, slot already assigned */
    int *free_stack, n_free;
    int loading, pf_inflight, pf_inflight_max, starved, stop, n_ready;
    uint64_t now, seq, rng, completions, ticks, pf_dropped, io_errors, hold_breaks;
    double decay, hot_heat, decay_tab[DECAY_TAB];
    int n_files, rr;
    hx_file **files;
    int *outstanding;
    hx_thread **readers;
    int n_readers;
    int sync_ok;            /* mutex/conds initialised */
    int opened;             /* open completed: close may write usage_out */
    hx_store_read_hook_fn hook;
    void *hook_ctx;
    hx_mutex mu;
    hx_cond work_cv, done_cv;
    hx_store_stats st;
    char *usage_out;
};

/* ------------------------------------------------------------- helpers */

static uint64_t xorshift(uint64_t *x) {
    uint64_t v = *x;
    v ^= v << 13;
    v ^= v >> 7;
    v ^= v << 17;
    return *x = v;
}

static double decay_pow(const hx_store *s, uint64_t dt) {
    return dt < DECAY_TAB ? s->decay_tab[dt] : pow(s->decay, (double)dt);
}

static double eff_heat(const hx_store *s, int key) {
    return s->heat[key] * decay_pow(s, s->now - s->heat_tick[key]);
}

static int is_held(const hx_store *s, const slot *sl) { return sl->hold_tick == s->now + 1; }
static int is_protected(const hx_store *s, const slot *sl) { return sl->prot_tick == s->now + 1; }

static int evictable(const hx_store *s, const slot *sl) {
    return sl->state == SLOT_READY && sl->refs == 0 && !sl->pinned && !is_held(s, sl);
}

/* 1 if slot a is a better eviction victim than slot b. */
static int cheaper(const hx_store *s, int a, int b) {
    const slot *x = &s->slots[a], *y = &s->slots[b];
    int px = is_protected(s, x), py = is_protected(s, y);
    if (px != py) return px < py;
    if (s->policy == HEARTH_POLICY_LFU) {
        double hx_ = eff_heat(s, x->key), hy = eff_heat(s, y->key);
        if (hx_ != hy) return hx_ < hy;
    }
    if (x->last_seq != y->last_seq) return x->last_seq < y->last_seq;
    return a < b;
}

/* LFU: lowest heat among a few sampled slots, full scan if the sample found
 * nothing unprotected. LRU: least recently used by full scan. Pinned slots
 * occupy [0, n_pinned) and are never sampled. Returns -1 if nothing qualifies. */
static int pick_victim(hx_store *s, int allow_protected) {
    int lo = s->n_pinned, n = s->n_slots - lo, best = -1;
    if (n <= 0) return -1;
    if (s->policy == HEARTH_POLICY_LFU) {
        int k = n < LFU_SAMPLES ? n : LFU_SAMPLES;
        for (int i = 0; i < k; i++) {
            int c = lo + (int)(xorshift(&s->rng) % (uint64_t)n);
            if (evictable(s, &s->slots[c]) && (best < 0 || cheaper(s, c, best))) best = c;
        }
        if (best >= 0 && !is_protected(s, &s->slots[best])) return best;
    }
    for (int c = lo; c < s->n_slots; c++)
        if (evictable(s, &s->slots[c]) && (best < 0 || cheaper(s, c, best))) best = c;
    if (best >= 0 && !allow_protected && is_protected(s, &s->slots[best])) return -1;
    return best;
}

static int slot_for_demand(hx_store *s) {
    if (s->n_free) return s->free_stack[--s->n_free];
    return pick_victim(s, 1);
}

/* Called only with the demand queue empty (next_job), so a prefetch never takes
 * a slot a demand needs. It never evicts a protected/held/in-use slot, nor a hot
 * one (LFU: heat of an expert used in about half of all tokens; LRU: used
 * during this token). */
static int slot_for_prefetch(hx_store *s) {
    if (s->n_free) return s->free_stack[--s->n_free];
    int v = pick_victim(s, 0);
    if (v < 0) return -1;
    const slot *sl = &s->slots[v];
    if (s->policy == HEARTH_POLICY_LFU ? eff_heat(s, sl->key) >= s->hot_heat : sl->last_tick == s->now) return -1;
    return v;
}

static void assign(hx_store *s, int si, int key, int demand) {
    slot *sl = &s->slots[si];
    if (sl->state == SLOT_READY) {
        s->slot_of[sl->key] = -1;
        s->n_ready--;
        s->st.evictions++;
        if (sl->pf_unused) s->st.prefetch_wasted++;
    }
    sl->key = key;
    sl->state = SLOT_LOADING;
    sl->refs = 0;
    sl->pf_unused = (uint8_t)!demand;
    sl->hold_tick = demand ? s->now + 1 : 0;
    sl->prot_tick = demand ? 0 : s->now + 1;
    s->slot_of[key] = si;
    s->kq[key] = Q_NONE;
    s->loading++;
}

static int next_job(hx_store *s, int *si, int *key, int *kind) {
    if (s->pre_len) {
        *si = s->pre[s->pre_head++];
        s->pre_len--;
        *key = s->slots[*si].key;
        *kind = JOB_PRE;
        return 1;
    }
    if (s->dq_len) {
        int v = slot_for_demand(s);
        if (v < 0) { s->starved = 1; return 0; }
        int k = s->dq[s->dq_head];
        s->dq_head = (s->dq_head + 1) % s->nkeys;
        s->dq_len--;
        s->starved = 0;
        assign(s, v, k, 1);
        *si = v; *key = k; *kind = JOB_DEMAND;
        return 1;
    }
    s->starved = 0;
    while (s->pfq_len && s->pf_inflight < s->pf_inflight_max) {
        int k = s->pfq[0];
        s->pfq_len--;
        memmove(s->pfq, s->pfq + 1, sizeof *s->pfq * (size_t)s->pfq_len);
        s->kq[k] = Q_NONE;
        int v = slot_for_prefetch(s);
        if (v < 0) { s->pf_dropped++; continue; }
        assign(s, v, k, 0);
        s->pf_inflight++;
        s->st.prefetch_issued++;
        *si = v; *key = k; *kind = JOB_PF;
        return 1;
    }
    return 0;
}

/* Fewest outstanding reads; ties rotate so equal mirrors share the load. */
static int pick_file(hx_store *s) {
    int best = -1;
    for (int i = 0; i < s->n_files; i++) {
        int f = (s->rr + i) % s->n_files;
        if (best < 0 || s->outstanding[f] < s->outstanding[best]) best = f;
    }
    s->rr = (s->rr + 1) % s->n_files;
    return best;
}

/* The caller asked for key and has not collected it: its completion is news. */
static void mark_fresh(hx_store *s, int key) {
    if (s->miss_pending[key] && s->fresh[key] != s->wgen + 1) {
        s->fresh[key] = s->wgen + 1;
        s->n_fresh++;
    }
}

/* The caller is looking at key now. */
static void unfresh(hx_store *s, int key) {
    if (s->fresh[key] == s->wgen + 1) {
        s->fresh[key] = 0;
        s->n_fresh--;
    }
}

static void finish(hx_store *s, int si, int key, int kind, int ok, uint64_t busy) {
    slot *sl = &s->slots[si];
    mark_fresh(s, key);
    s->loading--;
    if (kind == JOB_PF) s->pf_inflight--;
    s->st.read_ns += busy;
    if (ok) {
        sl->state = SLOT_READY;
        sl->last_seq = ++s->seq;
        sl->last_tick = s->now;
        s->n_ready++;
        s->st.reads++;
        s->st.bytes_read += s->mf->experts[key].nbytes;
    } else {
        s->slot_of[key] = -1;
        sl->state = SLOT_FREE;
        sl->key = -1;
        sl->refs = 0;
        sl->hold_tick = sl->prot_tick = 0;
        sl->pf_unused = 0;
        if (!sl->pinned) s->free_stack[s->n_free++] = si;
        if (kind != JOB_PF) {   /* a failed prefetch is simply dropped */
            if (!(s->failed[key] & FAIL_NOW)) s->n_failed_now++;
            s->failed[key] |= FAIL_NOW | FAIL_EVER;
            s->miss_pending[key] = 0;   /* a retry after the tick is a new miss */
            s->io_errors++;
        }
    }
    s->completions++;
    hx_cond_broadcast(&s->done_cv);
    if (s->starved) hx_cond_broadcast(&s->work_cv);
}

static void *reader_main(void *arg) {
    hx_store *s = (hx_store *)arg;
    hx_mutex_lock(&s->mu);
    while (!s->stop) {
        int si, key, kind;
        if (!next_job(s, &si, &key, &kind)) {
            hx_cond_wait(&s->work_cv, &s->mu);
            continue;
        }
        const hx_expert_entry *ent = &s->mf->experts[key];
        uint8_t *dst = s->arena + (size_t)si * s->slot_bytes;
        int ok = 0, f = pick_file(s);
        int lvl_warn = (s->failed[key] & FAIL_EVER) ? HX_LOG_DEBUG : HX_LOG_WARN;
        uint64_t busy = 0;
        for (int attempt = 0; attempt < READ_ATTEMPTS; attempt++) {
            hx_store_read_hook_fn hook = s->hook;
            void *hook_ctx = s->hook_ctx;
            s->outstanding[f]++;
            hx_mutex_unlock(&s->mu);
            uint64_t t0 = hx_now_ns();
            int64_t got = hook && hook(hook_ctx, key / s->E, key % s->E, f, hx_file_is_direct(s->files[f]))
                              ? -1
                              : hx_file_pread(s->files[f], dst, (size_t)ent->nbytes, ent->offset);
            uint64_t t1 = hx_now_ns();
            hx_mutex_lock(&s->mu);
            s->outstanding[f]--;
            busy += t1 - t0;
            if (got == (int64_t)ent->nbytes) { ok = 1; break; }
            hx_log(lvl_warn, "expert (%d, %d): read of %llu bytes at offset %llu from file %d returned %lld",
                   key / s->E, key % s->E, (unsigned long long)ent->nbytes, (unsigned long long)ent->offset, f,
                   (long long)got);
            if (s->stop) break;
            f = (f + 1) % s->n_files;
        }
        if (!ok)
            hx_log(lvl_warn == HX_LOG_WARN ? HX_LOG_ERROR : HX_LOG_DEBUG,
                   "expert (%d, %d): giving up after %d read attempts (retried after the next token)", key / s->E,
                   key % s->E, READ_ATTEMPTS);
        finish(s, si, key, kind, ok, busy);
    }
    hx_mutex_unlock(&s->mu);
    return NULL;
}

/* ------------------------------------------------------- caller side */

static int key_of(const hx_store *s, int layer, int expert) {
    if (!s || !s->n_slots || layer < 0 || layer >= s->L || expert < 0 || expert >= s->E) return -1;
    int k = layer * s->E + expert;
    return s->mf->experts[k].nbytes ? k : -1;
}

static const void *take(hx_store *s, int si, int key) {
    slot *sl = &s->slots[si];
    sl->refs++;
    sl->hold_tick = 0;
    sl->prot_tick = 0;
    sl->last_seq = ++s->seq;
    sl->last_tick = s->now;
    if (sl->pf_unused) { sl->pf_unused = 0; s->st.prefetch_used++; }
    if (s->miss_pending[key]) s->miss_pending[key] = 0;
    else s->st.hits++;
    s->heat[key] = eff_heat(s, key) + 1.0;
    s->heat_tick[key] = s->now;
    s->count[key] += 1.0;
    s->count_f[key] = (float)s->count[key];
    return s->arena + (size_t)si * s->slot_bytes;
}

static void pfq_remove(hx_store *s, int key) {
    for (int i = 0; i < s->pfq_len; i++)
        if (s->pfq[i] == key) {
            s->pfq_len--;
            memmove(s->pfq + i, s->pfq + i + 1, sizeof *s->pfq * (size_t)(s->pfq_len - i));
            return;
        }
}

static void request_demand(hx_store *s, int key) {
    if (s->failed[key] & FAIL_NOW) return;
    if (!s->miss_pending[key]) { s->miss_pending[key] = 1; s->st.misses++; }
    int si = s->slot_of[key];
    if (si >= 0) { s->slots[si].hold_tick = s->now + 1; return; }   /* in flight: promote */
    if (s->kq[key] == Q_DEM) return;
    if (s->kq[key] == Q_PF) pfq_remove(s, key);
    s->kq[key] = Q_DEM;
    s->dq[(s->dq_head + s->dq_len) % s->nkeys] = key;
    s->dq_len++;
    hx_cond_signal(&s->work_cv);
}

const void *hx_store_try_acquire(hx_store *s, int layer, int expert) {
    int key = key_of(s, layer, expert);
    if (key < 0) return NULL;
    const void *p = NULL;
    hx_mutex_lock(&s->mu);
    unfresh(s, key);
    int si = s->slot_of[key];
    if (si >= 0 && s->slots[si].state == SLOT_READY) p = take(s, si, key);
    else request_demand(s, key);
    hx_mutex_unlock(&s->mu);
    return p;
}

/* A blocked caller cannot collect the slabs it demanded earlier, so if every
 * slot a demand could use is held for such a slab, the holds are what stands
 * between it and a deadlock (e.g. a batch demanded more experts than there
 * are slots, then waits for one of the last). Drop them: those slabs become
 * ordinary victims and are re-read if evicted before they are collected. */
static void break_holds(hx_store *s) {
    int broke = 0;
    for (int i = s->n_pinned; i < s->n_slots; i++)
        if (is_held(s, &s->slots[i])) { s->slots[i].hold_tick = 0; broke = 1; }
    s->hold_breaks += (uint64_t)broke;
    hx_cond_broadcast(&s->work_cv);   /* also covers holds that expired at a tick */
}

const void *hx_store_acquire(hx_store *s, int layer, int expert) {
    int key = key_of(s, layer, expert);
    if (key < 0) return NULL;
    const void *p = NULL;
    uint64_t t0 = 0, last_progress = 0, seen = 0;
    int warned = 0, failed = 0;
    hx_mutex_lock(&s->mu);
    for (;;) {
        int si = s->slot_of[key];
        if (si >= 0 && s->slots[si].state == SLOT_READY) { p = take(s, si, key); break; }
        if ((s->failed[key] & FAIL_NOW) || s->stop) { failed = s->failed[key] & FAIL_NOW; break; }
        request_demand(s, key);
        if (s->starved && s->loading == 0) break_holds(s);
        uint64_t now = hx_now_ns();
        if (!t0) { t0 = last_progress = now; seen = s->completions; }
        if (s->completions != seen) { seen = s->completions; last_progress = now; }
        if (!warned && now - last_progress > 10000000000ull) {
            hx_log(HX_LOG_WARN, "expert (%d, %d): no read has completed for 10 s; are all %d cache slots acquired?",
                   layer, expert, s->n_slots - s->n_pinned);
            warned = 1;
        }
        hx_cond_timedwait(&s->done_cv, &s->mu, 100000);
    }
    unfresh(s, key);
    if (t0) s->st.stall_ns += hx_now_ns() - t0;
    hx_mutex_unlock(&s->mu);
    if (failed) hx_log(HX_LOG_DEBUG, "expert (%d, %d) could not be read this token", layer, expert);
    return p;
}

void hx_store_release(hx_store *s, int layer, int expert) {
    int key = key_of(s, layer, expert);
    if (key < 0) return;
    hx_mutex_lock(&s->mu);
    int si = s->slot_of[key];
    if (si < 0 || s->slots[si].state != SLOT_READY || s->slots[si].refs <= 0) {
        hx_log(HX_LOG_WARN, "hx_store_release(%d, %d) without a matching acquire", layer, expert);
    } else if (--s->slots[si].refs == 0 && s->starved) {
        hx_cond_broadcast(&s->work_cv);
    }
    hx_mutex_unlock(&s->mu);
}

void hx_store_wait_any(hx_store *s, uint32_t timeout_us) {
    if (!s || !s->n_slots || timeout_us == 0) return;
    hx_mutex_lock(&s->mu);
    uint64_t t0 = hx_now_ns();
    /* A read the caller has not seen finished already: return at once (without
     * breaking holds: the caller is about to collect it). Otherwise wait for the next
     * completion, or out the timeout even with nothing pending (e.g. only a failed
     * expert is missing), so a try_acquire + wait_any polling loop cannot spin. */
    if (!s->n_fresh) {
        if (s->starved && s->loading == 0) break_holds(s);   /* nothing could complete otherwise */
        uint64_t c0 = s->completions, deadline = t0 + (uint64_t)timeout_us * 1000;
        while (s->completions == c0 && !s->stop) {
            uint64_t now = hx_now_ns();
            if (now >= deadline) break;
            hx_cond_timedwait(&s->done_cv, &s->mu, (uint32_t)((deadline - now + 999) / 1000));
        }
    }
    s->wgen++;   /* every completion so far is reported */
    s->n_fresh = 0;
    s->st.stall_ns += hx_now_ns() - t0;
    hx_mutex_unlock(&s->mu);
}

void hx_store_prefetch(hx_store *s, int layer, const int *experts, int n) {
    if (!s || !s->n_slots || !experts || n <= 0) return;
    int added = 0;
    hx_mutex_lock(&s->mu);
    for (int i = 0; i < n; i++) {
        int key = key_of(s, layer, experts[i]);
        if (key < 0) continue;
        int si = s->slot_of[key];
        if (si >= 0) { s->slots[si].prot_tick = s->now + 1; continue; }   /* resident or in flight */
        if ((s->failed[key] & FAIL_NOW) || s->kq[key] != Q_NONE) continue;
        if (s->pfq_len >= s->pfq_max) { s->pf_dropped++; continue; }
        s->pfq[s->pfq_len++] = key;
        s->kq[key] = Q_PF;
        added = 1;
    }
    if (added) hx_cond_signal(&s->work_cv);
    hx_mutex_unlock(&s->mu);
}

void hx_store_tick(hx_store *s) {
    if (!s) return;
    hx_mutex_lock(&s->mu);
    s->now++;
    s->ticks++;
    if (s->n_failed_now) {   /* failed experts become readable again */
        for (int k = 0; k < s->nkeys; k++) s->failed[k] &= (uint8_t)~FAIL_NOW;
        s->n_failed_now = 0;
    }
    if (s->starved) hx_cond_broadcast(&s->work_cv);   /* expired holds may free a slot */
    hx_mutex_unlock(&s->mu);
}

int hx_store_is_resident(hx_store *s, int layer, int expert) {
    int key = key_of(s, layer, expert);
    if (key < 0) return 0;
    hx_mutex_lock(&s->mu);
    int si = s->slot_of[key];
    int r = si >= 0 && s->slots[si].state == SLOT_READY;
    hx_mutex_unlock(&s->mu);
    return r;
}

void hx_store_get_stats(hx_store *s, hx_store_stats *out) {
    if (!out) return;
    memset(out, 0, sizeof *out);
    if (!s) return;
    hx_mutex_lock(&s->mu);
    *out = s->st;
    out->n_slots = s->n_slots;
    out->resident = s->n_ready;
    out->pinned = s->n_pinned;
    out->slot_bytes = s->slot_bytes;
    hx_mutex_unlock(&s->mu);
}

void hx_store_reset_stats(hx_store *s) {
    if (!s) return;
    hx_mutex_lock(&s->mu);
    memset(&s->st, 0, sizeof s->st);
    hx_mutex_unlock(&s->mu);
}

const float *hx_store_heat(hx_store *s) { return s ? s->count_f : NULL; }

/* Test seam, see the top of this file. */
void hx_store_set_read_hook(hx_store *s, hx_store_read_hook_fn fn, void *ctx);
void hx_store_set_read_hook(hx_store *s, hx_store_read_hook_fn fn, void *ctx) {
    if (!s || !s->sync_ok) return;
    hx_mutex_lock(&s->mu);
    s->hook = fn;
    s->hook_ctx = ctx;
    hx_mutex_unlock(&s->mu);
}

/* ----------------------------------------------------------- usage file */

static void w32le(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static uint32_t r32le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd64le(const uint8_t *p) { return (uint64_t)r32le(p) | ((uint64_t)r32le(p + 4) << 32); }

/* FORMAT.md §8. The profile only steers caching, so a missing, mismatched or corrupt
 * one is ignored (with a warning; usage_out, if it names the same file, replaces it on
 * close). Everything is validated before anything is taken from it. */
static void load_usage(hx_store *s, const char *path) {
    if (!hx_path_exists(path)) {
        hx_log(HX_LOG_INFO, "usage profile %s not found; starting without one", path);
        return;
    }
    char why[400];
    uint8_t *buf = NULL;
    uint64_t want = USAGE_HDR + 4 * (uint64_t)s->n_heat;
    hx_file *f = hx_file_open(path, HX_FILE_READ, why, sizeof why);
    if (!f) goto bad;
    int64_t size = hx_file_size(f);
    if (size != (int64_t)want) {
        hx_fail(why, sizeof why, "it is %lld bytes, expected %llu for %d layers x %d experts", (long long)size,
                (unsigned long long)want, s->L, s->E);
        goto bad;
    }
    buf = (uint8_t *)malloc((size_t)want);
    if (!buf || hx_file_pread(f, buf, (size_t)want, 0) != (int64_t)want) {
        hx_fail(why, sizeof why, buf ? "read failed" : "out of memory");
        goto bad;
    }
    if (r32le(buf) != USAGE_MAGIC || r32le(buf + 4) != USAGE_VERSION) {
        hx_fail(why, sizeof why, "bad magic/version");
        goto bad;
    }
    if (r32le(buf + 8) != (uint32_t)s->L || r32le(buf + 12) != (uint32_t)s->E) {
        hx_fail(why, sizeof why, "it is for %u layers x %u experts, the model has %d x %d", r32le(buf + 8),
                r32le(buf + 12), s->L, s->E);
        goto bad;
    }
    for (int k = 0; k < s->n_heat; k++) {
        float h;
        uint32_t u = r32le(buf + USAGE_HDR + 4 * (size_t)k);
        memcpy(&h, &u, sizeof h);
        if (!isfinite(h) || h < 0.0f) {
            hx_fail(why, sizeof why, "heat entry %d is not a finite non-negative number", k);
            goto bad;
        }
    }
    uint64_t tokens = rd64le(buf + 16);
    for (int k = 0; k < s->n_heat; k++) {
        uint32_t u = r32le(buf + USAGE_HDR + 4 * (size_t)k);
        memcpy(&s->count_f[k], &u, sizeof u);
        if (k >= s->nkeys) continue;
        double h = s->count_f[k];
        s->count[k] = h;
        /* As a per-token rate scaled to the steady state of the decayed counter. */
        s->heat[k] = (tokens > 0 && s->decay < 1.0) ? h / (double)tokens / (1.0 - s->decay) : h;
    }
    s->seed_tokens = tokens;
    free(buf);
    hx_file_close(f);
    return;
bad:
    hx_log(HX_LOG_WARN, "usage profile %s ignored: %s", path, why);
    free(buf);
    if (f) hx_file_close(f);
}

static void save_usage(hx_store *s) {
    size_t n = USAGE_HDR + 4 * (size_t)s->n_heat;
    uint8_t *buf = (uint8_t *)calloc(1, n);
    char err[300];
    if (!buf) { hx_log(HX_LOG_WARN, "usage profile not written: out of memory"); return; }
    uint64_t tokens = s->seed_tokens > UINT64_MAX - s->ticks ? UINT64_MAX : s->seed_tokens + s->ticks;
    w32le(buf, USAGE_MAGIC);
    w32le(buf + 4, USAGE_VERSION);
    w32le(buf + 8, (uint32_t)s->L);
    w32le(buf + 12, (uint32_t)s->E);
    w32le(buf + 16, (uint32_t)tokens);
    w32le(buf + 20, (uint32_t)(tokens >> 32));
    for (int k = 0; k < s->n_heat; k++) {
        uint32_t u;
        memcpy(&u, &s->count_f[k], sizeof u);
        w32le(buf + USAGE_HDR + 4 * (size_t)k, u);
    }
    hx_file *f = hx_file_open(s->usage_out, HX_FILE_WRITE | HX_FILE_CREATE, err, sizeof err);
    if (!f) hx_log(HX_LOG_WARN, "usage profile not written: %s", err);
    else {
        if (hx_file_pwrite(f, buf, n, 0) != (int64_t)n) hx_log(HX_LOG_WARN, "usage profile %s: write failed", s->usage_out);
        hx_file_close(f);
    }
    free(buf);
}

/* ------------------------------------------------------------- mirrors */

static int same_range(hx_file *a, hx_file *b, uint64_t off, uint64_t n, uint8_t *ba, uint8_t *bb, size_t cap) {
    while (n) {
        size_t k = n < cap ? (size_t)n : cap;
        if (hx_file_pread(a, ba, k, off) != (int64_t)k || hx_file_pread(b, bb, k, off) != (int64_t)k) return 0;
        if (memcmp(ba, bb, k)) return 0;
        off += k;
        n -= k;
    }
    return 1;
}

/* A mirror must be byte-identical. Compared: size, preamble, metadata, both
 * directories, and the first page of up to MIRROR_SPOT slabs spread over the file. */
static int check_mirror(const hx_store *s, const char *path, char *err, size_t errlen) {
    const hx_modelfile *mf = s->mf;
    char e2[300];
    hx_file *a = hx_file_open(mf->path, HX_FILE_READ, err, errlen), *b = NULL;
    uint8_t *ba = NULL, *bb = NULL;
    int ok = 0;
    const size_t cap = 1 << 20;
    if (!a) return 0;
    b = hx_file_open(path, HX_FILE_READ, e2, sizeof e2);
    if (!b) { hx_fail(err, errlen, "mirror %s", e2); goto done; }
    if (hx_file_size(b) != (int64_t)mf->file_size || hx_file_size(a) != (int64_t)mf->file_size) {
        hx_fail(err, errlen, "mirror %s: size %lld differs from %s (%llu)", path, (long long)hx_file_size(b), mf->path,
                (unsigned long long)mf->file_size);
        goto done;
    }
    ba = (uint8_t *)malloc(cap);
    bb = (uint8_t *)malloc(cap);
    if (!ba || !bb) { hx_fail(err, errlen, "out of memory"); goto done; }
    if (!same_range(a, b, 0, 64, ba, bb, cap)) { hx_fail(err, errlen, "mirror %s: preamble differs", path); goto done; }
    uint8_t pre[64];
    memcpy(pre, ba, 64);
    uint64_t sec[3][2] = {{rd64le(pre + 8), rd64le(pre + 16)}, {rd64le(pre + 24), rd64le(pre + 32) * 128},
                          {rd64le(pre + 40), rd64le(pre + 48) * 32}};
    for (int i = 0; i < 3; i++) {
        uint64_t off = sec[i][0], n = sec[i][1];
        if (off > mf->file_size || n > mf->file_size - off || !same_range(a, b, off, n, ba, bb, cap)) {
            hx_fail(err, errlen, "mirror %s: header sections differ from %s", path, mf->path);
            goto done;
        }
    }
    if (s->nkeys) {
        int spots = 0;
        for (int i = 0; i < MIRROR_SPOT; i++) {
            int k = (int)(((int64_t)i * s->nkeys) / MIRROR_SPOT);
            while (k < s->nkeys && !mf->experts[k].nbytes) k++;
            if (k >= s->nkeys) break;
            if (!same_range(a, b, mf->experts[k].offset, HX_SLAB_ALIGN, ba, bb, cap)) {
                hx_fail(err, errlen, "mirror %s: expert data differs from %s (expert %d, %d)", path, mf->path, k / s->E,
                        k % s->E);
                goto done;
            }
            spots++;
        }
        (void)spots;
    }
    ok = 1;
done:
    free(ba);
    free(bb);
    if (b) hx_file_close(b);
    hx_file_close(a);
    return ok;
}

/* ----------------------------------------------------------------- open */

typedef struct { double c; int k; } ranked;

static int cmp_ranked(const void *a, const void *b) {
    const ranked *x = (const ranked *)a, *y = (const ranked *)b;
    if (x->c != y->c) return x->c > y->c ? -1 : 1;
    return (x->k > y->k) - (x->k < y->k);
}

/* Pins the hottest experts into slots [0, n_pin), warm-loads the next ones,
 * and waits for those reads. Needs the readers running. */
static int preload(hx_store *s, float pin_fraction, int warm, int n_nonempty, char *err, size_t errlen) {
    ranked *r = (ranked *)malloc(sizeof *r * (size_t)(s->nkeys ? s->nkeys : 1));
    int nr = 0, ok = 1;
    if (!r) { hx_fail(err, errlen, "out of memory"); return 0; }
    for (int k = 0; k < s->nkeys; k++)
        if (s->mf->experts[k].nbytes && s->count[k] > 0) { r[nr].c = s->count[k]; r[nr].k = k; nr++; }
    qsort(r, (size_t)nr, sizeof *r, cmp_ranked);
    /* int(pin_fraction * slots) in double precision, as hearth.sim computes it, for the
     * decimal value the caller wrote: the float option is rounded to 6 decimals first
     * (0.7f * 100 is 69.99999881; 0.7 * 100 is 70). */
    double pf = floor((double)pin_fraction * 1e6 + 0.5) / 1e6;
    int n_pin = (int)(pf * (double)s->n_slots);
    if (s->n_slots < n_nonempty && n_pin > s->n_slots - s->min_slots) n_pin = s->n_slots - s->min_slots;
    if (n_pin > nr) n_pin = nr;
    if (n_pin < 0) n_pin = 0;
    int n_warm = warm ? nr - n_pin : 0;
    if (n_warm > s->n_slots - n_pin) n_warm = s->n_slots - n_pin;

    hx_mutex_lock(&s->mu);
    s->n_pinned = n_pin;
    s->n_free = 0;
    for (int si = s->n_slots - 1; si >= n_pin; si--) s->free_stack[s->n_free++] = si;
    for (int i = 0; i < n_pin + n_warm; i++) {
        int si = i < n_pin ? i : s->free_stack[--s->n_free], k = r[i].k;
        slot *sl = &s->slots[si];
        sl->key = k;
        sl->state = SLOT_LOADING;
        sl->pinned = (uint8_t)(i < n_pin);
        s->slot_of[k] = si;
        s->loading++;
        s->pre[s->pre_len++] = si;
    }
    if (s->pre_len) hx_cond_broadcast(&s->work_cv);
    while (s->loading > 0 || s->pre_len > 0) hx_cond_wait(&s->done_cv, &s->mu);
    for (int i = 0; i < n_pin + n_warm; i++)
        if (s->failed[r[i].k] & FAIL_NOW) {
            hx_fail(err, errlen, "%s: cannot read expert (%d, %d) at open", s->mf->path, r[i].k / s->E, r[i].k % s->E);
            ok = 0;
            break;
        }
    hx_mutex_unlock(&s->mu);
    if (n_pin + n_warm)
        hx_log(HX_LOG_INFO, "expert store: pinned %d and warm-loaded %d experts from the usage profile", n_pin, n_warm);
    free(r);
    return ok;
}

hx_store *hx_store_open(const hx_modelfile *mf, const hx_store_opts *o, char *err, size_t errlen) {
    if (err && errlen) err[0] = 0;
    if (!mf || !o) return (hx_store *)hx_fail(err, errlen, "hx_store_open: NULL model or options");
    if (o->policy != HEARTH_POLICY_LRU && o->policy != HEARTH_POLICY_LFU)
        return (hx_store *)hx_fail(err, errlen, "hx_store_open: unknown eviction policy %d", o->policy);
    if (o->n_mirrors < 0 || (o->n_mirrors > 0 && !o->mirrors))
        return (hx_store *)hx_fail(err, errlen, "hx_store_open: bad mirror list");
    hx_store *s = (hx_store *)calloc(1, sizeof *s);
    if (!s) return (hx_store *)hx_fail(err, errlen, "out of memory");
    const hx_config *c = &mf->cfg;
    s->mf = mf;
    s->L = c->n_layers;
    s->E = c->n_experts;
    s->nkeys = mf->experts ? s->L * s->E : 0;
    s->n_heat = s->E > 0 ? s->L * s->E : 0;
    s->policy = o->policy;
    s->n_io = o->n_io_threads < 1 ? 1 : o->n_io_threads > MAX_IO ? MAX_IO : o->n_io_threads;
    if (s->n_io != o->n_io_threads)
        hx_log(HX_LOG_WARN, "expert store: n_io_threads %d is outside 1..%d, using %d (minimum slots follow)",
               o->n_io_threads, MAX_IO, s->n_io);
    s->top_k = c->top_k > 0 ? c->top_k : 1;
    s->slot_bytes = s->nkeys ? mf->slab_bytes_max : 0;
    s->decay = (o->heat_decay > 0.0f && o->heat_decay <= 1.0f) ? (double)o->heat_decay : 0.995;
    s->hot_heat = s->decay < 1.0 ? 0.5 / (1.0 - s->decay) : HUGE_VAL;
    for (int i = 0; i < DECAY_TAB; i++) s->decay_tab[i] = pow(s->decay, (double)i);
    s->rng = 0x9E3779B97F4A7C15ull;
    s->min_slots = 2 * s->top_k + s->n_io + 2;
    float pin_fraction = o->pin_fraction > 0.0f ? (o->pin_fraction < 0.9f ? o->pin_fraction : 0.9f) : 0.0f;

    int n_nonempty = 0;
    for (int k = 0; k < s->nkeys; k++) n_nonempty += mf->experts[k].nbytes != 0;
    if (n_nonempty) {
        uint64_t want = o->cache_bytes / s->slot_bytes;
        if (want < (uint64_t)s->min_slots) want = (uint64_t)s->min_slots;
        if (want > (uint64_t)n_nonempty) want = (uint64_t)n_nonempty;   /* everything fits: never evicts */
        if (want > SIZE_MAX / s->slot_bytes) {
            hx_store_close(s);
            return (hx_store *)hx_fail(err, errlen, "expert cache too large for this platform");
        }
        s->n_slots = (int)want;
    }

    hx_mutex_init(&s->mu);
    hx_cond_init(&s->work_cv);
    hx_cond_init(&s->done_cv);
    s->sync_ok = 1;

    size_t nk = (size_t)(s->nkeys ? s->nkeys : 1), nh = (size_t)(s->n_heat ? s->n_heat : 1);
    size_t ns = (size_t)(s->n_slots ? s->n_slots : 1);
    s->slots = (slot *)calloc(ns, sizeof *s->slots);
    s->slot_of = (int *)malloc(nk * sizeof *s->slot_of);
    s->kq = (uint8_t *)calloc(nk, 1);
    s->miss_pending = (uint8_t *)calloc(nk, 1);
    s->failed = (uint8_t *)calloc(nk, 1);
    s->heat = (double *)calloc(nk, sizeof *s->heat);
    s->heat_tick = (uint64_t *)calloc(nk, sizeof *s->heat_tick);
    s->count = (double *)calloc(nk, sizeof *s->count);
    s->count_f = (float *)calloc(nh, sizeof *s->count_f);
    s->fresh = (uint64_t *)calloc(nk, sizeof *s->fresh);
    s->dq = (int *)malloc(nk * sizeof *s->dq);
    s->pfq_max = 2 * s->top_k > 2 * s->n_io ? 2 * s->top_k : 2 * s->n_io;
    if (s->pfq_max < 4) s->pfq_max = 4;
    s->pfq = (int *)malloc((size_t)s->pfq_max * sizeof *s->pfq);
    s->pre = (int *)malloc(ns * sizeof *s->pre);
    s->free_stack = (int *)malloc(ns * sizeof *s->free_stack);
    s->pf_inflight_max = s->n_io > 1 ? s->n_io - 1 : 1;
    if (!s->slots || !s->slot_of || !s->kq || !s->miss_pending || !s->failed || !s->heat || !s->heat_tick ||
        !s->count || !s->count_f || !s->fresh || !s->dq || !s->pfq || !s->pre || !s->free_stack) {
        hx_store_close(s);
        return (hx_store *)hx_fail(err, errlen, "out of memory");
    }
    for (size_t k = 0; k < nk; k++) s->slot_of[k] = -1;
    for (size_t i = 0; i < ns; i++) s->slots[i].key = -1;
    for (int i = s->n_slots - 1; i >= 0; i--) s->free_stack[s->n_free++] = i;

    if (o->usage_out && *o->usage_out) {
        size_t n = strlen(o->usage_out) + 1;
        s->usage_out = (char *)malloc(n);
        if (!s->usage_out) { hx_store_close(s); return (hx_store *)hx_fail(err, errlen, "out of memory"); }
        memcpy(s->usage_out, o->usage_out, n);
    }
    if (o->usage_in && *o->usage_in) load_usage(s, o->usage_in);

    if (s->n_slots) {
        s->arena_bytes = (size_t)s->n_slots * (size_t)s->slot_bytes;
        s->arena = (uint8_t *)hx_alloc_large(s->arena_bytes, 1);
        if (!s->arena) {
            hx_store_close(s);
            return (hx_store *)hx_fail(err, errlen, "cannot allocate %.2f GiB for %d expert cache slots",
                                       (double)s->n_slots * (double)s->slot_bytes / (1u << 30), s->n_slots);
        }
        s->n_files = 1 + o->n_mirrors;
        s->files = (hx_file **)calloc((size_t)s->n_files, sizeof *s->files);
        s->outstanding = (int *)calloc((size_t)s->n_files, sizeof *s->outstanding);
        if (!s->files || !s->outstanding) { hx_store_close(s); return (hx_store *)hx_fail(err, errlen, "out of memory"); }
        int flags = HX_FILE_READ | (o->direct_io ? HX_FILE_DIRECT : 0);
        for (int i = 0; i < s->n_files; i++) {
            const char *p = i == 0 ? mf->path : o->mirrors[i - 1];
            if (!p || !*p) { hx_store_close(s); return (hx_store *)hx_fail(err, errlen, "mirror %d: empty path", i - 1); }
            if (i > 0 && !check_mirror(s, p, err, errlen)) { hx_store_close(s); return NULL; }
            s->files[i] = hx_file_open(p, flags, err, errlen);
            if (!s->files[i]) { hx_store_close(s); return NULL; }
            if (o->direct_io && !hx_file_is_direct(s->files[i]))
                hx_log(HX_LOG_INFO, "%s: unbuffered I/O unavailable, using buffered reads", p);
        }
        s->readers = (hx_thread **)calloc((size_t)s->n_io, sizeof *s->readers);
        if (!s->readers) { hx_store_close(s); return (hx_store *)hx_fail(err, errlen, "out of memory"); }
        for (int i = 0; i < s->n_io; i++) {
            if (hx_thread_create(&s->readers[i], reader_main, s) != 0) {
                hx_store_close(s);
                return (hx_store *)hx_fail(err, errlen, "cannot start expert reader thread %d", i);
            }
            s->n_readers++;
        }
        if ((pin_fraction > 0.0f || o->warm_start) && !preload(s, pin_fraction, o->warm_start, n_nonempty, err, errlen)) {
            hx_store_close(s);
            return NULL;
        }
    }
    hx_log(HX_LOG_INFO, "expert store: %d slots x %.2f MiB = %.2f GiB, %d pinned, %d reader(s), %s I/O, %d file(s), %s",
           s->n_slots, (double)s->slot_bytes / (1 << 20), (double)s->arena_bytes / (1u << 30), s->n_pinned, s->n_io,
           o->direct_io ? "direct" : "buffered", s->n_files, s->policy == HEARTH_POLICY_LFU ? "LFU" : "LRU");
    s->opened = 1;
    return s;
}

void hx_store_close(hx_store *s) {
    if (!s) return;
    if (s->sync_ok) {
        hx_mutex_lock(&s->mu);
        s->stop = 1;
        hx_cond_broadcast(&s->work_cv);
        hx_cond_broadcast(&s->done_cv);
        hx_mutex_unlock(&s->mu);
    }
    for (int i = 0; i < s->n_readers; i++) hx_thread_join(s->readers[i]);
    if (s->opened)
        hx_log(HX_LOG_DEBUG, "expert store closed: %llu prefetches dropped, %llu hold breaks, %llu unreadable experts",
               (unsigned long long)s->pf_dropped, (unsigned long long)s->hold_breaks, (unsigned long long)s->io_errors);
    if (s->opened && s->usage_out) save_usage(s);
    for (int i = 0; s->files && i < s->n_files; i++)
        if (s->files[i]) hx_file_close(s->files[i]);
    if (s->arena) hx_free_large(s->arena, s->arena_bytes);
    if (s->sync_ok) {
        hx_cond_destroy(&s->work_cv);
        hx_cond_destroy(&s->done_cv);
        hx_mutex_destroy(&s->mu);
    }
    free(s->readers);
    free(s->files);
    free(s->outstanding);
    free(s->slots);
    free(s->slot_of);
    free(s->kq);
    free(s->miss_pending);
    free(s->failed);
    free(s->heat);
    free(s->heat_tick);
    free(s->count);
    free(s->count_f);
    free(s->fresh);
    free(s->dq);
    free(s->pfq);
    free(s->pre);
    free(s->free_stack);
    free(s->usage_out);
    free(s);
}
