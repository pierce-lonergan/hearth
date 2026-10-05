"""Expert-cache policies and the trace-driven cache simulation.

Keys are ``moe_layer * n_experts + expert``. The simulation walks a *schedule*
(one group per (forward step, MoE layer); a group is the set of distinct experts
that step needs in that layer - k for plain decode, the union over the window for
speculative verification) and classifies every needed expert as

  vram    resident in the static VRAM tier (GPU what-if)
  hit     resident in the DRAM cache before the step needed it
  pfhit   resident because the previous layer's prefetch brought it in
  miss    demand read from storage (``bypass`` counts misses that could not get a slot)
  skip    not evaluated at all (lossy cache-aware routing what-if)

Rules shared by every online policy (mirroring engine/src/hx_store.h):
  * a slot is taken for every demand miss (no bypass) unless every resident slab is
    protected - then the read goes to a scratch buffer (``bypass``);
  * experts of the group being computed are protected (refcount > 0);
  * prefetched experts occupy slots and are protected until used or the forward step
    ends (hx_store_tick): a demand miss evicts one only as a last resort (store.c), a
    prefetch never does; prefetches that would need a protected victim are dropped.

Belady is the exception: it solves the relaxed problem in which any resident may be
evicted at any time, a missed expert may bypass the cache, and the cache may start
with any contents. Every online policy's run (without prefetch and without the lossy
skip knob) is a feasible schedule of that problem, so Belady's hit count over the
measured window is an upper bound for all of them; see _Belady.
"""
from __future__ import annotations

import math
import numbers
import random
from collections import OrderedDict
from dataclasses import dataclass
from heapq import heapify, heappop, heappush, merge

import numpy as np

POLICIES = ("lru", "lru-layer", "lfu", "pinned", "lfu-pinned", "belady")
ONLINE_POLICIES = POLICIES[:-1]
POLICY_HELP = {
    "lru": "global LRU over all (layer, expert) slabs (engine HEARTH_POLICY_LRU baseline)",
    "lru-layer": "LRU partitioned per MoE layer (slots split evenly across layers)",
    "lfu": "decayed-frequency heat, evict the coldest unprotected slab (engine HEARTH_POLICY_LFU)",
    "pinned": "static: hottest slabs from the profile pinned, small LRU scratch for the rest",
    "lfu-pinned": "pin_fraction of slots pinned from the profile, LFU (profile-seeded heat) for the rest",
    "belady": "offline optimum of a relaxed problem (no protection, bypass, warm start): upper bound, not a policy",
}
REJECT, FREE = -2, -1
MAX_IO_THREADS = 64           # hx_store_opts.n_io_threads range is 1..64; store.c clamps larger values


def engine_min_slots(top_k: int, io_threads: int = 8) -> int:
    """hx_store_open's floor on the slot count (hx_store.h), with n_io_threads clamped to 64 as store.c does."""
    if io_threads < 1:
        raise ValueError(f"io_threads must be >= 1, got {io_threads}")
    return 2 * top_k + min(int(io_threads), MAX_IO_THREADS) + 2


# ---- schedule -------------------------------------------------------------

@dataclass
class Schedule:
    acc: np.ndarray        # int64 keys, all groups concatenated
    ptr: np.ndarray        # int64 [G+1] group boundaries
    minrank: np.ndarray    # int32 per access: best rank at which the expert was routed in the window
    pairs: np.ndarray      # int32 [G] (token, expert) routings in the group (compute work)
    step_start: np.ndarray # int64 [S] first trace position of each forward step
    step_npos: np.ndarray  # int32 [S] positions evaluated by the step (1 = plain decode)
    step_nacc: np.ndarray  # int32 [S] tokens produced by the step
    n_moe_layers: int
    n_experts: int
    top_k: int

    @property
    def n_groups(self) -> int:
        return len(self.ptr) - 1

    @property
    def n_steps(self) -> int:
        return len(self.step_start)

    @property
    def max_union(self) -> int:
        return int(np.diff(self.ptr).max()) if self.n_groups else 0


def build_schedule(ids: np.ndarray, n_experts: int, spec_k: int = 0, spec_alpha: float = 0.0,
                   seed: int = 0) -> Schedule:
    """Forward steps over a trace ids[T, L, k]. spec_k > 0: each step verifies a window of up to
    spec_k+1 positions; the number of accepted drafts is geometric (P(accept j drafts or more) =
    alpha**j, truncated at the window) and the next window starts after the last produced token."""
    ids = np.asarray(ids)
    if ids.ndim != 3:
        raise ValueError(f"ids must be [tokens, moe_layers, top_k], got shape {ids.shape}")
    T, L, k = ids.shape
    E = int(n_experts)
    if int(spec_k) < 0 or not 0.0 <= float(spec_alpha) <= 1.0:
        raise ValueError("need spec_k >= 0 and 0 <= spec_alpha <= 1")
    if ids.size:
        if int(ids.min()) < 0 or int(ids.max()) >= E:
            raise ValueError(f"expert ids must be in [0, {E})")
        if k > 1 and not (np.diff(np.sort(ids.astype(np.int64), axis=-1), axis=-1) > 0).all():
            raise ValueError("an expert appears twice in one token's top-k (routing is without replacement)")
    offs = (np.arange(L, dtype=np.int64) * E)
    if spec_k <= 0:
        acc = (ids.astype(np.int64) + offs[None, :, None]).ravel()
        ptr = np.arange(0, T * L * k + 1, k, dtype=np.int64)
        minrank = np.tile(np.arange(k, dtype=np.int32), T * L)
        ones = np.ones(T, dtype=np.int32)
        return Schedule(acc, ptr, minrank, np.full(T * L, k, dtype=np.int32), np.arange(T, dtype=np.int64),
                        ones, ones.copy(), L, E, k)

    rng = np.random.default_rng([seed, 0x5BEC])
    starts, npos, nacc = [], [], []
    pos = 0
    while pos < T:
        n = min(spec_k + 1, T - pos)
        a = 0
        while a < n - 1 and rng.random() < spec_alpha:
            a += 1
        starts.append(pos)
        npos.append(n)
        nacc.append(a + 1)
        pos += a + 1
    accs, lens, mrs = [], [], []
    rank = np.arange(k, dtype=np.int64)
    for s, n in zip(starts, npos):
        w = ids[s:s + n].astype(np.int64)                         # [n, L, k]
        enc = (w * k + rank[None, None, :]).transpose(1, 0, 2).reshape(L, n * k)
        enc.sort(axis=1)
        e, r = enc // k, enc % k
        first = np.ones(e.shape, dtype=bool)
        first[:, 1:] = e[:, 1:] != e[:, :-1]
        lens.append(first.sum(axis=1))
        accs.append((e + offs[:, None])[first])
        mrs.append(r[first])
    acc = np.concatenate(accs)
    ptr = np.zeros(len(starts) * L + 1, dtype=np.int64)
    np.cumsum(np.concatenate(lens), out=ptr[1:])
    pairs = np.repeat(np.asarray(npos, dtype=np.int32) * k, L)
    return Schedule(acc, ptr, np.concatenate(mrs).astype(np.int32), pairs, np.asarray(starts, dtype=np.int64),
                    np.asarray(npos, dtype=np.int32), np.asarray(nacc, dtype=np.int32), L, E, k)


# ---- policies -------------------------------------------------------------

class _LRU:
    """Policy interface (all policies): initial() -> keys resident at the start; touch(hit positions, miss
    positions) once per group; admit(key, next_use, prot) -> FREE, REJECT or the evicted key, never evicting
    a key in `prot` or a held one; hold(keys) / release(keys) bracket a prefetch's protection;
    admit_last_resort(...) may evict a held slab (a demand's last resort); tick(tokens) after each step.

    Held slabs (prefetched, not used yet) are kept out of the recency order, so a cache full of them rejects
    without a scan; released unused, they go back to the position of their admission (the engine stamps a
    prefetched slot when it arrives). Every event gets a unique stamp from a clock shared by the partitions
    of _LRULayer."""
    needs_future = False

    def __init__(self, cap: int, acc: list, skip_mask=None, clock=None):
        self.cap = max(0, int(cap))
        self.od = OrderedDict()            # key -> stamp of its last use or admission, oldest first
        self.held = OrderedDict()          # key -> admission stamp, oldest first
        self.acc = acc
        self.skip = skip_mask              # keys the policy never manages (pinned)
        self.clock = [0] if clock is None else clock

    def initial(self) -> list:
        return []

    def touch(self, hpos, mpos):
        od, held, acc, skip = self.od, self.held, self.acc, self.skip
        t = self.clock[0]
        for i in hpos:
            key = acc[i]
            if skip is not None and skip[key]:
                continue
            t += 1
            try:
                od.move_to_end(key)
            except KeyError:               # a held prefetch, used now
                del held[key]
            od[key] = t
        self.clock[0] = t

    def _put(self, key):
        self.clock[0] = t = self.clock[0] + 1
        self.od[key] = t

    def admit(self, key, nu, prot):
        od = self.od
        if len(od) + len(self.held) < self.cap:
            self._put(key)
            return FREE
        for v in od:
            if v not in prot:
                break
        else:
            return REJECT
        del od[v]
        self._put(key)
        return v

    def admit_last_resort(self, key, nu, prot):
        held = self.held
        for v in held:
            if v not in prot:
                break
        else:
            return REJECT
        del held[v]
        self._put(key)
        return v

    def hold(self, keys):
        od, held = self.od, self.held
        for key in keys:
            held[key] = od.pop(key)

    def release(self, keys):
        held = self.held
        if not held:
            return
        rel = sorted((held.pop(k), k) for k in keys if k in held)
        if not rel:
            return
        od, s0, tail = self.od, rel[0][0], []
        while od:                          # merge by stamp: only the slabs used since rel[0] was admitted move
            k, s = od.popitem()
            if s < s0:
                od[k] = s
                break
            tail.append((s, k))
        tail.reverse()
        for s, k in merge(tail, rel):
            od[k] = s

    def tick(self, n):
        pass


class _LRULayer:
    needs_future = False

    def __init__(self, slots: int, acc: list, n_layers: int, n_experts: int):
        base, extra = divmod(int(slots), n_layers)
        self.clock = [0]
        self.parts = [_LRU(base + (1 if ly < extra else 0), acc, clock=self.clock) for ly in range(n_layers)]
        self.E = n_experts
        self.acc = acc

    def initial(self) -> list:
        return []

    def touch(self, hpos, mpos):
        acc, parts, E = self.acc, self.parts, self.E
        t = self.clock[0]
        for i in hpos:
            key = acc[i]
            p = parts[key // E]
            od = p.od
            t += 1
            try:
                od.move_to_end(key)
            except KeyError:
                del p.held[key]
            od[key] = t
        self.clock[0] = t

    def admit(self, key, nu, prot):
        return self.parts[key // self.E].admit(key, nu, prot)

    def admit_last_resort(self, key, nu, prot):
        return self.parts[key // self.E].admit_last_resort(key, nu, prot)

    def hold(self, keys):
        parts, E = self.parts, self.E
        for key in keys:
            parts[key // E].hold((key,))

    def release(self, keys):
        by, E = {}, self.E
        for key in keys:
            by.setdefault(key // E, []).append(key)
        for ly, ks in by.items():
            self.parts[ly].release(ks)

    def tick(self, n):
        pass


class _LFU:
    """Decayed-frequency heat. heat(t) = sum over uses s of decay**(t - s); stored scaled by
    decay**-t (same order, no per-token sweep). Heat is kept for every expert, resident or not,
    like hx_store_heat. samples > 0 evicts the coldest of `samples` random residents (the engine's
    sampled eviction); samples == 0 evicts the exact coldest."""
    needs_future = False
    _RESCALE = 1e100

    def __init__(self, cap: int, acc: list, n_keys: int, decay: float = 0.995, samples: int = 0,
                 seed: int = 0, heat0=None, pinned=()):
        if not 0.0 < float(decay) <= 1.0:
            raise ValueError(f"LFU decay must be in (0, 1], got {decay}")
        if samples < 0:
            raise ValueError(f"LFU samples must be >= 0, got {samples}")
        self.cap = max(0, int(cap))
        self.acc = acc
        self.decay = float(decay)
        self.heat = [0.0] * n_keys if heat0 is None else [float(h) for h in heat0]
        self.inc = 1.0
        self.res = bytearray(n_keys)       # unpinned residents managed here
        self.rset = set()
        self.heap = []
        self.samples = int(samples)
        self.rng = random.Random(seed)
        self.rlist, self.rpos = [], {}
        self.pinned = list(pinned)
        self.held = set()                  # prefetched, unused (protected): kept out of the heap
        self.hheap = []                    # (heat, key) of held slabs, for the last resort (exact mode)
        self.deferred = []                 # admitted during the current loop (protected): heap entry on flush()
        self.rejected = None               # prot of the last REJECT; it repeats until another method runs

    def flush(self):
        self.rejected = None
        if self.deferred:
            heap, heat, res, held = self.heap, self.heat, self.res, self.held
            for key in self.deferred:
                if res[key] and key not in held:
                    heappush(heap, (heat[key], key))
            self.deferred.clear()

    def hold(self, keys):
        self.rejected = None
        self.held.update(keys)
        if not self.samples:
            heat, hh = self.heat, self.hheap
            for key in keys:
                heappush(hh, (heat[key], key))

    def release(self, keys):
        self.rejected = None
        held, heat, res = self.held, self.heat, self.res
        for key in keys:
            if key in held:
                held.discard(key)
                if res[key] and not self.samples:
                    heappush(self.heap, (heat[key], key))
        if not held:
            self.hheap.clear()

    def initial(self) -> list:
        return self.pinned + list(self.rset)

    def _add(self, key):
        self.res[key] = 1
        self.rset.add(key)
        if self.samples:
            self.rpos[key] = len(self.rlist)
            self.rlist.append(key)
        else:
            self.deferred.append(key)

    def _remove(self, key):
        self.res[key] = 0
        self.rset.discard(key)
        if self.samples:
            i = self.rpos.pop(key)
            last = self.rlist.pop()
            if last != key:
                self.rlist[i] = last
                self.rpos[last] = i

    def touch(self, hpos, mpos):
        self.rejected = None
        heat, inc, acc, res = self.heat, self.inc, self.acc, self.res
        if self.samples:
            for i in hpos:
                heat[acc[i]] += inc
        else:
            heap = self.heap
            for i in hpos:
                key = acc[i]
                h = heat[key] + inc
                heat[key] = h
                if res[key]:
                    heappush(heap, (h, key))
            if len(heap) > 4 * len(self.rset) + 4096:
                self._rebuild()
        for i in mpos:
            heat[acc[i]] += inc

    def _rebuild(self):
        heat, held = self.heat, self.held
        self.heap = [(heat[k], k) for k in self.rset if k not in held]
        heapify(self.heap)
        self.hheap = [(heat[k], k) for k in held]
        heapify(self.hheap)

    def _victim(self, prot, held_protected: bool = True) -> int:
        if self.samples:
            rl, heat, rng = self.rlist, self.heat, self.rng
            hp = self.held if held_protected else ()
            n = len(rl)
            if n == 0:
                return -1
            best, bh = -1, 0.0
            for _ in range(self.samples):
                key = rl[int(rng.random() * n)]
                if key in prot or key in hp:
                    continue
                h = heat[key]
                if best < 0 or h < bh:
                    best, bh = key, h
            if best < 0:                       # all samples protected: exact scan, unless nothing is unprotected
                res = self.res
                if n - len(hp) - sum(1 for x in prot if res[x] and x not in hp) <= 0:
                    return -1
                for key in rl:
                    if key not in prot and key not in hp and (best < 0 or heat[key] < bh):
                        best, bh = key, heat[key]
            return best
        heap, heat, res, held = self.heap, self.heat, self.res, self.held
        aside = []
        v = -1
        while heap:
            h, key = heap[0]
            if not res[key] or heat[key] != h or key in held:   # held keys get a fresh entry on release
                heappop(heap)
                continue
            if key in prot:
                aside.append(heappop(heap))
                continue
            heappop(heap)
            v = key
            break
        for e in aside:
            heappush(heap, e)
        return v

    def _held_victim(self, prot) -> int:
        hh, heat, held = self.hheap, self.heat, self.held
        aside = []
        v = -1
        while hh:
            h, key = heappop(hh)
            if key not in held:
                continue
            if heat[key] != h:
                heappush(hh, (heat[key], key))
                continue
            if key in prot:
                aside.append((h, key))
                continue
            v = key
            break
        for e in aside:
            heappush(hh, e)
        return v

    def admit(self, key, nu, prot):
        # A REJECT under the same (unmodified) `prot` set repeats until any other call changes the state
        if prot is self.rejected:
            if self.samples and self.rlist:
                for _ in range(self.samples):  # the draws _victim would make, so the stream stays the same
                    self.rng.random()
            return REJECT
        self.rejected = None
        if len(self.rset) < self.cap:
            self._add(key)
            return FREE
        v = self._victim(prot)
        if v < 0:
            self.rejected = prot
            return REJECT
        self._remove(v)
        self._add(key)
        return v

    def admit_last_resort(self, key, nu, prot):
        self.rejected = None
        # sampled mode draws its samples even with nothing held, like any other eviction attempt
        v = self._victim(prot, False) if self.samples else self._held_victim(prot)
        if v < 0:
            return REJECT
        self.held.discard(v)
        self._remove(v)
        self._add(key)
        return v

    def tick(self, n):
        self.rejected = None
        self.inc /= self.decay ** n
        if self.inc > self._RESCALE:
            s = 1.0 / self.inc
            self.heat = [h * s for h in self.heat]
            self.inc = 1.0
            if not self.samples:
                self._rebuild()


class _Pinned(_LRU):
    def __init__(self, cap: int, acc: list, pinned: list, n_keys: int):
        mask = bytearray(n_keys)
        for key in pinned:
            mask[key] = 1
        super().__init__(cap, acc, skip_mask=mask)
        self.pinned = list(pinned)

    def initial(self) -> list:
        return list(self.pinned)


class _Belady:
    """Offline optimum of the relaxed problem (no protection, bypass allowed, free initial contents),
    counting only hits at access positions >= `start` (the measured window).

    Per group the state may become any subset of (state | group) of at most `cap` slabs, and hits are
    the group's slabs resident at its start. Keeping the `cap` slabs whose next use is earliest is
    optimal (the exchange argument behind MIN, applied group by group). Before `start` the next use
    that matters is the first one at or after `start`, and the cache starts with the `cap` slabs used
    earliest from `start` on, so the state at `start` is the best possible one. Tests check it against
    exhaustive search over groups with k > 1, several layers and speculative windows.

    `vram`: mask of keys in the static VRAM tier. run_cache serves them before looking at the cache, so
    they never enter it (in any policy); they are kept out of the warm start, where they would hold slots
    for the whole run."""
    needs_future = True

    def __init__(self, cap: int, acc_np: np.ndarray, acc: list, n_keys: int, start: int = 0, vram=None):
        self.cap = max(0, int(cap))
        self.acc = acc
        A = len(acc_np)
        self.INF = A + 1
        self.start = start = min(max(int(start), 0), A)
        order = np.argsort(acc_np, kind="stable")
        skeys = acc_np[order]
        nxt = np.full(A, self.INF, dtype=np.int64)
        same = skeys[1:] == skeys[:-1]
        nxt[order[:-1][same]] = order[1:][same]
        first = np.full(n_keys, self.INF, dtype=np.int64)     # first use at or after start
        if start < A:
            u, i = np.unique(acc_np[start:], return_index=True)
            first[u] = i + start
        if vram is not None:
            first[np.frombuffer(bytes(vram), np.uint8).astype(bool)] = self.INF
        if start > 0:
            nxt[:start] = first[acc_np[:start]]
        self.nxt = nxt.tolist()
        self._order, self._skeys, self._first = order, skeys, first
        self.cur = [-1] * n_keys           # next use of each resident, -1 if not resident
        self.heap = []
        self.n = 0

    def initial(self) -> list:
        fin = np.flatnonzero(self._first < self.INF)
        keys = fin[np.argsort(self._first[fin], kind="stable")][:self.cap].tolist()
        for key in keys:
            nu = int(self._first[key])
            self.cur[key] = nu
            self.heap.append((-nu, key))
        heapify(self.heap)
        self.n = len(keys)
        return keys

    def next_at(self, key: int, pos: int) -> int:
        pos = max(int(pos), self.start)
        a = np.searchsorted(self._skeys, key, side="left")
        b = np.searchsorted(self._skeys, key, side="right")
        ps = self._order[a:b]
        j = np.searchsorted(ps, pos, side="left")
        return int(ps[j]) if j < len(ps) else self.INF

    def touch(self, hpos, mpos):
        acc, nxt, cur, heap = self.acc, self.nxt, self.cur, self.heap
        for i in hpos:
            key = acc[i]
            nu = nxt[i]
            cur[key] = nu
            heappush(heap, (-nu, key))
        if len(heap) > 4 * self.n + 4096:
            self.heap = [(-cur[k], k) for k in range(len(cur)) if cur[k] >= 0]
            heapify(self.heap)

    def admit(self, key, nu, prot):
        # `prot` is ignored on purpose: the relaxation is what makes this an upper bound
        cur, heap = self.cur, self.heap
        if self.n < self.cap:
            self.n += 1
            cur[key] = nu
            heappush(heap, (-nu, key))
            return FREE
        while heap and cur[heap[0][1]] != -heap[0][0]:
            heappop(heap)
        if not heap or nu >= -heap[0][0]:
            return REJECT
        _, v = heappop(heap)
        cur[v] = -1
        cur[key] = nu
        heappush(heap, (-nu, key))
        return v

    def admit_last_resort(self, key, nu, prot):
        return REJECT                      # admit() already ignored protection

    def hold(self, keys):
        pass

    def release(self, keys):
        pass

    def tick(self, n):
        pass


# ---- simulation -------------------------------------------------------------

@dataclass
class CacheCounts:
    """Per-group counters (index = step * n_moe_layers + layer)."""
    hit: np.ndarray
    hit_cold: np.ndarray
    pfhit: np.ndarray
    pfhit_cold: np.ndarray
    miss: np.ndarray
    miss_cold: np.ndarray
    bypass: np.ndarray
    vram: np.ndarray
    skip: np.ndarray
    pf_issued: np.ndarray          # prefetch reads issued during this layer (for the next layer)
    pf_issued_cold: np.ndarray
    pf_wasted: np.ndarray          # of those, not used by the next layer
    pf_wasted_cold: np.ndarray
    evictions: int
    slots: int
    n_pinned: int
    policy: str


@dataclass
class Prefetch:
    """Next-layer prediction: each expert the next layer really needs is predicted with probability
    `recall`; the predictor emits (needed - found + extra) wrong experts (at most the experts left),
    drawn from the layer's observed popularity. Predictions are prefetched while the current layer
    computes."""
    recall: float = 0.8
    extra: int = 0

    def __post_init__(self):
        if not (is_real(self.recall) and 0.0 <= self.recall <= 1.0):
            raise ValueError(f"prefetch recall must be in [0, 1], got {self.recall!r}")
        self.recall = float(self.recall)
        self.extra = as_count("prefetch extra", self.extra)


def is_real(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def as_count(name: str, v, lo: int = 0) -> int:
    """v as an int >= lo: integral reals (4.0) are accepted; inf, NaN, 4.5, True and strings are not."""
    if not (is_real(v) and math.isfinite(v) and v == int(v) and v >= lo):
        raise ValueError(f"{name} must be an integer >= {lo}, got {v!r}")
    return int(v)


def profile_order(heat: np.ndarray, exclude=None) -> np.ndarray:
    """Keys sorted hottest first (stable), optionally excluding a mask."""
    order = np.argsort(-np.asarray(heat, dtype=np.float64), kind="stable")
    if exclude is not None:
        order = order[~np.asarray(exclude, dtype=bool)[order]]
    return order


def make_policy(name: str, slots: int, sched: Schedule, acc: list, *, profile=None, vram=None,
                lfu_decay: float = 0.995, lfu_samples: int = 0, pin_fraction: float = 0.5,
                io_threads: int = 8, seed: int = 0, measure_step: int = 0, heat_scale: float = 1.0):
    nk = sched.n_moe_layers * sched.n_experts
    scratch = min(slots, max(engine_min_slots(sched.top_k, io_threads), sched.max_union + sched.top_k))
    if name == "lru":
        return _LRU(slots, acc)
    if name == "lru-layer":
        return _LRULayer(slots, acc, sched.n_moe_layers, sched.n_experts)
    if name == "lfu":
        return _LFU(slots, acc, nk, decay=lfu_decay, samples=lfu_samples, seed=seed)
    if name in ("pinned", "lfu-pinned"):
        if profile is None:
            raise ValueError(f"policy {name!r} needs a heat profile")
        order = profile_order(profile, exclude=None if vram is None else np.frombuffer(bytes(vram), np.uint8))
        if name == "pinned":
            n_pin = max(0, slots - scratch)
            pins = order[:n_pin].tolist()
            return _Pinned(slots - len(pins), acc, pins, nk)
        n_pin = max(0, min(int(pin_fraction * slots), slots - scratch))
        pins = order[:n_pin].tolist()
        return _LFU(slots - n_pin, acc, nk, decay=lfu_decay, samples=lfu_samples, seed=seed,
                    heat0=np.asarray(profile, dtype=np.float64) * heat_scale, pinned=pins)
    if name == "belady":
        start = int(sched.ptr[min(max(int(measure_step), 0), sched.n_steps) * sched.n_moe_layers])
        return _Belady(slots, sched.acc, acc, nk, start=start, vram=vram)
    raise ValueError(f"unknown policy {name!r}; known: {', '.join(POLICIES)}")


def run_cache(sched: Schedule, policy: str, slots: int, *, profile=None, vram_keys=None, cold_keys=None,
              prefetch: Prefetch | None = None, skip_rank: int | None = None, popularity=None,
              lfu_decay: float = 0.995, lfu_samples: int = 0, pin_fraction: float = 0.5,
              io_threads: int = 8, seed: int = 0, measure_step: int = 0, heat_scale: float = 1.0) -> CacheCounts:
    """Simulate one policy over the schedule. `profile`: heat per key (pinned policies, VRAM tier); pins
    are ranked by it and lfu-pinned seeds its LFU heat with profile * heat_scale.
    `vram_keys` / `cold_keys`: iterables of keys in the static VRAM tier / stored at low precision.
    `popularity`: [L, E] weights for drawing wrong predictions (defaults to uniform).
    `measure_step`: first forward step whose hits count; Belady optimises hits from there on."""
    L, E = sched.n_moe_layers, sched.n_experts
    nk = L * E
    G = sched.n_groups
    acc = sched.acc.tolist()
    ptr = sched.ptr.tolist()
    mr = sched.minrank.tolist() if skip_rank is not None else None
    vram = bytearray(nk)
    if vram_keys is not None:
        for key in vram_keys:
            vram[int(key)] = 1
    cold = bytearray(nk)
    if cold_keys is not None:
        for key in cold_keys:
            cold[int(key)] = 1
    pol = make_policy(policy, slots, sched, acc, profile=profile, vram=vram if vram_keys is not None else None,
                      lfu_decay=lfu_decay, lfu_samples=lfu_samples, pin_fraction=pin_fraction,
                      io_threads=io_threads, seed=seed, measure_step=measure_step, heat_scale=heat_scale)
    fut = pol.needs_future
    nxt = pol.nxt if fut else None
    admit, touch, last_resort = pol.admit, pol.touch, pol.admit_last_resort
    hold, release = pol.hold, pol.release
    flush = getattr(pol, "flush", None)       # call after each admission loop (admitted slabs were protected)

    res = bytearray(nk)
    for key in pol.initial():
        res[key] = 1
    n_pinned = len(getattr(pol, "pinned", ()))

    # prefetch prediction streams (deterministic)
    pf_on = prefetch is not None and (prefetch.recall > 0 or prefetch.extra > 0)
    if pf_on:
        prng = np.random.default_rng([seed, 0x9F37])
        found = (prng.random(len(acc)) < prefetch.recall).tolist()
        pop = np.ones((L, E)) if popularity is None else np.asarray(popularity, dtype=np.float64) + 1e-3
        pop = pop / pop.sum(axis=1, keepdims=True)
        logpop = np.log(pop)
        frng = np.random.default_rng([seed, 0x9F38])     # exact refill when the stream runs dry
        n_draw = 8192            # per layer; consumed cyclically
        fstream = []
        for ly in range(L):
            c = np.cumsum(pop[ly])
            c[-1] = 1.0
            fstream.append((np.searchsorted(c, prng.random(n_draw), side="right").clip(0, E - 1) + ly * E).tolist())
        fptr = [0] * L
    pfp = bytearray(nk)
    pf_live = set()              # prefetched this step and not used yet: protected until used or the step ends

    z = [0] * G
    c_hit, c_hitc, c_ph, c_phc, c_m, c_mc, c_by, c_v, c_sk = (z[:] for _ in range(9))
    c_pf, c_pfc, c_pw, c_pwc = z[:], z[:], z[:], z[:]
    evictions = 0
    nacc = sched.step_nacc.tolist()
    pf_prev = []
    for g in range(G):
        a, b = ptr[g], ptr[g + 1]
        ly = g % L
        hpos, mpos = [], []
        nh = nhc = nph = nphc = nv = nsk = 0
        for i in range(a, b):
            key = acc[i]
            if vram[key]:
                nv += 1
            elif res[key]:
                if pfp[key]:
                    pfp[key] = 0
                    nph += 1
                    nphc += cold[key]
                else:
                    nh += 1
                    nhc += cold[key]
                hpos.append(i)
            elif mr is not None and mr[i] >= skip_rank:
                nsk += 1
            else:
                mpos.append(i)
        touch(hpos, mpos)
        if pf_live:
            used = pf_live.intersection(acc[a:b])
            if used:
                pf_live -= used
                release(used)
        nm = nmc = nby = 0
        if mpos:
            gprot = set(acc[a:b])            # unused prefetches (pf_live) are held: protected by the policy
            live = bool(pf_live)
            for i in mpos:
                key = acc[i]
                nu = nxt[i] if fut else 0
                v = admit(key, nu, gprot)
                if v == REJECT and live:
                    # last resort (store.c): a demand may evict an unused prefetch, a prefetch never does
                    v = last_resort(key, nu, gprot)
                nm += 1
                nmc += cold[key]
                if v == REJECT:
                    nby += 1
                    continue
                res[key] = 1
                if v >= 0:
                    res[v] = 0
                    evictions += 1
                    pf_live.discard(v)
            if flush:
                flush()
        if pf_prev:
            nw = nwc = 0
            for key in pf_prev:
                if pfp[key]:
                    pfp[key] = 0
                    nw += 1
                    nwc += cold[key]
            c_pw[g - 1], c_pwc[g - 1] = nw, nwc
            pf_prev = []
        c_hit[g], c_hitc[g], c_ph[g], c_phc[g] = nh, nhc, nph, nphc
        c_m[g], c_mc[g], c_by[g], c_v[g], c_sk[g] = nm, nmc, nby, nv, nsk

        if pf_on and ly < L - 1:
            a2, b2 = b, ptr[g + 2]
            true = acc[a2:b2]
            pred = [acc[i] for i in range(a2, b2) if found[i]]
            n_false = min((b2 - a2) - len(pred) + prefetch.extra, E - (b2 - a2))
            if n_false > 0:
                tset = set(true)
                fs, fp = fstream[ly + 1], fptr[ly + 1]
                chosen = set()
                tries, lim = 0, 8 * n_false + 64
                while len(chosen) < n_false and tries < lim:
                    key = fs[fp]
                    fp = fp + 1 if fp + 1 < len(fs) else 0
                    tries += 1
                    if key not in tset and key not in chosen:
                        chosen.add(key)
                fptr[ly + 1] = fp
                need = n_false - len(chosen)
                if need > 0:                   # Gumbel-top-k over the experts not yet taken
                    base = (ly + 1) * E
                    gum = logpop[ly + 1] + frng.gumbel(size=E)
                    gum[[key - base for key in tset | chosen]] = -np.inf
                    chosen.update((np.argpartition(-gum, need - 1)[:need] + base).tolist())
                pred.extend(sorted(chosen))
            prot2 = set(acc[a:b])
            prot2.update(pred)
            issued = []
            npf = npfc = 0
            for key in pred:
                if vram[key] or res[key]:
                    continue
                v = admit(key, pol.next_at(key, a2) if fut else 0, prot2)
                if v == REJECT:
                    continue
                hold((key,))
                res[key] = 1
                pfp[key] = 1
                if v >= 0:
                    res[v] = 0
                    evictions += 1
                issued.append(key)
                npf += 1
                npfc += cold[key]
            c_pf[g], c_pfc[g] = npf, npfc
            pf_prev = issued
            pf_live.update(issued)
            if flush:
                flush()
        if ly == L - 1:
            if pf_live:
                release(pf_live)
                pf_live.clear()
            pol.tick(nacc[g // L])

    arr = lambda x: np.asarray(x, dtype=np.int32)  # noqa: E731
    return CacheCounts(arr(c_hit), arr(c_hitc), arr(c_ph), arr(c_phc), arr(c_m), arr(c_mc), arr(c_by),
                       arr(c_v), arr(c_sk), arr(c_pf), arr(c_pfc), arr(c_pw), arr(c_pwc), evictions, int(slots), n_pinned,
                       policy)
