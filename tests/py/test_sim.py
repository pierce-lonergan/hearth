"""Tests for hearth.sim (task T04-sim)."""
from __future__ import annotations

import dataclasses
import itertools
import json
import math
import struct
import time

import numpy as np
import pytest

from hearth import presets
from hearth.sim import (HARDWARE, ONLINE_POLICIES, POLICIES, Calibration, Lossy, Prefetch, Spec, Trace,
                        build_schedule, calibrate, engine_min_slots, feasibility, get_hardware, levers, load_usage,
                        main, model_costs, parse_nvme, retime, run_cache, save_usage, simulate, slab_bytes, sweep,
                        synthetic, synthetic_for)
from hearth.sim.timing import COMPONENTS, evaluate
from hearth.sim.trace import zipf_popularity

SMALL = "olmoe-1b-7b"     # 16 MoE layers x 64 experts, top-8: fast


def small(**kw):
    base = dict(tokens=600, zipf=1.2, reuse=0.0, seed=3)
    base.update(kw)
    return simulate(SMALL, "this-pc", **base)


# ---- traces ------------------------------------------------------------------

def test_trace_file_round_trip(tmp_path):
    tr = synthetic(57, 5, 40, 3, zipf=0.9, reuse=0.3, seed=7, n_layers=6)
    p = tr.save(tmp_path / "t.hrtr")
    raw = p.read_bytes()
    # FORMAT.md sec. 9: 6 x u32 header then u16 ids, exactly as the golden tests read it
    assert len(raw) == 24 + 57 * 5 * 3 * 2
    assert struct.unpack_from("<6I", raw, 0) == (0x52545248, 1, 6, 40, 3, 5)
    assert np.array_equal(np.frombuffer(raw, "<u2", offset=24).reshape(-1, 5, 3), tr.ids)
    back = Trace.load(p)
    assert np.array_equal(back.ids, tr.ids)
    assert (back.n_layers, back.n_experts, back.top_k, back.n_moe_layers, back.n_tokens) == (6, 40, 3, 5, 57)


def test_trace_load_rejects_malformed(tmp_path):
    tr = synthetic(4, 2, 8, 2, seed=1)
    good = tr.save(tmp_path / "g.hrtr").read_bytes()
    cases = {"magic": b"XXXX" + good[4:], "version": good[:4] + struct.pack("<I", 2) + good[8:],
             "short": good[:10], "ragged": good[:-1], "k_gt_e": good[:16] + struct.pack("<I", 9) + good[20:]}
    for name, blob in cases.items():
        p = tmp_path / f"{name}.hrtr"
        p.write_bytes(blob)
        with pytest.raises(ValueError):
            Trace.load(p)
    with pytest.raises(ValueError):
        Trace(np.full((2, 2, 2), 8, dtype=np.uint16), n_experts=8)


def test_synthetic_deterministic_and_valid():
    a = synthetic(300, 6, 32, 4, zipf=1.1, reuse=0.3, seed=11)
    b = synthetic(300, 6, 32, 4, zipf=1.1, reuse=0.3, seed=11)
    c = synthetic(300, 6, 32, 4, zipf=1.1, reuse=0.3, seed=12)
    assert np.array_equal(a.ids, b.ids)
    assert not np.array_equal(a.ids, c.ids)
    s = np.sort(a.ids.astype(int), axis=-1)
    assert (np.diff(s, axis=-1) > 0).all(), "top-k must be without replacement"
    assert a.ids.max() < 32
    z = synthetic(300, 6, 32, 4, zipf=1.1, reuse=0.0, seed=11)
    assert (np.diff(np.sort(z.ids.astype(int), axis=-1), axis=-1) > 0).all()


def test_synthetic_matches_successive_sampling():
    """Stream-dedupe sampler == Plackett-Luce / Gumbel-top-k marginals."""
    E, k, T, seed = 10, 3, 30000, 5
    tr = synthetic(T, 1, E, k, zipf=1.4, reuse=0.0, seed=seed)
    p = zipf_popularity(E, 1, 1.4, np.random.default_rng(seed))[0]   # same first draw as synthetic()
    emp = np.bincount(tr.ids.ravel(), minlength=E) / T
    g = np.log(p)[None, :] + np.random.default_rng(1).gumbel(size=(T, E))
    ref = np.bincount(np.argpartition(-g, k - 1, axis=1)[:, :k].ravel(), minlength=E) / T
    assert np.abs(emp - ref).max() < 0.02
    # popularity order is respected: the most popular expert is chosen most often
    assert int(np.argmax(emp)) == int(np.argmax(p))


def test_trace_stats():
    hi = synthetic(800, 4, 64, 4, zipf=1.3, reuse=0.0, seed=2).stats()
    flat = synthetic(800, 4, 64, 4, zipf=0.0, reuse=0.0, seed=2).stats()
    sticky = synthetic(800, 4, 64, 4, zipf=0.0, reuse=0.9, seed=2).stats()
    assert flat.entropy_norm.min() > 0.97
    assert hi.entropy_norm.max() < flat.entropy_norm.min()
    assert hi.mass_top[0.10] > flat.mass_top[0.10] + 0.2
    assert abs(flat.reuse_rate.mean() - 4 / 64) < 0.03            # chance overlap only
    assert sticky.reuse_rate.mean() > 0.88
    assert flat.union_per_window[1] == 4.0
    assert sticky.union_per_window[5] < flat.union_per_window[5]
    assert hi.slots_for_mass[0.5] < flat.slots_for_mass[0.5]
    assert flat.slots_for_mass[0.95] <= 4 * 64


# ---- cache policies -------------------------------------------------------------

def _opt_group_hits(groups, cap, start_group=0):
    """Exhaustive optimum of the relaxed problem Belady solves: any initial contents (<= cap slabs); per group,
    hits = |group & state| (counted from start_group on) and the next state is any subset of state | group
    with <= cap slabs. Maximal subsets suffice, since a superset state is never worse."""
    keys = sorted({x for grp in groups for x in grp})
    states = {frozenset(c): 0 for c in itertools.combinations(keys, min(cap, len(keys)))}
    for g, grp in enumerate(groups):
        G = frozenset(grp)
        new = {}
        for C, h in states.items():
            h2 = h + (len(G & C) if g >= start_group else 0)
            pool = sorted(C | G)
            for nxt in itertools.combinations(pool, min(cap, len(pool))):
                nxt = frozenset(nxt)
                if new.get(nxt, -1) < h2:
                    new[nxt] = h2
        states = new
    return max(states.values())


def _random_ids(rng, T, L, E, k, zipf=0.0):
    p = (np.arange(E) + 1.0) ** -zipf
    p /= p.sum()
    return np.array([[rng.choice(E, size=k, replace=False, p=p) for _ in range(L)] for _ in range(T)],
                    dtype=np.uint16)


def _groups(sched):
    return [sched.acc[sched.ptr[g]:sched.ptr[g + 1]].tolist() for g in range(sched.n_groups)]


def test_belady_is_exactly_optimal_on_small_schedules():
    """Belady == exhaustive optimum of the relaxed group problem with k > 1, several layers, speculative
    windows and a warm-up prefix, and >= every online policy (no prefetch) over the measured groups."""
    ex = build_schedule(np.array([[2, 1], [1, 0], [2, 1], [1, 0]], dtype=np.uint16).reshape(4, 1, 2), 3)
    assert int(run_cache(ex, "belady", 1).hit.sum()) == _opt_group_hits(_groups(ex), 1)
    assert int(run_cache(ex, "belady", 1).hit.sum()) >= int(run_cache(ex, "lru", 1).hit.sum()) == 2
    rng = np.random.default_rng(0)
    n_spec = n_multi = 0
    for trial in range(80):
        L, E = int(rng.integers(1, 3)), int(rng.integers(2, 5))
        k = int(rng.integers(1, E + 1))
        ids = _random_ids(rng, int(rng.integers(3, 10)), L, E, k, zipf=float(rng.uniform(0, 1.5)))
        spec = (int(rng.integers(1, 4)), float(rng.uniform(0.2, 0.9))) if trial % 3 == 0 else (0, 0.0)
        sched = build_schedule(ids, E, *spec, seed=trial)
        cap = int(rng.integers(1, min(L * E, 3) + 1))
        ms = int(rng.integers(0, sched.n_steps))
        g0 = ms * L
        got = int(run_cache(sched, "belady", cap, measure_step=ms).hit[g0:].sum())
        assert got == _opt_group_hits(_groups(sched), cap, g0), (trial, ids.tolist(), spec, cap, ms)
        prof = rng.random(L * E)
        for pol in ONLINE_POLICIES:
            assert int(run_cache(sched, pol, cap, profile=prof, io_threads=1).hit[g0:].sum()) <= got, (trial, pol)
        n_spec += spec[0] > 0
        n_multi += k > 1 and L > 1
    assert n_spec > 20 and n_multi > 10


def test_belady_bounds_online_policies_fuzz():
    """Larger random schedules (pinned slabs, sampled LFU, speculative windows, warm-up): Belady's hits over
    the measured groups are never below any online policy's."""
    rng = np.random.default_rng(7)
    for trial in range(120):
        L, E = int(rng.integers(1, 4)), int(rng.integers(4, 13))
        k = int(rng.integers(1, min(E, 4) + 1))
        ids = _random_ids(rng, int(rng.integers(5, 40)), L, E, k, zipf=float(rng.uniform(0, 1.5)))
        spec = (int(rng.integers(1, 4)), float(rng.uniform(0.2, 0.9))) if trial % 2 else (0, 0.0)
        sched = build_schedule(ids, E, *spec, seed=trial)
        cap = int(rng.integers(1, L * E))
        ms = int(rng.integers(0, sched.n_steps))
        g0 = ms * L
        best = int(run_cache(sched, "belady", cap, measure_step=ms).hit[g0:].sum())
        prof = rng.random(L * E)
        for pol in ONLINE_POLICIES:
            for kw in ({}, {"lfu_samples": 2, "lfu_decay": 0.8, "pin_fraction": 0.9}):
                c = run_cache(sched, pol, cap, profile=prof, io_threads=1, seed=trial, **kw)
                assert int(c.hit[g0:].sum()) <= best, (trial, pol, kw)


def test_belady_upper_bound_through_simulate_on_a_tiny_shape():
    """The verifier's counterexample class: one MoE layer, E=17, k=3, engine-minimum 9 slots, speculative
    windows, with and without warm-up."""
    # found by random search: generation 1's protected, cold-start Belady scored 0.692 here, lru 0.697
    ids = [[12, 1, 4], [16, 12, 3], [1, 9, 17], [1, 15, 13], [6, 2, 17], [8, 0, 3], [6, 0, 7], [6, 4, 18],
           [16, 10, 14], [3, 18, 17], [11, 0, 3], [14, 1, 19], [19, 1, 6], [13, 5, 1], [6, 10, 0], [11, 6, 4],
           [1, 16, 4], [13, 10, 9], [18, 16, 10], [16, 4, 0], [5, 1, 14], [7, 3, 2], [19, 5, 1], [17, 18, 0],
           [4, 19, 9], [10, 4, 19], [5, 8, 12], [4, 8, 2], [10, 14, 3], [16, 14, 12], [3, 4, 1], [4, 11, 19],
           [10, 5, 17], [12, 19, 5], [4, 6, 5], [8, 11, 18], [0, 12, 2]]
    shape20 = dataclasses.replace(presets.get(SMALL), name="tiny20", n_layers=1, n_experts=20, top_k=3)
    kw = dict(trace=Trace(np.array(ids, dtype=np.uint16)[:, None, :], 20), cache_gb=0.0, io_threads=1,
              spec=(3, 0.3051467409487659), warmup=0, seed=1347)
    res = {p: simulate(shape20, "this-pc", policy=p, **kw) for p in POLICIES}
    assert res["belady"].slots == 9 and res["lru"].hit_rate > 0.69
    for p in ONLINE_POLICIES:
        assert res["belady"].hit_rate >= res[p].hit_rate, p
    tiny = dataclasses.replace(presets.get(SMALL), name="tiny", n_layers=1, n_experts=17, top_k=3)
    for seed in range(6):
        tr = synthetic_for(tiny, 120, zipf=0.4 + 0.15 * seed, reuse=0.3, seed=seed)
        for spec, warmup in (((3, 0.5), 0), (None, 30), ((2, 0.8), 12)):
            res = {p: simulate(tiny, "this-pc", trace=tr, policy=p, cache_gb=0.0, io_threads=1, spec=spec,
                               warmup=warmup) for p in POLICIES}
            assert res["belady"].slots == 9
            for p in ONLINE_POLICIES:
                assert res["belady"].hit_rate >= res[p].hit_rate - 1e-12, (seed, spec, warmup, p)


@pytest.mark.parametrize("cache_gb,spec", [(0.0, None), (0.2, None), (0.6, None), (2.0, None),
                                           (0.3, Spec(3, 0.7)), (1.0, Spec(4, 0.5))])
def test_belady_upper_bounds_every_online_policy(cache_gb, spec):
    res = {p: small(policy=p, cache_gb=cache_gb, spec=spec) for p in POLICIES}
    for p in ONLINE_POLICIES:
        assert res["belady"].hit_rate >= res[p].hit_rate - 1e-12, (p, res[p].hit_rate, res["belady"].hit_rate)
    assert len({r.slots for r in res.values()}) == 1


def test_lru_cyclic_pathology():
    shape = presets.get(SMALL)
    working_set = shape.n_moe_layers * shape.top_k                    # 128 slabs per token
    lru = small(policy="lru", cache_gb=0.2)
    lfu = small(policy="lfu", cache_gb=0.2)
    assert lru.slots < working_set
    assert lru.hit_rate < 0.005
    assert lfu.hit_rate > 0.2
    # once the cache holds a token's working set, LRU recovers
    big = small(policy="lru", cache_gb=1.0)
    assert big.slots > working_set and big.hit_rate > 0.4


def test_min_slots_and_cache_clamp():
    r = small(cache_gb=0.0)
    assert r.slots == engine_min_slots(presets.get(SMALL).top_k, 8) == 26
    huge = small(cache_gb=10_000.0)
    assert huge.slots == 16 * 64 and huge.hit_rate > 0.99
    assert any("every expert is cached" in w for w in huge.warnings)
    assert huge.feasible and huge.cache_gib < 4.0                     # what is allocated fits, not what was asked


def test_lfu_variants():
    exact = small(policy="lfu", cache_gb=0.5)
    sampled = small(policy="lfu", cache_gb=0.5, lfu_samples=64)
    assert abs(exact.hit_rate - sampled.hit_rate) < 0.03
    # strong decay forces the heat rescale path (inc > 1e100) many times
    fast = small(policy="lfu", cache_gb=0.5, lfu_decay=0.5)
    assert 0.0 < fast.hit_rate <= 1.0
    lfu_pin = small(policy="lfu-pinned", cache_gb=0.5, pin_fraction=0.5)
    assert lfu_pin.n_pinned == int(0.5 * lfu_pin.slots)
    pinned = small(policy="pinned", cache_gb=0.5)
    assert pinned.n_pinned == pinned.slots - max(26, 8 + 8)


# ---- byte accounting and timing ------------------------------------------------------

def test_byte_accounting_without_prefetch():
    r = small(cache_gb=0.5)
    c = r._costs
    shape = presets.get(SMALL)
    assert r.loads_per_token == shape.n_moe_layers * shape.top_k
    assert math.isclose(r.hit_rate + r.prefetch_rate + r.miss_rate + r.skip_rate, 1.0)
    b = r.bytes_per_token
    assert math.isclose(b["nvme"], r.miss_rate * r.loads_per_token * r.slab_bytes, rel_tol=1e-9)
    assert b["nvme_prefetch"] == 0 and b["nvme_prefetch_wasted"] == 0
    dense = c.global_bytes + shape.n_moe_layers * (c.layer_bytes + c.shared_bytes)
    assert math.isclose(b["dram"], r.loads_per_token * r.slab_bytes + dense, rel_tol=1e-9)
    assert math.isclose(sum(r.time_per_token_ms.values()), r.ms_per_token, rel_tol=1e-9)
    assert math.isclose(1e3 / r.ms_per_token, r.tok_s, rel_tol=1e-9)
    assert r.bottleneck == max(r.time_per_token_ms, key=r.time_per_token_ms.get)


def test_timing_parts_sum_to_step_time_everywhere():
    for kw in (dict(prefetch=Prefetch(0.7, 2)), dict(spec=Spec(3, 0.6)), dict(gpu="dense+experts", dense_bits=4.25),
               dict(lossy=Lossy(cold_bits=2.5, cold_frac=0.5)), dict()):
        r = small(cache_gb=0.4, **kw)
        tm = evaluate(r._counts, r._sched, r._costs, r._hw, r.calibration, r.settings["gpu"])
        assert np.allclose(tm.parts.sum(axis=1), tm.step_s, rtol=1e-9), kw
        assert (tm.parts >= -1e-12).all()


def test_all_hits_is_dram_bound_and_matches_hand_calculation():
    hw = get_hardware("this-pc", nvme_gbs=1e-3)              # storage almost unusable
    r = simulate(SMALL, hw, tokens=300, zipf=1.0, cache_gb=10_000.0, warmup=299)
    # last token: every expert resident (after warm-up), so time = fixed + backbone + experts from DRAM
    c, cal = r._costs, r.calibration
    assert r.miss_rate == 0.0
    bw = hw.dram_gbs * 1e9
    want = (cal.overhead_ms * 1e-3 + cal.layer_overhead_us * 1e-6 * c.n_layers
            + (c.global_bytes + c.n_moe_layers * (c.layer_bytes + c.shared_bytes)) / bw
            + r.loads_per_token * c.slab / bw)
    assert math.isclose(r.ms_per_token * 1e-3, want, rel_tol=1e-9)
    assert r.bottleneck == "dram"


def test_more_storage_bandwidth_never_hurts():
    one = small(cache_gb=0.3)
    two = simulate(SMALL, get_hardware("this-pc", nvme_count=2), tokens=600, zipf=1.2, reuse=0.0, seed=3,
                   cache_gb=0.3)
    assert two.hit_rate == one.hit_rate
    assert two.tok_s > one.tok_s
    assert one.bottleneck == "nvme"


def test_prefetch_accounting():
    base = small(cache_gb=0.3)
    pf = small(cache_gb=0.3, prefetch=Prefetch(recall=1.0, extra=0))
    assert pf.prefetch_rate > 0.5 and pf.miss_rate < base.miss_rate
    assert math.isclose(pf.hit_rate + pf.prefetch_rate + pf.miss_rate, 1.0)
    b = pf.bytes_per_token
    assert math.isclose(b["nvme"], b["nvme_demand"] + b["nvme_prefetch"])
    assert b["nvme_prefetch_wasted"] == 0.0                     # recall 1, no extras: no wrong guesses
    noisy = small(cache_gb=0.3, prefetch=Prefetch(recall=0.5, extra=3))
    assert noisy.bytes_per_token["nvme_prefetch_wasted"] > 0
    assert 0.0 <= noisy.late_prefetch_rate <= 1.0
    # with storage effectively infinite, prefetches are never late
    fast = simulate(SMALL, get_hardware("this-pc", nvme_gbs=1e6), tokens=600, zipf=1.2, reuse=0.0, seed=3,
                    cache_gb=0.3, prefetch=Prefetch(1.0, 0))
    assert fast.late_prefetch_rate == 0.0


# ---- prefetch byte accounting, hand-computed (H04: council round 1, T04 finding 1) -------------------

def test_prefetch_counters_hand_traced():
    """Every counter of a 4-token, 3-layer run, traced by hand. E = 2, k = 1, Prefetch(1.0, 1): each layer
    predicts the next layer's true expert, then the only other one (deterministic). LRU, 3 slots; cold keys
    2 and 5. Keys: layer 0 = {0, 1}, layer 1 = {2, 3}, layer 2 = {4, 5}.
      t0: miss 0; prefetch 2, 3. L1 uses 2 (3 wasted); prefetch 4 evicts 0, 5 is dropped (2 is protected).
          L2 uses 4. Step end releases 3.                                    residents {3, 2, 4}
      t1: miss 1 evicts 3; prefetch 3 evicts 4 (2 is the true one, already resident). L1 hits 2 (3 wasted);
          prefetch 5 evicts 1, 4 is dropped. L2 uses 5.                       residents {3, 2, 5}
      t2: miss 0 evicts 3; prefetch 3 evicts 5. L1 uses 3; prefetch 5 evicts 2, 4 evicts 0. L2 uses 5 (4
          wasted).                                                           residents {3, 4, 5}
      t3: miss 0 evicts 3; prefetch 2 evicts 4, 3 evicts 5. L1 uses 2 (3 wasted); prefetch 4 evicts 0, 5 is
          dropped. L2 uses 4.
    Admissions: 4 misses + 11 prefetches = 15 into 3 slots, so 12 evictions."""
    ids = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0]], dtype=np.uint16).reshape(4, 3, 1)
    sched = build_schedule(ids, 2)
    assert sched.acc.tolist() == [0, 2, 4, 1, 2, 5, 0, 3, 5, 0, 2, 4]
    c = run_cache(sched, "lru", 3, prefetch=Prefetch(1.0, 1), cold_keys=[2, 5])
    zero = [0] * 12
    want = dict(hit=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], hit_cold=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                pfhit=[0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1], pfhit_cold=[0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                miss=[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], miss_cold=zero, bypass=zero, vram=zero, skip=zero,
                pf_issued=[2, 1, 0, 1, 1, 0, 1, 2, 0, 2, 1, 0], pf_issued_cold=[1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
                pf_wasted=[1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0], pf_wasted_cold=zero)
    for name, w in want.items():
        assert getattr(c, name).tolist() == w, name
    assert c.evictions == 12 and c.slots == 3 and c.n_pinned == 0


def _tiny_prefetch_run(hw, **kw):
    """2 MoE layers x 2 experts, top-1, 3 tokens: t0 = (0, 2), t1 = (1, 2), t2 = (0, 3). Every expert fits
    (slots = 4), Prefetch(1.0, 1) predicts both layer-1 experts; the array profile makes keys 1 and 2 (layer 0
    expert 1, layer 1 expert 0) the cold half. By hand:
      t0: L0 misses 0 (hot) and prefetches 2 (cold, needed) and 3 (hot, wrong); L1 uses the prefetched 2.
      t1: L0 misses 1 (cold); nothing left to prefetch; L1 hits 2 (cold).
      t2: L0 hits 0, L1 hits 3 (both hot)."""
    shape = dataclasses.replace(presets.get(SMALL), name="tiny-pf", n_layers=2, n_experts=2, top_k=1)
    tr = Trace(np.array([[0, 0], [1, 0], [0, 1]], dtype=np.uint16).reshape(3, 2, 1), 2)
    base = dict(trace=tr, policy="lru", cache_gb=1.0, io_threads=1, prefetch=Prefetch(1.0, 1), warmup=0,
                lossy=Lossy(cold_bits=2.25, cold_frac=0.5), profile=np.array([3.0, 0.0, 0.0, 2.0]), context=0,
                calib=Calibration(overhead_ms=0.5, layer_overhead_us=10.0))
    return simulate(shape, hw, **{**base, **kw})


def test_prefetch_bytes_per_token_hand_computed():
    """bytes_per_token, rates and step times of _tiny_prefetch_run, with the prefetch window too short for all
    prefetched bytes (0 < f < 1, so the needed cold slab is partly promoted to a demand read) and with fast
    storage (f = 1). nvme_demand + nvme_prefetch - nvme_prefetch_wasted equals the bytes of misses + prefetch
    hits; dram is every expert computed plus the backbone; all per accepted token."""
    hw = get_hardware("this-pc", nvme_latency_us=80.0)
    r = _tiny_prefetch_run(hw)
    c, cal = r._costs, r.calibration
    S, Sc = float(c.slab), float(c.slab_cold)
    assert (c.slab, c.slab_cold) == (3342336, 1769472)            # FORMAT sec. 5 at 4.25 and 2.25 bpw
    assert r.slots == 4 and r.tokens == 3 and r.steps == 3
    cnt = r._counts
    assert cnt.miss.tolist() == [1, 0, 1, 0, 0, 0] and cnt.miss_cold.tolist() == [0, 0, 1, 0, 0, 0]
    assert cnt.pfhit.tolist() == [0, 1, 0, 0, 0, 0] and cnt.pfhit_cold.tolist() == [0, 1, 0, 0, 0, 0]
    assert cnt.hit.tolist() == [0, 0, 0, 1, 1, 1] and cnt.hit_cold.tolist() == [0, 0, 0, 1, 0, 0]
    assert cnt.pf_issued.tolist() == [2, 0, 0, 0, 0, 0] and cnt.pf_issued_cold.tolist() == [1, 0, 0, 0, 0, 0]
    assert cnt.pf_wasted.tolist() == [1, 0, 0, 0, 0, 0] and cnt.pf_wasted_cold.tolist() == [0] * 6

    bw_d, bw_io, ops, lat = 60e9, 6.5e9, 5e12, 80e-6
    ch = max(S / bw_d, 2.0 * c.expert_params / ops)
    cc = max(Sc / bw_d, 2.0 * c.expert_params / ops)
    A = max(c.layer_bytes / bw_d, 2.0 * c.layer_params / ops)
    G = max(c.global_bytes / bw_d, 2.0 * c.global_params / ops)
    assert (cal.overhead_ms, cal.layer_overhead_us) == (0.5, 10.0)
    F = 0.5e-3 + 10e-6 * 2
    # t0, L0: one hot miss, nothing computed before it lands; storage is then idle for ch until L1's MoE
    moe00 = lat + S / bw_io + ch
    f = (ch + A) * bw_io / (Sc + S)                                   # share of the 2 prefetched slabs in time
    assert 0.0 < f < 1.0
    # t0, L1: a share f of the needed (cold) prefetch arrived in time; the rest is promoted to a demand read
    m = 1 - f                                                         # late share, a demand read
    moe01 = max(f * cc + m * cc, lat + m * Sc / bw_io + cc, lat + Sc / bw_io + m * cc)   # sec. 3.4, W0 = f * cc
    steps = [F + G + 2 * A + moe00 + moe01,
             F + G + 2 * A + (lat + Sc / bw_io + cc) + cc,
             F + G + 2 * A + ch + ch]
    assert math.isclose(r.ms_per_token, sum(steps) / 3 * 1e3, rel_tol=1e-12)
    nvme_wait = (lat + S / bw_io) + (moe01 - cc) + (lat + Sc / bw_io)
    assert math.isclose(r.time_per_token_ms["nvme"], nvme_wait / 3 * 1e3, rel_tol=1e-12)
    b = r.bytes_per_token
    backbone = c.global_bytes + 2 * (c.layer_bytes + c.shared_bytes)
    assert math.isclose(b["nvme_demand"], (S + (1 - f) * Sc + Sc) / 3, rel_tol=1e-12)
    assert math.isclose(b["nvme_prefetch"], f * (Sc + S) / 3, rel_tol=1e-12)
    assert math.isclose(b["nvme_prefetch_wasted"], f * S / 3, rel_tol=1e-12)
    assert math.isclose(b["nvme"], b["nvme_demand"] + b["nvme_prefetch"], rel_tol=1e-15)
    assert math.isclose(b["nvme_demand"] + b["nvme_prefetch"] - b["nvme_prefetch_wasted"], (S + 2 * Sc) / 3,
                        rel_tol=1e-12)                                # misses (S + Sc) + prefetch hit (Sc)
    assert math.isclose(b["dram"], (3 * S + 3 * Sc) / 3 + backbone, rel_tol=1e-12)
    assert b["vram"] == 0.0
    assert (r.hit_rate, r.prefetch_rate, r.miss_rate, r.skip_rate, r.vram_rate, r.bypass_rate) == \
        (3 / 6, 1 / 6, 2 / 6, 0.0, 0.0, 0.0)
    assert math.isclose(r.late_prefetch_rate, 1 - f, rel_tol=1e-12)
    assert (r.loads_per_token, r.tokens_per_step) == (2.0, 1.0)

    fast = _tiny_prefetch_run(get_hardware("this-pc", nvme_gbs=1e4))      # every prefetch in time
    b = fast.bytes_per_token
    assert fast._counts.pfhit.tolist() == cnt.pfhit.tolist()
    assert (b["nvme_demand"], b["nvme_prefetch"], b["nvme_prefetch_wasted"]) == ((S + Sc) / 3, (Sc + S) / 3, S / 3)
    assert math.isclose(b["dram"], r.bytes_per_token["dram"], rel_tol=1e-12) and fast.late_prefetch_rate == 0.0
    # warm-up: only t1 and t2 count (2 tokens); t0's prefetch is outside the window
    w = _tiny_prefetch_run(hw, warmup=1)
    b = w.bytes_per_token
    assert (w.tokens, b["nvme_demand"], b["nvme_prefetch"], b["nvme_prefetch_wasted"]) == (2, Sc / 2, 0.0, 0.0)
    assert math.isclose(b["dram"], (2 * S + 2 * Sc) / 2 + backbone, rel_tol=1e-12)
    assert (w.hit_rate, w.prefetch_rate, w.miss_rate) == (3 / 4, 0.0, 1 / 4)


@pytest.mark.parametrize("kw", [dict(prefetch=Prefetch(0.5, 3), lossy=Lossy(cold_bits=2.25, cold_frac=0.6)),
                                dict(prefetch=Prefetch(0.8, 1), spec=Spec(3, 0.6),
                                     lossy=Lossy(skip_miss_rank=6, cold_bits=3.25, cold_frac=0.3)),
                                dict(prefetch=Prefetch(0.7, 2), gpu="dense+experts",
                                     hardware=get_hardware("this-pc", vram_gib=3.0))])
def test_prefetch_byte_identities_on_realistic_runs(kw):
    """The per-token byte identities on olmoe runs with prefetch, cold slabs, skips, speculative windows, a
    VRAM tier and a warm-up, from the run's own counters (measured steps only):
      nvme_demand + nvme_prefetch - nvme_prefetch_wasted = (misses + prefetch hits) x slab / tokens;
      nvme_prefetch = issued prefetch bytes / tokens when storage is fast enough for all of them;
      dram = (loads - vram - skips) x slab + backbone per step, / tokens; vram = VRAM-tier loads x slab / tokens."""
    kw = dict(kw)
    hw = kw.pop("hardware", get_hardware("this-pc"))
    r = simulate(SMALL, hw, tokens=600, zipf=1.2, reuse=0.2, seed=3, cache_gb=0.3, **kw)
    c, cnt, sched = r._costs, r._counts, r._sched
    L = sched.n_moe_layers
    g = np.repeat(sched.step_start >= r.warmup_tokens, L)
    S, Sc = float(c.slab), float(c.slab_cold)

    def nbytes(n, n_cold):
        return float((n[g] - n_cold[g]).sum()) * S + float(n_cold[g].sum()) * Sc
    tok = r.tokens
    assert tok > 400 and r.prefetch_rate > 0.05 and cnt.pf_wasted[g].sum() > 0
    b = r.bytes_per_token
    used = nbytes(cnt.miss, cnt.miss_cold) + nbytes(cnt.pfhit, cnt.pfhit_cold)
    assert math.isclose(b["nvme_demand"] + b["nvme_prefetch"] - b["nvme_prefetch_wasted"], used / tok, rel_tol=1e-9)
    assert math.isclose(b["nvme"], b["nvme_demand"] + b["nvme_prefetch"], rel_tol=1e-12)
    bb = c.global_bytes + L * (c.layer_bytes + c.shared_bytes)
    computed = nbytes(cnt.hit, cnt.hit_cold) + nbytes(cnt.pfhit, cnt.pfhit_cold) + nbytes(cnt.miss, cnt.miss_cold)
    on_gpu = kw.get("gpu", "off") != "off"
    assert math.isclose(b["dram"], (computed + (0.0 if on_gpu else bb * r.steps)) / tok, rel_tol=1e-9)
    assert math.isclose(b["vram"], (float(cnt.vram[g].sum()) * S + (bb * r.steps if on_gpu else 0.0)) / tok,
                        rel_tol=1e-9)
    skips = kw.get("lossy") is not None and kw["lossy"].skip_miss_rank is not None
    assert (cnt.vram[g].sum() > 0) == on_gpu and (cnt.skip[g].sum() > 0) == skips
    loads = int((cnt.hit + cnt.pfhit + cnt.miss + cnt.vram + cnt.skip)[g].sum())
    assert math.isclose(r.loads_per_token, loads / tok, rel_tol=1e-12)
    # fast storage: every prefetch arrives, so prefetch bytes are exactly the issued ones
    fast = simulate(SMALL, get_hardware(hw, nvme_gbs=1e5), tokens=600, zipf=1.2, reuse=0.2, seed=3, cache_gb=0.3,
                    **kw)
    fc = fast._counts
    assert fc.pf_issued.tolist() == cnt.pf_issued.tolist() and fast.late_prefetch_rate == 0.0
    fb = fast.bytes_per_token
    assert math.isclose(fb["nvme_prefetch"], nbytes(fc.pf_issued, fc.pf_issued_cold) / tok, rel_tol=1e-9)
    assert math.isclose(fb["nvme_prefetch_wasted"], nbytes(fc.pf_wasted, fc.pf_wasted_cold) / tok, rel_tol=1e-9)
    assert math.isclose(fb["nvme_demand"], nbytes(fc.miss, fc.miss_cold) / tok, rel_tol=1e-9)
    assert r.prefetch_rate == fast.prefetch_rate


def test_speculative_union_amortisation():
    tr = synthetic_for(presets.get(SMALL), 600, zipf=0.0, reuse=1.0, seed=4)   # every token reuses its experts
    plain = simulate(SMALL, trace=tr, cache_gb=0.0)
    spec = simulate(SMALL, trace=tr, cache_gb=0.0, spec=Spec(4, 0.9))
    assert plain.loads_per_token == 128.0
    assert spec.tokens_per_step > 3.0
    assert math.isclose(spec.loads_per_token * spec.tokens_per_step, 128.0, rel_tol=0.02)
    assert spec.tok_s > 2 * plain.tok_s
    # independent tokens: verification loads the union, so reads per accepted token go UP at low alpha
    indep = synthetic_for(presets.get(SMALL), 600, zipf=0.0, reuse=0.0, seed=4)
    lo = simulate(SMALL, trace=indep, cache_gb=0.0, spec=Spec(4, 0.3))
    assert lo.loads_per_token > 128.0


def test_speculative_acceptance_statistics():
    import inspect
    assert inspect.signature(simulate).parameters["tokens"].default == 2000      # documented default trace
    assert small(spec=Spec(0, 0.5)).tokens_per_step == 1.0                     # k = 0 means off
    assert Spec(4, 0.0).expected_tokens() == 1.0
    assert Spec(4, 1.0).expected_tokens() == 5.0
    assert math.isclose(Spec(4, 0.6).expected_tokens(), (1 - 0.6 ** 5) / 0.4)
    ids = np.zeros((20000, 1, 1), dtype=np.uint16)
    s = build_schedule(ids, 1, spec_k=4, spec_alpha=0.6, seed=1)
    assert abs(s.step_nacc.mean() - Spec(4, 0.6).expected_tokens()) < 0.05
    assert s.step_nacc.max() <= 5 and s.step_nacc.min() >= 1
    assert (np.diff(s.step_start) == s.step_nacc[:-1]).all()
    assert int(s.step_nacc.sum()) == 20000


def test_determinism():
    kw = dict(cache_gb=0.4, prefetch=Prefetch(0.7, 1), spec=Spec(3, 0.6), policy="lfu", lfu_samples=8)
    a, b = small(**kw), small(**kw)
    assert a.tok_s == b.tok_s and a.hit_rate == b.hit_rate and a.bytes_per_token == b.bytes_per_token
    c = small(**{**kw, "seed": 4})
    assert c.tok_s != a.tok_s


def test_lossy_knobs_are_labelled_and_reduce_traffic():
    base = small(cache_gb=0.3)
    assert base.lossy == []
    topk = small(cache_gb=0.3, lossy=Lossy(topk=6))
    assert topk.loads_per_token == 16 * 6 and any("LOSSY" in s for s in topk.lossy)
    skip = small(cache_gb=0.3, lossy=Lossy(skip_miss_rank=6))
    assert skip.skip_rate > 0 and skip.bytes_per_token["nvme"] < base.bytes_per_token["nvme"]
    assert math.isclose(skip.hit_rate + skip.prefetch_rate + skip.miss_rate + skip.skip_rate, 1.0)
    assert skip.loads_per_token == 16 * 8
    cold = small(cache_gb=0.3, lossy=Lossy(cold_bits=2.25, cold_frac=0.75))
    assert cold.bytes_per_token["nvme"] < base.bytes_per_token["nvme"] and cold.hit_rate == base.hit_rate
    with pytest.raises(ValueError):
        small(lossy=Lossy(topk=0))


def test_gpu_modes_and_feasibility():
    k2 = presets.get("kimi-k2")
    r = simulate("kimi-k2", "this-pc", tokens=300, gpu="dense+experts", dense_bits=4.25)
    assert r.feasible and r.vram_rate > 0 and r.bytes_per_token["vram"] > 0
    assert r.feasibility.backbone_location == "vram"
    q8 = simulate("kimi-k2", "this-pc", tokens=300, gpu="dense")
    assert not q8.feasible and any("VRAM" in w for w in q8.warnings)
    with pytest.raises(ValueError):
        simulate("kimi-k2", "laptop-16gb", tokens=300, gpu="dense")
    mac = simulate("kimi-k2", "mac-studio-192gb", tokens=300, gpu="dense+experts")
    cpu = simulate("kimi-k2", "mac-studio-192gb", tokens=300)
    assert mac.feasibility.backbone_location == "unified" and mac.vram_rate == 0.0
    assert mac.hit_rate == cpu.hit_rate and mac.tok_s > cpu.tok_s
    assert k2.n_moe_layers == 60


def _resident_bytes_from_presets(shape, dense_bits=8.25, embed_bits=8.25):
    """FORMAT.md sec. 4.1 footprint from Shape.dense_params(): the embedding table at embed_bits, routers and
    norms F32, every other resident matrix (LM head, attention, shared/dense FFN, latent projections) at
    dense_bits."""
    D, routers = shape.d_model, shape.n_moe_layers * shape.n_experts * shape.d_model
    matrices = shape.dense_params() - shape.vocab * D - routers
    return (shape.vocab * D * embed_bits / 8 + matrices * dense_bits / 8 + routers * 4
            + shape.n_layers * 2 * D * 4 + D * 4)


def test_backbone_accounting_matches_presets_without_double_counting():
    """Shape.dense_params() includes the latent-MoE projections since commit 8ea46c0; the simulator counts them
    once. Its resident footprint and parameter totals must equal the presets' for every model."""
    for name, shape in presets.PRESETS.items():
        c = model_costs(shape)
        assert c.resident_params == shape.dense_params(), name
        assert c.total_params == shape.params_total(), name
        assert math.isclose(c.resident_bytes, _resident_bytes_from_presets(shape), rel_tol=1e-12), name
        # what one step reads (KV aside) is the whole backbone except the embedding table
        per_step = c.global_bytes + c.n_moe_layers * (c.layer_bytes + c.shared_bytes) \
            - c.n_layers * c.kv_bytes_per_layer
        assert math.isclose(per_step, c.resident_bytes - c.embed_bytes, rel_tol=1e-12), name
    k3 = presets.get("kimi-k3")
    c3 = model_costs(k3, context=0)
    latent = 2 * 7168 * 3584
    assert c3.total_params == 2_752_693_600_256                     # ~2.75e12; 2.748e12 without the projections
    assert c3.resident_params == 29_952_770_048 and k3.n_moe_layers * latent == 4_726_980_608
    assert c3.layer_params == k3.attn_params() + latent + 896 * 7168
    assert any("latent MoE: experts in 3584-d, 51.4M projection params" in n for n in c3.notes)
    # an explicit expert_d equal to d_model still has projections in Shape.dense_params(), so here too
    same = dataclasses.replace(presets.get(SMALL), name="explicit-d", expert_d=2048)
    want = presets.get(SMALL).dense_params() + 16 * 2 * 2048 ** 2
    assert model_costs(same).resident_params == same.dense_params() == want


def test_feasibility_exact_kimi_k2_on_this_pc():
    k2, hw, GiB = presets.get("kimi-k2"), HARDWARE["this-pc"], 2.0 ** 30
    c = model_costs(k2)
    f = feasibility(c, hw)
    assert (f.model, f.hardware, f.gpu_mode, f.fits, f.reasons, f.backbone_location) == \
        ("kimi-k2", "this-pc", "off", True, [], "ram")
    resident = _resident_bytes_from_presets(k2)
    kv = (512 + 64) * 4.0 * 4096 * 61                                  # MLA latent + rope, f32, max_seq, all layers
    slab = 23_396_352                                                  # FORMAT sec. 5, 7168 x 2048 at Q4
    assert c.slab == slab and kv == 575_668_224
    assert math.isclose(f.backbone_gib, resident / GiB, rel_tol=1e-12) and round(f.backbone_gib, 4) == 11.7172
    assert f.kv_gib == kv / GiB
    free = 61.6 - 6.0 - 0.5 - resident / GiB - kv / GiB
    assert math.isclose(f.ram_free_for_cache_gib, free, rel_tol=1e-12) and round(free, 4) == 42.8466
    assert f.min_cache_gib == (2 * 8 + 8 + 2) * slab / GiB                # 26 slots, 0.5665 GiB
    assert math.isclose(f.min_ram_gib, 6.0 + 0.5 + (resident + kv + 26 * slab) / GiB, rel_tol=1e-12)
    assert round(f.min_ram_gib, 4) == 19.3199
    assert f.recommended_cache_gib == f.ram_free_for_cache_gib
    assert f.expert_total_gib == 502.03125 == 60 * 384 * slab / GiB
    assert f.vram_expert_slots == 0 and f.file_fits_drive
    assert math.isclose(f.file_gb, (resident + 60 * 384 * slab) / 1e9, rel_tol=1e-12)
    want = {"Q3 (3.25 bpw, roadmap R03)": 424.79748096, "Q4 (4.25 bpw)": 551.63323392,
            "Q8 (8.25 bpw)": 1058.97624576, "BF16": 2041.9533312}
    assert list(f.disk_gb_by_format) == list(want)
    for label, gb in want.items():
        assert math.isclose(f.disk_gb_by_format[label], gb, rel_tol=1e-12), label
    assert f.disk_gb_by_format["Q4 (4.25 bpw)"] == f.file_gb
    for label, bits in (("Q3 (3.25 bpw, roadmap R03)", 3.25), ("Q8 (8.25 bpw)", 8.25), ("BF16", 16.0)):
        assert math.isclose(f.disk_gb_by_format[label], (resident + 23040 * slab_bytes(7168, 2048, bits)) / 1e9,
                            rel_tol=1e-12)
    assert (f.params_total, f.params_resident) == (k2.params_total(), k2.dense_params()) == \
        (1_026_407_202_816, 11_721_179_136)
    d = f.to_dict()
    assert d == dataclasses.asdict(f) and d["disk_gb_by_format"]["BF16"] == f.disk_gb_by_format["BF16"]


def test_feasibility_exact_kimi_k3_latent_experts():
    """Kimi-K3's experts are sized at expert_d = 3584; disk sizes are exact and the file does not fit 1 TB."""
    k3 = presets.get("kimi-k3")
    c = model_costs(k3)
    f = feasibility(c, HARDWARE["laptop-16gb"])
    resident = _resident_bytes_from_presets(k3)
    assert c.slab == slab_bytes(3584, 3072, 4.25) == 17_547_264
    want = {"Q3 (3.25 bpw, roadmap R03)": 1138.761771008, "Q4 (4.25 bpw)": 1479.104374784,
            "Q8 (8.25 bpw)": 2840.474789888, "BF16": 5478.129969152}
    for label, gb in want.items():
        assert math.isclose(f.disk_gb_by_format[label], gb, rel_tol=1e-12), label
    assert math.isclose(f.disk_gb_by_format["Q4 (4.25 bpw)"], (resident + 92 * 896 * 17_547_264) / 1e9, rel_tol=1e-12)
    assert f.file_gb == f.disk_gb_by_format["Q4 (4.25 bpw)"] and not f.file_fits_drive and not f.fits
    assert f.reasons[-1] == "model file 1479 GB exceeds one drive (1000 GB); each mirror is a full copy"
    assert f.reasons[0].startswith("only -19.7 GiB RAM left for experts") and len(f.reasons) == 2
    assert f.recommended_cache_gib == f.min_cache_gib == (2 * 16 + 8 + 2) * 17_547_264 / 2 ** 30
    assert f.params_total == 2_752_693_600_256
    big = feasibility(c, get_hardware("laptop-16gb", nvme_capacity_gb=f.file_gb))   # exactly the file: it fits
    assert big.file_fits_drive and len(big.reasons) == 1


def test_feasibility_gpu_modes_exact():
    ol, GiB = presets.get(SMALL), 2.0 ** 30
    c = model_costs(ol)
    bb, emb, kv = c.resident_bytes / GiB, c.embed_bytes / GiB, c.kv_resident_bytes / GiB
    # no GPU: a GPU mode is a reason on its own (olmoe otherwise fits the laptop)
    lap = HARDWARE["laptop-16gb"]
    assert feasibility(c, lap).fits
    for mode in ("dense", "dense+experts"):
        f = feasibility(c, lap, gpu_mode=mode)
        assert not f.fits and f.reasons == [f"gpu_mode={mode} but laptop-16gb has no GPU"]
        assert f.backbone_location == "ram" and f.vram_expert_slots == 0
        assert f.ram_free_for_cache_gib == feasibility(c, lap).ram_free_for_cache_gib
    # discrete GPU: the backbone minus the embedding table (and the KV cache) moves to VRAM
    pc = HARDWARE["this-pc"]
    dense, both = feasibility(c, pc, gpu_mode="dense"), feasibility(c, pc, gpu_mode="dense+experts")
    for f in (dense, both):
        assert f.fits and f.backbone_location == "vram"
        assert f.ram_free_for_cache_gib == 61.6 - (6.0 + 0.5 + emb)
        assert f.min_ram_gib == 6.0 + 0.5 + emb + f.min_cache_gib
    assert dense.vram_expert_slots == 0                                # only dense+experts has a VRAM tier
    assert both.vram_expert_slots == int((12.0 - (bb - emb + kv + 1.0)) * GiB // c.slab) == 3095
    assert feasibility(c, pc).ram_free_for_cache_gib == 61.6 - (6.0 + 0.5 + (bb + kv))
    assert feasibility(c, pc).recommended_cache_gib == c.expert_total_bytes / GiB == 3.1875   # all experts fit
    # unified memory: the backbone stays in (shared) RAM, no VRAM tier
    mac = HARDWARE["mac-studio-192gb"]
    for mode in ("dense", "dense+experts"):
        f = feasibility(c, mac, gpu_mode=mode)
        assert f.fits and f.backbone_location == "unified" and f.vram_expert_slots == 0
        assert f.ram_free_for_cache_gib == 192.0 - (8.0 + 0.5 + (bb + kv)) == feasibility(c, mac).ram_free_for_cache_gib
    # VRAM boundary: exactly enough for backbone + KV + headroom fits (with no tier); a byte less does not
    need = bb - emb + kv + 1.0
    edge = feasibility(c, get_hardware("this-pc", vram_gib=need), gpu_mode="dense+experts")
    assert edge.fits and edge.vram_expert_slots == 0
    short = feasibility(c, get_hardware("this-pc", vram_gib=need - 2.0 ** -30), gpu_mode="dense")
    assert short.reasons == [f"backbone {bb - emb:.1f} GiB + KV {kv:.1f} GiB + 1.0 GiB headroom does not fit "
                             f"{need - 2.0 ** -30:.0f} GiB VRAM"]
    k2 = feasibility(model_costs(presets.get("kimi-k2")), pc, gpu_mode="dense")
    assert k2.reasons == ["backbone 10.6 GiB + KV 0.5 GiB + 1.0 GiB headroom does not fit 12 GiB VRAM"]


def test_feasibility_ram_boundary_and_hint():
    """Synthetic costs in powers of two, so the boundary is exact: RAM left == engine minimum fits."""
    GiB = 2.0 ** 30
    c = dataclasses.replace(model_costs(presets.get(SMALL)), top_k=1, resident_bytes=2 * GiB, embed_bytes=0.5 * GiB,
                            kv_resident_bytes=0.25 * GiB, slab=int(GiB / 8), expert_total_bytes=64 * GiB)
    min_cache = (2 * 1 + 1 + 2) / 8                                    # 5 slots of 1/8 GiB at io_threads=1
    hw = get_hardware("laptop-16gb", ram_gib=1.0 + 0.5 + 2.25 + min_cache, os_reserve_gib=1.0)
    f = feasibility(c, hw, io_threads=1)
    assert f.ram_free_for_cache_gib == min_cache == f.min_cache_gib and f.fits
    assert f.min_ram_gib == hw.ram_gib and f.recommended_cache_gib == min_cache
    tight = feasibility(c, get_hardware(hw, ram_gib=hw.ram_gib - 1 / 64), io_threads=1)
    assert not tight.fits and tight.recommended_cache_gib == min_cache
    assert tight.reasons == ["only 0.6 GiB RAM left for experts after backbone/KV/OS reserve (a Q4 backbone, "
                             "--dense-bits 4.25, roughly halves it; LOSSY vs the Q8 default); engine minimum cache "
                             "is 0.62 GiB"]
    q4 = feasibility(dataclasses.replace(c, dense_bits=4.25), get_hardware(hw, ram_gib=hw.ram_gib - 1 / 64),
                     io_threads=1)
    assert q4.reasons == ["only 0.6 GiB RAM left for experts after backbone/KV/OS reserve; engine minimum cache is "
                          "0.62 GiB"]
    # io_threads raise the minimum like hx_store_open, up to the 64-thread clamp
    assert feasibility(c, hw, io_threads=64).min_cache_gib == feasibility(c, hw, io_threads=500).min_cache_gib \
        == (2 + 64 + 2) / 8
    # the file boundary: a drive exactly the size of the file holds it
    file_gb = (2 * GiB + 64 * GiB) / 1e9
    assert feasibility(c, get_hardware(hw, nvme_capacity_gb=file_gb), io_threads=1).file_fits_drive
    assert not feasibility(c, get_hardware(hw, nvme_capacity_gb=file_gb * (1 - 1e-12)), io_threads=1).file_fits_drive


def test_cli_feasibility_output_and_json(capsys):
    assert main(["--model", SMALL, "--hw", "laptop-16gb", "--gpu", "dense", "--feasibility"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "olmoe-1b-7b on laptop-16gb:"
    assert "fits                       NO - gpu_mode=dense but laptop-16gb has no GPU" in out
    assert main(["--model", "kimi-k2", "--feasibility", "--json"]) == 0
    js = json.loads(capsys.readouterr().out)["feasibility"]
    want = feasibility(model_costs(presets.get("kimi-k2")), HARDWARE["this-pc"]).to_dict()
    assert js == json.loads(json.dumps(want))
    assert js["disk_gb_by_format"]["Q4 (4.25 bpw)"] == 551.63323392 and js["params_total"] == 1_026_407_202_816
    # every encoding flag reaches the feasibility path
    assert main(["--model", "kimi-k2", "--feasibility", "--json", "--dense-bits", "4.25", "--embed-bits", "4.25",
                 "--max-seq", "1024", "--kv-bytes", "2", "--expert-bits", "3.25", "--io-threads", "4"]) == 0
    js = json.loads(capsys.readouterr().out)["feasibility"]
    c = model_costs(presets.get("kimi-k2"), dense_bits=4.25, embed_bits=4.25, max_seq=1024, kv_elem_bytes=2.0,
                    expert_bits=3.25)
    assert js == json.loads(json.dumps(feasibility(c, HARDWARE["this-pc"], io_threads=4).to_dict()))
    assert main(["--model", "kimi-k2", "--feasibility"]) == 0
    assert "parameters                 1026 B, of which 11.72 B resident" in capsys.readouterr().out


def test_slab_bytes_follow_format():
    # FORMAT.md sec. 5 by hand for Kimi-K2 at Q4: D=7168, F=2048
    gate = 2048 * (34 * 7168 // 64)
    up_off = (gate + 63) // 64 * 64
    down_off = (up_off + gate + 63) // 64 * 64
    end = down_off + 7168 * (34 * 2048 // 64)
    assert slab_bytes(7168, 2048, 4.25) == (end + 4095) // 4096 * 4096
    assert slab_bytes(7168, 2048, 4.25) == presets.get("kimi-k2").expert_bytes(4.25)
    assert slab_bytes(64, 64, 8.25) == 16384                       # 3 x 64 rows x 66 B = 12672 -> 4096-aligned
    k3 = model_costs(presets.get("kimi-k3"))
    assert k3.slab == slab_bytes(3584, 3072, 4.25)                 # latent experts use expert_d
    assert any("latent" in n for n in k3.notes)


def test_calibrate_recovers_known_scalars():
    pts = [small(cache_gb=g, **kw) for g, kw in ((0.2, {}), (0.6, {}), (1.5, {}),
                                                 (0.4, {"spec": Spec(3, 0.7)}))]
    true = Calibration(dram_eff=0.7, io_eff=0.55, overhead_ms=4.0)
    measured = [(r, retime(r, true).tok_s) for r in pts]
    cal, rep = calibrate(measured)
    assert rep["rms_log_error"] < 1e-3
    for r, m in measured:
        assert math.isclose(retime(r, cal).tok_s, m, rel_tol=2e-3)
    assert math.isclose(cal.io_eff, 0.55, rel_tol=0.05) and math.isclose(cal.dram_eff, 0.7, rel_tol=0.05)
    assert math.isclose(cal.overhead_ms, 4.0, rel_tol=0.1)
    with pytest.raises(ValueError):
        calibrate([])


def test_usage_profile_round_trip_and_pinning(tmp_path):
    shape = presets.get(SMALL)
    heat = np.random.default_rng(0).random((shape.n_layers, shape.n_experts)).astype(np.float32)
    p = save_usage(tmp_path / "m.usage", heat, tokens_observed=123)
    back, tok = load_usage(p)
    assert tok == 123 and np.array_equal(back, heat.astype(np.float64))
    r = small(policy="pinned", cache_gb=0.5, profile=str(p))
    assert r.n_pinned > 0
    tr = synthetic_for(shape, 600, zipf=1.2, reuse=0.0, seed=3)
    freq = tr.frequencies()
    hot = save_usage(tmp_path / "hot.usage", freq.astype(np.float32))
    oracle = small(policy="pinned", cache_gb=0.5, profile=str(hot))
    assert oracle.hit_rate > r.hit_rate


def test_trace_geometry_mismatch_is_an_error(tmp_path):
    tr = synthetic(10, 3, 8, 2, seed=0)
    with pytest.raises(ValueError):
        simulate(SMALL, trace=tr)
    with pytest.raises(ValueError):
        simulate(SMALL, policy="fifo")


def test_hardware_profiles_and_overrides():
    assert set(HARDWARE) >= {"this-pc", "laptop-16gb", "desktop-32gb", "workstation-128gb-2nvme", "mac-studio-192gb"}
    pc = HARDWARE["this-pc"]
    assert (pc.dram_gbs, pc.nvme_gbs, pc.vram_gib, pc.vram_gbs) == (60.0, 6.5, 12.0, 672.0)
    assert parse_nvme("2x6.5") == (2, 6.5) and parse_nvme("7") == (1, 7.0)
    with pytest.raises(ValueError):
        parse_nvme("fast")
    h = get_hardware("this-pc", nvme_count=2, io_cap_gbs=10.0)
    assert h.io_gbs == 10.0 and h.name == "this-pc*"
    assert HARDWARE["workstation-128gb-2nvme"].io_gbs == 24.0
    assert HARDWARE["mac-studio-192gb"].unified
    with pytest.raises(KeyError):
        get_hardware("this-pc", warp_drive=1)
    assert get_hardware("mac-studio-192gb", vram_gbs=0.5).vram_gbs == 0.5
    with pytest.raises(ValueError):
        get_hardware("mac-studio-192gb", vram_gbs=0.0)                # unified memory needs a GPU bandwidth
    assert get_hardware("laptop-16gb", vram_gbs=0.0, pcie_gbs=0.0).has_gpu is False


def test_sweep_reuses_trace_and_orders_results():
    rs = sweep({"cache_gb": [0.2, 0.6, 1.5]}, model=SMALL, tokens=400, zipf=1.2, reuse=0.0)
    hits = [r.hit_rate for r in rs]
    assert hits == sorted(hits) and hits[0] < hits[-1]
    assert all(r.trace_source == rs[0].trace_source for r in rs)


# ---- CLI -------------------------------------------------------------------------

def test_cli_smoke(capsys, tmp_path):
    assert main(["--list"]) == 0
    assert "kimi-k3" in capsys.readouterr().out
    assert main(["--model", SMALL, "--tokens", "300", "--cache-gb", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "SIMULATED" in out and "tok/s" in out
    assert main(["--model", SMALL, "--tokens", "300", "--compare-policies", "--markdown", "--cache-gb", "0.3"]) == 0
    out = capsys.readouterr().out
    assert out.count("| ") > 6 and "belady" in out and "lru-layer" in out
    tpath = tmp_path / "x.hrtr"
    assert main(["--model", SMALL, "--tokens", "200", "--save-trace", str(tpath), "--stats"]) == 0
    assert "entropy" in capsys.readouterr().out
    assert main(["--model", SMALL, "--trace", str(tpath), "--json", "--spec-k", "3", "--prefetch", "0.8",
                 "--nvme", "2x7", "--policy", "lfu-pinned"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["result"]["settings"]["hw"]["nvme_count"] == 2 and js["result"]["policy"] == "lfu-pinned"
    assert main(["--model", "kimi-k3", "--hw", "laptop-16gb", "--feasibility"]) == 0
    assert "NO" in capsys.readouterr().out
    assert main(["--model", SMALL, "--tokens", "300", "--sweep-zipf", "0.5,1.2", "--sweep-cache", "0.3,1"]) == 0
    assert "0.3GiB zipf=0.5" in capsys.readouterr().out
    assert main(["--model", SMALL, "--tokens", "300", "--levers"]) == 0
    assert "baseline" in capsys.readouterr().out
    assert main(["--model", "no-such-model"]) == 2


def _one_layer_counts(n_hit, n_miss):
    from hearth.sim.cache import CacheCounts
    z = lambda v=0: np.array([v], dtype=np.int32)   # noqa: E731
    return CacheCounts(z(n_hit), z(), z(), z(), z(n_miss), z(), z(), z(), z(), z(), z(), z(), z(), 0, 0, 0, "lfu")


@pytest.mark.parametrize("nvme_gbs,lat_us,h,term", [(2.0, 50.0, 3, 1),     # storage-bound: last arrival + compute
                                                    (400.0, 50.0, 3, 0),   # compute-bound
                                                    (400.0, 500.0, 0, 2)])  # latency + first arrival + all compute
def test_pipeline_formula_exact(nvme_gbs, lat_us, h, term):
    import dataclasses
    shape = dataclasses.replace(presets.get(SMALL), name="one-layer", n_layers=1)
    k = shape.top_k
    sched = build_schedule(np.arange(k, dtype=np.uint16).reshape(1, 1, k), shape.n_experts)
    costs = model_costs(shape, context=0)
    hw = get_hardware("this-pc", nvme_gbs=nvme_gbs, nvme_latency_us=lat_us)
    cal = Calibration(overhead_ms=0.5, layer_overhead_us=10.0)
    m = k - h
    tm = evaluate(_one_layer_counts(h, m), sched, costs, hw, cal)
    bw, io, ops, lat = hw.dram_gbs * 1e9, nvme_gbs * 1e9, hw.cpu_int8_tops * 1e12, lat_us * 1e-6
    S = costs.slab
    c = max(S / bw, 2.0 * costs.expert_params / ops)
    shared = max(costs.shared_bytes / bw, 2.0 * costs.shared_params / ops)
    W0 = shared + h * c
    terms = [W0 + m * c, lat + m * S / io + c, lat + S / io + m * c]
    assert int(np.argmax(terms)) == term
    moe = max(terms)
    A = max(costs.layer_bytes / bw, 2.0 * costs.layer_params / ops)
    glob = max(costs.global_bytes / bw, 2.0 * costs.global_params / ops)
    want = 0.5e-3 + 10e-6 * 1 + glob + A + moe
    assert math.isclose(float(tm.step_s[0]), want, rel_tol=1e-12)
    assert math.isclose(float(tm.nvme_demand_bytes[0]), m * S)
    stall = moe - (W0 + m * c)
    assert math.isclose(float(tm.parts[0, COMPONENTS.index("nvme")]), stall, rel_tol=1e-9, abs_tol=1e-15)
    no_miss = evaluate(_one_layer_counts(k, 0), sched, costs, hw, cal)
    assert math.isclose(float(no_miss.step_s[0]), 0.5e-3 + 10e-6 + glob + A + shared + k * c, rel_tol=1e-12)


def test_compare_policies_slicing_and_cli_calib(tmp_path, capsys):
    from hearth.sim import compare_policies
    rs = compare_policies(["lru", "belady"], model=SMALL, tokens=300, cache_gb=0.2)
    assert [r.policy for r in rs] == ["lru", "belady"] and rs[1].hit_rate >= rs[0].hit_rate
    tr = synthetic(50, 2, 8, 2, seed=0)
    part = tr[10:20]
    assert part.n_tokens == 10 and np.array_equal(part.ids, tr.ids[10:20]) and part.n_experts == 8
    with pytest.raises(TypeError):
        tr[3]
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"overhead_ms": 50.0, "io_eff": 0.5}), encoding="utf-8")
    args = ["--model", SMALL, "--tokens", "300", "--cache-gb", "0.3", "--json"]
    assert main(args) == 0
    base = json.loads(capsys.readouterr().out)["result"]
    assert main(args + ["--calib", str(cal)]) == 0
    slow = json.loads(capsys.readouterr().out)["result"]
    assert slow["calibration"]["overhead_ms"] == 50.0 and slow["calibration"]["io_eff"] == 0.5
    assert slow["tok_s"] < base["tok_s"] and slow["hit_rate"] == base["hit_rate"]


# ---- targeted invariants (added after mutation testing) ------------------------------

def _counts(**per_group):
    from hearth.sim.cache import CacheCounts
    n = len(next(iter(per_group.values())))
    names = ["hit", "hit_cold", "pfhit", "pfhit_cold", "miss", "miss_cold", "bypass", "vram", "skip", "pf_issued",
             "pf_issued_cold", "pf_wasted", "pf_wasted_cold"]
    arrs = {k: np.asarray(per_group.get(k, [0] * n), dtype=np.int32) for k in names}
    return CacheCounts(**arrs, evictions=0, slots=0, n_pinned=0, policy="lfu")


def test_slots_equal_to_group_size_keep_exactly_the_previous_group():
    """With as many slots as experts per layer (L=1), every correct policy ends each group holding exactly
    that group, so hits == overlap with the previous token. Exercises protection in every policy."""
    tr = synthetic(300, 1, 6, 2, zipf=0.5, reuse=0.3, seed=9)
    sched = build_schedule(tr.ids, 6)
    want = sum(len(set(tr.ids[t, 0]) & set(tr.ids[t - 1, 0])) for t in range(1, 300))
    prof = np.zeros(6)
    for pol, kw in [("lru", {}), ("lru-layer", {}), ("lfu", {}), ("lfu", {"lfu_samples": 1}),
                    ("pinned", {"profile": prof}), ("lfu-pinned", {"profile": prof})]:
        c = run_cache(sched, pol, 2, **kw)
        assert int(c.hit.sum()) == want, pol
        assert int(c.bypass.sum()) == 0 and c.n_pinned == 0


def test_lru_recency_and_cold_cache_accounting():
    seq = np.array([0, 1, 0, 2, 0], dtype=np.uint16).reshape(-1, 1, 1)   # LRU keeps 0; FIFO would evict it
    sched = build_schedule(seq, 3)
    assert int(run_cache(sched, "lru", 2).hit.sum()) == 2
    assert int(run_cache(sched, "belady", 2).hit.sum()) == 4        # warm start {0, 1}; only 2's one use misses
    assert int(run_cache(sched, "belady", 2, measure_step=3).hit[3:].sum()) == 2   # holds {2, 0} by then
    tr = synthetic(200, 3, 16, 4, zipf=0.8, seed=2)
    big = run_cache(build_schedule(tr.ids, 16), "lfu", 3 * 16)
    assert big.evictions == 0
    assert int(big.miss.sum()) == len(np.unique(build_schedule(tr.ids, 16).acc))
    zero = run_cache(build_schedule(tr.ids, 16), "lfu", 0, lfu_samples=4)
    assert int(zero.hit.sum()) == 0 and int(zero.bypass.sum()) == int(zero.miss.sum()) == 200 * 3 * 4


def test_prefetch_bookkeeping_invariants():
    L, E, k = 3, 64, 4
    tr = synthetic(200, L, E, k, zipf=0.0, reuse=0.0, seed=5)
    sched = build_schedule(tr.ids, E)
    c = run_cache(sched, "lru", 10_000, prefetch=Prefetch(recall=0.6, extra=2), popularity=tr.frequencies())
    issued, used, wasted = (x.reshape(-1, L) for x in (c.pf_issued, c.pfhit, c.pf_wasted))
    assert (issued[:, :-1] == used[:, 1:] + wasted[:, :-1]).all()
    assert (issued[:, -1] == 0).all() and (used[:, 0] == 0).all()
    # recall 0: only wrong guesses; in a cold cache the first step prefetches k + extra distinct experts per layer
    wrong = run_cache(sched, "lru", 10_000, prefetch=Prefetch(recall=0.0, extra=4), popularity=tr.frequencies())
    first = wrong.pf_issued.reshape(-1, L)[0]
    assert list(first) == [k + 4, k + 4, 0]
    assert int(wrong.pfhit.sum()) == 0
    # wrong guesses are drawn across the whole popularity distribution, not a fixed few
    assert int(wrong.pf_issued.reshape(-1, L)[:, 0].sum()) >= 30
    assert int(run_cache(sched, "lru", 10_000, prefetch=Prefetch(0.0, 1)).pf_issued.sum()) > 0


def test_prefetch_timing_exact():
    """Two MoE layers, one step: layer 1's prefetched experts arrive only partly in time."""
    import dataclasses
    shape = dataclasses.replace(presets.get(SMALL), name="two-layer", n_layers=2)
    k = shape.top_k
    sched = build_schedule(np.arange(2 * k, dtype=np.uint16).reshape(1, 2, k), shape.n_experts)
    costs = model_costs(shape, context=0)
    hw = get_hardware("this-pc", nvme_gbs=2.0, nvme_latency_us=50.0)
    cal = Calibration(overhead_ms=0.0, layer_overhead_us=0.0)
    h0, m0, h1, p1, m1, w0 = 6, 2, 3, 3, 2, 2
    cnt = _counts(hit=[h0, h1], miss=[m0, m1], pfhit=[0, p1], pf_issued=[p1 + w0, 0], pf_wasted=[w0, 0])
    tm = evaluate(cnt, sched, costs, hw, cal)
    bw, io, ops, lat, S = hw.dram_gbs * 1e9, 2e9, hw.cpu_int8_tops * 1e12, 50e-6, costs.slab
    c = max(S / bw, 2.0 * costs.expert_params / ops)
    A = max(costs.layer_bytes / bw, 2.0 * costs.layer_params / ops)
    glob = max(costs.global_bytes / bw, 2.0 * costs.global_params / ops)

    def moe(W0, m):
        return max(W0 + m * c, lat + m * S / io + c, lat + S / io + m * c)
    t0 = moe(h0 * c, m0)
    slack = t0 - (lat + m0 * S / io)
    f = min(1.0, (slack + A) * io / ((p1 + w0) * S))
    assert 0.0 < f < 1.0
    t1 = moe((h1 + f * p1) * c, m1 + (1 - f) * p1)
    assert math.isclose(float(tm.step_s[0]), glob + 2 * A + t0 + t1, rel_tol=1e-12)
    assert math.isclose(float(tm.nvme_prefetch_bytes[0]), f * (p1 + w0) * S, rel_tol=1e-12)
    assert math.isclose(float(tm.nvme_wasted_bytes[0]), f * w0 * S, rel_tol=1e-12)
    assert math.isclose(float(tm.nvme_demand_bytes[0]), (m0 + m1 + (1 - f) * p1) * S, rel_tol=1e-12)
    assert math.isclose(float(tm.late_prefetch[0]), (1 - f) * p1, rel_tol=1e-12)


def test_compute_bound_roofline():
    import dataclasses
    shape = dataclasses.replace(presets.get(SMALL), name="one-layer", n_layers=1)
    k = shape.top_k
    sched = build_schedule(np.arange(k, dtype=np.uint16).reshape(1, 1, k), shape.n_experts)
    costs = model_costs(shape, context=0)
    hw = get_hardware("this-pc", cpu_int8_tops=0.002)
    tm = evaluate(_counts(hit=[k], miss=[0]), sched, costs, hw, Calibration(overhead_ms=0.0, layer_overhead_us=0.0))
    ops = 0.002e12
    want = 2.0 * (costs.global_params + costs.layer_params + k * costs.expert_params) / ops
    assert math.isclose(float(tm.step_s[0]), want, rel_tol=1e-12)
    cpu = float(tm.parts[0, COMPONENTS.index("cpu")])
    assert cpu > 0.9 * want
    half = evaluate(_counts(hit=[k], miss=[0]), sched, costs, hw, Calibration(overhead_ms=0.0, layer_overhead_us=0.0,
                                                                           compute_eff=0.5))
    assert math.isclose(float(half.step_s[0]), 2 * want, rel_tol=1e-12)
    gpu = evaluate(_counts(hit=[k], miss=[0]), sched, costs, hw, Calibration(), "dense")
    assert float(gpu.parts[0, COMPONENTS.index("cpu")]) < cpu      # backbone compute moved off the CPU


def test_model_cost_formulas():
    k2, k3, ol = presets.get("kimi-k2"), presets.get("kimi-k3"), presets.get(SMALL)
    c2 = model_costs(k2, context=0)
    b = 8.25 / 8
    assert math.isclose(c2.layer_bytes, k2.attn_params() * b + 384 * 7168 * 4 + 2 * 7168 * 4)
    assert math.isclose(c2.shared_bytes, 3 * 7168 * 2048 * b)
    assert math.isclose(c2.global_bytes, (163840 * 7168 + k2.attn_params() + 3 * 7168 * 18432) * b + 2 * 7168 * 4
                        + 7168 * 4)
    c3 = model_costs(k3, context=0)
    assert math.isclose(c3.layer_bytes, (k3.attn_params() + 2 * 7168 * 3584) * b + 896 * 7168 * 4 + 2 * 7168 * 4)
    assert c3.expert_params == 3 * 3584 * 3072 and any("KDA" in n for n in c3.notes)
    assert not any("KDA" in n for n in c2.notes) and not any("latent" in n for n in c2.notes)
    co = model_costs(ol, context=1000, kv_elem_bytes=2.0)
    assert co.kv_bytes_per_layer == 2 * 16 * 128 * 2.0 * 1000
    assert model_costs(k2, context=1000).kv_bytes_per_layer == (512 + 64) * 4.0 * 1000
    assert slab_bytes(7168, 2048, 8.25) == ((2048 * 7392 * 2 + 7168 * 2112) + 4095) // 4096 * 4096


def test_gpu_expert_tier_attribution():
    r = simulate("kimi-k2", "this-pc", tokens=300, gpu="dense+experts", dense_bits=4.25)
    dram_ms = (1 - r.vram_rate) * r.loads_per_token * r.slab_bytes / (60e9) * 1e3
    assert math.isclose(r.time_per_token_ms["dram"], dram_ms, rel_tol=1e-9)


def test_trace_stats_exact():
    ids = np.array([[0, 1], [0, 2], [0, 2], [3, 0]], dtype=np.uint16).reshape(4, 1, 2)
    st = Trace(ids, n_experts=5).stats(top_fracs=(0.2, 0.4), windows=(1, 2, 7), mass_quantiles=(0.5, 0.75, 0.9))
    assert math.isclose(float(st.reuse_rate[0]), 4 / 6)
    assert math.isclose(float(st.entropy_bits[0]), 1.75)                 # expert 4 unused: no NaN
    assert math.isclose(float(st.entropy_norm[0]), 1.75 / math.log2(5))
    assert st.mass_top_pooled == {0.2: 0.5, 0.4: 0.75} and st.mass_top == {0.2: 0.5, 0.4: 0.75}
    assert st.slots_for_mass == {0.5: 1, 0.75: 2, 0.9: 4}
    assert st.union_per_window == {1: 2.0, 2: 3.0}                       # w=7 > tokens is skipped
    assert set(synthetic(20, 2, 8, 2).stats().union_per_window) == {1, 2, 3, 5}


def test_trace_edge_cases(tmp_path):
    one = synthetic(5, 2, 4, 1, seed=1, n_layers=3)
    back = Trace.load(one.save(tmp_path / "k1.hrtr"))
    assert np.array_equal(back.ids, one.ids) and back.n_layers == 3
    raw = (tmp_path / "k1.hrtr").read_bytes()
    for off in (16, 20, 12):                                           # top_k = 0, n_moe = 0, n_experts = 0
        bad = tmp_path / f"z{off}.hrtr"
        bad.write_bytes(raw[:off] + struct.pack("<I", 0) + raw[off + 4:])
        with pytest.raises(ValueError):
            Trace.load(bad)
    assert synthetic(0, 2, 4, 2, n_layers=7).n_layers == 7
    assert synthetic(10, 2, 4, 2, reuse=0.0, n_layers=7).n_layers == 7
    assert synthetic(10, 2, 4, 2, reuse=0.5, n_layers=7).n_layers == 7
    assert synthetic(3, 1, 65536, 1, zipf=0.0).ids.shape == (3, 1, 1)
    with pytest.raises(ValueError):
        synthetic(3, 1, 65537, 1)
    with pytest.raises(ValueError):
        synthetic(3, 1, 4, 5)
    with pytest.raises(ValueError):
        synthetic(3, 1, 4, 0)
    with pytest.raises(ValueError):
        synthetic(3, 1, 2 ** 40, 1)                 # rejected before any per-expert array is allocated
    assert model_costs(presets.get(SMALL), expert_bits=32.0, dense_bits=32.0, embed_bits=32.0).slab > 0
    for bad in (dict(expert_bits=32.5), dict(dense_bits=0.0), dict(embed_bits=33.0), dict(kv_elem_bytes=9.0)):
        with pytest.raises(ValueError):
            model_costs(presets.get(SMALL), **bad)
    save_usage(tmp_path / "u.usage", np.ones((2, 4), np.float32))
    blob = (tmp_path / "u.usage").read_bytes()
    for name, b in {"magic": b"XXXX" + blob[4:], "version": blob[:4] + struct.pack("<I", 2) + blob[8:],
                    "size": blob[:-4]}.items():
        (tmp_path / f"{name}.usage").write_bytes(b)
        with pytest.raises(ValueError):
            load_usage(tmp_path / f"{name}.usage")


def test_synthetic_rank_order_and_exact_refill():
    seed = 6
    tr = synthetic(3000, 1, 32, 4, zipf=1.5, reuse=0.0, seed=seed)
    p = zipf_popularity(32, 1, 1.5, np.random.default_rng(seed))[0]
    assert p[tr.ids[:, 0, 0]].mean() > 2 * p[tr.ids[:, 0, 3]].mean()   # rank 0 = most popular on average
    # extreme skew: most rows run out of distinct draws and use the exact Gumbel refill
    E, k, T = 12, 6, 20000
    for reuse in (0.0, 0.4):
        tr = synthetic(T, 1, E, k, zipf=3.0, reuse=reuse, seed=seed)
        assert (np.diff(np.sort(tr.ids.astype(int), axis=-1), axis=-1) > 0).all()
        if reuse == 0.0:
            p = zipf_popularity(E, 1, 3.0, np.random.default_rng(seed))[0]
            emp = np.bincount(tr.ids.ravel(), minlength=E) / T
            g = np.log(p)[None, :] + np.random.default_rng(2).gumbel(size=(T, E))
            ref = np.bincount(np.argpartition(-g, k - 1, axis=1)[:, :k].ravel(), minlength=E) / T
            assert np.abs(emp - ref).max() < 0.02


def test_policy_registry_and_tied_embeddings():
    assert ONLINE_POLICIES == ("lru", "lru-layer", "lfu", "pinned", "lfu-pinned")
    assert POLICIES == ONLINE_POLICIES + ("belady",)
    import dataclasses
    base = presets.get(SMALL)
    tied = dataclasses.replace(base, name="tied", tie_embeddings=True)
    cu, ct = model_costs(base), model_costs(tied)
    assert ct.global_params == cu.global_params - base.vocab * base.d_model
    assert math.isclose(cu.resident_bytes - ct.resident_bytes, base.vocab * base.d_model * 8.25 / 8)


def test_exact_step_cpu_vs_gpu_backbone():
    """One MoE layer with shared experts, compute-bound CPU (tiny int8 rate), all experts cached."""
    import dataclasses
    shape = dataclasses.replace(presets.get("qwen1.5-moe-a2.7b"), name="one-layer", n_layers=1)
    k, E = shape.top_k, shape.n_experts
    sched = build_schedule(np.arange(k, dtype=np.uint16).reshape(1, 1, k), E)
    costs = model_costs(shape, context=0)
    hw = get_hardware("this-pc", cpu_int8_tops=0.01)
    cal = Calibration(overhead_ms=0.25, layer_overhead_us=5.0)
    bw, vbw, ops = hw.dram_gbs * 1e9, hw.vram_gbs * 1e9, 0.01e12
    fixed = 0.25e-3 + 5e-6
    c = max(costs.slab / bw, 2.0 * costs.expert_params / ops)
    cpu_want = (fixed + max(costs.global_bytes / bw, 2.0 * costs.global_params / ops)
                + max(costs.layer_bytes / bw, 2.0 * costs.layer_params / ops)
                + max(costs.shared_bytes / bw, 2.0 * costs.shared_params / ops) + k * c)
    cpu = evaluate(_counts(hit=[k]), sched, costs, hw, cal)
    assert math.isclose(float(cpu.step_s[0]), cpu_want, rel_tol=1e-12)
    # attribution with shared experts: the parts add up, and dram is every byte streamed at DRAM bandwidth
    assert math.isclose(float(cpu.parts[0].sum()), cpu_want, rel_tol=1e-12)
    streamed = costs.global_bytes + costs.layer_bytes + costs.shared_bytes + k * costs.slab
    assert costs.shared_bytes > 0 and math.isclose(float(cpu.parts[0, COMPONENTS.index("dram")]), streamed / bw,
                                                   rel_tol=1e-12)
    gpu_want = (fixed + (costs.global_bytes + costs.layer_bytes + costs.shared_bytes) / vbw
                + 2 * (hw.gpu_sync_us * 1e-6 + 4.0 * shape.d_model / (hw.pcie_gbs * 1e9)) + k * c)
    gpu = evaluate(_counts(hit=[k]), sched, costs, hw, cal, "dense")
    assert math.isclose(float(gpu.step_s[0]), gpu_want, rel_tol=1e-12)
    assert math.isclose(float(gpu.vram_bytes[0]), costs.global_bytes + costs.layer_bytes + costs.shared_bytes)


def test_batched_window_compute_uses_tokens_per_expert():
    import dataclasses
    shape = dataclasses.replace(presets.get(SMALL), name="one-layer", n_layers=1)
    ids = np.array([list(range(0, 8)), list(range(4, 12)), list(range(8, 16))], dtype=np.uint16).reshape(3, 1, 8)
    sched = build_schedule(ids, shape.n_experts, spec_k=2, spec_alpha=1.0)
    assert sched.n_steps == 1 and int(sched.step_npos[0]) == 3 and int(sched.pairs[0]) == 24
    assert int(np.diff(sched.ptr)[0]) == 16                                  # union of the window
    costs = model_costs(shape, context=0)
    hw = get_hardware("this-pc", cpu_int8_tops=0.002)
    tm = evaluate(_counts(hit=[16]), sched, costs, hw, Calibration(overhead_ms=0.0, layer_overhead_us=0.0))
    bw, ops = hw.dram_gbs * 1e9, 0.002e12
    want = (max(costs.global_bytes / bw, 3 * 2.0 * costs.global_params / ops)
            + max(costs.layer_bytes / bw, 3 * 2.0 * costs.layer_params / ops)
            + 16 * max(costs.slab / bw, 1.5 * 2.0 * costs.expert_params / ops))
    assert math.isclose(float(tm.step_s[0]), want, rel_tol=1e-12)


def test_simulate_profiles_warnings_and_rates(tmp_path):
    whole = small(policy="pinned", cache_gb=0.5, warmup=0)
    assert any("WHOLE trace" in w for w in whole.warnings) and whole.n_pinned > 0
    assert any("upper bound" in w for w in small(policy="belady", cache_gb=0.5).warnings)
    assert not any("upper bound" in w for w in small(policy="lfu", cache_gb=0.5).warnings)
    short = small(tokens=150)
    assert any("noisy" in w for w in short.warnings)
    assert not any("noisy" in w for w in small(tokens=600).warnings)
    full = small(cache_gb=0.3, lossy=Lossy(cold_bits=2.25, cold_frac=1.0))
    assert full.bytes_per_token["nvme"] < small(cache_gb=0.3).bytes_per_token["nvme"]
    none = small(cache_gb=0.3, lossy=Lossy(cold_bits=2.25, cold_frac=0.0))
    assert none.bytes_per_token == small(cache_gb=0.3).bytes_per_token
    with pytest.raises(ValueError):
        small(lossy=Lossy(cold_bits=2.25, cold_frac=1.5))
    # a usage profile with one row per transformer layer (dense rows included) is accepted
    k2 = presets.get("kimi-k2")
    heat = np.zeros((k2.n_layers, k2.n_experts), np.float32)
    heat[1:, :8] = 100.0
    p = save_usage(tmp_path / "k2.usage", heat)
    r = simulate("kimi-k2", "this-pc", tokens=300, policy="pinned", cache_gb=2.0, profile=str(p))
    assert r.n_pinned > 0
    with pytest.raises(ValueError):
        simulate("kimi-k2", "this-pc", tokens=300, policy="pinned", profile=np.zeros(5))


def test_gpu_rates_and_vram_bytes():
    r = simulate("kimi-k2", "this-pc", tokens=300, gpu="dense+experts", dense_bits=4.25)
    assert r.vram_rate > 0 and r.hit_rate > r.vram_rate
    assert math.isclose(r.hit_rate + r.prefetch_rate + r.miss_rate + r.skip_rate, 1.0)
    d = simulate("kimi-k2", "this-pc", tokens=300, gpu="dense", dense_bits=4.25)
    c = d._costs
    assert d.vram_rate == 0.0
    assert math.isclose(d.bytes_per_token["vram"], c.global_bytes + c.n_moe_layers * (c.layer_bytes + c.shared_bytes))


def test_levers_rows():
    from hearth.sim import levers
    rows = dict(levers("kimi-k2", "this-pc", tokens=300))
    base = rows["baseline: lfu, all free RAM as cache, prefetch off, no speculative decoding"]
    nv = rows["+1 NVMe drive (2x6.5 GB/s)"]
    assert nv.hit_rate == base.hit_rate and nv.tok_s > base.tok_s
    assert rows["prefetch next layer (recall 0.8)"].prefetch_rate > 0
    assert rows["prefetch next layer (recall 0.8)"].settings["prefetch"] == {"recall": 0.8, "extra": 0}
    assert rows["Q3 experts 3.25 bpw (LOSSY vs Q4, roadmap R03)"].slab_bytes < base.slab_bytes
    assert rows["LOSSY top-k 8->6"].loads_per_token == 60 * 6
    assert rows["policy lru (global)"].policy == "lru"
    gpu = rows["GPU: Q4 backbone (LOSSY vs Q8) + hot experts on GPU (roadmap R01)"]   # Q8 does not fit 12 GiB
    assert gpu.feasible and gpu.settings["dense_bits"] == 4.25 and gpu.vram_rate > 0 and gpu.lossy
    assert rows["speculative k=4, alpha=0.8"].tokens_per_step > rows["speculative k=4, alpha=0.6"].tokens_per_step
    assert rows["2x RAM (123 GiB)"].cache_gib > base.cache_gib
    q4 = rows["dense backbone Q4 instead of Q8 (LOSSY vs the Q8 default)"]
    assert q4.settings["dense_bits"] == 4.25 and any("LOSSY dense backbone" in x for x in q4.lossy)
    assert base.lossy == [] and rows["Q3 experts 3.25 bpw (LOSSY vs Q4, roadmap R03)"].lossy
    combo = rows["prefetch 0.8 + spec k=4 a=0.6 + 1 more NVMe"]
    assert combo.prefetch_rate > 0 and combo.tokens_per_step > 1 and combo.settings["hw"]["nvme_count"] == 2
    small_rows = dict(levers(SMALL, "this-pc", tokens=300))
    assert "GPU: backbone + hot experts on GPU (roadmap R01)" in small_rows
    assert not any(k.startswith("GPU") for k in dict(levers(SMALL, "laptop-16gb", tokens=300)))


def test_levers_labels_follow_the_baseline():
    rows = levers(SMALL, "this-pc", tokens=300, policy="lru", cache_gb=0.2, spec=Spec(4, 0.6),
                  prefetch=Prefetch(0.5, 0))
    base = rows[0][1]
    assert rows[0][0] == ("baseline: lru, 0.2 GiB cache, prefetch recall 0.5 + 0 extra, "
                          "speculative k=4 alpha=0.6")
    d = dict(rows)
    assert d["policy lfu"].policy == "lfu"
    assert d["prefetch off"].settings["prefetch"] is None and d["prefetch off"].settings["spec"] is not None
    assert d["speculative decoding off"].settings["spec"] is None
    ram = d["2x RAM (123 GiB), cache 0.2 -> 61.8 GiB"]
    assert ram.cache_gib > 10 * base.cache_gib
    for label, r in rows[1:]:
        assert r.settings != base.settings, label            # every row changes something
    assert not any(label.startswith("prefetch 0.8") for label, _ in rows)   # nothing left to combine
    gpu = dict(levers(SMALL, "this-pc", tokens=300, gpu="dense"))
    assert gpu["GPU off (CPU only)"].settings["gpu"] == "off"


def test_low_bit_encodings_are_labelled_lossy(capsys):
    assert small(cache_gb=0.3).lossy == [] and small(cache_gb=0.3, dense_bits=16.0).lossy == []
    assert any("LOSSY dense backbone at 4.25" in x for x in small(cache_gb=0.3, dense_bits=4.25).lossy)
    assert any("LOSSY experts at 3.25" in x for x in small(cache_gb=0.3, expert_bits=3.25).lossy)
    assert any("LOSSY embeddings" in x for x in small(cache_gb=0.3, embed_bits=4.25).lossy)
    assert main(["--model", SMALL, "--tokens", "300", "--dense-bits", "4.25"]) == 0
    assert "LOSSY dense backbone" in capsys.readouterr().out


def test_prefetched_experts_stay_protected_until_used_or_step_end():
    """hx_store.h: experts prefetched for the current token are protected until used or the token ends.
    Token 0 prefetches one wrong guess for layer 1 (recall 0; three guesses, one free slot). Layer 1's two demand misses
    must evict layer 0's experts, not the unused prefetch, so token 1 finds neither of them."""
    ids = np.array([[[0, 1], [2, 3]], [[0, 1], [2, 3]]], dtype=np.uint16)
    sched = build_schedule(ids, 16)
    c = run_cache(sched, "lfu", 3, prefetch=Prefetch(0.0, 1), popularity=np.ones((2, 16)))
    assert c.pf_issued.tolist()[0] == 1 and c.pf_wasted.tolist()[0] == 1
    assert c.miss.tolist()[:3] == [2, 2, 2] and int(c.hit[2]) == 0
    assert int(c.bypass.sum()) == 0                            # protection ended with token 0
    # bookkeeping invariants still hold with protection on a realistic run
    tr = synthetic(200, 3, 64, 4, zipf=1.0, reuse=0.2, seed=5)
    s2 = build_schedule(tr.ids, 64)
    c2 = run_cache(s2, "lfu", 40, prefetch=Prefetch(0.5, 3), popularity=tr.frequencies(), io_threads=1)
    issued, used, wasted = (x.reshape(-1, 3) for x in (c2.pf_issued, c2.pfhit, c2.pf_wasted))
    assert (issued[:, :-1] == used[:, 1:] + wasted[:, :-1]).all()


def test_demand_evicts_unused_prefetch_only_as_last_resort():
    """store.c: a demand may evict a protected (unused) prefetch as a last resort; it never bypasses while one
    exists. One speculative window: layer 0 needs {0}, layer 1 needs {1, 2, 3}; 3 slots. Layer 0 prefetches
    two wrong guesses into the free slots, so layer 1's first miss evicts expert 0 and the next two must
    evict the wrong guesses."""
    ids = np.array([[[0], [1]], [[0], [2]], [[0], [3]]], dtype=np.uint16)
    sched = build_schedule(ids, 16, spec_k=2, spec_alpha=1.0)
    assert np.diff(sched.ptr).tolist() == [1, 3]
    for pol in ("lru", "lfu"):
        c = run_cache(sched, pol, 3, prefetch=Prefetch(0.0, 1), popularity=np.ones((2, 16)))
        assert c.pf_issued.tolist() == [2, 0] and c.pf_wasted.tolist() == [2, 0], pol
        assert c.miss.tolist() == [1, 3] and c.bypass.tolist() == [0, 0] and c.evictions == 3, pol


def test_prefetch_extra_is_clamped_and_fast():
    L, E, k = 3, 64, 4
    tr = synthetic(200, L, E, k, zipf=1.2, reuse=0.0, seed=5)
    sched = build_schedule(tr.ids, E)
    c = run_cache(sched, "lru", 10_000, prefetch=Prefetch(recall=0.0, extra=10 ** 6), popularity=tr.frequencies())
    assert list(c.pf_issued.reshape(-1, L)[0]) == [E - k, E - k, 0]     # every other expert, once
    t = time.perf_counter()
    r = small(tokens=100, cache_gb=0.5, prefetch=Prefetch(0.5, 100_000))
    assert time.perf_counter() - t < 30.0 and r.prefetch_rate > 0      # was > 10 min before the clamp


@pytest.mark.parametrize("kw", [
    dict(cache_gb=-5), dict(io_threads=0), dict(io_threads=-20), dict(lfu_decay=0.0), dict(lfu_decay=1.5),
    dict(lfu_samples=-3), dict(pin_fraction=2.0), dict(context=-100), dict(expert_bits=0.0),
    dict(dense_bits=float("nan")), dict(embed_bits=-1.0), dict(spec=(2, 1.5)), dict(spec=(-1, 0.5)),
    dict(prefetch=(1.5, 0)), dict(prefetch=(0.5, -1)), dict(lossy=Lossy(cold_bits=0.0)),
    dict(hw_overrides={"dram_gbs": 0.0}), dict(hw_overrides={"nvme_count": 0}), dict(hw_overrides={"nvme_gbs": 0.0}),
    dict(hw_overrides={"cpu_int8_tops": 0.0}), dict(hw_overrides={"ram_gib": float("inf")}),
    dict(hw_overrides={"pcie_gbs": 0.0}), dict(tokens=-5), dict(zipf=float("nan")), dict(prefetch=-0.5),
    dict(prefetch=1.5), dict(spec=(4.5, 0.5)), dict(warmup=-1), dict(warmup=5.5), dict(io_threads=2.5),
    dict(lfu_samples=1.5)])
def test_simulate_rejects_bad_inputs(kw):
    with pytest.raises(ValueError):
        simulate(SMALL, "this-pc", **{"tokens": 60, **kw})


def test_cli_rejects_bad_inputs_without_traceback(capsys):
    for opt in (["--dram-gbs", "0"], ["--lfu-decay", "0"], ["--expert-bits", "0"], ["--int8-tops", "0"],
                ["--nvme", "0x5"], ["--nvme", "2x0"], ["--cache-gb", "-5"], ["--pin-fraction", "2"],
                ["--io-threads", "-20"], ["--context", "-100"], ["--spec-k", "2", "--spec-alpha", "1.5"],
                ["--lfu-decay", "1.5"], ["--lfu-samples", "-3"], ["--dram-eff", "0"], ["--tokens", "-5"],
                ["--prefetch", "1.5"], ["--feasibility", "--io-threads", "0"], ["--sweep-cache", "1,-1"],
                ["--spec-k", "-1"], ["--prefetch-extra", "-1"], ["--prefetch", "-0.5"], ["--spec-alpha", "1.5"],
                ["--prefetch-extra", "-1", "--compare-policies"], ["--spec-k", "-2", "--levers"]):
        assert main(["--model", SMALL, "--tokens", "60"] + opt) == 2, opt
        assert "hearth sim: error" in capsys.readouterr().err, opt
    with pytest.raises(ValueError):
        Calibration(io_eff=0.0)
    with pytest.raises(ValueError):
        Calibration(overhead_ms=-1.0)


def test_trace_rejects_invalid_ids_and_huge_headers(tmp_path):
    with pytest.raises(ValueError):
        Trace(np.zeros((5, 2, 3), np.uint16), 8)                         # an expert twice in one top-k
    with pytest.raises(ValueError):
        Trace(np.full((3, 2, 2), -1), 8)                                 # negative ids
    with pytest.raises(ValueError):
        Trace(np.array([[[0.5, 1.0]]]), 4)                               # non-integer ids
    with pytest.raises(ValueError):
        Trace(np.zeros((1, 1, 1), np.uint16), 70000)
    with pytest.raises(ValueError):
        build_schedule(np.zeros((2, 1, 2), np.uint16), 4)
    with pytest.raises(ValueError):
        build_schedule(np.array([[[0, 4]]]), 4)                          # id == n_experts
    for bad in ((-1, 0.5), (2, 1.5), (2, -0.1)):
        with pytest.raises(ValueError):
            build_schedule(np.zeros((3, 1, 1), np.uint16), 4, *bad)
    assert Trace(np.zeros((4, 3, 1), np.uint16), 2).n_tokens == 4          # k = 1 needs no duplicate check
    with pytest.raises(ValueError):
        Trace(np.zeros((0, 2, 5), np.uint16), 4)                         # top_k > n_experts (no ids to check)
    # (n_layers, n_experts, top_k, n_moe): each violates exactly one bound
    for dims in ((10 ** 9,) * 4, (100, 70000, 2, 1), (4096, 65536, 1, 4096), (2048, 8192, 1024, 2048),
                 (4, 8, 9, 2), (4, 0, 1, 2)):
        hdr = struct.pack("<6I", 0x52545248, 1, dims[0], dims[1], dims[2], dims[3])   # header only
        (tmp_path / "h.hrtr").write_bytes(hdr)
        with pytest.raises(ValueError):
            Trace.load(tmp_path / "h.hrtr")


def test_pcie_bandwidth_sets_gpu_hand_off_time():
    slow_hw = get_hardware("this-pc", pcie_gbs=0.05)
    kw = dict(tokens=600, zipf=1.2, reuse=0.0, seed=3, cache_gb=0.3)
    fast, slow = small(cache_gb=0.3, gpu="dense"), simulate(SMALL, slow_hw, gpu="dense", **kw)
    assert slow.tok_s < fast.tok_s and slow.hit_rate == fast.hit_rate
    assert simulate(SMALL, slow_hw, **kw).tok_s == small(cache_gb=0.3).tok_s      # CPU-only: PCIe unused
    mac = simulate(SMALL, get_hardware("mac-studio-192gb"), gpu="dense", **kw)
    assert mac.tok_s > 0                                                         # unified: no PCIe term


def _belady_reference_hits(sched, cap, start_group):
    """Plain re-implementation of the relaxed optimum (no heap): hits per group."""
    acc, ptr = sched.acc.tolist(), sched.ptr.tolist()
    start = ptr[start_group]
    pos = {}
    for i, key in enumerate(acc):
        pos.setdefault(key, []).append(i)

    def next_use(key, after):
        for q in pos[key]:
            if q >= max(after, start):
                return q
        return math.inf

    keys = sorted(pos, key=lambda x: next_use(x, 0))
    state = {x for x in keys[:cap] if next_use(x, 0) < math.inf}
    hits = []
    for g in range(sched.n_groups):
        grp = acc[ptr[g]:ptr[g + 1]]
        hits.append(sum(x in state for x in grp))
        cand = sorted(state | set(grp), key=lambda x: (next_use(x, ptr[g + 1]), x))
        state = set(cand[:cap])
    return hits


def test_belady_matches_reference_on_long_traces():
    """Long enough for Belady's heap to be rebuilt (> 4 * residents + 4096 entries); also next_at()."""
    from hearth.sim.cache import make_policy
    tr = synthetic(2500, 2, 10, 3, zipf=1.0, reuse=0.3, seed=4)
    for spec, cap, ms in (((0, 0.0), 7, 0), ((0, 0.0), 12, 400), ((3, 0.6), 9, 100)):
        sched = build_schedule(tr.ids, 10, *spec, seed=1)
        c = run_cache(sched, "belady", cap, measure_step=ms)
        assert int(c.hit.sum()) > 4096 + 4 * cap
        want = _belady_reference_hits(sched, cap, ms * 2)
        assert c.hit[ms * 2:].tolist() == want[ms * 2:], spec
    pol = make_policy("belady", 7, sched, sched.acc.tolist(), measure_step=ms)
    start = int(sched.ptr[ms * 2])
    rng = np.random.default_rng(0)
    for key, p in zip(rng.integers(0, 20, 50), rng.integers(0, len(sched.acc), 50)):
        later = np.flatnonzero(sched.acc[max(p, start):] == key)
        assert pol.next_at(int(key), int(p)) == (int(later[0]) + max(p, start) if len(later) else pol.INF)
    r = small(policy="belady", cache_gb=0.3, prefetch=Prefetch(0.8, 1))
    assert r.prefetch_rate > 0 and any("NOT an upper bound" in w for w in r.warnings)
    assert not any("NOT an upper bound" in w for w in small(policy="belady", cache_gb=0.3).warnings)


def test_run_cache_defaults_follow_the_engine():
    tr = synthetic(100, 2, 32, 4, zipf=1.0, seed=1)
    sched = build_schedule(tr.ids, 32)
    c = run_cache(sched, "pinned", 40, profile=tr.frequencies().ravel())
    assert c.n_pinned == 40 - max(engine_min_slots(4, 8), sched.max_union + 4)


def test_eviction_accounting_with_prefetch():
    tr = synthetic(300, 3, 64, 4, zipf=1.0, reuse=0.2, seed=8)
    sched = build_schedule(tr.ids, 64)
    for pol in ("lru", "lfu"):
        c = run_cache(sched, pol, 40, prefetch=Prefetch(0.6, 3), popularity=tr.frequencies())
        admitted = int((c.miss - c.bypass).sum() + c.pf_issued.sum())
        assert int(c.pf_issued.sum()) > 0 and c.evictions == admitted - 40, pol   # the cache starts empty, ends full


def test_wrong_guess_refill_draws_from_the_next_layers_popularity():
    """Layer 1 has 10 hot experts; the other wrong guesses must come from layer 1's (flat) remainder, not from
    layer 0's hot set, so over many tokens almost every layer-1 expert gets prefetched once (big cache)."""
    T, E, k = 60, 64, 4
    ids = np.zeros((T, 2, k), dtype=np.uint16)
    ids[:, 0] = [0, 1, 2, 3]
    ids[:, 1] = [60, 61, 62, 63]
    pop = np.zeros((2, E))
    pop[0, 50:60] = 1000.0
    pop[1, 0:10] = 1000.0
    c = run_cache(build_schedule(ids, E), "lru", 10_000, prefetch=Prefetch(0.0, 16), popularity=pop)
    issued = c.pf_issued.reshape(T, 2)[:, 0]
    assert issued[0] == 20 and int(issued.sum()) > 50                 # 10 hot + most of the 50 cold ones


# ---- VRAM tier vs Belady, LFU heat, partitions, held prefetches, input checks ----------------

def _hw_with_vram_slots(shape, n_slots, **kw):
    """this-pc with exactly enough VRAM for the backbone plus n_slots expert slabs (gpu=dense+experts)."""
    from hearth.sim.feasibility import VRAM_HEADROOM_GIB
    c = model_costs(shape)
    need = (c.resident_bytes - c.embed_bytes + c.kv_resident_bytes) / 2 ** 30 + VRAM_HEADROOM_GIB
    return get_hardware("this-pc", vram_gib=need + (n_slots + 0.5) * c.slab / 2 ** 30, **kw)


def test_belady_bounds_online_policies_with_a_vram_tier():
    """VRAM keys are served before the cache is consulted, so Belady must solve the problem without them: its
    hits on the other keys equal the exhaustive optimum of that sub-problem, and no online policy beats it.
    (Generation 2 warm-started Belady with VRAM keys, which then held slots for the whole run.)"""
    rng = np.random.default_rng(11)
    n_vram_heavy = 0
    for trial in range(90):
        L, E = int(rng.integers(1, 3)), int(rng.integers(3, 6))
        k = int(rng.integers(1, min(E, 3) + 1))
        ids = _random_ids(rng, int(rng.integers(4, 10)), L, E, k, zipf=float(rng.uniform(0, 1.5)))
        spec = (int(rng.integers(1, 3)), 0.6) if trial % 3 == 0 else (0, 0.0)
        sched = build_schedule(ids, E, *spec, seed=trial)
        vram = rng.choice(L * E, size=int(rng.integers(1, L * E // 2 + 1)), replace=False)
        cap = int(rng.integers(1, 4))
        ms = int(rng.integers(0, sched.n_steps))
        g0 = ms * L
        c = run_cache(sched, "belady", cap, vram_keys=vram, measure_step=ms)
        vset = set(vram.tolist())
        rest = [[x for x in grp if x not in vset] for grp in _groups(sched)]
        if any(rest):
            assert int(c.hit[g0:].sum()) == _opt_group_hits(rest, cap, g0), (trial, ids.tolist(), vram, cap, ms)
        assert int(c.vram.sum()) == sum(len(g) - len(r) for g, r in zip(_groups(sched), rest))
        prof = rng.random(L * E)
        for pol in ONLINE_POLICIES:
            for kw in ({}, {"lfu_samples": 2, "lfu_decay": 0.8, "pin_fraction": 0.9}):
                o = run_cache(sched, pol, cap, profile=prof, vram_keys=vram, io_threads=1, seed=trial, **kw)
                assert int(o.hit[g0:].sum()) <= int(c.hit[g0:].sum()), (trial, pol, kw)
        n_vram_heavy += int(c.vram.sum()) > 0
    assert n_vram_heavy > 60
    # larger schedules: bound only
    for trial in range(60):
        L, E, k = int(rng.integers(1, 4)), int(rng.integers(6, 16)), int(rng.integers(1, 4))
        ids = _random_ids(rng, int(rng.integers(10, 50)), L, E, k, zipf=float(rng.uniform(0.3, 1.5)))
        sched = build_schedule(ids, E, *((2, 0.5) if trial % 2 else (0, 0.0)), seed=trial)
        vram = rng.choice(L * E, size=int(rng.integers(1, L * E // 2)), replace=False)
        cap, ms = int(rng.integers(1, L * E // 2 + 1)), int(rng.integers(0, sched.n_steps))
        best = int(run_cache(sched, "belady", cap, vram_keys=vram, measure_step=ms).hit[ms * L:].sum())
        prof = rng.random(L * E)
        for pol in ONLINE_POLICIES:
            o = run_cache(sched, pol, cap, profile=prof, vram_keys=vram, io_threads=1, seed=trial)
            assert int(o.hit[ms * L:].sum()) <= best, (trial, pol)


def test_belady_upper_bound_with_the_gpu_expert_tier_through_simulate():
    # the verifier's reproduction: lru-layer 68.2% > belady 63.4% in generation 2
    hw = get_hardware("this-pc", vram_gib=3.0)
    kw = dict(tokens=400, zipf=1.2, reuse=0.3, seed=2, cache_gb=0.0, gpu="dense+experts")
    res = {p: simulate(SMALL, hw, policy=p, **kw) for p in POLICIES}
    assert res["belady"].vram_rate > 0.1 and len({r.vram_rate for r in res.values()}) == 1
    for p in ONLINE_POLICIES:
        assert res["belady"].hit_rate >= res[p].hit_rate, (p, res[p].hit_rate, res["belady"].hit_rate)
    # tiny shape: 2 MoE layers x 16 experts, top-3, 7 VRAM slabs, engine-minimum cache (9 slots)
    tiny = dataclasses.replace(presets.get(SMALL), name="tiny-vram", n_layers=2, n_dense_layers=0, n_experts=16,
                               top_k=3)
    hw7 = _hw_with_vram_slots(tiny, 7)
    for seed in range(5):
        tr = synthetic_for(tiny, 150, zipf=0.6 + 0.2 * seed, reuse=0.3, seed=seed)
        for spec, warmup in ((None, 15), ((3, 0.6), 0)):
            res = {p: simulate(tiny, hw7, trace=tr, policy=p, cache_gb=0.0, io_threads=1, spec=spec, warmup=warmup,
                               gpu="dense+experts") for p in POLICIES}
            assert res["belady"].feasibility.vram_expert_slots == 7 and res["belady"].slots == 9
            for p in ONLINE_POLICIES:
                assert res["belady"].hit_rate >= res[p].hit_rate - 1e-12, (seed, spec, p)


def test_lfu_decay_weights_recent_uses_more():
    """One layer, k=1, 2 slots; A A A B C B A. At token 4, C must evict A or B. Decayed heat at decay 1/2
    (scaled by 2**t): A = 1+2+4 = 7 < B = 8, so A goes; pure frequency (decay 1) keeps A (3 uses) and evicts B."""
    seq = np.array([0, 0, 0, 1, 2, 1, 0], dtype=np.uint16).reshape(-1, 1, 1)
    sched = build_schedule(seq, 3)
    recent = run_cache(sched, "lfu", 2, lfu_decay=0.5)
    assert recent.hit.tolist() == [0, 1, 1, 0, 0, 1, 0]           # t5: B hit; t6: A evicted at t4, misses
    frequent = run_cache(sched, "lfu", 2, lfu_decay=1.0)
    assert frequent.hit.tolist() == [0, 1, 1, 0, 0, 0, 1]         # t5: B evicted, misses; t6: A hit
    # sampled eviction: 64 samples of 2 residents see both (P(miss one) = 2**-63), so it matches exact LFU
    assert run_cache(sched, "lfu", 2, lfu_decay=0.5, lfu_samples=64).hit.tolist() == recent.hit.tolist()


def _lfu_exact_reference(sched, cap):
    """LFU at decay 1/2 in exact integers (heat scaled by 2**t is a sum of powers of two), with the simulator's
    rules: the group's experts are protected, ties go to the smaller key, no free slot and nothing evictable
    means a bypass. Returns (hits, bypasses) per group."""
    acc, ptr, L = sched.acc.tolist(), sched.ptr.tolist(), sched.n_moe_layers
    heat, res, t, hits, byp = {}, set(), 0, [], []
    for g in range(sched.n_groups):
        grp = acc[ptr[g]:ptr[g + 1]]
        hits.append(sum(x in res for x in grp))
        for x in grp:
            heat[x] = heat.get(x, 0) + 2 ** t
        nb = 0
        for x in grp:
            if x in res:
                continue
            if len(res) < cap:
                res.add(x)
                continue
            cand = [y for y in res if y not in grp]
            if not cand:
                nb += 1
                continue
            res.discard(min(cand, key=lambda y: (heat[y], y)))
            res.add(x)
        byp.append(nb)
        if g % L == L - 1:
            t += int(sched.step_nacc[g // L])
    return hits, byp


def test_lfu_matches_exact_reference_across_heat_rescales():
    """decay 1/2 rescales the stored heat every ~333 tokens (1e100); 1500-2000 tokens rescale 4-6 times. Plain
    and speculative schedules (tick(n) with n > 1)."""
    for seed, (T, L, E, k, cap, spec) in enumerate([(1500, 2, 10, 3, 12, (0, 0.0)), (1800, 3, 12, 2, 9, (0, 0.0)),
                                                    (2000, 2, 10, 3, 11, (3, 0.7))]):
        tr = synthetic(T, L, E, k, zipf=0.8, reuse=0.2, seed=seed)
        sched = build_schedule(tr.ids, E, *spec, seed=seed)
        c = run_cache(sched, "lfu", cap, lfu_decay=0.5)
        hits, byp = _lfu_exact_reference(sched, cap)
        assert c.hit.tolist() == hits and c.bypass.tolist() == byp, seed
        assert int(sched.step_nacc.sum()) * math.log10(2) > 4 * 100   # at least 4 rescales happened
    # no per-token sweep: the stored scale grows until it passes 1e100 (2**333 at decay 1/2), then resets
    from hearth.sim.cache import _LFU
    pol = _LFU(4, [0, 1], 2, decay=0.5)
    pol.touch([0], [])
    for _ in range(332):
        pol.tick(1)
    assert pol.inc == 2.0 ** 332 and pol.heat[0] == 1.0
    pol.tick(1)
    assert pol.inc == 1.0 and pol.heat[0] == 2.0 ** -333


def test_sampled_lfu_victims_are_uniform_over_unprotected_residents():
    from hearth.sim.cache import FREE, _LFU
    n = 10
    pol = _LFU(n, list(range(n)), n, decay=1.0, samples=1, seed=5)
    for key in range(n):
        assert pol.admit(key, 0, set()) == FREE
        pol.heat[key] = 10.0 + key                                 # key 0 is the coldest
    draws = 20000
    got = np.bincount([pol._victim(set()) for _ in range(draws)], minlength=n) / draws
    assert np.abs(got - 1 / n).max() < 0.015, got
    # protected (key 0, 2) and held (key 1) slabs are never chosen; a protected sample falls back to the
    # coldest unprotected resident (key 3): P = 1/n per free key, + 3/n for key 3
    pol.hold([1])
    got = np.bincount([pol._victim({0, 2}) for _ in range(draws)], minlength=n) / draws
    want = np.array([0, 0, 0, 4, 1, 1, 1, 1, 1, 1]) / n
    assert np.abs(got - want).max() < 0.015, got
    assert pol._victim(set(range(n))) == -1


def test_lru_layer_partition_capacities():
    from hearth.sim.cache import make_policy
    tr = synthetic(5, 4, 8, 2, seed=0)
    sched = build_schedule(tr.ids, 8)
    pol = make_policy("lru-layer", 11, sched, sched.acc.tolist())
    assert [p.cap for p in pol.parts] == [3, 3, 3, 2]
    assert [p.cap for p in make_policy("lru-layer", 8, sched, sched.acc.tolist()).parts] == [2, 2, 2, 2]
    # behaviour: 3 layers, each cycling over 3 experts, 7 slots = 3 + 2 + 2. Layer 0 keeps its cycle; the
    # others are LRU's cyclic worst case and never hit.
    T = 30
    ids = np.array([[[t % 3]] * 3 for t in range(T)], dtype=np.uint16)
    c = run_cache(build_schedule(ids, 4), "lru-layer", 7)
    assert c.hit.reshape(T, 3).sum(axis=0).tolist() == [T - 3, 0, 0]


class _NaiveLRU:
    """Reference for the held-prefetch rules: a slab's stamp is its last use or its admission; held slabs are
    skipped by admit and are the only candidates of the last resort."""

    def __init__(self, cap):
        self.cap, self.stamp, self.held, self.t = cap, {}, set(), 0

    def _now(self):
        self.t += 1
        return self.t

    def use(self, keys):
        for key in keys:
            self.stamp[key] = self._now()
            self.held.discard(key)

    def _pick(self, cand):
        return min(cand, key=self.stamp.get)

    def admit(self, key, prot, last_resort=False):
        from hearth.sim.cache import FREE, REJECT
        if not last_resort and len(self.stamp) < self.cap:
            self.stamp[key] = self._now()
            return FREE
        pool = self.held if last_resort else [x for x in self.stamp if x not in self.held]
        cand = [x for x in pool if x not in prot]
        if not cand:
            return REJECT
        v = self._pick(cand)
        self.held.discard(v)
        del self.stamp[v]
        self.stamp[key] = self._now()
        return v


class _NaiveLFU(_NaiveLRU):
    def __init__(self, cap, decay):
        super().__init__(cap)
        self.heat, self.inc, self.decay = {}, 1.0, decay

    def warm(self, keys):
        for key in keys:
            self.heat[key] = self.heat.get(key, 0.0) + self.inc

    def _pick(self, cand):
        return min(cand, key=lambda x: (self.heat.get(x, 0.0), x))


def test_held_prefetch_rules_match_a_naive_model():
    """Random group / prefetch / step-end sequences, driven through the policy interface exactly as run_cache
    drives it, against _NaiveLRU / _NaiveLFU: every admit and last resort must pick the same victim. Covers the
    out-of-order held slabs of lru, lru-layer and exact lfu (generation 3 keeps them out of the eviction order)."""
    from hearth.sim.cache import REJECT, make_policy
    rng = np.random.default_rng(3)
    n_last = n_rej = 0
    for trial in range(150):
        name = ("lru", "lru-layer", "lfu")[trial % 3]
        L = 2 if name == "lru-layer" else 1
        E = int(rng.integers(4, 9))
        cap = int(rng.integers(L, L * E))
        sched = build_schedule(np.zeros((1, L, 1), np.uint16), E)       # only the geometry is used
        acc = list(range(L * E))                                        # position i is key i
        pol = make_policy(name, cap, sched, acc, lfu_decay=0.5)
        if name == "lru-layer":
            refs = [_NaiveLRU(p.cap) for p in pol.parts]
            ref_of = lambda key: refs[key // E]                         # noqa: E731
        else:
            ref = _NaiveLFU(cap, 0.5) if name == "lfu" else _NaiveLRU(cap)
            refs, ref_of = [ref], (lambda key, r=ref: r)
        res, live = set(), set()
        for _ in range(40):
            ly = int(rng.integers(0, L))
            grp = set((ly * E + rng.choice(E, size=int(rng.integers(1, 4)), replace=False)).tolist())
            hits, miss = sorted(x for x in grp if x in res), sorted(x for x in grp if x not in res)
            pol.touch(hits, miss)
            for r in refs:
                r.use([x for x in hits if ref_of(x) is r])
                if isinstance(r, _NaiveLFU):
                    r.warm(hits + miss)
            used = live & grp
            live -= used
            pol.release(used)
            had_live = bool(live)
            for key in miss:
                v, w = pol.admit(key, 0, grp), ref_of(key).admit(key, grp)
                assert v == w, (trial, name, "admit", key, v, w)
                if v == REJECT and had_live:
                    v, w = pol.admit_last_resort(key, 0, grp), ref_of(key).admit(key, grp, last_resort=True)
                    assert v == w, (trial, name, "last resort", key, v, w)
                    n_last += v >= 0
                if v != REJECT:
                    res.add(key)
                    res.discard(v)
                    live.discard(v)
            if hasattr(pol, "flush"):
                pol.flush()
            if rng.random() < 0.7:                                      # prefetch for some layer
                ly2 = int(rng.integers(0, L))
                pred = set((ly2 * E + rng.choice(E, size=int(rng.integers(1, E)), replace=False)).tolist())
                prot2 = grp | pred
                for key in sorted(pred - res):
                    v, w = pol.admit(key, 0, prot2), ref_of(key).admit(key, prot2)
                    assert v == w, (trial, name, "prefetch", key, v, w)
                    if v == REJECT:
                        n_rej += 1
                        continue
                    pol.hold((key,))
                    ref_of(key).held.add(key)
                    res.add(key)
                    res.discard(v)
                    live.add(key)
                if hasattr(pol, "flush"):
                    pol.flush()
            if rng.random() < 0.35:                                     # the forward step ends
                pol.release(set(live))
                for r in refs:
                    r.held.clear()
                live.clear()
                pol.tick(1)
                for r in refs:
                    if isinstance(r, _NaiveLFU):
                        r.inc /= r.decay
    assert n_last > 100 and n_rej > 100


def test_a_cache_full_of_held_prefetches_rejects_without_a_scan():
    from hearth.sim.cache import REJECT, _LFU, _LRU
    lru = _LRU(4, list(range(8)))
    for key in range(4):
        lru.admit(key, 0, set())
    lru.hold([0, 1, 2, 3])
    assert len(lru.od) == 0 and lru.admit(5, 0, set()) == REJECT     # nothing to walk past
    assert lru.admit_last_resort(5, 0, set()) == 0                   # oldest held slab
    lfu = _LFU(3, list(range(8)), 8, samples=0)
    for key in range(3):
        lfu.admit(key, 0, set())
    lfu.flush()
    lfu.hold([0, 1, 2])
    assert lfu.admit(5, 0, set()) == REJECT and lfu.heap == []      # held slabs are not in the eviction heap
    # the last resort takes the coldest held slab not in prot, also after a heap rebuild
    big = _LFU(6, list(range(8)), 8, samples=0)
    for key, h in zip(range(6), (5.0, 3.0, 9.0, 1.0, 7.0, 2.0)):
        big.admit(key, 0, set())
        big.heat[key] = h
    big.flush()
    big.hold([0, 1, 2, 3, 4, 5])
    big._rebuild()
    assert big.admit_last_resort(6, 0, {3}) == 5                      # 3 is the coldest but protected
    assert big.admit_last_resort(7, 0, set()) == 3                    # ...and still a candidate afterwards
    assert big.admit_last_resort(5, 0, set()) == 1
    # a repeated REJECT under the same protection is answered from the memo, and draws the same samples
    s1, s2 = _LFU(2, [0, 1, 2], 3, samples=4, seed=9), _LFU(2, [0, 1, 2], 3, samples=4, seed=9)
    for p in (s1, s2):
        p.admit(0, 0, set())
        p.admit(1, 0, set())
    prot = {0, 1}
    assert s1.admit(2, 0, prot) == REJECT and s1.admit(2, 0, prot) == REJECT
    assert s2._victim(prot) == -1 and s2._victim(prot) == -1
    assert s1.rng.random() == s2.rng.random()


def test_prefetch_spec_forms_and_warning_boundaries():
    assert small(tokens=60, cache_gb=0.3, prefetch=0.8).settings["prefetch"] == {"recall": 0.8, "extra": 0}
    extras_only = small(tokens=60, cache_gb=0.3, prefetch=Prefetch(0.0, 2))           # wrong guesses only
    assert extras_only.settings["prefetch"] == {"recall": 0.0, "extra": 2}
    assert extras_only.bytes_per_token["nvme_prefetch_wasted"] > 0
    assert small(tokens=60, cache_gb=0.3, prefetch=Prefetch(0.0, 0)).settings["prefetch"] is None
    # "noisy" below 200 measured tokens, not at 200
    assert any("only 199 measured" in w for w in small(tokens=221, warmup=22, cache_gb=0.3).warnings)
    assert not any("noisy" in w for w in small(tokens=222, warmup=22, cache_gb=0.3).warnings)
    # a model that does not fit gets its own reasons, not a cache warning on top
    lap = simulate("kimi-k2", "laptop-16gb", tokens=100, cache_gb=0.0)
    assert not lap.feasible and not any(w.startswith("INFEASIBLE: cache") for w in lap.warnings)
    assert any(w.startswith("INFEASIBLE: only") for w in lap.warnings)


def test_warmup_must_leave_a_measured_step():
    # Spec(4, 1.0) on 10 tokens: steps start at 0 and 5; nothing starts at or after token 9
    with pytest.raises(ValueError, match="no forward step"):
        simulate(SMALL, tokens=10, spec=Spec(4, 1.0), warmup=9, cache_gb=0.3)
    r = simulate(SMALL, tokens=30, spec=Spec(4, 1.0), warmup=12, cache_gb=0.3)
    assert r.tokens == 15 and r.steps == 3 and r.tok_s > 0
    assert any("only 15 measured tokens" in w for w in r.warnings)
    assert main(["--model", SMALL, "--tokens", "10", "--spec-k", "4", "--spec-alpha", "1", "--warmup", "9"]) == 2


def test_integral_floats_are_normalised():
    a = simulate(SMALL, tokens=60, spec=Spec(4.0, 0.6), cache_gb=0.3)
    b = simulate(SMALL, tokens=60, spec=Spec(4, 0.6), cache_gb=0.3)
    assert a.tok_s == b.tok_s and type(Spec(4.0).k) is int and a.settings["spec"]["k"] == 4
    r = simulate(SMALL, tokens=60, io_threads=8.0, cache_gb=0.0)
    assert r.slots == 26 and type(r.slots) is int and type(r.settings["io_threads"]) is int
    assert type(simulate(SMALL, tokens=60, lfu_samples=4.0, cache_gb=0.3).settings["lfu_samples"]) is int


def test_calibrate_rejects_bad_measurements():
    r = simulate(SMALL, tokens=60, cache_gb=0.3)
    for bad in (0.0, -2.0, float("nan"), float("inf"), "3", True, None):
        with pytest.raises(ValueError, match="point 0"):
            calibrate([(r, bad)])
    with pytest.raises(ValueError, match="point 1"):
        calibrate([(r, 1.0), (r, 0.0)])
    with pytest.raises(ValueError, match="point 0"):
        calibrate([r])
    cal, rep = calibrate([(r, np.float32(r.tok_s))], fit=("io_eff",))          # numpy scalars are fine
    assert rep["rms_log_error"] < 1e-3


def test_trace_validates_n_layers(tmp_path):
    with pytest.raises(ValueError):
        Trace(np.array([[[3, 3]]] * 4, np.uint16), 8)                   # duplicate in a one-layer top-2 trace
    ids = np.zeros((3, 5, 1), np.uint16)
    for bad in (2, 4, -1, 2 ** 32, 6.5):
        with pytest.raises(ValueError):
            Trace(ids, 4, n_layers=bad)
    assert Trace(ids, 4, n_layers=0).n_layers == 5 and Trace(ids, 4, n_layers=5.0).n_layers == 5
    big = Trace(ids, 4, n_layers=2 ** 32 - 1)
    assert Trace.load(big.save(tmp_path / "b.hrtr")).n_layers == 2 ** 32 - 1


def test_cache_beyond_free_ram_is_infeasible(capsys):
    ok = simulate("kimi-k2", "this-pc", tokens=100, cache_gb=40.0)
    over = simulate("kimi-k2", "this-pc", tokens=100, cache_gb=200.0)
    free = over.feasibility.ram_free_for_cache_gib
    assert ok.feasible and ok.cache_gib <= free
    assert over.feasibility.fits and over.cache_gib > free and not over.feasible
    assert any(w.startswith("INFEASIBLE: cache") for w in over.warnings)
    assert simulate("kimi-k2", "this-pc", tokens=100).feasible                # default: all free RAM
    assert main(["--model", "kimi-k2", "--tokens", "100", "--sweep-cache", "40,200"]) == 0
    rows = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(("40GiB", "200GiB"))]
    assert [("INFEASIBLE" in ln) for ln in rows] == [False, True], rows


# ---- H04: engine parity, profiles, CLI and input checks (council round 1, T04 minor findings) --------

def test_io_threads_clamped_to_64_like_store_c():
    assert engine_min_slots(8, 64) == 2 * 8 + 64 + 2 == engine_min_slots(8, 65) == engine_min_slots(8, 10 ** 6)
    assert engine_min_slots(8, 63) == 2 * 8 + 63 + 2 and engine_min_slots(1, 1) == 5
    with pytest.raises(ValueError):
        engine_min_slots(8, 0)
    r = small(tokens=60, cache_gb=0.0, io_threads=100)
    assert r.slots == 2 * 8 + 64 + 2
    assert any(w.startswith("io_threads 100 > 64: the engine clamps it to 64") for w in r.warnings)
    assert not any("clamps" in w for w in small(tokens=60, cache_gb=0.0, io_threads=64).warnings)
    assert small(tokens=60, cache_gb=0.0, io_threads=64).slots == r.slots


def test_lfu_pinned_heat_is_seeded_like_the_engine(tmp_path):
    """store.c seeds LFU heat with count / tokens_observed / (1 - decay); raw counts recorded over 6000 tokens
    would dwarf the live heat (the council's case: hit rate 0.487 simulated vs 0.611 engine-like)."""
    from hearth.sim.core import lfu_seed_scale
    assert lfu_seed_scale(6000, 0.995) == 1.0 / 6000 / (1.0 - 0.995)
    assert lfu_seed_scale(0, 0.995) == 1.0 and lfu_seed_scale(6000, 1.0) == 1.0
    shape = presets.get(SMALL)
    other = synthetic_for(shape, 6000, zipf=1.2, reuse=0.2, seed=9).frequencies().astype(np.float32)
    path = save_usage(tmp_path / "o.usage", other, tokens_observed=6000)
    tr = synthetic_for(shape, 600, zipf=1.2, reuse=0.2, seed=3)
    sched = build_schedule(tr.ids, 64)
    counts = other.ravel().astype(np.float64)
    for pf, prof in ((0.0, str(path)), (0.5, path)):                  # a str or a Path
        r = simulate(SMALL, trace=tr, policy="lfu-pinned", cache_gb=0.4, profile=prof, pin_fraction=pf)
        engine = run_cache(sched, "lfu-pinned", r.slots, profile=counts * lfu_seed_scale(6000, 0.995), pin_fraction=pf)
        raw = run_cache(sched, "lfu-pinned", r.slots, profile=counts, pin_fraction=pf)
        assert r._counts.hit.tolist() == engine.hit.tolist(), pf
        assert r.settings["profile"] == str(path)
        assert int(engine.hit.sum()) > 1.5 * int(raw.hit.sum()), pf
    # a profile without a token count (an array) seeds raw values; warm-up and oracle profiles their window
    r = simulate(SMALL, trace=tr, policy="lfu-pinned", cache_gb=0.4, profile=counts)
    assert r._counts.hit.tolist() == run_cache(sched, "lfu-pinned", r.slots, profile=counts).hit.tolist()
    warm = tr.ids[:60]
    wc = np.bincount((warm.astype(np.int64) + (np.arange(16) * 64)[None, :, None]).ravel(), minlength=1024)
    r = simulate(SMALL, trace=tr, policy="lfu-pinned", cache_gb=0.4, warmup=60, lfu_decay=0.98)
    seed = wc * lfu_seed_scale(60, 0.98)
    assert r._counts.hit.tolist() == run_cache(sched, "lfu-pinned", r.slots, profile=seed, lfu_decay=0.98).hit.tolist()
    r = simulate(SMALL, trace=tr, policy="lfu-pinned", cache_gb=0.4, profile="oracle", lfu_decay=0.98)
    whole = tr.frequencies().ravel() * lfu_seed_scale(600, 0.98)
    assert r._counts.hit.tolist() == run_cache(sched, "lfu-pinned", r.slots, profile=whole, lfu_decay=0.98).hit.tolist()


def test_profiles_must_be_finite_and_non_negative(tmp_path):
    shape = presets.get(SMALL)
    good = np.ones((shape.n_layers, shape.n_experts), np.float32)
    for bad in (np.nan, -1.0, np.inf, -1e30):
        h = good.copy()
        h[3, 7] = bad
        p = save_usage(tmp_path / "bad.usage", h, tokens_observed=100)
        for pol in ("pinned", "lfu-pinned"):
            with pytest.raises(ValueError, match="heat entry 199 is not a finite non-negative number"):
                small(tokens=60, policy=pol, cache_gb=0.3, profile=str(p))
        with pytest.raises(ValueError, match="heat entry 199"):
            small(tokens=60, policy="pinned", cache_gb=0.3, profile=h.ravel())
    with pytest.raises(ValueError, match="heat entry 0"):
        small(tokens=60, cache_gb=0.3, lossy=Lossy(cold_bits=2.25), profile=np.full(1024, np.nan))
    assert small(tokens=60, policy="pinned", cache_gb=0.3, profile=np.zeros(1024)).n_pinned > 0   # zeros are fine
    # profiles are only read when something uses them
    assert small(tokens=60, policy="lru", cache_gb=0.3, profile=np.full(1024, np.nan)).hit_rate >= 0
    assert main(["--model", SMALL, "--tokens", "60", "--policy", "pinned", "--profile",
                 str(tmp_path / "bad.usage")]) == 2


def test_oracle_and_array_profiles_pin_the_hottest_slabs():
    tr = synthetic_for(presets.get(SMALL), 600, zipf=1.2, reuse=0.0, seed=3)
    freq = tr.frequencies().ravel().astype(np.float64)
    oracle = small(policy="pinned", cache_gb=0.5, profile="oracle")
    assert any("oracle profile" in w for w in oracle.warnings) and oracle.settings["profile"] == "oracle"
    arr = small(policy="pinned", cache_gb=0.5, profile=freq)
    assert arr.settings["profile"] == "array" and not any("oracle" in w for w in arr.warnings)
    assert arr._counts.hit.tolist() == oracle._counts.hit.tolist()            # same heat, same pins
    sched = build_schedule(tr.ids, 64)
    assert oracle._counts.hit.tolist() == run_cache(sched, "pinned", oracle.slots, profile=freq).hit.tolist()
    warm = small(policy="pinned", cache_gb=0.5)
    assert oracle.hit_rate > warm.hit_rate                                   # it sees the future
    with pytest.raises(ValueError, match="entries"):
        small(policy="pinned", cache_gb=0.5, profile=freq[:-1])


def test_cli_empty_sweep_lists_exit_2(capsys):
    for opt in (["--sweep-cache", ","], ["--sweep-zipf", ","], ["--sweep-cache", " "]):
        assert main(["--model", SMALL, "--tokens", "60"] + opt) == 2, opt
        assert "hearth sim: error: empty list" in capsys.readouterr().err, opt


def test_cli_overrides_every_hardware_and_encoding_field(capsys):
    args = ["--model", SMALL, "--tokens", "60", "--cache-gb", "0.3", "--json", "--gpu-sync-us", "40",
            "--cpu-cores", "6", "--embed-bits", "4.25", "--max-seq", "1024", "--gpu", "dense"]
    assert main(args) == 0
    r = json.loads(capsys.readouterr().out)["result"]
    s = r["settings"]
    assert (s["hw"]["gpu_sync_us"], s["hw"]["cpu_cores"], s["embed_bits"], s["max_seq"]) == (40.0, 6, 4.25, 1024)
    want = simulate(SMALL, get_hardware("this-pc", gpu_sync_us=40.0, cpu_cores=6), tokens=60, cache_gb=0.3,
                    embed_bits=4.25, max_seq=1024, gpu="dense")
    assert r["tok_s"] == want.tok_s and r["feasibility"]["kv_gib"] == want.feasibility.kv_gib
    assert any("LOSSY embeddings" in x for x in r["lossy"])
    assert main(["--model", SMALL, "--tokens", "60", "--json", "--hw", "laptop-16gb", "--unified", "--vram-gbs",
                 "200", "--gpu", "dense"]) == 0
    s = json.loads(capsys.readouterr().out)["result"]["settings"]
    assert s["hw"]["unified"] is True and s["gpu"] == "dense"
    assert main(["--model", SMALL, "--tokens", "60", "--json", "--hw", "mac-studio-192gb", "--no-unified",
                 "--gpu", "dense"]) == 2                                   # no GPU left
    assert "has no GPU" in capsys.readouterr().err


def test_hardware_dicts_round_trip():
    r = small(tokens=60, cache_gb=0.3, hw_overrides={"nvme_count": 2, "io_cap_gbs": 9.0})
    hw = get_hardware(r.settings["hw"])
    assert hw == r._hw and hw.io_gbs == 9.0
    again = simulate(SMALL, r.settings["hw"], tokens=600, zipf=1.2, reuse=0.0, seed=3, cache_gb=0.3)
    assert again.tok_s == simulate(SMALL, r._hw, tokens=600, zipf=1.2, reuse=0.0, seed=3, cache_gb=0.3).tok_s
    with pytest.raises(ValueError, match="io_gbs"):
        get_hardware({**r.settings["hw"], "io_gbs": 13.0})
    with pytest.raises(KeyError, match="warp_drive"):
        get_hardware({**r.settings["hw"], "warp_drive": 1})


@pytest.mark.parametrize("make", [
    lambda: Spec(math.inf, 0.5), lambda: Spec(math.nan, 0.5), lambda: Spec(2.5, 0.5), lambda: Spec(True, 0.5),
    lambda: Spec(2, math.nan), lambda: Spec(2, "0.5"), lambda: Prefetch(0.5, math.inf), lambda: Prefetch(0.5, 1.5),
    lambda: Prefetch(math.nan, 0), lambda: Prefetch("0.5", 0), lambda: Lossy(topk=6.5), lambda: Lossy(topk=math.inf),
    lambda: Lossy(topk=0), lambda: Lossy(skip_miss_rank=math.nan), lambda: Lossy(cold_bits="3"),
    lambda: Lossy(cold_frac=None)])
def test_feature_specs_reject_non_finite_and_fractional_values(make):
    with pytest.raises(ValueError):
        make()


def test_feature_specs_normalise_integral_floats():
    assert Lossy(topk=6.0).topk == 6 and type(Lossy(topk=6.0).topk) is int
    assert type(Lossy(skip_miss_rank=np.int64(5)).skip_miss_rank) is int
    assert Prefetch(1, 2.0).extra == 2 and type(Prefetch(1, 2.0).extra) is int and Prefetch(1, 0).recall == 1.0
    assert small(tokens=60, cache_gb=0.3, lossy=Lossy(topk=6.0)).loads_per_token == 16 * 6     # was a TypeError
    with pytest.raises(ValueError):
        small(tokens=60, warmup=math.inf)


def test_gpu_expert_tier_decides_the_step_exactly():
    """dense+experts on a discrete GPU with slow VRAM: the GPU's expert tier, not the CPU, sets T_moe."""
    shape = dataclasses.replace(presets.get(SMALL), name="one-layer", n_layers=1)
    k = shape.top_k
    sched = build_schedule(np.arange(k, dtype=np.uint16).reshape(1, 1, k), shape.n_experts)
    costs = model_costs(shape, context=0)
    hw = get_hardware("this-pc", vram_gbs=2.0)
    cal = Calibration(overhead_ms=0.0, layer_overhead_us=0.0)
    v, h = 6, 2
    tm = evaluate(_counts(hit=[h], vram=[v]), sched, costs, hw, cal, "dense+experts")
    S, vbw = costs.slab, 2e9
    sync = 2 * (15e-6 + 4.0 * shape.d_model / 50e9)
    c = max(S / 60e9, 2.0 * costs.expert_params / 5e12)
    t_gpu, t_cpu = v * S / vbw + sync, h * c
    assert t_gpu > t_cpu
    bb = (costs.global_bytes + costs.layer_bytes + costs.shared_bytes) / vbw
    assert math.isclose(float(tm.step_s[0]), bb + sync + t_gpu, rel_tol=1e-12)
    assert math.isclose(float(tm.parts[0, COMPONENTS.index("gpu")]), bb + sync + t_gpu - t_cpu, rel_tol=1e-12)
    assert math.isclose(float(tm.vram_bytes[0]), v * S + costs.global_bytes + costs.layer_bytes + costs.shared_bytes)
    assert float(tm.dram_bytes[0]) == h * S
    cpu_only = evaluate(_counts(hit=[h], vram=[0]), sched, costs, hw, cal, "dense+experts")
    assert math.isclose(float(cpu_only.step_s[0]), bb + sync + t_cpu, rel_tol=1e-12)


def test_ranks_beyond_int16_are_kept():
    """Schedule.minrank is int32: a 1-layer top-40000 trace (valid geometry) must not wrap ranks >= 32768."""
    k, E = 40_000, 40_001
    perm = np.random.default_rng(0).permutation(E)[:k].astype(np.uint16)
    ids = np.tile(perm, (3, 1)).reshape(3, 1, k)
    for spec in ((0, 0.0), (1, 1.0)):
        sched = build_schedule(ids, E, *spec)
        assert sched.minrank.dtype == np.int32 and int(sched.minrank.max()) == k - 1
        c = run_cache(sched, "lru", 10, skip_rank=1)
        # only the rank-0 expert is ever loaded: missed once, then a hit in every later step
        assert int(c.skip.sum()) == sched.n_steps * (k - 1)
        assert (int(c.miss.sum()), int(c.hit.sum())) == (1, sched.n_steps - 1)


def test_synthetic_refill_rows_keep_reused_experts():
    """Each previous expert is kept with probability `reuse`, so P(e in token t | e in token t-1) >= reuse for
    every expert, also when the row needs the exact refill (extreme skew, many rows). Expert 0 is one of the
    rarest here (rank 8 of 12): dropping it from refilled rows (`> 0` for `>= 0`) gave 0.35."""
    E, k, reuse, seed = 12, 6, 0.5, 6
    tr = synthetic(12000, 1, E, k, zipf=3.0, reuse=reuse, seed=seed)
    p = zipf_popularity(E, 1, 3.0, np.random.default_rng(seed))[0]
    assert int((p > p[0]).sum()) == 8
    inn = np.stack([(tr.ids[:, 0, :] == e).any(-1) for e in range(E)])        # [E, T]
    prev, nxt = inn[:, :-1], inn[:, 1:]
    n = prev.sum(axis=1)
    stay = (prev & nxt).sum(axis=1) / np.maximum(n, 1)
    assert n[0] > 1500
    assert (stay[n > 500] > reuse - 0.03).all(), stay


def test_simulate_boundaries_and_default_warmup():
    """Boundary values of simulate()'s own checks, and the documented default warm-up (10% of the trace)."""
    from hearth.sim.core import lfu_seed_scale
    r = small()
    assert r.warmup_tokens == 60 == r.settings["warmup"] and small(tokens=599).warmup_tokens == 59
    for kw in (dict(lfu_decay=1.0), dict(pin_fraction=0.9, policy="lfu-pinned"), dict(lossy=Lossy(topk=1)),
               dict(lossy=Lossy(skip_miss_rank=1))):
        assert small(tokens=60, cache_gb=0.3, **kw).tokens > 0, kw            # accepted
    assert small(tokens=60, cache_gb=0.3, lossy=Lossy(topk=1)).loads_per_token == 16
    with pytest.raises(ValueError, match="lfu_decay"):
        small(tokens=60, policy="lru", lfu_decay=0.0)                          # also when no LFU runs
    with pytest.raises(ValueError, match="io_threads must be an integer >= 1"):
        small(tokens=60, io_threads=0)
    assert lfu_seed_scale(1, 0.5) == 2.0
    # a one-token warm-up is the profile's source; only warm-up 0 falls back to the whole trace
    assert not any("WHOLE trace" in w for w in small(tokens=60, policy="pinned", cache_gb=0.3, warmup=1).warnings)


def test_rates_with_bypasses_and_one_load_windows():
    """bypass_rate = bypassed / demand misses; tokens_per_step; the rates of windows small enough to do by hand."""
    shape = dataclasses.replace(presets.get(SMALL), name="tiny-bypass", n_layers=1, n_experts=6, top_k=1)
    ids = np.array([0, 1, 2, 3, 4, 5] * 2, dtype=np.uint16).reshape(12, 1, 1)
    # windows of 6 tokens, 5 slots (engine minimum at top-1, one I/O thread): in each window the 6th expert finds
    # every slot held by the window's own experts and is read into a scratch buffer
    r = simulate(shape, trace=Trace(ids, 6), spec=Spec(5, 1.0), cache_gb=0.0, io_threads=1, warmup=6)
    assert (r.slots, r.steps, r.tokens, r.tokens_per_step, r.loads_per_token) == (5, 1, 6, 6.0, 1.0)
    assert (r.hit_rate, r.miss_rate, r.bypass_rate) == (5 / 6, 1 / 6, 1.0)
    one = simulate(shape, trace=Trace(ids[:2], 6), cache_gb=0.0, io_threads=1, warmup=1)   # a single load
    assert (one.tokens, one.loads_per_token, one.miss_rate, one.hit_rate, one.bypass_rate) == (1, 1.0, 1.0, 0.0, 0.0)
    # many misses, some bypassed: olmoe verification windows wider than the engine-minimum cache
    big = small(cache_gb=0.0, spec=Spec(4, 0.9))
    c, g = big._counts, np.repeat(big._sched.step_start >= big.warmup_tokens, 16)
    byp, miss = int(c.bypass[g].sum()), int(c.miss[g].sum())
    assert 0 < byp < miss and big.bypass_rate == byp / miss


def test_public_defaults():
    """The documented defaults of the cache-level API: 8 I/O threads, no prefetch extras, seed 0 and measuring
    from the first step."""
    assert engine_min_slots(8) == engine_min_slots(8, 8) == 26
    assert Prefetch(0.8) == Prefetch(0.8, 0)
    sched = build_schedule(synthetic(40, 2, 8, 2, seed=1).ids, 8)
    pf = Prefetch(0.5, 2)
    assert run_cache(sched, "lfu", 5, prefetch=pf).pf_issued.tolist() == \
        run_cache(sched, "lfu", 5, prefetch=pf, seed=0).pf_issued.tolist()
    assert run_cache(sched, "belady", 5).hit.tolist() == run_cache(sched, "belady", 5, measure_step=0).hit.tolist()
