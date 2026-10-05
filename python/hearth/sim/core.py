"""simulate / sweep / retime / levers - the public entry points of hearth.sim."""
from __future__ import annotations

import itertools
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .. import presets
from . import feasibility as feas_mod
from .cache import (ONLINE_POLICIES, POLICIES, CacheCounts, Prefetch, Schedule, build_schedule,
                    engine_min_slots, profile_order, run_cache)
from .hardware import Hardware, get_hardware
from .timing import COMPONENTS, GPU_MODES, Calibration, ModelCosts, Timing, evaluate, model_costs
from .trace import Trace, load_usage, synthetic_for

GIB = float(1 << 30)
# converter defaults (PYTHON_API convert(): Q4 experts, Q8 dense/embedding/head); fewer bits change output
DEFAULT_BITS = {"expert_bits": 4.25, "dense_bits": 8.25, "embed_bits": 8.25}


def bits_labels(**bits) -> list:
    """LOSSY labels for encodings below the converter defaults (INV-LOSSLESS-DEFAULT)."""
    names = {"expert_bits": "experts", "dense_bits": "dense backbone", "embed_bits": "embeddings"}
    return [f"LOSSY {names[k]} at {v:g} bpw (converter default {DEFAULT_BITS[k]:g})"
            for k, v in bits.items() if v < DEFAULT_BITS[k]]


@dataclass(frozen=True)
class Spec:
    """Speculative decoding: k drafted tokens per step, each accepted with probability alpha
    (geometric: the first rejection ends the run). The step verifies k+1 positions in one batch."""
    k: int = 4
    alpha: float = 0.6

    def __post_init__(self):
        if int(self.k) != self.k or self.k < 0:
            raise ValueError(f"speculative k must be an integer >= 0, got {self.k}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"speculative alpha must be in [0, 1], got {self.alpha}")
        object.__setattr__(self, "k", int(self.k))          # 4.0 -> 4: k is used as a slice bound
        object.__setattr__(self, "alpha", float(self.alpha))

    def expected_tokens(self) -> float:
        a = self.alpha
        return float(self.k + 1) if a >= 1.0 else (1 - a ** (self.k + 1)) / (1 - a)


@dataclass(frozen=True)
class Lossy:
    """What-if knobs that CHANGE MODEL OUTPUT (INV-LOSSLESS-DEFAULT: off by default, never in the engine's
    default path). topk: keep only the first `topk` ranks. skip_miss_rank: experts routed only at ranks
    >= this are skipped when not already cached. cold_bits: the coldest `cold_frac` of experts (by the
    profile) are stored at cold_bits/weight (slots stay full-size, as in the engine)."""
    topk: int | None = None
    skip_miss_rank: int | None = None
    cold_bits: float | None = None
    cold_frac: float = 0.5

    def labels(self) -> list:
        out = []
        if self.topk is not None:
            out.append(f"LOSSY top-k reduced to {self.topk}")
        if self.skip_miss_rank is not None:
            out.append(f"LOSSY uncached experts at rank >= {self.skip_miss_rank} skipped")
        if self.cold_bits is not None:
            out.append(f"LOSSY coldest {self.cold_frac:.0%} of experts at {self.cold_bits} bpw")
        return out


@dataclass(frozen=True)
class TraceSpec:
    tokens: int = 2000
    zipf: float = 1.1
    reuse: float = 0.25
    seed: int = 0


@dataclass
class Result:
    model: str
    hardware: str
    policy: str
    cache_gib: float
    slots: int
    slab_bytes: int
    tokens: int                 # accepted tokens measured (after warm-up)
    steps: int
    warmup_tokens: int
    tok_s: float
    ms_per_token: float
    hit_rate: float             # needed expert loads served without any storage read (DRAM + VRAM)
    prefetch_rate: float        # ... whose storage read was a prefetch
    miss_rate: float            # ... that needed a demand read
    skip_rate: float            # ... skipped (lossy)
    vram_rate: float            # part of hit_rate served from the VRAM tier
    bypass_rate: float          # demand reads that could not get a slot
    late_prefetch_rate: float   # prefetched loads that had not arrived when needed
    loads_per_token: float      # distinct expert loads per accepted token
    tokens_per_step: float
    bytes_per_token: dict       # tier -> bytes per accepted token
    time_per_token_ms: dict     # component -> ms per accepted token
    bottleneck: str
    feasible: bool
    warnings: list
    lossy: list
    settings: dict
    feasibility: feas_mod.Feasibility
    calibration: Calibration
    trace_source: str
    n_pinned: int = 0
    _counts: CacheCounts | None = field(default=None, repr=False)
    _sched: Schedule | None = field(default=None, repr=False)
    _costs: ModelCosts | None = field(default=None, repr=False)
    _hw: Hardware | None = field(default=None, repr=False)

    def label(self) -> str:
        s = self.settings
        bits = [self.model, self.hardware, self.policy, f"{self.cache_gib:.1f}GiB"]
        if s.get("prefetch"):
            bits.append(f"pf{s['prefetch']['recall']}+{s['prefetch']['extra']}")
        if s.get("spec"):
            bits.append(f"spec{s['spec']['k']}@{s['spec']['alpha']}")
        if s.get("gpu", "off") != "off":
            bits.append("gpu:" + s["gpu"])
        if self.lossy:
            bits.append("LOSSY")
        return " ".join(bits)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        d["feasibility"] = self.feasibility.to_dict()
        d["calibration"] = self.calibration.to_dict()
        return d


# ---- trace resolution (with a tiny cache so sweeps reuse synthetic traces) ----------

_TRACE_CACHE: "OrderedDict[tuple, Trace]" = OrderedDict()


def _synthetic_cached(shape, ts: TraceSpec) -> Trace:
    key = (shape.n_moe_layers, shape.n_experts, shape.top_k, shape.n_layers, ts.tokens, ts.zipf, ts.reuse, ts.seed)
    tr = _TRACE_CACHE.get(key)
    if tr is None:
        tr = synthetic_for(shape, ts.tokens, zipf=ts.zipf, reuse=ts.reuse, seed=ts.seed)
        _TRACE_CACHE[key] = tr
        while len(_TRACE_CACHE) > 6:
            _TRACE_CACHE.popitem(last=False)
    return tr


def resolve_trace(shape, trace=None, *, tokens=2000, zipf=1.1, reuse=0.25, seed=0) -> Trace:
    if isinstance(trace, Trace):
        tr = trace
    elif isinstance(trace, (str, Path)):
        tr = Trace.load(trace)
    elif isinstance(trace, TraceSpec):
        tr = _synthetic_cached(shape, trace)
    elif isinstance(trace, dict):
        tr = _synthetic_cached(shape, TraceSpec(**trace))
    elif trace is None:
        tr = _synthetic_cached(shape, TraceSpec(tokens, zipf, reuse, seed))
    else:
        raise TypeError(f"trace must be a Trace, a path, a TraceSpec/dict or None, not {type(trace).__name__}")
    if (tr.n_moe_layers, tr.n_experts, tr.top_k) != (shape.n_moe_layers, shape.n_experts, shape.top_k):
        raise ValueError(f"trace geometry (moe_layers={tr.n_moe_layers}, experts={tr.n_experts}, top_k={tr.top_k}) "
                         f"does not match {shape.name} ({shape.n_moe_layers}, {shape.n_experts}, {shape.top_k})")
    return tr


def _resolve_profile(profile, shape, ids: np.ndarray, warmup: int, warnings: list) -> np.ndarray:
    L, E = shape.n_moe_layers, shape.n_experts
    offs = (np.arange(L, dtype=np.int64) * E)[None, :, None]

    def heat_of(sl):
        return np.bincount((ids[sl].astype(np.int64) + offs).ravel(), minlength=L * E).astype(np.float64)

    if profile is None or (isinstance(profile, str) and profile == "warmup"):
        if warmup > 0:
            return heat_of(slice(0, warmup))
        warnings.append("profile from the WHOLE trace (warmup=0): pinned/VRAM/cold choices see the future")
        return heat_of(slice(None))
    if isinstance(profile, str) and profile == "oracle":
        warnings.append("oracle profile: pinned/VRAM/cold choices see the whole trace (upper bound)")
        return heat_of(slice(None))
    if isinstance(profile, (str, Path)):
        heat, _ = load_usage(profile)
        if heat.shape == (shape.n_layers, E):
            heat = heat[shape.n_dense_layers:]
        if heat.shape != (L, E):
            raise ValueError(f"usage profile shape {heat.shape} does not match {shape.name} ({L}, {E})")
        return heat.ravel().astype(np.float64)
    arr = np.asarray(profile, dtype=np.float64)
    if arr.size != L * E:
        raise ValueError(f"profile must have {L}*{E} entries, got {arr.size}")
    return arr.ravel()


def _as_prefetch(p) -> Prefetch | None:
    if p is None or p is False:
        return None
    if isinstance(p, Prefetch):
        return p if (p.recall > 0 or p.extra > 0) else None
    if isinstance(p, (int, float)):
        return _as_prefetch(Prefetch(float(p), 0))
    if isinstance(p, dict):
        return _as_prefetch(Prefetch(**p))
    if isinstance(p, (tuple, list)):
        return _as_prefetch(Prefetch(*p))
    raise TypeError(f"bad prefetch spec {p!r}")


def _as_spec(s) -> Spec | None:
    if s is None or s is False:
        return None
    if isinstance(s, Spec):
        return s if s.k > 0 else None
    if isinstance(s, dict):
        return _as_spec(Spec(**s))
    if isinstance(s, (tuple, list)):
        return _as_spec(Spec(*s))
    raise TypeError(f"bad spec {s!r}")


def _as_lossy(x) -> Lossy:
    if x is None:
        return Lossy()
    if isinstance(x, Lossy):
        return x
    if isinstance(x, dict):
        return Lossy(**x)
    raise TypeError(f"bad lossy spec {x!r}")


# ---- simulate ---------------------------------------------------------------------

def simulate(model="kimi-k2", hardware="this-pc", trace=None, *, policy: str = "lfu", cache_gb: float | None = None,
             prefetch=None, spec=None, lossy=None, gpu: str = "off", warmup: int | None = None, profile="warmup",
             expert_bits: float = 4.25, dense_bits: float = 8.25, embed_bits: float = 8.25, context: int = 1024,
             max_seq: int = 4096, kv_elem_bytes: float = 4.0, io_threads: int = 8, lfu_decay: float = 0.995,
             lfu_samples: int = 0, pin_fraction: float = 0.5, calib: Calibration | None = None, seed: int = 0,
             tokens: int = 2000, zipf: float = 1.1, reuse: float = 0.25, hw_overrides: dict | None = None) -> Result:
    """Simulate decoding `model` on `hardware` over a routing trace.

    trace: a Trace, a .hrtr path, a TraceSpec/dict, or None (synthetic from tokens/zipf/reuse/seed).
    cache_gb: DRAM expert cache in GiB (None = all RAM left after backbone, KV and OS reserve).
    prefetch: None/0 (off), a recall in (0, 1], Prefetch(recall, extra) or a dict.
    spec: None, Spec(k, alpha), (k, alpha) or a dict.  gpu: "off" | "dense" | "dense+experts" (roadmap R01).
    warmup: leading trace tokens excluded from statistics (default 10%); also the heat profile source
    for the pinned policies, the VRAM tier and cold experts unless `profile` is given.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; known: {', '.join(POLICIES)}")
    if gpu not in GPU_MODES:
        raise ValueError(f"gpu must be one of {GPU_MODES}")
    if cache_gb is not None and not 0.0 <= cache_gb < float("inf"):
        raise ValueError(f"cache_gb must be >= 0 (or None for all free RAM), got {cache_gb}")
    if int(io_threads) != io_threads or io_threads < 1:
        raise ValueError(f"io_threads must be an integer >= 1, got {io_threads}")
    if not 0.0 < lfu_decay <= 1.0:
        raise ValueError(f"lfu_decay must be in (0, 1], got {lfu_decay}")
    if int(lfu_samples) != lfu_samples or lfu_samples < 0:
        raise ValueError(f"lfu_samples must be an integer >= 0, got {lfu_samples}")
    if not 0.0 <= pin_fraction <= 0.9:
        raise ValueError(f"pin_fraction must be in [0, 0.9] (hx_store_opts), got {pin_fraction}")
    if warmup is not None and (int(warmup) != warmup or warmup < 0):
        raise ValueError(f"warmup must be an integer >= 0, got {warmup}")
    io_threads, lfu_samples = int(io_threads), int(lfu_samples)
    shape = presets.get(model)
    hw = get_hardware(hardware, **(hw_overrides or {}))
    if gpu != "off" and not hw.has_gpu:
        raise ValueError(f"gpu={gpu!r} but hardware {hw.name} has no GPU")
    calib = calib or Calibration()
    pf = _as_prefetch(prefetch)
    sp = _as_spec(spec)
    lo = _as_lossy(lossy)
    warnings: list = []
    tr = resolve_trace(shape, trace, tokens=tokens, zipf=zipf, reuse=reuse, seed=seed)
    T = tr.n_tokens
    if T < 2:
        raise ValueError("trace needs at least 2 tokens")
    W = T // 10 if warmup is None else int(warmup)
    if not 0 <= W < T:
        raise ValueError(f"warmup must be in [0, {T})")
    ids = tr.ids
    k_eff = shape.top_k
    if lo.topk is not None:
        if not 1 <= lo.topk <= shape.top_k:
            raise ValueError(f"lossy topk must be in [1, {shape.top_k}]")
        k_eff = lo.topk
        ids = ids[:, :, :k_eff]
    if lo.skip_miss_rank is not None and not 1 <= lo.skip_miss_rank < k_eff:
        raise ValueError(f"skip_miss_rank must be in [1, {k_eff})")
    if lo.cold_bits is not None and not 0.0 <= lo.cold_frac <= 1.0:
        raise ValueError("cold_frac must be in [0, 1]")

    costs = model_costs(shape, expert_bits=expert_bits, dense_bits=dense_bits, embed_bits=embed_bits,
                        cold_bits=lo.cold_bits, context=context, max_seq=max_seq, kv_elem_bytes=kv_elem_bytes)
    fz = feas_mod.check(costs, hw, gpu_mode=gpu, io_threads=io_threads, shape=shape)
    if not fz.fits:
        warnings.extend("INFEASIBLE: " + r for r in fz.reasons)
    min_slots = engine_min_slots(shape.top_k, io_threads)
    want_gib = fz.recommended_cache_gib if cache_gb is None else float(cache_gb)
    n_keys = shape.n_moe_layers * shape.n_experts
    slots = int(max(want_gib, 0.0) * GIB // costs.slab)
    if slots < min_slots:
        slots = min_slots
    if slots > n_keys and cache_gb is not None:
        warnings.append(f"cache {cache_gb:g} GiB requested; all {n_keys} expert slabs take "
                        f"{n_keys * costs.slab / GIB:.1f} GiB, so every expert is cached")
    slots = min(slots, n_keys)
    cache_fits = slots * costs.slab / GIB <= max(fz.ram_free_for_cache_gib, 0.0) + 1e-9
    if fz.fits and not cache_fits:
        warnings.append(f"INFEASIBLE: cache {slots * costs.slab / GIB:.1f} GiB exceeds the "
                        f"{max(fz.ram_free_for_cache_gib, 0):.1f} GiB of RAM left after backbone/KV/OS reserve")

    needs_profile = policy in ("pinned", "lfu-pinned") or lo.cold_bits is not None or \
        (gpu == "dense+experts" and fz.vram_expert_slots > 0)
    heat = _resolve_profile(profile, shape, ids, W, warnings) if needs_profile else None
    vram_keys = None
    if gpu == "dense+experts" and fz.vram_expert_slots > 0:
        vram_keys = profile_order(heat)[:fz.vram_expert_slots]
    cold_keys = None
    if lo.cold_bits is not None and lo.cold_frac > 0:
        n_cold = int(round(lo.cold_frac * n_keys))
        cold_keys = profile_order(heat)[n_keys - n_cold:] if n_cold else None

    sched = build_schedule(ids, shape.n_experts, sp.k if sp else 0, sp.alpha if sp else 0.0, seed)
    popularity = tr.frequencies() if pf is not None else None
    measure_step = int(np.searchsorted(sched.step_start, W, side="left"))
    if measure_step >= sched.n_steps:
        raise ValueError(f"no forward step starts at or after the {W}-token warm-up (trace of {T} tokens, "
                         f"last step starts at token {int(sched.step_start[-1])}): shorten the warm-up")
    n_measured = int(sched.step_nacc[measure_step:].sum())
    counts = run_cache(sched, policy, slots, profile=heat, vram_keys=vram_keys, cold_keys=cold_keys,
                       prefetch=pf, skip_rank=lo.skip_miss_rank, popularity=popularity, lfu_decay=lfu_decay,
                       lfu_samples=lfu_samples, pin_fraction=pin_fraction, io_threads=io_threads, seed=seed,
                       measure_step=measure_step)
    if policy == "belady":
        warnings.append("belady is an offline upper bound (knows the future, may bypass the cache and evict "
                        "in-use slabs, starts with the best possible cache contents)")
        if pf is not None or lo.skip_miss_rank is not None:
            warnings.append("belady with prefetch or skip_miss_rank is NOT an upper bound: it bounds demand-only "
                            "caching of the same request stream")
    settings = dict(model=shape.name, hardware=hw.name, policy=policy, cache_gb=cache_gb, gpu=gpu,
                    prefetch=asdict(pf) if pf else None, spec=asdict(sp) if sp else None, lossy=asdict(lo),
                    warmup=W, profile=profile if isinstance(profile, str) else "array",
                    expert_bits=expert_bits, dense_bits=dense_bits, embed_bits=embed_bits, context=context,
                    max_seq=max_seq, kv_elem_bytes=kv_elem_bytes, io_threads=io_threads, lfu_decay=lfu_decay,
                    lfu_samples=lfu_samples, pin_fraction=pin_fraction, seed=seed, trace_tokens=T,
                    hw=hw.to_dict(), cost_notes=list(costs.notes))
    if n_measured < 200:
        warnings.append(f"only {n_measured} measured tokens: hit rates are noisy")
    res = Result(model=shape.name, hardware=hw.name, policy=policy, cache_gib=slots * costs.slab / GIB, slots=slots,
                 slab_bytes=costs.slab, tokens=0, steps=0, warmup_tokens=W, tok_s=0.0, ms_per_token=0.0,
                 hit_rate=0.0, prefetch_rate=0.0, miss_rate=0.0, skip_rate=0.0, vram_rate=0.0, bypass_rate=0.0,
                 late_prefetch_rate=0.0, loads_per_token=0.0, tokens_per_step=0.0, bytes_per_token={},
                 time_per_token_ms={}, bottleneck="", feasible=fz.fits and cache_fits, warnings=warnings,
                 lossy=bits_labels(expert_bits=expert_bits, dense_bits=dense_bits, embed_bits=embed_bits)
                 + lo.labels(),
                 settings=settings, feasibility=fz, calibration=calib, trace_source=tr.source,
                 n_pinned=counts.n_pinned, _counts=counts, _sched=sched, _costs=costs, _hw=hw)
    return _aggregate(res, evaluate(counts, sched, costs, hw, calib, gpu))


def _aggregate(res: Result, tm: Timing) -> Result:
    sched, c = res._sched, res._counts
    L = sched.n_moe_layers
    mask = sched.step_start >= res.warmup_tokens
    g = np.repeat(mask, L)
    tokens = int(sched.step_nacc[mask].sum())
    steps = int(mask.sum())
    tsum = float(tm.step_s[mask].sum())
    hit, ph, miss = int(c.hit[g].sum()), int(c.pfhit[g].sum()), int(c.miss[g].sum())
    vram, skip, byp = int(c.vram[g].sum()), int(c.skip[g].sum()), int(c.bypass[g].sum())
    need = max(hit + ph + miss + vram + skip, 1)
    tok = max(tokens, 1)
    parts = tm.parts[mask].sum(axis=0) / tok * 1e3
    tpt = {name: float(v) for name, v in zip(COMPONENTS, parts)}
    nv_d = float(tm.nvme_demand_bytes[mask].sum()) / tok
    nv_p = float(tm.nvme_prefetch_bytes[mask].sum()) / tok
    return replace(
        res, tokens=tokens, steps=steps, tok_s=tokens / tsum if tsum > 0 else 0.0,
        ms_per_token=tsum / tok * 1e3, hit_rate=(hit + vram) / need, prefetch_rate=ph / need, miss_rate=miss / need,
        skip_rate=skip / need, vram_rate=vram / need, bypass_rate=byp / max(miss, 1),
        late_prefetch_rate=float(tm.late_prefetch[mask].sum()) / max(ph, 1), loads_per_token=need / tok,
        tokens_per_step=tokens / max(steps, 1),
        bytes_per_token={"nvme": nv_d + nv_p, "nvme_demand": nv_d, "nvme_prefetch": nv_p,
                         "nvme_prefetch_wasted": float(tm.nvme_wasted_bytes[mask].sum()) / tok,
                         "dram": float(tm.dram_bytes[mask].sum()) / tok,
                         "vram": float(tm.vram_bytes[mask].sum()) / tok},
        time_per_token_ms=tpt, bottleneck=max(tpt, key=tpt.get))


def retime(res: Result, calib: Calibration) -> Result:
    """Re-run only the timing model with different calibration scalars (cache counts are reused)."""
    tm = evaluate(res._counts, res._sched, res._costs, res._hw, calib, res.settings.get("gpu", "off"))
    return _aggregate(replace(res, calibration=calib), tm)


def sweep(grid: dict, **base) -> list:
    """Cartesian product of `grid` (param -> list of values) over simulate(**base). Synthetic traces are
    generated once and reused. Hardware fields can be swept via hw_overrides dicts."""
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        kw = dict(base)
        kw.update(zip(keys, combo))
        out.append(simulate(**kw))
    return out


def compare_policies(policies=POLICIES, **kw) -> list:
    return sweep({"policy": list(policies)}, **kw)


def _base_label(kw: dict) -> str:
    cg = kw.get("cache_gb")
    pf, sp, lo = _as_prefetch(kw.get("prefetch")), _as_spec(kw.get("spec")), _as_lossy(kw.get("lossy"))
    bits = [kw.get("policy", "lfu"), "all free RAM as cache" if cg is None else f"{cg:g} GiB cache",
            "prefetch off" if pf is None else f"prefetch recall {pf.recall:g} + {pf.extra} extra",
            "no speculative decoding" if sp is None else f"speculative k={sp.k} alpha={sp.alpha:g}"]
    if kw.get("gpu", "off") != "off":
        bits.append(f"gpu {kw['gpu']}")
    for name, what in (("expert_bits", "experts"), ("dense_bits", "backbone")):
        v = kw.get(name, DEFAULT_BITS[name])
        if v != DEFAULT_BITS[name]:
            bits.append(f"{what} {v:g} bpw")
    bits += bits_labels(**{k: kw.get(k, d) for k, d in DEFAULT_BITS.items()}) + lo.labels()
    return "baseline: " + ", ".join(bits)


def levers(model="kimi-k2", hardware="this-pc", **base) -> list:
    """One-at-a-time what-ifs against the baseline given by `base` (simulate() keywords; by default lfu,
    all free RAM as cache, no prefetch, no speculative decoding). Each row toggles one thing relative to
    that baseline and is labelled from the actual settings. Returns [(label, Result)]. Lossy and roadmap
    items are labelled as such."""
    hw = get_hardware(hardware, **(base.pop("hw_overrides", None) or {}))
    shape = presets.get(model)
    kw = dict(model=model, hardware=hw, **base)
    pf, sp, lo = _as_prefetch(kw.get("prefetch")), _as_spec(kw.get("spec")), _as_lossy(kw.get("lossy"))
    policy, cg, gpu = kw.get("policy", "lfu"), kw.get("cache_gb"), kw.get("gpu", "off")
    eb, db = kw.get("expert_bits", DEFAULT_BITS["expert_bits"]), kw.get("dense_bits", DEFAULT_BITS["dense_bits"])
    run = lambda **ch: simulate(**{**kw, **ch})                     # noqa: E731
    more_nvme = get_hardware(hw, nvme_count=hw.nvme_count + 1)
    rows = [(_base_label(kw), simulate(**kw))]
    alt = "lru" if policy != "lru" else "lfu"
    rows.append((f"policy {alt}" + (" (global)" if alt == "lru" else ""), run(policy=alt)))
    if pf is None:
        rows.append(("prefetch next layer (recall 0.8)", run(prefetch=Prefetch(0.8, 0))))
    else:
        rows.append(("prefetch off", run(prefetch=None)))
    if sp is None:
        rows.append(("speculative k=4, alpha=0.6", run(spec=Spec(4, 0.6))))
        rows.append(("speculative k=4, alpha=0.8", run(spec=Spec(4, 0.8))))
    else:
        rows.append(("speculative decoding off", run(spec=None)))
    rows.append((f"+1 NVMe drive ({more_nvme.nvme_count}x{hw.nvme_gbs:g} GB/s)", run(hardware=more_nvme)))
    big = get_hardware(hw, ram_gib=2 * hw.ram_gib)
    if cg is None:
        rows.append((f"2x RAM ({big.ram_gib:.0f} GiB)", run(hardware=big)))
    else:
        rows.append((f"2x RAM ({big.ram_gib:.0f} GiB), cache {cg:g} -> {cg + hw.ram_gib:g} GiB",
                     run(hardware=big, cache_gb=cg + hw.ram_gib)))
    if db > 4.25:
        what = "Q4 instead of Q8" if db == 8.25 else f"4.25 bpw instead of {db:g}"
        rows.append((f"dense backbone {what} (LOSSY vs the Q8 default)", run(dense_bits=4.25)))
    if eb > 3.25:
        what = "Q3 experts 3.25 bpw (LOSSY vs Q4" if eb == 4.25 else f"experts 3.25 bpw (LOSSY vs {eb:g}"
        rows.append((what + ", roadmap R03)", run(expert_bits=3.25)))
    if lo.topk is None:
        k2 = max(1, shape.top_k * 3 // 4)
        rows.append((f"LOSSY top-k {shape.top_k}->{k2}", run(lossy=replace(lo, topk=k2))))
    if gpu != "off":
        rows.append(("GPU off (CPU only)", run(gpu="off")))
    elif hw.has_gpu:
        r = run(gpu="dense+experts")
        label = "GPU: backbone + hot experts on GPU (roadmap R01)"
        if not r.feasible and not hw.unified and db > 4.25:
            r = run(gpu="dense+experts", dense_bits=4.25)
            label = "GPU: Q4 backbone (LOSSY vs Q8) + hot experts on GPU (roadmap R01)"
        rows.append((label, r))
    combo, parts = {"hardware": more_nvme}, []
    if pf is None:
        combo["prefetch"] = Prefetch(0.8, 0)
        parts.append("prefetch 0.8")
    if sp is None:
        combo["spec"] = Spec(4, 0.6)
        parts.append("spec k=4 a=0.6")
    if parts:
        rows.append((" + ".join(parts + ["1 more NVMe"]), run(**combo)))
    return rows


ALL_POLICIES = POLICIES
__all__ = ["simulate", "sweep", "retime", "compare_policies", "levers", "Result", "Spec", "Lossy", "TraceSpec",
           "Prefetch", "ONLINE_POLICIES", "ALL_POLICIES", "resolve_trace"]
