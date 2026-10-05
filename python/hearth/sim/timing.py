"""Byte/FLOP costs of a model and the per-step timing model (docs/SIMULATOR.md sec. 3).

Everything is explicit and vectorised over forward steps; the cache simulation
supplies per-(step, layer) counts, this module turns them into seconds.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import asdict, dataclass, replace

import numpy as np

from .cache import CacheCounts, Schedule
from .hardware import Hardware

GPU_MODES = ("off", "dense", "dense+experts")


# ---- byte sizes (docs/FORMAT.md sec. 5, sec. 6) ------------------------------------

def row_bytes(bits: float, n: int) -> int:
    """Bytes of one stored row of n weights; exact for Q4 (34 B / 64) and Q8 (66 B / 64) rows, FORMAT.md sec. 6."""
    return int(math.ceil(bits * n / 8.0))


def slab_bytes(d_in: int, ffn: int, bits: float) -> int:
    """One expert slab: gate [F,D], up [F,D], down [D,F] with the sec. 5 alignment rules."""
    a64 = lambda x: (x + 63) & ~63          # noqa: E731
    off_up = a64(ffn * row_bytes(bits, d_in))
    off_down = a64(off_up + ffn * row_bytes(bits, d_in))
    end = off_down + d_in * row_bytes(bits, ffn)
    return (end + 4095) & ~4095


@dataclass(frozen=True)
class ModelCosts:
    name: str
    n_layers: int
    n_moe_layers: int
    n_dense_layers: int
    n_experts: int
    top_k: int
    expert_params: int
    slab: int                    # bytes of one expert slab at expert_bits
    slab_cold: int               # bytes at cold_bits (== slab when unused)
    expert_bits: float
    dense_bits: float
    # per step, read once from wherever the backbone lives
    global_bytes: float          # LM head, final norm, leading dense layers (attention + FFN + KV)
    global_params: float
    layer_bytes: float           # one MoE layer: attention, norms, router (F32), latent projections, KV
    layer_params: float
    shared_bytes: float          # one MoE layer's shared experts
    shared_params: float
    kv_bytes_per_layer: float    # KV read per layer per step at `context`
    # footprint
    resident_bytes: float        # whole backbone incl. embedding table
    embed_bytes: float           # the embedding table alone (one row read per token)
    kv_resident_bytes: float     # KV cache allocated for `max_seq`
    expert_total_bytes: float    # all slabs (disk and the upper bound for the cache)
    context: int
    act_bytes: float = 0.0       # one position's hidden state (F32) crossing PCIe in a GPU hand-off
    notes: tuple = ()


def model_costs(shape, *, expert_bits: float = 4.25, dense_bits: float = 8.25, embed_bits: float = 8.25,
                cold_bits: float | None = None, context: int = 1024, max_seq: int = 4096,
                kv_elem_bytes: float = 4.0) -> ModelCosts:
    """Byte/parameter accounting from a hearth.presets.Shape. Matrices at dense_bits, routers and
    norms F32 (FORMAT.md sec. 4.1). For latent MoE (expert_d != d_model) the per-layer down/up projections
    into the expert space (2*D*expert_d) are added to the backbone - Shape.dense_params() omits them."""
    for name, v in (("expert_bits", expert_bits), ("dense_bits", dense_bits), ("embed_bits", embed_bits)):
        if not 0.0 < v <= 32.0:
            raise ValueError(f"{name} must be in (0, 32], got {v}")
    if cold_bits is not None and not 0.0 < cold_bits <= 32.0:
        raise ValueError(f"cold_bits must be in (0, 32], got {cold_bits}")
    if not (context >= 0 and max_seq >= 0 and 0.0 <= kv_elem_bytes <= 8.0):
        raise ValueError(f"need context >= 0, max_seq >= 0 and 0 <= kv_elem_bytes <= 8 "
                         f"(got {context}, {max_seq}, {kv_elem_bytes})")
    D = shape.d_model
    De = shape.expert_d or D
    b = dense_bits / 8.0
    notes = []
    attn = shape.attn_params()
    if shape.attn == "mla":
        kv_per_pos = (shape.kv_lora_rank + shape.qk_rope_dim) * kv_elem_bytes
    else:
        kv_per_pos = 2 * shape.n_kv_heads * shape.head_dim * kv_elem_bytes
    norms = 2 * D * 4
    router = shape.n_experts * D * 4
    latent = 2 * D * De if De != D else 0
    if latent:
        notes.append(f"latent MoE: +{2 * D * De / 1e6:.1f}M projection params per MoE layer (not in Shape.dense_params)")
    shared = 3 * D * shape.shared_ffn
    dense_ffn = 3 * D * shape.dense_ffn
    head_params = 0 if shape.tie_embeddings else shape.vocab * D
    kv_layer = kv_per_pos * context
    global_params = head_params + shape.n_dense_layers * (attn + dense_ffn)
    global_bytes = head_params * b + shape.n_dense_layers * ((attn + dense_ffn) * b + norms + kv_layer) + D * 4
    layer_params = attn + latent + shape.n_experts * D
    layer_bytes = (attn + latent) * b + router + norms + kv_layer
    embed_bytes = shape.vocab * D * embed_bits / 8.0
    resident = (embed_bytes + head_params * b + shape.n_layers * (attn * b + norms)
                + shape.n_moe_layers * (router + (latent + shared) * b) + shape.n_dense_layers * dense_ffn * b + D * 4)
    slab = slab_bytes(De, shape.expert_ffn, expert_bits)
    slab_cold = slab_bytes(De, shape.expert_ffn, cold_bits) if cold_bits else slab
    if shape.arch == "kimi_k3":
        notes.append("kimi-k3: KV term assumes MLA in every layer (69 of 93 are KDA linear attention) - overstates KV")
    return ModelCosts(
        name=shape.name, n_layers=shape.n_layers, n_moe_layers=shape.n_moe_layers,
        n_dense_layers=shape.n_dense_layers, n_experts=shape.n_experts, top_k=shape.top_k,
        expert_params=3 * De * shape.expert_ffn, slab=slab, slab_cold=slab_cold, expert_bits=expert_bits,
        dense_bits=dense_bits, global_bytes=global_bytes, global_params=global_params, layer_bytes=layer_bytes,
        layer_params=layer_params, shared_bytes=shared * b, shared_params=shared, kv_bytes_per_layer=kv_layer,
        resident_bytes=resident, embed_bytes=embed_bytes, kv_resident_bytes=kv_per_pos * max_seq * shape.n_layers,
        expert_total_bytes=float(slab) * shape.n_moe_layers * shape.n_experts, context=context,
        act_bytes=4.0 * D, notes=tuple(notes))


# ---- calibration scalars -----------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    """Scalars fitted to engine measurements (``calibrate``). Defaults are uncalibrated guesses."""
    dram_eff: float = 1.0          # x hw.dram_gbs
    io_eff: float = 1.0            # x aggregate storage BW
    compute_eff: float = 1.0       # x hw.cpu_int8_tops
    overhead_ms: float = 1.0       # fixed per forward step (sampling, dispatch, Python)
    layer_overhead_us: float = 20.0  # per transformer layer per step (pool fork/join, small ops)

    def __post_init__(self):
        for f in ("dram_eff", "io_eff", "compute_eff"):
            if not 0.0 < getattr(self, f) < math.inf:
                raise ValueError(f"calibration {f} must be a finite number > 0, got {getattr(self, f)}")
        for f in ("overhead_ms", "layer_overhead_us"):
            if not 0.0 <= getattr(self, f) < math.inf:
                raise ValueError(f"calibration {f} must be a finite number >= 0, got {getattr(self, f)}")

    def to_dict(self) -> dict:
        return asdict(self)


COMPONENTS = ("overhead", "dram", "cpu", "nvme", "gpu")


@dataclass
class Timing:
    step_s: np.ndarray             # [S] seconds per forward step
    parts: np.ndarray              # [S, len(COMPONENTS)] attribution of step_s
    nvme_demand_bytes: np.ndarray  # [S] demand reads incl. promoted late prefetches
    nvme_prefetch_bytes: np.ndarray  # [S] prefetch reads completed in time (incl. wrong guesses)
    nvme_wasted_bytes: np.ndarray  # [S] the wrong-guess part of nvme_prefetch_bytes
    dram_bytes: np.ndarray         # [S] bytes the CPU streamed from DRAM
    vram_bytes: np.ndarray         # [S]
    late_prefetch: np.ndarray      # [S] prefetched experts that had not arrived in time (fractional)


def evaluate(counts: CacheCounts, sched: Schedule, costs: ModelCosts, hw: Hardware,
             calib: Calibration = Calibration(), gpu_mode: str = "off") -> Timing:
    if gpu_mode not in GPU_MODES:
        raise ValueError(f"gpu_mode must be one of {GPU_MODES}")
    S, L = sched.n_steps, sched.n_moe_layers
    f = lambda x: np.asarray(x, dtype=np.float64).reshape(S, L)   # noqa: E731
    hit, hitc, ph, phc = f(counts.hit), f(counts.hit_cold), f(counts.pfhit), f(counts.pfhit_cold)
    miss, missc, vram, skip = f(counts.miss), f(counts.miss_cold), f(counts.vram), f(counts.skip)
    pfw, pfwc = f(counts.pf_wasted), f(counts.pf_wasted_cold)
    pairs = f(sched.pairs)
    npos = sched.step_npos.astype(np.float64)

    bw_d = hw.dram_gbs * 1e9 * calib.dram_eff
    bw_io = max(hw.io_gbs * 1e9 * calib.io_eff, 1.0)
    ops = hw.cpu_int8_tops * 1e12 * calib.compute_eff
    lat = hw.nvme_latency_us * 1e-6
    on_gpu = gpu_mode != "off"
    bw_bb = hw.vram_gbs * 1e9 if on_gpu else bw_d          # where the backbone is read from
    # two hand-offs per MoE layer (to the GPU and back), each a latency plus the step's activations over PCIe
    xfer = npos * costs.act_bytes / (hw.pcie_gbs * 1e9) if (on_gpu and not hw.unified) else 0.0
    sync = 2 * (hw.gpu_sync_us * 1e-6 + xfer)

    def mem_cpu(nbytes, params, bw, cpu_side=True):
        mem = nbytes / bw
        if not cpu_side:
            return mem, np.zeros_like(npos)
        comp = npos * 2.0 * params / ops
        return mem, np.maximum(comp - mem, 0.0)

    g_mem, g_cpu = mem_cpu(costs.global_bytes, costs.global_params, bw_bb, not on_gpu)
    a_mem, a_cpu = mem_cpu(costs.layer_bytes, costs.layer_params, bw_bb, not on_gpu)
    s_mem, s_cpu = mem_cpu(costs.shared_bytes, costs.shared_params, bw_bb, not on_gpu)
    A = (a_mem + a_cpu) + ((s_mem + s_cpu + sync) if on_gpu else 0.0)       # [S]
    W_sh = 0.0 if on_gpu else (s_mem + s_cpu)                            # shared overlaps expert I/O on CPU

    union = hit + ph + miss + vram + skip
    tau = np.divide(pairs, union, out=np.ones_like(pairs), where=union > 0)
    cpu_e = tau * 2.0 * costs.expert_params / ops                      # [S, L] compute per expert
    # unified memory + GPU experts: the GPU computes cached experts straight from shared memory
    bw_e = hw.vram_gbs * 1e9 if (hw.unified and gpu_mode == "dense+experts") else bw_d
    mh, mc = costs.slab / bw_e, costs.slab_cold / bw_e
    ch, cc = np.maximum(mh, cpu_e), np.maximum(mc, cpu_e)               # per-expert compute time
    xh, xc = ch - mh, cc - mc                                           # compute excess over memory
    sh, sc = float(costs.slab), float(costs.slab_cold)

    step = np.zeros(S)
    parts = np.zeros((S, len(COMPONENTS)))
    nv_dem = np.zeros(S)
    nv_pf = np.zeros(S)
    nv_waste = np.zeros(S)
    dram_b = np.zeros(S)
    vram_b = np.zeros(S)
    late_n = np.zeros(S)
    slack = np.zeros(S)
    for ly in range(L):
        # Prefetches for layer l were queued behind layer l-1's demand reads; they can use the storage
        # channel from the end of those reads until layer l's MoE starts. Needed ones that have not
        # arrived by then are promoted to demand reads; stale wrong guesses are cancelled.
        ph_h, ph_c = ph[:, ly] - phc[:, ly], phc[:, ly]
        true_b = ph_h * sh + ph_c * sc
        false_b = ((pfw[:, ly - 1] - pfwc[:, ly - 1]) * sh + pfwc[:, ly - 1] * sc) if ly else np.zeros(S)
        pf_b = true_b + false_b
        budget = slack + A
        frac = np.where(pf_b > 0, np.minimum(1.0, budget * bw_io / np.maximum(pf_b, 1.0)), 1.0)
        late_h, late_c = (1 - frac) * ph_h, (1 - frac) * ph_c
        hit_h, hit_c = hit[:, ly] - hitc[:, ly], hitc[:, ly]
        mis_h, mis_c = miss[:, ly] - missc[:, ly], missc[:, ly]
        W0 = W_sh + (hit_h + frac * ph_h) * ch[:, ly] + (hit_c + frac * ph_c) * cc[:, ly]
        m = mis_h + mis_c + late_h + late_c
        Mb = (mis_h + late_h) * sh + (mis_c + late_c) * sc
        Cm = (mis_h + late_h) * ch[:, ly] + (mis_c + late_c) * cc[:, ly]
        has = m > 0
        msafe = np.where(has, m, 1.0)
        io = np.where(has, lat + Mb / bw_io, 0.0)
        t_cpu = np.where(has, np.maximum.reduce([W0 + Cm, io + Cm / msafe, lat + (Mb / msafe) / bw_io + Cm]), W0)
        t_gpu = np.where(vram[:, ly] > 0, vram[:, ly] * sh / (hw.vram_gbs * 1e9 if hw.vram_gbs else 1.0) + sync, 0.0)
        t_moe = np.maximum(t_cpu, t_gpu)
        slack = np.maximum(t_moe - io, 0.0)
        step += A + t_moe
        # attribution
        n_h = hit_h + frac * ph_h + mis_h + late_h
        n_c = hit_c + frac * ph_c + mis_c + late_c
        exp_mem = n_h * mh + n_c * mc
        exp_cpu = n_h * xh[:, ly] + n_c * xc[:, ly]
        parts[:, 1] += exp_mem + (0.0 if on_gpu else s_mem)
        parts[:, 2] += exp_cpu + (0.0 if on_gpu else s_cpu)
        parts[:, 3] += np.maximum(t_cpu - (W0 + Cm), 0.0)
        parts[:, 4] += np.maximum(t_gpu - t_cpu, 0.0)
        # bytes actually moved
        nv_dem += Mb
        nv_pf += frac * pf_b
        nv_waste += frac * false_b
        dram_b += (n_h * sh + n_c * sc)
        vram_b += vram[:, ly] * sh
        late_n += late_h + late_c
    fixed = calib.overhead_ms * 1e-3 + calib.layer_overhead_us * 1e-6 * costs.n_layers
    step += fixed + g_mem + g_cpu
    parts[:, 0] += fixed
    bb_mem = g_mem + L * a_mem
    bb_cpu = g_cpu + L * a_cpu
    if on_gpu:
        parts[:, 4] += bb_mem + L * (s_mem + sync)
        vram_b += costs.global_bytes + L * (costs.layer_bytes + costs.shared_bytes)
    else:
        parts[:, 1] += bb_mem
        parts[:, 2] += bb_cpu
        dram_b += costs.global_bytes + L * (costs.layer_bytes + costs.shared_bytes)
    return Timing(step, parts, nv_dem, nv_pf, nv_waste, dram_b, vram_b, late_n)


# ---- calibration ----------------------------------------------------------------

def _nelder_mead(fun, x0, step=0.3, iters=400, tol=1e-10):
    n = len(x0)
    pts = [np.asarray(x0, dtype=np.float64)]
    for i in range(n):
        p = pts[0].copy()
        p[i] += step
        pts.append(p)
    vals = [fun(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = fun(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = fun(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = fun(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [fun(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


FITTABLE = ("dram_eff", "io_eff", "compute_eff", "overhead_ms", "layer_overhead_us")


def calibrate(points, fit=("dram_eff", "io_eff", "overhead_ms"), start: Calibration | None = None):
    """Fit calibration scalars to measured engine numbers.

    points: iterable of (Result, measured_tok_per_s) - each Result from ``simulate`` with the
    same model/hardware/settings/trace the engine ran (ideally a recorded trace via --trace).
    Minimises the squared error of log(tok/s) over the chosen scalars (all kept positive).
    Returns (Calibration, report dict with per-point predicted/measured and the RMS log error).
    """
    from .core import retime   # late import: core depends on this module

    pts = list(points)
    if not pts:
        raise ValueError("calibrate needs at least one (Result, measured tok/s) point")
    for i, pt in enumerate(pts):
        if not (isinstance(pt, (tuple, list)) and len(pt) == 2 and hasattr(pt[0], "_counts")):
            raise ValueError(f"calibration point {i} must be (Result from simulate(), measured tok/s), got {pt!r}")
        r, meas = pt
        if isinstance(meas, bool) or not isinstance(meas, numbers.Real) or not 0.0 < meas < math.inf:
            raise ValueError(f"calibration point {i} ({r.label()}): measured tok/s must be a finite number > 0, "
                             f"got {meas!r}")
        if not r.tok_s > 0.0:
            raise ValueError(f"calibration point {i} ({r.label()}): the simulated run has no measured steps")
    bad = set(fit) - set(FITTABLE)
    if bad:
        raise ValueError(f"cannot fit {sorted(bad)}; fittable: {FITTABLE}")
    base = start or pts[0][0].calibration
    x0 = np.log([max(getattr(base, k), 1e-6) for k in fit])

    def calib_of(x):
        return replace(base, **{k: float(np.exp(v)) for k, v in zip(fit, x)})

    def loss(x):
        c = calib_of(x)
        err = 0.0
        for r, meas in pts:
            pred = retime(r, c).tok_s
            err += (math.log(pred) - math.log(meas)) ** 2
        return err / len(pts)

    x, v = _nelder_mead(loss, x0)
    for step in (0.1, 0.02):                 # restarts: Nelder-Mead can stall on a flat valley
        x, v = _nelder_mead(loss, x, step=step)
    cal = calib_of(x)
    rows = [{"label": r.label(), "measured_tok_s": meas, "predicted_tok_s": retime(r, cal).tok_s,
             "uncalibrated_tok_s": r.tok_s} for r, meas in pts]
    return cal, {"rms_log_error": math.sqrt(v), "fit": list(fit), "points": rows}
