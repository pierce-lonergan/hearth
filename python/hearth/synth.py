"""Synthetic containers: tiny random models for tests, real-shaped ones for benchmarks.

make_tiny builds a small model whose weights exercise every code path of an
architecture flavour and checks, by running hearth.reference on a fixed token
sequence (the golden-test sequence), that every routing decision has a clear margin
— the engine/reference routing-equality test must never hinge on a float near-tie.
Routers of layers with a tight decision are redrawn until the margins are clear.

make_shaped writes a benchmark container with a real model's geometry and random
weights (synthetic — label results as such, INV-HONEST). Encoded rows are drawn
from small random pools so even trillion-parameter shapes stream to disk quickly;
`physical_experts` aliases slabs (FORMAT.md §5 flag bit 0) to bound the file size.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import presets, quant
from .format import ContainerWriter, slab_layout
from .quant import BF16, F16, F32, Q4, Q8
from .reference import Reference

ARCHS = ("qwen3_moe", "olmoe", "mixtral", "qwen2_moe", "deepseek_v3")

# The golden tests' token sequence (wrapped into the vocabulary when it is smaller).
CHECK_TOKENS = (1, 17, 42, 99, 3, 250, 7, 7, 64, 128, 5, 200)
_SOFTMAX_REL_GAP = 2e-3     # min relative gap between consecutive top-(k+1) softmax scores
_SIGMOID_ABS_GAP = 3e-4     # min absolute gap between consecutive top-(k+1) sigmoid(+bias) scores
_GROUP_ABS_GAP = 6e-4       # min gap at the topk_group boundary of group scores
_MAX_REDRAWS = 64

f32 = np.float32

__all__ = ["make_tiny", "make_shaped", "ARCHS", "CHECK_TOKENS"]


def _inv_freq(rope_dim: int, theta: float) -> np.ndarray:
    j = np.arange(0, rope_dim, 2, dtype=np.float64)
    return (1.0 / theta ** (j / rope_dim)).astype(f32)


def _groups(E: int, K: int) -> tuple[int, int]:
    for ng in (4, 2):
        if E % ng == 0 and E // ng >= 2:
            tg = max(1, ng // 2)
            while tg * (E // ng) < K:
                tg += 1
            if tg < ng:
                return ng, tg
    return 1, 1


def _tiny_meta(arch, L, D, E, K, F, V, max_seq, dtype) -> dict:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose one of {', '.join(ARCHS)}")
    if not 0 < K <= E:
        raise ValueError("need 0 < top_k <= n_experts")
    H = 4 if D >= 64 else 2
    m = dict(arch=arch, n_layers=L, d_model=D, vocab_size=V, max_seq=max_seq, norm_eps=1e-6, n_heads=H,
             n_experts=E, top_k=K, expert_ffn_dim=F, source="synthetic", bos_id=0, eos_ids=[V - 1],
             expert_dtype=dtype, layer_kind=[1] * L)
    hd = D // H
    if arch == "qwen3_moe":
        hd = 2 * D // H                      # head_dim != d_model / n_heads, as in the real models
        m.update(n_kv_heads=H // 2, head_dim=hd, qk_norm=1, norm_topk_prob=1)
        theta = 1e6
    elif arch == "olmoe":
        m.update(n_kv_heads=H, head_dim=hd, qk_norm=2, norm_eps=1e-5)
        theta = 1e4
    elif arch == "mixtral":
        m.update(n_kv_heads=H // 2, head_dim=hd, norm_topk_prob=1, norm_eps=1e-5)
        theta = 1e6
    elif arch == "qwen2_moe":
        m.update(n_kv_heads=H // 2, head_dim=hd, qkv_bias=1, shared_ffn_dim=2 * F, shared_gate=1)
        theta = 1e6
    else:  # deepseek_v3
        ng, tg = _groups(E, K)
        nope, rope_d, vd = 32, 16, 32
        mscale = 0.1 * math.log(4.0) + 1.0
        m.update(attn_kind=1, q_lora_rank=64, kv_lora_rank=64, qk_nope_dim=nope, qk_rope_dim=rope_d,
                 v_head_dim=vd, rope_dim=rope_d, rope_style=1, rope_attn_factor=1.1,
                 attn_scale=(nope + rope_d) ** -0.5 * mscale * mscale, dense_ffn_dim=2 * D,
                 shared_ffn_dim=F, score_fn=1, score_bias=1, n_group=ng, topk_group=tg,
                 norm_topk_prob=1, routed_scale=2.5)
        m["layer_kind"] = [0] + [1] * (L - 1)
        m["_theta"] = 1e4
        return m
    m["rope_dim"] = hd
    m["attn_scale"] = hd ** -0.5
    m["_theta"] = theta
    return m


def _mat_dtype(dtype: int, cols: int) -> int:
    return F16 if quant.is_quant(dtype) and cols % quant.QK else dtype


def _tiny_weights(m: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    L, D, V, H = m["n_layers"], m["d_model"], m["vocab_size"], m["n_heads"]

    def lin(o, i, gain=1.0):
        return (rng.standard_normal((o, i)) * (gain / math.sqrt(i))).astype(f32)

    def normw(n):
        return np.clip(1.0 + 0.15 * rng.standard_normal(n), 0.5, 1.5).astype(f32)

    def bias(n):
        return (0.1 * rng.standard_normal(n)).astype(f32)

    t: dict[str, np.ndarray] = {}
    t["tok_embd"] = rng.standard_normal((V, D)).astype(f32)
    t["rope_inv_freq"] = _inv_freq(m["rope_dim"], m["_theta"])
    t["out_norm"] = normw(D)
    t["lm_head"] = lin(V, D, 2.0)
    for i in range(L):
        p = f"blk.{i}."
        t[p + "attn_norm"], t[p + "ffn_norm"] = normw(D), normw(D)
        if m.get("attn_kind", 0) == 1:
            nope, rd, vd, C, ql = m["qk_nope_dim"], m["qk_rope_dim"], m["v_head_dim"], m["kv_lora_rank"], m["q_lora_rank"]
            if ql:
                t[p + "attn_q_a"], t[p + "attn_q_a_norm"] = lin(ql, D), normw(ql)
                t[p + "attn_q_b"] = lin(H * (nope + rd), ql)
            else:
                t[p + "attn_q"] = lin(H * (nope + rd), D)
            t[p + "attn_kv_a"], t[p + "attn_kv_a_norm"] = lin(C + rd, D), normw(C)
            t[p + "attn_kv_b"] = lin(H * (nope + vd), C)
            t[p + "attn_o"] = lin(D, H * vd)
        else:
            Hkv, hd = m["n_kv_heads"], m["head_dim"]
            t[p + "attn_q"], t[p + "attn_k"], t[p + "attn_v"] = lin(H * hd, D), lin(Hkv * hd, D), lin(Hkv * hd, D)
            t[p + "attn_o"] = lin(D, H * hd)
            if m.get("qkv_bias"):
                t[p + "attn_q_bias"], t[p + "attn_k_bias"], t[p + "attn_v_bias"] = bias(H * hd), bias(Hkv * hd), bias(Hkv * hd)
            if m.get("qk_norm") == 1:
                t[p + "attn_q_norm"], t[p + "attn_k_norm"] = normw(hd), normw(hd)
            elif m.get("qk_norm") == 2:
                t[p + "attn_q_norm"], t[p + "attn_k_norm"] = normw(H * hd), normw(Hkv * hd)
        if m["layer_kind"][i] == 0:
            Fd = m["dense_ffn_dim"]
            t[p + "ffn_gate"], t[p + "ffn_up"], t[p + "ffn_down"] = lin(Fd, D), lin(Fd, D), lin(D, Fd)
            continue
        t.update(_router(m, i, rng))
        Fs = m.get("shared_ffn_dim", 0)
        if Fs:
            t[p + "shexp_gate"], t[p + "shexp_up"], t[p + "shexp_down"] = lin(Fs, D), lin(Fs, D), lin(D, Fs)
            if m.get("shared_gate"):
                t[p + "shexp_gate_inp"] = lin(1, D, 2.0)
    return t


def _router(m: dict, i: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    E, D, p = m["n_experts"], m["d_model"], f"blk.{i}."
    gain = 2.5 if m.get("score_fn", 0) == 0 else 2.0
    out = {p + "moe_router": (rng.standard_normal((E, D)) * (gain / math.sqrt(D))).astype(f32)}
    if m.get("score_bias"):
        out[p + "moe_router_bias"] = (0.1 * rng.standard_normal(E)).astype(f32)
    return out


def _tensor_dtype(name: str, arr: np.ndarray, dtype: int) -> int:
    base = name.split(".")[-1]
    if arr.ndim == 1 or base in ("moe_router", "shexp_gate_inp"):
        return F32
    d = Q8 if dtype == Q4 else dtype
    return _mat_dtype(d, arr.shape[-1])


def _margin_problem(sel: np.ndarray, gscore, m: dict) -> bool:
    K, E = m["top_k"], m["n_experts"]
    s = np.sort(sel.astype(np.float64))[::-1]
    n = min(K + 1, E)
    gaps = s[:n - 1] - s[1:n]
    if m.get("score_fn", 0) == 0:
        if np.any(gaps < _SOFTMAX_REL_GAP * np.maximum(s[:n - 1], 1e-30)):
            return True
    elif np.any(gaps < _SIGMOID_ABS_GAP):
        return True
    if gscore is not None:
        tg, ng = m["topk_group"], m["n_group"]
        g = np.sort(gscore.astype(np.float64))[::-1]
        if tg < ng and g[tg - 1] - g[tg] < _GROUP_ABS_GAP:
            return True
    return False


def _write_tiny(path, m, enc, experts, edtype):
    meta = {k: v for k, v in m.items() if not k.startswith("_")}
    moe = [i for i in range(m["n_layers"]) if m["layer_kind"][i] == 1]
    with ContainerWriter(path, meta) as w:
        for name, (shape, dt, b) in enc.items():
            w.declare_tensor(name, shape, dt)
        if moe:
            w.declare_experts(m["n_layers"], m["n_experts"], m["d_model"], m["expert_ffn_dim"], edtype, moe)
        for name, (shape, dt, b) in enc.items():
            w.write_tensor(name, b)
        for (li, e), mats in experts.items():
            w.write_expert(li, e, *mats)


def make_tiny(path, *, arch: str = "qwen3_moe", dtype: int = F32, seed: int = 0, n_layers: int = 3,
              d_model: int = 128, n_experts: int = 8, top_k: int = 2, expert_ffn: int = 64,
              vocab: int = 256, max_seq: int = 256) -> Path:
    """Tiny random model of `arch`; `dtype` applies to experts and big dense matrices
    (norms/router/biases stay F32; for Q4 the dense matrices use Q8). Q8/Q4 need the
    native quantizer."""
    path = Path(path)
    dtype = quant.dtype_of(dtype)
    if dtype not in (F32, F16, BF16, Q8, Q4):
        raise ValueError("dtype must be F32, F16, BF16, Q8 or Q4")
    m = _tiny_meta(arch, n_layers, d_model, n_experts, top_k, expert_ffn, vocab, max_seq, dtype)
    rng = np.random.default_rng(seed)
    w = _tiny_weights(m, rng)
    D, F = d_model, expert_ffn
    edtype = _mat_dtype(_mat_dtype(dtype, D), F)
    experts = {}
    for i in range(n_layers):
        if m["layer_kind"][i] == 1:
            for e in range(n_experts):
                g = (rng.standard_normal((F, D)) / math.sqrt(D)).astype(f32)
                u = (rng.standard_normal((F, D)) / math.sqrt(D)).astype(f32)
                dn = (rng.standard_normal((D, F)) / math.sqrt(F)).astype(f32)
                experts[(i, e)] = tuple(quant.quantize(a, edtype) for a in (g, u, dn))
    enc = {name: (a.shape, _tensor_dtype(name, a, dtype), quant.quantize(a, _tensor_dtype(name, a, dtype)))
           for name, a in w.items()}
    toks = [t % vocab for t in CHECK_TOKENS][:max_seq]     # the golden sequence, as far as max_seq allows
    moe_layers = [i for i in range(n_layers) if m["layer_kind"][i] == 1]
    redraws = 0
    try:
        while True:
            _write_tiny(path, m, enc, experts, edtype)
            if not moe_layers:
                return path
            with Reference(path, emulate_act_quant=True) as ref:
                ref.router_trace = []
                logits = ref.eval(toks)
                bad = sorted({li for li, sel, gs in ref.router_trace if _margin_problem(sel, gs, m)})
            if not np.isfinite(logits).all() or float(logits.std(axis=1).min()) < 0.1:
                raise RuntimeError(f"make_tiny({arch}): degenerate logits; adjust the weight scales")
            if not bad:
                return path
            redraws += 1
            if redraws > _MAX_REDRAWS:
                raise RuntimeError(f"make_tiny({arch}, seed={seed}): could not obtain clear routing margins")
            li = bad[0]
            r = _router(m, li, np.random.default_rng([seed, li, redraws]))
            for name, a in r.items():
                enc[name] = (a.shape, F32, quant.quantize(a, F32))
    except BaseException:
        path.unlink(missing_ok=True)     # never leave a container that failed its checks
        raise


# ---------------------------------------------------------------- benchmark shapes

def _shaped_meta(s: presets.Shape, expert_dtype: int, max_seq: int) -> dict:
    L, D = s.n_layers, s.d_model
    m = dict(arch=s.arch, n_layers=L, d_model=D, vocab_size=s.vocab, max_seq=max_seq, norm_eps=1e-6,
             n_heads=s.n_heads, n_experts=s.n_experts, top_k=s.top_k, expert_ffn_dim=s.expert_ffn,
             dense_ffn_dim=s.dense_ffn, shared_ffn_dim=s.shared_ffn, tie_embeddings=int(s.tie_embeddings),
             layer_kind=[0] * s.n_dense_layers + [1] * (L - s.n_dense_layers),
             source=f"synthetic:{s.name}", expert_dtype=expert_dtype)
    if s.attn == "mla":
        m.update(attn_kind=1, q_lora_rank=s.q_lora_rank, kv_lora_rank=s.kv_lora_rank, qk_nope_dim=s.qk_nope_dim,
                 qk_rope_dim=s.qk_rope_dim, v_head_dim=s.v_head_dim, rope_dim=s.qk_rope_dim, rope_style=1,
                 attn_scale=(s.qk_nope_dim + s.qk_rope_dim) ** -0.5)
    else:
        m.update(n_kv_heads=s.n_kv_heads, head_dim=s.head_dim, rope_dim=s.head_dim, attn_scale=s.head_dim ** -0.5)
    if s.arch == "qwen3_moe":
        m.update(qk_norm=1, norm_topk_prob=1)
    elif s.arch == "olmoe":
        m.update(qk_norm=2)
    elif s.arch == "mixtral":
        m.update(norm_topk_prob=1)
    elif s.arch == "qwen2_moe":
        m.update(qkv_bias=1, shared_gate=1 if s.shared_ffn else 0)
    elif s.arch == "deepseek_v3":
        kimi = "kimi" in s.name
        ng, tg = (1, 1) if kimi or s.n_experts % 8 else (8, 4)
        m.update(score_fn=1, score_bias=1, n_group=ng, topk_group=tg, norm_topk_prob=1,
                 routed_scale=2.827 if kimi else 2.5)
    return m


def _random_rows(rng: np.random.Generator, dtype: int, cols: int, n: int) -> np.ndarray:
    """n valid encoded rows of `cols` weights (std ~ 1/sqrt(cols)), as uint8 [n, row_bytes]."""
    sd = 1.0 / math.sqrt(cols)
    if dtype in (F32, F16, BF16):
        x = (rng.standard_normal((n, cols)) * sd).astype(f32)
        return np.frombuffer(quant.quantize(x, dtype), dtype=np.uint8).reshape(n, -1)
    nb = cols // quant.QK
    if dtype == Q8:
        blk = np.zeros((n, nb), dtype=quant._Q8_BLOCK)
        blk["d"] = np.float16(3.0 * sd / 127.0)
        blk["q"] = np.clip(np.rint(rng.standard_normal((n, nb, quant.QK)) * 42.0), -127, 127).astype(np.int8)
    elif dtype == Q4:
        blk = np.zeros((n, nb), dtype=quant._Q4_BLOCK)
        blk["d"] = np.float16(3.0 * sd / 7.0)
        q = np.clip(np.rint(rng.standard_normal((n, nb, quant.QK)) * 2.5), -8, 7).astype(np.int16) + 8
        blk["qs"] = (q[..., :32] | (q[..., 32:] << 4)).astype(np.uint8)
    else:
        raise ValueError(f"unsupported dtype {dtype}")
    return blk.view(np.uint8).reshape(n, -1)


class _LazyRandom:
    """A random F32 tensor generated only when written (keeps big routers out of RAM)."""

    def __init__(self, shape, sd, seed):
        self.shape, self.sd, self.seed = shape, sd, seed

    def __call__(self):
        return (np.random.default_rng(self.seed).standard_normal(self.shape) * self.sd).astype(f32)


class _Pools:
    def __init__(self, rng, n=97):
        self.rng, self.n, self.p = rng, n, {}

    def rows(self, dtype, cols, r0, r1, salt=0) -> bytes:
        key = (dtype, cols)
        if key not in self.p:
            self.p[key] = _random_rows(self.rng, dtype, cols, self.n)
        pool = self.p[key]
        idx = (np.arange(r0, r1, dtype=np.int64) * 31 + salt) % self.n
        return pool[idx].tobytes()


def make_shaped(path, *, preset: str | dict, expert_dtype=Q4, physical_experts: int | None = None,
                seed: int = 0, dense_dtype=Q8, max_seq: int = 4096) -> Path:
    """Benchmark container with `preset`'s geometry and synthetic random weights.

    physical_experts: distinct slabs stored per MoE layer (default: all). Entries
    e >= physical_experts alias slab (layer, e % physical_experts) with flag bit 0."""
    s = presets.get(preset)
    if s.arch not in ARCHS:
        raise ValueError(f"preset {s.name!r} has arch {s.arch!r}, which the engine does not implement; "
                         f"make_shaped supports {', '.join(ARCHS)}")
    if s.expert_d and s.expert_d != s.d_model:
        raise ValueError(f"preset {s.name!r} runs experts in a latent space (expert_d={s.expert_d}); not supported")
    if s.n_experts <= 0 or s.top_k <= 0:
        raise ValueError(f"preset {s.name!r} has no routed experts")
    path = Path(path)
    edt, ddt = quant.dtype_of(expert_dtype), quant.dtype_of(dense_dtype)
    E = s.n_experts
    P = E if physical_experts is None else int(physical_experts)
    if not 1 <= P <= E:
        raise ValueError(f"physical_experts must be in 1..{E}")
    m = _shaped_meta(s, edt, max_seq)
    L, D, V, H = s.n_layers, s.d_model, s.vocab, s.n_heads
    rng = np.random.default_rng(seed)
    pools = _Pools(rng)

    mats: list[tuple[str, tuple[int, int]]] = [("tok_embd", (V, D))]
    if not s.tie_embeddings:
        mats.append(("lm_head", (V, D)))
    vecs: dict[str, np.ndarray] = {"rope_inv_freq": _inv_freq(m["rope_dim"], 1e4 if s.attn == "mla" else 1e6),
                                   "out_norm": np.ones(D, dtype=f32)}
    for i in range(L):
        p = f"blk.{i}."
        vecs[p + "attn_norm"] = vecs[p + "ffn_norm"] = np.ones(D, dtype=f32)
        if s.attn == "mla":
            nope, rd, vd, C, ql = s.qk_nope_dim, s.qk_rope_dim, s.v_head_dim, s.kv_lora_rank, s.q_lora_rank
            if ql:
                mats += [(p + "attn_q_a", (ql, D)), (p + "attn_q_b", (H * (nope + rd), ql))]
                vecs[p + "attn_q_a_norm"] = np.ones(ql, dtype=f32)
            else:
                mats.append((p + "attn_q", (H * (nope + rd), D)))
            mats += [(p + "attn_kv_a", (C + rd, D)), (p + "attn_kv_b", (H * (nope + vd), C)), (p + "attn_o", (D, H * vd))]
            vecs[p + "attn_kv_a_norm"] = np.ones(C, dtype=f32)
        else:
            Hkv, hd = s.n_kv_heads, s.head_dim
            mats += [(p + "attn_q", (H * hd, D)), (p + "attn_k", (Hkv * hd, D)), (p + "attn_v", (Hkv * hd, D)),
                     (p + "attn_o", (D, H * hd))]
            if m.get("qkv_bias"):
                vecs[p + "attn_q_bias"] = np.zeros(H * hd, dtype=f32)
                vecs[p + "attn_k_bias"] = vecs[p + "attn_v_bias"] = np.zeros(Hkv * hd, dtype=f32)
            if m.get("qk_norm") == 1:
                vecs[p + "attn_q_norm"] = vecs[p + "attn_k_norm"] = np.ones(hd, dtype=f32)
            elif m.get("qk_norm") == 2:
                vecs[p + "attn_q_norm"] = np.ones(H * hd, dtype=f32)
                vecs[p + "attn_k_norm"] = np.ones(Hkv * hd, dtype=f32)
        if m["layer_kind"][i] == 0:
            Fd = s.dense_ffn
            mats += [(p + "ffn_gate", (Fd, D)), (p + "ffn_up", (Fd, D)), (p + "ffn_down", (D, Fd))]
        else:
            vecs[p + "moe_router"] = _LazyRandom((E, D), 1.0 / math.sqrt(D), (seed, i, 1))
            if m.get("score_bias"):
                vecs[p + "moe_router_bias"] = np.zeros(E, dtype=f32)
            if s.shared_ffn:
                Fs = s.shared_ffn
                mats += [(p + "shexp_gate", (Fs, D)), (p + "shexp_up", (Fs, D)), (p + "shexp_down", (D, Fs))]
                if m.get("shared_gate"):
                    vecs[p + "shexp_gate_inp"] = _LazyRandom((1, D), 1.0 / math.sqrt(D), (seed, i, 2))

    mat_dt = {name: _mat_dtype(ddt, shp[1]) for name, shp in mats}
    moe = [i for i in range(L) if m["layer_kind"][i] == 1]
    e_dt = _mat_dtype(_mat_dtype(edt, D), s.expert_ffn)
    F = s.expert_ffn
    with ContainerWriter(path, m) as w:
        for name, shp in mats:
            w.declare_tensor(name, shp, mat_dt[name])
        for name, a in vecs.items():
            w.declare_tensor(name, a.shape, F32)
        w.declare_experts(L, E, D, F, e_dt, moe, alias=lambda li, e: (li, e % P) if e >= P else None)
        for name, a in vecs.items():
            w.write_tensor(name, a() if isinstance(a, _LazyRandom) else a)
        for k, (name, (rows, cols)) in enumerate(mats):
            rb = quant.row_bytes(mat_dt[name], cols)
            step = max(1, (64 << 20) // rb)
            for r0 in range(0, rows, step):
                r1 = min(rows, r0 + step)
                w.write_tensor_rows(name, r0, pools.rows(mat_dt[name], cols, r0, r1, salt=k))
        payloads = []
        for j in range(4):
            payloads.append((pools.rows(e_dt, D, 0, F, salt=3 * j), pools.rows(e_dt, D, 0, F, salt=3 * j + 1),
                             pools.rows(e_dt, F, 0, D, salt=3 * j + 2)))
        n = 0
        for li in moe:
            for e in range(P):
                w.write_expert(li, e, *payloads[n % len(payloads)])
                n += 1
    return path


def shaped_bytes(preset: str | dict, expert_dtype=Q4, physical_experts: int | None = None) -> int:
    """Approximate size of make_shaped's output (expert slabs dominate)."""
    s = presets.get(preset)
    edt = quant.dtype_of(expert_dtype)
    P = s.n_experts if physical_experts is None else int(physical_experts)
    slab = slab_layout(_mat_dtype(_mat_dtype(edt, s.d_model), s.expert_ffn), s.d_model, s.expert_ffn)[3]
    return s.n_moe_layers * P * slab + int(s.dense_params() * 66 / 64)
