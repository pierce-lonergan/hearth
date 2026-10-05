"""Numpy reference forward pass — the numerical ground truth (docs/NUMERICS.md §5).

Everything is float32 and follows the canonical operation order of NUMERICS.md:
dot16/sum16 reductions, RMSNorm/softmax/RoPE/SwiGLU exactly as specified, rank-order
expert accumulation, sequential attention-value sums, the absorbed MLA form of §5.2.
Transcendentals (exp, sin, cos) are evaluated in float64 and rounded to float32,
i.e. (almost always) the correctly rounded value a good libm returns.

With ``emulate_act_quant=True`` (default), matrices stored as Q8/Q4 are applied the way
the engine applies them (§2 activation quantization, exact per-block integer dots,
float32 accumulation in block order), so the reference tracks the engine closely on
quantized containers. Otherwise quantized weights are dequantized exactly and used
like F32 weights.

Tokens are always processed one at a time (by definition, a batch equals the
sequence of single-token evaluations — INV-DET-2).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np

from . import quant
from .format import ContainerReader, canonical_tensors

f32 = np.float32
_ROW_CHUNK = 2048
_CANON_NAN = np.uint32(0x7FC00000)     # the kernels store every NaN output as this quiet NaN

__all__ = ["Reference", "sum16", "dot16_rows", "rmsnorm", "softmax", "silu", "sigmoid", "rope"]


# ---------------------------------------------------------------- canonical primitives

def sum16(a: np.ndarray) -> np.ndarray:
    """NUMERICS §1 sum16 along the last axis (16 lane accumulators, then a tree)."""
    a = np.asarray(a, dtype=f32)
    n = a.shape[-1]
    m = (n + 15) // 16
    if m * 16 != n:
        pad = [(0, 0)] * (a.ndim - 1) + [(0, m * 16 - n)]
        a = np.pad(a, pad)          # +0.0 lanes never change a lane value
    a = a.reshape(a.shape[:-1] + (m, 16))
    L = np.zeros(a.shape[:-2] + (16,), dtype=f32)
    for k in range(m):
        L = L + a[..., k, :]
    for s in (8, 4, 2, 1):
        L = L[..., :s] + L[..., s:2 * s]
    return L[..., 0]


def dot16_rows(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y[r] = dot16(W[r], x) for a float32 matrix W [R, n]."""
    R = W.shape[0]
    y = np.empty(R, dtype=f32)
    for r0 in range(0, R, _ROW_CHUNK):
        y[r0:r0 + _ROW_CHUNK] = sum16(W[r0:r0 + _ROW_CHUNK] * x)
    return y


def _exp(x):
    with np.errstate(over="ignore"):      # overflow -> inf, exactly like expf
        return np.exp(np.asarray(x, dtype=np.float64)).astype(f32)


def rmsnorm(x: np.ndarray, w: np.ndarray, eps) -> np.ndarray:
    """Over the last axis: ms = sum16(x*x)/n; r = 1/sqrt(ms+eps); y = (x*r)*w."""
    x = np.asarray(x, dtype=f32)
    ms = sum16(x * x) / f32(x.shape[-1])
    r = f32(1.0) / np.sqrt(ms + f32(eps))
    return (x * r[..., None]) * w


def softmax(x: np.ndarray) -> np.ndarray:
    """Over the last axis: m = max; e = exp(x-m); s = sum16(e); p = e/s."""
    x = np.asarray(x, dtype=f32)
    e = _exp(x - x.max(axis=-1, keepdims=True))
    return e / sum16(e)[..., None]


def sigmoid(x):
    return f32(1.0) / (f32(1.0) + _exp(-np.asarray(x, dtype=f32)))


def silu(x):
    x = np.asarray(x, dtype=f32)
    return x / (f32(1.0) + _exp(-x))


def rope(x: np.ndarray, pos: int, inv_freq: np.ndarray, rope_dim: int, style: int, factor) -> np.ndarray:
    """Rotate the first rope_dim dims of each head (last axis) at position pos (NUMERICS §4)."""
    x = np.array(x, dtype=f32, copy=True)
    half = rope_dim // 2
    th = f32(pos) * inv_freq[:half]
    c = np.cos(th.astype(np.float64)).astype(f32) * f32(factor)
    s = np.sin(th.astype(np.float64)).astype(f32) * f32(factor)
    if style == 0:
        ia, ib = np.arange(half), np.arange(half) + half
    else:
        ia, ib = np.arange(half) * 2, np.arange(half) * 2 + 1
    a, b = x[..., ia].copy(), x[..., ib].copy()
    x[..., ia] = a * c - b * s
    x[..., ib] = b * c + a * s
    return x


# ---------------------------------------------------------------- weights

class _Mat:
    """A weight matrix [rows, cols] that applies itself like the engine does."""
    __slots__ = ("dtype", "rows", "cols", "w", "q", "d")

    def __init__(self, dtype, rows, cols, w=None, q=None, d=None):
        self.dtype, self.rows, self.cols, self.w, self.q, self.d = dtype, rows, cols, w, q, d

    @classmethod
    def from_bytes(cls, buf, dtype, rows, cols, emulate):
        if quant.is_quant(dtype) and emulate:
            q, d = quant.split_blocks(buf, dtype, (rows, cols))
            return cls(dtype, rows, cols, q=q.astype(f32), d=d)
        w = quant.dequantize(buf, dtype, (rows, cols)).astype(f32, copy=False)
        return cls(dtype, rows, cols, w=w)

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.w, self.q, self.d) if a is not None)

    def slice_rows(self, r0, r1) -> "_Mat":
        if self.w is not None:
            return _Mat(self.dtype, r1 - r0, self.cols, w=self.w[r0:r1])
        return _Mat(self.dtype, r1 - r0, self.cols, q=self.q[r0:r1], d=self.d[r0:r1])

    def dense(self) -> np.ndarray:
        if self.w is not None:
            return self.w
        nb = self.cols // quant.QK
        return (self.q.reshape(self.rows, nb, quant.QK) * self.d[:, :, None]).reshape(self.rows, self.cols)

    def row(self, i) -> np.ndarray:
        if self.w is not None:
            return self.w[i].copy()
        nb = self.cols // quant.QK
        return (self.q[i].reshape(nb, quant.QK) * self.d[i][:, None]).reshape(-1)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        y = self._matvec(x)
        nan = np.isnan(y)
        if nan.any():
            y.view(np.uint32)[nan] = _CANON_NAN
        return y

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=f32)
        if x.shape != (self.cols,):
            raise ValueError(f"matvec: x has shape {x.shape}, expected ({self.cols},)")
        if self.w is not None:
            return dot16_rows(self.w, x)
        # NUMERICS §3, quantized weights: exact per-block integer dots, block-order f32 accumulation
        xq, xd = quant.act_quant_q8(x)
        nb = self.cols // quant.QK
        xqf = xq.astype(f32).reshape(nb, quant.QK)
        acc = np.zeros(self.rows, dtype=f32)
        for r0 in range(0, self.rows, _ROW_CHUNK):
            r1 = min(self.rows, r0 + _ROW_CHUNK)
            # integer products and partial sums stay below 2**24: exact in float32, any order
            isum = np.einsum("rgi,gi->rg", self.q[r0:r1].reshape(r1 - r0, nb, quant.QK), xqf)
            a = np.zeros(r1 - r0, dtype=f32)
            for g in range(nb):
                a = a + isum[:, g] * (self.d[r0:r1, g] * xd[g])
            acc[r0:r1] = a
        return acc


class Reference:
    """Reference model over a .hearth container (path or ContainerReader)."""

    def __init__(self, path_or_reader, emulate_act_quant: bool = True, expert_cache_bytes: int = 4 << 30):
        if isinstance(path_or_reader, ContainerReader):
            self.reader, self._own = path_or_reader, False
        else:
            self.reader, self._own = ContainerReader(path_or_reader), True
        self.emulate = bool(emulate_act_quant)
        c = self.cfg = self.reader.config
        self.L, self.D, self.V = c["n_layers"], c["d_model"], c["vocab_size"]
        self.mla = c["attn_kind"] == 1
        self.H = c["n_heads"]
        self.Hkv = c["n_kv_heads"]
        self.hd = c["head_dim"]
        self.rope_dim = c["rope_dim"]
        self.E, self.K = c["n_experts"], c["top_k"]
        self.eps = f32(c["norm_eps"])
        self.moe_layers = [i for i in range(self.L) if c["layer_kind"][i] == 1]
        self._moe_index = {li: j for j, li in enumerate(self.moe_layers)}
        self._replay: np.ndarray | None = None
        self._mats: dict[str, _Mat] = {}
        self._vecs: dict[str, np.ndarray] = {}
        self._experts: OrderedDict = OrderedDict()
        self._exp_bytes, self._exp_cap = 0, int(expert_cache_bytes)
        try:
            self._validate_tensors()
            self.inv_freq = self._vec("rope_inv_freq") if self.rope_dim > 0 else np.zeros(0, dtype=f32)
            self.tok_embd = self._mat("tok_embd")
            self.lm_head = self.tok_embd if c["tie_embeddings"] else self._mat("lm_head")
            self.out_norm = self._vec("out_norm")
        except BaseException:
            self.close()                          # a rejected container is not left open (locked on Windows)
            raise
        self.router_trace: list | None = None   # set to [] to record (layer, sel, group_scores) per decision
        self.reset()

    # -- weights
    def _validate_tensors(self) -> None:
        """Every tensor the forward pass reads exists with its FORMAT.md §4.1 shape (and is F32
        where the format requires it), so a malformed container never yields numbers."""
        tens = self.reader.tensors
        for name, (shape, f32_only) in canonical_tensors(self.cfg).items():
            info = tens.get(name)
            if info is None:
                raise ValueError(f"{self.reader.path}: container lacks tensor {name!r}")
            if info.shape != shape:
                raise ValueError(f"{self.reader.path}: tensor {name!r} has shape {info.shape}, expected {shape}")
            if f32_only and info.dtype != quant.F32:
                raise ValueError(f"{self.reader.path}: tensor {name!r} is {quant.NAMES[info.dtype]}, must be F32")

    def _mat(self, name, emulate=None) -> _Mat:
        key = name if emulate is None else f"{name}#{int(emulate)}"
        m = self._mats.get(key)
        if m is None:
            info = self.reader.tensors[name]
            rows, cols = info.shape                   # every matrix is 2-D (checked by _validate_tensors)
            m = _Mat.from_bytes(self.reader.read_tensor(name, dequant=False), info.dtype, rows, cols,
                                self.emulate if emulate is None else emulate)
            self._mats[key] = m
        return m

    def _vec(self, name) -> np.ndarray:
        v = self._vecs.get(name)
        if v is None:
            v = np.asarray(self.reader.read_tensor(name), dtype=f32).reshape(-1)
            self._vecs[name] = v
        return v

    def _expert(self, layer, e):
        key = (layer, e)
        hit = self._experts.get(key)
        if hit is not None:
            self._experts.move_to_end(key)
            return hit
        ent = self.reader.expert_entry(layer, e)
        F, D = self.cfg["expert_ffn_dim"], self.D
        g, u, dn = self.reader.read_expert(layer, e, dequant=False)
        mats = (_Mat.from_bytes(g, ent.dtype, F, D, self.emulate), _Mat.from_bytes(u, ent.dtype, F, D, self.emulate),
                _Mat.from_bytes(dn, ent.dtype, D, F, self.emulate))
        nb = sum(m.nbytes for m in mats)
        while self._experts and self._exp_bytes + nb > self._exp_cap:
            _, old = self._experts.popitem(last=False)
            self._exp_bytes -= sum(m.nbytes for m in old)
        self._experts[key] = mats
        self._exp_bytes += nb
        return mats

    # -- state
    def reset(self) -> None:
        self.pos = 0
        self._kv = [None] * self.L
        self._routes: list[np.ndarray] = []

    def close(self) -> None:
        if self._own:
            self.reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False

    def routing_history(self) -> np.ndarray:
        """uint16 [tokens since reset, n_moe_layers, top_k], each top-k in rank order."""
        if not self._routes:
            return np.zeros((0, len(self.moe_layers), self.K), dtype=np.uint16)
        return np.stack(self._routes).astype(np.uint16)

    def replay_routes(self, routes) -> None:
        """Use these expert ids instead of the computed top-k: routes [positions, n_moe_layers,
        top_k] (e.g. another model's routing_history()), indexed by position since reset().
        Gate weights still come from this model's own scores. Separates numerical error from
        discrete routing flips when comparing two models. None restores normal routing; the
        setting survives reset()."""
        if routes is None:
            self._replay = None
            return
        r = np.asarray(routes)
        want = (len(self.moe_layers), self.K)
        if r.ndim != 3 or r.shape[1:] != want or not np.issubdtype(r.dtype, np.integer):
            raise ValueError(f"routes must be an integer array [positions, {want[0]}, {want[1]}], got {r.shape}")
        r = r.astype(np.int64)
        if r.size and (r.min() < 0 or r.max() >= self.E):
            raise ValueError(f"routes hold expert ids outside [0, {self.E})")
        s = np.sort(r, axis=-1)
        if (s[..., 1:] == s[..., :-1]).any():
            raise ValueError("routes repeat an expert within one top-k")
        self._replay = r

    def eval(self, tokens: Sequence[int]) -> np.ndarray:
        toks = [int(t) for t in np.asarray(tokens).reshape(-1)]
        out = np.empty((len(toks), self.V), dtype=f32)
        for i, t in enumerate(toks):
            out[i] = self._forward(t)
        return out

    # -- forward
    def _append(self, layer, *rows):
        st = self._kv[layer]
        if st is None:
            st = self._kv[layer] = [np.zeros((16,) + r.shape, dtype=f32) for r in rows]
        if self.pos >= st[0].shape[0]:
            st[:] = [np.concatenate([a, np.zeros_like(a)]) for a in st]
        for a, r in zip(st, rows):
            a[self.pos] = r
        return [a[:self.pos + 1] for a in st]

    def _forward(self, token: int) -> np.ndarray:
        c = self.cfg
        if not 0 <= token < self.V:
            raise ValueError(f"token {token} outside vocabulary of {self.V}")
        if self.pos >= c["max_seq"]:
            raise ValueError(f"position {self.pos} exceeds max_seq {c['max_seq']}")
        rs = f32(c["residual_scale"])
        h = f32(c["emb_scale"]) * self.tok_embd.row(token)
        route = []
        for i in range(self.L):
            p = f"blk.{i}."
            x = rmsnorm(h, self._vec(p + "attn_norm"), self.eps)
            a = self._attn_mla(i, x) if self.mla else self._attn_gqa(i, x)
            h = h + rs * a
            x = rmsnorm(h, self._vec(p + "ffn_norm"), self.eps)
            if c["layer_kind"][i] == 1:
                f, ids = self._moe(i, x)
                route.append(ids)
            else:
                f = self._ffn(p + "ffn_", x)
            h = h + rs * f
        h = rmsnorm(h, self.out_norm, self.eps)
        logits = f32(c["logit_scale"]) * self.lm_head.matvec(h)
        self._routes.append(np.array(route, dtype=np.uint16).reshape(len(self.moe_layers), self.K))
        self.pos += 1
        return logits

    def _lin(self, name, x, bias=None):
        y = self._mat(name).matvec(x)
        if bias is not None:
            y = y + self._vec(bias)
        return y

    def _attn_gqa(self, i, x):
        c, p = self.cfg, f"blk.{i}."
        H, Hkv, hd = self.H, self.Hkv, self.hd
        qb = c["qkv_bias"] == 1
        q = self._lin(p + "attn_q", x, p + "attn_q_bias" if qb else None)
        k = self._lin(p + "attn_k", x, p + "attn_k_bias" if qb else None)
        v = self._lin(p + "attn_v", x, p + "attn_v_bias" if qb else None)
        if c["qk_norm"] == 1:
            q = rmsnorm(q.reshape(H, hd), self._vec(p + "attn_q_norm"), self.eps).reshape(-1)
            k = rmsnorm(k.reshape(Hkv, hd), self._vec(p + "attn_k_norm"), self.eps).reshape(-1)
        elif c["qk_norm"] == 2:
            q = rmsnorm(q, self._vec(p + "attn_q_norm"), self.eps)
            k = rmsnorm(k, self._vec(p + "attn_k_norm"), self.eps)
        q = rope(q.reshape(H, hd), self.pos, self.inv_freq, self.rope_dim, c["rope_style"], c["rope_attn_factor"])
        k = rope(k.reshape(Hkv, hd), self.pos, self.inv_freq, self.rope_dim, c["rope_style"], c["rope_attn_factor"])
        K, V = self._append(i, k, v.reshape(Hkv, hd))
        grp = np.arange(H) // (H // Hkv)
        Kh = K[:, grp, :].transpose(1, 0, 2)            # [H, T, hd]
        scores = f32(c["attn_scale"]) * sum16(q[:, None, :] * Kh)
        pr = softmax(scores)                              # [H, T]
        o = np.zeros((H, hd), dtype=f32)
        Vh = V[:, grp, :]                                 # [T, H, hd]
        for t in range(Vh.shape[0]):
            o = o + pr[:, t, None] * Vh[t]
        return self._mat(p + "attn_o").matvec(o.reshape(-1))

    def _attn_mla(self, i, x):
        c, p = self.cfg, f"blk.{i}."
        H, nope, rd, vd, C = self.H, c["qk_nope_dim"], c["qk_rope_dim"], c["v_head_dim"], c["kv_lora_rank"]
        if c["q_lora_rank"]:
            qa = rmsnorm(self._mat(p + "attn_q_a").matvec(x), self._vec(p + "attn_q_a_norm"), self.eps)
            q = self._mat(p + "attn_q_b").matvec(qa)
        else:
            q = self._mat(p + "attn_q").matvec(x)
        q = q.reshape(H, nope + rd)
        q_nope = q[:, :nope]
        # NUMERICS §4: the first rope_dim (<= qk_rope_dim) dims of the rope part are rotated
        rdim, st, fac = self.rope_dim, c["rope_style"], c["rope_attn_factor"]
        q_pe = rope(q[:, nope:], self.pos, self.inv_freq, rdim, st, fac)
        kva = self._mat(p + "attn_kv_a").matvec(x)
        lat = rmsnorm(kva[:C], self._vec(p + "attn_kv_a_norm"), self.eps)
        kpe = rope(kva[C:C + rd], self.pos, self.inv_freq, rdim, st, fac)
        Cc, Kpe = self._append(i, lat, kpe)               # [T, C], [T, rd]
        kvb = self._mat(p + "attn_kv_b")
        wk = self._wuk(p, kvb, H, nope, vd)               # [H, nope, C]
        q_lat = np.zeros((H, C), dtype=f32)
        for n in range(nope):
            q_lat = q_lat + q_nope[:, n, None] * wk[:, n, :]
        s_lat = sum16(q_lat[:, None, :] * Cc[None, :, :])   # [H, T]
        s_pe = sum16(q_pe[:, None, :] * Kpe[None, :, :])
        pr = softmax(f32(c["attn_scale"]) * (s_lat + s_pe))
        o_lat = np.zeros((H, C), dtype=f32)
        for t in range(Cc.shape[0]):
            o_lat = o_lat + pr[:, t, None] * Cc[t][None, :]
        o = np.empty((H, vd), dtype=f32)
        for h in range(H):
            r0 = h * (nope + vd) + nope
            o[h] = kvb.slice_rows(r0, r0 + vd).matvec(o_lat[h])
        return self._mat(p + "attn_o").matvec(o.reshape(-1))

    def _wuk(self, p, kvb, H, nope, vd):
        key = p + "#wuk"
        w = self._vecs.get(key)
        if w is None:
            full = kvb.dense().reshape(H, nope + vd, -1)
            w = self._vecs[key] = np.ascontiguousarray(full[:, :nope, :])
        return w

    def _ffn(self, prefix, x):
        g = self._mat(prefix + "gate").matvec(x)
        u = self._mat(prefix + "up").matvec(x)
        return self._mat(prefix + "down").matvec(silu(g) * u)

    def _route(self, i, x):
        c, p, E, K = self.cfg, f"blk.{i}.", self.E, self.K
        logits = self._mat(p + "moe_router", emulate=False).matvec(x)
        score = softmax(logits) if c["score_fn"] == 0 else sigmoid(logits)
        sel = score + self._vec(p + "moe_router_bias") if c["score_bias"] == 1 else score.copy()
        ng, gscore = c["n_group"], None
        if ng > 1:
            g = sel.reshape(ng, E // ng)
            top = -np.sort(-g, axis=1)[:, :2]
            gscore = top[:, 0] + top[:, 1] if top.shape[1] > 1 else top[:, 0]
            keep = np.lexsort((np.arange(ng), -gscore))[:c["topk_group"]]
            mask = np.ones(ng, dtype=bool)
            mask[keep] = False
            g = g.copy()
            g[mask] = f32(0.0)
            sel = g.reshape(E)
        if self.router_trace is not None:
            self.router_trace.append((i, sel.copy(), None if gscore is None else gscore.copy()))
        if self._replay is None:
            ids = np.lexsort((np.arange(E), -sel))[:K]
        elif self.pos < self._replay.shape[0]:
            ids = self._replay[self.pos, self._moe_index[i]]
        else:
            raise ValueError(f"replayed routes cover {self._replay.shape[0]} positions; position {self.pos} has none")
        w = score[ids].astype(f32)
        if c["norm_topk_prob"] == 1:
            s = f32(0.0)
            for j in range(K):
                s = f32(s + w[j])
            w = w / f32(s + f32(1e-20))
        w = w * f32(c["routed_scale"])
        return ids, w

    def _moe(self, i, x):
        c, p = self.cfg, f"blk.{i}."
        ids, w = self._route(i, x)
        out = np.zeros(self.D, dtype=f32)
        for j, e in enumerate(ids):
            g, u, dn = self._expert(i, int(e))
            y = dn.matvec(silu(g.matvec(x)) * u.matvec(x))
            out = out + w[j] * y
        if c["shared_ffn_dim"] > 0:
            s = self._ffn(p + "shexp_", x)
            if c["shared_gate"] == 1:
                gi = self._mat(p + "shexp_gate_inp", emulate=False)
                s = s * sigmoid(dot16_rows(gi.dense(), x)[0])
            out = out + s
        return out, ids
