"""Routing traces: the .hrtr file format (docs/FORMAT.md sec. 9), a synthetic
generator, and trace statistics.

A trace is ``ids[token, moe_layer, rank]`` (uint16): the top-k experts each
evaluated token was routed to in every MoE layer, sorted by rank. Real traces
come from ``hearth_trace_start`` or ``Reference.routing_history()``.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TRACE_MAGIC = 0x52545248      # "HRTR"
MAX_EXPERTS = 65536           # ids are u16
MAX_KEYS = 1 << 24            # moe_layers * n_experts: bounds every per-(layer, expert) array
MAX_ROW = 1 << 20             # moe_layers * top_k ids per token
USAGE_MAGIC = 0x53555248      # "HRUS"
_HDR = struct.Struct("<6I")   # magic, version, n_layers, n_experts, top_k, n_moe_layers


@dataclass
class Trace:
    ids: np.ndarray            # uint16 [tokens, n_moe_layers, top_k], rank order
    n_experts: int
    n_layers: int = 0          # total transformer layers (0 -> n_moe_layers)
    source: str = ""

    def __post_init__(self):
        ids = np.asarray(self.ids)
        if ids.ndim != 3:
            raise ValueError(f"trace ids must be [tokens, moe_layers, top_k], got shape {ids.shape}")
        if ids.size and not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f"trace ids must be integers, got dtype {ids.dtype}")
        self.n_experts = int(self.n_experts)
        _check_geometry(ids.shape[1], self.n_experts, ids.shape[2])
        if ids.size:
            lo, hi = int(ids.min()), int(ids.max())
            if lo < 0 or hi >= self.n_experts:
                raise ValueError(f"trace has expert id {lo if lo < 0 else hi} outside [0, {self.n_experts})")
            if ids.shape[2] > 1 and not (np.diff(np.sort(ids, axis=-1), axis=-1) > 0).all():
                raise ValueError("trace has an expert twice in one token's top-k (routing is without replacement)")
        self.ids = np.ascontiguousarray(ids, dtype=np.uint16)
        if int(self.n_layers) != self.n_layers:
            raise ValueError(f"n_layers must be an integer, got {self.n_layers}")
        self.n_layers = int(self.n_layers) or self.n_moe_layers
        if not self.n_moe_layers <= self.n_layers <= 0xFFFFFFFF:      # the file header stores it as u32
            raise ValueError(f"n_layers must be 0 (= moe_layers) or in [{self.n_moe_layers}, 2**32), "
                             f"got {self.n_layers}")

    @property
    def n_tokens(self) -> int:
        return self.ids.shape[0]

    @property
    def n_moe_layers(self) -> int:
        return self.ids.shape[1]

    @property
    def top_k(self) -> int:
        return self.ids.shape[2]

    def __len__(self) -> int:
        return self.n_tokens

    def __getitem__(self, sl) -> "Trace":
        if not isinstance(sl, slice):
            raise TypeError("Trace supports slicing only")
        return Trace(self.ids[sl], self.n_experts, self.n_layers, self.source)

    # ---- file I/O (FORMAT.md sec. 9) ---------------------------------------
    def save(self, path) -> Path:
        path = Path(path)
        with open(path, "wb") as f:
            f.write(_HDR.pack(TRACE_MAGIC, 1, self.n_layers, self.n_experts, self.top_k, self.n_moe_layers))
            f.write(self.ids.astype("<u2", copy=False).tobytes())
        return path

    @classmethod
    def load(cls, path) -> "Trace":
        raw = Path(path).read_bytes()
        if len(raw) < _HDR.size:
            raise ValueError(f"{path}: too short for a routing trace")
        magic, ver, n_layers, n_experts, top_k, n_moe = _HDR.unpack_from(raw, 0)
        if magic != TRACE_MAGIC:
            raise ValueError(f"{path}: bad magic 0x{magic:08x} (want HRTR)")
        if ver != 1:
            raise ValueError(f"{path}: unsupported trace version {ver}")
        if top_k == 0 or n_moe == 0 or n_moe > n_layers:
            raise ValueError(f"{path}: inconsistent header (layers={n_layers} moe={n_moe} E={n_experts} k={top_k})")
        try:
            _check_geometry(n_moe, n_experts, top_k)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None
        per_tok = n_moe * top_k * 2
        body = len(raw) - _HDR.size
        if body % per_tok:
            raise ValueError(f"{path}: body of {body} bytes is not a whole number of tokens ({per_tok} B each)")
        ids = np.frombuffer(raw, dtype="<u2", offset=_HDR.size).reshape(-1, n_moe, top_k)
        return cls(ids, n_experts, n_layers, source=str(path))

    # ---- statistics -----------------------------------------------------
    def frequencies(self) -> np.ndarray:
        """Activation counts [n_moe_layers, n_experts]."""
        L, E = self.n_moe_layers, self.n_experts
        flat = (self.ids.astype(np.int64) + (np.arange(L, dtype=np.int64) * E)[None, :, None]).ravel()
        return np.bincount(flat, minlength=L * E).reshape(L, E)

    def stats(self, top_fracs=(0.01, 0.05, 0.10, 0.20), windows=(1, 2, 3, 5),
              mass_quantiles=(0.5, 0.8, 0.9, 0.95)) -> "TraceStats":
        return trace_stats(self, top_fracs, windows, mass_quantiles)


def _check_geometry(n_moe: int, n_experts: int, top_k: int) -> None:
    if not 1 <= n_experts <= MAX_EXPERTS:
        raise ValueError(f"n_experts must be in [1, {MAX_EXPERTS}], got {n_experts}")
    if top_k > n_experts or n_moe * n_experts > MAX_KEYS or n_moe * top_k > MAX_ROW:
        raise ValueError(f"unsupported trace geometry (moe_layers={n_moe}, experts={n_experts}, top_k={top_k}): "
                         f"need top_k <= experts, moe_layers*experts <= {MAX_KEYS}, moe_layers*top_k <= {MAX_ROW}")


@dataclass
class TraceStats:
    n_tokens: int
    n_moe_layers: int
    n_experts: int
    top_k: int
    entropy_bits: np.ndarray            # [L] empirical entropy of each layer's expert frequency
    entropy_norm: np.ndarray            # [L] entropy / log2(E) (1 = perfectly balanced)
    reuse_rate: np.ndarray              # [L] fraction of a token's experts also used by the previous token
    mass_top: dict                      # x -> mean over layers of the access mass in the top x of experts
    mass_top_pooled: dict               # x -> mass in the top x of all (layer, expert) pairs
    slots_for_mass: dict                # q -> (layer, expert) slabs needed to cover q of all accesses
    union_per_window: dict              # w -> mean distinct experts per layer over w consecutive tokens
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "tokens": self.n_tokens, "moe_layers": self.n_moe_layers, "experts": self.n_experts,
            "top_k": self.top_k,
            "entropy_norm_mean": float(self.entropy_norm.mean()),
            "entropy_norm_min": float(self.entropy_norm.min()),
            "reuse_rate_mean": float(self.reuse_rate.mean()),
            "mass_top": {f"{x:.0%}": round(v, 4) for x, v in self.mass_top.items()},
            "mass_top_pooled": {f"{x:.0%}": round(v, 4) for x, v in self.mass_top_pooled.items()},
            "slots_for_mass": {f"{q:.0%}": v for q, v in self.slots_for_mass.items()},
            "union_per_window": {w: round(v, 3) for w, v in self.union_per_window.items()},
        }


def trace_stats(tr: Trace, top_fracs=(0.01, 0.05, 0.10, 0.20), windows=(1, 2, 3, 5),
                mass_quantiles=(0.5, 0.8, 0.9, 0.95)) -> TraceStats:
    T, L, k, E = tr.n_tokens, tr.n_moe_layers, tr.top_k, tr.n_experts
    freq = tr.frequencies().astype(np.float64)
    tot = freq.sum(axis=1, keepdims=True)
    p = np.divide(freq, tot, out=np.zeros_like(freq), where=tot > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)
    ent_norm = ent / math.log2(E) if E > 1 else np.zeros(L)

    reuse = np.zeros(L)
    if T > 1:
        acc = np.zeros(L)
        step = max(1, int(4_000_000 // max(1, L * k * k)))
        for a in range(1, T, step):
            b = min(T, a + step)
            cur = tr.ids[a:b, :, :, None]
            prev = tr.ids[a - 1:b - 1, :, None, :]
            acc += (cur == prev).any(-1).sum(axis=(0, 2))
        reuse = acc / ((T - 1) * k)

    srt = -np.sort(-freq, axis=1)
    mass_top = {}
    for x in top_fracs:
        n = max(1, int(math.ceil(x * E)))
        mass_top[x] = float((srt[:, :n].sum(axis=1) / np.maximum(tot[:, 0], 1)).mean())
    pooled = -np.sort(-freq.ravel())
    total = pooled.sum()
    cum = np.cumsum(pooled) / max(total, 1)
    mass_pooled = {}
    for x in top_fracs:
        n = max(1, int(math.ceil(x * L * E)))
        mass_pooled[x] = float(cum[min(n, len(cum)) - 1])
    slots_for = {q: int(np.searchsorted(cum, q - 1e-12) + 1) for q in mass_quantiles}

    union = {}
    for w in windows:
        if w > T:
            continue
        nwin = T // w
        blk = tr.ids[:nwin * w].reshape(nwin, w, L, k).transpose(0, 2, 1, 3).reshape(nwin, L, w * k)
        s = np.sort(blk, axis=-1)
        distinct = 1 + (np.diff(s.astype(np.int32), axis=-1) != 0).sum(-1)
        union[w] = float(distinct.mean())
    return TraceStats(T, L, E, k, ent, ent_norm, reuse, mass_top, mass_pooled, slots_for, union)


# ---- synthetic traces ---------------------------------------------------

def zipf_popularity(n_experts: int, n_layers: int, exponent: float, rng: np.random.Generator) -> np.ndarray:
    """[L, E] probabilities: p(rank r) ~ (r+1)^-exponent, ranks randomly permuted per layer."""
    base = (np.arange(n_experts, dtype=np.float64) + 1.0) ** (-float(exponent))
    base /= base.sum()
    p = np.empty((n_layers, n_experts))
    for ly in range(n_layers):
        p[ly, rng.permutation(n_experts)] = base
    return p


def _first_occurrence(stream: np.ndarray) -> np.ndarray:
    """Boolean mask, True where a value appears for the first time in its row."""
    idx = np.argsort(stream, axis=-1, kind="stable")
    s = np.take_along_axis(stream, idx, axis=-1)
    first = np.ones(s.shape, dtype=bool)
    first[..., 1:] = s[..., 1:] != s[..., :-1]
    out = np.empty_like(first)
    np.put_along_axis(out, idx, first, axis=-1)
    return out


def synthetic(n_tokens: int, n_moe_layers: int, n_experts: int, top_k: int, *, zipf: float = 1.1,
              reuse: float = 0.25, seed: int = 0, n_layers: int = 0) -> Trace:
    """Deterministic synthetic routing trace.

    Per layer, expert popularity follows a power law with the given exponent over a random
    permutation of the experts. For every token and layer, each of the previous token's experts
    is kept independently with probability `reuse` (temporal locality); the remaining slots are
    filled by sampling without replacement proportional to popularity (successive sampling,
    excluding experts already chosen). Ranks are ordered by a noisy popularity score
    (log p + Gumbel). The measured reuse rate is >= `reuse` because popular experts recur anyway.
    """
    T, L, E, k = int(n_tokens), int(n_moe_layers), int(n_experts), int(top_k)
    if k <= 0:
        raise ValueError(f"need top_k > 0, got {k}")
    if not 0.0 <= reuse <= 1.0:
        raise ValueError("reuse must be in [0, 1]")
    if not 0.0 <= zipf < math.inf:
        raise ValueError(f"zipf exponent must be a finite number >= 0, got {zipf}")
    if T < 0:
        raise ValueError(f"n_tokens must be >= 0, got {T}")
    _check_geometry(L, E, k)
    rng = np.random.default_rng(seed)
    p = zipf_popularity(E, L, zipf, rng)
    logp = np.log(p)
    cdf = np.cumsum(p, axis=1)
    cdf[:, -1] = 1.0
    m = min((6 if zipf > 0.9 else 4) * k, 4096)   # draws per row; rows left short get an exact refill
    ids = np.empty((T, L, k), dtype=np.uint16)
    if T == 0:
        return Trace(ids, E, n_layers or L, source="synthetic")

    def draw_stream(nt: int) -> np.ndarray:
        u = rng.random((nt, L, m))
        out = np.empty((nt, L, m), dtype=np.int32)
        for ly in range(L):
            out[:, ly, :] = np.searchsorted(cdf[ly], u[:, ly, :], side="right")
        np.minimum(out, E - 1, out=out)
        return out

    def finish_rows(sel: np.ndarray) -> np.ndarray:
        # rank order by a noisy popularity score
        score = np.take_along_axis(np.broadcast_to(logp, sel.shape[:-2] + logp.shape), sel, axis=-1) \
            + rng.gumbel(size=sel.shape)
        order = np.argsort(-score, axis=-1, kind="stable")
        return np.take_along_axis(sel, order, axis=-1)

    def exact_fill(ly: int, exclude: np.ndarray, need: int) -> np.ndarray:
        g = logp[ly] + rng.gumbel(size=E)
        g[exclude] = -np.inf
        return np.argpartition(-g, need - 1)[:need] if need > 0 else np.empty(0, dtype=np.int64)

    CH = max(1, int(8_000_000 // max(1, L * m)))
    if reuse == 0.0:
        for a in range(0, T, CH):
            b = min(T, a + CH)
            st = draw_stream(b - a)
            take = _first_occurrence(st)
            cs = np.cumsum(take, axis=-1)
            take &= cs <= k
            got = take.sum(-1)
            comb = np.where(take, st, -1)
            order = np.argsort(comb < 0, axis=-1, kind="stable")[..., :k]
            sel = np.take_along_axis(comb, order, axis=-1)
            for t, ly in zip(*np.nonzero(got < k)):
                have = sel[t, ly][sel[t, ly] >= 0]
                sel[t, ly] = np.concatenate([have, exact_fill(ly, have, k - len(have))])
            ids[a:b] = finish_rows(sel)
        return Trace(ids, E, n_layers or L, source=f"synthetic(zipf={zipf},reuse={reuse},seed={seed})")

    prev = None
    rows = np.arange(L)
    for a in range(0, T, CH):
        b = min(T, a + CH)
        st_all = draw_stream(b - a)
        first_all = _first_occurrence(st_all)
        keep_all = rng.random((b - a, L, k)) < reuse
        for t in range(b - a):
            st, first = st_all[t], first_all[t]
            if prev is None:
                forced = np.full((L, k), -1, dtype=np.int64)
            else:
                forced = np.where(keep_all[t], prev, -1)
            need = k - (forced >= 0).sum(axis=1)
            in_forced = (st[:, :, None] == forced[:, None, :]).any(-1)
            valid = first & ~in_forced
            take = valid & (np.cumsum(valid, axis=1) <= need[:, None])
            comb = np.concatenate([forced, np.where(take, st, -1)], axis=1)
            order = np.argsort(comb < 0, axis=1, kind="stable")[:, :k]
            sel = np.take_along_axis(comb, order, axis=1)
            for ly in rows[take.sum(1) < need]:
                have = sel[ly][sel[ly] >= 0]
                sel[ly] = np.concatenate([have, exact_fill(ly, have, k - len(have))])
            sel = finish_rows(sel)
            ids[a + t] = sel
            prev = sel.astype(np.int64)
    return Trace(ids, E, n_layers or L, source=f"synthetic(zipf={zipf},reuse={reuse},seed={seed})")


def synthetic_for(shape, n_tokens: int = 2000, *, zipf: float = 1.1, reuse: float = 0.25, seed: int = 0) -> Trace:
    """Synthetic trace with a preset's geometry (n_moe_layers, n_experts, top_k)."""
    return synthetic(n_tokens, shape.n_moe_layers, shape.n_experts, shape.top_k,
                     zipf=zipf, reuse=reuse, seed=seed, n_layers=shape.n_layers)


# ---- usage (heat) profiles, FORMAT.md sec. 8 ---------------------------------

def load_usage(path) -> tuple[np.ndarray, int]:
    """Returns (heat [n_layers, n_experts] float64, tokens_observed)."""
    raw = Path(path).read_bytes()
    if len(raw) < 24:
        raise ValueError(f"{path}: too short for a usage profile")
    magic, ver, n_layers, n_experts = struct.unpack_from("<4I", raw, 0)
    (tokens,) = struct.unpack_from("<Q", raw, 16)
    if magic != USAGE_MAGIC or ver != 1:
        raise ValueError(f"{path}: not a version-1 HRUS usage profile")
    n = n_layers * n_experts
    if len(raw) != 24 + 4 * n:
        raise ValueError(f"{path}: expected {24 + 4 * n} bytes, got {len(raw)}")
    heat = np.frombuffer(raw, dtype="<f4", offset=24).astype(np.float64).reshape(n_layers, n_experts)
    return heat, int(tokens)


def save_usage(path, heat: np.ndarray, tokens_observed: int = 0) -> Path:
    heat = np.asarray(heat, dtype="<f4")
    L, E = heat.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<4IQ", USAGE_MAGIC, 1, L, E, int(tokens_observed)))
        f.write(heat.tobytes())
    return Path(path)
