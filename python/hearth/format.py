"""The .hearth container (docs/FORMAT.md): streaming writer and validating reader."""
from __future__ import annotations

import math
import operator
import os
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import numpy as np

from . import quant
from .quant import BF16, F16, F32, I32, Q4, Q8, U8

MAGIC = 0x48545248
VERSION = 1
SLAB_ALIGN = 4096
TENSOR_ALIGN = 64
PREAMBLE_BYTES = 64
TDIR_ENTRY = 128
EDIR_ENTRY = 32
NAME_LEN = 80
FLAG_ALIASED = 1

T_U32, T_F32, T_U64, T_STR, T_U32A, T_F32A, T_U8A = 1, 2, 3, 4, 5, 6, 7
_TYPE_NAMES = {T_U32: "u32", T_F32: "f32", T_U64: "u64", T_STR: "str",
               T_U32A: "u32[]", T_F32A: "f32[]", T_U8A: "u8[]"}

_U32_KEYS = """n_layers d_model vocab_size max_seq attn_kind n_heads n_kv_heads head_dim qk_norm qkv_bias
q_lora_rank kv_lora_rank qk_nope_dim qk_rope_dim v_head_dim rope_dim rope_style dense_ffn_dim n_experts top_k
expert_ffn_dim shared_ffn_dim shared_gate score_fn score_bias n_group topk_group norm_topk_prob tie_embeddings
bos_id expert_dtype""".split()
_F32_KEYS = "norm_eps rope_attn_factor attn_scale routed_scale emb_scale residual_scale logit_scale".split()
_STR_KEYS = "arch tokenizer chat_template source".split()

# FORMAT.md §3.1 key -> type
SCHEMA: dict[str, int] = {**{k: T_U32 for k in _U32_KEYS}, **{k: T_F32 for k in _F32_KEYS},
                          **{k: T_STR for k in _STR_KEYS}, "layer_kind": T_U8A, "eos_ids": T_U32A}

REQUIRED = ("n_layers", "d_model", "vocab_size", "n_heads")
NO_BOS = 0xFFFFFFFF

# Acceptance limits shared with the C reader (engine/src/modelfile.c, hx_modelfile.h).
MAX_LAYERS = 512
META_MAX = 1 << 30
TENSORS_MAX = 1 << 20
TOKENIZER_MAX = 259
_DIM_MAX, _FFN_MAX, _VOCAB_MAX, _SEQ_MAX, _HEADS_MAX, _EXPERTS_MAX = 1 << 20, 1 << 22, 1 << 27, 1 << 28, 65536, 65536

__all__ = ["ContainerWriter", "ContainerReader", "TensorInfo", "ExpertEntry", "slab_layout",
           "encode_meta", "decode_meta", "apply_defaults", "canonical_tensors", "SCHEMA", "MAGIC", "VERSION",
           "FLAG_ALIASED", "MAX_LAYERS"]


def _align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def slab_layout(dtype: int, D: int, F: int) -> tuple[int, int, int, int]:
    """(off_gate, off_up, off_down, padded nbytes) of one expert slab (FORMAT.md §5)."""
    rd, rf = quant.row_bytes(dtype, D), quant.row_bytes(dtype, F)
    off_gate = 0
    off_up = _align(off_gate + F * rd, 64)
    off_down = _align(off_up + F * rd, 64)
    end = off_down + D * rf
    return off_gate, off_up, off_down, _align(end, SLAB_ALIGN)


# ---------------------------------------------------------------- metadata

def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _is_float(v) -> bool:
    return isinstance(v, (float, np.floating))


def _infer_type(key: str, v) -> int:
    if key in SCHEMA:
        return SCHEMA[key]
    if isinstance(v, (bool, np.bool_)):
        return T_U32
    if _is_int(v):
        if 0 <= int(v) <= 0xFFFFFFFF:
            return T_U32
        if 0 <= int(v) < 1 << 64:
            return T_U64
        raise ValueError(f"metadata {key!r}: integer {v} does not fit u32/u64")
    if _is_float(v):
        return T_F32
    if isinstance(v, str):
        return T_STR
    if isinstance(v, (bytes, bytearray)):
        return T_U8A
    if isinstance(v, np.ndarray):
        if v.dtype == np.uint8:
            return T_U8A
        if np.issubdtype(v.dtype, np.integer):
            return T_U32A
        if np.issubdtype(v.dtype, np.floating):
            return T_F32A
    if isinstance(v, (list, tuple)):
        if all(_is_int(x) or isinstance(x, (bool, np.bool_)) for x in v):
            return T_U32A
        if all(_is_int(x) or _is_float(x) for x in v):
            return T_F32A
    raise ValueError(f"metadata {key!r}: cannot encode value of type {type(v).__name__}")


def _enc_value(key: str, t: int, v) -> bytes:
    try:
        if t == T_U32:
            iv = int(v)
            if key == "bos_id" and iv < 0:
                iv = NO_BOS
            if not 0 <= iv <= 0xFFFFFFFF:
                raise ValueError(f"{iv} out of u32 range")
            return struct.pack("<I", iv)
        if t == T_U64:
            return struct.pack("<Q", int(v))
        if t == T_F32:
            return struct.pack("<f", float(v))
        if t == T_STR:
            b = str(v).encode("utf-8")
            return struct.pack("<I", len(b)) + b
        if t == T_U32A:
            a = np.asarray(list(v) if not isinstance(v, np.ndarray) else v, dtype=np.int64).reshape(-1)
            if a.size and (a.min() < 0 or a.max() > 0xFFFFFFFF):
                raise ValueError("element out of u32 range")
            return struct.pack("<I", a.size) + a.astype("<u4").tobytes()
        if t == T_F32A:
            a = np.asarray(v, dtype="<f4").reshape(-1)
            return struct.pack("<I", a.size) + a.tobytes()
        if t == T_U8A:
            a = np.frombuffer(bytes(v), dtype=np.uint8) if isinstance(v, (bytes, bytearray)) \
                else np.asarray(v, dtype=np.int64).reshape(-1)
            if a.size and (a.min() < 0 or a.max() > 255):
                raise ValueError("element out of u8 range")
            return struct.pack("<I", a.size) + a.astype(np.uint8).tobytes()
    except (TypeError, ValueError, struct.error, OverflowError) as e:
        raise ValueError(f"metadata {key!r}: {e}") from None
    raise ValueError(f"metadata {key!r}: unknown type {t}")


def encode_meta(meta: dict) -> bytes:
    out = bytearray()
    for key, v in meta.items():
        if v is None:
            continue
        kb = str(key).encode("ascii")
        if not kb or len(kb) > 0xFFFF:
            raise ValueError(f"bad metadata key {key!r}")
        t = _infer_type(key, v)
        out += struct.pack("<H", len(kb)) + kb + struct.pack("<B", t) + _enc_value(key, t, v)
    return bytes(out)


def decode_meta(buf: bytes) -> dict:
    """Parse a metadata section; raises ValueError on any malformed entry."""
    meta, p, n = {}, 0, len(buf)

    def take(k):
        nonlocal p
        if p + k > n:
            raise ValueError("metadata entry runs past the end of the metadata section")
        b = buf[p:p + k]
        p += k
        return b

    while p < n:
        (klen,) = struct.unpack("<H", take(2))
        kb = take(klen)
        try:
            key = kb.decode("ascii")
        except UnicodeDecodeError:
            raise ValueError("metadata key is not ASCII") from None
        (t,) = struct.unpack("<B", take(1))
        if t == T_U32:
            v = struct.unpack("<I", take(4))[0]
        elif t == T_F32:
            v = struct.unpack("<f", take(4))[0]
        elif t == T_U64:
            v = struct.unpack("<Q", take(8))[0]
        elif t == T_STR:
            (ln,) = struct.unpack("<I", take(4))
            try:
                v = take(ln).decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError(f"metadata {key!r}: invalid UTF-8") from None
        elif t in (T_U32A, T_F32A, T_U8A):
            (cnt,) = struct.unpack("<I", take(4))
            width = 1 if t == T_U8A else 4
            raw = take(cnt * width)
            dt = {T_U32A: "<u4", T_F32A: "<f4", T_U8A: "u1"}[t]
            v = np.frombuffer(raw, dtype=dt).tolist()
        else:
            raise ValueError(f"metadata {key!r}: unknown type {t}")
        want = SCHEMA.get(key)
        if want is not None and t != want:
            raise ValueError(f"metadata {key!r} has type {_TYPE_NAMES[t]}, expected {_TYPE_NAMES[want]}")
        meta[key] = v
    return meta


def _as_int(key: str, v) -> int:
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    try:
        return operator.index(v)
    except TypeError:
        raise ValueError(f"metadata {key!r} must be an integer, got {type(v).__name__}") from None


def _int_list(key: str, v) -> list[int]:
    if not isinstance(v, (list, tuple, bytes, bytearray, np.ndarray)):
        raise ValueError(f"metadata {key!r} must be an array, got {type(v).__name__}")
    return [_as_int(key, x) for x in (v.reshape(-1).tolist() if isinstance(v, np.ndarray) else v)]


def apply_defaults(meta: dict) -> dict:
    """FORMAT.md §3.1 defaults plus the C reader's acceptance rules (engine/src/modelfile.c
    build_config): value ranges, head/MLA/rope/routing consistency, special-token ids.
    Raises ValueError for anything the C reader would reject."""
    for k in REQUIRED:
        if k not in meta:
            raise ValueError(f"metadata is missing required key {k!r}")
    c = dict(meta)

    def u32(key, default, lo, hi):
        v = _as_int(key, c[key]) if key in c else default
        if not lo <= v <= hi:
            raise ValueError(f"metadata {key} = {v} out of range [{lo}, {hi}]" + ("" if key in c else " (default)"))
        c[key] = v
        return v

    def f32(key, default):
        v = c.get(key, default)
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise ValueError(f"metadata {key!r} must be a number, got {type(v).__name__}")
        with np.errstate(over="ignore"):
            v = np.float32(v)
        if not np.isfinite(v):
            raise ValueError(f"metadata {key} is not finite")
        c[key] = float(v)
        return c[key]

    L = u32("n_layers", 0, 1, MAX_LAYERS)
    D = u32("d_model", 0, 1, _DIM_MAX)
    V = u32("vocab_size", 0, 1, _VOCAB_MAX)
    u32("max_seq", 4096, 1, _SEQ_MAX)
    if f32("norm_eps", 1e-6) < 0.0:
        raise ValueError("metadata norm_eps is negative")
    mla = u32("attn_kind", 0, 0, 1) == 1
    H = u32("n_heads", 0, 1, _HEADS_MAX)
    Hkv = u32("n_kv_heads", H, 1, _HEADS_MAX)
    hd = u32("head_dim", D // H, 0, _HEADS_MAX)
    u32("qk_norm", 0, 0, 2)
    u32("qkv_bias", 0, 0, 1)
    u32("q_lora_rank", 0, 0, _DIM_MAX)
    C = u32("kv_lora_rank", 0, 0, _DIM_MAX)
    nope, rd = u32("qk_nope_dim", 0, 0, _HEADS_MAX), u32("qk_rope_dim", 0, 0, _HEADS_MAX)
    vd = u32("v_head_dim", 0, 0, _HEADS_MAX)
    if mla:
        if C < 1 or vd < 1 or nope + rd < 1:
            raise ValueError("MLA needs kv_lora_rank, v_head_dim and qk_nope_dim + qk_rope_dim >= 1")
        qk_dim = nope + rd
    else:
        if hd < 1:
            raise ValueError(f"head_dim is 0 (d_model {D} < n_heads {H} and no head_dim key)")
        if H % Hkv:
            raise ValueError(f"n_heads {H} is not a multiple of n_kv_heads {Hkv}")
        qk_dim = hd
    cap = rd if mla else hd
    if u32("rope_dim", cap, 0, cap) % 2:
        raise ValueError(f"rope_dim {c['rope_dim']} is odd")
    u32("rope_style", 0, 0, 1)
    f32("rope_attn_factor", 1.0)
    f32("attn_scale", np.float32(1.0) / np.sqrt(np.float32(qk_dim)))
    dense_ffn = u32("dense_ffn_dim", 0, 0, _FFN_MAX)
    E = u32("n_experts", 0, 0, _EXPERTS_MAX)
    K = u32("top_k", 0, 0, _EXPERTS_MAX)
    F = u32("expert_ffn_dim", 0, 0, _FFN_MAX)
    u32("shared_ffn_dim", 0, 0, _FFN_MAX)
    for key in ("shared_gate", "score_fn", "score_bias"):
        u32(key, 0, 0, 1)
    ng, tg = u32("n_group", 1, 1, _EXPERTS_MAX), u32("topk_group", 1, 1, _EXPERTS_MAX)
    u32("norm_topk_prob", 0, 0, 1)
    for key in ("routed_scale", "emb_scale", "residual_scale", "logit_scale"):
        f32(key, 1.0)
    u32("tie_embeddings", 0, 0, 1)

    lk = _int_list("layer_kind", c["layer_kind"]) if "layer_kind" in c else [1 if E > 0 else 0] * L
    if len(lk) != L:
        raise ValueError(f"layer_kind has {len(lk)} entries for {L} layers")
    if any(k not in (0, 1) for k in lk):
        raise ValueError("layer_kind entries must be 0 or 1")
    c["layer_kind"] = lk
    n_moe = c["n_moe_layers"] = sum(lk)
    if n_moe:
        if E < 1:
            raise ValueError(f"layer_kind marks {n_moe} MoE layers but n_experts is 0")
        if not 1 <= K <= E:
            raise ValueError(f"top_k {K} out of range [1, n_experts {E}]")
        if F < 1:
            raise ValueError("MoE layers need expert_ffn_dim >= 1")
        if E % ng:
            raise ValueError(f"n_experts {E} is not a multiple of n_group {ng}")
        if tg > ng:
            raise ValueError(f"topk_group {tg} > n_group {ng}")
        if K > tg * (E // ng):
            raise ValueError(f"top_k {K} exceeds the experts in topk_group {tg} groups")
    if n_moe < L and dense_ffn < 1:
        raise ValueError("dense layers need dense_ffn_dim >= 1")

    bos = _as_int("bos_id", c.get("bos_id", NO_BOS))
    bos = NO_BOS if bos < 0 else bos            # the writer encodes a negative bos_id as "none"
    if bos != NO_BOS and bos >= V:
        raise ValueError(f"bos_id {bos} >= vocab_size {V}")
    c["bos_id"] = bos
    eos = _int_list("eos_ids", c.get("eos_ids", []))
    for i, t in enumerate(eos):
        if not 0 <= t < V:
            raise ValueError(f"eos_ids[{i}] = {t} outside the vocabulary of {V}")
    c["eos_ids"] = eos
    for key in ("arch", "tokenizer", "chat_template", "source"):
        c.setdefault(key, "")
        if not isinstance(c[key], str):
            raise ValueError(f"metadata {key!r} must be a string, got {type(c[key]).__name__}")
    if len(c["tokenizer"].encode("utf-8")) > TOKENIZER_MAX or "\0" in c["tokenizer"]:
        raise ValueError(f"tokenizer path must be at most {TOKENIZER_MAX} bytes without NUL")
    return c


def canonical_tensors(c: dict) -> dict[str, tuple[tuple[int, ...], bool]]:
    """FORMAT.md §4.1 for a config from apply_defaults: name -> (shape, must be F32) of every
    tensor the forward pass reads (rope_inv_freq only when rope_dim > 0; MLA ignores
    qkv_bias/qk_norm; shexp_gate_inp only when a gated shared expert exists)."""
    D, V, H = c["d_model"], c["vocab_size"], c["n_heads"]
    t: dict[str, tuple[tuple[int, ...], bool]] = {"tok_embd": ((V, D), False), "out_norm": ((D,), True)}
    if not c["tie_embeddings"]:
        t["lm_head"] = ((V, D), False)
    if c["rope_dim"] > 0:
        t["rope_inv_freq"] = ((c["rope_dim"] // 2,), True)
    Fd, Fs, E = c["dense_ffn_dim"], c["shared_ffn_dim"], c["n_experts"]
    for i in range(c["n_layers"]):
        p = f"blk.{i}."
        t[p + "attn_norm"] = ((D,), True)
        t[p + "ffn_norm"] = ((D,), True)
        if c["attn_kind"] == 1:
            nope, rd, vd, C, ql = (c["qk_nope_dim"], c["qk_rope_dim"], c["v_head_dim"], c["kv_lora_rank"],
                                   c["q_lora_rank"])
            if ql:
                t[p + "attn_q_a"] = ((ql, D), False)
                t[p + "attn_q_a_norm"] = ((ql,), True)
                t[p + "attn_q_b"] = ((H * (nope + rd), ql), False)
            else:
                t[p + "attn_q"] = ((H * (nope + rd), D), False)
            t[p + "attn_kv_a"] = ((C + rd, D), False)
            t[p + "attn_kv_a_norm"] = ((C,), True)
            t[p + "attn_kv_b"] = ((H * (nope + vd), C), False)
            t[p + "attn_o"] = ((D, H * vd), False)
        else:
            hd, Hkv = c["head_dim"], c["n_kv_heads"]
            t[p + "attn_q"] = ((H * hd, D), False)
            t[p + "attn_k"] = ((Hkv * hd, D), False)
            t[p + "attn_v"] = ((Hkv * hd, D), False)
            t[p + "attn_o"] = ((D, H * hd), False)
            if c["qkv_bias"]:
                t[p + "attn_q_bias"] = ((H * hd,), True)
                t[p + "attn_k_bias"] = ((Hkv * hd,), True)
                t[p + "attn_v_bias"] = ((Hkv * hd,), True)
            if c["qk_norm"]:
                t[p + "attn_q_norm"] = ((hd,) if c["qk_norm"] == 1 else (H * hd,), True)
                t[p + "attn_k_norm"] = ((hd,) if c["qk_norm"] == 1 else (Hkv * hd,), True)
        if c["layer_kind"][i] == 0:
            t[p + "ffn_gate"] = ((Fd, D), False)
            t[p + "ffn_up"] = ((Fd, D), False)
            t[p + "ffn_down"] = ((D, Fd), False)
            continue
        t[p + "moe_router"] = ((E, D), True)
        if c["score_bias"]:
            t[p + "moe_router_bias"] = ((E,), True)
        if Fs:
            t[p + "shexp_gate"] = ((Fs, D), False)
            t[p + "shexp_up"] = ((Fs, D), False)
            t[p + "shexp_down"] = ((D, Fs), False)
            if c["shared_gate"]:
                t[p + "shexp_gate_inp"] = ((1, D), False)
    return t


def _byte_view(data, what: str) -> memoryview:
    """A flat unsigned-byte view of a bytes-like object (len() of a typed memoryview counts items)."""
    mv = memoryview(data)
    if not mv.c_contiguous:
        raise ValueError(f"{what}: buffer is not C-contiguous")
    return mv.cast("B")


# ---------------------------------------------------------------- writer

@dataclass
class _TDecl:
    name: str
    dtype: int
    shape: tuple
    nbytes: int
    offset: int = 0
    written: bool = False
    next_row: int = 0


class ContainerWriter:
    """Streaming .hearth writer: declare everything, then write tensors/slabs in any order
    (positional writes into a pre-sized file), then close()."""

    def __init__(self, path, meta: dict):
        self.path = Path(path)
        self.meta = dict(meta)
        self._tensors: dict[str, _TDecl] = {}
        self._exp: dict | None = None
        self._frozen = False
        self._closed = False
        self._f = open(self.path, "w+b")

    # -- declarations
    def declare_tensor(self, name: str, shape: tuple, dtype: int) -> None:
        self._check_open(declaring=True)
        if not isinstance(name, str) or not name or len(name.encode("ascii")) > NAME_LEN - 1:
            raise ValueError(f"tensor name {name!r} must be 1..{NAME_LEN - 1} ASCII chars")
        if name in self._tensors:
            raise ValueError(f"tensor {name!r} declared twice")
        shape = tuple(int(s) for s in shape)
        if not 1 <= len(shape) <= 4 or any(s < 1 or s > 0xFFFFFFFF for s in shape):
            raise ValueError(f"tensor {name!r}: bad shape {shape}")
        dtype = quant.dtype_of(dtype)
        rows = int(np.prod(shape[:-1], dtype=np.int64)) if len(shape) > 1 else 1
        nbytes = rows * quant.row_bytes(dtype, shape[-1])
        self._tensors[name] = _TDecl(name, dtype, shape, nbytes)

    def declare_experts(self, n_layers: int, n_experts: int, d_model: int, ffn_dim: int,
                        dtype: int | Callable[[int, int], int], moe_layers: Sequence[int],
                        alias: Callable[[int, int], tuple[int, int] | None] | None = None) -> None:
        self._check_open(declaring=True)
        if self._exp is not None:
            raise ValueError("declare_experts called twice")
        L, E, D, F = int(n_layers), int(n_experts), int(d_model), int(ffn_dim)
        if L <= 0 or E <= 0 or D <= 0 or F <= 0:
            raise ValueError("declare_experts: all sizes must be positive")
        for key, val in (("n_layers", L), ("n_experts", E), ("d_model", D), ("expert_ffn_dim", F)):
            if key in self.meta and int(self.meta[key]) != val:
                raise ValueError(f"declare_experts: {key}={val} but metadata says {self.meta[key]}")
        moe = sorted(set(int(i) for i in moe_layers))
        if any(not 0 <= i < L for i in moe):
            raise ValueError("declare_experts: moe layer index out of range")
        dt = {}
        for li in moe:
            for e in range(E):
                d = quant.dtype_of(dtype(li, e) if callable(dtype) else dtype)
                if d in (I32, U8):
                    raise ValueError("expert slabs must use a float or block-quant dtype")
                dt[(li, e)] = d
        target: dict[tuple[int, int], tuple[int, int]] = {}
        if alias is not None:
            for li in moe:
                for e in range(E):
                    a = alias(li, e)
                    if a is None or tuple(a) == (li, e):
                        continue
                    a = (int(a[0]), int(a[1]))
                    if a not in dt:
                        raise ValueError(f"expert ({li},{e}) aliases {a}, which is not a routed expert")
                    target[(li, e)] = a
            for k, a in target.items():
                if a in target:
                    raise ValueError(f"expert {k} aliases {a}, which is itself an alias")
        self._exp = {"L": L, "E": E, "D": D, "F": F, "moe": moe, "dtype": dt, "alias": target,
                     "slabs": {}, "written": set()}

    # -- layout
    def _check_consistency(self) -> None:
        """Refuse metadata/expert declarations the reader would reject."""
        c = apply_defaults({k: v for k, v in self.meta.items() if v is not None})   # as encode_meta sees it
        moe = [i for i, k in enumerate(c["layer_kind"]) if k == 1]
        if moe:                                    # apply_defaults guarantees n_experts >= 1 then
            if self._exp is None:
                raise ValueError("metadata declares MoE layers but declare_experts was not called")
            if self._exp["moe"] != moe:
                raise ValueError(f"declare_experts moe_layers {self._exp['moe']} != layer_kind MoE layers {moe}")
        if self._exp is not None and (self._exp["L"], self._exp["E"]) != (c["n_layers"], c["n_experts"]):
            raise ValueError(f"declare_experts ({self._exp['L']} layers x {self._exp['E']} experts) disagrees "
                             f"with metadata n_layers/n_experts ({c['n_layers']}, {c['n_experts']})")

    def _freeze(self) -> None:
        if self._frozen:
            return
        self._check_consistency()
        meta_b = encode_meta(self.meta)
        self._meta_b = meta_b
        meta_off = PREAMBLE_BYTES
        tdir_off = meta_off + len(meta_b)
        n_entries = self._exp["L"] * self._exp["E"] if self._exp else 0
        edir_off = tdir_off + len(self._tensors) * TDIR_ENTRY
        pos = _align(edir_off + n_entries * EDIR_ENTRY, TENSOR_ALIGN)
        for t in self._tensors.values():
            t.offset = pos
            pos = _align(pos + t.nbytes, TENSOR_ALIGN)
        end = pos if not self._tensors else max(t.offset + t.nbytes for t in self._tensors.values())
        if self._exp:
            pos = _align(end, SLAB_ALIGN)
            ex = self._exp
            for li in ex["moe"]:
                for e in range(ex["E"]):
                    if (li, e) in ex["alias"]:
                        continue
                    lay = slab_layout(ex["dtype"][(li, e)], ex["D"], ex["F"])
                    ex["slabs"][(li, e)] = (pos, lay)
                    pos += lay[3]
            end = max(end, pos)
        self._layout = (meta_off, tdir_off, edir_off, n_entries)
        self._size = end
        self._f.truncate(end)
        self._frozen = True

    def _check_open(self, declaring=False):
        if self._closed:
            raise ValueError("writer is closed")
        if declaring and self._frozen:
            raise ValueError("cannot declare after the first write")

    def _pwrite(self, off: int, data) -> None:
        self._f.seek(off)
        self._f.write(data)

    # -- data
    def write_tensor(self, name: str, data) -> None:
        self._check_open()
        if name not in self._tensors:
            raise KeyError(f"tensor {name!r} was not declared")
        self._freeze()
        t = self._tensors[name]
        buf = self._encode(data, t.dtype, t.shape, f"tensor {name!r}")
        self._pwrite(t.offset, buf)
        t.written = True

    def write_tensor_rows(self, name: str, row0: int, data) -> None:
        """Stream a big tensor in row chunks (2-D view [rows, cols]); chunks must arrive in
        order. The tensor counts as written once its last row is in."""
        self._check_open()
        if name not in self._tensors:
            raise KeyError(f"tensor {name!r} was not declared")
        self._freeze()
        t = self._tensors[name]
        rows = int(np.prod(t.shape[:-1], dtype=np.int64)) if len(t.shape) > 1 else 1
        cols, rb = t.shape[-1], quant.row_bytes(t.dtype, t.shape[-1])
        if int(row0) != t.next_row:
            raise ValueError(f"tensor {name!r}: chunk starts at row {row0}, expected {t.next_row}")
        if isinstance(data, (bytes, bytearray, memoryview)):
            data = _byte_view(data, f"tensor {name!r}")
            n = data.nbytes // rb
        else:
            n = int(np.asarray(data).size) // cols
        if n <= 0 or row0 + n > rows:
            raise ValueError(f"tensor {name!r}: bad chunk of {n} rows at {row0} (tensor has {rows})")
        buf = self._encode(data, t.dtype, (n, cols), f"tensor {name!r} rows {row0}..{row0 + n}")
        self._pwrite(t.offset + row0 * rb, buf)
        t.next_row = row0 + n
        t.written = t.next_row == rows

    @staticmethod
    def _encode(data, dtype: int, shape: tuple, what: str):
        rows = int(np.prod(shape[:-1], dtype=np.int64)) if len(shape) > 1 else 1
        nbytes = rows * quant.row_bytes(dtype, shape[-1])
        if isinstance(data, (bytes, bytearray, memoryview)):
            b = _byte_view(data, what)
            if b.nbytes != nbytes:
                raise ValueError(f"{what}: got {b.nbytes} bytes, expected {nbytes}")
            return b
        a = np.asarray(data)
        if a.size != rows * shape[-1] or (a.ndim > 1 and a.shape[-1] != shape[-1]):
            raise ValueError(f"{what}: got array of shape {a.shape}, expected {shape}")
        if dtype in (I32, U8):
            return quant.quantize(a.reshape(rows, shape[-1]), dtype)
        if not np.issubdtype(a.dtype, np.floating):
            raise ValueError(f"{what}: expected a float array for dtype {quant.NAMES[dtype]}")
        enc = quant.quantize(a.reshape(rows, shape[-1]).astype(np.float32, copy=False), dtype)
        if len(enc) != nbytes:
            raise ValueError(f"{what}: encoder produced {len(enc)} bytes, expected {nbytes}")
        return enc

    def write_expert(self, layer: int, expert: int, gate, up, down) -> None:
        self._check_open()
        if self._exp is None:
            raise ValueError("no experts declared")
        key = (int(layer), int(expert))
        ex = self._exp
        if key not in ex["dtype"]:
            raise KeyError(f"expert {key} was not declared (layer not MoE or index out of range)")
        if key in ex["alias"]:
            raise ValueError(f"expert {key} is an alias of {ex['alias'][key]}; write the target instead")
        self._freeze()
        off, (og, ou, od, nb) = ex["slabs"][key]
        dt, D, F = ex["dtype"][key], ex["D"], ex["F"]
        slab = bytearray(nb)
        for o, m, shape, nm in ((og, gate, (F, D), "gate"), (ou, up, (F, D), "up"), (od, down, (D, F), "down")):
            b = self._encode(m, dt, shape, f"expert {key} {nm}")
            slab[o:o + len(b)] = b
        self._pwrite(off, slab)
        ex["written"].add(key)

    # -- finish
    def close(self) -> None:
        if self._closed:
            return
        try:
            self._freeze()
            missing = [n for n, t in self._tensors.items() if not t.written]
            if missing:
                raise ValueError(f"tensors declared but never written: {missing[:8]}"
                                 + (" ..." if len(missing) > 8 else ""))
            if self._exp:
                miss = [k for k in self._exp["slabs"] if k not in self._exp["written"]]
                if miss:
                    raise ValueError(f"{len(miss)} expert slabs never written, e.g. {miss[:4]}")
            meta_off, tdir_off, edir_off, n_entries = self._layout
            tdir = bytearray()
            for t in self._tensors.values():
                shp = list(t.shape) + [1] * (4 - len(t.shape))
                tdir += struct.pack("<80sII4IQQQ", t.name.encode("ascii"), t.dtype, len(t.shape),
                                    *shp, t.offset, t.nbytes, 0)
            edir = bytearray()
            if self._exp:
                ex = self._exp
                for li in range(ex["L"]):
                    for e in range(ex["E"]):
                        k = (li, e)
                        if k in ex["alias"]:
                            off, lay = ex["slabs"][ex["alias"][k]]
                            edir += struct.pack("<QQIIQ", off, lay[3], ex["dtype"][ex["alias"][k]], FLAG_ALIASED, 0)
                        elif k in ex["slabs"]:
                            off, lay = ex["slabs"][k]
                            edir += struct.pack("<QQIIQ", off, lay[3], ex["dtype"][k], 0, 0)
                        else:
                            edir += struct.pack("<QQIIQ", 0, 0, 0, 0, 0)
            pre = struct.pack("<IIQQQQQQII", MAGIC, VERSION, meta_off, len(self._meta_b), tdir_off,
                              len(self._tensors), edir_off, n_entries, SLAB_ALIGN, 0)
            assert len(pre) == PREAMBLE_BYTES
            self._pwrite(meta_off, self._meta_b)
            self._pwrite(tdir_off, bytes(tdir))
            self._pwrite(edir_off, bytes(edir))
            self._f.flush()
            self._pwrite(0, pre)       # preamble last: an interrupted write never looks valid
            self._f.flush()
            os.fsync(self._f.fileno())
        except BaseException:
            self.abort()
            raise
        self._f.close()
        self._closed = True

    def abort(self) -> None:
        """Close without finalizing and delete the partial file."""
        if self._closed:
            return
        self._closed = True
        try:
            self._f.close()
        finally:
            try:
                self.path.unlink()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            self.close()
        else:
            self.abort()
        return False


# ---------------------------------------------------------------- reader

@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: int
    shape: tuple
    offset: int
    nbytes: int


class ExpertEntry(NamedTuple):
    offset: int
    nbytes: int
    dtype: int
    flags: int


class ContainerReader:
    """Parses and validates a .hearth file; reads tensors and expert slabs on demand."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._f = open(self.path, "rb")
        try:
            self._parse()
        except BaseException as e:
            self._f.close()
            if isinstance(e, (struct.error, OverflowError, UnicodeDecodeError, TypeError)):
                raise ValueError(f"{self.path}: malformed container ({e})") from None
            raise

    def _read_at(self, off: int, n: int) -> bytes:
        if off < 0 or n < 0 or off + n > self.file_size:
            raise ValueError(f"read [{off}, {off + n}) outside file of {self.file_size} bytes")
        with self._lock:
            self._f.seek(off)
            b = self._f.read(n)
        if len(b) != n:
            raise ValueError("short read (file changed or truncated)")
        return b

    def _parse(self) -> None:
        self._f.seek(0, os.SEEK_END)
        size = self.file_size = self._f.tell()
        if size < PREAMBLE_BYTES:
            raise ValueError(f"{self.path}: file too small ({size} bytes) to be a .hearth container")
        (magic, ver, meta_off, meta_bytes, tdir_off, n_t, edir_off, n_e, align, _res) = \
            struct.unpack("<IIQQQQQQII", self._read_at(0, PREAMBLE_BYTES))
        if magic != MAGIC:
            raise ValueError(f"{self.path}: bad magic 0x{magic:08x}")
        if ver != VERSION:
            raise ValueError(f"{self.path}: unsupported version {ver}")
        if align != SLAB_ALIGN:
            raise ValueError(f"{self.path}: slab alignment {align} != {SLAB_ALIGN}")

        def region(off, n, what):
            if off + n > size:                      # Python ints: no overflow, same as C's in_file
                raise ValueError(f"{self.path}: {what} [{off}, +{n}) lies outside the file ({size} bytes)")

        region(meta_off, meta_bytes, "metadata")
        if meta_bytes > META_MAX:
            raise ValueError(f"{self.path}: metadata section of {meta_bytes} bytes is implausibly large")
        if n_t > size // TDIR_ENTRY or n_t > TENSORS_MAX:
            raise ValueError(f"{self.path}: tensor count {n_t} impossible for file size")
        region(tdir_off, n_t * TDIR_ENTRY, "tensor directory")
        if n_e > size // EDIR_ENTRY:
            raise ValueError(f"{self.path}: expert entry count {n_e} impossible for file size")
        region(edir_off, n_e * EDIR_ENTRY, "expert directory")

        self.meta = decode_meta(self._read_at(meta_off, meta_bytes))
        self.config = apply_defaults(self.meta)
        c = self.config

        self.tensors: dict[str, TensorInfo] = {}
        raw = self._read_at(tdir_off, n_t * TDIR_ENTRY)
        for i in range(n_t):
            name_b, dt, ndim, s0, s1, s2, s3, off, nb, _r = struct.unpack_from("<80sII4IQQQ", raw, i * TDIR_ENTRY)
            if b"\0" not in name_b:
                raise ValueError(f"{self.path}: tensor entry {i}: name not NUL-terminated")
            name = name_b[:name_b.index(b"\0")].decode("ascii")
            if not name:
                raise ValueError(f"{self.path}: tensor entry {i}: empty name")
            if name in self.tensors:
                raise ValueError(f"{self.path}: duplicate tensor {name!r}")
            if dt not in quant.NAMES:
                raise ValueError(f"{self.path}: tensor {name!r}: unknown dtype {dt}")
            if not 1 <= ndim <= 4:
                raise ValueError(f"{self.path}: tensor {name!r}: ndim {ndim}")
            dims = (s0, s1, s2, s3)
            shape = dims[:ndim]
            if any(s < 1 for s in shape) or any(s != 1 for s in dims[ndim:]):
                raise ValueError(f"{self.path}: tensor {name!r}: bad shape {dims}")
            if off % TENSOR_ALIGN:
                raise ValueError(f"{self.path}: tensor {name!r}: offset {off} not 64-byte aligned")
            rows = math.prod(shape[:-1])
            try:
                want = rows * quant.row_bytes(dt, shape[-1])
            except ValueError as e:
                raise ValueError(f"{self.path}: tensor {name!r}: {e}") from None
            if nb != want:
                raise ValueError(f"{self.path}: tensor {name!r}: nbytes {nb} != {want}")
            region(off, nb, f"tensor {name!r}")
            self.tensors[name] = TensorInfo(name, dt, tuple(shape), off, nb)
        rf = self.tensors.get("rope_inv_freq")
        if rf is not None and c["rope_dim"] > 0 and (rf.dtype != F32 or rf.shape != (c["rope_dim"] // 2,)):
            raise ValueError(f"{self.path}: tensor 'rope_inv_freq' must be F32 [{c['rope_dim'] // 2}]")

        L, E = c["n_layers"], c["n_experts"]
        has_moe = E > 0 and c["n_moe_layers"] > 0
        if has_moe and n_e != L * E:
            raise ValueError(f"{self.path}: {n_e} expert entries, expected n_layers*n_experts = {L * E}")
        if not has_moe and n_e not in (0, L * E):
            raise ValueError(f"{self.path}: {n_e} expert entries for a model without routed experts")
        self._experts: list[ExpertEntry] = []
        raw = self._read_at(edir_off, n_e * EDIR_ENTRY)
        D, F = c["d_model"], c["expert_ffn_dim"]
        for i in range(n_e):
            off, nb, dt, fl, _r = struct.unpack_from("<QQIIQ", raw, i * EDIR_ENTRY)
            layer = i // E                          # n_e > 0 implies E > 0 (n_e == n_layers * E)
            if off % SLAB_ALIGN or nb % SLAB_ALIGN:
                raise ValueError(f"{self.path}: expert entry {i}: offset/size not 4096-aligned")
            region(off, nb, f"expert slab {i}")
            if c["layer_kind"][layer] == 1:
                if nb == 0:
                    raise ValueError(f"{self.path}: expert entry {i}: empty slab in MoE layer {layer}")
                if dt not in (F32, F16, BF16, Q8, Q4):
                    raise ValueError(f"{self.path}: expert entry {i}: bad dtype {dt}")
                try:
                    need = slab_layout(dt, D, F)[3]
                except ValueError as e:
                    raise ValueError(f"{self.path}: expert entry {i}: {e}") from None
                if nb < need:
                    raise ValueError(f"{self.path}: expert entry {i}: slab {nb} bytes < {need} needed")
            elif nb:
                raise ValueError(f"{self.path}: expert entry {i}: non-empty slab in dense layer {layer}")
            self._experts.append(ExpertEntry(off, nb, dt, fl))
        self._check_regions(((meta_off, meta_bytes, "metadata"), (tdir_off, n_t * TDIR_ENTRY, "tensor directory"),
                             (edir_off, n_e * EDIR_ENTRY, "expert directory")))

    def _check_regions(self, sections) -> None:
        """No two of preamble, sections, tensors and physical slabs overlap; entries sharing a
        slab agree on size and dtype and at most one lacks the aliased flag (as the C reader)."""
        regs = [(0, PREAMBLE_BYTES, "preamble")] + [(o, o + n, w) for o, n, w in sections if n]
        regs += [(t.offset, t.offset + t.nbytes, f"tensor {t.name!r}") for t in self.tensors.values()]
        groups: dict[int, list[int]] = {}
        for i, ent in enumerate(self._experts):
            if ent.nbytes:
                groups.setdefault(ent.offset, []).append(i)
        for off, idx in groups.items():
            first = self._experts[idx[0]]
            for j in idx[1:]:
                if self._experts[j][1:3] != first[1:3]:
                    raise ValueError(f"{self.path}: expert entries {idx[0]} and {j} share offset {off} "
                                     f"but differ in size or dtype")
            plain = [j for j in idx if not self._experts[j].flags & FLAG_ALIASED]
            if len(plain) > 1:
                raise ValueError(f"{self.path}: expert entries {plain[0]} and {plain[1]} point at the same "
                                 f"slab without the aliased flag")
            regs.append((off, off + first.nbytes, f"expert slab of entry {idx[0]}"))
        regs.sort(key=lambda r: r[0])        # regions are non-empty: equal starts always overlap
        m = 0
        for i in range(1, len(regs)):
            if regs[i][0] < regs[m][1]:
                raise ValueError(f"{self.path}: {regs[m][2]} overlaps {regs[i][2]}")
            if regs[i][1] > regs[m][1]:
                m = i

    # -- access
    def read_tensor(self, name: str, dequant: bool = True):
        t = self.tensors.get(name)
        if t is None:
            raise KeyError(name)
        b = self._read_at(t.offset, t.nbytes)
        return quant.dequantize(b, t.dtype, t.shape) if dequant else b

    def has(self, name: str) -> bool:
        return name in self.tensors

    def expert_entry(self, layer: int, e: int) -> ExpertEntry:
        E = self.config["n_experts"]
        if not (0 <= layer < self.config["n_layers"] and 0 <= e < E) or not self._experts:
            raise IndexError(f"no expert entry ({layer}, {e})")
        return self._experts[layer * E + e]

    def read_expert(self, layer: int, e: int, dequant: bool = True):
        ent = self.expert_entry(layer, e)
        if ent.nbytes == 0:
            raise ValueError(f"layer {layer} has no routed experts")
        D, F = self.config["d_model"], self.config["expert_ffn_dim"]
        og, ou, od, _ = slab_layout(ent.dtype, D, F)
        slab = self._read_at(ent.offset, ent.nbytes)
        rd, rf = quant.row_bytes(ent.dtype, D), quant.row_bytes(ent.dtype, F)
        parts = (slab[og:og + F * rd], slab[ou:ou + F * rd], slab[od:od + D * rf])
        if not dequant:
            return parts
        return (quant.dequantize(parts[0], ent.dtype, (F, D)), quant.dequantize(parts[1], ent.dtype, (F, D)),
                quant.dequantize(parts[2], ent.dtype, (D, F)))

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False

    def __del__(self):
        try:
            self._f.close()
        except Exception:
            pass
