"""Hugging Face checkpoint -> .hearth container (docs/FORMAT.md §4.1).

Supported model_type: qwen3_moe, olmoe, mixtral, qwen2_moe, deepseek_v3, kimi_k2
(the DeepseekV3 architecture). Tensors are read lazily from *.safetensors (through
safetensors.safe_open + torch when installed, else a small numpy reader); FP8 e4m3
weights with `weight_scale_inv` block scales are dequantized to float32 first.
RoPE inverse frequencies and the attention factor come from transformers'
ROPE_INIT_FUNCTIONS when available, so the engine never needs to know
YaRN/NTK/Llama3 variants. Routed experts are streamed layer by layer and quantized
on a thread pool (the native quantizer releases the GIL).

Usage: python -m hearth.convert SRC_DIR DST.hearth [--experts q4] [--dense q8] ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import quant
from .format import ContainerWriter, apply_defaults, canonical_tensors
from .quant import BF16, F16, F32, Q4, Q8

SUPPORTED = ("qwen3_moe", "olmoe", "mixtral", "qwen2_moe", "deepseek_v3", "kimi_k2")
MAX_EOS = 8
_CHUNK_ELEMS = 1 << 26      # 2-D tensors with more elements are converted in row chunks
_CHUNK_TARGET = 1 << 24     # elements per chunk (whole FP8 block rows, at least one)

f32 = np.float32

__all__ = ["convert", "SUPPORTED", "HFCheckpoint", "fp8_e4m3_to_f32", "fp8_block_dequant",
           "hearth_meta", "tensor_plan", "rope_params"]


# ---------------------------------------------------------------- FP8

def _e4m3_table() -> np.ndarray:
    v = np.arange(256, dtype=np.int64)
    s = np.where(v & 0x80, -1.0, 1.0)
    e = (v >> 3) & 0xF
    m = v & 0x7
    val = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1.0 + m / 8.0) * 2.0 ** (e - 7.0))
    val = s * val
    val[(e == 0xF) & (m == 0x7)] = np.nan          # e4m3fn: no infinities, S.1111.111 is NaN
    return val.astype(f32)


_E4M3 = _e4m3_table()


def fp8_e4m3_to_f32(raw: np.ndarray) -> np.ndarray:
    """Decode float8_e4m3fn bytes (uint8 array) exactly."""
    return _E4M3[np.asarray(raw, dtype=np.uint8)]


def fp8_block_dequant(w: np.ndarray, scale_inv: np.ndarray, block=(128, 128), row0: int = 0) -> np.ndarray:
    """w (float32 values of an FP8 tensor, rows [row0, row0+R) of the full matrix) times the
    per-block scale: out[r, c] = w[r, c] * scale_inv[(row0+r)//bR, c//bC] (float32)."""
    bR, bC = int(block[0]), int(block[1])
    R, C = w.shape
    rows = (np.arange(row0, row0 + R) // bR)
    cols = np.arange(C) // bC
    s = np.asarray(scale_inv, dtype=f32)
    if s.ndim != 2 or rows.max(initial=0) >= s.shape[0] or cols.max(initial=0) >= s.shape[1]:
        raise ValueError(f"scale_inv shape {s.shape} does not cover a {w.shape} tensor in {block} blocks")
    return (w.astype(f32) * s[rows][:, cols]).astype(f32)


# ---------------------------------------------------------------- safetensors

_ST_NP = {"F32": "<f4", "F16": "<f2", "F64": "<f8", "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "i1",
          "U8": "u1", "BOOL": "u1", "BF16": "<u2", "F8_E4M3": "u1"}


class _NumpyST:
    """Minimal lazy safetensors reader (memory-mapped)."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(n))
        self.base = 8 + n
        hdr.pop("__metadata__", None)
        self.hdr = hdr
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def keys(self):
        return list(self.hdr)

    def info(self, name):
        h = self.hdr[name]
        return h["dtype"], tuple(h["shape"])

    def get(self, name, r0=None, r1=None) -> np.ndarray:
        dt, shape = self.info(name)
        if dt not in _ST_NP:
            raise ValueError(f"{self.path.name}:{name}: unsupported safetensors dtype {dt}")
        a, b = self.hdr[name]["data_offsets"]
        arr = self.mm[self.base + a:self.base + b].view(_ST_NP[dt]).reshape(shape)
        if r0 is not None:
            arr = arr[r0:r1]
        if dt == "BF16":
            return quant.bf16_bits_to_f32(np.ascontiguousarray(arr))
        if dt == "F8_E4M3":
            return fp8_e4m3_to_f32(arr)
        return np.asarray(arr, dtype=f32) if dt.startswith("F") else np.array(arr)


class HFCheckpoint:
    """Lazy access to the tensors of a local Hugging Face model directory."""

    def __init__(self, src, backend: str = "auto"):
        self.src = Path(src)
        idx = self.src / "model.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
            files = sorted(set(wm.values()))
        else:
            files = sorted(p.name for p in self.src.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no *.safetensors files in {self.src}")
        if backend == "auto":
            try:
                import safetensors  # noqa: F401
                import torch  # noqa: F401
                backend = "torch"
            except ImportError:
                backend = "numpy"
        if backend not in ("torch", "numpy"):
            raise ValueError("backend must be auto, torch or numpy")
        self.backend = backend
        self._lock = threading.Lock()
        self._h: dict[str, object] = {}
        self.where: dict[str, str] = {}
        for fn in files:
            for k in self._handle(fn).keys():
                self.where[k] = fn

    def _handle(self, fn):
        h = self._h.get(fn)
        if h is None:
            if self.backend == "torch":
                from safetensors import safe_open
                h = safe_open(str(self.src / fn), framework="pt")
            else:
                h = _NumpyST(self.src / fn)
            self._h[fn] = h
        return h

    def names(self) -> list[str]:
        return list(self.where)

    def __contains__(self, name):
        return name in self.where

    def info(self, name) -> tuple[str, tuple]:
        h = self._handle(self.where[name])
        if self.backend == "numpy":
            return h.info(name)
        sl = h.get_slice(name)
        return sl.get_dtype(), tuple(sl.get_shape())

    def shape(self, name) -> tuple:
        return self.info(name)[1]

    def _raw(self, name, r0=None, r1=None) -> np.ndarray:
        h = self._handle(self.where[name])
        if self.backend == "numpy":
            with self._lock:
                return h.get(name, r0, r1)
        import torch
        with self._lock:
            t = h.get_tensor(name) if r0 is None else h.get_slice(name)[r0:r1]
        if t.dtype in (torch.float8_e4m3fn,):
            return t.float().numpy()
        if t.is_floating_point():
            return t.to(torch.float32).numpy()
        return t.numpy()

    def load(self, name, r0=None, r1=None, block=(128, 128)) -> np.ndarray:
        """float32 tensor (rows [r0, r1) if given), FP8 block scales applied."""
        a = self._raw(name, r0, r1)
        sname = name + "_scale_inv"
        if sname in self.where:
            if self.info(name)[0] not in ("F8_E4M3", "float8_e4m3fn", "torch.float8_e4m3fn"):
                raise ValueError(f"{name} has a {sname} but is not FP8 e4m3")
            a = fp8_block_dequant(a.reshape(a.shape[0], -1), self._raw(sname), block, row0=r0 or 0)
        elif self.info(name)[0] in ("F8_E4M3", "float8_e4m3fn"):
            raise ValueError(f"{name} is FP8 without a weight_scale_inv")
        return np.ascontiguousarray(a, dtype=f32)


# ---------------------------------------------------------------- config

def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _arch(cfg: dict) -> str:
    mt = cfg.get("model_type", "")
    if mt == "kimi_k2" or (mt not in SUPPORTED and "DeepseekV3ForCausalLM" in (cfg.get("architectures") or [])):
        return "deepseek_v3"
    if mt not in SUPPORTED:
        raise ValueError(f"unsupported model_type {mt!r}; supported: {', '.join(SUPPORTED)}")
    return mt


def _tf_config(cfg: dict):
    """A transformers config object for `cfg`, or None without transformers."""
    try:
        from transformers import (DeepseekV3Config, MixtralConfig, OlmoeConfig, Qwen2MoeConfig,
                                  Qwen3MoeConfig)
    except ImportError:
        return None
    cls = {"qwen3_moe": Qwen3MoeConfig, "olmoe": OlmoeConfig, "mixtral": MixtralConfig,
           "qwen2_moe": Qwen2MoeConfig, "deepseek_v3": DeepseekV3Config}[_arch(cfg)]
    d = {k: v for k, v in cfg.items() if k not in ("model_type", "auto_map", "architectures")}
    return cls.from_dict(d)


def _head_dim(cfg: dict) -> int:
    return int(cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"])


def _rot_dim(cfg: dict) -> int:
    if _arch(cfg) == "deepseek_v3":
        return int(cfg["qk_rope_head_dim"])
    return int(_head_dim(cfg) * float(cfg.get("partial_rotary_factor", 1.0)))


def _rope_fallback(cfg: dict, dim: int) -> tuple[np.ndarray, float]:
    """default / linear / yarn in float32 like transformers (used only without it)."""
    import math as _m
    base = float(cfg.get("rope_theta", 10000.0))
    rs = cfg.get("rope_scaling") or {}
    rt = rs.get("rope_type", rs.get("type")) or "default"
    expo = (np.arange(0, dim, 2, dtype=np.int64).astype(f32) / f32(dim)).astype(f32)
    pos_freqs = (f32(base) ** expo).astype(f32)
    inv = (f32(1.0) / pos_freqs).astype(f32)
    if rt == "default":
        return inv, 1.0
    if rt == "linear":
        return (inv / f32(rs["factor"])).astype(f32), 1.0
    if rt != "yarn":
        raise ValueError(f"rope type {rt!r} needs transformers installed")
    factor = float(rs["factor"])
    mpe = cfg.get("max_position_embeddings")
    if "original_max_position_embeddings" in rs:
        omax = rs["original_max_position_embeddings"]
        factor = mpe / omax
    else:
        omax = mpe

    def get_mscale(scale, m=1.0):
        return 1.0 if scale <= 1 else 0.1 * m * _m.log(scale) + 1.0

    af = rs.get("attention_factor")
    if af is None:
        ms, msa = rs.get("mscale"), rs.get("mscale_all_dim")
        af = float(get_mscale(factor, ms) / get_mscale(factor, msa)) if ms and msa else get_mscale(factor)
    bf, bs = rs.get("beta_fast") or 32, rs.get("beta_slow") or 1

    def corr_dim(nrot):
        return (dim * _m.log(omax / (nrot * 2 * _m.pi))) / (2 * _m.log(base))

    low = max(_m.floor(corr_dim(bf)), 0)
    high = min(_m.ceil(corr_dim(bs)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=f32) - f32(low)) / f32(high - low), 0, 1).astype(f32)
    extra = (f32(1) - ramp).astype(f32)
    interp = (f32(1.0) / (f32(factor) * pos_freqs)).astype(f32)
    out = (interp * (f32(1) - extra) + inv * extra).astype(f32)
    return out, float(af)


def rope_params(cfg: dict) -> tuple[np.ndarray, float]:
    """(inv_freq float32 [rope_dim/2], attention factor) as transformers computes them."""
    dim = _rot_dim(cfg)
    rs = cfg.get("rope_scaling") or {}
    rt = rs.get("rope_type", rs.get("type")) or "default"
    tcfg = _tf_config(cfg)
    if tcfg is not None:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if rt not in ROPE_INIT_FUNCTIONS:
            raise ValueError(f"unknown rope type {rt!r}")
        if _arch(cfg) == "deepseek_v3":
            tcfg.head_dim = dim
        elif not getattr(tcfg, "head_dim", None):
            tcfg.head_dim = _head_dim(cfg)
        inv, af = ROPE_INIT_FUNCTIONS[rt](tcfg, "cpu")
        inv = inv.float().numpy().astype(f32)
    else:
        inv, af = _rope_fallback(cfg, dim)
    if rt in ("dynamic", "longrope"):
        print(f"[hearth.convert] warning: rope type {rt!r} depends on sequence length; "
              f"storing the base inverse frequencies", file=sys.stderr)
    if inv.shape != (dim // 2,):
        raise ValueError(f"rope inv_freq has shape {inv.shape}, expected ({dim // 2},)")
    return inv, float(af)


def _yarn_get_mscale(scale=1.0, mscale=1.0):
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def _layer_kinds(cfg: dict, arch: str) -> list[int]:
    L = int(cfg["num_hidden_layers"])
    if arch == "deepseek_v3":
        k = int(cfg.get("first_k_dense_replace", 0))
        return [0 if i < k else 1 for i in range(L)]
    if arch in ("qwen3_moe", "qwen2_moe"):
        mlp_only = set(cfg.get("mlp_only_layers") or [])
        step = int(cfg.get("decoder_sparse_step", 1) or 1)
        E = int(cfg.get("num_experts", 0))
        return [1 if (i not in mlp_only and E > 0 and (i + 1) % step == 0) else 0 for i in range(L)]
    return [1] * L


def hearth_meta(cfg: dict, max_seq: int | None = None) -> dict:
    """FORMAT.md §3.1 metadata for a Hugging Face config dict."""
    arch = _arch(cfg)
    L, D, H = int(cfg["num_hidden_layers"]), int(cfg["hidden_size"]), int(cfg["num_attention_heads"])
    m: dict = dict(arch=arch, n_layers=L, d_model=D, vocab_size=int(cfg["vocab_size"]),
                   norm_eps=float(cfg.get("rms_norm_eps", 1e-6)), n_heads=H,
                   tie_embeddings=int(bool(cfg.get("tie_word_embeddings", False))))
    ms = int(max_seq or cfg.get("max_position_embeddings") or 4096)
    sw = cfg.get("sliding_window")
    if sw and (arch == "mixtral" or cfg.get("use_sliding_window")) and sw < ms:
        print(f"[hearth.convert] warning: sliding window {sw} < max_seq {ms}; clamping max_seq "
              f"(the engine implements full attention only)", file=sys.stderr)
        ms = int(sw)
    m["max_seq"] = ms
    if cfg.get("attention_bias") and arch != "qwen2_moe":
        raise ValueError(f"{arch}: attention_bias=True (output-projection biases) is not supported")
    if arch == "olmoe" and cfg.get("clip_qkv") is not None:
        raise ValueError("olmoe: clip_qkv is not supported")
    if arch == "deepseek_v3":
        if cfg.get("scoring_func", "sigmoid") != "sigmoid" or cfg.get("topk_method", "noaux_tc") not in ("noaux_tc",):
            raise ValueError("deepseek_v3: only sigmoid scoring with noaux_tc routing is supported")
        nope, rd, vd = int(cfg["qk_nope_head_dim"]), int(cfg["qk_rope_head_dim"]), int(cfg["v_head_dim"])
        scale = (nope + rd) ** -0.5
        rs = cfg.get("rope_scaling")
        if rs:
            mad = rs.get("mscale_all_dim", 0)
            if mad:
                ms_ = _yarn_get_mscale(rs["factor"], mad)
                scale = scale * ms_ * ms_
        E = int(cfg["n_routed_experts"])
        m.update(attn_kind=1, q_lora_rank=int(cfg.get("q_lora_rank") or 0), kv_lora_rank=int(cfg["kv_lora_rank"]),
                 qk_nope_dim=nope, qk_rope_dim=rd, v_head_dim=vd, rope_dim=rd,
                 rope_style=1 if cfg.get("rope_interleave", True) else 0, attn_scale=scale,
                 dense_ffn_dim=int(cfg.get("intermediate_size", 0)), n_experts=E,
                 top_k=int(cfg["num_experts_per_tok"]), expert_ffn_dim=int(cfg["moe_intermediate_size"]),
                 shared_ffn_dim=int(cfg["moe_intermediate_size"]) * int(cfg.get("n_shared_experts") or 0),
                 score_fn=1, score_bias=1, n_group=int(cfg.get("n_group") or 1),
                 topk_group=int(cfg.get("topk_group") or 1), norm_topk_prob=int(bool(cfg.get("norm_topk_prob", True))),
                 routed_scale=float(cfg.get("routed_scaling_factor", 1.0)))
    else:
        hd = _head_dim(cfg)
        m.update(n_kv_heads=int(cfg.get("num_key_value_heads") or H), head_dim=hd, rope_dim=_rot_dim(cfg),
                 rope_style=0, attn_scale=hd ** -0.5)
        if arch == "qwen3_moe":
            m.update(qk_norm=1, n_experts=int(cfg["num_experts"]), top_k=int(cfg["num_experts_per_tok"]),
                     expert_ffn_dim=int(cfg["moe_intermediate_size"]), dense_ffn_dim=int(cfg.get("intermediate_size", 0)),
                     norm_topk_prob=int(bool(cfg.get("norm_topk_prob", False))))
        elif arch == "olmoe":
            m.update(qk_norm=2, n_experts=int(cfg["num_experts"]), top_k=int(cfg["num_experts_per_tok"]),
                     expert_ffn_dim=int(cfg["intermediate_size"]),
                     norm_topk_prob=int(bool(cfg.get("norm_topk_prob", False))))
        elif arch == "mixtral":
            m.update(n_experts=int(cfg["num_local_experts"]), top_k=int(cfg["num_experts_per_tok"]),
                     expert_ffn_dim=int(cfg["intermediate_size"]), norm_topk_prob=1)
        elif arch == "qwen2_moe":
            m.update(qkv_bias=int(bool(cfg.get("qkv_bias", True))), n_experts=int(cfg["num_experts"]),
                     top_k=int(cfg["num_experts_per_tok"]), expert_ffn_dim=int(cfg["moe_intermediate_size"]),
                     dense_ffn_dim=int(cfg.get("intermediate_size", 0)),
                     shared_ffn_dim=int(cfg.get("shared_expert_intermediate_size", 0)),
                     shared_gate=1 if cfg.get("shared_expert_intermediate_size", 0) else 0,
                     norm_topk_prob=int(bool(cfg.get("norm_topk_prob", False))))
    m["layer_kind"] = _layer_kinds(cfg, arch)
    if not any(m["layer_kind"]):
        m["n_experts"] = 0
    return m


# ---------------------------------------------------------------- name mapping

def tensor_plan(meta: dict, has_router_bias: bool = True) -> list[tuple[str, str, str]]:
    """[(hearth_name, hf_name, role)] for every dense tensor. role: embed | head | mat | vec."""
    arch, L = meta["arch"], meta["n_layers"]
    plan = [("tok_embd", "model.embed_tokens.weight", "embed"), ("out_norm", "model.norm.weight", "vec")]
    if not meta.get("tie_embeddings"):
        plan.append(("lm_head", "lm_head.weight", "head"))
    for i in range(L):
        p, h = f"blk.{i}.", f"model.layers.{i}."
        a = h + "self_attn."
        plan += [(p + "attn_norm", h + "input_layernorm.weight", "vec"),
                 (p + "ffn_norm", h + "post_attention_layernorm.weight", "vec")]
        if meta.get("attn_kind") == 1:
            if meta.get("q_lora_rank"):
                plan += [(p + "attn_q_a", a + "q_a_proj.weight", "mat"), (p + "attn_q_a_norm", a + "q_a_layernorm.weight", "vec"),
                         (p + "attn_q_b", a + "q_b_proj.weight", "mat")]
            else:
                plan.append((p + "attn_q", a + "q_proj.weight", "mat"))
            plan += [(p + "attn_kv_a", a + "kv_a_proj_with_mqa.weight", "mat"),
                     (p + "attn_kv_a_norm", a + "kv_a_layernorm.weight", "vec"),
                     (p + "attn_kv_b", a + "kv_b_proj.weight", "mat"), (p + "attn_o", a + "o_proj.weight", "mat")]
        else:
            plan += [(p + "attn_q", a + "q_proj.weight", "mat"), (p + "attn_k", a + "k_proj.weight", "mat"),
                     (p + "attn_v", a + "v_proj.weight", "mat"), (p + "attn_o", a + "o_proj.weight", "mat")]
            if meta.get("qkv_bias"):
                plan += [(p + "attn_q_bias", a + "q_proj.bias", "vec"), (p + "attn_k_bias", a + "k_proj.bias", "vec"),
                         (p + "attn_v_bias", a + "v_proj.bias", "vec")]
            if meta.get("qk_norm"):
                plan += [(p + "attn_q_norm", a + "q_norm.weight", "vec"), (p + "attn_k_norm", a + "k_norm.weight", "vec")]
        if meta["layer_kind"][i] == 0:
            plan += [(p + "ffn_gate", h + "mlp.gate_proj.weight", "mat"), (p + "ffn_up", h + "mlp.up_proj.weight", "mat"),
                     (p + "ffn_down", h + "mlp.down_proj.weight", "mat")]
            continue
        moe = h + ("block_sparse_moe." if arch == "mixtral" else "mlp.")
        plan.append((p + "moe_router", moe + "gate.weight", "vec"))
        if meta.get("score_bias") and has_router_bias:
            plan.append((p + "moe_router_bias", moe + "gate.e_score_correction_bias", "vec"))
        if meta.get("shared_ffn_dim"):
            sh = moe + ("shared_experts." if arch == "deepseek_v3" else "shared_expert.")
            plan += [(p + "shexp_gate", sh + "gate_proj.weight", "mat"), (p + "shexp_up", sh + "up_proj.weight", "mat"),
                     (p + "shexp_down", sh + "down_proj.weight", "mat")]
            if meta.get("shared_gate"):
                plan.append((p + "shexp_gate_inp", moe + "shared_expert_gate.weight", "vec"))
    return plan


def expert_names(meta: dict, layer: int, e: int) -> tuple[str, str, str]:
    h = f"model.layers.{layer}."
    if meta["arch"] == "mixtral":
        b = h + f"block_sparse_moe.experts.{e}."
        return b + "w1.weight", b + "w3.weight", b + "w2.weight"
    b = h + f"mlp.experts.{e}."
    return b + "gate_proj.weight", b + "up_proj.weight", b + "down_proj.weight"


def _ignorable(name: str, L: int) -> bool:
    if name.endswith("rotary_emb.inv_freq") or name.endswith("_scale_inv"):
        return True
    if name.startswith("model.layers."):
        try:
            return int(name.split(".")[2]) >= L          # e.g. DeepSeek's multi-token-prediction layer
        except ValueError:
            return False
    return False


# ---------------------------------------------------------------- conversion

def _progress(on, msg):
    if on:
        print(f"[hearth.convert] {msg}", file=sys.stderr, flush=True)


def _tokenizer_meta(src: Path, dst: Path, meta: dict) -> Path | None:
    """Sets meta tokenizer/chat_template; returns the tokenizer.json to copy next to dst."""
    tok = src / "tokenizer.json"
    if tok.exists():
        meta["tokenizer"] = dst.stem + ".tokenizer.json"
    else:
        print(f"[hearth.convert] warning: {src} has no tokenizer.json; the container will not reference "
              f"a tokenizer", file=sys.stderr)
    tc = _load_json(src / "tokenizer_config.json") or {}
    tmpl = tc.get("chat_template")
    if isinstance(tmpl, list):
        named = {t.get("name"): t.get("template") for t in tmpl if isinstance(t, dict)}
        tmpl = named.get("default") or next(iter(named.values()), None)
    if not tmpl and (src / "chat_template.jinja").exists():
        tmpl = (src / "chat_template.jinja").read_text(encoding="utf-8")
    if not tmpl and (src / "chat_template.json").exists():
        tmpl = (_load_json(src / "chat_template.json") or {}).get("chat_template")
    if tmpl:
        meta["chat_template"] = str(tmpl)
    return tok if tok.exists() else None


def _special_ids(src: Path, cfg: dict, meta: dict) -> None:
    """bos_id / eos_ids from config.json and generation_config.json. Ids outside the
    vocabulary (which the container readers reject) are dropped with a warning."""
    gen = _load_json(src / "generation_config.json") or {}
    V = meta["vocab_size"]

    def ok(x) -> bool:
        if 0 <= int(x) < V:
            return True
        print(f"[hearth.convert] warning: special token id {x} is outside the vocabulary of {V}; dropped",
              file=sys.stderr)
        return False

    bos = cfg.get("bos_token_id", gen.get("bos_token_id"))
    if isinstance(bos, list):
        bos = bos[0] if bos else None
    if bos is not None and ok(bos):
        meta["bos_id"] = int(bos)
    eos: list[int] = []
    for v in (cfg.get("eos_token_id"), gen.get("eos_token_id")):
        for x in (v if isinstance(v, list) else [v]):
            if x is not None and int(x) not in eos and ok(x):
                eos.append(int(x))
    meta["eos_ids"] = eos[:MAX_EOS]


def convert(src: str | Path, dst: str | Path, *, expert_dtype=Q4, dense_dtype=Q8, embed_dtype=Q8,
            head_dtype=Q8, max_seq: int | None = None, threads: int = 0, progress=True) -> Path:
    """Convert a local Hugging Face model directory into a .hearth container."""
    src, dst = Path(src), Path(dst)
    cfg = _load_json(src / "config.json")
    if cfg is None:
        raise FileNotFoundError(f"{src / 'config.json'} not found")
    edt, ddt, mdt, hdt = (quant.dtype_of(x) for x in (expert_dtype, dense_dtype, embed_dtype, head_dtype))
    for d in (edt, ddt, mdt, hdt):
        if d not in (F32, F16, BF16, Q8, Q4):
            raise ValueError("dtypes must be F32, F16, BF16, Q8 or Q4")
    if any(quant.is_quant(d) for d in (edt, ddt, mdt, hdt)):
        from . import _native
        _native.lib()    # Q8/Q4 need the native quantizer: fail before doing any work
    threads = threads if threads > 0 else min(os.cpu_count() or 1, 16)
    t0 = time.time()
    meta = hearth_meta(cfg, max_seq)
    arch, L, D = meta["arch"], meta["n_layers"], meta["d_model"]
    ck = HFCheckpoint(src)
    qc = cfg.get("quantization_config") or {}
    block = tuple(qc.get("weight_block_size") or (128, 128))

    has_bias = any(n.endswith("gate.e_score_correction_bias") for n in ck.names())
    if meta.get("score_bias") and not has_bias:
        meta["score_bias"] = 0
    plan = tensor_plan(meta, has_bias)
    moe = [i for i in range(L) if meta["layer_kind"][i] == 1]
    E, F = meta.get("n_experts", 0), meta.get("expert_ffn_dim", 0)

    # every mapped tensor must exist with the right shape; every checkpoint tensor must be used
    canon = canonical_tensors(apply_defaults(meta))
    used = set()
    for hname, hf, role in plan:
        if hf not in ck:
            raise ValueError(f"checkpoint lacks {hf} (needed for {hname})")
        used.add(hf)
        want = canon[hname][0]
        got = ck.shape(hf)
        if tuple(got) != want:
            raise ValueError(f"{hf}: shape {got}, expected {want}")
    for li in moe:
        for e in range(E):
            for k, nm in enumerate(expert_names(meta, li, e)):
                if nm not in ck:
                    raise ValueError(f"checkpoint lacks {nm}")
                want = (F, D) if k < 2 else (D, F)
                if tuple(ck.shape(nm)) != want:
                    raise ValueError(f"{nm}: shape {ck.shape(nm)}, expected {want}")
                used.add(nm)
    unused = [n for n in ck.names() if n not in used and not _ignorable(n, L)]
    if unused:
        raise ValueError(f"checkpoint tensors not understood by the {arch} mapping: {unused[:8]}"
                         + (" ..." if len(unused) > 8 else ""))

    inv_freq, af = rope_params(cfg)
    meta["rope_attn_factor"] = af
    meta["expert_dtype"] = edt
    meta["source"] = str(cfg.get("_name_or_path") or src.name)
    _special_ids(src, cfg, meta)
    tok = _tokenizer_meta(src, dst, meta)

    def mat_dtype(role, cols):
        d = {"embed": mdt, "head": hdt, "mat": ddt}.get(role, F32)
        return F16 if quant.is_quant(d) and cols % quant.QK else d

    e_dt = edt if not (quant.is_quant(edt) and (D % quant.QK or F % quant.QK)) else F16
    shapes = {hname: tuple(ck.shape(hf)) for hname, hf, _ in plan}
    _progress(progress, f"{arch}: {L} layers, D={D}, {E} experts x {len(moe)} MoE layers -> {dst}")
    with ContainerWriter(dst, meta) as w:
        for hname, hf, role in plan:
            w.declare_tensor(hname, shapes[hname], mat_dtype(role, shapes[hname][-1]))
        w.declare_tensor("rope_inv_freq", inv_freq.shape, F32)
        if moe:
            w.declare_experts(L, E, D, F, e_dt, moe)
        w.write_tensor("rope_inv_freq", inv_freq)
        for hname, hf, role in plan:
            shp = shapes[hname]
            dt = mat_dtype(role, shp[-1])
            if len(shp) == 2 and shp[0] * shp[1] > _CHUNK_ELEMS:
                step = max(block[0], _CHUNK_TARGET // shp[1] // block[0] * block[0])
                for r0 in range(0, shp[0], step):
                    r1 = min(shp[0], r0 + step)
                    a = ck.load(hf, r0, r1, block)
                    w.write_tensor_rows(hname, r0, quant.quantize(a, dt, threads) if quant.is_quant(dt) else a)
            else:
                a = ck.load(hf, block=block)
                w.write_tensor(hname, quant.quantize(a, dt, threads) if quant.is_quant(dt) else a)
            if hname.endswith("attn_norm"):
                _progress(progress, f"dense tensors of layer {hname.split('.')[1]}")

        def job(li, e):
            mats = [ck.load(nm, block=block) for nm in expert_names(meta, li, e)]
            return [quant.quantize(m, e_dt, 1) for m in mats]

        if moe:
            with ThreadPoolExecutor(threads) as ex:
                for li in moe:
                    t = time.time()
                    futs = []
                    for e in range(E):
                        futs.append(ex.submit(job, li, e))
                        if len(futs) > threads + 2:
                            w.write_expert(li, e - len(futs) + 1, *futs.pop(0).result())
                    while futs:
                        w.write_expert(li, E - len(futs), *futs.pop(0).result())
                    _progress(progress, f"layer {li}: {E} experts in {time.time() - t:.1f}s")
    if tok is not None:
        shutil.copyfile(tok, dst.parent / meta["tokenizer"])
    _progress(progress, f"done in {time.time() - t0:.1f}s: {dst} ({dst.stat().st_size / 2**30:.2f} GiB)")
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hearth.convert", description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--experts", default="q4")
    ap.add_argument("--dense", default="q8")
    ap.add_argument("--embed", default="q8")
    ap.add_argument("--head", default="q8")
    ap.add_argument("--max-seq", type=int, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    convert(a.src, a.dst, expert_dtype=a.experts, dense_dtype=a.dense, embed_dtype=a.embed, head_dtype=a.head,
            max_seq=a.max_seq, threads=a.threads, progress=not a.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
