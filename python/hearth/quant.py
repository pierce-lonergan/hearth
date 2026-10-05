"""Weight dtypes and block encodings (docs/FORMAT.md §6, docs/NUMERICS.md §2, §6).

Encoding: F32/F16/BF16 are done in numpy (round-to-nearest-even). Q8/Q4 call the C
quantizer (`hearth_quantize`), which is authoritative. Decoding is exact pure numpy.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

F32, F16, BF16, Q8, Q4, I32, U8 = 0, 1, 2, 3, 4, 5, 6
QK = 64

NAMES = {F32: "F32", F16: "F16", BF16: "BF16", Q8: "Q8", Q4: "Q4", I32: "I32", U8: "U8"}
BY_NAME = {v.lower(): k for k, v in NAMES.items()}

_Q8_BLOCK = np.dtype([("d", "<f2"), ("q", "i1", (QK,))])          # 66 bytes
_Q4_BLOCK = np.dtype([("d", "<f2"), ("qs", "u1", (QK // 2,))])    # 34 bytes

__all__ = ["F32", "F16", "BF16", "Q8", "Q4", "I32", "U8", "QK", "NAMES", "BY_NAME",
           "dtype_of", "is_quant", "row_bytes", "quantize", "dequantize", "split_blocks",
           "act_quant_q8", "f32_to_bf16_bits", "f32_to_f16_bits", "bf16_bits_to_f32", "f16_bits_to_f32"]


def dtype_of(x) -> int:
    """Accept an int id or a name ('q4', 'F16', ...)."""
    if isinstance(x, str):
        try:
            return BY_NAME[x.lower()]
        except KeyError:
            raise ValueError(f"unknown dtype {x!r}") from None
    x = int(x)
    if x not in NAMES:
        raise ValueError(f"unknown dtype id {x}")
    return x


def is_quant(dtype: int) -> bool:
    return dtype in (Q8, Q4)


def row_bytes(dtype: int, n: int) -> int:
    """Bytes per row of n elements. Raises ValueError for Q8/Q4 when n % 64 != 0."""
    dtype, n = dtype_of(dtype), int(n)
    if n < 0:
        raise ValueError("negative row length")
    if dtype in (F32, I32):
        return 4 * n
    if dtype in (F16, BF16):
        return 2 * n
    if dtype == U8:
        return n
    if n % QK:
        raise ValueError(f"{NAMES[dtype]} rows need a multiple of {QK} elements, got {n}")
    return (66 if dtype == Q8 else 34) * (n // QK)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit patterns (uint16), round-to-nearest-even; NaN stays (quiet) NaN."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    nan = (u & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    safe = np.where(nan, np.uint32(0), u)
    r = ((safe >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    out = ((safe + r) >> np.uint32(16)).astype(np.uint16)
    if nan.any():
        out[nan] = ((u[nan] >> np.uint32(16)) | np.uint32(0x0040)).astype(np.uint16)
    return out


def f32_to_f16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> binary16 bit patterns, round-to-nearest-even, overflow -> inf, NaN -> quiet NaN."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        h = x.astype(np.float16).view(np.uint16)
    nan = np.isnan(x)
    if nan.any():   # quiet NaN with the payload's top bits, like vcvtps2ph
        u = x[nan].view(np.uint32)
        h = h.copy()
        h[nan] = (((u >> 16) & 0x8000) | 0x7E00 | ((u >> 13) & 0x3FF)).astype(np.uint16)
    return h


def bf16_bits_to_f32(b: np.ndarray) -> np.ndarray:
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32)


def f16_bits_to_f32(b: np.ndarray) -> np.ndarray:
    """binary16 bit patterns -> float32, exactly like vcvtph2ps / the C q_f16 (NaNs come back quiet)."""
    out = np.asarray(b, dtype="<u2").view("<f2").astype(np.float32)
    nan = np.isnan(out)
    if nan.any():
        out.view(np.uint32)[nan] |= np.uint32(0x00400000)
    return out


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 0:
        raise ValueError("cannot encode a scalar")
    return x.reshape(1, -1) if x.ndim == 1 else x.reshape(-1, x.shape[-1])


def quantize(x: np.ndarray, dtype: int, threads: int = 0) -> bytes:
    """Encode x (float32, [..., cols]) row by row into `dtype`.

    Q8/Q4 use the native quantizer (raises hearth._native.HearthLibNotFound if the
    library is unavailable); `threads` (0 = auto) splits rows across Python threads —
    the native call releases the GIL."""
    dtype = dtype_of(dtype)
    x2 = _as_2d(x)
    if dtype == F32:
        return np.ascontiguousarray(x2, dtype="<f4").tobytes()
    if dtype == F16:
        return f32_to_f16_bits(x2).astype("<u2").tobytes()
    if dtype == BF16:
        return f32_to_bf16_bits(x2.astype(np.float32, copy=False)).astype("<u2").tobytes()
    if dtype == I32:
        return np.ascontiguousarray(x2, dtype="<i4").tobytes()
    if dtype == U8:
        return np.ascontiguousarray(x2, dtype=np.uint8).tobytes()
    return _quantize_native(x2, dtype, threads).tobytes()


def _quantize_native(x2: np.ndarray, dtype: int, threads: int = 0) -> np.ndarray:
    from . import _native
    x2 = np.ascontiguousarray(x2, dtype=np.float32)
    rows, cols = x2.shape
    rb = row_bytes(dtype, cols)
    out = np.empty(rows * rb, dtype=np.uint8)
    if rows == 0:
        return out
    if threads <= 0:
        threads = min(os.cpu_count() or 1, 16)
    threads = max(1, min(threads, rows, (rows * cols) // (1 << 18) or 1))
    if threads == 1:
        _native.quantize_into(dtype, x2, out, 1)
        return out
    bounds = np.linspace(0, rows, threads + 1).astype(np.int64)
    view = out.reshape(rows, rb)

    def work(i):
        r0, r1 = int(bounds[i]), int(bounds[i + 1])
        if r1 > r0:
            _native.quantize_into(dtype, x2[r0:r1], view[r0:r1].reshape(-1), 1)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(work, range(threads)))
    return out


def _buffer(buf, nbytes: int) -> np.ndarray:
    if isinstance(buf, np.ndarray):
        a = np.ascontiguousarray(buf).reshape(-1).view(np.uint8)
    else:
        a = np.frombuffer(buf, dtype=np.uint8)
    if a.nbytes < nbytes:
        raise ValueError(f"buffer has {a.nbytes} bytes, need {nbytes}")
    return a[:nbytes]


def _rows_cols(shape) -> tuple[int, int]:
    shape = tuple(int(s) for s in (shape if isinstance(shape, (tuple, list)) else (shape,)))
    if not shape:
        raise ValueError("empty shape")
    cols = shape[-1]
    rows = int(np.prod(shape[:-1], dtype=np.int64)) if len(shape) > 1 else 1
    return rows, cols


def split_blocks(buf, dtype: int, shape) -> tuple[np.ndarray, np.ndarray]:
    """Q8/Q4 storage -> (q int8 [rows, cols] with Q4 nibbles already minus 8,
    d float32 [rows, cols/64]) such that w = d * q exactly."""
    dtype = dtype_of(dtype)
    rows, cols = _rows_cols(shape)
    a = _buffer(buf, rows * row_bytes(dtype, cols))
    nb = cols // QK
    if dtype == Q8:
        blk = a.view(_Q8_BLOCK).reshape(rows, nb)
        q = blk["q"].reshape(rows, cols).astype(np.int8)
    elif dtype == Q4:
        blk = a.view(_Q4_BLOCK).reshape(rows, nb)
        qs = blk["qs"]
        lo = (qs & 0x0F).astype(np.int8) - 8
        hi = (qs >> 4).astype(np.int8) - 8
        q = np.concatenate([lo, hi], axis=-1).reshape(rows, cols)
    else:
        raise ValueError(f"split_blocks needs Q8 or Q4, got {NAMES[dtype]}")
    d = f16_bits_to_f32(blk["d"].view("<u2"))
    return q, d


def dequantize(buf, dtype: int, shape) -> np.ndarray:
    """Exact decode to float32 (I32/U8 come back as int32/uint8) with the given shape."""
    dtype = dtype_of(dtype)
    shape = tuple(int(s) for s in (shape if isinstance(shape, (tuple, list)) else (shape,)))
    rows, cols = _rows_cols(shape)
    a = _buffer(buf, rows * row_bytes(dtype, cols))
    if dtype == F32:
        out = a.view("<f4").astype(np.float32)
    elif dtype == F16:
        out = f16_bits_to_f32(a.view("<u2"))
    elif dtype == BF16:
        out = bf16_bits_to_f32(a.view("<u2"))
    elif dtype == I32:
        return a.view("<i4").astype(np.int32).reshape(shape)
    elif dtype == U8:
        return a.copy().reshape(shape)
    else:
        q, d = split_blocks(a, dtype, (rows, cols))
        # f16 scale (11 significant bits) times |q| <= 127 is exact in float32; inf/NaN scales as in C
        with np.errstate(invalid="ignore"):
            out = (q.reshape(rows, cols // QK, QK).astype(np.float32) * d[:, :, None])
    return out.reshape(shape)


def act_quant_q8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NUMERICS §2 for the last axis (length % 64 == 0):
    returns (q int8 [..., n], d float32 [..., n/64])."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[-1]
    if n % QK:
        raise ValueError(f"activation length {n} is not a multiple of {QK}")
    xb = x.reshape(x.shape[:-1] + (n // QK, QK))
    a = np.abs(xb)
    amax = np.where(np.isnan(a), np.float32(0.0), a).max(axis=-1)    # NaN never wins `a > amax` in C
    d = (amax / np.float32(127.0)).astype(np.float32)
    nz = d != 0
    idv = np.zeros_like(d)
    np.divide(np.float32(1.0), d, out=idv, where=nz)
    with np.errstate(invalid="ignore"):       # inf * 0 -> NaN -> 0, like the engine
        v = xb * idv[..., None]
    v = np.clip(np.where(np.isnan(v), np.float32(0.0), v), -127.0, 127.0)   # degenerate inputs, as the engine
    q = np.rint(v).astype(np.int8)                     # rint: round half to even
    return q.reshape(x.shape), d
