"""Locate and load the Hearth shared library; raw kernel bindings.

Search order (first hit wins):
  1. ``$HEARTH_LIB`` — a library file, or a directory containing one;
  2. ``python/hearth/lib/`` next to this file;
  3. ``<data dir>/build/**/`` — newest first. Libraries that export the full engine
     API (``hearth_open``) are preferred over kernels-only builds. Candidates that
     are inspected but not chosen are unloaded again (no lingering file locks).

Engine bindings (``hearth_open`` & co.) live in ``hearth.engine``; this module only
declares the quantization / kernel utilities of ``hearth.h``.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
from pathlib import Path

import numpy as np

__all__ = ["HearthLibNotFound", "lib", "available", "data_dir", "matmul", "quantize_into",
           "dequantize", "row_bytes", "cpu_isa", "version", "lib_path"]


class HearthLibNotFound(RuntimeError):
    """The Hearth shared library could not be found or loaded."""


if sys.platform == "win32":
    _NAMES = ("hearth.dll",)
elif sys.platform == "darwin":
    _NAMES = ("libhearth.dylib", "hearth.dylib")
else:
    _NAMES = ("libhearth.so", "hearth.so")

_lock = threading.Lock()
_lib: ctypes.CDLL | None = None
_lib_path: Path | None = None
_load_error: str | None = None


def data_dir() -> Path:
    """$HEARTH_DATA, else %LOCALAPPDATA%/hearth (Windows) or ~/.cache/hearth."""
    env = os.environ.get("HEARTH_DATA")
    if env:
        return Path(env)
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "hearth"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "hearth"


def _in_dir(d: Path) -> list[Path]:
    return [d / n for n in _NAMES if (d / n).is_file()]


def _candidates() -> list[tuple[Path, bool]]:
    """(path, explicit) pairs in search order; explicit = HEARTH_LIB or the package lib dir."""
    env = os.environ.get("HEARTH_LIB")
    if env:
        p = Path(env)
        out = _in_dir(p) if p.is_dir() else ([p] if p.is_file() else [])
        if not out:
            raise HearthLibNotFound(f"HEARTH_LIB={env!r} does not name a Hearth library")
        return [(q, True) for q in out]
    out = [(q, True) for q in _in_dir(Path(__file__).resolve().parent / "lib")]
    build = data_dir() / "build"
    if build.is_dir():
        found = []
        for n in _NAMES:
            for q in build.rglob(n):
                try:
                    found.append((q.stat().st_mtime, q))
                except OSError:
                    pass
        found.sort(key=lambda t: t[0], reverse=True)
        out += [(q, False) for _, q in found]
    return out


def _declare(L: ctypes.CDLL) -> None:
    c_int, c_i64, c_vp, c_fp = ctypes.c_int, ctypes.c_int64, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)
    sigs = {
        "hearth_version": (ctypes.c_char_p, []),
        "hearth_row_bytes": (ctypes.c_size_t, [c_int, c_i64]),
        "hearth_quantize": (c_int, [c_int, c_fp, c_i64, c_i64, c_vp, c_int]),
        "hearth_dequantize": (c_int, [c_int, c_vp, c_i64, c_i64, c_fp]),
        "hearth_matmul": (c_int, [c_int, c_vp, c_i64, c_i64, c_fp, c_int, c_fp, c_int]),
        "hearth_cpu_isa": (c_int, []),
    }
    for name, (res, args) in sigs.items():
        fn = getattr(L, name, None)
        if fn is not None:
            fn.restype, fn.argtypes = res, args


def _try_load(p: Path) -> ctypes.CDLL | None:
    try:
        return ctypes.CDLL(str(p))
    except OSError:
        return None


def _release(L) -> None:
    """Unload a candidate that was inspected but not chosen, so its file is not locked
    for the rest of the process (Windows refuses to overwrite or delete a loaded DLL)."""
    import _ctypes
    handle = getattr(L, "_handle", None)
    if handle:
        (_ctypes.FreeLibrary if sys.platform == "win32" else _ctypes.dlclose)(handle)


def lib() -> ctypes.CDLL:
    """The loaded library (cached). Raises HearthLibNotFound."""
    global _lib, _lib_path, _load_error
    if _lib is not None:
        return _lib
    with _lock:
        if _lib is not None:
            return _lib
        try:
            cands = _candidates()
        except HearthLibNotFound as e:
            _load_error = str(e)
            raise
        chosen, fallback = None, None
        for p, explicit in cands:
            L = _try_load(p)
            if L is None:
                continue
            if not hasattr(L, "hearth_quantize"):
                _release(L)
                continue
            if explicit or hasattr(L, "hearth_open"):
                chosen = (L, p)
                break
            if fallback is None:
                fallback = (L, p)
            else:
                _release(L)
        if chosen is None:
            chosen = fallback
        elif fallback is not None:
            _release(fallback[0])
        if chosen is None:
            where = ", ".join(str(c) for c, _ in cands) or "no candidates"
            _load_error = f"Hearth shared library not found or not loadable ({where}); " \
                          f"build it (scripts/hxcc.py --shared) or set HEARTH_LIB"
            raise HearthLibNotFound(_load_error)
        _declare(chosen[0])
        _lib, _lib_path = chosen
        return _lib


def lib_path() -> Path | None:
    """Path of the loaded library (None if not loaded / not found)."""
    try:
        lib()
    except HearthLibNotFound:
        return None
    return _lib_path


def available() -> bool:
    try:
        lib()
        return True
    except HearthLibNotFound:
        return False


def _fn(name: str):
    fn = getattr(lib(), name, None)
    if fn is None:
        raise HearthLibNotFound(f"{_lib_path} does not export {name}")
    return fn


def version() -> str:
    return _fn("hearth_version")().decode()


def cpu_isa() -> int:
    """Best ISA of this CPU: 1 scalar, 2 avx2, 3 avx512."""
    return int(_fn("hearth_cpu_isa")())


def row_bytes(dtype: int, n: int) -> int:
    return int(_fn("hearth_row_bytes")(int(dtype), int(n)))


def _as_ptr(buf):
    """(pointer, keepalive) for bytes / bytearray / numpy buffers."""
    if isinstance(buf, np.ndarray):
        a = np.ascontiguousarray(buf)
        return ctypes.c_void_p(a.ctypes.data), a
    if isinstance(buf, (bytes, bytearray, memoryview)):
        a = np.frombuffer(buf, dtype=np.uint8)
        return ctypes.c_void_p(a.ctypes.data), a
    raise TypeError(f"expected bytes or ndarray, got {type(buf).__name__}")


def quantize_into(dtype: int, x: np.ndarray, out: np.ndarray, n_threads: int = 1) -> None:
    """hearth_quantize: x float32 [rows, cols] (C-contiguous) into out (uint8 buffer of the
    exact size). Releases the GIL, so Python threads can quantize disjoint row ranges."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("quantize_into expects a 2-D array")
    rows, cols = x.shape
    need = rows * row_bytes(dtype, cols)
    if need == 0 and rows * cols:
        raise ValueError(f"dtype {dtype} cannot encode rows of {cols}")
    if out.nbytes != need or not out.flags.c_contiguous:
        raise ValueError(f"output buffer must be {need} contiguous bytes, got {out.nbytes}")
    rc = _fn("hearth_quantize")(int(dtype), x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                rows, cols, ctypes.c_void_p(out.ctypes.data), int(n_threads))
    if rc != 0:
        raise RuntimeError(f"hearth_quantize(dtype={dtype}, {rows}x{cols}) failed with {rc}")


def dequantize(dtype: int, buf, rows: int, cols: int) -> np.ndarray:
    """hearth_dequantize (the numpy version in hearth.quant is the usual path)."""
    ptr, keep = _as_ptr(buf)
    if keep.nbytes < rows * row_bytes(dtype, cols):
        raise ValueError("buffer too small")
    y = np.empty((rows, cols), dtype=np.float32)
    rc = _fn("hearth_dequantize")(int(dtype), ptr, int(rows), int(cols),
                                  y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    if rc != 0:
        raise RuntimeError(f"hearth_dequantize failed with {rc}")
    return y


def matmul(dtype, W, rows, cols, X: np.ndarray, isa: int = 0) -> np.ndarray | None:
    """Y[T, rows] = X[T, cols] · W^T through the dispatched kernels (NUMERICS §3).
    Returns None if this CPU cannot run `isa` (1 scalar, 2 avx2, 3 avx512; 0 = auto);
    any other kernel failure raises RuntimeError."""
    if isa not in (0, 1, 2, 3):
        raise ValueError(f"isa must be 0 (auto), 1 (scalar), 2 (avx2) or 3 (avx512), got {isa!r}")
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[1] != cols:
        raise ValueError(f"X must be [T, {cols}], got {X.shape}")
    rb = row_bytes(dtype, cols)
    if rb == 0:
        raise ValueError(f"dtype {dtype} cannot encode rows of {cols}")
    wptr, wkeep = _as_ptr(W)
    if wkeep.nbytes != rows * rb:
        raise ValueError(f"W has {wkeep.nbytes} bytes, expected {rows * rb}")
    if isa and isa > cpu_isa():
        return None
    T = X.shape[0]
    Y = np.empty((T, rows), dtype=np.float32)
    rc = _fn("hearth_matmul")(int(dtype), wptr, int(rows), int(cols),
                              X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(T),
                              Y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(isa))
    if rc == -2 and isa:       # engine/src/quant.c: the CPU cannot run the requested ISA
        return None
    if rc != 0:
        raise RuntimeError(f"hearth_matmul(dtype={dtype}, {rows}x{cols}, T={T}, isa={isa}) failed with {rc}")
    return Y
