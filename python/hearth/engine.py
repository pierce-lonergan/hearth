"""ctypes bindings for the C engine (engine/include/hearth.h).

The Structure classes below mirror the C structs field for field; the test
suite parses hearth.h and checks names, order and types so ABI drift is caught
without compiling anything.
"""
from __future__ import annotations

import ctypes
import os
import threading
from ctypes import (POINTER, Structure, byref, c_char, c_char_p, c_double, c_float, c_int,
                    c_int32, c_size_t, c_uint64, c_void_p)

import numpy as np

__all__ = ["Engine", "HearthError", "HearthOptions", "HearthModelInfo", "HearthStats",
           "POLICIES", "PREFETCH", "ISAS", "parse_policy", "parse_prefetch", "parse_isa",
           "isa_name", "MAX_MIRRORS"]

MAX_MIRRORS = 8
_KEEP_FLOATS = 1 << 24  # logits buffer kept between calls (64 MiB); larger requests get a one-off array


class HearthError(RuntimeError):
    """An engine call failed (open error, KV capacity exceeded, ...)."""


class HearthOptions(Structure):
    _fields_ = [
        ("model_path", c_char_p),
        ("mirror_paths", c_char_p * MAX_MIRRORS),
        ("n_mirrors", c_int),
        ("cache_gb", c_double),
        ("n_threads", c_int),
        ("n_io_threads", c_int),
        ("direct_io", c_int),
        ("policy", c_int),
        ("prefetch", c_int),
        ("prefetch_extra", c_int),
        ("usage_in", c_char_p),
        ("usage_out", c_char_p),
        ("pin_fraction", c_float),
        ("warm_start", c_int),
        ("max_seq", c_int),
        ("max_batch", c_int),
        ("isa", c_int),
        ("verbose", c_int),
    ]


class HearthModelInfo(Structure):
    _fields_ = [
        ("arch", c_char * 32),
        ("n_layers", c_int), ("d_model", c_int), ("vocab_size", c_int), ("max_seq", c_int),
        ("n_heads", c_int), ("n_kv_heads", c_int), ("head_dim", c_int), ("attn_kind", c_int),
        ("n_experts", c_int), ("top_k", c_int), ("expert_ffn_dim", c_int), ("n_moe_layers", c_int),
        ("bos_id", c_int),
        ("n_eos", c_int), ("eos_ids", c_int * 8),
        ("dense_bytes", c_uint64),
        ("expert_bytes", c_uint64),
        ("slab_bytes_max", c_uint64),
        ("params_total", c_double),
        ("params_active", c_double),
        ("cache_slots", c_int),
        ("isa", c_int),
        ("n_threads", c_int), ("n_io_threads", c_int),
    ]


class HearthStats(Structure):
    _fields_ = [
        ("tokens", c_uint64),
        ("forward_calls", c_uint64),
        ("wall_s", c_double),
        ("attn_s", c_double), ("moe_s", c_double), ("dense_s", c_double), ("stall_s", c_double),
        ("expert_uses", c_uint64),
        ("expert_loads_unique", c_uint64),
        ("cache_hits", c_uint64), ("cache_misses", c_uint64),
        ("prefetch_issued", c_uint64), ("prefetch_used", c_uint64), ("prefetch_wasted", c_uint64),
        ("bytes_read", c_uint64),
        ("read_s", c_double),
        ("evictions", c_uint64),
        ("cache_slots", c_int), ("cache_resident", c_int), ("cache_pinned", c_int),
        ("read_errors", c_uint64),
    ]


POLICIES = {"lru": 0, "lfu": 1}
PREFETCH = {"off": 0, "none": 0, "next": 1, "shared": 2}
ISAS = {"auto": 0, "scalar": 1, "avx2": 2, "avx512": 3}


def _parse_enum(value, table: dict, what: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{what}: expected str or int, got bool")
    if isinstance(value, (int, np.integer)):
        v = int(value)
        if v not in set(table.values()):
            raise ValueError(f"{what}: invalid value {v} (valid: {sorted(set(table.values()))})")
        return v
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "")
        if key in table:
            return table[key]
        raise ValueError(f"{what}: unknown {value!r} (valid: {', '.join(table)})")
    raise TypeError(f"{what}: expected str or int, got {type(value).__name__}")


def parse_policy(v) -> int:
    return _parse_enum(v, POLICIES, "policy")


def parse_prefetch(v) -> int:
    return _parse_enum(v, PREFETCH, "prefetch")


def parse_isa(v) -> int:
    return _parse_enum(v, ISAS, "isa")


def isa_name(v: int) -> str:
    for k, i in ISAS.items():
        if i == v:
            return k
    return f"isa{v}"


# The shared CDLL is also used by other modules (hearth._native); prototypes are
# set on private function objects (lib[name] returns a fresh one each time) so
# nothing here can clash with argtypes chosen elsewhere.
_PROTOS = {
    "hearth_version": (c_char_p, []),
    "hearth_default_options": (None, [POINTER(HearthOptions)]),
    "hearth_open": (c_void_p, [POINTER(HearthOptions), c_char_p, c_size_t]),
    "hearth_close": (None, [c_void_p]),
    "hearth_info": (c_int, [c_void_p, POINTER(HearthModelInfo)]),
    "hearth_eval": (c_int, [c_void_p, POINTER(c_int32), c_int, POINTER(c_float), c_int]),
    "hearth_pos": (c_int, [c_void_p]),
    "hearth_reset": (c_int, [c_void_p]),
    "hearth_rewind": (c_int, [c_void_p, c_int]),
    "hearth_get_stats": (c_int, [c_void_p, POINTER(HearthStats)]),
    "hearth_reset_stats": (None, [c_void_p]),
    "hearth_trace_start": (c_int, [c_void_p, c_char_p]),
    "hearth_trace_stop": (c_int, [c_void_p]),
    "hearth_route_replay": (c_int, [c_void_p, c_char_p]),
    "hearth_cpu_isa": (c_int, []),
}

_api_cache: dict[int, "_Api"] = {}
_api_lock = threading.Lock()


class _Api:
    def __init__(self, cdll):
        self.lib = cdll
        for name, (restype, argtypes) in _PROTOS.items():
            try:
                fn = cdll[name]
            except AttributeError:
                fn = None
            if fn is not None:
                fn.restype = restype
                fn.argtypes = argtypes
            setattr(self, name[len("hearth_"):], fn)

    def need(self, name: str):
        fn = getattr(self, name)
        if fn is None:
            raise HearthError(f"native library does not export hearth_{name}")
        return fn


def _api() -> _Api:
    from hearth import _native  # lazy: importing hearth.engine never loads the DLL
    cdll = _native.lib()
    with _api_lock:
        api = _api_cache.get(id(cdll))
        if api is None or api.lib is not cdll:
            api = _api_cache[id(cdll)] = _Api(cdll)
        return api


def native_version() -> str:
    v = _api().need("version")()
    return v.decode("utf-8", "replace") if v else ""


def cpu_isa() -> int:
    return int(_api().need("cpu_isa")())


def _encode_path(p) -> bytes:
    return os.fspath(p).encode("utf-8")


_C_INT_MAX = 0x7FFFFFFF


def _check_c_int(name: str, v) -> int:
    """A non-negative value for a C int option field (ctypes would silently wrap larger ones)."""
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {v!r}")
    if not 0 <= int(v) <= _C_INT_MAX:
        raise ValueError(f"{name} must be in [0, {_C_INT_MAX}], got {v!r}")
    return int(v)


def _check_real(name: str, v, hi) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number, got {v!r}") from None
    if not np.isfinite(f) or f < 0 or (hi is not None and f > hi):
        want = "a finite number >= 0" if hi is None else f"in [0, {hi}]"
        raise ValueError(f"{name} must be {want}, got {v!r}")
    return f


class Engine:
    """One open model. All native calls are serialised by an internal lock, so
    concurrent use is memory-safe (it is not concurrent: one forward at a time)."""

    def __init__(self, model_path, *, cache_gb=8.0, threads=0, io_threads=0, direct_io=True,
                 policy="lfu", prefetch="shared", prefetch_extra=0, usage_in=None, usage_out=None,
                 pin_fraction=0.0, warm_start=False, max_seq=0, max_batch=0, isa="auto",
                 mirrors=(), verbose=0):
        self._h = None
        self._lock = threading.Lock()
        if model_path is None:
            raise TypeError("model_path is required")
        if isinstance(mirrors, (str, bytes, os.PathLike)):
            mirrors = [mirrors]
        mirrors = list(mirrors or ())
        if len(mirrors) > MAX_MIRRORS:
            raise ValueError(f"at most {MAX_MIRRORS} mirrors are supported, got {len(mirrors)}")
        cache_gb = _check_real("cache_gb", cache_gb, None)
        pin_fraction = _check_real("pin_fraction", pin_fraction, 1.0)
        ints = {name: _check_c_int(name, v) for name, v in (
            ("threads", threads), ("io_threads", io_threads), ("prefetch_extra", prefetch_extra),
            ("max_seq", max_seq), ("max_batch", max_batch),
            ("verbose", int(verbose) if isinstance(verbose, bool) else verbose))}

        self.model_path = os.fspath(model_path)
        self.options = dict(cache_gb=cache_gb, threads=ints["threads"], io_threads=ints["io_threads"],
                            direct_io=bool(direct_io), policy=parse_policy(policy),
                            prefetch=parse_prefetch(prefetch), prefetch_extra=ints["prefetch_extra"],
                            usage_in=None if usage_in is None else os.fspath(usage_in),
                            usage_out=None if usage_out is None else os.fspath(usage_out),
                            pin_fraction=pin_fraction, warm_start=bool(warm_start),
                            max_seq=ints["max_seq"], max_batch=ints["max_batch"], isa=parse_isa(isa),
                            mirrors=[os.fspath(m) for m in mirrors], verbose=ints["verbose"])
        api = _api()
        self._api = api
        o = self.options

        opt = HearthOptions()
        api.need("default_options")(byref(opt))
        # Encoded paths must outlive hearth_open: c_char_p fields do not own memory.
        keep = [_encode_path(self.model_path)]
        opt.model_path = keep[0]
        for i, m in enumerate(o["mirrors"]):
            keep.append(_encode_path(m))
            opt.mirror_paths[i] = keep[-1]
        opt.n_mirrors = len(o["mirrors"])
        opt.cache_gb = o["cache_gb"]
        opt.n_threads = o["threads"]
        opt.n_io_threads = o["io_threads"]
        opt.direct_io = 1 if o["direct_io"] else 0
        opt.policy = o["policy"]
        opt.prefetch = o["prefetch"]
        opt.prefetch_extra = o["prefetch_extra"]
        if o["usage_in"] is not None:
            keep.append(_encode_path(o["usage_in"]))
            opt.usage_in = keep[-1]
        if o["usage_out"] is not None:
            keep.append(_encode_path(o["usage_out"]))
            opt.usage_out = keep[-1]
        opt.pin_fraction = o["pin_fraction"]
        opt.warm_start = 1 if o["warm_start"] else 0
        opt.max_seq = o["max_seq"]
        opt.max_batch = o["max_batch"]
        opt.isa = o["isa"]
        opt.verbose = o["verbose"]

        err = ctypes.create_string_buffer(1024)
        h = api.need("open")(byref(opt), err, len(err))
        del keep
        if not h:
            msg = err.value.decode("utf-8", "replace").strip()
            raise HearthError(msg or f"hearth_open failed for {self.model_path!r}")
        self._h = c_void_p(h)

        mi = HearthModelInfo()
        rc = api.need("info")(self._h, byref(mi))
        if rc < 0:
            self.close()
            raise HearthError(f"hearth_info failed (code {rc})")
        self.info = _info_dict(mi)
        self._vocab = int(self.info["vocab_size"])
        if self._vocab <= 0:
            self.close()
            raise HearthError(f"engine reports vocab_size {self._vocab}")
        self._buf = np.empty(0, dtype=np.float32)

    # ---- lifecycle -------------------------------------------------------
    def close(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            h, self._h = getattr(self, "_h", None), None
            if h is not None:
                self._api.need("close")(h)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def closed(self) -> bool:
        return self._h is None

    def _handle(self):
        if self._h is None:
            raise HearthError("engine is closed")
        return self._h

    # ---- evaluation ------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._vocab

    @property
    def kv_capacity(self) -> int:
        """Maximum number of positions the KV cache holds (hearth_options.max_seq rule)."""
        model_max = int(self.info["max_seq"]) or 4096
        req = self.options["max_seq"]
        return min(req, model_max) if req > 0 else min(model_max, 4096)

    def eval(self, tokens, all_logits: bool = False) -> np.ndarray:
        """Append tokens to the KV cache; return float32 logits of the last token
        ([vocab]) or of every token ([n, vocab]) when all_logits."""
        if isinstance(tokens, (int, np.integer)):
            tokens = [int(tokens)]
        arr = np.asarray(tokens).reshape(-1)
        if arr.size and not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(f"token ids must be integers, got dtype {arr.dtype}")
        n = int(arr.size)
        V = self._vocab
        if n == 0:
            if all_logits:
                return np.empty((0, V), dtype=np.float32)
            raise ValueError("eval needs at least one token")
        if int(arr.min()) < 0 or int(arr.max()) >= V:
            raise ValueError(f"token id out of range [0, {V}): min {int(arr.min())}, max {int(arr.max())}")
        if n > 0x7FFFFFFF:
            raise ValueError("too many tokens for one eval call")
        ids = np.ascontiguousarray(arr, dtype=np.int32)
        need = (n if all_logits else 1) * V
        with self._lock:
            h = self._handle()
            if need > self._buf.size and need <= _KEEP_FLOATS:
                self._buf = np.empty(need, dtype=np.float32)
            reuse = need <= self._buf.size
            buf = self._buf if reuse else np.empty(need, dtype=np.float32)
            rc = self._api.eval(h, ids.ctypes.data_as(POINTER(c_int32)), n,
                                buf.ctypes.data_as(POINTER(c_float)), 1 if all_logits else 0)
            if rc < 0:
                pos = int(self._api.pos(h))
                raise HearthError(f"hearth_eval failed (code {rc}) evaluating {n} tokens at position {pos} "
                                  f"(KV capacity {self.kv_capacity})")
            out = buf[:need].copy() if reuse else buf
        return out.reshape(n, V) if all_logits else out

    @property
    def pos(self) -> int:
        with self._lock:
            return int(self._api.pos(self._handle()))

    def reset(self) -> None:
        with self._lock:
            rc = self._api.reset(self._handle())
        if rc < 0:
            raise HearthError(f"hearth_reset failed (code {rc})")

    def rewind(self, pos: int) -> None:
        pos = int(pos)
        with self._lock:
            h = self._handle()
            cur = int(self._api.pos(h))
            if pos < 0 or pos > cur:
                raise ValueError(f"rewind position {pos} outside [0, {cur}]")
            rc = self._api.rewind(h, pos)
        if rc < 0:
            raise HearthError(f"hearth_rewind({pos}) failed (code {rc})")

    # ---- telemetry -------------------------------------------------------
    def stats(self) -> dict:
        s = HearthStats()
        with self._lock:
            rc = self._api.get_stats(self._handle(), byref(s))
        if rc < 0:
            raise HearthError(f"hearth_get_stats failed (code {rc})")
        d = {name: getattr(s, name) for name, _ in HearthStats._fields_}
        d.update(derived_stats(d))
        return d

    def reset_stats(self) -> None:
        with self._lock:
            self._api.reset_stats(self._handle())

    def trace_start(self, path) -> None:
        b = _encode_path(path)
        with self._lock:
            rc = self._api.need("trace_start")(self._handle(), b)
        if rc < 0:
            raise HearthError(f"hearth_trace_start({os.fspath(path)!r}) failed (code {rc})")

    def trace_stop(self) -> None:
        with self._lock:
            rc = self._api.need("trace_stop")(self._handle())
        if rc < 0:
            raise HearthError(f"hearth_trace_stop failed (code {rc})")

    def route_replay(self, path) -> None:
        b = _encode_path(path)
        with self._lock:
            rc = self._api.need("route_replay")(self._handle(), b)
        if rc < 0:
            raise HearthError(f"hearth_route_replay({os.fspath(path)!r}) failed (code {rc})")

    def __repr__(self) -> str:
        state = "closed" if self._h is None else f"pos={self.pos}"
        return f"Engine({self.model_path!r}, {state})"


def _info_dict(mi: HearthModelInfo) -> dict:
    d = {}
    for name, _ in HearthModelInfo._fields_:
        v = getattr(mi, name)
        if name == "arch":
            v = bytes(v).split(b"\0", 1)[0].decode("utf-8", "replace")
        elif name == "eos_ids":
            v = list(v)
        d[name] = v
    n = max(0, min(int(d["n_eos"]), 8))
    d["eos_ids"] = d["eos_ids"][:n]
    d["isa_name"] = isa_name(int(d["isa"]))
    return d


def derived_stats(s: dict) -> dict:
    """Ratios that every front end wants; safe on all-zero stats."""
    hits, misses = s.get("cache_hits", 0), s.get("cache_misses", 0)
    wall = s.get("wall_s", 0.0)
    issued = s.get("prefetch_issued", 0)
    return {
        "hit_rate": hits / (hits + misses) if hits + misses else 0.0,
        "stall_frac": s.get("stall_s", 0.0) / wall if wall > 0 else 0.0,
        "tok_per_s": s.get("tokens", 0) / wall if wall > 0 else 0.0,
        "prefetch_accuracy": s.get("prefetch_used", 0) / issued if issued else 0.0,
        "gb_read": s.get("bytes_read", 0) / 1e9,
    }
