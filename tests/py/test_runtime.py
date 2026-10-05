"""Runtime tests: ctypes ABI vs hearth.h, sampling, lossless speculative decoding,
chat helpers, HTTP server, CLI. Engine-dependent tests use a pure-Python
FakeEngine; integration tests with the real library skip when it is absent."""
from __future__ import annotations

import ctypes
import http.client
import json
import re
import struct
import subprocess
import sys
import threading
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from hearth import engine as heng
from hearth.chat import (ChatTemplate, Conversation, StreamDecoder, TextStream, _parse_metadata,
                         normalize_messages, read_metadata)
from hearth.generate import (GenStats, NgramIndex, Sampler, accept_point_mass, check_speculative, common_prefix,
                             generate, generate_ids, sync_prefix)

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "engine" / "include" / "hearth.h"


# ---------------------------------------------------------------------------------
# ABI: parse the C structs and compare with the ctypes mirrors
# ---------------------------------------------------------------------------------

def parse_c_structs(text: str) -> dict[str, list[tuple[str, str, int, int | None]]]:
    """{typedef name: [(field, base type, pointer depth, array length or None)]}."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    out = {}
    for m in re.finditer(r"typedef\s+struct\s+\w+\s*\{(.*?)\}\s*(\w+)\s*;", text, re.S):
        fields = []
        for decl in m.group(1).split(";"):
            decl = " ".join(decl.split())
            if not decl:
                continue
            dm = re.match(r"^((?:const\s+)?(?:unsigned\s+)?\w+)\s*(.*)$", decl)
            assert dm, decl
            base, rest = dm.groups()
            for d in rest.split(","):
                d = d.strip()
                ptr = d.count("*")
                d = d.replace("*", "").strip()
                am = re.match(r"^(\w+)\s*(?:\[\s*(\d+)\s*\])?$", d)
                assert am, f"cannot parse declarator {d!r} in {decl!r}"
                fields.append((am.group(1), base, ptr, int(am.group(2)) if am.group(2) else None))
        out[m.group(2)] = fields
    return out


C_TYPES = {
    ("const char", 1): ctypes.c_char_p, ("char", 1): ctypes.c_char_p, ("char", 0): ctypes.c_char,
    ("int", 0): ctypes.c_int, ("float", 0): ctypes.c_float, ("double", 0): ctypes.c_double,
    ("uint64_t", 0): ctypes.c_uint64, ("int64_t", 0): ctypes.c_int64, ("uint32_t", 0): ctypes.c_uint32,
    ("int32_t", 0): ctypes.c_int32, ("size_t", 0): ctypes.c_size_t,
}


def test_parser_handles_header_idioms():
    s = parse_c_structs("typedef struct a { const char *p[8]; int x, y; /* c */ int n; int e[8];\n"
                        "uint64_t z; } a_t;")
    assert s["a_t"] == [("p", "const char", 1, 8), ("x", "int", 0, None), ("y", "int", 0, None),
                        ("n", "int", 0, None), ("e", "int", 0, 8), ("z", "uint64_t", 0, None)]


@pytest.mark.parametrize("cname,cls", [("hearth_options", heng.HearthOptions),
                                       ("hearth_model_info", heng.HearthModelInfo),
                                       ("hearth_stats", heng.HearthStats)])
def test_ctypes_structs_match_header(cname, cls):
    structs = parse_c_structs(HEADER.read_text(encoding="utf-8"))
    assert cname in structs, f"{cname} not found in hearth.h"
    want = structs[cname]
    got = cls._fields_
    assert [f[0] for f in got] == [w[0] for w in want], "field names/order differ from hearth.h"
    for (name, ctype), (_, base, ptr, n) in zip(got, want):
        elem = C_TYPES.get((base, ptr))
        assert elem is not None, f"{cname}.{name}: unmapped C type {base}{'*' * ptr}"
        if n is None:
            assert ctype is elem, f"{cname}.{name}: ctypes {ctype} != C {base}{'*' * ptr}"
        else:
            assert issubclass(ctype, ctypes.Array), f"{cname}.{name} should be an array"
            assert ctype._length_ == n and ctype._type_ is elem, f"{cname}.{name}: array type mismatch"


def test_struct_sizes_follow_natural_alignment():
    # Independent layout computation from the parsed header (LP64/LLP64 agree for these types).
    structs = parse_c_structs(HEADER.read_text(encoding="utf-8"))
    for cname, cls in (("hearth_options", heng.HearthOptions), ("hearth_model_info", heng.HearthModelInfo),
                       ("hearth_stats", heng.HearthStats)):
        off, align = 0, 1
        for name, base, ptr, n in structs[cname]:
            t = C_TYPES[(base, ptr)]
            sz, al = ctypes.sizeof(t), ctypes.alignment(t)
            off = (off + al - 1) // al * al
            assert getattr(cls, name).offset == off, f"{cname}.{name} offset"
            off += sz * (n or 1)
            align = max(align, al)
        assert ctypes.sizeof(cls) == (off + align - 1) // align * align


def test_enum_parsing():
    assert heng.parse_policy("LRU") == 0 and heng.parse_policy("lfu") == 1 and heng.parse_policy(1) == 1
    assert [heng.parse_prefetch(x) for x in ("off", "none", "next", "shared")] == [0, 0, 1, 2]
    assert [heng.parse_isa(x) for x in ("auto", "scalar", "avx2", "avx512", "AVX-512")] == [0, 1, 2, 3, 3]
    with pytest.raises(ValueError):
        heng.parse_policy("fifo")
    with pytest.raises(ValueError):
        heng.parse_isa(7)
    with pytest.raises(TypeError):
        heng.parse_prefetch(1.5)


def test_engine_constant_values_match_header():
    text = HEADER.read_text(encoding="utf-8")
    consts = {k: int(v) for k, v in re.findall(r"HEARTH_(\w+)\s*=\s*(\d+)", text)}
    assert heng.POLICIES["lru"] == consts["POLICY_LRU"] and heng.POLICIES["lfu"] == consts["POLICY_LFU"]
    assert heng.PREFETCH["off"] == consts["PREFETCH_OFF"] and heng.PREFETCH["next"] == consts["PREFETCH_NEXT"]
    assert heng.PREFETCH["shared"] == consts["PREFETCH_SHARED"]
    for k in ("auto", "scalar", "avx2", "avx512"):
        assert heng.ISAS[k] == consts["ISA_" + k.upper()]


class FakeNativeApi:
    """Pure-Python stand-in for hearth.engine._Api: receives the same ctypes
    arguments as the C functions and writes logits through the pointer.
    `fail` names calls that return an error code; `closed` records every
    hearth_close handle; `opts` is what hearth_open received."""

    def __init__(self, vocab=11, max_seq=64, fail=(), open_error=b"cannot open: simulated failure"):
        self.V, self.max_seq, self.kv = vocab, max_seq, []
        self.fail, self.open_error = set(fail), open_error
        self.opened, self.closed, self.opts = 0, [], None
        self.arch, self.n_eos, self.eos = b"fake", 0, []
        self.stats_reset, self.trace = 0, None

    def need(self, name):
        return getattr(self, name)

    def default_options(self, popt):
        popt._obj.n_threads = -77           # the Engine must overwrite every field it owns

    def open(self, popt, err, errlen):
        o = popt._obj
        self.opts = {name: getattr(o, name) for name, _ in heng.HearthOptions._fields_}
        self.opts["mirror_paths"] = list(o.mirror_paths)
        if "open" in self.fail:
            err.value = self.open_error
            return None
        self.opened += 1
        return 4096

    def close(self, h):
        self.closed.append(h.value)

    def info(self, h, pinfo):
        if "info" in self.fail:
            return -5
        mi = pinfo._obj
        mi.arch = self.arch
        mi.vocab_size, mi.max_seq, mi.n_eos, mi.isa = self.V, self.max_seq, self.n_eos, 3
        for i, t in enumerate(self.eos):
            mi.eos_ids[i] = t
        return 0

    def eval(self, h, ids, n, out, all_logits):
        if "eval" in self.fail:
            return -3
        toks = np.ctypeslib.as_array(ids, shape=(n,)).tolist()
        rows = np.ctypeslib.as_array(out, shape=((n if all_logits else 1) * self.V,)).reshape(-1, self.V)
        for i, t in enumerate(toks):
            self.kv.append(int(t))
            if all_logits or i == n - 1:
                seed = zlib.crc32(np.asarray(self.kv, dtype=np.int32).tobytes())
                rows[i if all_logits else 0] = np.random.default_rng(seed).standard_normal(self.V)
        return 0

    def pos(self, h):
        return len(self.kv)

    def rewind(self, h, p):
        if "rewind" in self.fail:
            return -1
        del self.kv[p:]
        return 0

    def reset(self, h):
        if "reset" in self.fail:
            return -1
        self.kv.clear()
        return 0

    def get_stats(self, h, ps):
        if "get_stats" in self.fail:
            return -1
        s = ps._obj
        s.tokens, s.cache_hits, s.cache_misses, s.wall_s, s.stall_s = len(self.kv), 3, 1, 2.0, 0.5
        s.bytes_read, s.prefetch_issued, s.prefetch_used, s.cache_pinned = 1_500_000_000, 8, 6, 4
        return 0

    def reset_stats(self, h):
        self.stats_reset += 1

    def trace_start(self, h, path):
        self.trace = path
        return -1 if "trace_start" in self.fail else 0

    def trace_stop(self, h):
        return -1 if "trace_stop" in self.fail else 0

    def route_replay(self, h, path):
        self.trace = path
        return -1 if "route_replay" in self.fail else 0

    def version(self):
        return None if "version" in self.fail else b"9.9.9-fake"

    def cpu_isa(self):
        return 2


def fake_api(monkeypatch, **kw) -> FakeNativeApi:
    api = FakeNativeApi(**kw)
    monkeypatch.setattr(heng, "_api", lambda: api)
    return api


def test_engine_options_reach_hearth_options(monkeypatch):
    import os
    api = fake_api(monkeypatch)
    model = Path("models") / "m é.hearth"
    with heng.Engine(model):
        pass
    o = api.opts
    assert o["model_path"] == os.fspath(model).encode("utf-8") and o["n_mirrors"] == 0
    assert o["mirror_paths"] == [None] * 8 and o["usage_in"] is None and o["usage_out"] is None
    assert (o["cache_gb"], o["n_threads"], o["n_io_threads"], o["direct_io"], o["policy"], o["prefetch"],
            o["prefetch_extra"], o["pin_fraction"], o["warm_start"], o["max_seq"], o["max_batch"], o["isa"],
            o["verbose"]) == (8.0, 0, 0, 1, 1, 2, 0, 0.0, 0, 0, 0, 0, 0)
    e = heng.Engine("m", cache_gb=1, threads=np.int64(3), io_threads=4, direct_io=False, policy="lru", prefetch=1,
                    prefetch_extra=2, usage_in="ui", usage_out=Path("uo"), pin_fraction=1.0, warm_start=True,
                    max_seq=0x7FFFFFFF, max_batch=7, isa="scalar", mirrors="D:/mirror", verbose=True)
    o = api.opts
    assert (o["cache_gb"], o["n_threads"], o["n_io_threads"], o["direct_io"], o["policy"], o["prefetch"],
            o["prefetch_extra"], o["usage_in"], o["usage_out"], o["pin_fraction"], o["warm_start"], o["max_seq"],
            o["max_batch"], o["isa"], o["verbose"]) == (1.0, 3, 4, 0, 0, 1, 2, b"ui", b"uo", 1.0, 1, 0x7FFFFFFF,
                                                        7, 1, 1)
    assert o["n_mirrors"] == 1 and o["mirror_paths"][:2] == [b"D:/mirror", None]
    assert e.model_path == "m" and e.options["mirrors"] == ["D:/mirror"] and e.options["usage_out"] == "uo"
    heng.Engine("m", mirrors=[f"{c}:/m" for c in "DEFGHIJK"]).close()        # the maximum, 8
    assert api.opts["n_mirrors"] == 8 and api.opts["mirror_paths"][7] == b"K:/m"


@pytest.mark.parametrize("kw,exc", [
    (dict(cache_gb=-0.5), ValueError), (dict(cache_gb=float("nan")), ValueError), (dict(cache_gb=float("inf")), ValueError),
    (dict(cache_gb="lots"), TypeError), (dict(pin_fraction=1.01), ValueError), (dict(pin_fraction=-0.1), ValueError),
    (dict(max_seq=2 ** 31), ValueError), (dict(max_seq=2 ** 32 + 100), ValueError), (dict(threads=-1), ValueError),
    (dict(io_threads=2 ** 31), ValueError), (dict(prefetch_extra=-1), ValueError), (dict(max_batch=2 ** 40), ValueError),
    (dict(verbose=-1), ValueError), (dict(verbose=2 ** 31), ValueError), (dict(threads=True), TypeError),
    (dict(max_seq=2.5), TypeError), (dict(max_batch="10"), TypeError), (dict(mirrors=["m"] * 9), ValueError),
    (dict(policy="fifo"), ValueError), (dict(isa=9), ValueError), (dict(prefetch=None), TypeError)])
def test_engine_rejects_bad_options_before_opening(monkeypatch, kw, exc):
    """ctypes silently wraps ints that do not fit a C int (2**32 + 100 -> 100), so
    every option is range-checked before hearth_open sees it."""
    api = fake_api(monkeypatch)
    with pytest.raises(exc):
        heng.Engine("m.hearth", **kw)
    assert api.opts is None
    with pytest.raises(TypeError):
        heng.Engine(None)


def test_engine_closes_the_native_handle_exactly_once(monkeypatch):
    import gc
    api = fake_api(monkeypatch)
    with heng.Engine("a") as e:
        assert not e.closed and api.closed == []
    assert e.closed and api.closed == [4096]
    e.close()
    assert api.closed == [4096]                         # close() is idempotent
    with pytest.raises(KeyError):
        with heng.Engine("b"):
            raise KeyError("boom")                      # __exit__ closes and does not swallow the error
    assert len(api.closed) == 2
    e = heng.Engine("c")
    e.close()
    e.close()
    assert len(api.closed) == 3
    e = heng.Engine("d")
    del e                                               # __del__ releases an engine nobody closed
    gc.collect()
    assert len(api.closed) == 4
    api.fail = {"info"}
    with pytest.raises(heng.HearthError, match="hearth_info failed") as ei:
        heng.Engine("e")
    assert len(api.closed) == 5                         # released at once, not left to __del__ ...
    del ei
    gc.collect()
    assert len(api.closed) == 5                         # ... and only once
    api.fail, api.V = set(), 0
    with pytest.raises(heng.HearthError, match="vocab_size 0") as ei:
        heng.Engine("f")
    assert len(api.closed) == 6
    del ei
    gc.collect()
    assert len(api.closed) == 6
    api.V = 1
    with heng.Engine("one") as e:                       # a one-token vocabulary is odd but valid
        assert e.vocab_size == 1
    api.V, api.fail = 11, {"open"}
    with pytest.raises(heng.HearthError, match="simulated failure"):
        heng.Engine("g")
    api.open_error = b""
    with pytest.raises(heng.HearthError, match="hearth_open failed for 'h'"):
        heng.Engine("h")
    gc.collect()
    assert len(api.closed) == 7 and api.opened == 7
    bare = heng.Engine.__new__(heng.Engine)             # __init__ never ran: close/__del__ must not raise
    bare.close()
    bare.__del__()


def test_engine_kv_capacity_follows_the_max_seq_rule(monkeypatch):
    """hearth_options.max_seq: 0 = min(model max_seq, 4096); otherwise clamped to the model's maximum."""
    # The C engine never reports a model max_seq of 0; unknown is treated as 4096 here.
    for model_max, req, want in ((64, 0, 64), (64, 1, 1), (64, 10, 10), (64, 64, 64), (64, 100, 64), (10000, 0, 4096),
                                 (10000, 5000, 5000), (0, 0, 4096), (0, 100, 100), (0, 5000, 4096)):
        fake_api(monkeypatch, max_seq=model_max)
        with heng.Engine("m", max_seq=req) as e:
            assert e.kv_capacity == want, (model_max, req)


def test_engine_calls_and_error_codes(monkeypatch):
    import os
    api = fake_api(monkeypatch)
    api.arch, api.n_eos, api.eos = b"fake_moe\0junk", 2, [5, 6, 7]
    e = heng.Engine("m")
    assert e.vocab_size == 11 and e.info["arch"] == "fake_moe" and e.info["eos_ids"] == [5, 6]
    assert e.info["isa_name"] == "avx512" and e.info["max_seq"] == 64 and "pos=0" in repr(e)
    assert e.eval([], all_logits=True).shape == (0, 11)
    for bad, exc in (([], ValueError), ([1.5], TypeError), ([-1], ValueError), ([11], ValueError)):
        with pytest.raises(exc):
            e.eval(bad)
    assert e.pos == 0
    assert e.eval(3).shape == (11,) and e.pos == 1                  # a bare int is one token
    assert e.eval(np.array([[4], [5]]), all_logits=True).shape == (2, 11) and api.kv == [3, 4, 5]
    e.rewind(3)
    e.rewind(0)                                                     # back to an empty cache
    assert e.pos == 0 and api.kv == []
    e.eval([1, 2, 3])
    for bad in (-1, 4):
        with pytest.raises(ValueError, match=r"outside \[0, 3\]"):
            e.rewind(bad)
    e.rewind(1)
    assert api.kv == [1]
    e.reset()
    assert e.pos == 0
    e.eval([1, 2])
    st = e.stats()
    assert st["tokens"] == 2 and st["cache_pinned"] == 4 and st["hit_rate"] == 0.75 and st["stall_frac"] == 0.25
    assert st["tok_per_s"] == 1.0 and st["prefetch_accuracy"] == 0.75 and st["gb_read"] == 1.5
    e.reset_stats()
    assert api.stats_reset == 1
    e.trace_start(Path("t") / "x.hrtr")
    assert api.trace == os.fspath(Path("t") / "x.hrtr").encode()
    e.trace_stop()
    e.route_replay("r.hrtr")
    assert api.trace == b"r.hrtr"
    api.fail = {"eval", "rewind", "reset", "get_stats", "trace_start", "trace_stop", "route_replay"}
    with pytest.raises(heng.HearthError, match=r"code -3\) evaluating 1 tokens at position 2 \(KV capacity 64\)"):
        e.eval([1])
    for call, what in ((lambda: e.rewind(1), "hearth_rewind"), (e.reset, "hearth_reset"),
                       (e.stats, "hearth_get_stats"), (lambda: e.trace_start("t"), "hearth_trace_start"),
                       (e.trace_stop, "hearth_trace_stop"), (lambda: e.route_replay("t"), "hearth_route_replay")):
        with pytest.raises(heng.HearthError, match=what):
            call()
    e.close()
    assert "closed" in repr(e)
    for call in (lambda: e.eval([1]), lambda: e.pos, e.reset, lambda: e.rewind(0), e.stats, e.reset_stats):
        with pytest.raises(heng.HearthError, match="closed"):
            call()


def test_engine_info_and_stats_helpers(monkeypatch):
    for n_eos, want in ((-1, []), (0, []), (3, [1, 2, 3]), (12, [1, 2, 3, 4, 5, 6, 7, 8])):
        api = fake_api(monkeypatch)
        api.n_eos, api.eos = n_eos, [1, 2, 3, 4, 5, 6, 7, 8]
        with heng.Engine("m") as e:
            assert e.info["eos_ids"] == want, n_eos
    assert heng.derived_stats({}) == {"hit_rate": 0.0, "stall_frac": 0.0, "tok_per_s": 0.0, "prefetch_accuracy": 0.0,
                                      "gb_read": 0.0}
    d = heng.derived_stats({"cache_hits": 1, "cache_misses": 3, "wall_s": 4.0, "stall_s": 1.0, "tokens": 10,
                            "prefetch_issued": 5, "prefetch_used": 2, "bytes_read": 3e9})
    assert d == {"hit_rate": 0.25, "stall_frac": 0.25, "tok_per_s": 2.5, "prefetch_accuracy": 0.4, "gb_read": 3.0}
    assert heng.derived_stats({"cache_hits": 2, "cache_misses": 2})["hit_rate"] == 0.5
    # partial dicts: every missing counter counts as 0
    assert heng.derived_stats({"cache_hits": 3, "wall_s": 0.5, "stall_s": 0.25, "tokens": 1, "prefetch_issued": 4}) == \
           {"hit_rate": 1.0, "stall_frac": 0.5, "tok_per_s": 2.0, "prefetch_accuracy": 0.0, "gb_read": 0.0}
    assert heng.derived_stats({"wall_s": 2.0, "prefetch_used": 2}) == \
           {"hit_rate": 0.0, "stall_frac": 0.0, "tok_per_s": 0.0, "prefetch_accuracy": 0.0, "gb_read": 0.0}
    assert [heng.isa_name(i) for i in (0, 1, 2, 3, 7)] == ["auto", "scalar", "avx2", "avx512", "isa7"]
    api = fake_api(monkeypatch)
    assert heng.native_version() == "9.9.9-fake" and heng.cpu_isa() == 2
    api.fail = {"version"}
    assert heng.native_version() == ""


def test_native_api_prototypes_and_cache(monkeypatch):
    from hearth import _native

    class Fn:
        pass

    class FakeCDLL:
        def __init__(self, names):
            self.fns = {n: Fn() for n in names}

        def __getitem__(self, name):
            if name not in self.fns:
                raise AttributeError(name)
            return self.fns[name]

    lib = FakeCDLL(["hearth_open", "hearth_eval", "hearth_rewind"])
    api = heng._Api(lib)
    assert api.open is lib.fns["hearth_open"] and api.open.restype is ctypes.c_void_p
    assert api.open.argtypes == [ctypes.POINTER(heng.HearthOptions), ctypes.c_char_p, ctypes.c_size_t]
    assert api.eval.restype is ctypes.c_int and api.rewind.argtypes == [ctypes.c_void_p, ctypes.c_int]
    assert api.version is None and api.need("eval") is lib.fns["hearth_eval"]
    with pytest.raises(heng.HearthError, match="does not export hearth_version"):
        api.need("version")
    monkeypatch.setattr(heng, "_api_cache", {})
    monkeypatch.setattr(_native, "lib", lambda: lib)
    a1 = heng._api()
    assert heng._api() is a1 and a1.lib is lib                     # one _Api per loaded library
    lib2 = FakeCDLL([])
    monkeypatch.setattr(_native, "lib", lambda: lib2)
    a2 = heng._api()
    assert a2 is not a1 and a2.lib is lib2 and heng._api() is a2


def test_engine_eval_returns_independent_copies(monkeypatch):
    """Engine.eval fills a reusable buffer; every result must be a copy that
    later calls cannot overwrite."""
    monkeypatch.setattr(heng, "_api", lambda: FakeNativeApi())
    e = heng.Engine("fake.hearth")
    a = e.eval([1, 2, 3])
    b = e.eval([4, 5], all_logits=True)
    c = e.eval([6])
    keep = [x.copy() for x in (a, b, c)]
    d = e.eval([7, 8], all_logits=True)
    f = e.eval([9])
    assert a.dtype == np.float32 and a.shape == (11,) and b.shape == (2, 11) and d.shape == (2, 11)
    for x, x0 in zip((a, b, c), keep):
        assert np.array_equal(x, x0), "an earlier eval result was overwritten by a later call"
    assert not np.array_equal(c, f) and not np.array_equal(b[1], d[1])
    e.rewind(3)
    assert np.array_equal(e.eval([4, 5], all_logits=True), keep[1])
    monkeypatch.setattr(heng, "_KEEP_FLOATS", 15)         # one-off (non-retained) buffer path
    e.rewind(3)
    big = e.eval([4, 5], all_logits=True)
    assert np.array_equal(big, keep[1]) and e._buf.size == 22   # retained buffer left as it was
    assert np.array_equal(b, keep[1])
    monkeypatch.setattr(heng, "_KEEP_FLOATS", 33)
    e.rewind(3)
    e.eval([4, 5, 6], all_logits=True)                      # exactly the limit: still kept for reuse
    assert e._buf.size == 33
    with pytest.raises(ValueError):
        e.eval([11])
    e.close()
    e.close()
    with pytest.raises(heng.HearthError):
        e.eval([1])


def test_import_is_lazy():
    code = ("import sys, hearth, hearth.engine, hearth.generate, hearth.chat, hearth.server, hearth.cli;"
            "bad=[m for m in ('tokenizers','jinja2','torch','transformers') if m in sys.modules];"
            "print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                         env=_env()).stdout.strip()
    assert out == "[]"


def _env():
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    return env


# ---------------------------------------------------------------------------------
# FakeEngine: deterministic toy LM with the Engine's eval/rewind/pos semantics
# ---------------------------------------------------------------------------------

class FakeEngine:
    """Deterministic toy LM over its own KV list. Logit terms:
    - last two tokens: text quickly becomes repetitive, which prompt lookup exploits;
    - position: a KV position bug changes the output;
    - crc_amp: running CRC-32 of the whole KV in order, so reused KV that holds
      the wrong tokens changes the output. It makes text aperiodic (prompt lookup
      then rarely hits), so it is off by default and on for prefix-reuse tests."""

    def __init__(self, vocab=64, max_seq=4096, seed=0, bonus=4.0, pos_amp=0.3, crc_amp=0.0, lo=0, hi=None,
                 eos=None, eos_at=None):
        rng = np.random.default_rng(seed + 1000)
        self.V, self.seed, self.bonus, self.pos_amp, self.crc_amp = vocab, seed, bonus, pos_amp, crc_amp
        self.lo, self.hi = lo, (hi if hi is not None else vocab)
        self.table = rng.standard_normal((257, vocab)).astype(np.float32)
        self.pos_table = rng.standard_normal((61, vocab)).astype(np.float32)
        self.kv_table = rng.standard_normal((257, vocab)).astype(np.float32)
        self.eos, self.eos_at = eos, eos_at
        self.info = {"vocab_size": vocab, "max_seq": max_seq, "eos_ids": [] if eos is None else [eos],
                     "bos_id": -1, "arch": "fake", "params_total": 0}
        self.kv_capacity = max_seq
        self.kv: list[int] = []
        self._crc = [seed & 0xFFFFFFFF]   # _crc[i]: running CRC-32 of kv[:i]
        self.calls = 0
        self.evaluated = 0
        self.entered = threading.Event()
        self.release = None  # threading.Event: eval blocks until set

    @property
    def pos(self):
        return len(self.kv)

    def _logits(self):
        ctx = self.kv
        a = ctx[-1]
        b = ctx[-2] if len(ctx) > 1 else 0
        x = self.table[(a * 31 + b * 17 + self.seed) % 257].copy()
        x[self.lo + (a * 7 + b * 3 + self.seed) % (self.hi - self.lo)] += self.bonus
        x += self.pos_amp * self.pos_table[len(ctx) % 61]
        if self.crc_amp:
            x += self.crc_amp * self.kv_table[self._crc[len(ctx)] % 257]
        if self.eos_at is not None and len(ctx) >= self.eos_at:
            x[self.eos] += 100.0
        return x

    def eval(self, tokens, all_logits=False):
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(10), "test engine was never released"
        tokens = [int(t) for t in np.asarray(tokens).reshape(-1)]
        if not tokens:
            raise ValueError("empty")
        if len(self.kv) + len(tokens) > self.kv_capacity:
            raise heng.HearthError("KV capacity exceeded")
        rows = []
        for t in tokens:
            assert 0 <= t < self.V
            self.kv.append(t)
            self._crc.append(zlib.crc32(t.to_bytes(4, "little"), self._crc[-1]))
            rows.append(self._logits())
        self.calls += 1
        self.evaluated += len(tokens)
        return np.stack(rows) if all_logits else rows[-1]

    def rewind(self, pos):
        assert 0 <= pos <= len(self.kv)
        del self.kv[pos:]
        del self._crc[pos + 1:]

    def reset(self):
        self.kv.clear()
        del self._crc[1:]

    def stats(self):
        return {"tokens": self.evaluated, "forward_calls": self.calls, "cache_hits": 0, "cache_misses": 0,
                "bytes_read": 0}


class FakeTokenizer:
    """Byte-level: id == byte value."""
    eos_ids: list[int] = []
    bos_id = -1

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=True):
        return bytes(int(i) for i in ids).decode("utf-8", "replace")

    def token_to_id(self, s):
        b = s.encode("utf-8")
        return b[0] if len(b) == 1 else None


# ---------------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------------

def test_greedy_is_argmax_first_index_on_ties():
    s = Sampler.greedy()
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.standard_normal(100).astype(np.float32)
        assert s.sample(x) == int(np.argmax(x))
    x = np.array([1.0, 3.0, 3.0, 0.0], dtype=np.float32)
    assert s.sample(x) == 1
    p = s.probs(x)
    assert p[1] == 1.0 and p.sum() == 1.0


def test_top_k_mask():
    x = np.array([0.1, 2.0, -1.0, 1.5, 0.3, 1.9], dtype=np.float32)
    p = Sampler(temperature=1.0, top_k=3).probs(x)
    assert set(np.flatnonzero(p)) == {1, 3, 5}
    assert abs(p.sum() - 1) < 1e-12
    ref = np.exp(x[[1, 3, 5]] - x.max())
    assert np.allclose(p[[1, 3, 5]], ref / ref.sum())


def test_top_p_keeps_smallest_nucleus():
    probs = np.array([0.5, 0.25, 0.15, 0.1])
    x = np.log(probs)
    assert set(np.flatnonzero(Sampler(top_p=0.5).probs(x))) == {0}
    assert set(np.flatnonzero(Sampler(top_p=0.6).probs(x))) == {0, 1}
    assert set(np.flatnonzero(Sampler(top_p=0.75).probs(x))) == {0, 1}
    assert set(np.flatnonzero(Sampler(top_p=0.76).probs(x))) == {0, 1, 2}
    assert set(np.flatnonzero(Sampler(top_p=1.0).probs(x))) == {0, 1, 2, 3}
    p = Sampler(top_p=0.8).probs(x)
    assert np.allclose(p[:3], probs[:3] / probs[:3].sum())


def test_min_p_and_temperature():
    probs = np.array([0.6, 0.3, 0.06, 0.04])
    x = np.log(probs)
    p = Sampler(min_p=0.09).probs(x)
    assert set(np.flatnonzero(p)) == {0, 1, 2}
    assert np.allclose(p[:3], probs[:3] / probs[:3].sum()) and abs(p.sum() - 1) < 1e-12   # renormalised
    assert set(np.flatnonzero(Sampler(min_p=0.11).probs(x))) == {0, 1}
    hot = Sampler(temperature=2.0).probs(x)
    cold = Sampler(temperature=0.5).probs(x)
    assert hot[0] < 0.6 < cold[0]
    assert np.allclose(hot, np.sqrt(probs) / np.sqrt(probs).sum())        # p ** (1 / T), renormalised
    assert np.allclose(cold, probs ** 2 / (probs ** 2).sum())


def test_tiny_temperature_tends_to_greedy_without_nan():
    import warnings
    x = np.array([5.0, 1.0, -3.0, 0.5], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                  # no overflow / invalid-value warnings either
        for t in (1e-320, 1e-300, 1e-30):
            s = Sampler(temperature=t, seed=0)
            assert s.probs(x).tolist() == [1.0, 0.0, 0.0, 0.0], t
            assert [s.sample(x) for _ in range(20)] == [0] * 20
        tie = Sampler(temperature=1e-320).probs(np.array([2.0, 2.0, 1.0]))
        assert tie.tolist() == [0.5, 0.5, 0.0]
        masked = Sampler(temperature=1.0).probs(np.array([-np.inf, 0.0, 0.0]))
        assert masked.tolist() == [0.0, 0.5, 0.5]           # -inf is a masked token, not an error


def test_non_finite_logits_are_rejected():
    for bad in ([np.nan, 1.0, 0.0], [np.inf, 1.0, 0.0], [-np.inf, -np.inf, -np.inf]):
        for s in (Sampler.greedy(), Sampler(temperature=0.7, seed=1)):
            with pytest.raises(ValueError, match="NaN"):
                s.sample(np.array(bad))
            with pytest.raises(ValueError):
                s.probs(np.array(bad))


def test_repetition_penalty():
    x = np.array([2.0, 1.9, -1.0, -1.2], dtype=np.float32)
    s = Sampler(temperature=0.0, repetition_penalty=1.5)
    assert s.sample(x, context=[]) == 0
    assert s.sample(x, context=[0]) == 1          # 2.0/1.5 < 1.9
    y = s._penalised(x, [2])
    assert y[2] == pytest.approx(-1.5)            # negative logits are multiplied


def test_seeded_sampling_is_deterministic():
    rng = np.random.default_rng(3)
    logits = [rng.standard_normal(50).astype(np.float32) for _ in range(30)]

    def run(seed):
        s = Sampler(temperature=0.9, top_k=20, top_p=0.95, seed=seed)
        return [s.sample(x) for x in logits]

    assert run(7) == run(7)
    assert run(7) != run(8)


def test_sampling_frequencies_match_distribution():
    x = np.log(np.array([0.5, 0.3, 0.2]))
    s = Sampler(temperature=1.0, seed=1)
    n = 20000
    counts = np.bincount([s.sample(x) for _ in range(n)], minlength=3) / n
    assert np.abs(counts - [0.5, 0.3, 0.2]).max() < 0.015


class StubRng:
    """rng.random() returns these values in order."""

    def __init__(self, vals):
        self.vals = list(vals)

    def random(self):
        return self.vals.pop(0)


def test_sampling_edge_cases_and_defaults():
    from types import SimpleNamespace
    from hearth.generate import kv_capacity
    x = np.array([2.0, 1.9, -1.0], dtype=np.float32)
    pen = Sampler(temperature=0.0, repetition_penalty=1.5)
    assert pen.sample(x) == 0 and pen.sample(x, context=None) == 0 and pen.sample(x, context=[0]) == 1
    # u = random() * total can round up to the total: the draw must still be a token with mass
    assert accept_point_mass(np.array([0.3, 0.2, 0.5, 0.0]), 0, StubRng([0.99, 1.0])) == (False, 2)
    assert accept_point_mass(np.array([0.3, 0.2, 0.5, 0.0]), 0, StubRng([0.29])) == (True, 0)
    s = Sampler(temperature=1.0)
    s.rng = StubRng([1.0, 0.0])
    assert s.draw(np.array([0.5, 0.5, 0.0])) == 1 and s.draw(np.array([0.0, 0.5, 0.5])) == 1
    assert GenStats().as_dict() == {
        "prompt_tokens": 0, "reused_tokens": 0, "new_tokens": 0, "decode_tokens": 0, "forwards": 0, "draft_steps": 0,
        "drafted": 0, "accepted": 0, "prefill_s": 0.0, "decode_s": 0.0, "finish_reason": None, "stop_id": None,
        "acceptance_rate": 0.0, "tokens_per_forward": 0.0, "prefill_tok_s": 0.0, "decode_tok_s": 0.0}
    assert kv_capacity(SimpleNamespace(kv_capacity=77, info={"max_seq": 5})) == 77
    assert kv_capacity(SimpleNamespace(info={"max_seq": 64})) == 64        # engines without kv_capacity
    assert kv_capacity(SimpleNamespace(info={})) == kv_capacity(SimpleNamespace()) == 1 << 30


def test_sampler_validation():
    for kw in (dict(temperature=-1), dict(top_p=0.0), dict(top_p=1.5), dict(top_k=-1), dict(min_p=2),
               dict(repetition_penalty=0)):
        with pytest.raises(ValueError):
            Sampler(**kw)


# ---------------------------------------------------------------------------------
# Speculative decoding
# ---------------------------------------------------------------------------------

def test_ngram_index_proposals():
    idx = NgramIndex(3, [5, 6, 7, 8, 9, 5, 6, 7])
    assert idx.propose(3) == [8, 9, 5]
    assert idx.propose(1) == [8]
    idx.append(1)                                  # suffix (6,7,1),(7,1),(1) never seen before
    assert idx.propose(4) == []
    idx2 = NgramIndex(3, [1, 2, 3, 9, 2, 3, 4, 7, 2, 3])
    assert idx2.propose(2) == [4, 7]               # most recent occurrence of the longest match
    idx3 = NgramIndex(2, [4, 4, 4])
    assert idx3.propose(5) == [4]                  # never matches itself, only earlier text
    assert NgramIndex(3).propose(4) == []


def test_common_prefix_and_sync():
    assert common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix([1, 2], [1, 2, 3]) == 2
    assert common_prefix([], [1]) == 0
    e = FakeEngine()
    e.eval([1, 2, 3, 4])
    assert sync_prefix(e, [1, 2, 3, 4], [1, 2, 9]) == 2 and e.pos == 2
    e.eval([7, 7])
    assert sync_prefix(e, [1, 2, 7, 7], [1, 2, 7, 7]) == 3 and e.pos == 3  # last token re-evaluated
    assert sync_prefix(e, None, [1]) == 0 and e.pos == 0
    e.eval([1, 2])
    assert sync_prefix(e, [1], [1, 2, 3]) == 0 and e.pos == 0             # held shorter than pos -> reset


PROMPTS = [
    [1, 2, 3, 4, 5, 1, 2, 3],
    [10, 11, 12, 13, 10, 11, 12, 13, 10, 11],
    [7],
    [3, 3, 3, 3, 3, 3],
    list(range(40)) + list(range(20)),
]


@pytest.mark.parametrize("pos_amp,bonus,crc_amp", [(0.3, 4.0, 0.0), (2.5, 1.0, 0.0), (0.3, 4.0, 2.0)])
def test_speculative_greedy_is_lossless(pos_amp, bonus, crc_amp):
    total_drafted = total_accepted = 0
    for seed in range(12):
        for prompt in PROMPTS:
            for draft_len, ngram_n in ((4, 3), (1, 1), (8, 2)):
                kw = dict(seed=seed, pos_amp=pos_amp, bonus=bonus, crc_amp=crc_amp)
                e1, e2 = FakeEngine(**kw), FakeEngine(**kw)
                plain, st1 = generate_ids(e1, prompt, 48, Sampler.greedy())
                spec, st2 = generate_ids(e2, prompt, 48, Sampler.greedy(), speculative="ngram",
                                         draft_len=draft_len, ngram_n=ngram_n)
                assert spec == plain, (seed, prompt, draft_len, ngram_n)
                assert st1.finish_reason == st2.finish_reason == "length"
                seq = prompt + spec
                assert e2.kv == seq[:e2.pos] and e2.pos == len(seq) - 1
                total_drafted += st2.drafted
                total_accepted += st2.accepted
    assert total_drafted > 0
    if crc_amp == 0.0:  # whole-KV hashing makes text aperiodic, so drafts are then mostly rejected
        assert total_accepted > 0


def test_speculative_reduces_forwards_on_repetitive_text():
    e1, e2 = FakeEngine(seed=1), FakeEngine(seed=1)
    prompt = [10, 11, 12, 13, 10, 11, 12, 13, 10, 11]
    plain, st1 = generate_ids(e1, prompt, 100, Sampler.greedy())
    spec, st2 = generate_ids(e2, prompt, 100, Sampler.greedy(), speculative="ngram", draft_len=6)
    assert spec == plain
    assert st2.acceptance_rate > 0.3, st2
    assert st2.forwards < st1.forwards / 2
    assert st2.tokens_per_forward > 2.0
    assert st1.tokens_per_forward == 1.0


def test_speculative_lossless_with_stop_tokens_penalty_and_capacity():
    for seed in range(10):
        prompt = [5, 6, 7, 8, 5, 6, 7]
        # stop token: first id the plain run produces after a few steps
        ref, _ = generate_ids(FakeEngine(seed=seed), prompt, 30, Sampler.greedy())
        stop = [ref[min(7, len(ref) - 1)]]
        a, sa = generate_ids(FakeEngine(seed=seed), prompt, 30, Sampler.greedy(), stop)
        b, sb = generate_ids(FakeEngine(seed=seed), prompt, 30, Sampler.greedy(), stop, "ngram", 5, 3)
        assert a == b and sa.finish_reason == sb.finish_reason == "stop" and sa.stop_id == sb.stop_id
        # context-dependent processing (repetition penalty) under greedy
        pen = dict(temperature=0.0, repetition_penalty=1.3)
        a, _ = generate_ids(FakeEngine(seed=seed), prompt, 40, Sampler(**pen))
        b, _ = generate_ids(FakeEngine(seed=seed), prompt, 40, Sampler(**pen), (), "ngram", 4, 3)
        assert a == b
        # KV capacity ends generation identically
        a, sa = generate_ids(FakeEngine(seed=seed, max_seq=20), prompt, 100, Sampler.greedy())
        e = FakeEngine(seed=seed, max_seq=20)
        b, sb = generate_ids(e, prompt, 100, Sampler.greedy(), (), "ngram", 6, 3)
        assert a == b and len(a) == 20 - len(prompt) + 1 and sa.finish_reason == sb.finish_reason == "length"
        assert e.pos <= 20


REPETITIVE = [10, 11, 12, 13, 10, 11, 12, 13, 10, 11]


def test_speculative_lossless_with_penalty_on_short_contexts():
    """A small vocabulary and short prompts make accepted drafts long relative to
    the context, so the penalty context of the token after a draft matters."""
    accepted = 0
    for seed in range(20):
        for penalty in (1.5, 2.5, 4.0):
            for prompt in ([2, 0, 3, 0, 2], [1, 2, 3, 1, 2], [1, 2, 1, 3, 1, 2], [7, 7]):
                kw = dict(seed=seed, vocab=16, bonus=3.0)
                s = dict(temperature=0.0, repetition_penalty=penalty)
                a, _ = generate_ids(FakeEngine(**kw), prompt, 30, Sampler(**s))
                b, st = generate_ids(FakeEngine(**kw), prompt, 30, Sampler(**s), (), "ngram", 4, 1)
                assert a == b, (seed, penalty, prompt)
                accepted += st.accepted
    assert accepted > 0


def test_generate_closed_at_any_point_leaves_consistent_kv():
    mid_burst = 0
    for k in range(1, 40):
        e = FakeEngine(seed=1)
        gen = generate(e, REPETITIVE, 60, Sampler.greedy(), speculative="ngram", draft_len=6)
        got = [next(gen).id for _ in range(k)]
        mid_burst += e.pos > len(REPETITIVE) + k        # suspended inside an accepted draft burst
        gen.close()
        seq = REPETITIVE + got
        assert e.pos <= len(seq) and e.kv == seq[:e.pos] and e.pos >= len(seq) - 1
    assert mid_burst > 0


def test_generate_token_fields_and_stats_are_consistent():
    toks = list(generate(FakeEngine(seed=1), REPETITIVE, 60, Sampler.greedy(), speculative="ngram", draft_len=6))
    st = toks[-1].stats
    steps: dict = {}
    for t in toks:
        steps.setdefault(t.step, []).append(t)
    assert [(t.drafted, t.accepted) for t in steps[0]] == [(0, 0)]       # prefill
    for ts in steps.values():
        d, a = ts[0].drafted, ts[0].accepted
        assert all((t.drafted, t.accepted) == (d, a) for t in ts) and 0 <= a <= d and len(ts) <= a + 1
    assert st.drafted == sum(ts[0].drafted for ts in steps.values())
    assert st.accepted == sum(ts[0].accepted for ts in steps.values())
    assert st.draft_steps == sum(1 for ts in steps.values() if ts[0].drafted) > 0
    assert st.forwards == max(steps) and st.decode_tokens == st.new_tokens - 1 == 59
    assert 0 <= st.prefill_s < 60 and 0 < st.decode_s < 60
    assert 0 < st.decode_tok_s and 0 < st.prefill_tok_s
    plain = list(generate(FakeEngine(seed=1), REPETITIVE, 20))
    assert [(t.step, t.drafted, t.accepted) for t in plain] == [(i, 0, 0) for i in range(20)]
    assert plain[-1].stats.forwards == 19 and plain[-1].stats.draft_steps == 0


def test_generate_edge_cases():
    e = FakeEngine()
    out, st = generate_ids(e, [1, 2], 0)
    assert out == [] and st.finish_reason == "length"
    with pytest.raises(ValueError):
        generate_ids(e, [], 4)
    with pytest.raises(ValueError):
        generate_ids(FakeEngine(max_seq=4), [1, 2, 3, 4, 5], 4)
    with pytest.raises(ValueError):
        generate_ids(e, [1], 4, speculative="medusa")
    with pytest.raises(ValueError):
        generate_ids(e, [1], 4, speculative="ngram", ngram_n=0)
    with pytest.raises(ValueError):
        generate_ids(e, [1], -1)
    out, st = generate_ids(FakeEngine(eos=3, eos_at=4), [1, 2], 10, None, [3])
    assert st.finish_reason == "stop" and st.stop_id == 3 and 3 not in out


def test_check_speculative():
    assert check_speculative("none", 4, 3) == "none" and check_speculative(None, 0, 1) == "none"
    assert check_speculative("ngram", 0, 1) == "ngram" and check_speculative(True, np.int64(4), 3) == "ngram"
    for args in (("medusa", 4, 3), ("ngram", -1, 3), ("ngram", 4, 0), ("none", 4, -2), ("ngram", 2.5, 3),
                 ("ngram", 4, True), ("ngram", "4", 3)):
        with pytest.raises(ValueError):
            check_speculative(*args)


def test_generate_zero_tokens_leaves_a_prompt_prefix():
    """max_new_tokens == 0 evaluates nothing but must still leave the engine
    holding a prefix of the new prompt: callers track the KV from that."""
    e = FakeEngine()
    a, b = [1, 2, 3, 4, 5, 6], [1, 2, 9, 9]
    e.eval(a)
    out, st = generate_ids(e, b, 0, held=a)
    assert out == [] and st.finish_reason == "length" and st.reused_tokens == 2
    assert e.kv == [1, 2] and e.calls == 1                    # rewound, nothing evaluated
    out, st = generate_ids(e, [1, 2, 7], 0, held=[1, 2])
    assert e.kv == [1, 2] and st.reused_tokens == 2
    out, st = generate_ids(e, [1, 2], 0, held=None)           # unknown KV content: reset
    assert e.pos == 0 and st.reused_tokens == 0


def test_accept_point_mass_is_exact():
    p = np.array([0.45, 0.25, 0.2, 0.1, 0.0])
    n = 40000
    for d in (0, 2, 4):
        rng = np.random.default_rng(d)
        draws = np.array([accept_point_mass(p, d, rng)[1] for _ in range(n)])
        freq = np.bincount(draws, minlength=p.size) / n
        assert np.abs(freq - p).max() < 4 * np.sqrt(0.25 / n) + 1e-3, (d, freq)
    rng = np.random.default_rng(0)
    assert accept_point_mass(np.array([0.0, 1.0]), 1, rng) == (True, 1)


def _exact_plain_distribution(make, prompt, n_new, sampler_kw):
    """Exact distribution of n_new sampled tokens via enumeration."""
    dist = {}

    def rec(seq, prob):
        if len(seq) - len(prompt) == n_new:
            dist[tuple(seq[len(prompt):])] = dist.get(tuple(seq[len(prompt):]), 0.0) + prob
            return
        e = make()
        lg = e.eval(seq)
        p = Sampler(**sampler_kw).probs(lg, seq)
        for t in np.flatnonzero(p):
            rec(seq + [int(t)], prob * p[t])

    rec(list(prompt), 1.0)
    return dist


@pytest.mark.parametrize("broken", [False, True])
def test_speculative_sampling_preserves_distribution(broken, monkeypatch):
    if broken:  # negative control: accepting every draft must be detected
        import hearth.generate as g
        monkeypatch.setattr(g, "accept_point_mass", lambda p, d, rng: (True, int(d)))

    def make():
        return FakeEngine(vocab=4, seed=5, bonus=1.0, pos_amp=0.5)

    prompt = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    kw = dict(temperature=1.0)
    exact = _exact_plain_distribution(make, prompt, 3, kw)
    n = 6000
    counts: dict = {}
    drafted = 0
    for seed in range(n):
        out, st = generate_ids(make(), prompt, 3, Sampler(seed=seed, **kw), (), "ngram", 2, 2)
        counts[tuple(out)] = counts.get(tuple(out), 0) + 1
        drafted += st.drafted
    assert drafted > 0
    keys = [k for k, p in exact.items() if p * n >= 5]
    chi2 = sum((counts.get(k, 0) - exact[k] * n) ** 2 / (exact[k] * n) for k in keys)
    rest_exp = n - sum(exact[k] * n for k in keys)
    rest_obs = n - sum(counts.get(k, 0) for k in keys)
    if rest_exp >= 5:
        chi2 += (rest_obs - rest_exp) ** 2 / rest_exp
    df = len(keys)
    limit = df + 6 * np.sqrt(2 * df)
    if broken:
        assert chi2 > limit, f"negative control not detected: chi2={chi2:.1f} limit={limit:.1f}"
    else:
        assert chi2 < limit, f"chi2={chi2:.1f} limit={limit:.1f} df={df}"


# ---------------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------------

def _write_container(path: Path, meta: list[tuple[str, int, bytes]]):
    body = b""
    for key, typ, payload in meta:
        k = key.encode()
        body += struct.pack("<H", len(k)) + k + bytes([typ]) + payload
    pre = struct.pack("<IIQQQQQQII", 0x48545248, 1, 64, len(body), 64 + len(body), 0, 64 + len(body), 0, 4096, 0)
    path.write_bytes(pre + body)


def test_metadata_fallback_parser(tmp_path):
    tpl = "{{ messages[0]['content'] }}"
    p = tmp_path / "m.hearth"
    _write_container(p, [
        ("arch", 4, struct.pack("<I", 4) + b"test"),
        ("n_layers", 1, struct.pack("<I", 3)),
        ("norm_eps", 2, struct.pack("<f", 0.5)),
        ("big", 3, struct.pack("<Q", 1 << 40)),
        ("eos_ids", 5, struct.pack("<III", 2, 7, 9)),
        ("rope", 6, struct.pack("<If", 1, 0.25)),
        ("layer_kind", 7, struct.pack("<I", 3) + bytes([0, 1, 1])),
        ("chat_template", 4, struct.pack("<I", len(tpl)) + tpl.encode()),
        ("tokenizer", 4, struct.pack("<I", 8) + b"tok.json"),
    ])
    m = _parse_metadata(p)
    assert m == {"arch": "test", "n_layers": 3, "norm_eps": 0.5, "big": 1 << 40, "eos_ids": [7, 9],
                 "rope": [0.25], "layer_kind": [0, 1, 1], "chat_template": tpl, "tokenizer": "tok.json"}
    bad = tmp_path / "bad.hearth"
    bad.write_bytes(p.read_bytes()[:90])
    with pytest.raises(ValueError):
        _parse_metadata(bad)
    junk = tmp_path / "junk.hearth"
    junk.write_bytes(b"XXXX" + bytes(60))
    with pytest.raises(ValueError):
        _parse_metadata(junk)
    lying = tmp_path / "lying.hearth"
    data = bytearray(p.read_bytes())
    struct.pack_into("<Q", data, 16, 1 << 40)          # meta_bytes far past the end of the file
    lying.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        _parse_metadata(lying)


def _write_raw_container(path: Path, body: bytes, meta_bytes: int, meta_off: int = 64):
    pre = struct.pack("<IIQQQQQQII", 0x48545248, 1, meta_off, meta_bytes, 64 + len(body), 0, 64 + len(body), 0,
                      4096, 0)
    path.write_bytes(pre + body + bytes(64))   # bytes after the section exist: only its declared end counts


def test_metadata_parser_rejects_entries_cut_off_by_the_section_end(tmp_path):
    entries = [("s", 4, struct.pack("<I", 3) + b"abc"), ("u32", 1, struct.pack("<I", 7)),
               ("f32", 2, struct.pack("<f", 0.5)), ("u64", 3, struct.pack("<Q", 9)),
               ("au32", 5, struct.pack("<II", 1, 2)), ("af32", 6, struct.pack("<If", 1, 1.5)),
               ("au8", 7, struct.pack("<I", 2) + b"\x01\x02")]
    p = tmp_path / "m.hearth"
    for key, typ, payload in entries:
        k = key.encode()
        body = struct.pack("<H", len(k)) + k + bytes([typ]) + payload
        _write_raw_container(p, body, len(body))
        assert len(_parse_metadata(p)) == 1                   # the whole entry parses
        for cut in range(1, len(body)):
            _write_raw_container(p, body, cut)
            with pytest.raises(ValueError, match="malformed metadata entry"):
                _parse_metadata(p)
    _write_raw_container(p, b"", 0, meta_off=1 << 20)          # section starts past the end of the file
    with pytest.raises(ValueError, match="out of bounds"):
        _parse_metadata(p)
    body = struct.pack("<H", 1) + b"k" + bytes([1]) + struct.pack("<I", 42)
    pre = struct.pack("<IIQQQQQQII", 0x48545248, 1, 200, len(body), 0, 0, 0, 0, 4096, 0)
    p.write_bytes(pre + bytes(136) + body)                     # the section need not follow the preamble
    assert _parse_metadata(p) == {"k": 42}
    data = bytearray(p.read_bytes())
    struct.pack_into("<Q", data, 16, len(body) + 1)            # one byte longer than the file
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="out of bounds"):
        _parse_metadata(p)
    body = struct.pack("<H", 1) + b"k" + bytes([9]) + bytes(4)
    _write_raw_container(p, body, len(body))
    with pytest.raises(ValueError, match="unknown metadata type 9"):
        _parse_metadata(p)
    data = bytearray(p.read_bytes())
    struct.pack_into("<I", data, 4, 2)                        # version 2
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="version"):
        _parse_metadata(p)
    p.write_bytes(bytes(63))
    with pytest.raises(ValueError, match="too short"):
        _parse_metadata(p)


def test_tokenizer_wrapper_special_tokens(tmp_path):
    from hearth.chat import Tokenizer
    raw = _byte_level_tokenizer()
    path = tmp_path / "tok.json"
    raw.save(str(path))
    tok = Tokenizer.from_file(path, 5, [7, 9])
    assert tok.bos_id == 5 and tok.eos_ids == [7, 9] and tok.vocab_size == 256
    assert tok.bos_token == raw.id_to_token(5) and tok.eos_token == raw.id_to_token(7)
    assert tok.token_to_id(raw.id_to_token(3)) == 3 and tok.id_to_token(np.int64(3)) == raw.id_to_token(3)
    plain = Tokenizer.from_file(path)
    assert plain.bos_id == -1 and plain.bos_token is None and plain.eos_token is None
    assert Tokenizer(raw, 0).bos_token == raw.id_to_token(0)          # id 0 is a real token
    assert Tokenizer(raw, None).bos_id == -1 and Tokenizer(raw, -3).bos_id == -1
    assert Tokenizer(raw).bos_id == -1 and Tokenizer(raw).eos_ids == []
    tpl = {"chat_template": "{{ bos_token }}|{% for m in messages %}{{ m['content'] }}{% endfor %}|{{ eos_token }}"}
    msgs = [{"role": "user", "content": "hi"}]
    assert ChatTemplate.from_metadata(tpl, tok).render(msgs) == f"{raw.id_to_token(5)}|hi|{raw.id_to_token(7)}"
    over = dict(tpl, bos_token="<s>", eos_token="</s>")
    assert ChatTemplate.from_metadata(over, tok).render(msgs) == "<s>|hi|</s>"   # metadata strings win
    assert ChatTemplate.from_metadata(tpl, None).render(msgs) == "|hi|"
    assert ChatTemplate.from_metadata({}, tok).is_fallback


def test_read_metadata_closes_the_container_reader(monkeypatch):
    fmt = pytest.importorskip("hearth.format")
    closed = []

    class Reader:
        def __init__(self, path):
            self.meta = {"k": 1}

        def close(self):
            closed.append(True)

    class NoClose:
        def __init__(self, path):
            self.meta = {"a": 2}

    monkeypatch.setattr(fmt, "ContainerReader", Reader)
    assert read_metadata("m.hearth") == {"k": 1} and closed == [True]   # no open handle left behind
    monkeypatch.setattr(fmt, "ContainerReader", NoClose)
    assert read_metadata("m.hearth") == {"a": 2}


def test_read_metadata_agrees_with_container_reader(tmp_path):
    synth = pytest.importorskip("hearth.synth")
    pytest.importorskip("hearth.format")
    try:
        p = synth.make_tiny(tmp_path / "t.hearth", seed=0)
    except Exception as e:  # needs the native quantizer for some dtypes; F32 should not
        pytest.skip(f"make_tiny unavailable: {e}")
    fast = _parse_metadata(p)
    full = read_metadata(p)
    assert set(fast) == set(full)
    for k in fast:
        a, b = fast[k], full[k]
        if isinstance(a, float):
            assert a == pytest.approx(b)
        elif isinstance(a, list):
            assert list(map(float, a)) == pytest.approx(list(map(float, b)))
        else:
            assert a == b, k


def test_chat_template_rendering():
    pytest.importorskip("jinja2")
    tpl = ("{{ bos_token }}{% for m in messages %}{% if m['role'] == 'system' %}[S]{{ m['content'] }}"
           "{% elif m['role'] == 'user' %}[U]{{ m['content'] }}{% else %}[A]{% generation %}{{ m['content'] }}"
           "{% endgeneration %}{{ eos_token }}{% endif %}{% endfor %}"
           "{% if add_generation_prompt %}[A]{% endif %}{{ {'k': 'é'} | tojson }}")
    t = ChatTemplate(tpl, bos_token="<s>", eos_token="</s>")
    msgs = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"}, {"role": "user", "content": [{"type": "text", "text": "a"},
                                                                                 {"type": "text", "text": "b"}]}]
    assert t.render(msgs) == '<s>[S]be nice[U]hi[A]yo</s>[U]ab[A]{"k": "é"}'
    assert t.render(msgs, add_generation_prompt=False).endswith('[U]ab{"k": "é"}')
    bad = ChatTemplate("{{ raise_exception('nope') }}")
    with pytest.raises(Exception, match="nope"):
        bad.render(msgs)
    with pytest.raises(Exception):
        ChatTemplate("{{ ().__class__.__mro__ }}{{ messages.append(1) }}").render(msgs)  # sandboxed


def test_chat_template_matches_transformers_rendering():
    pytest.importorskip("transformers")
    from transformers.utils.chat_template_utils import render_jinja_template
    tpl = ("{%- for m in messages %}\n  {{- '<|im_start|>' + m.role + '\\n' + m.content | trim + '<|im_end|>\\n' }}\n"
           "{%- endfor %}\n{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
           "{{ strftime_now('%Y') | length }}{{ bos_token }}")
    msgs = [{"role": "user", "content": " hello "}, {"role": "assistant", "content": "x"}]
    ours = ChatTemplate(tpl, bos_token="<B>").render(msgs, add_generation_prompt=True)
    theirs, _ = render_jinja_template(conversations=[msgs], chat_template=tpl, add_generation_prompt=True,
                                      bos_token="<B>")
    assert ours == theirs[0]


def test_chatml_fallback():
    t = ChatTemplate()
    assert t.is_fallback and t.end_of_turn == "<|im_end|>"
    assert t.render([{"role": "user", "content": "hi"}]) == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"


def test_normalize_messages():
    assert normalize_messages([{"role": "developer", "content": None}]) == [{"role": "system", "content": ""}]
    with pytest.raises(ValueError):
        normalize_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}])
    with pytest.raises(ValueError):
        normalize_messages([{"content": "x"}])


def _byte_level_tokenizer():
    tk = pytest.importorskip("tokenizers")
    from tokenizers import decoders, models, pre_tokenizers
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    t = tk.Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    return t


def test_stream_decoder_multibyte(tmp_path):
    from hearth.chat import Tokenizer
    raw = _byte_level_tokenizer()
    path = tmp_path / "tok.json"
    raw.save(str(path))
    tok = Tokenizer.from_file(path)
    text = "héllo 世界, ok 🙂!"
    ids = tok.encode(text)
    assert len(ids) == len(text.encode("utf-8"))
    dec = StreamDecoder(tok)
    parts = [dec.push(i) for i in ids] + [dec.flush()]
    assert "".join(parts) == text
    assert all("\ufffd" not in p for p in parts)


def test_text_stream_stop_strings_are_held_back():
    tok = FakeTokenizer()
    ts = TextStream(tok, ["STOP", "zzz"])
    out = []
    for i in tok.encode("hello ST"):
        out.append(ts.push(i))
    assert "".join(out) == "hello "                 # "ST" might begin a stop string
    for i in tok.encode("OP world"):
        out.append(ts.push(i))
        if ts.stopped:
            break
    assert ts.stopped == "STOP" and "".join(out) == "hello "
    ts3 = TextStream(tok, ["abc"])
    assert "".join(ts3.push(i) for i in tok.encode("xxab")) == "xx"
    assert ts3.push(tok.encode("x")[0]) == "abx"
    ts2 = TextStream(tok, ["STOP"])
    s = "".join(ts2.push(i) for i in tok.encode("a STOMP b")) + ts2.finish()
    assert s == "a STOMP b" and ts2.stopped is None
    assert ts.push(tok.encode("x")[0]) == "" and ts.finish() == ""      # nothing after a stop
    ts5 = TextStream(tok, ["ab"])
    assert ts5.push(tok.encode("a")[0]) + ts5.push(tok.encode("b")[0]) == "" and ts5.stopped == "ab"   # at offset 0

    class WordTokenizer:                    # one token can complete several stop strings at once
        words = ["x", "xabcd", "q"]

        def decode(self, ids, skip_special_tokens=True):
            return "".join(self.words[i] for i in ids)

    for stops, want in ((["ab", "abc"], "ab"), (["abc", "ab"], "abc"), (["bcd", "abc"], "abc")):
        ts4 = TextStream(WordTokenizer(), stops)
        out = ts4.push(0) + ts4.push(1) + ts4.push(2)
        assert ts4.stopped == want and out == "xx", stops       # earliest match; ties go to the first listed


def chat_engine(**kw):
    """Letters-only toy LM whose output depends on the exact KV content."""
    return FakeEngine(vocab=128, seed=3, lo=97, hi=123, crc_amp=1.5, **kw)


def test_conversation_reuses_prefix_and_matches_fresh_run():
    tok = FakeTokenizer()
    e = chat_engine()
    conv = Conversation(e, tok, ChatTemplate())
    r1 = "".join(conv.say("hello", max_new_tokens=12))
    assert len(r1) == 12 and conv.messages[-1] == {"role": "assistant", "content": r1}
    assert conv.held == e.kv
    e.evaluated = 0
    r2 = "".join(conv.say("again", max_new_tokens=12))
    prompt2 = conv.prompt_ids(conv.messages[:-1])
    assert conv.last_stats.reused_tokens > 0 and conv.held == e.kv
    assert e.evaluated < len(prompt2)               # only the new suffix was prefilled
    fresh, _ = generate_ids(chat_engine(), prompt2, 12)
    assert r2 == tok.decode(fresh)


def test_conversation_recovers_after_engine_error():
    tok = FakeTokenizer()
    e = chat_engine(max_seq=60)
    conv = Conversation(e, tok, ChatTemplate())
    with pytest.raises(ValueError):
        "".join(conv.say("x" * 80, max_new_tokens=4))   # prompt longer than the KV cache
    assert conv.held == e.kv
    conv.reset()
    assert len("".join(conv.say("hi", max_new_tokens=4))) == 4 and conv.held == e.kv


def test_conversation_say_rolls_back_a_failed_turn_and_keeps_a_cut_one():
    tok = FakeTokenizer()
    e = chat_engine(max_seq=300)
    conv = Conversation(e, tok, ChatTemplate())
    r1 = "".join(conv.say("hi", max_new_tokens=5))
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": r1}]
    assert conv.messages == history
    with pytest.raises(ValueError, match="KV cache holds 300"):
        "".join(conv.say("x" * 400, max_new_tokens=5))          # does not fit: the turn is discarded
    assert conv.messages == history and conv.held == e.kv
    orig, calls = e.eval, [0]

    def fail_third(tokens, all_logits=False):
        calls[0] += 1
        if calls[0] == 3:
            raise heng.HearthError("simulated I/O failure")
        return orig(tokens, all_logits)

    e.eval = fail_third
    got = []
    with pytest.raises(heng.HearthError):
        for d in conv.say("again", max_new_tokens=8):
            got.append(d)
    e.eval = orig
    assert got and conv.messages == history                       # streamed text, but the turn failed
    stream = conv.say("more", max_new_tokens=8)
    part = next(stream) + next(stream)
    stream.close()                                                # the consumer stopped early: keep what it got
    assert conv.messages == history + [{"role": "user", "content": "more"}, {"role": "assistant", "content": part}]
    reply = "".join(conv.say("last", max_new_tokens=4))
    fresh, _ = generate_ids(chat_engine(), conv.prompt_ids(conv.messages[:-1]), 4)
    assert reply == tok.decode(fresh) and conv.held == e.kv


def test_conversation_zero_token_turn_keeps_kv_in_sync():
    """Regression: generating 0 tokens used to leave `held` describing the new
    prompt while the engine still held the previous one, so the next turn
    reused the wrong KV."""
    tok = FakeTokenizer()
    e = chat_engine()
    conv = Conversation(e, tok, ChatTemplate())
    pa = conv.prompt_ids([{"role": "user", "content": "A" * 40}])
    pb = conv.prompt_ids([{"role": "user", "content": "B" * 60}])
    shared = common_prefix(pa, pb)
    assert shared == len("<|im_start|>user\n")
    assert len(list(conv.generate_ids(pa, 5))) == 5 and conv.held == e.kv
    assert list(conv.generate_ids(pb, 0)) == [] and conv.held == e.kv
    assert conv.last_stats.reused_tokens == shared and e.pos == shared
    got = [t.id for t in conv.generate_ids(pb, 10)]
    assert conv.held == e.kv and conv.last_stats.reused_tokens == shared
    assert got == generate_ids(chat_engine(), pb, 10)[0]
    conv.reset()
    assert "".join(conv.say("hi", max_new_tokens=0)) == "" and conv.held == e.kv
    assert conv.messages[-1] == {"role": "assistant", "content": ""}
    reply = "".join(conv.say("again", max_new_tokens=6))
    fresh, _ = generate_ids(chat_engine(), conv.prompt_ids(conv.messages[:-1]), 6)
    assert reply == tok.decode(fresh) and conv.held == e.kv


def test_conversation_interrupt_keeps_kv_tracking_consistent(monkeypatch):
    import hearth.generate as g
    tok = FakeTokenizer()
    e = chat_engine()
    conv = Conversation(e, tok, ChatTemplate())
    pa = conv.prompt_ids([{"role": "user", "content": "A" * 30}])
    pb = conv.prompt_ids([{"role": "user", "content": "B" * 50}])   # longer than pa + its reply
    list(conv.generate_ids(pa, 5))

    def interrupt(engine):
        raise KeyboardInterrupt

    # Ctrl-C before generate() touched the engine: it still holds the previous turn.
    with monkeypatch.context() as m:
        m.setattr(g, "kv_capacity", interrupt)
        with pytest.raises(KeyboardInterrupt):
            list(conv.generate_ids(pb, 5))
    assert conv.held == e.kv
    got = [t.id for t in conv.generate_ids(pb, 8)]
    assert got == generate_ids(chat_engine(), pb, 8)[0] and conv.held == e.kv

    # Ctrl-C mid-decode, after an engine call returned: the generated prefix is kept, not reset.
    orig, calls = e.eval, [0]

    def eval_then_interrupt(tokens, all_logits=False):
        r = orig(tokens, all_logits)
        calls[0] += 1
        if calls[0] == 4:
            raise KeyboardInterrupt
        return r

    e.eval = eval_then_interrupt
    with pytest.raises(KeyboardInterrupt):
        list(conv.generate_ids(pa, 20))
    e.eval = orig
    assert conv.held == e.kv and len(conv.held) == len(pa) + 3
    got = [t.id for t in conv.generate_ids(pb, 8)]
    assert got == generate_ids(chat_engine(), pb, 8)[0] and conv.held == e.kv
    assert conv.last_stats.reused_tokens == common_prefix(pa, pb)


def test_conversation_closed_mid_burst_keeps_its_cache():
    """A consumer that stops early (client disconnect, stop string) must leave the
    generated prefix cached, also when it stops inside an accepted draft burst."""
    mid_burst = 0
    for k in range(1, 45):
        e = FakeEngine(seed=1)
        conv = Conversation(e, FakeTokenizer(), ChatTemplate(), stop_ids=[])
        it = conv.generate_ids(REPETITIVE, 60, None, None, "ngram", 6)
        ids = [next(it).id for _ in range(k)]
        mid_burst += e.pos > len(REPETITIVE) + k
        it.close()
        assert conv.held == e.kv == (REPETITIVE + ids)[:e.pos]
        assert e.pos >= len(REPETITIVE) + k - 1, k
    assert mid_burst > 0


def test_generation_defaults_agree_across_front_ends():
    import inspect
    from hearth import chat, server
    from hearth.cli import build_parser
    fns = (generate, generate_ids, chat.Conversation.generate_ids, chat.Conversation.say, server.App.__init__,
           server.serve)
    params = [inspect.signature(f).parameters for f in fns]
    for name in ("speculative", "draft_len", "ngram_n"):
        assert {p[name].default for p in params} == {inspect.signature(generate).parameters[name].default}, name
    a = build_parser().parse_args(["serve", "m"])
    assert (a.speculative, a.draft_len, a.ngram_n) == ("none", 4, 3) == tuple(
        params[0][n].default for n in ("speculative", "draft_len", "ngram_n"))
    for name, cli_value in (("max_queue", a.max_queue), ("timeout", a.timeout)):
        assert params[4][name].default == params[5][name].default == cli_value, name
    assert (a.max_queue, a.timeout) == (8, 60.0)
    assert params[2]["max_new_tokens"].default == params[0]["max_new_tokens"].default == 256


# ---------------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------------

_DEFAULT = object()


@contextmanager
def running_server(engine, tokenizer=_DEFAULT, template=None, host="127.0.0.1", **kw):
    from hearth.server import make_server
    tok = FakeTokenizer() if tokenizer is _DEFAULT else tokenizer
    srv = make_server(engine, tok, template, host=host, port=0, **kw)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(5)


def assert_synced(app, timeout=10.0):
    """The server's record of the KV cache (Conversation.held) equals the FakeEngine's real KV."""
    deadline = time.time() + timeout
    while app.gate.busy or app.gate.waiting:   # the gate is released just after the reply is written
        assert time.time() < deadline, "server never became idle"
        time.sleep(0.002)
    assert app.conv.held == app.engine.kv, "Conversation.held disagrees with the engine's KV cache"


def request(srv, method, path, body=None, headers=None, raw=None, check=True):
    """One request on a fresh connection. check: afterwards assert the KV
    bookkeeping matches the FakeEngine (callers running requests concurrently
    pass False)."""
    host, port = srv.server_address[:2]
    c = http.client.HTTPConnection(host, port, timeout=20)
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    payload = r.read()
    c.close()
    if check and hasattr(srv.app.engine, "kv"):
        assert_synced(srv.app)
    return r.status, dict(r.getheaders()), payload


def sse_events(payload: bytes):
    events = []
    for block in payload.decode("utf-8").split("\n\n"):
        if not block.strip():
            continue
        ev, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        events.append((ev, data if data == "[DONE]" else json.loads(data)))
    return events


def server_engine(**kw):
    return FakeEngine(vocab=128, seed=4, lo=97, hi=123, crc_amp=1.5, **kw)


MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello there"}]


def expected_chat(msgs, n, **kw):
    tok = FakeTokenizer()
    ids = tok.encode(ChatTemplate().render(msgs))
    out, st = generate_ids(server_engine(**kw), ids, n)
    return ids, out, tok.decode(out), st


def test_server_basic_routes():
    with running_server(server_engine(), model_name="tiny") as srv:
        st, _, body = request(srv, "GET", "/health")
        assert st == 200 and json.loads(body)["status"] == "ok"
        st, _, body = request(srv, "GET", "/v1/models")
        assert st == 200 and json.loads(body)["data"][0]["id"] == "tiny"
        st, _, body = request(srv, "GET", "/stats")
        assert st == 200 and "server" in json.loads(body)
        st, _, body = request(srv, "GET", "/nope")
        assert st == 404 and "error" in json.loads(body)
        st, _, body = request(srv, "POST", "/v1/chat/completions", raw=b"{not json")
        assert st == 400 and json.loads(body)["error"]["type"] == "invalid_request_error"
        st, _, body = request(srv, "POST", "/v1/chat/completions", {"messages": []})
        assert st == 400
        st, _, body = request(srv, "POST", "/v1/chat/completions", raw=b"")       # Content-Length: 0
        assert st == 400 and "JSON object" in json.loads(body)["error"]["message"]
        st, _, body = request(srv, "POST", "/v1/messages", {"messages": [{"role": "user", "content": "x"}]})
        assert st == 400 and json.loads(body)["type"] == "error"


def test_openai_chat_non_stream():
    ids, out, text, _ = expected_chat(MSGS, 10)
    with running_server(server_engine()) as srv:
        st, _, body = request(srv, "POST", "/v1/chat/completions",
                              {"model": "x", "messages": MSGS, "max_tokens": 10, "temperature": 0})
    assert st == 200
    r = json.loads(body)
    assert r["object"] == "chat.completion" and r["id"].startswith("chatcmpl-") and len(r["id"]) > 20
    assert r["choices"][0]["message"] == {"role": "assistant", "content": text}
    assert r["choices"][0]["finish_reason"] == "length"
    assert r["usage"]["prompt_tokens"] == len(ids) and r["usage"]["completion_tokens"] == 10
    assert r["usage"]["total_tokens"] == len(ids) + 10


def test_openai_chat_stream_matches_non_stream():
    _, _, text, _ = expected_chat(MSGS, 15)
    with running_server(server_engine()) as srv:
        st, h, body = request(srv, "POST", "/v1/chat/completions",
                              {"messages": MSGS, "max_tokens": 15, "temperature": 0, "stream": True,
                               "stream_options": {"include_usage": True}})
    assert st == 200 and h["Content-Type"].startswith("text/event-stream")
    ev = sse_events(body)
    assert ev[-1] == (None, "[DONE]")
    chunks = [d for _, d in ev[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert content == text
    fin = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert len(fin) == 1 and fin[0]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 15
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_openai_stop_and_eos():
    _, _, text, _ = expected_chat(MSGS, 20)
    stop = text[5:8]
    cut = text.find(stop)
    with running_server(server_engine()) as srv:
        st, _, body = request(srv, "POST", "/v1/chat/completions",
                              {"messages": MSGS, "max_tokens": 20, "temperature": 0, "stop": [stop]})
        r = json.loads(body)
        assert r["choices"][0]["message"]["content"] == text[:cut]
        assert r["choices"][0]["finish_reason"] == "stop"
    ids = FakeTokenizer().encode(ChatTemplate().render(MSGS))
    with running_server(server_engine(eos=10, eos_at=len(ids) + 6)) as srv:
        st, _, body = request(srv, "POST", "/v1/chat/completions",
                              {"messages": MSGS, "max_tokens": 50, "temperature": 0})
        r = json.loads(body)
        assert r["choices"][0]["finish_reason"] == "stop"
        assert len(r["choices"][0]["message"]["content"]) == 6
        assert r["usage"]["completion_tokens"] == 7          # 6 text tokens + end-of-sequence


def test_server_prefix_reuse_across_requests():
    e = server_engine()
    with running_server(e) as srv:
        st, _, body = request(srv, "POST", "/v1/chat/completions",
                              {"messages": MSGS, "max_tokens": 8, "temperature": 0})
        reply = json.loads(body)["choices"][0]["message"]["content"]
        msgs2 = MSGS + [{"role": "assistant", "content": reply}, {"role": "user", "content": "more"}]
        e.evaluated = 0
        st, _, body = request(srv, "POST", "/v1/chat/completions",
                              {"messages": msgs2, "max_tokens": 8, "temperature": 0})
        r = json.loads(body)
        ids2, _, text2, _ = expected_chat(msgs2, 8)
        assert r["choices"][0]["message"]["content"] == text2
        assert r["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        assert e.evaluated < len(ids2)
        stats = json.loads(request(srv, "GET", "/stats")[2])["server"]
        ids1 = FakeTokenizer().encode(ChatTemplate().render(MSGS))
        assert stats["completed"] == stats["requests"] == 2 and stats["completion_tokens"] == 16
        assert stats["prompt_tokens"] == len(ids1) + len(ids2)
        assert stats["cached_tokens"] == r["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
        assert stats["held_tokens"] == len(ids2) + 7 and not stats["busy"] and stats["queue_waiting"] == 0


def test_server_speculative_matches_plain():
    _, _, text, _ = expected_chat(MSGS, 30)
    with running_server(server_engine(), speculative="ngram", draft_len=5) as srv:
        st, _, body = request(srv, "POST", "/v1/chat/completions",
                              {"messages": MSGS, "max_tokens": 30, "temperature": 0})
    assert json.loads(body)["choices"][0]["message"]["content"] == text


def test_completions_endpoint():
    tok = FakeTokenizer()
    out, _ = generate_ids(server_engine(), tok.encode("abc"), 6)
    with running_server(server_engine()) as srv:
        st, _, body = request(srv, "POST", "/v1/completions", {"prompt": "abc", "max_tokens": 6, "temperature": 0})
        r = json.loads(body)
        assert st == 200 and r["object"] == "text_completion" and r["id"].startswith("cmpl-")
        assert r["choices"][0]["text"] == tok.decode(out) and r["usage"]["prompt_tokens"] == 3
        for same in (["abc"], [tok.encode("abc")]):              # a batch of one is unwrapped
            st, _, body = request(srv, "POST", "/v1/completions", {"prompt": same, "max_tokens": 6, "temperature": 0})
            assert st == 200 and json.loads(body)["choices"][0]["text"] == tok.decode(out)
        for bad in (["a", "b"], [[1], [2]], [], [1, "x"], [True, 2], None):
            st, _, body = request(srv, "POST", "/v1/completions", {"prompt": bad, "max_tokens": 2})
            assert st == 400, bad
        st, _, body = request(srv, "POST", "/v1/completions", {"prompt": [0, 1], "max_tokens": 2, "temperature": 0})
        assert st == 200                                          # token id 0 is valid
        st, _, body = request(srv, "POST", "/v1/completions",
                              {"prompt": tok.encode("abc"), "max_tokens": 6, "temperature": 0, "echo": True})
        assert json.loads(body)["choices"][0]["text"] == "abc" + tok.decode(out)
        st, _, body = request(srv, "POST", "/v1/completions",
                              {"prompt": "abc", "max_tokens": 6, "temperature": 0, "stream": True})
        ev = sse_events(body)
        assert ev[-1][1] == "[DONE]"
        assert "".join(d["choices"][0]["text"] for _, d in ev[:-1]) == tok.decode(out)
        st, _, _ = request(srv, "POST", "/v1/completions", {"prompt": [1, 999], "max_tokens": 2})
        assert st == 400
        st, _, body = request(srv, "POST", "/v1/completions", {"prompt": "x" * 5000, "max_tokens": 2})
        assert st == 400 and "context" in json.loads(body)["error"]["message"]


def test_anthropic_messages():
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hello there"}]}]
    full = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello there"}]
    ids, _, text, _ = expected_chat(full, 9)
    with running_server(server_engine()) as srv:
        st, _, body = request(srv, "POST", "/v1/messages",
                              {"model": "m", "system": "sys", "messages": msgs, "max_tokens": 9, "temperature": 0})
        r = json.loads(body)
        assert st == 200 and r["type"] == "message" and r["role"] == "assistant" and r["id"].startswith("msg_")
        assert r["content"] == [{"type": "text", "text": text}]
        assert r["stop_reason"] == "max_tokens" and r["stop_sequence"] is None
        assert r["usage"]["input_tokens"] == len(ids) and r["usage"]["output_tokens"] == 9

        st, _, body = request(srv, "POST", "/v1/messages",
                              {"system": [{"type": "text", "text": "sys"}], "messages": msgs, "max_tokens": 9,
                               "temperature": 0, "stream": True})
        ev = sse_events(body)
        names = [e for e, _ in ev]
        assert names[:3] == ["message_start", "content_block_start", "ping"]
        assert names[-3:] == ["content_block_stop", "message_delta", "message_stop"]
        assert set(names[3:-3]) == {"content_block_delta"}
        assert all(e == d["type"] for e, d in ev)
        assert all(d["index"] == 0 for e, d in ev if e.startswith("content_block"))
        assert "".join(d["delta"]["text"] for e, d in ev if e == "content_block_delta") == text
        md = [d for e, d in ev if e == "message_delta"][0]
        assert md["delta"]["stop_reason"] == "max_tokens" and md["usage"]["output_tokens"] == 9
        assert ev[0][1]["message"]["usage"]["input_tokens"] == len(ids)

        stop = text[3:6]
        st, _, body = request(srv, "POST", "/v1/messages",
                              {"system": "sys", "messages": msgs, "max_tokens": 9, "temperature": 0,
                               "stop_sequences": [stop]})
        r = json.loads(body)
        assert r["stop_reason"] == "stop_sequence" and r["stop_sequence"] == stop
        assert r["content"][0]["text"] == text[:text.find(stop)]


def test_server_api_key():
    with running_server(server_engine(), api_key="sekrit") as srv:
        body = {"messages": MSGS, "max_tokens": 2, "temperature": 0}
        assert request(srv, "POST", "/v1/chat/completions", body)[0] == 401
        assert request(srv, "POST", "/v1/chat/completions", body, {"Authorization": "Bearer nope"})[0] == 401
        assert request(srv, "POST", "/v1/chat/completions", body, {"Authorization": "Bearer sekrit"})[0] == 200
        st, _, b = request(srv, "POST", "/v1/messages", {"messages": [{"role": "user", "content": "x"}],
                                                         "max_tokens": 2}, {"x-api-key": "bad"})
        assert st == 401 and json.loads(b)["error"]["type"] == "authentication_error"
        assert request(srv, "POST", "/v1/messages", {"messages": [{"role": "user", "content": "x"}],
                                                     "max_tokens": 2}, {"x-api-key": "sekrit"})[0] == 200
        assert request(srv, "GET", "/v1/models")[0] == 401
        assert request(srv, "GET", "/health")[0] == 200


def _post_async(srv, body, results, key):
    def go():
        results[key] = request(srv, "POST", "/v1/chat/completions", body, check=False)
    th = threading.Thread(target=go, daemon=True)
    th.start()
    return th


def test_server_429_when_queue_full():
    e = server_engine()
    e.release = threading.Event()
    body = {"messages": MSGS, "max_tokens": 3, "temperature": 0}
    results = {}
    with running_server(e, max_queue=1) as srv:
        ta = _post_async(srv, body, results, "a")
        assert e.entered.wait(10)                       # A holds the engine
        tb = _post_async(srv, body, results, "b")
        deadline = time.time() + 10
        while srv.app.gate.waiting < 1 and time.time() < deadline:
            time.sleep(0.01)
        assert srv.app.gate.waiting == 1                # B is queued
        st, h, b = request(srv, "POST", "/v1/chat/completions", body, check=False)
        assert st == 429 and h.get("Retry-After") == "1"
        assert json.loads(b)["error"]["type"] == "rate_limit_exceeded"
        st, _, b = request(srv, "POST", "/v1/messages", {"messages": [{"role": "user", "content": "x"}],
                                                         "max_tokens": 2}, check=False)
        assert st == 429 and json.loads(b)["error"]["type"] == "rate_limit_error"
        e.release.set()
        ta.join(10)
        tb.join(10)
        assert results["a"][0] == 200 and results["b"][0] == 200
        assert json.loads(request(srv, "GET", "/stats")[2])["server"]["rejected_busy"] == 2


def test_engine_gate_is_fifo_and_bounded():
    from hearth.server import Busy, EngineGate
    g = EngineGate(max_queue=3)
    g.acquire()                                     # the running request
    order, threads = [], []
    for i in range(3):
        def waiter(i=i):
            g.acquire()
            order.append(i)
            time.sleep(0.01)
            g.release()
        th = threading.Thread(target=waiter, daemon=True)
        th.start()
        threads.append(th)
        deadline = time.time() + 10
        while g.waiting < i + 1:                    # enqueue strictly one after another
            assert time.time() < deadline
            time.sleep(0.002)
    assert g.busy and g.waiting == 3
    with pytest.raises(Busy):
        g.acquire()                                 # queue full
    assert g.waiting == 3
    g.release()
    for th in threads:
        th.join(10)
    assert order == [0, 1, 2]                       # served in arrival order
    assert not g.busy and g.waiting == 0
    g.acquire()                                     # an idle gate is taken without waiting
    g.release()
    g0 = EngineGate(max_queue=0)
    g0.acquire()
    with pytest.raises(Busy):
        g0.acquire()


def test_engine_gate_cancelled_waiter_hands_over():
    """A waiter that is woken for its turn and then fails (exception inside wait)
    must wake the next one, or the engine sits idle with a queue behind it."""
    from hearth.server import EngineGate
    g = EngineGate(max_queue=4)
    b_rewaiting = threading.Event()

    class Cond(threading.Condition):
        def wait(self, timeout=None):
            name = threading.current_thread().name
            if name == "B" and not g._active:
                b_rewaiting.set()                  # B saw A still at the head and sleeps again
            r = super().wait(timeout)
            if name == "A" and not g._active:      # A's turn came: fail only once B is asleep again
                self.release()
                assert b_rewaiting.wait(10)
                self.acquire()
                raise RuntimeError("cancelled while waiting")
            return r

    g._cv = Cond()
    g.acquire()
    errors, served = [], []

    def a():
        try:
            g.acquire()
        except RuntimeError as e:
            errors.append(e)

    def b():
        g.acquire()
        served.append("B")
        g.release()

    threads = []
    for name, fn in (("A", a), ("B", b)):
        th = threading.Thread(target=fn, name=name, daemon=True)
        th.start()
        threads.append(th)
        deadline = time.time() + 10
        while g.waiting < len(threads):
            assert time.time() < deadline
            time.sleep(0.002)
    g.release()
    for th in threads:
        th.join(10)
    assert len(errors) == 1 and served == ["B"], (errors, served)
    assert not g.busy and g.waiting == 0


def test_server_client_disconnect_mid_stream():
    import socket
    e = server_engine()
    orig = e.eval

    def slow(tokens, all_logits=False):
        time.sleep(0.002)
        return orig(tokens, all_logits)

    e.eval = slow
    with running_server(e) as srv:
        body = json.dumps({"messages": MSGS, "max_tokens": 400, "temperature": 0, "stream": True}).encode()
        s = socket.create_connection(srv.server_address[:2], timeout=10)
        head = (b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                b"Content-Length: %d\r\n\r\n" % len(body))
        s.sendall(head + body)
        got = b""
        while got.count(b'"content"') < 4:          # role chunk + a few generated deltas
            got += s.recv(4096)
        assert got.startswith(b"HTTP/1.1 200")
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))  # close with RST
        s.close()
        deadline = time.time() + 20
        while time.time() < deadline and (srv.app.gate.busy or srv.app.counters["completed"] < 1):
            time.sleep(0.01)
        assert not srv.app.gate.busy and srv.app.counters["completed"] == 1
        assert srv.app.counters["completion_tokens"] < 400  # generation stopped at the disconnect
        e.eval = orig
        held = srv.app.conv.held
        assert len(held) == e.pos and e.kv == held   # cleanup ran before the engine was released
        _, _, text, _ = expected_chat(MSGS, 7)
        st, _, b = request(srv, "POST", "/v1/chat/completions", {"messages": MSGS, "max_tokens": 7, "temperature": 0})
        assert st == 200 and json.loads(b)["choices"][0]["message"]["content"] == text


def _read_until_closed(s) -> bytes:
    data = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            return data
        data += chunk


def test_server_drops_a_client_that_stops_reading():
    """Regression: a streaming client that stopped reading blocked the server in
    a send while it held the engine, so every other request queued or got 429."""
    import socket
    e = server_engine(max_seq=40000)
    n = 30000
    with running_server(e, timeout=0.5, max_queue=1) as srv:
        body = json.dumps({"prompt": "abc", "max_tokens": n, "temperature": 0, "stream": True}).encode()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        s.connect(srv.server_address[:2])
        try:
            s.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
            assert e.entered.wait(10)                   # never read: the socket buffers fill up
            deadline = time.time() + 20
            while (srv.app.gate.busy or srv.app.counters["completed"] < 1) and time.time() < deadline:
                time.sleep(0.02)
            assert not srv.app.gate.busy, "a client that stopped reading still holds the engine"
            c = srv.app.counters
            assert c["completed"] == 1 and c["completion_tokens"] < n and c["errors"] == 0
            st, _, b = request(srv, "POST", "/v1/completions", {"prompt": "abc", "max_tokens": 3, "temperature": 0})
            assert st == 200 and len(json.loads(b)["choices"][0]["text"]) == 3
        finally:
            s.close()


def test_server_closes_idle_connections_after_the_timeout():
    import socket
    with running_server(server_engine(), timeout=0.3) as srv:
        s = socket.create_connection(srv.server_address[:2], timeout=10)
        t0 = time.time()
        try:
            assert s.recv(16) == b"" and time.time() - t0 < 8   # closed by the server, not by our timeout
        finally:
            s.close()
        st, _, _ = request(srv, "GET", "/health")
        assert st == 200


def test_server_writes_non_stream_replies_after_releasing_the_engine(monkeypatch):
    from hearth import server as hs
    seen = []
    orig = hs.Handler._send_json

    def spy(self, status, obj, headers=None):
        seen.append((status, self.app.gate.busy))
        return orig(self, status, obj, headers)

    monkeypatch.setattr(hs.Handler, "_send_json", spy)
    with running_server(server_engine()) as srv:
        for path, body in (("/v1/chat/completions", {"messages": MSGS, "max_tokens": 3}),
                           ("/v1/completions", {"prompt": "ab", "max_tokens": 3}),
                           ("/v1/messages", {"messages": MSGS[1:], "max_tokens": 3})):
            assert request(srv, "POST", path, body)[0] == 200
    assert [st for st, _ in seen] == [200] * 3
    assert not any(busy for _, busy in seen), "a reply was written while the engine was held"


def test_server_streams_to_http10_clients_without_chunked_encoding():
    import socket
    text = expected_chat(MSGS, 6)[2]
    with running_server(server_engine()) as srv:
        body = json.dumps({"messages": MSGS, "max_tokens": 6, "temperature": 0, "stream": True}).encode()
        for version in (b"HTTP/1.0", b"HTTP/1.1"):
            s = socket.create_connection(srv.server_address[:2], timeout=10)
            s.sendall(b"POST /v1/chat/completions " + version + b"\r\nHost: x\r\nConnection: close\r\n"
                      b"Content-Length: %d\r\n\r\n" % len(body) + body)
            data = _read_until_closed(s)
            s.close()
            head, _, payload = data.partition(b"\r\n\r\n")
            assert head.startswith(b"HTTP/1.1 200"), head
            chunked = b"transfer-encoding: chunked" in head.lower()
            assert chunked == (version == b"HTTP/1.1"), head
            if chunked:
                payload = _dechunk(payload)
            else:                           # close-delimited: the payload is the bare event stream
                assert b"connection: close" in head.lower()
            ev = sse_events(payload)
            assert ev[-1] == (None, "[DONE]")
            assert "".join(d["choices"][0]["delta"].get("content", "") for _, d in ev[:-1]) == text
        assert_synced(srv.app)


def _dechunk(payload: bytes) -> bytes:
    out, p = b"", 0
    while True:
        eol = payload.index(b"\r\n", p)
        n = int(payload[p:eol], 16)
        if n == 0:
            assert payload[eol:] == b"\r\n\r\n", payload[eol:]
            return out
        out += payload[eol + 2:eol + 2 + n]
        assert payload[eol + 2 + n:eol + 4 + n] == b"\r\n"
        p = eol + 4 + n


def test_server_rejects_non_boolean_flags():
    with running_server(server_engine()) as srv:
        for path, base in (("/v1/chat/completions", {"messages": MSGS, "max_tokens": 2}),
                           ("/v1/completions", {"prompt": "ab", "max_tokens": 2}),
                           ("/v1/messages", {"messages": MSGS[1:], "max_tokens": 2})):
            for bad, param in (({"stream": "false"}, "stream"), ({"stream": 0}, "stream")):
                st, h, body = request(srv, "POST", path, dict(base, **bad))
                assert st == 400 and h["Content-Type"].startswith("application/json"), (path, bad)
                err = json.loads(body)["error"]
                assert param in err["message"] and "boolean" in err["message"]
            st, h, _ = request(srv, "POST", path, dict(base, stream=None))
            assert st == 200 and h["Content-Type"].startswith("application/json")   # null = the default
        for bad, param in (({"echo": "yes"}, "echo"),):
            st, _, body = request(srv, "POST", "/v1/completions", {"prompt": "ab", "max_tokens": 2, **bad})
            assert st == 400 and json.loads(body)["error"]["param"] == param
        for opts in ("x", {"include_usage": "true"}):
            st, _, body = request(srv, "POST", "/v1/chat/completions",
                                  {"messages": MSGS, "max_tokens": 2, "stream": True, "stream_options": opts})
            assert st == 400 and json.loads(body)["error"]["param"] in ("stream_options", "include_usage")
        st, _, body = request(srv, "POST", "/v1/chat/completions",
                              {"messages": MSGS, "max_tokens": 2, "stream": True, "stream_options": None})
        assert st == 200 and sse_events(body)[-1] == (None, "[DONE]")


def test_server_binds_ipv6_hosts():
    import socket
    try:
        probe = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        probe.bind(("::1", 0))
        probe.close()
    except (OSError, AttributeError):
        pytest.skip("no IPv6 loopback")
    with running_server(server_engine(), host="::1") as srv:
        port = srv.server_address[1]
        assert srv.url == f"http://[::1]:{port}"
        c = http.client.HTTPConnection("::1", port, timeout=10)
        c.request("GET", "/health")
        r = c.getresponse()
        assert r.status == 200 and json.loads(r.read())["status"] == "ok"
        c.close()
    with running_server(server_engine()) as srv:
        assert srv.url == f"http://127.0.0.1:{srv.server_address[1]}"


def test_server_is_quiet_about_vanished_clients(capsys):
    """Clients that reset a keep-alive connection (the openai/anthropic SDKs do)
    must not print tracebacks; genuine handler errors still do."""
    import socket
    errors = []
    with running_server(server_engine()) as srv:
        orig = srv.handle_error

        def spy(request_, addr):
            errors.append(sys.exc_info()[1])
            orig(request_, addr)

        srv.handle_error = spy
        host, port = srv.server_address[:2]
        c = http.client.HTTPConnection(host, port, timeout=10)
        c.request("GET", "/health")
        assert c.getresponse().read()
        c.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        c.sock.close()                               # RST while the server waits for the next request
        deadline = time.time() + 10
        while not errors and time.time() < deadline:
            time.sleep(0.01)
        assert errors and isinstance(errors[0], ConnectionError), errors
        for exc in (TimeoutError("slow"), BrokenPipeError(32, "pipe")):
            try:
                raise exc
            except OSError:
                orig(None, ("127.0.0.1", 1))
        assert "Traceback" not in capsys.readouterr().err
        try:
            raise RuntimeError("a real bug")
        except RuntimeError:
            orig(None, ("127.0.0.1", 1))
        assert "RuntimeError: a real bug" in capsys.readouterr().err


def test_server_defaults_and_edge_parameters():
    import socket
    ids = FakeTokenizer().encode(ChatTemplate().render(MSGS))
    with running_server(server_engine(max_seq=120), timeout=2) as srv:
        st, _, body = request(srv, "POST", "/v1/chat/completions", {"messages": MSGS, "temperature": 0})
        r = json.loads(body)                            # no max_tokens: fill the context
        assert st == 200 and r["usage"]["completion_tokens"] == 120 - len(ids)
        assert r["choices"][0]["finish_reason"] == "length"
        assert r["choices"][0]["message"]["content"] == expected_chat(MSGS, 120 - len(ids), max_seq=120)[2]
        assert request(srv, "POST", "/v1/chat/completions", {"messages": MSGS, "max_tokens": 2, "seed": 0})[0] == 200
        st, _, body = request(srv, "POST", "/v1/chat/completions", {"messages": MSGS, "max_tokens": 2, "seed": -1})
        assert st == 400 and json.loads(body)["error"]["param"] == "seed"
        blocks = [{"type": "text", "text": "s"}, {"type": "image", "text": "IGNORED"}, "junk", {"type": "text", "text": "ys"}]
        st, _, body = request(srv, "POST", "/v1/messages", {"system": blocks, "messages": MSGS[1:], "max_tokens": 4,
                                                             "temperature": 0})
        assert st == 200 and json.loads(body)["content"][0]["text"] == expected_chat(MSGS, 4, max_seq=120)[2]
        for length in (b"-5", b"12abc"):
            s = socket.create_connection(srv.server_address[:2], timeout=10)
            s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: " + length + b"\r\n\r\n")
            data = _read_until_closed(s)
            s.close()
            assert data.startswith(b"HTTP/1.1 400") and b"bad Content-Length" in data, data[:200]


def test_server_generation_failures_on_every_endpoint():
    e = server_engine()
    with running_server(e) as srv:
        orig = e.eval

        def broken(tokens, all_logits=False):
            raise heng.HearthError("simulated I/O failure")

        e.eval = broken
        for path, body, kind in (("/v1/chat/completions", {"messages": MSGS, "max_tokens": 3}, "server_error"),
                                 ("/v1/completions", {"prompt": "ab", "max_tokens": 3}, "server_error"),
                                 ("/v1/messages", {"messages": MSGS[1:], "max_tokens": 3}, "api_error")):
            st, _, b = request(srv, "POST", path, body)
            r = json.loads(b)
            assert st == 500 and r["error"]["type"] == kind and "simulated" in r["error"]["message"], path
            st, h, b = request(srv, "POST", path, dict(body, stream=True))
            assert st == 200 and h["Content-Type"].startswith("text/event-stream")
            errs = [d for _, d in sse_events(b) if isinstance(d, dict) and "error" in d]
            assert len(errs) == 1 and errs[0]["error"]["type"] == kind and "simulated" in errs[0]["error"]["message"]
        e.eval = orig
        assert json.loads(request(srv, "GET", "/stats")[2])["server"]["errors"] == 6
        assert request(srv, "POST", "/v1/completions", {"prompt": "ab", "max_tokens": 3})[0] == 200


def test_server_timeout_is_validated(tmp_path):
    from hearth.server import App, serve
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="timeout"):
            App(server_engine(), FakeTokenizer(), timeout=bad)
        with pytest.raises(ValueError, match="timeout"):
            serve(str(tmp_path / "missing.hearth"), timeout=bad)       # before the container is read
    assert App(server_engine(), FakeTokenizer(), timeout=None).timeout is None
    assert App(server_engine(), FakeTokenizer()).timeout == 60.0
    with pytest.raises(OSError):                         # 0.5 s is valid: it gets as far as reading the file
        serve(str(tmp_path / "missing.hearth"), timeout=0.5)


def test_server_rejects_messages_it_cannot_render():
    pytest.importorskip("jinja2")
    image = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    with running_server(server_engine()) as srv:
        for path, extra in (("/v1/chat/completions", {}), ("/v1/messages", {})):
            st, _, body = request(srv, "POST", path, dict(messages=image, max_tokens=2, **extra))
            r = json.loads(body)
            assert st == 400 and "unsupported content part 'image_url'" in r["error"]["message"], (path, r)
    with running_server(server_engine(), template=ChatTemplate("{{ raise_exception('roles must alternate') }}")) as srv:
        for path in ("/v1/chat/completions", "/v1/messages"):
            st, _, body = request(srv, "POST", path, {"messages": MSGS[1:], "max_tokens": 2})
            r = json.loads(body)
            assert st == 400 and r["error"]["message"] == "chat template failed: roles must alternate", (path, r)


def test_server_engine_failure_is_reported_and_recovers():
    e = server_engine(max_seq=200)
    with running_server(e) as srv:
        orig = e.eval
        calls = {"n": 0}

        def flaky(tokens, all_logits=False):
            calls["n"] += 1
            if calls["n"] in (3, 6):        # one failure in each of the first two requests
                raise heng.HearthError("simulated I/O failure")
            return orig(tokens, all_logits)

        e.eval = flaky
        st, _, b = request(srv, "POST", "/v1/chat/completions", {"messages": MSGS, "max_tokens": 5, "temperature": 0})
        assert st == 500 and "simulated" in json.loads(b)["error"]["message"]
        st, _, b = request(srv, "POST", "/v1/chat/completions",
                           {"messages": MSGS, "max_tokens": 5, "temperature": 0, "stream": True})
        assert st == 200                    # headers were already sent: the failure is an SSE error event
        errs = [d["error"] for _, d in sse_events(b) if isinstance(d, dict) and "error" in d]
        assert len(errs) == 1 and "simulated" in errs[0]["message"] and errs[0]["type"] == "server_error"
        _, _, text, _ = expected_chat(MSGS, 5, max_seq=200)
        st, _, b = request(srv, "POST", "/v1/chat/completions", {"messages": MSGS, "max_tokens": 5, "temperature": 0})
        assert st == 200 and json.loads(b)["choices"][0]["message"]["content"] == text
        assert json.loads(request(srv, "GET", "/stats")[2])["server"]["errors"] == 2


def test_server_zero_max_tokens_keeps_kv_in_sync():
    """Regression: a max_tokens=0 request used to leave the server's KV record
    describing its prompt while the engine still held the previous request,
    so the next request reused the wrong KV and returned wrong text."""
    a = [{"role": "user", "content": "A" * 40}]
    b = [{"role": "user", "content": "B" * 60}]
    with running_server(server_engine()) as srv:
        def chat(msgs, **kw):
            st, _, body = request(srv, "POST", "/v1/chat/completions", dict(messages=msgs, temperature=0, **kw))
            return st, json.loads(body)

        assert chat(a, max_tokens=5)[0] == 200
        st, r = chat(b, max_tokens=0)                   # as OpenAI: chat needs max_tokens >= 1
        assert st == 400 and r["error"]["param"] == "max_tokens"
        st, r = chat(b, max_completion_tokens=0)
        assert st == 400 and r["error"]["param"] == "max_completion_tokens"
        st, _, body = request(srv, "POST", "/v1/completions",   # legacy completions accept 0
                              {"prompt": ChatTemplate().render(b), "max_tokens": 0, "temperature": 0})
        r = json.loads(body)
        assert st == 200 and r["choices"][0]["text"] == "" and r["usage"]["completion_tokens"] == 0
        assert r["choices"][0]["finish_reason"] == "length"
        st, r = chat(b, max_tokens=10)
        assert st == 200 and r["choices"][0]["message"]["content"] == expected_chat(b, 10)[2]
        assert r["usage"]["prompt_tokens_details"]["cached_tokens"] == len("<|im_start|>user\n")


def test_server_alternating_prompts_match_fresh_runs():
    """Prefix reuse across unrelated, diverging and extended prompts on every
    endpoint: each reply equals a fresh engine's and the KV record stays exact."""
    a = [{"role": "user", "content": "A" * 30}]
    b = [{"role": "system", "content": "sys"}, {"role": "user", "content": "B" * 45}]
    tok = FakeTokenizer()
    with running_server(server_engine()) as srv:
        for msgs in (a, b, a, b[:1] + [{"role": "user", "content": "B" * 20}], a + [{"role": "user", "content": "x"}]):
            st, _, body = request(srv, "POST", "/v1/chat/completions",
                                  {"messages": msgs, "max_tokens": 6, "temperature": 0})
            assert json.loads(body)["choices"][0]["message"]["content"] == expected_chat(msgs, 6)[2]
            st, _, body = request(srv, "POST", "/v1/chat/completions",
                                  {"messages": msgs, "max_tokens": 4, "temperature": 0, "stream": True})
            got = "".join(d["choices"][0]["delta"].get("content", "") for _, d in sse_events(body)[:-1])
            assert got == expected_chat(msgs, 4)[2]
            prompt = tok.encode(ChatTemplate().render(msgs))[:-3]
            st, _, body = request(srv, "POST", "/v1/completions", {"prompt": prompt, "max_tokens": 5, "temperature": 0})
            assert json.loads(body)["choices"][0]["text"] == tok.decode(generate_ids(server_engine(), prompt, 5)[0])
        st, _, body = request(srv, "POST", "/v1/messages", {"system": "sys", "messages": b[1:], "max_tokens": 7,
                                                             "temperature": 0})
        assert json.loads(body)["content"][0]["text"] == expected_chat(b, 7)[2]


def test_server_keepalive_survives_error_replies():
    """Error replies sent before the body was read (401, 404, GET with a body)
    must still consume it, or the next request on the connection is garbage."""
    with running_server(server_engine(), api_key="k") as srv:
        host, port = srv.server_address[:2]
        c = http.client.HTTPConnection(host, port, timeout=20)
        body = json.dumps({"messages": MSGS, "max_tokens": 3, "temperature": 0}).encode()
        json_h = {"Content-Type": "application/json"}
        auth = dict(json_h, Authorization="Bearer k")
        want = expected_chat(MSGS, 3)[2]

        def call(method, path, data, headers):
            c.request(method, path, body=data, headers=headers)
            r = c.getresponse()
            return r.status, r.read()

        assert call("POST", "/v1/chat/completions", body, json_h)[0] == 401
        sock = c.sock
        assert sock is not None
        for method, path, data, headers, status in (
                ("POST", "/v1/chat/completions", body, auth, 200),
                ("POST", "/v1/nope", body, auth, 404),
                ("POST", "/v1/chat/completions", body, auth, 200),
                ("GET", "/health", b"ignored body", {}, 200),
                ("GET", "/health", b"x", {}, 200),
                ("GET", "/health", b"", {}, 200),
                ("GET", "/v1/models", b"x" * 100, {}, 401),
                ("POST", "/v1/messages", body, dict(json_h, **{"x-api-key": "bad"}), 401),
                ("POST", "/v1/chat/completions", json.dumps({"messages": MSGS, "max_tokens": 0}).encode(), auth, 400),
                ("POST", "/v1/chat/completions", body, auth, 200)):
            st, payload = call(method, path, data, headers)
            assert st == status, (method, path, payload[:200])
            if status == 200 and path == "/v1/chat/completions":
                assert json.loads(payload)["choices"][0]["message"]["content"] == want
            assert c.sock is sock, "the keep-alive connection was dropped"
        c.close()
        assert_synced(srv.app)


def test_server_closes_connection_rather_than_read_a_large_unwanted_body():
    import socket
    from hearth.server import DRAIN_UNAUTH
    with running_server(server_engine(), api_key="k") as srv:
        for framing in (b"Content-Length: %d\r\n" % (DRAIN_UNAUTH + 1), b"Transfer-Encoding: chunked\r\n",
                        b"Content-Length: abc\r\n"):
            s = socket.create_connection(srv.server_address[:2], timeout=5)
            s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n" + framing + b"\r\n")
            data = b""
            while True:
                chunk = s.recv(65536)            # times out (fails) if the server waits for the body
                if not chunk:
                    break
                data += chunk
            s.close()
            assert data.startswith(b"HTTP/1.1 401"), data[:100]


def test_server_request_validation_and_routing_details():
    e = server_engine(max_seq=120)
    with running_server(e, model_name="tiny", cors="*") as srv:
        st, h, body = request(srv, "GET", "/v1/models/tiny")
        assert st == 200 and json.loads(body)["id"] == "tiny" and h["Access-Control-Allow-Origin"] == "*"
        assert request(srv, "GET", "/v1/models/other")[0] == 404
        st, h, _ = request(srv, "OPTIONS", "/v1/chat/completions")
        assert st == 204 and h["Access-Control-Allow-Origin"] == "*"
        st, h, body = request(srv, "POST", "/v1/chat/completions/?trace=1",      # query string, trailing slash
                              {"messages": MSGS, "max_tokens": 2, "temperature": 0.5, "seed": 1})
        assert st == 200 and h["Access-Control-Allow-Origin"] == "*"
        assert json.loads(body)["usage"]["completion_tokens"] == 2
        for bad in ({"max_tokens": 2.5}, {"seed": 1.5}, {"n": 2}, {"temperature": -1}, {"top_p": 1.5},
                    {"temperature": "hot"}, {"max_tokens": True}, {"stop": 5}, {"stop": ["s"] * 17}):
            st, _, body = request(srv, "POST", "/v1/chat/completions", dict({"messages": MSGS, "max_tokens": 2}, **bad))
            assert st == 400 and json.loads(body)["error"]["type"] == "invalid_request_error", bad
        for path, req in (("/v1/chat/completions", {"messages": MSGS}), ("/v1/completions", {"prompt": "x"})):
            st, _, body = request(srv, "POST", path, dict(req, n=2, max_tokens=2))
            assert st == 400 and json.loads(body)["error"]["message"] == "only n=1 is supported"
        # KV capacity 120: a full prompt cannot generate; one free slot gives exactly one token
        st, _, body = request(srv, "POST", "/v1/completions", {"prompt": [97] * 120, "max_tokens": 5})
        assert st == 400 and "context" in json.loads(body)["error"]["message"]
        st, _, body = request(srv, "POST", "/v1/completions", {"prompt": [97] * 119, "max_tokens": 5, "temperature": 0})
        r = json.loads(body)
        assert st == 200 and r["usage"]["completion_tokens"] == 1 and r["choices"][0]["finish_reason"] == "length"
        for bad in ({"messages": []}, {"messages": "hi"}, {"messages": [{"role": "system", "content": "x"}]},
                    {"messages": [{"role": "user", "content": "x"}], "system": 5}):
            st, _, body = request(srv, "POST", "/v1/messages", dict(bad, max_tokens=2))
            assert st == 400 and json.loads(body)["type"] == "error", bad
        assert json.loads(request(srv, "GET", "/stats")[2])["server"]["errors"] == 0   # 4xx are not errors


def test_anthropic_assistant_prefill_continues_the_turn():
    msgs = [{"role": "user", "content": "hello there"}, {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "more"}, {"role": "assistant", "content": "Sure"}]
    tok = FakeTokenizer()
    ids = tok.encode(ChatTemplate().render([{"role": "system", "content": "sys"}] + msgs[:3]) + "Sure")
    want = tok.decode(generate_ids(server_engine(), ids, 6)[0])
    with running_server(server_engine()) as srv:
        st, _, body = request(srv, "POST", "/v1/messages",
                              {"system": "sys", "messages": msgs, "max_tokens": 6, "temperature": 0})
        r = json.loads(body)
        assert st == 200 and r["content"][0]["text"] == want and r["usage"]["input_tokens"] == len(ids)


def test_server_sampling_parameters_reach_the_sampler():
    tok = FakeTokenizer()
    ids = tok.encode(ChatTemplate().render(MSGS))

    def want(n, **kw):
        return tok.decode(generate_ids(server_engine(), ids, n, Sampler(**kw))[0])

    greedy = want(8, temperature=0.0)
    assert want(8, temperature=1.0, seed=3) != greedy          # the checks below can tell them apart
    full = dict(temperature=1.0, seed=3, top_k=5, top_p=0.9, min_p=0.05, repetition_penalty=1.2)
    with running_server(server_engine()) as srv:
        def chat(**kw):
            st, _, body = request(srv, "POST", "/v1/chat/completions", dict(dict(messages=MSGS, max_tokens=8), **kw))
            assert st == 200, body
            return json.loads(body)["choices"][0]["message"]["content"]

        assert chat(temperature=1.0, seed=3) == want(8, temperature=1.0, seed=3)
        assert chat(**full) == want(8, **full)
        assert chat(temperature=1.5, top_p=0) == greedy            # an empty nucleus means greedy
        assert chat(seed=3) == want(8, temperature=1.0, seed=3)    # OpenAI default temperature is 1
        assert len(chat(max_completion_tokens=2, max_tokens=5, temperature=0)) == 2
        assert len(chat(max_completion_tokens=1, temperature=0)) == 1


def test_server_verbose_request_log(capsys):
    with running_server(server_engine(), verbose=True) as srv:
        assert request(srv, "GET", "/health")[0] == 200
    err = capsys.readouterr().err
    assert "[hearth.server]" in err and "GET /health" in err


def test_serve_wires_container_tokenizer_and_engine(tmp_path, monkeypatch):
    import hearth.engine
    import hearth.server as hs
    monkeypatch.setitem(sys.modules, "hearth.format", None)     # metadata via the built-in parser
    _byte_level_tokenizer().save(str(tmp_path / "tok.json"))
    with_bos, no_bos = tmp_path / "m.hearth", tmp_path / "plain.hearth"
    _write_container(with_bos, [("bos_id", 1, struct.pack("<I", 7)), ("eos_ids", 5, struct.pack("<II", 1, 3)),
                                ("tokenizer", 4, struct.pack("<I", 8) + b"tok.json")])
    _write_container(no_bos, [("eos_ids", 5, struct.pack("<II", 1, 4))])
    seen = []

    class FakeServedEngine(FakeEngine):
        def __init__(self, path, **kw):
            super().__init__(vocab=256)
            self.path, self.kw, self.closed = path, kw, False

        def close(self):
            self.closed = True

    def fake_forever(self, *a, **k):
        seen.append(self.app)
        raise KeyboardInterrupt

    monkeypatch.setattr(hearth.engine, "Engine", FakeServedEngine)
    monkeypatch.setattr(hs.HearthServer, "serve_forever", fake_forever)
    hs.serve(with_bos, port=0, engine_kwargs={"cache_gb": 1.5}, api_key="k", speculative="ngram")
    app = seen[-1]
    assert app.engine.kw == {"cache_gb": 1.5} and app.engine.closed and app.engine.path == with_bos
    assert app.tokenizer.bos_id == 7 and app.tokenizer.eos_ids == [3] and 3 in app.conv.stop_ids
    assert app.model_name == "m" and app.api_key == "k" and app.speculative == "ngram"
    assert app.conv.template.is_fallback
    hs.serve(no_bos, port=0, tokenizer_path=tmp_path / "tok.json", model_name="named")
    app = seen[-1]
    assert app.tokenizer.bos_id == -1 and app.tokenizer.eos_ids == [4] and app.model_name == "named"


def test_server_without_tokenizer_serves_token_id_completions():
    prompt = [104, 105, 33, 10, 104]
    want, _ = generate_ids(server_engine(), prompt, 6)
    with running_server(server_engine(), tokenizer=None) as srv:
        st, _, body = request(srv, "POST", "/v1/completions", {"prompt": prompt, "max_tokens": 6, "temperature": 0})
        r = json.loads(body)
        assert st == 200, r
        assert r["choices"][0]["token_ids"] == want and r["choices"][0]["text"] == ""
        assert r["usage"] == {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}
        st, _, body = request(srv, "POST", "/v1/completions",
                              {"prompt": prompt + [99], "max_tokens": 4, "temperature": 0, "stream": True})
        ev = sse_events(body)
        assert st == 200 and ev[-1][1] == "[DONE]"
        ids = [t for _, d in ev[:-1] for t in d["choices"][0]["token_ids"]]
        assert ids == generate_ids(server_engine(), prompt + [99], 4)[0]
        assert ev[-2][1]["choices"][0]["finish_reason"] == "length"
        for path, req, needle in (
                ("/v1/completions", {"prompt": "text", "max_tokens": 2}, "token ids"),
                ("/v1/completions", {"prompt": prompt, "max_tokens": 2, "stop": ["x"]}, "tokenizer"),
                ("/v1/completions", {"prompt": prompt, "max_tokens": 2, "echo": True}, "tokenizer"),
                ("/v1/chat/completions", {"messages": MSGS, "max_tokens": 2}, "token-id"),
                ("/v1/messages", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 2}, "tokenizer")):
            st, _, body = request(srv, "POST", path, req)
            r = json.loads(body)
            assert st == 400 and needle in r["error"]["message"], (path, r)


def test_bad_speculative_settings_fail_at_startup(tmp_path, capsys):
    from hearth.cli import main
    from hearth.server import App, serve
    for kw in (dict(speculative="ngram", ngram_n=0), dict(draft_len=-1), dict(speculative="medusa"),
               dict(draft_len=2.5)):
        with pytest.raises(ValueError):
            App(server_engine(), FakeTokenizer(), **kw)
    app = App(server_engine(), FakeTokenizer(), speculative="prompt_lookup", draft_len=np.int64(3))
    assert app.speculative == "ngram" and app.draft_len == 3
    missing = str(tmp_path / "missing.hearth")
    with pytest.raises(ValueError, match="ngram_n"):
        serve(missing, ngram_n=0)                     # rejected before the container is even read
    for argv in (["serve", missing, "--ngram-n", "0"], ["serve", missing, "--speculative", "ngram", "--draft-len", "-1"],
                 ["run", missing, "--ids", "1,2", "--ngram-n", "0"], ["chat", missing, "--draft-len", "-3"],
                 ["run", missing, "--ids", "1,2", "-n", "-1"]):
        assert main(argv) == 2, argv
        err = capsys.readouterr().err
        assert "must be >= " in err and "missing.hearth" not in err, err


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------

def test_cli_parsing():
    from hearth.cli import build_parser, engine_kwargs
    p = build_parser()
    a = p.parse_args(["bench", "m.hearth", "--cache-gb", "2.5", "--threads", "4", "--policy", "lru",
                      "--prefetch", "off", "--buffered", "--mirror", "D:/m.hearth", "--mirror", "E:/m.hearth",
                      "--isa", "avx2", "--gen-tokens", "16", "--ids", "fixed", "-vv"])
    kw = engine_kwargs(a)
    assert kw["cache_gb"] == 2.5 and kw["threads"] == 4 and kw["policy"] == "lru" and kw["prefetch"] == "off"
    assert kw["direct_io"] is False and kw["mirrors"] == ["D:/m.hearth", "E:/m.hearth"] and kw["isa"] == "avx2"
    assert kw["verbose"] == 2 and a.gen_tokens == 16 and a.ids == "fixed"
    import inspect
    sig = inspect.signature(heng.Engine.__init__)
    assert set(kw) <= set(sig.parameters), "CLI engine flags must map onto Engine keyword arguments"
    a = p.parse_args(["run", "m.hearth", "-p", "hi", "--speculative", "ngram", "--temperature", "0.5", "-n", "9"])
    assert a.speculative == "ngram" and a.temperature == 0.5 and a.max_tokens == 9 and a.fn.__name__ == "cmd_run"
    a = p.parse_args(["serve", "m.hearth", "--port", "9000", "--api-key", "k", "--max-queue", "2"])
    assert a.port == 9000 and a.api_key == "k" and a.max_queue == 2
    a = p.parse_args(["synth", "tiny", "x.hearth", "--arch", "deepseek_v3", "--dtype", "q4"])
    assert a.kind == "tiny" and a.arch == "deepseek_v3" and a.dtype == "q4"
    a = p.parse_args(["synth", "shaped", "y.hearth", "--preset", "kimi-k2", "--physical-experts", "16"])
    assert a.kind == "shaped" and a.physical_experts == 16
    a = p.parse_args(["convert", "src", "dst.hearth", "--expert-dtype", "q8"])
    assert a.expert_dtype == "q8" and a.dense_dtype == "q8"
    for cmd in (["chat", "m"], ["info", "m", "--json"], ["doctor"]):
        p.parse_args(cmd)
    with pytest.raises(SystemExit):
        p.parse_args(["bench", "m", "--policy", "fifo"])


def test_cli_doctor_and_version(capsys):
    from hearth.cli import main
    assert main(["doctor", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["deps"]["numpy"]["present"] and "native_lib" in rep and "data_dir" in rep
    r = subprocess.run([sys.executable, "-m", "hearth", "--version"], capture_output=True, text=True, cwd=ROOT,
                       env=_env())
    assert r.returncode == 0 and r.stdout.startswith("hearth ")


def test_cli_output_path_policy(tmp_path, monkeypatch):
    from hearth import cli
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "data"))
    p = cli.output_path("tiny.hearth", force=False)
    assert p == tmp_path / "data" / "models" / "tiny.hearth"
    assert cli.is_cloud_synced("C:/Users/x/OneDrive/Desktop/m.hearth")
    assert not cli.is_cloud_synced(str(tmp_path / "m.hearth")) or "onedrive" in str(tmp_path).lower()
    with pytest.raises(cli.CLIError):
        cli.output_path(str(Path("C:/Users/x/OneDrive - Org/m.hearth")), force=False)


def test_cli_info_on_metadata(tmp_path, capsys, monkeypatch):
    from hearth.cli import main
    monkeypatch.setitem(sys.modules, "hearth.format", None)   # exercise the built-in metadata parser
    p = tmp_path / "m.hearth"
    _write_container(p, [("arch", 4, struct.pack("<I", 4) + b"test"), ("n_layers", 1, struct.pack("<I", 3))])
    assert main(["info", str(p), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metadata"]["arch"] == "test" and out["chat_template"] is False


class CliEngine(FakeEngine):
    """FakeEngine behind the Engine(path, **options) constructor the CLI calls.
    Byte-level tokenizer ids 64..89 are 'a'..'z'; the bonus makes replies lowercase text."""
    instances: list = []
    fail_at: int | None = None          # eval call (1-based) that raises HearthError
    interrupt_at: int | None = None     # eval call that raises KeyboardInterrupt
    eos_kw: dict = {}
    bytes_per_token = 1_000_000

    def __init__(self, path, **kw):
        super().__init__(vocab=256, seed=2, lo=64, hi=90, bonus=8.0, max_seq=kw.get("max_seq") or 4096,
                         **self.eos_kw)
        self.path, self.kw, self.closed = path, kw, False
        self.fed, self.outs, self.traces, self._base, self.attempts = [], [], [], 0, 0
        self.info.update(params_total=2.5e9, params_active=5e8, dense_bytes=3 << 20, expert_bytes=5 << 30,
                         slab_bytes_max=4096, cache_slots=12, isa=3, isa_name="avx512", n_threads=4, n_io_threads=8)
        type(self).instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        self.closed = True

    def eval(self, tokens, all_logits=False):
        self.attempts += 1
        n = self.attempts
        if n == self.fail_at:
            raise heng.HearthError("simulated engine failure")
        if n == self.interrupt_at:
            raise KeyboardInterrupt
        self.fed.append([int(t) for t in np.asarray(tokens).reshape(-1)])
        out = super().eval(tokens, all_logits)
        self.outs.append(out)
        return out

    def reset_stats(self):
        self._base = self.evaluated

    def stats(self):
        d = {name: 0 for name, _ in heng.HearthStats._fields_}
        n = self.evaluated - self._base
        d.update(tokens=n, cache_hits=3 * n, cache_misses=n, bytes_read=self.bytes_per_token * n, wall_s=0.001 * n,
                 prefetch_issued=2 * n, prefetch_used=n, evictions=7)
        d.update(heng.derived_stats(d))
        return d

    def trace_start(self, path):
        self.traces.append(("start", path))

    def trace_stop(self):
        self.traces.append(("stop",))

    def route_replay(self, path):
        self.traces.append(("replay", path))


@pytest.fixture
def cli_engine(monkeypatch):
    import hearth.engine
    monkeypatch.setitem(sys.modules, "hearth.format", None)     # metadata via the built-in parser
    monkeypatch.setattr(CliEngine, "instances", [])
    monkeypatch.setattr(hearth.engine, "Engine", CliEngine)
    return CliEngine


def _cli_container(tmp_path: Path, tokenizer: bool = True, extra=(), name="m.hearth") -> Path:
    meta = list(extra)
    if tokenizer:
        _byte_level_tokenizer().save(str(tmp_path / "tok.json"))
        meta.append(("tokenizer", 4, struct.pack("<I", 8) + b"tok.json"))
    p = tmp_path / name
    _write_container(p, meta)
    return p


def _tpl_meta(tpl: str):
    b = tpl.encode()
    return ("chat_template", 4, struct.pack("<I", len(b)) + b)


class ScriptedStdin:
    """input() reads these lines; KeyboardInterrupt entries are raised (Ctrl-C at the prompt)."""

    def __init__(self, lines):
        self.lines = list(lines)

    def readline(self):
        if not self.lines:
            return ""
        x = self.lines.pop(0)
        if x is KeyboardInterrupt:
            raise KeyboardInterrupt
        return x + "\n"


def run_chat(monkeypatch, argv, lines):
    from hearth import chat as hchat
    from hearth.cli import main
    convs = []

    class SpyConversation(hchat.Conversation):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            convs.append(self)

    monkeypatch.setattr(hchat, "Conversation", SpyConversation)
    monkeypatch.setattr(sys, "stdin", ScriptedStdin(lines))
    rc = main(argv)
    return rc, convs[-1]


def test_cli_chat_survives_a_full_context(tmp_path, monkeypatch, capsys, cli_engine):
    """Regression: a turn that no longer fit the KV cache ended the REPL with
    exit status 1, and the failed turn stayed in the history."""
    m = _cli_container(tmp_path)
    # ChatML, 1 token per byte: "hello" turn 55 prompt tokens + 8 reply; the 'x' turn needs 174 > 140;
    # "third turn" needs 134, leaving room for 7 of the 8 requested tokens.
    rc, conv = run_chat(monkeypatch, ["chat", str(m), "--max-seq", "140", "-n", "8", "--temperature", "0"],
                        ["hello", "x" * 50, "third turn", "/exit", "never read"])
    out, err = capsys.readouterr()
    assert rc == 0
    assert "[context full: this turn needs 174 prompt tokens, the KV cache holds 140;" in err
    assert err.count("[context full") == 1
    assert "[reply cut off: the KV cache (140 tokens) is full; /reset clears the conversation]" in err
    assert [(x["role"], len(x["content"])) for x in conv.messages] == [("user", 5), ("assistant", 8),
                                                                      ("user", 10), ("assistant", 7)]
    assert conv.messages[2]["content"] == "third turn"
    eng = cli_engine.instances[-1]
    assert eng.pos == 140 and eng.closed and sys.stdin.lines == ["never read"]
    assert re.search(r"^\[8 tok, decode [\d.]+ tok/s, prefill 55 tok [\d.]+ tok/s, hit 75%, read [\d.]+ GB, length\]$",
                     err, re.M), err
    # the third turn reuses the KV of turn one (55 + 7 tokens) and prefills the rest of its 134
    assert re.search(r"^\[7 tok, decode [\d.]+ tok/s, prefill 72 tok [\d.]+ tok/s \(62 cached\), hit 75%, "
                     r"read [\d.]+ GB, length\]$", err, re.M), err
    assert out.startswith(">>> ") and conv.messages[1]["content"] in out


def test_cli_chat_commands_and_failed_turns(tmp_path, monkeypatch, capsys, cli_engine):
    m = _cli_container(tmp_path)
    monkeypatch.setattr(CliEngine, "fail_at", 2)        # the first decode step of the first reply fails
    rc, conv = run_chat(monkeypatch, ["chat", str(m), "-n", "6", "--temperature", "0", "-q"],
                        ["hello", "/help", "/bogus", "", "again", "/stats", KeyboardInterrupt, "/system Be brief",
                         "hi", "/reset", "/system", "bye"])
    out, err = capsys.readouterr()
    assert rc == 0                                      # EOF ends the session normally
    assert "[turn failed and was discarded: simulated engine failure]" in err
    assert "/reset  clear history" in err and "unknown command /bogus; /help lists them" in err
    assert "[system prompt set; conversation cleared]" in err and "[conversation cleared]" in err
    assert "tok/s" not in err                           # -q: no stats lines
    stats = json.loads(out[out.index("{"):out.index("}") + 1])
    assert stats["tokens"] > 0
    assert conv.system is None and [x["role"] for x in conv.messages] == ["user", "assistant"]
    assert conv.messages[0]["content"] == "bye"
    eng = cli_engine.instances[-1]
    text = Path(tmp_path / "tok.json")
    from hearth.chat import Tokenizer
    tok = Tokenizer.from_file(text)
    prompts = [tok.decode(f) for f in eng.fed if len(f) > 1]
    assert "again" in prompts[1] and "hello" not in prompts[1]      # the failed turn left no trace
    assert any("Be brief" in p and "hi" in p for p in prompts)


def test_cli_chat_interrupt_and_template_errors(tmp_path, monkeypatch, capsys, cli_engine):
    m = _cli_container(tmp_path)
    monkeypatch.setattr(CliEngine, "interrupt_at", 4)   # Ctrl-C during the first reply, after 3 tokens
    rc, conv = run_chat(monkeypatch, ["chat", str(m), "-n", "6", "--temperature", "0", "--system", "S"],
                        ["hello", "more"])
    err = capsys.readouterr().err
    assert rc == 0 and "[interrupted]" in err and conv.system == "S"
    assert [(x["role"], len(x["content"])) for x in conv.messages] == [("user", 5), ("assistant", 3),
                                                                      ("user", 4), ("assistant", 6)]
    bad = _cli_container(tmp_path, extra=[_tpl_meta("{{ raise_exception('roles must alternate') }}")], name="b.hearth")
    rc, conv = run_chat(monkeypatch, ["chat", str(bad)], ["hello", "/exit"])
    err = capsys.readouterr().err
    assert rc == 0 and "[cannot build the prompt: roles must alternate]" in err and conv.messages == []
    assert "no chat template" not in err
    rc, _ = run_chat(monkeypatch, ["chat", str(m)], [])
    assert rc == 0 and "note: the container has no chat template; using ChatML" in capsys.readouterr().err


def test_cli_run_prompts_ids_and_outputs(tmp_path, monkeypatch, capsys, cli_engine):
    from hearth.chat import ChatTemplate, Tokenizer
    from hearth.cli import main
    m = _cli_container(tmp_path)
    tok = Tokenizer.from_file(tmp_path / "tok.json")

    def ref(ids, n, **kw):
        return generate_ids(CliEngine("ref", **kw), ids, n)[0]

    js = tmp_path / "out.json"
    assert main(["run", str(m), "--ids", "5, 6 7", "-n", "6", "--json-out", str(js)]) == 0
    out, err = capsys.readouterr()
    want = ref([5, 6, 7], 6)
    assert out == tok.decode(want) + "\n"                             # the container has a tokenizer: text
    assert re.fullmatch(r"\[6 tok, decode [\d.]+ tok/s, prefill 3 tok [\d.]+ tok/s, hit 75%, read [\d.]+ GB, length\]\n",
                        err), err
    rec = json.loads(js.read_text())
    assert rec["prompt_ids"] == [5, 6, 7] and rec["output_ids"] == want and rec["stats"]["new_tokens"] == 6
    assert main(["run", str(m), "-p", "hello", "-n", "5", "-q"]) == 0
    out, err = capsys.readouterr()
    assert out == tok.decode(ref(tok.encode("hello", add_special_tokens=True), 5)) + "\n" and err == ""
    (tmp_path / "p.txt").write_text("from a file", encoding="utf-8")
    assert main(["run", str(m), "--prompt-file", str(tmp_path / "p.txt"), "-n", "4", "-q", "--print-ids"]) == 0
    assert capsys.readouterr().out.split() == [str(t) for t in ref(tok.encode("from a file"), 4)]
    assert main(["run", str(m), "-p", "hi", "--chat", "--system", "S", "-n", "5", "-q", "--speculative", "ngram"]) == 0
    assert cli_engine.instances[-1].closed
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    ids = tok.encode(ChatTemplate().render(msgs), add_special_tokens=False)
    assert capsys.readouterr().out == tok.decode(ref(ids, 5)) + "\n"
    samp = dict(temperature=3.0, top_k=20, top_p=0.9, min_p=0.05, repetition_penalty=1.3, seed=7)
    flags = [x for k, v in samp.items() for x in (f"--{k.replace('_', '-')}", str(v))]
    assert main(["run", str(m), "--ids", "1,2,3", "-n", "8", "-q", "--print-ids", *flags]) == 0
    got = [int(t) for t in capsys.readouterr().out.split()]
    assert got == generate_ids(CliEngine("ref"), [1, 2, 3], 8, Sampler(**samp))[0] != ref([1, 2, 3], 8)
    monkeypatch.setattr(CliEngine, "eos_kw", dict(eos=10, eos_at=6))
    full = ref([1, 2, 3], 8)
    assert full[3] == 10                                                # the model emits end-of-sequence 4th
    assert main(["run", str(m), "--ids", "1,2,3", "-n", "8", "-q", "--print-ids"]) == 0
    assert capsys.readouterr().out.split() == [str(t) for t in full[:3]]
    assert main(["run", str(m), "--ids", "1,2,3", "-n", "8", "-q", "--ignore-eos", "--print-ids"]) == 0
    assert capsys.readouterr().out.split() == [str(t) for t in full]
    monkeypatch.setattr(CliEngine, "eos_kw", {})
    monkeypatch.setattr(CliEngine, "interrupt_at", 3)
    assert main(["run", str(m), "--ids", "1,2", "-n", "8", "--print-ids"]) == 0
    out, err = capsys.readouterr()
    assert len(out.split()) == 2 and "[interrupted]" in err and "[2 tok" in err


def test_cli_run_tokenizer_flag_and_raw_ids(tmp_path, capsys, cli_engine):
    from hearth.chat import Tokenizer
    from hearth.cli import main
    plain = _cli_container(tmp_path, tokenizer=False, extra=[
        ("bos_id", 1, struct.pack("<I", 7)), _tpl_meta("{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}")])
    _byte_level_tokenizer().save(str(tmp_path / "other.json"))
    assert main(["run", str(plain), "-p", "abc", "-n", "3", "-q"]) == 2
    assert "no tokenizer for" in capsys.readouterr().err
    assert main(["run", str(plain), "-p", "abc", "-n", "3", "-q", "--tokenizer", str(tmp_path / "other.json")]) == 0
    tok = Tokenizer.from_file(tmp_path / "other.json")
    assert cli_engine.instances[-1].fed[0] == tok.encode("abc")
    want = generate_ids(CliEngine("ref"), tok.encode("abc"), 3)[0]
    assert capsys.readouterr().out == tok.decode(want) + "\n"
    assert main(["run", str(plain), "-p", "abc", "--chat", "-n", "2", "-q", "--tokenizer", str(tmp_path / "other.json")]) == 0
    assert cli_engine.instances[-1].fed[0] == [7] + tok.encode("abc")    # the container's BOS id, via the template
    capsys.readouterr()
    assert main(["run", str(plain), "--ids", "4,5", "-n", "2", "-q"]) == 0       # no tokenizer needed for ids
    junk = tmp_path / "junk.hearth"
    junk.write_bytes(b"not a container")
    assert main(["run", str(junk), "--ids", "4,5", "-n", "2", "-q"]) == 0        # the engine validates it
    capsys.readouterr()
    assert main(["run", str(junk), "-p", "x", "-n", "2"]) == 1
    assert "hearth run: ValueError:" in capsys.readouterr().err
    assert main(["run", str(plain), "-n", "2"]) == 2
    assert "give --prompt, --prompt-file or --ids" in capsys.readouterr().err
    assert main(["run", str(plain), "--ids", "1,x"]) == 2


def test_cli_bench_feeds_the_requested_ids_and_reports_consistent_rates(tmp_path, capsys, cli_engine):
    from hearth.cli import bench_ids, main
    assert bench_ids("fixed", 3, 100, 0) == [13, 32, 51] and bench_ids("fixed", 2, 100, 0, offset=1) == [32, 51]
    r = bench_ids("random", 50, 256, 1)
    assert r == bench_ids("random", 50, 256, 1) and all(0 <= t < 256 for t in r)
    assert r != bench_ids("random", 50, 256, 2) and r != bench_ids("random", 50, 256, 1, offset=3)
    m = _cli_container(tmp_path, tokenizer=False)
    for kind in ("random", "fixed", "greedy"):
        t0 = time.perf_counter()
        assert main(["bench", str(m), "--prompt-tokens", "12", "--gen-tokens", "5", "--warmup", "3", "--seed", "4",
                     "--ids", kind, "--json", "--max-seq", "64", "--threads", "2"]) == 0
        elapsed = time.perf_counter() - t0
        res = json.loads(capsys.readouterr().out)
        assert 0 < res["prefill_s"] and 0 < res["decode_s"] and res["prefill_s"] + res["decode_s"] < elapsed
        eng = cli_engine.instances[-1]
        feed = "fixed" if kind == "greedy" else kind
        assert eng.fed[0] == bench_ids(feed, 3, 256, 4, offset=10_000_000)
        assert eng.fed[1] == bench_ids(feed, 12, 256, 4)
        assert [len(f) for f in eng.fed[2:]] == [1] * 5
        dec = [f[0] for f in eng.fed[2:]]
        if kind == "greedy":       # each decode token is the argmax of the previous forward's logits
            assert dec == [int(np.argmax(o)) for o in eng.outs[1:-1]]
        else:
            assert dec == bench_ids(kind, 5, 256, 4, offset=12)
        assert res["prefill_tokens"] == 12 and res["decode_tokens"] == 5 and res["ids"] == kind
        assert res["prefill_tok_s"] == pytest.approx(12 / res["prefill_s"])
        assert res["decode_tok_s"] == pytest.approx(5 / res["decode_s"])
        assert res["prefill_stats"]["tokens"] == 12 and res["decode_stats"]["tokens"] == 5
        assert res["options"]["max_seq"] == 64 and res["threads"] == 4 and res["arch"] == "fake" and eng.closed
        assert eng.kw["threads"] == 2 and eng.traces == []
    assert main(["bench", str(m), "--prompt-tokens", "4", "--gen-tokens", "2", "--trace", "t.hrtr",
                 "--route-replay", "r.hrtr"]) == 0
    out = capsys.readouterr().out
    assert cli_engine.instances[-1].traces == [("replay", "r.hrtr"), ("start", "t.hrtr"), ("stop",)]
    assert "(routing replayed from trace: synthetic workload)" in out and "arch=fake  isa=avx512" in out
    assert re.search(r"^  prefill       4 tok .* tok/s  hit 75\.0%  read 0\.00 GB  stall 0\.0%$", out, re.M), out
    assert re.search(r"^  decode        2 tok .* prefetch used 2/4$", out, re.M), out
    assert "  decode   1.0 MB read per token, 7 evictions" in out
    CliEngine.bytes_per_token = 0
    try:
        assert main(["bench", str(m), "--prompt-tokens", "4", "--gen-tokens", "2"]) == 0
    finally:
        CliEngine.bytes_per_token = 1_000_000
    out = capsys.readouterr().out
    assert "decode        2 tok" in out and "MB read per token" not in out     # nothing read: no I/O line
    assert main(["bench", str(m), "--prompt-tokens", "0", "--gen-tokens", "3", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["prefill_tok_s"] == 0.0 and res["decode_tokens"] == 3
    assert main(["bench", str(m), "--prompt-tokens", "0", "--gen-tokens", "0"]) == 0
    assert "prefill" not in capsys.readouterr().out.split("\n", 1)[1]
    assert main(["bench", str(m), "--prompt-tokens", "60", "--gen-tokens", "5", "--max-seq", "64"]) == 2
    assert "warmup + prompt + gen = 65 tokens exceeds the KV capacity 64" in capsys.readouterr().err


def test_cli_output_survives_a_console_that_cannot_encode_it(tmp_path, monkeypatch):
    import io
    from hearth.cli import main
    monkeypatch.setitem(sys.modules, "hearth.format", None)
    p = tmp_path / "m.hearth"
    arch = "tést".encode()
    _write_container(p, [("arch", 4, struct.pack("<I", len(arch)) + arch)])
    out = io.TextIOWrapper(io.BytesIO(), encoding="ascii")       # e.g. a legacy code page console
    monkeypatch.setattr(sys, "stdout", out)
    assert main(["info", str(p)]) == 0
    out.flush()
    assert b"  arch             t?st" in out.buffer.getvalue()


def test_cli_info_text_and_engine(tmp_path, capsys, cli_engine):
    from hearth.cli import main
    m = _cli_container(tmp_path, extra=[("arch", 4, struct.pack("<I", 4) + b"test"), ("n_layers", 1, struct.pack("<I", 3)),
                                        _tpl_meta("{{ messages }}")])
    assert main(["info", str(m), "--open", "--cache-gb", "2"]) == 0
    out = capsys.readouterr().out.splitlines()
    size = m.stat().st_size
    assert out == [f"{m}  ({size} B)", "  arch             test", "  n_layers         3",
                   f"  tokenizer        {tmp_path / 'tok.json'}", "  chat_template    yes", "engine:",
                   "  params total 2.50 B, active 0.50 B", "  resident 3.0 MiB, experts 5.0 GiB, largest slab 4.0 KiB",
                   "  cache slots 12, isa avx512, threads 4, io threads 8"]
    assert cli_engine.instances[-1].kw["cache_gb"] == 2.0 and cli_engine.instances[-1].closed
    plain = _cli_container(tmp_path, tokenizer=False, name="p.hearth")
    assert main(["info", str(plain)]) == 0
    assert capsys.readouterr().out.splitlines()[1:] == ["  tokenizer        -", "  chat_template    no (ChatML fallback)"]
    assert main(["info", str(m), "--open", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["engine"]["cache_slots"] == 12 and rep["file_bytes"] == size and rep["chat_template"] is True


def test_cli_serve_options(monkeypatch, capsys):
    import hearth.server as hs
    from hearth.cli import main
    seen = []
    monkeypatch.setattr(hs, "serve", lambda model, **kw: seen.append(dict(kw, model=model)))
    monkeypatch.delenv("HEARTH_API_KEY", raising=False)
    assert main(["serve", "m.hearth", "--host", "0.0.0.0", "--port", "9001", "--timeout", "2.5", "--max-queue", "3",
                 "--cors", "*", "--log-requests", "--model-name", "x", "--speculative", "ngram", "--draft-len", "6",
                 "--ngram-n", "2", "--tokenizer", "t.json", "--cache-gb", "1.5"]) == 0
    kw = seen[-1]
    assert (kw["model"], kw["host"], kw["port"], kw["timeout"], kw["max_queue"], kw["cors"], kw["verbose"],
            kw["model_name"], kw["speculative"], kw["draft_len"], kw["ngram_n"], kw["tokenizer_path"], kw["api_key"]) == \
           ("m.hearth", "0.0.0.0", 9001, 2.5, 3, "*", True, "x", "ngram", 6, 2, "t.json", None)
    assert kw["engine_kwargs"]["cache_gb"] == 1.5
    assert "warning: listening on 0.0.0.0 without an API key" in capsys.readouterr().err
    monkeypatch.setenv("HEARTH_API_KEY", "envkey")
    assert main(["serve", "m.hearth", "--host", "0.0.0.0"]) == 0
    kw = seen[-1]
    assert kw["api_key"] == "envkey" and (kw["timeout"], kw["max_queue"], kw["port"], kw["speculative"]) == \
           (60.0, 8, 8080, "none") and kw["verbose"] is False
    assert main(["serve", "m.hearth", "--api-key", "k", "--host", "0.0.0.0"]) == 0 and seen[-1]["api_key"] == "k"
    monkeypatch.delenv("HEARTH_API_KEY")
    for host in ("127.0.0.1", "localhost", "::1"):
        assert main(["serve", "m.hearth", "--host", host]) == 0
    assert capsys.readouterr().err == ""
    assert main(["serve", "m.hearth", "--timeout", "0"]) == 2
    assert "--timeout must be > 0" in capsys.readouterr().err


def test_cli_serve_reports_a_port_in_use(tmp_path, capsys, cli_engine):
    import socket
    from hearth.cli import main
    m = _cli_container(tmp_path)
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    try:
        rc = main(["serve", str(m), "--port", str(blocker.getsockname()[1])])
    finally:
        blocker.close()
    err = capsys.readouterr().err
    assert rc == 1 and "hearth serve: OSError" in err and "Traceback" not in err
    assert cli_engine.instances[-1].closed


def test_cli_convert_synth_and_sim_dispatch(tmp_path, monkeypatch, capsys):
    import inspect
    import hearth.convert
    import hearth.sim
    import hearth.synth
    from hearth.cli import main
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "data"))
    real_make_tiny = hearth.synth.make_tiny
    calls = []

    def rec(name):
        def fn(*a, **kw):
            calls.append((name, a, kw))
            return a[1] if name == "convert" else a[0]
        return fn

    monkeypatch.setattr(hearth.convert, "convert", rec("convert"))
    monkeypatch.setattr(hearth.synth, "make_tiny", rec("tiny"))
    monkeypatch.setattr(hearth.synth, "make_shaped", rec("shaped"))
    models = tmp_path / "data" / "models"
    assert main(["convert", "hf_dir", "out.hearth", "--expert-dtype", "q8", "--dense-dtype", "bf16", "--embed-dtype",
                 "f16", "--head-dtype", "f32", "--max-seq", "77", "--threads", "3", "--quiet"]) == 0
    assert calls[-1] == ("convert", ("hf_dir", models / "out.hearth"),
                         dict(expert_dtype=3, dense_dtype=2, embed_dtype=1, head_dtype=0, max_seq=77, threads=3,
                              progress=False))
    assert capsys.readouterr().out.strip() == str(models / "out.hearth")
    assert main(["convert", "hf_dir", "o2.hearth"]) == 0
    assert calls[-1][2] == dict(expert_dtype=4, dense_dtype=3, embed_dtype=3, head_dtype=3, max_seq=None, threads=0,
                                progress=True)
    assert main(["synth", "tiny", "t.hearth", "--arch", "olmoe", "--dtype", "q4", "--seed", "5", "--n-layers", "2",
                 "--d-model", "64", "--n-experts", "4", "--top-k", "1", "--expert-ffn", "32", "--vocab", "300",
                 "--max-seq", "99"]) == 0
    assert calls[-1] == ("tiny", (models / "t.hearth",), dict(arch="olmoe", dtype=4, seed=5, n_layers=2, d_model=64,
                                                             n_experts=4, top_k=1, expert_ffn=32, vocab=300, max_seq=99))
    assert main(["synth", "tiny", "d.hearth"]) == 0          # CLI defaults are make_tiny's own defaults
    sig = inspect.signature(real_make_tiny).parameters
    assert set(calls[-1][2]) == set(sig) - {"path"}
    assert calls[-1][2] == {k: sig[k].default for k in calls[-1][2]}
    assert main(["synth", "shaped", "s.hearth", "--preset", "kimi-k2", "--expert-dtype", "q8", "--physical-experts",
                 "16", "--seed", "2"]) == 0
    assert calls[-1] == ("shaped", (models / "s.hearth",), dict(preset="kimi-k2", expert_dtype=3, physical_experts=16,
                                                               seed=2))
    assert main(["synth", "shaped", "s.hearth", "--preset", "p", "--dense-dtype", "f16", "--max-seq", "512"]) == 0
    assert calls[-1][2] == dict(preset="p", expert_dtype=4, physical_experts=None, seed=0, dense_dtype=1, max_seq=512)
    capsys.readouterr()
    cloud = tmp_path / "OneDrive" / "x.hearth"
    n = len(calls)
    assert main(["synth", "tiny", str(cloud)]) == 2 and len(calls) == n          # refused before writing
    assert "cloud-synced folder" in capsys.readouterr().err
    assert main(["synth", "tiny", str(cloud), "--force"]) == 0 and calls[-1][1] == (cloud,) and cloud.parent.is_dir()
    deep = tmp_path / "new" / "dir" / "y.hearth"
    assert main(["convert", "hf", str(deep)]) == 0 and calls[-1][1] == ("hf", deep) and deep.parent.is_dir()
    monkeypatch.chdir(tmp_path)
    assert main(["synth", "tiny", str(Path("rel") / "z.hearth")]) == 0          # a relative path with a directory
    assert calls[-1][1] == (Path("rel") / "z.hearth",) and (tmp_path / "rel").is_dir()
    sim_rc = [3]
    monkeypatch.setattr(hearth.sim, "main", lambda argv: calls.append(("sim", argv)) or sim_rc[0])
    assert main(["sim", "--trace", "x", "--help"]) == 3 and calls[-1] == ("sim", ["--trace", "x", "--help"])
    sim_rc[0] = None
    assert main(["sim"]) == 0 and calls[-1] == ("sim", [])


def test_cli_doctor_text_report(tmp_path, monkeypatch, capsys):
    import importlib.util
    import os
    from hearth import _native, cli
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "OneDrive" / "hearth"))
    real_find = importlib.util.find_spec
    missing = {"torch"}
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name in missing else real_find(name, *a))

    class FakeLib:
        _name = "fake/hearth.dll"

    monkeypatch.setattr(_native, "lib", lambda: FakeLib())
    api = fake_api(monkeypatch)
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^\[ok  \] python             \d+\.\d+", out, re.M)
    assert re.search(r"^\[ok  \] numpy              \d", out, re.M)
    assert re.search(r"^\[warn\] torch              missing \(optional\)$", out, re.M)
    assert "[ok  ] native library     fake/hearth.dll  version 9.9.9-fake\n" in out
    assert "[ok  ] cpu isa (engine)   avx2\n" in out
    assert re.search(r"^\[ok  \] ram                [\d.]+ \w+ total, [\d.]+ \w+ available$", out, re.M)
    feats = cli._cpu_features_py()
    if os.name == "nt":
        assert set(feats) == {"avx2", "avx512f"}
    if feats:
        assert f"[ok  ] cpu features       {', '.join(k for k, v in feats.items() if v) or 'no AVX2/AVX-512'}\n" in out
    assert re.search(r"^\[warn\] data dir .*OneDrive.hearth \(not created yet\), [\d.]+ \w+ free  -- cloud-synced: "
                     r"models must not live here \(INV-DATA\)$", out, re.M), out
    missing.add("numpy")
    monkeypatch.setattr(api, "version", lambda: (_ for _ in ()).throw(OSError("bad build")))
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] numpy              missing\n" in out and "[warn] native library     bad build\n" in out
    assert cli.main(["doctor", "--json"]) == 1
    rep = json.loads(capsys.readouterr().out)
    assert rep["deps"]["numpy"] == {"present": False, "version": None, "required": True}
    assert rep["deps"]["torch"]["required"] is False and rep["native_lib"]["error"] == "bad build"
    assert rep["data_dir"]["cloud_synced"] is True and rep["data_dir"]["exists"] is False and rep["data_dir"]["free"] > 0
    assert rep["ram"]["total"] >= rep["ram"].get("available", 0) > 0
    missing.clear()

    def nolib():
        raise _native.HearthLibNotFound("no library here")

    monkeypatch.setattr(_native, "lib", nolib)
    import shutil
    probed, real_usage = [], shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage", lambda p: probed.append(Path(p)) or real_usage(p))
    assert cli.main(["doctor", "--json"]) == 0
    assert probed[-1] == tmp_path                       # the nearest existing ancestor of the data dir
    capsys.readouterr()
    (tmp_path / "OneDrive" / "hearth").mkdir(parents=True)
    assert cli.main(["doctor"]) == 0
    assert probed[-1] == tmp_path / "OneDrive" / "hearth"
    out = capsys.readouterr().out
    assert "[warn] native library     not found (no library here); build it with" in out
    assert "(not created yet)" not in out
    assert cli.main(["doctor", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["native_lib"] == {"found": False, "error": "no library here"}


def test_cli_posix_probes_parse_proc_files(tmp_path, monkeypatch):
    """The Linux branches of doctor's CPU / RAM probes, run against fixture files
    (os.name and the /proc paths are redirected; no Linux machine is needed)."""
    import types
    from hearth import cli
    proc = tmp_path / "proc"
    proc.mkdir()
    real_path = cli.Path
    monkeypatch.setattr(cli, "Path", lambda p: real_path(str(proc) + p[5:]) if str(p).startswith("/proc") else real_path(p))
    pages = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 1000}

    def sysconf(name):
        if name not in pages:
            raise ValueError(f"unrecognized configuration name {name!r}")    # as os.sysconf does
        return pages[name]

    monkeypatch.setattr(cli, "os", types.SimpleNamespace(name="posix", sysconf=sysconf))
    assert cli._cpu_features_py() == {} and cli._ram() == {"total": 4096 * 1000}     # no /proc: sysconf only
    (proc / "cpuinfo").write_text("processor\t: 0\nmodel name\t: x\nflags\t\t: fpu sse2 avx2 avx512f avx512bw\n\n"
                                  "processor\t: 1\nflags\t\t: fpu\n")
    (proc / "meminfo").write_text("MemTotal:       16384 kB\nMemFree:          100 kB\nMemAvailable:    8192 kB\n")
    assert cli._cpu_features_py() == {"avx2": True, "avx512f": True}                # the first CPU's flags
    assert cli._ram() == {"total": 16384 * 1024, "available": 8192 * 1024}
    (proc / "cpuinfo").write_text("processor\t: 0\nflags\t\t: fpu sse2\n")
    assert cli._cpu_features_py() == {"avx2": False, "avx512f": False}
    pages.clear()                                                                   # sysconf unsupported
    (proc / "meminfo").unlink()
    assert cli._ram() == {}


def test_cli_data_dir_without_the_native_module(tmp_path, monkeypatch):
    import os
    import hearth
    from hearth import cli
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path / "d"))
    assert cli.data_dir() == tmp_path / "d"
    monkeypatch.delattr(hearth, "_native", raising=False)
    monkeypatch.setitem(sys.modules, "hearth._native", None)
    assert cli.data_dir() == tmp_path / "d"
    monkeypatch.delenv("HEARTH_DATA")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cli.data_dir() == (tmp_path / "local" / "hearth" if os.name == "nt" else tmp_path / "xdg" / "hearth")


def test_cli_helpers():
    from hearth import cli
    assert [cli._fmt_bytes(n) for n in (0, 1023, 1024, 1536, 5 * 1024 ** 3, 3 * 1024 ** 5)] == \
           ["0 B", "1023 B", "1.0 KiB", "1.5 KiB", "5.0 GiB", "3072.0 TiB"]
    st = GenStats(prompt_tokens=10, reused_tokens=4, new_tokens=5, decode_tokens=4, forwards=2, draft_steps=1,
                  drafted=4, accepted=2, prefill_s=2.0, decode_s=2.0, finish_reason="length")
    before = {"cache_hits": 10, "cache_misses": 5, "bytes_read": 1_000_000_000}
    after = {"cache_hits": 13, "cache_misses": 6, "bytes_read": 2_500_000_000}
    assert cli._stats_line(st, before, after) == \
           "[5 tok, decode 2.00 tok/s, prefill 6 tok 3.0 tok/s (4 cached), accept 50% (2.00 tok/fwd), hit 75%, " \
           "read 1.50 GB, length]"
    st2 = GenStats(prompt_tokens=3, new_tokens=1, prefill_s=1.0)
    same = {"cache_hits": 1, "cache_misses": 1, "bytes_read": 5}
    for b, a in ((None, None), (same, same), (same, None)):
        assert cli._stats_line(st2, b, a) == "[1 tok, decode 0.00 tok/s, prefill 3 tok 3.0 tok/s]"
    even = {"cache_hits": 3, "cache_misses": 3, "bytes_read": 5}
    assert cli._stats_line(st2, same, even) == "[1 tok, decode 0.00 tok/s, prefill 3 tok 3.0 tok/s, hit 50%]"
    assert cli._parse_ids("1, 2 3,4") == [1, 2, 3, 4]
    for bad in ("", " , ", "1,x"):
        with pytest.raises(cli.CLIError):
            cli._parse_ids(bad)


def test_cli_main_maps_errors_to_exit_codes(monkeypatch, capsys):
    from hearth import _native, cli

    def raising(exc):
        def fn(args):
            raise exc
        return fn

    for exc, rc, msg in ((cli.CLIError("bad flag"), 2, "hearth doctor: bad flag\n"),
                         (heng.HearthError("engine broke"), 1, "hearth doctor: HearthError: engine broke\n"),
                         (OSError("disk gone"), 1, "hearth doctor: OSError: disk gone\n"),
                         (ValueError("bad value"), 1, "hearth doctor: ValueError: bad value\n"),
                         (ImportError("no tokenizers"), 1, "hearth doctor: ImportError: no tokenizers\n"),
                         (_native.HearthLibNotFound("no lib"), 1, "hearth doctor: HearthLibNotFound: no lib\n"),
                         (KeyboardInterrupt(), 130, "interrupted\n")):
        monkeypatch.setattr(cli, "cmd_doctor", raising(exc))
        assert cli.main(["doctor"]) == rc
        assert capsys.readouterr().err == msg
    monkeypatch.setattr(cli, "cmd_doctor", raising(RuntimeError("a bug")))
    with pytest.raises(RuntimeError, match="a bug"):
        cli.main(["doctor"])
    monkeypatch.setattr(cli, "cmd_doctor", lambda args: None)
    assert cli.main(["doctor"]) == 0
    with pytest.raises(SystemExit):
        cli.main([])


# ---------------------------------------------------------------------------------
# Compiled ABI stub: implements hearth.h in C so the ctypes bindings are exercised
# against the compiler's own struct layout (runs in a subprocess so the stub never
# becomes the cached library of this test session)
# ---------------------------------------------------------------------------------

ABI_STUB_C = r'''#include "hearth.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct hearth_engine {
    hearth_options opt;
    char model_path[512];
    char mirrors[8][512];
    char usage_in[512], usage_out[512];
    int32_t *kv;
    int pos, cap, vocab;
    hearth_stats st;
    int tracing;
};

static hearth_options g_last;          /* copy of the last options seen (strings re-pointed) */
static char g_paths[11][512];

HEARTH_API const char *hearth_version(void) { return "0.1.0-abi-stub"; }

HEARTH_API void hearth_default_options(hearth_options *o) {
    memset(o, 0, sizeof *o);
    o->cache_gb = 8.0; o->direct_io = 1; o->policy = HEARTH_POLICY_LFU; o->prefetch = HEARTH_PREFETCH_SHARED;
    o->n_threads = -77; /* sentinel: python must overwrite every field it owns */
}

static void cp(char *dst, const char *src) {
    if (src) { strncpy(dst, src, 511); dst[511] = 0; } else dst[0] = 0;
}

HEARTH_API hearth_engine *hearth_open(const hearth_options *o, char *err, size_t errlen) {
    if (!o || !o->model_path) { if (err) snprintf(err, errlen, "no model path"); return NULL; }
    if (strstr(o->model_path, "fail")) { if (err) snprintf(err, errlen, "cannot open '%s': simulated failure", o->model_path); return NULL; }
    if (o->isa == HEARTH_ISA_AVX512 && strstr(o->model_path, "noavx512")) { if (err) snprintf(err, errlen, "CPU lacks AVX-512"); return NULL; }
    hearth_engine *e = (hearth_engine *)calloc(1, sizeof *e);
    if (!e) return NULL;
    e->opt = *o;
    cp(e->model_path, o->model_path);
    g_last = *o;
    cp(g_paths[0], o->model_path); g_last.model_path = g_paths[0];
    for (int i = 0; i < 8; i++) { cp(g_paths[1 + i], i < o->n_mirrors ? o->mirror_paths[i] : NULL); g_last.mirror_paths[i] = i < o->n_mirrors ? g_paths[1 + i] : NULL; }
    cp(g_paths[9], o->usage_in); g_last.usage_in = o->usage_in ? g_paths[9] : NULL;
    cp(g_paths[10], o->usage_out); g_last.usage_out = o->usage_out ? g_paths[10] : NULL;
    e->cap = o->max_seq > 0 ? (o->max_seq < 300 ? o->max_seq : 300) : 300;
    e->vocab = 97;
    e->kv = (int32_t *)calloc((size_t)e->cap, sizeof(int32_t));
    return e;
}

HEARTH_API void hearth_close(hearth_engine *e) { if (e) { free(e->kv); free(e); } }

HEARTH_API int hearth_info(hearth_engine *e, hearth_model_info *m) {
    memset(m, 0, sizeof *m);
    snprintf(m->arch, sizeof m->arch, "stub_moe");
    m->n_layers = 3; m->d_model = 128; m->vocab_size = e->vocab; m->max_seq = 300;
    m->n_heads = 4; m->n_kv_heads = 2; m->head_dim = 32; m->attn_kind = 0;
    m->n_experts = 8; m->top_k = 2; m->expert_ffn_dim = 64; m->n_moe_layers = 3;
    m->bos_id = -1; m->n_eos = 2; m->eos_ids[0] = 5; m->eos_ids[1] = 6; m->eos_ids[2] = 99;
    m->dense_bytes = 123456789012ULL; m->expert_bytes = 9876543210123ULL; m->slab_bytes_max = 4096 * 3;
    m->params_total = 1.0e12; m->params_active = 3.2e10;
    m->cache_slots = 77; m->isa = e->opt.isa ? e->opt.isa : 3; m->n_threads = e->opt.n_threads; m->n_io_threads = 8;
    return 0;
}

static float logit(const int32_t *kv, int p, int v) {
    int a = kv[p], b = p > 0 ? kv[p - 1] : 0;
    float x = (float)sin((double)(a * 31 + b * 17 + v * 7)) + 0.3f * (float)cos((double)((p % 61) * 13 + v));
    if (v == (a * 7 + b * 3) % 97) x += 4.0f;
    return x;
}

HEARTH_API int hearth_eval(hearth_engine *e, const int32_t *t, int n, float *logits, int all) {
    if (n <= 0) return -1;
    if (e->pos + n > e->cap) return -2;
    for (int i = 0; i < n; i++) { if (t[i] < 0 || t[i] >= e->vocab) return -3; e->kv[e->pos + i] = t[i]; }
    for (int i = 0; i < n; i++) {
        float *row = all ? logits + (size_t)i * e->vocab : logits;
        if (logits && (all || i == n - 1))
            for (int v = 0; v < e->vocab; v++) row[v] = logit(e->kv, e->pos + i, v);
    }
    e->pos += n;
    e->st.tokens += (uint64_t)n; e->st.forward_calls++; e->st.cache_hits += 3; e->st.cache_misses += 1;
    e->st.bytes_read += 4096; e->st.wall_s += 0.5; e->st.stall_s += 0.125;
    return 0;
}

HEARTH_API int hearth_pos(hearth_engine *e) { return e->pos; }
HEARTH_API int hearth_reset(hearth_engine *e) { e->pos = 0; return 0; }
HEARTH_API int hearth_rewind(hearth_engine *e, int pos) { if (pos < 0 || pos > e->pos) return -1; e->pos = pos; return 0; }
HEARTH_API int hearth_get_stats(hearth_engine *e, hearth_stats *s) {
    *s = e->st; s->cache_slots = 77; s->cache_resident = 12; s->cache_pinned = 4; s->evictions = 1ULL << 40;
    s->prefetch_issued = 10; s->prefetch_used = 7; s->prefetch_wasted = 3; s->read_s = 0.25;
    return 0;
}
HEARTH_API void hearth_reset_stats(hearth_engine *e) { memset(&e->st, 0, sizeof e->st); }
HEARTH_API int hearth_trace_start(hearth_engine *e, const char *p) { if (!p || strstr(p, "fail")) return -1; e->tracing = 1; return 0; }
HEARTH_API int hearth_trace_stop(hearth_engine *e) { if (!e->tracing) return -1; e->tracing = 0; return 0; }
HEARTH_API int hearth_route_replay(hearth_engine *e, const char *p) { (void)e; return p && !strstr(p, "fail") ? 0 : -1; }
HEARTH_API size_t hearth_row_bytes(int dtype, int64_t n) { return dtype == 0 ? (size_t)(4 * n) : 0; }
HEARTH_API int hearth_quantize(int d, const float *s, int64_t r, int64_t c, void *dst, int nt) { (void)d; (void)s; (void)r; (void)c; (void)dst; (void)nt; return -1; }
HEARTH_API int hearth_cpu_isa(void) { return 3; }

HEARTH_API void stub_last_options(hearth_options *o) { *o = g_last; }
'''

ABI_DRIVER = r'''
import ctypes, gc
import numpy as np
from hearth import _native
from hearth import engine as heng
from hearth.generate import Sampler, generate_ids

L = _native.lib()
fn = L["stub_layout"]
fn.restype = ctypes.c_int
fn.argtypes = [ctypes.POINTER(ctypes.c_size_t)]
o = (ctypes.c_size_t * 1024)()
got = list(o[:fn(o)])
want = []
for cls in (heng.HearthOptions, heng.HearthModelInfo, heng.HearthStats):
    want.append(ctypes.sizeof(cls))
    for name, t in cls._fields_:
        want += [getattr(cls, name).offset, ctypes.sizeof(t)]
assert got == want, ("C layout (size, then offset/size per field) differs from ctypes", got, want)

P = dict(model="C:/models/ünïcode model.hearth", m1="D:/mirror one.hearth", m2="E:/m2.hearth",
         ui="C:/u in.usage", uo="C:/u out.usage")
e = heng.Engine(P["model"], cache_gb=2.5, threads=3, io_threads=5, direct_io=False, policy="lru",
                prefetch="next", prefetch_extra=2, usage_in=P["ui"], usage_out=P["uo"], pin_fraction=0.25,
                warm_start=True, max_seq=250, max_batch=64, isa="avx2", mirrors=[P["m1"], P["m2"]], verbose=1)
gc.collect()
lo = heng.HearthOptions()
f2 = L["stub_last_options"]
f2.argtypes = [ctypes.POINTER(heng.HearthOptions)]
f2(ctypes.byref(lo))
assert lo.model_path.decode("utf-8") == P["model"]
assert lo.n_mirrors == 2 and lo.mirror_paths[0].decode() == P["m1"] and lo.mirror_paths[1].decode() == P["m2"]
assert lo.mirror_paths[2] is None
assert lo.usage_in.decode() == P["ui"] and lo.usage_out.decode() == P["uo"]
assert (lo.cache_gb, lo.n_threads, lo.n_io_threads, lo.direct_io, lo.policy, lo.prefetch, lo.prefetch_extra) ==        (2.5, 3, 5, 0, 0, 1, 2)
assert (lo.pin_fraction, lo.warm_start, lo.max_seq, lo.max_batch, lo.isa, lo.verbose) == (0.25, 1, 250, 64, 2, 1)

i = e.info
assert i["arch"] == "stub_moe" and i["vocab_size"] == 97 and i["eos_ids"] == [5, 6] and i["n_eos"] == 2
assert i["dense_bytes"] == 123456789012 and i["expert_bytes"] == 9876543210123 and i["params_total"] == 1e12
assert i["cache_slots"] == 77 and i["isa"] == 2 and i["isa_name"] == "avx2" and i["n_threads"] == 3
assert e.kv_capacity == 250

a = e.eval([1, 2, 3])
assert a.dtype == np.float32 and a.shape == (97,) and e.pos == 3
b = e.eval([4, 5], all_logits=True)
assert b.shape == (2, 97)
a0, b0 = a.copy(), b.copy()
e.rewind(3)
again = e.eval(np.array([4, 5], dtype=np.int64), all_logits=True)
assert np.array_equal(again, b0)
nxt = e.eval([6])                                   # same buffer as `again` and `b`
assert np.array_equal(a, a0) and np.array_equal(b, b0) and np.array_equal(again, b0), "results must be copies"
assert not np.array_equal(nxt, b0[0])               # so an overwrite would have shown
keep = heng._KEEP_FLOATS
heng._KEEP_FLOATS = 100                             # force the one-off (non-retained) buffer path
e.rewind(3)
big = e.eval([4, 5, 6], all_logits=True)
assert np.array_equal(big[:2], b0) and e._buf.size == 2 * 97
heng._KEEP_FLOATS = keep
try:
    e.eval([97]); raise SystemExit("expected ValueError")
except ValueError:
    pass
e.reset()
e.eval(list(range(90)) * 2)
try:
    e.eval([1] * 100); raise SystemExit("expected HearthError")
except heng.HearthError as ex:
    assert "code -2" in str(ex)
st = e.stats()
assert st["evictions"] == 1 << 40 and st["cache_pinned"] == 4 and st["read_s"] == 0.25
assert abs(st["hit_rate"] - 0.75) < 1e-12 and abs(st["stall_frac"] - 0.25) < 1e-12
e.reset_stats()
assert e.stats()["tokens"] == 0
e.trace_start("C:/t.hrtr"); e.trace_stop(); e.route_replay("C:/ok.hrtr")
for bad in (lambda: e.trace_stop(), lambda: e.trace_start("fail.hrtr"), lambda: e.route_replay("fail")):
    try:
        bad(); raise SystemExit("expected HearthError")
    except heng.HearthError:
        pass

for prompt in ([10, 11, 12, 13, 10, 11, 12, 13, 10, 11], [1, 2]):
    plain, s1 = generate_ids(e, prompt, 300)
    spec, s2 = generate_ids(e, prompt, 300, None, (), "ngram", 6, 3)
    assert plain == spec and len(plain) == 250 - len(prompt) + 1 and s2.accepted > 0
    assert s2.forwards < s1.forwards

with e:
    pass
e.close(); e.close()
try:
    e.eval([1]); raise SystemExit("expected HearthError")
except heng.HearthError:
    pass
for path, kw, msg in (("C:/fail.hearth", {}, "simulated failure"), ("C:/noavx512.hearth", {"isa": "avx512"}, "AVX-512")):
    try:
        heng.Engine(path, **kw); raise SystemExit("expected HearthError")
    except heng.HearthError as ex:
        assert msg in str(ex), ex
try:
    heng.Engine("C:/x.hearth", policy="bogus")
except ValueError:
    pass
gc.collect()
print("ALL OK")
'''


def abi_layout_c() -> str:
    """C function reporting sizeof(struct) then offsetof/sizeof of every field in header order."""
    structs = parse_c_structs(HEADER.read_text(encoding="utf-8"))
    lines = ["HEARTH_API int stub_layout(size_t *o) {", "    int i = 0;"]
    for name in ("hearth_options", "hearth_model_info", "hearth_stats"):
        lines.append(f"    o[i++] = sizeof({name});")
        for field, *_ in structs[name]:
            lines.append(f"    o[i++] = offsetof({name}, {field}); o[i++] = sizeof((({name} *)0)->{field});")
    lines += ["    return i;", "}"]
    return "\n".join(lines) + "\n"


def test_engine_against_compiled_abi_stub(tmp_path):
    import os
    src = tmp_path / "abi_stub.c"
    src.write_text(ABI_STUB_C + abi_layout_c(), encoding="utf-8")
    out = tmp_path / ("hearth_abi_stub.dll" if os.name == "nt" else "libhearth_abi_stub.so")
    try:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "hxcc.py"), "--shared", "-o", str(out), str(src)],
                           capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"no C toolchain: {e}")
    if r.returncode != 0 or not out.exists():
        pytest.skip(f"could not compile the ABI stub (no C toolchain?): {(r.stdout + r.stderr)[-400:]}")
    assert "warning" not in (r.stdout + r.stderr).lower(), r.stdout + r.stderr
    env = _env()
    env["HEARTH_LIB"] = str(out)
    d = subprocess.run([sys.executable, "-c", ABI_DRIVER], capture_output=True, text=True, env=env, cwd=ROOT,
                       timeout=300)
    assert d.returncode == 0 and "ALL OK" in d.stdout, d.stdout + d.stderr


# ---------------------------------------------------------------------------------
# Integration with the real engine (skipped without the library / synth)
# ---------------------------------------------------------------------------------

def _real_engine_available() -> bool:
    try:
        from hearth import _native
        lib = _native.lib()
    except Exception:
        return False
    return hasattr(lib, "hearth_open")


real = pytest.mark.skipif(not _real_engine_available(), reason="native Hearth library with the engine API not built")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    synth = pytest.importorskip("hearth.synth")
    from hearth import quant
    d = tmp_path_factory.mktemp("rt_models")
    return synth.make_tiny(d / "tiny_q4.hearth", arch="qwen3_moe", dtype=quant.Q4, seed=11, max_seq=256)


@real
def test_real_engine_basics(tiny_model):
    meta = read_metadata(tiny_model)
    with heng.Engine(tiny_model, cache_gb=0.0, threads=2) as e:
        info = e.info
        assert info["vocab_size"] == meta["vocab_size"] and info["n_layers"] == meta["n_layers"]
        assert isinstance(info["arch"], str) and e.pos == 0
        lg = e.eval([1, 2, 3])
        assert lg.dtype == np.float32 and lg.shape == (info["vocab_size"],) and e.pos == 3
        all_lg = e.eval([4, 5], all_logits=True)
        assert all_lg.shape == (2, info["vocab_size"])
        lg0, all0 = lg.copy(), all_lg.copy()
        e.rewind(3)
        again = e.eval([4, 5], all_logits=True)
        assert np.array_equal(again, all0)                # deterministic re-evaluation after rewind
        nxt = e.eval([6])                                 # reuses the buffer behind `again`
        for x, x0 in ((lg, lg0), (all_lg, all0), (again, all0)):
            assert np.array_equal(x, x0), "eval results must be copies, not views of the reusable buffer"
        assert not np.array_equal(nxt, all0[0])           # so an overwrite would have shown
        st = e.stats()
        assert {n for n, _ in heng.HearthStats._fields_} <= set(st) and st["tokens"] >= 7
        e.reset()
        assert e.pos == 0
        with pytest.raises(ValueError):
            e.eval([info["vocab_size"]])
        with pytest.raises(ValueError):
            e.rewind(1)
    e.close()
    e.close()
    with pytest.raises(heng.HearthError):
        e.eval([1])


@real
def test_real_engine_kv_capacity_and_rewind_to_zero(tiny_model):
    """Engine.kv_capacity is what the C engine enforces (the model allows 256)."""
    for req, want in ((0, 256), (100, 100), (1000, 256)):
        with heng.Engine(tiny_model, cache_gb=0.0, max_seq=req) as e:
            assert e.kv_capacity == want and e.info["max_seq"] == 256
            e.eval([1] * want)                          # exactly full
            with pytest.raises(heng.HearthError):
                e.eval([1])
            e.rewind(0)
            assert e.pos == 0
            first = e.eval([3, 4])
            e.rewind(0)
            assert np.array_equal(e.eval([3, 4]), first)
    with pytest.raises(ValueError, match="max_seq"):
        heng.Engine(tiny_model, max_seq=2 ** 32 + 100)  # ctypes would have passed 100


@real
def test_real_engine_open_errors(tmp_path):
    with pytest.raises(heng.HearthError):
        heng.Engine(tmp_path / "missing.hearth")
    bad = tmp_path / "bad.hearth"
    bad.write_bytes(b"HRTH" + bytes(100))
    with pytest.raises(heng.HearthError):
        heng.Engine(bad)
    with pytest.raises(ValueError):
        heng.Engine(bad, policy="fifo")


@real
def test_real_engine_speculative_lossless(tiny_model):
    prompt = [5, 9, 17, 5, 9, 17, 5, 9]
    with heng.Engine(tiny_model, cache_gb=0.0) as e:
        plain, _ = generate_ids(e, prompt, 40)
        spec, st = generate_ids(e, prompt, 40, None, (), "ngram", 4, 3)
    assert spec == plain
    assert st.forwards > 0


@real
def test_real_server_prefix_reuse_stays_correct(tiny_model):
    """Regression (found on the real engine): a max_tokens=0 request desynced the
    server's KV record, and the next request reused the wrong KV. Token ids in
    and out (no tokenizer) so the comparison is exact."""
    pre = "<|im_start|>user\n"
    pa, pb = list((pre + "A" * 40).encode()), list((pre + "B" * 60).encode())
    pc = list(b"unrelated: nothing in common")        # reuses nothing: the server rewinds to 0
    with heng.Engine(tiny_model, cache_gb=0.0) as ref:
        want_b = generate_ids(ref, pb, 10)[0]
        want_a = generate_ids(ref, pa + [10], 6)[0]
        want_c = generate_ids(ref, pc, 5)[0]
    with heng.Engine(tiny_model, cache_gb=0.0) as e, running_server(e, tokenizer=None, stop_ids=[]) as srv:
        def comp(prompt, n):
            st, _, body = request(srv, "POST", "/v1/completions", {"prompt": prompt, "max_tokens": n, "temperature": 0})
            r = json.loads(body)
            assert st == 200, r
            return r["choices"][0]["token_ids"], srv.app.last_generation
        assert len(comp(pa, 5)[0]) == 5
        assert comp(pb, 0)[0] == []
        ids, gen = comp(pb, 10)
        assert ids == want_b and gen["reused_tokens"] == len(pre)
        ids, gen = comp(pa + [10], 6)
        assert ids == want_a and gen["reused_tokens"] == len(pre)
        assert srv.app.conv.held == (pa + [10] + ids)[:e.pos]
        ids, gen = comp(pc, 5)
        assert ids == want_c and gen["reused_tokens"] == 0
        assert srv.app.conv.held == (pc + ids)[:e.pos]
        st, _, body = request(srv, "POST", "/v1/completions", {"prompt": [1] * 300, "max_tokens": 2})
        assert st == 400 and "the context holds 256" in json.loads(body)["error"]["message"]


@real
def test_real_engine_bench_cli(tiny_model, capsys):
    from hearth.cli import main
    assert main(["bench", str(tiny_model), "--prompt-tokens", "16", "--gen-tokens", "8", "--json",
                 "--cache-gb", "0"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["decode_tokens"] == 8 and res["decode_stats"]["tokens"] == 8
