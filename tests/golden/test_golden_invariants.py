"""Golden invariant tests — MAINTAINER-OWNED. Do not modify (INV-VERIFY).

These encode governance/INVARIANTS.md as executable checks against the real
engine (C shared library) and the numpy reference. Hash-locked by
tests/golden/MANIFEST.sha256 (governance/tools/check_golden.py).
"""
from __future__ import annotations

import itertools
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from hearth import quant
from hearth.engine import Engine, HearthError
from hearth.reference import Reference
from hearth.synth import make_tiny

ARCHS = ["qwen3_moe", "olmoe", "mixtral", "qwen2_moe", "deepseek_v3"]
TOKENS = [1, 17, 42, 99, 3, 250, 7, 7, 64, 128, 5, 200]


def read_trace(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    magic, ver, n_layers, n_experts, top_k, n_moe = struct.unpack_from("<6I", raw, 0)
    assert magic == 0x52545248 and ver == 1
    ids = np.frombuffer(raw, dtype="<u2", offset=24)
    return ids.reshape(-1, n_moe, top_k)


@pytest.fixture(scope="session")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("golden_models")
    out = {}
    for arch in ARCHS:
        out[(arch, "f32")] = make_tiny(d / f"{arch}_f32.hearth", arch=arch, dtype=quant.F32, seed=1)
        out[(arch, "q4")] = make_tiny(d / f"{arch}_q4.hearth", arch=arch, dtype=quant.Q4, seed=2)
    out["big_q4"] = make_tiny(d / "big_q4.hearth", arch="deepseek_v3", dtype=quant.Q4, seed=3,
                              n_layers=4, n_experts=32, top_k=4)
    return out


def run(path, tokens, one_by_one=True, **kw):
    with Engine(path, **kw) as e:
        if one_by_one:
            return np.stack([e.eval([t]) for t in tokens])
        return e.eval(tokens, all_logits=True)


@pytest.mark.parametrize("arch", ARCHS)
def test_engine_matches_reference(models, arch, tmp_path):
    """INV-NUM-1 / INV-NUM-2 on F32 containers."""
    path = models[(arch, "f32")]
    ref = Reference(path)
    want = np.stack([ref.eval([t])[0] for t in TOKENS])
    want_routes = ref.routing_history()
    trace = tmp_path / "t.hrtr"
    with Engine(path) as e:
        e.trace_start(trace)
        got = np.stack([e.eval([t]) for t in TOKENS])
        e.trace_stop()
    scale = max(1.0, float(np.abs(want).max()))
    err = float(np.abs(got - want).max())
    assert err <= 1e-3 * scale, f"{arch}: max abs logit error {err:.3e} (scale {scale:.2f})"
    got_routes = read_trace(trace)
    assert np.array_equal(got_routes, want_routes), f"{arch}: routing differs from reference"


@pytest.mark.parametrize("arch", ARCHS)
def test_quantized_close_to_reference(models, arch):
    """Q4 experts / Q8 dense: reference emulates activation quantization, so the
    only differences are float-order effects."""
    path = models[(arch, "q4")]
    ref = Reference(path, emulate_act_quant=True)
    want = np.stack([ref.eval([t])[0] for t in TOKENS[:6]])
    got = run(path, TOKENS[:6])
    scale = max(1.0, float(np.abs(want).max()))
    assert float(np.abs(got - want).max()) <= 2e-2 * scale


def test_scheduling_independence(models, tmp_path):
    """INV-DET-1: where weights came from and how work was scheduled never changes output."""
    path = models["big_q4"]
    base = run(path, TOKENS, threads=1, io_threads=1, cache_gb=64.0, prefetch="off", direct_io=False)
    usage = tmp_path / "u.usage"
    run(path, TOKENS, usage_out=str(usage))
    assert usage.exists() and usage.stat().st_size > 24
    configs = [
        dict(threads=3, io_threads=4, cache_gb=0.0, policy="lru", prefetch="off"),
        dict(threads=8, io_threads=2, cache_gb=0.0, policy="lfu", prefetch="shared", prefetch_extra=2),
        dict(threads=2, io_threads=8, cache_gb=0.002, policy="lfu", prefetch="next", direct_io=True),
        dict(threads=5, cache_gb=64.0, usage_in=str(usage), pin_fraction=0.5, warm_start=True),
        dict(threads=4, cache_gb=0.0, mirrors=[str(path)], io_threads=3),
    ]
    for kw in configs:
        got = run(path, TOKENS, **kw)
        assert np.array_equal(got, base), f"output changed under {kw}"


@pytest.mark.parametrize("arch", ["deepseek_v3", "qwen3_moe"])
def test_batch_equals_sequential(models, arch):
    """INV-DET-2."""
    path = models[(arch, "q4")]
    seq = run(path, TOKENS, one_by_one=True)
    batch = run(path, TOKENS, one_by_one=False)
    assert np.array_equal(seq, batch)
    chunked = run(path, TOKENS, one_by_one=False, max_batch=5)
    assert np.array_equal(seq, chunked)


def test_rewind(models):
    path = models[("qwen3_moe", "q4")]
    with Engine(path) as e:
        full = e.eval(TOKENS, all_logits=True)
        e.rewind(5)
        assert e.pos == 5
        again = e.eval(TOKENS[5:], all_logits=True)
    assert np.array_equal(full[5:], again)


def test_isa_identity(models):
    """INV-DET-3 end to end."""
    path = models["big_q4"]
    outs = {}
    for isa in ["scalar", "avx2", "avx512"]:
        try:
            outs[isa] = run(path, TOKENS[:6], isa=isa)
        except HearthError:
            continue  # CPU cannot run this ISA
    assert "scalar" in outs
    for isa, o in outs.items():
        assert np.array_equal(o, outs["scalar"]), f"{isa} differs from scalar"


@pytest.mark.parametrize("dtype", [quant.F32, quant.F16, quant.BF16, quant.Q8, quant.Q4])
def test_kernel_isa_and_batch_identity(dtype):
    """INV-DET-3 + INV-DET-2 at kernel level through hearth_matmul."""
    from hearth import _native
    rng = np.random.default_rng(dtype)
    rows, cols, T = 37, 256, 5
    W = rng.standard_normal((rows, cols)).astype(np.float32)
    X = rng.standard_normal((T, cols)).astype(np.float32)
    Wq = quant.quantize(W, dtype)
    results = {}
    for isa in (1, 2, 3):
        Y = _native.matmul(dtype, Wq, rows, cols, X, isa=isa)
        if Y is None:
            continue
        Y1 = np.stack([_native.matmul(dtype, Wq, rows, cols, X[t:t + 1], isa=isa)[0] for t in range(T)])
        assert np.array_equal(Y, Y1), f"isa {isa}: batched != single"
        results[isa] = Y
    assert 1 in results
    for isa, Y in results.items():
        assert np.array_equal(Y, results[1]), f"isa {isa} != scalar for dtype {dtype}"
    # and the numbers are actually right
    deq = quant.dequantize(Wq, dtype, (rows, cols))
    ref = X.astype(np.float64) @ deq.astype(np.float64).T
    tol = 1e-4 if dtype in (quant.F32,) else 2e-2
    assert np.abs(results[1] - ref).max() <= tol * max(1.0, np.abs(ref).max())


def test_malformed_files_fail_cleanly(models, tmp_path):
    """INV-FMT / INV-SAFE: truncated or corrupted containers raise, never crash."""
    src = models[("qwen3_moe", "q4")]
    data = src.read_bytes()
    cases = {
        "empty": b"",
        "magic": b"XXXX" + data[4:],
        "truncated_half": data[: len(data) // 2],
        "truncated_dirs": data[:100],
        "flipped_counts": data[:32] + struct.pack("<Q", 1 << 40) + data[40:],
    }
    for name, blob in cases.items():
        p = tmp_path / f"{name}.hearth"
        p.write_bytes(blob)
        with pytest.raises(HearthError):
            Engine(p).close()
