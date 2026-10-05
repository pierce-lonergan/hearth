"""Container writer/reader (docs/FORMAT.md) and dtype encodings (FORMAT.md §6, NUMERICS.md §2)."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from hearth import _native, quant
from hearth.format import (FLAG_ALIASED, ContainerReader, ContainerWriter, decode_meta, encode_meta,
                           slab_layout)
from hearth.quant import BF16, F16, F32, I32, Q4, Q8, U8

needs_native = pytest.mark.skipif(not _native.available(), reason="Hearth native library not available "
                                  "(build it with scripts/hxcc.py --shared or set HEARTH_LIB)")

BASE_META = dict(arch="synthetic", n_layers=3, d_model=64, vocab_size=96, n_heads=2, dense_ffn_dim=64)
MOE_META = dict(BASE_META, n_experts=4, top_k=2, expert_ffn_dim=64)


def _random_quant_bytes(rng, dtype, rows, cols):
    """Valid Q8/Q4 encodings built without the native quantizer."""
    nb = cols // 64
    if dtype == Q8:
        blk = np.zeros((rows, nb), dtype=[("d", "<f2"), ("q", "i1", (64,))])
        blk["q"] = rng.integers(-127, 128, size=(rows, nb, 64))
    else:
        blk = np.zeros((rows, nb), dtype=[("d", "<f2"), ("qs", "u1", (32,))])
        blk["qs"] = rng.integers(0, 256, size=(rows, nb, 32))
    blk["d"] = rng.uniform(-0.05, 0.05, size=(rows, nb)).astype(np.float16)
    return blk.tobytes()


def _slow_decode(buf, dtype, rows, cols):
    """Element-by-element decode straight from the FORMAT.md §6 table."""
    out = np.zeros((rows, cols), dtype=np.float64)
    bsz = 66 if dtype == Q8 else 34
    nb = cols // 64
    for r in range(rows):
        for b in range(nb):
            o = (r * nb + b) * bsz
            d = float(np.frombuffer(buf[o:o + 2], dtype="<f2")[0])
            for j in range(64):
                if dtype == Q8:
                    q = struct.unpack("b", buf[o + 2 + j:o + 3 + j])[0]
                else:
                    byte = buf[o + 2 + (j % 32)]
                    q = (byte & 15 if j < 32 else byte >> 4) - 8
                out[r, b * 64 + j] = d * q
    return out


# ---------------------------------------------------------------- quant helpers

def test_row_bytes():
    assert quant.row_bytes(F32, 10) == 40 and quant.row_bytes(I32, 3) == 12
    assert quant.row_bytes(F16, 10) == 20 and quant.row_bytes(BF16, 10) == 20 and quant.row_bytes(U8, 7) == 7
    assert quant.row_bytes(Q8, 128) == 132 and quant.row_bytes(Q4, 128) == 68
    with pytest.raises(ValueError):
        quant.row_bytes(Q4, 100)
    with pytest.raises(ValueError):
        quant.dtype_of(9)
    assert quant.dtype_of("q4") == Q4 and quant.dtype_of("BF16") == BF16


def test_bf16_rne_and_specials():
    x = np.array([1.0, 1.00390625, 1.01171875, 1.0 + 2 ** -8 + 2 ** -20, -2.5, np.inf, -np.inf, 3.4e38, 0.0, -0.0],
                 dtype=np.float32)
    b = quant.f32_to_bf16_bits(x)
    # ties to even: 1+2^-8 -> 1.0 (even), 1+3*2^-8 -> 1+2^-6; just above a tie rounds up
    assert list(b[:4]) == [0x3F80, 0x3F80, 0x3F82, 0x3F81]
    assert b[4] == 0xC020 and b[5] == 0x7F80 and b[6] == 0xFF80
    assert b[7] == 0x7F80                       # rounds past the largest finite bf16 -> inf
    assert b[8] == 0x0000 and b[9] == 0x8000
    nan = np.array([np.nan], dtype=np.float32)
    nan_bits = np.array([0x7F800001, 0xFFC12345], dtype=np.uint32).view(np.float32)
    for v in (nan, nan_bits):
        out = quant.bf16_bits_to_f32(quant.f32_to_bf16_bits(v))
        assert np.isnan(out).all()


def test_f16_encode_rne_and_nan():
    x = np.array([1.0 + 2 ** -11, 1.0 + 3 * 2 ** -11, 65504, 65520, 1e-8, np.nan], dtype=np.float32)
    h = np.frombuffer(quant.quantize(x, F16), dtype="<u2")
    assert h[0] == 0x3C00 and h[1] == 0x3C02 and h[2] == 0x7BFF and h[3] == 0x7C00 and h[4] == 0
    assert (h[5] & 0x7E00) == 0x7E00          # quiet NaN
    # NaN payloads: quieted, top 10 payload bits kept (vcvtps2ph semantics, as the native converter)
    nans = np.array([0x7F8014E8, 0x7F8CFC76, 0xFFBB9945, 0x7FC00000], dtype=np.uint32).view(np.float32)
    assert list(np.frombuffer(quant.quantize(nans, F16), dtype="<u2")) == [0x7E00, 0x7E67, 0xFFDC, 0x7E00]


def _f16_bits_definition(h: np.ndarray) -> np.ndarray:
    """binary16 -> float32 bit patterns, vcvtph2ps semantics (NaNs quiet, payload kept)."""
    h = h.astype(np.uint32)
    sign, e, m = (h & 0x8000) << 16, (h >> 10) & 0x1F, h & 0x3FF
    sub = (m.astype(np.float32) * np.float32(2.0 ** -24)).view(np.uint32)
    return np.where(e == 31, sign | 0x7F800000 | (m << 13) | np.where(m != 0, 0x00400000, 0),
                    np.where(e > 0, sign | ((e + 112) << 23) | (m << 13), sign | sub)).astype(np.uint32)


def test_f16_decode_all_patterns():
    h = np.arange(1 << 16, dtype=np.uint16)
    want = _f16_bits_definition(h)
    assert np.array_equal(quant.dequantize(h.tobytes(), F16, (1 << 16,)).view(np.uint32), want)
    assert np.array_equal(quant.f16_bits_to_f32(h).view(np.uint32), want)
    # Q8/Q4 block scales decode the same way (q = 1 shows d itself)
    blk = np.zeros(1 << 16, dtype=[("d", "<u2"), ("q", "i1", (64,))])
    blk["d"], blk["q"] = h, 1
    _, d = quant.split_blocks(blk.tobytes(), Q8, (1 << 16, 64))
    assert np.array_equal(d[:, 0].view(np.uint32), want)
    if _native.available():
        assert np.array_equal(_native.dequantize(F16, h, 1, 1 << 16).view(np.uint32), want.reshape(1, -1))
        blk["q"] = np.arange(64) - 32                   # includes q = 0: 0*inf and 0*NaN cases
        for dt, buf in ((Q8, blk.tobytes()), (Q4, _q4_all_scales(h))):
            ours = quant.dequantize(buf, dt, (1 << 16, 64)).view(np.uint32)
            assert np.array_equal(ours, _native.dequantize(dt, buf, 1 << 16, 64).view(np.uint32)), quant.NAMES[dt]


def _q4_all_scales(h):
    blk = np.zeros(len(h), dtype=[("d", "<u2"), ("qs", "u1", (32,))])
    blk["d"] = h
    blk["qs"] = (np.arange(32) % 16) | ((15 - np.arange(32) % 16) << 4)
    return blk.tobytes()


def test_act_quant_q8_ignores_nan_like_the_engine():
    """C (`if (a > amax)`) never lets a NaN win the block max; NaN products quantize to 0."""
    x = np.linspace(-2, 2, 192).astype(np.float32)
    x[10] = np.nan
    x[64:128] = np.nan                                     # an all-NaN block: d = 0, q = 0
    x[130] = np.inf                                        # amax = inf: d = inf, id = 0, every q = 0
    q, d = quant.act_quant_q8(x)
    clean = x[:64].copy()
    clean[10] = 0.0
    q0, d0 = quant.act_quant_q8(clean)
    assert d[0] == d0[0] and np.isfinite(d[0]) and np.array_equal(q[:64], q0)
    assert d[1] == 0 and (q[64:128] == 0).all()
    assert d[2] == np.inf and (q[128:] == 0).all()


def test_act_quant_q8_matches_definition():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((3, 192)) * [[0.001], [1.0], [300.0]]).astype(np.float32)
    x[1, 64:128] = 0.0                                    # an all-zero block: d = 0, q = 0
    x[2, 5] = np.float32(127.5) * (np.abs(x[2, :64]).max() / np.float32(127.0))  # far from a tie, still valid
    q, d = quant.act_quant_q8(x)
    assert q.dtype == np.int8 and d.dtype == np.float32 and q.shape == x.shape and d.shape == (3, 3)
    for r in range(3):
        for b in range(3):
            blk = x[r, b * 64:(b + 1) * 64]
            amax = np.float32(np.abs(blk).max())
            dd = np.float32(amax / np.float32(127.0))
            idd = np.float32(1.0) / dd if dd != 0 else np.float32(0.0)
            want = [int(np.rint(np.clip(np.float32(v * idd), -127, 127))) for v in blk]
            assert d[r, b] == dd
            assert list(q[r, b * 64:(b + 1) * 64]) == want
    assert (q[1, 64:128] == 0).all() and d[1, 1] == 0
    half = np.zeros(64, dtype=np.float32)
    half[0], half[1], half[2] = 127.0, 0.5, 1.5                # d = 1: exact ties go to even
    qh, _ = quant.act_quant_q8(half)
    assert list(qh[:3]) == [127, 0, 2]


def test_q8_q4_decode_layout():
    rng = np.random.default_rng(0)
    for dt in (Q8, Q4):
        buf = _random_quant_bytes(rng, dt, 3, 128)
        got = quant.dequantize(buf, dt, (3, 128))
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got.astype(np.float64), _slow_decode(buf, dt, 3, 128))
        q, d = quant.split_blocks(buf, dt, (3, 128))
        np.testing.assert_array_equal(np.repeat(d, 64, axis=1) * q.astype(np.float32), got)


def test_short_buffers_raise():
    with pytest.raises(ValueError):
        quant.dequantize(b"\0" * 65, Q8, (1, 64))
    with pytest.raises(ValueError):
        quant.split_blocks(np.zeros(33, np.uint8), Q4, (64,))
    assert quant.dequantize(b"\0" * 70, Q8, (1, 64)).shape == (1, 64)   # extra bytes are ignored


def test_float_encodings_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 70)).astype(np.float32)
    np.testing.assert_array_equal(quant.dequantize(quant.quantize(x, F32), F32, x.shape), x)
    np.testing.assert_array_equal(quant.dequantize(quant.quantize(x, F16), F16, x.shape), x.astype(np.float16).astype(np.float32))
    b = quant.dequantize(quant.quantize(x, BF16), BF16, x.shape)
    assert np.abs(b - x).max() <= np.abs(x).max() * 2 ** -8
    cube = rng.standard_normal((2, 3, 64)).astype(np.float32)      # rows are the last axis for any ndim
    for dt in (F32, F16, BF16):
        assert quant.quantize(cube, dt) == quant.quantize(cube.reshape(6, 64), dt)
    enc = np.frombuffer(quant.quantize(cube, F16), np.uint8)
    spaced = np.zeros(2 * enc.size, np.uint8)
    spaced[::2] = enc                                                # a non-contiguous ndarray buffer works too
    assert not spaced[::2].flags.c_contiguous
    assert np.array_equal(quant.dequantize(spaced[::2], F16, cube.shape), quant.dequantize(enc, F16, cube.shape))


@needs_native
def test_native_quantize_q8_q4_error_bounds():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((16, 256)).astype(np.float32)
    xb = x.reshape(16, 4, 64).astype(np.float64)
    y8 = quant.dequantize(quant.quantize(x, Q8), Q8, x.shape).reshape(16, 4, 64)
    amax = np.abs(xb).max(axis=2, keepdims=True)
    assert (np.abs(y8 - xb) / amax).max() <= 0.5 / 127 * 1.01
    # Q4 (NUMERICS §6): never worse, per block, than the plain absmax scale xm/7 stored as f16
    y4 = quant.dequantize(quant.quantize(x, Q4), Q4, x.shape).reshape(16, 4, 64)
    idx = np.abs(xb).argmax(axis=2)[..., None]
    d0 = np.take_along_axis(xb, idx, 2).astype(np.float32) / np.float32(7.0)
    d0 = d0.astype(np.float16).astype(np.float64)
    base = d0 * np.clip(np.rint(xb / d0), -8, 7)
    assert (((y4 - xb) ** 2).sum(axis=2) <= ((base - xb) ** 2).sum(axis=2) * (1 + 1e-9)).all()
    assert np.sqrt(((y4 - xb) ** 2).mean()) < 0.12 * np.sqrt((xb ** 2).mean())
    big = rng.standard_normal((1031, 1024)).astype(np.float32)   # large enough to use 4 threads
    for dt in (Q8, Q4):   # threaded quantization is identical to single-threaded
        one = quant.quantize(big, dt, threads=1)
        assert quant.quantize(big, dt, threads=4) == one and quant.quantize(big, dt) == one
        assert quant.quantize(big[:5], dt) == one[:5 * quant.row_bytes(dt, 1024)]
        assert quant.quantize(big[:6].reshape(2, 3, 1024), dt) == one[:6 * quant.row_bytes(dt, 1024)]   # N-D


# ---------------------------------------------------------------- metadata

def test_meta_roundtrip_all_types(tmp_path):
    meta = dict(BASE_META, n_experts=2, top_k=1, expert_ffn_dim=64, norm_eps=1e-5, rope_attn_factor=1.25,
                layer_kind=[0, 1, 1], eos_ids=[5, 6],
                bos_id=-1, tokenizer="m.tokenizer.json", chat_template="{{ messages }} é中",
                x_u32=7, x_u64=1 << 40, x_u64max=(1 << 64) - 1, x_f32=0.1, x_str="hi", x_u32a=[0, 2, 0xFFFFFFFF],
                x_f32a=[0.5, 1.5], x_f32nd=np.array([0.5, -2.0], np.float32), x_u32nd=np.array([3, 0]),
                x_boola=[True, False],
                x_u8a=b"\x01\x02\xff", x_bool=True, x_empty=[], x_none=None)
    p = tmp_path / "m.hearth"
    with ContainerWriter(p, meta) as w:
        w.declare_tensor("t", (4,), F32)
        w.declare_experts(3, 2, 64, 64, F32, [1, 2])
        w.write_tensor("t", np.arange(4, dtype=np.float32))
        for li in (1, 2):
            for e in range(2):
                w.write_expert(li, e, *(np.full((64, 64), e, np.float32),) * 3)
    with ContainerReader(p) as r:
        m = r.meta
        assert m["n_layers"] == 3 and m["arch"] == "synthetic"
        assert m["norm_eps"] == float(np.float32(1e-5)) and m["rope_attn_factor"] == 1.25
        assert m["layer_kind"] == [0, 1, 1] and m["eos_ids"] == [5, 6] and m["bos_id"] == 0xFFFFFFFF
        assert m["chat_template"] == meta["chat_template"]
        assert m["x_u32"] == 7 and m["x_u64"] == 1 << 40 and m["x_f32"] == float(np.float32(0.1))
        assert m["x_u64max"] == (1 << 64) - 1 and m["x_f32nd"] == [0.5, -2.0] and m["x_u32nd"] == [3, 0]
        assert m["x_boola"] == [1, 0]
        assert m["x_str"] == "hi" and m["x_u32a"] == [0, 2, 0xFFFFFFFF] and m["x_f32a"] == [0.5, 1.5]
        assert m["x_u8a"] == [1, 2, 255] and m["x_bool"] == 1 and m["x_empty"] == [] and "x_none" not in m
        c = r.config
        assert c["n_kv_heads"] == 2 and c["head_dim"] == 32 and c["rope_dim"] == 32 and c["max_seq"] == 4096
        assert c["n_group"] == 1 and c["routed_scale"] == 1.0 and c["n_moe_layers"] == 2
        assert abs(c["attn_scale"] - 32 ** -0.5) < 1e-7
    # raw type codes per the FORMAT.md §3.1 schema
    raw = encode_meta({"layer_kind": [1], "eos_ids": [2], "n_layers": 1, "norm_eps": 0.5, "arch": "a", "x": 1 << 33})
    types, pos = [], 0
    while pos < len(raw):
        (kl,) = struct.unpack_from("<H", raw, pos)
        t = raw[pos + 2 + kl]
        types.append((raw[pos + 2:pos + 2 + kl].decode(), t))
        pos += 3 + kl + {1: 4, 2: 4, 3: 8}.get(t, 0)
        if t in (4, 5, 6, 7):
            (n,) = struct.unpack_from("<I", raw, pos)
            pos += 4 + n * (1 if t in (4, 7) else 4)
    assert dict(types) == {"layer_kind": 7, "eos_ids": 5, "n_layers": 1, "norm_eps": 2, "arch": 4, "x": 3}
    assert decode_meta(raw)["x"] == 1 << 33


def test_meta_rejects_bad_values():
    for bad in ({"n_layers": -1}, {"n_layers": 1 << 33}, {"x": object()}, {"layer_kind": [300]}, {"x": -5},
                {"": 1}, {"k" * 0x10000: 1}, {"x": 1 << 64}):
        with pytest.raises(ValueError):
            encode_meta(bad)


# ---------------------------------------------------------------- tensors and experts

def _container(path, rng, alias=None, dtype_fn=None, native=False):
    """A small container with dense tensors of every dtype and a routed-expert section."""
    D, F, E, L = 64, 128, 4, 3
    meta = dict(MOE_META, expert_ffn_dim=F, layer_kind=[0, 1, 1])
    tens = {
        "a_f32": (rng.standard_normal((3, 5)).astype(np.float32), F32),
        "a_f16": (rng.standard_normal((2, 3, 4)).astype(np.float32), F16),
        "a_bf16": (rng.standard_normal((7,)).astype(np.float32), BF16),
        "a_i32": (rng.integers(-1000, 1000, size=(2, 2, 2, 2)).astype(np.int32), I32),
        "a_u8": (rng.integers(0, 256, size=(9,)).astype(np.uint8), U8),
        "a_q8": (_random_quant_bytes(rng, Q8, 3, 128), Q8),
        "a_q4": (_random_quant_bytes(rng, Q4, 2, 64), Q4),
    }
    shapes = {"a_q8": (3, 128), "a_q4": (2, 64)}
    dtype_fn = dtype_fn or (lambda li, e: [F32, F16, BF16, Q8 if native else F16][e])
    experts = {}
    with ContainerWriter(path, meta) as w:
        for n, (a, dt) in tens.items():
            w.declare_tensor(n, shapes.get(n, getattr(a, "shape", None)), dt)
        w.declare_experts(L, E, D, F, dtype_fn, [1, 2], alias=alias)
        for n, (a, dt) in tens.items():
            w.write_tensor(n, a)
        for li in (1, 2):
            for e in range(E):
                if alias and alias(li, e):
                    continue
                mats = (rng.standard_normal((F, D)).astype(np.float32), rng.standard_normal((F, D)).astype(np.float32),
                        rng.standard_normal((D, F)).astype(np.float32))
                w.write_expert(li, e, *mats)
                experts[(li, e)] = mats
    return tens, experts


def test_tensor_and_expert_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    p = tmp_path / "c.hearth"
    tens, experts = _container(p, rng, native=_native.available())
    with ContainerReader(p) as r:
        for n, (a, dt) in tens.items():
            info = r.tensors[n]
            assert info.dtype == dt and info.offset % 64 == 0
            got = r.read_tensor(n)
            raw = r.read_tensor(n, dequant=False)
            assert len(raw) == info.nbytes
            if dt in (Q8, Q4):
                assert raw == a
                np.testing.assert_array_equal(got, quant.dequantize(a, dt, info.shape))
            elif dt == F32 or dt in (I32, U8):
                np.testing.assert_array_equal(got, a)
                assert got.dtype == a.dtype
            else:
                np.testing.assert_array_equal(got, quant.dequantize(quant.quantize(a, dt), dt, a.shape))
        for li in range(3):
            for e in range(4):
                ent = r.expert_entry(li, e)
                if li == 0:
                    assert ent == (0, 0, 0, 0)
                    continue
                assert ent.offset % 4096 == 0 and ent.nbytes % 4096 == 0 and ent.flags == 0
                assert ent.nbytes == slab_layout(ent.dtype, 64, 128)[3]
                got = r.read_expert(li, e)
                for g, want in zip(got, experts[(li, e)]):
                    np.testing.assert_array_equal(g, quant.dequantize(quant.quantize(want, ent.dtype), ent.dtype, want.shape))
                # slab padding bytes are zero
                og, ou, od, nb = slab_layout(ent.dtype, 64, 128)
                end = od + 64 * quant.row_bytes(ent.dtype, 128)
                pad = r._read_at(ent.offset + end, nb - end)
                assert pad == bytes(len(pad))
        with pytest.raises(ValueError):
            r.read_expert(0, 1)


def test_file_layout_bytes(tmp_path):
    rng = np.random.default_rng(6)
    p = tmp_path / "c.hearth"
    _container(p, rng)
    raw = p.read_bytes()
    assert raw[:4] == b"HRTH"
    magic, ver, meta_off, meta_bytes, tdir_off, n_t, edir_off, n_e, align, res = struct.unpack_from("<IIQQQQQQII", raw, 0)
    assert (magic, ver, align, res) == (0x48545248, 1, 4096, 0)
    assert meta_off == 64 and tdir_off == meta_off + meta_bytes and edir_off == tdir_off + n_t * 128
    assert n_t == 7 and n_e == 12
    name, dt, nd, s0, s1, s2, s3, off, nb, rsv = struct.unpack_from("<80sII4IQQQ", raw, tdir_off)
    assert name.rstrip(b"\0") == b"a_f32" and (dt, nd, s0, s1, s2, s3) == (F32, 2, 3, 5, 1, 1) and nb == 60 and rsv == 0
    assert np.array_equal(np.frombuffer(raw[off:off + nb], "<f4").reshape(3, 5),
                          ContainerReader(p).read_tensor("a_f32"))
    # the writer packs: dense tensors 64-aligned right after the directories, slabs back to back
    with ContainerReader(p) as r:
        infos = sorted(r.tensors.values(), key=lambda t: t.offset)
        pos = -(-(edir_off + n_e * 32) // 64) * 64
        for t in infos:
            assert t.offset == pos
            pos = -(-(t.offset + t.nbytes) // 64) * 64
        slabs = sorted({r.expert_entry(li, e)[:2] for li in (1, 2) for e in range(4)})
        assert slabs[0][0] == -(-(infos[-1].offset + infos[-1].nbytes) // 4096) * 4096
        for (o1, n1), (o2, _) in zip(slabs, slabs[1:]):
            assert o2 == o1 + n1
        assert slabs[-1][0] + slabs[-1][1] == len(raw)


def test_slab_layout_numbers():
    assert slab_layout(Q4, 64, 128) == (0, 4352, 8704, 16384)
    assert slab_layout(Q8, 128, 64) == (0, 8448, 16896, 28672)
    assert slab_layout(F16, 33, 3) == (0, 256, 512, 4096)
    assert slab_layout(F32, 64, 64) == (0, 16384, 32768, 49152)


@pytest.mark.parametrize("native", [False, True])
def test_aliasing(tmp_path, native):
    if native and not _native.available():
        pytest.skip("native library not available")
    rng = np.random.default_rng(7)
    alias = lambda li, e: (li, e % 2) if e >= 2 else None  # noqa: E731
    p = tmp_path / "a.hearth"
    dt = Q4 if native else F16
    _container(p, rng, alias=alias, dtype_fn=lambda li, e: dt)
    full = tmp_path / "f.hearth"
    _container(full, np.random.default_rng(7), dtype_fn=lambda li, e: dt)
    assert p.stat().st_size < full.stat().st_size
    with ContainerReader(p) as r:
        for li in (1, 2):
            for e in range(4):
                ent = r.expert_entry(li, e)
                if e >= 2:
                    tgt = r.expert_entry(li, e % 2)
                    assert ent.flags & FLAG_ALIASED and ent.offset == tgt.offset and ent.nbytes == tgt.nbytes
                    for a, b in zip(r.read_expert(li, e), r.read_expert(li, e % 2)):
                        np.testing.assert_array_equal(a, b)
                else:
                    assert ent.flags == 0


def test_writer_validation(tmp_path):
    p = tmp_path / "v.hearth"
    w = ContainerWriter(p, BASE_META)
    w.declare_tensor("a", (2, 64), F32)
    with pytest.raises(ValueError):
        w.declare_tensor("a", (2, 64), F32)                      # duplicate
    with pytest.raises(ValueError):
        w.declare_tensor("q", (2, 100), Q8)                     # not a multiple of 64
    with pytest.raises(ValueError):
        w.declare_tensor("x" * 80, (2,), F32)                   # name too long
    with pytest.raises(ValueError):
        w.declare_tensor("z", (0, 3), F32)
    with pytest.raises(ValueError):
        w.declare_tensor("z", (1 << 32, 1), U8)
    with pytest.raises(ValueError):
        w.declare_tensor("z", (1, 1, 1, 1, 1), F32)             # at most 4 dimensions
    w.declare_tensor("u32max", (0xFFFFFFFF,), U8)                # largest legal dimension (never written)
    for bad in ((3, 0, 64, 64), (0, 4, 64, 64), (3, 4, 0, 64), (3, 4, 64, 0)):
        with pytest.raises(ValueError):
            w.declare_experts(*bad, F32, [1])
    del w._tensors["u32max"]
    with pytest.raises(ValueError):
        w.write_tensor("a", np.zeros((3, 64), np.float32))      # wrong size
    with pytest.raises(ValueError):
        w.write_tensor("a", b"\0" * 7)
    with pytest.raises(KeyError):
        w.write_tensor("nope", np.zeros(2, np.float32))
    with pytest.raises(ValueError):
        w.declare_tensor("late", (2,), F32)                     # declarations end at the first write
    with pytest.raises(ValueError, match="never written"):
        w.close()
    assert not p.exists()                                       # a failed close leaves nothing behind

    with pytest.raises(ValueError):
        with ContainerWriter(p, BASE_META) as w:
            w.declare_experts(3, 4, 64, 64, F32, [1], alias=lambda li, e: (li, 1) if e == 0 else ((li, 0) if e == 1 else None))
    with pytest.raises(ValueError, match="n_experts"):
        with ContainerWriter(p, dict(MOE_META, n_experts=8, layer_kind=[0, 1, 0])) as w:
            w.declare_experts(3, 4, 64, 64, F32, [1])           # disagrees with metadata
    with pytest.raises(ValueError, match="declare_experts"):
        with ContainerWriter(p, dict(MOE_META, layer_kind=[0, 1, 1])) as w:
            w.declare_experts(3, 4, 64, 64, F32, [1])           # layer 2 is MoE too
            w.write_expert(1, 0, *(np.ones((64, 64), np.float32),) * 3)
    assert not p.exists()
    with ContainerWriter(p, dict(MOE_META, layer_kind=[0, 1, 0])) as w:
        w.declare_experts(3, 4, 64, 64, F32, [1], alias=lambda li, e: (li, 0) if e else None)
        with pytest.raises(ValueError):
            w.write_expert(1, 2, *(np.zeros((64, 64), np.float32),) * 3)    # alias entries are not writable
        with pytest.raises(KeyError):
            w.write_expert(0, 0, *(np.zeros((64, 64), np.float32),) * 3)    # dense layer
        with pytest.raises(ValueError):
            w.write_expert(1, 0, np.zeros((64, 64), np.float32), np.zeros((64, 64), np.float32), np.zeros((32, 64), np.float32))
        w.write_expert(1, 0, *(np.ones((64, 64), np.float32),) * 3)
    assert ContainerReader(p).expert_entry(1, 3).flags == FLAG_ALIASED


def test_writer_lifecycle_and_names(tmp_path):
    p = tmp_path / "l.hearth"
    w = ContainerWriter(p, dict(MOE_META, n_layers=1, layer_kind=[1], dense_ffn_dim=0))
    for bad in ("", 5, None, "é", "x" * 80):
        with pytest.raises(ValueError):
            w.declare_tensor(bad, (2,), F32)
    w.declare_tensor("x" * 79, (2,), F32)
    w.declare_experts(1, 4, 64, 64, F16, [0])               # the smallest legal model: one MoE layer
    w.write_tensor("x" * 79, np.ones(2, np.float32))
    with pytest.raises(ValueError, match="cannot declare after the first write"):
        w.declare_experts(1, 4, 64, 64, F16, [0])
    for e in range(4):
        w.write_expert(0, e, *(np.full((64, 64), e, np.float32),) * 3)
    w.close()
    w.close()                                               # idempotent
    for call in (lambda: w.write_tensor("x" * 79, np.ones(2, np.float32)),
                 lambda: w.write_tensor_rows("x" * 79, 0, np.ones(2, np.float32)),
                 lambda: w.write_expert(0, 0, *(np.ones((64, 64), np.float32),) * 3),
                 lambda: w.declare_tensor("late", (1,), F32),
                 lambda: w.declare_experts(1, 4, 64, 64, F16, [0])):
        with pytest.raises(ValueError, match="writer is closed"):
            call()
    with ContainerReader(p) as r:
        assert r.tensors["x" * 79].shape == (2,) and r.read_expert(0, 3)[2][0, 0] == 3.0


def test_streamed_rows(tmp_path):
    rng = np.random.default_rng(8)
    a = rng.standard_normal((10, 64)).astype(np.float32)
    c = rng.standard_normal((2, 3, 64)).astype(np.float32)
    p = tmp_path / "s.hearth"
    with ContainerWriter(p, BASE_META) as w:
        w.declare_tensor("big", (10, 64), F16)
        w.declare_tensor("cube", (2, 3, 64), F32)              # 3-D: streamed as 6 rows of 64
        w.write_tensor_rows("big", 0, a[:4])
        with pytest.raises(ValueError):
            w.write_tensor_rows("big", 6, a[6:])                 # out of order
        with pytest.raises(ValueError):
            w.write_tensor_rows("big", 4, np.zeros((7, 64), np.float32))   # runs past the last row
        with pytest.raises(ValueError):
            w.write_tensor_rows("big", 4, np.zeros((0, 64), np.float32))
        with pytest.raises(ValueError):
            w.write_tensor("cube", np.zeros((3, 128), np.float32))         # right size, wrong row length
        w.write_tensor_rows("big", 4, quant.quantize(a[4:], F16))
        w.write_tensor_rows("cube", 0, c[0])
        w.write_tensor_rows("cube", 3, c[1].reshape(-1))
    assert w._f.closed
    np.testing.assert_array_equal(ContainerReader(p).read_tensor("big"),
                                  quant.dequantize(quant.quantize(a, F16), F16, a.shape))
    np.testing.assert_array_equal(ContainerReader(p).read_tensor("cube"), c)


# ---------------------------------------------------------------- malformed files

def _patch(raw: bytes, off: int, fmt: str, *vals) -> bytes:
    b = bytearray(raw)
    struct.pack_into(fmt, b, off, *vals)
    return bytes(b)


def _ov(a: str, b: str) -> str:
    """Regex for 'a overlaps b' in either order."""
    import re
    return f"{re.escape(a)} overlaps {re.escape(b)}|{re.escape(b)} overlaps {re.escape(a)}"


def _slab(raw, edir_off, i, off=None, nbytes=None, dtype=None, flags=None):
    """Patch fields of expert directory entry i."""
    e = edir_off + 32 * i
    for pos, fmt, v in ((0, "<Q", off), (8, "<Q", nbytes), (16, "<I", dtype), (20, "<I", flags)):
        if v is not None:
            raw = _patch(raw, e + pos, fmt, v)
    return raw


def test_malformed_files_raise(tmp_path):
    rng = np.random.default_rng(9)
    good = tmp_path / "g.hearth"
    _container(good, rng)
    raw = good.read_bytes()
    _, _, meta_off, meta_bytes, tdir_off, n_t, edir_off, n_e, _, _ = struct.unpack_from("<IIQQQQQQII", raw, 0)
    t0, t1 = tdir_off, tdir_off + 128               # entries of a_f32 [3, 5] and a_f16 [2, 3, 4]
    e1 = edir_off + 4 * 32                          # first entry of layer 1 (a real slab)
    off = [struct.unpack_from("<Q", raw, edir_off + 32 * i)[0] for i in range(n_e)]
    f32_off = struct.unpack_from("<Q", raw, t0 + 104)[0]
    assert [struct.unpack_from("<I", raw, edir_off + 32 * i + 16)[0] for i in (5, 6, 7)] == [F16, BF16, F16]
    cases = {   # name: (bytes, which rule must fire)
        "empty": (b"", "too small"),
        "short": (raw[:63], "too small"),
        "magic": (b"XXXX" + raw[4:], "bad magic"),
        "version": (_patch(raw, 4, "<I", 2), "unsupported version"),
        "align": (_patch(raw, 56, "<I", 512), "slab alignment"),
        "truncated_half": (raw[:len(raw) // 2], "outside the file"),
        "truncated_dirs": (raw[:100], "metadata .* outside the file"),
        "truncated_tail": (raw[:-1], "expert slab 11 .* outside the file"),
        "meta_off": (_patch(raw, 8, "<Q", len(raw)), "metadata .* outside"),
        "meta_bytes": (_patch(raw, 16, "<Q", 1 << 62), "metadata .* outside"),
        "meta_overrun": (_patch(raw, 16, "<Q", meta_bytes - 1), "runs past the end"),
        "tdir_off": (_patch(raw, 24, "<Q", len(raw) - 10), "tensor directory .* outside"),
        "n_tensors_huge": (_patch(raw, 32, "<Q", 1 << 40), "tensor count"),
        "n_tensors_more": (_patch(raw, 32, "<Q", n_t + 1), "empty name"),
        "n_tensors_over_size": (_patch(raw, 32, "<Q", len(raw) // 128 + 1), "tensor count"),
        "n_entries_over_size": (_patch(raw, 48, "<Q", len(raw) // 32 + 1), "expert entry count"),
        "edir_off": (_patch(raw, 40, "<Q", 1 << 63), "expert directory .* outside"),
        "n_entries": (_patch(raw, 48, "<Q", n_e - 1), "11 expert entries, expected"),
        "tensor_offset_range": (_patch(raw, t0 + 104, "<Q", len(raw)), "a_f32' .* outside"),
        "tensor_offset_align": (_patch(raw, t0 + 104, "<Q", f32_off + 4), "not 64-byte aligned"),
        "tensor_nbytes": (_patch(raw, t0 + 112, "<Q", 61), "nbytes 61 != 60"),
        "tensor_dtype": (_patch(raw, t0 + 80, "<I", 9), "unknown dtype 9"),
        "tensor_ndim": (_patch(raw, t0 + 84, "<I", 5), "ndim 5"),
        "tensor_ndim_0": (_patch(raw, t0 + 84, "<I", 0), "ndim 0"),
        "tensor_shape_zero": (_patch(raw, t0 + 88, "<I", 0), "bad shape"),
        "tensor_shape_zero_nbytes_zero": (_patch(_patch(raw, t0 + 88, "<I", 0), t0 + 112, "<Q", 0), "bad shape"),
        "tensor_unused_dim": (_patch(raw, t0 + 96, "<I", 5), "bad shape \\(3, 5, 5, 1\\)"),
        "tensor_name_unterminated": (_patch(raw, t0, "80s", b"y" * 80), "not NUL-terminated"),
        "tensor_name_duplicate": (_patch(raw, t1, "80s", b"a_f32"), "duplicate tensor 'a_f32'"),
        "slab_align": (_patch(raw, e1, "<Q", off[4] + 64), "not 4096-aligned"),
        "slab_size_align": (_patch(raw, e1 + 8, "<Q", 98304 + 64), "not 4096-aligned"),
        "slab_range": (_patch(raw, e1, "<Q", (len(raw) // 4096) * 4096), "expert slab 4 .* outside"),
        "slab_size": (_patch(raw, e1 + 8, "<Q", 4096), "4096 bytes < 98304 needed"),
        "slab_dtype": (_patch(raw, e1 + 16, "<I", 7), "bad dtype 7"),
        "slab_dtype_i32": (_patch(raw, e1 + 16, "<I", I32), "bad dtype 5"),
        "slab_missing": (_patch(raw, e1 + 8, "<Q", 0), "empty slab in MoE layer 1"),
        "slab_in_dense_layer": (_patch(_patch(raw, edir_off, "<Q", off[4]), edir_off + 8, "<Q", 4096),
                                "non-empty slab in dense layer 0"),
        "meta_type": (_patch(raw, meta_off + 2 + struct.unpack_from("<H", raw, meta_off)[0], "<B", 99), "unknown type 99"),
        # regions may not overlap (engine/src/modelfile.c check_regions)
        "tensor_on_tensor": (_patch(raw, t1 + 104, "<Q", f32_off), _ov("tensor 'a_f32'", "tensor 'a_f16'")),
        "tensor_on_preamble": (_patch(raw, t0 + 104, "<Q", 0), _ov("preamble", "tensor 'a_f32'")),
        "tensor_on_metadata": (_patch(raw, t0 + 104, "<Q", 64), _ov("metadata", "tensor 'a_f32'")),
        "tensor_on_tdir": (_patch(raw, t0 + 104, "<Q", 256), _ov("tensor directory", "tensor 'a_f32'")),
        "tensor_on_edir": (_patch(raw, t0 + 104, "<Q", edir_off // 64 * 64 + 64),
                           _ov("expert directory", "tensor 'a_f32'")),
        "slab_on_header": (_slab(raw, edir_off, 4, off=0), _ov("preamble", "expert slab of entry 4")),
        "slab_on_slab": (_slab(raw, edir_off, 5, off=off[4] + 4096),
                         _ov("expert slab of entry 4", "expert slab of entry 5")),
        "slab_shared_unflagged": (_slab(raw, edir_off, 7, off=off[5]), "entries 5 and 7 point at the same slab"),
        "slab_shared_dtype": (_slab(raw, edir_off, 6, off=off[5], flags=FLAG_ALIASED), "differ in size or dtype"),
        "slab_shared_size": (_slab(raw, edir_off, 7, off=off[5], nbytes=49152 + 4096, flags=FLAG_ALIASED),
                             "differ in size or dtype"),
    }
    for name, (blob, msg) in cases.items():
        p = tmp_path / f"{name}.hearth"
        p.write_bytes(blob)
        with pytest.raises(ValueError, match=msg):
            ContainerReader(p).close()
            pytest.fail(f"{name}: no error")
        p.unlink()                                  # the reader closed its handle (Windows refuses otherwise)
    # aliasing is fine with the flag on the extra entries (or on all of them), as in the C reader
    for name, blob in (("alias", _slab(raw, edir_off, 7, off=off[5], flags=FLAG_ALIASED)),
                       ("alias_all", _slab(_slab(raw, edir_off, 7, off=off[5], flags=FLAG_ALIASED), edir_off, 5,
                                           flags=FLAG_ALIASED)),
                       ("flag_alone", _slab(raw, edir_off, 9, flags=FLAG_ALIASED))):
        p = tmp_path / f"{name}.hearth"
        p.write_bytes(blob)
        with ContainerReader(p) as r:
            assert r.expert_entry(1, 3).offset == off[5] or name == "flag_alone"
            for a, b in zip(r.read_expert(1, 3, dequant=False), r.read_expert(1, 1, dequant=False)):
                assert (a == b) == (name != "flag_alone")


def test_expert_entry_counts(tmp_path):
    """n_expert_entries is 0 or n_layers*n_experts when no layer is MoE, exactly the latter otherwise."""
    p = tmp_path / "d.hearth"
    with ContainerWriter(p, dict(MOE_META, layer_kind=[0, 0, 0])) as w:     # experts declared, no MoE layer
        w.declare_tensor("t", (64,), F32)
        w.declare_experts(3, 4, 64, 64, F32, [])
        w.write_tensor("t", np.zeros(64, np.float32))
    raw = p.read_bytes()
    assert struct.unpack_from("<Q", raw, 48)[0] == 12
    for n, ok in ((12, True), (0, True), (5, False), (13, False)):
        p.write_bytes(_patch(raw, 48, "<Q", n))
        if ok:
            ContainerReader(p).close()
        else:
            with pytest.raises(ValueError, match=f"{n} expert entries for a model without routed experts"):
                ContainerReader(p)
    with ContainerWriter(p, BASE_META) as w:                               # n_experts = 0
        w.declare_tensor("t", (64,), F32)
        w.write_tensor("t", np.zeros(64, np.float32))
    raw = p.read_bytes()
    p.write_bytes(_patch(raw, 40, "<Q", len(raw)))                         # an empty directory may sit at EOF
    ContainerReader(p).close()
    p.write_bytes(_patch(raw, 40, "<Q", len(raw) + 1))
    with pytest.raises(ValueError, match="expert directory .* outside"):
        ContainerReader(p)
    p.write_bytes(_patch(_patch(raw, 48, "<Q", 5), 40, "<Q", 64))
    with pytest.raises(ValueError, match="5 expert entries for a model without routed experts"):
        ContainerReader(p)


def test_reader_caps(tmp_path, monkeypatch):
    import hearth.format as hf
    assert (hf.TENSORS_MAX, hf.META_MAX) == (1 << 20, 1 << 30)            # engine/src/modelfile.c
    p = tmp_path / "c.hearth"
    _container(p, np.random.default_rng(3))                                # 7 tensors
    meta_bytes = struct.unpack_from("<Q", p.read_bytes(), 16)[0]
    monkeypatch.setattr(hf, "TENSORS_MAX", 7)
    monkeypatch.setattr(hf, "META_MAX", meta_bytes)
    ContainerReader(p).close()
    monkeypatch.setattr(hf, "TENSORS_MAX", 6)
    with pytest.raises(ValueError, match="tensor count 7"):
        ContainerReader(p)
    monkeypatch.setattr(hf, "TENSORS_MAX", 7)
    monkeypatch.setattr(hf, "META_MAX", meta_bytes - 1)
    with pytest.raises(ValueError, match="implausibly large"):
        ContainerReader(p)


def _entry(key: str, t: int, payload: bytes) -> bytes:
    kb = key.encode("ascii")
    return struct.pack("<H", len(kb)) + kb + bytes([t]) + payload


def _arr(fmt: str, vals) -> bytes:
    return struct.pack("<I", len(vals)) + struct.pack(f"<{len(vals)}{fmt}", *vals)


def _forge(path, monkeypatch, meta_bytes: bytes):
    """A container with exactly this metadata section (the writer's own checks bypassed)."""
    import hearth.format as hf
    with monkeypatch.context() as mp:
        mp.setattr(hf, "encode_meta", lambda meta: meta_bytes)
        mp.setattr(hf.ContainerWriter, "_check_consistency", lambda self: None)
        with ContainerWriter(path, {}) as w:
            w.declare_tensor("t", (4,), F32)
            w.write_tensor("t", np.zeros(4, np.float32))
    return path


U, FL, U64, S, UA, FA, BA = 1, 2, 3, 4, 5, 6, 7
_GOOD = b"".join(_entry(k, U, struct.pack("<I", v)) for k, v in
                 (("n_layers", 2), ("d_model", 64), ("vocab_size", 96), ("n_heads", 4), ("dense_ffn_dim", 64)))


def _e(key, v, t=U):
    """One metadata entry: u32 by default; f32 for floats; arrays as (fmt, values)."""
    if t == U:
        return _entry(key, U, struct.pack("<I", v))
    if t == FL:
        return _entry(key, FL, struct.pack("<f", v))
    if t == U64:
        return _entry(key, U64, struct.pack("<Q", v))
    if t == S:
        b = v.encode("utf-8")
        return _entry(key, S, struct.pack("<I", len(b)) + b)
    return _entry(key, t, _arr({UA: "I", FA: "f", BA: "B"}[t], v))


# FORMAT.md §3.1 types and the C reader's ranges/consistency rules (engine/src/modelfile.c build_config)
BAD_META = {
    "n_heads_u32a": ([_e("n_heads", [4], UA)], "type u32\\[\\], expected u32"),
    "attn_kind_u32a": ([_e("attn_kind", [0], UA)], "has type"),
    "layer_kind_u32": ([_e("layer_kind", 0)], "has type u32, expected u8"),
    "layer_kind_u32a": ([_e("layer_kind", [0, 0], UA)], "has type"),
    "eos_ids_u32": ([_e("eos_ids", 5)], "has type u32, expected u32\\[\\]"),
    "n_heads_f32": ([_e("n_heads", 4.0, FL)], "has type f32, expected u32"),
    "norm_eps_u32": ([_e("norm_eps", 0)], "has type"),
    "arch_u32": ([_e("arch", 1)], "has type"),
    "tokenizer_u8a": ([_e("tokenizer", [65], BA)], "has type"),
    "bos_u64": ([_e("bos_id", 1, U64)], "has type"),
    "n_layers_0": ([_e("n_layers", 0)], "n_layers = 0 out of range"),
    "n_layers_513": ([_e("n_layers", 513)], "n_layers = 513 out of range"),
    "d_model_huge": ([_e("d_model", (1 << 20) + 1)], "d_model"),
    "vocab_huge": ([_e("vocab_size", (1 << 27) + 1)], "vocab_size"),
    "max_seq_0": ([_e("max_seq", 0)], "max_seq"),
    "n_heads_0": ([_e("n_heads", 0)], "n_heads = 0"),
    "n_kv_heads_0": ([_e("n_kv_heads", 0)], "n_kv_heads = 0"),
    "kv_not_divisor": ([_e("n_kv_heads", 3)], "not a multiple of n_kv_heads"),
    "head_dim_0": ([_e("head_dim", 0)], "head_dim is 0"),
    "head_dim_default_0": ([_e("n_heads", 128)], "head_dim is 0"),
    "attn_kind_2": ([_e("attn_kind", 2)], "attn_kind"),
    "qk_norm_3": ([_e("qk_norm", 3)], "qk_norm"),
    "qkv_bias_2": ([_e("qkv_bias", 2)], "qkv_bias"),
    "rope_dim_odd": ([_e("rope_dim", 3)], "odd"),
    "rope_dim_gt_head": ([_e("rope_dim", 18)], "rope_dim = 18 out of range"),
    "rope_style_2": ([_e("rope_style", 2)], "rope_style"),
    "norm_eps_neg": ([_e("norm_eps", -1.0, FL)], "negative"),
    "norm_eps_nan": ([_e("norm_eps", float("nan"), FL)], "not finite"),
    "attn_scale_inf": ([_e("attn_scale", float("inf"), FL)], "not finite"),
    "rope_factor_nan": ([_e("rope_attn_factor", float("nan"), FL)], "not finite"),
    "routed_scale_inf": ([_e("routed_scale", float("-inf"), FL)], "not finite"),
    "mla_no_latent": ([_e("attn_kind", 1), _e("qk_rope_dim", 16), _e("v_head_dim", 16)], "MLA needs"),
    "mla_no_v": ([_e("attn_kind", 1), _e("kv_lora_rank", 64), _e("qk_rope_dim", 16)], "MLA needs"),
    "mla_no_qk": ([_e("attn_kind", 1), _e("kv_lora_rank", 64), _e("v_head_dim", 16)], "MLA needs"),
    "moe_no_experts": ([_e("layer_kind", [1, 1], BA)], "n_experts is 0"),
    "top_k_0": ([_e("n_experts", 4), _e("expert_ffn_dim", 64)], "top_k 0 out of range"),
    "top_k_gt_e": ([_e("n_experts", 4), _e("top_k", 5), _e("expert_ffn_dim", 64)], "top_k 5 out of range"),
    "expert_ffn_0": ([_e("n_experts", 4), _e("top_k", 2)], "expert_ffn_dim"),
    "group_divides": ([_e("n_experts", 6), _e("top_k", 2), _e("expert_ffn_dim", 64), _e("n_group", 4)],
                      "not a multiple of n_group"),
    "topk_group_gt": ([_e("n_experts", 8), _e("top_k", 2), _e("expert_ffn_dim", 64), _e("n_group", 4),
                       _e("topk_group", 5)], "topk_group 5 > n_group 4"),
    "top_k_gt_groups": ([_e("n_experts", 8), _e("top_k", 5), _e("expert_ffn_dim", 64), _e("n_group", 4),
                         _e("topk_group", 2)], "exceeds the experts"),
    "n_group_0": ([_e("n_group", 0)], "n_group = 0"),
    "topk_group_0": ([_e("topk_group", 0)], "topk_group = 0"),
    "flags_2": ([_e("shared_gate", 2)], "shared_gate"),
    "score_fn_2": ([_e("score_fn", 2)], "score_fn"),
    "score_bias_2": ([_e("score_bias", 2)], "score_bias"),
    "norm_topk_2": ([_e("norm_topk_prob", 2)], "norm_topk_prob"),
    "tie_2": ([_e("tie_embeddings", 2)], "tie_embeddings"),
    "layer_kind_2": ([_e("layer_kind", [0, 2], BA)], "0 or 1"),
    "layer_kind_count": ([_e("layer_kind", [0], BA)], "1 entries for 2 layers"),
    "dense_ffn_0": ([_e("dense_ffn_dim", 0)], "dense_ffn_dim"),
    "dense_ffn_huge": ([_e("dense_ffn_dim", (1 << 22) + 1)], "dense_ffn_dim = 4194305 out of range"),
    "shared_ffn_huge": ([_e("shared_ffn_dim", (1 << 22) + 1)], "shared_ffn_dim = 4194305 out of range"),
    "expert_ffn_huge": ([_e("expert_ffn_dim", (1 << 22) + 1)], "expert_ffn_dim = 4194305 out of range"),
    "n_experts_huge": ([_e("n_experts", 65537)], "n_experts = 65537 out of range"),
    "kv_lora_huge": ([_e("kv_lora_rank", (1 << 20) + 1)], "kv_lora_rank"),
    "q_lora_huge": ([_e("q_lora_rank", (1 << 20) + 1)], "q_lora_rank"),
    "mla_rope_gt_qk_rope": ([_e("attn_kind", 1), _e("kv_lora_rank", 64), _e("qk_rope_dim", 16), _e("v_head_dim", 16),
                             _e("rope_dim", 18)], "rope_dim = 18 out of range .0, 16."),
    "bos_ge_vocab": ([_e("bos_id", 96)], "bos_id 96"),
    "eos_ge_vocab": ([_e("eos_ids", [5, 96], UA)], "eos_ids\\[1\\] = 96"),
    "tokenizer_long": ([_e("tokenizer", "t" * 260, S)], "tokenizer path"),
    "tokenizer_nul": ([_e("tokenizer", "a\0b", S)], "tokenizer path"),
}


@pytest.mark.parametrize("case", sorted(BAD_META))
def test_metadata_types_and_ranges_are_validated(tmp_path, monkeypatch, case):
    entries, msg = BAD_META[case]
    p = _forge(tmp_path / "m.hearth", monkeypatch, _GOOD + b"".join(entries))
    with pytest.raises(ValueError, match=msg):
        ContainerReader(p).close()


def test_metadata_validation_accepts_valid_variants(tmp_path, monkeypatch):
    good = [[], [_e("bos_id", 0xFFFFFFFF)], [_e("bos_id", 95), _e("eos_ids", [0, 95], UA)],
            [_e("n_kv_heads", 2), _e("rope_dim", 16), _e("rope_style", 1)], [_e("n_kv_heads", 1)],
            [_e("x_custom", [1.5], FA)],
            [_e("d_model", 64)], [_e("n_layers", 512), _e("layer_kind", [0] * 512, BA)],
            [_e("attn_kind", 1), _e("kv_lora_rank", 64), _e("qk_rope_dim", 16), _e("v_head_dim", 16),
             _e("rope_dim", 16)], [_e("tokenizer", "t" * 259, S)], [_e("norm_eps", 0.0, FL)],
            [_e("attn_kind", 1), _e("kv_lora_rank", 1), _e("qk_nope_dim", 1), _e("v_head_dim", 1)],   # MLA minimum
            [_e("attn_kind", 1), _e("kv_lora_rank", 64), _e("qk_rope_dim", 16), _e("v_head_dim", 16),
             _e("rope_dim", 0)], [_e("rope_dim", 0)], [_e("shared_ffn_dim", 1 << 22), _e("dense_ffn_dim", 1 << 22)],
            [_e("n_experts", 65536), _e("top_k", 1), _e("expert_ffn_dim", 1 << 22), _e("n_group", 65536),
             _e("layer_kind", [0, 0], BA)]]
    for extra in good:
        p = _forge(tmp_path / "m.hearth", monkeypatch, _GOOD + b"".join(extra))
        with ContainerReader(p) as r:
            assert r.config["n_heads"] == 4 and isinstance(r.config["n_heads"], int)
    for bos, want in ((0, 0), (95, 95), (None, 0xFFFFFFFF), (-1, 0xFFFFFFFF)):   # negative = "no BOS"
        p = tmp_path / "bos.hearth"
        with ContainerWriter(p, dict(BASE_META, bos_id=bos)) as w:
            w.declare_tensor("t", (1,), F32)
            w.write_tensor("t", np.zeros(1, np.float32))
        assert ContainerReader(p).config["bos_id"] == want
    # the writer refuses the same mistakes before writing anything
    for bad in ({"n_kv_heads": 0}, {"n_kv_heads": 3}, {"bos_id": 96}, {"eos_ids": [96]}, {"rope_dim": 3},
                {"norm_eps": float("nan")}, {"n_heads": 4.0}, {"layer_kind": 5}, {"arch": 5},
                {"tokenizer": "a\0b"}, {"n_layers": 513}, {"attn_scale": 1e39}, {"norm_eps": True},
                {"routed_scale": "2.5"}):
        p = tmp_path / "w.hearth"
        meta = dict(BASE_META, n_heads=4)
        meta.update(bad)
        with pytest.raises(ValueError):
            with ContainerWriter(p, meta) as w:
                w.declare_tensor("t", (1,), F32)
                w.write_tensor("t", np.zeros(1, np.float32))
        assert not p.exists()


def test_rope_inv_freq_shape_checked_when_present(tmp_path):
    p = tmp_path / "r.hearth"
    for shape, dt, ok in (((16,), F32, True), ((15,), F32, False), ((16,), F16, False), ((1, 16), F32, False),
                          (None, None, True)):
        with ContainerWriter(p, BASE_META) as w:            # rope_dim = head_dim = 32 -> [16]
            w.declare_tensor("t", (1,), F32)
            if shape:
                w.declare_tensor("rope_inv_freq", shape, dt)
            for n, info in w._tensors.items():
                w.write_tensor(n, np.ones(info.shape, np.float32))
        if ok:
            ContainerReader(p).close()
        else:
            with pytest.raises(ValueError, match="rope_inv_freq"):
                ContainerReader(p)
    with ContainerWriter(p, dict(BASE_META, rope_dim=0)) as w:     # rope_dim 0: an inv_freq is unused, not checked
        w.declare_tensor("rope_inv_freq", (3,), F32)
        w.write_tensor("rope_inv_freq", np.ones(3, np.float32))
    ContainerReader(p).close()


def test_huge_layer_count_is_rejected_cheaply(tmp_path, monkeypatch):
    """A tiny forged file must not make the reader build O(n_layers) lists before validating."""
    import time
    import tracemalloc
    for n in (10 ** 8, 0xFFFFFFFF):
        p = _forge(tmp_path / "h.hearth", monkeypatch, _GOOD + _e("n_layers", n) + _e("n_experts", 8))
        tracemalloc.start()
        t0 = time.perf_counter()
        with pytest.raises(ValueError, match="n_layers"):
            ContainerReader(p)
        dt = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert peak < (4 << 20) and dt < 1.0, (n, peak, dt)


def test_missing_required_meta_and_bad_layer_kind(tmp_path, monkeypatch):
    p = tmp_path / "m.hearth"
    bad = [dict(BASE_META, layer_kind=[1, 1])] + [{k: v for k, v in BASE_META.items() if k != req}
                                                  for req in ("n_layers", "d_model", "vocab_size", "n_heads")]
    for meta in bad:                                # the writer refuses all of them before writing anything
        with pytest.raises(ValueError, match="layer_kind has 2 entries|missing required key"):
            with ContainerWriter(p, meta) as w:
                w.declare_tensor("t", (1,), F32)
                w.write_tensor("t", np.zeros(1, np.float32))
        assert not p.exists()
        with pytest.raises(ValueError, match="layer_kind has 2 entries|missing required key"):
            w = ContainerWriter(p, meta)
            w.declare_tensor("t", (1,), F32)
            w.close()                               # closing without any write validates too
        assert not p.exists()
    import hearth.format as hf
    monkeypatch.setattr(hf.ContainerWriter, "_check_consistency", lambda self: None)
    for meta in bad:                                # forged files: the reader refuses them as well
        with ContainerWriter(p, meta) as w:
            w.declare_tensor("t", (1,), F32)
            w.write_tensor("t", np.zeros(1, np.float32))
        with pytest.raises(ValueError, match="layer_kind has 2 entries|missing required key"):
            ContainerReader(p)


def test_writer_byte_buffers_are_sized_in_bytes(tmp_path):
    """bytes, bytearray and memoryviews of any item type count bytes, never items."""
    p = tmp_path / "b.hearth"
    with ContainerWriter(p, dict(MOE_META, layer_kind=[0, 1, 1])) as w:
        w.declare_tensor("a", (64,), U8)
        w.declare_tensor("b", (64,), U8)
        w.declare_tensor("m", (2, 64), U8)
        w.declare_tensor("f", (4, 16), F32)
        w.declare_tensor("r", (6, 64), F16)
        w.declare_tensor("i", (3, 2), I32)
        w.declare_tensor("u", (2, 3), U8)
        w.declare_tensor("q4d", (1, 2, 3, 4), F32)
        w.declare_experts(3, 4, 64, 64, F32, [1, 2])
        w.write_tensor("i", np.array([[-5, 7], [1 << 30, 0], [3, -1]], np.int32))
        w.write_tensor("u", np.arange(6, dtype=np.uint8).reshape(2, 3))
        w.write_tensor("q4d", np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4))
        w.write_tensor("b", np.arange(64, dtype=np.uint8))
        with pytest.raises(ValueError, match="got 256 bytes, expected 64"):
            w.write_tensor("a", memoryview(np.full(64, 7, np.float32)))     # 64 items, 256 bytes
        with pytest.raises(ValueError, match="not C-contiguous"):
            w.write_tensor("a", memoryview(np.zeros((64, 2), np.uint8)[:, 0]))
        w.write_tensor("a", memoryview(np.full(16, 7, np.float32)))         # 16 items, 64 bytes
        w.write_tensor("m", memoryview(np.arange(128, dtype=np.uint8).reshape(2, 64)))
        f = np.arange(64, dtype=np.float32)
        w.write_tensor("f", memoryview(f.reshape(4, 16)))
        r = quant.quantize(np.ones((6, 64), np.float32), F16)
        w.write_tensor_rows("r", 0, memoryview(np.frombuffer(r, np.float16)[:2 * 64]))   # 2 rows as f16 items
        with pytest.raises(ValueError, match="bad chunk"):
            w.write_tensor_rows("r", 2, memoryview(np.zeros(16, np.float16)))          # 32 bytes: no whole row
        w.write_tensor_rows("r", 2, bytearray(r[2 * 128:]))
        mats = [np.full((64, 64), v, np.float32) for v in (1, 2, 3)]
        w.write_expert(1, 0, *(memoryview(m) for m in mats))
        for li in (1, 2):
            for e in range(4):
                if (li, e) != (1, 0):
                    w.write_expert(li, e, *(bytes(m) for m in mats))
    with ContainerReader(p) as r_:
        assert r_.read_tensor("b").tolist() == list(range(64))              # untouched by the rejected write
        assert (r_.read_tensor("a", dequant=False) == np.full(16, 7, np.float32).tobytes())
        assert r_.read_tensor("m").tolist() == np.arange(128).reshape(2, 64).tolist()
        assert r_.read_tensor("i").tolist() == [[-5, 7], [1 << 30, 0], [3, -1]]
        assert r_.read_tensor("u").tolist() == [[0, 1, 2], [3, 4, 5]]
        np.testing.assert_array_equal(r_.read_tensor("q4d"), np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4))
        np.testing.assert_array_equal(r_.read_tensor("f"), f.reshape(4, 16))
        np.testing.assert_array_equal(r_.read_tensor("r"), np.ones((6, 64), np.float32))
        for g, m in zip(r_.read_expert(1, 0), mats):
            np.testing.assert_array_equal(g, m)
    assert p.stat().st_size == ContainerReader(p).expert_entry(2, 3).offset + slab_layout(F32, 64, 64)[3]


# ---------------------------------------------------------------- benchmark-shaped containers

TOY_MLA = dict(name="toy-mla", arch="deepseek_v3", n_layers=3, d_model=128, vocab=300, n_experts=16, top_k=2,
               expert_ffn=64, n_dense_layers=1, dense_ffn=128, shared_ffn=64, attn="mla", n_heads=2,
               q_lora_rank=64, kv_lora_rank=64, qk_nope_dim=32, qk_rope_dim=16, v_head_dim=32)
TOY_GQA = dict(name="toy-gqa", arch="qwen2_moe", n_layers=2, d_model=128, vocab=256, n_experts=8, top_k=2,
               expert_ffn=128, shared_ffn=128, n_heads=4, n_kv_heads=2, head_dim=32)


@pytest.mark.parametrize("preset", [TOY_MLA, TOY_GQA])
@pytest.mark.parametrize("expert_dtype", [Q4, Q8, F16])
def test_make_shaped(tmp_path, preset, expert_dtype):
    from hearth import synth
    from hearth.presets import get
    from hearth.reference import Reference
    s = get(preset)
    full = synth.make_shaped(tmp_path / "full.hearth", preset=preset, expert_dtype=expert_dtype)
    small = synth.make_shaped(tmp_path / "small.hearth", preset=preset, expert_dtype=expert_dtype, physical_experts=3)
    assert small.stat().st_size < full.stat().st_size
    assert abs(synth.shaped_bytes(preset, expert_dtype, 3) - small.stat().st_size) < 0.1 * small.stat().st_size
    with ContainerReader(small) as r:
        c = r.config
        assert (c["n_layers"], c["d_model"], c["vocab_size"], c["n_experts"]) == (s.n_layers, s.d_model, s.vocab, s.n_experts)
        assert c["layer_kind"] == [0] * s.n_dense_layers + [1] * s.n_moe_layers
        assert c["source"].startswith("synthetic")
        for li in range(s.n_dense_layers, s.n_layers):
            for e in range(s.n_experts):
                ent = r.expert_entry(li, e)
                assert ent.dtype == expert_dtype
                if e >= 3:
                    assert ent.flags & FLAG_ALIASED and ent.offset == r.expert_entry(li, e % 3).offset
        if s.attn == "mla":
            assert r.tensors["blk.1.attn_kv_b"].shape == (2 * (32 + 32), 64) and c["attn_kind"] == 1
        else:
            assert c["qkv_bias"] == 1 and c["shared_gate"] == 1
        assert r.tensors["tok_embd"].dtype == Q8 and r.tensors["blk.1.moe_router"].dtype == F32
    logits = Reference(small).eval([1, 2, 3])
    assert np.isfinite(logits).all() and logits.std() > 0


@pytest.mark.parametrize("preset,want", [
    ("qwen3-30b-a3b", dict(qk_norm=1, norm_topk_prob=1, n_kv_heads=4, head_dim=128, attn_scale=128 ** -0.5)),
    ("olmoe-1b-7b", dict(qk_norm=2, n_kv_heads=16)),
    ("mixtral-8x22b", dict(norm_topk_prob=1, n_kv_heads=8)),
    ("qwen1.5-moe-a2.7b", dict(qkv_bias=1, shared_gate=1, shared_ffn_dim=5632)),
    ("deepseek-v3", dict(attn_kind=1, score_fn=1, score_bias=1, n_group=8, topk_group=4, routed_scale=2.5,
                         norm_topk_prob=1, rope_style=1, attn_scale=192 ** -0.5, dense_ffn_dim=18432)),
    ("kimi-k2", dict(attn_kind=1, n_group=1, topk_group=1, routed_scale=2.827, n_experts=384)),
])
def test_shaped_meta_per_arch(preset, want):
    from hearth import presets, synth
    s = presets.get(preset)
    m = synth._shaped_meta(s, Q4, 4096)
    for k, v in want.items():
        assert m[k] == pytest.approx(v), k
    assert m["layer_kind"] == [0] * s.n_dense_layers + [1] * s.n_moe_layers and m["expert_dtype"] == Q4


def test_shaped_random_rows_are_centred():
    from hearth import synth
    rng = np.random.default_rng(0)
    for dt in (Q4, Q8, F16):
        rows = synth._random_rows(rng, dt, 256, 64)
        w = quant.dequantize(rows.tobytes(), dt, (64, 256))
        assert (w < 0).mean() > 0.3 and (w > 0).mean() > 0.3 and abs(w.mean()) < 0.01
        assert 0.5 / 16 < w.std() < 2.0 / 16                     # ~ 1/sqrt(cols)


def test_make_shaped_rejects_unsupported_presets(tmp_path):
    from hearth import synth
    for name in ("glm-4.5", "gpt-oss-120b", "kimi-k3", "glm-5.2"):
        with pytest.raises(ValueError, match="arch|latent"):
            synth.make_shaped(tmp_path / "x.hearth", preset=name)
    with pytest.raises(ValueError):
        synth.make_shaped(tmp_path / "x.hearth", preset=TOY_GQA, physical_experts=0)
    with pytest.raises(ValueError):
        synth.make_shaped(tmp_path / "x.hearth", preset=TOY_GQA, physical_experts=9)
    for bad in (dict(top_k=0), dict(n_experts=0)):
        with pytest.raises(ValueError, match="no routed experts"):
            synth.make_shaped(tmp_path / "x.hearth", preset=dict(TOY_GQA, **bad))
    assert not (tmp_path / "x.hearth").exists()
    for P in (1, 8):                                           # the extremes: one slab per layer, no aliasing
        p = synth.make_shaped(tmp_path / f"p{P}.hearth", preset=TOY_GQA, expert_dtype=F16, physical_experts=P)
        with ContainerReader(p) as r:
            assert [bool(r.expert_entry(1, e).flags) for e in range(8)] == [e >= P for e in range(8)]


# ---------------------------------------------------------------- native library loader and bindings

def test_data_dir(monkeypatch, tmp_path):
    import os
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path))
    assert _native.data_dir() == tmp_path
    monkeypatch.delenv("HEARTH_DATA")
    if os.name == "nt":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
        assert _native.data_dir() == tmp_path / "lad" / "hearth"
    else:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert _native.data_dir() == tmp_path / "xdg" / "hearth"


def test_library_search_order(monkeypatch, tmp_path):
    import os
    name = _native._NAMES[0]
    monkeypatch.delenv("HEARTH_LIB", raising=False)
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path))
    old, new = tmp_path / "build" / "old" / name, tmp_path / "build" / "x" / "y" / name
    for p in (old, new):
        p.parent.mkdir(parents=True)
        p.write_bytes(b"not a library")
    os.utime(old, (1e9, 1e9))
    os.utime(new, (2e9, 2e9))
    assert [p for p, explicit in _native._candidates() if not explicit] == [new, old]     # newest first
    monkeypatch.setenv("HEARTH_LIB", str(old))
    assert _native._candidates() == [(old, True)]
    monkeypatch.setenv("HEARTH_LIB", str(old.parent))
    assert _native._candidates() == [(old, True)]
    monkeypatch.setenv("HEARTH_LIB", str(tmp_path / "missing"))
    with pytest.raises(_native.HearthLibNotFound):
        _native._candidates()
    monkeypatch.setattr(_native, "_lib", None)
    monkeypatch.setattr(_native, "_lib_path", None)
    monkeypatch.setenv("HEARTH_LIB", str(old))                    # exists but cannot be loaded
    assert not _native.available() and _native.lib_path() is None
    with pytest.raises(_native.HearthLibNotFound, match="not loadable"):
        _native.lib()


def test_loader_prefers_full_engine_builds(monkeypatch, tmp_path):
    import ctypes
    from types import SimpleNamespace

    def fake(full):
        ns = SimpleNamespace(hearth_quantize=lambda *a: 0, hearth_row_bytes=lambda *a: 0)
        if full:
            ns.hearth_open = lambda *a: 0
        return ns

    kern, full, nokern = fake(False), fake(True), SimpleNamespace(hearth_open=lambda *a: 0)
    k2, k3 = fake(False), fake(False)
    libs = {tmp_path / "k": kern, tmp_path / "f": full, tmp_path / "n": nokern, tmp_path / "bad": None,
            tmp_path / "k2": k2, tmp_path / "k3": k3}
    names = {id(v): k.name for k, v in libs.items()}
    released = []
    monkeypatch.setattr(_native, "_try_load", lambda p: libs[p])
    monkeypatch.setattr(_native, "_release", lambda L: released.append(names[id(L)]))
    for cands, want, freed in (
            ([("k", False), ("f", False)], "f", ["k"]),               # full engine beats a newer kernels build
            ([("bad", False), ("k", False)], "k", []),                # kernels-only build is the fallback
            ([("k", True), ("f", False)], "k", []),                   # an explicit choice wins
            ([("n", False), ("k", False)], "k", ["n"]),               # no hearth_quantize: skipped
            ([("k", False), ("k2", False), ("k3", False)], "k", ["k2", "k3"]),
            ([("k", False), ("k2", False), ("f", False)], "f", ["k2", "k"])):
        monkeypatch.setattr(_native, "_lib", None)
        monkeypatch.setattr(_native, "_lib_path", None)
        monkeypatch.setattr(_native, "_candidates", lambda c=cands: [(tmp_path / p, e) for p, e in c])
        released.clear()
        assert _native.lib() is libs[tmp_path / want]                 # the first call loads and returns it
        assert released == freed, cands                               # everything inspected but not chosen is unloaded
        assert _native.lib_path() == tmp_path / want and _native.available()
        monkeypatch.setattr(_native, "_lib", None)
        monkeypatch.setattr(_native, "_lib_path", None)
        assert _native.lib_path() == tmp_path / want                  # lib_path() loads on demand
    assert kern.hearth_row_bytes.restype is ctypes.c_size_t           # prototypes were declared
    import sys
    want_name = {"win32": "hearth.dll", "darwin": "libhearth.dylib"}.get(sys.platform, "libhearth.so")
    assert _native._NAMES[0] == want_name


def test_loader_edge_cases(monkeypatch, tmp_path):
    import contextlib
    import os
    import sys
    from pathlib import Path
    junk = tmp_path / _native._NAMES[0]
    junk.write_bytes(b"MZ not really a library")
    assert _native._try_load(junk) is None and _native._try_load(tmp_path / "missing.dll") is None
    sysdll = {"win32": Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "kernel32.dll"}.get(sys.platform)
    if sysdll is None:
        import ctypes.util
        sysdll = ctypes.util.find_library("c")
    if sysdll is not None:
        assert _native._try_load(sysdll) is not None                 # a loadable library loads
    monkeypatch.setattr(_native, "_lib", None)
    monkeypatch.setattr(_native, "_lib_path", None)
    monkeypatch.setattr(_native, "_candidates", lambda: [])
    with pytest.raises(_native.HearthLibNotFound, match="no candidates"):
        _native.lib()
    sentinel = object()

    @contextlib.contextmanager
    def raced():                                    # another thread finished loading while we waited
        _native._lib = sentinel
        yield

    monkeypatch.setattr(_native, "_lock", raced())
    monkeypatch.setattr(_native, "_candidates", lambda: pytest.fail("must not scan again"))
    assert _native.lib() is sentinel


def test_matmul_rejects_unknown_isa():
    W = np.zeros(2 * 64 * 4, np.uint8)
    for isa in (4, 7, -1, 99, "avx2", None):        # checked before the library is even needed
        with pytest.raises(ValueError, match="isa must be"):
            _native.matmul(F32, W, 2, 64, np.ones((1, 64), np.float32), isa=isa)


@needs_native
def test_unchosen_libraries_are_unloaded(monkeypatch, tmp_path):
    """Scanning <data dir>/build must not keep the candidates it rejected locked (Windows
    cannot overwrite or delete a loaded DLL, which would break concurrent rebuilds)."""
    import os
    import shutil
    src = _native.lib_path()
    if hasattr(_native.lib(), "hearth_open"):
        pytest.skip("needs a kernels-only build: a full engine build is chosen without scanning further")
    monkeypatch.delenv("HEARTH_LIB", raising=False)
    monkeypatch.setenv("HEARTH_DATA", str(tmp_path))
    copies = []
    for i, sub in enumerate(("old", "mid", "new")):
        p = tmp_path / "build" / sub / _native._NAMES[0]
        p.parent.mkdir(parents=True)
        shutil.copyfile(src, p)
        os.utime(p, (1e9 + i, 1e9 + i))
        copies.append(p)
    monkeypatch.setattr(_native, "_lib", None)
    monkeypatch.setattr(_native, "_lib_path", None)
    assert _native.lib_path() == copies[2]                    # newest kernels build; all three were inspected
    chosen = _native.lib()
    copies[0].unlink()                                        # rejected candidates are not locked
    copies[1].unlink()
    _native._release(chosen)                                  # (the chosen one is; release it for cleanup)
    copies[2].unlink()


@needs_native
def test_native_bindings_validate(monkeypatch):
    W = quant.quantize(np.ones((3, 64), np.float32), Q8)
    X = np.ones((2, 64), np.float32)
    assert _native.matmul(Q8, W, 3, 64, X).shape == (2, 3)
    assert _native.matmul(Q8, np.frombuffer(W, np.uint8), 3, 64, X[0]).shape == (1, 3)
    for bad in ((Q8, W, 3, 64, np.ones((2, 32), np.float32)), (Q8, W[:-1], 3, 64, X),
                (Q8, b"", 1, 100, np.ones((1, 100), np.float32))):
        with pytest.raises(ValueError):
            _native.matmul(*bad)
    with pytest.raises(TypeError):
        _native.matmul(Q8, [1, 2], 3, 64, X)
    np.testing.assert_array_equal(_native.dequantize(Q8, W, 3, 64), quant.dequantize(W, Q8, (3, 64)))
    with pytest.raises(ValueError):
        _native.dequantize(Q8, W[:-1], 3, 64)
    for dt, n in ((F32, 10), (F16, 10), (BF16, 3), (Q8, 128), (Q4, 128)):
        assert _native.row_bytes(dt, n) == quant.row_bytes(dt, n)
    assert _native.version().startswith("0.") and _native.cpu_isa() in (1, 2, 3)
    for x, out in ((np.ones((1, 64), np.float32), np.empty(10, np.uint8)),
                   (np.ones(64, np.float32), np.empty(66, np.uint8)),
                   (np.ones((1, 100), np.float32), np.empty(0, np.uint8))):
        with pytest.raises(ValueError):
            _native.quantize_into(Q8, x, out)
    f = np.ones((2, 8), np.float32)
    for isa in (0, 1, 2, 3):                        # every valid id runs (or reports the CPU cannot)
        Y = _native.matmul(F32, f, 2, 8, np.ones((1, 8), np.float32), isa=isa)
        assert Y is None or np.array_equal(Y, np.full((1, 2), 8.0, np.float32))
    assert _native.matmul(F32, f, 2, 8, np.ones((1, 8), np.float32), isa=1) is not None   # scalar always runs
    monkeypatch.setattr(_native, "cpu_isa", lambda: 1)
    assert _native.matmul(F32, f, 2, 8, np.ones((1, 8), np.float32), isa=3) is None
    real = _native._fn
    for rc in (-2, -1, -3):     # only -2 (ISA unavailable) means "cannot run"; other failures must surface
        monkeypatch.setattr(_native, "_fn", lambda name, rc=rc: (lambda *a: rc) if name != "hearth_row_bytes" else real(name))
        if rc == -2:
            assert _native.matmul(F32, f, 2, 8, np.ones((1, 8), np.float32), isa=1) is None
        else:
            with pytest.raises(RuntimeError, match=f"failed with {rc}"):
                _native.matmul(F32, f, 2, 8, np.ones((1, 8), np.float32), isa=1)
        with pytest.raises(RuntimeError):
            _native.matmul(F32, f, 2, 8, np.ones((1, 8), np.float32))
    with pytest.raises(RuntimeError):
        _native.dequantize(Q8, W, 3, 64)
    with pytest.raises(RuntimeError):
        _native.quantize_into(Q8, np.ones((1, 64), np.float32), np.empty(66, np.uint8))
