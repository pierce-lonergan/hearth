"""hearth.convert: FP8 block dequantization, safetensors backends, name mapping, metadata/RoPE,
companion files, and quantized conversion quality."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from hearth import _native, quant
from hearth.convert import (HFCheckpoint, _ignorable, _rope_fallback, convert, expert_names, fp8_block_dequant,
                            fp8_e4m3_to_f32, hearth_meta, rope_params, tensor_plan)
from hearth.format import ContainerReader
from hearth.reference import Reference

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_reference import TOKENS, build_hf_model, hf_cases, hf_logits_and_routes  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
needs_native = pytest.mark.skipif(not _native.available(), reason="Hearth native library not available")
ALL = ["qwen3_moe", "olmoe", "mixtral", "qwen2_moe", "deepseek_v3", "deepseek_v3_no_q_lora"]
F32 = quant.F32


def _f32_convert(src, dst, **kw):
    return convert(src, dst, expert_dtype=F32, dense_dtype=F32, embed_dtype=F32, head_dtype=F32, progress=False, **kw)


# ---------------------------------------------------------------- FP8

def test_fp8_e4m3_decode():
    v = fp8_e4m3_to_f32(np.arange(256, dtype=np.uint8))
    assert v[0x00] == 0 and v[0x38] == 1.0 and v[0x7E] == 448.0 and v[0xFE] == -448.0
    assert v[0x01] == 2.0 ** -9 and v[0x08] == 2.0 ** -6 and np.signbit(v[0x80])
    assert np.isnan(v[0x7F]) and np.isnan(v[0xFF]) and np.isfinite(np.delete(v, [0x7F, 0xFF])).all()
    t = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    assert np.array_equal(v, t, equal_nan=True)


def test_fp8_block_dequant_synthetic():
    rng = np.random.default_rng(0)
    R, C, b = 300, 260, 128
    w = fp8_e4m3_to_f32(rng.integers(0, 256, size=(R, C)).astype(np.uint8))
    w[np.isnan(w)] = 1.0
    s = rng.uniform(1e-4, 1e-2, size=(3, 3)).astype(np.float32)
    want = np.empty((R, C), np.float32)
    for r in range(R):
        for c in range(C):
            want[r, c] = np.float32(w[r, c] * s[r // b, c // b])
    got = fp8_block_dequant(w, s, (b, b))
    assert np.array_equal(got, want)
    assert np.array_equal(fp8_block_dequant(w[128:300], s, (b, b), row0=128), want[128:])
    for short in (s[:2], s[:, :2], s[0]):                     # scales that do not cover the tensor
        with pytest.raises(ValueError):
            fp8_block_dequant(w, short, (b, b))
    small = w[:100, :100]                                     # smaller than one block: a 1x1 scale
    assert np.array_equal(fp8_block_dequant(small, s[:1, :1], (b, b)), small * s[0, 0])


def _fp8_checkpoint(model, out: Path, block=32):
    """Save `model` the way DeepSeek ships FP8 weights; returns the dequantized state for comparison."""
    from safetensors.torch import save_file
    sd, deq = {}, {}
    for name, t in model.state_dict().items():
        t = t.detach().float()
        fp8 = t.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name and not name.endswith("mlp.gate.weight")
        if not fp8:
            sd[name] = t.contiguous()
            continue
        R, C = t.shape
        nr, nc = -(-R // block), -(-C // block)
        pad = torch.zeros(nr * block, nc * block)
        pad[:R, :C] = t
        amax = pad.abs().reshape(nr, block, nc, block).amax(dim=(1, 3)).clamp(min=1e-12)
        s = (amax / 448.0).float()
        sfull = s.repeat_interleave(block, 0).repeat_interleave(block, 1)[:R, :C]
        q = (t / sfull).to(torch.float8_e4m3fn)
        sd[name] = q.contiguous()
        sd[name + "_scale_inv"] = s.contiguous()
        deq[name] = q.float() * sfull
    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    cfg = model.config.to_dict()
    cfg["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                                  "weight_block_size": [block, block]}
    (out / "config.json").write_text(json.dumps(cfg))
    return deq


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_fp8_checkpoint_end_to_end(tmp_path, backend, monkeypatch):
    model = build_hf_model("deepseek_v3", seed=3)
    deq = _fp8_checkpoint(model, tmp_path / "fp8")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in deq:
                p.copy_(deq[name])
    want, _ = hf_logits_and_routes(model, TOKENS)
    ck = HFCheckpoint(tmp_path / "fp8", backend=backend)
    name = "model.layers.1.mlp.experts.3.down_proj.weight"
    assert np.array_equal(ck.load(name, block=(32, 32)), deq[name].numpy())
    import hearth.convert as hc
    orig = hc.HFCheckpoint
    monkeypatch.setattr(hc, "HFCheckpoint", lambda src: orig(src, backend=backend))
    got = Reference(_f32_convert(tmp_path / "fp8", tmp_path / "m.hearth")).eval(TOKENS)
    scale = max(1.0, float(np.abs(want).max()))
    assert np.abs(got - want).max() <= 2e-3 * scale


@pytest.mark.parametrize("backend", ["torch", "numpy"])
@pytest.mark.parametrize("dense", ["f32", "q8"])
def test_chunked_tensor_path_matches_whole(tmp_path, monkeypatch, backend, dense):
    """Real models' embeddings, LM heads and big projections are converted in row chunks. With
    tiny thresholds every dense matrix here takes that path, FP8 ones with chunks starting in
    different 32-row scale blocks; the bytes must equal whole-tensor conversion."""
    if dense == "q8" and not _native.available():
        pytest.skip("Hearth native library not available")
    import hearth.convert as hc
    _fp8_checkpoint(build_hf_model("deepseek_v3", seed=3), tmp_path / "fp8", block=32)
    orig = hc.HFCheckpoint
    monkeypatch.setattr(hc, "HFCheckpoint", lambda src: orig(src, backend=backend))
    kw = dict(expert_dtype="f32", dense_dtype=dense, embed_dtype=dense, head_dtype="f32", progress=False)
    whole = convert(tmp_path / "fp8", tmp_path / "whole.hearth", **kw)
    calls = []
    real = hc.ContainerWriter.write_tensor_rows

    def spy(self, name, row0, data):
        calls.append((name, row0))
        return real(self, name, row0, data)

    monkeypatch.setattr(hc.ContainerWriter, "write_tensor_rows", spy)
    monkeypatch.setattr(hc, "_CHUNK_ELEMS", 2048)
    monkeypatch.setattr(hc, "_CHUNK_TARGET", 2048)
    chunked = convert(tmp_path / "fp8", tmp_path / "chunked.hearth", **kw)
    assert chunked.read_bytes() == whole.read_bytes()
    starts = {}
    for name, r0 in calls:
        starts.setdefault(name, []).append(r0)
    assert starts["tok_embd"] == starts["lm_head"] == list(range(0, 320, 32))
    assert starts["blk.1.attn_q_b"] == [0, 32, 64, 96]           # FP8 [128, 64]: four scale-block rows
    calls.clear()
    monkeypatch.setattr(hc, "_CHUNK_ELEMS", 128 * 64)            # strictly larger tensors only
    monkeypatch.setattr(hc, "_CHUNK_TARGET", 4096)               # two 32-row blocks per chunk
    again = convert(tmp_path / "fp8", tmp_path / "again.hearth", **kw)
    assert again.read_bytes() == whole.read_bytes()
    starts = {}
    for name, r0 in calls:
        starts.setdefault(name, []).append(r0)
    assert starts["tok_embd"] == list(range(0, 320, 64)) and "blk.1.attn_q_b" not in starts
    assert "model.layers.1.self_attn.q_b_proj.weight_scale_inv" in HFCheckpoint(tmp_path / "fp8").names()
    assert "blk.0.attn_norm" not in starts                       # 1-D tensors are never chunked


def test_safetensors_backends_agree(tmp_path):
    model = build_hf_model("qwen2_moe", seed=2).to(torch.bfloat16)
    model.save_pretrained(tmp_path / "bf16")
    from safetensors.torch import save_file
    save_file({"extra.ids": torch.arange(6, dtype=torch.int64).reshape(2, 3),
               "extra.half": torch.randn(4, 5).half()}, str(tmp_path / "bf16" / "zz-extra.safetensors"))
    a, b = HFCheckpoint(tmp_path / "bf16", backend="torch"), HFCheckpoint(tmp_path / "bf16", backend="numpy")
    assert np.array_equal(a.load("extra.ids"), np.arange(6, dtype=np.float32).reshape(2, 3))
    assert sorted(a.names()) == sorted(b.names())
    for n in a.names():
        assert a.shape(n) == b.shape(n)
        x, y = a.load(n), b.load(n)
        assert x.dtype == y.dtype == np.float32 and np.array_equal(x, y), n
        if len(a.shape(n)) == 2 and a.shape(n)[0] > 3:
            assert np.array_equal(a.load(n, 1, 3), x[1:3])
            assert np.array_equal(b.load(n, 1, 3), x[1:3])


# ---------------------------------------------------------------- mapping and metadata

@pytest.mark.parametrize("name", ALL)
def test_name_mapping_covers_checkpoint(name, tmp_path):
    model = build_hf_model(name)
    model.save_pretrained(tmp_path / "hf")
    cfg = json.loads((tmp_path / "hf" / "config.json").read_text())
    meta = hearth_meta(cfg)
    ck = HFCheckpoint(tmp_path / "hf")
    mapped = {hf for _, hf, _ in tensor_plan(meta)}
    for li in range(meta["n_layers"]):
        if meta["layer_kind"][li]:
            for e in range(meta["n_experts"]):
                mapped.update(expert_names(meta, li, e))
    names = set(ck.names())
    assert mapped <= names, sorted(mapped - names)[:5]
    assert all(_ignorable(n, meta["n_layers"]) for n in names - mapped), sorted(names - mapped)[:5]
    L = meta["n_layers"]                       # layers past n_layers (DeepSeek MTP) and scale tensors are skipped
    assert _ignorable(f"model.layers.{L}.mlp.gate.weight", L) and _ignorable("x.weight_scale_inv", L)
    assert not _ignorable(f"model.layers.{L - 1}.mlp.gate.weight", L) and not _ignorable("model.layers.x.y", L)
    hearth_names = [h for h, _, _ in tensor_plan(meta)]
    assert len(set(hearth_names)) == len(hearth_names)
    assert all(len(h) <= 79 for h in hearth_names)
    from hearth.format import apply_defaults, canonical_tensors
    canon = canonical_tensors(apply_defaults(meta))          # the plan is exactly FORMAT.md §4.1 for this config
    assert set(canon) == set(hearth_names) | {"rope_inv_freq"}
    for h, hf, _ in tensor_plan(meta):
        assert tuple(ck.shape(hf)) == canon[h][0], h


def test_unknown_or_missing_tensors_are_errors(tmp_path):
    from safetensors.torch import save_file
    model = build_hf_model("mixtral")
    model.save_pretrained(tmp_path / "hf")
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    extra = dict(sd, **{"model.layers.0.self_attn.o_proj.bias": torch.zeros(64)})
    save_file(extra, str(tmp_path / "hf" / "model.safetensors"))
    with pytest.raises(ValueError, match="not understood"):
        _f32_convert(tmp_path / "hf", tmp_path / "x.hearth")
    many = dict(sd, **{f"model.extra{i}.weight": torch.zeros(2) for i in range(9)})
    save_file(many, str(tmp_path / "hf" / "model.safetensors"))
    with pytest.raises(ValueError, match=r"not understood .* \.\.\.$"):        # more than 8: elided
        _f32_convert(tmp_path / "hf", tmp_path / "x.hearth")
    sd.pop("model.layers.2.block_sparse_moe.experts.1.w3.weight")
    save_file(sd, str(tmp_path / "hf" / "model.safetensors"))
    with pytest.raises(ValueError, match="lacks"):
        _f32_convert(tmp_path / "hf", tmp_path / "x.hearth")
    assert not (tmp_path / "x.hearth").exists()


def test_expert_width_differs_from_d_model(tmp_path):
    """F != D (every hf_cases model has F == D == 64): expert matrices keep their orientation, and
    a width that is not a multiple of 64 falls back to F16 slabs when Q4 is requested."""
    import transformers as tf
    from test_reference import _base
    case = (tf.MixtralForCausalLM, tf.MixtralConfig(**_base(intermediate_size=96, num_local_experts=4,
                                                             num_experts_per_tok=2, rope_theta=1e6)))
    model = build_hf_model("mixtral", seed=2, case=case)
    model.save_pretrained(tmp_path / "hf")
    want, want_routes = hf_logits_and_routes(model, TOKENS)
    path = _f32_convert(tmp_path / "hf", tmp_path / "f.hearth")
    with ContainerReader(path) as r:
        assert r.config["expert_ffn_dim"] == 96 and r.expert_entry(0, 0).dtype == F32
        g, u, d = r.read_expert(1, 2)
        sd = model.state_dict()
        assert np.array_equal(d, sd["model.layers.1.block_sparse_moe.experts.2.w2.weight"].numpy())   # [64, 96]
        assert np.array_equal(u, sd["model.layers.1.block_sparse_moe.experts.2.w3.weight"].numpy())   # [96, 64]
    got = Reference(path).eval(TOKENS)
    assert np.abs(got - want).max() <= 2e-3 * max(1.0, float(np.abs(want).max()))
    if _native.available():
        q = convert(tmp_path / "hf", tmp_path / "q.hearth", progress=False)          # Q4 experts requested
        with ContainerReader(q) as r:
            assert r.expert_entry(0, 0).dtype == quant.F16 and r.tensors["blk.0.attn_q"].dtype == quant.Q8
            assert r.tensors["blk.0.attn_q"].shape == (64, 64)


def test_unsupported_configs_raise(tmp_path):
    with pytest.raises(ValueError, match="unsupported model_type"):
        hearth_meta({"model_type": "llama", "num_hidden_layers": 1})
    cfg = hf_cases()["qwen3_moe"][1].to_dict()
    cfg["attention_bias"] = True
    with pytest.raises(ValueError, match="attention_bias"):
        hearth_meta(cfg)


def test_sliding_window_clamps_max_seq():
    mx = hf_cases()["mixtral"][1].to_dict()
    mx["sliding_window"] = 32
    assert hearth_meta(mx)["max_seq"] == 32 and hearth_meta(mx, 16)["max_seq"] == 16
    q3 = hf_cases()["qwen3_moe"][1].to_dict()
    q3.update(sliding_window=32, use_sliding_window=False)
    assert hearth_meta(q3)["max_seq"] == 64
    q3["use_sliding_window"] = True
    assert hearth_meta(q3)["max_seq"] == 32


def test_layer_kinds_from_config():
    q3 = hearth_meta(hf_cases()["qwen3_moe"][1].to_dict())
    assert q3["layer_kind"] == [1, 0, 1] and q3["dense_ffn_dim"] == 128
    q2 = hf_cases()["qwen2_moe"][1].to_dict()
    assert hearth_meta(q2)["layer_kind"] == [1, 1, 0]
    q2.update(mlp_only_layers=[], decoder_sparse_step=2)
    assert hearth_meta(q2)["layer_kind"] == [0, 1, 0]
    ds = hearth_meta(hf_cases()["deepseek_v3"][1].to_dict())
    assert ds["layer_kind"] == [0, 1, 1] and ds["shared_ffn_dim"] == 64 and ds["routed_scale"] == 2.5


@pytest.mark.parametrize("name", ["deepseek_v3", "deepseek_v3_no_q_lora", "qwen3_moe", "mixtral"])
def test_rope_and_attn_scale_match_transformers(name, tmp_path):
    model = build_hf_model(name)
    model.save_pretrained(tmp_path / "hf")
    path = _f32_convert(tmp_path / "hf", tmp_path / "m.hearth")
    r = ContainerReader(path)
    rot = model.model.rotary_emb
    assert np.array_equal(r.read_tensor("rope_inv_freq"), rot.inv_freq.float().numpy())
    assert r.config["rope_attn_factor"] == float(np.float32(rot.attention_scaling))
    attn = model.model.layers[0].self_attn
    scaling = getattr(attn, "scaling", None) or attn.head_dim ** -0.5
    assert r.config["attn_scale"] == float(np.float32(scaling))


@pytest.mark.parametrize("rs", [None, {"type": "linear", "factor": 2.0},
                                {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16,
                                 "mscale": 1.0, "mscale_all_dim": 0.8, "beta_fast": 32, "beta_slow": 1},
                                {"rope_type": "yarn", "factor": 8.0}])
def test_rope_fallback_matches_transformers(rs):
    cfg = hf_cases()["deepseek_v3"][1].to_dict()
    cfg["rope_scaling"] = rs
    inv, af = rope_params(cfg)
    inv2, af2 = _rope_fallback(cfg, cfg["qk_rope_head_dim"])
    np.testing.assert_allclose(inv2, inv, rtol=2e-7, atol=0)
    assert math.isclose(af, af2, rel_tol=1e-12)


def test_kimi_k2_model_type(tmp_path):
    model = build_hf_model("deepseek_v3")
    model.save_pretrained(tmp_path / "hf")
    cfgp = tmp_path / "hf" / "config.json"
    cfg = json.loads(cfgp.read_text())
    cfg.update(model_type="kimi_k2", architectures=["DeepseekV3ForCausalLM"],
               auto_map={"AutoConfig": "configuration_deepseek.DeepseekV3Config"})
    cfgp.write_text(json.dumps(cfg))
    want, _ = hf_logits_and_routes(model, TOKENS)
    r = Reference(_f32_convert(tmp_path / "hf", tmp_path / "k.hearth"))
    assert r.cfg["arch"] == "deepseek_v3" and r.cfg["attn_kind"] == 1
    assert np.abs(r.eval(TOKENS) - want).max() <= 2e-3 * max(1.0, float(np.abs(want).max()))


def test_tokenizer_template_and_special_ids(tmp_path):
    model = build_hf_model("olmoe")
    src = tmp_path / "hf"
    model.save_pretrained(src)
    (src / "tokenizer.json").write_text('{"version": "1.0", "model": {"type": "BPE"}}')
    tmpl = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
    (src / "tokenizer_config.json").write_text(json.dumps({"chat_template": [{"name": "default", "template": tmpl},
                                                                             {"name": "tool_use", "template": "x"}]}))
    gen = json.loads((src / "generation_config.json").read_text())
    gen["eos_token_id"] = [7, 9, 50279]
    (src / "generation_config.json").write_text(json.dumps(gen))
    cfg = json.loads((src / "config.json").read_text())
    cfg.update(eos_token_id=300, bos_token_id=11)
    (src / "config.json").write_text(json.dumps(cfg))
    dst = tmp_path / "out" / "olmoe.hearth"
    dst.parent.mkdir()
    _f32_convert(src, dst, max_seq=48)
    c = ContainerReader(dst).config
    assert c["tokenizer"] == "olmoe.tokenizer.json"
    assert (dst.parent / "olmoe.tokenizer.json").read_text() == (src / "tokenizer.json").read_text()
    assert c["chat_template"] == tmpl and c["max_seq"] == 48
    assert c["eos_ids"] == [300, 7, 9] and c["bos_id"] == 11      # 50279 >= vocab 320: dropped
    assert c["source"] == (cfg.get("_name_or_path") or "hf")      # provenance: the HF id, else the folder name
    cfg["_name_or_path"] = "allenai/OLMoE-1B-7B-0924"
    (src / "config.json").write_text(json.dumps(cfg))
    assert ContainerReader(_f32_convert(src, dst)).config["source"] == "allenai/OLMoE-1B-7B-0924"
    cfg.update(bos_token_id=320)
    (src / "config.json").write_text(json.dumps(cfg))
    assert ContainerReader(_f32_convert(src, dst)).config["bos_id"] == 0xFFFFFFFF


# ---------------------------------------------------------------- quantized conversion

@needs_native
@pytest.mark.parametrize("name,seed", [("qwen3_moe", 1), ("deepseek_v3", 1), ("deepseek_v3", 3), ("qwen2_moe", 5)])
def test_q4_conversion_close_to_f32(name, seed, tmp_path):
    """Q4 experts + Q8 dense track the F32 conversion. Quantization noise legitimately flips
    near-tied routing decisions, and one flip can move a token's logits by as much as their
    own magnitude (measured), so the error bounds are taken with the Q4 model replaying the F32 model's routing;
    free-running routing must still agree on most decisions."""
    model = build_hf_model(name, seed=seed)
    model.save_pretrained(tmp_path / "hf")
    f = _f32_convert(tmp_path / "hf", tmp_path / "f.hearth")
    q = convert(tmp_path / "hf", tmp_path / "q.hearth", threads=4, progress=False)   # Q4 experts, Q8 dense
    with ContainerReader(q) as r:
        li = r.config["layer_kind"].index(1)
        assert r.expert_entry(li, 0).dtype == quant.Q4
        assert r.tensors["tok_embd"].dtype == quant.Q8 and r.tensors["lm_head"].dtype == quant.Q8
        assert r.tensors[f"blk.{li}.attn_o"].dtype == quant.Q8 and r.tensors[f"blk.{li}.moe_router"].dtype == F32
        assert r.tensors[f"blk.{li}.ffn_norm"].dtype == F32 and r.tensors["rope_inv_freq"].dtype == F32
    assert q.stat().st_size < 0.35 * f.stat().st_size
    with Reference(f) as ra, Reference(q, emulate_act_quant=True) as rb:
        a = ra.eval(TOKENS)
        routes = ra.routing_history()
        rb.eval(TOKENS)
        free = rb.routing_history()
        rb.reset()
        rb.replay_routes(routes)
        b = rb.eval(TOKENS)
        assert np.array_equal(rb.routing_history(), routes)
    agree = np.mean([set(x) == set(y) for x, y in zip(routes.reshape(-1, routes.shape[-1]).tolist(),
                                                      free.reshape(-1, free.shape[-1]).tolist())])
    assert agree >= 0.6, agree                     # 0.79..1.0 over 6 archs x 8 seeds
    # bounds vs worst of 6 archs x 8 seeds: rel RMSE 0.141, corr 0.9899, per-token 0.244
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.98
    assert np.sqrt(((a - b) ** 2).mean()) < 0.2 * np.sqrt((a ** 2).mean())
    assert (np.abs(a - b).max(axis=1) <= 0.35 * np.abs(a).max(axis=1)).all()
    # threads do not change the bytes
    q1 = convert(tmp_path / "hf", tmp_path / "q1.hearth", threads=1, progress=False)
    assert q1.read_bytes() == q.read_bytes()


@pytest.mark.parametrize("where", ["jinja", "json", "tokenizer_config_wins"])
def test_chat_template_fallback_files(tmp_path, where):
    model = build_hf_model("mixtral")
    src = tmp_path / "hf"
    model.save_pretrained(src)
    tmpl = "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"
    tc = {"model_max_length": 64}
    if where == "tokenizer_config_wins":
        tc["chat_template"] = tmpl
        (src / "chat_template.jinja").write_text("other", encoding="utf-8")
        (src / "chat_template.json").write_text(json.dumps({"chat_template": "other"}))
    elif where == "jinja":
        (src / "chat_template.jinja").write_text(tmpl, encoding="utf-8")
    else:
        (src / "chat_template.json").write_text(json.dumps({"chat_template": tmpl}))
    (src / "tokenizer_config.json").write_text(json.dumps(tc))
    c = ContainerReader(_f32_convert(src, tmp_path / "m.hearth")).config
    assert c["chat_template"] == tmpl and c["tokenizer"] == ""


def test_cli_and_progress(tmp_path, capsys):
    from hearth.convert import main
    model = build_hf_model("qwen2_moe")
    model.save_pretrained(tmp_path / "hf")
    dst = tmp_path / "cli.hearth"
    assert main([str(tmp_path / "hf"), str(dst), "--experts", "f32", "--dense", "f16", "--embed", "f32",
                 "--head", "bf16", "--max-seq", "40", "--threads", "2"]) == 0
    err = capsys.readouterr().err
    assert "dense tensors of layer 0" in err and "experts in" in err and "done in" in err
    r = ContainerReader(dst)
    assert r.config["max_seq"] == 40 and r.tensors["blk.0.attn_q"].dtype == quant.F16
    assert r.tensors["lm_head"].dtype == quant.BF16 and r.expert_entry(0, 0).dtype == F32
