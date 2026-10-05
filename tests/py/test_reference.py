"""hearth.reference: canonical numerics (docs/NUMERICS.md), engine-kernel emulation, routing,
and agreement with Hugging Face transformers (INV-NUM-1)."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from hearth import _native, quant, synth
from hearth.format import ContainerReader
from hearth.reference import Reference, _Mat, rmsnorm, rope, sum16

TOKENS = [5, 17, 300, 42, 99, 3, 250, 7, 7, 64, 128, 1, 200, 31, 77, 2]
TINY = [t % 256 for t in TOKENS]          # make_tiny's default vocabulary is 256
needs_native = pytest.mark.skipif(not _native.available(), reason="Hearth native library not available")


# ---------------------------------------------------------------- primitives

def _sum16_loop(a):
    L = [np.float32(0.0)] * 16
    for i, v in enumerate(a):
        L[i & 15] = np.float32(L[i & 15] + np.float32(v))
    for s in (8, 4, 2, 1):
        for j in range(s):
            L[j] = np.float32(L[j] + L[j + s])
    return L[0]


@pytest.mark.parametrize("n", [1, 15, 16, 17, 100, 257])
def test_sum16_matches_definition(n):
    a = (np.random.default_rng(n).standard_normal(n) * 1000).astype(np.float32)
    assert sum16(a) == _sum16_loop(a)
    assert np.array_equal(sum16(np.stack([a, -a])), np.array([_sum16_loop(a), _sum16_loop(-a)], dtype=np.float32))


def test_rmsnorm_and_rope_match_definition():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(40).astype(np.float32)
    w = rng.standard_normal(40).astype(np.float32)
    ms = np.float32(_sum16_loop(x * x) / np.float32(40))
    r = np.float32(1.0) / np.sqrt(np.float32(ms + np.float32(1e-6)))
    assert np.array_equal(rmsnorm(x, w, 1e-6), (x * r) * w)
    inv = (1.0 / 10000 ** (np.arange(0, 12, 2) / 12)).astype(np.float32)
    v = rng.standard_normal(16).astype(np.float32)          # rope_dim 12 < head dim 16: partial rotary
    for style in (0, 1):
        got = rope(v, 7, inv, 12, style, 1.25)
        want = v.copy()
        for j in range(6):
            th = np.float32(np.float32(7) * inv[j])
            c = np.float32(np.float32(math.cos(th)) * np.float32(1.25))
            s = np.float32(np.float32(math.sin(th)) * np.float32(1.25))
            ia, ib = (j, j + 6) if style == 0 else (2 * j, 2 * j + 1)
            a, b = v[ia], v[ib]
            want[ia], want[ib] = np.float32(a * c - b * s), np.float32(b * c + a * s)
        assert np.array_equal(got, want)
        assert np.array_equal(got[12:], v[12:])


@needs_native
@pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")     # deliberate NaN/inf inputs
@pytest.mark.parametrize("dtype", [quant.F32, quant.F16, quant.BF16, quant.Q8, quant.Q4])
def test_matvec_bit_identical_to_engine_kernels(dtype):
    """The reference applies every weight dtype exactly like hearth_matmul (NUMERICS §2-§3)."""
    rng = np.random.default_rng(dtype)
    rows, cols = 45, 320
    W = rng.standard_normal((rows, cols)).astype(np.float32)
    X = (rng.standard_normal((9, cols)) * np.array([[1e-3], [1.0], [50.0]] + [[1.0]] * 6)).astype(np.float32)
    X[3, :64] = 0.0
    X[4, :64] = (np.arange(64) - 31.5) * 0.5      # with amax 63.5, d = 0.5 and x/d is an exact .5 tie
    X[4, 0] = 63.5
    X[4, 64:128] = np.arange(64) - 31.5           # amax 127 -> d = 1: ties again
    X[4, 64] = 127.0
    X[5, 10] = np.nan                             # degenerate inputs: NaN must not win the block max
    X[6, 64:128] = np.nan
    X[7, 3], X[7, 200] = np.inf, -np.inf
    X[8, 5], X[8, 6] = np.inf, np.nan
    Wq = quant.quantize(W, dtype)
    m = _Mat.from_bytes(Wq, dtype, rows, cols, emulate=True)
    ours = np.stack([m.matvec(x) for x in X])
    if quant.is_quant(dtype):
        assert np.isfinite(ours[5:7]).all()       # NaN activations quantize to 0 (inf ones give d = inf)
    for isa in (1, 2, 3):
        Y = _native.matmul(dtype, Wq, rows, cols, X, isa=isa)
        if Y is not None:
            assert np.array_equal(ours.view(np.uint32), Y.view(np.uint32)), f"isa {isa}"
    if not quant.is_quant(dtype):                 # NaN weights: the canonical quiet NaN, like the kernels
        Wn = W.copy()
        Wn[3, 7] = np.nan
        Wq = quant.quantize(Wn, dtype)
        ours = _Mat.from_bytes(Wq, dtype, rows, cols, emulate=True).matvec(X[1])
        assert ours.view(np.uint32)[3] == 0x7FC00000
        Y = _native.matmul(dtype, Wq, rows, cols, X[1], isa=1)
        assert np.array_equal(ours.view(np.uint32), Y[0].view(np.uint32))


# ---------------------------------------------------------------- reference behaviour on synthetic models

@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    d = tmp_path_factory.mktemp("ref_tiny")
    return {a: synth.make_tiny(d / f"{a}.hearth", arch=a, seed=4) for a in synth.ARCHS}


@pytest.mark.parametrize("arch", synth.ARCHS)
def test_kv_cache_across_calls_and_reset(tiny_models, arch):
    ref = Reference(tiny_models[arch])
    full = ref.eval(TINY[:10])
    hist = ref.routing_history()
    ref.reset()
    assert ref.routing_history().shape[0] == 0
    a = ref.eval(TINY[:4])
    b = np.concatenate([ref.eval([t]) for t in TINY[4:7]] + [ref.eval(TINY[7:10])])
    assert np.array_equal(np.concatenate([a, b]), full)
    assert np.array_equal(ref.routing_history(), hist)
    cfg = ref.cfg
    assert hist.dtype == np.uint16 and hist.shape == (10, cfg["n_moe_layers"], cfg["top_k"])
    assert (hist < cfg["n_experts"]).all()
    assert all(len(set(r)) == cfg["top_k"] for r in hist.reshape(-1, cfg["top_k"]).tolist())
    # F32 container: emulation has nothing to emulate
    ref2 = Reference(tiny_models[arch], emulate_act_quant=False)
    assert np.array_equal(ref2.eval(TINY[:10]), full)
    # the KV cache grows past its initial capacity (16 positions) transparently
    long = [(7 * i + 3) % 256 for i in range(40)]
    ref.reset()
    ref2.reset()
    a = ref.eval(long)
    assert np.array_equal(np.concatenate([ref2.eval(long[:17]), ref2.eval(long[17:])]), a)
    assert np.array_equal(a[:10], ref.reset() or ref.eval(long[:10]))


def test_make_tiny_deterministic_and_deepseek_flavour(tmp_path):
    a = synth.make_tiny(tmp_path / "a.hearth", arch="deepseek_v3", seed=11)
    b = synth.make_tiny(tmp_path / "b.hearth", arch="deepseek_v3", seed=11)
    assert a.read_bytes() == b.read_bytes()
    cfg = Reference(a).cfg
    assert cfg["layer_kind"][0] == 0 and cfg["attn_kind"] == 1 and cfg["q_lora_rank"] > 0
    assert cfg["n_group"] > 1 and cfg["score_fn"] == 1 and cfg["score_bias"] == 1 and cfg["rope_style"] == 1
    assert cfg["routed_scale"] != 1.0 and cfg["rope_attn_factor"] != 1.0 and cfg["shared_ffn_dim"] > 0
    assert cfg["topk_group"] < cfg["n_group"]                  # group selection is a real decision


# make_tiny's routing-margin contract, as literals: the tests must not read the thresholds back from synth
SOFTMAX_REL, SIGMOID_ABS, GROUP_ABS = 2e-3, 3e-4, 6e-4


def _margins(path) -> dict:
    """Smallest routing margins of a container on the golden token sequence, as multiples of
    the contract thresholds (>= 1 means clear): 'sel' over the top-(k+1) selection scores of
    every decision, 'grp' at the topk_group boundary of the group scores."""
    out = {"sel": math.inf, "grp": math.inf}
    with Reference(path) as r:
        r.router_trace = []
        r.eval([t % r.V for t in synth.CHECK_TOKENS][:r.cfg["max_seq"]])
        c = r.cfg
        for _, sel, gs in r.router_trace:
            s = np.sort(sel.astype(np.float64))[::-1][:c["top_k"] + 1]
            gap = s[:-1] - s[1:]
            thr = SOFTMAX_REL * s[:-1] if c["score_fn"] == 0 else np.full_like(gap, SIGMOID_ABS)
            out["sel"] = min(out["sel"], float((gap / thr).min()))
            tg, ng = c["topk_group"], c["n_group"]
            if gs is not None and tg < ng:
                g = np.sort(gs.astype(np.float64))[::-1]
                out["grp"] = min(out["grp"], (g[tg - 1] - g[tg]) / GROUP_ABS)
    return out


def _mcfg(score_fn, K, E, ng=1, tg=1):
    return dict(score_fn=score_fn, top_k=K, n_experts=E, n_group=ng, topk_group=tg)


def test_margin_problem_branches():
    mp = synth._margin_problem
    perm = np.array([3, 0, 5, 1, 4, 2])                       # scores arrive unsorted
    base = np.array([0.50, 0.40, 0.30, 0.20, 0.10, 0.05])
    for score_fn, thr in ((0, lambda hi: SOFTMAX_REL * hi), (1, lambda hi: SIGMOID_ABS)):
        m = _mcfg(score_fn, 3, 6)
        assert not mp(base[perm], None, m)
        for j in range(3):                                    # every rank boundary of the top-(k+1)
            for f, want in ((0.9, True), (1.1, False)):
                s = base.copy()
                s[j + 1] = s[j] - f * thr(s[j])
                assert mp(s[perm], None, m) == want, (score_fn, j, f)
        s = base.copy()
        s[4] = s[3]                                           # ranks k+1 and k+2 are not a decision
        assert not mp(s[perm], None, m)
    s = np.array([0.5, 0.5 - 5e-4, 0.1, 0.0])                 # 5e-4: inside softmax's 1e-3, outside sigmoid's 3e-4
    assert mp(s, None, _mcfg(0, 2, 4)) and not mp(s, None, _mcfg(1, 2, 4))
    assert mp(np.array([0.5, 0.5 - 1e-4]), None, _mcfg(1, 2, 2))      # top_k == n_experts: only rank order
    assert not mp(np.array([0.5, 0.4]), None, _mcfg(1, 2, 2))
    m = _mcfg(1, 2, 8, ng=4, tg=2)
    sel = np.linspace(0.9, 0.2, 8)
    gperm = np.array([2, 0, 3, 1])
    for f, want in ((0.9, True), (1.1, False)):               # boundary between the 2nd and 3rd best group
        g = np.array([1.0, 0.9, 0.9 - f * GROUP_ABS, 0.5])
        assert mp(sel, g[gperm], m) == want, f
    assert not mp(sel, np.array([1.0, 1.0, 0.5, 0.5]), m)    # ties away from the boundary are harmless
    assert not mp(sel, np.ones(4), _mcfg(1, 2, 8, ng=4, tg=4))         # every group kept: no group decision
    assert mp(sel, np.ones(4), m)
    # a gap of exactly the threshold is clear (gaps chosen so the float64 arithmetic is exact)
    assert not mp(np.array([1000.0, 998.0]), None, _mcfg(0, 1, 2))      # 2e-3 * 1000 == 2.0
    assert not mp(np.array([SIGMOID_ABS, 0.0]), None, _mcfg(1, 1, 2))
    assert not mp(sel, np.array([1.0, GROUP_ABS, 0.0, 0.0]), m)


BIG = dict(arch="deepseek_v3", dtype=quant.Q4, n_layers=4, n_experts=32, top_k=4)    # golden big_q4 geometry
# (make_tiny kwargs, seed, branch): seeds whose first draw has a near-tie that only this branch catches
REDRAW_SEEDS = [(dict(arch="mixtral"), 1, "softmax"), (dict(arch="qwen3_moe"), 0, "softmax"),
                (dict(arch="olmoe"), 8, "softmax"), (dict(arch="deepseek_v3"), 4, "sigmoid"),
                (dict(arch="deepseek_v3"), 5, "sigmoid"), (dict(arch="deepseek_v3"), 9, "sigmoid"),
                (dict(arch="deepseek_v3"), 14, "sigmoid"), (dict(arch="deepseek_v3"), 7, "group"),
                (dict(arch="deepseek_v3"), 16, "group"), (BIG, 2, "sigmoid"), (BIG, 3, "group"), (BIG, 4, "sigmoid")]
_BRANCH = {"softmax": ("_SOFTMAX_REL_GAP", "sel"), "sigmoid": ("_SIGMOID_ABS_GAP", "sel"),
           "group": ("_GROUP_ABS_GAP", "grp")}


@pytest.mark.parametrize("kw,seed,branch", REDRAW_SEEDS,
                         ids=[f"{k['arch']}{'-big' if k is BIG else ''}-{s}-{b}" for k, s, b in REDRAW_SEEDS])
def test_make_tiny_redraws_near_ties(tmp_path, monkeypatch, kw, seed, branch):
    """INV-NUM-2 needs clear routing margins: make_tiny must redraw these seeds' routers."""
    if kw.get("dtype") == quant.Q4 and not _native.available():
        pytest.skip("Hearth native library not available")
    got = _margins(synth.make_tiny(tmp_path / "ok.hearth", seed=seed, **kw))
    assert min(got.values()) >= 1.0, got
    const, key = _BRANCH[branch]
    monkeypatch.setattr(synth, const, -1.0)                   # this branch never asks for a redraw
    raw = _margins(synth.make_tiny(tmp_path / "raw.hearth", seed=seed, **kw))
    assert raw[key] < 1.0, f"seed {seed} no longer needs a {branch} redraw ({raw}); pick another seed"


@pytest.mark.parametrize("arch", synth.ARCHS)
def test_golden_f32_fixtures_have_clear_margins(tmp_path, arch):
    assert min(_margins(synth.make_tiny(tmp_path / "g.hearth", arch=arch, seed=1)).values()) >= 1.0
    for have, want in zip((synth._SOFTMAX_REL_GAP, synth._SIGMOID_ABS_GAP, synth._GROUP_ABS_GAP),
                          (SOFTMAX_REL, SIGMOID_ABS, GROUP_ABS)):
        assert have >= want


def test_make_tiny_edge_cases(tmp_path, monkeypatch):
    assert synth._groups(8, 2) == (4, 2) and synth._groups(32, 4) == (4, 2)
    assert synth._groups(6, 2) == (2, 1) and synth._groups(7, 2) == (1, 1) and synth._groups(4, 3) == (1, 1)
    for bad in (dict(top_k=0), dict(top_k=9), dict(arch="llama"), dict(dtype=quant.I32)):
        with pytest.raises(ValueError):
            synth.make_tiny(tmp_path / "bad.hearth", **bad)
    p = synth.make_tiny(tmp_path / "dense.hearth", arch="deepseek_v3", n_layers=1)    # no MoE layer at all
    assert p == tmp_path / "dense.hearth" and ContainerReader(p).config["n_moe_layers"] == 0
    p = synth.make_tiny(tmp_path / "d64.hearth", arch="mixtral", d_model=64, vocab=128, seed=3)
    assert ContainerReader(p).config["n_heads"] == 4
    orig = synth._tiny_weights

    def flat_head(m, rng):
        w = orig(m, rng)
        w["lm_head"][:] = 0.0
        return w

    monkeypatch.setattr(synth, "_tiny_weights", flat_head)
    with pytest.raises(RuntimeError, match="degenerate"):
        synth.make_tiny(tmp_path / "flat.hearth", arch="olmoe")
    assert not (tmp_path / "flat.hearth").exists()            # a container that failed its checks is removed
    monkeypatch.setattr(synth, "_tiny_weights", orig)
    monkeypatch.setattr(synth, "_SOFTMAX_REL_GAP", 10.0)      # no router can satisfy this
    monkeypatch.setattr(synth, "_MAX_REDRAWS", 2)
    with pytest.raises(RuntimeError, match="margins"):
        synth.make_tiny(tmp_path / "tight.hearth", arch="mixtral")
    assert not (tmp_path / "tight.hearth").exists()


@pytest.mark.parametrize("arch", synth.ARCHS)
def test_make_tiny_short_max_seq(tmp_path, arch):
    for ms in (1, 8):
        p = synth.make_tiny(tmp_path / f"s{ms}.hearth", arch=arch, max_seq=ms, n_layers=2, d_model=64, vocab=64)
        with Reference(p) as r:
            assert r.cfg["max_seq"] == ms
            r.eval(list(range(ms)))
            with pytest.raises(ValueError, match="max_seq"):
                r.eval([1])
    with pytest.raises(ValueError, match="max_seq"):
        synth.make_tiny(tmp_path / "s0.hearth", arch=arch, max_seq=0)
    assert not (tmp_path / "s0.hearth").exists()


def test_replay_routes(tiny_models):
    ref = Reference(tiny_models["deepseek_v3"])
    want = ref.eval(TINY[:6])
    hist = ref.routing_history()
    ref.reset()
    ref.replay_routes(hist)                     # replaying a model's own routing changes nothing
    assert np.array_equal(ref.eval(TINY[:6]), want) and np.array_equal(ref.routing_history(), hist)
    with pytest.raises(ValueError, match="cover 6 positions"):
        ref.eval([1])
    E, K = ref.E, ref.K
    alt = np.array([[[e for e in range(E) if e not in row][:K] for row in tok] for tok in hist.tolist()])
    ref.reset()                                 # the replay setting survives reset()
    ref.replay_routes(alt)
    got = ref.eval(TINY[:6])
    assert np.array_equal(ref.routing_history(), alt) and not np.array_equal(got, want)
    ref.replay_routes(None)
    ref.reset()
    assert np.array_equal(ref.eval(TINY[:6]), want)
    for bad in (hist[:, :1], hist[..., :1], hist.astype(np.float32), np.full_like(hist, E),
                np.repeat(hist[..., :1], K, axis=-1), hist[0], hist.astype(np.int64) - E, hist.astype(np.int64) + E):
        with pytest.raises(ValueError):
            ref.replay_routes(bad)
    ref.replay_routes(np.array([[[0, 1], [1, 0]]]))           # the same experts in both MoE layers is fine
    ref.reset()
    ref.eval([3])
    assert ref.routing_history().tolist() == [[[0, 1], [1, 0]]]


def test_make_tiny_arch_flavours(tiny_models):
    want = {"qwen3_moe": dict(qk_norm=1, norm_topk_prob=1), "olmoe": dict(qk_norm=2),
            "mixtral": dict(norm_topk_prob=1), "qwen2_moe": dict(qkv_bias=1, shared_gate=1)}
    for arch, kv in want.items():
        c = ContainerReader(tiny_models[arch]).config
        for k, v in kv.items():
            assert c[k] == v, (arch, k)
    for arch in ("qwen3_moe", "mixtral"):
        c = ContainerReader(tiny_models[arch]).config
        assert c["n_kv_heads"] < c["n_heads"]
    c = ContainerReader(tiny_models["olmoe"]).config            # PYTHON_API defaults of make_tiny
    assert (c["n_layers"], c["d_model"], c["n_experts"], c["top_k"], c["expert_ffn_dim"], c["vocab_size"],
            c["max_seq"]) == (3, 128, 8, 2, 64, 256, 256)


def _with_router(ref, layer, W, bias=None):
    E, D = W.shape
    ref._mats[f"blk.{layer}.moe_router#0"] = _Mat(quant.F32, E, D, w=W.astype(np.float32))
    if bias is not None:
        ref._vecs[f"blk.{layer}.moe_router_bias"] = np.asarray(bias, dtype=np.float32)


def test_routing_ties_pick_lower_index(tiny_models):
    ref = Reference(tiny_models["qwen3_moe"])
    D, E = ref.D, ref.E
    _with_router(ref, 0, np.zeros((E, D)))
    ids, w = ref._route(0, np.ones(D, dtype=np.float32))
    assert list(ids) == [0, 1]
    assert np.allclose(w, 0.5)                       # norm_topk_prob=1
    W = np.zeros((E, D))
    W[5, 0] = W[2, 0] = 1.0                          # experts 2 and 5 tie at the top
    _with_router(ref, 0, W)
    ids, _ = ref._route(0, np.ones(D, dtype=np.float32))
    assert list(ids) == [2, 5]


def test_group_routing_masks_dropped_groups(tiny_models):
    ref = Reference(tiny_models["deepseek_v3"])
    c = ref.cfg
    E, D, ng, tg = ref.E, ref.D, c["n_group"], c["topk_group"]
    assert (E, ng, tg, ref.K) == (8, 4, 2, 2)
    # sigmoid(0) = 0.5 for all; bias decides: group 3 gets the single best expert but a poor partner,
    # groups 0 and 1 have the best top-2 sums -> expert 7 must not be chosen although its sel is largest
    bias = np.array([0.30, 0.29, 0.28, 0.27, 0.0, 0.0, -0.4, 0.35])
    _with_router(ref, 1, np.zeros((E, D)), bias)
    ref.router_trace = []
    ids, w = ref._route(1, np.ones(D, dtype=np.float32))
    assert list(ids) == [0, 1]
    sel = ref.router_trace[-1][1]
    assert (sel[4:] == 0.0).all()                    # dropped groups -> 0.0
    assert np.allclose(w, 0.5 * c["routed_scale"])  # weights come from the unbiased scores
    # ties between groups keep the lower group index
    _with_router(ref, 1, np.zeros((E, D)), np.zeros(E))
    ids, _ = ref._route(1, np.ones(D, dtype=np.float32))
    assert list(ids) == [0, 1]


@needs_native
def test_quantized_emulation_tracks_float(tmp_path):
    p = synth.make_tiny(tmp_path / "q8.hearth", arch="qwen2_moe", dtype=quant.Q8, seed=5)
    a = Reference(p, emulate_act_quant=True).eval(TINY[:8])
    b = Reference(p, emulate_act_quant=False).eval(TINY[:8])
    assert not np.array_equal(a, b)
    # activation quantization is a small perturbation (an occasional token may route differently)
    assert np.median(np.abs(a - b).max(axis=1)) <= 0.05 * max(1.0, np.abs(b).max())


# ---------------------------------------------------------------- transformers agreement (INV-NUM-1)

def _base(**kw):
    d = dict(vocab_size=320, hidden_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
             max_position_embeddings=64, rms_norm_eps=1e-6)
    d.update(kw)
    return d


def hf_cases():
    """name -> (model class, config) for tiny transformers models; dims are multiples of 64 where Q8/Q4 apply."""
    import transformers as tf
    yarn = dict(type="yarn", factor=4.0, original_max_position_embeddings=16, mscale=1.0, mscale_all_dim=0.8,
                beta_fast=32, beta_slow=1)
    ds = dict(num_key_value_heads=4, intermediate_size=128, moe_intermediate_size=64, n_routed_experts=8,
              kv_lora_rank=64, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16)
    return {
        "qwen3_moe": (tf.Qwen3MoeForCausalLM, tf.Qwen3MoeConfig(**_base(
            head_dim=32, intermediate_size=128, moe_intermediate_size=64, num_experts=8, num_experts_per_tok=2,
            norm_topk_prob=True, mlp_only_layers=[1], rope_theta=1e6))),
        "olmoe": (tf.OlmoeForCausalLM, tf.OlmoeConfig(**_base(
            num_key_value_heads=4, intermediate_size=64, num_experts=8, num_experts_per_tok=2))),
        "mixtral": (tf.MixtralForCausalLM, tf.MixtralConfig(**_base(
            intermediate_size=64, num_local_experts=4, num_experts_per_tok=2, rope_theta=1e6))),
        "qwen2_moe": (tf.Qwen2MoeForCausalLM, tf.Qwen2MoeConfig(**_base(
            intermediate_size=128, moe_intermediate_size=64, shared_expert_intermediate_size=64, num_experts=8,
            num_experts_per_tok=2, mlp_only_layers=[2]))),
        "deepseek_v3": (tf.DeepseekV3ForCausalLM, tf.DeepseekV3Config(**_base(
            **ds, num_experts_per_tok=2, n_group=4, topk_group=2, first_k_dense_replace=1, routed_scaling_factor=2.5,
            q_lora_rank=64, n_shared_experts=1, rope_scaling=yarn))),
        "deepseek_v3_no_q_lora": (tf.DeepseekV3ForCausalLM, tf.DeepseekV3Config(**_base(
            **ds, num_experts_per_tok=3, n_group=1, topk_group=1, first_k_dense_replace=0, routed_scaling_factor=1.5,
            q_lora_rank=None, n_shared_experts=2, rope_interleave=False))),
    }


def build_hf_model(name, seed=1, case=None):
    """A float32 eager transformers model with well-conditioned random weights (clear routing margins);
    case = (model class, config) instead of hf_cases()[name]."""
    import torch
    cls, cfg = case or hf_cases()[name]
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cls(cfg).eval().float()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for pname, p in model.named_parameters():
            if p.ndim == 1:
                v = (1 + 0.15 * torch.randn(p.shape, generator=g)) if "norm" in pname else 0.1 * torch.randn(p.shape, generator=g)
            elif "embed_tokens" in pname:
                v = torch.randn(p.shape, generator=g)
            else:
                router = pname.endswith("mlp.gate.weight") or pname.endswith("block_sparse_moe.gate.weight")
                gain = 2.5 if router else (2.0 if "lm_head" in pname else 1.0)
                v = torch.randn(p.shape, generator=g) * (gain / math.sqrt(p.shape[1]))
            p.copy_(v)
        for bname, b in model.named_buffers():
            if bname.endswith("e_score_correction_bias"):
                b.copy_(0.1 * torch.randn(b.shape, generator=g))
    return model


def hf_logits_and_routes(model, tokens):
    """Full-sequence logits and the selected expert *sets* per (token, MoE layer)."""
    import torch
    routes = []
    hooks = []
    for layer in model.model.layers:
        moe = getattr(layer, "block_sparse_moe", None) or layer.mlp
        gate = getattr(moe, "gate", None)
        if gate is None or not hasattr(moe, "experts"):
            continue
        k = model.config.num_experts_per_tok

        def hook(mod, inp, out, k=k):
            if isinstance(out, tuple):                 # DeepseekV3TopkRouter -> (indices, weights)
                routes.append([set(r) for r in out[0].tolist()])
            else:                                      # router logits -> softmax top-k
                routes.append([set(r) for r in torch.topk(torch.softmax(out.float(), -1), k, -1)[1].tolist()])
        hooks.append(gate.register_forward_hook(hook))
    try:
        with torch.no_grad():
            logits = model(torch.tensor([tokens])).logits[0].float().numpy()
    finally:
        for h in hooks:
            h.remove()
    return logits, [[routes[li][t] for li in range(len(routes))] for t in range(len(tokens))]


def convert_f32(model, tmp_path):
    from hearth.convert import convert
    model.save_pretrained(tmp_path / "hf")
    F = quant.F32
    return convert(tmp_path / "hf", tmp_path / "m.hearth", expert_dtype=F, dense_dtype=F, embed_dtype=F,
                   head_dtype=F, progress=False)


@pytest.mark.parametrize("name", ["qwen3_moe", "olmoe", "mixtral", "qwen2_moe", "deepseek_v3", "deepseek_v3_no_q_lora"])
def test_reference_matches_transformers(name, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = build_hf_model(name)
    want, want_routes = hf_logits_and_routes(model, TOKENS)
    path = convert_f32(model, tmp_path)
    ref = Reference(path)
    one = np.stack([ref.eval([t])[0] for t in TOKENS])
    hist = ref.routing_history()
    ref.reset()
    full = ref.eval(TOKENS)
    scale = max(1.0, float(np.abs(want).max()))
    for got in (one, full):
        err = float(np.abs(got - want).max())
        assert err <= 2e-3 * scale, f"{name}: max abs logit error {err:.3e} (scale {scale:.2f})"
    assert hist.shape[1] == len(want_routes[0])
    for t in range(len(TOKENS)):
        for li, s in enumerate(want_routes[t]):
            assert set(hist[t, li].tolist()) == s, f"{name}: token {t}, MoE layer {li}"
    c = ref.cfg
    if name == "deepseek_v3":
        assert c["rope_style"] == 1 and c["rope_attn_factor"] != 1.0 and c["n_group"] == 4
    if name == "deepseek_v3_no_q_lora":
        assert c["q_lora_rank"] == 0 and c["rope_style"] == 0 and "blk.0.attn_q" in ref.reader.tensors


def _rewrite(src, dst, meta_updates=None, drop=(), replace=None, put=None):
    """Copy a container byte for byte through ContainerReader/ContainerWriter with edits;
    put = {name: (shape, dtype, float array)} adds or overrides tensors."""
    from hearth.format import ContainerWriter
    replace, put = replace or {}, put or {}
    with ContainerReader(src) as r:
        meta = dict(r.meta, **(meta_updates or {}))
        c = r.config
        moe = [i for i in range(c["n_layers"]) if c["layer_kind"][i] == 1]
        with ContainerWriter(dst, meta) as w:
            names = [n for n in r.tensors if n not in drop and n not in put]
            for n in names:
                t = r.tensors[replace.get(n, n)]
                w.declare_tensor(n, t.shape, t.dtype)
            for n, (shape, dt, _) in put.items():
                w.declare_tensor(n, shape, dt)
            if moe:
                w.declare_experts(c["n_layers"], c["n_experts"], c["d_model"], c["expert_ffn_dim"],
                                  lambda li, e: r.expert_entry(li, e).dtype, moe)
            for n in names:
                w.write_tensor(n, r.read_tensor(replace.get(n, n), dequant=False))
            for n, (_, _, a) in put.items():
                w.write_tensor(n, np.asarray(a, dtype=np.float32))
            for li in moe:
                for e in range(c["n_experts"]):
                    w.write_expert(li, e, *r.read_expert(li, e, dequant=False))
    return dst


def test_canonical_tensor_table_matches_make_tiny(tiny_models):
    from hearth.format import canonical_tensors
    for arch, path in tiny_models.items():
        with ContainerReader(path) as r:
            want = canonical_tensors(r.config)
            assert set(want) == set(r.tensors), arch            # make_tiny writes exactly what §4.1 asks for
            for n, (shape, f32_only) in want.items():
                assert r.tensors[n].shape == shape and (not f32_only or r.tensors[n].dtype == quant.F32), (arch, n)


def test_reference_validates_tensors_against_format(tmp_path):
    """FORMAT.md §4.1 shapes and F32-only classes are enforced before any number is produced."""
    ds = synth.make_tiny(tmp_path / "ds.hearth", arch="deepseek_v3", seed=6)
    q2 = synth.make_tiny(tmp_path / "q2.hearth", arch="qwen2_moe", seed=6)
    one = (1,), quant.F32, np.ones(1)
    cases = [(ds, dict(put={"blk.0.attn_norm": one}), "attn_norm' has shape"),
             (ds, dict(put={"blk.1.moe_router_bias": one}), "moe_router_bias' has shape"),
             (ds, dict(put={"out_norm": one}), "out_norm' has shape"),
             (ds, dict(put={"blk.1.attn_kv_a_norm": one}), "attn_kv_a_norm' has shape"),
             (ds, dict(put={"tok_embd": ((512, 64), quant.F32, np.zeros((512, 64)))}), "tok_embd' has shape"),
             (ds, dict(put={"blk.0.ffn_norm": ((128,), quant.F16, np.ones(128))}), "ffn_norm' is F16, must be F32"),
             (ds, dict(put={"blk.2.moe_router": ((8, 128), quant.BF16, np.zeros((8, 128)))}), "must be F32"),
             (ds, dict(drop={"blk.2.attn_q_b"}), "lacks tensor 'blk.2.attn_q_b'"),
             (ds, dict(drop={"rope_inv_freq"}), "lacks tensor 'rope_inv_freq'"),
             (q2, dict(drop={"blk.1.attn_k_bias"}), "lacks tensor 'blk.1.attn_k_bias'"),     # qkv_bias=1
             (q2, dict(drop={"blk.2.shexp_gate_inp"}), "shexp_gate_inp"),
             (q2, dict(put={"blk.0.attn_q_bias": ((64,), quant.F32, np.zeros(64))}), "attn_q_bias' has shape")]
    for k, (src, edit, msg) in enumerate(cases):
        bad = _rewrite(src, tmp_path / f"bad{k}.hearth", **edit)
        with pytest.raises(ValueError, match=msg):
            Reference(bad)
        bad.unlink()                                              # the rejected file was closed again
    import hearth.reference as hr
    closed = []

    class Spy(ContainerReader):
        def close(self):
            closed.append(self.path.name)
            super().close()

    with pytest.MonkeyPatch.context() as mp:                     # closed explicitly, not left to the GC
        mp.setattr(hr, "ContainerReader", Spy)
        with pytest.raises(ValueError, match="lacks tensor"):
            Reference(_rewrite(ds, tmp_path / "spy.hearth", drop={"lm_head"}))
    assert closed == ["spy.hearth"]
    shared = ContainerReader(_rewrite(ds, tmp_path / "shared.hearth", put={"out_norm": one}))
    with pytest.raises(ValueError, match="out_norm"):
        Reference(shared)
    assert not shared._f.closed                                   # a caller's reader stays the caller's
    shared.close()
    # the same edits applied consistently are fine, and unknown extra tensors are ignored
    extra = _rewrite(ds, tmp_path / "extra.hearth", put={"my.notes": ((3,), quant.F32, np.ones(3)),
                                                          "blk.0.ffn_norm": ((128,), quant.F32, np.ones(128))})
    assert np.isfinite(Reference(extra).eval(TINY[:2])).all()
    tied = _rewrite(ds, tmp_path / "tied.hearth", {"tie_embeddings": 1}, drop={"lm_head"})
    assert Reference(tied).eval(TINY[:1]).shape == (1, 256)
    nogate = _rewrite(q2, tmp_path / "ng.hearth", {"shared_gate": 0}, drop={f"blk.{i}.shexp_gate_inp" for i in range(3)})
    assert np.isfinite(Reference(nogate).eval(TINY[:2])).all()


def _ref_logits(path, n=8):
    with Reference(path) as r:
        return r.eval(TINY[:n])


@pytest.mark.parametrize("arch", ["qwen3_moe", "deepseek_v3"])
def test_rope_dim_zero_means_no_rotation(tmp_path, arch):
    """rope_dim = 0 (accepted by the C reader, which then needs no rope_inv_freq) rotates
    nothing: identical to a rotation by zero angles."""
    src = synth.make_tiny(tmp_path / "src.hearth", arch=arch, seed=3)
    with ContainerReader(src) as r:
        n = r.tensors["rope_inv_freq"].shape[0]
    flat = {"rope_attn_factor": 1.0}                          # cos(0) * 1 = 1, sin(0) * 1 = 0: exact identity
    want = _ref_logits(_rewrite(src, tmp_path / "z.hearth", flat, put={"rope_inv_freq": ((n,), quant.F32, np.zeros(n))}))
    none = _rewrite(src, tmp_path / "n.hearth", dict(flat, rope_dim=0), drop={"rope_inv_freq"})
    assert np.array_equal(_ref_logits(none), want)
    junk = _rewrite(src, tmp_path / "j.hearth", dict(flat, rope_dim=0), put={"rope_inv_freq": ((3,), quant.F32, np.ones(3))})
    assert np.array_equal(_ref_logits(junk), want)            # an unused rope_inv_freq is ignored
    assert not np.array_equal(_ref_logits(_rewrite(src, tmp_path / "f.hearth", flat)), want)


@pytest.mark.parametrize("rope_dim", [2, 8])
def test_mla_partial_rotary(tmp_path, rope_dim):
    """MLA with rope_dim < qk_rope_dim rotates only the first rope_dim dims of q_pe / k_pe
    (GPT-J pairs): the same as rotating all of them with zero frequencies past rope_dim."""
    src = synth.make_tiny(tmp_path / "src.hearth", arch="deepseek_v3", seed=3)
    with ContainerReader(src) as r:
        inv = r.read_tensor("rope_inv_freq")
        assert r.config["rope_style"] == 1 and r.config["qk_rope_dim"] == 2 * inv.size == 16
    h = rope_dim // 2
    padded = np.concatenate([inv[:h], np.zeros(inv.size - h, np.float32)])
    flat = {"rope_attn_factor": 1.0}
    want = _ref_logits(_rewrite(src, tmp_path / "w.hearth", flat, put={"rope_inv_freq": ((inv.size,), quant.F32, padded)}))
    part = _rewrite(src, tmp_path / "p.hearth", dict(flat, rope_dim=rope_dim), put={"rope_inv_freq": ((h,), quant.F32, inv[:h])})
    assert np.array_equal(_ref_logits(part), want)


def test_scales_and_tied_embeddings(tmp_path):
    src = synth.make_tiny(tmp_path / "src.hearth", arch="qwen2_moe", seed=6)
    base = _rewrite(src, tmp_path / "base.hearth", {"norm_eps": 0.0})
    assert ContainerReader(base).read_tensor("tok_embd").tobytes() == ContainerReader(src).read_tensor("tok_embd").tobytes()
    want = Reference(base).eval(TINY[:6])
    # power-of-two scales commute exactly with RMSNorm (eps = 0): logits must not change by a single bit
    scaled = _rewrite(src, tmp_path / "s.hearth", {"norm_eps": 0.0, "emb_scale": 2.0, "residual_scale": 2.0,
                                                    "logit_scale": 0.5})
    assert np.array_equal(Reference(scaled).eval(TINY[:6]), want * np.float32(0.5))
    only_emb = _rewrite(src, tmp_path / "e.hearth", {"norm_eps": 0.0, "emb_scale": 2.0})
    assert not np.array_equal(Reference(only_emb).eval(TINY[:6]), want)
    tied = _rewrite(src, tmp_path / "t.hearth", {"tie_embeddings": 1}, drop={"lm_head"})
    untied = _rewrite(src, tmp_path / "u.hearth", replace={"lm_head": "tok_embd"})
    assert "lm_head" not in ContainerReader(tied).tensors
    assert np.array_equal(Reference(tied).eval(TINY[:6]), Reference(untied).eval(TINY[:6]))


@pytest.mark.parametrize("bad", [dict(top_k=0), dict(top_k=9), dict(n_group=3, topk_group=1),
                                 dict(n_group=4, topk_group=0), dict(n_group=4, topk_group=5),
                                 dict(n_kv_heads=0), dict(n_kv_heads=3)])
def test_reference_rejects_bad_config(tmp_path, monkeypatch, bad):
    from hearth.format import ContainerWriter
    src = synth.make_tiny(tmp_path / "src.hearth", arch="mixtral", seed=2)
    with pytest.raises(ValueError):                  # the writer refuses it ...
        _rewrite(src, tmp_path / "w.hearth", bad)
    monkeypatch.setattr(ContainerWriter, "_check_consistency", lambda self: None)
    forged = _rewrite(src, tmp_path / "bad.hearth", bad)
    with pytest.raises(ValueError):                  # ... and a forged file never reaches the forward pass
        Reference(forged)


def test_expert_cache_is_lru_and_transparent(tiny_models, tmp_path):
    path = tiny_models["olmoe"]
    want = Reference(path).eval(TINY[:5])
    one = sum(m.nbytes for m in Reference(path)._expert(0, 0))
    assert Reference(path)._exp_cap == 4 << 30                     # default expert cache budget: 4 GiB
    small = Reference(path, expert_cache_bytes=2 * one)
    assert np.array_equal(small.eval(TINY[:5]), want)          # eviction never changes results
    assert small._exp_bytes <= 2 * one and len(small._experts) <= 2
    assert small._exp_bytes == sum(m.nbytes for mats in small._experts.values() for m in mats)
    small._experts.clear()
    small._exp_bytes = 0
    a, b, c = (0, 1), (0, 2), (0, 3)
    small._expert(*a), small._expert(*b), small._expert(*a)      # a is now most recently used
    small._expert(*c)
    assert list(small._experts) == [a, c]
    with Reference(path) as r:
        r.eval(TINY[:1])
    assert r.reader._f.closed
    with pytest.raises(ValueError, match="outside vocabulary"):    # the context manager never swallows errors
        with Reference(path) as r:
            r.eval([256])
