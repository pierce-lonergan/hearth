"""Shapes of real (and clearly-labelled hypothetical) MoE models.

Used by the simulator (hearth.sim) and by hearth.synth.make_shaped to build
benchmark containers with a real model's geometry and random weights.

`verified` is True only once the numbers were checked against the model's
published config.json (see docs/BENCHMARKS.md for sources).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Shape:
    name: str
    arch: str                      # hearth arch flavour (qwen3_moe, deepseek_v3, ...)
    n_layers: int
    d_model: int
    vocab: int
    n_experts: int
    top_k: int
    expert_ffn: int
    n_dense_layers: int = 0        # leading dense FFN layers (DeepSeek first_k_dense_replace)
    dense_ffn: int = 0
    shared_ffn: int = 0            # total shared-expert width
    expert_d: int = 0              # expert input/output dim if experts run in a latent space (0 = d_model)
    # attention
    attn: str = "gqa"              # "gqa" | "mla"
    n_heads: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_dim: int = 0
    qk_rope_dim: int = 0
    v_head_dim: int = 0
    tie_embeddings: bool = False
    verified: bool = False
    hypothetical: bool = False
    notes: str = ""

    # ---- derived -------------------------------------------------------
    @property
    def n_moe_layers(self) -> int:
        return self.n_layers - self.n_dense_layers

    def attn_params(self) -> int:
        D = self.d_model
        if self.attn == "mla":
            H = self.n_heads
            qk = self.qk_nope_dim + self.qk_rope_dim
            q = (D * self.q_lora_rank + self.q_lora_rank * H * qk) if self.q_lora_rank else D * H * qk
            kv = D * (self.kv_lora_rank + self.qk_rope_dim) + self.kv_lora_rank * H * (self.qk_nope_dim + self.v_head_dim)
            o = H * self.v_head_dim * D
            return q + kv + o
        hd = self.head_dim
        return D * self.n_heads * hd * 2 + D * self.n_kv_heads * hd * 2

    def expert_params(self) -> int:
        return 3 * (self.expert_d or self.d_model) * self.expert_ffn

    def dense_params(self) -> int:
        """Everything that stays resident: embeddings, head, attention, routers, shared and dense FFNs."""
        D = self.d_model
        emb = self.vocab * D * (1 if self.tie_embeddings else 2)
        per_layer_attn = self.attn_params()
        router = self.n_experts * D
        shared = 3 * D * self.shared_ffn
        dense_ffn = 3 * D * self.dense_ffn
        return (emb + self.n_layers * per_layer_attn + self.n_moe_layers * (router + shared)
                + self.n_dense_layers * dense_ffn)

    def params_total(self) -> int:
        return self.dense_params() + self.n_moe_layers * self.n_experts * self.expert_params()

    def params_active(self) -> int:
        return self.dense_params() - self.vocab * self.d_model * (0 if self.tie_embeddings else 1) \
            + self.n_moe_layers * self.top_k * self.expert_params()

    def expert_bytes(self, bits_per_weight: float = 4.25) -> int:
        """Bytes of one expert slab at the given bits/weight, padded to 4096."""
        raw = self.expert_params() * bits_per_weight / 8.0
        return int(-(-raw // 4096) * 4096)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(params_total=self.params_total(), params_active=self.params_active(),
                 n_moe_layers=self.n_moe_layers)
        return d


PRESETS: dict[str, Shape] = {s.name: s for s in [
    # Verified against config.json on Hugging Face (2026-10-04).
    Shape("olmoe-1b-7b", "olmoe", n_layers=16, d_model=2048, vocab=50304, n_experts=64, top_k=8,
          expert_ffn=1024, n_heads=16, n_kv_heads=16, head_dim=128, verified=True),
    Shape("qwen1.5-moe-a2.7b", "qwen2_moe", n_layers=24, d_model=2048, vocab=151936, n_experts=60, top_k=4,
          expert_ffn=1408, shared_ffn=5632, n_heads=16, n_kv_heads=16, head_dim=128, verified=True),
    Shape("qwen3-30b-a3b", "qwen3_moe", n_layers=48, d_model=2048, vocab=151936, n_experts=128, top_k=8,
          expert_ffn=768, n_heads=32, n_kv_heads=4, head_dim=128, verified=True),
    Shape("qwen3-235b-a22b", "qwen3_moe", n_layers=94, d_model=4096, vocab=151936, n_experts=128, top_k=8,
          expert_ffn=1536, n_heads=64, n_kv_heads=4, head_dim=128, verified=True),
    Shape("mixtral-8x22b", "mixtral", n_layers=56, d_model=6144, vocab=32000, n_experts=8, top_k=2,
          expert_ffn=16384, n_heads=48, n_kv_heads=8, head_dim=128, verified=True),
    Shape("glm-4.5", "glm4_moe", n_layers=92, d_model=5120, vocab=151552, n_experts=160, top_k=8,
          expert_ffn=1536, n_dense_layers=3, dense_ffn=12288, shared_ffn=1536,
          n_heads=96, n_kv_heads=8, head_dim=128, verified=True,
          notes="partial rotary 0.5, attention bias, qk-norm; simulation only for now"),
    Shape("gpt-oss-120b", "gpt_oss", n_layers=36, d_model=2880, vocab=201088, n_experts=128, top_k=4,
          expert_ffn=2880, n_heads=64, n_kv_heads=8, head_dim=64, verified=True,
          notes="sinks, sliding window, MXFP4; simulation only"),
    Shape("deepseek-v3", "deepseek_v3", n_layers=61, d_model=7168, vocab=129280, n_experts=256, top_k=8,
          expert_ffn=2048, n_dense_layers=3, dense_ffn=18432, shared_ffn=2048, attn="mla", n_heads=128,
          q_lora_rank=1536, kv_lora_rank=512, qk_nope_dim=128, qk_rope_dim=64, v_head_dim=128, verified=True),
    Shape("kimi-k2", "deepseek_v3", n_layers=61, d_model=7168, vocab=163840, n_experts=384, top_k=8,
          expert_ffn=2048, n_dense_layers=1, dense_ffn=18432, shared_ffn=2048, attn="mla", n_heads=64,
          q_lora_rank=1536, kv_lora_rank=512, qk_nope_dim=128, qk_rope_dim=64, v_head_dim=128, verified=True),
    Shape("glm-5.2", "glm_moe_dsa", n_layers=78, d_model=6144, vocab=154880, n_experts=256, top_k=8,
          expert_ffn=2048, n_dense_layers=3, dense_ffn=12288, shared_ffn=2048, attn="mla", n_heads=64,
          q_lora_rank=2048, kv_lora_rank=512, qk_nope_dim=192, qk_rope_dim=64, v_head_dim=256, verified=True,
          notes="DSA sparse-attention indexer + MTP; simulation only for now"),
    Shape("kimi-k3", "kimi_k3", n_layers=93, d_model=7168, vocab=163840, n_experts=896, top_k=16,
          expert_ffn=3072, expert_d=3584, n_dense_layers=1, dense_ffn=33792, shared_ffn=6144, attn="mla",
          n_heads=64, q_lora_rank=1536, kv_lora_rank=512, qk_nope_dim=128, qk_rope_dim=64, v_head_dim=128,
          verified=True,
          notes="2.8T: latent MoE (experts in 3584-d), 24 MLA + 69 KDA linear-attention layers, MXFP4 experts; "
                "attention params approximated as MLA for all layers; simulation only"),
]}


def get(name_or_dict) -> Shape:
    if isinstance(name_or_dict, Shape):
        return name_or_dict
    if isinstance(name_or_dict, dict):
        return Shape(**name_or_dict)
    key = str(name_or_dict).lower()
    if key not in PRESETS:
        raise KeyError(f"unknown preset {name_or_dict!r}; known: {', '.join(sorted(PRESETS))}")
    return PRESETS[key]
