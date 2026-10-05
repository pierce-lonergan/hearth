# The `.hearth` container format (version 1)

**Status: normative.** The C reader (`engine/src/modelfile.c`), the Python writer
(`python/hearth/format.py`) and every tool must agree with this document byte for
byte. Changing it is a constraint-class change (see `governance/INVARIANTS.md`):
bump `version` and keep the old reader path working.

A `.hearth` file is one self-describing binary file. It is designed so that:

* the **resident** part of the model (embeddings, attention, norms, routers,
  shared experts, dense FFNs, LM head) is one contiguous region read once at open;
* every **routed expert** is one contiguous, 4096-byte-aligned *slab* that can be
  fetched with a single direct-I/O read (`O_DIRECT` / `FILE_FLAG_NO_BUFFERING`);
* the same file can be copied to several drives and read in parallel (mirrors).

All integers are **little-endian**. There is no implicit padding anywhere except
where this document says "pad".

## 1. File layout

```
offset 0            preamble (64 bytes)
preamble.meta_off   metadata key/value section  (meta_bytes long)
preamble.tdir_off   tensor directory            (n_tensors * 128 bytes)
preamble.edir_off   expert directory            (n_expert_entries * 32 bytes)
...                 dense data region           (each tensor 64-byte aligned)
...                 expert data region          (each slab 4096-byte aligned)
```

Writers place the sections in the order shown, but readers MUST only use the
offsets in the preamble and directories (never assume adjacency).

## 2. Preamble (64 bytes at offset 0)

| off | type | name | value |
|----:|------|------|-------|
| 0  | u32 | magic | `0x48545248` (bytes `H R T H`) |
| 4  | u32 | version | `1` |
| 8  | u64 | meta_off | absolute offset of metadata section |
| 16 | u64 | meta_bytes | size of metadata section |
| 24 | u64 | tdir_off | absolute offset of tensor directory |
| 32 | u64 | n_tensors | number of tensor entries |
| 40 | u64 | edir_off | absolute offset of expert directory |
| 48 | u64 | n_expert_entries | must equal `n_layers * n_experts` (0 if no MoE) |
| 56 | u32 | align | slab alignment, must be `4096` |
| 60 | u32 | reserved | `0` |

## 3. Metadata section

A sequence of entries until `meta_bytes` is consumed. Each entry:

```
u16  key_len
u8   key[key_len]        ASCII, no NUL
u8   type
...  payload
```

| type | name | payload |
|-----:|------|---------|
| 1 | u32 | 4 bytes |
| 2 | f32 | 4 bytes IEEE-754 |
| 3 | u64 | 8 bytes |
| 4 | str | `u32 len` + `len` bytes UTF-8 (no NUL) |
| 5 | u32[] | `u32 count` + `count*4` bytes |
| 6 | f32[] | `u32 count` + `count*4` bytes |
| 7 | u8[]  | `u32 count` + `count` bytes |

Readers ignore unknown keys. Missing keys take the default below.

### 3.1 Keys

| key | type | default | meaning |
|-----|------|---------|---------|
| `arch` | str | `""` | informational (`qwen3_moe`, `olmoe`, `mixtral`, `qwen2_moe`, `deepseek_v3`, `synthetic`) |
| `n_layers` | u32 | required | transformer blocks |
| `d_model` | u32 | required | hidden size D |
| `vocab_size` | u32 | required | V |
| `max_seq` | u32 | 4096 | max positions the model supports |
| `norm_eps` | f32 | 1e-6 | RMSNorm epsilon |
| `attn_kind` | u32 | 0 | 0 = GQA/MHA, 1 = MLA (DeepSeek-V2/V3, Kimi-K2) |
| `n_heads` | u32 | required | attention heads H |
| `n_kv_heads` | u32 | = n_heads | GQA kv heads (ignored for MLA) |
| `head_dim` | u32 | D/H | GQA head dim (ignored for MLA) |
| `qk_norm` | u32 | 0 | 0 none, 1 per-head RMSNorm over head_dim (Qwen3), 2 full-width RMSNorm over H*head_dim / Hkv*head_dim (OLMoE) |
| `qkv_bias` | u32 | 0 | 1 if q/k/v projections have biases (Qwen2-MoE) |
| `q_lora_rank` | u32 | 0 | MLA: 0 = no q compression |
| `kv_lora_rank` | u32 | 0 | MLA latent rank C |
| `qk_nope_dim` | u32 | 0 | MLA |
| `qk_rope_dim` | u32 | 0 | MLA |
| `v_head_dim` | u32 | 0 | MLA |
| `rope_dim` | u32 | head_dim (GQA) / qk_rope_dim (MLA) | number of rotated dims per head (partial rotary) |
| `rope_style` | u32 | 0 | 0 = NEOX/rotate-half (pairs `i`, `i+rope_dim/2`), 1 = GPT-J/interleaved (pairs `2i`, `2i+1`) |
| `rope_attn_factor` | f32 | 1.0 | multiplier applied to cos and sin (YaRN "attention_factor"/mscale) |
| `attn_scale` | f32 | 1/sqrt(qk_dim) | softmax scale applied to q·k |
| `dense_ffn_dim` | u32 | 0 | FFN width of dense (non-MoE) layers |
| `n_experts` | u32 | 0 | routed experts per MoE layer E |
| `top_k` | u32 | 0 | experts per token K |
| `expert_ffn_dim` | u32 | 0 | routed expert FFN width F |
| `shared_ffn_dim` | u32 | 0 | total shared-expert FFN width (0 = none) |
| `shared_gate` | u32 | 0 | 1 = shared output scaled by `sigmoid(shexp_gate_inp · x)` (Qwen2-MoE) |
| `score_fn` | u32 | 0 | 0 = softmax over all E logits, 1 = sigmoid |
| `score_bias` | u32 | 0 | 1 = `moe_router_bias` added to scores for *selection only* (DeepSeek-V3 `e_score_correction_bias`) |
| `n_group` | u32 | 1 | group-limited routing groups |
| `topk_group` | u32 | 1 | groups kept |
| `norm_topk_prob` | u32 | 0 | 1 = renormalise selected weights to sum to 1 |
| `routed_scale` | f32 | 1.0 | multiplier on routed weights (DeepSeek `routed_scaling_factor`) |
| `emb_scale` | f32 | 1.0 | embedding multiplier |
| `residual_scale` | f32 | 1.0 | `h += residual_scale * sublayer(h)` |
| `logit_scale` | f32 | 1.0 | logits multiplier (Granite: `1/logits_scaling`) |
| `tie_embeddings` | u32 | 0 | 1 = LM head is `tok_embd` |
| `layer_kind` | u8[] | all 1 if n_experts>0 else all 0 | per layer: 0 dense FFN, 1 MoE |
| `bos_id` | u32 | 0xFFFFFFFF | |
| `eos_ids` | u32[] | empty | |
| `tokenizer` | str | `""` | path of tokenizer.json relative to the container's directory |
| `chat_template` | str | `""` | Jinja chat template copied from the source model |
| `source` | str | `""` | provenance (HF repo id / "synthetic") |
| `expert_dtype` | u32 | — | informational: dtype of most slabs |

## 4. Tensor directory

`n_tensors` entries of exactly 128 bytes:

| off | type | name |
|----:|------|------|
| 0   | char[80] | name, NUL-padded (max 79 chars) |
| 80  | u32 | dtype (§6) |
| 84  | u32 | ndim (1..4) |
| 88  | u32[4] | shape, outermost first (`[rows, cols]` for matrices); unused = 1 |
| 104 | u64 | offset (absolute, 64-byte aligned) |
| 112 | u64 | nbytes |
| 120 | u64 | reserved = 0 |

Matrices are stored **row-major**, `shape = [out_features, in_features]`
(the PyTorch `nn.Linear.weight` convention). A quantized matrix stores each row
as `in_features/64` consecutive blocks; `in_features` must be a multiple of 64
for Q8/Q4 (writers fall back to F16 otherwise). `nbytes` must equal
`rows * row_bytes(dtype, cols)`.

### 4.1 Canonical tensor names

Global: `tok_embd [V,D]`, `out_norm [D]`, `lm_head [V,D]` (absent when tied),
`rope_inv_freq [rope_dim/2]` (F32, **present iff rope_dim > 0**; the converter computes it
with the source framework's own RoPE-scaling code so the engine never needs to
know YaRN/NTK/Llama3 variants).

Per block `blk.{i}.` (i = 0..n_layers-1):

| name | shape | when |
|------|-------|------|
| `attn_norm`, `ffn_norm` | [D] | always |
| `attn_q` | [H*hd, D] | GQA; MLA when q_lora_rank = 0 (then [H*(nope+rope), D]) |
| `attn_k`, `attn_v` | [Hkv*hd, D] | GQA |
| `attn_q_bias` | [H*hd] | qkv_bias (GQA only; MLA ignores qkv_bias and qk_norm) |
| `attn_k_bias`, `attn_v_bias` | [Hkv*hd] | qkv_bias |
| `attn_q_norm` | [hd] (qk_norm=1) or [H*hd] (qk_norm=2) | qk_norm |
| `attn_k_norm` | [hd] (qk_norm=1) or [Hkv*hd] (qk_norm=2) | qk_norm |
| `attn_q_a` | [q_lora, D] | MLA, q_lora_rank>0 |
| `attn_q_a_norm` | [q_lora] | MLA, q_lora_rank>0 |
| `attn_q_b` | [H*(nope+rope), q_lora] | MLA, q_lora_rank>0 |
| `attn_kv_a` | [C+rope, D] | MLA (`kv_a_proj_with_mqa`) |
| `attn_kv_a_norm` | [C] | MLA |
| `attn_kv_b` | [H*(nope+v), C] | MLA; per head h rows `h*(nope+v) .. +nope` are W_UK, next `v` rows are W_UV |
| `attn_o` | [D, H*hd] (GQA) / [D, H*v] (MLA) | always |
| `ffn_gate`, `ffn_up` | [Fd, D] | layer_kind = 0 |
| `ffn_down` | [D, Fd] | layer_kind = 0 |
| `moe_router` | [E, D] (F32) | layer_kind = 1 |
| `moe_router_bias` | [E] (F32) | score_bias |
| `shexp_gate`, `shexp_up` | [Fs, D] | shared_ffn_dim > 0 |
| `shexp_down` | [D, Fs] | shared_ffn_dim > 0 |
| `shexp_gate_inp` | [1, D] | shared_gate and shared_ffn_dim > 0 |

Norm weights, biases, `rope_inv_freq`, `moe_router` and `moe_router_bias` are
always F32. Other matrices may be F32, F16, BF16, Q8 or Q4.

## 5. Expert directory and slabs

`n_layers * n_experts` entries of 32 bytes, index `layer * n_experts + e`:

| off | type | name |
|----:|------|------|
| 0  | u64 | offset (absolute, multiple of 4096) |
| 8  | u64 | nbytes (slab size, multiple of 4096; 0 for dense layers) |
| 16 | u32 | dtype of the three matrices in this slab |
| 20 | u32 | flags (bit 0: aliased — synthetic benchmark containers may point several entries at one physical slab) |
| 24 | u64 | reserved = 0 |

Different experts MAY use different dtypes (per-expert mixed precision).

The preamble, metadata, directories, tensors and physical slabs must not overlap.
Entries sharing an offset must have equal `nbytes` and `dtype`, and at most one of them
may have flag bit 0 clear.

A slab holds one routed expert's SwiGLU FFN: `gate [F,D]`, `up [F,D]`, `down [D,F]`.

```
off_gate = 0
off_up   = align64(off_gate + F * row_bytes(dtype, D))
off_down = align64(off_up   + F * row_bytes(dtype, D))
end      = off_down + D * row_bytes(dtype, F)
nbytes   = align4096(end)          (padding bytes are zero)
```

Expert output: `down( silu(gate·x) ⊙ (up·x) )`.

## 6. Dtypes and block encodings

| id | name | layout | bits/weight |
|---:|------|--------|------------:|
| 0 | F32  | IEEE-754 binary32 | 32 |
| 1 | F16  | IEEE-754 binary16 | 16 |
| 2 | BF16 | bfloat16 | 16 |
| 3 | Q8   | block of 64: `u16 d (f16)`, `i8 q[64]` = 66 bytes; `w = d*q` | 8.25 |
| 4 | Q4   | block of 64: `u16 d (f16)`, `u8 qs[32]` = 34 bytes; byte `j` holds element `j` in its low nibble and element `j+32` in its high nibble, each stored as `q+8` with `q ∈ [-8,7]`; `w = d*q` | 4.25 |
| 5 | I32  | int32 | 32 |
| 6 | U8   | uint8 | 8 |

`row_bytes(F32,n)=4n`, `F16/BF16: 2n`, `Q8: 66*n/64`, `Q4: 34*n/64`.

Reserved for future versions: 7 = Q3 (E8-lattice / Hadamard-rotated), 8 = Q2.

## 7. Companion files

* `<name>.tokenizer.json` — Hugging Face tokenizer, referenced by `tokenizer`.
* `<name>.usage` — expert heat profile (written by the engine; §8).
* Mirrors are byte-identical copies of the `.hearth` file on other drives.

## 8. Usage (heat) profile

```
u32 magic 0x53555248 ("HRUS")   u32 version 1
u32 n_layers  u32 n_experts      u64 tokens_observed
f32 heat[n_layers * n_experts]   (activation counts; may be decayed)
```

## 9. Routing trace (for the simulator)

```
u32 magic 0x52545248 ("HRTR")   u32 version 1
u32 n_layers  u32 n_experts  u32 top_k  u32 n_moe_layers
then per evaluated token: u16 ids[n_moe_layers * top_k]   (MoE layers in order, each top-k sorted by rank)
```

Readers (route replay) require version 1, `n_experts`, `top_k` and `n_moe_layers`
equal to the model's, and `n_layers >= n_moe_layers`.
