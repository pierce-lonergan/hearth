# Hearth numerics (normative)

Hearth guarantees **bit-identical output** no matter *where* a weight was when it
was used (NVMe, DRAM cache, pinned, prefetched), how many threads ran, which SIMD
ISA was dispatched, or whether tokens were evaluated one at a time or as a batch.
That only works if every kernel follows the *canonical* order of floating-point
operations below. Any SIMD/GPU kernel must reproduce the scalar definition
exactly. In particular:

* **No FMA contraction** for canonical float math. Build flags: MSVC `/fp:precise`
  (default, no `/fp:contract`); GCC/Clang `-ffp-contract=off`. Never `-ffast-math`.
* Integer partial sums are exact; float accumulation order is fixed.
* `expf`, `sinf`, `cosf`, `sqrtf` come from the platform libm and are called the
  same way from every code path.

## 1. Canonical reductions

`sum16(a[0..n))` — 16 lane accumulators `L[0..15] = 0.0f`;
for `i` in `0..n-1`: `L[i & 15] = L[i & 15] + a[i]`; then for `s` in `8,4,2,1`:
for `j < s`: `L[j] = L[j] + L[j+s]`; result `L[0]`.

`dot16(w, x, n)` — same as `sum16` with `a[i] = w[i] * x[i]` (product rounded to
f32, then added — never fused).

## 2. Activation quantization (for Q8/Q4 weights)

Input `x` (f32, length n, n % 64 == 0) becomes `n/64` blocks
`hx_act_q8 { float d; int32 sum; int8 q[64]; }`:

```
amax = max_i |x_i|                       (exact)
d    = amax / 127.0f
id   = d != 0 ? 1.0f / d : 0.0f
q_i  = (int8) nearbyintf(x_i * id)       (round-half-to-even; |q_i| <= 127)
sum  = Σ q_i                             (exact int32)
```

## 3. Matrix-vector product `y = W x`

**Q8 / Q4 weights** (with activation blocks from §2), for each row r:

```
acc = 0.0f
for g in 0 .. n/64-1:
    isum = Σ_i wq[g][i] * xq[g][i]        exact int32;  Q4: wq = nibble - 8
    acc  = acc + (float)isum * (dw[g] * dx[g])     dw = f16→f32 of block scale
y[r] = acc
```

(For Q4 an implementation may compute `Σ nibble*xq - 8*sum` — it is the same exact integer.)

**F32 / F16 / BF16 weights**: `y[r] = dot16(row_r_as_f32, x, n)`.

Bias (if any) is added afterwards: `y[r] = y[r] + b[r]`.

A NaN result is stored as the canonical quiet NaN `0x7fc00000`. (IEEE 754 does not
fix which NaN operand an add/mul propagates and compilers commute SIMD operands, so
raw payloads would depend on ISA and batching.) Bit-identity therefore covers NaNs.

**Batched** `Y = W X` (T tokens) must produce, for every (token, row), exactly the
value the single-token matvec produces. This is what makes prefill, speculative
verification and decode agree bit for bit.

## 4. Elementwise ops

* RMSNorm: `ms = sum16(x_i*x_i) / (float)n`; `r = 1.0f / sqrtf(ms + eps)`;
  `y_i = (x_i * r) * w_i`.
* SiLU: `x / (1.0f + expf(-x))`. SwiGLU: `silu(g_i) * u_i`.
* Sigmoid: `1.0f / (1.0f + expf(-x))`.
* Softmax over `x[0..n)`: `m = max`; `e_i = expf(x_i - m)`; `s = sum16(e)`; `p_i = e_i / s`.
* RoPE (pair index `j < rope_dim/2`, position `p`): `θ = (float)p * inv_freq[j]`;
  `c = cosf(θ) * rope_attn_factor`; `s = sinf(θ) * rope_attn_factor`;
  `(a, b) → (a*c - b*s, b*c + a*s)` where `(a,b)` = `(x[j], x[j+rope_dim/2])`
  for NEOX style, `(x[2j], x[2j+1])` for GPT-J style. Only the first `rope_dim`
  dims of each head are rotated. MLA: the first `rope_dim` (<= `qk_rope_dim`) dims of
  `q_pe` and `kpe` are rotated; `rope_dim = 0` rotates nothing.

## 5. Forward pass (one token at position `pos`)

```
h = emb_scale * dequant(tok_embd[token])
for layer i:
    x = rmsnorm(h, attn_norm)
    h = h + residual_scale * attention(x, pos)        (elementwise: h_i + (rs * a_i))
    x = rmsnorm(h, ffn_norm)
    h = h + residual_scale * ffn_i(x)
h = rmsnorm(h, out_norm)
logits = logit_scale * (lm_head · h)                  (tok_embd if tied)
```

### 5.1 GQA attention
`q = Wq x (+bq)`, `k = Wk x (+bk)`, `v = Wv x (+bv)`; qk_norm (per head or full
width) is applied before RoPE; RoPE on q and k; append (k, v) to the cache.
For head h, kv head `g = h / (H / Hkv)`:
`score_t = attn_scale * dot16(q_h, k_{g,t}, hd)` for t = 0..pos; `p = softmax(score)`;
`o_h[d] = Σ_t p_t * v_{g,t}[d]` accumulated sequentially in t starting from 0.0f.
`out = Wo concat_h(o_h)`.

### 5.2 MLA attention (absorbed form)
```
q   = q_lora ? Wqb · rmsnorm(Wqa x, q_a_norm) : Wq x     per head: [q_nope (nope) | q_pe (rope)]
kva = Wkva x;  c = rmsnorm(kva[0:C], kv_a_norm);  kpe = kva[C:C+rope]
RoPE(q_pe per head), RoPE(kpe);  cache (c, kpe) for this position
per head h:  W_UK = attn_kv_b rows [h*(nope+v), +nope)   W_UV = next v rows
   q_lat[c] = Σ_n q_nope[n] * W_UK[n][c]     (n ascending, sequential, from 0.0f)
   score_t  = attn_scale * (dot16(q_lat, c_t, C) + dot16(q_pe, kpe_t, rope))
   p = softmax(score);  o_lat[c] = Σ_t p_t * c_t[c]  (sequential in t)
   o_h = W_UV · o_lat   (matvec, canonical §3)
out = Wo concat_h(o_h)
```
This is algebraically identical to the reference (non-absorbed) MLA; results
match the reference within float tolerance, not bit-exactly.

### 5.3 FFN
Dense: `down(swiglu(gate x, up x))`.

MoE:
```
logits = router · x                                (F32, dot16)
score  = softmax(logits) | sigmoid(logits)
sel    = score + router_bias (score_bias) | score
if n_group > 1: gscore_g = sum of the top-2 sel in group g; keep the topk_group best
                groups (ties → lower index); sel of experts in dropped groups := 0.0f
pick the top_k experts by sel, descending (ties → lower expert index)  → ids[0..K)
w_j = score[ids_j];  if norm_topk_prob: w_j = w_j / (Σ_j w_j + 1e-20f)
w_j = w_j * routed_scale
out = 0; for j in 0..K-1 (rank order): out_i = out_i + w_j * expert_{ids_j}(x)_i
if shared: s = shexp(x); if shared_gate: s = s * sigmoid(dot16(shexp_gate_inp, x))
           out_i = out_i + s_i
```
The rank-order accumulation is why expert results that arrive from NVMe in a
different order still give identical output: each expert writes its own buffer
and the sum is done in rank order at the end.

## 6. Quantization (weights)

The C quantizer (`hearth_quantize`) is authoritative and deterministic
(scalar code, identical on every ISA). Q8: per block `d = amax/127` (stored f16),
`q = nearbyint(x / d_f16)` clamped to [-127,127]. Q4: candidate-scale search
minimising squared error over each 64-block (see `engine/src/quant.c`), stored
as f16 scale + nibbles per FORMAT.md §6. Dequantization is exact: `w = f16(d) * q`.
