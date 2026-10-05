# The Hearth simulator (`hearth.sim`)

`hearth.sim` predicts decode speed and expert-cache behaviour for any MoE model
in `hearth.presets` on any hardware profile. It drives a cache simulation from a
**routing trace** (recorded by the engine, or synthetic) and turns the per-layer
hit/miss counts into time with an explicit bandwidth model. Use it to compare
cache policies and features, size a machine, or decide what to build next.

> **Every number this tool prints is SIMULATED, not measured** (INV-HONEST). Until
> the scalars of §4 are fitted to engine measurements (task T09), treat absolute
> tokens/s as order-of-magnitude estimates. Relative comparisons on the same trace
> are more trustworthy, because they share every assumption.

Only numpy is needed. A 2000-token Kimi-K2 trace (60 MoE layers × top-8 = 960k
expert requests) runs through all five online policies in about 2.6 s on this-pc's
Ryzen 9 9950X; see §9 for conditions.

## 1. Quick start

```
python -m hearth sim --model kimi-k2 --hw this-pc                     # one run, defaults below
python -m hearth sim --model kimi-k2 --cache-gb 40 --compare-policies
python -m hearth sim --model kimi-k2 --cache-gb 40 --spec-k 4 --spec-alpha 0.6 --prefetch 0.8
python -m hearth sim --model kimi-k3 --hw laptop-16gb --feasibility
python -m hearth sim --model kimi-k2 --levers --markdown              # one-at-a-time what-ifs
python -m hearth sim --model kimi-k2 --sweep-zipf 0,0.5,1,1.4 --sweep-cache 8,16,40
python -m hearth sim --model kimi-k2 --trace run.hrtr                 # a recorded trace
python -m hearth sim --list                                           # presets, profiles, policies
```

`python -m hearth.sim` works too. The runtime wires `python -m hearth sim` to
`hearth.sim.main(argv)`. Hardware fields can be overridden from the CLI
(`--ram-gib`, `--dram-gbs`, `--nvme 2x6.5`, `--nvme-latency-us`, `--io-cap-gbs`,
`--int8-tops`, `--os-reserve-gib`, `--vram-gib`, `--vram-gbs`, `--pcie-gbs`), as can the
model encoding (`--expert-bits`, `--dense-bits`, `--context`, `--kv-bytes`).
`--json` emits everything machine-readably.

Out-of-range inputs are rejected with an error rather than simulated:

* non-positive bandwidths, capacities or int8 rate, and zero drives;
* LFU decay outside (0, 1], and `pin_fraction` outside [0, 0.9] (the `hx_store` range);
* acceptance or recall outside [0, 1], also when the feature is otherwise off
  (`--spec-alpha 1.5` without `--spec-k`);
* negative cache, context, sample counts, draft length (`--spec-k`) or prefetch
  extras, and fewer than one I/O thread;
* non-integer draft length, I/O threads, sample count or warm-up (`4.0` is accepted
  as 4);
* a warm-up after which no forward step starts, so nothing would be measured;
* bits/weight outside (0, 32];
* a trace whose `n_layers` is below its MoE layer count or does not fit the
  file's u32 field;
* calibration points whose measured tok/s is not a finite number > 0 (§4).

```python
from hearth.sim import simulate, sweep, calibrate, Spec, Prefetch, Lossy
r = simulate("kimi-k2", "this-pc", policy="lfu", cache_gb=40, prefetch=Prefetch(0.8, 0), spec=Spec(4, 0.6))
r.tok_s, r.hit_rate, r.bytes_per_token["nvme"], r.bottleneck, r.time_per_token_ms
rs = sweep({"policy": ["lru", "lfu", "belady"], "cache_gb": [8, 16, 40]}, model="kimi-k2")
```

Defaults: a synthetic trace of 2000 tokens with Zipf 1.1 and reuse 0.25, seed 0.
The first 10% of tokens are warm-up and are excluded from statistics. Policy `lfu`
with decay 0.995 per token. The cache gets all RAM left after the backbone, KV
cache, 0.5 GiB of workspace and the OS reserve. Experts are Q4 (4.25 bpw); the
backbone and embeddings are Q8 (8.25 bpw), the converter's defaults. Context is
1024 with f32 KV. Speculative decoding, the GPU and all lossy knobs are off.

**Prefetch is off by default in the simulator, but the engine's default is
`HEARTH_PREFETCH_SHARED`** (hearth.h, `Engine(prefetch="shared")`). The simulator
cannot know that predictor's recall, so it does not assume one. For a run that
resembles the engine's default, pass `--prefetch 0.8`; that recall is an
*assumption*, and §8 labels it as one wherever it is used.

## 2. Inputs

### 2.1 Models

The simulator reads the shapes in `hearth.presets` (read-only for this module).
Backbone bytes are counted per FORMAT.md §4.1:

* matrices at `--dense-bits`;
* routers (`moe_router`, F32) and norms at 4 bytes;
* the embedding table resident at `embed_bits`, but only one row is read per token;
* the LM head read in full every step;
* expert slabs sized exactly per FORMAT.md §5 (64-byte sub-alignment, 4096-byte slab
  alignment). For Q4/Q8 this matches `Shape.expert_bytes`.

For latent MoE (Kimi-K3, `expert_d = 3584`) experts are sized in the latent width.
The two per-layer projections into and out of that space (2·D·expert_d parameters
per MoE layer) are **added** to the backbone, because `Shape.dense_params()` omits
them (see the handover's contract issues).

### 2.2 Hardware profiles

Capacities are in GiB; bandwidths are *achievable* GB/s (10⁹ B/s), not datasheet
peaks. Every field can be overridden.

| profile | RAM GiB | DRAM GB/s | storage | int8 TOPS | GPU | OS reserve GiB |
|---|---|---|---|---|---|---|
| `this-pc` | 61.6 | 60 | 1 × 6.5 GB/s, 80 µs, 2 TB | 5.0 | RTX 5070 12 GiB, 672 GB/s, PCIe 50 GB/s | 6 |
| `laptop-16gb` | 16 | 45 | 1 × 4.5 GB/s, 1 TB | 1.5 | none | 4 |
| `desktop-32gb` | 32 | 38 | 1 × 5.0 GB/s, 2 TB | 1.0 | RTX 3060 12 GiB, 360 GB/s | 4 |
| `workstation-128gb-2nvme` | 128 | 55 | 2 × 12 GB/s, 4 TB each | 5.0 | 24 GiB, 1000 GB/s | 6 |
| `mac-studio-192gb` | 192 | 250 (CPU) | 1 × 6.0 GB/s, 2 TB | 2.0 | unified, 800 GB/s | 8 |

Assumptions behind them:

* **this-pc**:
  * Ryzen 9 9950X, 64 GB DDR5-6000 (61.6 GiB usable). AIDA64 reads about 78 GB/s;
    streaming kernels are assumed to reach about 60.
  * Samsung 990 PRO: 7.45 GB/s spec, about 6.5 achievable with direct I/O.
  * int8: about 20 TOPS of AVX-512 VNNI peak, of which Q4/Q8 matmul kernels are
    assumed to sustain about 25%.
  * RTX 5070: PCIe 5.0 x16, about 50 GB/s achievable.
* **laptop-16gb**: 8 cores, dual-channel DDR5-5600 (89.6 GB/s peak). One PCIe 4.0
  drive; thermal throttling is ignored.
* **desktop-32gb**: 8-core AVX2 CPU (no VNNI), DDR4-3200 dual channel (51.2 GB/s peak).
* **workstation-128gb-2nvme**:
  * AM5 16-core with 4×32 GB DDR5. Four DIMMs run at about 5200 on two channels.
  * Two PCIe 5.0 drives on CPU lanes, about 12 GB/s achievable each.
  * GPU at PCIe x8.
* **mac-studio-192gb**: M2 Ultra.
  * 800 GB/s is the unified-memory peak. The CPU cores alone are assumed to reach
    about 250 GB/s, which is the most uncertain number here.
  * Internal SSD at about 6 GB/s. The engine has no Metal backend yet (R02).

Storage bandwidth is `nvme_count × nvme_gbs`, optionally capped by `io_cap_gbs`.
Use the cap for drives that share a chipset uplink. Mirrors are full copies, so the
model file must fit on *each* drive.

`pcie_gbs` matters only in the GPU what-ifs. Each host-to-GPU hand-off costs
`gpu_sync_us` plus the time to move the step's activations (n positions × d_model ×
4 bytes) at `pcie_gbs`. A discrete GPU therefore needs `pcie_gbs > 0`. Unified
memory has no PCIe term.

### 2.3 Routing traces

**Real traces** use the format of FORMAT.md §9: a `HRTR` header of 6 × u32, then
u16 ids `[token][moe_layer][rank]`. They are recorded with `Engine.trace_start` /
`hearth_trace_start`, or from `Reference.routing_history()` wrapped in
`hearth.sim.Trace(ids, n_experts, n_layers)`. `Trace.save/load` round-trip the
format exactly. Loading validates the magic, version, header consistency and body
size. Every trace, loaded or built in memory, must also satisfy:

* integer ids in `[0, n_experts)`;
* no expert twice in one token's top-k, because routing is without replacement;
* bounded geometry: `n_experts ≤ 65536`, `moe_layers × n_experts ≤ 2^24` and
  `moe_layers × top_k ≤ 2^20`. A corrupt header therefore cannot make the
  statistics allocate exabytes.

The **synthetic generator** (`hearth.sim.synthetic`, deterministic per seed) works
per layer:

* Expert popularity `p(rank r) ∝ (r+1)^(-zipf)` over a random permutation of the
  experts. `zipf = 0` is perfectly balanced routing.
* For each token, each of the previous token's experts is kept independently with
  probability `reuse` (temporal locality).
* The remaining slots are filled by sampling *without replacement* proportional to
  popularity (successive sampling, the same distribution as Gumbel-top-k).
  Implementation: an i.i.d. stream deduplicated in order; rows that run short are
  refilled exactly.
* Ranks are ordered by a noisy popularity score (log p + Gumbel).
* Layers are independent of each other. The measured reuse rate is ≥ `reuse`,
  because popular experts recur by chance.

**Statistics** (`Trace.stats()`, `--stats`):

* per-layer entropy, normalised by log2 E;
* reuse rate from the previous token;
* access mass in the top x% of experts, per layer and pooled over all
  (layer, expert) slabs;
* slabs needed to cover 50/80/90/95% of all loads, which is a direct cache-sizing
  number;
* mean distinct experts per layer over w consecutive tokens. This union is what a
  speculative verification step must load.

## 3. The model

### 3.1 Forward steps and groups

The trace is cut into **forward steps**. In plain decode each step is one token.
With speculative decoding (`Spec(k, alpha)`), a step verifies the window
`[pos, pos+k]` in one batched `hearth_eval`:

* The number of accepted drafts `a` is geometric: P(a ≥ j) = α^j, truncated at the
  window.
* The step produces `a+1` tokens (the accepted drafts plus the verifier's token), and
  the next window starts at `pos+a+1`.
* The expected tokens per step is (1−α^(k+1))/(1−α).
* Rejected positions are routed as the trace's true next tokens. That approximates
  the real draft tokens' routing.

Each (step, MoE layer) is a **group**: the distinct experts the step needs in that
layer. That is `top_k` experts for decode, and the union over the window for
verification. Batched evaluation reads each needed expert once for all tokens in
the window (INV-DET-2 makes this legal).

### 3.2 Cache simulation

Keys are `(moe_layer, expert)` slabs. The slot count is `floor(cache_GiB · 2³⁰ / slab)`.
It is raised to the engine minimum `2·top_k + io_threads + 2` (hx_store.h) and
capped at the total number of experts.

For every group, each needed expert is classified as:

| class | meaning |
|---|---|
| `vram` | in the static VRAM tier (GPU what-if) |
| `hit` | resident in DRAM before the step needed it |
| `pfhit` | resident because the previous layer's prefetch brought it in |
| `miss` | demand read from storage |
| `skip` | not evaluated (lossy what-if) |

Rules shared by every online policy mirror `hx_store`:

* every demand miss takes a slot; there is no bypass;
* the group's experts are protected while it computes (refcount > 0);
* prefetched experts take slots and stay protected until they are used or the
  forward step ends (`hx_store_tick`). A wrong guess for layer l+1 therefore stays
  protected for the rest of the step. As in `engine/src/store.c`, a demand miss
  evicts such a slab only as a last resort, when nothing unprotected is left;
* prefetches never evict a protected slab; a prefetch that would need one is
  dropped;
* if every resident slab is in use by the current group (or pinned), a demand read
  goes to a scratch buffer and is counted in `bypass`.

Implementation note: unused prefetched slabs are *held* by the policy. They are kept
out of the LRU order and the LFU heap until they are used or the step ends. An
unused one then returns to the LRU position of its admission, which is when the
engine stamps a prefetched slot. A cache full of held slabs therefore rejects a
prefetch without scanning, and the demand's last resort takes the oldest (LRU) or
coldest (LFU) held slab directly. §9 gives the speed-up, and how the change was
checked to give identical results.

| policy | rule |
|---|---|
| `lru` | global LRU over all slabs (engine `HEARTH_POLICY_LRU`) |
| `lru-layer` | LRU partitioned per MoE layer, slots split evenly |
| `lfu` | decayed heat, as in the engine's `HEARTH_POLICY_LFU` (details below) |
| `pinned` | static: the hottest slabs by profile are pinned; a small LRU scratch area (`max(engine minimum, max group + top_k)`) handles the rest |
| `lfu-pinned` | `pin_fraction` (0.5) of slots pinned from the profile; LFU over the rest, heat seeded from the profile |
| `belady` | offline optimum of a relaxed problem: an upper bound, not a policy (details below) |

`lfu` in detail:

* Heat is h(t) = Σ over uses s of decay^(t−s), with decay 0.995 per token. It is
  stored scaled by decay^(−t) and rescaled near 1e100, so ordering is exact with no
  per-token sweep.
* Heat is kept for every slab, cached or not, like `hx_store_heat`.
* The coldest unprotected slab is evicted exactly, or the coldest of
  `--lfu-samples N` random residents to mimic the engine's sampled eviction.

`belady` in detail. It solves a *relaxed* problem, which makes its hit rate an upper
bound for every online policy:

* **The relaxed problem.** The state before each group may be any set of at most
  `slots` slabs. Hits are the group's slabs that are resident when it starts. The
  next state is any subset of (state ∪ group). There is no protection, so an in-use
  slab may be evicted and a missed slab may bypass the cache. The cache may start
  with any contents.
* **Why it is a bound.** Every run of an online policy without prefetch is one
  feasible schedule of this problem. That includes protection, pinning, pins loaded
  at open, and bypass when every slab is protected. So Belady's optimum is ≥ each
  of them.
* **The rule.** After each group, Belady keeps the `slots` slabs whose next use is
  earliest. This is MIN's exchange argument applied group by group. Ties inside one
  future group are symmetric, so they do not matter.
* **The measured window.** Statistics count only the steps after the warm-up, and
  Belady maximises hits over exactly that window. Before it, a slab's next use is
  its first use inside the window. The cache starts with the `slots` slabs that are
  used earliest there.
* **The VRAM tier.** With `gpu="dense+experts"`, VRAM slabs are served before the
  DRAM cache is consulted, in every policy, so they never enter it. Belady therefore
  solves the relaxed problem over the other slabs only, and VRAM slabs are kept out
  of its warm start. Generation 2 warm-started Belady with them. They then held
  DRAM slots for the whole run, and online policies could beat the "bound"
  (verifier: lru-layer 68.2% against Belady 63.4% on olmoe with 3 GiB of VRAM).
* **Tests.** They compare Belady with an exhaustive search of the relaxed problem,
  with top-k > 1, several layers, speculative windows, a warm-up prefix and a VRAM
  tier. They also check it against every online policy on random and synthetic
  traces, with and without the VRAM tier.
* **Not a bound with prefetch or `skip_miss_rank`.** Those change what reaches the
  cache. `simulate` warns when Belady is combined with them.

The **profile** for `pinned`, `lfu-pinned`, the VRAM tier and cold experts is the
access count over the warm-up prefix by default. Alternatives are `--profile oracle`
(the whole trace, flagged as seeing the future) or a `.usage` file (FORMAT.md §8).

### 3.3 Prefetch

`Prefetch(recall, extra)` models next-layer prediction, issued while layer l
computes, for layer l+1 of the same step:

* each expert layer l+1 really needs is predicted with probability `recall`;
* the predictor also emits `needed − found + extra` wrong experts, drawn without
  replacement from the layer's observed popularity. There are never more than the
  layer's remaining experts;
* predictions that are not resident are inserted into the cache (they occupy slots)
  and read at low priority. They stay protected until used or until the step ends
  (§3.2). Many wrong guesses therefore crowd out other slabs for the rest of the
  step.

The prefetch reads for layer l get the storage channel from the moment layer l−1's
demand reads finish until layer l's MoE phase starts: `slack(l−1) + A(l)`, below.
The fraction `f` of the prefetched bytes that fits in that window arrives in time.
The rest of the *needed* ones is promoted to demand reads, as `hx_store_try_acquire`
does. The rest of the wrong guesses is cancelled. For layer 0 of a step nothing is
predicted.

### 3.4 Timing equations

Symbols:

| symbol | meaning |
|---|---|
| n | positions in the step (1, or k+1 for verification) |
| L | MoE layers |
| S, S_c | slab bytes, hot and cold |
| P_e | parameters of one expert |
| BW_d | `dram_gbs · dram_eff` |
| BW_io | `io_gbs · io_eff` |
| OPS | `cpu_int8_tops · compute_eff` |
| λ | storage latency |
| BW_bb | where the backbone is read from: BW_d, or the VRAM bandwidth in GPU modes |

```
T_step  = T_fixed + T_global + Σ_l ( A_l + T_moe,l )
T_fixed = overhead_ms + layer_overhead_us · n_layers
sync    = 2 · (gpu_sync_us + n · 4 · d_model / BW_pcie)   GPU modes only, per MoE layer; no PCIe term on unified memory
T_global = max(B_global / BW_bb, n·2·P_global / OPS)      LM head, leading dense layers (+ their KV)
A_l      = max(B_layer  / BW_bb, n·2·P_layer  / OPS)      attention, norms, router (F32), latent projections,
                                                          KV read = context · kv_bytes_per_position
per expert:  c = max(S / BW_d, τ·2·P_e / OPS),  τ = n·top_k / |group|   (tokens per expert)
W0 = T_shared + (h + f·p) · c                             shared experts + experts already in DRAM
m  = d + (1−f)·p                                          demand misses incl. late prefetches;  M = m·S
T_moe = W0                                                if m = 0
      = max( W0 + m·c,  λ + M/BW_io + c,  λ + S/BW_io + m·c )   otherwise
slack_l = T_moe,l − (λ + M_l/BW_io)                       storage idle time after layer l's demand reads
f_l     = min(1, (slack_{l−1} + A_l) · BW_io / B_prefetch,l)
tok/s   = Σ accepted tokens / Σ T_step                    over steps after the warm-up
```

`T_moe` is the exact completion time of a two-stage pipeline:

* stage 1 is storage reads, serving misses back to back after one latency;
* stage 2 is compute: the shared experts and hits first, then each miss as it lands,
  in completion order (ARCHITECTURE.md).

Shared-expert compute overlaps outstanding reads, as in the engine's MoE schedule.
Compute is a roofline per expert: whichever is larger of the time to stream the
slab from DRAM and the time to do the int8 work. The int8 term matters only for
batched verification.

**Attribution.** Each step's time is split into `overhead`, `dram` (bandwidth-bound
compute), `cpu` (int8-bound excess), `nvme` (time the compute waited for storage)
and `gpu`. The parts sum exactly to `T_step`, which is tested. The `bottleneck` is
the largest part.

**Bytes per accepted token.**

* `nvme` = demand bytes (including promoted late prefetches) + prefetch bytes that
  arrived in time. Of the latter, `nvme_prefetch_wasted` were wrong guesses.
* `dram` = every expert computed on the CPU, plus the backbone and KV.
* `vram` = what the GPU read.

`hit_rate + prefetch_rate + miss_rate + skip_rate = 1` over expert loads. `hit_rate`
counts loads that needed no storage read, VRAM tier included.

### 3.5 Feasibility

`hearth.sim.feasibility(costs, hw)` and `--feasibility` report:

* **Backbone location.** RAM; VRAM in GPU modes (the embedding table stays in RAM,
  plus 1 GiB of VRAM headroom); or unified memory.
* **Expert cache.** RAM left for it = RAM − OS reserve − 0.5 GiB workspace − backbone − KV.
  The KV cache is sized at `max_seq` = 4096.
* **Minimum.** The engine-minimum cache, and the minimum RAM to run at all.
* **Recommended cache.** All of the RAM left. On every trace simulated here the hit
  rate kept rising with cache size, so no smaller "knee" is recommended.
* **Disk.** The model file size at Q3/Q4/Q8/BF16 experts, and whether it fits one
  drive.

`simulate` always runs. An infeasible configuration is flagged in `Result.feasible`
and in the warnings with the reason, and the report suggests a Q4 backbone where
that helps. Tables mark such rows `(INFEASIBLE)`.

A configuration is infeasible when the model does not fit (the reasons above), or
when the cache actually allocated is larger than the RAM left for it. "Allocated"
means after the engine minimum and the cap at all experts. So `--cache-gb 200` for
Kimi-K2 on this-pc is infeasible: 200 GiB is more than the 42.8 GiB left. But
`--cache-gb 10000` for a model whose experts all fit in 3.2 GiB is feasible, and a
warning says that every expert is cached.

### 3.6 What-if knobs

**Lossy** (`Lossy(...)`, `--lossy-*`). They change model output, are off by default,
and are labelled `LOSSY` everywhere (INV-LOSSLESS-DEFAULT):

* `topk=k'` keeps only the first k' ranks;
* `skip_miss_rank=r` skips experts routed only at ranks ≥ r when they are not cached
  (cache-aware routing, R06);
* `cold_bits=b, cold_frac=x` stores the coldest x of experts, by profile, at b
  bits/weight (R03). Slots stay full-size, as in the engine, so this cuts bytes but
  not slot count.

Q3 experts for all slabs is simply `--expert-bits 3.25`. That is lossy against Q4
and on the roadmap (R03). Any encoding below the converter defaults adds a
`LOSSY ...` entry to `Result.lossy`: experts below 4.25 bpw, or a backbone or
embeddings below 8.25 bpw. A Q4 backbone is therefore labelled LOSSY too.

**GPU** (`gpu="dense"` or `"dense+experts"`) is roadmap R01. The engine is CPU-only
today.

* `dense`: the backbone is read at VRAM bandwidth, with two activation hand-offs
  per MoE layer (`sync` in §3.4).
* `dense+experts`: also pins the hottest slabs into the VRAM left over. The GPU
  computes them concurrently with the CPU's experts:
  `T_moe = max(T_cpu, n_vram·S/BW_vram + sync)`.
* On unified memory, `dense+experts` computes all cached experts at the GPU's
  memory bandwidth instead.
* Asking for a GPU mode on a profile without a GPU is an error.

## 4. Calibration

The defaults are guesses: overhead 1 ms per step, 20 µs per layer, and every
efficiency at 1.0. They live in `Calibration(dram_eff, io_eff, compute_eff,
overhead_ms, layer_overhead_us)`.

`calibrate(points, fit=("dram_eff", "io_eff", "overhead_ms"))` fits the chosen
scalars to measured engine numbers. It minimises the squared error of log tok/s
with Nelder–Mead in log space, restarted twice from its own optimum. `points` is a list
of `(Result, measured_tok_s)`. A point that is not such a pair, or whose measured
tok/s is not a finite number > 0, is rejected with an error that names its index and
label.
Refitting does not rerun the cache simulation: `retime(result, calib)` re-evaluates
only the timing from the stored counts. The tests show it recovers known scalars
from synthetic "measurements" to within 5%, and to within 0.1% in practice.

Procedure for T09:

1. Record a routing trace on the real model with `Engine.trace_start`. On a
   synthetic-weight benchmark container (`hearth.synth.make_shaped`), replay a trace
   with `hearth_route_replay` so routing skew is realistic.
2. Run the engine at several settings. Vary `cache_gb` across the LRU cliff, `policy`
   between lru and lfu, `prefetch` between off and shared, and the I/O-bound vs
   DRAM-bound regime. Record tok/s and `hearth_stats`.
3. Check the cache model first, without timing: compare `cache_hits/misses` from
   `hearth_stats` with the simulator's `hit_rate` on the same trace, using
   `--lfu-samples` equal to the engine's sample count.
4. Fit the scalars with the code below and keep the JSON for `--calib FILE.json`
   (override single fields with `--io-eff` etc.).

```python
pts = [(simulate("kimi-k2", "this-pc", trace="k2.hrtr", cache_gb=g, policy=p), measured[g, p]) for g, p in runs]
cal, report = calibrate(pts)
json.dump(cal.to_dict(), open("calib-this-pc.json", "w"))
```

## 5. Validation

`tests/py/test_sim.py` has 112 tests and runs in about 21–26 s under moderate
machine load (§9). It checks:

* **Belady is the optimum of the relaxed problem (§3.2).**
  * It equals an exhaustive search on 80 random small schedules: top-k from 1 to 4,
    one or two layers, speculative windows in a third of them, and a random warm-up
    prefix. This includes the verifier's two-group counterexample to generation 1.
  * A plain re-implementation without a heap agrees on 2500-token traces, which are
    long enough to exercise the heap rebuild. `next_at` is checked against brute
    force.
* **Belady bounds every online policy over the measured window.** Checked on:
  * 120 random schedules, with pinned slabs, sampled LFU, speculative windows and
    warm-up;
  * a one-layer shape at the engine-minimum 9 slots through `simulate`, including
    the trace on which generation 1's Belady scored 0.692 against 0.697 for LRU (and
    for LRU-per-layer and pinned); the new Belady scores 0.810;
  * six olmoe cache/speculative configurations;
  * with a VRAM expert tier: 90 small random schedules, where Belady also equals the
    exhaustive optimum over the non-VRAM slabs, plus 60 larger ones; and through
    `simulate(gpu="dense+experts")`, the verifier's olmoe reproduction and a tiny
    shape with exactly 7 VRAM slabs.
* **LFU heat.**
  * Direction of the decay: on a hand-worked 7-token sequence, decay 1/2 evicts the
    older-but-more-frequent slab and pure frequency (decay 1) evicts the other one.
    The exact per-token hits are asserted for both.
  * An exact-integer LFU reference (at decay 1/2 the scaled heat is a sum of powers
    of two) agrees hit for hit and bypass for bypass over 1500–2000-token traces.
    Those runs pass 4–6 heat rescales, and include a speculative schedule.
  * The stored scale grows until it passes 1e100 and only then resets, so there is
    no per-token sweep.
  * Sampled eviction with one sample is uniform over the unprotected residents
    (20 000 draws). With protected and held slabs, it follows the expected mixture
    of a uniform sample and the exact fallback.
* **`lru-layer` partitions.** With slots not divisible by the layer count, the first
  layers get one more slot. A cyclic trace checks the resulting per-layer hits.
* **Held prefetches.** Random group / prefetch / step-end sequences are driven through
  the policy interface exactly as `run_cache` drives it. They are compared with a
  naive model (timestamps and an explicit held set) for `lru`, `lru-layer` and exact
  `lfu`: every admission and last-resort victim must match. White-box checks:
  * a cache full of held slabs rejects with an empty LRU order and an empty LFU heap;
  * the LFU last resort picks the coldest held slab not in `prot`, also after a heap
    rebuild;
  * the repeated-REJECT shortcut draws the same random samples as a full search.
* **LRU cyclic pathology.** With the cache below one token's working set, global LRU
  has a hit rate below 0.5% while LFU is above 20%. LRU recovers once the cache holds
  the working set.
* **Byte accounting.** The NVMe and DRAM equalities hold. Evictions equal admissions
  minus slots, with prefetch on. Time attributions sum to the step time in every
  mode. A hand-computed all-hit DRAM-bound step time matches.
* **Timing formulas.** `T_moe` matches §3.4 exactly in each of its three regimes,
  as do the GPU step with its PCIe hand-off and a compute-bound step at half
  `compute_eff`.
* **Prefetch.**
  * Accounting: no wrong-guess bytes at recall 1 with no extras, and no late
    prefetches with unlimited storage.
  * Protection: an unused prefetch survives the next layer's demand misses and
    expires at the step end. A demand evicts one only as a last resort, and never
    bypasses while one exists.
  * Wrong guesses: `extra` is clamped to the layer's remaining experts (10⁶ extras
    finish), and the exact refill draws from the next layer's popularity.
* **Speculative decoding.** Union amortisation: reads per accepted token are
  ≈ L·k / tokens-per-step when every token reuses its experts, and they *increase*
  at low acceptance on independent tokens. The geometric acceptance statistics are
  checked.
* **Input validation.**
  * 31 bad `simulate` inputs raise `ValueError`.
  * 24 bad CLI flag combinations exit with code 2 and an error line, without a
    traceback. These include negative `--spec-k` and `--prefetch-extra`, which
    generation 2 silently treated as "off".
  * A warm-up that leaves no measured step is an error. The "noisy" warning counts
    the tokens actually measured.
  * Integral floats (`Spec(4.0)`, `io_threads=8.0`) become ints.
  * `calibrate` names the bad point.
  * `Trace` rejects an `n_layers` that it could not save or load back.
  * A cache larger than the RAM left is infeasible and flagged in tables.
* **Traces.**
  * Files round-trip.
  * Malformed files, duplicate, negative or non-integer ids, and oversized
    geometries (each bound separately) are rejected.
  * The sampler matches Gumbel-top-k marginals, and entropy, reuse, mass and union
    are checked.
* **Labels.** Encodings below the converter defaults are LOSSY, and `levers()` labels
  follow the actual baseline.
* **Other.** Determinism for a given seed, feasibility arithmetic, FORMAT §5 slab
  sizes, calibration recovery, usage-file round trip, hardware overrides and CLI
  smoke tests.

**Mutation testing** (`governance/tools/mutate.py`). Generation 1 sampled 30 mutants
per whole file with seed 1: `cache.py` 86.7%, `timing.py` 86.7%, `trace.py` 93.3%,
`core.py` 86.7%. Generation 2 sampled the regions it changed, with seed 2:

| file (changed regions) | mutants | first run | after new tests | survivors (all equivalent) |
|---|---|---|---|---|
| `cache.py` | 30 | 63.3% | 90.0% | 3 |
| `core.py` | 24 | 91.7% | 100% | 0 |
| `trace.py` | 16 | 68.8% | 87.5% (+1 killed since) | 1 |
| `hardware.py` | 16 | 93.8% | 100% | 0 |
| `timing.py` | 16 | 87.5% | 100% | 0 |

The remaining survivors were inspected:

* Two Belady internals cannot change a result. One is a sign in the free-slot
  branch: with the optimal warm start, that branch only admits slabs that are never
  used in the window. The other is the initial value of `cur`: phantom heap entries
  carry next use 1, so they are never chosen as victims.
* LFU's `flush` may push stale or held entries. These are filtered when popped.
* `Trace.load`'s per-token byte count can be mutated, and the reshape still rejects
  ragged bodies.

The first runs found real gaps, now covered by tests:

* `build_schedule`'s own validation;
* the skip count;
* eviction counting on the prefetch path;
* the refill's popularity row;
* Belady's heap rebuild and `next_at`;
* boundary values of the new input checks.

**Generation 3.** The verifier showed that the gen-2 figures covered only sampled
regions. Its runs scored 0% on `_LFU.tick` (9 of 9 mutants survived, among them an
inverted decay that turns recency weighting upside down) and 65% on the LRU,
`lru-layer`, sampled-victim and `profile_order` code. The tests above (LFU heat,
partitions, held prefetches) were written against those survivors. Results on the
final code, with the test command
`{python} -m pytest -q -x -p no:cacheprovider tests/py/test_sim.py`:

| file: region (final line numbers) | seed | mutants | score | survivors |
|---|---|---|---|---|
| `cache.py`: `_LFU.tick` (487-496) | 3 | 9 | 88.9% (gen 2: 0%) | 1 |
| `cache.py`: `_LRU`, `_LRULayer`, LFU rebuild/sampled victim, `profile_order` (145-287, 389-418, 653-659) | 5 | 20 | 95.0% (gen 2: 65%) | 1 |
| `cache.py`: all policy code (145-496) | 3 | 40 | 87.5% (82.5% before two more white-box checks) | 5 |
| `cache.py`: LFU admit and memo, Belady, `make_policy`, `run_cache` loops (457-496, 510-612, 661-690, 773-866) | 4 | 40 | 82.5% | 7 |
| `core.py`: changed lines | 4 | 24 | 95.8% (79.2% first run) | 1 |
| `timing.py`: `calibrate` validation | 4 | 12 | 100% | 0 |
| `trace.py`: `n_layers` and duplicate checks | 4 | 10 | 100% (80.0% first run) | 0 |

The survivors were inspected. None changes a hit count or a timing:

* **Stamp clock** (4 mutants). The LRU stamp clock's start value or step size;
  any strictly increasing unique stamps give the same order.
* **Performance-only** (4 mutants):
  * the sign inside the sampled-LFU "nothing unprotected" pre-check (the exact scan
    after it then returns the same -1);
  * Belady's heap-rebuild threshold;
  * `>` vs `>=` against 1e100 in `tick`, which can only move a rescale by one tick;
  * `and` vs `or` in the memo's sample draw, which differs only with an empty
    resident list, where the generator is never used again.
* **Equal next use in Belady** (`>=` vs `>`). It decides between two slabs that are
  both never used again.
* **Tie rule** (1 mutant). Sampled LFU keeps the *first* of equally hot samples;
  `<=` keeps the last. That is an arbitrary tie rule, not a property.
* **Wrong-guess draw** (4 mutants, pre-existing code). They change how many stream
  draws the wrong-guess generator tries before its exact Gumbel refill, or call the
  refill for zero experts. They alter the random stream but not the distribution,
  which is sampling without replacement proportional to popularity either way.
* **Placeholder** (1 mutant). `Result(tokens=0)` before `_aggregate` overwrites it.

## 6. Limitations

The simulator is deliberately simple.

**Synthetic traces.**

* Stationary popularity, independent layers, and one locality knob.
* No topic or prompt drift, no prefill phase, and no correlation between layers.
  Real routing has all of these, and they matter for prefetch recall.
* The Zipf exponent and reuse of real Kimi/DeepSeek routing are **unknown** until
  traces are recorded. DeepSeek-style auxiliary-loss-free balancing pushes
  batch-level load toward uniform. Per-sequence skew may be much lower than
  Zipf 1.1, hence §8.3.

**Storage.**

* Perfect striping over drives and one latency per layer.
* Demand reads pre-empt prefetches instantly.
* No queue-depth, file-system, thermal or SLC-cache effects.
* NVMe DMA does not consume DRAM bandwidth.
* Cancelled wrong-guess prefetches still occupy a slot in the cache simulation
  ("phantom" residency; second-order).
* The engine's store (`engine/src/store.c`, in progress) also refuses a prefetch
  whose victim is "hot": under LFU, heat of an expert used in about half of all
  tokens; under LRU, used during this token. It picks the slot only when a reader
  starts the read. The simulator models neither, so it may let wrong guesses
  displace more useful slabs than the engine would.

**Compute.**

* A bandwidth/int8 roofline per expert; threads are assumed to reach the profile's
  DRAM bandwidth.
* No attention FLOPs beyond reading the KV cache, which is read once per step.
* No NUMA. No prefill.

**Speculative decoding.** Drafting is free (prompt lookup), acceptance is i.i.d.
geometric, and rejected positions use the true tokens' routing.

**Kimi-K3 specifics.**

* Attention is approximated as MLA in every layer, as in the preset. 69 of its 93
  layers are linear attention (KDA), so the KV term is overstated, and is small at
  context 1024.
* The latent projections are an addition the simulator makes itself.

**Other.**

* Exact LFU by default; the engine samples (`--lfu-samples`).
* Belady may bypass, evict in-use slabs and start warm. The engine can do none of
  these, so Belady is a bound, not a policy.
* Feasibility uses a fixed OS reserve and no swap.

## 7. Example outputs (real runs of this code, 2026-10-04)

```
$ python -m hearth sim --model kimi-k2 --hw this-pc --cache-gb 40 --policy lfu --zipf 1.1 --tokens 2000 --spec-k 4 --spec-alpha 0.6
SIMULATED (hearth.sim model, not a measurement)
model kimi-k2; hardware this-pc (DRAM 60 GB/s, RAM 61.6 GiB, NVMe 1x6.5 GB/s, int8 5 TOPS); trace synthetic(zipf=1.1,reuse=0.25,seed=0) (2000 tokens, first 200 = warm-up); experts 4.25 bpw, backbone 8.25 bpw, context 1024; prefetch off; speculative k=4 alpha=0.6

policy          lfu  (1835 slots x 22.3 MiB = 40.0 GiB)
decode speed    0.834 tok/s  (1199.8 ms/token, 2.32 tokens/step)
expert loads    638.8 per token: hit 51.1%, prefetched 0.0%, demand miss 48.9%
NVMe per token  7.303 GB (demand 7.303, prefetch 0.000 of which wasted 0.000)
DRAM per token  19.904 GB
time per token  overhead 1.0 ms, dram 331.7 ms, nvme 867.1 ms
bottleneck      nvme

$ python -m hearth sim --model kimi-k2 --hw this-pc --cache-gb 40 --compare-policies
SIMULATED (hearth.sim model, not a measurement)
model kimi-k2; hardware this-pc (DRAM 60 GB/s, RAM 61.6 GiB, NVMe 1x6.5 GB/s, int8 5 TOPS); trace synthetic(zipf=1.1,reuse=0.25,seed=0) (2000 tokens, first 200 = warm-up); experts 4.25 bpw, backbone 8.25 bpw, context 1024; prefetch off; speculative off

case        cache GiB    hit  prefetch   miss  NVMe GB/tok  DRAM GB/tok  tok/s  ms/tok  bottleneck
----------  ---------  -----  --------  -----  -----------  -----------  -----  ------  ----------
lru              40.0  58.7%      0.0%  41.3%         4.64        22.74   1.12     892        nvme
lru-layer        40.0  59.6%      0.0%  40.4%         4.54        22.74   1.14     877        nvme
lfu              40.0  61.2%      0.0%  38.8%         4.36        22.74   1.18     851        nvme
pinned           40.0  59.7%      0.0%  40.3%         4.53        22.74   1.14     876        nvme
lfu-pinned       40.0  61.2%      0.0%  38.8%         4.36        22.74   1.18     850        nvme
belady           40.0  76.6%      0.0%  23.4%         2.62        22.74   1.66     603        dram

  ! belady is an offline upper bound (knows the future, may bypass the cache and evict in-use slabs, starts with the best possible cache contents)

$ python -m hearth sim --model kimi-k2 --hw this-pc --feasibility
kimi-k2 on this-pc:
fits                       yes
backbone                   11.7 GiB in ram; KV 0.54 GiB
RAM left for expert cache  42.8 GiB (engine minimum 0.57 GiB)
minimum RAM                19.3 GiB
recommended cache          42.8 GiB of 502 GiB of experts (8.5%)
model file                 552 GB
disk by expert format      Q3 (3.25 bpw, roadmap R03): 425 GB, Q4 (4.25 bpw): 552 GB, Q8 (8.25 bpw): 1059 GB, BF16: 2042 GB

$ python -m hearth sim --model kimi-k2 --stats --tokens 2000 --cache-gb 40 --prefetch 0.8
trace statistics (synthetic(zipf=1.1,reuse=0.25,seed=0)):
tokens x layers x top-k                 2000 x 60 x 8 (E = 384)
entropy / log2(E)                       mean 0.781, min 0.774
reuse from previous token               0.395
mass in top x% (per layer)              1%: 0.273, 5%: 0.543, 10%: 0.657, 20%: 0.768
mass in top x% (all slabs)              1%: 0.267, 5%: 0.537, 10%: 0.654, 20%: 0.768
slabs to cover x% of loads              50%: 931, 80%: 5623, 90%: 10629, 95%: 14836
distinct experts / layer over w tokens  w=1: 8.0, w=2: 12.84, w=3: 17.123, w=5: 24.722

SIMULATED (hearth.sim model, not a measurement)
model kimi-k2; hardware this-pc (DRAM 60 GB/s, RAM 61.6 GiB, NVMe 1x6.5 GB/s, int8 5 TOPS); trace synthetic(zipf=1.1,reuse=0.25,seed=0) (2000 tokens, first 200 = warm-up); experts 4.25 bpw, backbone 8.25 bpw, context 1024; prefetch recall 0.8 + 0 extra; speculative off

policy           lfu  (1835 slots x 22.3 MiB = 40.0 GiB)
decode speed     1.33 tok/s  (752.1 ms/token, 1.00 tokens/step)
expert loads     480.0 per token: hit 60.8%, prefetched 30.8%, demand miss 8.4%
NVMe per token   4.630 GB (demand 3.682, prefetch 0.948 of which wasted 0.229)
DRAM per token   22.744 GB
time per token   overhead 2.2 ms, dram 379.1 ms, nvme 370.8 ms
bottleneck       dram
late prefetches  79.2%
```

How to read the last run:

* 30.8% of loads were predicted, but 79% of those predictions could not be read
  before their layer started. The single drive is busy with demand reads, so the
  needed ones were promoted to demand reads.
* When storage is the bottleneck, prefetch only fills the storage gaps during
  attention. It cannot create bandwidth.
* The wrong guesses (0.23 GB per token) occupy slots and stay protected for the
  rest of the step (§3.2). They displace slabs that later layers would have hit, and
  the hit rate falls from 61.2% without prefetch to 60.8%.

## 8. What does it take to run Kimi-K2 (1T) and Kimi-K3 (2.8T)? — SIMULATED

**All numbers in this section are SIMULATED** by the code above on 2026-10-04, with
uncalibrated scalars.

Trace assumptions: synthetic, 2000 tokens, Zipf exponent 1.1, reuse 0.25, seed 0,
first 200 tokens as warm-up.

Encoding and defaults:

* Q4 experts, and a Q8 backbone. Where Q8 does not fit, a Q4 backbone is used and
  marked **LOSSY** (vs the converter's Q8 default).
* LFU with decay 0.995; the cache gets all RAM left; context 1024.
* **The baseline has prefetch OFF.** This differs from the engine, whose default is
  `HEARTH_PREFETCH_SHARED`. The predictor's real recall is unknown, so the tables
  show prefetch separately, at an *assumed* recall of 0.8 ("+prefetch 0.8"). A
  configuration like the engine's default is closer to that column than to the
  baseline.

No engine has run these models yet. The skew of real Kimi routing is unknown, so
§8.3 shows how the answers move with it.

What one token costs (Kimi-K2, Q4 experts, Q8 backbone):

* 60 × 8 = 480 expert slabs of 22.3 MiB each, 11.2 GB in total;
* about 11.5 GB of backbone: 6.2 B attention, 2.6 B shared-expert and 1.2 B LM-head
  parameters at Q8, plus 0.66 GB of F32 routers.

So even with every expert in DRAM, this-pc's 60 GB/s caps Kimi-K2 at about
**2.6 tok/s**.

Kimi-K3 needs 92 × 16 = 1472 slabs of 16.7 MiB (25.8 GB) plus 31.7 GB of backbone
per token. That backbone includes 12.2 B shared-expert and 4.7 B latent-projection
parameters, and it caps Kimi-K3 at about **1.0 tok/s**. Storage only makes things
slower than these ceilings.

### 8.1 Does it fit?

| model | hardware | backbone | backbone GiB | cache GiB | cache/experts | min RAM GiB | file GB (Q4) | fits drive | runs |
|---|---|---|---|---|---|---|---|---|---|
| kimi-k2 | this-pc | Q8 | 11.7 | 42.8 | 8.5% | 19 | 552 | yes | yes |
| kimi-k2 | laptop-16gb | Q4 (LOSSY vs Q8) | 6.9 | 4.1 | 0.8% | 12 | 546 | yes | yes |
| kimi-k2 | desktop-32gb | Q8 | 11.7 | 15.2 | 3.0% | 17 | 552 | yes | yes |
| kimi-k2 | workstation-128gb-2nvme | Q8 | 11.7 | 109.2 | 21.8% | 19 | 552 | yes | yes |
| kimi-k2 | mac-studio-192gb | Q8 | 11.7 | 171.2 | 34.1% | 21 | 552 | yes | yes |
| kimi-k3 | this-pc | Q8 | 30.4 | 23.9 | 1.8% | 38 | 1479 | yes | yes |
| kimi-k3 | laptop-16gb | - | 17.3 | -6.6 | 0.0% | 23 | 1465 | NO | NO |
| kimi-k3 | desktop-32gb | Q4 (LOSSY vs Q8) | 17.3 | 9.4 | 0.7% | 23 | 1465 | yes | yes |
| kimi-k3 | workstation-128gb-2nvme | Q8 | 30.4 | 90.3 | 6.7% | 38 | 1479 | yes | yes |
| kimi-k3 | mac-studio-192gb | Q8 | 30.4 | 152.3 | 11.3% | 40 | 1479 | yes | yes |

The 16 GB laptop runs Kimi-K2 only with a Q4 backbone, which is lossy against Q8.
It cannot run Kimi-K3: the backbone alone exceeds its RAM, and the 1.48 TB file
exceeds its 1 TB drive. Kimi-K3 at Q4 experts fits a 2 TB drive.

### 8.2 Speed by profile (Zipf 1.1, reuse 0.25)

| model | hardware | backbone | cache GiB | LFU hit | NVMe GB/tok | tok/s LFU | tok/s +prefetch 0.8 | tok/s ceiling (all experts in DRAM) | bottleneck |
|---|---|---|---|---|---|---|---|---|---|
| kimi-k2 | this-pc | Q8 | 43 | 62% | 4.2 | 1.20 | 1.37 | 2.62 | nvme |
| kimi-k2 | laptop-16gb | Q4 (LOSSY) | 4 | 23% | 8.7 | 0.48 | 0.45 | 2.55 | nvme |
| kimi-k2 | desktop-32gb | Q8 | 15 | 45% | 6.2 | 0.66 | 0.72 | 1.66 | nvme |
| kimi-k2 | workstation-128gb-2nvme | Q8 | 109 | 77% | 2.5 | 2.40 | 2.41 | 2.41 | dram |
| kimi-k2 | mac-studio-192gb | Q8 | 171 | 84% | 1.8 | 2.85 | 3.01 | 10.73 | nvme |
| kimi-k3 | this-pc | Q8 | 24 | 39% | 15.7 | 0.36 | 0.38 | 1.04 | nvme |
| kimi-k3 | laptop-16gb | - | does not fit |  |  |  |  |  |  |
| kimi-k3 | desktop-32gb | Q4 (LOSSY) | 9 | 24% | 19.5 | 0.24 | 0.22 | 0.87 | nvme |
| kimi-k3 | workstation-128gb-2nvme | Q8 | 90 | 61% | 10.1 | 0.95 | 0.95 | 0.95 | dram |
| kimi-k3 | mac-studio-192gb | Q8 | 152 | 69% | 8.0 | 0.70 | 0.73 | 4.30 | nvme |

The "ceiling" column assumes every expert is already in DRAM:
`1 / (T_fixed + (backbone + L·k·S) / BW_dram)`.

* With one ordinary NVMe drive, both models are **storage-bound** everywhere.
* With two fast drives (the workstation), they become **DRAM-bandwidth-bound** and
  reach the ceiling. From there only a smaller backbone, a GPU or batching helps.
* The Mac Studio has the bandwidth for 10 tok/s on Kimi-K2, but one 6 GB/s SSD
  holds it to about 3.
* On the smallest caches, prefetch **costs** speed: the laptop goes from 0.48 to
  0.45 tok/s on Kimi-K2, and the 32 GB desktop from 0.24 to 0.22 on Kimi-K3. Wrong
  guesses stay protected until the step ends, and in a 4 GiB or 9 GiB cache they
  displace slabs that would have hit.

### 8.3 Sensitivity to routing skew (this-pc, reuse 0.25)

| model | zipf | cache GiB | LRU hit | LFU hit | Belady hit | NVMe GB/tok (LFU) | tok/s LRU | tok/s LFU | tok/s LFU+prefetch |
|---|---|---|---|---|---|---|---|---|---|
| kimi-k2 | 0 | 43 | 31% | 12% | 53% | 9.9 | 0.73 | 0.59 | 0.63 |
| kimi-k2 | 0.5 | 43 | 34% | 25% | 57% | 8.4 | 0.76 | 0.68 | 0.73 |
| kimi-k2 | 0.8 | 43 | 44% | 44% | 67% | 6.3 | 0.88 | 0.87 | 0.95 |
| kimi-k2 | 1 | 43 | 54% | 56% | 74% | 4.9 | 1.04 | 1.07 | 1.20 |
| kimi-k2 | 1.2 | 43 | 65% | 68% | 81% | 3.6 | 1.29 | 1.35 | 1.55 |
| kimi-k2 | 1.4 | 43 | 74% | 77% | 86% | 2.6 | 1.57 | 1.67 | 1.95 |
| kimi-k3 | 0 | 24 | 8% | 3% | - | 25.1 | 0.25 | 0.24 | 0.25 |
| kimi-k3 | 0.5 | 24 | 8% | 10% | - | 23.1 | 0.25 | 0.26 | 0.27 |
| kimi-k3 | 0.8 | 24 | 8% | 24% | - | 19.7 | 0.25 | 0.30 | 0.31 |
| kimi-k3 | 1 | 24 | 8% | 34% | - | 17.0 | 0.25 | 0.34 | 0.35 |
| kimi-k3 | 1.2 | 24 | 9% | 44% | - | 14.5 | 0.25 | 0.39 | 0.40 |
| kimi-k3 | 1.4 | 24 | 9% | 52% | - | 12.4 | 0.25 | 0.44 | 0.46 |

Across plausible skews:

* Kimi-K2 on this-pc spans **0.6–1.7 tok/s** with prefetch off, and 0.6–2.0 with
  prefetch at the assumed recall of 0.8.
* Kimi-K3 spans **0.24–0.44 tok/s**, and 0.25–0.46 with prefetch.

Skew is the largest unknown, so recording real traces is the first thing T09 should
do.

### 8.4 Cache policy: when LRU fails and when LFU does

* **LRU fails.** Kimi-K3's 23.9 GiB cache is 1461 slots, just *below* one token's
  working set of 1472 slabs. Global LRU collapses to about 8% at every skew: the
  cyclic pathology of ARCHITECTURE.md, on a real configuration. For Kimi-K2 the
  same happens at 4–8 GiB (§8.6: LRU 0%, LFU 22–34%).
* **LFU fails.** Under weak skew (Zipf ≤ 0.5) with temporal locality, LFU at the
  engine's default decay of 0.995 *loses* to LRU (Kimi-K2: 12% vs 31%). Its
  half-life of about 140 tokens rewards long-run frequency, which is noise when
  popularity is flat, and it evicts the experts the next token is likely to reuse.
* **Headroom.** On Kimi-K2, Belady's bound stays 9–23 points above the better of
  LRU and LFU, so better online policies may have room. The bound is loose by
  construction (§3.2: no protection, free warm start), so not all of that gap can
  be closed.

A decay sweep (this-pc; cells are hit rate / tok/s):

| model | zipf | LRU hit | LFU d=0.995 (engine default) | d=0.98 | d=0.9 | d=0.7 |
|---|---|---|---|---|---|---|
| kimi-k2 | 0 | 31% | 12% / 0.59 | 16% / 0.61 | 31% / 0.73 | 31% / 0.73 |
| kimi-k2 | 0.5 | 34% | 25% / 0.68 | 26% / 0.68 | 35% / 0.77 | 34% / 0.76 |
| kimi-k2 | 1.1 | 60% | 62% / 1.20 | 62% / 1.19 | 64% / 1.23 | 61% / 1.17 |
| kimi-k3 | 0 | 8% | 3% / 0.24 | 4% / 0.24 | 9% / 0.25 | 11% / 0.26 |
| kimi-k3 | 0.5 | 8% | 10% / 0.26 | 10% / 0.25 | 12% / 0.26 | 12% / 0.26 |
| kimi-k3 | 1.1 | 8% | 39% / 0.36 | 39% / 0.36 | 37% / 0.35 | 32% / 0.33 |

On these synthetic traces, decay 0.9 is the robust choice: as good as LRU under weak
skew, and within 2 points of 0.995 under strong skew. This is an open question for
the engine's default, to settle on real traces.

### 8.5 Speculative decoding needs temporal locality (Kimi-K2, this-pc, Zipf 1.1)

| reuse | expert loads/token plain | spec a=0.6 | spec a=0.8 | tok/s plain | tok/s spec k=4 a=0.6 | tok/s spec k=4 a=0.8 |
|---|---|---|---|---|---|---|
| 0 | 480 | 746 | 507 | 1.21 | 0.71 | 1.04 |
| 0.25 | 480 | 639 | 434 | 1.20 | 0.86 | 1.25 |
| 0.5 | 480 | 518 | 352 | 1.20 | 1.09 | 1.60 |
| 0.75 | 480 | 379 | 257 | 1.24 | 1.57 | 2.28 |
| 0.9 | 480 | 281 | 191 | 1.41 | 2.37 | 3.41 |

Verification loads the *union* of the window's experts. Rejected drafts cost
expert reads, and the union grows almost linearly when consecutive tokens share few
experts. Speculative decoding therefore **slows** a storage-bound MoE unless
acceptance and expert reuse are both high:

* at α = 0.8 it roughly breaks even at reuse 0.25 and gains 1.3× from reuse 0.5;
* at α = 0.6 it needs reuse of about 0.75 before it helps at all.

When the machine is DRAM-bound it still pays off, because the backbone is read once
per step: Kimi-K3 on the workstation gains 1.70× at α = 0.8 (§8.7).

### 8.6 Cache size (Kimi-K2 on this-pc, Zipf 1.1)

The 100 and 200 GiB rows do not fit this-pc's RAM (`Result.feasible` is false). They
show what a machine with that much free RAM and this-pc's other numbers would reach.

| cache GiB | slots | fits this-pc | LRU hit | LFU hit | tok/s LRU | tok/s LFU | bottleneck (LFU) |
|---|---|---|---|---|---|---|---|
| 4.0 | 183 | yes | 0% | 22% | 0.53 | 0.66 | nvme |
| 8.0 | 367 | yes | 0% | 34% | 0.53 | 0.76 | nvme |
| 16.0 | 734 | yes | 39% | 46% | 0.82 | 0.90 | nvme |
| 32.0 | 1468 | yes | 55% | 57% | 1.04 | 1.09 | nvme |
| 42.8 (all free RAM) | 1966 | yes | 60% | 62% | 1.16 | 1.20 | nvme |
| 100.0 | 4589 | **NO** (INFEASIBLE) | 74% | 76% | 1.58 | 1.63 | dram |
| 200.0 | 9178 | **NO** (INFEASIBLE) | 86% | 87% | 2.08 | 2.13 | dram |

### 8.7 Which levers matter most

Each row changes one thing against the baseline named in its first row; the labels
are built from the actual settings. LOSSY marks rows that change model output.
INFEASIBLE marks configurations that do not fit; their speed is what they *would*
reach. The baseline has prefetch off, unlike the engine (see the start of §8).
"hit" counts loads served without any storage read; prefetched loads are reads and
are not included, so rows with prefetch can show a lower hit rate and still be
faster.

**kimi-k2 on this-pc**

| lever | tok/s | vs base | hit | NVMe GB/tok | bottleneck |
|---|---|---|---|---|---|
| baseline: lfu, all free RAM as cache, prefetch off, no speculative decoding | 1.2 | 1.00x | 62.3% | 4.23 | nvme |
| policy lru (global) | 1.16 | 0.96x | 60.5% | 4.44 | nvme |
| prefetch next layer (recall 0.8) | 1.37 | 1.13x | 62.0% | 4.50 | dram |
| speculative k=4, alpha=0.6 | 0.857 | 0.71x | 52.6% | 7.09 | nvme |
| speculative k=4, alpha=0.8 | 1.25 | 1.04x | 52.2% | 4.86 | nvme |
| +1 NVMe drive (2x6.5 GB/s) | 1.94 | 1.61x | 62.3% | 4.23 | dram |
| 2x RAM (123 GiB) | 1.66 | 1.38x | 76.6% | 2.63 | dram |
| dense backbone Q4 instead of Q8 (LOSSY vs the Q8 default) | 1.36 | 1.13x | 64.1% | 4.03 | nvme |
| Q3 experts 3.25 bpw (LOSSY vs Q4, roadmap R03) | 1.62 | 1.35x | 66.7% | 2.86 | dram |
| LOSSY top-k 8->6 | 2.01 | 1.67x | 78.2% | 1.84 | dram |
| GPU: Q4 backbone (LOSSY vs Q8) + hot experts on GPU (roadmap R01) | 1.65 | 1.37x | 67.4% | 3.66 | nvme |
| prefetch 0.8 + spec k=4 a=0.6 + 1 more NVMe | 1.67 | 1.39x | 50.6% | 7.58 | dram |

**kimi-k3 on this-pc**

| lever | tok/s | vs base | hit | NVMe GB/tok | bottleneck |
|---|---|---|---|---|---|
| baseline: lfu, all free RAM as cache, prefetch off, no speculative decoding | 0.361 | 1.00x | 39.2% | 15.70 | nvme |
| policy lru (global) | 0.25 | 0.69x | 8.4% | 23.66 | nvme |
| prefetch next layer (recall 0.8) | 0.379 | 1.05x | 36.6% | 16.84 | nvme |
| speculative k=4, alpha=0.6 | 0.244 | 0.68x | 24.8% | 25.61 | nvme |
| speculative k=4, alpha=0.8 | 0.359 | 1.00x | 24.7% | 17.43 | nvme |
| +1 NVMe drive (2x6.5 GB/s) | 0.64 | 1.77x | 39.2% | 15.70 | dram |
| 2x RAM (123 GiB) | 0.515 | 1.43x | 60.2% | 10.29 | nvme |
| dense backbone Q4 instead of Q8 (LOSSY vs the Q8 default) | 0.426 | 1.18x | 46.5% | 13.81 | nvme |
| Q3 experts 3.25 bpw (LOSSY vs Q4, roadmap R03) | 0.485 | 1.34x | 43.7% | 11.13 | nvme |
| LOSSY top-k 16->12 | 0.561 | 1.56x | 52.2% | 9.26 | nvme |
| GPU: Q4 backbone (LOSSY vs Q8) + hot experts on GPU (roadmap R01) (INFEASIBLE) | 0.515 | 1.43x | 52.8% | 12.20 | nvme |
| prefetch 0.8 + spec k=4 a=0.6 + 1 more NVMe | 0.423 | 1.17x | 11.6% | 30.47 | nvme |

**kimi-k3 on workstation-128gb-2nvme**

| lever | tok/s | vs base | hit | NVMe GB/tok | bottleneck |
|---|---|---|---|---|---|
| baseline: lfu, all free RAM as cache, prefetch off, no speculative decoding | 0.953 | 1.00x | 61.0% | 10.07 | dram |
| policy lru (global) | 0.953 | 1.00x | 58.8% | 10.63 | dram |
| prefetch next layer (recall 0.8) | 0.954 | 1.00x | 60.7% | 12.80 | dram |
| speculative k=4, alpha=0.6 | 1.11 | 1.16x | 50.4% | 16.89 | dram |
| speculative k=4, alpha=0.8 | 1.62 | 1.70x | 50.0% | 11.56 | dram |
| +1 NVMe drive (3x12 GB/s) | 0.954 | 1.00x | 61.0% | 10.07 | dram |
| 2x RAM (256 GiB) | 0.954 | 1.00x | 74.5% | 6.59 | dram |
| dense backbone Q4 instead of Q8 (LOSSY vs the Q8 default) | 1.25 | 1.32x | 63.2% | 9.52 | dram |
| Q3 experts 3.25 bpw (LOSSY vs Q4, roadmap R03) | 1.07 | 1.12x | 65.2% | 6.87 | dram |
| LOSSY top-k 16->12 | 1.07 | 1.13x | 78.4% | 4.19 | dram |
| GPU: Q4 backbone (LOSSY vs Q8) + hot experts on GPU (roadmap R01) | 2.17 | 2.28x | 66.3% | 8.71 | dram |
| prefetch 0.8 + spec k=4 a=0.6 + 1 more NVMe | 1.15 | 1.21x | 48.2% | 22.28 | dram |

### 8.8 Answer (SIMULATED, under the assumptions above)

* **Kimi-K2 (1T) on this-pc.** About 1.2 tok/s with prefetch off. With the engine's
  default prefetch at an assumed recall of 0.8, about 1.37 tok/s. Depending on real
  routing skew, the range is 0.6–1.7 (0.6–2.0 with prefetch). It is storage-bound,
  against a DRAM-bandwidth ceiling of about 2.6 tok/s.
  * A second NVMe drive is the biggest lossless lever: 1.6×, to about 1.9 tok/s,
    which reaches the DRAM ceiling.
  * More RAM comes next: 1.4× at 123 GiB.
  * Putting the backbone and hot experts on the GPU also gives 1.4×. It is roadmap
    R01, and it needs a Q4 backbone to fit 12 GiB, which is LOSSY vs Q8.
  * Next-layer prefetch is worth about 1.13× at the assumed recall. The engine's
    default already prefetches, so part of this gain is already in the engine's
    default configuration rather than an extra lever.
  * Lossy levers change output: a Q4 backbone gives 1.13×, Q3 experts 1.35×, and
    top-k 8→6 1.67×.
* **Kimi-K3 (2.8T) on this-pc.** About 0.36 tok/s (0.24–0.44 over skew). Prefetch
  adds only 1.05×. A second NVMe drive (1.8×) and more RAM (1.4×) matter most. Its
  Q8 backbone does not fit a 12 GiB GPU, and even a Q4 one does not.
* **Kimi-K3 at about 1 tok/s and above** needs a workstation-class box: 128 GB and
  two PCIe 5.0 drives give 0.95 tok/s, DRAM-bound. Beyond that, two levers remain:
  * a GPU holding a Q4 backbone and hot experts: 2.3×, to about 2.2 tok/s. This is
    roadmap R01 and LOSSY vs Q8;
  * speculative decoding with high acceptance: 1.7× at α = 0.8.
* **Small machines.**
  * A 16 GB laptop runs Kimi-K2 only with a Q4 backbone (LOSSY), at about
    0.5 tok/s; prefetch lowers that to 0.45. It cannot hold Kimi-K3.
  * A 32 GB desktop runs Kimi-K2 at about 0.7 tok/s, and Kimi-K3 with a Q4
    backbone (LOSSY) at about 0.24 tok/s.
* **Levers that matter more than they look:**
  * the cache policy, whenever the cache is near or below one token's working set
    (Kimi-K3 on 64 GB: LFU 0.36 vs LRU 0.25 tok/s);
  * the LFU decay, under weak skew (§8.4).
* **Levers that matter less than hoped:**
  * prefetch, while storage is saturated. On small caches it hurts, because
    protected wrong guesses displace useful slabs;
  * speculative decoding, unless routing has strong temporal locality. On
    Kimi-K3/this-pc a verification window needs about 4500 slabs per step, three
    times the 1461-slot cache. Combined with prefetch, 70% of loads arrive as
    prefetches (hence the 12% "hit" in the last lever row), but storage still has
    to move them.

## 9. Simulator performance

All times are from real runs on this-pc (Python 3.13, numpy 2.2), measured on
2026-10-04 after the generation-3 changes. Other engineers were building and
testing at the same time. Windows reported 47-85% CPU load around these runs, so
treat the times as **noisy**; they were about 20% lower at lighter load earlier the
same day.

| run | time |
|---|---|
| synthetic trace: Kimi-K2 / Kimi-K3, 2000 tokens | 0.63 s / 1.9 s |
| Kimi-K2, 2000 tokens, all 5 online policies | 3.2 s total (0.43-0.94 s each); 2.6 s at lighter load |
| Kimi-K2, Belady | 0.68 s |
| Kimi-K2, LFU + prefetch (0.8, 0) / (0.5, 4), 40 GiB | 2.3 s / 3.5 s |
| Kimi-K2, `Prefetch(0.5, 64)`, 4 GiB: LRU / sampled LFU (8) / exact LFU | 6.9 s / 14.9 s / 6.1 s |
| Kimi-K3, one simulation including trace generation | 4.6 s |
| `--compare-policies` (6 policies, CLI, wall) | 4.5 s |
| `tests/py/test_sim.py` (112 tests) | 21 s |
| all of §8 regenerated in generation 2 (about 160 simulations, 6 processes) | 155 s |

**Protected prefetches.** Engine-style protection keeps wrong guesses resident and
protected until the step ends. In a small cache with many extras, the cache fills
with them, and every later admission used to walk past all of them before it was
rejected. In one paired run (both versions back to back, while a 5-job mutation
run was active), the `Prefetch(0.5, 64)` / 4 GiB case took:

| version | LRU | sampled LFU (8) | exact LFU |
|---|---|---|---|
| generation 2 | 50.6 s | 44.1 s | 21.6 s |
| generation 3 | 7.9 s | 15.1 s | 7.0 s |

Generation 3 changes three things:

* held slabs are kept out of the LRU order and the LFU heap (§3.2);
* the demand's last resort takes the oldest or coldest held slab directly;
* LFU answers a repeated REJECT under the same protection set without searching
  again. Sampled LFU still draws the samples a search would draw, so its random
  stream is unchanged. That draw is most of its remaining cost.

The table at the top of this section is a separate, later run. The default
configuration is unaffected.

These are pure optimisations. A scratch harness ran the generation-2 and
generation-3 `run_cache` side by side on random schedules: 1-4 layers, top-k 1-6,
speculative windows, VRAM and cold keys, the skip knob, a warm-up, and six prefetch
settings up to (0.5, 64). It covered every policy (Belady without VRAM) and four
LFU settings. On the final code, all 53 100 runs (6 seeds) gave identical counters,
including about 163 000 last-resort evictions. As a check that the harness can see a
difference, it reported 138 differing runs of 3 540 when the LRU last resort was
deliberately changed to evict the newest held slab instead.

Peak Python-traced memory for one default simulation (tracemalloc, 2000 tokens),
measured in generation 1: 107 MB for Kimi-K2 and 196 MB for Kimi-K3.
