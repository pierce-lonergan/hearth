# Hearth

**Frontier-scale Mixture-of-Experts models on the computer you already own.**

Hearth is a zero-dependency C11 inference engine (plus a numpy-only Python toolkit)
for running very large Mixture-of-Experts LLMs — hundreds of billions to trillions of
parameters — on ordinary PCs. The dense backbone stays in RAM; the routed experts live
on your NVMe drive and are streamed into a DRAM expert cache on demand, with
next-layer prediction and prefetch. Insufficient RAM makes Hearth slower, never wrong.

> **Status: pre-alpha (v0.1, October 2026).** The engine works end to end on real
> weights and is heavily tested, but it is CPU-only today, has been developed on one
> Windows machine, and quiet-machine benchmarks are still being collected. Numbers below
> say whether they were *measured* or *simulated*, on what hardware.

## What makes it different

* **Bit-identical by construction.** Every kernel follows one canonical floating-point
  order ([docs/NUMERICS.md](docs/NUMERICS.md)), so output does not depend on whether a
  weight came from NVMe, the cache or a prefetch, on the thread count, on the SIMD ISA
  (scalar / AVX2 / AVX-512), or on batching. Prefill, speculative verification and
  decode agree bit for bit — which makes speculative decoding exactly lossless and makes
  every performance optimisation testable against a fixed answer.
* **Streaming that respects the hardware.** One contiguous, 4096-aligned slab per
  expert ([docs/FORMAT.md](docs/FORMAT.md)), direct I/O, overlapped reads on Windows,
  mirrors across drives, a persistent thread pool instead of OpenMP fork/join.
* **One generic forward pass.** GQA and MLA attention, softmax / sigmoid / grouped
  routers, shared experts, dense prefix layers — Qwen3-MoE, OLMoE, Mixtral, Qwen2-MoE,
  DeepSeek-V3 and Kimi-K2 (the 1T-parameter class) share one engine, configured by data.
* **A simulator you can ask questions.** `python -m hearth sim` predicts decode speed,
  cache hit rate and the bottleneck for any preset model on any hardware profile, from
  synthetic or *recorded* routing traces.
* **Built to be maintained by humans and AI agents indefinitely.** See
  [How this project is built](#how-this-project-is-built).

## Results so far

All on one machine: AMD Ryzen 9 9950X (16 cores, AVX-512), 64 GB DDR5, Samsung 990 PRO
(PCIe 4.0 NVMe), CPU only.

| what | result | kind |
|---|---|---|
| Qwen3-30B-A3B (real weights), first 4 layers vs Hugging Face `transformers` fp32, 96 real tokens | max logit error 1.3e-5 (6e-7 relative), top-1 100%, expert routing 100%, batch == sequential bit-identical | measured |
| Qwen3-30B-A3B full model, Q4 experts / Q8 dense (17 GB) vs BF16 (61 GB), 4,095 tokens | perplexity 11.45 vs 11.16 (+2.6%), mean KL 0.025, top-1 agreement 93.2% | measured |
| Full **BF16** Qwen3-30B-A3B (61 GB) on this **64 GB** PC with a 30 GB expert cache | runs; 38 tok/s prefill, 195 GB streamed from NVMe without error | measured |
| Qwen3-30B-A3B Q4 decode, cold cache, machine busy with other jobs | ~19 tok/s | measured, preliminary |
| Kimi-K2-*shaped* container (1.03 T params, random weights, 102 GB on disk) | decodes; ~1.2–1.8 s/token in a worst-case smoke test | measured, preliminary |
| Expert-cache policies on real Qwen3 routing traces | LRU beat our original LFU default by 8–13 points; fast-decay LFU ≈ LRU; static pinning goes stale ([write-up](docs/results/qwen3-routing-and-cache-policy.md)) | simulated on real traces |

Test suite at the time of writing: 21/21 maintainer "golden" invariant tests, 644 Python
tests, 6/6 C test suites (MSVC, plus gcc sanitizers via WSL for parts of the engine).

## Quick start

Requirements: a C11 compiler (MSVC 2022, GCC or Clang), CMake ≥ 3.20, Python ≥ 3.9 with
numpy. For conversion and chat also `safetensors` and `tokenizers` (`torch` only for FP8
checkpoints; `transformers` only for tests and exotic RoPE variants).

```bash
git clone https://github.com/pierce-lonergan/hearth && cd hearth
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release
```

Point Python at the library (`build/Release/hearth.dll` on Windows,
`build/libhearth.so` on Linux) and use the CLI:

```bash
export PYTHONPATH=python HEARTH_LIB=build/libhearth.so
python -m hearth doctor                                   # CPU, RAM, disks, library
python -m hearth convert /models/Qwen3-30B-A3B qwen3.hearth   # HF dir -> Q4 experts / Q8 dense
python -m hearth chat qwen3.hearth --cache-gb 8           # interactive chat
python -m hearth serve qwen3.hearth --port 8080           # OpenAI + Anthropic compatible API
python -m hearth bench qwen3.hearth --cache-gb 4          # speed, hit rate, I/O
python -m hearth sim --model kimi-k2 --hw this-pc --compare-policies   # what-if analysis
```

Model files are large: keep them on a fast local NVMe drive, not in a cloud-synced
folder.

## How it works

```
 tokens ─► embed ─► [ attention (RAM) ─► router ─► routed experts (NVMe → DRAM cache) ] × L ─► head
                                          │  ▲
                        predict layer L+1 │  │ compute experts in arrival order,
                        and prefetch      ▼  │ sum in canonical rank order
                                   expert store: slots · LFU/LRU · pins · reader threads
                                   demand queue (high) / prefetch queue (low) · direct I/O
```

Read more: [ARCHITECTURE](docs/ARCHITECTURE.md) · [FORMAT](docs/FORMAT.md) ·
[NUMERICS](docs/NUMERICS.md) · [PYTHON_API](docs/PYTHON_API.md) ·
[SIMULATOR](docs/SIMULATOR.md).

## How this project is built

Hearth was bootstrapped by a team of AI agents under an explicit governance model, and
is designed so that humans and agents can keep improving it without the codebase
decaying over time ([governance/](governance/README.md)):

* **Invariants are a zero-loss lane.** [governance/INVARIANTS.md](governance/INVARIANTS.md)
  holds the rules that must never be paraphrased away; maintainer-owned **golden tests**
  enforce them and are hash-locked (`governance/tools/check_golden.py`).
* **Verification is separated from implementation.** Every task was built by one agent
  and attacked by an independent verifier that rebuilt it, wrote its own adversarial
  tests and could fail it; contributor tests must survive **mutation testing**
  (`governance/tools/mutate.py`).
* **Sessions end in hand-overs, not transcripts.** Each work session emits a
  schema-validated manifest ([governance/handovers/](governance/handovers/)) — what was
  verified, what was falsified, exact procedures, open mandates — and successors start
  from that.
* **Work is a DAG.** [governance/tasks.json](governance/tasks.json) with disjoint file
  ownership; `governance/tools/waves.py` computes what can run in parallel.
* **Beliefs get revised in the open.** When real routing traces falsified our original
  cache-policy assumption, the claim was corrected in the docs with the evidence
  ([example](docs/results/qwen3-routing-and-cache-policy.md)).

## Roadmap

* Quiet-machine benchmark suite and published ablations (in progress)
* Multi-layer lookahead prefetch and prediction-aware eviction (motivated by the real traces)
* CUDA / HIP backend with a VRAM expert tier; Vulkan and Metal
* Lower-bit experts (Hadamard-rotated Q3/Q4, heat-aware per-expert precision), opt-in and measured
* Expert-major chunked prefill with VNNI/AMX tiled GEMM
* Swarm mode: pool RAM and NVMe across machines on a LAN
* More architectures (GLM-4.5/5, gpt-oss, latent-MoE models such as Kimi-K3)

## Prior art and credits

Hearth stands on a lot of public work. Closest in spirit is
[Colibrì](https://github.com/JustVugg/colibri) (Apache-2.0), which showed that streaming
experts from NVMe makes frontier MoE models usable on consumer machines. Related systems
and ideas: [llama.cpp](https://github.com/ggml-org/llama.cpp) (MoE CPU offload, n-gram
speculation), [KTransformers](https://github.com/kvcache-ai/ktransformers),
[MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity),
[Fiddler](https://github.com/efeslab/fiddler), HOBBIT and EdgeMoE (mixed-precision
experts), AdapMoE, ProMoE, Pre-gated MoE, mixtral-offloading, PowerInfer,
*LLM in a flash*, and prompt-lookup decoding. Hearth's code is original; no code was
copied from these projects.

## License

MIT — see [LICENSE](LICENSE). Model weights you convert keep their own licenses.
