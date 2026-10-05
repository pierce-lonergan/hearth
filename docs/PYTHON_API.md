# Python package contract (`python/hearth`)

The Python layer prepares models, provides the numerical ground truth, and
drives the C engine. **numpy is the only hard dependency.** `torch`,
`safetensors`, `transformers`, `tokenizers` are optional (conversion / chat).
The package is importable with `PYTHONPATH=python`.

| module | owner role | purpose |
|---|---|---|
| `hearth._native` | model | locate + load the shared library via ctypes; raw kernel bindings (quantize, matmul) |
| `hearth.quant` | model | dtype constants, `row_bytes`, `quantize` (native), numpy `dequantize`, `act_quant_q8` emulation |
| `hearth.format` | model | `ContainerWriter`, `ContainerReader` (docs/FORMAT.md) |
| `hearth.reference` | model | numpy forward pass — numerical ground truth (docs/NUMERICS.md) |
| `hearth.convert` | model | Hugging Face → `.hearth` (qwen3_moe, olmoe, mixtral, qwen2_moe, deepseek_v3/kimi_k2) |
| `hearth.synth` | model | random containers: tiny (tests) and huge-shaped (benchmarks, may alias slabs) |
| `hearth.engine` | runtime | `Engine` class wrapping the C API; ctypes mirrors of `hearth_options` / `hearth_model_info` / `hearth_stats` |
| `hearth.generate` | runtime | sampling, prompt-lookup speculative decoding |
| `hearth.chat` / `hearth.server` | runtime | tokenizer + chat template, OpenAI-compatible HTTP server |
| `hearth.presets` | maintainers | real model shapes (verified configs) |
| `hearth.sim` | sim | trace-driven cache/IO/roofline simulator |
| `hearth.cli` | runtime | `python -m hearth <cmd>` |

## `hearth._native`

```python
def lib() -> ctypes.CDLL            # cached; search order: $HEARTH_LIB, python/hearth/lib/,
                                    # <data dir>/build/**/hearth.(dll|so|dylib); raises HearthLibNotFound
def available() -> bool
def data_dir() -> pathlib.Path      # $HEARTH_DATA or %LOCALAPPDATA%/hearth or ~/.cache/hearth
def matmul(dtype, W: bytes, rows, cols, X: np.ndarray, isa: int = 0) -> np.ndarray | None
                                    # hearth_matmul; X float32 [T, cols] -> Y float32 [T, rows];
                                    # returns None if the CPU cannot run `isa` (1 scalar, 2 avx2, 3 avx512)
class HearthLibNotFound(RuntimeError)
```

## `hearth.quant`

```python
F32, F16, BF16, Q8, Q4, I32, U8 = 0, 1, 2, 3, 4, 5, 6
def row_bytes(dtype: int, n: int) -> int
def quantize(x: np.ndarray, dtype: int) -> bytes        # x float32 [rows, cols]; Q8/Q4 call the C quantizer
                                                        # (authoritative); F32/F16/BF16 done in numpy (RNE)
def dequantize(buf: bytes | np.ndarray, dtype: int, shape: tuple) -> np.ndarray   # exact, pure numpy
def act_quant_q8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]  # NUMERICS §2 emulation: (q int8 [..., n], d float32 [..., n/64])
```

## `hearth.format`

```python
class ContainerWriter:
    def __init__(self, path, meta: dict): ...
    # Types are taken from a schema table of FORMAT.md §3.1 keys (layer_kind -> u8[], eos_ids -> u32[], ...);
    # unknown keys: int -> u32/u64, float -> f32, str -> str, list[int] -> u32[], list[float] -> f32[].
    def declare_tensor(self, name: str, shape: tuple[int, ...], dtype: int) -> None
    def declare_experts(self, n_layers: int, n_experts: int, d_model: int, ffn_dim: int,
                        dtype: int | Callable[[int, int], int], moe_layers: Sequence[int],
                        alias: Callable[[int, int], tuple[int, int] | None] | None = None) -> None
    def write_tensor(self, name: str, data: np.ndarray | bytes) -> None      # float array -> encoded to declared dtype
    def write_expert(self, layer: int, expert: int, gate, up, down) -> None  # float32 arrays [F,D],[F,D],[D,F] or encoded bytes per matrix
    def close(self) -> None   # writes preamble, metadata, directories; verifies every declared item was written
    # context manager; positional writes so huge models stream without buffering

class ContainerReader:
    def __init__(self, path): ...
    meta: dict; config: dict           # config = meta with FORMAT.md defaults applied
    tensors: dict[str, TensorInfo]     # TensorInfo(name, dtype, shape, offset, nbytes)
    def read_tensor(self, name, dequant=True) -> np.ndarray | bytes
    def expert_entry(self, layer, e) -> ExpertEntry   # (offset, nbytes, dtype, flags)
    def read_expert(self, layer, e, dequant=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

## `hearth.reference`

```python
class Reference:
    def __init__(self, path_or_reader, emulate_act_quant: bool = True): ...
    # emulate_act_quant: quantize activations exactly like the engine for Q8/Q4 weights
    def reset(self) -> None
    def eval(self, tokens: Sequence[int]) -> np.ndarray     # logits [n, vocab] float32, appends to its KV cache
    def routing_history(self) -> np.ndarray                # uint16 [tokens since reset, n_moe_layers, top_k], rank order
```

## `hearth.synth`

```python
def make_tiny(path, *, arch: str = "qwen3_moe", dtype: int = F32, seed: int = 0, n_layers: int = 3,
              d_model: int = 128, n_experts: int = 8, top_k: int = 2, expert_ffn: int = 64,
              vocab: int = 256, max_seq: int = 256) -> Path
    # Random weights exercising every code path of `arch` (qwen3_moe, olmoe, mixtral, qwen2_moe,
    # deepseek_v3 = MLA + sigmoid/bias/group routing + shared expert + leading dense layer).
    # dtype applies to experts and big dense matrices (norms/router stay F32; for Q4 the dense
    # matrices use Q8). Router weights are scaled so routing has clear margins (no near-ties).
def make_shaped(path, *, preset: str | dict, expert_dtype=Q4, physical_experts: int | None = None,
                seed: int = 0) -> Path
    # Benchmark container with a real model's *shape* (e.g. "qwen3-30b-a3b", "kimi-k2") and random
    # weights. physical_experts < logical experts => slabs are aliased (flag bit 0) to fit on disk.
```

## `hearth.convert`

```python
def convert(src: str | Path, dst: str | Path, *, expert_dtype=Q4, dense_dtype=Q8, embed_dtype=Q8,
            head_dtype=Q8, max_seq: int | None = None, threads: int = 0, progress=True) -> Path
```
Reads a local Hugging Face directory (config.json + *.safetensors; FP8 block-scaled
weights are dequantized), maps tensor names to FORMAT.md §4.1, computes
`rope_inv_freq`/`rope_attn_factor`/`attn_scale` with the framework's own RoPE code
when transformers is available, copies tokenizer.json and the chat template.

## `hearth.engine`

```python
class Engine:
    def __init__(self, model_path, *, cache_gb=8.0, threads=0, io_threads=0, direct_io=True,
                 policy="lfu", prefetch="shared", prefetch_extra=0, usage_in=None, usage_out=None,
                 pin_fraction=0.0, warm_start=False, max_seq=0, max_batch=0, isa="auto",
                 mirrors=(), verbose=0): ...
    # int options must be in [0, 2**31-1] (ValueError; TypeError for non-integers);
    # pin_fraction in [0, 1]; cache_gb finite >= 0
    info: dict
    def eval(self, tokens, all_logits=False) -> np.ndarray   # [vocab] or [n, vocab]
    pos: int
    kv_capacity: int   # max_seq > 0 ? min(max_seq, info['max_seq']) : min(info['max_seq'], 4096)
    def reset(self); def rewind(self, pos)
    def stats(self) -> dict; def reset_stats(self)
    def trace_start(self, path); def trace_stop(self); def route_replay(self, path)
    def close(self)   # also context manager
```

## `hearth.server`

```python
def serve(model_path, *, host='127.0.0.1', port=8080, model_name=None, api_key=None, max_queue=8,
          speculative='none', draft_len=4, ngram_n=3, cors=None, verbose=False, engine_kwargs=None,
          tokenizer_path=None, timeout=60.0) -> None
```
Clients stalling a read/write for `timeout` s are dropped; non-boolean `stream`/`echo` give 400;
without a tokenizer `/v1/completions` accepts token-id prompts and returns a non-standard
`token_ids` list; chat needs `max_tokens >= 1`. `hearth.chat.Tokenizer.from_container(model_path,
meta=None, path=None)`; `Conversation.say` rolls a turn back when generation raises.
