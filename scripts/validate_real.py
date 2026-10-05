#!/usr/bin/env python3
"""Validate Hearth on real weights.

  truncated  Keep the first N layers of a Hugging Face MoE checkpoint, run it in
             transformers (float32, eager) and in Hearth (F32 container), and compare
             logits and per-layer routing token by token. Real weight statistics,
             real tokenizer, small enough for the reference to fit in RAM.

  quality    Compare two containers of the same model (e.g. Q4 vs BF16) on a text:
             perplexity, mean KL(ref || test), top-1 agreement, and throughput.

Examples:
  python scripts/validate_real.py truncated --hf D:/models/Qwen3-30B-A3B --layers 4 --work D:/hearth-data/val
  python scripts/validate_real.py quality --test q4.hearth --ref bf16.hearth --tokens 4096
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

SAMPLE_TEXT = (
    "Mixture-of-Experts language models route each token to a small subset of expert networks. "
    "Because only a few experts are active per token, the model can hold far more parameters than "
    "it computes with. Running such a model on a personal computer is mostly a memory problem: the "
    "experts that are not needed right now can live on a fast solid-state drive and be streamed into "
    "RAM on demand. def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n"
    "    return a\n\nThe capital of France is Paris, and the Rhine flows through Germany."
)


def _routing_hooks(model, top_k: int, store: list):
    """Capture router top-k ids per layer (Qwen/OLMoE/Mixtral style `mlp.gate` linear)."""
    import torch
    handles = []
    for li, layer in enumerate(model.model.layers):
        gate = getattr(getattr(layer, "mlp", None), "gate", None)
        if gate is None or not hasattr(gate, "weight"):
            continue

        def hook(mod, inp, out, li=li):
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits.float(), dim=-1)
            store.append((li, torch.topk(probs, top_k, dim=-1).indices.cpu().numpy()))
        handles.append(gate.register_forward_hook(hook))
    return handles


def cmd_truncated(a) -> int:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from hearth import quant
    from hearth.convert import convert
    from hearth.engine import Engine

    hf, work = Path(a.hf), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    trunc = work / f"hf_first{a.layers}"
    cfg = AutoConfig.from_pretrained(hf)
    cfg.num_hidden_layers = a.layers
    if getattr(cfg, "mlp_only_layers", None):
        cfg.mlp_only_layers = [i for i in cfg.mlp_only_layers if i < a.layers]
    tok = AutoTokenizer.from_pretrained(hf)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(hf, config=cfg, torch_dtype=torch.float32,
                                                 attn_implementation="eager", low_cpu_mem_usage=True)
    model.eval()
    print(f"loaded first {a.layers} layers in {time.time() - t0:.0f}s", flush=True)
    if not (trunc / "config.json").exists():
        model.save_pretrained(trunc, safe_serialization=True)
        tok.save_pretrained(trunc)

    ids = tok(SAMPLE_TEXT)["input_ids"][: a.max_tokens]
    routes: list = []
    hooks = _routing_hooks(model, cfg.num_experts_per_tok, routes)
    with torch.no_grad():
        want = model(torch.tensor([ids])).logits[0].float().numpy()
    for h in hooks:
        h.remove()
    want_routes = np.stack([r for _, r in sorted(routes, key=lambda x: x[0])], axis=1)  # [T, L, K]
    del model

    results = {}
    for name, dt in (("f32", quant.F32), ("bf16", quant.BF16), ("q4", quant.Q4)):
        path = work / f"first{a.layers}_{name}.hearth"
        if not path.exists():
            t0 = time.time()
            kw = dict(expert_dtype=dt, dense_dtype=quant.Q8 if dt == quant.Q4 else dt,
                      embed_dtype=quant.Q8 if dt == quant.Q4 else dt, head_dtype=quant.Q8 if dt == quant.Q4 else dt)
            convert(trunc, path, progress=False, **kw)
            print(f"converted {name} in {time.time() - t0:.0f}s", flush=True)
        trace = work / f"first{a.layers}_{name}.hrtr"
        with Engine(path, cache_gb=64.0) as e:
            e.trace_start(trace)
            got = e.eval(ids, all_logits=True)
            e.trace_stop()
            one = None
            if name == "f32":  # INV-DET-2 on real weights: one-at-a-time must equal the batch
                e.reset()
                one = np.stack([e.eval([t]) for t in ids[:8]])
        raw = trace.read_bytes()
        n_moe, k = np.frombuffer(raw[:24], "<u4")[5], np.frombuffer(raw[:24], "<u4")[4]
        got_routes = np.frombuffer(raw, "<u2", offset=24).reshape(-1, n_moe, k)
        scale = max(1.0, float(np.abs(want).max()))
        lp_w = want - np.logaddexp.reduce(want, axis=-1, keepdims=True)
        lp_g = got - np.logaddexp.reduce(got, axis=-1, keepdims=True)
        kl = float(np.mean(np.sum(np.exp(lp_w) * (lp_w - lp_g), axis=-1)))
        same_set = np.mean([[set(got_routes[t, l]) == set(want_routes[t, l]) for l in range(n_moe)]
                            for t in range(len(ids))])
        results[name] = dict(
            max_abs_err=float(np.abs(got - want).max()), rel_err=float(np.abs(got - want).max() / scale),
            top1_agree=float(np.mean(got.argmax(-1) == want.argmax(-1))), mean_kl=kl,
            routing_set_agree=float(same_set), tokens=len(ids))
        if one is not None:
            results[name]["batch_vs_sequential_bit_identical"] = bool(np.array_equal(one, got[:8]))
        print(name, json.dumps(results[name]), flush=True)
    (work / f"truncated_{a.layers}_results.json").write_text(json.dumps(results, indent=1))
    return 0


def _token_stream(tokenizer_dir: Path | None, n: int) -> list[int]:
    from tokenizers import Tokenizer
    text = "\n\n".join((ROOT / p).read_text(encoding="utf-8") for p in
                       ["docs/ARCHITECTURE.md", "docs/NUMERICS.md", "docs/FORMAT.md", "governance/README.md",
                        "python/hearth/generate.py", "engine/src/hx_store.h"])
    tk = Tokenizer.from_file(str(tokenizer_dir))
    return tk.encode(text).ids[:n]


def cmd_quality(a) -> int:
    from hearth.chat import find_tokenizer, read_metadata
    from hearth.engine import Engine
    tpath = find_tokenizer(a.ref, read_metadata(a.ref))
    ids = _token_stream(tpath, a.tokens)
    out = {}
    logprobs = {}
    for name, path in (("ref", a.ref), ("test", a.test)):
        with Engine(path, cache_gb=a.cache_gb, max_batch=a.chunk) as e:
            t0 = time.time()
            lps, nll = [], 0.0
            for s in range(0, len(ids) - 1, a.chunk):
                chunk = ids[s:s + a.chunk]
                lg = e.eval(chunk, all_logits=True).astype(np.float64)
                lp = lg - np.logaddexp.reduce(lg, axis=-1, keepdims=True)
                tgt = ids[s + 1:s + 1 + len(chunk)]
                nll -= float(sum(lp[i, t] for i, t in enumerate(tgt)))
                lps.append(lp[: len(tgt)].astype(np.float32))
            dt = time.time() - t0
            st = e.stats()
        n = len(ids) - 1
        logprobs[name] = np.concatenate(lps)
        out[name] = dict(path=str(path), ppl=float(np.exp(nll / n)), tokens=n, seconds=dt,
                         prefill_tok_s=n / dt, gb_read=st.get("bytes_read", 0) / 1e9,
                         hit_rate=st.get("cache_hits", 0) / max(1, st.get("cache_hits", 0) + st.get("cache_misses", 0)))
        print(name, json.dumps(out[name]), flush=True)
    r, t = logprobs["ref"], logprobs["test"]
    out["mean_kl_ref_test"] = float(np.mean(np.sum(np.exp(r) * (r - t), axis=-1)))
    out["top1_agree"] = float(np.mean(r.argmax(-1) == t.argmax(-1)))
    print(json.dumps({k: out[k] for k in ("mean_kl_ref_test", "top1_agree")}), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("truncated")
    s.add_argument("--hf", required=True)
    s.add_argument("--layers", type=int, default=4)
    s.add_argument("--work", required=True)
    s.add_argument("--max-tokens", type=int, default=96)
    s = sub.add_parser("quality")
    s.add_argument("--test", required=True)
    s.add_argument("--ref", required=True)
    s.add_argument("--tokens", type=int, default=4096)
    s.add_argument("--chunk", type=int, default=512)
    s.add_argument("--cache-gb", type=float, default=40.0)
    s.add_argument("--out")
    a = ap.parse_args()
    return {"truncated": cmd_truncated, "quality": cmd_quality}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
