#!/usr/bin/env python3
"""Reproducible Hearth benchmark matrix on one model.

Method (so every configuration does identical work):
  1. Generate a reference token sequence once: prompt + N greedy tokens, model fully cached.
  2. For each configuration, open a fresh engine (cold cache; direct I/O bypasses the OS
     cache), prefill the prompt as one batch, then feed the generated tokens one at a time
     (teacher forcing) and time each step.
  3. Assert the logits of every configuration are bit-identical to the reference run
     (INV-DET-1 on real weights) and report decode tok/s (all steps and the warm second
     half), hit rate, GB read, stall fraction and prefetch accuracy.

A heat profile for pinning is recorded from a *different* text, so pinned runs do not
get an oracle profile of the measured sequence.

  python scripts/bench_suite.py --model qwen3_q4.hearth --out bench.json --decode 128 \
      --matrix default           # or: --matrix quick | cache | policy | full
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

PROMPT = ("You are a careful senior engineer. Explain, step by step, how a CPU cache works, why "
          "memory bandwidth limits large language model inference, and how a Mixture-of-Experts "
          "model differs from a dense model. Then write a short Python function that computes a "
          "moving average over a list of numbers.")
PROFILE_TEXT = ("The history of the printing press begins with movable type. Gutenberg combined a "
                "screw press, oil-based ink and metal type cast in a hand mould. Within decades, "
                "presses spread across Europe, books became cheaper, and literacy rose. " * 6)


def matrices(total_gb: float) -> dict[str, list[dict]]:
    small = [g for g in (2.0, 4.0, 8.0) if g < total_gb * 0.8]
    base = dict(policy="lfu", prefetch="shared", direct_io=True)
    out = {
        "quick": [dict(name="all-cached", cache_gb=total_gb * 1.1, **base)]
                 + [dict(name=f"cache-{g:g}GB", cache_gb=g, **base) for g in small[-1:]],
        "cache": [dict(name=f"cache-{g:g}GB", cache_gb=g, **base) for g in small]
                 + [dict(name="all-cached", cache_gb=total_gb * 1.1, **base)],
    }
    pol = []
    for g in small[-2:]:
        pol += [
            dict(name=f"{g:g}GB naive (LRU, no prefetch, buffered)", cache_gb=g, policy="lru", prefetch="off", direct_io=False),
            dict(name=f"{g:g}GB LRU + direct", cache_gb=g, policy="lru", prefetch="off", direct_io=True),
            dict(name=f"{g:g}GB LFU + direct", cache_gb=g, policy="lfu", prefetch="off", direct_io=True),
            dict(name=f"{g:g}GB LFU + next-layer prefetch", cache_gb=g, policy="lfu", prefetch="next", direct_io=True),
            dict(name=f"{g:g}GB LFU + shared-corrected prefetch", cache_gb=g, policy="lfu", prefetch="shared", direct_io=True),
            dict(name=f"{g:g}GB LFU + prefetch + pinned profile", cache_gb=g, policy="lfu", prefetch="shared",
                 direct_io=True, pin=0.5),
        ]
    out["policy"] = pol
    out["default"] = out["cache"] + pol
    out["full"] = out["default"]
    return out


def system_info() -> dict:
    from hearth._native import lib  # noqa: F401  (ensures the DLL resolves)
    from hearth.engine import cpu_isa, native_version
    info = dict(python=sys.version.split()[0], platform=platform.platform(), processor=platform.processor(),
                hearth=native_version(), isa=cpu_isa())
    try:
        import psutil  # optional
        info["ram_gb"] = psutil.virtual_memory().total / 2**30
    except Exception:
        pass
    return info


def tokenize(model: Path, text: str) -> list[int]:
    from hearth.chat import Tokenizer
    return Tokenizer.from_container(model).encode(text)


def run_config(model: Path, cfg: dict, prompt: list[int], gen: list[int], ref_logits: np.ndarray | None,
               threads: int, usage: Path | None) -> dict:
    from hearth.engine import Engine
    kw = dict(cache_gb=cfg["cache_gb"], policy=cfg["policy"], prefetch=cfg["prefetch"],
              direct_io=cfg["direct_io"], threads=threads)
    if cfg.get("pin") and usage is not None:
        kw.update(usage_in=str(usage), pin_fraction=cfg["pin"])
    t_open = time.perf_counter()
    with Engine(model, **kw) as e:
        open_s = time.perf_counter() - t_open
        info = e.info
        t0 = time.perf_counter()
        e.eval(prompt)
        prefill_s = time.perf_counter() - t0
        e.reset_stats()
        steps, logits = [], []
        for t in gen:
            t1 = time.perf_counter()
            logits.append(e.eval([t]))
            steps.append(time.perf_counter() - t1)
        st = e.stats()
    logits = np.stack(logits)
    half = len(steps) // 2
    res = dict(cfg, open_s=open_s, prefill_tokens=len(prompt), prefill_tok_s=len(prompt) / prefill_s,
               decode_tokens=len(gen), decode_tok_s=len(steps) / sum(steps),
               decode_tok_s_warm=(len(steps) - half) / sum(steps[half:]),
               p50_ms=1e3 * float(np.median(steps)), hit_rate=st["hit_rate"], gb_read=st["gb_read"],
               gb_per_token=st["gb_read"] / max(1, len(gen)), stall_frac=st["stall_frac"],
               prefetch_accuracy=st["prefetch_accuracy"], cache_slots=info.get("cache_slots"),
               read_errors=st.get("read_errors", 0))
    if ref_logits is not None:
        res["bit_identical_to_reference"] = bool(np.array_equal(logits, ref_logits))
    return res, logits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--matrix", default="default", choices=["quick", "cache", "policy", "default", "full"])
    ap.add_argument("--trace", help="also record the reference run's routing trace here (for the simulator)")
    a = ap.parse_args()

    from hearth.engine import Engine
    model = Path(a.model)
    out = Path(a.out)
    with Engine(model, cache_gb=1.0) as e:
        info = e.info
    total_gb = (info["expert_bytes"]) / 2**30
    print(f"model {model.name}: {info['arch']}, experts {total_gb:.1f} GiB, dense {info['dense_bytes']/2**30:.1f} GiB", flush=True)

    prompt = tokenize(model, PROMPT)
    usage = out.with_suffix(".usage")
    # 1. reference sequence + heat profile from a different text
    with Engine(model, cache_gb=total_gb * 1.1, usage_out=str(usage), threads=a.threads) as e:
        e.eval(tokenize(model, PROFILE_TEXT))
    with Engine(model, cache_gb=total_gb * 1.1, threads=a.threads) as e:
        if a.trace:
            e.trace_start(a.trace)
        lg = e.eval(prompt)
        gen = []
        for _ in range(a.decode):
            t = int(np.argmax(lg))
            gen.append(t)
            lg = e.eval([t])
        if a.trace:
            e.trace_stop()
    print(f"reference: {len(prompt)} prompt + {len(gen)} generated tokens", flush=True)

    report = dict(system=system_info(), model=str(model), model_info=info, prompt_tokens=len(prompt),
                  decode_tokens=len(gen), runs=[])
    ref_logits = None
    for cfg in [dict(name="reference (all cached)", cache_gb=total_gb * 1.1, policy="lfu", prefetch="shared",
                     direct_io=True)] + matrices(total_gb)[a.matrix]:
        res, logits = run_config(model, cfg, prompt, gen, ref_logits, a.threads, usage)
        if ref_logits is None:
            ref_logits = logits
        report["runs"].append(res)
        print(f"{res['name']:48s} decode {res['decode_tok_s']:7.2f} tok/s (warm {res['decode_tok_s_warm']:7.2f})"
              f"  hit {res['hit_rate']*100:5.1f}%  {res['gb_per_token']:.3f} GB/tok  stall {res['stall_frac']*100:4.1f}%"
              f"  prefill {res['prefill_tok_s']:7.1f} tok/s  identical={res.get('bit_identical_to_reference', '-')}",
              flush=True)
        out.write_text(json.dumps(report, indent=1))
    bad = [r["name"] for r in report["runs"] if r.get("bit_identical_to_reference") is False]
    if bad:
        print("INV-DET-1 VIOLATION in:", bad, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
