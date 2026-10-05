#!/usr/bin/env python3
"""Record real routing traces (docs/FORMAT.md section 9) from a model on diverse chat prompts.

  python scripts/record_traces.py --model qwen3_q4.hearth --out traces/ --new-tokens 256

Writes one .hrtr per prompt plus all.hrtr (concatenated, same header), and prints
routing statistics (per-layer entropy, top-10% expert mass, token-to-token reuse).
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

PROMPTS = {
    "code": "Write a Python class implementing an LRU cache with get and put in O(1), with type hints and a short docstring. /no_think",
    "math": "A train leaves at 9:40 and travels 210 km at 84 km/h, then 150 km at 100 km/h. When does it arrive? Show the steps. /no_think",
    "story": "Write a short story about a lighthouse keeper who discovers that the lamp has been signalling to someone. /no_think",
    "facts": "Explain how vaccines train the immune system, mentioning antigens, B cells, T cells and memory cells. /no_think",
    "translate": "Translate into French and German: 'The library opens at nine, but the reading room stays closed until noon.' /no_think",
}


def read_trace(p: Path):
    raw = p.read_bytes()
    hdr = struct.unpack_from("<6I", raw, 0)
    ids = np.frombuffer(raw, "<u2", offset=24).reshape(-1, hdr[5], hdr[4])
    return hdr, ids


def stats(ids: np.ndarray, n_experts: int) -> dict:
    T, L, K = ids.shape
    ent, top10, reuse = [], [], []
    for l in range(L):
        c = np.bincount(ids[:, l].ravel(), minlength=n_experts).astype(np.float64)
        p = c / c.sum()
        nz = p[p > 0]
        ent.append(float(-(nz * np.log2(nz)).sum()))
        top10.append(float(np.sort(p)[::-1][: max(1, n_experts // 10)].sum()))
        if T > 1:
            prev, cur = ids[:-1, l], ids[1:, l]
            reuse.append(float(np.mean([len(set(a) & set(b)) / K for a, b in zip(prev, cur)])))
    return dict(tokens=T, entropy_bits=float(np.mean(ent)), uniform_entropy_bits=float(np.log2(n_experts)),
                top10pct_expert_mass=float(np.mean(top10)), next_token_reuse=float(np.mean(reuse)) if reuse else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--cache-gb", type=float, default=24.0)
    a = ap.parse_args()
    from hearth.chat import load_chat
    from hearth.engine import Engine
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, tmpl, meta = load_chat(a.model)
    stop = set(tok.eos_ids)
    parts = []
    with Engine(a.model, cache_gb=a.cache_gb) as e:
        for name, text in PROMPTS.items():
            ids = tok.encode(tmpl.render([{"role": "user", "content": text}], add_generation_prompt=True))
            e.reset()
            path = out / f"{name}.hrtr"
            e.trace_start(path)
            lg = e.eval(ids)
            n = 0
            while n < a.new_tokens:
                t = int(np.argmax(lg))
                if t in stop:
                    break
                lg = e.eval([t])
                n += 1
            e.trace_stop()
            hdr, tr = read_trace(path)
            parts.append((hdr, path.read_bytes()[24:]))
            print(f"{name:10s} prompt {len(ids):4d} + {n:4d} generated  {stats(tr, hdr[3])}", flush=True)
    hdr = parts[0][0]
    (out / "all.hrtr").write_bytes(struct.pack("<6I", *hdr) + b"".join(p for _, p in parts))
    _, allids = read_trace(out / "all.hrtr")
    print("ALL", stats(allids, hdr[3]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
