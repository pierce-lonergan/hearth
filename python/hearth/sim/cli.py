"""Command line: ``python -m hearth.sim`` (and ``python -m hearth sim`` once the runtime wires it)."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from .. import presets
from . import report
from .cache import ONLINE_POLICIES, POLICIES, POLICY_HELP, Prefetch
from .core import Lossy, Spec, TraceSpec, levers, resolve_trace, simulate, sweep
from .feasibility import check
from .hardware import HARDWARE, get_hardware, parse_nvme
from .timing import GPU_MODES, Calibration, model_costs


def _floats(s: str) -> list:
    return [float(x) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hearth sim",
        description="Trace-driven simulator of Hearth decode speed and expert-cache behaviour. "
                    "All outputs are SIMULATED estimates (docs/SIMULATOR.md).")
    p.add_argument("--list", action="store_true", help="list model presets, hardware profiles and policies")
    g = p.add_argument_group("model and hardware")
    g.add_argument("--model", default="kimi-k2", help="preset name (hearth.presets), default kimi-k2")
    g.add_argument("--hw", default="this-pc", help="hardware profile, default this-pc")
    g.add_argument("--ram-gib", type=float)
    g.add_argument("--dram-gbs", type=float, help="sustained DRAM read GB/s for the CPU kernels")
    g.add_argument("--nvme", help="drives x GB/s each, e.g. 2x6.5")
    g.add_argument("--nvme-latency-us", type=float)
    g.add_argument("--nvme-capacity-gb", type=float)
    g.add_argument("--io-cap-gbs", type=float, help="platform cap on aggregate storage GB/s")
    g.add_argument("--int8-tops", type=float, dest="cpu_int8_tops")
    g.add_argument("--os-reserve-gib", type=float)
    g.add_argument("--vram-gib", type=float)
    g.add_argument("--vram-gbs", type=float)
    g.add_argument("--pcie-gbs", type=float)
    g.add_argument("--expert-bits", type=float, default=4.25, help="expert bits/weight (Q4 = 4.25)")
    g.add_argument("--dense-bits", type=float, default=8.25, help="backbone bits/weight (Q8 = 8.25)")
    g.add_argument("--context", type=int, default=1024, help="average context length during decode")
    g.add_argument("--kv-bytes", type=float, default=4.0, help="bytes per KV element")
    g = p.add_argument_group("trace")
    g.add_argument("--trace", help="routing trace (.hrtr, docs/FORMAT.md sec. 9) instead of a synthetic one")
    g.add_argument("--tokens", type=int, default=2000, help="synthetic trace length")
    g.add_argument("--zipf", type=float, default=1.1, help="synthetic per-layer popularity exponent")
    g.add_argument("--reuse", type=float, default=0.25, help="synthetic P(expert reused from previous token)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--save-trace", help="write the (synthetic) trace to this .hrtr file")
    g.add_argument("--warmup", type=int, help="leading tokens excluded from stats (default 10%%)")
    g.add_argument("--profile", default="warmup", help="heat source for pinning: warmup | oracle | FILE.usage")
    g.add_argument("--stats", action="store_true", help="print trace statistics")
    g = p.add_argument_group("cache")
    g.add_argument("--policy", default="lfu", choices=POLICIES)
    g.add_argument("--compare-policies", action="store_true", help="run every policy (incl. belady)")
    g.add_argument("--no-belady", action="store_true", help="with --compare-policies: online policies only")
    g.add_argument("--cache-gb", type=float, help="DRAM expert cache GiB (default: all free RAM)")
    g.add_argument("--sweep-cache", help="comma-separated cache sizes in GiB")
    g.add_argument("--sweep-zipf", help="comma-separated Zipf exponents (synthetic traces)")
    g.add_argument("--lfu-decay", type=float, default=0.995)
    g.add_argument("--lfu-samples", type=int, default=0, help="sampled eviction like the engine (0 = exact)")
    g.add_argument("--pin-fraction", type=float, default=0.5)
    g.add_argument("--io-threads", type=int, default=8)
    g = p.add_argument_group("features")
    g.add_argument("--prefetch", type=float, default=0.0, help="next-layer prediction recall (0 = off)")
    g.add_argument("--prefetch-extra", type=int, default=0, help="extra predicted experts per layer")
    g.add_argument("--spec-k", type=int, default=0, help="speculative draft length (0 = off)")
    g.add_argument("--spec-alpha", type=float, default=0.6, help="per-draft acceptance probability")
    g.add_argument("--gpu", default="off", choices=GPU_MODES, help="GPU what-if (roadmap R01)")
    g = p.add_argument_group("LOSSY what-ifs (change model output)")
    g.add_argument("--lossy-topk", type=int)
    g.add_argument("--lossy-skip-rank", type=int)
    g.add_argument("--lossy-cold-bits", type=float)
    g.add_argument("--lossy-cold-frac", type=float, default=0.5)
    g = p.add_argument_group("calibration")
    g.add_argument("--calib", help="JSON file with Calibration fields")
    g.add_argument("--overhead-ms", type=float)
    g.add_argument("--layer-overhead-us", type=float)
    g.add_argument("--dram-eff", type=float)
    g.add_argument("--io-eff", type=float)
    g.add_argument("--compute-eff", type=float)
    g = p.add_argument_group("output")
    g.add_argument("--feasibility", action="store_true", help="only report whether/how the model fits")
    g.add_argument("--levers", action="store_true", help="one-at-a-time what-if table")
    g.add_argument("--markdown", action="store_true")
    g.add_argument("--json", action="store_true")
    return p


def _hw_overrides(a) -> dict:
    ov = dict(ram_gib=a.ram_gib, dram_gbs=a.dram_gbs, nvme_latency_us=a.nvme_latency_us,
              nvme_capacity_gb=a.nvme_capacity_gb, io_cap_gbs=a.io_cap_gbs, cpu_int8_tops=a.cpu_int8_tops,
              os_reserve_gib=a.os_reserve_gib, vram_gib=a.vram_gib, vram_gbs=a.vram_gbs, pcie_gbs=a.pcie_gbs)
    if a.nvme:
        ov["nvme_count"], ov["nvme_gbs"] = parse_nvme(a.nvme)
    return {k: v for k, v in ov.items() if v is not None}


def _calib(a) -> Calibration:
    c = Calibration()
    if a.calib:
        with open(a.calib, encoding="utf-8") as f:
            c = Calibration(**json.load(f))
    ov = dict(overhead_ms=a.overhead_ms, layer_overhead_us=a.layer_overhead_us, dram_eff=a.dram_eff,
              io_eff=a.io_eff, compute_eff=a.compute_eff)
    return replace(c, **{k: v for k, v in ov.items() if v is not None})


def _list() -> str:
    out = ["models (hearth.presets):"]
    for n, s in presets.PRESETS.items():
        out.append(f"  {n:20s} {s.params_total() / 1e9:7.0f}B total, {s.params_active() / 1e9:5.1f}B active, "
                   f"{s.n_moe_layers} MoE layers x {s.n_experts} experts, top-{s.top_k}")
    out.append("hardware:")
    for n, h in HARDWARE.items():
        out.append(f"  {n:24s} {h.notes}")
    out.append("policies:")
    out += [f"  {n:11s} {POLICY_HELP[n]}" for n in POLICIES]
    return "\n".join(out)


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return _run(a)
    except (KeyError, ValueError, TypeError, OSError) as e:
        print(f"hearth sim: error: {e}", file=sys.stderr)
        return 2


def _run(a) -> int:
    if a.list:
        print(_list())
        return 0
    md = a.markdown
    hw = get_hardware(a.hw, **_hw_overrides(a))
    shape = presets.get(a.model)
    lossy = Lossy(a.lossy_topk, a.lossy_skip_rank, a.lossy_cold_bits, a.lossy_cold_frac)
    # always constructed so that out-of-range values are rejected; recall 0 + extra 0 and k = 0 mean off
    base = dict(model=a.model, hardware=hw, policy=a.policy, cache_gb=a.cache_gb,
                prefetch=Prefetch(a.prefetch, a.prefetch_extra), spec=Spec(a.spec_k, a.spec_alpha), lossy=lossy, gpu=a.gpu,
                warmup=a.warmup, profile=a.profile, expert_bits=a.expert_bits, dense_bits=a.dense_bits,
                context=a.context, kv_elem_bytes=a.kv_bytes, io_threads=a.io_threads, lfu_decay=a.lfu_decay,
                lfu_samples=a.lfu_samples, pin_fraction=a.pin_fraction, calib=_calib(a), seed=a.seed)
    trace = a.trace if a.trace else TraceSpec(a.tokens, a.zipf, a.reuse, a.seed)
    out = []
    payload = {}

    if a.feasibility:
        costs = model_costs(shape, expert_bits=a.expert_bits, dense_bits=a.dense_bits, context=a.context,
                            kv_elem_bytes=a.kv_bytes)
        f = check(costs, hw, gpu_mode=a.gpu, io_threads=a.io_threads, shape=shape)
        out.append(f"{shape.name} on {hw.name}:")
        out.append(report.format_feasibility(f, md))
        payload["feasibility"] = f.to_dict()
        return _emit(a, out, payload)

    if a.stats or a.save_trace:
        tr = resolve_trace(shape, trace)
        if a.save_trace:
            tr.save(a.save_trace)
            out.append(f"wrote {a.save_trace} ({tr.n_tokens} tokens)")
        if a.stats:
            st = tr.stats()
            out.append(f"trace statistics ({tr.source}):")
            out.append(report.format_stats(st, md))
            payload["trace_stats"] = st.summary()
        out.append("")

    if a.levers:
        rows = levers(**{**base, "trace": trace})
        out.append(report.SIM_BANNER)
        out.append(report.describe_setup(rows[0][1]))
        out.append("")
        out.append(report.format_levers(rows, md, title=f"levers: {shape.name} on {hw.name}"))
        payload["levers"] = [{"lever": lbl, **r.to_dict()} for lbl, r in rows]
        return _emit(a, out, payload)

    grid = {}
    if a.compare_policies:
        grid["policy"] = list(ONLINE_POLICIES if a.no_belady else POLICIES)
    if a.sweep_cache:
        grid["cache_gb"] = _floats(a.sweep_cache)
    if a.sweep_zipf:
        if a.trace:
            raise ValueError("--sweep-zipf needs a synthetic trace (no --trace)")
        grid["trace"] = [TraceSpec(a.tokens, z, a.reuse, a.seed) for z in _floats(a.sweep_zipf)]
    if grid:
        results = sweep(grid, **{**base, **({"trace": trace} if "trace" not in grid else {})})
        keys = list(grid)
        import itertools
        labels = []
        for combo in itertools.product(*(grid[k] for k in keys)):
            bits = []
            for k, v in zip(keys, combo):
                if k == "trace":
                    bits.append(f"zipf={v.zipf:g}")
                elif k == "cache_gb":
                    bits.append(f"{v:g}GiB")
                else:
                    bits.append(str(v))
            labels.append(" ".join(bits))
        out.append(report.SIM_BANNER)
        out.append(report.describe_setup(results[0]))
        out.append("")
        out.append(report.format_results(results, labels, md))
        payload["results"] = [dict(case=ly, **r.to_dict()) for ly, r in zip(labels, results)]
        return _emit(a, out, payload)

    r = simulate(trace=trace, **base)
    out.append(report.format_result(r, md))
    payload["result"] = r.to_dict()
    return _emit(a, out, payload)


def _emit(a, out, payload) -> int:
    if a.json:
        print(json.dumps(payload, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    else:
        print("\n".join(out).rstrip())
    return 0
