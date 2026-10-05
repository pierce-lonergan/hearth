"""hearth.sim - trace-driven simulator of expert caching, storage I/O and decode speed.

Predicts tokens/s, cache hit rates and bytes per token for any MoE preset on any
hardware profile, from synthetic or recorded routing traces. All outputs are
SIMULATED estimates; see docs/SIMULATOR.md for the equations and limitations.

    from hearth.sim import simulate
    r = simulate("kimi-k2", "this-pc", policy="lfu", cache_gb=40, spec=(4, 0.6))
    print(r.tok_s, r.hit_rate, r.bottleneck)
"""
from .cache import ONLINE_POLICIES, POLICIES, Prefetch, build_schedule, engine_min_slots, run_cache
from .core import Lossy, Result, Spec, TraceSpec, compare_policies, levers, resolve_trace, retime, simulate, sweep
from .feasibility import Feasibility
from .feasibility import check as feasibility
from .hardware import HARDWARE, Hardware, get_hardware, parse_nvme
from .timing import Calibration, ModelCosts, calibrate, model_costs, slab_bytes
from .trace import Trace, TraceStats, load_usage, save_usage, synthetic, synthetic_for, trace_stats


def main(argv=None) -> int:
    from .cli import main as _main
    return _main(argv)


__all__ = [
    "simulate", "sweep", "retime", "compare_policies", "levers", "calibrate", "feasibility", "main",
    "Result", "Spec", "Lossy", "TraceSpec", "Prefetch", "Calibration", "Hardware", "HARDWARE", "get_hardware",
    "parse_nvme", "Trace", "TraceStats", "synthetic", "synthetic_for", "trace_stats", "load_usage", "save_usage",
    "POLICIES", "ONLINE_POLICIES", "ModelCosts", "model_costs", "slab_bytes", "Feasibility", "build_schedule",
    "run_cache", "engine_min_slots", "resolve_trace",
]
