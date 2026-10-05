"""Plain-text and Markdown rendering of simulator results."""
from __future__ import annotations

from .timing import COMPONENTS

GB = 1e9
SIM_BANNER = "SIMULATED (hearth.sim model, not a measurement)"


def table(headers, rows, markdown: bool = False) -> str:
    rows = [[str(c) for c in r] for r in rows]
    if markdown:
        out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    fmt = lambda r: "  ".join(c.rjust(w[i]) if i else c.ljust(w[i]) for i, c in enumerate(r))  # noqa: E731
    return "\n".join([fmt(headers), fmt(["-" * x for x in w])] + [fmt(r) for r in rows])


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


RESULT_HEADERS = ["case", "cache GiB", "hit", "prefetch", "miss", "NVMe GB/tok", "DRAM GB/tok", "tok/s",
                  "ms/tok", "bottleneck"]


def result_row(r, label: str | None = None) -> list:
    flag = "" if r.feasible else " (INFEASIBLE)"
    return [(label or r.policy) + flag, f"{r.cache_gib:.1f}", _pct(r.hit_rate), _pct(r.prefetch_rate),
            _pct(r.miss_rate), f"{r.bytes_per_token['nvme'] / GB:.2f}", f"{r.bytes_per_token['dram'] / GB:.2f}",
            f"{r.tok_s:.3g}", f"{r.ms_per_token:.0f}", r.bottleneck]


def format_results(results, labels=None, markdown: bool = False, title: str | None = None) -> str:
    rows = [result_row(r, labels[i] if labels else None) for i, r in enumerate(results)]
    out = []
    if title:
        out.append(("### " if markdown else "") + title)
    out.append(table(RESULT_HEADERS, rows, markdown))
    notes = sorted({w for r in results for w in r.warnings + r.lossy})
    if notes:
        out.append("")
        out.extend(("- " if markdown else "  ! ") + n for n in notes)
    return "\n".join(out)


def describe_setup(r) -> str:
    s = r.settings
    hw = s["hw"]
    parts = [f"model {r.model}", f"hardware {r.hardware} (DRAM {hw['dram_gbs']:g} GB/s, RAM {hw['ram_gib']:g} GiB, "
             f"NVMe {hw['nvme_count']}x{hw['nvme_gbs']:g} GB/s, int8 {hw['cpu_int8_tops']:g} TOPS)",
             f"trace {r.trace_source} ({s['trace_tokens']} tokens, first {r.warmup_tokens} = warm-up)",
             f"experts {s['expert_bits']} bpw, backbone {s['dense_bits']} bpw, context {s['context']}"]
    pf = s.get("prefetch")
    parts.append("prefetch " + (f"recall {pf['recall']} + {pf['extra']} extra" if pf else "off"))
    sp = s.get("spec")
    parts.append("speculative " + (f"k={sp['k']} alpha={sp['alpha']}" if sp else "off"))
    if s.get("gpu", "off") != "off":
        parts.append(f"gpu {s['gpu']} (roadmap R01, not in the engine yet)")
    return "; ".join(parts)


def format_result(r, markdown: bool = False) -> str:
    b = r.bytes_per_token
    t = r.time_per_token_ms
    lines = [f"{SIM_BANNER}", describe_setup(r), ""]
    kv = [
        ("policy", f"{r.policy}  ({r.slots} slots x {r.slab_bytes / 2**20:.1f} MiB = {r.cache_gib:.1f} GiB"
                   + (f", {r.n_pinned} pinned" if r.n_pinned else "") + ")"),
        ("decode speed", f"{r.tok_s:.3g} tok/s  ({r.ms_per_token:.1f} ms/token, {r.tokens_per_step:.2f} tokens/step)"),
        ("expert loads", f"{r.loads_per_token:.1f} per token: hit {_pct(r.hit_rate)}"
                         + (f" (VRAM {_pct(r.vram_rate)})" if r.vram_rate else "")
                         + f", prefetched {_pct(r.prefetch_rate)}, demand miss {_pct(r.miss_rate)}"
                         + (f", skipped {_pct(r.skip_rate)}" if r.skip_rate else "")),
        ("NVMe per token", f"{b['nvme'] / GB:.3f} GB (demand {b['nvme_demand'] / GB:.3f}, prefetch "
                           f"{b['nvme_prefetch'] / GB:.3f} of which wasted {b['nvme_prefetch_wasted'] / GB:.3f})"),
        ("DRAM per token", f"{b['dram'] / GB:.3f} GB" + (f";  VRAM {b['vram'] / GB:.3f} GB" if b['vram'] else "")),
        ("time per token", ", ".join(f"{c} {t[c]:.1f} ms" for c in COMPONENTS if t.get(c, 0) > 0.05)),
        ("bottleneck", r.bottleneck),
    ]
    if r.prefetch_rate:
        kv.append(("late prefetches", _pct(r.late_prefetch_rate)))
    if markdown:
        lines += ["| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in kv]
    else:
        w = max(len(k) for k, _ in kv)
        lines += [f"{k.ljust(w)}  {v}" for k, v in kv]
    notes = r.lossy + r.warnings + list(r.settings.get("cost_notes", []))
    if notes:
        lines.append("")
        lines += [("- " if markdown else "! ") + n for n in notes]
    return "\n".join(lines)


def format_feasibility(f, markdown: bool = False) -> str:
    kv = [
        ("fits", "yes" if f.fits else "NO - " + "; ".join(f.reasons)),
        ("parameters", f"{f.params_total / 1e9:.0f} B, of which {f.params_resident / 1e9:.2f} B resident "
                       "(backbone incl. embeddings)"),
        ("backbone", f"{f.backbone_gib:.1f} GiB in {f.backbone_location}; KV {f.kv_gib:.2f} GiB"),
        ("RAM left for expert cache", f"{f.ram_free_for_cache_gib:.1f} GiB (engine minimum {f.min_cache_gib:.2f} GiB)"),
        ("minimum RAM", f"{f.min_ram_gib:.1f} GiB"),
        ("recommended cache", f"{f.recommended_cache_gib:.1f} GiB of {f.expert_total_gib:.0f} GiB of experts "
                              f"({100 * f.recommended_cache_gib / max(f.expert_total_gib, 1e-9):.1f}%)"),
        ("model file", f"{f.file_gb:.0f} GB" + ("" if f.file_fits_drive else " (does NOT fit one drive)")),
    ]
    if f.vram_expert_slots:
        kv.append(("VRAM expert tier", f"{f.vram_expert_slots} slabs"))
    if f.disk_gb_by_format:
        kv.append(("disk by expert format", ", ".join(f"{k}: {v:.0f} GB" for k, v in f.disk_gb_by_format.items())))
    if markdown:
        return "\n".join(["| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in kv])
    w = max(len(k) for k, _ in kv)
    return "\n".join(f"{k.ljust(w)}  {v}" for k, v in kv)


def format_stats(st, markdown: bool = False) -> str:
    s = st.summary()
    kv = [
        ("tokens x layers x top-k", f"{s['tokens']} x {s['moe_layers']} x {s['top_k']} (E = {s['experts']})"),
        ("entropy / log2(E)", f"mean {s['entropy_norm_mean']:.3f}, min {s['entropy_norm_min']:.3f}"),
        ("reuse from previous token", f"{s['reuse_rate_mean']:.3f}"),
        ("mass in top x% (per layer)", ", ".join(f"{k}: {v:.3f}" for k, v in s["mass_top"].items())),
        ("mass in top x% (all slabs)", ", ".join(f"{k}: {v:.3f}" for k, v in s["mass_top_pooled"].items())),
        ("slabs to cover x% of loads", ", ".join(f"{k}: {v}" for k, v in s["slots_for_mass"].items())),
        ("distinct experts / layer over w tokens", ", ".join(f"w={k}: {v}" for k, v in s["union_per_window"].items())),
    ]
    if markdown:
        return "\n".join(["| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in kv])
    w = max(len(k) for k, _ in kv)
    return "\n".join(f"{k.ljust(w)}  {v}" for k, v in kv)


def format_levers(rows, markdown: bool = False, title: str | None = None) -> str:
    base = rows[0][1].tok_s
    hdr = ["lever", "tok/s", "vs base", "hit", "NVMe GB/tok", "bottleneck"]
    body = []
    for label, r in rows:
        flag = ("" if r.feasible else " (INFEASIBLE)") + (" [LOSSY]" if r.lossy and "LOSSY" not in label else "")
        body.append([label + flag, f"{r.tok_s:.3g}", f"{r.tok_s / base:.2f}x" if base else "-", _pct(r.hit_rate),
                     f"{r.bytes_per_token['nvme'] / GB:.2f}", r.bottleneck])
    out = []
    if title:
        out.append(("### " if markdown else "") + title)
    out.append(table(hdr, body, markdown))
    return "\n".join(out)
