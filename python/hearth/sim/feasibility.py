"""Can this model run on this machine, and with how much expert cache?"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .cache import engine_min_slots
from .hardware import Hardware
from .timing import ModelCosts, slab_bytes

GIB = float(1 << 30)
WORKSPACE_GIB = 0.5          # activations, KV-independent scratch, thread stacks
VRAM_HEADROOM_GIB = 1.0      # driver/context/activations on the GPU
DISK_BITS = {"Q3 (3.25 bpw, roadmap R03)": 3.25, "Q4 (4.25 bpw)": 4.25, "Q8 (8.25 bpw)": 8.25, "BF16": 16.0}


@dataclass
class Feasibility:
    model: str
    hardware: str
    gpu_mode: str
    fits: bool
    reasons: list = field(default_factory=list)
    backbone_gib: float = 0.0        # resident backbone incl. embeddings
    backbone_location: str = "ram"   # "ram" | "vram" | "unified"
    kv_gib: float = 0.0              # KV cache at max_seq
    ram_free_for_cache_gib: float = 0.0
    min_cache_gib: float = 0.0       # engine minimum slot count
    min_ram_gib: float = 0.0         # RAM needed for the backbone + minimum cache + OS reserve
    recommended_cache_gib: float = 0.0
    expert_total_gib: float = 0.0
    vram_expert_slots: int = 0       # static VRAM tier (gpu_mode dense+experts)
    file_gb: float = 0.0             # .hearth file at the chosen expert bits (decimal GB)
    file_fits_drive: bool = True
    disk_gb_by_format: dict = field(default_factory=dict)   # file size (GB) with all experts at each format
    params_total: int = 0            # Shape.params_total()
    params_resident: int = 0         # Shape.dense_params(): the backbone, embeddings included

    def to_dict(self) -> dict:
        return asdict(self)


def check(costs: ModelCosts, hw: Hardware, *, gpu_mode: str = "off", io_threads: int = 8) -> Feasibility:
    reasons = []
    backbone = costs.resident_bytes / GIB
    kv = costs.kv_resident_bytes / GIB
    embed = costs.embed_bytes / GIB
    loc = "ram"
    ram_used = hw.os_reserve_gib + WORKSPACE_GIB
    vram_left = 0.0
    if gpu_mode != "off":
        if not hw.has_gpu:
            reasons.append(f"gpu_mode={gpu_mode} but {hw.name} has no GPU")
        elif hw.unified:
            loc = "unified"
        else:
            loc = "vram"
            need = backbone - embed + kv + VRAM_HEADROOM_GIB      # embedding table stays in RAM
            vram_left = hw.vram_gib - need
            if vram_left < 0:
                reasons.append(f"backbone {backbone - embed:.1f} GiB + KV {kv:.1f} GiB + {VRAM_HEADROOM_GIB} GiB "
                               f"headroom does not fit {hw.vram_gib:.0f} GiB VRAM")
            ram_used += embed
    if loc != "vram":                                             # RAM, or unified memory (also the RAM)
        ram_used += backbone + kv
    free = hw.ram_gib - ram_used
    min_cache = engine_min_slots(costs.top_k, io_threads) * costs.slab / GIB
    total = costs.expert_total_bytes / GIB
    if free < min_cache:
        hint = (" (a Q4 backbone, --dense-bits 4.25, roughly halves it; LOSSY vs the Q8 default)"
                if costs.dense_bits > 4.25 else "")
        reasons.append(f"only {free:.1f} GiB RAM left for experts after backbone/KV/OS reserve{hint}; "
                       f"engine minimum cache is {min_cache:.2f} GiB")
    vslots = int(max(vram_left, 0.0) * GIB // costs.slab) if (gpu_mode == "dense+experts" and loc == "vram") else 0
    n_slabs = costs.n_moe_layers * costs.n_experts
    disk = {label: (costs.resident_bytes + slab_bytes(costs.expert_d, costs.expert_ffn, bits) * n_slabs) / 1e9
            for label, bits in DISK_BITS.items()}
    file_gb = (costs.resident_bytes + costs.expert_total_bytes) / 1e9
    fits_drive = file_gb <= hw.nvme_capacity_gb
    if not fits_drive:
        reasons.append(f"model file {file_gb:.0f} GB exceeds one drive ({hw.nvme_capacity_gb:.0f} GB); "
                       "each mirror is a full copy")
    return Feasibility(
        model=costs.name, hardware=hw.name, gpu_mode=gpu_mode, fits=not reasons, reasons=reasons,
        backbone_gib=backbone, backbone_location=loc, kv_gib=kv, ram_free_for_cache_gib=free,
        min_cache_gib=min_cache, min_ram_gib=ram_used + min_cache,
        recommended_cache_gib=max(min(free, total), min_cache), expert_total_gib=total, vram_expert_slots=vslots,
        file_gb=file_gb, file_fits_drive=fits_drive, disk_gb_by_format=disk, params_total=costs.total_params,
        params_resident=costs.resident_params)
