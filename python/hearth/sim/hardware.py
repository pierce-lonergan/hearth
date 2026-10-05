"""Hardware profiles for the simulator.

Units: capacities in GiB (2**30 bytes), bandwidths in GB/s (1e9 bytes/s),
latencies in microseconds. Every number is an *assumption* about sustained,
achievable throughput (not the datasheet peak) and can be overridden from the
CLI or via ``get_hardware(name, **overrides)``. The rationale for each built-in
is in ``notes`` and docs/SIMULATOR.md.
"""
from __future__ import annotations

import math
import numbers
import re
from dataclasses import asdict, dataclass, fields, replace


@dataclass(frozen=True)
class Hardware:
    name: str
    ram_gib: float                 # usable system RAM
    dram_gbs: float                # sustained read BW achievable by the CPU kernels
    nvme_count: int = 1            # drives holding the model (mirrors / striping)
    nvme_gbs: float = 6.5          # achievable sequential read per drive
    nvme_latency_us: float = 80.0  # first-byte latency of one large direct read
    nvme_capacity_gb: float = 2000.0   # per drive, decimal GB
    io_cap_gbs: float = 0.0        # platform cap on aggregate storage BW (0 = none)
    cpu_cores: int = 16
    cpu_int8_tops: float = 5.0     # sustained int8 throughput of the quantized matmul kernels (2 ops/MAC)
    os_reserve_gib: float = 4.0    # RAM left to the OS and other programs
    gpu_name: str = ""
    vram_gib: float = 0.0          # 0 = no GPU
    pcie_gbs: float = 0.0          # host<->GPU achievable
    vram_gbs: float = 0.0          # GPU memory BW achievable by a GEMV
    unified: bool = False          # GPU memory is the system RAM (Apple silicon)
    gpu_sync_us: float = 15.0      # one host<->GPU activation hand-off (latency; transfer at pcie_gbs on top)
    notes: str = ""

    def __post_init__(self):
        positive = ("ram_gib", "dram_gbs", "nvme_gbs", "nvme_capacity_gb", "cpu_int8_tops")
        non_negative = ("nvme_latency_us", "io_cap_gbs", "os_reserve_gib", "vram_gib", "pcie_gbs", "vram_gbs",
                        "gpu_sync_us")
        for f in positive:
            v = getattr(self, f)
            if not (isinstance(v, numbers.Real) and not isinstance(v, bool) and 0.0 < v < math.inf):
                raise ValueError(f"hardware {f} must be a finite number > 0, got {v!r}")
        for f in non_negative:
            v = getattr(self, f)
            if not (isinstance(v, numbers.Real) and not isinstance(v, bool) and 0.0 <= v < math.inf):
                raise ValueError(f"hardware {f} must be a finite number >= 0, got {v!r}")
        for f in ("nvme_count", "cpu_cores"):
            v = getattr(self, f)
            if not (isinstance(v, numbers.Integral) and not isinstance(v, bool) and v >= 1):
                raise ValueError(f"hardware {f} must be an integer >= 1, got {v!r}")
        if self.vram_gib > 0 and not (self.vram_gbs > 0 and self.pcie_gbs > 0):
            raise ValueError("a discrete GPU (vram_gib > 0) needs vram_gbs > 0 and pcie_gbs > 0")
        if self.unified and not self.vram_gbs > 0:
            raise ValueError("unified memory needs vram_gbs > 0 (the GPU's memory bandwidth)")

    @property
    def has_gpu(self) -> bool:
        return self.vram_gib > 0 or self.unified

    @property
    def io_gbs(self) -> float:
        """Aggregate storage read bandwidth (perfect striping over mirrors, then platform cap)."""
        bw = self.nvme_count * self.nvme_gbs
        return min(bw, self.io_cap_gbs) if self.io_cap_gbs > 0 else bw

    def to_dict(self) -> dict:
        d = asdict(self)
        d["io_gbs"] = self.io_gbs
        return d


HARDWARE: dict[str, Hardware] = {h.name: h for h in [
    Hardware(
        "this-pc", ram_gib=61.6, dram_gbs=60.0, nvme_count=1, nvme_gbs=6.5, nvme_latency_us=80.0,
        nvme_capacity_gb=2000.0, cpu_cores=16, cpu_int8_tops=5.0, os_reserve_gib=6.0,
        gpu_name="RTX 5070", vram_gib=12.0, pcie_gbs=50.0, vram_gbs=672.0,
        notes="Ryzen 9 9950X (16C Zen 5, AVX-512 VNNI), 64 GB DDR5-6000 (61.6 GiB usable; AIDA64 ~78 GB/s "
              "read, ~60 GB/s assumed for streaming kernels), 1x Samsung 990 PRO 2TB (7.45 GB/s spec, "
              "~6.5 achievable), RTX 5070 12 GB (PCIe 5.0 x16 ~50 GB/s achievable, 672 GB/s VRAM). "
              "int8: ~20 TOPS VNNI peak, ~25% assumed sustained by Q4/Q8 kernels."),
    Hardware(
        "laptop-16gb", ram_gib=16.0, dram_gbs=45.0, nvme_count=1, nvme_gbs=4.5, nvme_latency_us=90.0,
        nvme_capacity_gb=1000.0, cpu_cores=8, cpu_int8_tops=1.5, os_reserve_gib=4.0,
        notes="8-core laptop, 16 GB dual-channel DDR5-5600 (89.6 GB/s peak, ~45 achievable), "
              "1x PCIe 4.0 NVMe 1 TB (~4.5 GB/s achievable, thermal throttling ignored), no dGPU."),
    Hardware(
        "desktop-32gb", ram_gib=32.0, dram_gbs=38.0, nvme_count=1, nvme_gbs=5.0, nvme_latency_us=80.0,
        nvme_capacity_gb=2000.0, cpu_cores=8, cpu_int8_tops=1.0, os_reserve_gib=4.0,
        gpu_name="RTX 3060 12GB", vram_gib=12.0, pcie_gbs=24.0, vram_gbs=360.0,
        notes="Mainstream gaming desktop: 8-core AVX2 CPU (no VNNI), 32 GB DDR4-3200 dual channel "
              "(51.2 GB/s peak, ~38 achievable), 1x PCIe 4.0 NVMe 2 TB (~5 GB/s), RTX 3060 12 GB."),
    Hardware(
        "workstation-128gb-2nvme", ram_gib=128.0, dram_gbs=55.0, nvme_count=2, nvme_gbs=12.0,
        nvme_latency_us=70.0, nvme_capacity_gb=4000.0, cpu_cores=16, cpu_int8_tops=5.0, os_reserve_gib=6.0,
        gpu_name="24GB GPU", vram_gib=24.0, pcie_gbs=25.0, vram_gbs=1000.0,
        notes="AM5 16-core (9950X class), 128 GB = 4x32 GB DDR5 (4 DIMMs run ~5200; 2 channels, ~55 GB/s "
              "achievable), 2x PCIe 5.0 NVMe 4 TB on CPU lanes (~14 GB/s spec, ~12 achievable each), "
              "24 GB GPU at PCIe x8."),
    Hardware(
        "mac-studio-192gb", ram_gib=192.0, dram_gbs=250.0, nvme_count=1, nvme_gbs=6.0, nvme_latency_us=60.0,
        nvme_capacity_gb=2000.0, cpu_cores=24, cpu_int8_tops=2.0, os_reserve_gib=8.0,
        gpu_name="M2 Ultra GPU (unified)", vram_gib=0.0, pcie_gbs=0.0, vram_gbs=800.0, unified=True,
        gpu_sync_us=5.0,
        notes="Mac Studio M2 Ultra, 192 GB unified LPDDR5 (800 GB/s peak; CPU cores alone are assumed to "
              "reach ~250 GB/s - uncertain), internal SSD ~6 GB/s achievable, NEON int8 ~2 TOPS sustained."),
]}


def parse_nvme(spec: str) -> tuple[int, float]:
    """'2x6.5' -> (2, 6.5); '7' -> (1, 7.0)."""
    m = re.fullmatch(r"\s*(?:(\d+)\s*[xX*]\s*)?([0-9]*\.?[0-9]+)\s*", spec)
    if not m:
        raise ValueError(f"bad NVMe spec {spec!r}; expected COUNTxGBPS, e.g. 2x6.5")
    return int(m.group(1) or 1), float(m.group(2))


def get_hardware(name_or_hw="this-pc", **overrides) -> Hardware:
    """A built-in profile (or a Hardware / dict, e.g. Hardware.to_dict() or Result.settings["hw"]) with field
    overrides applied (None values ignored)."""
    valid = {f.name for f in fields(Hardware)}
    if isinstance(name_or_hw, Hardware):
        hw = name_or_hw
    elif isinstance(name_or_hw, dict):
        d = dict(name_or_hw)
        io = d.pop("io_gbs", None)                 # derived (to_dict adds it); must agree with the fields
        bad = set(d) - valid
        if bad:
            raise KeyError(f"unknown hardware field(s): {', '.join(sorted(bad))}")
        hw = Hardware(**d)
        if io is not None and not math.isclose(io, hw.io_gbs, rel_tol=1e-12):
            raise ValueError(f"io_gbs {io} does not match nvme_count x nvme_gbs (capped by io_cap_gbs) = {hw.io_gbs}")
    else:
        key = str(name_or_hw).lower()
        if key not in HARDWARE:
            raise KeyError(f"unknown hardware {name_or_hw!r}; known: {', '.join(HARDWARE)}")
        hw = HARDWARE[key]
    ov = {k: v for k, v in overrides.items() if v is not None}
    bad = set(ov) - valid
    if bad:
        raise KeyError(f"unknown hardware field(s): {', '.join(sorted(bad))}")
    if ov:
        hw = replace(hw, **ov)
        if "name" not in ov:
            hw = replace(hw, name=hw.name + "*")   # mark as modified
    return hw
