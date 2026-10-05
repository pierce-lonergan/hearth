# Expert reads from an external USB hard disk

**Measured** 2026-10-05 with `test_store --bench --gib 1 --dir D:\hearth-data\bench`
(build of commit e0102aa), Ryzen 9 9950X, Windows 11. Drive: WD My Passport 5 TB
(USB, spinning disk, NTFS). Machine shared with other jobs on the NVMe drive (this test
used only the USB disk).

| expert slab | readers 1 | 2 | 4 | 8 | 16 | time per read (1 reader) |
|---|---|---|---|---|---|---|
| 2.39 MiB (Qwen3-30B-A3B Q4), direct | 0.03 GB/s | 0.03 | 0.03 | 0.03 | 0.03 | 75 ms |
| 22.3 MiB (DeepSeek-V3 / Kimi-K2 Q4), direct | 0.04 GB/s | 0.04 | 0.04 | 0.04 | 0.04 | 584 ms |
| buffered and prefetch modes | same within noise | | | | | |

Sequential unbuffered write while building the test file: 0.04 GB/s.

For comparison, the internal Samsung 990 PRO (PCIe 4.0 NVMe) delivers 6.5–7.2 GB/s with
the same benchmark: **~175–200x faster**. Parallel readers do not help a single spindle.

## What it means

* Streaming experts from a USB hard disk is not practical: a Kimi-K2-class model that
  misses even 1 GB of experts per token would spend ~25 s per token on I/O.
* External hard disks are good for **storing** model checkpoints (Hearth's own source
  checkpoint lives on one); copy the container to an NVMe drive before running it.
* The benchmark harness's 60 s no-progress watchdog fires while writing a 4 GiB test
  file at 0.04 GB/s; use `--gib 1` on slow devices (minor test limitation, recorded).
