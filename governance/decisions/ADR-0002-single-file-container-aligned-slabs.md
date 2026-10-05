# ADR-0002: Single-file container with 4096-aligned expert slabs and direct I/O

- Status: Accepted
- Date: 2026-10-04
- Supersedes: none
- Superseded-by: none

## Context

MoE decoding is memory-bound. Routed experts that miss the DRAM cache are read
from NVMe on the critical path, so every miss should be one large sequential
read with no extra copies (docs/ARCHITECTURE.md, "The physics"). Formats built
for dense models scatter one expert's tensors across the file and rely on mmap
and the OS page cache. That doubles the DRAM spent on hot weights and lets the
kernel evict exactly the pages we are about to need. We also want several
byte-identical copies on separate drives (mirrors) and reads that are safe on
hostile input.

## Decision

* A model is one self-describing `.hearth` file (docs/FORMAT.md, normative):
  a 64-byte preamble, a metadata section, a tensor directory and an expert
  directory, then a dense region (64-byte-aligned tensors, read once into one
  arena at open) and an expert region.
* Each routed expert is one contiguous slab (gate, up and down for one expert)
  starting at a multiple of 4096 bytes, with a size that is a multiple of 4096,
  so a miss is a single direct-I/O read: `O_DIRECT`,
  `FILE_FLAG_NO_BUFFERING` or `F_NOCACHE`, falling back to buffered I/O when
  the OS refuses.
* Readers use only the offsets in the preamble and directories, and validate
  every offset, size and alignment. Malformed input produces an error, never a
  crash (INV-FMT, INV-SAFE).
* Any format change bumps `version` and keeps the old reader path working.

## Consequences

* Models must be converted (`hearth.convert`); safetensors and GGUF files are
  not read directly.
* Each slab pays at most 4 KiB of padding. Real expert slabs are megabytes, so
  this is negligible; tiny test models pay proportionally more.
* Mirrors are plain file copies, and the cache tier owns its DRAM instead of
  sharing it with the page cache.
* The reader is a security boundary. It needs fuzz cases
  (`engine/tests/test_modelfile.c`) and the malformed-file golden test.
