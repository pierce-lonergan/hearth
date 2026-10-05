# ADR-0007: Windows with MSVC is a first-class target alongside POSIX

- Status: Accepted
- Date: 2026-10-04
- Supersedes: none
- Superseded-by: none

## Context

Most "ordinary PCs", the machines the mission is about, run Windows. The
reference development machine is Windows 11 with MSVC (docs and
scripts/hxcc.py). Engines that treat Windows as an afterthought tend to depend
on GCC extensions, VLAs, pthreads or a recent OpenMP. On Windows those either
do not exist or need extra runtimes. Direct I/O also works differently on each
OS: `FILE_FLAG_NO_BUFFERING`, `O_DIRECT` and `F_NOCACHE`.

## Decision

* Windows x64 with MSVC is tier 1, together with Linux (GCC and Clang) and
  macOS arm64 (Clang). CI builds and tests all of them.
* Engine code is portable C11 that MSVC accepts:
  * no VLAs, no GNU extensions, `<stdatomic.h>` via `/experimental:c11atomics`;
  * warning-clean at MSVC `/W3` and at GCC/Clang `-Wall -Wextra` (INV-SAFE).
* OS-specific code exists only in `platform_win.c` and `platform_posix.c`;
  compiler-specific code only in `hx_platform.h`.
* Module-level builds use `scripts/hxcc.py` (MSVC through vcvars, or
  `$CC`/clang/gcc). Full builds use CMake, with the Visual Studio generator on
  Windows.
* Build products and model files live in the data directory, never in the
  source tree, which may be cloud-synced (INV-DATA).

## Consequences

* POSIX code cannot be compiled on the Windows development machine. It is
  checked by careful review and by the Linux and macOS CI jobs, so a POSIX-only
  breakage shows up in CI, not locally.
* Some idioms need wrappers, for example aligned allocation, positional
  reads/writes and condition variables (`hx_platform.h`).
* MSVC's AddressSanitizer needs its runtime DLL on PATH. The Linux ASan job is
  the authoritative memory-safety check.
