# Hearth invariants — the zero-loss constraint lane (𝒦_C)

These are **constraints**, not opinions. In the knowledge-triage model
(governance/README.md) they have binary distortion: a paraphrase that loses a
qualifier is a violation. Agents and humans must carry them **verbatim** across
hand-overs, context compaction and generations. Changing one requires a
maintainer decision recorded in `governance/decisions/` and a golden-test update.

| id | invariant | enforced by |
|----|-----------|-------------|
| **INV-DET-1** | For a fixed binary and ISA, logits are **bit-identical** regardless of compute thread count, I/O thread count, cache budget, eviction policy, prefetch mode, direct vs buffered I/O, mirrors, pinning, warm start and usage profile. | `tests/golden/test_golden_invariants.py::test_scheduling_independence` |
| **INV-DET-2** | Evaluating T tokens in one `hearth_eval` call is **bit-identical** to evaluating them one at a time. (Makes prefill, speculative verification and decode agree.) | `test_batch_equals_sequential` |
| **INV-DET-3** | Scalar, AVX2 and AVX-512 kernels are **bit-identical** (docs/NUMERICS.md canonical order; no FMA contraction in canonical float math). | `test_isa_identity`, `engine/tests/test_quant.c` |
| **INV-NUM-1** | On F32 containers the engine matches `hearth.reference` with max abs logit error ≤ 1e-3·max(1, max\|logit\|); `hearth.reference` matches Hugging Face `transformers` for every supported architecture within 2e-3 relative. | `test_engine_matches_reference`, `tests/py/test_convert_*.py` |
| **INV-NUM-2** | Routing decisions (top-k ids per token per layer) of the engine equal the reference's on F32 containers. | `test_engine_matches_reference` |
| **INV-FMT** | `docs/FORMAT.md` is normative. Readers validate every offset/size/alignment; a malformed or truncated file yields an error, never a crash or out-of-bounds read (INV-SAFE). | `engine/tests/test_modelfile.c` fuzz cases |
| **INV-SAFE** | No out-of-bounds memory access on any input. Engine C code builds warning-clean at /W3 / -Wall -Wextra and runs clean under AddressSanitizer in tests. | CI asan job |
| **INV-DEP** | The engine core is C11 + OS APIs only: no third-party libraries, no C++ runtime, no OpenMP. Python: numpy is the only hard dependency. | review |
| **INV-DATA** | Model files, benchmark containers and build products never go into the repository or a cloud-synced folder; they live in the data dir (`$HEARTH_DATA`, default `%LOCALAPPDATA%/hearth` / `~/.cache/hearth`). | `.gitignore`, review |
| **INV-HONEST** | Every performance claim in docs states hardware, model, settings and whether it was **measured** or **simulated**. Synthetic-weight benchmarks are labelled as such. Speed-ups are reported against a named baseline measured on the same machine. | review |
| **INV-VERIFY** | Golden tests (`tests/golden/`) are authored by maintainers only. Contributors (human or agent) must not modify them; `governance/tools/check_golden.py` verifies `tests/golden/MANIFEST.sha256` in CI. New behaviour is covered by contributor tests, which must pass mutation testing (`governance/tools/mutate.py`). | CI |
| **INV-LOSSLESS-DEFAULT** | Defaults never change model output. Anything that trades accuracy for speed (adaptive top-k, cache-aware routing, lower-bit cold experts) is opt-in, off by default, and reports its measured quality impact. | review, `test_scheduling_independence` |
