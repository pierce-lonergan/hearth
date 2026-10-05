# First GitHub CI runs (2026-10-05)

The workflows written in round 1 had never run on GitHub. Results on `main`:

| run | commit | result |
|---|---|---|
| 37305848312 | 1e751ca (first public push) | 5 of 9 jobs failed |
| 37308785094 | 02fe0ef (director fixes) | all green except **macos arm64** |

## Fixed in 02fe0ef (director)

| finding | root cause | fix |
|---|---|---|
| Python ABI test failed on every OS | hearth.h amendment 8ea46c0 added `read_errors` to `hearth_stats` but the ctypes mirror was not updated, so `hearth_get_stats` wrote 8 bytes past the Python buffer. The director did not re-run the suite after amending the header. | mirror updated; procedure: re-run the full suite after any contract edit |
| UBSan: `quant_avx2.c:137` left shift of 32768 by 16 in `int` | `uint16_t` promoted to `int`, shifted into the sign bit (BF16 load) | unsigned bits + `memcpy` |
| gcc `-Werror=format-truncation` in test_modelfile.c / test_store.c | unchecked `snprintf` results | check return values |
| clang `-Wuninitialized-const-pointer` in test_modelfile.c | uninitialised struct passed as `const void *` | zero-initialise |
| jinja2-dependent tests failed where jinja2 is absent | optional dependency treated as required (INV-DEP) | `pytest.importorskip("jinja2")` |

## Open: macos arm64 (first time this code was ever compiled for Darwin/arm64)

1. `engine/tests/test_model.c:1157,1405` — engine output differs from the test's naive
   NUMERICS reference by ~1e-5 for GQA and MLA (RoPE paths) only; dense/MoE math is
   bit-exact. Leading hypothesis (unverified, no macOS machine here): Apple clang fuses
   `sinf(θ)` + `cosf(θ)` of the same argument into `__sincosf_stret`, whose results can
   differ by an ulp from separate calls, so code shape changes results. NUMERICS §4
   requires libm calls "the same way from every code path". Candidate fixes: route both
   engine and reference through one non-inlinable `hx_sincos` helper, or compile with an
   option that prevents the fusion; verify on the macOS runner.
2. `engine/tests/test_platform.c:858` — "earlier returned pointer still valid": the test
   assumes a pointer returned by `hx_env_str` stays valid after the variable is set again.
   On Darwin `getenv` storage is not stable across `setenv`. Either document the lifetime
   in hx_platform.h or make `platform_posix.c` return stable copies (bounded cache).
3. `test_pool` takes 720–1000 s on 2–3 core CI runners (spin-calibration and 200k-dispatch
   stress); add a CI-sized mode.
