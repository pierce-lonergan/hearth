# Contributing to Hearth

Hearth is maintained by humans and AI agents under one operating model. Read
**[governance/README.md](governance/README.md)** before your first change; it
covers picking a task, path ownership, tests, handovers and decisions. The
non-negotiables:

* `governance/INVARIANTS.md` is binding. Never modify `tests/golden/`
  (INV-VERIFY).
* Edit only the paths your task owns (`governance/tasks.json`). Report
  contract problems; do not patch contracts.
* New behaviour needs contributor tests that kill mutants
  (`governance/tools/mutate.py`).
* Original code only, under the MIT license (ADR-0006). No third-party code in
  the engine (INV-DEP).
* No model files or build products in the repository or a cloud-synced folder
  (INV-DATA). Use the data directory: `$HEARTH_DATA`, default
  `%LOCALAPPDATA%\hearth` or `~/.cache/hearth`.

## Build and test

Python needs `numpy` and `pytest`. `torch`, `transformers`, `safetensors` and
`tokenizers` are optional; tests that need them skip when they are missing.

### Windows (MSVC)

Quick module builds with `scripts/hxcc.py`. It finds MSVC through vcvars and
writes products to the data directory:

```bat
python scripts/hxcc.py --platform --run -o test_pool.exe engine/src/pool.c engine/tests/test_pool.c
python scripts/hxcc.py --shared -o hearth.dll <sources...>
```

Full build with CMake and the Visual Studio generator. Keep the build
directory outside the source tree if the checkout is in OneDrive or a similar
synced folder:

```bat
cmake -S . -B %LOCALAPPDATA%\hearth\build\cmake -G "Visual Studio 17 2022" -A x64
cmake --build %LOCALAPPDATA%\hearth\build\cmake --config Release
ctest --test-dir %LOCALAPPDATA%\hearth\build\cmake -C Release --output-on-failure
set HEARTH_LIB=%LOCALAPPDATA%\hearth\build\cmake\Release\hearth.dll
python -m pytest tests/py -q
python -E governance/tools/check_golden.py --run
```

`check_golden.py --run` is how golden tests run, locally and in CI: it
verifies their hash lock, runs a temporary copy of only the locked files,
isolated from conftest.py, ini and `__init__.py` files, plugins and bytecode
outside the lock, and fails if any test is skipped.

### Linux / macOS

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
export HEARTH_LIB=$PWD/build/libhearth.so          # .dylib on macOS
PYTHONPATH=python python -m pytest tests/py -q
python -E governance/tools/check_golden.py --run

# AddressSanitizer + UBSan
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DHEARTH_ASAN=ON && cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

`scripts/hxcc.py` also works here; it uses `$CC`, else clang, else gcc.

## Before you open a pull request

```sh
python governance/tools/check_golden.py
python governance/tools/waves.py validate
python governance/tools/validate_handover.py
python governance/tools/adr.py lint
python -m pytest tests/py -q
```

With a built library, also `python -E governance/tools/check_golden.py --run`.
Then run `mutate.py` on the files you changed, and write your handover
manifest (`python governance/tools/validate_handover.py --new <TASK_ID> --write`
creates `governance/handovers/<TASK_ID>-gen-<NNN>.json`). Every performance
number states the hardware, model and settings, and whether it was measured
or simulated (INV-HONEST).
