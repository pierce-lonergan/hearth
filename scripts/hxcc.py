#!/usr/bin/env python3
"""hxcc — compile a handful of Hearth C files without CMake.

Fast module-level builds for contributors (human or agent):

    python scripts/hxcc.py -o test_quant.exe engine/src/quant.c engine/src/quant_avx2.c \
        engine/src/quant_avx512.c engine/src/platform_win.c engine/tests/test_quant.c
    python scripts/hxcc.py --shared -o hearth.dll <sources...>
    python scripts/hxcc.py --run -o t.exe <sources...> -- arg1 arg2   # build then run

* Windows: MSVC via vcvars64 (environment cached in the hearth data dir).
* Elsewhere: $CC, else clang, else gcc.
* Files named *_avx2.c / *_avx512.c automatically get the matching ISA flags.
* `--platform` adds the right platform_*.c for this OS.
* Relative -o paths go to the build dir (default: <data dir>/build/hxcc), keeping
  build products out of the (possibly cloud-synced) source tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IS_WIN = os.name == "nt"


def data_dir() -> Path:
    if os.environ.get("HEARTH_DATA"):
        return Path(os.environ["HEARTH_DATA"])
    if IS_WIN:
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "hearth"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "hearth"


def msvc_env() -> dict:
    cache = data_dir() / "msvc_env.json"
    if cache.exists():
        try:
            env = json.loads(cache.read_text())
            if shutil.which("cl", path=env.get("PATH", "")):
                return env
        except Exception:
            pass
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        sys.exit("hxcc: Visual Studio not found (no vswhere.exe). Install VS Build Tools with the C++ workload.")
    inst = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                           "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    if not inst:
        sys.exit("hxcc: no Visual Studio installation with MSVC x64 tools found.")
    vcvars = Path(inst[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    env = dict(os.environ)
    env["PATH"] = str(vswhere.parent) + os.pathsep + env.get("PATH", "")
    out = subprocess.run(f'"{vcvars}" >nul 2>&1 && set', shell=True, capture_output=True, text=True, env=env).stdout
    res = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            res[k] = v
    if not shutil.which("cl", path=res.get("PATH", "")):
        sys.exit("hxcc: vcvars64.bat did not put cl.exe on PATH.")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(res))
    return res


def isa_flags(src: Path, msvc: bool) -> list[str]:
    name = src.name
    if name.endswith("_avx512.c"):
        return ["/arch:AVX512"] if msvc else ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512vnni", "-mavx2", "-mfma", "-mf16c"]
    if name.endswith("_avx2.c"):
        return ["/arch:AVX2"] if msvc else ["-mavx2", "-mfma", "-mf16c"]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--shared", action="store_true", help="build a shared library (defines HEARTH_BUILD_DLL)")
    ap.add_argument("--debug", action="store_true", help="no optimisation, debug info")
    ap.add_argument("--asan", action="store_true", help="AddressSanitizer")
    ap.add_argument("--platform", action="store_true", help="add platform_win.c / platform_posix.c")
    ap.add_argument("--run", action="store_true", help="run the executable after building")
    ap.add_argument("-D", action="append", default=[], help="extra define")
    ap.add_argument("sources", nargs="+")
    argv = sys.argv[1:]
    run_args: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, run_args = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)

    srcs = [Path(s).resolve() for s in a.sources]
    if a.platform:
        srcs.append(ROOT / "engine" / "src" / ("platform_win.c" if IS_WIN else "platform_posix.c"))
    for s in srcs:
        if not s.exists():
            sys.exit(f"hxcc: no such source {s}")

    out = Path(a.out)
    if not out.is_absolute():
        out = data_dir() / "build" / "hxcc" / out
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Per-process object dir: concurrent builds of the same output name must not collide.
    tag = hashlib.sha1(f"{out}|{os.getpid()}".encode()).hexdigest()[:10]
    objdir = out.parent / f".obj-{out.stem}-{tag}"
    objdir.mkdir(parents=True, exist_ok=True)

    inc = [ROOT / "engine" / "include", ROOT / "engine" / "src"]
    defines = list(a.D) + (["HEARTH_BUILD_DLL"] if a.shared else [])

    if IS_WIN and not os.environ.get("CC"):
        env = msvc_env()
        cl_exe = shutil.which("cl", path=env["PATH"])
        link_exe = shutil.which("link", path=env["PATH"])
        common = ["/nologo", "/std:c17", "/experimental:c11atomics", "/fp:precise", "/utf-8", "/W3",
                  "/D_CRT_SECURE_NO_WARNINGS", "/c"]
        common += ["/Od", "/Zi"] if a.debug else ["/O2", "/Oi"]
        if a.asan:
            common.append("/fsanitize=address")
        common += [f"/I{p}" for p in inc] + [f"/D{d}" for d in defines]
        objs = []
        for s in srcs:
            obj = objdir / (s.stem + ".obj")
            cmd = [cl_exe] + common + isa_flags(s, True) + [f"/Fo{obj}", f"/Fd{objdir / 'vc.pdb'}", str(s)]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            msg = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if l.strip() and l.strip() != s.name)
            if msg:
                print(msg)
            if r.returncode:
                print(f"hxcc: compile failed: {s}", file=sys.stderr)
                return r.returncode
            objs.append(str(obj))
        link = [link_exe, "/nologo", f"/OUT:{out}"] + (["/DLL"] if a.shared else []) + (["/DEBUG"] if a.debug or a.asan else []) + objs
        r = subprocess.run(link, env=env, capture_output=True, text=True)
        if r.stdout.strip():
            print(r.stdout.strip())
        if r.returncode:
            print(r.stderr, file=sys.stderr)
            print("hxcc: link failed", file=sys.stderr)
            return r.returncode
    else:
        env = dict(os.environ)
        cc = os.environ.get("CC") or shutil.which("clang") or shutil.which("gcc") or "cc"
        common = ["-std=c11", "-ffp-contract=off", "-Wall", "-Wextra", "-Wno-unused-parameter", "-pthread"]
        common += ["-O0", "-g"] if a.debug else ["-O2", "-g"]
        if a.asan:
            common += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        if a.shared:
            common += ["-fPIC", "-fvisibility=hidden"]
        common += [f"-I{p}" for p in inc] + [f"-D{d}" for d in defines]
        objs = []
        for s in srcs:
            obj = objdir / (s.stem + ".o")
            r = subprocess.run([cc] + common + isa_flags(s, False) + ["-c", str(s), "-o", str(obj)])
            if r.returncode:
                return r.returncode
            objs.append(str(obj))
        link = [cc] + (["-shared"] if a.shared else []) + objs + ["-o", str(out), "-lm", "-pthread"]
        if a.asan:
            link.append("-fsanitize=address,undefined")
        r = subprocess.run(link)
        if r.returncode:
            return r.returncode

    shutil.rmtree(objdir, ignore_errors=True)
    print(f"hxcc: built {out}", flush=True)
    if a.run:
        r = subprocess.run([str(out)] + run_args)
        return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
