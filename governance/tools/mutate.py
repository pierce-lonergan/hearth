#!/usr/bin/env python3
"""mutate — mutation testing for Hearth's C and Python code (standard library only).

A contributor test is trusted only if it notices when the code it covers is
broken. mutate.py makes small deliberate breaks ("mutants") in one source file,
runs a test command against each one and reports which mutants survived.

    python governance/tools/mutate.py --file engine/src/pool.c \\
        --test "python scripts/hxcc.py --platform --run -o {tmp}/mut_pool.exe engine/src/pool.c engine/tests/test_pool.c" \\
        --max-mutants 40 --jobs 4 [--lines 10-200] [--since origin/main] [--seed 1]
    python governance/tools/mutate.py --file python/hearth/quant.py \\
        --test "{python} -m pytest -q -x tests/py/test_format.py" --jobs 4
    python governance/tools/mutate.py --file engine/src/pool.c --list      # show the sampled mutants only

Safety: the working tree is never modified. Each job copies the repository
(without .git, build directories and model/binary files) to a temporary
directory, mutates the copy and runs the command there (cwd = the copy) with a
timeout; the copies are deleted afterwards. Copies have no .git, so tests
that need git cannot exercise their code inside a job.

Placeholders in --test / --build:
  {tmp}     a private scratch directory of this job (absolute, quoted), deleted
            at the end. Put build outputs there, as an argument: -o {tmp}/t.exe
            (hxcc then also keeps its object files there).
  {job}     a suffix unique to this job *and* this run: <run>-<index>. For
            outputs in shared places, e.g. hxcc -o mut_pool{job}.exe, which
            writes to <data dir>/build/hxcc; files there whose names carry the
            run id are deleted at the end.
  {run}     the id of this invocation: "mr" + 8 hex digits
  {python}  this Python interpreter (quoted)
  {root}    the job's private repository copy (quoted)
A fixed output name collides with the other jobs and with every other mutate.py
run on the machine (agents run them concurrently), so mutate.py warns when -o
is given without one of these placeholders. The command also sees
HEARTH_MUTATION_JOB (index), HEARTH_MUTATION_RUN, HEARTH_MUTATION_ROOT,
HEARTH_MUTATION_TMP and PYTHONDONTWRITEBYTECODE=1; PYTHONPATH entries inside
the repository are redirected into the copy.

Operators: relational (< <= > >= == !=), arithmetic (+ - * / % and compound
assignments), logical (&& <-> ||, and <-> or), negation (remove unary ! / not /
-), constant (integer literal +-1, 0 <-> 1), condition (if/while/for condition
negated), return (returned value replaced), call-delete (single-line call
statement removed). Comments, strings and preprocessor lines are never mutated;
neither is a line containing the word "nomutate".

C conditional compilation: code in an #if/#ifdef branch that is inactive on
this machine is not mutated, because such mutants can never be killed. #if 0
is always inactive. Other conditions are evaluated with three-valued logic
(true / false / unknown) from the platform's predefined macros (--pp-model:
auto = msvc on Windows unless $CC is set, else gnu; os = OS macros only; none),
-D/-U overrides, and the #define/#undef lines of the file and of the quoted
headers it includes. A branch that cannot be decided is mutated, and its
mutants carry the condition ("cond") in the report. C files are .c, .h and .inc;
an .inc fragment is read in the context of the files that #include it (their
earlier #defines, headers and typedefs; macros they disagree on are unknown).

Sampling is deterministic for a given --seed: candidates are grouped by
operator, each group is shuffled with the seed, and groups are drawn round-robin
so every operator class is represented.

Outcomes: killed (command failed), survived (command passed), timeout (counted
as detected), build-error (mutant did not compile; excluded from the score).
    score = (killed + timeout) / (killed + timeout + survived)
For C targets without --build, a failing command whose output matches
--build-error-regex (default: compiler/linker errors, hxcc failures) is a build
error. Python mutants that do not compile are discarded before sampling, so a
failing Python test run always counts as killed.

Timeouts: the unmutated baseline runs under the same --timeout as the mutants
(600 s when none is given) and must finish within it. A mutant's timeout is
never below max(2 x baseline, baseline + 5 s): a smaller --timeout is raised to
that floor with a warning, since a pass that merely runs long would otherwise
count as a kill. The default is max(10 s, 3 x baseline + 5 s). A mutant that
times out in the parallel phase is re-run once on its own afterwards
(--no-timeout-retry disables this), so load alone does not turn a slow pass
into a kill; the report marks retried mutants. If every scored mutant timed
out, the report warns that the score proves nothing and --min-score fails. A
timeout of the --build command is a build error, so for C a separate --build
keeps slow compiles out of the score.

Ctrl-C stops the run: queued mutants are cancelled, running commands are
killed and the job directories are removed.

Reports (JSON + Markdown with a diff for every survivor) go to --out-dir,
default <data dir>/mutation/<file>-s<seed>-<time>-<run>. Exit: 0 done, 1 score
below --min-score, 2 bad arguments or the unmutated code fails the test,
130 interrupted.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import keyword
import os
import platform
import queue
import random
import re
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tokenize
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = 2
OPERATORS = ("relational", "arithmetic", "logical", "negation", "constant", "condition", "return", "call-delete")
OUTCOMES = ("killed", "survived", "timeout", "build-error")
DEFAULT_BUILD_ERROR_RE = (r"hxcc: (compile|link) failed|\berror C\d{4}\b|\berror LNK\d{4}\b|"
                          r"^[^\n:]+\.(c|h|cc|cpp):\d+:(\d+:)? (fatal )?error:|^(SyntaxError|IndentationError): ")


class MutateError(Exception):
    pass


@dataclass
class Mutant:
    op: str
    line: int
    col: int
    edits: tuple          # ((start, end, replacement), ...) offsets into the source text
    before: str
    after: str
    idx: int = -1
    cond: str = ""        # C: undecidable #if condition the mutated code sits under

    def apply(self, src: str) -> str:
        out = src
        for start, end, text in sorted(self.edits, key=lambda e: e[0], reverse=True):
            out = out[:start] + text + out[end:]
        return out

    def key(self, rel: str) -> str:
        h = hashlib.sha1(f"{rel}:{self.op}:{self.line}:{self.col}:{self.before}->{self.after}".encode())
        return h.hexdigest()[:12]

    @property
    def name(self) -> str:
        return f"M{self.idx:04d}"


def _short(s: str, n: int = 70) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


REL_SWAP = {"<": ("<=", ">"), "<=": ("<", ">"), ">": (">=", "<"), ">=": (">", "<"), "==": ("!=",), "!=": ("==",)}
ARITH_SWAP = {"+": ("-",), "-": ("+",), "*": ("/",), "/": ("*",), "%": ("*",),
              "+=": ("-=",), "-=": ("+=",), "*=": ("/=",), "/=": ("*=",)}


def _int_variants(value: int) -> list:
    if value == 0:
        return [1]
    if value == 1:
        return [0, 2]
    return [value + 1, value - 1]


# ===================================================================== C
@dataclass
class Tok:
    kind: str     # id | num | str | op
    text: str
    start: int
    end: int
    line: int
    col: int


_C_OPS3 = {"<<=", ">>=", "..."}
_C_OPS2 = {"->", "++", "--", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "+=", "-=", "*=", "/=",
           "%=", "&=", "|=", "^=", "##"}
_C_IDENT = re.compile(r"[A-Za-z_]\w*")
_C_PPNUM = re.compile(r"\.?\d(?:[eEpP][+-]|[\w.'])*")
_C_INT = re.compile(r"^(0[xX][0-9a-fA-F]+|[1-9][0-9]*|0)([uU](?:ll|LL|l|L)?|(?:ll|LL|l|L)[uU]?)?$")
C_KEYWORDS = set("""auto break case char const continue default do double else enum extern float for goto
if inline int long register restrict return short signed sizeof static struct switch typedef union
unsigned void volatile while _Alignas _Alignof _Atomic _Bool _Complex _Generic _Imaginary _Noreturn
_Static_assert _Thread_local static_assert alignof""".split())
C_TYPE_WORDS = set("""void char short int long float double signed unsigned _Bool bool const volatile
restrict _Atomic struct union enum static extern register inline FILE memory_order""".split())
C_ATOMIC_TYPES = set("""atomic_bool atomic_char atomic_schar atomic_uchar atomic_short atomic_ushort atomic_int
atomic_uint atomic_long atomic_ulong atomic_llong atomic_ullong atomic_flag""".split())   # <stdatomic.h>; *_t names end in _t


def lex_c(src: str, directives: list = None) -> list:
    """Tokens of C source, skipping whitespace, comments and preprocessor directives.

    If `directives` is a list, (start, end, raw_text) of every directive is appended.
    """
    toks = []
    i, n = 0, len(src)
    line, bol = 1, 0
    code_on_line = False

    def advance(j):
        nonlocal line, bol
        k = src.count("\n", i, j)
        if k:
            line += k
            bol = src.rfind("\n", i, j) + 1

    def spliced(k):  # is the newline at k preceded by a backslash (line splice)?
        b = k - 1
        if b >= 0 and src[b] == "\r":
            b -= 1
        return b >= 0 and src[b] == "\\"

    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            bol = i
            code_on_line = False
            continue
        if c in " \t\r\f\v" or (c == "\\" and src[i + 1:i + 2] in ("\n", "\r")):
            i += 1
            continue
        if src.startswith("//", i):
            j = i
            while True:
                k = src.find("\n", j)
                if k < 0:
                    k = n
                    break
                if spliced(k):
                    j = k + 1
                    continue
                break
            advance(k)
            i = k
            continue
        if src.startswith("/*", i):
            k = src.find("*/", i + 2)
            k = n if k < 0 else k + 2
            advance(k)
            i = k
            continue
        if c == "#" and not code_on_line:
            j = i + 1
            while j < n:
                ch = src[j]
                if ch == "\n":
                    if spliced(j):
                        j += 1
                        continue
                    break
                if src.startswith("/*", j):
                    k = src.find("*/", j + 2)
                    j = n if k < 0 else k + 2
                    continue
                if src.startswith("//", j):
                    k = src.find("\n", j)
                    j = n if k < 0 else k
                    continue
                if ch == '"':
                    j += 1
                    while j < n and src[j] not in '"\n':
                        j += 2 if src[j] == "\\" else 1
                    j += 1
                    continue
                j += 1
            j = min(j, n)
            if directives is not None:
                directives.append((i, j, src[i:j]))
            advance(j)
            i = j
            continue
        code_on_line = True
        start, col, ln = i, i - bol + 1, line
        m = _C_IDENT.match(src, i)
        if m and not (m.group() in ("L", "u", "U", "u8") and src[m.end():m.end() + 1] in ("'", '"')):
            toks.append(Tok("id", m.group(), start, m.end(), ln, col))
            i = m.end()
            continue
        if m:  # string/char literal with an encoding prefix
            i = m.end()
            c = src[i]
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            toks.append(Tok("str", src[start:j], start, j, ln, col))
            advance(j)
            i = j
            continue
        if c.isdigit() or (c == "." and src[i + 1:i + 2].isdigit()):
            m = _C_PPNUM.match(src, i)
            toks.append(Tok("num", m.group(), start, m.end(), ln, col))
            i = m.end()
            continue
        for width, table in ((3, _C_OPS3), (2, _C_OPS2)):
            if src[i:i + width] in table:
                op = src[i:i + width]
                break
        else:
            op = c
        toks.append(Tok("op", op, start, start + len(op), ln, col))
        i += len(op)
    return toks


def c_typedef_names(toks: list) -> set:
    names = set()
    i = 0
    while i < len(toks):
        if toks[i].kind == "id" and toks[i].text == "typedef":
            depth, j, fnptr, last = 0, i + 1, None, None
            while j < len(toks):
                t = toks[j]
                if t.text in ("{", "(", "["):
                    if t.text == "(" and depth == 0 and j + 2 < len(toks) and toks[j + 1].text == "*" \
                            and toks[j + 2].kind == "id" and fnptr is None:
                        fnptr = toks[j + 2].text
                    depth += 1
                elif t.text in ("}", ")", "]"):
                    depth -= 1
                elif t.text == ";" and depth <= 0:
                    break
                elif t.kind == "id" and depth == 0:
                    last = t.text
                j += 1
            name = fnptr or last
            if name:
                names.add(name)
            i = j
        i += 1
    return names


def _local_headers(src: str, path: Path, root: Path, max_files: int = 64) -> list:
    """Quoted #include files reachable from src (transitively), for typedef names."""
    out, seen = [], set()
    todo = [(src, path)]
    while todo and len(out) < max_files:
        text, where = todo.pop()
        for inc in re.findall(r'^\s*#\s*include\s*"([^"]+)"', text, flags=re.M):
            for base in (where.parent, root / "engine" / "src", root / "engine" / "include"):
                p = (base / inc).resolve()
                if p.is_file():
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
                        try:
                            todo.append((p.read_text(encoding="utf-8", errors="replace"), p))
                        except OSError:
                            pass
                    break
    return out


# ========================================================== preprocessor
class _Mark:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


# Macro model values: an int (defined with that value), DEFINED (defined, value not
# an integer) or UNDEF (known not to be defined). A name missing from the model is
# unknown, because system headers and the compiler may define it.
DEFINED, UNDEF = _Mark("DEFINED"), _Mark("UNDEF")
PP_MODELS = ("auto", "msvc", "gnu", "os", "none")
_OS_MACROS = {
    "win32": ("_WIN32",),
    "linux": ("__linux__", "__gnu_linux__", "__unix__"),
    "darwin": ("__APPLE__", "__MACH__"),
}
_ANY_OS = ("_WIN32", "_WIN64", "__CYGWIN__", "__linux__", "__gnu_linux__", "__APPLE__", "__MACH__",
           "__FreeBSD__", "__NetBSD__", "__OpenBSD__", "__DragonFly__", "__ANDROID__")
_ANY_ARCH = ("_M_X64", "_M_AMD64", "_M_IX86", "_M_ARM64", "_M_ARM", "__x86_64__", "__amd64__", "__i386__",
             "__aarch64__", "__arm64__", "__arm__")
_ARCH_MACROS = {   # (msvc, gcc/clang)
    "x86_64": (("_M_X64", "_M_AMD64", "_WIN64"), ("__x86_64__", "__amd64__")),
    "arm64": (("_M_ARM64", "_WIN64"), ("__aarch64__",)),
}


def resolve_pp_model(model: str) -> str:
    if model != "auto":
        return model
    if sys.platform == "win32":
        return "os" if os.environ.get("CC") else "msvc"   # hxcc and CMake's VS generator use MSVC
    return "gnu"


def host_macros(model: str) -> dict:
    """Predefined macros of this machine's compiler, as far as #if evaluation needs them."""
    model = resolve_pp_model(model)
    plat = "win32" if sys.platform == "win32" else "darwin" if sys.platform == "darwin" else \
        "linux" if sys.platform.startswith("linux") else None
    defs: dict = {}
    if model == "none" or plat is None:
        return defs
    for m in _ANY_OS:
        defs[m] = UNDEF
    if plat == "win32":
        del defs["_WIN64"]
    for m in _OS_MACROS[plat]:
        defs[m] = 1
    machine = platform.machine().lower()
    arch = "x86_64" if machine in ("amd64", "x86_64", "x64") else "arm64" if machine in ("arm64", "aarch64") else None
    if model not in ("msvc", "gnu") or arch is None:
        return defs
    for m in _ANY_ARCH:
        defs[m] = UNDEF
    msvc_names, gnu_names = _ARCH_MACROS[arch]
    for m in msvc_names if model == "msvc" else gnu_names:
        if m != "_WIN64" or plat == "win32":
            defs[m] = 1
    if arch == "arm64" and model == "gnu" and plat == "darwin":
        defs["__arm64__"] = 1
    if model == "msvc":
        defs.update({"_MSC_VER": DEFINED, "__GNUC__": UNDEF, "__clang__": UNDEF})
    else:
        defs.update({"__GNUC__": DEFINED, "_MSC_VER": UNDEF})
    return defs


def parse_macro_args(defines, undefines) -> dict:
    out = {}
    for d in defines or []:
        name, _, value = d.partition("=")
        if not _C_IDENT.fullmatch(name):
            raise MutateError(f"-D {d!r}: not a macro name")
        v = pp_eval(value, {}) if value else 1
        out[name] = v if isinstance(v, int) else DEFINED
    for u in undefines or []:
        if not _C_IDENT.fullmatch(u):
            raise MutateError(f"-U {u!r}: not a macro name")
        out[u] = UNDEF
    return out


_PP_TOKEN = re.compile(r"(0[xX][0-9a-fA-F]+|[0-9]+)[uUlL]*|([A-Za-z_]\w*)|(&&|\|\||==|!=|<=|>=|<<|>>|[-+*/%()!~<>&|^?:,])")
_PP_LEVELS = (("||",), ("&&",), ("|",), ("^",), ("&",), ("==", "!="), ("<", ">", "<=", ">="), ("<<", ">>"),
              ("+", "-"), ("*", "/", "%"))


def _pp_binop(op, a, b):
    if op == "&&":
        return 0 if a == 0 or b == 0 else None if a is None or b is None else 1
    if op == "||":
        return 1 if a not in (0, None) or b not in (0, None) else None if a is None or b is None else 0
    if a is None or b is None:
        return None
    if op in ("/", "%"):
        if b == 0:
            return None
        q = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)   # C truncates toward zero
        return q if op == "/" else a - q * b
    if op in ("<<", ">>"):
        return None if not 0 <= b < 64 else (a << b if op == "<<" else a >> b)
    return {"|": a | b, "^": a ^ b, "&": a & b, "==": int(a == b), "!=": int(a != b), "<": int(a < b),
            ">": int(a > b), "<=": int(a <= b), ">=": int(a >= b), "+": a + b, "-": a - b, "*": a * b}[op]


def pp_eval(expr: str, defs: dict):
    """Evaluates an #if expression: an int, or None if it depends on unknown macros."""
    toks, pos, expr = [], 0, expr.strip()
    while pos < len(expr):
        if expr[pos].isspace():
            pos += 1
            continue
        m = _PP_TOKEN.match(expr, pos)
        if not m:
            return None
        pos = m.end()
        if m.group(1):
            t = m.group(1)
            try:
                toks.append(("num", int(t, 16) if t[:2].lower() == "0x" else int(t, 8) if len(t) > 1 and
                             t[0] == "0" else int(t)))
            except ValueError:
                return None
        elif m.group(2):
            toks.append(("id", m.group(2)))
        else:
            toks.append(("op", m.group(3)))
    i = 0

    def peek():
        return toks[i] if i < len(toks) else (None, None)

    def take():
        nonlocal i
        t = peek()
        i += 1
        return t

    def binary(level):
        if level == len(_PP_LEVELS):
            return unary()
        v = binary(level + 1)
        while peek()[0] == "op" and peek()[1] in _PP_LEVELS[level]:
            op = take()[1]
            v = _pp_binop(op, v, binary(level + 1))
        return v

    def unary():
        kind, x = peek()
        if kind == "op" and x in ("!", "~", "-", "+"):
            take()
            v = unary()
            return None if v is None else {"!": int(not v), "~": ~v, "-": -v, "+": v}[x]
        return primary()

    def primary():
        kind, x = take()
        if kind == "num":
            return x
        if (kind, x) == ("op", "("):
            v = binary(0)
            if take() != ("op", ")"):
                raise ValueError
            return v
        if kind != "id":
            raise ValueError
        if x == "defined":
            paren = peek() == ("op", "(")
            if paren:
                take()
            k, name = take()
            if k != "id" or (paren and take() != ("op", ")")):
                raise ValueError
            d = defs.get(name)
            return None if d is None else int(d is not UNDEF)
        if peek() == ("op", "("):   # function-like macro, __has_include(...)
            raise ValueError
        d = defs.get(x)
        return d if isinstance(d, int) else 0 if d is UNDEF else None

    try:
        v = binary(0)
        return v if i == len(toks) else None
    except (ValueError, IndexError, RecursionError):
        return None


def _not3(a):
    return None if a is None else int(not a)


def _and3(a, b):
    return 0 if a == 0 or b == 0 else None if a is None or b is None else 1


def _truth(v):
    return None if v is None else int(bool(v))


_DIRECTIVE = re.compile(r"#\s*([A-Za-z_]\w*)?(.*)", re.S)


def _directive_text(raw: str) -> str:
    """A directive with line splices joined and comments removed."""
    raw = re.sub(r"\\\r?\n", "", raw)
    out, i, n = [], 0, len(raw)
    while i < n:
        c = raw[i]
        if raw.startswith("/*", i):
            k = raw.find("*/", i + 2)
            i = n if k < 0 else k + 2
            out.append(" ")
            continue
        if raw.startswith("//", i):
            break
        if c in "\"'":
            j = i + 1
            while j < n and raw[j] != c:
                j += 2 if raw[j] == "\\" else 1
            out.append(raw[i:j + 1])
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out).strip()


class Preprocessor:
    """Conditional-compilation regions of C files under a three-valued macro model.

    resolve(name, from_path) -> (text, path) or None finds a quoted #include;
    each header is processed once, for its #define/#undef lines.
    """

    def __init__(self, defs: dict = None, resolve=None):
        self.defs = dict(defs or {})
        self.resolve = resolve
        self.seen: set = set()

    def regions(self, src: str, path=None, directives: list = None, base=1) -> list:
        """[(offset, state, cond)] from offset on: state 1 active, 0 inactive, None unknown."""
        if directives is None:
            directives = []
            lex_c(src, directives)
        parsed = []
        for start, end, raw in directives:
            m = _DIRECTIVE.match(_directive_text(raw))
            parsed.append((end, (m.group(1) or "") if m else "", m.group(2).strip() if m else ""))
        stack = []          # [state, taken, description] per open #if
        out = [(0, base, "")]
        for n, (end, name, rest) in enumerate(parsed):
            cur = self._state(stack, base)
            if name in ("if", "ifdef", "ifndef"):
                if name == "if":
                    v, desc = _truth(pp_eval(rest, self.defs)), f"#if {rest}"
                else:
                    macro = (rest.split() or [""])[0]
                    v = self._defined(macro, name == "ifndef")
                    guard = name == "ifndef" and n + 1 < len(parsed) and parsed[n + 1][1] == "define" and \
                        (parsed[n + 1][2].split() or [""])[0] == macro
                    if v is None and guard:
                        v = 1   # include guard
                    desc = f"#{name} {macro}"
                stack.append([v, v, desc])
            elif name in ("elif", "elifdef", "elifndef") and stack:
                f = stack[-1]
                if name == "elif":
                    v = _truth(pp_eval(rest, self.defs))
                else:   # C23
                    v = self._defined((rest.split() or [""])[0], name == "elifndef")
                f[0] = _and3(_not3(f[1]), v)
                f[1] = 1 if f[1] == 1 or v == 1 else None if None in (f[1], v) else 0
                f[2] = f"#{name} {rest}"
            elif name == "else" and stack:
                f = stack[-1]
                f[0], f[1] = _not3(f[1]), 1
                f[2] = f"#else of {f[2]}" if not f[2].startswith("#else") else f[2]
            elif name == "endif" and stack:
                stack.pop()
            elif cur != 0 and name in ("define", "undef"):
                self._define(name, rest, cur)
            elif cur != 0 and name == "include" and self.resolve is not None:
                inc = re.match(r'"([^"]+)"', rest)
                if inc:
                    self._include(inc.group(1), path, cur)
            state = self._state(stack, base)
            out.append((end, state, " && ".join(f[2] for f in stack if f[0] is None) if state is None else ""))
        return out

    def _defined(self, macro: str, negate: bool):
        d = self.defs.get(macro)
        v = None if d is None else int(d is not UNDEF)
        return _not3(v) if negate else v

    @staticmethod
    def _state(stack, base):
        v = base
        for f in stack:
            v = _and3(v, f[0])
        return v

    def _define(self, name, rest, cur):
        m = re.match(r"([A-Za-z_]\w*)(\()?(.*)", rest, re.S)
        if not m:
            return
        macro = m.group(1)
        if cur != 1:
            self.defs.pop(macro, None)
        elif name == "undef":
            self.defs[macro] = UNDEF
        elif m.group(2):
            self.defs[macro] = DEFINED
        else:
            v = pp_eval(m.group(3), {}) if m.group(3).strip() else None
            self.defs[macro] = v if isinstance(v, int) else DEFINED

    def _include(self, name, path, cur):
        found = self.resolve(name, path)
        if found is None:
            return
        text, hpath = found
        if hpath in self.seen:
            return
        self.seen.add(hpath)
        self.regions(text, hpath, base=cur)


def header_resolver(root: Path):
    """Finds quoted #include files next to the includer or in engine/src, engine/include."""
    def resolve(name, from_path):
        bases = ([Path(from_path).parent] if from_path else []) + [root / "engine" / "src", root / "engine" / "include"]
        for base in bases:
            p = (base / name).resolve()
            if p.is_file():
                try:
                    return p.read_text(encoding="utf-8", errors="replace"), p
                except OSError:
                    return None
        return None
    return resolve


def gen_c(src: str, types: set = frozenset(), defs: dict = None, resolve=None, path=None, stats: dict = None) -> list:
    """C mutants. defs: macro model for #if (None: nothing known, only literal
    conditions such as #if 0 are decided); resolve: header lookup (see Preprocessor)."""
    dirs: list = []
    toks = lex_c(src, dirs)
    types = set(types) | c_typedef_names(toks)
    n = len(toks)
    muts = []
    points = Preprocessor(defs, resolve).regions(src, path, dirs)
    region, r = [], 0
    for t in toks:
        while r + 1 < len(points) and points[r + 1][0] <= t.start:
            r += 1
        region.append(points[r])
    if stats is not None:
        stats.setdefault("inactive", 0)

    def is_type(k):
        t = toks[k]
        if t.kind != "id":
            return False
        if t.text in C_TYPE_WORDS or t.text in types or t.text.endswith("_t") or t.text in C_ATOMIC_TYPES:
            return True
        return k >= 1 and toks[k - 1].text in ("struct", "union", "enum")

    match = {}
    stack = []
    depth_at = []
    depth = 0
    for k, t in enumerate(toks):
        depth_at.append(depth)
        if t.kind != "op":
            continue
        if t.text in ("(", "[", "{"):
            stack.append(k)
            if t.text == "{":
                depth += 1
        elif t.text in (")", "]", "}"):
            if stack:
                o = stack.pop()
                match[o] = k
                match[k] = o
            if t.text == "}":
                depth = max(0, depth - 1)

    init0 = []          # file scope: inside an initializer (after "=", before "," or ";")
    in_init, pd = False, 0
    for k, t in enumerate(toks):
        if depth_at[k] == 0 and t.kind == "op":
            if t.text in ("(", "["):
                pd += 1
            elif t.text in (")", "]"):
                pd = max(0, pd - 1)
            elif pd == 0 and t.text == "=":
                in_init = True
            elif pd == 0 and t.text in (",", ";"):
                in_init = False
        init0.append(in_init)

    def is_cast(close):
        o = match.get(close)
        if o is None or o + 1 >= close:
            return False
        if o >= 1 and toks[o - 1].text in ("sizeof", "_Alignof", "alignof"):   # sizeof(T *) * n
            return False
        inner = range(o + 1, close)
        return is_type(o + 1) and all(toks[k].text == "*" or is_type(k) for k in inner)

    def star_declarator(k):
        """`*` that declares a pointer even though the type name is unknown (system typedefs)."""
        nxt = toks[k + 1] if k + 1 < n else None
        if nxt is None or nxt.text in (")", ",", "*", "const", "volatile", "restrict"):
            return True
        if depth_at[k] == 0:
            return not init0[k]
        p = toks[k - 1] if k else None
        if p is None or p.kind != "id" or nxt.kind != "id":
            return False
        pp = toks[k - 2] if k >= 2 else None
        starts = pp is None or pp.text in (";", "{", "}") or pp.text in C_TYPE_WORDS or \
            (pp.text == "(" and k >= 3 and toks[k - 3].text == "for")
        after = toks[k + 2] if k + 2 < n else None
        return starts and (after is None or after.text in ("=", ";", ",", "[", ")"))

    def binary(k):
        if k == 0:
            return False
        if toks[k].text == "*" and star_declarator(k):
            return False
        p = toks[k - 1]
        if p.kind in ("num", "str"):
            return True
        if p.kind == "id":
            return p.text not in C_KEYWORDS and not is_type(k - 1)
        if p.text == "]":
            return True
        if p.text == ")":
            return not is_cast(k - 1)
        return False

    no_const = set()
    for k, t in enumerate(toks):
        if t.kind == "id" and t.text in ("_Static_assert", "static_assert") and k + 1 < n and toks[k + 1].text == "(":
            no_const.update(range(k + 1, match.get(k + 1, k + 1) + 1))
        if t.text == "[" and k >= 2 and toks[k - 1].kind == "id" and (is_type(k - 2) or toks[k - 2].text == "*"):
            no_const.update(range(k, match.get(k, k) + 1))

    def add(op, k, edits, before, after):
        t = toks[k]
        state, cond = region[k][1], region[k][2]
        if state == 0:
            if stats is not None:
                stats["inactive"] += 1
            return
        muts.append(Mutant(op, t.line, t.col, tuple(edits), before, after, cond=cond))

    for k, t in enumerate(toks):
        x = t.text
        if t.kind == "op":
            if x in REL_SWAP:
                for r in REL_SWAP[x]:
                    add("relational", k, [(t.start, t.end, r)], x, r)
            elif x in ("&&", "||"):
                r = "||" if x == "&&" else "&&"
                add("logical", k, [(t.start, t.end, r)], x, r)
            elif x == "!":
                add("negation", k, [(t.start, t.end, "")], x, "")
            elif x in ARITH_SWAP:
                if x in ("+", "-", "*") and not binary(k):
                    if x == "-" and k + 1 < n:
                        add("negation", k, [(t.start, t.end, "")], "-" + toks[k + 1].text, toks[k + 1].text)
                    continue
                for r in ARITH_SWAP[x]:
                    add("arithmetic", k, [(t.start, t.end, r)], x, r)
        elif t.kind == "num" and k not in no_const:
            m = _C_INT.match(x)
            if not m:
                continue
            body, suffix = m.group(1), m.group(2) or ""
            if body.lower().startswith("0x"):
                value = int(body, 16)
                upper = any(ch in "ABCDEF" for ch in body[2:])
                fmt = (lambda v, p=body[:2], u=upper: p + (format(v, "X") if u else format(v, "x")))
            else:
                value = int(body)
                fmt = str
            for v in _int_variants(value):
                if v < 0:
                    continue
                r = fmt(v) + suffix
                add("constant", k, [(t.start, t.end, r)], x, r)
        elif t.kind == "id":
            if x in ("if", "while") and k + 1 < n and toks[k + 1].text == "(" and (k + 1) in match:
                o, c = toks[k + 1], toks[match[k + 1]]
                if c.start > o.end:
                    cond = src[o.end:c.start]
                    add("condition", k, [(o.end, o.end, "!("), (c.start, c.start, ")")],
                        f"{x} ({_short(cond, 50)})", f"{x} (!({_short(cond, 50)}))")
            elif x == "for" and k + 1 < n and toks[k + 1].text == "(" and (k + 1) in match:
                close = match[k + 1]
                semis = [j for j in range(k + 2, close) if toks[j].text == ";" and
                         sum(1 for q in range(k + 2, j) if toks[q].text in "([{") ==
                         sum(1 for q in range(k + 2, j) if toks[q].text in ")]}")]
                if len(semis) >= 2 and semis[1] > semis[0] + 1:
                    a, b = toks[semis[0] + 1], toks[semis[1] - 1]
                    cond = src[a.start:b.end]
                    add("condition", k, [(a.start, a.start, "!("), (b.end, b.end, ")")],
                        f"for (...; {_short(cond, 50)}; ...)", f"for (...; !({_short(cond, 50)}); ...)")
            elif x == "return" and depth_at[k] >= 1:
                j, d = k + 1, 0
                while j < n and not (toks[j].text == ";" and d == 0):
                    if toks[j].text in ("(", "[", "{"):
                        d += 1
                    elif toks[j].text in (")", "]", "}"):
                        d -= 1
                    j += 1
                if j >= n or j == k + 1:
                    continue
                a, b = toks[k + 1], toks[j - 1]
                expr = src[a.start:b.end]
                compact = "".join(expr.split())
                if compact in ("NULL", "(void*)0", "nullptr"):
                    continue
                r = {"0": "1", "1": "0", "true": "false", "false": "true"}.get(compact, "0")
                add("return", k, [(a.start, b.end, r)], f"return {_short(expr, 50)}", f"return {r}")
            elif (x not in C_KEYWORDS and depth_at[k] >= 1
                  and (k == 0 or toks[k - 1].line != t.line)
                  and (k == 0 or toks[k - 1].text in (";", "{", "}", ")", ":", "else"))):
                j = k + 1
                while j + 1 < n and toks[j].text in (".", "->") and toks[j + 1].kind == "id":
                    j += 2
                if j < n and toks[j].text == "(" and j in match:
                    c = match[j]
                    if (c + 1 < n and toks[c + 1].text == ";" and toks[c + 1].line == t.line
                            and (c + 2 >= n or toks[c + 2].line != t.line)):
                        stmt = src[t.start:toks[c + 1].end]
                        add("call-delete", k, [(t.start, toks[c + 1].end, "(void)0;")], _short(stmt), "(void)0;")
    return muts


# ================================================================ Python
_PY_INT = re.compile(r"^(0[xX][0-9a-fA-F_]+|[1-9][0-9_]*|0+)$")
# f-strings (3.12+) and t-strings (3.14+) are tokenized into parts; skip them whole
_FSTART = {getattr(tokenize, n) for n in ("FSTRING_START", "TSTRING_START") if hasattr(tokenize, n)}
_FEND = {getattr(tokenize, n) for n in ("FSTRING_END", "TSTRING_END") if hasattr(tokenize, n)}
_PY_VALUES = {"True", "False", "None"}


@dataclass
class PTok:
    type: int
    string: str
    start: int
    end: int
    row: int
    col: int
    end_row: int


def lex_py(src: str) -> list:
    starts = [0] + [m.end() for m in re.finditer("\n", src)]

    def off(rc):
        r, c = rc
        return starts[r - 1] + c if r - 1 < len(starts) else len(src)

    out = []
    fdepth, fstart = 0, None
    skip = (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING)
    for tk in tokenize.generate_tokens(io.StringIO(src).readline):
        if tk.type in _FSTART:
            if fdepth == 0:
                fstart = tk
            fdepth += 1
            continue
        if fdepth:
            if tk.type in _FEND:
                fdepth -= 1
                if fdepth == 0:
                    out.append(PTok(tokenize.STRING, "f-string", off(fstart.start), off(tk.end),
                                    fstart.start[0], fstart.start[1] + 1, tk.end[0]))
            continue
        if tk.type in skip:
            continue
        out.append(PTok(tk.type, tk.string, off(tk.start), off(tk.end), tk.start[0], tk.start[1] + 1, tk.end[0]))
    return out


def gen_python(src: str) -> list:
    toks = lex_py(src)
    n = len(toks)
    muts = []
    kw = set(keyword.kwlist)
    OP, NAME, NUMBER, STRING = tokenize.OP, tokenize.NAME, tokenize.NUMBER, tokenize.STRING
    structural = (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)

    def stmt_start(k):
        return k == 0 or toks[k - 1].type in structural or (toks[k - 1].type == OP and toks[k - 1].string == ";")

    def binary(k):
        if k == 0:
            return False
        p = toks[k - 1]
        if p.type in (NUMBER, STRING):
            return True
        if p.type == NAME:
            return p.string not in kw or p.string in _PY_VALUES
        return p.type == OP and p.string in (")", "]", "}")

    def match_close(k):
        d = 0
        for j in range(k, n):
            s = toks[j].string if toks[j].type == OP else ""
            if s in ("(", "[", "{"):
                d += 1
            elif s in (")", "]", "}"):
                d -= 1
                if d == 0:
                    return j
            elif toks[j].type == tokenize.NEWLINE:
                return None
        return None

    def add(op, k, edits, before, after):
        t = toks[k]
        muts.append(Mutant(op, t.row, t.col, tuple(edits), before, after))

    for k, t in enumerate(toks):
        s = t.string
        if t.type == OP:
            if s in REL_SWAP:
                for r in REL_SWAP[s]:
                    add("relational", k, [(t.start, t.end, r)], s, r)
            elif s in ARITH_SWAP or s == "//":
                if s in ("+", "-", "*", "/", "//", "%") and not binary(k):
                    if s == "-" and k + 1 < n:
                        add("negation", k, [(t.start, t.end, "")], "-" + toks[k + 1].string, toks[k + 1].string)
                    continue
                for r in (ARITH_SWAP.get(s) or ("*",)):
                    add("arithmetic", k, [(t.start, t.end, r)], s, r)
        elif t.type == NUMBER:
            if not _PY_INT.match(s):
                continue
            value = int(s.replace("_", ""), 0) if not re.match(r"^0+$", s) else 0
            for v in _int_variants(value):
                r = hex(v) if s[:2].lower() == "0x" else str(v)
                add("constant", k, [(t.start, t.end, r)], s, r)
        elif t.type == NAME:
            if s in ("and", "or"):
                r = "or" if s == "and" else "and"
                add("logical", k, [(t.start, t.end, r)], s, r)
            elif s == "not" and k + 1 < n:
                if (k > 0 and toks[k - 1].string == "is") or toks[k + 1].string == "in":
                    continue
                add("negation", k, [(t.start, toks[k + 1].start, "")], "not " + toks[k + 1].string, toks[k + 1].string)
            elif s in ("if", "elif", "while") and stmt_start(k):
                d, j = 0, k + 1
                while j < n and toks[j].type != tokenize.NEWLINE:
                    q = toks[j].string if toks[j].type == OP else ""
                    if q in ("(", "[", "{"):
                        d += 1
                    elif q in (")", "]", "}"):
                        d -= 1
                    elif q == ":" and d == 0:
                        break
                    j += 1
                if j < n and j > k + 1 and toks[j].string == ":":
                    a = toks[k + 1]
                    cond = src[a.start:toks[j - 1].end]
                    add("condition", k, [(a.start, a.start, "not ("), (toks[j].start, toks[j].start, ")")],
                        f"{s} {_short(cond, 50)}:", f"{s} not ({_short(cond, 50)}):")
            elif s == "return" and stmt_start(k):
                j = k + 1
                while j < n and toks[j].type != tokenize.NEWLINE and not (toks[j].type == OP and toks[j].string == ";"):
                    j += 1
                if j == k + 1:
                    continue
                a, b = toks[k + 1], toks[j - 1]
                expr = src[a.start:b.end]
                r = {"None": None, "True": "False", "False": "True"}.get(expr.strip(), "None")
                if r is None:
                    continue
                add("return", k, [(a.start, b.end, r)], f"return {_short(expr, 50)}", f"return {r}")
            elif s not in kw and stmt_start(k):
                j = k + 1
                while j + 1 < n and toks[j].type == OP and toks[j].string == "." and toks[j + 1].type == NAME:
                    j += 2
                if j < n and toks[j].type == OP and toks[j].string == "(":
                    c = match_close(j)
                    if (c is not None and c + 1 < n and toks[c + 1].type == tokenize.NEWLINE
                            and toks[c].end_row == t.row):
                        stmt = src[t.start:toks[c].end]
                        add("call-delete", k, [(t.start, toks[c].end, "pass")], _short(stmt), "pass")
    return muts


# ============================================================= selection
C_EXTENSIONS = (".c", ".h", ".inc")     # .inc: a C fragment #included by .c files (platform_common.inc)


def language_of(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in C_EXTENSIONS:
        return "c"
    if ext == ".py":
        return "python"
    raise MutateError(f"unsupported file type {ext!r} (C: {'/'.join(C_EXTENSIONS)}, Python: .py)")


_QUOTED_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"', re.M)


def includers(path: Path, root: Path) -> list:
    """[(file, text before its #include)] for the C files next to path or in engine/src that
    include path by a quoted name. A fragment such as an .inc is compiled only there, so
    their earlier #defines, headers and typedefs are its context."""
    path = Path(path).resolve()
    out = []
    for d in dict.fromkeys([path.parent, (root / "engine" / "src").resolve()]):
        for f in sorted(d.glob("*")) if d.is_dir() else []:
            if f.suffix.lower() not in C_EXTENSIONS or f.resolve() == path or not f.is_file():
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for m in _QUOTED_INCLUDE.finditer(text):
                found = header_resolver(root)(m.group(1), f)
                if found is not None and found[1] == path:
                    out.append((f, text[:m.start()]))
                    break
    return out


def fragment_context(path: Path, root: Path, defs: dict) -> tuple:
    """(macro model, typedef names, includer files) in effect where the includers include
    path. A macro that the includers leave in different states is unknown."""
    models, types = [], set()
    resolve = header_resolver(root)
    found = includers(path, root)
    for f, before in found:
        pp = Preprocessor(defs, resolve)
        state = pp.regions(before, f)[-1][1]
        if state == 0:                  # included only in a branch this machine does not compile
            continue
        models.append(pp.defs)
        types |= c_typedef_names(lex_c(before))
        for h in _local_headers(before, f, root):
            try:
                types |= c_typedef_names(lex_c(h.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
    files = [f for f, _ in found]
    if not models:
        return defs, types, files
    merged = {k: v for k, v in models[0].items() if all(k in m and m[k] == v for m in models[1:])}
    return merged, types, files


def generate(src: str, path: Path, root: Path, defs: dict = None, stats: dict = None) -> tuple:
    """Returns (candidates, invalid_discarded). C: defs is the macro model (see host_macros)."""
    lang = language_of(path)
    if lang == "c":
        types = set()
        if Path(path).suffix.lower() == ".inc":
            defs, types, files = fragment_context(path, root, defs or {})
            if stats is not None:
                stats["includers"] = [f.relative_to(root).as_posix() if f.is_relative_to(root) else str(f)
                                      for f in files]
        for h in _local_headers(src, path, root):
            try:
                types |= c_typedef_names(lex_c(h.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
        muts = gen_c(src, types, defs, header_resolver(root), path, stats)
    else:
        muts = gen_python(src)
    lines = src.split("\n")
    seen, out, invalid = set(), [], 0
    for m in sorted(muts, key=lambda m: (m.edits[0][0], OPERATORS.index(m.op), m.after)):
        sig = m.edits
        if sig in seen:
            continue
        seen.add(sig)
        if "nomutate" in lines[m.line - 1]:
            continue
        if lang == "python":
            try:
                compile(m.apply(src), str(path), "exec", dont_inherit=True)
            except (SyntaxError, ValueError):
                invalid += 1
                continue
        out.append(m)
    for i, m in enumerate(out):
        m.idx = i
    return out, invalid


def parse_line_spec(spec: str) -> set:
    lines = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise MutateError(f"bad --lines range {part!r} (use 10-200,250)")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        if b < a:
            raise MutateError(f"bad --lines range {part!r}")
        lines.update(range(a, b + 1))
    return lines


def changed_lines(root: Path, rel: str, rev: str):
    """Lines of `rel` added/changed since `rev` (working tree vs rev). None = every line (a file
    git does not track yet). Raises MutateError outside a git checkout or for an unknown rev."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60)

    if not rev or rev.startswith("-"):
        raise MutateError(f"--since {rev!r}: not a revision")
    try:
        if git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
            raise MutateError(f"--since needs a git checkout; {root} is not inside one")
        if git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").returncode != 0:
            raise MutateError(f"--since {rev}: no such commit in {root}")
        if git("ls-files", "--error-unmatch", "--", rel).returncode != 0:
            return None
        d = git("diff", "-U0", "--no-color", "--no-ext-diff", rev, "--", rel)
    except (OSError, subprocess.SubprocessError) as e:
        raise MutateError(f"--since needs git: {e}") from None
    if d.returncode != 0:
        raise MutateError(f"git diff {rev} failed: {d.stderr.strip()}")
    return hunk_lines(d.stdout)


def hunk_lines(diff: str) -> set:
    """New-side line numbers touched by a `git diff -U0` (pure deletions touch none)."""
    out = set()
    for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, flags=re.M):
        start, count = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
        out.update(range(start, start + count))
    return out


def select(cands: list, max_n: int, seed: int) -> list:
    if max_n <= 0 or len(cands) <= max_n:
        return list(cands)
    rng = random.Random(seed)
    groups = {}
    for m in cands:
        groups.setdefault(m.op, []).append(m)
    order = [op for op in OPERATORS if op in groups]
    for op in order:
        rng.shuffle(groups[op])
    picked = []
    while len(picked) < max_n:
        before = len(picked)
        for op in order:
            if groups[op] and len(picked) < max_n:
                picked.append(groups[op].pop())
        if len(picked) == before:
            break
    return sorted(picked, key=lambda m: m.idx)


def unified(src: str, mutated: str, rel: str) -> str:
    a = src.replace("\r\n", "\n").splitlines(keepends=True)
    b = mutated.replace("\r\n", "\n").splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, f"a/{rel}", f"b/{rel}", n=1))


# =============================================================== running
IGNORE_DIRS = {".git", ".hg", ".svn", "build", "out", "__pycache__", ".pytest_cache", ".venv", "venv",
               "node_modules", ".mypy_cache", ".ruff_cache", ".tox"}
IGNORE_EXT = {".hearth", ".safetensors", ".gguf", ".bin", ".usage", ".hrtr", ".exe", ".obj", ".o", ".dll",
              ".so", ".dylib", ".pdb", ".ilk", ".exp", ".lib", ".pyc"}
MAX_COPY_BYTES = 50 * 1024 * 1024


def data_dir() -> Path:
    if os.environ.get("HEARTH_DATA"):
        return Path(os.environ["HEARTH_DATA"])
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "hearth"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "hearth"


def copy_repo(root: Path, dst: Path) -> None:
    def ignore(d, names):
        out = []
        for nm in names:
            if nm in IGNORE_DIRS or nm.endswith(".egg-info") or Path(nm).suffix.lower() in IGNORE_EXT:
                out.append(nm)
                continue
            p = os.path.join(d, nm)
            try:
                if os.path.isfile(p) and os.path.getsize(p) > MAX_COPY_BYTES:
                    out.append(nm)
            except OSError:
                out.append(nm)
        return out
    shutil.copytree(root, dst, ignore=ignore, copy_function=shutil.copyfile, ignore_dangling_symlinks=True)


def rmtree(path: Path) -> None:
    def onerr(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
            func(p)
        except OSError:
            pass
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=onerr)
    else:
        shutil.rmtree(path, onerror=onerr)


def quote(s: str) -> str:
    if os.name == "nt":
        return f'"{s}"'
    return shlex.quote(s)


UNIQUE_PLACEHOLDERS = ("{job}", "{run}", "{tmp}", "{root}")   # make an output name private to one job


def new_run_id() -> str:
    return "mr" + secrets.token_hex(4)   # "mr" cannot occur inside hex tags such as hxcc's .obj-<stem>-<sha1>


def expand(cmd: str, job: int, copy_root: Path, tmp_dir: Path, run_id: str) -> str:
    values = {"{job}": f"{run_id}-{job}", "{run}": run_id, "{tmp}": quote(str(tmp_dir)),
              "{python}": quote(sys.executable), "{root}": quote(str(copy_root))}
    return re.sub(r"\{(?:job|run|tmp|python|root)\}", lambda m: values[m.group(0)], cmd)


def remove_build_products(run_id: str) -> int:
    """Deletes files and directories under <data dir>/build/hxcc whose names carry run_id."""
    base = data_dir() / "build" / "hxcc"
    if not run_id.startswith("mr") or len(run_id) < 10 or not base.is_dir():
        return 0
    removed = 0
    for dirpath, dirnames, filenames in os.walk(base):
        for d in [d for d in dirnames if run_id in d]:
            rmtree(Path(dirpath) / d)
            dirnames.remove(d)
            removed += 1
        for f in filenames:
            if run_id in f:
                try:
                    os.unlink(Path(dirpath) / f)
                    removed += 1
                except OSError:
                    pass
    return removed


@contextmanager
def sigint_ignored():
    """Lets cleanup finish when Ctrl-C is pressed again (main thread only)."""
    try:
        old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:
        old = None
    try:
        yield
    finally:
        if old is not None:
            signal.signal(signal.SIGINT, old)


class Cancelled(Exception):
    pass


def _kill_tree(p: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True, timeout=30)
        else:
            import signal
            os.killpg(p.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        p.kill()
    except OSError:
        pass


def run_shell(cmd: str, cwd: Path, env: dict, timeout: float, running: set = None, cancelled=None):
    """Returns (returncode, output, seconds, timed_out). The process is kept in
    `running` while it runs, so another thread can kill it after setting `cancelled`."""
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    t0 = time.monotonic()
    p = subprocess.Popen(cmd, shell=True, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)
    if running is not None:
        running.add(p)
    timed_out = False
    try:
        if cancelled is not None and cancelled.is_set():
            _kill_tree(p)
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(p)
            try:
                out, _ = p.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                out = b""
    finally:
        if running is not None:
            running.discard(p)
    return p.returncode, (out or b"").decode("utf-8", "replace"), time.monotonic() - t0, timed_out


def read_source(path: Path) -> str:
    with open(path, encoding="utf-8", errors="surrogateescape", newline="") as f:
        return f.read()


def write_source(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
        f.write(text)


class Runner:
    def __init__(self, root: Path, rel: str, src: str, test: str, build, timeout: float, build_re, jobs: int,
                 workdir: Path, verbose: bool = True, run_id: str = "mr00000000"):
        self.root, self.rel, self.src = root, rel, src
        self.test, self.build, self.timeout = test, build, timeout
        self.build_re = re.compile(build_re, re.M) if build_re else None
        self.workdir = workdir
        self.run_id = run_id
        self.copies = [workdir / f"job{j}" for j in range(jobs)]
        self.tmps = [workdir / f"tmp{j}" for j in range(jobs)]
        self.slots: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.done = 0
        self.verbose = verbose
        self.running: set = set()
        self.cancelled = threading.Event()

    def env(self, job: int) -> dict:
        env = dict(os.environ)
        copy_root = self.copies[job]
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["HEARTH_MUTATION_JOB"] = str(job)
        env["HEARTH_MUTATION_RUN"] = self.run_id
        env["HEARTH_MUTATION_ROOT"] = str(copy_root)
        env["HEARTH_MUTATION_TMP"] = str(self.tmps[job])
        if env.get("PYTHONPATH"):
            parts = []
            for e in env["PYTHONPATH"].split(os.pathsep):
                try:
                    parts.append(str(copy_root / Path(e).resolve().relative_to(self.root)))
                except (ValueError, OSError):
                    parts.append(e)
            env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    def prepare(self) -> None:
        for j, c in enumerate(self.copies):
            copy_repo(self.root, c)
            if not (c / self.rel).is_file():
                raise MutateError(f"{self.rel} was not copied into the job directory (ignored path?)")
            self.tmps[j].mkdir()
            self.slots.put(j)

    def cancel(self) -> None:
        """Stops queued work and kills running commands."""
        self.cancelled.set()
        for p in list(self.running):
            _kill_tree(p)

    def _shell(self, cmd: str, job: int, timeout: float):
        return run_shell(expand(cmd, job, self.copies[job], self.tmps[job], self.run_id), self.copies[job],
                         self.env(job), timeout, self.running, self.cancelled)

    def _clear_pycache(self, job: int) -> None:
        pc = (self.copies[job] / self.rel).parent / "__pycache__"
        if pc.is_dir():
            rmtree(pc)

    def run_once(self, job: int, text: str, timeout: float) -> dict:
        if self.cancelled.is_set():
            raise Cancelled()
        copy_root = self.copies[job]
        target = copy_root / self.rel
        write_source(target, text)
        self._clear_pycache(job)
        res = {"seconds": 0.0}
        try:
            if self.build:
                rc, out, dt, to = self._shell(self.build, job, timeout)
                res["seconds"] += dt
                if self.cancelled.is_set():
                    raise Cancelled()
                if to or rc != 0:
                    res.update(status="build-error", output=out)
                    return res
            rc, out, dt, to = self._shell(self.test, job, timeout)
            res["seconds"] += dt
            if self.cancelled.is_set():
                raise Cancelled()
            res["output"] = out
            if to:
                res["status"] = "timeout"
            elif rc == 0:
                res["status"] = "survived"
            elif not self.build and self.build_re and self.build_re.search(out):
                res["status"] = "build-error"
            else:
                res["status"] = "killed"
            res["returncode"] = rc
            return res
        finally:
            write_source(target, self.src)

    def baseline(self, timeout: float) -> dict:
        job = self.slots.get()
        try:
            return self.run_once(job, self.src, timeout)
        finally:
            self.slots.put(job)

    def run_mutant(self, m: Mutant, total: int, retry: bool = False) -> dict:
        job = self.slots.get()
        try:
            r = self.run_once(job, m.apply(self.src), self.timeout)
        finally:
            self.slots.put(job)
        with self.lock:
            if not retry:
                self.done += 1
            if self.verbose:
                where = "[retry]" if retry else f"[{self.done:>3}/{total}]"
                print(f"{where} {m.name} {m.op:<11} line {m.line:<5} "
                      f"{_short(m.before, 30)!r} -> {_short(m.after, 30)!r}: {r['status']} ({r['seconds']:.1f}s)",
                      file=sys.stderr, flush=True)
        return r

    def run_parallel(self, fn, items: list, jobs: int) -> list:
        """[fn(item)] on `jobs` threads. The main thread polls, so Ctrl-C is seen at
        once; it then cancels queued items and kills running commands."""
        ex = ThreadPoolExecutor(max_workers=jobs)
        futs = {ex.submit(fn, it): k for k, it in enumerate(items)}
        results = [None] * len(items)
        try:
            pending = set(futs)
            while pending:
                done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                for f in done:
                    results[futs[f]] = f.result()
        except BaseException:
            with sigint_ignored():
                self.cancel()
                ex.shutdown(wait=True, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
        return results


def seconds_arg(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        v = float("nan")
    if not (v > 0 and v != float("inf")):
        raise argparse.ArgumentTypeError(f"expected a positive number of seconds, got {text!r}")
    return v


def mutant_timeout(requested, baseline_s: float) -> float:
    """Per-mutant timeout: the requested one, but never so short that the unmutated code
    would time out (a slow pass must not count as a kill)."""
    floor = max(2.0 * baseline_s, baseline_s + 5.0)
    if requested is None:
        return max(10.0, 3.0 * baseline_s + 5.0)
    return max(requested, floor)


# =============================================================== reports
def score_of(counts: dict):
    detected = counts["killed"] + counts["timeout"]
    scored = detected + counts["survived"]
    return (detected / scored) if scored else None


def markdown(rep: dict) -> str:
    c = rep["counts"]
    sc = rep["score"]
    lines = [f"# Mutation report: `{rep['file']}`", ""]
    lines.append(f"- Test command: `{rep['test']}`" + (f"  \n- Build command: `{rep['build']}`" if rep["build"] else ""))
    lines.append(f"- Seed {rep['seed']}: {rep['selected']} of {rep['candidates']} candidate mutants "
                 f"(stratified by operator), {rep['jobs']} job(s), timeout {rep['timeout_s']:.0f} s, "
                 f"baseline {rep['baseline_s']:.1f} s")
    if rep.get("lines") or rep.get("since"):
        lines.append(f"- Restricted to lines {rep.get('lines') or ''} {('changed since ' + rep['since']) if rep.get('since') else ''}".rstrip())
    if rep.get("language") == "c":
        lines.append(f"- Preprocessor model `{rep['pp_model']}`: {rep['inactive_skipped']} mutant(s) in inactive "
                     "#if branches not generated")
    if rep.get("timeouts_retried"):
        lines.append(f"- {rep['timeouts_retried']} timed-out mutant(s) re-run alone; "
                     f"{rep['timeouts_cleared']} of them no longer timed out")
    for w in rep.get("warnings", []):
        lines.append(f"- **Warning:** {w}")
    if sc is None:
        lines.append("- **Mutation score: n/a** (no scorable mutants)")
    else:
        lines.append(f"- **Mutation score: {100 * sc:.1f}%** = (killed {c['killed']} + timeout {c['timeout']}) / "
                     f"{c['killed'] + c['timeout'] + c['survived']} scored; {c['build-error']} build error(s) excluded")
    lines += ["", "| operator | killed | timeout | survived | build-error |", "|---|---:|---:|---:|---:|"]
    for op, oc in rep["by_operator"].items():
        lines.append(f"| {op} | {oc['killed']} | {oc['timeout']} | {oc['survived']} | {oc['build-error']} |")
    surv = [m for m in rep["mutants"] if m["status"] == "survived"]
    lines += ["", f"## Surviving mutants ({len(surv)})", ""]
    if not surv:
        lines.append("None.")
    else:
        lines.append("Each survivor is either a gap in the tests or an equivalent mutant (same behaviour); "
                     "say which in your handover.")
    for m in surv:
        lines += ["", f"### {m['id']} {m['op']}, line {m['line']}: `{m['before']}` -> `{m['after']}`", ""]
        if m.get("cond"):
            lines += [f"Under `{m['cond']}`, which could not be evaluated: if that branch is not compiled "
                      "on this machine, the mutant cannot be killed here.", ""]
        lines += ["```diff", m["diff"].rstrip("\n"), "```"]
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", required=True, help="source file to mutate (repo-relative or absolute)")
    ap.add_argument("--test", help="shell command that exits 0 when the tests pass")
    ap.add_argument("--build", help="optional separate build command (failure => build-error)")
    ap.add_argument("--root", default=str(ROOT), help="repository root to copy (default: this checkout)")
    ap.add_argument("--max-mutants", type=int, default=40, help="sample size; 0 = all candidates")
    ap.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--lines", help="only mutate these lines, e.g. 10-200,250")
    ap.add_argument("--since", metavar="REV", help="only mutate lines changed since this git revision")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=seconds_arg,
                    help="seconds per mutant and for the baseline (default: max(10, 3*baseline+5); never below "
                         "max(2*baseline, baseline+5))")
    ap.add_argument("--build-error-regex", default=None,
                    help="output pattern that marks a failed run as build-error (default for C: compiler/linker "
                         "and hxcc failures; Python mutants are compile-checked up front, so none)")
    ap.add_argument("--out-dir", help="report directory (default: <data dir>/mutation/<file>-s<seed>-<time>)")
    ap.add_argument("--min-score", type=float, help="exit 1 if the score is below this fraction (0..1)")
    ap.add_argument("--list", action="store_true", help="only list the selected mutants")
    ap.add_argument("--json", action="store_true", help="print the JSON report (or list) to stdout")
    ap.add_argument("--keep-temp", action="store_true", help="keep the job directories for debugging")
    ap.add_argument("--no-timeout-retry", action="store_true", help="do not re-run timed-out mutants alone")
    ap.add_argument("--pp-model", choices=PP_MODELS, default="auto",
                    help="C: predefined-macro model for #if evaluation (default auto)")
    ap.add_argument("-D", "--define", action="append", metavar="NAME[=VALUE]", help="C: treat NAME as defined")
    ap.add_argument("-U", "--undefine", action="append", metavar="NAME", help="C: treat NAME as not defined")
    ap.add_argument("-q", "--quiet", action="store_true", help="no per-mutant progress on stderr")
    a = ap.parse_args(argv)

    try:
        root = Path(a.root).resolve()
        path = Path(a.file)
        path = (path if path.is_absolute() else root / path).resolve()
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            raise MutateError(f"{path} is not inside the repository root {root}") from None
        if not path.is_file():
            raise MutateError(f"no such file: {path}")
        src = read_source(path)
        defs = host_macros(a.pp_model)
        defs.update(parse_macro_args(a.define, a.undefine))
        gen_stats: dict = {}
        cands, invalid = generate(src, path, root, defs, gen_stats)
        n_cands = len(cands)
        allowed = None
        if a.lines:
            allowed = parse_line_spec(a.lines)
        if a.since:
            ch = changed_lines(root, rel, a.since)
            if ch is not None:
                allowed = ch if allowed is None else (allowed & ch)
        if allowed is not None:
            cands = [m for m in cands if m.line in allowed]
        selected = select(cands, a.max_mutants, a.seed)
    except (MutateError, OSError, SyntaxError, tokenize.TokenError) as e:
        print(f"mutate: {e}", file=sys.stderr)
        return 2

    if a.list:
        if a.json:
            print(json.dumps([dict({"id": m.name, "key": m.key(rel), "op": m.op, "line": m.line, "col": m.col,
                                    "before": m.before, "after": m.after}, **({"cond": m.cond} if m.cond else {}))
                              for m in selected], indent=2))
        else:
            for m in selected:
                print(f"{m.name} {m.op:<11} {m.line}:{m.col}  {m.before!r} -> {m.after!r}"
                      + (f"   [under {m.cond}]" if m.cond else ""))
            print(f"{len(selected)} selected of {len(cands)} in range ({n_cands} candidates, "
                  f"{invalid} invalid discarded, {gen_stats.get('inactive', 0)} in inactive #if branches)")
            if "includers" in gen_stats:
                print(f"fragment context: included by {', '.join(gen_stats['includers']) or 'nothing found'}")
        return 0
    if not a.test:
        print("mutate: --test is required (or use --list)", file=sys.stderr)
        return 2
    if not selected:
        print("mutate: no mutants to run (empty file, filters or range)", file=sys.stderr)
        return 2
    jobs = max(1, min(a.jobs, len(selected)))
    cmds = a.test + " " + (a.build or "")
    if not any(ph in cmds for ph in UNIQUE_PLACEHOLDERS) and re.search(r"(^|\s)(-o|--out)(\s|=)", cmds):
        print("mutate: warning: the -o output is shared by all jobs and by every other mutate.py run on this "
              "machine; write it to the job's own directory (-o {tmp}/name.exe) or add {job} to the name",
              file=sys.stderr)

    run_id = new_run_id()
    workdir = Path(tempfile.mkdtemp(prefix=f"hearth-mutate-{run_id}-"))
    t_start = time.monotonic()
    started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    build_re = a.build_error_regex
    if build_re is None and language_of(path) == "c":
        build_re = DEFAULT_BUILD_ERROR_RE
    runner = Runner(root, rel, src, a.test, a.build, 0.0, build_re, jobs, workdir,   # timeout set after baseline
                    verbose=not a.quiet, run_id=run_id)
    retried = 0
    try:
        runner.prepare()
        base_timeout = a.timeout or 600.0
        base = runner.run_parallel(lambda _: runner.baseline(base_timeout), [None], 1)[0]
        if base["status"] == "timeout":
            print(f"mutate: the test command did not finish within {base_timeout:g} s on the unmutated code"
                  + ("; raise --timeout" if a.timeout else "") + ". Output tail:\n" + base.get("output", "")[-3000:],
                  file=sys.stderr)
            return 2
        if base["status"] != "survived":
            print(f"mutate: the test command does not pass on the unmutated code ({base['status']}); "
                  "fix that first. Output tail:\n" + base.get("output", "")[-3000:], file=sys.stderr)
            return 2
        runner.timeout = mutant_timeout(a.timeout, base["seconds"])
        if a.timeout is not None and runner.timeout > a.timeout:
            print(f"mutate: warning: --timeout {a.timeout:g} s leaves no margin over the unmutated run "
                  f"({base['seconds']:.1f} s); using {runner.timeout:.1f} s so that a slow pass is not scored "
                  "as a kill", file=sys.stderr)
        if not a.quiet:
            print(f"mutate: {rel}: {len(selected)} mutants, {jobs} job(s), baseline {base['seconds']:.1f}s, "
                  f"timeout {runner.timeout:.0f}s, run {run_id}", file=sys.stderr)
        results = runner.run_parallel(lambda m: runner.run_mutant(m, len(selected)), selected, jobs)
        if not a.no_timeout_retry:
            for k, r in enumerate(results):
                if r["status"] == "timeout":
                    again = runner.run_parallel(lambda m: runner.run_mutant(m, len(selected), retry=True),
                                                [selected[k]], 1)[0]
                    again["first_status"] = "timeout"
                    results[k] = again
                    retried += 1
    except KeyboardInterrupt:
        print("mutate: interrupted; running commands killed, job directories removed", file=sys.stderr)
        return 130
    except (MutateError, OSError, shutil.Error) as e:
        print(f"mutate: {e}", file=sys.stderr)
        return 2
    finally:
        with sigint_ignored():
            runner.cancel()
            if a.keep_temp:
                print(f"mutate: job directories kept in {workdir}", file=sys.stderr)
            else:
                rmtree(workdir)
            if "{job}" in cmds or "{run}" in cmds:
                remove_build_products(run_id)

    counts = {k: 0 for k in OUTCOMES}
    by_op = {}
    records = []
    for m, r in zip(selected, results):
        counts[r["status"]] += 1
        by_op.setdefault(m.op, {k: 0 for k in OUTCOMES})[r["status"]] += 1
        rec = {"id": m.name, "key": m.key(rel), "op": m.op, "line": m.line, "col": m.col, "before": m.before,
               "after": m.after, "status": r["status"], "seconds": round(r["seconds"], 3),
               "diff": unified(src, m.apply(src), rel)}
        if m.cond:
            rec["cond"] = m.cond
        if r.get("first_status"):
            rec["retried_after"] = r["first_status"]
        if r["status"] in ("build-error", "timeout") or r.get("first_status"):
            rec["output_tail"] = r.get("output", "")[-1500:]
        records.append(rec)
    score = score_of(counts)
    warnings = []
    if counts["timeout"] and not counts["killed"] and not counts["survived"]:
        warnings.append("every scored mutant timed out, so the score proves nothing about the tests; "
                        "check the timeout and rerun with a larger --timeout")
    rep = {
        "tool": "governance/tools/mutate.py", "version": VERSION, "run": run_id, "started": started,
        "elapsed_s": round(time.monotonic() - t_start, 1), "file": rel, "language": language_of(path),
        "test": a.test, "build": a.build, "seed": a.seed, "jobs": jobs, "timeout_s": runner.timeout,
        "timeout_requested_s": a.timeout, "baseline_s": round(base["seconds"], 3), "lines": a.lines,
        "since": a.since,
        "pp_model": resolve_pp_model(a.pp_model), "inactive_skipped": gen_stats.get("inactive", 0),
        "included_by": gen_stats.get("includers"),
        "candidates": n_cands, "in_range": len(cands), "invalid_discarded": invalid, "selected": len(selected),
        "timeouts_retried": retried,
        "timeouts_cleared": sum(1 for r in results if r.get("first_status") and r["status"] != "timeout"),
        "counts": counts, "score": score, "by_operator": {op: by_op[op] for op in OPERATORS if op in by_op},
        "warnings": warnings, "mutants": records,
    }
    out_dir = Path(a.out_dir) if a.out_dir else (
        data_dir() / "mutation" / (re.sub(r"[^A-Za-z0-9]+", "_", rel).strip("_") +
                                   f"-s{a.seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{run_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mutation.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (out_dir / "mutation.md").write_text(markdown(rep), encoding="utf-8")

    if a.json:
        print(json.dumps(rep, indent=2))
    else:
        sc = "n/a" if score is None else f"{100 * score:.1f}%"
        print(f"mutation score {sc}: killed {counts['killed']}, timeout {counts['timeout']}, "
              f"survived {counts['survived']}, build-error {counts['build-error']} "
              f"({len(selected)} of {n_cands} candidates, seed {a.seed})")
        for r in records:
            if r["status"] == "survived":
                print(f"  survived {r['id']} {r['op']} line {r['line']}: {r['before']!r} -> {r['after']!r}")
        print(f"reports: {out_dir / 'mutation.md'}  {out_dir / 'mutation.json'}")
    for w in warnings:
        print(f"mutate: warning: {w}", file=sys.stderr)
    if a.min_score is not None and (score is None or score < a.min_score or warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
