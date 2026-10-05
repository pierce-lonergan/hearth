"""`python -m hearth <command>`: info, run, chat, serve, bench, convert, synth, sim, doctor.

Every subcommand imports what it needs when it runs, so `--help` and `doctor`
work without the native library or optional dependencies.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

__all__ = ["build_parser", "main", "engine_kwargs"]

DTYPES = {"f32": 0, "f16": 1, "bf16": 2, "q8": 3, "q4": 4}
SYNCED_MARKERS = ("onedrive", "dropbox", "icloud", "google drive", "googledrive", "mobile documents", "box sync")


class CLIError(Exception):
    pass


# ---- shared argument groups ------------------------------------------------------

def add_engine_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("engine")
    g.add_argument("--cache-gb", type=float, default=8.0, help="DRAM budget for routed experts in GiB (default 8)")
    g.add_argument("--threads", type=int, default=0, help="compute threads (0 = physical cores)")
    g.add_argument("--io-threads", type=int, default=0, help="expert reader threads (0 = 8)")
    g.add_argument("--buffered", action="store_true", help="use the OS page cache instead of direct I/O")
    g.add_argument("--policy", choices=["lru", "lfu"], default="lfu", help="expert cache eviction policy")
    g.add_argument("--prefetch", choices=["off", "next", "shared"], default="shared", help="next-layer prefetch mode")
    g.add_argument("--prefetch-extra", type=int, default=0, help="predict top_k + N experts for the next layer")
    g.add_argument("--usage-in", metavar="PATH", help="heat profile to seed / pin the cache")
    g.add_argument("--usage-out", metavar="PATH", help="write the updated heat profile here on close")
    g.add_argument("--pin-fraction", type=float, default=0.0, help="fraction of cache slots pinned at open")
    g.add_argument("--warm-start", action="store_true", help="fill the cache with the hottest experts at open")
    g.add_argument("--max-seq", type=int, default=0, help="KV capacity in tokens (0 = min(model, 4096))")
    g.add_argument("--max-batch", type=int, default=0, help="max tokens per forward chunk (0 = 512)")
    g.add_argument("--isa", choices=["auto", "scalar", "avx2", "avx512"], default="auto")
    g.add_argument("--mirror", action="append", default=[], metavar="PATH",
                   help="byte-identical copy of the model on another drive (repeatable, max 8)")
    g.add_argument("-v", "--verbose", action="count", default=0, help="engine log level (-v info, -vv debug)")


def engine_kwargs(args) -> dict:
    return dict(cache_gb=args.cache_gb, threads=args.threads, io_threads=args.io_threads,
                direct_io=not args.buffered, policy=args.policy, prefetch=args.prefetch,
                prefetch_extra=args.prefetch_extra, usage_in=args.usage_in, usage_out=args.usage_out,
                pin_fraction=args.pin_fraction, warm_start=args.warm_start, max_seq=args.max_seq,
                max_batch=args.max_batch, isa=args.isa, mirrors=list(args.mirror), verbose=args.verbose)


def add_sampling_args(p: argparse.ArgumentParser, temperature: float, max_tokens: int) -> None:
    g = p.add_argument_group("sampling")
    g.add_argument("-n", "--max-tokens", type=int, default=max_tokens, help=f"tokens to generate (default {max_tokens})")
    g.add_argument("--temperature", type=float, default=temperature, help=f"0 = greedy (default {temperature})")
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--min-p", type=float, default=0.0)
    g.add_argument("--repetition-penalty", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=None)
    add_spec_args(g)


def add_spec_args(g) -> None:
    g.add_argument("--speculative", choices=["none", "ngram"], default="none",
                   help="prompt-lookup speculative decoding (lossless)")
    g.add_argument("--draft-len", type=int, default=4, help="max draft tokens per forward (default 4)")
    g.add_argument("--ngram-n", type=int, default=3, help="longest n-gram matched for drafting (default 3)")


def sampler_from(args):
    from hearth.generate import Sampler
    return Sampler(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, min_p=args.min_p,
                   repetition_penalty=args.repetition_penalty, seed=args.seed)


def check_generation_args(args) -> None:
    """Reject bad generation flags before a (possibly huge) model is opened."""
    from hearth.generate import check_speculative
    try:
        check_speculative(args.speculative, args.draft_len, args.ngram_n)
    except ValueError as e:
        raise CLIError(str(e)) from None
    if getattr(args, "max_tokens", 0) < 0:
        raise CLIError(f"--max-tokens must be >= 0, got {args.max_tokens}")


# ---- helpers ---------------------------------------------------------------------

def data_dir() -> Path:
    try:
        from hearth._native import data_dir as _dd
        return Path(_dd())
    except ImportError:
        pass
    if os.environ.get("HEARTH_DATA"):
        return Path(os.environ["HEARTH_DATA"])
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "hearth"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "hearth"


def is_cloud_synced(path) -> bool:
    parts = [s.lower() for s in Path(os.path.abspath(path)).parts]
    return any(any(part.startswith(m) for m in SYNCED_MARKERS) for part in parts)


def output_path(p: str, force: bool) -> Path:
    """Bare file names go to <data dir>/models; cloud-synced destinations are
    refused (INV-DATA) unless forced."""
    path = Path(p)
    if not path.is_absolute() and path.parent == Path("."):
        path = data_dir() / "models" / path
    if is_cloud_synced(path) and not force:
        raise CLIError(f"refusing to write a model into a cloud-synced folder ({path}); "
                       f"use a path under {data_dir()} or pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return str(n)


def _stats_line(st, eng_before: dict | None, eng_after: dict | None) -> str:
    parts = [f"{st.new_tokens} tok", f"decode {st.decode_tok_s:.2f} tok/s"]
    n_pref = st.prompt_tokens - st.reused_tokens
    parts.append(f"prefill {n_pref} tok {st.prefill_tok_s:.1f} tok/s" +
                 (f" ({st.reused_tokens} cached)" if st.reused_tokens else ""))
    if st.drafted:
        parts.append(f"accept {st.acceptance_rate:.0%} ({st.tokens_per_forward:.2f} tok/fwd)")
    if eng_before is not None and eng_after is not None:
        h = eng_after["cache_hits"] - eng_before["cache_hits"]
        m = eng_after["cache_misses"] - eng_before["cache_misses"]
        if h + m:
            parts.append(f"hit {h / (h + m):.0%}")
        gb = (eng_after["bytes_read"] - eng_before["bytes_read"]) / 1e9
        if gb:
            parts.append(f"read {gb:.2f} GB")
    if st.finish_reason:
        parts.append(st.finish_reason)
    return "[" + ", ".join(parts) + "]"


def _engine_stats(engine):
    fn = getattr(engine, "stats", None)
    try:
        return fn() if callable(fn) else None
    except Exception:
        return None


def _parse_ids(s: str) -> list[int]:
    try:
        ids = [int(x) for x in s.replace(",", " ").split()]
    except ValueError:
        raise CLIError(f"--ids must be integers separated by commas or spaces, got {s!r}") from None
    if not ids:
        raise CLIError("--ids is empty")
    return ids


def _reconfigure_stdio() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


# ---- commands ----------------------------------------------------------------------

def cmd_info(args) -> int:
    from hearth.chat import find_tokenizer, read_metadata
    path = Path(args.model)
    meta = read_metadata(path)
    out = {"path": str(path), "file_bytes": path.stat().st_size}
    summary_keys = ["arch", "source", "n_layers", "d_model", "vocab_size", "max_seq", "attn_kind", "n_heads",
                    "n_kv_heads", "head_dim", "n_experts", "top_k", "expert_ffn_dim", "shared_ffn_dim",
                    "dense_ffn_dim", "expert_dtype", "bos_id", "eos_ids"]
    out["metadata"] = {k: meta[k] for k in summary_keys if k in meta}
    tok = find_tokenizer(path, meta)
    out["tokenizer"] = str(tok) if tok else None
    out["chat_template"] = bool(meta.get("chat_template"))
    if args.open:
        from hearth.engine import Engine
        with Engine(path, **engine_kwargs(args)) as e:
            out["engine"] = dict(e.info)
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    print(f"{path}  ({_fmt_bytes(out['file_bytes'])})")
    for k, v in out["metadata"].items():
        print(f"  {k:16s} {v}")
    print(f"  {'tokenizer':16s} {out['tokenizer'] or '-'}")
    print(f"  {'chat_template':16s} {'yes' if out['chat_template'] else 'no (ChatML fallback)'}")
    if "engine" in out:
        i = out["engine"]
        print("engine:")
        print(f"  params total {i['params_total'] / 1e9:.2f} B, active {i['params_active'] / 1e9:.2f} B")
        print(f"  resident {_fmt_bytes(i['dense_bytes'])}, experts {_fmt_bytes(i['expert_bytes'])}, "
              f"largest slab {_fmt_bytes(i['slab_bytes_max'])}")
        print(f"  cache slots {i['cache_slots']}, isa {i.get('isa_name', i['isa'])}, threads {i['n_threads']}, "
              f"io threads {i['n_io_threads']}")
    return 0


def _load_text_stack(args, need_tokenizer: bool):
    from hearth.chat import ChatTemplate, Tokenizer, read_metadata
    try:
        meta = read_metadata(args.model)
    except (OSError, ValueError):
        if need_tokenizer:
            raise
        meta = {}  # raw token ids: the engine validates the container itself
    tok = None
    try:
        tok = Tokenizer.from_container(args.model, meta, path=getattr(args, "tokenizer", None) or None)
    except FileNotFoundError:
        if need_tokenizer:
            raise CLIError(f"no tokenizer for {args.model}: pass --tokenizer PATH (or --ids for raw token ids)")
    return meta, tok, ChatTemplate.from_metadata(meta, tok)


def cmd_run(args) -> int:
    from hearth.chat import TextStream, default_stop_ids
    from hearth.engine import Engine
    from hearth.generate import generate

    check_generation_args(args)
    raw_ids = _parse_ids(args.ids) if args.ids else None
    if raw_ids is None and args.prompt is None and args.prompt_file is None:
        raise CLIError("give --prompt, --prompt-file or --ids")
    meta, tok, template = _load_text_stack(args, need_tokenizer=raw_ids is None)
    if raw_ids is not None:
        ids = raw_ids
    else:
        text = args.prompt if args.prompt is not None else Path(args.prompt_file).read_text(encoding="utf-8")
        if args.chat:
            msgs = ([{"role": "system", "content": args.system}] if args.system else []) + \
                   [{"role": "user", "content": text}]
            ids = tok.encode(template.render(msgs, add_generation_prompt=True), add_special_tokens=False)
        else:
            ids = tok.encode(text, add_special_tokens=True)
    sampler = sampler_from(args)
    with Engine(args.model, **engine_kwargs(args)) as eng:
        stop_ids = [] if args.ignore_eos else default_stop_ids(eng, tok, template if args.chat else None)
        before = _engine_stats(eng)
        ts = TextStream(tok, [template.end_of_turn] if args.chat and template.end_of_turn else []) \
            if tok is not None and not args.print_ids else None
        gen = generate(eng, ids, args.max_tokens, sampler, stop_ids, args.speculative, args.draft_len, args.ngram_n)
        st, out_ids = None, []
        try:
            while True:
                try:
                    t = next(gen)
                except StopIteration as e:
                    st = e.value
                    break
                st = t.stats
                out_ids.append(t.id)
                if ts is None:
                    print(t.id, end=" ", flush=True)
                    continue
                d = ts.push(t.id)
                if d:
                    sys.stdout.write(d)
                    sys.stdout.flush()
                if ts.stopped is not None:
                    break
            if ts is not None and ts.stopped is None:
                sys.stdout.write(ts.finish())
        except KeyboardInterrupt:
            _stderr("\n[interrupted]")
        finally:
            gen.close()
        print(flush=True)
        if not args.quiet and st is not None:
            _stderr(_stats_line(st, before, _engine_stats(eng)))
        if args.json_out:
            Path(args.json_out).write_text(json.dumps({"prompt_ids": ids, "output_ids": out_ids,
                                                       "stats": st.as_dict() if st else None}, indent=1))
    return 0


def _chat_turn(conv, line: str, args, sampler, cap: int) -> None:
    """One REPL turn. A turn that cannot run (context full, engine error) is
    reported and rolled back; the session continues."""
    from hearth.engine import HearthError
    eng = conv.engine
    try:
        need = len(conv.prompt_ids(conv.messages + [{"role": "user", "content": line}]))
    except Exception as e:  # chat template errors (raise_exception, undefined access)
        _stderr(f"[cannot build the prompt: {e}]")
        return
    if need >= cap:
        _stderr(f"[context full: this turn needs {need} prompt tokens, the KV cache holds {cap}; "
                f"/reset clears the conversation, or restart with a larger --max-seq]")
        return
    before = _engine_stats(eng)
    stream = conv.say(line, args.max_tokens, sampler, args.speculative, args.draft_len, args.ngram_n)
    try:
        for d in stream:
            sys.stdout.write(d)
            sys.stdout.flush()
    except KeyboardInterrupt:
        _stderr("\n[interrupted]")
    except (ValueError, HearthError) as e:
        print(flush=True)
        _stderr(f"[turn failed and was discarded: {e}]")
        return
    finally:
        stream.close()
    print(flush=True)
    st = conv.last_stats
    if st is None:
        return
    if not args.quiet:
        _stderr(_stats_line(st, before, _engine_stats(eng)))
    if st.finish_reason == "length" and st.new_tokens < args.max_tokens:
        _stderr(f"[reply cut off: the KV cache ({cap} tokens) is full; /reset clears the conversation]")


def cmd_chat(args) -> int:
    from hearth.chat import Conversation
    from hearth.engine import Engine
    from hearth.generate import kv_capacity

    check_generation_args(args)
    meta, tok, template = _load_text_stack(args, need_tokenizer=True)
    if template.is_fallback:
        _stderr("note: the container has no chat template; using ChatML")
    sampler = sampler_from(args)
    with Engine(args.model, **engine_kwargs(args)) as eng:
        conv = Conversation(eng, tok, template, system=args.system)
        cap = kv_capacity(eng)
        info = eng.info
        _stderr(f"hearth chat - {Path(args.model).name} ({info.get('arch', '?')}, "
                f"{info.get('params_total', 0) / 1e9:.1f}B params). /help for commands, Ctrl-D or /exit to quit.")
        while True:
            try:
                line = input(">>> ")
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                cmd, _, rest = line.partition(" ")
                if cmd in ("/exit", "/quit", "/q"):
                    break
                if cmd == "/reset":
                    conv.reset()
                    _stderr("[conversation cleared]")
                elif cmd == "/system":
                    conv.reset(system=rest.strip() or None)
                    _stderr("[system prompt set; conversation cleared]")
                elif cmd == "/stats":
                    print(json.dumps(_engine_stats(eng), indent=1, default=str))
                elif cmd == "/help":
                    _stderr("/reset  clear history   /system TEXT  set system prompt   /stats  engine counters   "
                            "/exit  quit")
                else:
                    _stderr(f"unknown command {cmd}; /help lists them")
                continue
            _chat_turn(conv, line, args, sampler, cap)
    return 0


def cmd_serve(args) -> int:
    from hearth.server import serve
    check_generation_args(args)
    api_key = args.api_key or os.environ.get("HEARTH_API_KEY") or None
    if args.host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        _stderr(f"warning: listening on {args.host} without an API key (set --api-key or HEARTH_API_KEY)")
    if not args.timeout > 0:
        raise CLIError(f"--timeout must be > 0, got {args.timeout}")
    serve(args.model, host=args.host, port=args.port, model_name=args.model_name, api_key=api_key,
          max_queue=args.max_queue, speculative=args.speculative, draft_len=args.draft_len, ngram_n=args.ngram_n,
          cors=args.cors, verbose=args.log_requests, engine_kwargs=engine_kwargs(args), tokenizer_path=args.tokenizer,
          timeout=args.timeout)
    return 0


def bench_ids(kind: str, n: int, vocab: int, seed: int, offset: int = 0) -> list[int]:
    import numpy as np
    if kind == "random":
        return np.random.default_rng((seed, offset)).integers(0, vocab, size=n).tolist()
    return [((i + offset) * 7919 + 13) % vocab for i in range(n)]


def cmd_bench(args) -> int:
    import numpy as np

    from hearth.engine import Engine

    with Engine(args.model, **engine_kwargs(args)) as eng:
        info = eng.info
        V = int(info["vocab_size"])
        total = args.warmup + args.prompt_tokens + args.gen_tokens
        if total > eng.kv_capacity:
            raise CLIError(f"warmup + prompt + gen = {total} tokens exceeds the KV capacity {eng.kv_capacity}; "
                           f"raise --max-seq or lower the token counts")
        if args.route_replay:
            eng.route_replay(args.route_replay)
        if args.trace:
            eng.trace_start(args.trace)
        feed = "fixed" if args.ids == "greedy" else args.ids
        if args.warmup:
            eng.eval(bench_ids(feed, args.warmup, V, args.seed, offset=10_000_000))
        prompt = bench_ids(feed, args.prompt_tokens, V, args.seed)
        eng.reset_stats()
        t0 = time.perf_counter()
        logits = eng.eval(prompt) if prompt else None
        prefill_s = time.perf_counter() - t0
        sp = eng.stats()
        eng.reset_stats()
        dec = bench_ids(feed, args.gen_tokens, V, args.seed, offset=args.prompt_tokens)
        nxt = int(np.argmax(logits)) if (args.ids == "greedy" and logits is not None) else None
        t0 = time.perf_counter()
        for i in range(args.gen_tokens):
            tok = nxt if nxt is not None else dec[i]
            lg = eng.eval([tok])
            if args.ids == "greedy":
                nxt = int(np.argmax(lg))
        decode_s = time.perf_counter() - t0
        sd = eng.stats()
        if args.trace:
            eng.trace_stop()

    res = {
        "model": str(args.model), "arch": info.get("arch"), "params_total": info.get("params_total"),
        "params_active": info.get("params_active"), "isa": info.get("isa_name"), "threads": info.get("n_threads"),
        "io_threads": info.get("n_io_threads"), "cache_slots": info.get("cache_slots"),
        "options": engine_kwargs(args), "ids": args.ids, "route_replay": args.route_replay,
        "prefill_tokens": args.prompt_tokens, "prefill_s": prefill_s,
        "prefill_tok_s": args.prompt_tokens / prefill_s if prefill_s > 0 and args.prompt_tokens else 0.0,
        "decode_tokens": args.gen_tokens, "decode_s": decode_s,
        "decode_tok_s": args.gen_tokens / decode_s if decode_s > 0 and args.gen_tokens else 0.0,
        "prefill_stats": sp, "decode_stats": sd,
    }
    if args.json:
        print(json.dumps(res, indent=1, default=str))
        return 0
    label = " (routing replayed from trace: synthetic workload)" if args.route_replay else ""
    print(f"hearth bench: {Path(args.model).name}  arch={res['arch']}  isa={res['isa']}  threads={res['threads']}  "
          f"io={res['io_threads']}  cache_slots={res['cache_slots']}{label}")
    if args.prompt_tokens:
        print(f"  prefill  {args.prompt_tokens:6d} tok  {prefill_s:8.3f} s  {res['prefill_tok_s']:9.2f} tok/s  "
              f"hit {sp['hit_rate']:.1%}  read {sp['gb_read']:.2f} GB  stall {sp['stall_frac']:.1%}")
    if args.gen_tokens:
        print(f"  decode   {args.gen_tokens:6d} tok  {decode_s:8.3f} s  {res['decode_tok_s']:9.2f} tok/s  "
              f"hit {sd['hit_rate']:.1%}  read {sd['gb_read']:.2f} GB  stall {sd['stall_frac']:.1%}  "
              f"prefetch used {sd['prefetch_used']}/{sd['prefetch_issued']}")
        if sd["bytes_read"]:
            print(f"  decode   {sd['bytes_read'] / args.gen_tokens / 1e6:.1f} MB read per token, "
                  f"{sd['evictions']} evictions")
    return 0


def cmd_convert(args) -> int:
    from hearth.convert import convert
    dst = output_path(args.dst, args.force)
    out = convert(args.src, dst, expert_dtype=DTYPES[args.expert_dtype], dense_dtype=DTYPES[args.dense_dtype],
                  embed_dtype=DTYPES[args.embed_dtype], head_dtype=DTYPES[args.head_dtype], max_seq=args.max_seq,
                  threads=args.threads, progress=not args.quiet)
    print(out)
    return 0


def cmd_synth(args) -> int:
    from hearth import synth
    dst = output_path(args.path, args.force)
    if args.kind == "tiny":
        out = synth.make_tiny(dst, arch=args.arch, dtype=DTYPES[args.dtype], seed=args.seed, n_layers=args.n_layers,
                              d_model=args.d_model, n_experts=args.n_experts, top_k=args.top_k,
                              expert_ffn=args.expert_ffn, vocab=args.vocab, max_seq=args.max_seq)
    else:
        extra = {}
        if args.dense_dtype is not None:
            extra["dense_dtype"] = DTYPES[args.dense_dtype]
        if args.max_seq is not None:
            extra["max_seq"] = args.max_seq
        out = synth.make_shaped(dst, preset=args.preset, expert_dtype=DTYPES[args.expert_dtype],
                                physical_experts=args.physical_experts, seed=args.seed, **extra)
    print(out)
    return 0


def _cpu_features_py() -> dict:
    feats = {}
    if os.name == "nt":
        try:
            import ctypes
            ipf = ctypes.windll.kernel32.IsProcessorFeaturePresent
            feats = {"avx2": bool(ipf(40)), "avx512f": bool(ipf(41))}
        except Exception:
            pass
    elif Path("/proc/cpuinfo").exists():
        try:
            flags = set()
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].split())
                    break
            feats = {"avx2": "avx2" in flags, "avx512f": "avx512f" in flags}
        except OSError:
            pass
    return feats


def _ram() -> dict:
    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return {"total": m.ullTotalPhys, "available": m.ullAvailPhys}
        return {}
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                info["total" if k == "MemTotal" else "available"] = int(v.split()[0]) * 1024
        return info
    except OSError:
        pass
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return {"total": total}
    except (ValueError, OSError, AttributeError):
        return {}


def cmd_doctor(args) -> int:
    import importlib.metadata as md
    import importlib.util
    import platform
    import shutil

    report: dict = {"python": sys.version.split()[0], "platform": platform.platform(),
                    "machine": platform.machine(), "cpu_count": os.cpu_count()}
    deps = {}
    for name, required in (("numpy", True), ("torch", False), ("safetensors", False), ("transformers", False),
                           ("tokenizers", False), ("jinja2", False)):
        present = importlib.util.find_spec(name) is not None
        ver = None
        if present:
            try:
                ver = md.version(name)
            except md.PackageNotFoundError:
                ver = "?"
        deps[name] = {"present": present, "version": ver, "required": required}
    report["deps"] = deps

    lib = {"found": False}
    try:
        from hearth import _native
        cdll = _native.lib()
        lib = {"found": True, "path": getattr(cdll, "_name", None)}
        from hearth.engine import cpu_isa, isa_name, native_version
        try:
            lib["version"] = native_version()
            lib["cpu_isa"] = isa_name(cpu_isa())
        except Exception as e:
            lib["error"] = str(e)
    except ImportError as e:
        lib["error"] = f"hearth._native unavailable: {e}"
    except Exception as e:
        lib["error"] = str(e)
    report["native_lib"] = lib
    report["cpu_features"] = _cpu_features_py()
    report["ram"] = _ram()

    dd = data_dir()
    probe = dd
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        du = shutil.disk_usage(probe)
        disk = {"total": du.total, "free": du.free}
    except OSError:
        disk = {}
    report["data_dir"] = {"path": str(dd), "exists": dd.exists(), "cloud_synced": is_cloud_synced(dd), **disk}

    if args.json:
        print(json.dumps(report, indent=1, default=str))
        return 0 if deps["numpy"]["present"] else 1

    def line(status, what, detail=""):
        print(f"[{status:4s}] {what:18s} {detail}")

    line("ok", "python", f"{report['python']} on {report['platform']}")
    for name, d in deps.items():
        if d["present"]:
            line("ok", name, d["version"])
        else:
            line("FAIL" if d["required"] else "warn", name, "missing" + ("" if d["required"] else " (optional)"))
    if lib.get("found"):
        line("ok", "native library", f"{lib.get('path')}  version {lib.get('version', '?')}")
        if "cpu_isa" in lib:
            line("ok", "cpu isa (engine)", lib["cpu_isa"])
        if "error" in lib:
            line("warn", "native library", lib["error"])
    else:
        line("warn", "native library", f"not found ({lib.get('error', '')}); build it with "
                                        f"scripts/build.ps1 or 'python scripts/hxcc.py --shared'")
    cf = report["cpu_features"]
    if cf:
        line("ok", "cpu features", ", ".join(k for k, v in cf.items() if v) or "no AVX2/AVX-512")
    ram = report["ram"]
    if ram:
        line("ok", "ram", f"{_fmt_bytes(ram.get('total', 0))} total, {_fmt_bytes(ram.get('available', 0))} available"
             if "available" in ram else f"{_fmt_bytes(ram.get('total', 0))} total")
    d = report["data_dir"]
    status = "warn" if d["cloud_synced"] else "ok"
    free = f", {_fmt_bytes(d['free'])} free" if "free" in d else ""
    line(status, "data dir", f"{d['path']}{'' if d['exists'] else ' (not created yet)'}{free}"
         + ("  -- cloud-synced: models must not live here (INV-DATA)" if d["cloud_synced"] else ""))
    return 0 if deps["numpy"]["present"] else 1


# ---- parser --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    from hearth import __version__
    p = argparse.ArgumentParser(prog="hearth", description="Run huge Mixture-of-Experts models on an ordinary PC.")
    p.add_argument("--version", action="version", version=f"hearth {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="<command>")
    sub.required = True

    s = sub.add_parser("info", help="show a model container's metadata (and engine info with --open)")
    s.add_argument("model")
    s.add_argument("--open", action="store_true", help="open the engine and print hearth_model_info")
    s.add_argument("--json", action="store_true")
    add_engine_args(s)
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("run", help="generate a completion for one prompt")
    s.add_argument("model")
    src = s.add_mutually_exclusive_group()
    src.add_argument("-p", "--prompt", help="prompt text")
    src.add_argument("--prompt-file", help="read the prompt from a UTF-8 file")
    src.add_argument("--ids", help="prompt as token ids, e.g. '1,17,42'")
    s.add_argument("--chat", action="store_true", help="wrap the prompt in the chat template as a user message")
    s.add_argument("--system", help="system prompt (with --chat)")
    s.add_argument("--tokenizer", help="tokenizer.json to use instead of the container's")
    s.add_argument("--print-ids", action="store_true", help="print token ids instead of text")
    s.add_argument("--ignore-eos", action="store_true", help="do not stop at end-of-sequence tokens")
    s.add_argument("--json-out", metavar="PATH", help="write prompt/output ids and stats as JSON")
    s.add_argument("-q", "--quiet", action="store_true", help="no stats line")
    add_sampling_args(s, temperature=0.0, max_tokens=128)
    add_engine_args(s)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("chat", help="interactive chat (streaming)")
    s.add_argument("model")
    s.add_argument("--system", help="system prompt")
    s.add_argument("--tokenizer", help="tokenizer.json to use instead of the container's")
    s.add_argument("-q", "--quiet", action="store_true", help="no stats line after replies")
    add_sampling_args(s, temperature=0.7, max_tokens=1024)
    add_engine_args(s)
    s.set_defaults(fn=cmd_chat)

    s = sub.add_parser("serve", help="OpenAI/Anthropic-compatible HTTP server")
    s.add_argument("model")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--api-key", help="require this bearer key (default: $HEARTH_API_KEY, else none)")
    s.add_argument("--max-queue", type=int, default=8, help="requests allowed to wait; more get HTTP 429")
    s.add_argument("--model-name", help="model id reported by /v1/models (default: file stem)")
    s.add_argument("--cors", metavar="ORIGIN", help="send Access-Control-Allow-Origin: ORIGIN")
    s.add_argument("--tokenizer", help="tokenizer.json to use instead of the container's")
    s.add_argument("--timeout", type=float, default=60.0,
                   help="seconds a client socket may stall a read or write before it is dropped (default 60)")
    s.add_argument("--log-requests", action="store_true")
    add_spec_args(s.add_argument_group("speculative decoding"))
    add_engine_args(s)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("bench", help="measure prefill/decode speed, cache hit rate and I/O")
    s.add_argument("model")
    s.add_argument("--prompt-tokens", type=int, default=128)
    s.add_argument("--gen-tokens", type=int, default=64)
    s.add_argument("--ids", choices=["random", "fixed", "greedy"], default="random",
                   help="token ids fed: seeded random, a fixed sequence, or greedy argmax feedback")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--warmup", type=int, default=0, help="tokens evaluated before measuring")
    s.add_argument("--trace", metavar="OUT", help="record the routing trace (docs/FORMAT.md section 9)")
    s.add_argument("--route-replay", metavar="TRACE", help="replay routing from a trace (synthetic containers)")
    s.add_argument("--json", action="store_true")
    add_engine_args(s)
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("convert", help="convert a Hugging Face model directory to .hearth")
    s.add_argument("src")
    s.add_argument("dst", help="output path (a bare file name goes to <data dir>/models)")
    for name, default in (("expert", "q4"), ("dense", "q8"), ("embed", "q8"), ("head", "q8")):
        s.add_argument(f"--{name}-dtype", choices=list(DTYPES), default=default)
    s.add_argument("--max-seq", type=int, default=None)
    s.add_argument("--threads", type=int, default=0)
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--force", action="store_true", help="allow writing into a cloud-synced folder")
    s.set_defaults(fn=cmd_convert)

    s = sub.add_parser("synth", help="write a random-weight container (tests / benchmarks)")
    ss = s.add_subparsers(dest="kind", metavar="<kind>")
    ss.required = True
    t = ss.add_parser("tiny", help="small model exercising every code path of an architecture")
    t.add_argument("path")
    t.add_argument("--arch", default="qwen3_moe", choices=["qwen3_moe", "olmoe", "mixtral", "qwen2_moe", "deepseek_v3"])
    t.add_argument("--dtype", choices=list(DTYPES), default="f32")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--n-layers", type=int, default=3)
    t.add_argument("--d-model", type=int, default=128)
    t.add_argument("--n-experts", type=int, default=8)
    t.add_argument("--top-k", type=int, default=2)
    t.add_argument("--expert-ffn", type=int, default=64)
    t.add_argument("--vocab", type=int, default=256)
    t.add_argument("--max-seq", type=int, default=256)
    t.add_argument("--force", action="store_true", help="allow writing into a cloud-synced folder")
    t = ss.add_parser("shaped", help="a real model's shape with random weights (synthetic benchmark)")
    t.add_argument("path")
    t.add_argument("--preset", required=True, help="e.g. qwen3-30b-a3b, kimi-k2 (see hearth.presets)")
    t.add_argument("--expert-dtype", choices=list(DTYPES), default="q4")
    t.add_argument("--dense-dtype", choices=list(DTYPES), default=None, help="dense matrices (default: synth's)")
    t.add_argument("--max-seq", type=int, default=None)
    t.add_argument("--physical-experts", type=int, default=None,
                   help="store fewer distinct slabs and alias the rest (fits huge shapes on disk)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--force", action="store_true", help="allow writing into a cloud-synced folder")
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("sim", help="trace-driven cache / I/O simulator (arguments go to hearth.sim)",
                       add_help=False)
    s.add_argument("sim_args", nargs=argparse.REMAINDER)
    s.set_defaults(fn=None)

    s = sub.add_parser("doctor", help="check the environment: CPU, RAM, disk, native library, Python deps")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_doctor)
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _reconfigure_stdio()
    if argv and argv[0] == "sim":
        from hearth import sim
        rc = sim.main(argv[1:])
        return int(rc or 0)
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except CLIError as e:
        _stderr(f"hearth {args.cmd}: {e}")
        return 2
    except KeyboardInterrupt:
        _stderr("interrupted")
        return 130
    except Exception as e:
        from hearth.engine import HearthError
        lib_missing = type(e).__name__ == "HearthLibNotFound"
        # OSError: unreadable files, a port in use, an unresolvable --host
        if isinstance(e, (HearthError, OSError, ImportError, ValueError)) or lib_missing:
            _stderr(f"hearth {args.cmd}: {type(e).__name__}: {e}")
            return 1
        raise
