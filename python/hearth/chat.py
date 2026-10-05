"""Tokenizer, chat template and multi-turn conversations with KV-cache reuse.

Optional dependencies: `tokenizers` (Hugging Face tokenizer.json) and `jinja2`
(chat templates). Neither is needed to import this module.
"""
from __future__ import annotations

import json
import os
import struct
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence

from hearth.generate import GenStats, Sampler, Token, generate

__all__ = ["read_metadata", "find_tokenizer", "Tokenizer", "ChatTemplate", "StreamDecoder", "TextStream",
           "Conversation", "normalize_messages", "load_chat", "CHATML_TEMPLATE"]

_MAGIC = 0x48545248
_SCALARS = {1: ("<I", 4), 2: ("<f", 4), 3: ("<Q", 8)}
_ARRAYS = {5: ("<u4", 4), 6: ("<f4", 4), 7: ("u1", 1)}


def _parse_metadata(path) -> dict:
    """Minimal reader for the preamble + metadata section (docs/FORMAT.md §2-3)."""
    import numpy as np
    with open(path, "rb") as f:
        pre = f.read(64)
        if len(pre) < 64:
            raise ValueError(f"{path}: too short for a .hearth preamble")
        magic, version, meta_off, meta_bytes = struct.unpack_from("<IIQQ", pre, 0)
        if magic != _MAGIC:
            raise ValueError(f"{path}: not a .hearth file (bad magic)")
        if version != 1:
            raise ValueError(f"{path}: unsupported .hearth version {version}")
        size = os.fstat(f.fileno()).st_size
        if meta_off > size or meta_bytes > size - meta_off or meta_bytes > (1 << 30):
            raise ValueError(f"{path}: metadata section out of bounds")
        f.seek(meta_off)
        buf = f.read(meta_bytes)
    if len(buf) != meta_bytes:
        raise ValueError(f"{path}: truncated metadata")
    meta, p, end = {}, 0, len(buf)

    def need(n):
        if p + n > end:
            raise ValueError(f"{path}: malformed metadata entry at offset {p}")

    while p < end:
        need(2)
        (klen,) = struct.unpack_from("<H", buf, p)
        p += 2
        need(klen + 1)
        key = buf[p:p + klen].decode("ascii", "replace")
        p += klen
        typ = buf[p]
        p += 1
        if typ in _SCALARS:
            fmt, n = _SCALARS[typ]
            need(n)
            meta[key] = struct.unpack_from(fmt, buf, p)[0]
            p += n
        elif typ == 4 or typ in _ARRAYS:
            need(4)
            (count,) = struct.unpack_from("<I", buf, p)
            p += 4
            width = 1 if typ == 4 else _ARRAYS[typ][1]
            need(count * width)
            raw = buf[p:p + count * width]
            p += count * width
            if typ == 4:
                meta[key] = raw.decode("utf-8", "replace")
            else:
                meta[key] = np.frombuffer(raw, dtype=_ARRAYS[typ][0]).tolist()
        else:
            raise ValueError(f"{path}: unknown metadata type {typ} for key {key!r}")
    return meta


def read_metadata(path) -> dict:
    """Container metadata (raw keys, no defaults applied)."""
    try:
        from hearth.format import ContainerReader
    except ImportError:
        return _parse_metadata(path)
    r = ContainerReader(path)
    try:
        return dict(r.meta)
    finally:
        close = getattr(r, "close", None)
        if callable(close):
            close()


def find_tokenizer(model_path, meta: dict | None = None) -> Path | None:
    model_path = Path(model_path)
    cands = []
    rel = (meta or {}).get("tokenizer") or ""
    if rel:
        cands.append(model_path.parent / rel)
    cands += [model_path.with_name(model_path.stem + ".tokenizer.json"), model_path.parent / "tokenizer.json"]
    for c in cands:
        if c.is_file():
            return c
    return None


class Tokenizer:
    """Thin wrapper over a Hugging Face `tokenizers.Tokenizer`."""

    def __init__(self, tk, bos_id: int = -1, eos_ids: Sequence[int] = ()):
        self._tk = tk
        self.bos_id = int(bos_id) if bos_id is not None and int(bos_id) >= 0 else -1
        self.eos_ids = [int(e) for e in eos_ids]

    @classmethod
    def from_file(cls, path, bos_id: int = -1, eos_ids: Sequence[int] = ()) -> "Tokenizer":
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except ImportError as e:
            raise ImportError("chat needs the 'tokenizers' package: pip install tokenizers") from e
        return cls(HFTokenizer.from_file(os.fspath(path)), bos_id, eos_ids)

    @classmethod
    def from_container(cls, model_path, meta: dict | None = None, path=None) -> "Tokenizer":
        """The container's tokenizer (or `path`, overriding the search) with the
        container's BOS / EOS ids."""
        meta = read_metadata(model_path) if meta is None else meta
        path = find_tokenizer(model_path, meta) if path is None else path
        if path is None:
            raise FileNotFoundError(f"no tokenizer found for {model_path} (metadata 'tokenizer' = "
                                    f"{meta.get('tokenizer', '')!r})")
        bos = meta.get("bos_id", 0xFFFFFFFF)
        return cls.from_file(path, -1 if bos == 0xFFFFFFFF else bos, meta.get("eos_ids", []) or [])

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self._tk.encode(text, add_special_tokens=add_special_tokens).ids)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._tk.decode([int(i) for i in ids], skip_special_tokens=skip_special_tokens)

    def token_to_id(self, s: str) -> int | None:
        return self._tk.token_to_id(s)

    def id_to_token(self, i: int) -> str | None:
        return self._tk.id_to_token(int(i))

    @property
    def bos_token(self) -> str | None:
        return self.id_to_token(self.bos_id) if self.bos_id >= 0 else None

    @property
    def eos_token(self) -> str | None:
        return self.id_to_token(self.eos_ids[0]) if self.eos_ids else None


CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _chatml(messages, add_generation_prompt=True, **_):
    out = []
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
    return "".join(out)


class ChatTemplate:
    """Renders a Hugging Face Jinja chat template with the same environment
    conventions transformers uses (sandboxed, trim_blocks/lstrip_blocks,
    loopcontrols, tojson, raise_exception, strftime_now, {% generation %})."""

    def __init__(self, source: str | None = None, bos_token: str | None = None, eos_token: str | None = None):
        self.source = source or ""
        self.bos_token = bos_token
        self.eos_token = eos_token
        self._compiled = None
        if self.source:
            self._compiled = _compile(self.source)

    @property
    def is_fallback(self) -> bool:
        return not self.source

    @property
    def end_of_turn(self) -> str | None:
        """Marker that ends an assistant turn when the ChatML fallback is used."""
        return "<|im_end|>" if self.is_fallback else None

    @classmethod
    def from_metadata(cls, meta: dict, tokenizer: Tokenizer | None = None) -> "ChatTemplate":
        bos = tokenizer.bos_token if tokenizer is not None else None
        eos = tokenizer.eos_token if tokenizer is not None else None
        # Optional string keys (not in FORMAT.md §3.1 yet) win when a converter writes them.
        if isinstance(meta.get("bos_token"), str):
            bos = meta["bos_token"]
        if isinstance(meta.get("eos_token"), str):
            eos = meta["eos_token"]
        return cls(meta.get("chat_template") or None, bos, eos)

    def render(self, messages: Sequence[dict], add_generation_prompt: bool = True, **kwargs) -> str:
        msgs = normalize_messages(messages)
        if self._compiled is None:
            return _chatml(msgs, add_generation_prompt)
        ctx = dict(kwargs)
        if self.bos_token is not None:
            ctx.setdefault("bos_token", self.bos_token)
        if self.eos_token is not None:
            ctx.setdefault("eos_token", self.eos_token)
        ctx.setdefault("tools", None)
        ctx.setdefault("documents", None)
        return self._compiled.render(messages=msgs, add_generation_prompt=add_generation_prompt, **ctx)


def _compile(source: str):
    try:
        import jinja2
        from jinja2 import nodes
        from jinja2.ext import Extension
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as e:
        raise ImportError("this model's chat template needs 'jinja2': pip install jinja2") from e

    class _Generation(Extension):
        # {% generation %}...{% endgeneration %} only marks assistant spans for
        # training masks; for inference the body renders unchanged.
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            return nodes.CallBlock(self.call_method("_render", []), [], [], body).set_lineno(lineno)

        def _render(self, caller):
            return caller()

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt):
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[_Generation, jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env.from_string(source)


def normalize_messages(messages) -> list[dict]:
    """Accept OpenAI/Anthropic style messages; text content parts are joined.
    Raises ValueError on unsupported content (images, tool blocks)."""
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or "role" not in m:
            raise ValueError(f"message {i} must be an object with a 'role'")
        role = str(m["role"])
        if role == "developer":
            role = "system"
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                    parts.append(str(part.get("text", "")))
                else:
                    kind = part.get("type") if isinstance(part, dict) else type(part).__name__
                    raise ValueError(f"message {i}: unsupported content part {kind!r} (text only)")
            content = "".join(parts)
        elif content is None:
            content = ""
        elif not isinstance(content, str):
            raise ValueError(f"message {i}: content must be a string or a list of text parts")
        mm = {k: v for k, v in m.items() if k not in ("role", "content")}
        mm["role"] = role
        mm["content"] = content
        out.append(mm)
    return out


class StreamDecoder:
    """Incremental detokenisation: emits text only once it is stable, i.e. not
    ending in an incomplete UTF-8 sequence, and decodes with a little left
    context so tokenizers that strip leading spaces still produce them."""

    def __init__(self, tokenizer, skip_special_tokens: bool = True):
        self.tok = tokenizer
        self.skip = skip_special_tokens
        self.ids: list[int] = []
        self._prefix = 0
        self._read = 0

    def _dec(self, ids):
        return self.tok.decode(ids, skip_special_tokens=self.skip)

    def push(self, tid: int) -> str:
        self.ids.append(int(tid))
        prefix_text = self._dec(self.ids[self._prefix:self._read])
        new_text = self._dec(self.ids[self._prefix:])
        if len(new_text) > len(prefix_text) and not new_text.endswith("\ufffd"):
            delta = new_text[len(prefix_text):]
            self._prefix, self._read = self._read, len(self.ids)
            return delta
        return ""

    def flush(self) -> str:
        prefix_text = self._dec(self.ids[self._prefix:self._read])
        new_text = self._dec(self.ids[self._prefix:])
        self._prefix = self._read = len(self.ids)
        return new_text[len(prefix_text):] if len(new_text) > len(prefix_text) else ""


class TextStream:
    """Token ids -> text deltas with stop strings. Text that could be the start
    of a stop string is held back until it is disambiguated."""

    def __init__(self, tokenizer, stop: Sequence[str] = (), skip_special_tokens: bool = True):
        self.dec = StreamDecoder(tokenizer, skip_special_tokens)
        self.stops = [s for s in stop if s]
        self.hold = max((len(s) for s in self.stops), default=0)
        self.text = ""
        self.sent = 0
        self.stopped: str | None = None

    def _scan(self, final: bool) -> str:
        if self.stops:
            start = max(0, self.sent - self.hold)
            best = None
            for s in self.stops:
                i = self.text.find(s, start)
                if i >= 0 and (best is None or i < best[0]):
                    best = (i, s)
            if best is not None:
                i, s = best
                self.stopped = s
                out = self.text[self.sent:i]
                self.text = self.text[:i]
                self.sent = i
                return out
        limit = len(self.text) if final else max(self.sent, len(self.text) - self._partial())
        out = self.text[self.sent:limit]
        self.sent = limit
        return out

    def _partial(self) -> int:
        """Length of the longest unsent suffix that is a proper prefix of a stop string."""
        best = 0
        tail = self.text[self.sent:]
        for s in self.stops:
            for n in range(min(len(s) - 1, len(tail)), best, -1):
                if tail.endswith(s[:n]):
                    best = n
                    break
        return best

    def push(self, tid: int) -> str:
        if self.stopped is not None:
            return ""
        self.text += self.dec.push(tid)
        return self._scan(False)

    def finish(self) -> str:
        if self.stopped is not None:
            return ""
        self.text += self.dec.flush()
        return self._scan(True)


class Conversation:
    """Keeps one engine's KV cache across turns and requests: each generation
    evaluates only the suffix of the prompt the engine does not already hold.

    `held` lists the tokens whose KV the engine holds (len(held) == engine.pos).
    The engine must not be used by anything else between calls."""

    def __init__(self, engine, tokenizer=None, template: ChatTemplate | None = None, system: str | None = None,
                 stop_ids: Sequence[int] | None = None):
        self.engine = engine
        self.tokenizer = tokenizer
        self.template = template if template is not None else ChatTemplate()
        self.system = system
        self.messages: list[dict] = []
        self.held: list[int] = []
        self.last_stats: GenStats | None = None
        if stop_ids is None:
            stop_ids = default_stop_ids(engine, tokenizer, self.template)
        self.stop_ids = list(stop_ids)
        self._synced = False

    def reset(self, system: str | None = ...) -> None:
        if system is not ...:
            self.system = system
        self.messages = []

    def forget_cache(self) -> None:
        self.held = []
        self._synced = False

    def generate_ids(self, prompt_ids: Sequence[int], max_new_tokens: int = 256, sampler: Sampler | None = None,
                     stop_ids: Sequence[int] | None = None, speculative: str = "none", draft_len: int = 4,
                     ngram_n: int = 3) -> Iterator[Token]:
        prompt = [int(t) for t in prompt_ids]
        held = self.held if self._synced else None
        self.last_stats = None
        gen = generate(self.engine, prompt, max_new_tokens, sampler,
                       self.stop_ids if stop_ids is None else stop_ids, speculative, draft_len, ngram_n, held=held)
        seq = list(prompt)
        # generate() guarantees the engine holds a prefix of `seq` only once it
        # has yielded or returned; an exception or interrupt (e.g. Ctrl-C during
        # prefill) before that leaves the KV content unknown.
        synced = failed = False
        try:
            while True:
                try:
                    t = next(gen)
                except StopIteration as e:
                    synced = True
                    self.last_stats = e.value
                    break
                synced = True
                seq.append(t.id)
                self.last_stats = t.stats
                yield t
        except Exception:
            failed = True  # engine state after a failed call is unspecified
            raise
        finally:
            gen.close()
            self._after(seq, synced and not failed)

    def _after(self, seq: list[int], known: bool) -> None:
        if known:
            try:
                pos = self.engine.pos
                if pos <= len(seq):
                    self.held = seq[:pos]
                    self._synced = True
                    return
            except Exception:
                pass
        self.forget_cache()
        try:
            self.engine.reset()
        except Exception:
            pass

    def prompt_ids(self, messages: Sequence[dict] | None = None, add_generation_prompt: bool = True) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("Conversation has no tokenizer")
        msgs = list(self.messages if messages is None else messages)
        if self.system and not (msgs and msgs[0].get("role") == "system"):
            msgs = [{"role": "system", "content": self.system}] + msgs
        text = self.template.render(msgs, add_generation_prompt=add_generation_prompt)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def say(self, text: str, max_new_tokens: int = 512, sampler: Sampler | None = None,
            speculative: str = "none", draft_len: int = 4, ngram_n: int = 3,
            stop: Sequence[str] = ()) -> Iterator[str]:
        """Add a user message, stream the assistant reply as text deltas and
        record it in `messages`. A reply cut short (the consumer closed the
        stream, Ctrl-C) is recorded as far as it got; if generation raises
        (e.g. the prompt no longer fits the KV cache) the turn is rolled back,
        so `messages` is as it was before the call."""
        n_before = len(self.messages)
        self.messages.append({"role": "user", "content": text})
        reply, gen, failed = [], None, False
        try:
            ids = self.prompt_ids()
            stops = list(stop)
            if self.template.end_of_turn:
                stops.append(self.template.end_of_turn)
            ts = TextStream(self.tokenizer, stops)
            gen = self.generate_ids(ids, max_new_tokens, sampler, None, speculative, draft_len, ngram_n)
            for t in gen:
                d = ts.push(t.id)
                if d:
                    reply.append(d)
                    yield d
                if ts.stopped is not None:
                    break
            d = ts.finish()
            if d:
                reply.append(d)
                yield d
        except Exception:
            failed = True
            raise
        finally:
            if gen is not None:
                gen.close()
            if failed:
                del self.messages[n_before:]
            else:
                self.messages.append({"role": "assistant", "content": "".join(reply)})


def default_stop_ids(engine, tokenizer=None, template: ChatTemplate | None = None) -> list[int]:
    info = getattr(engine, "info", {}) or {}
    ids = [int(e) for e in info.get("eos_ids", []) or []]
    if tokenizer is not None:
        ids += [int(e) for e in getattr(tokenizer, "eos_ids", []) or []]
        if template is not None and template.end_of_turn:
            tid = tokenizer.token_to_id(template.end_of_turn) if hasattr(tokenizer, "token_to_id") else None
            if tid is not None:
                ids.append(int(tid))
    return sorted(set(ids))


def load_chat(model_path, meta: dict | None = None) -> tuple[Tokenizer, ChatTemplate, dict]:
    """Tokenizer and chat template for a container (no engine needed)."""
    meta = read_metadata(model_path) if meta is None else meta
    tok = Tokenizer.from_container(model_path, meta)
    return tok, ChatTemplate.from_metadata(meta, tok), meta
