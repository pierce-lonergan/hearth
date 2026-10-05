"""OpenAI- and Anthropic-compatible HTTP server (standard library only).

Endpoints: GET /health, GET /v1/models, GET /stats, POST /v1/chat/completions,
POST /v1/completions, POST /v1/messages. One engine serves one request at a
time; further requests wait in a bounded FIFO queue and get HTTP 429 when it is
full. A client socket that stalls a read or write for `timeout` seconds is
dropped, so a client that stops reading a stream cannot hold the engine.
Non-streaming replies are written after the engine is released. Consecutive
requests share the KV cache through a Conversation, so a follow-up turn only
evaluates the new suffix of its prompt. Without a tokenizer only
/v1/completions with a token-id prompt is served; its choices then carry the
output as a non-standard "token_ids" list (text is "").
"""
from __future__ import annotations

import hmac
import json
import os
import socket
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator, Sequence

from hearth.chat import ChatTemplate, Conversation, TextStream, default_stop_ids, normalize_messages
from hearth.generate import GenStats, Sampler, check_speculative, kv_capacity

__all__ = ["App", "EngineGate", "Busy", "HearthServer", "make_server", "serve"]

MAX_BODY = 32 << 20
DRAIN_UNAUTH = 1 << 20  # largest unwanted body read (not refused by closing) before an error reply


class Busy(Exception):
    pass


class ClientGone(Exception):
    """Writing to the client failed (disconnect, reset, or a write that stalled past the timeout)."""


class HTTPError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error", param: str | None = None):
        super().__init__(message)
        self.status, self.message, self.kind, self.param = status, message, kind, param


class EngineGate:
    """Mutual exclusion with FIFO hand-off and a bounded number of waiters."""

    def __init__(self, max_queue: int = 8):
        self.max_queue = max(0, int(max_queue))
        self._cv = threading.Condition()
        self._active = False
        self._queue: deque = deque()

    @property
    def waiting(self) -> int:
        with self._cv:
            return len(self._queue)

    @property
    def busy(self) -> bool:
        with self._cv:
            return self._active

    def acquire(self) -> None:
        with self._cv:
            if not self._active and not self._queue:
                self._active = True
                return
            if len(self._queue) >= self.max_queue:
                raise Busy()
            ticket = object()
            self._queue.append(ticket)
            try:
                while self._active or self._queue[0] is not ticket:
                    self._cv.wait()
            except BaseException:
                self._queue.remove(ticket)
                self._cv.notify_all()
                raise
            self._queue.popleft()
            self._active = True

    def release(self) -> None:
        with self._cv:
            self._active = False
            self._cv.notify_all()


class Run:
    """One generation: iterate for text deltas; outcome fields are set at the end.
    Without a tokenizer every generated token yields "" and only `token_ids`
    carries the output."""

    def __init__(self, app: "App", prompt_ids: list[int], max_tokens: int, sampler: Sampler, stop: Sequence[str],
                 chat: bool):
        self.app = app
        self.chat = chat
        self.prompt_ids = prompt_ids
        self.max_tokens = max_tokens
        self.sampler = sampler
        self.user_stops = [s for s in stop if s]
        self.finish = "length"          # "eos" | "stop_sequence" | "length"
        self.stop_sequence: str | None = None
        self.completion_tokens = 0
        self.token_ids: list[int] = []  # generated ids (end-of-sequence token excluded)
        self.stats: GenStats | None = None

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_ids)

    @property
    def cached_tokens(self) -> int:
        return self.stats.reused_tokens if self.stats else 0

    def __iter__(self) -> Iterator[str]:
        app = self.app
        eot = app.conv.template.end_of_turn if self.chat else None
        stops = self.user_stops + ([eot] if eot and eot not in self.user_stops else [])
        ts = TextStream(app.tokenizer, stops) if app.tokenizer is not None else None
        app.conv.last_stats = None
        gen = app.conv.generate_ids(self.prompt_ids, self.max_tokens, self.sampler, None,
                                    app.speculative, app.draft_len, app.ngram_n)
        try:
            for t in gen:
                self.stats = t.stats
                self.completion_tokens += 1
                self.token_ids.append(t.id)
                if ts is None:
                    yield ""
                    continue
                d = ts.push(t.id)
                if d:
                    yield d
                if ts.stopped is not None:
                    break
            if ts is not None and ts.stopped is None:
                d = ts.finish()
                if d:
                    yield d
        finally:
            gen.close()
            if app.conv.last_stats is not None:
                self.stats = app.conv.last_stats
            st = self.stats
            if ts is not None and ts.stopped is not None:
                if ts.stopped in self.user_stops:
                    self.finish, self.stop_sequence = "stop_sequence", ts.stopped
                else:
                    self.finish = "eos"
            elif st is not None and st.finish_reason == "stop":
                self.finish = "eos"
                self.completion_tokens += 1  # the end-of-sequence token was generated too
            app.record(self)


class App:
    def __init__(self, engine, tokenizer, template: ChatTemplate | None = None, *, model_name: str = "hearth",
                 api_key: str | None = None, max_queue: int = 8, stop_ids: Sequence[int] | None = None,
                 speculative: str = "none", draft_len: int = 4, ngram_n: int = 3, cors: str | None = None,
                 verbose: bool = False, timeout: float | None = 60.0):
        # A bad speculative configuration must fail here, not on every request.
        speculative = check_speculative(speculative, draft_len, ngram_n)
        if timeout is not None and not timeout > 0:
            raise ValueError(f"timeout must be > 0 seconds or None, got {timeout!r}")
        self.timeout = timeout
        self.engine = engine
        self.tokenizer = tokenizer
        template = template if template is not None else ChatTemplate()
        if stop_ids is None:
            stop_ids = default_stop_ids(engine, tokenizer, template)
        self.conv = Conversation(engine, tokenizer, template, stop_ids=stop_ids)
        self.model_name = model_name
        self.api_key = api_key or None
        self.gate = EngineGate(max_queue)
        self.speculative, self.draft_len, self.ngram_n = speculative, int(draft_len), int(ngram_n)
        self.cors = cors
        self.verbose = verbose
        self.created = int(time.time())
        self._lock = threading.Lock()
        self.counters = dict(requests=0, completed=0, rejected_busy=0, errors=0, prompt_tokens=0,
                             completion_tokens=0, cached_tokens=0)
        self.last_generation: dict | None = None

    @property
    def capacity(self) -> int:
        return kv_capacity(self.engine)

    def count(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.counters[key] += n

    def record(self, run: Run) -> None:
        with self._lock:
            c = self.counters
            c["completed"] += 1
            c["prompt_tokens"] += run.prompt_tokens
            c["completion_tokens"] += run.completion_tokens
            c["cached_tokens"] += run.cached_tokens
            if run.stats is not None:
                self.last_generation = run.stats.as_dict()

    def check_auth(self, headers) -> None:
        if not self.api_key:
            return
        given = headers.get("x-api-key")
        auth = headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            given = auth[7:].strip()
        if not given or not hmac.compare_digest(given.encode(), self.api_key.encode()):
            raise HTTPError(401, "invalid or missing API key", "authentication_error")

    # ---- request -> generation parameters ---------------------------------
    def prepare(self, prompt_ids: list[int], max_tokens) -> int:
        cap = self.capacity
        if not prompt_ids:
            raise HTTPError(400, "prompt is empty")
        if len(prompt_ids) >= cap:
            raise HTTPError(400, f"prompt has {len(prompt_ids)} tokens; the context holds {cap} "
                                 f"(at least one more is needed to generate)", param="messages")
        room = cap - len(prompt_ids)
        if max_tokens is None:
            return room
        return min(int(max_tokens), room)

    def run(self, prompt_ids, max_tokens, sampler, stop, chat: bool) -> Run:
        return Run(self, prompt_ids, max_tokens, sampler, stop, chat)

    def stats(self) -> dict:
        out = {"server": dict(self.counters, queue_waiting=self.gate.waiting, busy=self.gate.busy,
                              max_queue=self.gate.max_queue, held_tokens=len(self.conv.held)),
               "last_generation": self.last_generation}
        fn = getattr(self.engine, "stats", None)
        if callable(fn):
            try:
                out["engine"] = fn()
            except Exception as e:  # closed engine etc.
                out["engine_error"] = str(e)
        return out


# ---- parameter parsing --------------------------------------------------------

def _num(body: dict, key: str, default, lo=None, hi=None, integer=False):
    v = body.get(key, default)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise HTTPError(400, f"'{key}' must be a number", param=key)
    if integer:
        if isinstance(v, float) and not v.is_integer():
            raise HTTPError(400, f"'{key}' must be an integer", param=key)
        v = int(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        want = f">= {lo}" if hi is None else f"in [{lo}, {hi}]"
        raise HTTPError(400, f"'{key}' must be {want}, got {v}", param=key)
    return v


def _bool(body: dict, key: str, default: bool = False) -> bool:
    v = body.get(key)
    if v is None:
        return default
    if not isinstance(v, bool):  # "false" is truthy: never guess
        raise HTTPError(400, f"'{key}' must be a boolean", param=key)
    return v


def _stops(v, key="stop") -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list) and all(isinstance(s, str) for s in v):
        if len(v) > 16:
            raise HTTPError(400, f"at most 16 '{key}' strings", param=key)
        return list(v)
    raise HTTPError(400, f"'{key}' must be a string or a list of strings", param=key)


def _sampler(body: dict) -> Sampler:
    temperature = float(_num(body, "temperature", 1.0, 0.0, 100.0))
    top_p = float(_num(body, "top_p", 1.0, 0.0, 1.0))
    if top_p == 0.0:  # an empty nucleus means "most likely token only"
        temperature, top_p = 0.0, 1.0
    try:
        return Sampler(temperature=temperature, top_p=top_p,
                       top_k=int(_num(body, "top_k", 0, 0, None, integer=True)),
                       min_p=float(_num(body, "min_p", 0.0, 0.0, 1.0)),
                       repetition_penalty=float(_num(body, "repetition_penalty", 1.0, 1e-3, 100.0)),
                       seed=_num(body, "seed", None, 0, None, integer=True))
    except ValueError as e:
        raise HTTPError(400, str(e)) from e


def _new_id(prefix: str) -> str:
    return prefix + uuid.uuid4().hex[:24]


def _sse(data, event: str | None = None) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    head = f"event: {event}\n" if event else ""
    return (head + f"data: {payload}\n\n").encode("utf-8")


# ---- HTTP -------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "hearth"
    sys_version = ""

    @property
    def app(self) -> App:
        return self.server.app

    def setup(self):
        self.timeout = self.server.app.timeout  # StreamRequestHandler applies it to the socket
        super().setup()

    def log_message(self, fmt, *args):
        if self.app.verbose:
            sys.stderr.write("[hearth.server] %s - %s\n" % (self.address_string(), fmt % args))

    # ---- plumbing --------------------------------------------------------------
    def _cors(self):
        if self.app.cors:
            self.send_header("Access-Control-Allow-Origin", self.app.cors)
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, x-api-key, "
                                                             "anthropic-version")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _write(self, data: bytes, flush: bool = False):
        try:
            self.wfile.write(data)
            if flush:
                self.wfile.flush()
        except OSError as e:  # reset, broken pipe, or the socket timeout expired
            self.close_connection = True
            raise ClientGone(str(e)) from e

    def _send_json(self, status: int, obj, headers: dict | None = None):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self._cors()
        try:
            self.end_headers()
        except OSError as e:
            self.close_connection = True
            raise ClientGone(str(e)) from e
        self._write(data)

    def _error(self, err: HTTPError, anthropic: bool):
        headers = {"Retry-After": "1"} if err.status == 429 else None
        if anthropic:
            kinds = {401: "authentication_error", 404: "not_found_error", 413: "request_too_large",
                     429: "rate_limit_error", 500: "api_error"}
            body = {"type": "error", "error": {"type": kinds.get(err.status, err.kind), "message": err.message}}
        else:
            kinds = {429: "rate_limit_exceeded", 500: "server_error"}
            body = {"error": {"message": err.message, "type": kinds.get(err.status, err.kind),
                              "param": err.param, "code": None}}
        self._send_json(err.status, body, headers)

    def _body(self) -> dict:
        self._body_done = True  # read below, or the connection is closed
        n = self.headers.get("Content-Length")
        if n is None:
            self.close_connection = True
            raise HTTPError(411, "Content-Length required")
        try:
            n = int(n)
        except ValueError:
            n = -1
        if n < 0:
            self.close_connection = True
            raise HTTPError(400, "bad Content-Length")
        if n > MAX_BODY:
            self.close_connection = True
            raise HTTPError(413, f"request body larger than {MAX_BODY} bytes", "request_too_large")
        raw = self.rfile.read(n)
        try:
            body = json.loads(raw.decode("utf-8") if raw else "null")
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise HTTPError(400, f"invalid JSON body: {e}") from None
        if not isinstance(body, dict):
            raise HTTPError(400, "request body must be a JSON object")
        return body

    def _start_stream(self):
        # HTTP/1.0 has no chunked encoding (RFC 9112 §6.1): the stream ends when the connection closes.
        self._chunked = self.request_version != "HTTP/1.0"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        if self._chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Connection", "close")
            self.close_connection = True
        self._cors()
        try:
            self.end_headers()
        except OSError as e:
            self.close_connection = True
            raise ClientGone(str(e)) from e

    def _chunk(self, data: bytes):
        if data:
            self._write(b"%x\r\n%s\r\n" % (len(data), data) if self._chunked else data, flush=True)

    def _end_stream(self):
        if self._chunked:
            self._write(b"0\r\n\r\n", flush=True)

    # ---- routing ---------------------------------------------------------------
    def _drain(self, limit: int = MAX_BODY) -> None:
        """Consume a request body that was not read, so the next request on a
        keep-alive connection starts at the right byte; close the connection
        when the body is too large or its length unknown."""
        if self._body_done:
            return
        self._body_done = True
        n = self.headers.get("Content-Length")
        if n is None:
            if self.headers.get("Transfer-Encoding"):
                self.close_connection = True
            return
        try:
            n = int(n)
        except ValueError:
            n = -1
        if 0 < n <= limit:
            self.rfile.read(n)
        elif n != 0:
            self.close_connection = True

    def do_OPTIONS(self):
        self._body_done = False
        self._drain(DRAIN_UNAUTH)
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        try:
            self._get()
        except ClientGone:
            self.close_connection = True

    def do_POST(self):
        try:
            self._post()
        except ClientGone:
            self.close_connection = True

    def _get(self):
        self._body_done = False
        self._drain(DRAIN_UNAUTH)  # GET bodies carry nothing, but must not be left on the connection
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        anthropic = False
        try:
            if path == "/health":
                self._send_json(200, {"status": "ok", "model": self.app.model_name,
                                      "busy": self.app.gate.busy, "queue_waiting": self.app.gate.waiting})
                return
            self.app.check_auth(self.headers)
            if path in ("/v1/models", "/models"):
                self._send_json(200, {"object": "list", "data": [self._model_card()]})
            elif path.startswith("/v1/models/"):
                if path[len("/v1/models/"):] != self.app.model_name:
                    raise HTTPError(404, "model not found", "not_found_error")
                self._send_json(200, self._model_card())
            elif path == "/stats":
                self._send_json(200, self.app.stats())
            else:
                raise HTTPError(404, f"no route for GET {path}", "not_found_error")
        except HTTPError as e:
            self._error(e, anthropic)

    def _post(self):
        self._body_done = False
        path = self.path.split("?", 1)[0].rstrip("/")
        anthropic = path in ("/v1/messages", "/messages")
        try:
            try:
                self.app.check_auth(self.headers)
            except HTTPError:
                self._drain(DRAIN_UNAUTH)
                raise
            routes = {"/v1/chat/completions": self._chat, "/chat/completions": self._chat,
                      "/v1/completions": self._completions, "/completions": self._completions,
                      "/v1/messages": self._messages, "/messages": self._messages}
            fn = routes.get(path)
            if fn is None:
                raise HTTPError(404, f"no route for POST {path}", "not_found_error")
            body = self._body()
            self.app.count("requests")
            fn(body)
        except HTTPError as e:
            if e.status >= 500:
                self.app.count("errors")
            self._drain()
            self._error(e, anthropic)

    def _model_card(self) -> dict:
        return {"id": self.app.model_name, "object": "model", "created": self.app.created, "owned_by": "hearth"}

    # ---- generation driver -----------------------------------------------------
    def _locked(self, fn):
        """Run fn while holding the engine. A dict it returns is the JSON reply,
        sent after the engine is released so a slow reader cannot hold it."""
        app = self.app
        try:
            app.gate.acquire()
        except Busy:
            app.count("rejected_busy")
            raise HTTPError(429, f"server busy: {app.gate.max_queue} requests already queued",
                            "rate_limit_error") from None
        try:
            reply = fn()
        finally:
            app.gate.release()
        if reply is not None:
            self._send_json(200, reply)

    def _encode_chat(self, messages) -> list[int]:
        app = self.app
        if app.tokenizer is None:
            raise HTTPError(400, "this server has no tokenizer; only token-id completions are available")
        try:
            msgs = normalize_messages(messages)
            text = app.conv.template.render(msgs, add_generation_prompt=True)
        except ValueError as e:
            raise HTTPError(400, str(e), param="messages") from None
        except Exception as e:  # template errors (raise_exception, undefined access)
            raise HTTPError(400, f"chat template failed: {e}", param="messages") from None
        return app.tokenizer.encode(text, add_special_tokens=False)

    def _stream_guard(self, gen_fn, on_error):
        """Run a streaming generation; a client that disconnects or stops
        reading ends it (the generator is closed while the engine is held)."""
        try:
            gen_fn()
        except ClientGone:
            self.close_connection = True
        except Exception as e:  # engine failure after headers were sent
            self.app.count("errors")
            try:
                on_error(str(e))
                self._end_stream()
            except ClientGone:
                pass
            self.close_connection = True

    # ---- OpenAI chat -----------------------------------------------------------
    def _chat(self, body: dict):
        app = self.app
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise HTTPError(400, "'messages' must be a non-empty list", param="messages")
        if _num(body, "n", 1, 1, None, integer=True) != 1:
            raise HTTPError(400, "only n=1 is supported", param="n")
        sampler = _sampler(body)
        stop = _stops(body.get("stop"))
        max_tokens = _num(body, "max_completion_tokens", None, 1, None, integer=True)
        if max_tokens is None:
            max_tokens = _num(body, "max_tokens", None, 1, None, integer=True)
        ids = self._encode_chat(msgs)
        max_tokens = app.prepare(ids, max_tokens)
        stream = _bool(body, "stream")
        opts = body.get("stream_options")
        if opts is not None and not isinstance(opts, dict):
            raise HTTPError(400, "'stream_options' must be an object", param="stream_options")
        include_usage = _bool(opts or {}, "include_usage")
        rid, created, model = _new_id("chatcmpl-"), int(time.time()), app.model_name

        def usage(run: Run):
            return {"prompt_tokens": run.prompt_tokens, "completion_tokens": run.completion_tokens,
                    "total_tokens": run.prompt_tokens + run.completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": run.cached_tokens}}

        def finish(run: Run):
            return "length" if run.finish == "length" else "stop"

        def chunk(delta, fin=None):
            return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": fin}]}

        def work():
            run = app.run(ids, max_tokens, sampler, stop, chat=True)
            if not stream:
                try:
                    text = "".join(run)
                except Exception as e:
                    app.conv.forget_cache()
                    raise HTTPError(500, f"generation failed: {e}", "server_error") from None
                return {"id": rid, "object": "chat.completion", "created": created, "model": model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                     "logprobs": None, "finish_reason": finish(run)}],
                        "usage": usage(run)}

            def go():
                self._start_stream()
                self._chunk(_sse(chunk({"role": "assistant", "content": ""})))
                with closing(iter(run)) as it:  # cleanup must happen while the gate is held
                    for d in it:
                        self._chunk(_sse(chunk({"content": d})))
                self._chunk(_sse(chunk({}, finish(run))))
                if include_usage:
                    u = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [], "usage": usage(run)}
                    self._chunk(_sse(u))
                self._chunk(_sse("[DONE]"))
                self._end_stream()

            self._stream_guard(go, lambda msg: self._chunk(
                _sse({"error": {"message": msg, "type": "server_error", "param": None, "code": None}})))

        self._locked(work)

    # ---- OpenAI legacy completions -------------------------------------------------
    def _completions(self, body: dict):
        app = self.app
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], (str, list)):
            prompt = prompt[0]
        if isinstance(prompt, str):
            if app.tokenizer is None:
                raise HTTPError(400, "this server has no tokenizer; send token ids as 'prompt'", param="prompt")
            ids = app.tokenizer.encode(prompt, add_special_tokens=True)
            prompt_text = prompt
        elif isinstance(prompt, list) and prompt and all(isinstance(t, int) and not isinstance(t, bool)
                                                         for t in prompt):
            vocab = int((getattr(app.engine, "info", {}) or {}).get("vocab_size", 0)) or None
            if vocab is not None and any(t < 0 or t >= vocab for t in prompt):
                raise HTTPError(400, f"token id out of range [0, {vocab})", param="prompt")
            ids = list(prompt)
            prompt_text = app.tokenizer.decode(ids) if app.tokenizer is not None else ""
        else:
            raise HTTPError(400, "'prompt' must be a string or a list of token ids (batches are not supported)",
                            param="prompt")
        if _num(body, "n", 1, 1, None, integer=True) != 1:
            raise HTTPError(400, "only n=1 is supported", param="n")
        sampler = _sampler(body)
        stop = _stops(body.get("stop"))
        max_tokens = app.prepare(ids, _num(body, "max_tokens", 16, 0, None, integer=True))
        echo = _bool(body, "echo")
        stream = _bool(body, "stream")
        # Without a tokenizer the output is token ids only (non-standard "token_ids" field, text "").
        tokenless = app.tokenizer is None
        if tokenless and stop:
            raise HTTPError(400, "'stop' strings need a tokenizer; this server has none", param="stop")
        if tokenless and echo:
            raise HTTPError(400, "'echo' needs a tokenizer; this server has none", param="echo")
        rid, created, model = _new_id("cmpl-"), int(time.time()), app.model_name

        def choice(text, fin, token_ids=None):
            c = {"index": 0, "text": text, "logprobs": None, "finish_reason": fin}
            if tokenless:
                c["token_ids"] = token_ids or []
            return {"id": rid, "object": "text_completion", "created": created, "model": model, "choices": [c]}

        def work():
            run = app.run(ids, max_tokens, sampler, stop, chat=False)
            if not stream:
                try:
                    text = "".join(run)
                except Exception as e:
                    app.conv.forget_cache()
                    raise HTTPError(500, f"generation failed: {e}", "server_error") from None
                out = choice((prompt_text if echo else "") + text, "length" if run.finish == "length" else "stop",
                             run.token_ids)
                out["usage"] = {"prompt_tokens": run.prompt_tokens, "completion_tokens": run.completion_tokens,
                                "total_tokens": run.prompt_tokens + run.completion_tokens}
                return out

            def go():
                self._start_stream()
                if echo and prompt_text:
                    self._chunk(_sse(choice(prompt_text, None)))
                sent = 0
                with closing(iter(run)) as it:
                    for d in it:
                        if tokenless:
                            new, sent = run.token_ids[sent:], len(run.token_ids)
                            self._chunk(_sse(choice("", None, new)))
                        else:
                            self._chunk(_sse(choice(d, None)))
                self._chunk(_sse(choice("", "length" if run.finish == "length" else "stop")))
                self._chunk(_sse("[DONE]"))
                self._end_stream()

            self._stream_guard(go, lambda msg: self._chunk(
                _sse({"error": {"message": msg, "type": "server_error", "param": None, "code": None}})))

        self._locked(work)

    # ---- Anthropic Messages ----------------------------------------------------------
    def _messages(self, body: dict):
        app = self.app
        if body.get("max_tokens") is None:
            raise HTTPError(400, "max_tokens: field required")
        max_tokens = _num(body, "max_tokens", None, 1, None, integer=True)
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise HTTPError(400, "messages: must be a non-empty list")
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
                raise HTTPError(400, "messages: each role must be 'user' or 'assistant'")
        system = body.get("system")
        full = []
        if system:
            if isinstance(system, list):
                system = "".join(b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text")
            elif not isinstance(system, str):
                raise HTTPError(400, "system: must be a string or a list of text blocks")
            full.append({"role": "system", "content": system})
        full += msgs
        sampler = _sampler(body)
        stop = _stops(body.get("stop_sequences"), "stop_sequences")
        if msgs[-1]["role"] == "assistant":
            # A trailing assistant message is a prefill: continue it rather than open a new turn.
            if app.tokenizer is None:
                raise HTTPError(400, "this server has no tokenizer")
            try:
                last = normalize_messages([msgs[-1]])[0]["content"]
                text = app.conv.template.render(normalize_messages(full[:-1]), add_generation_prompt=True)
            except ValueError as e:
                raise HTTPError(400, str(e), param="messages") from None
            except Exception as e:
                raise HTTPError(400, f"chat template failed: {e}", param="messages") from None
            ids = app.tokenizer.encode(text + last, add_special_tokens=False)
        else:
            ids = self._encode_chat(full)
        max_tokens = app.prepare(ids, max_tokens)
        stream = _bool(body, "stream")
        mid, model = _new_id("msg_"), app.model_name
        reasons = {"eos": "end_turn", "stop_sequence": "stop_sequence", "length": "max_tokens"}

        def work():
            run = app.run(ids, max_tokens, sampler, stop, chat=True)
            if not stream:
                try:
                    text = "".join(run)
                except Exception as e:
                    app.conv.forget_cache()
                    raise HTTPError(500, f"generation failed: {e}", "api_error") from None
                return {"id": mid, "type": "message", "role": "assistant", "model": model,
                        "content": [{"type": "text", "text": text}],
                        "stop_reason": reasons[run.finish], "stop_sequence": run.stop_sequence,
                        "usage": {"input_tokens": run.prompt_tokens, "output_tokens": run.completion_tokens,
                                  "cache_read_input_tokens": run.cached_tokens}}

            def go():
                self._start_stream()
                self._chunk(_sse({"type": "message_start", "message": {
                    "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": run.prompt_tokens, "output_tokens": 0}}}, "message_start"))
                self._chunk(_sse({"type": "content_block_start", "index": 0,
                                  "content_block": {"type": "text", "text": ""}}, "content_block_start"))
                self._chunk(_sse({"type": "ping"}, "ping"))
                with closing(iter(run)) as it:
                    for d in it:
                        self._chunk(_sse({"type": "content_block_delta", "index": 0,
                                          "delta": {"type": "text_delta", "text": d}}, "content_block_delta"))
                self._chunk(_sse({"type": "content_block_stop", "index": 0}, "content_block_stop"))
                self._chunk(_sse({"type": "message_delta",
                                  "delta": {"stop_reason": reasons[run.finish], "stop_sequence": run.stop_sequence},
                                  "usage": {"output_tokens": run.completion_tokens}}, "message_delta"))
                self._chunk(_sse({"type": "message_stop"}, "message_stop"))
                self._end_stream()

            self._stream_guard(go, lambda msg: self._chunk(
                _sse({"type": "error", "error": {"type": "api_error", "message": msg}}, "error")))

        self._locked(work)


class HearthServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server bind a port that is in use.
    allow_reuse_address = os.name != "nt"

    def __init__(self, addr, app: App):
        if ":" in addr[0]:
            self.address_family = socket.AF_INET6
        self.app = app
        super().__init__(addr, Handler)

    def handle_error(self, request, client_address):
        # Clients that vanish (reset keep-alive connections, aborted SDK calls) are routine.
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def make_server(engine, tokenizer, template: ChatTemplate | None = None, *, host: str = "127.0.0.1",
                port: int = 8080, **app_kwargs) -> HearthServer:
    return HearthServer((host, port), App(engine, tokenizer, template, **app_kwargs))


def serve(model_path, *, host: str = "127.0.0.1", port: int = 8080, model_name: str | None = None,
          api_key: str | None = None, max_queue: int = 8, speculative: str = "none", draft_len: int = 4,
          ngram_n: int = 3, cors: str | None = None, verbose: bool = False, engine_kwargs: dict | None = None,
          tokenizer_path=None, timeout: float | None = 60.0) -> None:
    from pathlib import Path

    from hearth.chat import ChatTemplate as _CT
    from hearth.chat import Tokenizer, read_metadata
    from hearth.engine import Engine

    check_speculative(speculative, draft_len, ngram_n)  # before the (possibly slow) model open
    if timeout is not None and not timeout > 0:
        raise ValueError(f"timeout must be > 0 seconds or None, got {timeout!r}")
    meta = read_metadata(model_path)
    try:
        tok = Tokenizer.from_container(model_path, meta, path=tokenizer_path or None)
    except FileNotFoundError as e:
        if tokenizer_path:
            raise
        print(f"warning: {e}; only /v1/completions with a token-id prompt will work "
              f"(replies carry 'token_ids', no text)", file=sys.stderr)
        tok = None
    template = _CT.from_metadata(meta, tok)
    if tok is not None and template.is_fallback:
        print("note: container has no chat template; using ChatML", file=sys.stderr)
    engine = Engine(model_path, **(engine_kwargs or {}))
    try:
        srv = make_server(engine, tok, template, host=host, port=port,
                          model_name=model_name or Path(model_path).stem, api_key=api_key, max_queue=max_queue,
                          speculative=speculative, draft_len=draft_len, ngram_n=ngram_n, cors=cors, verbose=verbose,
                          timeout=timeout)
        print(f"hearth: serving {srv.app.model_name} on {srv.url} (OpenAI: {srv.url}/v1, Anthropic: {srv.url}/v1/messages)",
              file=sys.stderr, flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            srv.server_close()
    finally:
        engine.close()
