"""Token sampling and generation with lossless prompt-lookup speculative decoding.

Speculative decoding here drafts tokens by looking the current suffix up in the
context itself (no draft model), verifies `[last] + draft` in one batched
forward and rewinds the KV cache past the first rejected position. Greedy
output is token-for-token identical to plain greedy decoding because the
engine's batched evaluation is bit-identical to sequential evaluation
(INV-DET-2). With sampling, drafts are accepted by exact speculative sampling
for a point-mass proposal, so the output distribution is unchanged (the random
stream is consumed differently, so a fixed seed gives a different sample than
plain decoding).

`engine` is anything with `eval(tokens, all_logits)`, `pos`, `rewind(pos)` and
`reset()` (hearth.engine.Engine or a test double).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, NamedTuple, Sequence

import numpy as np

__all__ = ["Sampler", "GenStats", "Token", "generate", "generate_ids", "NgramIndex",
           "common_prefix", "sync_prefix", "accept_point_mass", "kv_capacity", "check_speculative",
           "SPECULATIVE_MODES"]

SPECULATIVE_MODES = ("none", "ngram")


@dataclass
class Sampler:
    """Logit processing in the usual order: repetition penalty -> temperature ->
    top-k -> top-p -> min-p. temperature == 0 means greedy (argmax, first index
    wins ties)."""
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not np.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if not (0.0 <= self.min_p <= 1.0):
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}")
        if not np.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")
        self.rng = np.random.default_rng(self.seed)

    @classmethod
    def greedy(cls) -> "Sampler":
        return cls(temperature=0.0)

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    @property
    def uses_context(self) -> bool:
        return self.repetition_penalty != 1.0

    def _penalised(self, logits, context) -> np.ndarray:
        x = np.array(logits, dtype=np.float64).reshape(-1)
        # -inf (a masked token) is allowed; NaN or +inf would make every choice arbitrary.
        if np.isnan(x).any() or np.isposinf(x).any() or np.isneginf(x).all():
            raise ValueError("logits contain NaN or +inf, or are all -inf")
        if self.repetition_penalty != 1.0 and context is not None and len(context):
            ids = np.unique(np.asarray(context, dtype=np.int64))
            ids = ids[(ids >= 0) & (ids < x.size)]
            v = x[ids]
            x[ids] = np.where(v > 0, v / self.repetition_penalty, v * self.repetition_penalty)
        return x

    def argmax(self, logits, context=None) -> int:
        return int(np.argmax(self._penalised(logits, context)))

    def probs(self, logits, context=None) -> np.ndarray:
        """The final sampling distribution (float64, sums to 1). Greedy -> one-hot."""
        x = self._penalised(logits, context)
        V = x.size
        if self.is_greedy:
            p = np.zeros(V)
            p[int(np.argmax(x))] = 1.0
            return p
        # Shift before scaling: a tiny temperature then tends to a point mass
        # instead of overflowing to inf - inf = NaN.
        with np.errstate(over="ignore"):
            x = (x - np.max(x)) / self.temperature
        if 0 < self.top_k < V:
            kth = np.partition(x, V - self.top_k)[V - self.top_k]
            x[x < kth] = -np.inf
        p = np.exp(x)
        p /= p.sum()
        if self.top_p < 1.0:
            order = np.argsort(-p, kind="stable")
            ps = p[order]
            before = np.cumsum(ps) - ps
            drop = order[before >= self.top_p]
            p[drop] = 0.0
            p /= p.sum()
        if self.min_p > 0.0:
            p[p < self.min_p * p.max()] = 0.0
            p /= p.sum()
        return p

    def draw(self, p: np.ndarray) -> int:
        """Inverse-CDF draw from a normalised distribution."""
        c = np.cumsum(p)
        u = self.rng.random() * c[-1]
        i = int(np.searchsorted(c, u, side="right"))
        i = min(i, p.size - 1)
        while p[i] <= 0.0 and i > 0:  # u landed exactly on a boundary into a zero-mass tail
            i -= 1
        return i

    def sample(self, logits, context=None) -> int:
        if self.is_greedy:
            return self.argmax(logits, context)
        return self.draw(self.probs(logits, context))


def accept_point_mass(p: np.ndarray, d: int, rng: np.random.Generator) -> tuple[bool, int]:
    """Speculative sampling with a deterministic draft d (proposal q = delta_d).
    Accept with probability min(1, p[d]/q[d]) = p[d]; on rejection draw from the
    residual max(0, p - q) normalised, i.e. p with d removed. The result is
    distributed exactly as p. Returns (accepted, token)."""
    pd = float(p[d])
    if rng.random() < pd:
        return True, int(d)
    r = np.array(p, dtype=np.float64)
    r[d] = 0.0
    s = r.sum()
    if s <= 0.0:  # p is (numerically) a point mass on d
        return True, int(d)
    c = np.cumsum(r)
    u = rng.random() * c[-1]
    i = min(int(np.searchsorted(c, u, side="right")), r.size - 1)
    while r[i] <= 0.0 and i > 0:
        i -= 1
    return False, i


class NgramIndex:
    """Incremental prompt-lookup index over a growing token sequence.

    For every n in 1..max_n it maps an n-gram to the most recent start position
    that is followed by at least one token, so the current suffix never matches
    itself."""

    def __init__(self, max_n: int, tokens: Sequence[int] = ()):
        if max_n < 1:
            raise ValueError("ngram_n must be >= 1")
        self.max_n = max_n
        self.seq: list[int] = []
        self._maps: list[dict] = [dict() for _ in range(max_n + 1)]
        for t in tokens:
            self.append(t)

    def append(self, tok: int) -> None:
        s = self.seq
        s.append(int(tok))
        j = len(s) - 1  # the n-grams ending at j-1 now have a follower (s[j])
        for n in range(1, self.max_n + 1):
            start = j - n
            if start < 0:
                break
            self._maps[n][tuple(s[start:j])] = start

    def propose(self, k: int) -> list[int]:
        """Up to k tokens that followed the most recent earlier occurrence of the
        longest matching suffix (longest n first)."""
        s = self.seq
        if k <= 0 or not s:
            return []
        for n in range(min(self.max_n, len(s)), 0, -1):
            start = self._maps[n].get(tuple(s[len(s) - n:]))
            if start is not None:
                return s[start + n: start + n + k]
        return []


@dataclass
class GenStats:
    prompt_tokens: int = 0
    reused_tokens: int = 0         # prompt tokens whose KV was already held (prefix reuse)
    new_tokens: int = 0            # tokens yielded (stop token excluded)
    decode_tokens: int = 0         # tokens produced by decode forwards (incl. a final stop token)
    forwards: int = 0              # decode forward calls (prefill excluded)
    draft_steps: int = 0           # forwards that verified a non-empty draft
    drafted: int = 0
    accepted: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    finish_reason: str | None = None   # "stop" | "length"
    stop_id: int | None = None

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def tokens_per_forward(self) -> float:
        return self.decode_tokens / self.forwards if self.forwards else 0.0

    @property
    def prefill_tok_s(self) -> float:
        n = self.prompt_tokens - self.reused_tokens
        return n / self.prefill_s if self.prefill_s > 0 else 0.0

    @property
    def decode_tok_s(self) -> float:
        return self.decode_tokens / self.decode_s if self.decode_s > 0 else 0.0

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        for k in ("acceptance_rate", "tokens_per_forward", "prefill_tok_s", "decode_tok_s"):
            d[k] = getattr(self, k)
        return d


class Token(NamedTuple):
    id: int
    step: int          # forward that produced it (0 = prefill)
    drafted: int       # draft length verified in that forward
    accepted: int      # draft tokens accepted in that forward
    stats: GenStats    # running totals (same object for the whole generation)


def kv_capacity(engine) -> int:
    cap = getattr(engine, "kv_capacity", None)
    if cap is None:
        info = getattr(engine, "info", {}) or {}
        cap = int(info.get("max_seq", 0)) or (1 << 30)
    return int(cap)


def common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    x = np.asarray(a[:n], dtype=np.int64)
    y = np.asarray(b[:n], dtype=np.int64)
    diff = np.flatnonzero(x != y)
    return int(diff[0]) if diff.size else n


def sync_prefix(engine, held: Sequence[int] | None, prompt: Sequence[int]) -> int:
    """Rewind the engine to the longest prefix of `prompt` it already holds and
    return its length. `held` lists the tokens whose KV the engine holds
    (None = unknown: reset). At least one prompt token is always left to
    evaluate, since its logits are needed."""
    if held is None:
        engine.reset()
        return 0
    pos = engine.pos
    if pos > len(held):
        engine.reset()
        return 0
    k = min(common_prefix(held[:pos], prompt), len(prompt) - 1, pos)
    if k < pos:
        engine.rewind(k)
    return k


def _mode(speculative) -> str:
    if speculative in (None, False, "none", "off", ""):
        return "none"
    if speculative in (True, "ngram", "prompt", "prompt_lookup"):
        return "ngram"
    raise ValueError(f"speculative must be one of {SPECULATIVE_MODES}, got {speculative!r}")


def check_speculative(speculative, draft_len: int, ngram_n: int) -> str:
    """Validate speculative-decoding settings (so front ends can reject a bad
    configuration at startup); returns the normalised mode."""
    mode = _mode(speculative)
    for name, v, lo in (("draft_len", draft_len, 0), ("ngram_n", ngram_n, 1)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{name} must be an integer, got {v!r}")
        if v < lo:
            raise ValueError(f"{name} must be >= {lo}, got {v}")
    return mode


def generate(engine, prompt_ids: Sequence[int], max_new_tokens: int = 256, sampler: Sampler | None = None,
             stop_ids: Sequence[int] = (), speculative: str = "none", draft_len: int = 4, ngram_n: int = 3,
             *, held: Sequence[int] | None = None) -> Iterator[Token]:
    """Yield generated tokens one at a time (stop token not yielded).

    held: tokens whose KV the engine currently holds (prefix reuse); None resets
    the engine first. Once the generator has yielded a token or returned
    (including max_new_tokens == 0), and from then on even if it is closed or
    interrupted, the engine holds a prefix of prompt + yielded tokens of length
    engine.pos (all but the last token after a normal finish). If it raises
    before that, the engine's KV content is unspecified. Returns the GenStats
    (StopIteration.value).
    """
    sampler = sampler if sampler is not None else Sampler.greedy()
    mode = check_speculative(speculative, draft_len, ngram_n)
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    prompt = [int(t) for t in prompt_ids]
    if not prompt:
        raise ValueError("prompt_ids must not be empty")
    cap = kv_capacity(engine)
    if len(prompt) > cap:
        raise ValueError(f"prompt has {len(prompt)} tokens but the KV cache holds {cap}")
    stop = frozenset(int(s) for s in stop_ids)
    st = GenStats(prompt_tokens=len(prompt))
    if max_new_tokens == 0:
        # Nothing is evaluated, but the engine must still end up holding a
        # prefix of this prompt: callers that track the KV (Conversation) rely on it.
        st.reused_tokens = sync_prefix(engine, held, prompt)
        st.finish_reason = "length"
        return st

    seq = list(prompt)
    index = NgramIndex(ngram_n, seq) if mode == "ngram" else None
    use_ctx = sampler.uses_context

    try:
        k = sync_prefix(engine, held, prompt)
        st.reused_tokens = k
        t0 = time.perf_counter()
        logits = engine.eval(prompt[k:])
        new = [sampler.sample(logits, seq)]
        st.prefill_s = time.perf_counter() - t0
        step, drafted, n_acc = 0, 0, 0

        while True:
            for tok in new:
                if step > 0:
                    st.decode_tokens += 1
                if tok in stop:
                    st.finish_reason, st.stop_id = "stop", tok
                    return st
                seq.append(tok)
                if index is not None:
                    index.append(tok)
                st.new_tokens += 1
                yield Token(tok, step, drafted, n_acc, st)
                if st.new_tokens >= max_new_tokens:
                    st.finish_reason = "length"
                    return st

            base = engine.pos  # holds seq[:-1]; seq[-1] is fed next
            if base != len(seq) - 1:
                raise RuntimeError(f"engine position {base} out of sync with sequence length {len(seq)}")
            if base + 1 > cap:
                st.finish_reason = "length"
                return st

            room = min(draft_len, max_new_tokens - st.new_tokens - 1, cap - base - 1)
            draft = index.propose(room) if index is not None and room > 0 else []
            t0 = time.perf_counter()
            step += 1
            st.forwards += 1
            if not draft:
                logits = engine.eval([seq[-1]])
                new = [sampler.sample(logits, seq)]
                drafted, n_acc = 0, 0
            else:
                rows = engine.eval([seq[-1]] + draft, all_logits=True)
                full = seq + draft if use_ctx else seq
                n_acc, final = 0, None
                for i, d in enumerate(draft):
                    ctx = full[: len(seq) + i] if use_ctx else None
                    if sampler.is_greedy:
                        t = sampler.argmax(rows[i], ctx)
                        ok = t == d
                    else:
                        ok, t = accept_point_mass(sampler.probs(rows[i], ctx), d, sampler.rng)
                    if not ok:
                        final = t
                        break
                    n_acc += 1
                if final is None:
                    ctx = full[: len(seq) + len(draft)] if use_ctx else None
                    final = sampler.sample(rows[len(draft)], ctx)
                engine.rewind(base + 1 + n_acc)
                new = draft[:n_acc] + [final]
                drafted = len(draft)
                st.draft_steps += 1
                st.drafted += drafted
                st.accepted += n_acc
            st.decode_s += time.perf_counter() - t0
    finally:
        # Keep the documented post-condition even when closed mid-step.
        try:
            if engine.pos > len(seq):
                engine.rewind(len(seq))
        except Exception:
            pass


def generate_ids(engine, prompt_ids, max_new_tokens=256, sampler=None, stop_ids=(), speculative="none",
                 draft_len=4, ngram_n=3, *, held=None) -> tuple[list[int], GenStats]:
    gen = generate(engine, prompt_ids, max_new_tokens, sampler, stop_ids, speculative, draft_len, ngram_n,
                   held=held)
    out = []
    while True:
        try:
            out.append(next(gen).id)
        except StopIteration as e:
            return out, e.value
