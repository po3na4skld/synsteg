"""Operating modes measured in the dissertation behind one encode / decode pair.

Mode = b (1-5), standard or enhanced (additional bits, 2.8) and the improvements of subsection 4.3:
  identity     the original word holds one of the 2^b codes, at c_i = PRF(K_s, nonce, i) (4.6)
  hamming      Hamming (7,3) syndrome coding over binary slots {c_i: x_i, 1-c_i: s_i}, <= 1 change per 3 bits
  variable_bits  b_i = min(b, floor(log2 |U_i|)) (4.5)
  ppl_choice   GPT-2 judge picks the synonym, the slot's codes are swapped so the block points to it (4.8)
  delta, gamma perplexity window and target (4.7)
  key_selection  K PRP permutations of the table, the one closest to human texts for a guide detector (4.3, alg. 4.1)
The receiver reads the mode from the key file ({"version": 2, "mode": ..., "key": ...}); old plain-list keys decode
as before. K_s (the passphrase) is never stored: identity / Hamming slots are stored without the original word and
key selection stores the canonical table plus k, so both need the passphrase to decode.
Ported from experiments/improvements (common.py, build_arms.py, ppl_window.py, profiles.py, select_key.py) and
experiments/ppl_opt/optimize.py.
"""
import copy
import hashlib
import hmac
import math
import secrets
import time
from dataclasses import dataclass, asdict, fields
from itertools import product

import numpy as np

from steganography import core, helper
from steganography.gpt import generate_container
from steganography.secret_key import binarize_synonyms, generate_secret_key_mp, align_container_and_secret_key
from models.pool_arguments import LLMConfig, SecretKeyGenerationBody

LEFT, RIGHT, MAX_RETRY = 20, 4, 10  # judge window of 4.8: 20 left GPT-2 tokens of the stegotext, 4 right container words
H = np.array([[(j >> r) & 1 for j in range(1, 8)] for r in range(3)])  # 3x7 parity-check matrix, column j = bin(j)


@dataclass
class Mode:
    bits_per_word: int = 5
    additional_bits: int = 0  # > 0: enhanced mode
    identity: bool = False
    hamming: bool = False
    variable_bits: bool = False
    ppl_choice: bool = False
    delta: float | None = None
    gamma: float | None = None
    key_selection: bool = False
    n_candidates: int = 16

    @property
    def needs_scorer(self):
        return self.ppl_choice or self.delta is not None or self.gamma is not None

    @property
    def needs_secret(self):
        return self.identity or self.hamming or self.key_selection

    @property
    def improved(self):
        return self.needs_scorer or self.needs_secret or self.variable_bits

    def check(self):
        if not 1 <= self.bits_per_word <= 5:
            raise ValueError("bits_per_word must be 1-5")
        if self.additional_bits and self.improved:
            raise ValueError("enhanced mode (additional bits) does not combine with the 4.3 improvements")
        if self.identity + self.hamming + self.variable_bits > 1:
            raise ValueError("identity code, Hamming code and variable bit width are alternative slot layouts")
        if self.key_selection and (self.needs_scorer or self.identity or self.hamming):
            raise ValueError("key selection permutes a standard / variable-width table; "
                             "it does not combine with perplexity choice, identity code or Hamming code")
        return self

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})


PROFILES = {  # table 4.8 of the dissertation
    "steganalysis": dict(bits_per_word=2, hamming=True, ppl_choice=True),
    "steganalysis_keep_capacity": dict(bits_per_word=2, identity=True, delta=0.05, ppl_choice=True),
    "ai_text_detectors": dict(bits_per_word=4),  # any b = 1-4, standard mode
    "detector_ensemble": dict(bits_per_word=5, key_selection=True),
    "reader_d1": dict(bits_per_word=5, delta=0.02, gamma=2.0, ppl_choice=True),
    "max_capacity": dict(bits_per_word=5, variable_bits=True),
}


def codes(b):
    return ["".join(c) for c in product("01", repeat=b)]


def prf(secret: bytes, *parts) -> int:
    return int.from_bytes(hmac.new(secret, "|".join(map(str, parts)).encode(), hashlib.sha256).digest(), "big")


def distinct_syns(base, mapping):
    """U_i: distinct substitutes in key order, case-insensitive, without the word itself."""
    seen, out = {base.lower()}, []
    for s in mapping.values():
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def untouched(base):
    return {base: {"0": base, "1": base}}  # has_duplicates -> encoder and decoder skip it


def visitable(container, key):
    """Can the encoder visit word i at all? Slots it cannot visit must not hold the word itself as a code."""
    out = []
    for token, slot in zip(container.split(), key):
        _, token = helper.check_endswith_special(token)
        out.append(helper.is_token_replacable(token, slot))
    return out


def safe_syns(words, i, U):
    """Drop substitutes the decoder would also match on the unchanged text (x_i + following words)."""
    s = helper.clean_container(" ".join(words[i:]))
    return [v for v in U if not (v.lower() == s[:len(v)].lower() and helper.check_is_end_of_word(s, len(v)))]


def walk(container, key):
    """Word indices the encoder visits (replaceable, usable slot), in order."""
    out = []
    for i, (token, slot) in enumerate(zip(container.split(), key)):
        _, token = helper.check_endswith_special(token)
        if helper.is_token_replacable(token, slot) and not helper.has_duplicates(slot):
            out.append(i)
    return out


def capacity(key):
    return sum(len(next(iter(m))) for s in key for m in s.values() if len(set(m.values())) == len(m))


def _walk(container, key, bits, choose=None):
    """core.embed with an optional chooser; returns (stego, key used, bits consumed, trace). A last block shorter than
    its slot is padded with random bits (the receiver cuts at n_bits), so the key itself is never shortened.
    choose(i, token, mapping, part, render, out) -> code to use, or None to keep the word (no bits)."""
    key, out, idx, trace = copy.deepcopy(key), [], 0, []
    for i, (token, slot) in enumerate(zip(container.split(), key)):
        special, token = helper.check_endswith_special(token)
        if helper.is_token_replacable(token, slot) and idx < len(bits) and not helper.has_duplicates(slot):
            orig, (token, cap) = token, helper.check_capitalization(token, slot)
            mapping = slot[token]
            size = len(next(iter(mapping)))
            part = bits[idx: idx + size]
            part += "".join(secrets.choice("01") for _ in range(size - len(part)))
            render = lambda s, cap=cap, special=special: (s.capitalize() if cap else s) + special
            code = part if choose is None else choose(i, token, mapping, part, render, out)
            if code is not None and code != part and code in mapping:  # chooser picked another synonym (4.8)
                if part in mapping:
                    mapping[part], mapping[code] = mapping[code], mapping[part]
                else:
                    mapping[part] = mapping.pop(code)
            if code is None or part not in mapping:  # no synonym for this block here: the word carries no bits
                out.append(orig + special)
                continue
            out.append(render(mapping[part]))
            trace.append((i, idx, size, mapping[part], token))
            idx += size
        else:
            out.append(token + special)
    return " ".join(out), key, idx, trace


def _first_bad(dec, want, trace):
    """(position, chosen word) where decoding first goes wrong, to ban it and re-encode."""
    p = next((k for k, (x, y) in enumerate(zip(dec, want)) if x != y), None)
    if p is None and len(dec) >= len(want):
        return None
    p = len(dec) if p is None else p
    t = next((t for t in trace if t[1] + t[2] > p), None)
    return None if t is None or t[3].lower() == t[4].lower() else (t[0], t[3])


def _syndromes(bits, G):
    return "".join("".join(map(str, (H @ np.array([int(x) for x in bits[7 * g: 7 * g + 7]])) % 2)) for g in range(G))


class _Deltas:
    """Delta_i(w) = ln PPL(container with w at i) - ln PPL(container) (4.7), computed on demand."""

    def __init__(self, scorer, container, key):
        self.sc, self.words, self.key = scorer, container.split(), key
        self.base, self.cache = scorer.lnppl([container])[0], {}

    def __call__(self, i, cands):
        todo = [w for w in cands if (i, w) not in self.cache]
        if todo:
            special, tok = helper.check_endswith_special(self.words[i])
            _, cap = helper.check_capitalization(tok, self.key[i])
            texts = [" ".join(self.words[:i] + [(w.capitalize() if cap else w) + special] + self.words[i + 1:]) for w in todo]
            for w, v in zip(todo, self.sc.lnppl(texts)):
                self.cache[(i, w)] = float(v) - self.base
        return {w: self.cache[(i, w)] for w in cands}


def prp_key(T, k, secret, nonce):
    """pi_k = PRP(K_s, k): the codes of every slot are ordered by HMAC-SHA256(K_s, nonce|k|i|j)."""
    out = []
    for i, slot in enumerate(T):
        base, mp = next(iter(slot.items()))
        vals = list(mp.values())
        if len(set(vals)) < len(vals):
            out.append(slot)
            continue
        order = sorted(range(len(vals)), key=lambda j: prf(secret, "prp", nonce, k, i, j))
        out.append({base: dict(zip(mp.keys(), [vals[j] for j in order]))})
    return out


def _table(raw, mode):
    """Standard (binarize_synonyms, 90 % rule) or variable-width (4.5) table from the raw synonym lists."""
    out = []
    for slot in raw:
        base, mp = next(iter(slot.items()))
        U = distinct_syns(base, mp)
        if not mode.variable_bits:
            out.append({base: binarize_synonyms(base, U, mode.bits_per_word)})
            continue
        b = min(mode.bits_per_word, int(math.log2(len(U)))) if U else 0
        out.append({base: dict(zip(codes(b), U[: 2 ** b]))} if b else untouched(base))
    return out


def _embed_once(container, raw, bits, mode, secret, nonce, scorer, D, ban):
    """One encoding pass (no key selection). Returns (stego, final key, message bits used, trace, ok, bad)."""
    words = container.split()
    vis = visitable(container, raw)
    best_of = lambda i, cands, render, out: (
        cands[0] if len(cands) == 1 or not mode.needs_scorer else
        cands[int(np.argmin(scorer.nll(out[-LEFT:], [render(v) for v in cands], words[i + 1: i + 1 + RIGHT])))])
    usable = lambda i, U: [v for v in U if (i, v) not in ban] or U

    if mode.identity or mode.hamming:
        key, A = [], {}
        for i, slot in enumerate(raw):
            base, mp = next(iter(slot.items()))
            U = safe_syns(words, i, distinct_syns(base, mp))
            if D is not None and U and vis[i]:
                Di = D(i, U)
                U = sorted([w for w in U if Di[w] <= mode.delta], key=lambda w: Di[w])
            b = 1 if mode.hamming else min(mode.bits_per_word, int(math.log2(len(U) + 1)))
            if not U or b == 0 or not vis[i]:
                key.append(untouched(base))
                continue
            A[i] = U
            vals = U[: 2 ** b - 1]
            vals.insert(prf(secret, "ham" if mode.hamming else "id", nonce, i) % 2 ** b, base)
            key.append({base: dict(zip(codes(b), vals))})

        def choose(i, token, mapping, part, render, out):
            if mapping[part].lower() == token.lower() or not mode.needs_scorer:
                return part  # the original word stays; no choice requested
            if mode.hamming:  # the flipped position: substitute chosen by the 4.8 judge
                mapping[part] = best_of(i, usable(i, A[i]), render, out)
                return part
            w = best_of(i, usable(i, [v for v in mapping.values() if v.lower() != token.lower()]), render, out)
            return next(c for c, s in mapping.items() if s == w)

        if mode.hamming:
            words_key = lambda i: next(iter(key[i]))  # the slot's word = its code c_i
            P = walk(container, key)
            n3 = -(-len(bits) // 3) * 3
            msg = bits.ljust(n3, "0")
            G = min(len(P) // 7, n3 // 3)
            v = np.array([int(next(c for c, w in key[i][words_key(i)].items() if w == words_key(i))) for i in P[: 7 * G]], int)
            for g in range(G):
                s = (H @ v[7 * g: 7 * g + 7] + np.array([int(x) for x in msg[3 * g: 3 * g + 3]])) % 2
                j = int(s[0] + 2 * s[1] + 4 * s[2])  # syndrome -> column index (1..7), 0 = no change
                if j:
                    v[7 * g + j - 1] ^= 1
            walk_bits = "".join(map(str, v))
            stego, key2, _, trace = _walk(container, key, walk_bits, choose)
            dec = core.decode_message(stego, key2, clean_output=False)[0][: 7 * G]
            got = _syndromes(dec, G) if len(dec) == 7 * G else ""
            used = min(3 * G, len(bits))
            ok = got[:used] == bits[:used]
            return stego, key2, used, trace, ok, None if ok else _first_bad(dec, walk_bits, trace)
    else:
        key = _table(raw, mode)
        P = walk(container, key)
        T = scorer.n_tokens(container) - 1 if mode.gamma is not None else 1
        E = [0.0]

        def choose(i, token, mapping, part, render, out):  # ppl_window.run_one
            if not mode.needs_scorer:
                return part
            low = [v.lower() for v in mapping.values()]
            vals = [v for v in mapping.values() if low.count(v.lower()) == 1]
            Di = D(i, vals) if D is not None else {}
            A = [v for v in vals if (D is None or Di[v] <= mode.delta) and (i, v) not in ban]
            if not A:
                if any(safe_syns(words, i, [v]) != [v] for v in vals):  # keeping x_i would be misread
                    same = [v for v in vals if v.lower() == token.lower()]
                    w = (same or [min(vals, key=lambda v: Di.get(v, math.inf))])[0]
                    return next(c for c, s in mapping.items() if s == w)
                return None
            nll = scorer.nll(out[-LEFT:], [render(v) for v in A] + [words[i]], words[i + 1: i + 1 + RIGHT])
            e = np.asarray(nll[:-1]) - nll[-1]
            if mode.gamma is None:
                j = int(np.argmin(e))
            else:  # linear schedule of the ln-PPL excess towards ln gamma (4.7)
                j = int(np.argmin(np.abs((E[0] + e) / T - math.log(mode.gamma) * (P.index(i) + 1) / len(P))))
            E[0] += e[j]
            return next(c for c, s in mapping.items() if s == A[j])

    stego, key2, used, trace = _walk(container, key, bits, choose)
    used = min(used, len(bits))
    dec = core.decode_message(stego, key2, clean_output=False)[0]
    ok = dec[:used] == bits[:used]
    return stego, key2, used, trace, ok, None if ok else _first_bad(dec, bits[:used], trace)


def _embed(container, raw, bits, mode, secret, nonce, scorer, selector):
    """Returns (stego, key to store, extra mode fields, message bits used, trace, capacity bits)."""
    if mode.key_selection:
        T = _table(raw, mode)
        detector, calibration = selector
        cal = np.sort(np.asarray(calibration, float))
        F = lambda s: (np.searchsorted(cal, s, "left") + np.searchsorted(cal, s, "right")) / (2 * len(cal))
        cands = []
        for k in range(mode.n_candidates):
            key = prp_key(T, k, secret, nonce)
            stego, _, used, trace = _walk(container, key, bits)
            used = min(used, len(bits))
            if core.decode_message(stego, key, clean_output=False)[0][:used] == bits[:used]:
                cands.append((k, stego, used, trace))
        if not cands:
            raise ValueError("no candidate key decodes")
        crit = np.abs(F(np.asarray(detector([c[1] for c in cands]), float)) - 0.5)  # two-sided criterion (4.3)
        k, stego, used, trace = cands[int(np.argmin(crit))]
        return stego, T, dict(k=k), used, trace, capacity(T)

    D = _Deltas(scorer, container, raw) if mode.delta is not None else None
    ban = set()
    for _ in range(MAX_RETRY + 1):  # a failed decode bans the choice at the failing position (ppl_opt/optimize.py)
        stego, key, used, trace, ok, bad = _embed_once(container, raw, bits, mode, secret, nonce, scorer, D, ban)
        if ok or bad is None:
            break
        ban.add(bad)
    if mode.hamming:
        cap = len(walk(container, key)) // 7 * 3
    else:
        cap = capacity(key)
    if mode.identity or mode.hamming:  # store the slots without the original word: its code c_i needs K_s
        key = [s if helper.has_duplicates(s) else
               {w: [v for v in m.values() if v != w]} for s in key for w, m in s.items()]
    return stego, key, {}, used, trace, cap


def _rebuild(key, mode, secret):
    if mode.get("key_selection"):
        return prp_key(key, mode["k"], secret, mode["nonce"])
    out = []
    for i, slot in enumerate(key):
        base, v = next(iter(slot.items()))
        if isinstance(v, list):
            b = int(math.log2(len(v) + 1))
            vals = list(v)
            vals.insert(prf(secret, "ham" if mode.get("hamming") else "id", mode["nonce"], i) % 2 ** b, base)
            slot = {base: dict(zip(codes(b), vals))}
        out.append(slot)
    return out


def decode(stegotext: str, keyfile: dict | list, passphrase: str | None = None, as_text: bool | None = None) -> str:
    """Message (text, or bits for as_text=False / a bit-string message) from a stegotext and its key file.
    A plain list is an old key (before version 2) and decodes exactly as core.decode_message."""
    if isinstance(keyfile, list):
        return core.decode_message(stegotext, keyfile, clean_output=as_text is not False)[0]
    mode = keyfile["mode"]
    if Mode.from_dict(mode).needs_secret:
        if not passphrase:
            raise ValueError("this key needs the passphrase used for encoding")
        key = _rebuild(keyfile["key"], mode, passphrase.encode())
    else:
        key = keyfile["key"]
    raw = core.decode_message(stegotext, key, clean_output=False)[0]
    n = mode["n_bits"]
    if mode.get("hamming"):
        G = -(-n // 3)
        bits = _syndromes(raw, G)[:n] if len(raw) >= 7 * G else ""
    else:
        bits = raw[:n]
    if not (mode.get("text", True) if as_text is None else as_text):
        return bits
    try:
        return helper.get_text_from_binary(bits)
    except (ValueError, UnicodeDecodeError):
        raise ValueError("the stegotext does not decode with this key (wrong passphrase or key?)") from None


def encode(message: str, mode: Mode = Mode(), llm: LLMConfig = LLMConfig(), container: str | None = None,
           passphrase: str | None = None, scorer=None, selector=None, binarize: bool = True) -> dict:
    """Encode a message in the given mode. container=None: the LLM writes one. scorer (4.7/4.8) and selector
    ((detector, calibration scores), 4.3) default to steganography.ml (GPT-2, RoBERTa; optional extra).
    Returns dict(container, stegotext, keyfile, n_bits, capacity_bits, bits_per_word, replaced_share, recovered,
    time_report, usage_report)."""
    mode.check()
    if mode.needs_secret and not passphrase:
        raise ValueError("this mode needs a passphrase (shared with the receiver, never stored in the key)")
    if mode.needs_scorer and scorer is None:
        from steganography import ml
        scorer = ml.get_scorer()
    if mode.key_selection and selector is None:
        from steganography import ml
        selector = ml.get_selector()
    bits = helper.binarize_message(message) if binarize else message
    nonce = secrets.token_hex(8)
    extra = dict(nonce=nonce, n_bits=len(bits), text=binarize)

    if not mode.improved:  # the published method, unchanged
        container, stego, key, time_report, usage_report = core.encode_message(
            bits, mode.bits_per_word, mode.additional_bits, binarize=False, container=container, llm=llm)
        trace = _walk(container, key, bits)[3]
        used, cap = len(bits), capacity(key)
    else:
        time_report, usage_report = {}, {}
        user_container = container is not None
        rate = 3 / 7 if mode.hamming else mode.bits_per_word
        length = len(bits) / rate * 1.5
        for attempt in range(core.MAX_CAPACITY_ATTEMPTS):
            sfx = f"_{attempt}" if attempt else ""
            s = time.time()
            if not user_container:
                container, usage_report[f"container_generation{sfx}"] = generate_container(length, llm)
                time_report[f"container_generation{sfx}"] = round(time.time() - s, 2)
            s = time.time()
            body = SecretKeyGenerationBody.from_list(
                helper.divide_chunks(container.split(), 5), mode.bits_per_word, 0, [], llm)
            raw, usage_report[f"secret_key_generation{sfx}"] = generate_secret_key_mp(body, raw=True)
            raw = align_container_and_secret_key(container, raw)
            time_report[f"secret_key_generation{sfx}"] = round(time.time() - s, 2)
            s = time.time()
            stego, key, more, used, trace, cap = _embed(container, raw, bits, mode, passphrase and passphrase.encode(),
                                                        nonce, scorer, selector)
            time_report[f"embedding{sfx}"] = round(time.time() - s, 2)
            extra.update(more)
            if used == len(bits):
                break
            if user_container:
                raise ValueError(f"Container holds {used} of {len(bits)} bits")
            length *= 1.2 * len(bits) / max(used, 1)
        else:
            raise ValueError(f"Message did not fit after {core.MAX_CAPACITY_ATTEMPTS} containers")

    keyfile = {"version": 2, "mode": {**asdict(mode), **extra}, "key": key}
    try:
        recovered = decode(stego, keyfile, passphrase, as_text=False) == bits
    except ValueError:
        recovered = False
    n_words = len(container.split())
    return dict(container=container, stegotext=stego, keyfile=keyfile, n_bits=len(bits), capacity_bits=cap,
                bits_per_word=len(bits) / n_words, replaced_share=sum(w.lower() != o.lower() for *_, w, o in trace) / n_words,
                recovered=recovered, time_report=time_report, usage_report=usage_report)
