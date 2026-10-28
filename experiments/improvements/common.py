"""Shared pieces for the 4.3 improvement experiments (offline, no API calls).

Corpus 3.7 = the 299 reports of reports/gpt_omni_reports that build_corpus.py keeps (same order, same filter).
Encoder = the walk of build_corpus.encode_with_key / steganography.core.embed for a fixed key, with the last block
shortened in the key itself (the decoder must see it) and with an optional per-position chooser.
Decoder = steganography.core.decode_message of the working tree, unchanged (imported with the API modules stubbed,
as ppl_opt/optimize.py does).
"""
import copy, glob, hashlib, hmac, json, pathlib, sys
from itertools import product
from unittest.mock import MagicMock

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AID = ROOT / "experiments/ai_detectors"
sys.path.insert(0, str(ROOT))
for m in ("steganography.gpt", "steganography.secret_key", "models", "models.pool_arguments"):
    sys.modules.setdefault(m, MagicMock())
from steganography import helper  # noqa: E402
from steganography.core import decode_message  # noqa: E402

ARMS = HERE / "arms"; ARMS.mkdir(exist_ok=True)
DETS = ["D1_roberta_openai", "D2_chatgpt_roberta", "D3_desklib", "D4_binoculars", "D5_fast_detectgpt"]


def reports():
    out = []
    for f in sorted(glob.glob(str(ROOT / "reports/gpt_omni_reports/*.json"))):
        d = json.load(open(f))
        if d["message"].startswith(d["decoded_message"]):
            out.append(d)
    return out


def codes(b):
    return ["".join(c) for c in product("01", repeat=b)]


def distinct_syns(base, mapping):
    """U_i: distinct substitutes in key order, case-insensitive, without the word itself."""
    seen, out = {base.lower()}, []
    for s in mapping.values():
        if s.lower() not in seen:
            seen.add(s.lower()); out.append(s)
    return out


def untouched(base):
    return {base: {"0": base, "1": base}}  # has_duplicates -> encoder and decoder skip it


def prf(secret: bytes, *parts) -> int:
    return int.from_bytes(hmac.new(secret, "|".join(map(str, parts)).encode(), hashlib.sha256).digest(), "big")


def encode(container, key, bits, choose=None, track=None):
    """Returns (stego, key actually used (deep copy, last block shortened), bits used, trace).
    choose(i, token, mapping, part, render) -> code to use, or None to leave the word (no bits).
    Default: the code given by the message block (the published encoder). track(i, word) sees every output word."""
    key, out, idx, trace = copy.deepcopy(key), [], 0, []
    words = container.split()
    for i, (token, slot) in enumerate(zip(words, key)):
        special, token = helper.check_endswith_special(token)
        if helper.is_token_replacable(token, slot) and idx < len(bits) and not helper.has_duplicates(slot):
            orig, (token, cap) = token, helper.check_capitalization(token, slot)
            mapping = slot[token]
            size = len(next(iter(mapping)))
            part = bits[idx: idx + size]
            if len(part) != size:
                mapping = slot[token] = {c: s for c, s in zip(codes(len(part)), list(mapping.values()))}
            render = lambda s: (s.capitalize() if cap else s) + special
            code = part if choose is None else choose(i, token, mapping, part, render)
            if code is None:
                out.append(orig + special)
                if track: track(i, out[-1])
                continue
            if code != part:  # chooser picked another synonym: swap codes so that phi(part) = chosen (4.8)
                mapping[part], mapping[code] = mapping[code], mapping[part]
            out.append(render(mapping[part])); trace.append((i, idx, len(part), mapping[part], token)); idx += len(part)
        else:
            out.append(token + special)
        if track: track(i, out[-1])
    return " ".join(out), key, idx, trace


def visitable(container, key):
    """Per position: can the encoder visit this word at all (word matches its slot)? Slots of words it cannot visit
    (e.g. 'trust?.') must not hold the word itself as a code, otherwise the decoder reads bits there."""
    out = []
    for token, slot in zip(container.split(), key):
        _, token = helper.check_endswith_special(token)
        out.append(helper.is_token_replacable(token, slot))
    return out


def safe_syns(words, i, U):
    """Drop substitutes that the decoder would also match on the unchanged text (x_i + following words, e.g.
    'such' + 'as' vs the substitute 'such as'): with x_i as a code such a slot is ambiguous."""
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


def decode_bits(stego, key):
    return decode_message(stego, key, clean_output=False)[0]


def recovered(stego, key, bits, used):
    return decode_bits(stego, key)[:used] == bits[:used]


def replaced_share(container, trace):
    """Share of container words the encoder replaced by a different word (case-insensitive)."""
    return sum(s.lower() != o.lower() for _, _, _, s, o in trace) / len(container.split())


def write_arm(name, recs):
    with open(ARMS / f"{name}.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_arm(name):
    return [json.loads(l) for l in open(ARMS / f"{name}.jsonl")]
