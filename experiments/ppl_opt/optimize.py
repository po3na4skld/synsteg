"""Perplexity-optimised SynSteg key (offline, stored GPT-4o reports, no API calls).

For every run of reports/gpt_omni_reports the message prefix that the stored container holds (the same bits the
published key embeds, build_corpus.encode_with_key) is re-embedded with a message-dependent key: left to right, at each
active position with block u, s* = argmin over the slot's synonyms of the GPT-2 (124M) summed NLL of
  [last LEFT tokens of the already-substituted stegotext] + s* + [next RIGHT container words],
then the slot's codes are permuted (swap s* with the synonym that had code u) so phi_i(u) = s*.
The synonym set of every slot is unchanged, so capacity is unchanged; recovery is checked with the project decoder
(steganography.core.decode_message). If decoding fails (prefix ambiguity of multi-word synonyms), the failing choice
is banned and the run is re-optimised (at most MAX_RETRY times).
Variants: K = 32 (all synonyms), 8, 4 (only the first K synonyms of a slot in the published order are candidates).
Usage: <detector venv python> optimize.py  -> series/opt_k{K}/<uuid>.json (report format + opt fields)
"""
import copy, glob, json, os, pathlib, sys, time
from unittest.mock import MagicMock
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments/ai_detectors"))
# decode_message only needs steganography.helper; stub the API/key-generation imports of core.py (no openai, no config)
for m in ("steganography.gpt", "steganography.secret_key", "models", "models.pool_arguments"):
    sys.modules.setdefault(m, MagicMock())
from steganography import helper  # noqa: E402
from steganography.core import decode_message  # noqa: E402
from build_corpus import encode_with_key  # noqa: E402

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
LEFT, RIGHT, MAX_RETRY = 20, 4, 10  # left context in GPT-2 tokens, right context in container words
KS = [int(k) for k in os.environ.get("KS", "32,8,4").split(",")]


class Scorer:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.m = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(DEV).eval()

    @torch.no_grad()
    def nll(self, left_words, cands, right_words):
        """Summed NLL of ' cand right...' after the left context, one value per candidate."""
        bos = [self.tok.eos_token_id]
        left = (bos + self.tok(" ".join(left_words)).input_ids)[-LEFT - 1:] if left_words else bos
        tail = (" " + " ".join(right_words)) if right_words else ""
        rights = [self.tok((" " if left_words else "") + c + tail).input_ids for c in cands]
        T = len(left) + max(map(len, rights))
        ids = torch.full((len(cands), T), self.tok.eos_token_id)
        tgt_mask = torch.zeros((len(cands), T))
        for j, r in enumerate(rights):
            ids[j, :len(left) + len(r)] = torch.tensor(left + r)
            tgt_mask[j, len(left):len(left) + len(r)] = 1
        ids, tgt_mask = ids.to(DEV), tgt_mask.to(DEV)
        lp = self.m(ids).logits[:, :-1].float().log_softmax(-1)
        ll = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) * tgt_mask[:, 1:]
        return (-ll.sum(1)).cpu().numpy()


def optimise(container, key, bits, sc, K=32, ban=frozenset()):
    """Greedy left-to-right choice; returns (stego, new key, bits used, [(word idx, start bit, size, choice, original word)])."""
    key, out, idx, trace = copy.deepcopy(key), [], 0, []
    words = container.split()
    for i, (token, slot) in enumerate(zip(words, key)):
        special, token = helper.check_endswith_special(token)
        if helper.is_token_replacable(token, slot) and idx < len(bits) and not helper.has_duplicates(slot):
            token, was_cap = helper.check_capitalization(token, slot)
            mapping = slot[token]
            size = len(next(iter(mapping)))
            part = bits[idx: idx + size]
            if len(part) != size:  # last block, as core.embed: shorten the slot in place (the decoder sees it)
                mapping = slot[token] = {c: s for c, s in zip(encode_codes(len(part)), list(mapping.values()))}
                size = len(part)
            vals = list(mapping.values())
            low = [v.lower() for v in vals]
            cands = [v for v in vals[:K] if low.count(v.lower()) == 1 and (i, v) not in ban]
            render = lambda s: (s.capitalize() if was_cap else s) + special
            best = cands[int(np.argmin(sc.nll(out[-LEFT:], [render(s) for s in cands], words[i + 1: i + 1 + RIGHT])))]
            c = next(c for c, s in mapping.items() if s == best)
            mapping[part], mapping[c] = mapping[c], mapping[part]
            out.append(render(best)); trace.append((i, idx, size, best, token)); idx += size
        else:
            out.append(token + special)
    return " ".join(out), key, idx, trace


def encode_codes(n):
    return [format(j, f"0{n}b") for j in range(2 ** n)]


def run(d, sc, K):
    t0, ban = time.time(), set()
    for attempt in range(MAX_RETRY + 1):
        stego, key, used, trace = optimise(d["container"], d["secret_key"], d["message"], sc, K, frozenset(ban))
        dec, _ = decode_message(stego, key, clean_output=False)
        if dec.startswith(d["message"][:used]):
            break
        bad = next((p for p, (a, b) in enumerate(zip(dec, d["message"][:used])) if a != b), len(dec))
        i, _, _, best, _ = next(t for t in trace if t[1] + t[2] > bad)
        ban.add((i, best))
    re, _ = encode_with_key(d["container"], key, d["message"])
    _, base_used = encode_with_key(d["container"], d["secret_key"], d["message"])
    return dict(uuid=d["uuid"], message=d["message"], container=d["container"], encoded_message=stego, secret_key=key,
                decoded_message=dec, used_bits=used, base_used_bits=base_used, recovered=dec.startswith(d["message"][:used]),
                reencode_ok=re == stego, retries=attempt, positions=len(trace),
                kept_original=sum(b.lower() == o.lower() for _, _, _, b, o in trace), opt_time=time.time() - t0, K=K)


def main():
    sc = Scorer()
    reps = [json.load(open(f)) for f in sorted(glob.glob(str(ROOT / "reports/gpt_omni_reports/*.json")))]
    for K in KS:
        od = HERE / f"series/opt_k{K}"; od.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for n, d in enumerate(reps):
            if (od / f"{d['uuid']}.json").exists():
                continue
            r = run(d, sc, K)
            json.dump(r, open(od / f"{d['uuid']}.json", "w"))
            if n % 25 == 0:
                print(f"K={K} {n}/{len(reps)} {time.time() - t0:.0f}s rec={r['recovered']} retries={r['retries']} t={r['opt_time']:.2f}s", flush=True)
        print(f"K={K} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
