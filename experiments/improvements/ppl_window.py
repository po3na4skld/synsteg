"""Item 4: perplexity window (4.7) combined with perplexity-guided synonym choice (4.8), corpus 3.7 keys at b = 5.

GPT-2 (124M) judge, ln PPL of the whole text exactly as quality.py (mean token NLL, no BOS).
  Delta_i(w) = ln PPL(x with w at i) - ln PPL(x)           on the container x (precomputed once, deltas.json)
  A_i = {w in S_i : Delta_i(w) <= delta}; A_i empty -> the word is left in place and carries no bits
        (the decoder of the working tree skips an original word left in place)
  choice: left to right, z_w = stego prefix + w + rest of the container; with g = ln PPL(z_w) - ln PPL(x)
        gamma = min : argmin g                               (4.8 with the whole text as context)
        gamma = G   : argmin |g - ln G * t/T|, t/T = share of visited positions done (linear schedule to the target)
  codes of the slot are then swapped so that phi_i(block) = chosen word (4.8); synonym sets unchanged.
Decoding: steganography.core.decode_message; on failure the choice at the failing position is banned and the text
is re-encoded (<= 10 retries, as ppl_opt/optimize.py).
Usage: <ai_detectors venv python> ppl_window.py deltas | run <delta|inf> <gamma|min> [...]  -> arms/w_d{delta}_g{gamma}.jsonl
"""
import json, math, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import HERE, reports, encode, decode_bits, walk, replaced_share, write_arm, ARMS

sys.path.insert(0, str(HERE.parents[0] / "ai_detectors"))
from build_corpus import restrict_key  # noqa: E402
sys.path.insert(0, str(HERE.parents[0] / "ppl_opt"))
from optimize import Scorer as _Scorer, LEFT, RIGHT  # noqa: E402  (the 4.8 judge window, reused)


class Scorer(_Scorer):
    """ppl_opt Scorer with candidates scored in chunks of 4 (memory rule); same values."""
    def nll(self, left_words, cands, right_words):
        return np.concatenate([super(Scorer, self).nll(left_words, cands[k:k + 4], right_words) for k in range(0, len(cands), 4)])

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DELTAS = HERE / "arms/deltas.json"
BS = 4  # memory rule: batch <= 4


class Judge:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained("openai-community/gpt2"); self.tok.pad_token = self.tok.eos_token
        self.m = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(DEV).eval()

    @torch.no_grad()
    def lnppl(self, texts):
        out = []
        for i in range(0, len(texts), BS):
            enc = self.tok(texts[i:i + BS], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
            lp = self.m(**enc).logits[:, :-1].float().log_softmax(-1)
            tgt = enc["input_ids"][:, 1:]; mask = enc["attention_mask"][:, 1:].float()
            ll = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            out += (-(ll * mask).sum(1) / mask.sum(1)).cpu().tolist()
        return np.array(out)


def slot_of(token, slot):
    from steganography import helper
    special, tok = helper.check_endswith_special(token)
    t2, cap = helper.check_capitalization(tok, slot)
    return special, t2, cap


def deltas(judge):
    out = json.load(open(DELTAS)) if DELTAS.exists() else {}
    t0 = time.time()
    for n, d in enumerate(reports()):
        if d["uuid"] in out:
            continue
        key = restrict_key(d["secret_key"], 5); words = d["container"].split()
        base = judge.lnppl([d["container"]])[0]; res = {}
        for i in walk(d["container"], key):
            special, t2, cap = slot_of(words[i], key[i])
            vals = sorted(set(key[i][t2].values()))
            texts = [" ".join(words[:i] + [(v.capitalize() if cap else v) + special] + words[i + 1:]) for v in vals]
            res[str(i)] = dict(zip(vals, (judge.lnppl(texts) - base).round(5).tolist()))
        out[d["uuid"]] = res
        if n % 25 == 0:
            print(f"deltas {n} {time.time() - t0:.0f}s", flush=True); json.dump(out, open(DELTAS, "w"))
    json.dump(out, open(DELTAS, "w"))


def skip_safe(words, i, token, vals):
    """Leaving x_i in place is decodable only if no synonym of the slot matches x_i (+ following words) at this point."""
    from steganography import helper
    s = helper.clean_container(" ".join(words[i:]))
    return not any(v.lower() == s[:len(v)].lower() and helper.check_is_end_of_word(s, len(v)) for v in vals)


def run_one(d, key, D, sc, T, delta, gamma, ban):
    """Window scoring of ppl_opt/optimize.py (Scorer.nll: 20 left tokens of the stegotext + candidate + 4 right
    container words). Running excess E = sum of window NLL(chosen) - NLL(original word), ln-ratio estimate (E + e_w)/T."""
    words = d["container"].split(); P = walk(d["container"], key)
    out, E = [], [0.0]

    def choose(i, token, mapping, part, render):
        low = [v.lower() for v in mapping.values()]
        vals = [v for v in mapping.values() if low.count(v.lower()) == 1]
        A = [v for v in vals if D[str(i)].get(v, math.inf) <= delta and (i, v) not in ban]
        if not A:
            if not skip_safe(words, i, token, vals):
                same = [v for v in vals if v.lower() == token.lower()]
                return next(c for c, s in mapping.items() if s == (same or [min(vals, key=lambda v: D[str(i)].get(v, math.inf))])[0])
            return None
        nll = sc.nll(out[-LEFT:], [render(v) for v in A] + [words[i]], words[i + 1: i + 1 + RIGHT])
        e = nll[:-1] - nll[-1]
        if gamma == "min":
            j = int(np.argmin(e))
        else:
            j = int(np.argmin(np.abs((E[0] + e) / T - math.log(float(gamma)) * (P.index(i) + 1) / len(P))))
        E[0] += e[j]
        return next(c for c, s in mapping.items() if s == A[j])

    def track(i, w):  # keeps the left context of the scorer equal to the stegotext built so far
        out.append(w)

    stego, k2, used, trace = encode(d["container"], key, d["message"], choose, track)
    return stego, k2, used, trace


def run(sc, tok, delta, gamma):
    D = json.load(open(DELTAS)); name = f"w_d{delta}_g{gamma}"
    path = ARMS / f"{name}.jsonl"
    done = [json.loads(l) for l in open(path)] if path.exists() else []
    # keep first-attempt successes; texts that needed a retry or failed are redone with the current chooser
    # (skip_safe was added after the first pass; it changes nothing for a text that decoded at the first attempt)
    done = {r["uuid"]: r for r in done if r["retries"] == 0 and r["recovered"]}
    with open(path, "w") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in done.values())
    t0 = time.time(); fh = open(path, "a")
    for n, d in enumerate(reports()):
        if d["uuid"] in done:
            continue
        key = restrict_key(d["secret_key"], 5); ban, ts = set(), time.time()
        for attempt in range(11):
            T = len(tok(d["container"]).input_ids) - 1
            stego, k2, used, trace = run_one(d, key, D[d["uuid"]], sc, T, float(delta), gamma, ban)
            dec = decode_bits(stego, k2)
            if dec[:used] == d["message"][:used]:
                break
            bad = next((p for p, (a, b) in enumerate(zip(dec, d["message"][:used])) if a != b), len(dec))
            i, _, _, w, _ = next(t for t in trace if t[1] + t[2] > bad)
            ban.add((i, w))
        r = dict(uuid=d["uuid"], arm=name, text=stego, bits=used, n_words=len(d["container"].split()), recovered=dec[:used] == d["message"][:used],
                 retries=attempt, active=len(trace), visited=len(walk(d["container"], key)), replaced=replaced_share(d["container"], trace),
                 sec=round(time.time() - ts, 2))
        fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
        if n % 50 == 0:
            print(f"{name} {n} {time.time() - t0:.0f}s rec={r['recovered']} retries={attempt}", flush=True)
    print(f"{name} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "deltas":
        deltas(Judge())
    else:
        sc = Scorer(); a = sys.argv[2:]
        for delta, gamma in zip(a[::2], a[1::2]):
            run(sc, sc.tok, delta, gamma)
