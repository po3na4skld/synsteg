"""optimize.py for the DeepStego corpus (reports/json_reports, 14.12.2023, 64/128/256 bits, slots of 2/4/8 synonyms).
Same greedy GPT-2 choice, window and ban+retry as optimize.py; only the codec differs: the 2023 encoder/decoder
(codec2023.py), because the current has_duplicates/decoder no longer skip slots whose synonyms contain the original word.
All synonyms of a slot are candidates (at most 8). Usage: <ai_detectors venv python> optimize_ds2023.py -> ../series/ds2023_opt/<uuid>.json
"""
import copy, json, os, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import codec2023 as C  # noqa: E402
from optimize import Scorer, LEFT, RIGHT, MAX_RETRY  # noqa: E402

REPORTS = HERE.parents[2] / "reports/json_reports"
OUT = HERE.parent / "series/ds2023_opt"


def optimise(container, key, bits, sc, ban=frozenset()):
    key, out, idx, trace = copy.deepcopy(key), [], 0, []
    words = C.words(container)
    for i, (token, slot) in enumerate(zip(words, key)):
        special, token = C.endswith_special(token)
        if C.replacable(token, slot) and idx < len(bits) and not C.has_duplicates(slot):
            token, cap = C.capitalization(token, slot)
            size = len(next(iter(slot[token])))
            part = bits[idx: idx + size]
            if len(part) != size:  # last block: shorten the slot in place, as the 2023 encoder did
                slot[token] = C.fix_size(slot[token], len(part)); size = len(part)
            mapping = slot[token]
            vals = list(mapping.values()); low = [v.lower() for v in vals]
            cands = [v for v in vals if low.count(v.lower()) == 1 and (i, v) not in ban]
            render = lambda s: (s.capitalize() if cap else s) + special
            best = cands[int(np.argmin(sc.nll(out[-LEFT:], [render(s) for s in cands], words[i + 1: i + 1 + RIGHT])))]
            c = next(c for c, s in mapping.items() if s == best)
            mapping[part], mapping[c] = mapping[c], mapping[part]
            out.append(render(best)); trace.append((i, idx, size, best, token)); idx += size
        else:
            out.append(token + special)
    return " ".join(out), key, idx, trace


def run(d, sc):
    t0, ban = time.time(), set()
    for attempt in range(MAX_RETRY + 1):
        stego, key, used, trace = optimise(d["container"], d["secret_key"], d["message"], sc, frozenset(ban))
        dec = C.decode(stego, key)
        if dec.startswith(d["message"][:used]):
            break
        bad = next((p for p, (a, b) in enumerate(zip(dec, d["message"][:used])) if a != b), len(dec))
        i, _, _, best, _ = next(t for t in trace if t[1] + t[2] > bad)
        ban.add((i, best))
    re, _ = C.encode(d["container"], copy.deepcopy(key), d["message"])
    _, base_used = C.encode(d["container"], copy.deepcopy(d["secret_key"]), d["message"])
    return dict(uuid=d["uuid"], message=d["message"], container=d["container"], encoded_message=stego, secret_key=key,
                decoded_message=dec, used_bits=used, base_used_bits=base_used, recovered=dec.startswith(d["message"][:used]),
                reencode_ok=re == stego, retries=attempt, positions=len(trace),
                kept_original=sum(b.lower() == o.lower() for _, _, _, b, o in trace), opt_time=time.time() - t0, K=8)


def main():
    sc = Scorer(); OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    fs = sorted(f for f in os.listdir(REPORTS) if f.endswith(".json"))
    for n, f in enumerate(fs):
        d = json.load(open(REPORTS / f))
        if (OUT / f).exists():
            continue
        r = run(d, sc); json.dump(r, open(OUT / f, "w"))
        if n % 25 == 0:
            print(f"{n}/{len(fs)} {time.time() - t0:.0f}s rec={r['recovered']} retries={r['retries']} t={r['opt_time']:.2f}s", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
