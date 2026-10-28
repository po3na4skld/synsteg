"""Configuration profiles built from the item 1-4 measurements, applied to two corpora:
  c37  corpus 3.7 (299 GPT-4o runs of reports/gpt_omni_reports, b = 5 tables)  -> D1-D5, GPT-2 PPL ratio
  ds   DeepStego corpus (300 runs of reports/json_reports, 2023 keys, 1-2 bits per slot, 64/128/256-bit messages)
       -> sentence pairs for protocol A' steganalysis (run_steganalysis.py), D1-D5, GPT-2 PPL ratio
Mechanisms (combined per profile):
  ham    Hamming (7,3) syndrome coding over visited positions, binary slot {c_i: x_i, 1-c_i: s_i}, 3 bits / 7 positions
  ident  identity code: x_i at the keyed code c_i plus 2^b - 1 substitutes (4.6); b_i lowered when fewer pass the window
  delta  window (4.7): only substitutes with Delta_i(w) <= delta (GPT-2, whole container) are used; the substitute(s)
         put in a slot are those with the lowest Delta_i
  gamma  'min' = choice by the 4.8 window judge among the slot's substitutes (code swap), None = no choice
Every stegotext is decoded with steganography.core.decode_message (Hamming: + syndrome), recovery k/n is reported.
Usage: <ai_detectors venv python> profiles.py deltas_ds | build <profile ...>
"""
import json, math, os, sys, time
import numpy as np
from common import (visitable, safe_syns, HERE, ROOT, reports, codes, distinct_syns, untouched, prf, encode, decode_bits, walk, replaced_share,
                    write_arm, ARMS)
from build_arms import H, KS
from ppl_window import Judge, Scorer, LEFT, RIGHT, restrict_key, run_one

sys.path.insert(0, str(HERE.parents[0] / "ppl_opt/ds2023"))
import codec2023 as C  # noqa: E402

DS_DELTAS = ARMS / "deltas_ds.json"
DATA = ROOT / "datasets/textsteganalysis_runs"
PROFILES = {  # chosen from the item 1-4 measurements, see 33_improvements_4_3.md
    "p1": dict(ham=True, delta=math.inf, gamma="min"),           # against steganalysis: fewest changes
    "p2": dict(window=True, delta=0.02, gamma="2"),              # against AI-text detectors: grid optimum w_d0.02_g2
    "p3": dict(ident=True, bmax=2, delta=0.05, gamma="min"),     # balanced: identity code + window + 4.8 choice
}


def ds_reports():
    R = ROOT / "reports/json_reports"
    out = []
    for f in (f for f in os.listdir(R) if f.endswith(".json")):  # os.listdir order, as ppl_opt/ds2023/corpus.py
        d = json.load(open(R / f)); d["_file"] = f
        d["container_w"] = " ".join(C.words(d["container"]))
        out.append(d)
    return out


def deltas_ds():
    """REV=1: second worker from the end of the list into deltas_ds_rev.json (merged by 'merge_ds')."""
    rev = os.environ.get("REV") == "1"; path = ARMS / "deltas_ds_rev.json" if rev else DS_DELTAS
    judge = Judge(); out = json.load(open(path)) if path.exists() else {}; t0 = time.time()
    reps = ds_reports()[::-1] if rev else ds_reports()
    for n, d in enumerate(reps):
        other = json.load(open(DS_DELTAS if rev else ARMS / "deltas_ds_rev.json")) if (DS_DELTAS if rev else ARMS / "deltas_ds_rev.json").exists() else {}
        if d["uuid"] in out or d["uuid"] in other:
            continue
        x = d["container_w"]; words = x.split(); base = judge.lnppl([x])[0]; res = {}
        from steganography import helper
        jobs = []  # all (position, substitute) texts of the document in one batched pass
        for i, (tok, slot) in enumerate(zip(words, d["secret_key"])):
            special, t = helper.check_endswith_special(tok)
            if not helper.is_token_replacable(t, slot):
                continue
            t2, cap = helper.check_capitalization(t, slot)
            for v in distinct_syns(*next(iter(slot.items()))):
                jobs.append((str(i), v, " ".join(words[:i] + [(v.capitalize() if cap else v) + special] + words[i + 1:])))
        vals = judge.lnppl([t for _, _, t in jobs]) - base if jobs else []
        for (i, v, _), x in zip(jobs, vals):
            res.setdefault(i, {})[v] = round(float(x), 5)
        out[d["uuid"]] = res
        print(f"deltas_ds {n} {time.time() - t0:.0f}s", flush=True)
        json.dump(out, open(path, "w"))
    json.dump(out, open(path, "w"))


def merge_ds():
    a = json.load(open(DS_DELTAS)); a.update(json.load(open(ARMS / "deltas_ds_rev.json")))
    json.dump(a, open(DS_DELTAS, "w")); print("deltas_ds", len(a))


def ranked(U, Di, delta):
    return sorted([w for w in U if Di.get(w, math.inf) <= delta], key=lambda w: Di[w])


def encode_profile(container, raw_key, msg, uuid, spec, D, sc, T):
    """Returns (stego, bits used, recovered, replaced share, capacity bits). Decoding failures (multi-word prefix
    ambiguity) ban the chosen substitute at the failing position and re-encode, <= 10 times (as ppl_opt)."""
    ban = set()
    if spec.get("window"):  # item 4 chooser (ppl_window.run_one), same ban/retry
        d = dict(container=container, message=msg)
        for attempt in range(11):
            stego, k2, used, trace = run_one(d, raw_key, D, sc, T, spec["delta"], spec["gamma"], ban)
            dec = decode_bits(stego, k2)
            if dec[:used] == msg[:used]:
                break
            bad = next((p for p, (a, b) in enumerate(zip(dec, msg[:used])) if a != b), len(dec))
            i, _, _, w, _ = next(t for t in trace if t[1] + t[2] > bad)
            ban.add((i, w))
        C_bits = sum(len(next(iter(next(iter(s.values()))))) for s in raw_key if len(set(next(iter(s.values())).values())) == len(next(iter(s.values()))))
        return stego, used, dec[:used] == msg[:used], replaced_share(container, trace), C_bits
    for attempt in range(11):
        r = _encode_profile(container, raw_key, msg, uuid, spec, D, sc, T, ban)
        if r[2] or r[5] is None:
            return r[:5]
        ban.add(r[5])
    return r[:5]


def _encode_profile(container, raw_key, msg, uuid, spec, D, sc, T, ban):
    words = container.split(); out = []
    track = lambda i, w: out.append(w)

    def best(i, cands, render):
        cands = [v for v in cands if (i, v) not in ban] or cands
        if len(cands) == 1 or spec.get("gamma") is None:
            return cands[0]
        return cands[int(np.argmin(sc.nll(out[-LEFT:], [render(v) for v in cands], words[i + 1: i + 1 + RIGHT])))]

    def first_bad(dec, want, trace):
        p = next((k for k, (x, y) in enumerate(zip(dec, want)) if x != y), None)
        if p is None and len(dec) >= len(want):
            return None
        p = len(dec) if p is None else p
        t = next((t for t in trace if t[1] + t[2] > p), None)
        return None if t is None or t[3].lower() == t[4].lower() else (t[0], t[3])

    vis = visitable(container, raw_key)
    if spec.get("ham"):
        key, ci, A = [], {}, {}
        for i, slot in enumerate(raw_key):
            base, mp = next(iter(slot.items()))
            A[i] = ranked(safe_syns(words, i, distinct_syns(base, mp)), D.get(str(i), {}), spec["delta"]) if spec["delta"] < math.inf else safe_syns(words, i, distinct_syns(base, mp))
            if not A[i] or not vis[i]:
                key.append(untouched(base)); continue
            c = prf(KS, "ham", uuid, i) % 2; ci[i] = c
            key.append({base: {str(c): base, str(1 - c): A[i][0]}})
        P = walk(container, key); G = min(len(P) // 7, len(msg) // 3)
        v = np.array([ci[i] for i in P[: 7 * G]], int)
        for g in range(G):
            s = (H @ v[7 * g: 7 * g + 7] + np.array([int(x) for x in msg[3 * g: 3 * g + 3]])) % 2
            j = int(s[0] + 2 * s[1] + 4 * s[2])
            if j:
                v[7 * g + j - 1] ^= 1
        bits = "".join(map(str, v))

        def choose(i, token, mapping, part, render):
            if mapping[part].lower() != token.lower():  # the flipped position: substitute chosen by the 4.8 judge
                mapping[part] = best(i, A[i], render)
            return part

        stego, k2, used, trace = encode(container, key, bits, choose, track)
        dec = decode_bits(stego, k2)[: 7 * G]
        got = "".join("".join(map(str, (H @ np.array([int(x) for x in dec[7 * g: 7 * g + 7]])) % 2)) for g in range(G)) if len(dec) == 7 * G else ""
        ok = got == msg[: 3 * G]
        return stego, 3 * G, ok, replaced_share(container, trace), len(P) // 7 * 3, None if ok else first_bad(dec, bits, trace)
    key, C_bits = [], 0
    for i, slot in enumerate(raw_key):
        base, mp = next(iter(slot.items()))
        b = min(len(next(iter(mp))), spec.get("bmax", 99))
        if spec.get("ident"):
            A = ranked(safe_syns(words, i, distinct_syns(base, mp)), D.get(str(i), {}), spec["delta"]) if spec["delta"] < math.inf else safe_syns(words, i, distinct_syns(base, mp))
            b = min(b, int(math.log2(len(A) + 1)))
            if b == 0 or not vis[i]:
                key.append(untouched(base)); continue
            vals = A[: 2 ** b - 1]; vals.insert(prf(KS, "id", uuid, i) % 2 ** b, base)
            key.append({base: dict(zip(codes(b), vals))}); C_bits += b
        else:
            key.append(slot); C_bits += b * (len(set(mp.values())) == len(mp))

    def choose(i, token, mapping, part, render):
        if mapping[part].lower() == token.lower() or spec.get("gamma") is None:
            return part  # identity code keeps x_i; no choice requested
        vals = [v for v in mapping.values() if v.lower() != token.lower()]
        w = best(i, vals, render)
        return next(c for c, s in mapping.items() if s == w)

    stego, k2, used, trace = encode(container, key, msg, choose, track)
    dec = decode_bits(stego, k2)
    ok = dec[:used] == msg[:used]
    return stego, used, ok, replaced_share(container, trace), C_bits, None if ok else first_bad(dec, msg[:used], trace)


def build(names):
    sc = Scorer()
    c37 = json.load(open(ARMS / "deltas.json")); ds = json.load(open(DS_DELTAS)) if DS_DELTAS.exists() else {}
    for name in names:
        spec = PROFILES[name]; t0 = time.time()
        for corpus, reps, DD in (("c37", reports(), c37), ("ds", ds_reports(), ds)):
            if os.environ.get("CORPUS", corpus) != corpus:
                continue
            recs, pairs = [], []
            for d in reps:
                x = d["container"] if corpus == "c37" else d["container_w"]
                # window profile walks the b = 5 tables of corpus 3.7 (as item 4); Hamming / identity build their own slots
                raw = restrict_key(d["secret_key"], 5) if corpus == "c37" and spec.get("window") else d["secret_key"]
                T = len(sc.tok(x).input_ids) - 1
                st, used, ok, rep, Cb = encode_profile(x, raw, d["message"], d["uuid"], spec, DD.get(d["uuid"], {}), sc, T)
                recs.append(dict(uuid=d["uuid"], arm=f"{corpus}_{name}", text=st, bits=used, n_words=len(x.split()), recovered=ok,
                                 replaced=rep, cap_bits=Cb, full_msg=used == len(d["message"])))
                if corpus == "ds":
                    c_s, s_s = d["container"].split("."), st.split(".")
                    pairs += [(a.strip(), b.strip()) for a, b in zip(c_s, s_s) if a.strip() and b.strip()]
            write_arm(f"{corpus}_{name}", recs)
            if corpus == "ds":
                write_pairs(f"imp_{name}", pairs)
            print(f"{corpus}_{name}: n={len(recs)} rec={sum(r['recovered'] for r in recs)} bpw={np.mean([r['bits'] / r['n_words'] for r in recs]):.3f} "
                  f"replaced={np.mean([r['replaced'] for r in recs]):.3f} {time.time() - t0:.0f}s", flush=True)


def write_pairs(arm, pairs):
    d = DATA / arm; d.mkdir(parents=True, exist_ok=True)
    open(d / "clean.txt", "w").write("\n".join(a.replace("\n", " ") for a, _ in pairs))
    open(d / "stego.txt", "w").write("\n".join(b.replace("\n", " ") for _, b in pairs))
    print(arm, "pairs", len(pairs), "identical", sum(a == b for a, b in pairs), flush=True)


def base_ds():
    """Control arm: published 2023 stegotexts (sentence pairs byte-identical to artifacts/synsteg_*), human texts for D1-D5."""
    import csv, re, copy
    reps = ds_reports(); recs, pairs = [], []
    for d in reps:
        st, used = C.encode(d["container"], copy.deepcopy(d["secret_key"]), d["message"])
        assert st == d["encoded_message"]
        dec = C.decode(st, copy.deepcopy(d["secret_key"]))
        chg = sum(a.lower() != b.lower() for a, b in zip(d["container_w"].split(), st.split())) if len(st.split()) == len(d["container_w"].split()) else None
        recs.append(dict(uuid=d["uuid"], arm="ds_base", text=st, bits=used, n_words=len(d["container_w"].split()), recovered=dec.startswith(d["message"][:used]),
                         replaced=None if chg is None else chg / len(d["container_w"].split()), full_msg=used == len(d["message"])))
        c_s, s_s = d["container"].split("."), st.split(".")
        pairs += [(a.strip(), b.strip()) for a, b in zip(c_s, s_s) if a.strip() and b.strip()]
    write_arm("ds_base", recs); write_pairs("imp_base", pairs)
    cl = [l.strip() for l in open(ROOT / "artifacts/synsteg_clean.txt")]; sl = [l.strip() for l in open(ROOT / "artifacts/synsteg_stego.txt")]
    print("base pairs equal to artifacts/synsteg_*:", pairs == [(a, b) for a, b in zip(cl, sl) if a and b])
    write_arm("ds_cover", [dict(uuid=d["uuid"], arm="ds_cover", text=d["container"], bits=0, n_words=len(d["container_w"].split())) for d in reps])
    with open(ROOT / "datasets/IMDB_sample.csv", newline="") as fh:
        reviews = [re.sub(r"\s+", " ", re.sub(r"<br\s*/?>", " ", r["review"])).strip() for r in csv.DictReader(fh)]
    write_arm("ds_human", [dict(uuid=d["uuid"], arm="ds_human", text=" ".join(rev.split()[: len(d["container_w"].split())]), bits=0, n_words=0)
                           for d, rev in zip(reps, reviews)])


if __name__ == "__main__":
    {"deltas_ds": lambda a: deltas_ds(), "merge_ds": lambda a: merge_ds(), "build": build, "base": lambda a: base_ds()}[sys.argv[1]](sys.argv[2:])
