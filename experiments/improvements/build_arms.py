"""Items 2 and 3: variable bit-width (4.5), identity-code regulator (4.6), Hamming (7,3) matrix embedding.
All arms re-encode the corpus 3.7 containers with keys derived from the stored GPT-4o keys (no API calls) and are
decoded with steganography.core.decode_message. Output: arms/<arm>.jsonl + arms/build_summary.json.

Arms (message = the stored message of the run, as build_corpus.py):
  fixb4, fixb5   build_corpus.restrict_key(key, b) (sanity: equal to corpus stego_b4 / stego_b5)
  varb           b_i = min(5, floor(log2 |U_i|)), slot = first 2^b_i distinct substitutes            (4.5)
  id_b{1..5}     slot = {x_i} + first 2^b-1 distinct substitutes, x_i at a keyed code c_i = PRF(K_s, uuid, i) (4.6)
  ham73          binary slots {c_i: x_i, 1-c_i: first substitute}; Hamming (7,3) syndrome coding over consecutive
                 groups of 7 visited positions, 3 message bits per group, <= 1 change per group
The regulator arms are also encoded with a uniform random message of the same length (rand_* rows of the summary,
not scored by detectors) to check rho(b) = 1 - 2^-b, which assumes a uniform (e.g. encrypted) message.
"""
import json, math, random, sys
import numpy as np
from common import (visitable, safe_syns, AID, reports, codes, distinct_syns, untouched, prf, encode, decode_bits, walk, replaced_share,
                    write_arm)

sys.path.insert(0, str(AID))
from build_corpus import restrict_key  # noqa: E402

KS = b"synsteg-4.3-demo-secret"  # shared secret of the experiment (any value; the receiver needs the same one)
H = np.array([[(j >> r) & 1 for j in range(1, 8)] for r in range(3)])  # 3x7 parity-check matrix, column j = bin(j)


def key_varb(key, bmax=5):
    out, C = [], 0
    for slot in key:
        base, mp = next(iter(slot.items()))
        U = distinct_syns(base, mp)
        b = min(bmax, int(math.log2(len(U)))) if U else 0
        if b == 0:
            out.append(untouched(base)); continue
        out.append({base: dict(zip(codes(b), U[: 2 ** b]))}); C += b
    return out, C


def key_ident(key, b, uuid, container):
    out, C = [], 0
    vis = visitable(container, key); words = container.split()
    for i, slot in enumerate(key):
        base, mp = next(iter(slot.items()))
        U = safe_syns(words, i, distinct_syns(base, mp))
        if len(U) < 2 ** b - 1 or not vis[i]:
            out.append(untouched(base)); continue
        cs = codes(b); ci = prf(KS, "id", uuid, i) % 2 ** b
        vals = U[: 2 ** b - 1]; vals.insert(ci, base)
        out.append({base: dict(zip(cs, vals))}); C += b
    return out, C


def ham_encode(d, msg):
    key, ci = [], {}
    vis = visitable(d["container"], d["secret_key"]); words = d["container"].split()
    for i, slot in enumerate(d["secret_key"]):
        base, mp = next(iter(slot.items()))
        U = safe_syns(words, i, distinct_syns(base, mp))
        if not U or not vis[i]:
            key.append(untouched(base)); continue
        c = prf(KS, "ham", d["uuid"], i) % 2; ci[i] = c
        key.append({base: {str(c): base, str(1 - c): U[0]}})
    P = walk(d["container"], key)
    G = min(len(P) // 7, len(msg) // 3)
    v = np.array([ci[i] for i in P[: 7 * G]], int)
    for g in range(G):
        blk = v[7 * g: 7 * g + 7]
        s = (H @ blk + np.array([int(x) for x in msg[3 * g: 3 * g + 3]])) % 2
        j = int(s[0] + 2 * s[1] + 4 * s[2])  # syndrome -> column index (1..7), 0 = no change
        if j:
            blk[j - 1] ^= 1
    bits = "".join(map(str, v))
    stego, used_key, used, trace = encode(d["container"], key, bits)
    dec = decode_bits(stego, used_key)[: 7 * G]
    got = "".join("".join(map(str, (H @ np.array([int(x) for x in dec[7 * g: 7 * g + 7]])) % 2)) for g in range(G)) if len(dec) == 7 * G else ""
    return stego, 3 * G, got == msg[: 3 * G], len(P) // 7 * 3, replaced_share(d["container"], trace)


def main():
    reps = reports(); n = len(reps)
    summary, arms = {}, {}
    for d in reps:
        words = len(d["container"].split()); rnd = random.Random(d["uuid"])
        rmsg = "".join(rnd.choice("01") for _ in d["message"])
        todo = {f"fixb{b}": (restrict_key(d["secret_key"], b), None) for b in range(1, 6)}
        todo["varb"] = key_varb(d["secret_key"])
        for b in range(1, 6):
            todo[f"id_b{b}"] = key_ident(d["secret_key"], b, d["uuid"], d["container"])
        for arm, (key, C) in todo.items():
            if C is None:
                C = sum(len(next(iter(next(iter(s.values()))))) for s in key if not len(set(next(iter(s.values())).values())) < len(next(iter(s.values()))))
            for mname, msg in (("", d["message"]), ("rand_", rmsg)):
                if mname and not arm.startswith("id_"):
                    continue
                stego, k2, used, trace = encode(d["container"], key, msg)
                ok = decode_bits(stego, k2)[:used] == msg[:used]
                act = len(trace)
                chg = sum(s.lower() != o.lower() for _, _, _, s, o in trace)
                arms.setdefault(mname + arm, []).append(dict(uuid=d["uuid"], arm=mname + arm, text=stego, bits=used, n_words=words,
                    cap_bits=C, active=act, changed_active=chg, replaced=replaced_share(d["container"], trace), recovered=ok,
                    full_msg=used == len(msg)))
        for mname, msg in (("", d["message"]), ("rand_", rmsg)):
            stego, used, ok, C, rep = ham_encode(d, msg)
            arms.setdefault(mname + "ham73", []).append(dict(uuid=d["uuid"], arm=mname + "ham73", text=stego, bits=used, n_words=words,
                cap_bits=C, active=used // 3 * 7, changed_active=None, replaced=rep, recovered=ok,
                full_msg=used == len(msg)))
    corpus = {}
    for l in open(AID / "corpus.jsonl"):
        r = json.loads(l)
        if r["arm"].startswith("stego_b"):
            corpus[(r["uuid"], r["arm"])] = r["text"]
    for arm, recs in arms.items():
        if not arm.startswith("rand_"):
            write_arm(arm, recs)
        s = dict(n=len(recs), recovered=sum(r["recovered"] for r in recs),
                 bpw=float(np.mean([r["bits"] / r["n_words"] for r in recs])),
                 cap_bpw=float(np.mean([r["cap_bits"] / r["n_words"] for r in recs])),
                 replaced=float(np.mean([r["replaced"] for r in recs])),
                 full_msg=sum(r["full_msg"] for r in recs))
        if recs[0]["changed_active"] is not None:
            s["changed_per_active"] = float(sum(r["changed_active"] for r in recs) / max(1, sum(r["active"] for r in recs)))
            s["active_per_word"] = float(np.mean([r["active"] / r["n_words"] for r in recs]))
        else:  # Hamming: changes per position inside complete groups
            s["changed_per_active"] = float(sum(r["replaced"] * r["n_words"] for r in recs) / max(1, sum(r["active"] for r in recs)))
        if arm.startswith("fixb"):
            s["equal_to_corpus37"] = sum(r["text"] == corpus[(r["uuid"], "stego_b" + arm[-1])] for r in recs)
        summary[arm] = s
        print(arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
    json.dump(summary, open("arms/build_summary.json", "w"), indent=1)
    # reference arms of corpus 3.7 (human, cover, 16 candidate permutations at b = 5): detector scores come from
    # ai_detectors/scores, only the GPT-2 judge is new
    ref = {"human": [], "cover": [], "cand": []}
    for l in open(AID / "corpus.jsonl"):
        r = json.loads(l)
        a = "cand" if r["arm"] == "cand_b5" else r["arm"]
        if a in ref:
            ref[a].append(dict(uuid=r["uuid"], arm=a, cand=r.get("cand", -1), text=r["text"], bits=r["bits"], n_words=r["n_words"]))
    for a, recs in ref.items():
        write_arm(a, recs)


if __name__ == "__main__":
    main()
