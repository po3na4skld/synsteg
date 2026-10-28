"""Item 1: detector-aware key selection, criterion (4.2) vs two-sided criterion (4.3) / algorithm 4.1, with held-out
human calibration (2-fold 50/50 split of the 299 IMDB texts) and leave-one-detector-out; PRP permutations
pi_k = PRP(K_s, k) with decoding by a receiver that knows only (T, K_s, k).
Inputs: corpus 3.7 scores (ai_detectors/scores), GPT-2 judge scores (scores/gpt2_ppl.jsonl from score.py).
Usage: <ai_detectors venv python> select_key.py -> select_key.json + markdown on stdout
"""
import json, random, sys
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from common import HERE, AID, DETS, reports, encode, decode_bits, prf

sys.path.insert(0, str(AID))
import build_corpus  # noqa: E402

K, KS = 16, b"synsteg-4.3-demo-secret"


def load():
    S = {}
    for d in DETS:
        x = defaultdict(dict)
        for r in map(json.loads, open(AID / "scores" / f"{d}.jsonl")):
            x[r["arm"]][(r["uuid"], r["cand"])] = r["score"]
        S[d] = x
    P = {}
    for r in map(json.loads, open(HERE / "scores/gpt2_ppl.jsonl")):
        a, u, c = r["id"].split("|"); P[(a, u, int(c))] = r["score"]
    return S, P


def ecdf(cal):
    cal = np.sort(cal)
    return lambda s: (np.searchsorted(cal, s, "left") + np.searchsorted(cal, s, "right")) / (2 * len(cal))  # ties = 1/2


def pick(uu, crit):
    """crit(u) -> array of K values; argmin per text."""
    return {u: int(np.argmin(crit(u))) for u in uu}


def metrics(h, x):
    auc = roc_auc_score([0] * len(h) + [1] * len(x), np.r_[h, x])
    return auc, abs(auc - .5), float(np.mean(x > np.quantile(h, .95)))


def replay_keys(reps):
    """Re-create the 16 candidate keys of build_corpus.py (random.seed(0), same call order) to decode them."""
    random.seed(0); keys = {}
    for d in reps:
        keys[d["uuid"]] = [build_corpus.restrict_key(d["secret_key"], 5, permute=True) for _ in range(K)]
    return keys


def prp_key(T, k):
    """pi_k = PRP(K_s, k) on the codes of every slot: codes ordered by HMAC-SHA256(K_s, k|i|j)."""
    out = []
    for i, slot in enumerate(T):
        base, mp = next(iter(slot.items()))
        vals = list(mp.values())
        if len(set(vals)) < len(vals):
            out.append(slot); continue
        order = sorted(range(len(vals)), key=lambda j: prf(KS, "prp", k, i, j))
        out.append({base: dict(zip(mp.keys(), [vals[j] for j in order]))})
    return out


def main():
    S, P = load()
    reps = reports(); uu = [d["uuid"] for d in reps]
    corpus = {}
    for l in open(AID / "corpus.jsonl"):
        r = json.loads(l)
        if r["arm"] == "cand_b5":
            corpus[(r["uuid"], r["cand"])] = r
    # capacity: identical bits for all candidates of a text
    same_bits = sum(len({corpus[(u, k)]["bits"] for k in range(K)}) == 1 for u in uu)
    bpw = float(np.mean([corpus[(u, 0)]["bits"] / corpus[(u, 0)]["n_words"] for u in uu]))
    # recovery of every stored candidate with its replayed key
    keys = replay_keys(reps); rec = {}
    for d in reps:
        for k in range(K):
            st, k2, used, _ = encode(d["container"], keys[d["uuid"]][k], d["message"])
            assert st == corpus[(d["uuid"], k)]["text"]
            rec[(d["uuid"], k)] = decode_bits(st, k2)[:used] == d["message"][:used]
    # PRP candidates: sender and receiver share only T (canonical b = 5 table), K_s and k
    prp_ok, prp_texts = 0, defaultdict(set)
    for d in reps:
        T = build_corpus.restrict_key(d["secret_key"], 5)
        for k in range(K):
            st, _, used, _ = encode(d["container"], prp_key(T, k), d["message"])
            rx = prp_key(T, k)  # receiver side: rebuilt from (T, K_s, k) only
            prp_ok += decode_bits(st, rx)[:used] == d["message"][:used]; prp_texts[d["uuid"]].add(st)
    prp = dict(n=len(reps) * K, recovered=prp_ok, distinct_texts_per_container=float(np.mean([len(v) for v in prp_texts.values()])))

    cand = lambda d, u: np.array([S[d]["cand_b5"][(u, k)] for k in range(K)])
    allc = sorted(S[DETS[0]]["cand_b5"])
    ranks = {d: dict(zip(allc, rankdata([S[d]["cand_b5"][c] for c in allc]))) for d in DETS}
    rng = np.random.default_rng(0); perm = rng.permutation(len(uu))
    halves = [[uu[i] for i in perm[:150]], [uu[i] for i in perm[150:]]]
    rows = defaultdict(lambda: defaultdict(list))  # name -> det -> [(auc, dev, tpr) per fold]
    sels = defaultdict(list)
    for f in range(2):
        cal, ev = halves[f], halves[1 - f]
        F = {d: ecdf(np.array([S[d]["human"][(u, -1)] for u in cal])) for d in DETS}
        Hev = {d: np.array([S[d]["human"][(u, -1)] for u in ev]) for d in DETS}
        two = lambda G: (lambda u: np.mean([np.abs(F[d](cand(d, u)) - .5) for d in G], 0))
        choice = {"rank-mean (4.2), D1-D5": pick(uu, lambda u: np.mean([[ranks[d][(u, k)] for k in range(K)] for d in DETS], 0)),
                  "two-sided (4.3), D1-D5": pick(uu, two(DETS))}
        for d in DETS:
            choice[f"two-sided (4.3), {d[:2]} only"] = pick(uu, two([d]))
        for name, ch in choice.items():
            sels[name].append(ch)
            for d in DETS:
                rows[name][d].append(metrics(Hev[d], np.array([S[d]["cand_b5"][(u, ch[u])] for u in uu])))
        for d in DETS:
            rows["published key (stego_b5)"][d].append(metrics(Hev[d], np.array([S[d]["stego_b5"][(u, -1)] for u in uu])))
            rows["random permutation (mean of k = 0..15)"][d].append(tuple(np.mean(
                [metrics(Hev[d], np.array([S[d]["cand_b5"][(u, k)] for u in uu])) for k in range(K)], 0)))
            ch = pick(uu, two([g for g in DETS if g != d]))  # leave-one-detector-out
            sels[f"LODO held-out {d[:2]}"].append(ch)
            rows["two-sided (4.3), LODO: 4 guides, held-out row"][d].append(metrics(Hev[d], np.array([S[d]["cand_b5"][(u, ch[u])] for u in uu])))
            ch = pick(uu, lambda u: np.mean([[ranks[g][(u, k)] for k in range(K)] for g in DETS if g != d], 0))
            rows["rank-mean (4.2), LODO"][d].append(metrics(Hev[d], np.array([S[d]["cand_b5"][(u, ch[u])] for u in uu])))
    ratio = lambda texts: float(np.median([P[t] / P[("cover", u, -1)] for t, u in texts]))
    out = {}
    for name, per in rows.items():
        r = {d: dict(zip(("auc", "dev", "tpr5"), np.mean(v, 0).round(4).tolist())) for d, v in per.items()}
        if name in sels:
            ch_all = sels[name]
            r["ppl_ratio"] = round(float(np.mean([ratio([(("cand", u, ch[u]), u) for u in uu]) for ch in ch_all])), 3)
            r["recovered"] = int(np.mean([sum(rec[(u, ch[u])] for u in uu) for ch in ch_all]))
        elif name.startswith("published"):
            r["ppl_ratio"] = round(ratio([(("fixb5", u, -1), u) for u in uu]), 3)
        elif name.startswith("random"):
            r["ppl_ratio"] = round(float(np.mean([ratio([(("cand", u, k), u) for u in uu]) for k in range(K)])), 3)
            r["recovered"] = round(sum(rec.values()) / K, 1)
        elif "LODO" in name and name.startswith("two"):
            r["ppl_ratio"] = {d[:2]: round(float(np.mean([ratio([(("cand", u, ch[u]), u) for u in uu]) for ch in sels[f"LODO held-out {d[:2]}"]])), 3) for d in DETS}
        r["mean_dev"] = round(float(np.mean([r[d]["dev"] for d in DETS])), 4)
        out[name] = r
    res = dict(n_stego=len(uu), n_human_eval=[len(h) for h in halves], K=K, bpw_all_candidates=bpw, same_bits_all_candidates=same_bits,
               candidates_recovered=f"{sum(rec.values())}/{len(rec)}", prp=prp, rows=out)
    json.dump(res, open(HERE / "select_key.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    print("| selection | " + " | ".join(f"{d[:2]} AUC / dev / TPR5" for d in DETS) + " | mean dev | PPL ratio | rec |")
    for name, r in out.items():
        print(f"| {name} | " + " | ".join(f'{r[d]["auc"]:.3f} / {r[d]["dev"]:.3f} / {r[d]["tpr5"]:.3f}' for d in DETS) + f' | {r["mean_dev"]:.3f} | {r.get("ppl_ratio")} | {r.get("recovered")} |')


if __name__ == "__main__":
    main()
