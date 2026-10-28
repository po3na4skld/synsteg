"""Aggregate detector scores into the tables used in the dissertation (chapter 3.7 / 4.3).

For every detector and arm: AUC(human vs arm), TPR at 5% FPR (threshold = 95th percentile of human scores),
mean score, and the paired bootstrap CI of (arm - cover) mean score.
Detector-aware key selection: guide = D1 (cheapest); stego_sel = per-uuid candidate with the lowest ensemble (mean-rank) score across all detectors;
stego_rand = candidate 0 (a random permutation, the control). Held-out detectors D2..D5 measure transfer.
"""
import json, pathlib, random, sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
GUIDE = "D1_roberta_openai"
ARMS = ["cover", "stego_b1", "stego_b2", "stego_b3", "stego_b4", "stego_b5", "stego_rand", "stego_sel"]


def load():
    S = {}
    for f in sorted((HERE / "scores").glob("*.jsonl")):
        d = defaultdict(dict)
        for l in open(f):
            r = json.loads(l)
            d[r["arm"]][(r["uuid"], r["cand"])] = r["score"]
        S[f.stem] = d
    return S


def select(S, guide=GUIDE):
    """Return {uuid: cand} chosen by the guide detector (lowest score = least AI-like)."""
    if guide == "ensemble":  # mean rank over all complete detectors
        from scipy.stats import rankdata
        dets = [k for k in S if "cand_b5" in S[k] and len(S[k]["cand_b5"]) == len(S[GUIDE]["cand_b5"])]
        keys = sorted(S[GUIDE]["cand_b5"])
        ranks = np.mean([rankdata([S[k]["cand_b5"][key] for key in keys]) for k in dets], axis=0)
        g = dict(zip(keys, ranks))
    else:
        g = S[guide]["cand_b5"]
    best = {}
    for (u, c), s in g.items():
        if u not in best or s < best[u][1]:
            best[u] = (c, s)
    return {u: c for u, (c, s) in best.items()}


def tpr_at_fpr(h, x, fpr=0.05):
    thr = np.quantile(h, 1 - fpr)
    return float(np.mean(np.array(x) > thr))


def boot_ci(diff, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    diff = np.array(diff)
    m = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return float(np.mean(diff)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    S = load()
    corpus = {}
    for l in open(HERE / "corpus.jsonl"):
        r = json.loads(l)
        corpus[(r["uuid"], r["arm"], r.get("cand", -1))] = r
    uuids = sorted({u for (u, a, c) in corpus if a == "human"})
    sel = select(S, "ensemble")  # the proposed procedure: ensemble guide
    rows = []
    for det, d in S.items():
        if "human" not in d or len(d["human"]) < len(uuids):
            continue
        h = np.array([d["human"][(u, -1)] for u in uuids])
        cover = np.array([d["cover"][(u, -1)] for u in uuids])
        for arm in ARMS:
            if arm == "stego_sel":
                if "cand_b5" not in d: continue
                x = np.array([d["cand_b5"][(u, sel[u])] for u in uuids])
            elif arm == "stego_rand":
                if "cand_b5" not in d: continue
                x = np.array([d["cand_b5"][(u, 0)] for u in uuids])
            else:
                if arm not in d: continue
                x = np.array([d[arm][(u, -1)] for u in uuids])
            auc = roc_auc_score([0] * len(h) + [1] * len(x), np.concatenate([h, x]))
            m, lo, hi = boot_ci(x - cover)
            rows.append(dict(detector=det, arm=arm, auc=round(auc, 4), tpr5=round(tpr_at_fpr(h, x), 4),
                             mean=round(float(x.mean()), 4), d_vs_cover=round(m, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             mean_human=round(float(h.mean()), 4)))
    # bits per word per arm
    bpw = {}
    for arm in ["stego_b1", "stego_b2", "stego_b3", "stego_b4", "stego_b5"]:
        recs = [corpus[(u, arm, -1)] for u in uuids]
        bpw[arm] = dict(bits=float(np.mean([r["bits"] for r in recs])), bpw=float(np.mean([r["bits"] / r["n_words"] for r in recs])),
                        edit_rate=float(np.mean([sum(a != b for a, b in zip(r["text"].split(), corpus[(u, "cover", -1)]["text"].split())) / r["n_words"] for u, r in zip(uuids, recs)])))
    # transfer matrix: guide detector (rows) x evaluated detector (cols), AUC of stego_sel vs human
    complete = [k for k in S if "cand_b5" in S[k] and len(S[k]["human"]) == len(uuids)]
    matrix = {}
    for guide in complete + ["ensemble"]:
        sg = select(S, guide)
        matrix[guide] = {}
        for det in complete:
            d = S[det]
            h = np.array([d["human"][(u, -1)] for u in uuids])
            x = np.array([d["cand_b5"][(u, sg[u])] for u in uuids])
            r = np.array([d["cand_b5"][(u, 0)] for u in uuids])
            matrix[guide][det] = dict(auc_sel=round(roc_auc_score([0] * len(h) + [1] * len(x), np.concatenate([h, x])), 4),
                                      auc_rand=round(roc_auc_score([0] * len(h) + [1] * len(r), np.concatenate([h, r])), 4),
                                      tpr5_sel=round(tpr_at_fpr(h, x), 4), tpr5_rand=round(tpr_at_fpr(h, r), 4))
    json.dump(dict(rows=rows, bpw=bpw, selected=sel, matrix=matrix), open(HERE / "results.json", "w"), indent=1)
    print("\ntransfer matrix (AUC sel / rand):")
    for g, row in matrix.items():
        print(f"{g:>20}", "  ".join(f"{det[:6]} {v['auc_sel']:.3f}/{v['auc_rand']:.3f}" for det, v in row.items()))
    import pandas as pd
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    for det, g in df.groupby("detector"):
        print("\n", det, "  mean human =", g.mean_human.iloc[0])
        print(g.drop(columns=["detector", "mean_human"]).to_string(index=False))
    print("\nbits/word", json.dumps(bpw, indent=1))


if __name__ == "__main__":
    main()
