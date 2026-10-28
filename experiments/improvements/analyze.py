"""Tables for items 2-4: per arm bits/word, replaced share, GPT-2 PPL ratio stego/container (median of per-text
ratios), D1-D5 AUC and TPR at 5 % FPR against all 299 human texts, exact recovery k/n.
Usage: <ai_detectors venv python> analyze.py [arm ...] -> analyze.json + markdown on stdout"""
import json, sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from common import HERE, ARMS, DETS, load_arm


def scores():
    S = defaultdict(dict)
    for name in DETS + ["gpt2_ppl"]:
        for r in map(json.loads, open(HERE / "scores" / f"{name}.jsonl")):
            a, u, c = r["id"].split("|"); S[name][(a, u, int(c))] = r["score"]  # last line wins (re-encoded text)
    return S


def row(arm, S):
    recs = load_arm(arm); uu = [r["uuid"] for r in recs]
    cov, hum = ("ds_cover", "ds_human") if arm.startswith("ds_") else ("cover", "human")
    out = dict(n=len(recs), recovered=sum(r.get("recovered", True) for r in recs),
               bpw=float(np.mean([r["bits"] / r["n_words"] for r in recs])),
               replaced=float(np.mean([r["replaced"] for r in recs])) if recs[0].get("replaced") is not None else None,
               ppl_ratio=float(np.median([S["gpt2_ppl"][(arm, u, -1)] / S["gpt2_ppl"][(cov, u, -1)] for u in uu])))
    for d in DETS:
        h = np.array([v for (a, _, _), v in S[d].items() if a == hum])
        x = np.array([S[d][(arm, u, -1)] for u in uu])
        auc = roc_auc_score([0] * len(h) + [1] * len(x), np.r_[h, x])
        out[d[:2]] = dict(auc=round(auc, 4), tpr5=round(float(np.mean(x > np.quantile(h, .95))), 4))
    return out


def main(arms):
    S = scores(); res = {}
    print("| arm | n | rec | bits/word | replaced | PPL ratio | " + " | ".join(f"{d[:2]} AUC / TPR5" for d in DETS) + " |")
    print("|---" * (7 + len(DETS)) + "|")
    for a in arms:
        try:
            r = res[a] = row(a, S)
        except KeyError as e:
            print(a, "not scored yet", e); continue
        print(f"| {a} | {r['n']} | {r['recovered']}/{r['n']} | {r['bpw']:.3f} | {r['replaced'] if r['replaced'] is None else round(r['replaced'], 3)} | {r['ppl_ratio']:.2f} | "
              + " | ".join(f"{r[d[:2]]['auc']:.3f} / {r[d[:2]]['tpr5']:.3f}" for d in DETS) + " |")
    old = json.load(open(HERE / "analyze.json")) if (HERE / "analyze.json").exists() else {}
    old.update(res); json.dump(old, open(HERE / "analyze.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["cover"] + sorted(p.stem for p in ARMS.glob("*.jsonl") if p.stem not in ("human", "cand", "cover", "ds_human")))
