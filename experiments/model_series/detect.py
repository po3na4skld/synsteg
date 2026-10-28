"""Five AI-text detectors (experiments/ai_detectors/detectors.py, D1-D5) on containers and stegotexts of each series.
Protocol of experiments/ai_detectors/analyze.py: negatives = the 299 human (IMDB) texts of corpus.jsonl;
AUC(human vs X) and TPR at 5 % FPR (threshold = 95th percentile of human scores). Also AUC(container vs stego).
Usage: <detector venv python> detect.py name=reports_dir ...  -> reports/detectors_<names>.json
"""
import json, glob, os, sys, gc
import numpy as np, torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
AID = os.path.join(HERE, "../ai_detectors")
sys.path.insert(0, AID)
from detectors import DETECTORS

series = dict(a.split("=", 1) for a in sys.argv[1:])
human = [json.loads(l)["text"] for l in open(os.path.join(AID, "corpus.jsonl")) if '"arm": "human"' in l]
texts = {}
for name, d in series.items():
    reps = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json")))]
    reps = [r for r in reps if r["encoded_message"]]
    texts[name] = dict(L=[len(r["message"]) for r in reps], container=[r["container"] for r in reps], stego=[r["encoded_message"] for r in reps])

res = {}
ONLY = os.environ.get("DETECTORS")  # e.g. DETECTORS=D4,D5 to run a subset
for det_name, ctor in DETECTORS.items():
    if ONLY and not any(det_name.startswith(p) for p in ONLY.split(",")):
        continue
    det = ctor()
    h = np.array(det.score(human))
    thr = np.quantile(h, 0.95)
    res[det_name] = {}
    for name, t in texts.items():
        c, s, Ls = np.array(det.score(t["container"])), np.array(det.score(t["stego"])), np.array(t["L"])
        row = {}
        for L in ["all"] + sorted(set(t["L"])):
            m = np.ones(len(Ls), bool) if L == "all" else Ls == L
            auc = lambda x: float(roc_auc_score([0] * len(h) + [1] * len(x), np.concatenate([h, x])))
            row[str(L)] = dict(n=int(m.sum()), auc_container=auc(c[m]), auc_stego=auc(s[m]),
                               tpr5_container=float((c[m] > thr).mean()), tpr5_stego=float((s[m] > thr).mean()),
                               auc_container_vs_stego=float(roc_auc_score([0] * m.sum() + [1] * m.sum(), np.concatenate([c[m], s[m]]))))
        res[det_name][name] = row
        print(det_name, name, {k: round(v, 3) for k, v in row["all"].items()}, flush=True)
    del det; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
out = os.path.join(HERE, "../../reports", "detectors_" + "_".join(series) + ".json")
json.dump(res, open(out, "w"), indent=1)
print("saved", out)
