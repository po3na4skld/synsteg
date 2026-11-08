"""Profiles of table 4.8 under protocol B (whole texts, split by cover-stego pair), as in the re-evaluation article
(Kuznetsov, Chernov et al., Computers, 2026): the DeepStego corpus (reports/json_reports) re-encoded with each profile
(arms/ds_*.jsonl, written by profiles.py), cover = arms/ds_cover.jsonl.
Detectors: TF-IDF (word 1-2 grams) + logistic regression, 5-fold CV grouped by pair; TS-CNN and
BERT-LSTM-Att (as in the article's protocol B) from ../steganalysis/steganalysis.py with its 80/20 pair split and seeds 0/1/2.
Usage: experiments/steganalysis/.venv_ts/bin/python profiles_protocol_b.py [--tfidf-only] -> protocol_b_summary.json
"""
import json, os, sys, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'steganalysis'))
OUT = os.path.join(HERE, 'protocol_b_summary.json')
ARMS = ['ds_base', 'ds_p1', 'ds_p2', 'ds_p3']


def read(arm):
    return {d['uuid']: d['text'] for d in map(json.loads, open(os.path.join(HERE, 'arms', f'{arm}.jsonl')))}


def tfidf(pairs, seed):
    X = [t for c, s, _ in pairs for t in (c, s)]; y = np.array([0, 1] * len(pairs)); g = np.repeat(np.arange(len(pairs)), 2)
    order = np.random.RandomState(seed).permutation(len(pairs)); g = order[g]
    p = np.zeros(len(X))
    for tr, te in GroupKFold(5).split(X, y, g):
        v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        m = LogisticRegression(C=10, max_iter=2000).fit(v.fit_transform([X[i] for i in tr]), y[tr])
        p[te] = m.predict_proba(v.transform([X[i] for i in te]))[:, 1]
    return {'acc': float(((p > .5) == y).mean()), 'auc': float(roc_auc_score(y, p))}


def main():
    cover = read('ds_cover')
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for arm in ARMS:
        st = read(arm)
        pairs = [(cover[u], st[u], 0) for u in sorted(st) if u in cover and cover[u] != st[u]]
        res.setdefault(arm, {})['n_pairs'] = len(pairs)
        res[arm]['tfidf_lr'] = [tfidf(pairs, s) for s in (0, 1, 2)]
        print(arm, len(pairs), 'tfidf', res[arm]['tfidf_lr'], flush=True)
        json.dump(res, open(OUT, 'w'), indent=1)
    if '--tfidf-only' in sys.argv:
        return
    import steganalysis as S
    for arm in ARMS:
        st = read(arm)
        pairs = [(cover[u], st[u], 0) for u in sorted(st) if u in cover and cover[u] != st[u]]
        for name in ('TS-CNN', 'BERT-LSTM-Att'):  # the two neural detectors of the article's protocol B
            for seed in S.SEEDS:
                k = f'{name}/s{seed}'
                if k in res[arm]:
                    continue
                tr, va, te = S.split_pairs(pairs, seed)
                Xtr, ytr, _ = S.flatten(pairs, tr); Xva, yva, _ = S.flatten(pairs, va); Xte, yte, Lte = S.flatten(pairs, te)
                t = time.time(); m, fz, info = S.fit(name, Xtr, ytr, Xva, yva, seed)
                res[arm][k] = {**S.evaluate(name, m, fz, Xte, yte, Lte, per_L=False), 'epochs': info['epochs_run'],
                               'sec': round(time.time() - t)}
                print(arm, k, res[arm][k], flush=True)
                json.dump(res, open(OUT, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
