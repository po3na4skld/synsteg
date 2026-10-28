"""Sanity checks: duplicate covers across runs; shallow TF-IDF + logistic regression with the same pair splits."""
import collections, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import steganalysis as S

for series, folder in S.SERIES.items():
    pairs = S.load(folder)
    first = collections.Counter(' '.join(c.split()[:8]).lower() for c, _, _ in pairs)
    print(series, 'covers:', len(pairs), 'distinct 8-word openings:', len(first), 'top:', first.most_common(3))
    accs = []
    for seed in S.SEEDS:
        tr, va, te = S.split_pairs(pairs, seed); tr = np.concatenate([tr, va])
        Xtr, ytr, _ = S.flatten(pairs, tr); Xte, yte, _ = S.flatten(pairs, te)
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(Xtr)
        accs.append(accuracy_score(yte, LogisticRegression(C=10, max_iter=2000).fit(v.transform(Xtr), ytr).predict(v.transform(Xte))))
    print(series, 'tfidf-LR acc per seed:', np.round(accs, 4))
