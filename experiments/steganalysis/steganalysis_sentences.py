"""Sentence-level steganalysis with a random text shuffle, as in the DeepStego study (cover=0 vs stego=1).

Same four detectors and training loop as steganalysis.py; differences: one sample = one sentence, and the
80/20 split shuffles individual sentences (a cover and its stego can land in different parts).
Corpora: artifacts/synsteg_{clean,stego}.txt (old 8 662-sentence corpus), artifacts/synsteg_gptomni_{clean,stego}.txt,
gpt-6-luna and Qwen series split into sentences (pairs kept only when cover and stego split into the same count).
Usage: python steganalysis_sentences.py   -> ../../reports/steganalysis_sentences.json
"""
import glob, json, re, time, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import steganalysis as sa

sa.MAXLEN = 64  # sentences: 40 words at most in the old corpus
BASE = os.environ.get('SYNSTEG_ROOT', os.path.expanduser('~/PycharmProjects/Personal/SynSteg'))
OUT = f'{BASE}/reports/steganalysis_sentences.json'
SENT = re.compile(r'(?<=[.!?])\s+')


def lines(name):
    c = [l.strip() for l in open(f'{BASE}/artifacts/{name}_clean.txt')]
    s = [l.strip() for l in open(f'{BASE}/artifacts/{name}_stego.txt')]
    return [(a, b) for a, b in zip(c, s) if a and b]


def series(folder):
    pairs, skipped = [], 0
    for f in sorted(glob.glob(f'{BASE}/reports/{folder}/*.json')):
        r = json.load(open(f))
        if not r.get('encoded_message'): continue
        c, s = SENT.split(r['container'].strip()), SENT.split(r['encoded_message'].strip())
        if len(c) != len(s): skipped += 1; continue
        pairs += [(a, b) for a, b in zip(c, s) if a and b]
    return pairs, skipped


def main():
    data, skipped = {'old_8662': lines('synsteg'), 'gpt4o_1231': lines('synsteg_gptomni')}, {}
    for k, v in (('gpt6_luna_b5', 'gpt6_luna_missing'), ('qwen3_30b_b3', 'qwen_missing_b3')):
        data[k], skipped[k] = series(v)
    res = {'config': {'maxlen': sa.MAXLEN, 'seeds': sa.SEEDS, 'split': 'random shuffle of sentences 80/20, val 15% of train',
                      'n_pairs': {k: len(v) for k, v in data.items()},
                      'identical_pairs': {k: sum(a == b for a, b in v) for k, v in data.items()},
                      'texts_skipped_sentence_mismatch': skipped}, 'runs': []}
    print(res['config'], flush=True); t0 = time.time()
    for name, pairs in data.items():
        X = [a for a, _ in pairs] + [b for _, b in pairs]; y = np.r_[np.zeros(len(pairs), int), np.ones(len(pairs), int)]
        for seed in sa.SEEDS:
            idx = np.arange(len(X))
            tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
            tr, va = train_test_split(tr, test_size=sa.VAL_FRAC, stratify=y[tr], random_state=seed)
            sel = lambda I: [X[i] for i in I]
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            p = LogisticRegression(max_iter=2000, C=10).fit(v.fit_transform(sel(tr)), y[tr]).predict(v.transform(sel(te)))
            res['runs'].append({'series': name, 'detector': 'TF-IDF+LR', 'seed': seed, 'test': sa.metrics(y[te], p)})
            for det in sa.DETECTORS:
                t = time.time(); m, fz, info = sa.fit(det, sel(tr), y[tr], sel(va), y[va], seed)
                r = sa.metrics(y[te], (sa.predict(m, *fz(sel(te))) > .5).astype(int))
                res['runs'].append({'series': name, 'detector': det, 'seed': seed, 'train': info, 'test': r, 'sec': round(time.time() - t, 1)})
                print(f'{name} s{seed} {det:14s} acc={r["acc"]:.3f} f1={r["f1"]:.3f} stego%={r["frac_pred_stego"]:.2f} ep={info["epochs_run"]} {time.time()-t:.0f}s', flush=True)
            json.dump(res, open(OUT, 'w'), indent=1, default=float)
    res['config']['runtime_sec'] = round(time.time() - t0)
    json.dump(res, open(OUT, 'w'), indent=1, default=float)
    print('wrote', OUT, res['config']['runtime_sec'], 's')


if __name__ == '__main__':
    main()
