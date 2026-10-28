"""Protocol A of 24_steganalysis_sentences.md (sentence level, random 80/20 shuffle, 3 seeds) on container vs stego,
baseline (published keys) vs GPT-2-optimised keys. Reuses experiments/steganalysis (same detectors, training, splits).
Usage: <detector venv python> steganalysis_opt.py -> steganalysis_opt.json
"""
import json, pathlib, sys, time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "steganalysis"))
import steganalysis_sentences as ss  # noqa: E402  (sets sa.MAXLEN = 64)
sa = ss.sa
OUT = HERE / "steganalysis_opt.json"


def main():
    data, skipped = {}, {}
    for k, v in (("gpt4o_base", "gpt_omni_reports"), ("gpt4o_pplopt", "../experiments/ppl_opt/series/opt_k32")):
        data[k], skipped[k] = ss.series(v)
    res = {"config": {"maxlen": sa.MAXLEN, "seeds": sa.SEEDS, "n_pairs": {k: len(v) for k, v in data.items()},
                      "identical_pairs": {k: sum(a == b for a, b in v) for k, v in data.items()}, "texts_skipped_sentence_mismatch": skipped}, "runs": []}
    print(res["config"], flush=True); t0 = time.time()
    for name, pairs in data.items():  # loop body as in steganalysis_sentences.main
        X = [a for a, _ in pairs] + [b for _, b in pairs]; y = np.r_[np.zeros(len(pairs), int), np.ones(len(pairs), int)]
        for seed in sa.SEEDS:
            tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=seed)
            tr, va = train_test_split(tr, test_size=sa.VAL_FRAC, stratify=y[tr], random_state=seed)
            sel = lambda I: [X[i] for i in I]
            v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            p = LogisticRegression(max_iter=2000, C=10).fit(v.fit_transform(sel(tr)), y[tr]).predict(v.transform(sel(te)))
            res["runs"].append({"series": name, "detector": "TF-IDF+LR", "seed": seed, "test": sa.metrics(y[te], p)})
            for det in sa.DETECTORS:
                t = time.time(); m, fz, info = sa.fit(det, sel(tr), y[tr], sel(va), y[va], seed)
                r = sa.metrics(y[te], (sa.predict(m, *fz(sel(te))) > .5).astype(int))
                res["runs"].append({"series": name, "detector": det, "seed": seed, "train": info, "test": r, "sec": round(time.time() - t, 1)})
                print(f'{name} s{seed} {det:14s} acc={r["acc"]:.3f} f1={r["f1"]:.3f} stego%={r["frac_pred_stego"]:.2f} ep={info["epochs_run"]} {time.time()-t:.0f}s', flush=True)
            json.dump(res, open(OUT, "w"), indent=1, default=float)
    res["config"]["runtime_sec"] = round(time.time() - t0)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    for name in data:
        for det in ["TF-IDF+LR"] + list(sa.DETECTORS):
            a = [r["test"]["acc"] for r in res["runs"] if r["series"] == name and r["detector"] == det]
            f1 = [r["test"]["f1"] for r in res["runs"] if r["series"] == name and r["detector"] == det]
            print(f"| {name} | {det} | {np.mean(a):.3f} ± {np.std(a, ddof=1):.3f} | {np.mean(f1):.3f} ± {np.std(f1, ddof=1):.3f} |")


if __name__ == "__main__":
    main()
