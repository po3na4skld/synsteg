"""Stage 1 check of the setup against the old appendix: Table B.1 (no-attack null check) and the WordNet rows of
Table B.2 (b = 1..3) with the random-replacement control at the same per-text number of replacements.

WordNet lexicon (reconstruction; the old code is lost): for an alphabetic word, the word itself followed by the
distinct lemma names of all its synsets in WordNet order (so code 0 keeps the word). A word with >= 2^b entries is a
position; b message bits pick one of the first 2^b. Of the variants tried (stopwords dropped or not, word first or
WordNet order) this one is closest to the old b = 1 row (0.289 replaced, 0.57 bits per word). Applied to both classes.
-> results/wordnet_check.json
"""
import json, random
import numpy as np
from nltk.corpus import wordnet as wn
from common import HERE, load, MeanG, PPL, split_special, match_case, random_control, evaluate, paired

_wn = {}


def lex(word):
    w = word.lower()
    if w not in _wn:
        names = [w]
        if w.isalpha():
            for s in wn.synsets(w):
                for n in s.lemma_names():
                    n = n.replace("_", " ")
                    if n.lower() not in [x.lower() for x in names]:
                        names.append(n)
        _wn[w] = names
    return _wn[w]


def wordnet_embed(text, b, rng):
    toks, out, bits, rep = text.split(), [], 0, 0
    for t in toks:
        w, sp = split_special(t)
        c = lex(w)
        if len(c) >= 2 ** b:
            new = c[rng.randrange(2 ** b)]  # b random message bits
            bits += b
            if new.lower() != w.lower():
                rep += 1
                w = match_case(new, w)
        out.append(w + sp)
    return " ".join(out), rep, bits


def main():
    wm, hu = load("wm.jsonl"), load("human.jsonl")
    G, P = MeanG(), PPL()
    res = {}
    g0 = {k: [G(r["text"]) for r in rows] for k, rows in (("wm", wm), ("hu", hu))}
    nw = [len(r["text"].split()) for r in wm]
    for k, n in (("wm", wm), ("hu", hu)):
        g = np.array(g0[k]); se = g.std(ddof=1) / np.sqrt(len(g))
        res[f"B1_{k}"] = dict(n=len(g), mean_g=g.mean(), sd=g.std(ddof=1), se=se, z=(g.mean() - 0.5) / se,
                              words=np.mean([len(r["text"].split()) for r in n]))
    base = float(np.mean(g0["wm"]))
    ppl0 = [P(r["text"]) for r in wm]
    rows = [evaluate("no attack", g0["wm"], g0["hu"], base, nw, ppl0, [0] * len(wm), [0] * len(wm))]
    per_text = {}
    for b in (1, 2, 3):
        rng = random.Random(b)
        att = {k: [wordnet_embed(r["text"], b, rng) for r in rows_] for k, rows_ in (("wm", wm), ("hu", hu))}
        ctl = {k: [random_control(r["text"], a[1], lambda i, w: lex(w), rng) for r, a in zip(rows_, att[k])]
               for k, rows_ in (("wm", wm), ("hu", hu))}
        for name, A in ((f"WordNet SynSteg b={b}", att), (f"WordNet random b={b}", ctl)):
            gp, gn = [G(t[0]) for t in A["wm"]], [G(t[0]) for t in A["hu"]]
            ppl = [P(t[0]) for t in A["wm"]]
            bits = [t[2] for t in A["wm"]] if "SynSteg" in name else [0] * len(wm)
            rows.append(evaluate(name, gp, gn, base, nw, ppl, [t[1] for t in A["wm"]], bits))
            per_text[name] = gp
            print(rows[-1], flush=True)
        res[f"B3_b{b}"] = paired(per_text[f"WordNet SynSteg b={b}"], per_text[f"WordNet random b={b}"], base)
    res["B2"] = rows
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(res, open(HERE / "results/wordnet_check.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "B2"}, indent=1, default=float))


if __name__ == "__main__":
    main()
