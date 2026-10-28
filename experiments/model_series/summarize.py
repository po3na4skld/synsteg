"""Summarise <series>_metrics.jsonl files: mean ± SD per message length. Usage: summarize.py name=path.jsonl ..."""
import json, sys
import numpy as np


def ms(x, p=2):
    x = np.array([v for v in x if v is not None], float)
    return f"{x.mean():.{p}f} ± {x.std():.{p}f}" if len(x) else "n/a"


def summary(rows):
    out = {}
    for L in sorted({r["L"] for r in rows}):
        R = [r for r in rows if r["L"] == L]
        G = [r for r in R if "exact" in r]
        Ls = np.mean([s for r in G for s in r["sents"]]); mc = np.mean([t for r in G for t in r["toklens"]]); k = np.mean([r["k"] for r in G])
        out[L] = dict(
            n=len(R), failed=len(R) - len(G), errors=sorted({r["error"][:120] for r in R if r["error"]}),
            exact=f'{sum(r["exact"] for r in G)}/{len(R)}', embedded=ms([100 * r["embedded"] for r in G], 1),
            words=ms([r["words"] for r in G]), bpw=ms([r["bpw"] for r in G]),
            capacity=ms([r["capacity"] for r in G]), capacity_pooled=f"{(Ls - 1) * k / (8 * Ls * mc) * 100:.2f}",
            t_container=ms([r["t_container"] for r in G]), t_key=ms([r["t_key"] for r in G]),
            t_total=ms([(r["t_container"] or 0) + (r["t_key"] or 0) for r in G]), t_decode_ms=ms([1000 * r["t_decode"] for r in G], 2),
            tok_in=ms([r["tok_in"] for r in G], 0), tok_out=ms([r["tok_out"] for r in G], 0),
            key_tok_in=ms([r["key_tok_in"] for r in G], 0), key_tok_out=ms([r["key_tok_out"] for r in G], 0),
            syn_positions=ms([100 * r["syn_positions"] for r in G], 1), syn_mean=ms([r["syn_mean"] for r in G]),
            syn_full=ms([100 * r["syn_full"] for r in G], 1), usable=ms([100 * r["usable"] for r in G], 1),
            edit_rate=ms([100 * r["edit_rate"] for r in G], 1),
            ppl_container=ms([r.get("ppl_container") for r in G], 1), ppl_stego=ms([r.get("ppl_stego") for r in G], 1),
            ppl_ratio_of_means=f'{np.mean([r["ppl_stego"] for r in G]) / np.mean([r["ppl_container"] for r in G]):.2f}',
            ppl_ratio_per_run=ms([r.get("ppl_ratio") for r in G]),
        )
    return out


if __name__ == "__main__":
    res = {}
    for a in sys.argv[1:]:
        name, path = a.split("=", 1)
        res[name] = summary([json.loads(l) for l in open(path)])
    print(json.dumps(res, indent=1, ensure_ascii=False))
