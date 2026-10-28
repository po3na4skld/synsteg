"""Stage 2: SynSteg attacks with the cached gpt-6-luna synonym lists (fetch_synonyms.py), both classes.

Keys: for b = 1..5 the cached 32-synonym lists of every 5-word chunk go through the e87605e pipeline unchanged
(secret_key.binarize_synonyms: first 2^b distinct synonyms, >= 90 % rule, prefix-free; align_container_and_secret_key).
Rows:
  synsteg b         random message, encoder = the e87605e embed rule, exact decoding checked with core.decode_message
  control b         random replacement from the first 2^b distinct cached synonyms, same number of replaced words per text
  pplopt b          synonym choice by GPT-2 NLL (formula 4.8, Scorer of ppl_opt/optimize.py), codes permuted
  window b d        per word only synonyms with Delta ln PPL <= d (fragment: 20 tokens of the original text before the
                    word, the word, 4 next words), then the e87605e key rule on the filtered list
  window+opt b d    the same filtered key, synonym chosen by formula 4.8
  control (window…) random replacement from all cached synonyms at the same per-text count
-> results/luna_rows.json, results/luna_paired.json, results/luna_texts.jsonl
Usage: .venv/bin/python luna_attacks.py [--cache cache] [--limit N]
"""
import argparse, copy, json, random, sys
import numpy as np, torch
from common import HERE, load, MeanG, random_control, evaluate, paired

sys.path.insert(0, str(HERE / "synsteg_e87605e"))
from steganography import helper, core  # noqa: E402
from steganography.secret_key import binarize_synonyms, align_container_and_secret_key, fix_token_container_size  # noqa: E402
from steganography.helper import clean_container  # noqa: E402

sys.path.insert(1, str(HERE.parent / "ppl_opt"))
from optimize import Scorer, LEFT, RIGHT, MAX_RETRY  # noqa: E402  (formula 4.8 scorer: GPT-2, 20 left tokens, 4 right words)

DELTAS = [0.0, 0.25, 0.5, 1.0, 2.0]
WINDOW_B = [1, 2]


def raw_chunks(cache, tid):
    """[(chunk words, [(model's word, synonym list)])] per cached request, in text order."""
    out = []
    for f in sorted((cache / tid).glob("*.json")):
        j = json.load(open(f))
        out.append((j["chunk"], [(k, [s for s in v if isinstance(s, str)] if isinstance(v, list) else [])
                                 for d in j["words"] if isinstance(d, dict) for k, v in d.items()]))
    return out


def raw_key(chunks, per_chunk=True):
    """Aligned per-position raw synonym lists (None = no table for this word), via the pipeline's alignment.
    per_chunk: align every 5-word request with its own answer and concatenate. The pipeline aligns the whole text at
    once, so one key word the model rewrote (e.g. a dropped quote mark) stalls alignment for the rest of the text;
    the number of positions this changes is reported."""
    groups = chunks if per_chunk else [([w for c, _ in chunks for w in c], [p for _, a in chunks for p in a])]
    al = []
    for words, pairs in groups:
        sk = [{k: {str(j): s for j, s in enumerate(v)} if len(v) >= 2 else {"0": k, "1": k}} for k, v in pairs]
        al += align_container_and_secret_key(" ".join(words), sk)
    res = []
    for slot in al:
        (tok, m), = slot.items()
        res.append((tok, None if list(m) == ["0", "1"] and m["0"] == m["1"] else list(m.values())))
    return res


def key_from_lists(rk, b, lists=None):
    """e87605e key: binarize_synonyms(word, list, b) per position; lists overrides the raw lists (window filter)."""
    key = []
    for j, (tok, raw) in enumerate(rk):
        raw = raw if lists is None else lists[j]
        key.append({tok: binarize_synonyms(tok, raw, b) if raw else {"0": tok, "1": tok}})
    return key


def distinct(tok, raw):
    """The synonyms binarize_synonyms would keep, in order, before the 2^b cut and the 90 % rule (same filters:
    repeats, the word itself, phrases starting with it, phrases starting with another synonym)."""
    norm = lambda w: clean_container(w).strip().lower()
    base, out, seen = norm(tok), [], {norm(tok)}
    for s in raw or []:
        if norm(s).startswith((base + " ", base + "-")) or norm(s) in seen:
            continue
        seen.add(norm(s)); out.append(s)
    return [s for s in out if not any(t is not s and norm(s).startswith(norm(t) + " ") for t in out)]


def embed(text, key, bits, choose=None, ban=frozenset()):
    """core.embed generalised with an optional chooser (formula 4.8). Returns stego, key, bits used, replaced words,
    trace [(word idx, start bit, size, synonym)]."""
    key, out, idx, trace = copy.deepcopy(key), [], 0, []
    words = text.split()
    for i, (token, slot) in enumerate(zip(words, key)):
        special, token = helper.check_endswith_special(token)
        if helper.is_token_replacable(token, slot) and idx < len(bits) and not helper.has_duplicates(slot):
            original, (token, cap) = token, helper.check_capitalization(token, slot)
            size = len(next(iter(slot[token])))
            part = bits[idx: idx + size]
            if len(part) != size:
                slot[token] = fix_token_container_size(slot[token], len(part)); size = len(part)
            mapping = slot[token]
            if choose is not None:
                vals = [v for v in mapping.values() if (i, v) not in ban]
                best = choose(out, vals, words[i + 1: i + 1 + RIGHT], cap, special)
                c = next(c for c, s in mapping.items() if s == best)
                if part in mapping:
                    mapping[part], mapping[c] = mapping[c], mapping[part]
                else:
                    mapping[part] = mapping.pop(c)
            new = mapping.get(part)
            if new is None:  # missing code: the word stays, no bits
                out.append(original + special); continue
            out.append((new.capitalize() if cap else new) + special)
            trace.append((i, idx, size, new)); idx += size
        else:
            out.append(token + special)
    return " ".join(out), key, idx, len(trace), trace


def encode(text, key, bits, choose=None):
    """-> (stego, replaced words, bits, decoded exactly). Embed and check exact decoding with the project decoder; on a decoding failure under a chooser the failing
    choice is banned and the text re-embedded (as optimize.run)."""
    ban = set()
    for _ in range(MAX_RETRY + 1):
        stego, k2, used, nrep, trace = embed(text, key, bits, choose, frozenset(ban))
        dec, _ = core.decode_message(stego, k2, clean_output=False)
        if dec == bits[:used]:
            return stego, nrep, used, True
        if choose is None:
            break
        bad = next((p for p, (a, b) in enumerate(zip(dec, bits[:used])) if a != b), len(dec))
        i, _, _, best = next(t for t in trace if t[1] + t[2] > bad)
        ban.add((i, best))
    return stego, nrep, used, False


class Judge:
    """One GPT-2 for everything (formula 4.8 choice, Delta ln PPL, whole-text PPL); candidates scored in batches of 4."""
    def __init__(self):
        self.sc = Scorer()
        self.tok = self.sc.tok
        nll = self.sc.nll
        self.sc.nll = lambda l, c, r: np.concatenate([nll(l, c[k:k + 4], r) for k in range(0, len(c), 4)])

    @torch.no_grad()
    def __call__(self, text):  # GPT-2 perplexity of the whole text (first 1024 tokens), as common.PPL
        ids = [self.tok.eos_token_id] + self.tok(text).input_ids[:1023]
        x = torch.tensor([ids], device=self.sc.m.device)
        return float(torch.exp(self.sc.m(x, labels=x).loss))

    def choose(self, out, vals, right, cap, special):
        render = lambda s: (s.capitalize() if cap else s) + special
        return vals[int(np.argmin(self.sc.nll(out[-LEFT:], [render(s) for s in vals], right)))]

    def deltas(self, words, i, cands):
        """Delta ln PPL of the fragment (20 original tokens before, word, 4 next words) for each candidate."""
        special, token = helper.check_endswith_special(words[i])
        render = lambda s: (s.capitalize() if token.istitle() else s) + special
        left, right = words[max(0, i - LEFT):i], words[i + 1: i + 1 + RIGHT]
        allc = [token + special] + [render(s) for s in cands]
        nll = self.sc.nll(left, allc, right)
        tail = (" " + " ".join(right)) if right else ""
        n = np.array([len(self.tok((" " if left else "") + c + tail).input_ids) for c in allc])
        lp = nll / n
        return list(lp[1:] - lp[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(HERE / "cache"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    from pathlib import Path
    cache, outdir = Path(a.cache), Path(a.out); outdir.mkdir(exist_ok=True)
    classes = {"wm": load("wm.jsonl")[:a.limit], "hu": load("human.jsonl")[:a.limit]}
    J, G = Judge(), MeanG()
    P = J
    rng = random.Random(0)

    # keys, sanity check against the pipeline's own order (binarize then align) on every text
    rks, align_stats = {}, []
    for c, rows in classes.items():
        for r in rows:
            ch = raw_chunks(cache, r["id"])
            assert [w for c, _ in ch for w in c] == r["text"].split(), r["id"]
            rk, glob_rk = raw_key(ch), raw_key(ch, per_chunk=False)
            for b in (1, 5):  # same key as the pipeline's order (binarize, then align the whole text)
                ref = align_container_and_secret_key(r["text"], [{k: binarize_synonyms(k, v, b) if len(v) >= 2 else {"0": k, "1": k}} for _, a in ch for k, v in a])
                assert ref == key_from_lists(glob_rk, b), r["id"]
            align_stats.append(dict(id=r["id"], words=len(rk), tables=sum(x is not None for _, x in rk),
                                    tables_global=sum(x is not None for _, x in glob_rk),
                                    usable_b5=sum(not helper.has_duplicates(s) for s in key_from_lists(rk, 5))))
            for tok, raw in rk:
                if raw:
                    for b in (1, 5):
                        ref = binarize_synonyms(tok, raw, b)
                        d = distinct(tok, raw)[:2 ** b]
                        assert (list(ref.values()) == d) if len(d) >= max(2, 0.9 * 2 ** b) else ref == {"0": tok, "1": tok}
            rks[r["id"]] = rk
    json.dump(align_stats, open(outdir / "alignment.json", "w"), indent=0)
    print("keys built", {k: sum(a[k] for a in align_stats) for k in ("words", "tables", "tables_global", "usable_b5")}, flush=True)

    # Delta ln PPL of every distinct cached synonym (one GPT-2 pass per word)
    dl = json.load(open(outdir / "deltas.json")) if (outdir / "deltas.json").exists() else {}
    for c, rows in classes.items():
        for r in rows:
            if r["id"] in dl:
                continue
            words = r["text"].split()
            dl[r["id"]] = [J.deltas(words, i, distinct(tok, raw)) if distinct(tok, raw) else []
                           for i, (tok, raw) in enumerate(rks[r["id"]])]
    print("deltas done", flush=True)
    json.dump(dl, open(outdir / "deltas.json", "w"))

    variants = {}  # name -> class -> list of (text, n_replaced, bits, decoded_ok)
    def run(name, fn):
        variants[name] = {c: [fn(r) for r in rows] for c, rows in classes.items()}
        print(name, flush=True)

    msg = lambda r: "".join(rng.choice("01") for _ in range(5 * len(r["text"].split()) + 5))
    for b in range(1, 6):
        run(f"synsteg b={b}", lambda r, b=b: encode(r["text"], key_from_lists(rks[r["id"]], b), msg(r)))
        run(f"pplopt b={b}", lambda r, b=b: encode(r["text"], key_from_lists(rks[r["id"]], b), msg(r), J.choose))
    for b in WINDOW_B:
        for d in DELTAS:
            def filt(r, d=d):
                return [[s for s, x in zip(distinct(tok, raw), dx) if x <= d] if raw else None
                        for (tok, raw), dx in zip(rks[r["id"]], dl[r["id"]])]
            run(f"window b={b} d={d}", lambda r, b=b, f=filt: encode(r["text"], key_from_lists(rks[r["id"]], b, f(r)), msg(r)))
            run(f"window+opt b={b} d={d}", lambda r, b=b, f=filt: encode(r["text"], key_from_lists(rks[r["id"]], b, f(r)), msg(r), J.choose))

    # controls: same number of replaced words per text, random positions, uniform synonym
    controls = {}
    for name, V in list(variants.items()):
        b = int(name.split("b=")[1].split()[0])
        full = not name.startswith(("synsteg", "pplopt"))
        cn = f"control for {name}"
        def lists_of(rid, full=full, b=b):
            return [(distinct(tok, raw) if full else distinct(tok, raw)[:2 ** b]) if raw else [] for tok, raw in rks[rid]]
        controls[cn] = {c: [random_control(r["text"], v[1], lambda i, w, L=lists_of(r["id"]): L[i], rng) + (0, True)
                            for r, v in zip(rows, V[c])] for c, rows in classes.items()}
    variants.update(controls)

    nw = [len(r["text"].split()) for r in classes["wm"]]
    g0 = {c: [G(r["text"]) for r in rows] for c, rows in classes.items()}
    base = float(np.mean(g0["wm"]))
    ppl0 = [P(r["text"]) for r in classes["wm"]]
    rows_out = [evaluate("no attack", g0["wm"], g0["hu"], base, nw, ppl0, [0] * len(nw), [0] * len(nw))]
    rows_out[0].update(ppl_ratio=1.0, decoded=None)
    pg, texts_out = {}, []
    for name, V in variants.items():
        gp, gn = [G(t[0]) for t in V["wm"]], [G(t[0]) for t in V["hu"]]
        ppl = [P(t[0]) for t in V["wm"]]
        row = evaluate(name, gp, gn, base, nw, ppl, [t[1] for t in V["wm"]], [t[2] if "control" not in name else 0 for t in V["wm"]])
        row.update(ppl_ratio=row["ppl"] / float(np.mean(ppl0)), ppl_ratio_median=float(np.median(np.array(ppl) / np.array(ppl0))),
                   decoded=sum(t[3] for c in V for t in V[c]), n_decoded=sum(len(V[c]) for c in V),
                   share_hu=float(np.sum([t[1] for t in V["hu"]]) / np.sum([len(r["text"].split()) for r in classes["hu"]])))
        rows_out.append(row); pg[name] = gp
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        for c, rows in classes.items():
            for r, t in zip(rows, V[c]):
                texts_out.append(dict(variant=name, id=r["id"], text=t[0], replaced=t[1], bits=t[2], decoded=t[3]))
    pairs = {name: paired(pg[name], pg[f"control for {name}"], base) for name in pg if not name.startswith("control")}
    json.dump(rows_out, open(outdir / "luna_rows.json", "w"), indent=1)
    json.dump(pairs, open(outdir / "luna_paired.json", "w"), indent=1)
    with open(outdir / "luna_texts.jsonl", "w") as f:
        for t in texts_out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
