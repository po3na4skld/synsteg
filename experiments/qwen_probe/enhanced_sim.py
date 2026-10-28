"""Offline enhanced-mode check on stored Qwen runs: keep the first 2^b distinct synonyms of each stored list,
build the enhanced key with the published binarize_synonyms_partially (b + a bits per position), encode with the
published loop, decode with the published decode_message. No LLM calls."""
import glob, json, os, sys, random, statistics as st
import numpy as np
REPO = os.environ.get('SYNSTEG_ROOT', os.path.expanduser('~/PycharmProjects/Personal/SynSteg'))
sys.path.insert(0, REPO + "/experiments/model_series/synsteg_683e8cb")
from steganography import helper
from steganography.core import decode_message
from steganography.secret_key import binarize_synonyms_partially, fix_token_container_size

def run(rep, b, a):
    msg, cont = rep["message"], rep["container"]
    chunks = list(helper.divide_chunks(msg, b + a)); ci = 0; key = []
    for e in rep["secret_key"]:
        base = list(e)[0]
        syn = list(dict.fromkeys(s for s in e[base].values() if s.lower() != base.lower()))[:2 ** b]
        if len(syn) == 2 ** b and ci < len(chunks):
            key.append({base: binarize_synonyms_partially(syn, chunks[ci])}); ci += 1
        else:
            key.append({base: {"0": base, "1": base}})
    out, idx, replaced = [], 0, 0
    for token, rt in zip(cont.split(), key):  # published encode loop
        sp, token = helper.check_endswith_special(token)
        if helper.is_token_replacable(token, rt) and idx < len(msg) and not helper.has_duplicates(rt):
            token, cap = helper.check_capitalization(token, rt)
            size = len(next(iter(rt[token]))); part = msg[idx:idx + size]
            if len(part) != size: rt[token] = fix_token_container_size(rt[token], len(part))
            new = rt[token].get(part)
            if new is None: return dict(ok=False, err="chunk missing")
            out.append((new.capitalize() if cap else new) + sp); idx += size; replaced += 1
        else:
            out.append(token + sp)
    stego = " ".join(out)
    dec, _ = decode_message(stego, key, clean_output=False)
    return dict(ok=dec == msg, embedded=min(idx, len(msg)) / len(msg), prefix=msg.startswith(dec),
                bpw=len(msg) / len(cont.split()), repl=replaced / len(cont.split()))

np.random.seed(0); random.seed(0)
reps = [json.load(open(f)) for f in sorted(glob.glob(REPO + "/reports/qwen_series/*.json"))]
print("b a | runs | full recovery | bits embedded | replaced words | bits/word")
for b in (2, 3):
    for a in (3, 5, 8):
        for L in (128, 256, 512):
            r = [run(x, b, a) for x in reps if len(x["message"]) == L]
            r = [x for x in r if "embedded" in x]
            print(f"{b} {a} L={L} | {len(r)} | {sum(x['ok'] for x in r)}/{len(r)} | {st.mean(x['embedded'] for x in r):.0%} | {st.mean(x['repl'] for x in r):.0%} | {st.mean(x['bpw'] for x in r):.2f}")
