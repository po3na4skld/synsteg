"""Run: .venv/bin/python tests_missing_synonyms.py
Offline check of the missing-synonyms rule on stored runs: rebuild each key from the stored raw synonym lists with
the new binarize_synonyms, embed with core.embed, decode with core.decode_message. No LLM calls."""
import glob, json, os, sys, random, statistics as st
REPO = os.path.expanduser("~/PycharmProjects/Personal/SynSteg"); sys.path.insert(0, REPO)
os.environ.setdefault("OPENAI_API_KEY", "offline")
from steganography import core, helper
from steganography.secret_key import binarize_synonyms

# self-check on a toy key: 30 of 32 distinct -> usable, codes 11110/11111 absent; 20 of 32 -> unusable
syn = [f"w{i}" for i in range(30)] + ["w0", "w1"]
k = binarize_synonyms("base", syn, 5); assert len(k) == 30 and "11111" not in k
assert binarize_synonyms("base", [f"w{i}" for i in range(20)] * 2, 5) == {"0": "base", "1": "base"}
assert "Up" not in binarize_synonyms("up.", ["Up", "up", "upward", "higher"], 2).values()  # punctuation/case of the word

random.seed(1)
for folder in ("gpt6_luna_series", "qwen_series"):
    rows = []
    for f in sorted(glob.glob(f"{REPO}/reports/{folder}/*.json")):
        d = json.load(open(f))
        if not d["secret_key"]: continue
        L = len(d["message"]); msg = "".join(random.choice("01") for _ in range(L))
        key = [{b: binarize_synonyms(b, list(m.values()), 5)} for e in d["secret_key"] for b, m in e.items()]
        stego, n = core.embed(d["container"], key, msg)
        dec, _ = core.decode_message(stego, key, clean_output=False)
        usable = sum(not helper.has_duplicates(e) for e in key) / len(key)
        rows.append(dict(L=L, n=n, ok=dec == msg[:n], full=n == L, usable=usable))
    print(folder, "runs", len(rows), "| decoded == embedded bits in", sum(r["ok"] for r in rows), "| full message fits", sum(r["full"] for r in rows),
          "| usable positions", f"{st.mean(r['usable'] for r in rows):.0%}")
    for L in (128, 256, 512):
        r = [x for x in rows if x["L"] == L]
        if r: print(f"   L={L}: fits {sum(x['full'] for x in r)}/{len(r)}, embedded {st.mean(x['n'] / L for x in r):.0%}")
