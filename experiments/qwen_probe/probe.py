"""How many distinct synonyms per word Qwen gives when SynSteg's prompt asks for N (2..32). Same prompt, model, temperature 0.7."""
import json, glob, os, sys, random, re, urllib.request, statistics as st
REPO = os.environ.get('SYNSTEG_ROOT', os.path.expanduser('~/PycharmProjects/Personal/SynSteg')); sys.path.insert(0, REPO)
from utils import prompts
MODEL = "qwen3:30b-a3b-instruct-2507-q4_K_M"
OUT = os.path.join(os.path.dirname(__file__), "probe_results.jsonl")
random.seed(0)
conts = [json.load(open(f))["container"] for f in sorted(glob.glob(REPO + "/reports/qwen_series/*.json"))]
chunks = []
for c in random.sample(conts, 10):
    w = c.split(); chunks += [w[i:i + 5] for i in (0, 10)]
def ask(n, chunk):
    schema = {"type": "object", "required": ["words"], "properties": {"words": {"type": "array", "items": {
        "type": "object", "additionalProperties": {"type": "array", "maxItems": n, "items": {"type": "string"}}}}}}
    body = dict(model=MODEL, stream=False, think=False, format=schema, options=dict(num_predict=4096, temperature=0.7),
                messages=[{"role": "system", "content": prompts.ALL_SYNONYMS_GENERATION_PROMPT.replace("N_SYNONYMS", str(n))},
                          {"role": "user", "content": prompts.ALL_SYNONYMS_GENERATION_INPUT.format(context=chunk)}])
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(),
                                                                {"Content-Type": "application/json"}), timeout=1200))
    return json.loads(r["message"]["content"])["words"]
with open(OUT, "w") as fo:
    for n in (2, 4, 8, 16, 32):
        for ch in chunks:
            try: words = ask(n, ch)
            except Exception as e: fo.write(json.dumps({"n": n, "error": str(e)}) + "\n"); continue
            for e in words:
                for base, syn in e.items():
                    syn = [s.strip() for s in syn]
                    d = list(dict.fromkeys(s.lower() for s in syn if s.lower() != base.lower()))
                    fo.write(json.dumps({"n": n, "base": base, "returned": len(syn), "distinct": len(d),
                                         "multiword": sum(1 for s in d if re.search(r"[\s-]", s))}, ensure_ascii=False) + "\n")
            fo.flush()
        print("done n =", n, flush=True)
