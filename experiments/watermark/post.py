"""After luna_attacks.py: (1) 95 % bootstrap CI of the removed share 1 - residual for every row (watermarked class,
texts resampled, the same resample for attacked and unattacked g); (2) perplexity by the held-out judge
Qwen2.5-1.5B (GPT-2 is the selector of the low-perplexity variants, so its ratio is optimistic), non-control rows.
-> results/luna_post.json
"""
import collections, gc, json
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import HERE, load, MeanG

wm = load("wm.jsonl")
ids = [r["id"] for r in wm]
texts = collections.defaultdict(dict)
for l in open(HERE / "results/luna_texts.jsonl"):
    t = json.loads(l)
    if t["id"].startswith("wm"):
        texts[t["variant"]][t["id"]] = t["text"]
texts = {"no attack": {r["id"]: r["text"] for r in wm}, **texts}

G = MeanG()
g = {v: np.array([G(d[i]) for i in ids]) for v, d in texts.items()}
rng = np.random.default_rng(0)
idx = rng.integers(0, len(ids), (20000, len(ids)))
out = {}
for v, x in g.items():
    rem = 1 - (x[idx].mean(1) - 0.5) / (g["no attack"][idx].mean(1) - 0.5)
    out[v] = dict(n=len(ids), removed=float(1 - (x.mean() - 0.5) / (g["no attack"].mean() - 0.5)),
                  removed_lo=float(np.percentile(rem, 2.5)), removed_hi=float(np.percentile(rem, 97.5)))
print("g done", flush=True)

dev = torch.device("mps")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16).to(dev).eval()


@torch.no_grad()
def ppl(text):  # batch 1
    x = torch.tensor([tok(text).input_ids[:1024]], device=dev)
    return float(torch.exp(m(x, labels=x).loss.float()))


base = np.array([ppl(texts["no attack"][i]) for i in ids])
for v, d in texts.items():
    if v.startswith("control"):
        continue
    p = base if v == "no attack" else np.array([ppl(d[i]) for i in ids])
    out[v].update(qwen_ppl=float(p.mean()), qwen_ratio=float(p.mean() / base.mean()), qwen_ratio_median=float(np.median(p / base)))
    print(v, round(out[v]["qwen_ratio"], 2), flush=True)
del m; gc.collect(); torch.mps.empty_cache()
json.dump(out, open(HERE / "results/luna_post.json", "w"), indent=1)
