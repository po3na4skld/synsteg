"""Text quality and extra capacity figures for dissertation tables 3.9, 3.11, 3.12.
Per configuration: bits/char per L (mean of ratios over capacity runs); Flesch (mean), GPT-2 perplexity (median) and
MiniLM coherence (mean cosine of adjacent sentences) of the 128-bit stego corpus and of its length-matched covers;
full-protocol words and encode time of completed sends. Usage: <ai_detectors venv python> quality.py -> results/quality.json"""
import json, os, re, statistics as st, sys, pathlib
import numpy as np, textstat
HERE = pathlib.Path(__file__).resolve().parent; R = HERE / 'results'
sys.path.insert(0, str(HERE.parent / 'model_series'))
import metrics  # GPT-2 124M perplexity ppl(), as in experiments/model_series/metrics.py and summarize.py
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained('openai-community/gpt2'); tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained('openai-community/gpt2').to(metrics.DEV).eval()
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
SENT = re.compile(r'(?<=[.!?])\s+')
def coherence(t):
    s = [x for x in SENT.split(t.strip()) if x.strip()]
    if len(s) < 2: return None
    e = emb.encode(s, normalize_embeddings=True); return float(np.mean([e[i] @ e[i + 1] for i in range(len(e) - 1)]))
load = lambda p: [json.loads(l) for l in open(p)]
out = {}
for c in ('default', 'u32', 'u1024'):
    cap = [r for r in load(R / f'capacity_{c}.jsonl') if not r.get('encode_error')]
    bpc = {L: float(np.mean([r['L'] / r['chars'] for r in cap if r['L'] == L])) for L in (128, 256, 512)}
    stego = [r['stegotext'] for r in load(R / f'corpus_{c}.jsonl') if not r.get('encode_error')][:400]
    cover = [r['text'] for r in load(R / f'covers_{c}.jsonl')][:400]
    q = {}
    for name, texts in (('stego', stego), ('cover', cover)):
        coh = [x for x in map(coherence, texts) if x is not None]
        q[name] = dict(flesch=float(np.mean([textstat.flesch_reading_ease(t) for t in texts])),
                       ppl_median=float(np.median(metrics.ppl(texts, tok, m))), coherence=float(np.mean(coh)), n=len(texts))
    ch = load(R / f'chain_{c}.jsonl'); done = [r for r in ch if r.get('completed')]
    codec128 = [r for r in cap if r['L'] == 128]
    out[c] = dict(bits_per_char=bpc, quality=q, chain=dict(
        completed=len(done), sends=len(ch),
        words=float(np.mean([r['words'] for r in done])) if done else None,
        encode_s=float(np.mean([r['encode_s'] for r in done])) if done else None,
        codec_words_128=float(np.mean([len(r['stegotext'].split()) for r in codec128]))))
    print(c, json.dumps(out[c], indent=1), flush=True)
json.dump(out, open(R / 'quality.json', 'w'), indent=1)
