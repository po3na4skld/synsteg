"""Per-run metrics for a folder of SynSteg JSON reports -> <folder>/../<name>_metrics.jsonl and a summary table.

Definitions follow notebooks/gpt_omni_metrics_creation.ipynb, which produced DeepStego Table 1:
  words   = len(secret_key)  (container words after alignment, "Words Required")
  BPW     = |m| / words      (get_throughput)
  capacity % = (Ls-1)*k / (8*Ls*mc) * 100   (calculate_average_er; Ls words/sentence and mc chars/word of the
            stegotext, k mean code length of the key); per run, and pooled per L as in the notebook
  recovery = decoded_message == message
Added: synonym statistics from the stored key, GPT-2 (124M) perplexity of container and stegotext.
Usage: <detector venv python> metrics.py <reports dir> [<reports dir> ...]
"""
import json, glob, os, sys, difflib
import numpy as np, torch, textstat
from transformers import AutoTokenizer, AutoModelForCausalLM

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
B = int(os.environ.get("SERIES_B", 5))  # bits per position of the series


def ident(base, mp):
    return all(v.lower() == base.lower() for v in mp.values())


def edit_rate(a, b):
    a, b = a.lower().split(), b.lower().split()
    return 1 - sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()) / len(a)


def run_metrics(r):
    L, sk, enc = len(r["message"]), r["secret_key"], r["encoded_message"]
    words = len(sk)
    out = dict(uuid=r["uuid"], L=L, error=r.get("error_message", ""), words=words, container_words=len(r["container"].split()))
    if not words or not enc:
        return out
    dec = r.get("decoded_message") or ""
    pre = next((i for i, (a, b) in enumerate(zip(dec, r["message"])) if a != b), min(len(dec), L))
    k = np.mean([len(next(iter(mp))) for e in sk for mp in e.values()])
    sents = [len(s.split()) for s in enc.split(".")]
    Ls, mc = np.mean(sents), np.mean([len(t) for t in enc.split()])
    syn = [(len(mp), len(set(v.lower() for v in mp.values()))) for e in sk for base, mp in e.items() if not ident(base, mp)]
    tr, ur = r["encoding_time_report"], r["encoding_usage_report"]
    key_u = ur.get("secret_key_generation", [])
    key_u = key_u if isinstance(key_u, list) else [key_u]
    out.update(
        exact=dec == r["message"], embedded=pre / L, bpw=L / words, k=k, Ls=Ls, mc=mc, sents=sents, toklens=[len(t) for t in enc.split()],
        capacity=(Ls - 1) * k / (8 * Ls * mc) * 100,
        t_container=tr.get("container_generation"), t_key=tr.get("secret_key_generation"), t_decode=r.get("decoding_time"),
        tok_in=ur.get("container_generation", {}).get("prompt_tokens", 0) + sum(u["prompt_tokens"] for u in key_u),
        tok_out=ur.get("container_generation", {}).get("completion_tokens", 0) + sum(u["completion_tokens"] for u in key_u),
        key_tok_in=sum(u["prompt_tokens"] for u in key_u), key_tok_out=sum(u["completion_tokens"] for u in key_u),
        syn_positions=len(syn) / words,                                   # share of container words that got a synonym list
        syn_mean=float(np.mean([d for _, d in syn])) if syn else 0.0,     # mean distinct synonyms per such word
        syn_full=float(np.mean([d >= 2 ** B for _, d in syn])) if syn else 0.0,  # share with >= 2^b distinct synonyms
        usable=float(np.mean([s == d for s, d in syn])) if syn else 0.0,  # share without duplicates (usable by the method)
        edit_rate=edit_rate(r["container"], enc),  # share of container words not kept (difflib alignment)
        flesch_container=textstat.flesch_reading_ease(r["container"]), flesch_stego=textstat.flesch_reading_ease(enc),
    )
    return out


@torch.no_grad()
def ppl(texts, tok, m, bs=16):
    res = []
    for i in range(0, len(texts), bs):
        e = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
        lg = m(**e).logits[:, :-1].float()
        tgt, mask = e["input_ids"][:, 1:], e["attention_mask"][:, 1:].float()
        ll = lg.log_softmax(-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        res += torch.exp(-(ll * mask).sum(1) / mask.sum(1)).cpu().tolist()
    return res


def main(dirs):
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2"); tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(DEV).eval()
    for d in dirs:
        reps = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json")))]
        rows = [run_metrics(r) for r in reps]
        ok = [i for i, r in enumerate(reps) if r["encoded_message"]]
        pc, ps = ppl([reps[i]["container"] for i in ok], tok, m), ppl([reps[i]["encoded_message"] for i in ok], tok, m)
        for i, a, b in zip(ok, pc, ps):
            rows[i].update(ppl_container=a, ppl_stego=b, ppl_ratio=b / a)
        with open(d.rstrip("/") + "_metrics.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(d, len(rows), "runs", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
