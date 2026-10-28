"""Text-quality side of the experiment: GPT-2 perplexity (judge independent of the generator), Flesch Reading Ease,
sentence-embedding cosine to the cover, and edit rate, for cover / stego_b1..b5 / stego_rand / stego_sel."""
import json, pathlib, sys
import numpy as np, torch, textstat
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

HERE = pathlib.Path(__file__).parent
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def ppl(texts, tok, m, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
        logits = m(**enc).logits[:, :-1].float()
        tgt = enc["input_ids"][:, 1:]; mask = enc["attention_mask"][:, 1:].float()
        ll = logits.log_softmax(-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        out += torch.exp(-(ll * mask).sum(1) / mask.sum(1)).cpu().tolist()
    return out


def main():
    res = json.load(open(HERE / "results.json"))
    sel = res["selected"]
    corpus = {}
    for l in open(HERE / "corpus.jsonl"):
        r = json.loads(l); corpus[(r["uuid"], r["arm"], r.get("cand", -1))] = r
    uuids = sorted({u for (u, a, c) in corpus if a == "human"})
    arms = {a: [corpus[(u, a, -1)]["text"] for u in uuids] for a in ["human", "cover", "stego_b1", "stego_b2", "stego_b3", "stego_b4", "stego_b5"]}
    arms["stego_rand"] = [corpus[(u, "cand_b5", 0)]["text"] for u in uuids]
    arms["stego_sel"] = [corpus[(u, "cand_b5", sel[u])]["text"] for u in uuids]

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2"); tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(DEV).eval()
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(DEV))
    cov_e = emb.encode(arms["cover"], normalize_embeddings=True)
    table = {}
    for a, texts in arms.items():
        p = np.array(ppl(texts, tok, m))
        fl = np.array([textstat.flesch_reading_ease(t) for t in texts])
        e = emb.encode(texts, normalize_embeddings=True)
        cos = (e * cov_e).sum(1)
        edit = np.array([np.mean([x != y for x, y in zip(t.split(), c.split())]) for t, c in zip(texts, arms["cover"])])
        table[a] = dict(ppl_mean=float(p.mean()), ppl_median=float(np.median(p)), ppl_ratio_vs_cover=float(np.median(p) / np.median(ppl(arms["cover"], tok, m)) if a != "cover" else 1.0),
                        flesch=float(fl.mean()), flesch_sd=float(fl.std()), cos_to_cover=float(cos.mean()), edit_rate=float(edit.mean()))
        print(a, {k: round(v, 3) for k, v in table[a].items()}, flush=True)
    json.dump(table, open(HERE / "quality.json", "w"), indent=1)


if __name__ == "__main__":
    main()
