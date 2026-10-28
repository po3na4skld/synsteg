"""Positive class: 60 Qwen2.5-0.5B-Instruct texts under the SynthID-Text key (CPU, seed 0, T = 1.0 with the model's top_k = 20 / top_p = 0.8, 224 new tokens), texts of <= 3 words dropped,
first 56 kept. Negative class: 56 IMDB test reviews matched one-to-one by word count (nearest, without replacement).
-> data/wm.jsonl, data/human.jsonl
"""
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import HERE, QWEN, WM_DEVICE, wm_config, normalize, save

import sys; sys.path.insert(0, str(HERE / "synsteg_e87605e"))
from utils.constants import TOPICS  # the 50 container topics of SynSteg + 10 more
TOPICS = TOPICS + ["The History of Coffee", "Life in a Small Fishing Village", "How Bees Make Honey", "The Joy of Cooking at Home",
                   "Learning to Play a Musical Instrument", "A Day at a Mountain Lake", "The Story of the Bicycle",
                   "Why People Keep Pets", "Public Libraries Today", "Gardening in the City"]
PROMPTS = [f"Write a short essay of about 200 words on the topic: {t}." for t in TOPICS][:60]


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(QWEN, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(QWEN).to(WM_DEVICE).eval()
    rows = []
    for i in range(0, len(PROMPTS), 6):
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in PROMPTS[i:i + 6]]
        enc = tok(chats, return_tensors="pt", padding=True).to(WM_DEVICE)
        out = model.generate(**enc, watermarking_config=wm_config(), do_sample=True, temperature=1.0, max_new_tokens=224)
        for j, o in enumerate(out[:, enc.input_ids.shape[1]:]):
            raw = tok.decode(o, skip_special_tokens=True)
            rows.append(dict(id=f"wm{i + j:02d}", prompt=PROMPTS[i + j], raw=raw, text=normalize(raw)))
        print(i, flush=True)
    good = [r for r in rows if len(r["text"].split()) > 3]
    print(f"generated {len(rows)}, degenerate {len(rows) - len(good)}")
    wm = good[:56]
    save("wm_all.jsonl", rows)
    save("wm.jsonl", wm)

    from datasets import load_dataset
    imdb = [normalize(t) for t in load_dataset("stanfordnlp/imdb", split="test")["text"]]
    rng = random.Random(0)
    pool = rng.sample(range(len(imdb)), 5000)
    human = []
    for k, r in enumerate(wm):
        n = len(r["text"].split())
        j = min(pool, key=lambda j: abs(len(imdb[j].split()) - n))
        pool.remove(j)
        human.append(dict(id=f"h{k:02d}", imdb_test_idx=j, text=imdb[j]))
    save("human.jsonl", human)


if __name__ == "__main__":
    main()
