"""Cover texts for steganalysis and perplexity: ordinary ancestral sampling from GPT-2 (full vocabulary, temperature 1,
no top-k/top-p) from the codec prompt, one cover per stegotext with the same number of GPT-2 tokens.

Usage: $PY experiments/conversation_bench/covers.py default u32 u1024
  reads  results/corpus_<cfg>.jsonl (first N_STEGO successful encodes)
  writes results/covers_<cfg>.jsonl
"""
import json, os, sys, collections
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.environ.get('SYNSTEG_ROOT', os.path.expanduser('~/PycharmProjects/Personal/SynSteg'))
RES = f'{ROOT}/experiments/conversation_bench/results'
EXAMPLE = f'{ROOT}/experiments/conversation_bench/conversation-steganography/conversation-stenography.example.json'
N_STEGO = int(os.environ.get('N_STEGO', 400))  # smaller only for smoke tests
SEED = 0


def load_stego(cfg):
    rows = [json.loads(l) for l in open(f'{RES}/corpus_{cfg}.jsonl')]
    ok = [r for r in rows if not r['encode_error']]
    return ok[:N_STEGO], len(rows), len(ok)


def main(cfgs):
    prompt = json.load(open(EXAMPLE))['prompt']
    tok = AutoTokenizer.from_pretrained('openai-community/gpt2')
    m = AutoModelForCausalLM.from_pretrained('openai-community/gpt2').eval()   # CPU: reproducible sampling
    ids = tok(prompt, return_tensors='pt').input_ids
    for ci, cfg in enumerate(cfgs):
        stego, _, _ = load_stego(cfg)
        assert len(stego) == N_STEGO, f'{cfg}: only {len(stego)} stegotexts'
        by_len = collections.defaultdict(list)
        for i, r in enumerate(stego):
            by_len[r['gpt2_tokens']].append(i)
        covers = [None] * len(stego)
        for n, idx in sorted(by_len.items()):
            torch.manual_seed(SEED + 1000 * ci + n)
            with torch.no_grad():
                out = m.generate(ids.repeat(len(idx), 1), attention_mask=torch.ones_like(ids).repeat(len(idx), 1),
                                 do_sample=True, top_k=0, top_p=1.0, temperature=1.0, min_new_tokens=n, max_new_tokens=n,
                                 pad_token_id=tok.eos_token_id)
            for i, row in zip(idx, out[:, ids.shape[1]:]):
                covers[i] = {'i': i, 'gpt2_tokens': n, 'text': tok.decode(row, clean_up_tokenization_spaces=False)}
        with open(f'{RES}/covers_{cfg}.jsonl', 'w') as f:
            for c in covers:
                f.write(json.dumps(c) + '\n')
        print(cfg, len(covers), 'covers', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
