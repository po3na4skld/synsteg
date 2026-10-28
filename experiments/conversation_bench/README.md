# Conversation-Steganography benchmark (Table 8)

Benchmark of the open generative steganography method Conversation-Steganography
(https://github.com/nethical6/conversation-steganography, commit `863a96e`). It produces the Conversation-Steganography
rows of Table 8. The original harness of 15.08.2026 was lost. This folder is a rebuild that follows the original protocol.

## Layout

| Path | Contents |
|---|---|
| `conversation-steganography/` | clone of the library at `863a96e`, changed only by `hf_model_no_grad.patch` |
| `hf_model_no_grad.patch` | `torch.inference_mode()` → `torch.no_grad()` in `python/hf_model.py`. On current PyTorch the inference-mode logits tensor is written to outside the block, so the first token raises an error. |
| `bench/` | Go harness (`main.go`, its own `go.mod` with a `replace` pointing to the clone) |
| `covers.py` | length-matched GPT-2 cover texts |
| `stegan.py` | steganalysis. The four detectors and `fit()`/`evaluate()` are imported from `experiments/steganalysis/steganalysis.py`. |
| `summarize.py` | Table 8 values → `results/summary.json`. Perplexity comes from `ppl()` in `experiments/model_series/metrics.py`. |
| `results/` | raw outputs, one JSONL line per run, plus `steganalysis.json` and `summary.json` |
| `logs/` | logs of the runs that produced `results/` |

Configurations (raw codec: a framed payload of L random bits, no encryption, no chaining). Every other setting
(prompt "The weather today is", temperature 1, strict style, candidate pool 8, finish tokens 32, length bias 0.1, …)
is read from `conversation-stenography.example.json`:

| `-config` | coding | top_n |
|---|---|---|
| `default` | arithmetic | 8 (as in the example config) |
| `u32` | uniform | 32 |
| `u1024` | uniform | 1024 |

The model is GPT-2 124M (`openai-community/gpt2`, revision `main`). It runs on the CPU in float32, with the same backend
arguments that `cmd/conversation-stenography/simulate.go` uses.

## Setup

Requirements: Go ≥ 1.22 (1.26.6 was used), and the Python environment B of `REPRODUCE.md` (`experiments/ai_detectors/.venv`:
torch 2.14, transformers 5.17, scikit-learn). The HF models `openai-community/gpt2` and `bert-base-uncased` are
downloaded on first use.

```bash
export SYNSTEG_ROOT=$(pwd)          # package root; every path below is relative to it
export PY=experiments/ai_detectors/.venv/bin/python PYTHONDONTWRITEBYTECODE=1
B=experiments/conversation_bench

# library clone (skip if the folder is already present)
git clone https://github.com/nethical6/conversation-steganography.git $B/conversation-steganography
git -C $B/conversation-steganography checkout 863a96e
git -C $B/conversation-steganography apply ../hf_model_no_grad.patch

# harness (the library package itself uses only the standard library; Go may fetch the golang.org/x/term
# module listed in the clone's go.mod if it is not in the module cache)
(cd $B/bench && go build -o ../bin/bench .)
```

`bin/bench` resolves its default paths against `$SYNSTEG_ROOT`: the clone, the example config, and `$PY` as the model
interpreter (`-python` overrides it). It appends to the output file and skips runs that are already finished, so an
interrupted `codec` job can be restarted with the same command. Payloads are deterministic: they are seeded by `-seed`
(default 20260815), the configuration, L and the run index. Stegotexts are deterministic as well, given the model.

Smoke test (2 runs, about 20 s):

```bash
$B/bin/bench codec -config default -lengths 128 -runs 2 -out /tmp/smoke.jsonl
```

## Steps

### 1. Capacity and speed: 30 runs × L ∈ {128, 256, 512} × 3 configurations (270 runs)

Run these sequentially, with nothing else on the machine, because the encode and decode times are part of the result.

```bash
for c in default u32 u1024; do
  $B/bin/bench codec -config $c -lengths 128,256,512 -runs 30 -out $B/results/capacity_$c.jsonl
done > $B/logs/capacity.log 2>&1
```

Each line records: `L`, `payload_hex`, `stegotext`, `words` (whitespace words), `gpt2_tokens`, `data_tokens`,
`finish_tokens`, `encode_s`, `decode_s`, `exact` (decoding returns the payload byte for byte), `encode_error` /
`decode_error`, and `mean_logit_regret` / `worst_logit_regret` from the library's `CarrierMetrics`. BPW = L / words,
because the text is its own cover.

### 2. Stego corpus (400 stegotexts per configuration, L = 128, encode only)

```bash
for c in default u32 u1024; do
  $B/bin/bench codec -config $c -lengths 128 -runs 400 -decode=false -seed 1 -out $B/results/corpus_$c.jsonl \
    > $B/logs/corpus_$c.log 2>&1 &
done; wait
```

The three processes may run in parallel because their timings are not used. `-seed 1` keeps the corpus payloads
separate from those of step 1. Encodes that fail are recorded with `encode_error`, and the analysis uses the first
400 successful ones. If fewer than 400 succeed, rerun the same command with a larger `-runs` (for example 405). Finished
runs are skipped, so only the missing ones are added.

### 3. Full protocol: 6 sends per configuration, 128-bit plaintexts, encryption and chaining

```bash
for c in default u32 u1024; do
  $B/bin/bench chain -config $c -sends 6 -bits 128 -out $B/results/chain_$c.jsonl > $B/logs/chain_$c.log 2>&1 &
done; wait
```

Two independent `ConversationChain` participants (Alice and Bob) share one key. They alternate sending, as in
`simulate.go`: `Send` (SIV encryption, packing, carrier trials, rolling transcript prompt), then `Receive` on the other
side. A send counts as *completed* if `Send` returns a carrier. It counts as *decoded* if `Receive` returns the plaintext
exactly. After a failed receive, the receiver's public transcript is resynchronised with `RestorePublic`, so one failure
does not cascade into the later sends. A chain run cannot be resumed: delete the file before rerunning.

### 4. Covers (about 1 min)

```bash
$PY $B/covers.py default u32 u1024
```

For each configuration there are 400 covers, one per stegotext of step 2. Each cover is sampled from GPT-2 by plain
ancestral sampling (full vocabulary, temperature 1, no top-k/top-p) from the same prompt. Each has exactly the same
number of new GPT-2 tokens as its stegotext. Seeds are fixed, and sampling runs on the CPU.

### 5. Steganalysis (about 30 min with the three configurations in parallel)

```bash
for c in default u32 u1024; do
  $PY $B/stegan.py $c > $B/logs/stegan_$c.log 2>&1 &     # -> results/steganalysis_<cfg>.json
done; wait
```

Per configuration: 400 stego (label 1) vs 400 covers (label 0), whole texts, random 80/20 split stratified by label,
and 15 % of the training part for validation. Seeds 0/1/2. Detectors: TS-CNN, TS-RNN, Bi-LSTM-Dense, BERT-LSTM-Att, with
the training loop of `steganalysis.py` (max 60 epochs, patience 8, texts truncated to 256 tokens).

### 6. Summary (about 1 min)

```bash
$PY $B/summarize.py        # -> results/summary.json, prints the Table 8 rows
```

Per configuration, `summary.json` holds:
- the per-L means of the capacity cells, and the ranges over L of BPW, encode time and logit regret;
- error counts;
- the steganalysis accuracy per detector (mean ± SD over seeds) and the range of the detector means;
- the perplexity ratio: median GPT-2 perplexity of the 400 stegotexts / median perplexity of the 400 covers;
- the full-protocol result (decoded / sends, with failure reasons).

## Notes

- Logit regret is the library's own definition: `candidates[0].LogProb − candidates[chosen].LogProb`. The backend returns
  raw logits, so this difference equals the log-probability difference, and it is ≥ 0. It is the top-ranked minus the
  chosen token, so its sign is opposite to "chosen minus most likely". It is averaged over the data tokens only: finish
  tokens are greedy, with regret 0. The "most likely" token is the top candidate after the strict-style and copy-safe
  filters, not the unfiltered arg-max of the vocabulary.
- Perplexity and steganalysis use the generated text alone, without the prompt, for both stego and cover texts.
- Covers are unfiltered GPT-2 samples. They can contain digits, brackets, quotes and newlines, which the library's
  strict-style filter removes from stegotexts. This is part of what the detectors see.
- BERT-LSTM-Att runs on MPS and is not bit-exact between runs. The word-level detectors run on the CPU.

## Results of the rebuild run (21.10.2026)

Apple M4 Pro, 14 cores, macOS. GPT-2 runs on the CPU in float32. Go 1.26.6, torch 2.14.0, transformers 5.17.0. Model
fingerprint: `hf:c868574b…` (it is stored in every JSONL line).

| config | bits/word (128-512) | encode s | logit regret | steganalysis accuracy (range of detector means) | perplexity vs cover | full protocol decoded |
|---|---|---|---|---|---|---|
| default arithmetic top-n 8 | 2.03-2.11 | 2.39-13.50 | 0.57-0.61 | 0.956-0.996 | 0.14× | 4/6 |
| uniform top-n 32 | 3.84-4.80 | 1.12-3.75 | 3.30-3.42 | 0.977-0.992 | 1.31× | 4/6 |
| uniform top-n 1024 | 6.73-8.84 | 4.88-14.38 | 5.97-7.34 | 0.973-0.994 | 10.99× | 6/6 |

- Ranges are taken over the per-L means (L = 128, 256, 512). Encode times are means over successful encodes only. Decode
  times are about equal to encode times (`summary.json`).
- Every encode that succeeded also decoded exactly: 266/270 runs exact. There were 4 encode errors at default L = 512:
  2× `only k of 64 model candidates passed strict visible-text filtering`, and 2× `index out of range in self`. The
  latter means the arithmetic coder ran past GPT-2's 1024-token context, after about 105 s. There was 1 encode error at
  u32 L = 512 (strict filtering) and none at u1024.
- Corpus (L = 128): 405 attempts for default and for u32 (5 strict-filtering errors each), 400 for u1024 (no errors).
- Full protocol failures (all in `Send`): default, send 3 "none of 2 carrier trials passed the human-writing checks" and
  send 4 `index out of range in self` (the rolling transcript prompt plus the carrier exceeded 1024 tokens); u32, sends
  0 and 2 human-writing checks. Every carrier that was sent was decoded and authenticated.
- Wall time: step 1 about 62 min (sequential); steps 2-3 about 36 min (6 processes); steps 4-5 about 40 min, overlapping
  with the end of step 2; step 6 about 1 min. In total about 2 h. Other applications were running on the machine during
  step 1.
