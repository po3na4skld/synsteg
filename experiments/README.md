# Experiments

Every folder below is self-contained research code: it reads the stored run reports (`reports/`) or corpora and writes
its results next to the scripts. Nothing here calls a paid API unless stated. Third-party code is fetched by
`setup_third_party.sh` (exact upstream commits plus the local patches in this tree).

Python environments and exact package versions: `requirements-synsteg.txt` (series runner),
`requirements-experiments.txt` (detectors, metrics, steganalysis, perplexity), `requirements-textsteganalysis.txt`
(original DeepStego steganalysis code). The commands are in `../REPRODUCE.md`, section 2.

| Folder | Dissertation | What it does | Main results |
|---|---|---|---|
| `model_series/` | 3.8 | series runner for any model (OpenAI or local ollama), metrics (capacity, recovery, perplexity, Flesch), AI-text detectors on a series | `../reports/*_metrics.jsonl`, `../reports/model_series_summary.json`, `../reports/detectors_*.json` |
| `qwen_probe/` | 3.8 | how many distinct synonyms a local Qwen3-30B returns per word; offline enhanced-mode check | `probe_results.jsonl` |
| `steganalysis/` | 3.5, 3.8 | protocol A (sentence pairs, random split, 4 detectors, 3 seeds) and protocol A′ (original TextSteganalysis code of the DeepStego study) | `../reports/steganalysis_sentences.json`, `../reports/textsteganalysis_summary.json` |
| `conversation_bench/` | 3.6 | generative baseline conversation-steganography on GPT-2: capacity, time, reliability, steganalysis, text quality | `results/summary.json` |
| `ai_detectors/` | 3.7, appendix Е | corpus of 299 GPT-4o containers × 7 groups + 16 key permutations; detectors D1-D5; AUC, TPR at 5 % FPR, bootstrap CIs, key-selection transfer | `results.json`, `quality.json`, `analyze.out`, `scores/` |
| `ppl_opt/` | 4.3.4 | synonym choice by GPT-2 perplexity (formula 4.8), quality with an independent judge, detectors, steganalysis; `ds2023/` – the same on the DeepStego corpus | `quality.json`, `detectors.json`, `ds2023/textsteganalysis_ds2023_summary.json` |
| `improvements/` | 4.3.1-4.3.3, 4.3.6 | two-sided key selection (4.3) and PRP keys (algorithm 4.1), variable bit width (4.5), substitution-share regulator and Hamming (7,3) (4.6), perplexity window (4.7), adversary profiles with steganalysis | `analyze.json`, `select_key.json`, `steganalysis_summary.json`, `arms/` |
| `watermark/` | 4.5, appendix Б | SynthID-Text watermark vs synonym embedding with gpt-6-luna synonyms; control = random replacement at the same count | `results/luna_rows.json`, `results/luna_paired.json`, `results/luna_post.json` |

Notes
- Steganalysis numbers comparable with the published 0.5000-0.7644 come only from protocol A / A′ (sentence pairs,
  random shuffle), as in the DeepStego study. `steganalysis/steganalysis.py` (protocol B, whole texts, split by pair) is
  a stricter separate test.
- The memory-heavy steps (detectors D4/D5, BERT-LSTM-Att, GPT-2 window search) were run one model at a time on an Apple
  M4 Pro with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5`.
- `watermark/fetch_synonyms.py` is the only script here that calls an API (gpt-6-luna, 5.48 USD for 112 texts); its
  responses are cached in `watermark/cache/`, so every other step reruns offline.
