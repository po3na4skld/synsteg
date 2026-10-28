# SynSteg: synonym-based linguistic steganography with large language models

Code, data and experiments of the PhD dissertation of Kyrylo Chernov, «Аналіз та розробка моделі та методу
стеганографічного захисту інформації з використанням глибинного навчання» (125 Cybersecurity, V. N. Karazin Kharkiv
National University, 2026).

A large language model writes a cover text (the container) and, for every word, a table of 2^b contextual synonyms
(the secret key). The message bits pick which synonym replaces each word. The receiver needs only the stegotext and
the key. Published results: [DeepStego, *Computers* 14(5):165, 2025](https://doi.org/10.3390/computers14050165) and
[ICYRIME 2025, CEUR-WS Vol-3984](https://ceur-ws.org/Vol-3984/p13.pdf).

## Repository map

| Path | Contents |
|---|---|
| `steganography/` | the method: container and key generation (`gpt.py`, `secret_key.py`), embedding and decoding (`core.py`), all modes of chapter 4 (`modes.py`), optional GPT-2 judge and RoBERTa detector (`ml.py`) |
| `app.py`, `local_runner.py` | Streamlit application and command-line runner, both exposing every mode |
| `tests/` | offline tests: a round trip for every mode and profile, the app, the CLI, decoding of the 300 stored GPT-4o reports |
| `reports/` | raw run reports, one JSON per run (message, container, key, stegotext, timings, token usage), and metric summaries |
| `artifacts/` | DeepStego steganalysis corpus (4 331 sentence pairs) and the GPT-4o sentence corpus |
| `datasets/` | IMDB sample (human texts), steganalysis splits and the sentence pairs of every steganalysis run |
| `experiments/` | every experiment of chapters 3-4, see [`experiments/README.md`](experiments/README.md) |
| `notebooks/` | metric notebooks of the published series and the original DeepStego steganalysis notebook |
| `docs/reports/` | HTML reports: comparison with conversation-steganography, first watermark measurement (WordNet, superseded by `experiments/watermark/`) |
| `papers/` | the DeepStego article |
| `REPRODUCE.md` | environments and commands to recompute the results |

## Dissertation → data and code

| Dissertation | Data | Code | Results |
|---|---|---|---|
| 3.1-3.4 capacity, time, cost, readability (GPT-4 series, 12.2023) | `reports/json_reports/` | `notebooks/metrics_creation.ipynb` | published in ICYRIME 2025 |
| 3.1-3.4 the same, GPT-4o series (09.2024) | `reports/gpt_omni_reports/` | `notebooks/gpt_omni_metrics_creation.ipynb`, `experiments/model_series/metrics.py` | published in DeepStego; `reports/gpt_omni_reports_metrics.jsonl` |
| 3.5 steganalysis | `artifacts/synsteg_{clean,stego}.txt` | `notebooks/steganalysis_training/`, `experiments/steganalysis/` | `reports/steganalysis_sentences.json`, `reports/textsteganalysis_summary.json` |
| 3.6 comparison with conversation-steganography | GPT-2 texts in `experiments/conversation_bench/results/` | `experiments/conversation_bench/` | `results/summary.json`, `docs/reports/synsteg_vs_conversation_steganography.html` |
| 3.7 AI-text detectors, key permutations; appendix Е | `experiments/ai_detectors/corpus.jsonl` | `experiments/ai_detectors/` | `results.json`, `quality.json`, `analyze.out` |
| 3.8 gpt-6-luna and local Qwen3-30B | `reports/gpt6_luna_missing/`, `reports/qwen_missing_b3/`, `reports/qwen_series/` | `experiments/model_series/`, `experiments/qwen_probe/` | `reports/*_metrics.jsonl`, `reports/detectors_luna_missing_qwen_missing_b3.json` |
| 4.3.1-4.3.3, 4.3.6 improvements and profiles | corpus 3.7, `reports/json_reports/` | `experiments/improvements/` | `analyze.json`, `select_key.json`, `steganalysis_summary.json` |
| 4.3.4 synonym choice by perplexity | `reports/gpt_omni_reports/` | `experiments/ppl_opt/` | `quality.json`, `detectors.json`, `ds2023/` |
| 4.5, appendix Б SynthID-Text watermark | `experiments/watermark/data/`, cached gpt-6-luna answers in `cache/` | `experiments/watermark/` | `results/luna_rows.json`, `results/luna_paired.json` |

Not in the repository: trained detector checkpoints (36 GB, retrain with the scripts), virtual environments, the full
IMDB dataset (download from [Maas et al.](https://ai.stanford.edu/~amaas/data/sentiment/) into `datasets/`), and the two
third-party code bases, which `experiments/setup_third_party.sh` fetches at the exact commits with the local patches.

## Install

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt           # method, app, tests
.venv/bin/pip install -r requirements-extra.txt     # torch + transformers: perplexity choice, window, key selection
.venv/bin/python -m pytest                          # offline, no API calls
```

## Streamlit application

```bash
export OPENAI_API_KEY=...        # or type it in the sidebar; or use a local model through ollama
.venv/bin/streamlit run app.py
```

Optional: `SYNSTEG_PASSWORD_SHA512` (SHA-512 of a password) turns on a password gate; `SYNSTEG_MONGODB` (JSON
settings) stores reports in MongoDB. Without them the app needs neither.

The sidebar selects:
- **model**: OpenAI `gpt-4o` (the published series) or `gpt-6-luna`, or a local model through
  [ollama](https://ollama.com) (default `qwen3:30b-a3b-instruct-2507-q4_K_M`, nothing leaves the computer);
- **b** = 1-5 bits per word, **standard** or **enhanced** mode, an optional own container;
- **improvements of subsection 4.3**, each independent: original word in the synonym set (4.6), Hamming (7,3) matrix
  embedding, variable bit width (4.5), synonym choice by GPT-2 perplexity (4.8), perplexity window δ and target γ
  (4.7), detector-guided key selection (4.3, algorithm 4.1);
- or a **profile** from table 4.8: `steganalysis`, `steganalysis_keep_capacity`, `ai_text_detectors`,
  `detector_ensemble`, `reader_d1`, `max_capacity`.

Encoding returns the stegotext and a key file. The key file records the mode (no secrets); modes that use the shared
passphrase (identity code, Hamming, key selection) ask for it again when decoding. Old plain-list keys from `reports/`
still decode.

## Command line

```bash
.venv/bin/python local_runner.py -container_path container.txt -provider ollama -profile steganalysis_keep_capacity -passphrase '...'
.venv/bin/python local_runner.py -container_path container.txt -model gpt-6-luna -bits_per_word 4 -message_length 256
```

`-h` lists every flag; flags override the chosen profile.

## Notes

- Steganalysis numbers comparable with the published 0.5000-0.7644 use sentence pairs with a random split, as in the
  DeepStego study (`experiments/steganalysis/`, protocol A / A′).
- `gpt-6-luna` accepts only temperature 1; the client drops the parameter for it.
- No API key or password is stored in the repository; `config.json` is ignored by git.
