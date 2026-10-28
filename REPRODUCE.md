# Reproducing the experiments of the article

Table numbers below refer to the article draft; the map from dissertation subsections to data and code is in `README.md`.

"SynSteg synonym steganography on new large language models: comparison with Conversation-Steganography and
perplexity-based synonym choice" (Chernov, draft of 21.10.2026).

The package holds the code (branch `missing-synonyms`, commit `e87605e`), the experiment scripts, the raw reports
of every series used in the article and the result files. Every table can be recomputed offline from the stored
reports. Only Tables 1-4 need new calls to a language model (OpenAI API or a local Ollama server).

Hardware used: Apple M4 Pro, 48 GB RAM, macOS, PyTorch on MPS. CUDA works too; see "Notes on determinism".

## 1. Layout

| Path | Contents |
|---|---|
| `steganography/`, `utils/`, `models/` | SynSteg code (branch `missing-synonyms`: words with ≥ 90 % distinct synonyms are used, missing codes keep the word, prefix-free synonym lists, adaptive container length) |
| `experiments/model_series/` | series runner (`run_series.py`), metrics (`metrics.py`, `summarize.py`), AI-text detectors on a series (`detect.py`) |
| `experiments/qwen_probe/` | probe of distinct synonyms returned by Qwen3-30B (Table 1) |
| `experiments/steganalysis/` | sentence-level steganalysis (`steganalysis_sentences.py`, Table 5), TextSteganalysis runner (`run_textsteganalysis.py`, Table 6), patch for TextSteganalysis |
| `experiments/ai_detectors/` | detectors D1-D5 (`detectors.py`) and the human IMDB texts used as negatives (`corpus.jsonl`, arm `human`) |
| `experiments/ppl_opt/` | perplexity-based synonym choice on the GPT-4o reports (Table 9, upper part); `ds2023/` – the same on the DeepStego study corpus (Table 9, lower part) |
| `reports/gpt6_luna_missing/`, `reports/qwen_missing_b3/` | raw reports of the gpt-6-luna and Qwen series (one JSON per run: message, key, container, stegotext, timings, token usage) |
| `reports/gpt_omni_reports/` | 300 GPT-4o reports (09.2024) used in Table 9 |
| `reports/json_reports/` | 300 reports (12.2023) behind the DeepStego study corpus |
| `reports/qwen_series/` | earlier Qwen series; the probe of Table 1 samples its containers |
| `artifacts/synsteg_{clean,stego}.txt` | DeepStego study corpus (4 331 sentence pairs); `synsteg_gptomni_*` – GPT-4o sentence corpus (1 231 pairs) |
| `notebooks/steganalysis_training/training_steganalysis_models.ipynb` | original DeepStego steganalysis notebook; cell 8 holds the TextSteganalysis config reused verbatim |
| `reports/*.json`, `reports/*_metrics.jsonl`, `experiments/**/**.json` | results as obtained for the article |

No API key or password is included. `config.json` is not part of the package and is not needed.

## 2. Environments

Three Python environments were used. Their exact package versions are in `experiments/requirements-*.txt`.

```bash
# A. series runner (Python 3.12): SynSteg code + OpenAI client
python3.12 -m venv .venv && .venv/bin/pip install -r experiments/requirements-synsteg.txt

# B. metrics, detectors, steganalysis, perplexity choice (Python 3.12, PyTorch 2.14, transformers 5.17)
python3.12 -m venv experiments/ai_detectors/.venv
experiments/ai_detectors/.venv/bin/pip install -r experiments/requirements-experiments.txt

# C. original DeepStego steganalysis code (Python 3.11, torch 2.5.1, transformers 4.46.2)
git clone https://github.com/yjs1224/TextSteganalysis.git experiments/steganalysis/TextSteganalysis
git -C experiments/steganalysis/TextSteganalysis checkout ff773a4
git -C experiments/steganalysis/TextSteganalysis apply ../textsteganalysis_local.patch
python3.11 -m venv experiments/steganalysis/.venv_ts
experiments/steganalysis/.venv_ts/bin/pip install -r experiments/requirements-textsteganalysis.txt
```

The patch changes only the device: MPS instead of CUDA in `main.py`, and two `.cuda()` calls in `models/lstmatt.py`
that now use the model's device. On a CUDA machine the patch is optional.

Hugging Face models are downloaded on first use: `gpt2`, `bert-base-uncased`, `Qwen/Qwen2.5-1.5B`,
`openai-community/roberta-base-openai-detector`, `Hello-SimpleAI/chatgpt-detector-roberta`,
`desklib/ai-text-detector-v1.01`. About 6 GB of disk.

Set the package root once per shell; the scripts read it:

```bash
export SYNSTEG_ROOT=$(pwd)
```

## 3. Tables

Times are for the hardware above. "Offline" means the step uses only the stored reports.

### Table 1 – distinct synonyms returned by Qwen3-30B (needs Ollama)

```bash
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M        # Ollama 0.31.1 was used
experiments/ai_detectors/.venv/bin/python experiments/qwen_probe/probe.py   # -> experiments/qwen_probe/probe_results.jsonl
```

Stored result: `experiments/qwen_probe/probe_results.jsonl`, log `probe.log`.

### Tables 2-4 – reliability, capacity, time, cost and text quality

New series (needs the OpenAI API for gpt-6-luna, about 10 USD for 300 runs, or Ollama for Qwen):

```bash
cd experiments/model_series
export SYNSTEG_CODE=$SYNSTEG_ROOT          # run the checked-out branch, not the published commit 683e8cb
# gpt-6-luna, b = 5, 100 runs per length
for L in 128 256 512; do
  OPENAI_API_KEY=... SERIES_MODEL=gpt-6-luna SERIES_B=5 \
    ../../.venv/bin/python run_series.py reports/gpt6_luna_missing $L 100   # output dir is relative to the package root
done
# Qwen3-30B via Ollama, b = 3, 25 runs per length
for L in 128 256 512; do
  SERIES_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M OLLAMA_URL=http://localhost:11434 SERIES_B=3 \
    ../../.venv/bin/python run_series.py reports/qwen_missing_b3 $L 25
done
```

Metrics (offline, about 5 min):

```bash
SERIES_B=5 experiments/ai_detectors/.venv/bin/python experiments/model_series/metrics.py reports/gpt6_luna_missing
SERIES_B=3 experiments/ai_detectors/.venv/bin/python experiments/model_series/metrics.py reports/qwen_missing_b3   # b of the series
experiments/ai_detectors/.venv/bin/python experiments/model_series/summarize.py \
  luna=reports/gpt6_luna_missing_metrics.jsonl qwen=reports/qwen_missing_b3_metrics.jsonl
```

Stored results: `reports/gpt6_luna_missing_metrics.jsonl`, `reports/qwen_missing_b3_metrics.jsonl`.
The GPT-4o rows of Tables 2-4 are the published values of the DeepStego paper.

### Table 5 – sentence-level steganalysis (offline, about 70 min)

```bash
cd experiments/steganalysis && ../ai_detectors/.venv/bin/python steganalysis_sentences.py
```

Protocol: each cover and stegotext is split into sentences (`(?<=[.!?])\s+`); cover sentence = label 0, stego sentence =
label 1; random 80/20 shuffle of sentences stratified by label, 15 % of the training part for validation; seeds 0, 1, 2;
sentences truncated to 64 tokens; BERT frozen. Corpora: `artifacts/synsteg_*` (DeepStego study corpus),
`artifacts/synsteg_gptomni_*` (GPT-4o, 1 231 pairs), and the two new series split into sentences.
Output: `reports/steganalysis_sentences.json`.

### Table 6 – original DeepStego steganalysis code (offline, about 25 min; environment C)

```bash
cd experiments/steganalysis && .venv_ts/bin/python run_textsteganalysis.py
```

Reads the config from cell 8 of `notebooks/steganalysis_training/training_steganalysis_models.ipynb` (60/20/20 random
split, 3 epochs, seed 42, one run). Output: `reports/textsteganalysis_summary.json`; logs and the generated CSV splits
in `datasets/textsteganalysis_runs/`. Control: on the DeepStego corpus BiRNN 0.5741 and Bi-LSTM-Dense 0.5026 match the
2024 notebook output exactly.

### Table 7 – AI-text detectors (offline, about 10 min)

```bash
cd experiments/model_series
DET_BS=2 ../ai_detectors/.venv/bin/python detect.py luna_missing=../../reports/gpt6_luna_missing qwen_missing_b3=../../reports/qwen_missing_b3
```

Negatives: the 299 human IMDB texts of `experiments/ai_detectors/corpus.jsonl`. TPR@5 uses the 95th percentile of the
human scores as threshold. `DET_BS=2` avoids running out of memory on Binoculars with 48 GB.
Output: `reports/detectors_luna_missing_qwen_missing_b3.json`.

### Table 8 – comparison with Conversation-Steganography

The SynSteg rows come from Tables 2-5. The Conversation-Steganography rows come from our benchmark of
https://github.com/nethical6/conversation-steganography; see `experiments/conversation_bench/README.md`.

### Table 9 – perplexity-based synonym choice (offline)

GPT-4o reports (upper part):

```bash
cd experiments/ppl_opt
PY=../ai_detectors/.venv/bin/python
$PY optimize.py                     # -> series/opt_k{32,8,4}/, about 20 min
$PY eval_quality.py                 # -> quality.json (perplexity ratios, Flesch, time), about 5 min
cd ../model_series && DET_BS=2 $PY detect.py gpt4o_base=../../reports/gpt_omni_reports gpt4o_pplopt=../ppl_opt/series/opt_k32
                                    # -> reports/detectors_gpt4o_base_gpt4o_pplopt.json (stored as ppl_opt/detectors.json)
```

DeepStego study corpus (lower part):

```bash
cd experiments/ppl_opt/ds2023
../../ai_detectors/.venv/bin/python optimize_ds2023.py          # -> ../series/ds2023_opt/, about 6 min
../../ai_detectors/.venv/bin/python eval_quality_ds2023.py      # -> quality_ds2023.json
../../steganalysis/.venv_ts/bin/python run_textsteganalysis_ds2023.py   # seeds 42/43/44 -> textsteganalysis_ds2023_summary.json, about 80 min
```

`codec2023.py` is the SynSteg encoder/decoder of 14.12.2023, which produced `reports/json_reports`; with the original
keys it reproduces the stored stegotexts in 300/300 runs, and `corpus.py` rebuilds `artifacts/synsteg_*` byte for byte.

## 4. Notes on determinism

- `optimize.py`, `optimize_ds2023.py`, `run_textsteganalysis_ds2023.py` and `run_detectors.py` resume: they skip runs whose
  output already exists. The package ships these outputs, so to recompute Table 9 from scratch delete them first:
  `rm -rf experiments/ppl_opt/series/opt_k* experiments/ppl_opt/series/ds2023_opt experiments/ppl_opt/ds2023/textsteganalysis_ds2023_summary.json`.

- LLM calls (Tables 1-4) are not deterministic: gpt-6-luna accepts only temperature 1, and API timing depends on load.
  Re-running gives new containers and keys; the stored reports are the ones analysed in the article.
- Steganalysis on MPS is not bit-exact between runs for BERT-based models (differences in the third decimal).
  On the DeepStego corpus the CNN, BiRNN and Bi-LSTM-Dense values of Table 6 reproduce exactly.
- Bi-LSTM-Dense in TextSteganalysis collapses to a single class on these corpora; its accuracy is the class share of the test set.

## 5. Verification of this package (21.10.2026)

The package was unpacked into an empty folder, the three environments were built with the commands of Section 2, and
every offline step of Section 3 was run from scratch (Table 9 outputs deleted first). Results against the stored files:

| Table | Result |
|---|---|
| 2-4 (metrics) | identical (60 344 values for gpt-6-luna, 21 686 for Qwen) |
| 5 (sentence-level steganalysis) | identical (578 values) |
| 6 (TextSteganalysis) | 26 of 30 values identical; LSTM-Att (BERT) differs by up to 0.007 (MPS non-determinism) |
| 7 (AI-text detectors) | identical (96 values) |
| 9, GPT-4o reports | optimised stegotexts identical in 300/300 runs for K = 32, 8, 4; perplexity, Flesch and detector values identical; median choice time 1.27 s (stored 1.28 s) |
| 9, DeepStego corpus | optimised stegotexts identical in 300/300 runs; perplexity values identical; TextSteganalysis CNN, BiRNN and Bi-LSTM-Dense identical, LSTM-Att and R-BiLSTM-C differ by up to 0.006 in the mean over three seeds |
| 8, Conversation-Steganography | harness rebuilt on 21.10.2026 (`experiments/conversation_bench/`); its `results/summary.json` holds the values now in Table 8; not re-run from a clean copy (about 2 h) |

Live check: two Qwen3-30B runs at 128 bits through Ollama with `run_series.py` recovered both messages exactly (about 70 s per run).
Offline wall time on the hardware above: about 3.5 hours.
