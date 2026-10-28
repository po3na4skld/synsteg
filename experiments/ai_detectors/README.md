# AI-generated-text detector resistance (dissertation 3.7 / 4.3.1)

Offline experiment on the 300 `reports/gpt_omni_reports` runs: no OpenAI calls, re-encoding from the stored keys.

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python torch transformers accelerate numpy scikit-learn scipy pandas textstat sentence-transformers matplotlib safetensors
.venv/bin/python build_corpus.py     # corpus.jsonl: human / cover / stego_b1..b5 / 16 key permutations
.venv/bin/python run_detectors.py    # scores/<detector>.jsonl (resumable), ~50 min on Apple Silicon
.venv/bin/python analyze.py          # results.json + tables (AUC, TPR@5%FPR, bootstrap CI, key-selection transfer matrix)
.venv/bin/python quality.py          # quality.json (GPT-2 perplexity, Flesch, MiniLM cosine, edit rate)
.venv/bin/python figures.py          # fig_auc.png fig_tpr_bpw.png fig_transfer.png
```
Run 2026-09-27: 299 usable reports, detectors D1 roberta-base-openai-detector, D2 Hello-SimpleAI/chatgpt-detector-roberta,
D3 desklib/ai-text-detector-v1.01, D4 Binoculars (Qwen2.5-1.5B / -Instruct), D5 Fast-DetectGPT (Qwen2.5-1.5B).
