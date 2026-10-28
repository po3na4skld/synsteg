# Steganalysis protocols

Two protocols; compare numbers only within one.

- **A, DeepStego (matches published 0.5000-0.7644):** `steganalysis_sentences.py`. One sample = one sentence (cover 0, stego 1),
  random 80/20 shuffle of sentences stratified by label, val 15 % of train, MAXLEN 64, seeds 0/1/2. Old corpus =
  `artifacts/synsteg_{clean,stego}.txt` (line-aligned, 4331 non-empty pairs = 8662 samples, unchanged pairs kept).
  Series JSONs are split into sentences with `(?<=[.!?])\s+`. Output `reports/steganalysis_sentences.json`.
- **B, whole texts, split by pair:** `steganalysis.py`. Cover and its stego always in the same part, MAXLEN 256.
  Much higher numbers; not comparable with the published ones. Output `reports/steganalysis_missing.json`.

Commands and expected outputs: `REPRODUCE.md` in the package root.
