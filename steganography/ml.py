"""Optional models of subsection 4.3 (pip install -r requirements-extra.txt): GPT-2 judge for 4.7 / 4.8 and the guide
detector for key selection (4.3). Imported lazily by steganography.modes; one instance per process."""
import json
import os
import pathlib
from functools import lru_cache

ROOT = pathlib.Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "experiments/ai_detectors/corpus.jsonl"
GUIDE_DETECTOR = "openai-community/roberta-base-openai-detector"  # D1; label 0 = Fake (AI)
BS = 4  # memory rule of the experiments: batch <= 4


def _torch():
    try:
        import torch
        import transformers
    except ImportError:
        raise ImportError("perplexity choice / window and key selection need torch and transformers: "
                          "pip install -r requirements-extra.txt") from None
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch, transformers, dev


class GPT2Scorer:
    """GPT-2 (124M): ln PPL of whole texts (4.7) and the 4.8 window NLL (experiments/ppl_opt/optimize.py)."""

    def __init__(self, name="openai-community/gpt2"):
        self.torch, tr, self.dev = _torch()
        self.tok = tr.AutoTokenizer.from_pretrained(name)
        self.tok.pad_token = self.tok.eos_token
        self.m = tr.AutoModelForCausalLM.from_pretrained(name).to(self.dev).eval()

    def n_tokens(self, text):
        return len(self.tok(text).input_ids)

    def lnppl(self, texts):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), BS):
                enc = self.tok(texts[i:i + BS], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.dev)
                lp = self.m(**enc).logits[:, :-1].float().log_softmax(-1)
                tgt, mask = enc["input_ids"][:, 1:], enc["attention_mask"][:, 1:].float()
                ll = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                out += (-(ll * mask).sum(1) / mask.sum(1)).cpu().tolist()
        return out

    def nll(self, left_words, cands, right_words, left=20):
        """Summed NLL of ' cand right...' after the last `left` tokens of the stegotext, one value per candidate."""
        torch, out = self.torch, []
        bos = [self.tok.eos_token_id]
        ctx = (bos + self.tok(" ".join(left_words)).input_ids)[-left - 1:] if left_words else bos
        tail = (" " + " ".join(right_words)) if right_words else ""
        with torch.no_grad():
            for k in range(0, len(cands), BS):
                rights = [self.tok((" " if left_words else "") + c + tail).input_ids for c in cands[k:k + BS]]
                T = len(ctx) + max(map(len, rights))
                ids = torch.full((len(rights), T), self.tok.eos_token_id)
                tgt_mask = torch.zeros((len(rights), T))
                for j, r in enumerate(rights):
                    ids[j, :len(ctx) + len(r)] = torch.tensor(ctx + r)
                    tgt_mask[j, len(ctx):len(ctx) + len(r)] = 1
                ids, tgt_mask = ids.to(self.dev), tgt_mask.to(self.dev)
                lp = self.m(ids).logits[:, :-1].float().log_softmax(-1)
                ll = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) * tgt_mask[:, 1:]
                out += (-ll.sum(1)).cpu().tolist()
        return out


class Detector:
    """Sequence classifier, score = P(AI) (experiments/ai_detectors/detectors.py, D1)."""

    def __init__(self, name=GUIDE_DETECTOR, ai_label=0):
        self.torch, tr, self.dev = _torch()
        self.tok = tr.AutoTokenizer.from_pretrained(name)
        self.m = tr.AutoModelForSequenceClassification.from_pretrained(name).to(self.dev).eval()
        self.ai = ai_label

    def __call__(self, texts):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), BS):
                enc = self.tok(texts[i:i + BS], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.dev)
                out += self.m(**enc).logits.softmax(-1)[:, self.ai].float().cpu().tolist()
        return out


def calibration_texts(path=None):
    """Human texts of corpus 3.7 (IMDB reviews), or None if the corpus is absent."""
    path = pathlib.Path(path or os.environ.get("SYNSTEG_CALIBRATION", CALIBRATION))
    if not path.exists():
        return None
    return [r["text"] for r in map(json.loads, open(path)) if r["arm"] == "human"]


@lru_cache(maxsize=1)
def get_scorer():
    return GPT2Scorer()


@lru_cache(maxsize=1)
def get_selector():
    """(guide detector, its scores on the human calibration texts) for key selection."""
    texts = calibration_texts()
    if not texts:
        raise FileNotFoundError(f"key selection needs human calibration texts ({CALIBRATION} or SYNSTEG_CALIBRATION); "
                                "none found, key selection skipped")
    det = Detector()
    return det, det(texts)
