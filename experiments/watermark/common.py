"""Shared pieces: SynthID-Text config (HF transformers, ngram_len = 5), mean-g detector, GPT-2 perplexity, text I/O.

The sampling table of the HF processor is drawn with a torch.Generator on the processor's device, so generation and
detection must use the same device: both run on CPU here.
"""
import json, pathlib, re
import numpy as np, torch

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
# the 9-key example of the SynthIDTextWatermarkingConfig docstring: it reproduces the old human-class spread of mean g
# (sd 0.011 here vs 0.0125 in Table B.1; the 30-key list of the HF blog gives 0.0069)
KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57]
NGRAM = 5
WM_DEVICE = torch.device("cpu")


def wm_config():
    from transformers import SynthIDTextWatermarkingConfig
    return SynthIDTextWatermarkingConfig(keys=KEYS, ngram_len=NGRAM)


def normalize(text: str) -> str:
    """One line, single spaces, no brackets (the SynSteg container prompt forbids them: align_container_and_secret_key
    strips them from the container but not from the model's keys, so "(word" stalls the key alignment for the rest of
    the text); tokens left empty by the cleaner are dropped (they would shift words against key positions)."""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[()\[\]{}]", "", text)
    return " ".join(t for t in text.split() if t.strip(",.?!()[]{}"))


def load(name):
    return [json.loads(l) for l in open(DATA / name)]


def save(name, rows):
    DATA.mkdir(exist_ok=True)
    with open(DATA / name, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class MeanG:
    """Unweighted mean g-value over all tournament layers; repeated (n-1)-gram contexts are masked as in the
    reference detector. Texts are re-tokenized with the generator's tokenizer."""

    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(QWEN)
        self.proc = wm_config().construct_processor(len(self.tok), WM_DEVICE)

    @torch.no_grad()
    def __call__(self, text: str) -> float:
        ids = torch.tensor([self.tok(text).input_ids], device=WM_DEVICE)
        if ids.shape[1] < NGRAM + 1:
            return float("nan")
        g = self.proc.compute_g_values(ids).float()                  # (1, L-4, depth)
        m = self.proc.compute_context_repetition_mask(ids).float()   # (1, L-4)
        return float((g * m[..., None]).sum() / (m.sum() * g.shape[-1]))


class PPL:
    """GPT-2 (124M) perplexity of a whole text (first 1024 tokens)."""

    def __init__(self, device=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.dev = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.m = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(self.dev).eval()

    @torch.no_grad()
    def __call__(self, text: str) -> float:
        ids = [self.tok.eos_token_id] + self.tok(text).input_ids[:1023]
        x = torch.tensor([ids], device=self.dev)
        return float(torch.exp(self.m(x, labels=x).loss))


def auc_tpr(pos, neg):
    """AUC (ties count 1/2) and TPR at FPR <= 1 % (threshold above the top 1 % of negatives; with 56 negatives that is
    strictly above the maximum negative score)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    auc = ((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())
    k = int(np.floor(0.01 * len(neg)))
    thr = np.sort(neg)[::-1][k]
    return float(auc), float((pos > thr).mean())


SPECIAL = [",", ".", "!", "?", "..."]  # utils.constants.SPECIAL_TOKENS: trailing punctuation kept by the encoder


def split_special(token):
    """As steganography.helper.check_endswith_special: (word, trailing punctuation)."""
    for s in SPECIAL:
        if token.endswith(s):
            return token[:-len(s)], s
    return token, ""


def match_case(new, old):
    return new.capitalize() if old.istitle() else new


def random_control(text, n_replace, cands_of, rng):
    """Replace n_replace randomly chosen eligible words by a uniformly drawn candidate (never the word itself).
    cands_of(i, word) -> candidate list of position i (the same lexicon as the attack)."""
    toks = text.split()
    elig = []
    for i, t in enumerate(toks):
        w, _ = split_special(t)
        c = [s for s in cands_of(i, w) if s.lower() != w.lower()]
        if c:
            elig.append((i, c))
    for i, c in rng.sample(elig, min(n_replace, len(elig))):
        w, sp = split_special(toks[i])
        toks[i] = match_case(rng.choice(c), w) + sp
    return " ".join(toks), min(n_replace, len(elig))


def evaluate(name, pos, neg, base, n_words, ppl, n_rep, bits):
    """pos/neg: mean-g lists of the attacked classes; base: unattacked mean g of the watermarked class (scalar);
    n_words/ppl/n_rep/bits: per watermarked text. Returns one table row."""
    a, t = auc_tpr(pos, neg)
    g = float(np.mean(pos))
    return dict(attack=name, n_pos=len(pos), n_neg=len(neg), auc=a, tpr1=t, mean_g=g, mean_g_neg=float(np.mean(neg)),
                residual=(g - 0.5) / (base - 0.5), share=float(np.sum(n_rep) / np.sum(n_words)),
                bits=float(np.mean(bits)), ppl=float(np.mean(ppl)), ppl_median=float(np.median(ppl)))


def paired(g_a, g_b, base, B=20000, seed=0):
    """Paired bootstrap of mean(g_a - g_b) over texts (Б.3)."""
    d = np.asarray(g_a) - np.asarray(g_b)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=len(d), diff=float(d.mean()), lo=float(lo), hi=float(hi), significant=bool(hi < 0 or lo > 0),
                a_better=int((d < 0).sum()), share=float(d.mean() / (base - 0.5)))
