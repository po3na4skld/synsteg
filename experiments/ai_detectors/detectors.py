"""Five AI-generated-text detectors behind one interface: score(texts) -> list[float], higher = more AI-like.

D1 roberta-base-openai-detector      supervised, GPT-2 outputs (OpenAI, 2019)
D2 chatgpt-detector-roberta          supervised, HC3 ChatGPT vs human (Guo et al., 2023)
D3 desklib/ai-text-detector-v1.01    supervised, DeBERTa-v3-large, RAID leaderboard (2025)
D4 Binoculars                        zero-shot, observer/performer cross-perplexity (Hans et al., 2024)
D5 Fast-DetectGPT (analytic)         zero-shot, conditional probability curvature (Bao et al., 2024)
"""
import os
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModel, AutoConfig, PreTrainedModel

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class _Seq:
    def __init__(self, name, ai_label):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(name).to(DEV).eval()
        self.ai = ai_label

    @torch.no_grad()
    def score(self, texts, bs=16):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
            p = self.m(**enc).logits.softmax(-1)[:, self.ai]
            out += p.float().cpu().tolist()
        return out


def roberta_openai():
    # label 0 = Fake (AI), 1 = Real
    return _Seq("openai-community/roberta-base-openai-detector", ai_label=0)


def chatgpt_roberta():
    # label 0 = Human, 1 = ChatGPT
    return _Seq("Hello-SimpleAI/chatgpt-detector-roberta", ai_label=1)


class _Desklib(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        h = self.model(input_ids, attention_mask=attention_mask)[0]
        m = attention_mask.unsqueeze(-1).expand(h.size()).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.classifier(pooled)


class Desklib:
    def __init__(self):
        name = "desklib/ai-text-detector-v1.01"
        self.tok = AutoTokenizer.from_pretrained(name)
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        self.m = _Desklib(AutoConfig.from_pretrained(name))
        missing, unexpected = self.m.load_state_dict(load_file(hf_hub_download(name, "model.safetensors")), strict=False)
        assert not [k for k in missing if not k.endswith("position_ids")], missing
        self.m = self.m.to(DEV).eval()

    @torch.no_grad()
    def score(self, texts, bs=int(os.environ.get("DET_BS", 8))):  # DET_BS=2 if MPS memory runs out
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=768, return_tensors="pt").to(DEV)
            out += torch.sigmoid(self.m(enc["input_ids"], enc["attention_mask"])).squeeze(-1).float().cpu().tolist()
        return out


def _causal(name):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(DEV).eval()
    return tok, m


class Binoculars:
    """score = log-PPL(observer) / cross-PPL(observer||performer); LOWER = more AI. We return the negative."""

    def __init__(self, observer="Qwen/Qwen2.5-1.5B", performer="Qwen/Qwen2.5-1.5B-Instruct"):
        self.tok, self.obs = _causal(observer)
        _, self.perf = _causal(performer)

    @torch.no_grad()
    def score(self, texts, bs=int(os.environ.get("DET_BS", 8))):  # DET_BS=2 if MPS memory runs out
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
            lo = self.obs(**enc).logits[:, :-1].float()
            lp = self.perf(**enc).logits[:, :-1].float()
            tgt = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].float()
            logp_o = lo.log_softmax(-1)
            ppl = -(logp_o.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * mask).sum(1) / mask.sum(1)
            xppl = -((lp.softmax(-1) * logp_o).sum(-1) * mask).sum(1) / mask.sum(1)
            out += (-(ppl / xppl)).cpu().tolist()
        return out


class FastDetectGPT:
    """Analytic sampling discrepancy with one model as both sampler and scorer. Higher = more AI."""

    def __init__(self, name="Qwen/Qwen2.5-1.5B"):
        self.tok, self.m = _causal(name)

    @torch.no_grad()
    def score(self, texts, bs=int(os.environ.get("DET_BS", 8))):  # DET_BS=2 if MPS memory runs out
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
            logits = self.m(**enc).logits[:, :-1].float()
            tgt = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].float()
            logp = logits.log_softmax(-1)
            p = logp.exp()
            ll = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            mean = (p * logp).sum(-1)
            var = (p * logp.square()).sum(-1) - mean.square()
            n = mask.sum(1)
            d = ((ll - mean) * mask).sum(1) / n
            s = ((var * mask).sum(1) / n).sqrt()
            out += (d / s).cpu().tolist()
        return out


DETECTORS = {
    "D1_roberta_openai": roberta_openai,
    "D2_chatgpt_roberta": chatgpt_roberta,
    "D3_desklib": Desklib,
    "D4_binoculars": Binoculars,
    "D5_fast_detectgpt": FastDetectGPT,
}
