"""Score every record of arms/*.jsonl with D1-D5 (ai_detectors/detectors.py) and the GPT-2 judge (quality.py ppl).
A text already scored in corpus 3.7 (ai_detectors/scores) reuses that score. Resumable: scores/<name>.jsonl keyed by
arm|uuid|cand. Usage: DET_BS=2 <ai_detectors venv python> score.py [arm ...] [--only D1_roberta_openai,...]
"""
import gc, hashlib, json, sys, time
import torch
from common import HERE, AID, ARMS, DETS

sys.path.insert(0, str(AID))
SC = HERE / "scores"; SC.mkdir(exist_ok=True)


def rid(r):
    return f'{r["arm"]}|{r["uuid"]}|{r.get("cand", -1)}'


def th(t):
    return hashlib.sha1(t.encode()).hexdigest()[:12]


def corpus37_cache(name):
    if not (AID / "scores" / f"{name}.jsonl").exists():
        return {}
    text = {}
    for l in open(AID / "corpus.jsonl"):
        r = json.loads(l); text[f'{r["uuid"]}|{r["arm"]}|{r.get("cand", -1)}'] = r["text"]
    return {text[s["id"]]: s["score"] for s in map(json.loads, open(AID / "scores" / f"{name}.jsonl"))}


class GPT2:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from quality import DEV
        self.tok = AutoTokenizer.from_pretrained("openai-community/gpt2"); self.tok.pad_token = self.tok.eos_token
        self.m = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(DEV).eval()

    def score(self, texts):
        from quality import ppl
        return ppl(texts, self.tok, self.m, bs=4)


def main(arms, only):
    recs = [r for a in arms for r in map(json.loads, open(ARMS / f"{a}.jsonl"))]
    for name in only:
        out = SC / f"{name}.jsonl"
        # a record counts as scored only for the text it had (re-encoded texts are scored again; the last line wins)
        done = {(x["id"], x.get("h")) for x in map(json.loads, open(out))} if out.exists() else set()
        todo = [r for r in recs if (rid(r), th(r["text"])) not in done and (rid(r), None) not in done]
        if not todo:
            continue
        cache = corpus37_cache(name)
        hit = [r for r in todo if r["text"] in cache]
        with open(out, "a") as fh:
            for r in hit:
                fh.write(json.dumps({"id": rid(r), "h": th(r["text"]), "score": cache[r["text"]], "cached": True}) + "\n")
        todo = [r for r in todo if r["text"] not in cache]
        print(f"{name}: {len(hit)} from corpus 3.7, {len(todo)} to score", flush=True)
        if not todo:
            continue
        if name == "gpt2_ppl":
            det = GPT2()
        else:
            from detectors import DETECTORS
            det = DETECTORS[name]()
        t0 = time.time()
        with open(out, "a") as fh:
            for i in range(0, len(todo), 32):
                chunk = todo[i:i + 32]
                bs = 4 if name[:2] in ("D1", "D2", "D3", "gp") else 2  # memory rule
                for r, s in zip(chunk, det.score([r["text"] for r in chunk], bs=bs) if name != "gpt2_ppl" else det.score([r["text"] for r in chunk])):
                    fh.write(json.dumps({"id": rid(r), "h": th(r["text"]), "score": s}) + "\n")
                fh.flush()
        print(f"  {name} {len(todo)} in {time.time() - t0:.0f}s", flush=True)
        del det; gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    args = sys.argv[1:]
    only = DETS + ["gpt2_ppl"]
    if "--only" in args:
        only = args[args.index("--only") + 1].split(","); args = args[:args.index("--only")]
    main(args or sorted(p.stem for p in ARMS.glob("*.jsonl")), only)
