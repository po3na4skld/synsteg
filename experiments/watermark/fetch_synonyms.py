"""Fetch gpt-6-luna synonym tables (2^5 = 32 synonyms per word) once per text, through the SynSteg code of commit e87605e.

Every text of data/wm.jsonl and data/human.jsonl is split into the same 5-word chunks as core.encode_message, and each
chunk is sent with steganography.secret_key.generate_synonyms (its prompt, its 3 format retries). The only change
is the one of model_series/run_series.py: the model id, and `temperature` dropped (gpt-6-luna accepts only 1).
Keys for b = 1..5 are derived later from these cached lists (first 2^b distinct synonyms, binarize_synonyms).

Cache: cache/<text id>/<chunk idx>.json, one file per answered request, so a rerun only sends what is missing.
Token usage of every API call (format retries included) is appended to cache/usage.jsonl.

Usage:
  .venv/bin/python fetch_synonyms.py --estimate        # cost estimate, no calls, no key needed
  .venv/bin/python fetch_synonyms.py                   # real run, OPENAI_API_KEY from the environment
  .venv/bin/python fetch_synonyms.py --mock --cache /tmp/x --limit 2   # offline test with a mocked client
"""
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "synsteg_e87605e"))
MODEL, CHUNK, BITS = "gpt-6-luna", 5, 5
PRICE_IN, PRICE_OUT = 0.10e-6, 0.50e-6  # USD per token
# per request, measured on the gpt-6-luna e87605e series (22_luna_missing_series.md): 2.44 M in + 19.20 M out over
# ~7.9 k requests of 5 words at 32 synonyms (~81 % of output is reasoning)
EST_IN, EST_OUT = 309, 2430

_lock = threading.Lock()


def texts(limit):
    rows = []
    for name in ("wm.jsonl", "human.jsonl"):
        rows += [json.loads(l) for l in open(HERE / "data" / name)][:limit]
    return rows


def install_client(usage_log: Path, mock: bool):
    import openai
    from openai.resources.chat.completions import Completions
    from openai.types.chat import ChatCompletion
    from utils import constants
    constants.OPENAI_MODEL_SYNONYMS = constants.OPENAI_MODEL_CONTAINER = MODEL
    real = Completions.create

    def fake(self, *, model, messages, **kw):  # echoes the chunk: 32 synthetic synonyms per word
        assert "temperature" not in kw and model == MODEL and kw["response_format"] == {"type": "json_object"}
        words = eval(messages[1]["content"][len("Text: "):])  # the prompt carries the chunk as a Python list repr
        content = json.dumps({"words": [{w: [f"{w.lower()}x{k}" for k in range(32)]} for w in words]})
        return ChatCompletion.model_validate(dict(
            id="mock", object="chat.completion", created=0, model=model,
            choices=[dict(index=0, finish_reason="stop", message=dict(role="assistant", content=content))],
            usage=dict(prompt_tokens=300, completion_tokens=2400, total_tokens=2700)))

    def create(self, *a, **kw):
        kw.pop("temperature", None)
        try:
            r = (fake if mock else real)(self, *a, **kw)
        except openai.APIStatusError as e:
            if 400 <= e.status_code < 500 and e.status_code != 429:  # would be retried forever by backoff
                raise RuntimeError(f"HTTP {e.status_code}: {e.message}") from None
            raise
        with _lock, open(usage_log, "a") as f:
            f.write(json.dumps(dict(t=time.time(), **{k: getattr(r.usage, k) for k in ("prompt_tokens", "completion_tokens")})) + "\n")
        return r
    Completions.create = create


def total_usage(usage_log: Path):
    pi = po = n = 0
    if usage_log.exists():
        for l in open(usage_log):
            u = json.loads(l); pi += u["prompt_tokens"]; po += u["completion_tokens"]; n += 1
    return n, pi, po, pi * PRICE_IN + po * PRICE_OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(HERE / "cache"))
    ap.add_argument("--limit", type=int, default=None, help="texts per class")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args()
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    usage_log = cache / "usage.jsonl"

    jobs = []
    for r in texts(a.limit):
        words = r["text"].split()
        for k in range(0, len(words), CHUNK):
            if not (cache / r["id"] / f"{k // CHUNK:03d}.json").exists():
                jobs.append((r["id"], k // CHUNK, words[k:k + CHUNK]))
    est = len(jobs) * (EST_IN * PRICE_IN + EST_OUT * PRICE_OUT)
    print(f"{len(jobs)} requests to send; estimated {len(jobs) * EST_IN / 1e6:.2f} M in + "
          f"{len(jobs) * EST_OUT / 1e6:.2f} M out tokens = {est:.2f} USD", flush=True)
    if a.estimate or not jobs:
        return
    if a.mock:
        os.environ["OPENAI_API_KEY"] = "mock-not-a-key"
    elif not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set")
    install_client(usage_log, a.mock)
    from steganography.secret_key import generate_synonyms
    from models.pool_arguments import PoolArguments

    def work(job):
        tid, idx, chunk = job
        words, usage = generate_synonyms(PoolArguments(container_split=chunk, bits_per_word=BITS))
        out = cache / tid / f"{idx:03d}.json"
        out.parent.mkdir(exist_ok=True)
        tmp = out.with_suffix(".tmp")
        json.dump(dict(chunk=chunk, words=words, usage=usage, model=MODEL), open(tmp, "w"), ensure_ascii=False)
        tmp.rename(out)  # atomic: an interrupted run never leaves a half-written cache file

    t0, done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        for f in as_completed(futs):
            try:
                f.result(); done += 1
            except Exception as e:  # not cached -> retried on the next run
                failed += 1
                print(f"FAILED {futs[f][0]}/{futs[f][1]}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            if (done + failed) % 100 == 0:
                n, pi, po, usd = total_usage(usage_log)
                print(f"{done + failed}/{len(jobs)} ok={done} failed={failed} {time.time() - t0:.0f}s "
                      f"calls={n} in={pi} out={po} spent={usd:.2f} USD", flush=True)
    n, pi, po, usd = total_usage(usage_log)
    print(f"done ok={done} failed={failed} in {time.time() - t0:.0f}s; cumulative calls={n} in={pi} out={po} "
          f"cost={usd:.2f} USD", flush=True)


if __name__ == "__main__":
    main()
