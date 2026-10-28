"""SynSteg series with another LLM, method unchanged: code of commit 683e8cb (the state that produced
reports/gpt_omni_reports: container = floor(|m|/5) words, key requests per 10 words, 32 synonyms, b=5, a=0),
loop as in statistics_collection.py. Only the model id (and, for a local model, the endpoint) is swapped.

Usage (any cwd):
  local Qwen via ollama:  SERIES_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M OLLAMA_URL=http://localhost:11434 \
                          python run_series.py reports/qwen_series 128 30
  OpenAI model:           SERIES_MODEL=gpt-6-luna python run_series.py reports/gpt6_luna_series 128 3   (OPENAI_API_KEY from env)
The override sits at module top level so multiprocessing (spawn) workers that re-import this file get it too.
"""
import os, sys, subprocess, time, random
from uuid import uuid4

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../.."))
CODE = os.environ.get("SYNSTEG_CODE") or os.path.join(HERE, "synsteg_683e8cb")  # SYNSTEG_CODE=<repo> runs the checked-out branch
if CODE.endswith("synsteg_683e8cb") and not os.path.isdir(CODE):
    os.makedirs(CODE)
    subprocess.run(f"git -C {REPO} archive 683e8cb steganography models utils | tar -x -C {CODE}", shell=True, check=True)
sys.path.insert(0, CODE)

MODEL, OLLAMA = os.environ["SERIES_MODEL"], os.environ.get("OLLAMA_URL")
if OLLAMA:
    os.environ.setdefault("OPENAI_API_KEY", "local-ollama")  # placeholder so openai.OpenAI() constructs; never sent anywhere
elif not os.environ.get("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY is not set")

from utils import constants
constants.OPENAI_MODEL_CONTAINER = constants.OPENAI_MODEL_SYNONYMS = MODEL

if MODEL.startswith("gpt-6"):  # API accepts only the default temperature (HTTP 400 otherwise); 400 must not be retried forever
    import openai
    from openai.resources.chat.completions import Completions
    _create = Completions.create

    def _create_default_t(self, *a, **kw):
        kw.pop("temperature", None)
        try:
            return _create(self, *a, **kw)
        except openai.BadRequestError as e:
            raise RuntimeError(f"BadRequest: {e.message}") from None
    Completions.create = _create_default_t

if OLLAMA:  # thin adapter: same request (messages, temperature, JSON mode) sent to the local ollama server
    import httpx
    from openai.types.chat import ChatCompletion
    from openai.resources.chat.completions import Completions
    # ollama 0.31 ignores format="json" for this model (no grammar); a schema of the shape the prompt asks for
    # is the working equivalent of OpenAI JSON mode (syntactically valid JSON guaranteed). maxItems = 2^b = 32 is the
    # prompt's own "strictly N_SYNONYMS" (b = 5 in these series): without it Qwen sometimes lists synonyms for one word
    # until the token limit (smoke test 2026-10-18: truncated JSON); the method never uses more than 2^b items anyway.
    WORDS_SCHEMA = {"type": "object", "required": ["words"], "properties": {"words": {"type": "array", "items": {
        "type": "object", "additionalProperties": {"type": "array", "maxItems": 32, "items": {"type": "string"}}}}}}

    def _ollama_create(self, *, model, messages, temperature=None, response_format=None, **_):
        body = dict(model=model, messages=messages, stream=False, think=False,
                    options=dict(num_predict=4096, **({} if temperature is None else dict(temperature=temperature))))
        if response_format:
            body["format"] = WORDS_SCHEMA
        r = httpx.post(OLLAMA + "/api/chat", json=body, timeout=1200)
        r.raise_for_status()
        r = r.json()
        pi, ci = r.get("prompt_eval_count", 0), r.get("eval_count", 0)
        return ChatCompletion.model_validate(dict(
            id="ollama", object="chat.completion", created=0, model=model,
            choices=[dict(index=0, finish_reason="length" if r.get("done_reason") == "length" else "stop",
                          message=dict(role="assistant", content=r["message"]["content"]))],
            usage=dict(prompt_tokens=pi, completion_tokens=ci, total_tokens=pi + ci)))
    Completions.create = _ollama_create

if __name__ == "__main__":
    from steganography import core
    from models.report import ReportModel

    out, L, n = os.path.join(REPO, sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    for _ in range(n):
        message = "".join(str(random.randint(0, 1)) for _ in range(L))
        kw = dict(uuid=str(uuid4()), message=message, container="", encoded_message="", secret_key=[],
                  encoding_time_report={}, encoding_usage_report={})
        t = time.time()
        try:
            container, enc, sk, tr, ur = core.encode_message(message, bits_per_word=int(os.environ.get("SERIES_B", 5)), additional_bits=int(os.environ.get("SERIES_A", 0)), binarize=False)
            dec, dt = core.decode_message(enc, sk, clean_output=False)
            kw.update(container=container, encoded_message=enc, secret_key=sk, encoding_time_report=tr,
                      encoding_usage_report=ur, decoding_time=dt, decoded_message=dec)
        except Exception as e:  # runner-level only: record and continue the series
            kw["error_message"] = f"{type(e).__name__}: {e}"
        ReportModel(**kw).to_json(out)
        print(MODEL, L, kw["uuid"], f"{time.time() - t:.1f}s",
              "exact" if kw.get("decoded_message") == message else f"dec={len(kw.get('decoded_message', ''))}/{L} {kw.get('error_message', '')}",
              flush=True)
