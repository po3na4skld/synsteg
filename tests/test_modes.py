import math
import json

import pytest

from steganography import core, helper, modes
from steganography.modes import Mode, PROFILES, encode, decode
from models.pool_arguments import LLMConfig

MSG = "test"
OLLAMA = LLMConfig(provider="ollama", model="qwen3:30b-a3b-instruct-2507-q4_K_M")


def roundtrip(mode, passphrase="pw", **kw):
    r = encode(MSG, mode, passphrase=passphrase, **kw)
    assert r["recovered"]
    kf = json.loads(json.dumps(r["keyfile"]))  # as written to / read from the key file
    assert decode(r["stegotext"], kf, passphrase) == MSG
    assert passphrase not in json.dumps(kf)
    return r


def test_default_equals_published_encoder(container):
    r = roundtrip(Mode(), container=container)
    _, stego, key, _, _ = core.encode_message(MSG, 5, container=container)
    assert r["stegotext"] == stego and r["keyfile"]["key"] == key
    assert core.decode_message(r["stegotext"], r["keyfile"]["key"])[0] == MSG


@pytest.mark.parametrize("b", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("llm", [LLMConfig(), LLMConfig(model="gpt-6-luna"), OLLAMA])
def test_standard_generated_container(b, llm):
    r = roundtrip(Mode(bits_per_word=b), llm=llm)
    assert 0 < r["replaced_share"] <= 1 and r["keyfile"]["version"] == 2


@pytest.mark.parametrize("b,a", [(2, 1), (3, 2)])
def test_enhanced(b, a, container):
    roundtrip(Mode(bits_per_word=b, additional_bits=a), container=container)


@pytest.mark.parametrize("b", [1, 2, 3, 5])
def test_identity(b, container):
    r = roundtrip(Mode(bits_per_word=b, identity=True), container=container)
    assert all(isinstance(v, list) or len(set(v.values())) < len(v)
               for s in r["keyfile"]["key"] for v in s.values())  # the original word is not in the stored key


def test_hamming(container):
    r = roundtrip(Mode(bits_per_word=2, hamming=True), container=container)
    assert r["replaced_share"] <= math.ceil(32 / 3) / len(container.split()) + 1e-9  # <= 1 change per group of 3 bits


def test_variable_bits(container):
    r = roundtrip(Mode(variable_bits=True), container=container)
    assert len({len(next(iter(m))) for s in r["keyfile"]["key"] for m in s.values()}) > 1


@pytest.mark.parametrize("kw", [dict(ppl_choice=True), dict(delta=0.3), dict(delta=0.3, gamma=2.0),
                                dict(gamma=1.5, ppl_choice=True), dict(variable_bits=True, ppl_choice=True)])
def test_perplexity(kw, container, scorer):
    roundtrip(Mode(**kw), container=container, scorer=scorer)


def test_key_selection(container, selector):
    r = roundtrip(Mode(key_selection=True), container=container, selector=selector)
    assert 0 <= r["keyfile"]["mode"]["k"] < 16


@pytest.mark.parametrize("name", PROFILES)
def test_profiles(name, container, scorer, selector):
    roundtrip(Mode(**PROFILES[name]), container=container, scorer=scorer, selector=selector, llm=OLLAMA)


def test_profile_generated_container(scorer):
    roundtrip(Mode(**PROFILES["steganalysis"]), scorer=scorer)


def test_passphrase_required_and_checked(container):
    with pytest.raises(ValueError):
        encode(MSG, Mode(identity=True), container=container)
    r = encode(MSG, Mode(identity=True), container=container, passphrase="right")
    with pytest.raises(ValueError):
        decode(r["stegotext"], r["keyfile"])
    try:
        assert decode(r["stegotext"], r["keyfile"], "wrong") != MSG
    except ValueError:
        pass


def test_bit_messages(container):
    bits = "1011001110001111010101"
    r = encode(bits, Mode(bits_per_word=3, identity=True), container=container, passphrase="pw", binarize=False)
    assert decode(r["stegotext"], r["keyfile"], "pw") == bits


def test_container_too_short():
    with pytest.raises(ValueError):
        encode("a longer message", Mode(hamming=True), container="Only a few words here.", passphrase="pw")


def test_invalid_combinations():
    for m in (Mode(additional_bits=1, identity=True), Mode(identity=True, hamming=True),
              Mode(key_selection=True, ppl_choice=True), Mode(bits_per_word=6)):
        with pytest.raises(ValueError):
            m.check()


def test_missing_extras_message(monkeypatch, container):
    import builtins
    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda n, *a, **k: (_ for _ in ()).throw(ImportError(n))
                        if n in ("torch", "transformers") else real(n, *a, **k))
    from steganography import ml
    with pytest.raises(ImportError, match="requirements-extra"):
        ml.GPT2Scorer()


def test_ollama_request(monkeypatch):
    from steganography import gpt
    sent = {}

    class R:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": '{"words": []}'}, "prompt_eval_count": 3, "eval_count": 4}

    monkeypatch.setattr(gpt.httpx, "post", lambda url, json, timeout: sent.update(url=url, body=json) or R())
    out, usage = gpt._ollama_chat(OLLAMA, [{"role": "user", "content": "x"}], 0.7, gpt.words_schema(4))
    assert sent["url"] == "http://localhost:11434/api/chat" and sent["body"]["format"]["required"] == ["words"]
    assert sent["body"]["options"]["temperature"] == 0.7 and usage["total_tokens"] == 7


def test_openai_temperature(monkeypatch):
    from steganography import gpt
    calls = []

    class Fake:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            calls.append(kw)
            from openai.types.chat import ChatCompletion
            return ChatCompletion.model_validate(dict(id="x", object="chat.completion", created=0, model=kw["model"],
                choices=[dict(index=0, finish_reason="stop", message=dict(role="assistant", content="{}"))],
                usage=dict(prompt_tokens=1, completion_tokens=1, total_tokens=2)))

    monkeypatch.setattr(gpt.openai, "OpenAI", Fake)
    gpt._openai_chat("gpt-4o", [], 0.7, True)
    gpt._openai_chat("gpt-6-luna", [], 0.7, True)
    assert calls[0]["temperature"] == 0.7 and "temperature" not in calls[1]
    assert calls[0]["response_format"] == {"type": "json_object"}
