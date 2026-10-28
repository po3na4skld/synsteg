"""Offline fakes: an LLM that answers like the real one (deterministic synonym lists, containers of the asked length),
a tiny perplexity scorer and a guide detector, so no network and no torch are needed."""
import ast
import json
import os
import re
import sys
import zlib

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OPENAI_API_KEY", "offline-test")  # openai.OpenAI() is never reached: chat() is faked

VOCAB = ("people often share simple stories about their daily work and small pleasures while friends listen "
         "carefully because good conversation builds trust across communities and families over many years "
         "technology changes how we learn travel cook and rest yet basic needs remain quite similar everywhere").split()


def make_container(n_words):
    words = [VOCAB[(7 * i) % len(VOCAB)] for i in range(n_words)]
    for i in range(0, n_words, 9):
        words[i] = words[i].capitalize()
    for i in range(8, n_words, 9):
        words[i] += "."
    for i in range(4, n_words, 13):
        words[i] += ","
    return " ".join(words)


def synonyms_for(word, n):
    w = re.sub(r"[^a-z]", "", word.lower())
    out = [f"{w}{chr(97 + j // 26)}{chr(97 + j % 26)}" for j in range(n)]
    if len(w) % 3 == 0 and n > 2:
        out[1] = out[0]  # a repeat, as models produce
    return out


H = lambda s: zlib.crc32(s.encode())


def fake_chat(llm, prompt, input_message, temperature=None, json_schema=None, default_model=None):
    usage = dict(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    if json_schema is None:  # container
        n = int(re.search(r"Length: (\d+) words", input_message).group(1))
        return make_container(n), usage
    n = int(re.search(r"strictly (\d+) synonyms", prompt).group(1))
    words = ast.literal_eval(input_message[len("Text: "):])
    return json.dumps({"words": [{w: synonyms_for(w, n)} for w in words]}), usage


class FakeScorer:
    def n_tokens(self, text):
        return len(text.split()) + 1

    def lnppl(self, texts):
        return [3 + H(t) % 1000 / 1000 for t in texts]

    def nll(self, left_words, cands, right_words):
        return [H(" ".join(left_words[-3:]) + c) % 997 / 100 for c in cands]


def fake_detector(texts):
    return [H(t) % 1000 / 1000 for t in texts]


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    from steganography import gpt, secret_key
    monkeypatch.setattr(gpt, "chat", fake_chat)
    monkeypatch.setattr(secret_key, "PARALLEL_PROVIDERS", set())  # Pool workers would not see the fake


@pytest.fixture
def scorer():
    return FakeScorer()


@pytest.fixture
def selector():
    return fake_detector, [i / 50 for i in range(50)]


@pytest.fixture
def container():
    return make_container(160)
