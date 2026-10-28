"""Old plain-list keys of the stored series still decode through modes.decode, and the missing-synonyms rule
(binarize_synonyms) on stored synonym lists (moved here from tests_missing_synonyms.py). Skipped without reports/."""
import glob
import json
import os
import random

import pytest

from steganography import core, helper
from steganography.modes import decode
from steganography.secret_key import binarize_synonyms

from conftest import ROOT

REPORTS = os.path.join(ROOT, "reports")


def load(folder):
    files = sorted(glob.glob(os.path.join(REPORTS, folder, "*.json")))
    if not files:
        pytest.skip(f"reports/{folder} absent")
    return [d for d in map(lambda f: json.load(open(f)), files) if d["secret_key"]]


def test_old_keys_decode():
    reps = load("gpt_omni_reports")
    for d in reps:  # the stored decoded_message is what the decoder gave at the time
        assert decode(d["encoded_message"], d["secret_key"], as_text=False) == d["decoded_message"], d["uuid"]
    assert len(reps) == 300


def test_binarize_rules():
    # 30 of 32 distinct -> usable, codes 11110/11111 absent; 20 of 32 -> unusable
    k = binarize_synonyms("base", [f"w{i}" for i in range(30)] + ["w0", "w1"], 5)
    assert len(k) == 30 and "11111" not in k
    assert binarize_synonyms("base", [f"w{i}" for i in range(20)] * 2, 5) == {"0": "base", "1": "base"}
    assert "Up" not in binarize_synonyms("up.", ["Up", "up", "upward", "higher"], 2).values()  # punctuation/case
    assert "by means of trade" not in binarize_synonyms(
        "through", ["via", "by means of", "by means of trade", "across", "over"], 2).values()  # prefix-free


@pytest.mark.parametrize("folder,b", [("gpt6_luna_missing", 5), ("qwen_missing_b3", 3)])
def test_missing_synonyms_stress(folder, b):
    """Keys rebuilt from the stored synonym lists with binarize_synonyms: 20 random messages per key decode."""
    rnd = random.Random(1)
    for d in load(folder):
        key = [{w: binarize_synonyms(w, list(m.values()), b)} for e in d["secret_key"] for w, m in e.items()]
        for _ in range(20):
            msg = "".join(rnd.choice("01") for _ in range(len(d["message"])))
            k = [{w: dict(m) for w, m in s.items()} for s in key]  # embed shortens the last used slot in place
            stego, n = core.embed(d["container"], k, msg)
            assert core.decode_message(stego, k, clean_output=False)[0] == msg[:n]
