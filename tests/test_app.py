"""The Streamlit app and the CLI runner with the fake LLM (no network)."""
import os

import pytest
from streamlit.testing.v1 import AppTest

import local_runner
from steganography.modes import Mode, PROFILES
from conftest import ROOT, make_container


def run_app(profile, provider="Local (ollama)", passphrase="pw"):
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60)
    at.run()
    at.radio(key="provider").set_value(provider)
    at.selectbox(key="profile").set_value(profile)
    at.run()
    at.text_input(key="passphrase").set_value(passphrase)
    at.main.text_input[0].set_value("test")
    at.main.text_area[0].set_value(make_container(160))
    at.button[0].click()
    at.run()
    return at


@pytest.mark.parametrize("profile", ["custom", "steganalysis_keep_capacity", "max_capacity"])
def test_app_encodes(profile, monkeypatch, scorer):
    from steganography import ml
    monkeypatch.setattr(ml, "get_scorer", lambda: scorer)
    at = run_app(profile)
    assert not at.exception and not at.error, (at.exception, at.error)
    assert any("Bits embedded: 32" in m.value for m in at.markdown)
    assert not at.warning  # the stegotext decodes back


def test_app_password_gate(monkeypatch):
    monkeypatch.setenv("SYNSTEG_PASSWORD_SHA512", "00" * 64)
    at = AppTest.from_file(os.path.join(ROOT, "app.py"))
    at.run()
    assert len(at.text_input) == 1 and not at.selectbox  # only the password field


def test_runner_flags():
    a = local_runner.parser.parse_args(["-container_path", "x", "-profile", "steganalysis", "-bits_per_word", "3"])
    assert local_runner.mode_from_args(a) == Mode(**{**PROFILES["steganalysis"], "bits_per_word": 3})
    a = local_runner.parser.parse_args(["-container_path", "x", "-identity", "-delta", "0.05", "-ppl_choice"])
    assert local_runner.mode_from_args(a) == Mode(identity=True, delta=0.05, ppl_choice=True)
