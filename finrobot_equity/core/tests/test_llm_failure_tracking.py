"""AI text sections that fall back to placeholder text must be reported to the caller."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules import text_generator_agents as tga
from modules.llm_providers import LLMSettings


def _settings(key="k"):
    return LLMSettings(provider="anthropic", api_key=key, model=None, base_url=None)


def test_success_is_not_recorded(monkeypatch):
    monkeypatch.setattr(tga, "generate_text", lambda *a, **k: "real text")
    failures = []
    out = tga.generate_text_section({}, "tagline", "k", "Acme", "ACME", settings=_settings(), failures=failures)
    assert out == "real text" and failures == []


def test_api_error_is_recorded(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("credit balance too low")
    monkeypatch.setattr(tga, "generate_text", boom)
    failures = []
    out = tga.generate_text_section({}, "risks", "k", "Acme", "ACME", settings=_settings(), failures=failures)
    assert failures == ["risks"]
    assert out  # placeholder text is still returned so the pipeline can finish


def test_missing_key_is_recorded():
    failures = []
    tga.generate_text_section({}, "tagline", "", "Acme", "ACME", settings=_settings(key=""), failures=failures)
    assert failures == ["tagline"]


def test_failures_argument_is_optional(monkeypatch):
    monkeypatch.setattr(tga, "generate_text", lambda *a, **k: "x")
    assert tga.generate_text_section({}, "tagline", "k", "Acme", "ACME", settings=_settings()) == "x"


def test_inr_prompts_ask_for_indian_units():
    from modules import currency
    data = {}
    try:
        currency.set_currency("INR")
        assert "crore" in tga._prepare_user_prompt(data, "company_overview", "Acme", "ACME")
        currency.set_currency("USD")
        assert "crore" not in tga._prepare_user_prompt(data, "company_overview", "Acme", "ACME")
    finally:
        currency.set_currency("USD")
