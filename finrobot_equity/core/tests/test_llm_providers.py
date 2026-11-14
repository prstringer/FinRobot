#!/usr/bin/env python
# coding: utf-8
"""LLM provider switch: settings resolution and the request each provider receives.

The SDK clients are replaced by fakes, so no network calls or API spend.
Run from finrobot_equity/core with:  PYTHONPATH=src pytest tests
"""

import configparser
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modules import llm_providers  # noqa: E402
from modules.llm_providers import LLMError, LLMSettings, generate_text, resolve_llm_settings  # noqa: E402


def make_config(**keys):
    cfg = configparser.ConfigParser()
    cfg["API_KEYS"] = keys
    return cfg


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# --- Settings resolution ---------------------------------------------------------
def test_defaults_to_openai_with_legacy_openai_settings():
    cfg = make_config(openai_api_key="sk-1", openai_model="gpt-4.1",
                      openai_base_url="https://proxy.example/v1")
    s = resolve_llm_settings(cfg)
    assert (s.provider, s.api_key, s.model_name, s.base_url) == ("openai", "sk-1", "gpt-4.1", "https://proxy.example/v1")


def test_config_provider_and_argument_override():
    cfg = make_config(llm_provider="gemini", gemini_api_key="g-1", anthropic_api_key="a-1")
    assert resolve_llm_settings(cfg).provider == "gemini"
    s = resolve_llm_settings(cfg, provider="claude")  # alias
    assert (s.provider, s.api_key, s.model_name) == ("anthropic", "a-1", "claude-opus-5-5")


def test_placeholder_keys_ignored_and_env_fallback(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "env-g")
    cfg = make_config(llm_provider="gemini", gemini_api_key="YOUR_GEMINI_API_KEY")
    assert resolve_llm_settings(cfg).api_key == "env-g"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-a")
    assert resolve_llm_settings(None, provider="anthropic").api_key == "env-a"


def test_llm_model_applies_to_any_provider_and_default_openai_url_dropped():
    cfg = make_config(llm_provider="anthropic", anthropic_api_key="a", llm_model="claude-sonnet-5-5",
                      openai_base_url="https://api.openai.com/v1")
    assert resolve_llm_settings(cfg).model_name == "claude-sonnet-5-5"
    assert resolve_llm_settings(make_config(openai_api_key="k", openai_base_url="https://api.openai.com/v1")).base_url is None


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        resolve_llm_settings(None, provider="llama")


def test_missing_key_raises():
    with pytest.raises(LLMError):
        generate_text(LLMSettings(provider="openai"), "sys", "user")


# --- Requests sent to each SDK ----------------------------------------------------
def test_openai_request(monkeypatch):
    import openai
    calls = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            calls["client"] = kw
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kw):
            calls["request"] = kw
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" Hello "))])

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    out = generate_text(LLMSettings("openai", "sk", base_url="https://p/v1"), "SYS", "USER")
    assert out == "Hello"
    assert calls["client"] == {"api_key": "sk", "base_url": "https://p/v1"}
    assert calls["request"]["model"] == "gpt-4o-mini"
    assert calls["request"]["messages"] == [{"role": "system", "content": "SYS"},
                                            {"role": "user", "content": "USER"}]


class _FakeAnthropicClient:
    def __init__(self, response, calls, fail_beta=None):
        self.calls, self.response, self.fail_beta = calls, response, fail_beta
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._beta_create))
        self.messages = SimpleNamespace(create=self._create)

    def _beta_create(self, **kw):
        self.calls["beta"] = kw
        if self.fail_beta:
            raise self.fail_beta
        return self.response

    def _create(self, **kw):
        self.calls["plain"] = kw
        return self.response


def _claude_response(text="Claude says hi", stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason, stop_details=SimpleNamespace(category="cyber"),
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)])


def test_anthropic_request_uses_fallbacks_and_no_temperature(monkeypatch):
    import anthropic
    calls = {}
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key: _FakeAnthropicClient(_claude_response(), calls))
    out = generate_text(LLMSettings("anthropic", "ak"), "SYS", "USER", max_tokens=1000, temperature=0.7)
    assert out == "Claude says hi"  # thinking blocks skipped
    req = calls["beta"]
    assert req["model"] == "claude-opus-5-5"
    assert req["system"] == "SYS" and req["messages"] == [{"role": "user", "content": "USER"}]
    assert req["betas"] == ["server-side-fallback-2026-07-01"] and req["fallbacks"] == "default"
    assert req["max_tokens"] >= 8000          # room for thinking
    assert "temperature" not in req           # rejected by current Claude models


def test_anthropic_refusal_raises(monkeypatch):
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key: _FakeAnthropicClient(_claude_response("", "refusal"), {}))
    with pytest.raises(LLMError, match="declined"):
        generate_text(LLMSettings("anthropic", "ak"), "S", "U")


def test_anthropic_retries_without_fallback_beta(monkeypatch):
    import anthropic
    calls = {}
    err = anthropic.BadRequestError(
        "fallbacks is not supported for this model",
        response=SimpleNamespace(status_code=400, headers={}, request=None), body=None)
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key: _FakeAnthropicClient(_claude_response("ok"), calls, fail_beta=err))
    assert generate_text(LLMSettings("anthropic", "ak", model="claude-haiku-4-5"), "S", "U") == "ok"
    assert "fallbacks" not in calls["plain"] and calls["plain"]["model"] == "claude-haiku-4-5"


def test_gemini_request(monkeypatch):
    from google import genai
    calls = {}

    class FakeModels:
        def generate_content(self, **kw):
            calls["request"] = kw
            return SimpleNamespace(text=" Gemini text ", candidates=[])

    class FakeClient:
        def __init__(self, api_key):
            calls["key"] = api_key
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", FakeClient)
    out = generate_text(LLMSettings("gemini", "gk"), "SYS", "USER", temperature=0.7)
    assert out == "Gemini text"
    req = calls["request"]
    assert calls["key"] == "gk" and req["model"] == "gemini-3.8-flash" and req["contents"] == "USER"
    assert req["config"].system_instruction == "SYS" and req["config"].temperature == 0.7


def test_gemini_empty_response_raises(monkeypatch):
    from google import genai

    class FakeClient:
        def __init__(self, api_key):
            self.models = SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(
                text=None, candidates=[SimpleNamespace(finish_reason="SAFETY")]))

    monkeypatch.setattr(genai, "Client", FakeClient)
    with pytest.raises(LLMError, match="SAFETY"):
        generate_text(LLMSettings("gemini", "gk"), "S", "U")


def test_generate_text_section_uses_selected_provider(monkeypatch):
    from modules import text_generator_agents as t
    seen = {}

    def fake_generate(settings, system_prompt, user_prompt, **kw):
        seen.update(provider=settings.provider, system=system_prompt, user=user_prompt)
        return "Net Interest Income Growth: fine."

    monkeypatch.setattr(t, "generate_text", fake_generate)
    out = t.generate_text_section({"is_bank": True, "data_notes": ["Amounts in INR"]}, "major_takeaways",
                                  None, "HDFC Bank", "HDFCBANK",
                                  settings=LLMSettings("gemini", "gk"))
    assert out.startswith("Net Interest Income Growth")
    assert seen["provider"] == "gemini"
    assert "bank analyst" in seen["system"]          # bank-specific prompt
    assert "Amounts in INR" in seen["user"]          # data notes reach the prompt
    # legacy positional call (OpenAI) still works and falls back on errors
    monkeypatch.setattr(t, "generate_text", lambda *a, **k: (_ for _ in ()).throw(LLMError("boom")))
    assert "Revenue Growth" in t.generate_text_section({}, "major_takeaways", "sk", "X", "X")


# ---- usage and cost logging -------------------------------------------------

def test_cost_usd_prices_each_token_category():
    from modules.llm_providers import cost_usd
    # Opus 5.5: $4 in / $20 out, cache read 0.05x, 5-minute cache write 1.25x
    c = cost_usd("claude-opus-5-5", input_tokens=1_000_000, output_tokens=1_000_000,
                 cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
    assert c == pytest.approx(4 + 20 + 4 * 0.05 + 4 * 1.25)
    assert cost_usd("claude-haiku-5-5", input_tokens=1_000_000) == pytest.approx(0.10)
    assert cost_usd("some-unknown-model", input_tokens=5) is None


def test_anthropic_usage_is_logged_with_section_and_thinking(monkeypatch):
    import anthropic
    resp = _claude_response("hello")
    resp.model = "claude-opus-5-5"
    resp.usage = SimpleNamespace(
        input_tokens=2000, output_tokens=5000, cache_read_input_tokens=0,
        cache_creation_input_tokens=0, output_tokens_details=SimpleNamespace(thinking_tokens=4600))
    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key: _FakeAnthropicClient(resp, {}))
    sink = []
    generate_text(LLMSettings("anthropic", "ak"), "S", "U", usage_sink=sink, section="risks")
    assert len(sink) == 1
    rec = sink[0]
    assert rec["section"] == "risks" and rec["model"] == "claude-opus-5-5"
    assert rec["thinking_tokens"] == 4600 and rec["output_tokens"] == 5000
    assert rec["cost_usd"] == pytest.approx((2000 * 4 + 5000 * 20) / 1_000_000)


def test_usage_logging_is_optional(monkeypatch):
    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key: _FakeAnthropicClient(_claude_response("ok"), {}))
    assert generate_text(LLMSettings("anthropic", "ak"), "S", "U") == "ok"  # response has no usage; must not matter


# ---- model tiers ------------------------------------------------------------

def test_anthropic_without_forced_model_is_tiered_per_section():
    s = LLMSettings("anthropic", "ak")
    assert s.tiered
    assert s.route("investment_overview") == ("claude-opus-5-5", "medium")
    assert s.route("risks") == ("claude-sonnet-5-5", "low")
    assert s.route("tagline") == ("claude-haiku-5-5", "low")
    assert s.route("some_new_section") == ("claude-opus-5-5", "medium")  # unknown sections get the strong tier


def test_forced_model_or_other_provider_is_not_tiered():
    forced = LLMSettings("anthropic", "ak", model="claude-sonnet-5-5")
    assert not forced.tiered and forced.route("tagline") == ("claude-sonnet-5-5", None)
    assert not LLMSettings("openai", "k").tiered
    assert not LLMSettings("gemini", "k").tiered


def test_tiered_request_sets_model_and_effort_and_skips_fallback_for_haiku(monkeypatch):
    import anthropic
    calls = {}
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key: _FakeAnthropicClient(_claude_response("ok"), calls))
    generate_text(LLMSettings("anthropic", "ak"), "S", "U", section="tagline")
    assert "beta" not in calls                      # Haiku: no fallback beta
    assert calls["plain"]["model"] == "claude-haiku-5-5"
    assert calls["plain"]["output_config"] == {"effort": "low"}
    calls.clear()
    generate_text(LLMSettings("anthropic", "ak"), "S", "U", section="valuation_overview")
    assert calls["beta"]["model"] == "claude-opus-5-5"
    assert calls["beta"]["output_config"] == {"effort": "medium"}
    assert calls["beta"]["fallbacks"] == "default"


def test_forced_model_sends_no_effort(monkeypatch):
    import anthropic
    calls = {}
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda api_key: _FakeAnthropicClient(_claude_response("ok"), calls))
    generate_text(LLMSettings("anthropic", "ak", model="claude-opus-4-8"), "S", "U", section="risks")
    assert calls["beta"]["model"] == "claude-opus-4-8" and "output_config" not in calls["beta"]
