#!/usr/bin/env python
# coding: utf-8
"""
LLM provider switch for the report's AI text sections.

Supported providers:

* ``openai``    - OpenAI Chat Completions (``openai`` package). Also works with
                  OpenAI-compatible proxies through ``openai_base_url``.
* ``anthropic`` - Claude through the Anthropic Messages API (``anthropic`` package).
* ``gemini``    - Google Gemini through the Gen AI SDK (``google-genai`` package).

Choose with ``--llm-provider`` on the command line, ``llm_provider`` in
config.ini, or the web app's AI model selector. Keys come from config.ini
([API_KEYS] openai_api_key / anthropic_api_key / gemini_api_key) or the
environment (OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY or GOOGLE_API_KEY).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

PROVIDERS = ("openai", "anthropic", "gemini")

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-5-5",
    "gemini": "gemini-3.8-flash",
}

# USD per million tokens: (input, output, cache-read multiplier). Used only for the
# per-report cost estimate; prompts up to 100K tokens (Haiku is dearer beyond that).
MODEL_PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.05),
    "claude-sonnet-5-5": (2.00, 10.00, 0.05),
    "claude-haiku-5-5": (0.10, 0.50, 0.10),
}


def cost_usd(model: str, input_tokens: int = 0, output_tokens: int = 0,
             cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float | None:
    """Estimated cost of one call in USD (None for a model without a known price).

    ``output_tokens`` includes thinking tokens, which are billed as output.
    Cache writes are priced at the 5-minute rate (1.25x input).
    """
    price = MODEL_PRICES.get(model)
    if price is None:
        return None
    in_rate, out_rate, read_mult = price
    return (input_tokens * in_rate + cache_write_tokens * in_rate * 1.25
            + cache_read_tokens * in_rate * read_mult + output_tokens * out_rate) / 1_000_000


def _usage_record(settings, response, section: str | None) -> dict:
    """Flatten an Anthropic response's usage into a loggable dict."""
    u = response.usage
    details = getattr(u, "output_tokens_details", None)
    rec = {
        "section": section,
        "model": getattr(response, "model", None) or settings.model_name,
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "cache_read_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_write_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "thinking_tokens": getattr(details, "thinking_tokens", None),
    }
    rec["cost_usd"] = cost_usd(rec["model"], rec["input_tokens"], rec["output_tokens"],
                               rec["cache_read_tokens"], rec["cache_write_tokens"])
    return rec


PROVIDER_LABELS = {"openai": "OpenAI", "anthropic": "Anthropic Claude", "gemini": "Google Gemini"}

_KEY_CONFIG = {"openai": "openai_api_key", "anthropic": "anthropic_api_key", "gemini": "gemini_api_key"}
_KEY_ENV = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}
_PLACEHOLDER_PREFIXES = ("YOUR_", "<")


class LLMError(RuntimeError):
    """A text-generation request failed or returned no usable text."""


@dataclass
class LLMSettings:
    """Which provider and model to use, and how to reach it."""

    provider: str = "openai"
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None  # OpenAI-compatible proxies only

    @property
    def model_name(self) -> str:
        return self.model or DEFAULT_MODELS[self.provider]

    @property
    def label(self) -> str:
        return f"{PROVIDER_LABELS[self.provider]} ({self.model_name})"


def _config_value(config, key: str) -> str | None:
    if config is None:
        return None
    try:
        value = config.get("API_KEYS", key, fallback=None)
    except Exception:
        return None
    if not value or not value.strip() or value.strip().upper().startswith(_PLACEHOLDER_PREFIXES):
        return None
    return value.strip()


def normalize_provider(name: str | None) -> str:
    """Map user input ('claude', 'Google', 'gpt') to a provider id."""
    n = (name or "openai").strip().lower()
    aliases = {"claude": "anthropic", "google": "gemini", "gpt": "openai", "chatgpt": "openai"}
    n = aliases.get(n, n)
    if n not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider {name!r}; choose one of {', '.join(PROVIDERS)}")
    return n


def resolve_llm_settings(config=None, provider: str | None = None, model: str | None = None) -> LLMSettings:
    """Work out provider, key, model and base URL from arguments, config.ini and the environment.

    Precedence: explicit argument > config.ini > environment variable > default.
    The OpenAI-only ``openai_model`` / ``openai_base_url`` settings are kept for
    existing config files; ``llm_model`` applies to whichever provider is chosen.
    """
    provider = normalize_provider(provider or _config_value(config, "llm_provider"))
    api_key = _config_value(config, _KEY_CONFIG[provider])
    if not api_key:
        api_key = next((os.environ[v] for v in _KEY_ENV[provider] if os.environ.get(v)), None)
    model = model or _config_value(config, "llm_model")
    base_url = None
    if provider == "openai":
        model = model or _config_value(config, "openai_model")
        base_url = _config_value(config, "openai_base_url")
        if base_url and base_url.rstrip("/") == "https://api.openai.com/v1":
            base_url = None  # the default; no need to override
    return LLMSettings(provider=provider, api_key=api_key, model=model, base_url=base_url)


def generate_text(settings: LLMSettings, system_prompt: str, user_prompt: str,
                  max_tokens: int = 1000, temperature: float = 0.7,
                  usage_sink: list | None = None, section: str | None = None) -> str:
    """Send one system + user prompt to the configured provider and return the text reply.

    If ``usage_sink`` is given, token usage and estimated cost of the call are appended to it
    (Anthropic only).
    """
    if not settings.api_key:
        raise LLMError(f"No API key for {PROVIDER_LABELS[settings.provider]}")
    if settings.provider == "openai":
        return _generate_openai(settings, system_prompt, user_prompt, max_tokens, temperature)
    if settings.provider == "anthropic":
        return _generate_anthropic(settings, system_prompt, user_prompt, max_tokens, usage_sink, section)
    return _generate_gemini(settings, system_prompt, user_prompt, max_tokens, temperature)


def _generate_openai(settings, system_prompt, user_prompt, max_tokens, temperature) -> str:
    from openai import OpenAI

    kwargs = {"api_key": settings.api_key}
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=settings.model_name,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise LLMError("OpenAI returned an empty response")
    return text


def _generate_anthropic(settings, system_prompt, user_prompt, max_tokens,
                        usage_sink=None, section=None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.api_key)
    # Current Claude models think before answering and that counts toward max_tokens,
    # so allow well beyond the length of a report section. Sampling parameters
    # (temperature) are not sent: current models reject them.
    request = {
        "model": settings.model_name,
        "max_tokens": max(max_tokens, 8000),
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    try:
        # Server-side fallback: if a safety classifier declines, the API retries on a
        # suitable fallback model within the same call instead of returning nothing.
        response = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request)
    except anthropic.BadRequestError as e:
        # Older models (or a gateway) may not accept the fallback beta: retry without it.
        if "fallback" not in str(e).lower():
            raise
        response = client.messages.create(**request)
    if usage_sink is not None:
        usage_sink.append(_usage_record(settings, response, section))
    if response.stop_reason == "refusal":
        category = getattr(getattr(response, "stop_details", None), "category", None)
        raise LLMError(f"Claude declined the request (category: {category})")
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise LLMError(f"Claude returned no text (stop_reason: {response.stop_reason})")
    return text


def _generate_gemini(settings, system_prompt, user_prompt, max_tokens, temperature) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.api_key)
    response = client.models.generate_content(
        model=settings.model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            # Gemini 2.5+ models think first and that counts toward the output limit
            max_output_tokens=max(max_tokens, 8000),
            temperature=temperature,
        ),
    )
    text = (response.text or "").strip()
    if not text:
        reason = None
        if getattr(response, "candidates", None):
            reason = getattr(response.candidates[0], "finish_reason", None)
        raise LLMError(f"Gemini returned no text (finish_reason: {reason})")
    return text
