import os
from typing import Any, Dict
from datetime import date, datetime, timezone

# Configuration settings shared across the application

# The Prime visualization/generated-code prototype is archived for this release.
# This is a release boundary, not an environment-controlled feature flag.
ARCHIVED_TEMPLATE_IDS = frozenset({"prime_visualization"})
ARCHIVED_EXECUTION_MESSAGE = (
    "Prime visualization and generated-code experiments are archived. "
    "Existing results remain available to inspect."
)


def is_archived_template(template_id: str, template_data: Any = None) -> bool:
    if template_id in ARCHIVED_TEMPLATE_IDS:
        return True
    evaluation = (template_data or {}).get("evaluation", {}) if isinstance(template_data, dict) else {}
    return isinstance(evaluation, dict) and bool(evaluation.get("artifact_generation"))


# Current text/vision generation models, verified against provider docs 2026-10-13.
# Preserve legacy routing/price metadata for old exports; do not offer retired IDs.
LEGACY_LLM_MODELS = [
    {"id": "gemini-2.0-flash-lite", "provider": "google_generative_ai"},
    {"id": "gemini-2.0-flash", "provider": "google_generative_ai"},
    {"id": "gemini-2.5-flash-lite", "provider": "google_generative_ai"},
    {"id": "gemini-2.5-flash", "provider": "google_generative_ai"},
    {"id": "gemini-2.5-pro", "provider": "google_generative_ai"},
    {"id": "gemini-3-flash-preview", "provider": "google_generative_ai"},
    {"id": "gemini-3.1-flash-lite-preview", "provider": "google_generative_ai"},
    {"id": "gpt-5.6-luna", "provider": "openai"},
    {"id": "gpt-5.6-terra", "provider": "openai"},
    {"id": "gpt-5.6-sol", "provider": "openai"},
]
LLM_MODELS = [
    {"id": "gemini-3.5-flash-lite", "name": "Gemini 3.5 Flash-Lite · Efficient", "provider": "google_generative_ai"},
    {"id": "gemini-3.8-flash", "name": "Gemini 3.8 Flash · Balanced", "provider": "google_generative_ai"},
    {"id": "gemini-4-argon", "name": "Gemini 4 Argon · Advanced reasoning", "provider": "google_generative_ai"},
    {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro · Preview", "provider": "google_generative_ai"},
    {"id": "gemini-3.6-flash", "name": "Gemini 3.6 Flash · Previous generation", "provider": "google_generative_ai"},
    {"id": "gpt-6-luna", "name": "GPT-6 Luna · Efficient", "provider": "openai"},
    {"id": "gpt-6.1-sol", "name": "GPT-6.1 Sol · Balanced", "provider": "openai"},
    {"id": "gpt-6-astra", "name": "GPT-6 Astra · Advanced reasoning", "provider": "openai"},
]
MODEL_PROVIDERS = {model["id"]: model["provider"] for model in LEGACY_LLM_MODELS + LLM_MODELS}
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_MODELS_BY_PROVIDER = {"google_generative_ai": DEFAULT_MODEL, "openai": "gpt-6-luna"}
# Template design is a deliberate, infrequent authoring step. Keep its defaults
# separate from inexpensive evolution-run defaults, and use the same catalog.
TEMPLATE_GENERATION_DEFAULTS_BY_PROVIDER = {
    "google_generative_ai": "gemini-4-argon",
    "openai": "gpt-6-astra",
}
DEFAULT_TEMPLATE_GENERATION_MODEL = TEMPLATE_GENERATION_DEFAULTS_BY_PROVIDER["google_generative_ai"]
MODEL_CATALOG_VERIFIED_AT = "2026-10-13"
OPENAI_PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
GEMINI_PRICING_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"


def get_model_provider(model_name: str) -> str:
    """Resolve current and historical model IDs without changing saved IDs."""
    name = str(model_name or "").removeprefix("models/")
    return MODEL_PROVIDERS.get(name, "openai" if name.startswith("gpt-") else "google_generative_ai")

# Default creative settings
DEFAULT_CREATIVE_TEMP = 1.5
DEFAULT_TOP_P = 0.90

# Thinking controls per model.
# Gemini 3 models use thinking_level. Gemini 2.5 models use thinking_budget.
THINKING_MODEL_CONFIG = {
    "gemini-2.0-flash-lite": {
        "supports_thinking": False,
    },
    "gemini-2.0-flash": {
        "supports_thinking": False,
    },
    "gemini-2.5-flash-lite": {
        "supports_thinking": True,
        "supports_thinking_level": False,
        "supports_thinking_budget": True,
        "allow_off": True,
        "allowed_levels": ["off", "low", "medium", "high"],
        "default_level": "off",
        "default_budget": 0,
        "min_budget": 512,
        "max_budget": 24576,
    },
    "gemini-2.5-flash": {
        "supports_thinking": True,
        "supports_thinking_level": False,
        "supports_thinking_budget": True,
        "allow_off": True,
        "allowed_levels": ["off", "low", "medium", "high"],
        "default_level": "high",
        "default_budget": -1,
        "min_budget": 0,
        "max_budget": 24576,
    },
    "gemini-2.5-pro": {
        "supports_thinking": True,
        "supports_thinking_level": False,
        "supports_thinking_budget": True,
        "allow_off": False,
        "allowed_levels": ["low", "medium", "high"],
        "default_level": "high",
        "default_budget": -1,
        "min_budget": 128,
        "max_budget": 32768,
    },
    "gemini-3-flash-preview": {
        "supports_thinking": True,
        "supports_thinking_level": True,
        "supports_thinking_budget": False,
        "allow_off": False,
        "off_replacement_level": "minimal",
        "allowed_levels": ["minimal", "low", "medium", "high"],
        "default_level": "low",
        "default_budget": None,
        "min_budget": 0,
        "max_budget": 24576,
    },
    "gemini-3.1-flash-lite-preview": {
        "supports_thinking": True,
        "supports_thinking_level": True,
        "supports_thinking_budget": False,
        "allow_off": False,
        "off_replacement_level": "minimal",
        "allowed_levels": ["minimal", "low", "medium", "high"],
        "default_level": "low",
        "default_budget": None,
        "min_budget": 0,
        "max_budget": 24576,
    },
    "gemini-3.1-pro-preview": {
        "supports_thinking": True,
        "supports_thinking_level": True,
        "supports_thinking_budget": False,
        "allow_off": False,
        "allowed_levels": ["low", "medium", "high"],
        "default_level": "low",
        "default_budget": None,
        "min_budget": 128,
        "max_budget": 32768,
    },
    "gemini-3.6-flash": {
        "supports_thinking": True,
        "supports_thinking_level": True,
        "supports_thinking_budget": False,
        "supports_sampling_parameters": False,
        "allow_off": False,
        "allowed_levels": ["low", "medium", "high"],
        "default_level": "medium",
        "default_budget": None,
        "min_budget": 0,
        "max_budget": 65536,
    },
    "gpt-5.6-luna": {
        "supports_thinking": True,
        "supports_thinking_level": True,
        "supports_thinking_budget": False,
        "supports_sampling_parameters": "reasoning_none_only",
        "allow_off": True,
        "allowed_levels": ["none", "low", "medium", "high", "xhigh", "max"],
        "default_level": "medium",
        "default_budget": None,
    },
    "gpt-5.6-terra": {
        "supports_thinking": True,
        "supports_thinking_level": True,
        "supports_thinking_budget": False,
        "supports_sampling_parameters": "reasoning_none_only",
        "allow_off": True,
        "allowed_levels": ["none", "low", "medium", "high", "xhigh", "max"],
        "default_level": "medium",
        "default_budget": None,
    },
    "gpt-5.6-sol": {
        "supports_thinking": True,
        "supports_thinking_level": True,
        "supports_thinking_budget": False,
        "supports_sampling_parameters": "reasoning_none_only",
        "allow_off": True,
        "allowed_levels": ["none", "low", "medium", "high", "xhigh", "max"],
        "default_level": "medium",
        "default_budget": None,
    },
}

# Generation capabilities are independent of the current picker, so legacy
# snapshots remain readable. Gemini 3.8/4 and reasoning Sol/Astra cannot turn off.
for _model, _levels, _default in (
    ("gemini-3.5-flash-lite", ["minimal", "low", "medium", "high"], "minimal"),
    ("gemini-3.8-flash", ["low", "medium", "high"], "medium"),
    ("gemini-4-argon", ["low", "medium", "high"], "medium"),
):
    THINKING_MODEL_CONFIG[_model] = {
        "supports_thinking": True, "supports_thinking_level": True,
        "supports_thinking_budget": False, "supports_sampling_parameters": False,
        "allow_off": False, "allowed_levels": _levels, "default_level": _default,
        "off_replacement_level": _levels[0], "default_budget": None,
        "min_budget": 0, "max_budget": 65536,
    }
THINKING_MODEL_CONFIG["gemini-4-argon"]["max_output_tokens"] = 32768
THINKING_MODEL_CONFIG["gemini-4-argon"]["http_timeout_ms"] = 900000
THINKING_MODEL_CONFIG["gemini-4-argon"]["retry_response_timeouts"] = False
THINKING_MODEL_CONFIG["gemini-3.1-pro-preview"]["supports_sampling_parameters"] = False
THINKING_MODEL_CONFIG["gemini-3.6-flash"]["allowed_levels"] = ["minimal", "low", "medium", "high"]
for _model, _allow_off in (("gpt-6-luna", True), ("gpt-6.1-sol", False), ("gpt-6-astra", False)):
    THINKING_MODEL_CONFIG[_model] = {
        "supports_thinking": True, "supports_thinking_level": True,
        "supports_thinking_budget": False,
        "supports_sampling_parameters": "reasoning_none_only" if _allow_off else False,
        "allow_off": _allow_off,
        "allowed_levels": (["none"] if _allow_off else []) + ["low", "medium", "high", "xhigh", "max"],
        "default_level": "medium", "off_replacement_level": "low", "default_budget": None,
    }

# Backwards-compatible budget-only view used by some helper flows.
THINKING_BUDGET_CONFIG = {
    model_id: {
        "min": cfg.get("min_budget", 0),
        "max": cfg.get("max_budget", 0),
        "default": cfg.get("default_budget", 0),
        "can_disable": bool(cfg.get("allow_off", False)),
    }
    for model_id, cfg in THINKING_MODEL_CONFIG.items()
    if cfg.get("supports_thinking_budget")
}

model_prices_per_million_tokens = {
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.3},
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4, "cached_input": 0.025},
    "gemini-2.5-flash-lite": {"input": 0.1, "output": 0.4, "cached_input": 0.01},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "cached_input": 0.03},
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.0,
        "cached_input": 0.125,
        "prompt_tiers": [
            {
                "max_prompt_tokens": 200_000,
                "input": 1.25,
                "output": 10.0,
                "cached_input": 0.125,
            },
            {
                "max_prompt_tokens": None,
                "input": 2.50,
                "output": 15.0,
                "cached_input": 0.25,
            },
        ],
    },
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00, "cached_input": 0.05},
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50, "cached_input": 0.025},
    "gemini-3.1-pro-preview": {
        "input": 2.00,
        "output": 12.00,
        "cached_input": 0.20,
        "prompt_tiers": [
            {
                "max_prompt_tokens": 200_000,
                "input": 2.00,
                "output": 12.00,
                "cached_input": 0.20,
            },
            {
                "max_prompt_tokens": None,
                "input": 4.00,
                "output": 18.00,
                "cached_input": 0.40,
            },
        ],
    },
    "gemini-3.6-flash": {
        "input": 1.50,
        "output": 7.50,
        "cached_input": 0.15,
    },
    "gpt-5.6-luna": {
        "input": 0.20,
        "output": 1.20,
        "cached_input": 0.02,
        "cache_write_input_multiplier": 1.25,
        "prompt_tiers": [
            {"max_prompt_tokens": 272_000, "input": 0.20, "output": 1.20, "cached_input": 0.02},
            {"max_prompt_tokens": None, "input": 0.40, "output": 1.80, "cached_input": 0.04},
        ],
    },
    "gpt-5.6-terra": {
        "input": 2.00,
        "output": 12.00,
        "cached_input": 0.20,
        "cache_write_input_multiplier": 1.25,
        "prompt_tiers": [
            {"max_prompt_tokens": 272_000, "input": 2.00, "output": 12.00, "cached_input": 0.20},
            {"max_prompt_tokens": None, "input": 4.00, "output": 18.00, "cached_input": 0.40},
        ],
    },
    "gpt-5.6-sol": {
        "input": 4.00,
        "output": 20.00,
        "cached_input": 0.40,
        "cache_write_input_multiplier": 1.25,
        "prompt_tiers": [
            {"max_prompt_tokens": 272_000, "input": 4.00, "output": 20.00, "cached_input": 0.40},
            {"max_prompt_tokens": None, "input": 8.00, "output": 30.00, "cached_input": 0.80},
        ],
    },
}

# Standard paid-tier rates. Published promotional expiry dates are automatic;
# per-response cost snapshots preserve the rate actually used at request time.
for _model in ("gemini-3.6-flash", "gemini-3.8-flash"):
    model_prices_per_million_tokens[_model] = {
        "input": 0.75, "output": 3.75, "cached_input": 0.075,
        "effective_from": "2026-10-13",
        "price_changes": [{"effective_from": "2027-01-01", "input": 1.50, "output": 7.50, "cached_input": 0.15}],
    }
model_prices_per_million_tokens["gemini-3.5-flash-lite"] = {"input": 0.30, "output": 2.50, "cached_input": 0.03}
model_prices_per_million_tokens["gemini-4-argon"] = {
    "input": 2.00, "output": 10.00, "cached_input": 0.10,
    "effective_from": "2026-10-13",
    "price_changes": [{"effective_from": "2026-12-01", "input": 4.00, "output": 20.00, "cached_input": 0.20}],
}
for _model, _input, _cached, _output in (
    ("gpt-6-luna", 0.10, 0.01, 0.50),
    ("gpt-6.1-sol", 2.00, 0.10, 10.00),
    ("gpt-6-astra", 10.00, 1.00, 50.00),
):
    model_prices_per_million_tokens[_model] = {
        "input": _input, "cached_input": _cached, "output": _output,
        "cache_write_input_multiplier": 1.25,
        "prompt_tiers": [
            {"max_prompt_tokens": 272_000, "input": _input, "cached_input": _cached, "output": _output},
            {"max_prompt_tokens": None, "input": _input * 2, "cached_input": _cached * 2, "output": _output * 1.5},
        ],
    }

# Unknown models are explicitly unpriced, never silently billed at a cheap rate.
DEFAULT_TEXT_MODEL_PRICE = {"input": 0.0, "output": 0.0}

def _safe_non_negative_int(value: Any) -> int:
    try:
        numeric = int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(numeric, 0)


def resolve_model_token_prices(model_name: str, prompt_token_count: int = 0, *, priced_at=None) -> Dict[str, Any]:
    """Resolve per-million-token prices for a model and request prompt tier."""
    model_name = str(model_name or "").removeprefix("models/")
    pricing_known = model_name in model_prices_per_million_tokens
    pricing = dict(model_prices_per_million_tokens.get(model_name, DEFAULT_TEXT_MODEL_PRICE))
    when = priced_at if priced_at is not None else datetime.now(timezone.utc)
    if isinstance(when, datetime):
        when = when.astimezone(timezone.utc).date() if when.tzinfo else when.date()
    elif isinstance(when, str):
        when = date.fromisoformat(when[:10])
    if not isinstance(when, date):
        raise ValueError("priced_at must be an ISO date, date, or datetime")
    effective_date = pricing.get("effective_from", MODEL_CATALOG_VERIFIED_AT)
    for change in sorted(pricing.get("price_changes", []), key=lambda row: row["effective_from"]):
        if when >= date.fromisoformat(change["effective_from"]):
            pricing.update(change)
            effective_date = change["effective_from"]
    resolved = {
        "pricing_known": pricing_known,
        "price_version": MODEL_CATALOG_VERIFIED_AT,
        "price_effective_date": effective_date,
        "price_source": (OPENAI_PRICING_SOURCE if get_model_provider(model_name) == "openai" else GEMINI_PRICING_SOURCE) if pricing_known else None,
        "pricing_regime": "standard",
        "input": float(pricing.get("input", DEFAULT_TEXT_MODEL_PRICE["input"])),
        "output": float(pricing.get("output", DEFAULT_TEXT_MODEL_PRICE["output"])),
        "cached_input": float(
            pricing.get(
                "cached_input",
                pricing.get("context_cache", pricing.get("input", DEFAULT_TEXT_MODEL_PRICE["input"])),
            )
        ),
        "large_prompt_tier": False,
        "cache_write_input_multiplier": float(
            pricing.get("cache_write_input_multiplier", 1.0)
        ),
    }

    prompt_tokens = _safe_non_negative_int(prompt_token_count)
    for tier in pricing.get("prompt_tiers", []) or []:
        max_prompt_tokens = tier.get("max_prompt_tokens")
        if max_prompt_tokens is None or prompt_tokens <= int(max_prompt_tokens):
            resolved.update(
                {
                    "input": float(tier.get("input", resolved["input"])),
                    "output": float(tier.get("output", resolved["output"])),
                    "cached_input": float(
                        tier.get(
                            "cached_input",
                            tier.get("context_cache", resolved["cached_input"]),
                        )
                    ),
                    "large_prompt_tier": max_prompt_tokens is None,
                }
            )
            break

    resolved["cache_write_input"] = (
        resolved["input"] * resolved["cache_write_input_multiplier"]
    )

    return resolved


def estimate_generation_token_cost(
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    thought_tokens: int = 0,
    cached_input_tokens: int = 0,
    tool_input_tokens: int = 0,
    cache_write_input_tokens: int = 0,
    *,
    priced_at=None,
) -> Dict[str, Any]:
    """Estimate text generation cost, including thinking tokens and prompt tiers."""
    input_tokens = _safe_non_negative_int(input_tokens)
    output_tokens = _safe_non_negative_int(output_tokens)
    thought_tokens = _safe_non_negative_int(thought_tokens)
    tool_input_tokens = _safe_non_negative_int(tool_input_tokens)
    billable_input_tokens = input_tokens + tool_input_tokens
    cached_input_tokens = min(_safe_non_negative_int(cached_input_tokens), billable_input_tokens)
    cache_write_input_tokens = min(
        _safe_non_negative_int(cache_write_input_tokens),
        billable_input_tokens - cached_input_tokens,
    )
    uncached_input_tokens = max(
        billable_input_tokens - cached_input_tokens - cache_write_input_tokens,
        0,
    )

    prices = resolve_model_token_prices(
        model_name,
        prompt_token_count=billable_input_tokens,
        priced_at=priced_at,
    )
    uncached_input_cost = prices["input"] * uncached_input_tokens / 1_000_000
    cached_input_cost = prices["cached_input"] * cached_input_tokens / 1_000_000
    cache_write_input_cost = (
        prices["cache_write_input"] * cache_write_input_tokens / 1_000_000
    )
    output_cost = prices["output"] * output_tokens / 1_000_000
    thought_cost = prices["output"] * thought_tokens / 1_000_000

    return {
        **{key: prices[key] for key in ("pricing_known", "price_source", "price_version", "price_effective_date", "pricing_regime")},
        "input_tokens": input_tokens,
        "tool_input_tokens": tool_input_tokens,
        "billable_input_tokens": billable_input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_input_tokens": cache_write_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "thought_tokens": thought_tokens,
        "input_rate": prices["input"],
        "cached_input_rate": prices["cached_input"],
        "cache_write_input_rate": prices["cache_write_input"],
        "output_rate": prices["output"],
        "large_prompt_tier": bool(prices.get("large_prompt_tier", False)),
        "uncached_input_cost": uncached_input_cost,
        "cached_input_cost": cached_input_cost,
        "cache_write_input_cost": cache_write_input_cost,
        "input_cost": uncached_input_cost + cached_input_cost + cache_write_input_cost,
        "output_cost": output_cost,
        "thought_cost": thought_cost,
        "total_cost": (
            uncached_input_cost
            + cached_input_cost
            + cache_write_input_cost
            + output_cost
            + thought_cost
        ),
    }

EMBEDDING_MODELS = [
    {
        "id": "gemini-embedding-2",
        "name": "Gemini Embedding 2",
        "description": "Multimodal embedding model for text, image, video, audio, and PDF inputs.",
    },
    {
        "id": "gemini-embedding-001",
        "name": "Gemini Embedding",
        "description": "Text-only embedding model.",
    },
]

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"

embedding_prices_per_million_tokens = {
    "text-embedding-3-small": {
        "input": 0.02,
    },
    "gemini-embedding-2": {
        "text_input": 0.20,
        "image_input": 0.45,
        "audio_input": 6.50,
        "video_input": 12.00,
    },
    "gemini-embedding-001": {
        "input": 0.15,
        "batch_input": 0.075,
    },
}

# SMTP Configuration
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "")

# Run coordination / resilience
RUN_LEASE_SECONDS = int(os.environ.get("RUN_LEASE_SECONDS", 90))
RUN_HEARTBEAT_SECONDS = int(os.environ.get("RUN_HEARTBEAT_SECONDS", 15))
RUN_STATE_WRITE_INTERVAL_SECONDS = float(os.environ.get("RUN_STATE_WRITE_INTERVAL_SECONDS", 1.0))
GLOBAL_MAX_ACTIVE_RUNS = int(os.environ.get("GLOBAL_MAX_ACTIVE_RUNS", 0))

# Progress update smoothing jitter. Keep default at 0 for throughput.
PROGRESS_JITTER_MAX_SECONDS = float(os.environ.get("PROGRESS_JITTER_MAX_SECONDS", 0.0))
