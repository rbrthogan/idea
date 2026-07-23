import os
from typing import Any, Dict

# Configuration settings shared across the application

# Available LLM models
LLM_MODELS = [
    {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite (Fast + Cheap)"},
    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash (Fast)"},
    {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash Lite (Fast + Cheap)"},
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
    {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash Preview"},
    {"id": "gemini-3.1-flash-lite-preview", "name": "Gemini 3.1 Flash-Lite Preview"},
    {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro Preview"},
    {"id": "gemini-3.6-flash", "name": "Gemini 3.6 Flash"},
]

# Default model
DEFAULT_MODEL = "gemini-2.5-flash-lite"

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
}

DEFAULT_TEXT_MODEL_PRICE = {"input": 0.1, "output": 0.4}


def _safe_non_negative_int(value: Any) -> int:
    try:
        numeric = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(numeric, 0)


def resolve_model_token_prices(model_name: str, prompt_token_count: int = 0) -> Dict[str, Any]:
    """Resolve per-million-token prices for a model and request prompt tier."""
    pricing = model_prices_per_million_tokens.get(model_name, DEFAULT_TEXT_MODEL_PRICE)
    resolved = {
        "input": float(pricing.get("input", DEFAULT_TEXT_MODEL_PRICE["input"])),
        "output": float(pricing.get("output", DEFAULT_TEXT_MODEL_PRICE["output"])),
        "cached_input": float(
            pricing.get(
                "cached_input",
                pricing.get("context_cache", pricing.get("input", DEFAULT_TEXT_MODEL_PRICE["input"])),
            )
        ),
        "large_prompt_tier": False,
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

    return resolved


def estimate_generation_token_cost(
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    thought_tokens: int = 0,
    cached_input_tokens: int = 0,
    tool_input_tokens: int = 0,
) -> Dict[str, Any]:
    """Estimate text generation cost, including thinking tokens and prompt tiers."""
    input_tokens = _safe_non_negative_int(input_tokens)
    output_tokens = _safe_non_negative_int(output_tokens)
    thought_tokens = _safe_non_negative_int(thought_tokens)
    tool_input_tokens = _safe_non_negative_int(tool_input_tokens)
    billable_input_tokens = input_tokens + tool_input_tokens
    cached_input_tokens = min(_safe_non_negative_int(cached_input_tokens), billable_input_tokens)
    uncached_input_tokens = max(billable_input_tokens - cached_input_tokens, 0)

    prices = resolve_model_token_prices(
        model_name,
        prompt_token_count=billable_input_tokens,
    )
    uncached_input_cost = prices["input"] * uncached_input_tokens / 1_000_000
    cached_input_cost = prices["cached_input"] * cached_input_tokens / 1_000_000
    output_cost = prices["output"] * output_tokens / 1_000_000
    thought_cost = prices["output"] * thought_tokens / 1_000_000

    return {
        "input_tokens": input_tokens,
        "tool_input_tokens": tool_input_tokens,
        "billable_input_tokens": billable_input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "thought_tokens": thought_tokens,
        "input_rate": prices["input"],
        "cached_input_rate": prices["cached_input"],
        "output_rate": prices["output"],
        "large_prompt_tier": bool(prices.get("large_prompt_tier", False)),
        "uncached_input_cost": uncached_input_cost,
        "cached_input_cost": cached_input_cost,
        "input_cost": uncached_input_cost + cached_input_cost,
        "output_cost": output_cost,
        "thought_cost": thought_cost,
        "total_cost": uncached_input_cost + cached_input_cost + output_cost + thought_cost,
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
