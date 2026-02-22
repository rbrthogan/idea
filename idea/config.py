# Configuration settings shared across the application

# Available LLM models
LLM_MODELS = [
    {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite (Fast + Cheap)"},
    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash (Fast)"},
    {"id": "gemini-2.5-flash-lite", "name": "Gemini 2.5 Flash Lite (Fast + Cheap)"},
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"},
    {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash Preview"},
    {"id": "gemini-3-pro-preview", "name": "Gemini 3 Pro Preview"}
]

# Default model
DEFAULT_MODEL = "gemini-2.5-flash-lite"

# Default creative settings
DEFAULT_CREATIVE_TEMP = 1.5
DEFAULT_TOP_P = 0.90

# Thinking controls per model.
# NOTE: The python SDK currently exposes thinking_budget directly. thinking_level
# is accepted by the app layer and mapped to budget when needed.
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
        "default_level": "off",
        "default_budget": 0,
        "min_budget": 128,
        "max_budget": 24576,
    },
    "gemini-2.5-pro": {
        "supports_thinking": True,
        "supports_thinking_level": False,
        "supports_thinking_budget": True,
        "allow_off": False,
        "default_level": "low",
        "default_budget": 1024,
        "min_budget": 128,
        "max_budget": 32768,
    },
    "gemini-3-flash-preview": {
        "supports_thinking": True,
        "supports_thinking_level": True,
        "supports_thinking_budget": True,
        "allow_off": True,
        "default_level": "off",
        "default_budget": 0,
        "min_budget": 0,
        "max_budget": 24576,
    },
    "gemini-3-pro-preview": {
        "supports_thinking": True,
        "supports_thinking_level": True,
        "supports_thinking_budget": True,
        "allow_off": False,
        "default_level": "low",
        "default_budget": 1024,
        "min_budget": 128,
        "max_budget": 32768,
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
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
    "gemini-2.5-flash-lite": {"input": 0.1, "output": 0.4},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00}
}

# SMTP Configuration
import os
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
