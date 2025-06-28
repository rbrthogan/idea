# Configuration settings shared across the application

# Available LLM models
LLM_MODELS = [
    {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite (Fast + Cheap)"},
    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash (Fast)"},
    {"id": "gemini-2.5-flash-lite-preview-06-17", "name": "Gemini 2.5 Flash Lite (Fast + Cheap)"},
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
    {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro"}
]

# Default model
DEFAULT_MODEL = "gemini-2.5-flash-lite"

# Default creative settings
DEFAULT_CREATIVE_TEMP = 1.5
DEFAULT_TOP_P = 0.90

# Thinking budget configuration for Gemini 2.5 models
THINKING_BUDGET_CONFIG = {
    "gemini-2.5-pro": {
        "min": 128,
        "max": 32768,
        "default": 128,  # Minimum possible (can't disable)
        "can_disable": False  # Cannot set to 0
    },
    "gemini-2.5-flash": {
        "min": 0,
        "max": 24576,
        "default": 0,  # Disabled thinking
        "can_disable": True  # Can set to 0
    },
    "gemini-2.5-flash-lite-preview-06-17": {
        "min": 512,
        "max": 24576,
        "default": 0,  # Disabled thinking
        "can_disable": True  # Can set to 0
    }
}

model_prices_per_million_tokens = {
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.3},
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
    "gemini-2.5-flash-lite-preview-06-17": {"input": 0.1, "output": 0.4},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0}
}
