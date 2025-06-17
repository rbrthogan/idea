# Configuration settings shared across the application

# Available LLM models
LLM_MODELS = [
    # {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro (Balanced + Higher Rate Limit)"},
    {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite (Fast + Cheap)"},
    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash (Fast)"},
    {"id": "gemini-2.0-pro-exp-02-05", "name": "Gemini 2.0 Pro (Experimental)"},
    {"id": "gemini-2.0-flash-thinking-exp-01-21", "name": "Gemini 2.0 Flash Thinking (Experimental)"},
    {"id": "gemini-2.5-flash-preview-05-20", "name": "Gemini 2.5 Flash"},
    {"id": "gemini-2.5-pro-preview-06-05", "name": "Gemini 2.5 Pro"}
]

# Default model
DEFAULT_MODEL = "gemini-2.0-flash-lite"

# Oracle configuration
ORACLE_MODES = [
    {"id": "add", "name": "Add New Idea (Grow Population)"},
    {"id": "replace", "name": "Replace Least Diverse Idea"}
]

DEFAULT_ORACLE_MODE = "replace"

model_prices_per_million_tokens = {
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.3},
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
    "gemini-2.0-pro-exp-02-05": {"input": 0.0, "output": 0.0},
    "gemini-2.0-flash-thinking-exp-01-21": {"input": 0.0, "output": 0.0},
    "gemini-2.5-flash-preview-05-20": {"input": 0.15, "output": 3.5},
    "gemini-2.5-pro-preview-06-05": {"input": 2.25, "output": 10.0}
}
