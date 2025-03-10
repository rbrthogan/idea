# Configuration settings shared across the application

# Available LLM models
LLM_MODELS = [
    {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro (Balanced + Higher Rate Limit)"},
    {"id": "gemini-2.0-flash-lite", "name": "Gemini 2.0 Flash Lite (Fast + Cheap)"},
    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash (Fast)"},
    {"id": "gemini-2.0-pro-exp-02-05", "name": "Gemini 2.0 Pro (Experimental)"},
    {"id": "gemini-2.0-flash-thinking-exp-01-21", "name": "Gemini 2.0 Flash Thinking (Experimental)"}
]

# Default model
DEFAULT_MODEL = "gemini-2.0-flash-lite"

# Task types
TASK_TYPES = [
    {"id": "airesearch", "name": "AI Research"},
    {"id": "game_design", "name": "Game Design"},
    {"id": "drabble", "name": "Drabble (100-word story)"}
]

model_prices_per_million_tokens = {
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.3},
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
    "gemini-2.0-pro-exp-02-05": {"input": 0.0, "output": 0.0},
    "gemini-2.0-flash-thinking-exp-01-21": {"input": 0.0, "output": 0.0}
}