# Configuration settings shared across the application

# Available LLM models
LLM_MODELS = [
    {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash (Fast)"},
    {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro (Balanced)"},
    {"id": "gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash (Experimental)"},
    {"id": "gemini-2.0-flash-thinking-exp-01-21", "name": "Gemini 2.0 Flash Thinking (Experimental)"}
]

# Default model
DEFAULT_MODEL = "gemini-1.5-flash"