import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    THINKING_MODEL_CONFIG,
    LLM_MODELS,
    embedding_prices_per_million_tokens,
    model_prices_per_million_tokens,
)
from idea.diversity import DiversityCalculator
from idea.viewer import _get_default_thinking_settings, _normalize_thinking_level


def test_gemini_31_flash_lite_is_available_with_pricing_and_thinking_defaults():
    model_id = "gemini-3.1-flash-lite-preview"
    model_ids = [model["id"] for model in LLM_MODELS]

    assert model_id in model_ids
    assert model_prices_per_million_tokens[model_id] == {
        "input": 0.25,
        "output": 1.50,
    }

    thinking_config = THINKING_MODEL_CONFIG[model_id]
    assert thinking_config["supports_thinking"] is True
    assert thinking_config["supports_thinking_level"] is True
    assert thinking_config["default_level"] == "minimal"
    assert thinking_config["allow_off"] is False
    assert _get_default_thinking_settings(model_id) == ("minimal", None)


def test_gemini_3_models_use_minimal_instead_of_off_thinking_level():
    assert _normalize_thinking_level("minimal") == "minimal"
    assert _get_default_thinking_settings("gemini-3-flash-preview") == (
        "minimal",
        None,
    )


def test_gemini_embedding_2_is_default_with_legacy_embedding_fallback():
    assert DEFAULT_EMBEDDING_MODEL == "gemini-embedding-2"
    assert EMBEDDING_MODELS[0]["id"] == "gemini-embedding-2"
    assert embedding_prices_per_million_tokens["gemini-embedding-2"]["text_input"] == 0.20
    assert embedding_prices_per_million_tokens["gemini-embedding-001"]["input"] == 0.15

    calculator = DiversityCalculator(api_key=None)

    assert calculator.embedding_model == "gemini-embedding-2"
    assert calculator._embedding_models[:2] == [
        "gemini-embedding-2",
        "models/gemini-embedding-2",
    ]
    assert "gemini-embedding-001" in calculator._embedding_models
