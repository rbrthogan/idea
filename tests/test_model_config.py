import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODELS,
    THINKING_MODEL_CONFIG,
    LLM_MODELS,
    embedding_prices_per_million_tokens,
    estimate_generation_token_cost,
    model_prices_per_million_tokens,
)
from idea.diversity import DiversityCalculator
from idea.viewer import _get_default_thinking_settings, _normalize_thinking_level


def test_gemini_31_flash_lite_is_available_with_pricing_and_thinking_defaults():
    model_id = "gemini-3.1-flash-lite-preview"
    model_ids = [model["id"] for model in LLM_MODELS]

    assert model_id in model_ids
    assert model_prices_per_million_tokens[model_id]["input"] == 0.25
    assert model_prices_per_million_tokens[model_id]["output"] == 1.50
    assert model_prices_per_million_tokens[model_id]["cached_input"] == 0.025

    thinking_config = THINKING_MODEL_CONFIG[model_id]
    assert thinking_config["supports_thinking"] is True
    assert thinking_config["supports_thinking_level"] is True
    assert thinking_config["default_level"] == "low"
    assert thinking_config["allow_off"] is False
    assert _get_default_thinking_settings(model_id) == ("low", None)


def test_gemini_31_pro_uses_current_level_defaults():
    model_id = "gemini-3.1-pro-preview"
    thinking_config = THINKING_MODEL_CONFIG[model_id]

    assert thinking_config["supports_thinking"] is True
    assert thinking_config["supports_thinking_level"] is True
    assert thinking_config["supports_thinking_budget"] is False
    assert thinking_config["allow_off"] is False
    assert thinking_config["allowed_levels"] == ["low", "medium", "high"]
    assert thinking_config["default_level"] == "low"
    assert _get_default_thinking_settings(model_id) == ("low", None)


def test_gemini_36_flash_is_available_with_current_pricing_and_thinking_defaults():
    model_id = "gemini-3.6-flash"
    model_ids = [model["id"] for model in LLM_MODELS]

    assert model_id in model_ids
    assert model_prices_per_million_tokens[model_id] == {
        "input": 1.50,
        "output": 7.50,
        "cached_input": 0.15,
    }

    thinking_config = THINKING_MODEL_CONFIG[model_id]
    assert thinking_config["supports_thinking"] is True
    assert thinking_config["supports_thinking_level"] is True
    assert thinking_config["supports_thinking_budget"] is False
    assert thinking_config["supports_sampling_parameters"] is False
    assert thinking_config["allowed_levels"] == ["low", "medium", "high"]
    assert thinking_config["default_level"] == "medium"
    assert _get_default_thinking_settings(model_id) == ("medium", None)


def test_pro_pricing_tips_to_large_prompt_regime_and_prices_thinking_as_output():
    cost = estimate_generation_token_cost(
        "gemini-2.5-pro",
        input_tokens=210_000,
        output_tokens=1_000,
        thought_tokens=2_000,
        cached_input_tokens=50_000,
    )

    assert cost["uncached_input_tokens"] == 160_000
    assert cost["cached_input_tokens"] == 50_000
    assert cost["input_rate"] == 2.50
    assert cost["cached_input_rate"] == 0.25
    assert cost["output_rate"] == 15.00
    assert cost["large_prompt_tier"] is True
    assert cost["thought_cost"] == 2_000 * 15.00 / 1_000_000
    assert cost["total_cost"] == (
        160_000 * 2.50 / 1_000_000
        + 50_000 * 0.25 / 1_000_000
        + 1_000 * 15.00 / 1_000_000
        + 2_000 * 15.00 / 1_000_000
    )


def test_gemini_31_pro_pricing_tips_to_large_prompt_regime():
    cost = estimate_generation_token_cost(
        "gemini-3.1-pro-preview",
        input_tokens=210_000,
        output_tokens=1_000,
        thought_tokens=2_000,
        cached_input_tokens=50_000,
    )

    assert cost["input_rate"] == 4.00
    assert cost["cached_input_rate"] == 0.40
    assert cost["output_rate"] == 18.00
    assert cost["large_prompt_tier"] is True
    assert cost["total_cost"] == (
        160_000 * 4.00 / 1_000_000
        + 50_000 * 0.40 / 1_000_000
        + 1_000 * 18.00 / 1_000_000
        + 2_000 * 18.00 / 1_000_000
    )


def test_gemini_3_models_default_to_low_thinking_level():
    assert _normalize_thinking_level("minimal") == "minimal"
    assert _get_default_thinking_settings("gemini-3-flash-preview") == (
        "low",
        None,
    )
    assert _get_default_thinking_settings("gemini-3.1-flash-lite-preview") == (
        "low",
        None,
    )
    assert _get_default_thinking_settings("gemini-3.1-pro-preview") == (
        "low",
        None,
    )
    assert _get_default_thinking_settings("gemini-3.6-flash") == (
        "medium",
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
